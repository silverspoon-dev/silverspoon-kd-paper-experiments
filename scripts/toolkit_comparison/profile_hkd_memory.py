"""
Memory profiling: hooks vs output_hidden_states for HKD.

Both approaches use identical manual training loops — same teacher, student,
optimizer, warmup, loss computation. The ONLY difference is how intermediate
features are obtained:
  A) output_hidden_states=True (what TextBrewer does)
  B) register_forward_hook on each layer (what SK does)

This isolates the memory impact of hooks vs native hidden state output.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/toolkit_comparison/profile_hkd_memory.py
"""
import gc
import sys
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR))

TEACHER_PATH = PROJECT_DIR / "runs" / "hf__standard__bert_base_cased__mnli__scratch" / "model"
TEACHER_LAYERS = [1, 3, 5, 7, 9, 11]  # teacher layers aligned to student [0..5]
BATCH_SIZE = 32
SEQ_LEN = 128
NUM_LABELS = 3
TEMPERATURE = 4.0
WARMUP_STEPS = 5

VOCAB_SIZE = None


def mb(b):
    return b / 1024 / 1024


def mem_now():
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated()


def peak_mem():
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated()


def reset(device):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)


def make_batch(device):
    return {
        "input_ids": torch.randint(1, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=device),
        "attention_mask": torch.ones(BATCH_SIZE, SEQ_LEN, dtype=torch.long, device=device),
        "token_type_ids": torch.zeros(BATCH_SIZE, SEQ_LEN, dtype=torch.long, device=device),
        "labels": torch.randint(0, NUM_LABELS, (BATCH_SIZE,), device=device),
    }


def create_t6_student(teacher_model):
    student_config = deepcopy(teacher_model.config)
    student_config.num_hidden_layers = 6
    student = AutoModelForSequenceClassification.from_config(student_config)
    student.bert.embeddings.load_state_dict(teacher_model.bert.embeddings.state_dict())
    for s_idx, t_idx in enumerate(TEACHER_LAYERS):
        student.bert.encoder.layer[s_idx].load_state_dict(
            teacher_model.bert.encoder.layer[t_idx].state_dict())
    student.bert.pooler.load_state_dict(teacher_model.bert.pooler.state_dict())
    student.classifier.load_state_dict(teacher_model.classifier.state_dict())
    return student


def compute_hkd_loss(t_hidden, s_hidden, t_logits, s_logits):
    """Shared loss: MSE on 6 hidden pairs + KL on logits.

    t_hidden and s_hidden are already aligned lists of length 6.
    """
    loss = torch.zeros(1, device=s_logits.device)
    for i in range(len(s_hidden)):
        loss = loss + F.mse_loss(s_hidden[i], t_hidden[i])
    loss = loss + F.kl_div(
        F.log_softmax(s_logits / TEMPERATURE, dim=-1),
        F.softmax(t_logits.detach() / TEMPERATURE, dim=-1),
        reduction="batchmean") * (TEMPERATURE ** 2)
    return loss


# ── Approach A: output_hidden_states ────────────────────────────────────────

def train_step_hidden_states(teacher, student, optimizer, batch):
    """One training step using output_hidden_states=True."""
    teacher_inputs = {k: v for k, v in batch.items() if k != "labels"}
    with torch.no_grad():
        t_out = teacher(**teacher_inputs)
    s_out = student(**batch)

    # t_out.hidden_states[0] = embeddings, [1..12] = layer outputs
    # s_out.hidden_states[0] = embeddings, [1..6] = layer outputs
    t_hidden = [t_out.hidden_states[t_idx + 1] for t_idx in TEACHER_LAYERS]
    s_hidden = [s_out.hidden_states[s_idx + 1] for s_idx in range(6)]

    loss = compute_hkd_loss(t_hidden, s_hidden, t_out.logits, s_out.logits)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss.item()


# ── Approach B: forward hooks ───────────────────────────────────────────────

class HookCapture:
    """Minimal hook-based capture — mirrors what ModuleCaptureEngine does."""

    def __init__(self, model, modules):
        self.captured = {}
        self._hooks = []
        for idx, mod in enumerate(modules):
            hook = mod.register_forward_hook(self._make_hook(idx))
            self._hooks.append(hook)

    def _make_hook(self, idx):
        def hook_fn(module, input, output):
            # Store the hidden state (first element if tuple)
            if isinstance(output, tuple):
                self.captured[idx] = output[0]
            else:
                self.captured[idx] = output
        return hook_fn

    def pop(self, idx):
        return self.captured.pop(idx)

    def clear(self):
        self.captured.clear()

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


def train_step_hooks(teacher, student, optimizer, batch,
                     teacher_capture, student_capture):
    """One training step using forward hooks to capture intermediates."""
    teacher_inputs = {k: v for k, v in batch.items() if k != "labels"}

    teacher_capture.clear()
    student_capture.clear()

    with torch.no_grad():
        t_out = teacher(**teacher_inputs)
    s_out = student(**batch)

    t_hidden = [teacher_capture.pop(i) for i in range(len(TEACHER_LAYERS))]
    s_hidden = [student_capture.pop(i) for i in range(6)]

    loss = compute_hkd_loss(t_hidden, s_hidden, t_out.logits, s_out.logits)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss.item()


# ── Profiling ───────────────────────────────────────────────────────────────

def profile_approach(name, step_fn, device):
    """Run warmup then measure peak memory on a single profiled step."""
    # Warmup: stabilize allocator + initialize optimizer states
    for _ in range(WARMUP_STEPS):
        batch = make_batch(device)
        step_fn(batch)
        del batch

    gc.collect()
    torch.cuda.empty_cache()

    # Profile
    torch.cuda.reset_peak_memory_stats(device)
    batch = make_batch(device)
    step_fn(batch)
    p = peak_mem()
    del batch
    print(f"  {name}: peak = {mb(p):.1f} MB")
    return p


def main():
    device = torch.device("cuda:0")
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"Config: batch={BATCH_SIZE}, seq={SEQ_LEN}, alignments=6 hidden + 1 KL")
    print(f"Warmup: {WARMUP_STEPS} steps before profiled step")
    print()

    print("Loading teacher...")
    teacher_base = AutoModelForSequenceClassification.from_pretrained(str(TEACHER_PATH))
    global VOCAB_SIZE
    VOCAB_SIZE = teacher_base.config.vocab_size
    print(f"Vocab: {VOCAB_SIZE}, Hidden: {teacher_base.config.hidden_size}")
    student_base = create_t6_student(teacher_base)

    # ── Approach A: output_hidden_states ──────────────────────────────────
    print("\n" + "=" * 60)
    print("  A) output_hidden_states=True (TextBrewer-style)")
    print("=" * 60)

    teacher_a = deepcopy(teacher_base).to(device).eval()
    student_a = deepcopy(student_base).to(device).train()
    teacher_a.config.output_hidden_states = True
    student_a.config.output_hidden_states = True
    optimizer_a = AdamW(student_a.parameters(), lr=1e-4)

    peak_a = profile_approach(
        "output_hidden_states",
        lambda batch: train_step_hidden_states(teacher_a, student_a, optimizer_a, batch),
        device,
    )

    del teacher_a, student_a, optimizer_a
    gc.collect()
    torch.cuda.empty_cache()

    # ── Approach B: forward hooks ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  B) forward hooks (SK-style)")
    print("=" * 60)

    teacher_b = deepcopy(teacher_base).to(device).eval()
    student_b = deepcopy(student_base).to(device).train()
    optimizer_b = AdamW(student_b.parameters(), lr=1e-4)

    # Hook the same layers as Approach A
    teacher_modules = [teacher_b.bert.encoder.layer[i] for i in TEACHER_LAYERS]
    student_modules = [student_b.bert.encoder.layer[i] for i in range(6)]
    # Also hook classifiers for KL
    teacher_modules.append(teacher_b.classifier)
    student_modules.append(student_b.classifier)

    teacher_capture = HookCapture(teacher_b, teacher_modules)
    student_capture = HookCapture(student_b, student_modules)

    peak_b = profile_approach(
        "forward hooks",
        lambda batch: train_step_hooks(
            teacher_b, student_b, optimizer_b, batch,
            teacher_capture, student_capture),
        device,
    )

    teacher_capture.remove_hooks()
    student_capture.remove_hooks()
    del teacher_b, student_b, optimizer_b
    gc.collect()
    torch.cuda.empty_cache()

    # ── Approach C: SK HolisticDistiller (full Trainer) ──────────────────
    print("\n" + "=" * 60)
    print("  C) SK HolisticDistiller (full Trainer, for reference)")
    print("=" * 60)

    try:
        import os
        from silverspoon_kd import HolisticDistiller, TrainingArguments
        from silverspoon_kd.alignments.alignment import Alignment
        from silverspoon_kd.losses.kl import kl_divergence_loss as sk_kl
        from torch.utils.data import Dataset

        teacher_c = deepcopy(teacher_base).to(device).eval()
        student_c = deepcopy(student_base).to(device).train()

        alignments = [
            Alignment(
                teacher_block=teacher_c.bert.encoder.layer[t_idx],
                student_block=student_c.bert.encoder.layer[s_idx],
                teacher_model_name="bert-base", student_model_name="bert-t6",
                teacher_module_name=f"bert.encoder.layer.{t_idx}",
                student_module_name=f"bert.encoder.layer.{s_idx}",
                loss_function="mse", auto_projector=False, loss_weight=1.0,
            )
            for s_idx, t_idx in enumerate(TEACHER_LAYERS)
        ]
        alignments.append(Alignment(
            teacher_block=teacher_c.classifier, student_block=student_c.classifier,
            teacher_model_name="bert-base", student_model_name="bert-t6",
            teacher_module_name="classifier", student_module_name="classifier",
            loss_function=sk_kl(temperature=TEMPERATURE),
            auto_projector=False, loss_weight=1.0,
        ))

        class DummyDS(Dataset):
            def __len__(self):
                return BATCH_SIZE * (WARMUP_STEPS + 2)
            def __getitem__(self, idx):
                return {
                    "input_ids": torch.randint(1, VOCAB_SIZE, (SEQ_LEN,)),
                    "attention_mask": torch.ones(SEQ_LEN, dtype=torch.long),
                    "token_type_ids": torch.zeros(SEQ_LEN, dtype=torch.long),
                    "labels": torch.randint(0, NUM_LABELS, ()),
                }

        output_dir = "/tmp/sk_hkd_profile"
        os.makedirs(output_dir, exist_ok=True)
        args = TrainingArguments(
            output_dir=output_dir, max_steps=WARMUP_STEPS + 1,
            per_device_train_batch_size=BATCH_SIZE, learning_rate=1e-4,
            lr_scheduler_type="linear", warmup_steps=0, max_grad_norm=1.0,
            report_to="none", logging_steps=9999,
            save_strategy="no", eval_strategy="no", disable_tqdm=False,
            alpha=0.0,
        )

        def prep_teacher(inputs):
            return {k: v for k, v in inputs.items()
                    if k in ("input_ids", "attention_mask", "token_type_ids")}

        hkd = HolisticDistiller(
            student_model=student_c, teacher_model=teacher_c,
            alignments=alignments, train_dataset=DummyDS(),
            args=args, prepare_teacher_inputs=prep_teacher,
        )

        reset(device)
        hkd.train()
        peak_c = peak_mem()
        print(f"  SK HolisticDistiller: peak = {mb(peak_c):.1f} MB")

        del teacher_c, student_c, hkd
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        import traceback
        traceback.print_exc()
        peak_c = 0
        print(f"  SK HolisticDistiller failed: {e}")

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  A) output_hidden_states:  {mb(peak_a):>8.1f} MB  (baseline)")
    print(f"  B) forward hooks:         {mb(peak_b):>8.1f} MB  ({mb(peak_b - peak_a):>+.1f} MB / {(peak_b - peak_a) / peak_a * 100:>+.1f}%)")
    if peak_c > 0:
        print(f"  C) SK HolisticDistiller:  {mb(peak_c):>8.1f} MB  ({mb(peak_c - peak_a):>+.1f} MB / {(peak_c - peak_a) / peak_a * 100:>+.1f}%)")
    print()
    if abs(peak_b - peak_a) < peak_a * 0.02:
        print("  Hooks vs output_hidden_states: <2% difference.")
        print("  The memory overhead in the TextBrewer comparison is NOT from hooks.")
        if peak_c > 0 and peak_c > peak_b * 1.05:
            print(f"  The extra {mb(peak_c - peak_b):.0f} MB in SK Trainer is from Trainer/distiller overhead.")
    else:
        print(f"  Hooks add {mb(peak_b - peak_a):.0f} MB ({(peak_b - peak_a) / peak_a * 100:.1f}%) vs output_hidden_states.")


if __name__ == "__main__":
    main()
