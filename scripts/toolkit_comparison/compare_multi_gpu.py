"""
Multi-GPU benchmark: silverspoon-kd vs torchdistill, TextBrewer, and DistillKit.

Demonstrates that silverspoon-kd is competitive in DDP throughput against every
major KD toolkit, then showcases teacher placement strategies that no other
toolkit supports.

Part 1 — Vision DDP (silverspoon-kd vs torchdistill)
  ResNet-50 → ResNet-18 / ImageNet, 2 GPUs, DDP.

Part 2 — NLP DDP (silverspoon-kd vs TextBrewer)
  BERT-base → BERT-T6 / MNLI, 2 GPUs, DDP.

Part 3 — LLM DDP (silverspoon-kd vs DistillKit)
  Qwen3-4B → Qwen3-0.6B / Dolma, 2 GPUs, DDP.

Part 4 — Teacher Placement (silverspoon-kd only)
  Qwen3-4B → Qwen3-0.6B / Dolma with replicated / FSDP-sharded / split-GPU.
  No other KD library supports FSDP teacher sharding or split-GPU placement.

Part 5 — OOM Boundary (silverspoon-kd only)
  Finds the maximum batch size for each teacher placement, then measures
  throughput at that max BS. Shows that advanced placement enables larger
  batch sizes → higher throughput than replicated-only toolkits.

Usage:
    # All parts (4 GPUs recommended, 2 minimum):
    CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/toolkit_comparison/compare_multi_gpu.py --part all

    # Individual parts:
    CUDA_VISIBLE_DEVICES=0,1 python scripts/toolkit_comparison/compare_multi_gpu.py --part vision
    CUDA_VISIBLE_DEVICES=0,1 python scripts/toolkit_comparison/compare_multi_gpu.py --part nlp
    CUDA_VISIBLE_DEVICES=0,1 python scripts/toolkit_comparison/compare_multi_gpu.py --part llm
    CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/toolkit_comparison/compare_multi_gpu.py --part placement
"""

import argparse
import gc
import json
import os
import socket
import sys
import tempfile
import time
from copy import deepcopy
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageClassification,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    DefaultDataCollator,
    get_linear_schedule_with_warmup,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR))

# Use SLURM's TMPDIR (node-local, more space) or fall back to /tmp.
_TMPDIR = os.environ.get("TMPDIR", "/tmp")

TEACHER_BERT_PATH = PROJECT_DIR / "runs" / "hf__standard__bert_base_cased__mnli__scratch" / "model"
TEACHER_BERT_HF = "textattack/bert-base-uncased-MNLI"  # fallback if local teacher not found


# ══════════════════════════════════════════════════════════════════════════════
# Shared utilities
# ══════════════════════════════════════════════════════════════════════════════


def _find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _setup_dist(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def _cleanup_dist():
    if dist.is_initialized():
        dist.destroy_process_group()
    # Clean env vars so next spawn starts fresh
    for k in ("MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK", "RANK", "WORLD_SIZE"):
        os.environ.pop(k, None)


def _reset_gpu(device):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)


def _peak_mb(device):
    torch.cuda.synchronize(device)
    return torch.cuda.max_memory_allocated(device) / 1024 / 1024


def _write_result(path, d):
    with open(path, "w") as f:
        json.dump(d, f)


def _read_result(path):
    with open(path) as f:
        return json.load(f)


def _spawn_run(worker_fn, world_size, args_dict, label="", visible_gpus=None):
    """Spawn *worker_fn* on *world_size* ranks.

    When more GPUs are visible than *world_size* (e.g. 4-GPU node running
    a 2-rank probe), unused devices can cause NCCL/FSDP interference.
    Pass *visible_gpus* (e.g. ``[0, 1]``) to restrict
    ``CUDA_VISIBLE_DEVICES`` for the spawned children.
    """
    port = _find_free_port()
    result_file = tempfile.mktemp(suffix=".json", prefix="bench_")
    print(f"\n{'=' * 70}")
    print(f"  {label}  ({world_size} GPUs)")
    print(f"{'=' * 70}")
    saved_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in visible_gpus)
    try:
        mp.spawn(worker_fn, args=(world_size, port, result_file, args_dict),
                 nprocs=world_size, join=True)
        result = _read_result(result_file)
        os.unlink(result_file)
        print(json.dumps(result, indent=2))
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  FAILED: {e}")
        return {"toolkit": label, "error": str(e)}
    finally:
        # Restore CUDA_VISIBLE_DEVICES so the next probe starts fresh
        if saved_cvd is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = saved_cvd
        elif visible_gpus is not None and "CUDA_VISIBLE_DEVICES" in os.environ:
            del os.environ["CUDA_VISIBLE_DEVICES"]


# ══════════════════════════════════════════════════════════════════════════════
# Part 1 — Vision: torchdistill vs silverspoon-kd (ResNet DDP)
# ══════════════════════════════════════════════════════════════════════════════


class ImageNetMapDataset(Dataset):
    def __init__(self, hf_dataset, transform):
        self.ds = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]
        img = sample["image"].convert("RGB")
        return {"pixel_values": self.transform(img), "labels": sample["label"]}


def _load_imagenet_train(num_shards=10):
    from datasets import load_dataset
    from torchvision.transforms import (
        Compose, Normalize, RandomHorizontalFlip, RandomResizedCrop, ToTensor,
    )
    base = "https://huggingface.co/datasets/ILSVRC/imagenet-1k/resolve/refs/convert/parquet/default"
    raw = load_dataset(
        "parquet",
        data_files={"train": [f"{base}/train/{i:04d}.parquet" for i in range(num_shards)]},
        split="train",
    )
    tf = Compose([RandomResizedCrop(224), RandomHorizontalFlip(), ToTensor(),
                  Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    return ImageNetMapDataset(raw, tf)


def _load_imagenet_eval():
    from datasets import load_dataset
    from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
    base = "https://huggingface.co/datasets/ILSVRC/imagenet-1k/resolve/refs/convert/parquet/default"
    raw = load_dataset("parquet",
                       data_files={"val": f"{base}/validation/0000.parquet"}, split="val[:500]")
    tf = Compose([Resize(256), CenterCrop(224), ToTensor(),
                  Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    return ImageNetMapDataset(raw, tf)


def _resnet50_teacher(device):
    m = AutoModelForImageClassification.from_pretrained("timm/resnet50.a1_in1k").to(device).eval()
    for p in m.parameters():
        p.requires_grad = False
    return m


def _resnet18_student(device):
    m = AutoModelForImageClassification.from_pretrained("timm/resnet18.a1_in1k")
    for mod in m.modules():
        if isinstance(mod, nn.Conv2d):
            nn.init.kaiming_normal_(mod.weight, mode="fan_out", nonlinearity="relu")
            if mod.bias is not None:
                nn.init.zeros_(mod.bias)
        elif isinstance(mod, nn.Linear):
            nn.init.normal_(mod.weight, 0, 0.01)
            if mod.bias is not None:
                nn.init.zeros_(mod.bias)
        elif isinstance(mod, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(mod.weight)
            nn.init.zeros_(mod.bias)
    return m.to(device)


@torch.no_grad()
def _eval_accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    for b in loader:
        logits = model(pixel_values=b["pixel_values"].to(device)).logits
        correct += (logits.argmax(-1) == b["labels"].to(device)).sum().item()
        total += b["labels"].size(0)
    return 100.0 * correct / total if total else 0.0


# --- Vision workers ---

def _w_torchdistill_vision(rank, world_size, port, result_file, a):
    _setup_dist(rank, world_size, port)
    device = torch.device(f"cuda:{rank}")
    from torchdistill.losses.mid_level import KDLoss

    teacher = _resnet50_teacher(device)
    student = DDP(_resnet18_student(device), device_ids=[rank])
    ds = _load_imagenet_train(a["num_shards"])
    sampler = DistributedSampler(ds, world_size, rank, shuffle=True)
    loader = DataLoader(ds, batch_size=a["batch_size"], sampler=sampler,
                        collate_fn=DefaultDataCollator(), num_workers=4, pin_memory=True)
    kd = KDLoss(student_module_path=".", student_module_io="output",
                teacher_module_path=".", teacher_module_io="output",
                temperature=a["T"], alpha=a["alpha"], beta=1 - a["alpha"], reduction="batchmean")
    opt = AdamW(student.parameters(), lr=a["lr"])
    sched = get_linear_schedule_with_warmup(opt, int(a["steps"] * a["warmup"]), a["steps"])

    student.train()
    _reset_gpu(device)
    t0 = time.perf_counter()
    step = 0
    while step < a["steps"]:
        sampler.set_epoch(step)
        for b in loader:
            if step >= a["steps"]:
                break
            pv = b["pixel_values"].to(device)
            labels = b["labels"].to(device)
            s_logits = student(pixel_values=pv).logits
            with torch.no_grad():
                t_logits = teacher(pixel_values=pv).logits
            loss = kd(student_io_dict={".": {"output": s_logits}},
                      teacher_io_dict={".": {"output": t_logits}}, targets=labels)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0
    mem = _peak_mb(device)

    if rank == 0:
        ev = _load_imagenet_eval()
        acc = _eval_accuracy(student.module, DataLoader(ev, 64, collate_fn=DefaultDataCollator()), device)
        _write_result(result_file, {
            "toolkit": f"torchdistill KD (DDP, {world_size} GPUs)",
            "test_accuracy_pct": round(acc, 2), "wall_clock_sec": round(elapsed, 2),
            "samples_per_sec": round(a["steps"] * a["batch_size"] * world_size / elapsed, 1),
            "peak_gpu_mb": round(mem, 0),
        })
    del teacher, student; gc.collect(); torch.cuda.empty_cache()
    _cleanup_dist()


def _w_silverspoon_vision(rank, world_size, port, result_file, a):
    _setup_dist(rank, world_size, port)
    device = torch.device(f"cuda:{rank}")
    from silverspoon_kd import ResponseBasedDistiller, TrainingArguments

    teacher = _resnet50_teacher(device)
    student = _resnet18_student(device)
    ds = _load_imagenet_train(a["num_shards"])
    out = f"{_TMPDIR}/sk_vision_ddp_{rank}"; os.makedirs(out, exist_ok=True)

    args = TrainingArguments(
        output_dir=out, run_name="sk_vision_ddp", max_steps=a["steps"],
        per_device_train_batch_size=a["batch_size"], learning_rate=a["lr"],
        lr_scheduler_type="linear", warmup_ratio=a["warmup"], max_grad_norm=1.0,
        bf16=False, fp16=False, report_to="none", logging_steps=99999,
        save_strategy="no", eval_strategy="no", disable_tqdm=True,
        alpha=a["alpha"],
        auto_dtype_match=True, optim="adamw_torch", dataloader_num_workers=4,
    )
    from silverspoon_kd.losses.kl import kl_divergence_loss
    distiller = ResponseBasedDistiller(
        student_model=student, teacher_model=teacher, train_dataset=ds,
        data_collator=DefaultDataCollator(), args=args,
        soft_loss_fn=kl_divergence_loss(temperature=a["T"]),
        prepare_teacher_inputs=lambda inp: {"pixel_values": inp["pixel_values"]},
    )
    _reset_gpu(device)
    t0 = time.perf_counter()
    distiller.train()
    torch.cuda.synchronize(device); elapsed = time.perf_counter() - t0; mem = _peak_mb(device)

    if rank == 0:
        ev = _load_imagenet_eval()
        mdl = distiller.model.module if hasattr(distiller.model, "module") else distiller.model
        acc = _eval_accuracy(mdl, DataLoader(ev, 64, collate_fn=DefaultDataCollator()), device)
        _write_result(result_file, {
            "toolkit": f"silverspoon-kd response-based KD (DDP, {world_size} GPUs)",
            "test_accuracy_pct": round(acc, 2), "wall_clock_sec": round(elapsed, 2),
            "samples_per_sec": round(a["steps"] * a["batch_size"] * world_size / elapsed, 1),
            "peak_gpu_mb": round(mem, 0),
        })
    del teacher, student, distiller; gc.collect(); torch.cuda.empty_cache()
    _cleanup_dist()


def _w_silverspoon_vision_hkd(rank, world_size, port, result_file, a):
    """silverspoon-kd HKD: ResNet50→18 with per-stage MSE alignment (DDP)."""
    _setup_dist(rank, world_size, port)
    device = torch.device(f"cuda:{rank}")
    from silverspoon_kd import HolisticDistiller, TrainingArguments
    from silverspoon_kd.alignments.alignment import Alignment

    teacher = _resnet50_teacher(device)
    student = _resnet18_student(device)
    ds = _load_imagenet_train(a["num_shards"])
    out = f"{_TMPDIR}/sk_vision_hkd_{rank}"; os.makedirs(out, exist_ok=True)

    # Align ResNet stages (layer1-4). timm/HF wraps as TimmWrapperForImageClassification
    # with inner timm_model.layer1..layer4.
    # Dimension mismatch: ResNet50 channels [256,512,1024,2048] vs ResNet18 [64,128,256,512]
    # auto_projector=True handles this.
    stage_names = ["timm_model.layer1", "timm_model.layer2",
                   "timm_model.layer3", "timm_model.layer4"]
    alignments = []
    for stage_name in stage_names:
        t_block = teacher
        s_block = student
        for attr in stage_name.split("."):
            t_block = getattr(t_block, attr)
            s_block = getattr(s_block, attr)
        alignments.append(Alignment(
            teacher_block=t_block, student_block=s_block,
            teacher_model_name="resnet50", student_model_name="resnet18",
            teacher_module_name=stage_name, student_module_name=stage_name,
            loss_function="mse", auto_projector=True, loss_weight=1.0,
            auto_device_match=True, auto_dtype_match=True,
        ))

    args = TrainingArguments(
        output_dir=out, run_name="sk_vision_hkd", max_steps=a["steps"],
        per_device_train_batch_size=a["batch_size"], learning_rate=a["lr"],
        lr_scheduler_type="linear", warmup_ratio=a["warmup"], max_grad_norm=1.0,
        bf16=False, fp16=False, report_to="none", logging_steps=99999,
        save_strategy="no", eval_strategy="no", disable_tqdm=True,
        auto_dtype_match=True, optim="adamw_torch", dataloader_num_workers=4,
    )
    distiller = HolisticDistiller(
        student_model=student, teacher_model=teacher, alignments=alignments,
        train_dataset=ds, data_collator=DefaultDataCollator(), args=args,
        prepare_teacher_inputs=lambda inp: {"pixel_values": inp["pixel_values"]},
    )
    _reset_gpu(device)
    t0 = time.perf_counter()
    distiller.train()
    torch.cuda.synchronize(device); elapsed = time.perf_counter() - t0; mem = _peak_mb(device)

    if rank == 0:
        ev = _load_imagenet_eval()
        mdl = distiller.model.module if hasattr(distiller.model, "module") else distiller.model
        acc = _eval_accuracy(mdl, DataLoader(ev, 64, collate_fn=DefaultDataCollator()), device)
        _write_result(result_file, {
            "toolkit": f"silverspoon-kd HKD (DDP, {world_size} GPUs)",
            "test_accuracy_pct": round(acc, 2), "wall_clock_sec": round(elapsed, 2),
            "samples_per_sec": round(a["steps"] * a["batch_size"] * world_size / elapsed, 1),
            "peak_gpu_mb": round(mem, 0),
        })
    del teacher, student, distiller; gc.collect(); torch.cuda.empty_cache()
    _cleanup_dist()


# ══════════════════════════════════════════════════════════════════════════════
# Part 2 — NLP: TextBrewer vs silverspoon-kd (BERT DDP)
# ══════════════════════════════════════════════════════════════════════════════


def _load_mnli(tokenizer, max_length=128):
    from datasets import load_dataset
    ds = load_dataset("glue", "mnli", split="train")
    ds = ds.map(lambda ex: tokenizer(ex["premise"], ex["hypothesis"],
                                     truncation=True, max_length=max_length),
                batched=True, remove_columns=["premise", "hypothesis", "idx"])
    ds.set_format("torch")
    return ds


def _bert_t6_student(teacher_model):
    teacher_layers = [1, 3, 5, 7, 9, 11]
    cfg = deepcopy(teacher_model.config); cfg.num_hidden_layers = 6
    student = AutoModelForSequenceClassification.from_config(cfg)
    student.bert.embeddings.load_state_dict(teacher_model.bert.embeddings.state_dict())
    for si, ti in enumerate(teacher_layers):
        student.bert.encoder.layer[si].load_state_dict(
            teacher_model.bert.encoder.layer[ti].state_dict())
    student.bert.pooler.load_state_dict(teacher_model.bert.pooler.state_dict())
    student.classifier.load_state_dict(teacher_model.classifier.state_dict())
    return student


# --- NLP workers ---

def _bert_collate_fn(tokenizer):
    """Collator returning a plain dict (not BatchEncoding) for TextBrewer compatibility.

    TextBrewer's get_outputs_from_batch checks `type(batch) is dict` (not isinstance),
    so BatchEncoding fails. Also strips non-model keys like 'label' that BERT rejects.
    """
    pad = DataCollatorWithPadding(tokenizer)
    model_keys = {"input_ids", "attention_mask", "token_type_ids"}
    def _fn(features):
        batch = pad(features)
        return {k: batch[k] for k in model_keys if k in batch}
    return _fn


def _w_textbrewer_nlp(rank, world_size, port, result_file, a):
    _setup_dist(rank, world_size, port)
    device = torch.device(f"cuda:{rank}")
    from textbrewer import TrainingConfig, DistillationConfig, GeneralDistiller

    teacher = AutoModelForSequenceClassification.from_pretrained(
        str(a["teacher_path"])).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False
    # TextBrewer wraps teacher with DDP; newer PyTorch rejects fully-frozen models.
    # Enable grad on one small param to satisfy DDP (demonstrates the wasteful pattern).
    teacher.classifier.bias.requires_grad_(True)
    student = _bert_t6_student(teacher).to(device)
    tokenizer = AutoTokenizer.from_pretrained(str(a["teacher_path"]))
    ds = _load_mnli(tokenizer, a["max_length"])
    collator = _bert_collate_fn(tokenizer)
    sampler = DistributedSampler(ds, world_size, rank, shuffle=True)
    loader = DataLoader(ds, batch_size=a["batch_size"], sampler=sampler,
                        collate_fn=collator, num_workers=0, pin_memory=True)

    def _adaptor(batch, out):
        return {"logits": out.logits}
    tcfg = TrainingConfig(device=device, local_rank=rank, log_dir=None,
                          output_dir=f"{_TMPDIR}/tb_ddp_{rank}",
                          ckpt_steps=a["steps"] + 1)
    dcfg = DistillationConfig(temperature=a["T"], kd_loss_type="ce",
                              kd_loss_weight=1.0, hard_label_weight=0.0)
    distiller = GeneralDistiller(tcfg, dcfg, teacher, student, _adaptor, _adaptor)
    opt = AdamW(student.parameters(), lr=a["lr"])

    _reset_gpu(device)
    t0 = time.perf_counter()
    distiller.train(optimizer=opt, dataloader=loader, num_steps=a["steps"],
                    scheduler_class=get_linear_schedule_with_warmup,
                    scheduler_args={"num_warmup_steps": int(a["steps"] * a["warmup"]),
                                    "num_training_steps": a["steps"]},
                    max_grad_norm=1.0)
    torch.cuda.synchronize(device); elapsed = time.perf_counter() - t0; mem = _peak_mb(device)

    if rank == 0:
        total = a["steps"] * a["batch_size"] * world_size
        _write_result(result_file, {
            "toolkit": f"TextBrewer (DDP, {world_size} GPUs)",
            "wall_clock_sec": round(elapsed, 2),
            "samples_per_sec": round(total / elapsed, 1),
            "peak_gpu_mb": round(mem, 0),
            "note": "TextBrewer wraps BOTH teacher and student with DDP",
        })
    del teacher, student, distiller; gc.collect(); torch.cuda.empty_cache()
    _cleanup_dist()


def _w_silverspoon_nlp(rank, world_size, port, result_file, a):
    _setup_dist(rank, world_size, port)
    device = torch.device(f"cuda:{rank}")
    from silverspoon_kd import ResponseBasedDistiller, TrainingArguments

    teacher = AutoModelForSequenceClassification.from_pretrained(
        str(a["teacher_path"])).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False
    student = _bert_t6_student(teacher).to(device)
    tokenizer = AutoTokenizer.from_pretrained(str(a["teacher_path"]))
    ds = _load_mnli(tokenizer, a["max_length"])
    collator = DataCollatorWithPadding(tokenizer)

    out = f"{_TMPDIR}/sk_nlp_ddp_{rank}"; os.makedirs(out, exist_ok=True)
    args = TrainingArguments(
        output_dir=out, run_name="sk_nlp_ddp", max_steps=a["steps"],
        per_device_train_batch_size=a["batch_size"], learning_rate=a["lr"],
        lr_scheduler_type="linear", warmup_ratio=a["warmup"], max_grad_norm=1.0,
        bf16=False, fp16=False, report_to="none", logging_steps=99999,
        save_strategy="no", eval_strategy="no", disable_tqdm=True,
        alpha=0.0, auto_dtype_match=True,
        optim="adamw_torch_fused", dataloader_num_workers=0,
    )
    from silverspoon_kd.losses.kl import kl_divergence_loss
    distiller = ResponseBasedDistiller(
        student_model=student, teacher_model=teacher, train_dataset=ds,
        data_collator=collator, args=args,
        soft_loss_fn=kl_divergence_loss(temperature=a["T"], chunk_size=1024),
    )
    _reset_gpu(device)
    t0 = time.perf_counter()
    distiller.train()
    torch.cuda.synchronize(device); elapsed = time.perf_counter() - t0; mem = _peak_mb(device)

    if rank == 0:
        total = a["steps"] * a["batch_size"] * world_size
        _write_result(result_file, {
            "toolkit": f"silverspoon-kd response-based KD (DDP, {world_size} GPUs)",
            "wall_clock_sec": round(elapsed, 2),
            "samples_per_sec": round(total / elapsed, 1),
            "peak_gpu_mb": round(mem, 0),
            "note": "Only student is wrapped with DDP; teacher stays unwrapped",
        })
    del teacher, student, distiller; gc.collect(); torch.cuda.empty_cache()
    _cleanup_dist()


def _w_silverspoon_nlp_hkd(rank, world_size, port, result_file, a):
    """silverspoon-kd HKD: BERT-base→T6 with per-layer MSE + output KL (DDP)."""
    _setup_dist(rank, world_size, port)
    device = torch.device(f"cuda:{rank}")
    from silverspoon_kd import HolisticDistiller, TrainingArguments
    from silverspoon_kd.alignments.alignment import Alignment
    from silverspoon_kd.losses.kl import kl_divergence_loss

    teacher = AutoModelForSequenceClassification.from_pretrained(
        str(a["teacher_path"])).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False
    student = _bert_t6_student(teacher).to(device)
    tokenizer = AutoTokenizer.from_pretrained(str(a["teacher_path"]))
    ds = _load_mnli(tokenizer, a["max_length"])
    collator = DataCollatorWithPadding(tokenizer)

    teacher_layers = [1, 3, 5, 7, 9, 11]
    # Hidden-state MSE alignments
    alignments = [
        Alignment(
            teacher_block=teacher.bert.encoder.layer[t_idx],
            student_block=student.bert.encoder.layer[s_idx],
            teacher_model_name="bert-base",
            student_model_name="bert-t6",
            teacher_module_name=f"bert.encoder.layer.{t_idx}",
            student_module_name=f"bert.encoder.layer.{s_idx}",
            loss_function="mse",
            auto_projector=False,
            loss_weight=1.0,
            auto_device_match=True,
            auto_dtype_match=True,
        )
        for s_idx, t_idx in enumerate(teacher_layers)
    ]
    # Output KL alignment on classifier (matches TextBrewer's kd_loss)
    alignments.append(Alignment(
        teacher_block=teacher.classifier,
        student_block=student.classifier,
        teacher_model_name="bert-base",
        student_model_name="bert-t6",
        teacher_module_name="classifier",
        student_module_name="classifier",
        loss_function=kl_divergence_loss(temperature=a["T"]),
        auto_projector=False,
        loss_weight=1.0,
        auto_device_match=True,
        auto_dtype_match=True,
    ))

    out = f"{_TMPDIR}/sk_nlp_hkd_{rank}"; os.makedirs(out, exist_ok=True)
    args = TrainingArguments(
        output_dir=out, run_name="sk_nlp_hkd", max_steps=a["steps"],
        per_device_train_batch_size=a["batch_size"], learning_rate=a["lr"],
        lr_scheduler_type="linear", warmup_ratio=a["warmup"], max_grad_norm=1.0,
        bf16=False, fp16=False, report_to="none", logging_steps=99999,
        save_strategy="no", eval_strategy="no", disable_tqdm=True,
        auto_dtype_match=True, optim="adamw_torch_fused", dataloader_num_workers=0,
    )
    distiller = HolisticDistiller(
        student_model=student, teacher_model=teacher, alignments=alignments,
        train_dataset=ds, data_collator=collator, args=args,
    )
    _reset_gpu(device)
    t0 = time.perf_counter()
    distiller.train()
    torch.cuda.synchronize(device); elapsed = time.perf_counter() - t0; mem = _peak_mb(device)

    if rank == 0:
        total = a["steps"] * a["batch_size"] * world_size
        _write_result(result_file, {
            "toolkit": f"silverspoon-kd HKD (DDP, {world_size} GPUs)",
            "wall_clock_sec": round(elapsed, 2),
            "samples_per_sec": round(total / elapsed, 1),
            "peak_gpu_mb": round(mem, 0),
        })
    del teacher, student, distiller; gc.collect(); torch.cuda.empty_cache()
    _cleanup_dist()


# ══════════════════════════════════════════════════════════════════════════════
# Part 3 — LLM: DistillKit vs silverspoon-kd (Qwen3 DDP)
# ══════════════════════════════════════════════════════════════════════════════


class TokenizedTextDataset(Dataset):
    column_names = ["input_ids", "attention_mask", "labels"]  # required by TRL SFTTrainer

    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {"input_ids": self.input_ids[idx],
                "attention_mask": self.attention_mask[idx],
                "labels": self.labels[idx]}


def _get_dolma_ids(tokenizer, max_length=1024, max_samples=3000):
    """Get tokenized Dolma data as a list of fixed-length id sequences."""
    import glob as _glob
    # Check for local torch cache first (fastest, written by this benchmark)
    bench_cache = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "datasets",
                               f"dolma_bench_seq{max_length}.pt")
    if os.path.isfile(bench_cache):
        rows = torch.load(bench_cache, weights_only=True).tolist()
        return rows[:max_samples]
    # Check for HF Dataset cache (written by data.py or other scripts)
    from datasets import Dataset as HFDataset
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "datasets")
    paths = sorted(_glob.glob(os.path.join(cache_dir, "dolma_tokenized_*")),
                   key=os.path.getmtime, reverse=True)
    if paths:
        ds = HFDataset.load_from_disk(paths[0])
        if len(ds) > max_samples:
            ds = ds.select(range(max_samples))
        rows = [[x if x is not None else 0 for x in row] for row in ds["input_ids"]]
        # Verify cached data matches requested seq length — skip if mismatched
        if rows and len(rows[0]) != max_length:
            print(f"  _get_dolma_ids: cached data has seq_len={len(rows[0])}, "
                  f"need {max_length} — skipping HF cache")
        else:
            return rows[:max_samples]
    # Stream from HuggingFace and save to local cache
    from datasets import load_dataset
    raw = load_dataset("allenai/dolma3_pool", split="train", streaming=True)
    all_ids, eos = [], tokenizer.eos_token_id
    for s in raw:
        if len(all_ids) // max_length >= max_samples:
            break
        text = s.get("text", "")
        if len(text) > 100:
            all_ids.extend(tokenizer(text, add_special_tokens=False)["input_ids"])
            all_ids.append(eos)
    n = min(len(all_ids) // max_length, max_samples)
    rows = [all_ids[i * max_length:(i + 1) * max_length] for i in range(n)]
    # Save to local cache so spawned processes don't need to re-stream
    os.makedirs(os.path.dirname(bench_cache), exist_ok=True)
    torch.save(torch.tensor(rows, dtype=torch.long), bench_cache)
    return rows


def _load_dolma(tokenizer, max_length=1024, max_samples=3000):
    """Load Dolma as a PyTorch Dataset (for silverspoon-kd)."""
    rows = _get_dolma_ids(tokenizer, max_length, max_samples)
    ids = torch.tensor(rows, dtype=torch.long)
    return TokenizedTextDataset(ids, torch.ones_like(ids), ids.clone())


def _load_dolma_hf(tokenizer, max_length=1024, max_samples=3000):
    """Load Dolma as an HF Dataset (for DistillKit/TRL)."""
    from datasets import Dataset as HFDataset
    rows = _get_dolma_ids(tokenizer, max_length, max_samples)
    return HFDataset.from_dict({
        "input_ids": rows,
        "attention_mask": [[1] * len(r) for r in rows],
        "labels": rows,
    })


# --- LLM workers ---

def _w_distillkit_llm(rank, world_size, port, result_file, a):
    _setup_dist(rank, world_size, port)
    device = torch.device(f"cuda:{rank}")

    # Import DistillKit (must go through main to avoid circular import)
    from distillkit.main import do_distill  # noqa: F401 — triggers correct import order
    from distillkit.trainer import DistillationTrainer
    from distillkit.signals import OnlineSignalSource
    from distillkit.configuration import (
        DistillationRunConfig, DatasetConfiguration, LocalDataset,
        TeacherModelConfig, LossFunctionConfig, LossFunction,
    )
    from trl import SFTConfig

    tokenizer = AutoTokenizer.from_pretrained(a["student"], trust_remote_code=True)
    teacher = AutoModelForCausalLM.from_pretrained(
        a["teacher"], torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    student = AutoModelForCausalLM.from_pretrained(
        a["student"], torch_dtype=torch.bfloat16, trust_remote_code=True)

    ds = _load_dolma_hf(tokenizer, a["max_length"])

    signal = OnlineSignalSource(teacher, vocab_size=tokenizer.vocab_size)

    config = DistillationRunConfig(
        model=a["student"],
        dataset=DatasetConfiguration(train_dataset=LocalDataset(disk_path=f"{_TMPDIR}/dummy")),
        teacher=TeacherModelConfig(path=a["teacher"]),
        sequence_length=a["max_length"],
        output_path=f"{_TMPDIR}/dk_llm_{rank}",
        loss_functions=[
            LossFunctionConfig(function=LossFunction.KL, weight=0.5, temperature=2.0),
            LossFunctionConfig(function=LossFunction.CROSS_ENTROPY, weight=0.5),
        ],
        trust_remote_code=True,
    )
    sft_args = SFTConfig(
        output_dir=f"{_TMPDIR}/dk_llm_{rank}", max_steps=a["steps"],
        per_device_train_batch_size=a["batch_size"], learning_rate=a["lr"],
        lr_scheduler_type="cosine", warmup_ratio=0.1, max_grad_norm=1.0,
        bf16=True, report_to="none", logging_steps=a["steps"],
        save_strategy="no", eval_strategy="no", disable_tqdm=True,
        gradient_checkpointing=False,
        dataset_text_field=None,
    )

    trainer = DistillationTrainer(
        model=student, config=config, signal_source=signal,
        true_vocab_size=tokenizer.vocab_size, args=sft_args,
        train_dataset=ds, processing_class=tokenizer,
    )
    _reset_gpu(device)
    t0 = time.perf_counter()
    result = trainer.train()
    torch.cuda.synchronize(device); elapsed = time.perf_counter() - t0; mem = _peak_mb(device)

    if rank == 0:
        total = a["steps"] * a["batch_size"] * world_size
        loss = result.training_loss if hasattr(result, "training_loss") else 0.0
        _write_result(result_file, {
            "toolkit": f"DistillKit (DDP, {world_size} GPUs)",
            "final_loss": round(loss, 4), "wall_clock_sec": round(elapsed, 2),
            "samples_per_sec": round(total / elapsed, 1),
            "peak_gpu_mb": round(mem, 0),
        })
    del teacher, student, trainer; gc.collect(); torch.cuda.empty_cache()
    _cleanup_dist()


def _w_silverspoon_llm(rank, world_size, port, result_file, a):
    _setup_dist(rank, world_size, port)
    device = torch.device(f"cuda:{rank}")
    from silverspoon_kd import ResponseBasedDistiller, TrainingArguments

    tokenizer = AutoTokenizer.from_pretrained(a["student"], trust_remote_code=True)
    teacher = AutoModelForCausalLM.from_pretrained(
        a["teacher"], torch_dtype=torch.bfloat16, trust_remote_code=True).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    student_cfg = AutoConfig.from_pretrained(a["student"], trust_remote_code=True)
    student = AutoModelForCausalLM.from_config(student_cfg).to(torch.bfloat16).to(device)

    ds = _load_dolma(tokenizer, a["max_length"])
    loss_mode = a.get("loss_mode", "chunked")
    out = f"{_TMPDIR}/sk_llm_ddp_{loss_mode}_{rank}"; os.makedirs(out, exist_ok=True)

    args = TrainingArguments(
        output_dir=out, run_name=f"sk_llm_ddp_{loss_mode}", max_steps=a["steps"],
        per_device_train_batch_size=a["batch_size"], learning_rate=a["lr"],
        lr_scheduler_type="cosine", warmup_ratio=0.1, max_grad_norm=1.0,
        bf16=True, fp16=False, report_to="none", logging_steps=a["steps"],
        save_strategy="no", eval_strategy="no", disable_tqdm=True,
        alpha=0.5, auto_dtype_match=True,
        gradient_checkpointing=False,
    )
    from silverspoon_kd.losses.kl import kl_divergence_loss
    def prep(inp):
        r = {"input_ids": inp["input_ids"]}
        if "attention_mask" in inp: r["attention_mask"] = inp["attention_mask"]
        return r

    distiller_kwargs = {}
    if loss_mode == "liger":
        soft_loss_fn = kl_divergence_loss(temperature=2.0)
        distiller_kwargs["use_liger_kernel"] = True
        distiller_kwargs["output_head_layer"] = "lm_head"
    elif loss_mode == "no_chunk":
        soft_loss_fn = kl_divergence_loss(temperature=2.0, chunk_size=0)
    else:
        soft_loss_fn = kl_divergence_loss(temperature=2.0, chunk_size=256)

    distiller = ResponseBasedDistiller(
        student_model=student, teacher_model=teacher, train_dataset=ds,
        args=args, prepare_teacher_inputs=prep,
        soft_loss_fn=soft_loss_fn, **distiller_kwargs,
    )
    _reset_gpu(device)
    t0 = time.perf_counter()
    result = distiller.train()
    torch.cuda.synchronize(device); elapsed = time.perf_counter() - t0; mem = _peak_mb(device)

    if rank == 0:
        total = a["steps"] * a["batch_size"] * world_size
        loss = result.training_loss if hasattr(result, "training_loss") else 0.0
        mode_labels = {"chunked": "chunked KL", "no_chunk": "no chunking", "liger": "Liger kernel"}
        mode_label = mode_labels.get(loss_mode, loss_mode)
        _write_result(result_file, {
            "toolkit": f"silverspoon-kd ReSKD ({mode_label}, DDP, {world_size} GPUs)",
            "final_loss": round(loss, 4), "wall_clock_sec": round(elapsed, 2),
            "samples_per_sec": round(total / elapsed, 1),
            "peak_gpu_mb": round(mem, 0),
        })
    del teacher, student, distiller; gc.collect(); torch.cuda.empty_cache()
    _cleanup_dist()


# ══════════════════════════════════════════════════════════════════════════════
# Part 4 — Teacher Placement (silverspoon-kd only)
# ══════════════════════════════════════════════════════════════════════════════


def _w_silverspoon_placement(rank, world_size, port, result_file, a):
    _setup_dist(rank, world_size, port)
    device = torch.device(f"cuda:{rank}")
    from silverspoon_kd import ResponseBasedDistiller, TrainingArguments
    mode = a["placement"]

    tokenizer = AutoTokenizer.from_pretrained(a["student"], trust_remote_code=True)

    # Teacher loading depends on placement mode
    if mode == "replicated":
        teacher = AutoModelForCausalLM.from_pretrained(
            a["teacher"], torch_dtype=torch.bfloat16, trust_remote_code=True).to(device).eval()
        placement_arg = None
    elif mode == "sharded":
        teacher = AutoModelForCausalLM.from_pretrained(
            a["teacher"], torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
        placement_arg = "sharded"
    elif mode == "split":
        teacher = AutoModelForCausalLM.from_pretrained(
            a["teacher"], torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
        placement_arg = {"teacher_only_devices": a["teacher_devices"], "strategy": "sharded"}
    else:
        raise ValueError(f"Unknown placement: {mode}")

    for p in teacher.parameters():
        p.requires_grad = False

    student_cfg = AutoConfig.from_pretrained(a["student"], trust_remote_code=True)
    student = AutoModelForCausalLM.from_config(student_cfg).to(torch.bfloat16).to(device)
    ds = _load_dolma(tokenizer, a["max_length"])

    out = f"{_TMPDIR}/sk_place_{mode}_{rank}"; os.makedirs(out, exist_ok=True)
    args = TrainingArguments(
        output_dir=out, run_name=f"sk_{mode}", max_steps=a["steps"],
        per_device_train_batch_size=a["batch_size"], learning_rate=a["lr"],
        lr_scheduler_type="cosine", warmup_ratio=0.1, max_grad_norm=1.0,
        bf16=True, fp16=False, report_to="none", logging_steps=a["steps"],
        save_strategy="no", eval_strategy="no", disable_tqdm=True,
        alpha=0.5, auto_dtype_match=True,
        auto_device_match=True, gradient_checkpointing=False,
        teacher_placement=placement_arg,
    )
    from silverspoon_kd.losses.kl import kl_divergence_loss
    def prep(inp):
        r = {"input_ids": inp["input_ids"]}
        if "attention_mask" in inp: r["attention_mask"] = inp["attention_mask"]
        return r
    distiller = ResponseBasedDistiller(
        student_model=student, teacher_model=teacher, train_dataset=ds,
        args=args, prepare_teacher_inputs=prep,
        soft_loss_fn=kl_divergence_loss(temperature=2.0),
        # Liger's fused KL kernel uses torch.compile, which fails on
        # cross-device tensors (teacher and student on disjoint GPU sets).
        use_liger_kernel=not mode.startswith("split"),
        output_head_layer="lm_head",
    )
    _reset_gpu(device)
    t0 = time.perf_counter()
    result = distiller.train()
    torch.cuda.synchronize(device); elapsed = time.perf_counter() - t0; mem = _peak_mb(device)

    if rank == 0:
        total = a["steps"] * a["batch_size"] * world_size
        loss = result.training_loss if hasattr(result, "training_loss") else 0.0
        label = f"silverspoon-kd ({mode}"
        if mode == "split":
            label += f", teacher GPUs {a['teacher_devices']}"
        label += f", {world_size} ranks)"
        _write_result(result_file, {
            "toolkit": label, "placement": mode,
            "final_loss": round(loss, 4), "wall_clock_sec": round(elapsed, 2),
            "samples_per_sec": round(total / elapsed, 1),
            "peak_gpu_mb": round(mem, 0),
        })
    del teacher, student, distiller; gc.collect(); torch.cuda.empty_cache()
    _cleanup_dist()


# ══════════════════════════════════════════════════════════════════════════════
# Part 5 — OOM Boundary: Max Batch Size & Throughput by Placement
# ══════════════════════════════════════════════════════════════════════════════


def _w_oom_probe(rank, world_size, port, result_file, a):
    """Try training for a few steps; report OOM or success."""
    _setup_dist(rank, world_size, port)
    device = torch.device(f"cuda:{rank}")
    from silverspoon_kd import ResponseBasedDistiller, TrainingArguments
    mode = a["placement"]

    tokenizer = AutoTokenizer.from_pretrained(a["student"], trust_remote_code=True)

    if mode == "replicated":
        teacher = AutoModelForCausalLM.from_pretrained(
            a["teacher"], torch_dtype=torch.bfloat16, trust_remote_code=True).to(device).eval()
        placement_arg = None
    elif mode == "sharded":
        teacher = AutoModelForCausalLM.from_pretrained(
            a["teacher"], torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
        placement_arg = "sharded"
    elif mode == "split":
        teacher = AutoModelForCausalLM.from_pretrained(
            a["teacher"], torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
        placement_arg = {"teacher_only_devices": a["teacher_devices"], "strategy": "sharded"}
    elif mode == "split_tp":
        teacher = AutoModelForCausalLM.from_pretrained(
            a["teacher"], torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
        placement_arg = {"teacher_only_devices": a["teacher_devices"], "strategy": "tp"}
    else:
        raise ValueError(f"Unknown placement: {mode}")

    for p in teacher.parameters():
        p.requires_grad = False

    student_cfg = AutoConfig.from_pretrained(a["student"], trust_remote_code=True)
    student = AutoModelForCausalLM.from_config(student_cfg).to(torch.bfloat16).to(device)
    ds = _load_dolma(tokenizer, a["max_length"])

    out = f"{_TMPDIR}/sk_oom_{mode}_{rank}"; os.makedirs(out, exist_ok=True)
    # Optimizations vs the original config (no methodological change, just
    # using SK's native fast paths that the other toolkits don't expose):
    #   - optim="adamw_torch_fused": fused AdamW kernel (5-10% / step)
    #   - auto_dtype_match=False, auto_device_match=False: skip per-forward
    #     dtype/device introspection (everything is already bf16+CUDA)
    args = TrainingArguments(
        output_dir=out, run_name=f"sk_oom_{mode}", max_steps=a["probe_steps"],
        per_device_train_batch_size=a["batch_size"], learning_rate=1e-3,
        lr_scheduler_type="cosine", warmup_ratio=0.0, max_grad_norm=1.0,
        bf16=True, fp16=False, report_to="none", logging_steps=99999,
        save_strategy="no", eval_strategy="no", disable_tqdm=True,
        optim="adamw_torch_fused",
        alpha=0.5, auto_dtype_match=False,
        auto_device_match=False, gradient_checkpointing=False,
        teacher_placement=placement_arg,
    )
    from silverspoon_kd.losses.kl import kl_divergence_loss
    def prep(inp):
        r = {"input_ids": inp["input_ids"]}
        if "attention_mask" in inp: r["attention_mask"] = inp["attention_mask"]
        return r

    oom = False
    mem = 0
    try:
        distiller = ResponseBasedDistiller(
            student_model=student, teacher_model=teacher, train_dataset=ds,
            args=args, prepare_teacher_inputs=prep,
            soft_loss_fn=kl_divergence_loss(temperature=2.0),
            # Liger's fused KL kernel uses torch.compile, which fails on
            # cross-device tensors (teacher and student on disjoint GPU sets).
            use_liger_kernel=not mode.startswith("split"),
            output_head_layer="lm_head",
        )
        _reset_gpu(device)
        distiller.train()
        torch.cuda.synchronize(device)
        mem = _peak_mb(device)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" in str(e).lower() or isinstance(e, torch.cuda.OutOfMemoryError):
            oom = True
        else:
            raise
    finally:
        gc.collect(); torch.cuda.empty_cache()

    if rank == 0:
        _write_result(result_file, {"oom": oom, "peak_gpu_mb": round(mem, 0),
                                     "batch_size": a["batch_size"]})
    _cleanup_dist()


def _w_throughput_at_bs(rank, world_size, port, result_file, a):
    """Run a throughput measurement at a specific batch size + placement."""
    _setup_dist(rank, world_size, port)
    device = torch.device(f"cuda:{rank}")
    from silverspoon_kd import ResponseBasedDistiller, TrainingArguments
    mode = a["placement"]

    tokenizer = AutoTokenizer.from_pretrained(a["student"], trust_remote_code=True)

    if mode == "replicated":
        teacher = AutoModelForCausalLM.from_pretrained(
            a["teacher"], torch_dtype=torch.bfloat16, trust_remote_code=True).to(device).eval()
        placement_arg = None
    elif mode == "sharded":
        teacher = AutoModelForCausalLM.from_pretrained(
            a["teacher"], torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
        placement_arg = "sharded"
    elif mode == "split":
        teacher = AutoModelForCausalLM.from_pretrained(
            a["teacher"], torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
        placement_arg = {"teacher_only_devices": a["teacher_devices"], "strategy": "sharded"}
    elif mode == "split_tp":
        teacher = AutoModelForCausalLM.from_pretrained(
            a["teacher"], torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
        placement_arg = {"teacher_only_devices": a["teacher_devices"], "strategy": "tp"}
    else:
        raise ValueError(f"Unknown placement: {mode}")

    for p in teacher.parameters():
        p.requires_grad = False

    student_cfg = AutoConfig.from_pretrained(a["student"], trust_remote_code=True)
    student = AutoModelForCausalLM.from_config(student_cfg).to(torch.bfloat16).to(device)
    ds = _load_dolma(tokenizer, a["max_length"])

    out = f"{_TMPDIR}/sk_tp_{mode}_{rank}"; os.makedirs(out, exist_ok=True)
    # Same optimizations as the OOM probe — see comment there.
    args = TrainingArguments(
        output_dir=out, run_name=f"sk_tp_{mode}", max_steps=a["steps"],
        per_device_train_batch_size=a["batch_size"], learning_rate=1e-3,
        lr_scheduler_type="cosine", warmup_ratio=0.1, max_grad_norm=1.0,
        bf16=True, fp16=False, report_to="none", logging_steps=99999,
        save_strategy="no", eval_strategy="no", disable_tqdm=True,
        optim="adamw_torch_fused",
        alpha=0.5, auto_dtype_match=False,
        auto_device_match=False, gradient_checkpointing=False,
        teacher_placement=placement_arg,
    )
    from silverspoon_kd.losses.kl import kl_divergence_loss
    def prep(inp):
        r = {"input_ids": inp["input_ids"]}
        if "attention_mask" in inp: r["attention_mask"] = inp["attention_mask"]
        return r
    distiller = ResponseBasedDistiller(
        student_model=student, teacher_model=teacher, train_dataset=ds,
        args=args, prepare_teacher_inputs=prep,
        soft_loss_fn=kl_divergence_loss(temperature=2.0),
        # Liger's fused KL kernel uses torch.compile, which fails on
        # cross-device tensors (teacher and student on disjoint GPU sets).
        use_liger_kernel=not mode.startswith("split"),
        output_head_layer="lm_head",
    )
    _reset_gpu(device)
    t0 = time.perf_counter()
    distiller.train()
    torch.cuda.synchronize(device); elapsed = time.perf_counter() - t0; mem = _peak_mb(device)

    if rank == 0:
        total = a["steps"] * a["batch_size"] * world_size
        tokens = total * a["max_length"]
        label = f"silverspoon-kd ({mode}, bs={a['batch_size']}, seq={a['max_length']})"
        _write_result(result_file, {
            "toolkit": label, "placement": mode,
            "batch_size": a["batch_size"], "max_length": a["max_length"],
            "wall_clock_sec": round(elapsed, 2),
            "samples_per_sec": round(total / elapsed, 1),
            "tokens_per_sec": round(tokens / elapsed, 0),
            "peak_gpu_mb": round(mem, 0),
        })
    del teacher, student, distiller; gc.collect(); torch.cuda.empty_cache()
    _cleanup_dist()


def _w_distillkit_oom_probe(rank, world_size, port, result_file, a):
    """Try DistillKit training for a few steps; report OOM or success."""
    _setup_dist(rank, world_size, port)
    device = torch.device(f"cuda:{rank}")

    from distillkit.main import do_distill  # noqa: F401
    from distillkit.trainer import DistillationTrainer
    from distillkit.signals import OnlineSignalSource
    from distillkit.configuration import (
        DistillationRunConfig, DatasetConfiguration, LocalDataset,
        TeacherModelConfig, LossFunctionConfig, LossFunction,
    )
    from trl import SFTConfig

    tokenizer = AutoTokenizer.from_pretrained(a["student"], trust_remote_code=True)
    teacher = AutoModelForCausalLM.from_pretrained(
        a["teacher"], torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
    for p in teacher.parameters():
        p.requires_grad = False
    student = AutoModelForCausalLM.from_pretrained(
        a["student"], torch_dtype=torch.bfloat16, trust_remote_code=True)
    ds = _load_dolma_hf(tokenizer, a["max_length"])

    signal = OnlineSignalSource(teacher, vocab_size=tokenizer.vocab_size)
    config = DistillationRunConfig(
        model=a["student"],
        dataset=DatasetConfiguration(train_dataset=LocalDataset(disk_path=f"{_TMPDIR}/dummy")),
        teacher=TeacherModelConfig(path=a["teacher"]),
        sequence_length=a["max_length"],
        output_path=f"{_TMPDIR}/dk_oom_{rank}",
        loss_functions=[
            LossFunctionConfig(function=LossFunction.KL, weight=0.5, temperature=2.0),
            LossFunctionConfig(function=LossFunction.CROSS_ENTROPY, weight=0.5),
        ],
        trust_remote_code=True,
    )
    sft_args = SFTConfig(
        output_dir=f"{_TMPDIR}/dk_oom_{rank}", max_steps=a["probe_steps"],
        per_device_train_batch_size=a["batch_size"], learning_rate=1e-3,
        lr_scheduler_type="cosine", warmup_ratio=0.0, max_grad_norm=1.0,
        bf16=True, report_to="none", logging_steps=99999,
        save_strategy="no", eval_strategy="no", disable_tqdm=True,
        gradient_checkpointing=False,
        dataset_text_field=None,
        max_length=a["max_length"],
    )

    oom = False
    mem = 0
    try:
        trainer = DistillationTrainer(
            model=student, config=config, signal_source=signal,
            true_vocab_size=tokenizer.vocab_size, args=sft_args,
            train_dataset=ds, processing_class=tokenizer,
        )
        _reset_gpu(device)
        trainer.train()
        torch.cuda.synchronize(device)
        mem = _peak_mb(device)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" in str(e).lower() or isinstance(e, torch.cuda.OutOfMemoryError):
            oom = True
        else:
            raise
    finally:
        gc.collect(); torch.cuda.empty_cache()

    if rank == 0:
        _write_result(result_file, {"oom": oom, "peak_gpu_mb": round(mem, 0),
                                     "batch_size": a["batch_size"]})
    _cleanup_dist()


def _w_distillkit_throughput_at_bs(rank, world_size, port, result_file, a):
    """Run DistillKit throughput measurement at a specific batch size."""
    _setup_dist(rank, world_size, port)
    device = torch.device(f"cuda:{rank}")

    from distillkit.main import do_distill  # noqa: F401
    from distillkit.trainer import DistillationTrainer
    from distillkit.signals import OnlineSignalSource
    from distillkit.configuration import (
        DistillationRunConfig, DatasetConfiguration, LocalDataset,
        TeacherModelConfig, LossFunctionConfig, LossFunction,
    )
    from trl import SFTConfig

    tokenizer = AutoTokenizer.from_pretrained(a["student"], trust_remote_code=True)
    teacher = AutoModelForCausalLM.from_pretrained(
        a["teacher"], torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
    for p in teacher.parameters():
        p.requires_grad = False
    student = AutoModelForCausalLM.from_pretrained(
        a["student"], torch_dtype=torch.bfloat16, trust_remote_code=True)
    ds = _load_dolma_hf(tokenizer, a["max_length"])

    signal = OnlineSignalSource(teacher, vocab_size=tokenizer.vocab_size)
    config = DistillationRunConfig(
        model=a["student"],
        dataset=DatasetConfiguration(train_dataset=LocalDataset(disk_path=f"{_TMPDIR}/dummy")),
        teacher=TeacherModelConfig(path=a["teacher"]),
        sequence_length=a["max_length"],
        output_path=f"{_TMPDIR}/dk_tp_{rank}",
        loss_functions=[
            LossFunctionConfig(function=LossFunction.KL, weight=0.5, temperature=2.0),
            LossFunctionConfig(function=LossFunction.CROSS_ENTROPY, weight=0.5),
        ],
        trust_remote_code=True,
    )
    sft_args = SFTConfig(
        output_dir=f"{_TMPDIR}/dk_tp_{rank}", max_steps=a["steps"],
        per_device_train_batch_size=a["batch_size"], learning_rate=1e-3,
        lr_scheduler_type="cosine", warmup_ratio=0.1, max_grad_norm=1.0,
        bf16=True, report_to="none", logging_steps=a["steps"],
        save_strategy="no", eval_strategy="no", disable_tqdm=True,
        gradient_checkpointing=False,
        dataset_text_field=None,
        max_length=a["max_length"],
    )

    trainer = DistillationTrainer(
        model=student, config=config, signal_source=signal,
        true_vocab_size=tokenizer.vocab_size, args=sft_args,
        train_dataset=ds, processing_class=tokenizer,
    )
    _reset_gpu(device)
    t0 = time.perf_counter()
    trainer.train()
    torch.cuda.synchronize(device); elapsed = time.perf_counter() - t0; mem = _peak_mb(device)

    if rank == 0:
        total = a["steps"] * a["batch_size"] * world_size
        tokens = total * a["max_length"]
        label = f"DistillKit (replicated, bs={a['batch_size']}, seq={a['max_length']})"
        _write_result(result_file, {
            "toolkit": label, "placement": "replicated",
            "batch_size": a["batch_size"], "max_length": a["max_length"],
            "wall_clock_sec": round(elapsed, 2),
            "samples_per_sec": round(total / elapsed, 1),
            "tokens_per_sec": round(tokens / elapsed, 0),
            "peak_gpu_mb": round(mem, 0),
        })
    del teacher, student, trainer; gc.collect(); torch.cuda.empty_cache()
    _cleanup_dist()


def _w_textbrewer_oom_probe(rank, world_size, port, result_file, a):
    """Try TextBrewer LLM training for a few steps; report OOM or success."""
    _setup_dist(rank, world_size, port)
    device = torch.device(f"cuda:{rank}")

    from textbrewer import TrainingConfig, DistillationConfig, GeneralDistiller
    from torch.optim import AdamW
    from torch.utils.data import DataLoader, DistributedSampler

    tokenizer = AutoTokenizer.from_pretrained(a["student"], trust_remote_code=True)
    teacher = AutoModelForCausalLM.from_pretrained(
        a["teacher"], torch_dtype=torch.bfloat16, trust_remote_code=True).eval().to(device)
    for p in teacher.parameters():
        p.requires_grad = False
    # TextBrewer wraps both models with DDP; PyTorch rejects fully-frozen models.
    # Enable grad on one small param to satisfy DDP.
    for p in teacher.parameters():
        if p.numel() < 200:
            p.requires_grad_(True)
            break
    student = AutoModelForCausalLM.from_pretrained(
        a["student"], torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)

    ds = _load_dolma(tokenizer, a["max_length"])
    sampler = DistributedSampler(ds, world_size, rank, shuffle=True)
    loader = DataLoader(ds, batch_size=a["batch_size"], sampler=sampler,
                        num_workers=0, pin_memory=True)

    # Adaptor returns 'losses' so TB can mix in hard-label CE loss
    # (matches SK's alpha=0.5 and DistillKit's KL+CE weighting).
    def _adaptor(batch, out):
        return {"logits": (out.logits,), "losses": [out.loss]}

    train_config = TrainingConfig(
        device=device, local_rank=rank, log_dir=None,
        output_dir=f"{_TMPDIR}/tb_oom_{rank}",
        ckpt_steps=a["probe_steps"] + 1)
    # 0.5 KD + 0.5 hard-label CE (matches SK's alpha=0.5 and DK's 0.5/0.5)
    distill_config = DistillationConfig(
        temperature=2, kd_loss_type="ce",
        kd_loss_weight=0.5, hard_label_weight=0.5)

    oom = False
    mem = 0
    try:
        distiller = GeneralDistiller(
            train_config, distill_config, teacher, student, _adaptor, _adaptor)
        optimizer = AdamW(student.parameters(), lr=1e-3)
        _reset_gpu(device)
        distiller.train(optimizer=optimizer, dataloader=loader,
                        num_steps=a["probe_steps"], max_grad_norm=1.0)
        torch.cuda.synchronize(device)
        mem = _peak_mb(device)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" in str(e).lower() or isinstance(e, torch.cuda.OutOfMemoryError):
            oom = True
        else:
            raise
    finally:
        gc.collect(); torch.cuda.empty_cache()

    if rank == 0:
        _write_result(result_file, {"oom": oom, "peak_gpu_mb": round(mem, 0),
                                     "batch_size": a["batch_size"]})
    _cleanup_dist()


def _w_textbrewer_throughput_at_bs(rank, world_size, port, result_file, a):
    """Run TextBrewer LLM throughput measurement at a specific batch size."""
    _setup_dist(rank, world_size, port)
    device = torch.device(f"cuda:{rank}")

    from textbrewer import TrainingConfig, DistillationConfig, GeneralDistiller
    from torch.optim import AdamW
    from torch.utils.data import DataLoader, DistributedSampler

    tokenizer = AutoTokenizer.from_pretrained(a["student"], trust_remote_code=True)
    teacher = AutoModelForCausalLM.from_pretrained(
        a["teacher"], torch_dtype=torch.bfloat16, trust_remote_code=True).eval().to(device)
    for p in teacher.parameters():
        p.requires_grad = False
    for p in teacher.parameters():
        if p.numel() < 200:
            p.requires_grad_(True)
            break
    student = AutoModelForCausalLM.from_pretrained(
        a["student"], torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)

    ds = _load_dolma(tokenizer, a["max_length"])
    sampler = DistributedSampler(ds, world_size, rank, shuffle=True)
    loader = DataLoader(ds, batch_size=a["batch_size"], sampler=sampler,
                        num_workers=0, pin_memory=True)

    # Adaptor returns 'losses' so TB can mix in hard-label CE loss
    # (matches SK's alpha=0.5 and DistillKit's KL+CE weighting).
    def _adaptor(batch, out):
        return {"logits": (out.logits,), "losses": [out.loss]}

    train_config = TrainingConfig(
        device=device, local_rank=rank, log_dir=None,
        output_dir=f"{_TMPDIR}/tb_tp_{rank}",
        ckpt_steps=a["steps"] + 1)
    # 0.5 KD + 0.5 hard-label CE (matches SK's alpha=0.5 and DK's 0.5/0.5)
    distill_config = DistillationConfig(
        temperature=2, kd_loss_type="ce",
        kd_loss_weight=0.5, hard_label_weight=0.5)

    distiller = GeneralDistiller(
        train_config, distill_config, teacher, student, _adaptor, _adaptor)
    optimizer = AdamW(student.parameters(), lr=1e-3)

    _reset_gpu(device)
    t0 = time.perf_counter()
    distiller.train(optimizer=optimizer, dataloader=loader,
                    num_steps=a["steps"], max_grad_norm=1.0)
    torch.cuda.synchronize(device); elapsed = time.perf_counter() - t0; mem = _peak_mb(device)

    if rank == 0:
        total = a["steps"] * a["batch_size"] * world_size
        tokens = total * a["max_length"]
        label = f"TextBrewer (replicated, bs={a['batch_size']}, seq={a['max_length']})"
        _write_result(result_file, {
            "toolkit": label, "placement": "replicated",
            "batch_size": a["batch_size"], "max_length": a["max_length"],
            "wall_clock_sec": round(elapsed, 2),
            "samples_per_sec": round(total / elapsed, 1),
            "tokens_per_sec": round(tokens / elapsed, 0),
            "peak_gpu_mb": round(mem, 0),
        })
    del teacher, student, distiller; gc.collect(); torch.cuda.empty_cache()
    _cleanup_dist()


def _find_max_batch_size_textbrewer(max_length, teacher, student, probe_steps=3,
                                     lo=1, hi=32, visible_gpus=None):
    """Binary search for TextBrewer's largest batch size (replicated only)."""
    world_size = 2
    base = {"max_length": max_length, "teacher": teacher, "student": student,
            "probe_steps": probe_steps}

    r = _spawn_run(_w_textbrewer_oom_probe, world_size, {**base, "batch_size": lo},
                   f"TextBrewer probe bs={lo} seq={max_length}",
                   visible_gpus=visible_gpus)
    if not r or r.get("oom") or "error" in r:
        print(f"  TextBrewer: even batch_size={lo} OOMs for seq={max_length}")
        return 0

    while True:
        r = _spawn_run(_w_textbrewer_oom_probe, world_size, {**base, "batch_size": hi},
                       f"TextBrewer probe bs={hi} seq={max_length}",
                       visible_gpus=visible_gpus)
        if not r or r.get("oom") or "error" in r:
            break
        lo = hi
        hi *= 2
        if hi > 256:
            return lo

    while hi - lo > 1:
        mid = (lo + hi) // 2
        r = _spawn_run(_w_textbrewer_oom_probe, world_size, {**base, "batch_size": mid},
                       f"TextBrewer probe bs={mid} seq={max_length}",
                       visible_gpus=visible_gpus)
        if r and not r.get("oom") and "error" not in r:
            lo = mid
        else:
            hi = mid
    print(f"  TextBrewer max batch size for seq={max_length}: {lo}")
    return lo


def _find_max_batch_size_distillkit(max_length, teacher, student, probe_steps=3,
                                     lo=1, hi=32, visible_gpus=None):
    """Binary search for DistillKit's largest batch size (replicated only)."""
    world_size = 2
    base = {"max_length": max_length, "teacher": teacher, "student": student,
            "probe_steps": probe_steps}

    r = _spawn_run(_w_distillkit_oom_probe, world_size, {**base, "batch_size": lo},
                   f"DistillKit probe bs={lo} seq={max_length}",
                   visible_gpus=visible_gpus)
    if not r or r.get("oom") or "error" in r:
        print(f"  DistillKit: even batch_size={lo} OOMs for seq={max_length}")
        return 0

    while True:
        r = _spawn_run(_w_distillkit_oom_probe, world_size, {**base, "batch_size": hi},
                       f"DistillKit probe bs={hi} seq={max_length}",
                       visible_gpus=visible_gpus)
        if not r or r.get("oom") or "error" in r:
            break
        lo = hi
        hi *= 2
        if hi > 256:
            return lo

    while hi - lo > 1:
        mid = (lo + hi) // 2
        r = _spawn_run(_w_distillkit_oom_probe, world_size, {**base, "batch_size": mid},
                       f"DistillKit probe bs={mid} seq={max_length}",
                       visible_gpus=visible_gpus)
        if r and not r.get("oom") and "error" not in r:
            lo = mid
        else:
            hi = mid
    print(f"  DistillKit max batch size for seq={max_length}: {lo}")
    return lo


def _find_max_batch_size(placement, max_length, teacher, student, probe_steps=3,
                          lo=1, hi=32, teacher_devices=None, visible_gpus=None):
    """Binary search for the largest batch size that fits without OOM."""
    world_size = 2
    base = {"placement": placement, "max_length": max_length,
            "teacher": teacher, "student": student, "probe_steps": probe_steps}
    if placement.startswith("split") and teacher_devices:
        base["teacher_devices"] = teacher_devices

    # Verify lo fits
    r = _spawn_run(_w_oom_probe, world_size, {**base, "batch_size": lo},
                   f"probe {placement} bs={lo} seq={max_length}",
                   visible_gpus=visible_gpus)
    if not r or r.get("oom") or "error" in r:
        print(f"  Even batch_size={lo} OOMs for {placement} seq={max_length}")
        return 0

    # Exponential growth to find an upper bound that OOMs
    while True:
        r = _spawn_run(_w_oom_probe, world_size, {**base, "batch_size": hi},
                       f"probe {placement} bs={hi} seq={max_length}",
                       visible_gpus=visible_gpus)
        if not r or r.get("oom") or "error" in r:
            break  # hi is OOM, proceed to binary search
        lo = hi
        hi *= 2  # keep growing
        if hi > 256:  # safety cap
            return lo

    # Binary search between lo (fits) and hi (OOM)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        r = _spawn_run(_w_oom_probe, world_size, {**base, "batch_size": mid},
                       f"probe {placement} bs={mid} seq={max_length}",
                       visible_gpus=visible_gpus)
        if r and not r.get("oom") and "error" not in r:
            lo = mid
        else:
            hi = mid
    print(f"  Max batch size for {placement} seq={max_length}: {lo}")
    return lo


def _save_partial(args, results, ngpu, section_key="oom_boundary"):
    """Atomically merge results under section_key into the output JSON.
    Lets a preempted job retain whatever measurements completed before kill."""
    out = args.output or str(PROJECT_DIR / "results" / "multi_gpu_comparison.json")
    existing = {}
    if os.path.exists(out):
        try:
            with open(out) as f:
                existing = json.load(f)
        except Exception:
            existing = {}
    existing.pop("config", None)
    # Merge into existing section so a partial run (e.g. one new teacher
    # added to an existing teacher_scaling sweep) doesn't wipe prior entries.
    existing.setdefault(section_key, {}).update(results)
    existing["config"] = {
        "num_gpus": ngpu, "part": f"{section_key}-partial",
        "teacher": args.teacher, "student": args.student,
    }
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(existing, f, indent=2)
    os.replace(tmp, out)
    print(f"  [partial save] {out}")


def run_oom_boundary(args):
    ngpu = torch.cuda.device_count()
    if ngpu < 2:
        print("SKIP: OOM boundary needs 2+ GPUs"); return {}

    seq_lengths = [int(x) for x in args.oom_seq_lengths.split(",")]
    modes = ["replicated", "sharded"]
    if ngpu >= 4:
        modes.append("split")

    # When more GPUs are visible than needed for 2-rank probes, restrict
    # CUDA_VISIBLE_DEVICES to prevent NCCL/FSDP interference from extra
    # devices.  Split probes need all GPUs; everything else uses 2.
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    all_gpus = [int(g) for g in cvd.split(",") if g.strip()] if cvd else list(range(ngpu))
    two_gpus = all_gpus[:2] if len(all_gpus) > 2 else None  # None = no restriction

    # Pre-cache Dolma data for each seq length to avoid HF rate limiting in probes
    tokenizer = AutoTokenizer.from_pretrained(args.student, trust_remote_code=True)
    for sl in seq_lengths:
        print(f"  Pre-caching Dolma tokens for seq_len={sl} ...")
        _get_dolma_ids(tokenizer, max_length=sl)
    del tokenizer

    results = {}
    for seq_len in seq_lengths:
        hi_default = args.oom_max_probe_bs

        # --- DistillKit OOM boundary (replicated only) ---
        print(f"\n--- Finding DistillKit max batch size: seq={seq_len} ---")
        dk_max_bs = _find_max_batch_size_distillkit(
            seq_len, args.teacher, args.student,
            probe_steps=3, lo=1, hi=hi_default, visible_gpus=two_gpus)
        if dk_max_bs == 0:
            results[f"distillkit_seq{seq_len}"] = {
                "toolkit": f"DistillKit (replicated, seq={seq_len})",
                "error": "OOM at batch_size=1"}
        else:
            for bs_attempt in [dk_max_bs, dk_max_bs - 1]:
                if bs_attempt < 1:
                    break
                shared = {"batch_size": bs_attempt, "max_length": seq_len,
                          "steps": args.oom_steps,
                          "teacher": args.teacher, "student": args.student}
                r = _spawn_run(_w_distillkit_throughput_at_bs, 2, shared,
                               f"DistillKit (replicated, bs={bs_attempt}, seq={seq_len})",
                               visible_gpus=two_gpus)
                if r and "error" not in r:
                    break
                print(f"  DistillKit throughput at bs={bs_attempt} failed, retrying bs={bs_attempt - 1}")
            results[f"distillkit_seq{seq_len}"] = r
        _save_partial(args, results, ngpu, "oom_boundary")

        # --- TextBrewer OOM boundary (replicated only) ---
        print(f"\n--- Finding TextBrewer max batch size: seq={seq_len} ---")
        tb_max_bs = _find_max_batch_size_textbrewer(
            seq_len, args.teacher, args.student,
            probe_steps=3, lo=1, hi=hi_default, visible_gpus=two_gpus)
        if tb_max_bs == 0:
            results[f"textbrewer_seq{seq_len}"] = {
                "toolkit": f"TextBrewer (replicated, seq={seq_len})",
                "error": "OOM at batch_size=1"}
        else:
            for bs_attempt in [tb_max_bs, tb_max_bs - 1]:
                if bs_attempt < 1:
                    break
                shared = {"batch_size": bs_attempt, "max_length": seq_len,
                          "steps": args.oom_steps,
                          "teacher": args.teacher, "student": args.student}
                r = _spawn_run(_w_textbrewer_throughput_at_bs, 2, shared,
                               f"TextBrewer (replicated, bs={bs_attempt}, seq={seq_len})",
                               visible_gpus=two_gpus)
                if r and "error" not in r:
                    break
                print(f"  TextBrewer throughput at bs={bs_attempt} failed, retrying bs={bs_attempt - 1}")
            results[f"textbrewer_seq{seq_len}"] = r
        _save_partial(args, results, ngpu, "oom_boundary")

        # --- silverspoon-kd OOM boundary (all placements) ---
        for mode in modes:
            td = [2, 3] if mode == "split" else None
            # Split needs all GPUs; replicated/sharded use only 2
            vg = None if mode == "split" else two_gpus
            print(f"\n--- Finding max batch size: {mode}, seq={seq_len} ---")
            max_bs = _find_max_batch_size(
                mode, seq_len, args.teacher, args.student,
                probe_steps=3, lo=1, hi=hi_default, teacher_devices=td,
                visible_gpus=vg)
            if max_bs == 0:
                results[f"{mode}_seq{seq_len}"] = {
                    "toolkit": f"silverspoon-kd ({mode}, seq={seq_len})",
                    "error": "OOM at batch_size=1"}
                _save_partial(args, results, ngpu, "oom_boundary")
                continue

            # Throughput run at max batch size (retry with max_bs-1 on failure)
            for bs_attempt in [max_bs, max_bs - 1]:
                if bs_attempt < 1:
                    break
                shared = {"placement": mode, "batch_size": bs_attempt,
                          "max_length": seq_len, "steps": args.oom_steps,
                          "teacher": args.teacher, "student": args.student}
                if td:
                    shared["teacher_devices"] = td
                r = _spawn_run(_w_throughput_at_bs, 2, shared,
                               f"silverspoon-kd ({mode}, bs={bs_attempt}, seq={seq_len})",
                               visible_gpus=vg)
                if r and "error" not in r:
                    break
                print(f"  Throughput run at bs={bs_attempt} failed, retrying with bs={bs_attempt - 1}")
            results[f"{mode}_seq{seq_len}"] = r
            _save_partial(args, results, ngpu, "oom_boundary")

    # Print results table
    for seq_len in seq_lengths:
        entries = []
        # Competitors first (DistillKit, TextBrewer)
        for prefix in ["distillkit", "textbrewer"]:
            key = f"{prefix}_seq{seq_len}"
            if key in results:
                entries.append(results[key])
        # Then SK placements
        for mode in modes:
            key = f"{mode}_seq{seq_len}"
            if key in results:
                entries.append(results[key])
        if not entries:
            continue
        print(f"\n{'=' * 95}")
        print(f"  PART 5: OOM BOUNDARY — seq_len={seq_len}")
        print(f"{'=' * 95}")
        hdr = f"{'Toolkit / Placement':<50} {'Max BS':>8} {'Samp/s':>10} {'Tok/s':>12} {'Peak MB':>10}"
        print(hdr)
        print("-" * 95)
        base_sps = None
        for e in entries:
            if "error" in e:
                print(f"{e.get('toolkit', '?'):<50} {'OOM':>8}")
                continue
            bs = e.get("batch_size", "?")
            sps = e.get("samples_per_sec", 0)
            tps = e.get("tokens_per_sec", 0)
            mem = e.get("peak_gpu_mb", 0)
            if base_sps is None:
                base_sps = sps
                gain = "(DK baseline)"
            elif base_sps > 0:
                gain = f"+{(sps - base_sps) / base_sps * 100:.0f}% vs DK"
            else:
                gain = ""
            print(f"{e.get('toolkit', '?'):<50} {bs:>8} {sps:>10.1f} {tps:>12.0f} {mem:>10.0f}  {gain}")

    if results:
        print("\n  NOTE: DistillKit is limited to 'replicated' teacher placement.")
        print("  silverspoon-kd achieves higher throughput by fitting larger batch sizes")
        print("  using memory freed by FSDP sharding or split-GPU teacher placement.")

    return results


def run_teacher_scaling(args):
    """Vary teacher size with student + seq fixed. The configuration where
    SK's FSDP-sharded teacher placement matters — TB/DK can only replicate
    the teacher across GPUs, so they OOM on teachers that exceed per-GPU
    memory once a student + activations are added."""
    ngpu = torch.cuda.device_count()
    if ngpu < 2:
        print("SKIP: teacher scaling needs 2+ GPUs"); return {}

    teachers = [t.strip() for t in args.teacher_scaling_models.split(",") if t.strip()]
    seq_len = args.teacher_scaling_seq
    modes = ["replicated", "sharded"]
    if ngpu >= 4:
        modes.append("split")

    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    all_gpus = [int(g) for g in cvd.split(",") if g.strip()] if cvd else list(range(ngpu))
    two_gpus = all_gpus[:2] if len(all_gpus) > 2 else None

    tokenizer = AutoTokenizer.from_pretrained(args.student, trust_remote_code=True)
    print(f"  Pre-caching Dolma tokens for seq_len={seq_len} ...")
    _get_dolma_ids(tokenizer, max_length=seq_len)
    del tokenizer

    results = {}
    for teacher in teachers:
        tname = teacher.split("/")[-1]
        hi_default = args.oom_max_probe_bs
        label = f"teacher={tname}, seq={seq_len}"

        # --- DistillKit (replicated only) ---
        print(f"\n--- Finding DistillKit max batch size: {label} ---")
        dk_max_bs = _find_max_batch_size_distillkit(
            seq_len, teacher, args.student,
            probe_steps=3, lo=1, hi=hi_default, visible_gpus=two_gpus)
        if dk_max_bs == 0:
            results[f"distillkit_{tname}"] = {
                "toolkit": f"DistillKit (replicated, {label})",
                "error": "OOM at batch_size=1"}
        else:
            for bs_attempt in [dk_max_bs, dk_max_bs - 1]:
                if bs_attempt < 1:
                    break
                shared = {"batch_size": bs_attempt, "max_length": seq_len,
                          "steps": args.oom_steps,
                          "teacher": teacher, "student": args.student}
                r = _spawn_run(_w_distillkit_throughput_at_bs, 2, shared,
                               f"DistillKit (replicated, bs={bs_attempt}, {label})",
                               visible_gpus=two_gpus)
                if r and "error" not in r:
                    break
                print(f"  DistillKit throughput at bs={bs_attempt} failed, retrying bs={bs_attempt - 1}")
            results[f"distillkit_{tname}"] = r
        _save_partial(args, results, ngpu, "teacher_scaling")

        # --- TextBrewer (replicated only) ---
        print(f"\n--- Finding TextBrewer max batch size: {label} ---")
        tb_max_bs = _find_max_batch_size_textbrewer(
            seq_len, teacher, args.student,
            probe_steps=3, lo=1, hi=hi_default, visible_gpus=two_gpus)
        if tb_max_bs == 0:
            results[f"textbrewer_{tname}"] = {
                "toolkit": f"TextBrewer (replicated, {label})",
                "error": "OOM at batch_size=1"}
        else:
            for bs_attempt in [tb_max_bs, tb_max_bs - 1]:
                if bs_attempt < 1:
                    break
                shared = {"batch_size": bs_attempt, "max_length": seq_len,
                          "steps": args.oom_steps,
                          "teacher": teacher, "student": args.student}
                r = _spawn_run(_w_textbrewer_throughput_at_bs, 2, shared,
                               f"TextBrewer (replicated, bs={bs_attempt}, {label})",
                               visible_gpus=two_gpus)
                if r and "error" not in r:
                    break
                print(f"  TextBrewer throughput at bs={bs_attempt} failed, retrying bs={bs_attempt - 1}")
            results[f"textbrewer_{tname}"] = r
        _save_partial(args, results, ngpu, "teacher_scaling")

        # --- silverspoon-kd (all placements) ---
        for mode in modes:
            td = [2, 3] if mode == "split" else None
            vg = None if mode == "split" else two_gpus
            print(f"\n--- Finding max batch size: {mode}, {label} ---")
            max_bs = _find_max_batch_size(
                mode, seq_len, teacher, args.student,
                probe_steps=3, lo=1, hi=hi_default, teacher_devices=td,
                visible_gpus=vg)
            if max_bs == 0:
                results[f"sk_{mode}_{tname}"] = {
                    "toolkit": f"silverspoon-kd ({mode}, {label})",
                    "error": "OOM at batch_size=1"}
                _save_partial(args, results, ngpu, "teacher_scaling")
                continue

            for bs_attempt in [max_bs, max_bs - 1]:
                if bs_attempt < 1:
                    break
                shared = {"placement": mode, "batch_size": bs_attempt,
                          "max_length": seq_len, "steps": args.oom_steps,
                          "teacher": teacher, "student": args.student}
                if td:
                    shared["teacher_devices"] = td
                r = _spawn_run(_w_throughput_at_bs, 2, shared,
                               f"silverspoon-kd ({mode}, bs={bs_attempt}, {label})",
                               visible_gpus=vg)
                if r and "error" not in r:
                    break
                print(f"  Throughput run at bs={bs_attempt} failed, retrying with bs={bs_attempt - 1}")
            results[f"sk_{mode}_{tname}"] = r
            _save_partial(args, results, ngpu, "teacher_scaling")

    # Print per-teacher tables
    for teacher in teachers:
        tname = teacher.split("/")[-1]
        entries = []
        for prefix in ["distillkit", "textbrewer"]:
            key = f"{prefix}_{tname}"
            if key in results:
                entries.append(results[key])
        for mode in modes:
            key = f"sk_{mode}_{tname}"
            if key in results:
                entries.append(results[key])
        if not entries:
            continue
        print(f"\n{'=' * 95}")
        print(f"  TEACHER SCALING — {tname}, seq={seq_len}")
        print(f"{'=' * 95}")
        hdr = f"{'Toolkit / Placement':<50} {'Max BS':>8} {'Samp/s':>10} {'Tok/s':>12} {'Peak MB':>10}"
        print(hdr)
        print("-" * 95)
        for e in entries:
            if "error" in e:
                print(f"{e.get('toolkit', '?'):<50} {'OOM':>8}")
                continue
            bs = e.get("batch_size", "?")
            sps = e.get("samples_per_sec", 0)
            tps = e.get("tokens_per_sec", 0)
            mem = e.get("peak_gpu_mb", 0)
            print(f"{e.get('toolkit', '?'):<50} {bs:>8} {sps:>10.1f} {tps:>12.0f} {mem:>10.0f}")

    return results


def run_split_comparison(args):
    """Compare SK split-FSDP vs split-TP teacher placement across teachers.

    Both modes dedicate GPUs [2, 3] to the teacher and place the student on
    GPUs [0, 1]. The difference is the teacher distribution strategy:
      - split (FSDP full_shard): all-gathers full teacher weights per forward
      - split_tp (tensor parallel): keeps weights split, no per-forward gather

    For an inference-only teacher, TP should win on throughput (no all-gather)
    while FSDP-shard wins on generality (works with any HF model).

    Saves to a new ``split_comparison`` section to avoid overwriting any
    existing oom_boundary or teacher_scaling entries."""
    ngpu = torch.cuda.device_count()
    if ngpu < 4:
        print("SKIP: split-placement comparison needs 4+ GPUs"); return {}

    teachers = [t.strip() for t in args.split_comparison_models.split(",") if t.strip()]
    seq_len = args.split_comparison_seq

    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    all_gpus = [int(g) for g in cvd.split(",") if g.strip()] if cvd else list(range(ngpu))
    teacher_devices = all_gpus[2:4]

    tokenizer = AutoTokenizer.from_pretrained(args.student, trust_remote_code=True)
    print(f"  Pre-caching Dolma tokens for seq_len={seq_len} ...")
    _get_dolma_ids(tokenizer, max_length=seq_len)
    del tokenizer

    results = {}
    for teacher in teachers:
        tname = teacher.split("/")[-1]
        hi_default = args.oom_max_probe_bs
        for mode in ("split", "split_tp"):
            label = f"teacher={tname}, seq={seq_len}, {mode}"
            print(f"\n--- Finding max batch size: {label} ---")
            max_bs = _find_max_batch_size(
                mode, seq_len, teacher, args.student,
                probe_steps=3, lo=1, hi=hi_default, teacher_devices=teacher_devices,
                visible_gpus=None)
            if max_bs == 0:
                results[f"sk_{mode}_{tname}"] = {
                    "toolkit": f"silverspoon-kd ({mode}, {label})",
                    "error": "OOM at batch_size=1"}
                _save_partial(args, results, ngpu, "split_comparison")
                continue
            for bs_attempt in [max_bs, max_bs - 1]:
                if bs_attempt < 1:
                    break
                shared = {"placement": mode, "batch_size": bs_attempt,
                          "max_length": seq_len, "steps": args.oom_steps,
                          "teacher": teacher, "student": args.student,
                          "teacher_devices": teacher_devices}
                r = _spawn_run(_w_throughput_at_bs, 2, shared,
                               f"silverspoon-kd ({mode}, bs={bs_attempt}, {label})",
                               visible_gpus=None)
                if r and "error" not in r:
                    break
                print(f"  Throughput run at bs={bs_attempt} failed, retrying with bs={bs_attempt - 1}")
            results[f"sk_{mode}_{tname}"] = r
            _save_partial(args, results, ngpu, "split_comparison")

    # Print summary
    for teacher in teachers:
        tname = teacher.split("/")[-1]
        print(f"\n{'=' * 95}")
        print(f"  SPLIT COMPARISON — {tname}, seq={seq_len}")
        print(f"{'=' * 95}")
        hdr = f"{'Mode':<18} {'Max BS':>8} {'Samp/s':>10} {'Tok/s':>12} {'Peak MB':>10}"
        print(hdr)
        print("-" * 60)
        for mode in ("split", "split_tp"):
            key = f"sk_{mode}_{tname}"
            e = results.get(key, {})
            if not e or "error" in e:
                print(f"{mode:<18} {'OOM':>8}")
                continue
            bs = e.get("batch_size", "?")
            sps = e.get("samples_per_sec", 0)
            tps = e.get("tokens_per_sec", 0)
            mem = e.get("peak_gpu_mb", 0)
            print(f"{mode:<18} {bs:>8} {sps:>10.1f} {tps:>12.0f} {mem:>10.0f}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Orchestration
# ══════════════════════════════════════════════════════════════════════════════


def _print_table(title, results, cols, key_field="toolkit"):
    """Pretty-print a comparison table."""
    print(f"\n{'=' * 90}")
    print(f"  {title}")
    print(f"{'=' * 90}")
    hdr = f"{'Configuration':<50}"
    for c in cols:
        hdr += f" {c:>12}"
    print(hdr)
    print("-" * (50 + 13 * len(cols)))
    for r in results:
        if "error" in r:
            print(f"{r.get(key_field, '?'):<50}  {'ERROR':>12}")
            continue
        row = f"{r[key_field]:<50}"
        for c in cols:
            v = r.get(c, "—")
            if isinstance(v, float):
                row += f" {v:>12.1f}"
            else:
                row += f" {str(v):>12}"
        print(row)


def run_vision(args):
    ngpu = min(torch.cuda.device_count(), 2)
    if ngpu < 2:
        print("SKIP: Vision DDP needs 2+ GPUs"); return {}
    shared = {"steps": args.vision_steps, "batch_size": args.vision_bs,
              "lr": 2e-4, "T": 2.0, "alpha": 0.5, "warmup": 0.1,
              "num_shards": args.num_shards}
    results = {}
    results["torchdistill"] = _spawn_run(_w_torchdistill_vision, ngpu, shared,
                                         "torchdistill KD (DDP)")
    results["silverspoon"] = _spawn_run(_w_silverspoon_vision, ngpu, shared,
                                        "silverspoon-kd response-based KD (DDP)")
    results["silverspoon_hkd"] = _spawn_run(_w_silverspoon_vision_hkd, ngpu, shared,
                                             "silverspoon-kd HKD (DDP)")
    _print_table(f"PART 1: VISION DDP — ResNet-50→18 / ImageNet ({args.vision_steps} steps)",
                 list(results.values()),
                 ["wall_clock_sec", "samples_per_sec", "peak_gpu_mb", "test_accuracy_pct"])
    return results


def run_nlp(args):
    ngpu = min(torch.cuda.device_count(), 2)
    if ngpu < 2:
        print("SKIP: NLP DDP needs 2+ GPUs"); return {}
    teacher_path = str(TEACHER_BERT_PATH) if TEACHER_BERT_PATH.exists() else TEACHER_BERT_HF
    print(f"  NLP teacher: {teacher_path}")
    shared = {"steps": args.nlp_steps, "batch_size": args.nlp_bs,
              "lr": 1e-4, "T": 4.0, "warmup": 0.1, "max_length": 128,
              "teacher_path": teacher_path}
    results = {}
    results["textbrewer"] = _spawn_run(_w_textbrewer_nlp, ngpu, shared,
                                       "TextBrewer (DDP)")
    results["silverspoon"] = _spawn_run(_w_silverspoon_nlp, ngpu, shared,
                                        "silverspoon-kd response-based KD (DDP)")
    # HKD: BERT layer-aligned MSE (same as single-GPU TextBrewer HKD comparison)
    results["silverspoon_hkd"] = _spawn_run(_w_silverspoon_nlp_hkd, ngpu, shared,
                                             "silverspoon-kd HKD (DDP)")
    _print_table(f"PART 2: NLP DDP — BERT-base→T6 / MNLI ({args.nlp_steps} steps)",
                 list(results.values()),
                 ["wall_clock_sec", "samples_per_sec", "peak_gpu_mb"])

    # Highlight TextBrewer's unnecessary teacher DDP wrapping
    tb = results.get("textbrewer", {})
    sk = results.get("silverspoon", {})
    if "error" not in tb and "error" not in sk and tb.get("peak_gpu_mb") and sk.get("peak_gpu_mb"):
        diff = tb["peak_gpu_mb"] - sk["peak_gpu_mb"]
        if diff > 0:
            print(f"\n  silverspoon-kd uses {diff:.0f} MB less memory per rank —")
            print("  TextBrewer DDP-wraps the teacher (unnecessary grad sync overhead)")
    return results


def run_llm(args):
    ngpu = min(torch.cuda.device_count(), 2)
    if ngpu < 2:
        print("SKIP: LLM DDP needs 2+ GPUs"); return {}
    shared = {"steps": args.llm_steps, "batch_size": args.llm_bs,
              "lr": 1e-3, "max_length": 1024,
              "teacher": args.teacher, "student": args.student}
    results = {}
    results["distillkit"] = _spawn_run(_w_distillkit_llm, ngpu, shared,
                                       "DistillKit (DDP)")
    # Run SK with all 3 loss modes
    for mode, label in [("no_chunk", "no chunking"), ("chunked", "chunked KL"), ("liger", "Liger kernel")]:
        results[f"silverspoon_{mode}"] = _spawn_run(
            _w_silverspoon_llm, ngpu, {**shared, "loss_mode": mode},
            f"silverspoon-kd ReSKD ({label}, DDP)")
    _print_table(f"PART 3: LLM DDP — {args.teacher}→{args.student} ({args.llm_steps} steps)",
                 list(results.values()),
                 ["wall_clock_sec", "samples_per_sec", "peak_gpu_mb", "final_loss"])
    return results


def run_placement(args):
    ngpu = torch.cuda.device_count()
    if ngpu < 2:
        print("SKIP: Placement needs 2+ GPUs"); return {}
    shared = {"steps": args.llm_steps, "batch_size": args.llm_bs,
              "lr": 1e-3, "max_length": 1024,
              "teacher": args.teacher, "student": args.student}
    results = {}

    results["replicated"] = _spawn_run(
        _w_silverspoon_placement, 2, {**shared, "placement": "replicated"},
        "silverspoon-kd (replicated teacher, 2 ranks)")

    results["sharded"] = _spawn_run(
        _w_silverspoon_placement, 2, {**shared, "placement": "sharded"},
        "silverspoon-kd (FSDP-sharded teacher, 2 ranks)")

    if ngpu >= 4:
        results["split"] = _spawn_run(
            _w_silverspoon_placement, 2,
            {**shared, "placement": "split", "teacher_devices": [2, 3]},
            "silverspoon-kd (split-GPU: teacher 2-3, student 0-1)")
    else:
        print(f"\n  Skipping split-GPU (needs 4 GPUs, found {ngpu})")

    _print_table(
        f"PART 4: TEACHER PLACEMENT — {args.teacher}→{args.student} ({args.llm_steps} steps)",
        list(results.values()),
        ["wall_clock_sec", "samples_per_sec", "peak_gpu_mb", "final_loss"])

    # Highlight memory savings
    base = results.get("replicated", {}).get("peak_gpu_mb", 0)
    if base > 0:
        for key in ("sharded", "split"):
            r = results.get(key, {})
            if "error" not in r and r.get("peak_gpu_mb"):
                pct = (base - r["peak_gpu_mb"]) / base * 100
                print(f"\n  {key} placement saves {pct:.1f}% GPU memory vs replicated")

    print("\n  NOTE: torchdistill, TextBrewer, and DistillKit only support")
    print("  replicated teacher placement. FSDP sharding and split-GPU")
    print("  teacher placement are unique to silverspoon-kd.")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════


def main():
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(
        description="Multi-GPU KD benchmark: silverspoon-kd vs torchdistill, TextBrewer, DistillKit")
    parser.add_argument("--part", choices=["vision", "nlp", "llm", "placement", "oom",
                                            "teacher_scaling", "split_comparison", "all"],
                        default="all")
    # Vision
    parser.add_argument("--vision_steps", type=int, default=2000)
    parser.add_argument("--vision_bs", type=int, default=32, help="batch size per GPU")
    parser.add_argument("--num_shards", type=int, default=10, help="ImageNet parquet shards")
    # NLP
    parser.add_argument("--nlp_steps", type=int, default=1000)
    parser.add_argument("--nlp_bs", type=int, default=32)
    # LLM
    parser.add_argument("--llm_steps", type=int, default=500)
    parser.add_argument("--llm_bs", type=int, default=2)
    parser.add_argument("--teacher", default="Qwen/Qwen3-4B")
    parser.add_argument("--student", default="Qwen/Qwen3-0.6B")
    # OOM boundary
    parser.add_argument("--oom_steps", type=int, default=50, help="steps for throughput run")
    parser.add_argument("--oom_seq_lengths", default="1024,2048", help="comma-separated seq lengths")
    parser.add_argument("--oom_max_probe_bs", type=int, default=32, help="initial upper bound for probe")
    # Teacher scaling
    parser.add_argument("--teacher_scaling_models",
                        default="Qwen/Qwen3-4B,Qwen/Qwen3-8B,Qwen/Qwen3-14B",
                        help="comma-separated teacher HF IDs to scale through")
    parser.add_argument("--teacher_scaling_seq", type=int, default=1024,
                        help="fixed seq length for teacher-scaling sweep")
    # Split-placement (TP vs FSDP) comparison — needs 4 GPUs
    parser.add_argument("--split_comparison_models",
                        default="Qwen/Qwen3-4B,Qwen/Qwen3-8B,Qwen/Qwen3-14B",
                        help="comma-separated teacher HF IDs for split TP-vs-FSDP comparison")
    parser.add_argument("--split_comparison_seq", type=int, default=1024,
                        help="fixed seq length for split TP-vs-FSDP comparison")
    # Output
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    ngpu = torch.cuda.device_count()
    print(f"Available GPUs: {ngpu}")
    for i in range(ngpu):
        print(f"  cuda:{i} — {torch.cuda.get_device_name(i)}")

    all_results = {}

    if args.part in ("vision", "all"):
        all_results["vision"] = run_vision(args)
    if args.part in ("nlp", "all"):
        all_results["nlp"] = run_nlp(args)
    if args.part in ("llm", "all"):
        all_results["llm"] = run_llm(args)
    if args.part in ("placement", "all"):
        all_results["placement"] = run_placement(args)
    if args.part in ("oom", "all"):
        all_results["oom_boundary"] = run_oom_boundary(args)
    if args.part == "teacher_scaling":
        all_results["teacher_scaling"] = run_teacher_scaling(args)
    if args.part == "split_comparison":
        all_results["split_comparison"] = run_split_comparison(args)

    # Final summary
    print(f"\n{'=' * 90}")
    print("  OVERALL SUMMARY")
    print(f"{'=' * 90}")
    for section, results in all_results.items():
        if not results:
            continue
        print(f"\n  {section.upper()}:")
        for key, r in results.items():
            if "error" in r:
                print(f"    {key}: FAILED — {r['error'][:80]}")
            else:
                fw = r.get("toolkit", key)
                sps = r.get("samples_per_sec", "?")
                mem = r.get("peak_gpu_mb", "?")
                print(f"    {fw}: {sps} samp/s, {mem} MB")

    # Merge with existing results if running individual parts
    out = args.output or str(PROJECT_DIR / "results" / "multi_gpu_comparison.json")
    if os.path.exists(out) and args.part != "all":
        try:
            with open(out) as f:
                existing = json.load(f)
            existing.pop("config", None)
            # Merge per-section so a partial run (e.g. adding one teacher to
            # the teacher_scaling sweep) doesn't wipe prior entries.
            for k, v in all_results.items():
                if isinstance(v, dict) and isinstance(existing.get(k), dict):
                    existing[k].update(v)
                else:
                    existing[k] = v
            all_results = existing
        except Exception:
            pass

    all_results["config"] = {
        "num_gpus": ngpu, "part": args.part,
        "vision_steps": args.vision_steps, "nlp_steps": args.nlp_steps,
        "llm_steps": args.llm_steps, "teacher": args.teacher, "student": args.student,
    }
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
