"""Dataset loading: text (Dolma, textbook, etc.), CIFAR-10, and ImageNet."""

import glob
import hashlib
import logging
import os
from typing import Callable, Optional, Union

import torch
from datasets import Dataset as HFDataset
from datasets import load_dataset
from omegaconf import DictConfig
from torch.utils.data import IterableDataset
from torchvision.transforms import (CenterCrop, Compose, Normalize, Pad,
                                     RandomCrop, RandomHorizontalFlip,
                                     RandomResizedCrop, Resize, ToTensor)

logger = logging.getLogger(__name__)
SEED = 42


# ── Streaming dataset for on-the-fly tokenization ────────────────────────────

class StreamingPackedDataset(IterableDataset):
    """Streams raw text from HuggingFace, tokenizes on-the-fly,
    and yields packed fixed-length sequences.  Zero disk usage.

    Uses explicit ``data_files`` URLs so load_dataset resolves in ~2s
    (bypassing the slow 273k-file discovery for large repos like Dolma).
    """

    def __init__(self, data_files, tokenizer, max_length, text_field="text",
                 add_labels=True):
        self.data_files = data_files
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_field = text_field
        self.add_labels = add_labels
        self.eos_id = tokenizer.eos_token_id or tokenizer.sep_token_id or tokenizer.pad_token_id

    def __iter__(self):
        files = self.data_files
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            # Shard files across DataLoader workers (no duplication)
            files = files[worker_info.id::worker_info.num_workers]
        buffer = []
        # Loop forever — training.max_steps controls when to stop.
        # Without this, finite shards exhaust mid-training and the
        # Trainer exits early.
        while True:
            hf_ds = load_dataset("json", data_files=files,
                                 streaming=True, split="train")
            for example in hf_ds:
                ids = self.tokenizer(example[self.text_field],
                                     add_special_tokens=False)["input_ids"]
                buffer.extend(ids)
                buffer.append(self.eos_id)
                while len(buffer) >= self.max_length:
                    chunk = buffer[:self.max_length]
                    buffer = buffer[self.max_length:]
                    sample = {"input_ids": torch.tensor(chunk, dtype=torch.long),
                              "attention_mask": torch.ones(self.max_length, dtype=torch.long)}
                    if self.add_labels:
                        sample["labels"] = torch.tensor(chunk, dtype=torch.long)
                    yield sample


# Dolma subsets known to exist at index 0010+ with 580 shards each (~40k docs/shard)
_DOLMA_SUBSETS = [
    "education_and_jobs", "entertainment", "art_and_design",
    "electronics_and_hardware", "crime_and_law", "fashion_and_beauty",
]


def _build_dolma_shard_urls(n_shards=100, offset=0):
    """Generate explicit Dolma shard URLs.  Spreads across subsets for diversity.
    Each shard ≈ 40k docs ≈ 37M tokens ≈ 36k packed-1024 samples."""
    base = "hf://datasets/allenai/dolma3_pool/data"
    urls = []
    shard_idx = offset
    subset_idx = 10  # known to exist
    while len(urls) < n_shards:
        for subset in _DOLMA_SUBSETS:
            if len(urls) >= n_shards:
                break
            urls.append(
                f"{base}/common_crawl-{subset}-{subset_idx:04d}"
                f"/shard_{shard_idx:08d}.jsonl.zst")
        shard_idx += 1
        if shard_idx >= 580:
            shard_idx = 0
            subset_idx += 1
    return urls


def _build_streaming_eval_set(data_files, tokenizer, max_length, eval_count,
                               text_field="text", add_labels=True):
    """Stream a shard to build a small fixed eval set in memory."""
    hf_ds = load_dataset("json", data_files=data_files,
                         streaming=True, split="train")
    eos_id = tokenizer.eos_token_id or tokenizer.sep_token_id or tokenizer.pad_token_id
    buffer, samples = [], []
    for example in hf_ds:
        ids = tokenizer(example[text_field],
                        add_special_tokens=False)["input_ids"]
        buffer.extend(ids)
        buffer.append(eos_id)
        while len(buffer) >= max_length and len(samples) < eval_count:
            chunk = buffer[:max_length]
            buffer = buffer[max_length:]
            sample = {"input_ids": chunk, "attention_mask": [1] * max_length}
            if add_labels:
                sample["labels"] = chunk
            samples.append(sample)
        if len(samples) >= eval_count:
            break
    return HFDataset.from_list(samples)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _resolve_size(size, total):
    if size == 0 or size == 0.0:
        return 0
    if isinstance(size, float) and 0 < size < 1:
        return int(total * size)
    if isinstance(size, int) and size >= 1:
        return min(size, total)
    return 0


def _tokenizer_cache_id(tokenizer) -> str:
    special = str(sorted(tokenizer.special_tokens_map.items()))
    return f"v{tokenizer.vocab_size}_s{hashlib.md5(special.encode()).hexdigest()[:8]}"


def _split_shuffled(dataset, eval_size, test_size):
    total = len(dataset)
    eval_n = _resolve_size(eval_size, total)
    test_n = _resolve_size(test_size, total)
    train_n = total - eval_n - test_n
    train = dataset.select(range(train_n))
    eval_ = dataset.select(range(train_n, train_n + eval_n))
    test = dataset.select(range(train_n + eval_n, total)) if test_n > 0 else HFDataset.from_list([])
    return train, eval_, test


def _retokenize_from_existing_cache(
    source_cache_path: str,
    source_tokenizer_name: str,
    target_tokenizer,
    max_length: int,
    max_samples: int,
    add_labels: bool,
    cache_path: str,
):
    """Build a tokenized cache for a new tokenizer by decoding an existing cache.

    This avoids re-downloading large datasets when only the tokenizer differs.
    Uses chunked processing and a generator to avoid holding all tokens in memory.
    """
    import tempfile
    from transformers import AutoTokenizer

    src_tok = AutoTokenizer.from_pretrained(source_tokenizer_name, trust_remote_code=True)
    src_ds = HFDataset.load_from_disk(source_cache_path)
    logger.info("Re-tokenizing %d samples from %s → target tokenizer", len(src_ds), source_cache_path)

    batch_size = 500
    eos_id = target_tokenizer.eos_token_id or target_tokenizer.sep_token_id or target_tokenizer.pad_token_id
    src_eos = src_tok.eos_token_id or src_tok.sep_token_id or 0

    # Phase 1: decode + re-tokenize into temporary chunked files to avoid OOM
    chunk_file_size = 100_000  # samples per temp shard
    temp_dir = tempfile.mkdtemp(prefix="retok_")
    residual_ids = []  # leftover tokens from previous batch
    shard_idx = 0
    total_samples = 0
    shard_paths = []

    for start in range(0, len(src_ds), batch_size):
        batch = src_ds[start:start + batch_size]["input_ids"]
        batch = [[src_eos if x is None else x for x in ids] for ids in batch]
        texts = src_tok.batch_decode(batch, skip_special_tokens=True)
        encoded = target_tokenizer(texts, add_special_tokens=False)
        for doc_ids in encoded["input_ids"]:
            residual_ids.extend(doc_ids)
            residual_ids.append(eos_id)

        # Flush complete chunks to a shard when enough accumulate
        n_ready = len(residual_ids) // max_length
        if n_ready >= chunk_file_size:
            n_flush = min(n_ready, chunk_file_size)
            flush_ids = residual_ids[:n_flush * max_length]
            residual_ids = residual_ids[n_flush * max_length:]
            chunks = [flush_ids[i * max_length:(i + 1) * max_length] for i in range(n_flush)]
            data = {"input_ids": chunks, "attention_mask": [[1] * max_length] * n_flush}
            if add_labels:
                data["labels"] = chunks
            shard_path = os.path.join(temp_dir, f"shard_{shard_idx:04d}")
            HFDataset.from_dict(data).save_to_disk(shard_path)
            shard_paths.append(shard_path)
            total_samples += n_flush
            shard_idx += 1
            logger.info("Re-tokenize progress: %d samples written (%d shards)", total_samples, shard_idx)
            del chunks, flush_ids, data

        if max_samples > 0 and total_samples + len(residual_ids) // max_length >= max_samples:
            break

    # Flush remaining residual
    n_remaining = len(residual_ids) // max_length
    if max_samples > 0:
        n_remaining = min(n_remaining, max_samples - total_samples)
    if n_remaining > 0:
        flush_ids = residual_ids[:n_remaining * max_length]
        chunks = [flush_ids[i * max_length:(i + 1) * max_length] for i in range(n_remaining)]
        data = {"input_ids": chunks, "attention_mask": [[1] * max_length] * n_remaining}
        if add_labels:
            data["labels"] = chunks
        shard_path = os.path.join(temp_dir, f"shard_{shard_idx:04d}")
        HFDataset.from_dict(data).save_to_disk(shard_path)
        shard_paths.append(shard_path)
        total_samples += n_remaining
        del chunks, flush_ids, data
    del residual_ids

    # Phase 2: concatenate shards
    from datasets import concatenate_datasets
    shards = [HFDataset.load_from_disk(p) for p in shard_paths]
    tok_ds = concatenate_datasets(shards) if len(shards) > 1 else shards[0]
    logger.info("Saving re-tokenized dataset (%d samples) to %s", len(tok_ds), cache_path)
    tok_ds.save_to_disk(cache_path)

    # Cleanup temp shards
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
    return tok_ds


# Map of cache prefixes to the tokenizer model name used to create them
_KNOWN_TOKENIZER_SOURCES = {
    "Qwen/Qwen3-0.6B": "v151936",   # Qwen vocab size prefix in cache ID
    "Qwen/Qwen3-8B": "v151936",
}


def _find_source_cache(dataset_type: str, cache_dir: str, target_tok_id: str):
    """Find an existing tokenized cache from a different tokenizer for re-tokenization."""
    pattern = os.path.join(cache_dir, f"{dataset_type}_tokenized_*")
    for path in sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True):
        if target_tok_id not in os.path.basename(path):
            # This is a cache from a different tokenizer — usable as source
            return path
    return None


def _format_messages_to_text(example, tokenizer):
    parts = []
    for msg in example["messages"]:
        role, content = msg["role"], msg["content"]
        if role == "user":
            parts.append(f"### User:\n{content}")
        elif role == "assistant":
            parts.append(f"### Assistant:\n{content}")
    return {"text": "\n\n".join(parts) + tokenizer.eos_token}


# ── Dataset registry ─────────────────────────────────────────────────────────

DATASET_REGISTRY = {
    "dolma": {
        "dataset_name": "allenai/dolma3_pool",
        "dataset_id": "dolma3_pool",
        "text_field": "text",
        "split": "train",
        "revision": "refs/convert/parquet",
        "always_stream_source": True,
        "use_overflowing_tokens": True,
        "add_labels": True,
    },
    "dolmino_wiki": {
        "dataset_name": "allenai/dolmino-mix-1124",
        "config_name": "wiki",
        "dataset_id": "dolmino_wiki",
        "text_field": "text",
        "split": "train",
        "always_stream_source": True,
        "use_overflowing_tokens": True,
        "add_labels": True,
    },
    "dolmino_pes2o": {
        "dataset_name": "allenai/dolmino-mix-1124",
        "config_name": "pes2o",
        "dataset_id": "dolmino_pes2o",
        "text_field": "text",
        "split": "train",
        "always_stream_source": True,
        "use_overflowing_tokens": True,
        "add_labels": True,
    },
    "dolmino_flan": {
        "dataset_name": "allenai/dolmino-mix-1124",
        "config_name": "flan",
        "dataset_id": "dolmino_flan",
        "text_field": "text",
        "split": "train",
        "always_stream_source": True,
        "use_overflowing_tokens": True,
        "add_labels": True,
    },
    "dolmino_stackexchange": {
        "dataset_name": "allenai/dolmino-mix-1124",
        "config_name": "stackexchange",
        "dataset_id": "dolmino_stackexchange",
        "text_field": "text",
        "split": "train",
        "always_stream_source": True,
        "use_overflowing_tokens": True,
        "add_labels": True,
    },
    "textbook": {
        "dataset_name": "open-phi/textbooks",
        "dataset_id": "textbook",
        "text_field": "markdown",
        "split": "train",
        "use_overflowing_tokens": True,
        "add_labels": True,
    },
    "tulu3_sft_personas_instruction_following": {
        "dataset_name": "allenai/tulu-3-sft-personas-instruction-following",
        "dataset_id": "tulu3_sft_personas_if",
        "text_field": "messages",
        "format_fn": _format_messages_to_text,
        "split": "train",
        "add_labels": True,
    },
    "tulu3_sft_mixture": {
        "dataset_name": "allenai/tulu-3-sft-mixture",
        "dataset_id": "tulu3_sft_mixture",
        "text_field": "messages",
        "format_fn": _format_messages_to_text,
        "split": "train",
        "add_labels": True,
    },
}


# ── Unified text dataset loader ──────────────────────────────────────────────

def get_text_dataset(
    tokenizer,
    dataset_type: str,
    *,
    max_length: int = 1024,
    safe_char_limit: int = -1,
    eval_size: Union[int, float] = 1000,
    test_size: Union[int, float] = 0.1,
    streaming: bool = False,
    buffer_size: int = 10000,
    map_batch_size: int = 1000,
    max_samples: int = -1,
    cache_dir: str = None,
    dataset_name: str = None,
    config_name: str = None,
    dataset_id: str = None,
    text_field: str = None,
    format_fn: Optional[Callable] = None,
    split: str = None,
    revision: str = None,
    always_stream_source: bool = None,
    use_overflowing_tokens: bool = None,
    add_labels: bool = None,
):
    """Load, tokenize, cache, and split any registered text dataset."""
    # Resolve registry defaults
    entry = DATASET_REGISTRY.get(dataset_type, {})
    dataset_name = dataset_name or entry.get("dataset_name")
    config_name = config_name if config_name is not None else entry.get("config_name")
    dataset_id = dataset_id or entry.get("dataset_id", dataset_type)
    text_field = text_field or entry.get("text_field", "text")
    format_fn = format_fn if format_fn is not None else entry.get("format_fn")
    split = split or entry.get("split", "train")
    revision = revision if revision is not None else entry.get("revision")
    if always_stream_source is None:
        always_stream_source = entry.get("always_stream_source", False)
    if use_overflowing_tokens is None:
        use_overflowing_tokens = entry.get("use_overflowing_tokens", False)
    if add_labels is None:
        add_labels = entry.get("add_labels", False)
    if dataset_name is None:
        raise ValueError(f"Unknown dataset_type '{dataset_type}'. Known: {list(DATASET_REGISTRY)}")

    needs_format = format_fn is not None

    def _format_example(example):
        return format_fn(example, tokenizer)

    def _pre_cut_text(example):
        if safe_char_limit > 0:
            example[text_field] = example[text_field][:safe_char_limit]
        return example

    def _tokenize_streaming(examples):
        text = examples["text"] if needs_format else examples[text_field]
        if safe_char_limit > 0:
            text = [t[:safe_char_limit] for t in text] if isinstance(text, list) else text[:safe_char_limit]
        encoded = tokenizer(text, add_special_tokens=False)
        eos_id = tokenizer.eos_token_id or tokenizer.sep_token_id or tokenizer.pad_token_id
        flat_ids = []
        for doc_ids in encoded["input_ids"]:
            flat_ids.extend(doc_ids)
            flat_ids.append(eos_id)
        n = len(flat_ids) // max_length
        flat_ids = flat_ids[:n * max_length]
        return {"input_ids": [flat_ids[i * max_length:(i + 1) * max_length] for i in range(n)]}

    def _tokenize_batched(examples):
        source = "text" if needs_format else text_field
        encoded = tokenizer(examples[source], add_special_tokens=False)
        eos_id = tokenizer.eos_token_id or tokenizer.sep_token_id or tokenizer.pad_token_id
        flat_ids = []
        for doc_ids in encoded["input_ids"]:
            flat_ids.extend(doc_ids)
            flat_ids.append(eos_id)
        n = len(flat_ids) // max_length
        flat_ids = flat_ids[:n * max_length]
        return {"input_ids": torch.tensor(flat_ids, dtype=torch.long).reshape(n, max_length)}

    # ── Streaming path: pure HF streaming + on-the-fly tokenization ─────────
    # No disk cache.  Uses explicit shard URLs for ~2s startup (vs 200s).
    if streaming:
        if dataset_type == "dolma":
            # 2 shards for eval, rest for training
            eval_urls = _build_dolma_shard_urls(n_shards=2, offset=0)
            train_urls = _build_dolma_shard_urls(n_shards=100, offset=2)
        else:
            raise ValueError(
                f"Streaming not yet supported for dataset_type='{dataset_type}'. "
                f"Only 'dolma' is supported.")

        eval_count = int(eval_size) if isinstance(eval_size, (int, float)) and eval_size >= 1 else 1000
        logger.info("Building eval set (%d samples) from %d shards...",
                    eval_count, len(eval_urls))
        eval_ds = _build_streaming_eval_set(
            eval_urls, tokenizer, max_length, eval_count,
            text_field=text_field, add_labels=add_labels)
        test_ds = HFDataset.from_list([])

        train_ds = StreamingPackedDataset(
            train_urls, tokenizer, max_length,
            text_field=text_field, add_labels=add_labels)
        logger.info("Streaming: %d train shards, %d eval samples",
                    len(train_urls), len(eval_ds))
        return train_ds, eval_ds, test_ds

    # ── Non-streaming path: tokenized cache ───────────────────────────────
    tok_id = _tokenizer_cache_id(tokenizer)
    parts = [dataset_id] + ([config_name] if config_name else [])
    parts += [tok_id, str(max_length), str(safe_char_limit), str(max_samples),
              f"labels={add_labels}", f"overflow={use_overflowing_tokens}", "packed_v5"]
    cache_key = hashlib.md5("_".join(parts).encode()).hexdigest()[:12]
    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "datasets")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{dataset_type}_tokenized_{cache_key}")

    if os.path.exists(cache_path):
        logger.info("Loading cached tokenized dataset from %s", cache_path)
        tok_ds = HFDataset.load_from_disk(cache_path)
        return _split_shuffled(tok_ds.shuffle(seed=SEED), eval_size, test_size)

    # Try re-tokenizing from an existing cache (avoids slow network download)
    source_cache = _find_source_cache(dataset_type, cache_dir, tok_id)
    if source_cache is not None:
        source_tok_name = "Qwen/Qwen3-0.6B"
        tok_ds = _retokenize_from_existing_cache(
            source_cache, source_tok_name, tokenizer,
            max_length, max_samples if max_samples > 0 else -1,
            add_labels, cache_path,
        )
        return _split_shuffled(tok_ds.shuffle(seed=SEED), eval_size, test_size)

    # Load raw dataset
    load_kw = {"split": split}
    if revision is not None:
        load_kw["revision"] = revision
    if always_stream_source:
        load_kw["streaming"] = True
    _load = lambda kw: load_dataset(dataset_name, config_name, **kw) if config_name else load_dataset(dataset_name, **kw)
    try:
        dataset = _load(load_kw)
    except Exception as e:
        if revision and ("revision" in str(e).lower() or "doesn't exist" in str(e).lower()):
            logger.warning("Revision '%s' failed, retrying without revision: %s", revision, e)
            load_kw.pop("revision", None)
            dataset = _load(load_kw)
        else:
            raise

    # Materialize from streaming
    if always_stream_source:
        limit = max_samples if max_samples > 0 else float("inf")
        dataset = HFDataset.from_generator(
            lambda: (s for i, s in enumerate(dataset) if i < limit),
            features=dataset.features,
        )
    elif max_samples > 0:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    if safe_char_limit > 0 and not needs_format:
        dataset = dataset.map(_pre_cut_text)
    if needs_format:
        dataset = dataset.map(_format_example, remove_columns=dataset.column_names)

    tok_ds = dataset.map(_tokenize_batched, batched=True, batch_size=map_batch_size,
                         remove_columns=dataset.column_names)
    logger.info("Saving tokenized dataset to %s", cache_path)
    tok_ds.save_to_disk(cache_path)
    return _split_shuffled(tok_ds.shuffle(seed=SEED), eval_size, test_size)


# ── Image datasets ───────────────────────────────────────────────────────────

def get_train_eval_cifar10_datasets(image_processor, batch_size, num_epochs, num_shards=8):
    """Load CIFAR-10 datasets for training and evaluation."""
    dataset = load_dataset("uoft-cs/cifar10", split="train")
    eval_dataset = load_dataset("uoft-cs/cifar10", split="test")
    num_train = len(dataset)
    dataset = dataset.to_iterable_dataset(num_shards=num_shards)
    eval_dataset = eval_dataset.to_iterable_dataset(num_shards=max(1, num_shards // 4))

    normalize = Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    train_tf = Compose([Pad(4), RandomCrop(32), RandomHorizontalFlip(), ToTensor(), normalize])
    val_tf = Compose([ToTensor(), normalize])

    def preprocess_train(ex):
        ex["pixel_values"] = [train_tf(img.convert("RGB")) for img in ex["img"]]
        ex["labels"] = ex["label"]
        return ex

    def preprocess_val(ex):
        ex["pixel_values"] = [val_tf(img.convert("RGB")) for img in ex["img"]]
        ex["labels"] = ex["label"]
        return ex

    train = dataset.map(preprocess_train, batched=True, remove_columns=["img"])
    eval_ = eval_dataset.map(preprocess_val, batched=True, remove_columns=["img"])
    return train, eval_, num_train // batch_size * num_epochs


def get_train_eval_cifar100_datasets(image_processor, batch_size, num_epochs, num_shards=8):
    """Load CIFAR-100 datasets for training and evaluation.

    Uses the same augmentation as Park et al. (CVPR 2019): random crop 32x32
    from zero-padded 40x40 images, random horizontal flip.
    """
    dataset = load_dataset("uoft-cs/cifar100", split="train")
    eval_dataset = load_dataset("uoft-cs/cifar100", split="test")
    num_train = len(dataset)
    dataset = dataset.to_iterable_dataset(num_shards=num_shards)
    eval_dataset = eval_dataset.to_iterable_dataset(num_shards=max(1, num_shards // 4))

    # CIFAR-100 normalization (per-channel mean/std from training set)
    normalize = Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    train_tf = Compose([Pad(4), RandomCrop(32), RandomHorizontalFlip(), ToTensor(), normalize])
    val_tf = Compose([ToTensor(), normalize])

    def preprocess_train(ex):
        ex["pixel_values"] = [train_tf(img.convert("RGB")) for img in ex["img"]]
        ex["labels"] = ex["fine_label"]
        return ex

    def preprocess_val(ex):
        ex["pixel_values"] = [val_tf(img.convert("RGB")) for img in ex["img"]]
        ex["labels"] = ex["fine_label"]
        return ex

    train = dataset.map(preprocess_train, batched=True, remove_columns=["img", "fine_label", "coarse_label"])
    eval_ = eval_dataset.map(preprocess_val, batched=True, remove_columns=["img", "fine_label", "coarse_label"])
    return train, eval_, num_train // batch_size * num_epochs


def get_train_eval_imagenet1k_subset_datasets(image_processor, batch_size, num_epochs, num_shards=32):
    """Load ImageNet-1k subset datasets for training and evaluation."""
    base = "https://huggingface.co/datasets/ILSVRC/imagenet-1k/resolve/refs/convert/parquet/default"
    dataset = load_dataset("parquet",
                           data_files={"train": [f"{base}/train/{i:04}.parquet" for i in range(100)]},
                           split="train")
    eval_dataset = load_dataset("parquet",
                                data_files={"validation": f"{base}/validation/0000.parquet"},
                                split="validation[:500]")
    num_train = len(dataset)
    dataset = dataset.to_iterable_dataset(num_shards=num_shards)
    eval_dataset = eval_dataset.to_iterable_dataset(num_shards=max(1, num_shards // 4))

    # Resolve image size
    if hasattr(image_processor, "size"):
        size = (image_processor.size.get("shortest_edge") or
                (image_processor.size["height"], image_processor.size["width"]))
    elif hasattr(image_processor, "crop_size"):
        size = (image_processor.crop_size["height"], image_processor.crop_size["width"])
    else:
        size = (224, 224)

    mean = getattr(image_processor, "image_mean", None) or getattr(image_processor, "mean", [0.485, 0.456, 0.406])
    std = getattr(image_processor, "image_std", None) or getattr(image_processor, "std", [0.229, 0.224, 0.225])
    normalize = Normalize(mean=mean, std=std)

    train_tf = Compose([RandomResizedCrop(size), RandomHorizontalFlip(), ToTensor(), normalize])
    val_tf = Compose([Resize(size), CenterCrop(size), ToTensor(), normalize])

    def preprocess_train(ex):
        ex["pixel_values"] = [train_tf(img.convert("RGB")) for img in ex["image"]]
        ex["labels"] = ex["label"]
        return ex

    def preprocess_val(ex):
        ex["pixel_values"] = [val_tf(img.convert("RGB")) for img in ex["image"]]
        ex["labels"] = ex["label"]
        return ex

    train = dataset.map(preprocess_train, batched=True, remove_columns=["image"])
    eval_ = eval_dataset.map(preprocess_val, batched=True, remove_columns=["image"])
    return train, eval_, num_train // batch_size * num_epochs


# ── Downstream task datasets ──────────────────────────────────────────────────

def get_glue_dataset(tokenizer, task_name, max_length=128):
    """Load and tokenize a GLUE dataset (e.g. MNLI)."""
    dataset = load_dataset("glue", task_name)
    if task_name == "mnli":
        train_ds = dataset["train"]
        eval_matched = dataset["validation_matched"]
        eval_mismatched = dataset["validation_mismatched"]

        def tokenize_fn(examples):
            return tokenizer(
                examples["premise"], examples["hypothesis"],
                truncation=True, max_length=max_length, padding=False,
            )

        cols_to_remove = [c for c in train_ds.column_names if c not in ("label",)]
        train_tok = train_ds.map(tokenize_fn, batched=True, remove_columns=cols_to_remove)
        train_tok = train_tok.rename_column("label", "labels")
        eval_m_tok = eval_matched.map(tokenize_fn, batched=True, remove_columns=cols_to_remove)
        eval_m_tok = eval_m_tok.rename_column("label", "labels")
        eval_mm_tok = eval_mismatched.map(tokenize_fn, batched=True, remove_columns=cols_to_remove)
        eval_mm_tok = eval_mm_tok.rename_column("label", "labels")
        # Store mismatched for separate evaluation
        train_tok._eval_mismatched = eval_mm_tok
        return train_tok, eval_m_tok
    else:
        raise ValueError(f"Unsupported GLUE task: {task_name}")


def get_squad_dataset(tokenizer, max_length=384, doc_stride=128):
    """Load and tokenize SQuAD v1.1 for extractive QA."""
    dataset = load_dataset("squad")

    def prepare_train(examples):
        tokenized = tokenizer(
            examples["question"], examples["context"],
            truncation="only_second", max_length=max_length,
            stride=doc_stride, return_overflowing_tokens=True,
            return_offsets_mapping=True, padding=False,
        )
        sample_mapping = tokenized.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized.pop("offset_mapping")

        start_positions = []
        end_positions = []
        for i, offsets in enumerate(offset_mapping):
            sample_idx = sample_mapping[i]
            answers = examples["answers"][sample_idx]
            if len(answers["answer_start"]) == 0:
                start_positions.append(0)
                end_positions.append(0)
                continue
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Find token span
            sequence_ids = tokenized.sequence_ids(i)
            # Find context start/end in token indices
            ctx_start = 0
            while ctx_start < len(sequence_ids) and sequence_ids[ctx_start] != 1:
                ctx_start += 1
            ctx_end = len(sequence_ids) - 1
            while ctx_end >= 0 and sequence_ids[ctx_end] != 1:
                ctx_end -= 1

            if ctx_start > ctx_end or offsets[ctx_start][0] > start_char or offsets[ctx_end][1] < end_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                token_start = ctx_start
                while token_start <= ctx_end and offsets[token_start][0] <= start_char:
                    token_start += 1
                start_positions.append(token_start - 1)
                token_end = ctx_end
                while token_end >= ctx_start and offsets[token_end][1] >= end_char:
                    token_end -= 1
                end_positions.append(token_end + 1)

        tokenized["start_positions"] = start_positions
        tokenized["end_positions"] = end_positions
        return tokenized

    def prepare_validation(examples):
        tokenized = tokenizer(
            examples["question"], examples["context"],
            truncation="only_second", max_length=max_length,
            stride=doc_stride, return_overflowing_tokens=True,
            return_offsets_mapping=True, padding=False,
        )
        sample_mapping = tokenized.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized["offset_mapping"]

        start_positions = []
        end_positions = []
        example_ids = []
        for i, offsets in enumerate(offset_mapping):
            sample_idx = sample_mapping[i]
            example_ids.append(examples["id"][sample_idx])
            answers = examples["answers"][sample_idx]
            if len(answers["answer_start"]) == 0:
                start_positions.append(0)
                end_positions.append(0)
                continue
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])
            sequence_ids = tokenized.sequence_ids(i)
            ctx_start = 0
            while ctx_start < len(sequence_ids) and sequence_ids[ctx_start] != 1:
                ctx_start += 1
            ctx_end = len(sequence_ids) - 1
            while ctx_end >= 0 and sequence_ids[ctx_end] != 1:
                ctx_end -= 1
            if ctx_start > ctx_end or offsets[ctx_start][0] > start_char or offsets[ctx_end][1] < end_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                token_start = ctx_start
                while token_start <= ctx_end and offsets[token_start][0] <= start_char:
                    token_start += 1
                start_positions.append(token_start - 1)
                token_end = ctx_end
                while token_end >= ctx_start and offsets[token_end][1] >= end_char:
                    token_end -= 1
                end_positions.append(token_end + 1)

        tokenized["start_positions"] = start_positions
        tokenized["end_positions"] = end_positions
        tokenized["example_id"] = example_ids
        return tokenized

    cols = dataset["train"].column_names
    train_tok = dataset["train"].map(prepare_train, batched=True, remove_columns=cols)
    eval_tok = dataset["validation"].map(prepare_validation, batched=True, remove_columns=cols)
    # Remove non-tensor columns that would break the data collator during training eval
    # (offset_mapping and example_id are only needed for full post-hoc evaluation in evaluate.py)
    eval_tok = eval_tok.remove_columns(["offset_mapping", "example_id"])
    return train_tok, eval_tok


CONLL_LABEL_LIST = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]


def get_conll_dataset(tokenizer, max_length=128):
    """Load and tokenize CoNLL-2003 for NER (token classification)."""
    dataset = load_dataset("BramVanroy/conll2003")

    def tokenize_and_align_labels(examples):
        tokenized = tokenizer(
            examples["tokens"], is_split_into_words=True,
            truncation=True, max_length=max_length, padding=False,
        )
        all_labels = []
        for i, ner_tags in enumerate(examples["ner_tags"]):
            word_ids = tokenized.word_ids(batch_index=i)
            labels = []
            previous_word_idx = None
            for word_idx in word_ids:
                if word_idx is None:
                    labels.append(-100)
                elif word_idx != previous_word_idx:
                    labels.append(ner_tags[word_idx])
                else:
                    labels.append(-100)  # subword tokens
                previous_word_idx = word_idx
            all_labels.append(labels)
        tokenized["labels"] = all_labels
        return tokenized

    cols = dataset["train"].column_names
    train_tok = dataset["train"].map(tokenize_and_align_labels, batched=True, remove_columns=cols)
    eval_tok = dataset["validation"].map(tokenize_and_align_labels, batched=True, remove_columns=cols)
    return train_tok, eval_tok


# ── Unified loader (called from train.py) ────────────────────────────────────

def load_datasets(cfg: DictConfig, tokenizer_or_processor):
    """Load train and eval datasets based on configuration."""
    data_type = cfg.data.type
    logger.info("Loading dataset: %s", data_type)

    # Downstream task datasets
    if data_type == "glue_mnli":
        return get_glue_dataset(tokenizer_or_processor, "mnli", cfg.data.get("max_length", 128))
    if data_type == "squad_v1":
        return get_squad_dataset(tokenizer_or_processor, cfg.data.get("max_length", 384), cfg.data.get("doc_stride", 128))
    if data_type == "conll2003":
        return get_conll_dataset(tokenizer_or_processor, cfg.data.get("max_length", 128))

    if data_type == "imagenet":
        return get_train_eval_imagenet1k_subset_datasets(
            tokenizer_or_processor, cfg.training.batch_size,
            cfg.training.num_epochs, cfg.data.get("num_shards", 32))

    if data_type == "cifar10":
        return get_train_eval_cifar10_datasets(
            tokenizer_or_processor, cfg.training.batch_size,
            cfg.training.num_epochs, cfg.data.get("num_shards", 8))

    if data_type == "cifar100":
        return get_train_eval_cifar100_datasets(
            tokenizer_or_processor, cfg.training.batch_size,
            cfg.training.num_epochs, cfg.data.get("num_shards", 8))

    # Text datasets
    if data_type not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {data_type}. Known: {['imagenet', 'cifar10'] + list(DATASET_REGISTRY)}")

    streaming = cfg.data.get("streaming", False)
    if streaming and cfg.training.max_steps <= 0:
        raise ValueError("data.streaming=true requires training.max_steps to be set")

    train, eval_, _ = get_text_dataset(
        tokenizer_or_processor, data_type,
        max_length=cfg.data.get("max_length", 1024),
        safe_char_limit=cfg.data.get("safe_char_limit", -1),
        eval_size=cfg.data.get("eval_size", 1000),
        test_size=cfg.data.get("test_size", 0.0),
        streaming=streaming,
        buffer_size=cfg.data.get("buffer_size", 10000),
        map_batch_size=cfg.data.get("map_batch_size", 1000),
        max_samples=cfg.data.get("max_samples", -1),
        cache_dir=cfg.data.get("cache_dir", None),
    )
    return train, eval_
