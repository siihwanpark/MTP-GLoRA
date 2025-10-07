from __future__ import annotations

import os, re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List, Literal, Tuple
from types import SimpleNamespace
from transformers import HfArgumentParser


@dataclass
class ModelArguments:
    model_path: str
    cache_dir: Optional[str] = None
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    tp_size: int = 1
    fuse_weights: bool = False


@dataclass
class DataArguments:
    train_data_path: str
    eval_data_path: Optional[str] = None
    shuffle_seed: int = 42
    build_dataset_num_proc: int = 4
    num_workers: int = 4
    dataset_cache_dir: Optional[str] = None
    dataset_cache_rebuild: bool = False
    shuffle: bool = True
    pin_memory: bool = True
    group_by_length: bool = True
    drop_last: bool = True


@dataclass
class MTPArguments:
    draft_length: int = 4
    chunk_size: int = 2048
    min_chunk_size: int = 1024


@dataclass
class LoRAArguments:
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_bias: bool = False
    lora_use_rslora: bool = False
    lora_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    ])


@dataclass
class TrainingArguments:
    max_steps: int = 50_000
    warmup_steps: int = 5_000
    per_device_batch_size: int = 1
    lr: float = 2e-4
    grad_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    logging_steps: int = 1
    save_steps: int = 1_000
    save_limit: int = 3
    output_dir: str = "results"
    save_dir: Optional[str] = None
    resume: bool = False
    checkpoint_dir: Optional[str] = None


@dataclass
class ReportingArguments:
    report_to: Literal["wandb", "tensorboard", "none"] = "none"
    tensorboard_log_dir: Optional[str] = None
    wandb_project: Optional[str] = None
    wandb_name: Optional[str] = None
    wandb_key: Optional[str] = None


@dataclass
class DistributedArguments:
    seed: int = 0
    verbose: bool = False
    dist_timeout: int = 20
    local_rank: int = -1


def _merge_to_namespace(*objs) -> SimpleNamespace:
    merged = {}
    for o in objs:
        merged.update(vars(o))
    return SimpleNamespace(**merged)


def parse_args() -> Tuple[
    ModelArguments, DataArguments, MTPArguments, LoRAArguments,
    TrainingArguments, ReportingArguments, DistributedArguments, SimpleNamespace
]:
    parser = HfArgumentParser((
        ModelArguments,
        DataArguments,
        MTPArguments,
        LoRAArguments,
        TrainingArguments,
        ReportingArguments,
        DistributedArguments,
    ))
    (margs, dargs, mtpargs, loraargs, targs, rargs, distargs) = parser.parse_args_into_dataclasses()

    merged = _merge_to_namespace(margs, dargs, mtpargs, loraargs, targs, rargs, distargs)
    return merged


def make_run_name(args, include_timestamp: bool = False) -> str:
    """
    concise experiment name:
      <model>-<data>-bsz<global>-lr<lr>-MTP<draft>-lora<rank>-SoftSCE-seed<seed>
    """

    def _basename_wo_ext(path: str) -> str:
        base = os.path.basename(str(path).rstrip("/"))
        name, _ = os.path.splitext(base)
        return name or base

    def _short_model_name(model_path: str) -> str:
        base = str(model_path).split("/")[-1]
        return re.sub(r"[^A-Za-z0-9._-]+", "", base) or "model"

    def _fmt_lr(x: float) -> str:
        return f"{x:.0e}" if (x != 0 and (x < 1e-3 or x >= 1e3)) else f"{x:g}"

    def _sanitize(s: str) -> str:
        s = re.sub(r"\s+", "", s)
        s = re.sub(r"[^A-Za-z0-9._-]", "-", s)
        s = re.sub(r"-{2,}", "-", s).strip("-")
        return s

    def _world_dp(tp_size: int) -> int:
        ws = int(os.environ.get("WORLD_SIZE", "1") or 1)
        tp = max(1, int(tp_size or 1))
        return max(1, ws // tp)

    model = _short_model_name(getattr(args, "model_path", "model"))
    data  = _basename_wo_ext(getattr(args, "train_data_path", "data"))

    tp    = int(getattr(args, "tp_size", 1))
    dp    = _world_dp(tp)
    pbs   = int(getattr(args, "per_device_batch_size", 1))
    ga    = int(getattr(args, "grad_accumulation_steps", 1))
    gbs   = max(1, dp * pbs * ga)

    lr    = _fmt_lr(float(getattr(args, "lr", 2e-4)))
    draft = int(getattr(args, "draft_length", 4))
    lora  = int(getattr(args, "lora_rank", 16))
    fuse  = int(getattr(args, "fuse_weights", False))
    seed  = int(getattr(args, "seed", 0))

    name = f"{model}-{data}-bsz{gbs}-lr{lr}-MTP{draft}-lora{lora}-SoftSCE"
    if fuse:
        name += "-fused"
    if include_timestamp:
        name += "-" + datetime.now().strftime("%m%d-%H%M")
    name += f"-seed{seed}"
    return _sanitize(name)