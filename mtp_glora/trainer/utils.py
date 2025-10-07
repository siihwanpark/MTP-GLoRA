# train/trainer_utils.py
from __future__ import annotations

import os
import json
from typing import Tuple

import torch
import torch.distributed as dist
from datasets import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer

from mtp_glora.models import AutoDistributedModelForCausalLM
from mtp_glora.models.layers import apply_lora_to_model

from mtp_glora.utils import init_distributed, get_dp_group, print_on_rank0, is_dist_main, barrier
from mtp_glora.data_utils import (
    build_mtp_dataset, prepare_mtp_dataloader,
    MTP_DATASET_BUILD_VERSION, cache_dir_for, load_cached_dataset, save_dataset_to_cache
)


def setup_distributed_training(args) -> Tuple[bool, int]:
    """
    Initialize DDP if WORLD_SIZE>1. Returns (is_distributed, local_rank).
    """
    requires_dist = "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1
    if requires_dist:
        init_distributed(timeout=args.dist_timeout, tp_size=args.tp_size)
        if args.local_rank >= 0:
            local_rank = args.local_rank
        else:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            args.local_rank = local_rank

        torch.cuda.set_device(local_rank)
        args.dp_size = dist.get_world_size() // args.tp_size
        print_on_rank0(
            f"Initialized: WORLD_SIZE={dist.get_world_size()}, TP={args.tp_size}, DP={args.dp_size}"
        )
        return True, local_rank
    else:
        print("Running in single GPU mode")
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        return False, 0


def load_tokenizer(args, append_mask_token: bool = True):
    tok = AutoTokenizer.from_pretrained(args.model_path)
    if '<mask>' not in tok.get_vocab() and append_mask_token:
        tok.add_special_tokens({'additional_special_tokens': ['<mask>']})
        print_on_rank0(f"Added <mask> token with ID: {tok.convert_tokens_to_ids('<mask>')}")
    return tok


def load_model_and_apply_lora(args, tokenizer, fuse_weights=False):
    device = f"cuda:{args.local_rank}" if args.local_rank >= 0 else ("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoDistributedModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=(
            torch.bfloat16 if args.dtype == "bfloat16"
            else torch.float16 if args.dtype == "float16"
            else torch.float32
        ),
        device=device,
        cache_dir=args.cache_dir if args.cache_dir is not None else os.getenv("HF_HOME", "~/.cache/huggingface/hub"),
        attn_implementation="flex_attention",
    )
    model.resize_token_embeddings(len(tokenizer))
    print_on_rank0("Loaded base model")

    if fuse_weights:
        model.fuse_weights()
        print_on_rank0("QKV / Gate-Up Projections are fused")

    apply_lora_to_model(
        model,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        lora_bias=args.lora_bias,
        use_rslora=args.lora_use_rslora,
        target_modules=args.lora_modules,
    )
    print_on_rank0("Applied GatedLoRA to base model")
    return model


def get_train_dataloader(args, tokenizer: PreTrainedTokenizer):
    # decide cache root (if not exist, create .mtp_cache/ under the train folder)
    default_cache_root = os.path.join(os.path.dirname(os.path.abspath(args.train_data_path)), ".mtp_cache")
    cache_root = getattr(args, "dataset_cache_dir", None) or default_cache_root
    os.makedirs(cache_root, exist_ok=True)

    cache_dir = cache_dir_for(
        train_data_path=args.train_data_path,
        tokenizer=tokenizer,
        draft_length=args.draft_length,
        shuffle_seed=args.shuffle_seed,
        cache_root=cache_root,
    )

    # force rebuild option
    force_rebuild = bool(getattr(args, "dataset_cache_rebuild", False))

    # 1) try cache
    mtp_dataset = None if force_rebuild else load_cached_dataset(cache_dir)

    if mtp_dataset is None:
        # 2) rank0 only build & save
        if is_dist_main():
            print(f"[cache] building MTP dataset → {cache_dir}")
            with open(args.train_data_path, "r") as f:
                data = json.load(f)
            train_data = data.get("results", data)
            ds_raw = Dataset.from_list(train_data)

            # build with existing function
            mtp_dataset = build_mtp_dataset(
                dataset=ds_raw,
                tokenizer=tokenizer,
                draft_length=args.draft_length,
                mask_token_id=tokenizer.convert_tokens_to_ids("<mask>"),
                shuffle_seed=args.shuffle_seed,
                num_proc=args.build_dataset_num_proc,
            )

            # meta information for saving
            meta = {
                "version": MTP_DATASET_BUILD_VERSION,
                "args": {
                    "draft_length": int(args.draft_length),
                    "shuffle_seed": int(args.shuffle_seed),
                },
            }
            save_dataset_to_cache(mtp_dataset, cache_dir, meta)
            print(f"[Dataset Cache] MTP dataset saved at {cache_dir}")
        
        # 3) all ranks synchronize and load
        barrier()

        if not is_dist_main():
            mtp_dataset = load_cached_dataset(cache_dir)
            if mtp_dataset is None:
                raise RuntimeError(f"Cache directory not found after barrier: {cache_dir}")
    else:
        if is_dist_main():
            print(f"[Dataset Cache] MTP dataset loaded from {cache_dir}")
        barrier()

    # torch format is released, so set_format again
    mtp_dataset.set_format(
        type="torch",
        columns=["input_ids", "position_ids", "gate_mask", "regular_token_mask", "total_len"],
    )

    # the rest is the same
    loader, sampler = prepare_mtp_dataloader(
        dataset=mtp_dataset,
        tokenizer=tokenizer,
        batch_size=args.per_device_batch_size,
        draft_length=args.draft_length,
        chunk_size=args.chunk_size,
        min_chunk_size=args.min_chunk_size,
        num_workers=args.num_workers,
        shuffle=args.shuffle,
        process_group=get_dp_group(),
        pin_memory=args.pin_memory,
        group_by_length=args.group_by_length,
        seed=args.shuffle_seed,
        drop_last=args.drop_last,
    )
    return loader, sampler


def accumulate_batches(dataloader, accum_steps: int, sampler=None):
    assert accum_steps >= 1
    it = iter(dataloader)
    while True:
        micro_batches = []
        while len(micro_batches) < accum_steps:
            try:
                micro_batches.append(next(it))
            except StopIteration:
                # epoch end -> next epoch
                if sampler is not None and hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(sampler.epoch + 1)
                it = iter(dataloader)
                # fill again with new epoch
                continue
        yield micro_batches