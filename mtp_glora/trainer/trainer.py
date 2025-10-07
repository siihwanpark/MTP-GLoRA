# train/trainer.py
from __future__ import annotations

import os
import re
import glob
import json
import time
import shutil
from typing import Dict, List, Tuple, Optional, Union

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from safetensors.torch import save_file as safetensors_save_file, load_file as safetensors_load_file

from mtp_glora.data_utils import StreamingKVCacheManager, DistributedLengthGroupedBatchSampler
from mtp_glora.utils import get_dp_group, sync_on_last_step, print_on_rank0, StepStatistics


def _unwrap_model(m: torch.nn.Module) -> torch.nn.Module:
    return m.module if isinstance(m, DDP) else m


class MTPTrainer:
    """
    Minimal HF-Trainer-like wrapper:
      - grad accumulation
      - chunked forward/backward with KV cache
      - 'last valid chunk' sync point for DDP
      - StepStatistics (DP reduce)
      - checkpoint save/load (rank0 IO + barriers)
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        *,
        grad_accum_steps: int,
        draft_length: int,
        is_distributed: bool,
        local_rank: int,
        distributed_length_sampler: Optional[Union[DistributedLengthGroupedBatchSampler, DistributedSampler]] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.distributed_length_sampler = distributed_length_sampler
        self.grad_accum_steps = max(1, grad_accum_steps)
        self.draft_length = draft_length
        self.is_distributed = is_distributed
        self.local_rank = local_rank
        self.device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")

        self.kv_cache_manager = StreamingKVCacheManager()
    

    @staticmethod
    def _last_valid_chunk_index(batch: List[Dict[str, torch.Tensor]]) -> Tuple[int, List[bool]]:
        flags = [bool(c["gate_mask"].clamp(min=0).sum().item()) for c in batch]
        return max(i for i, v in enumerate(flags) if v), flags


    def training_step(self, batch) -> Dict:
        """
        One optimizer step (with grad accumulation). Returns reduced metrics dict.
        Args:
            batch: (1) single batch (list[chunk dict]) or
                   (2) list of micro-batches[list[list[chunk dict]]]
        Returns:
            Dict: reduced metrics dict
        """
        
        device = self.device
        stats = StepStatistics()

        # micro-batch uniformization: [[chunks], [chunks], ...]
        if isinstance(batch, list) and batch and isinstance(batch[0], dict):
            micro_batches = [batch]  # single batch
        elif isinstance(batch, list) and batch and isinstance(batch[0], list):
            micro_batches = batch     # already multiple micro-batches
        else:
            raise TypeError("batch must be a list[chunk] or list[list[chunk]]")

        effective_accum = len(micro_batches)
        loss_scale = 1.0 / float(effective_accum)

        # zero_grad only within this function
        self.optimizer.zero_grad(set_to_none=True)
        
        for mi, micro in enumerate(micro_batches):
            is_last_micro = (mi == effective_accum - 1)

            stats.start_batch()
            self.kv_cache_manager.reset_cache()
            past_key_values = None

            last_valid_idx, valid_flags = self._last_valid_chunk_index(micro)
            chunk_loss_scale = loss_scale / float(last_valid_idx + 1) # loss scale by the number of valid chunks

            for chunk_idx, chunk in enumerate(micro):
                for k, v in chunk.items():
                    if isinstance(v, torch.Tensor):
                        chunk[k] = v.to(device, non_blocking=True)

                # Update chunk with KV cache information
                chunk = self.kv_cache_manager.prepare_data_with_kv_cache(chunk)

                # Sync only on the last valid chunk of the last micro-batch
                enable_sync = is_last_micro and (chunk_idx == last_valid_idx)

                if not valid_flags[chunk_idx]:
                    continue

                with sync_on_last_step(self.model, enable_sync):
                    out = self.model(
                        input_ids=chunk["input_ids"],
                        attention_mask=chunk["attention_mask"],
                        gate_mask=chunk["gate_mask"],
                        regular_token_mask=chunk["regular_token_mask"],
                        position_ids=chunk["position_ids"],
                        past_key_values=past_key_values,
                    )
                    loss = out["loss"] * chunk_loss_scale
                    loss.backward()

                stats.update_from_chunk(out, is_valid=True, loss_scale=chunk_loss_scale)

                if chunk_idx < len(micro) - 1:
                    past_key_values = self.kv_cache_manager.extract_regular_kv_cache_for_next_chunk(chunk, out["past_key_values"])

            stats.end_batch()

        # DP reduce metrics
        stats.reduce_dp_(group=get_dp_group())
        return stats.to_logdict(loss_avg_by="batch")


    def _trainable_state_dict(self) -> Dict[str, torch.Tensor]:
        m = _unwrap_model(self.model)
        trainable = {n for n, p in m.named_parameters() if p.requires_grad}
        full = m.state_dict()
        return {k: v.detach().cpu() for k, v in full.items() if k in trainable}


    @staticmethod
    def _list_checkpoints_with_steps(save_dir: str):
        """Save checkpoints in save_dir and return a list of (step, path, is_dir)."""
        items = []
        for name in os.listdir(save_dir):
            path = os.path.join(save_dir, name)
            # step-123, ckpt-step-00000123, step-123.pt, ckpt-step-00000123.pt all match
            m = re.match(r'^(?:ckpt-)?step-(\d+)(?:\.pt)?$', name)
            if m:
                step = int(m.group(1))
                items.append((step, path, os.path.isdir(path)))
        # sort by step
        items.sort(key=lambda x: x[0])
        return items


    def save_checkpoint(
        self,
        save_dir: str,
        config: Dict,
        step: int,
        *,
        max_to_keep: int = 3,
        include_optimizer: bool = True,
    ) -> str:
        """
        Save checkpoint for MTP model training.

        Args:
            save_dir (str): The directory to save the checkpoint.
            config (Dict): The config of the checkpoint.
            step (int): The global step of the checkpoint.
            max_to_keep (int): The maximum number of checkpoints to keep.
            include_optimizer (bool): Whether to include the optimizer state dict in the checkpoint.
        """
        if not save_dir:
            return

        group = get_dp_group()
        is_main = (not dist.is_initialized()) or (dist.get_rank(group) == 0)

        if is_main:
            os.makedirs(save_dir, exist_ok=True)
        if dist.is_initialized():
            dist.barrier(group)

        ckpt_dir = ""
        if is_main:
            # Create checkpoint directory
            ckpt_dir = os.path.join(save_dir, f"step-{step}")
            os.makedirs(ckpt_dir, exist_ok=True)

            # Save model parameters (safetensors)
            model_path = os.path.join(ckpt_dir, "model.safetensors")
            mstate = self._trainable_state_dict()
            safetensors_save_file(mstate, model_path)

            # Save additional states (optimizer/scheduler/RNG/meta)
            state = {
                "step": int(step),
                "weights_mode": "trainable",
                "rng_state_cpu": torch.get_rng_state(),
                "rng_state_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                "distributed_length_sampler": (self.distributed_length_sampler.state_dict() if self.distributed_length_sampler is not None and hasattr(self.distributed_length_sampler, "state_dict") else None),
            }
            if include_optimizer:
                state["optimizer"] = self.optimizer.state_dict()
                state["scheduler"] = self.scheduler.state_dict()
            torch.save(state, os.path.join(ckpt_dir, "state.pt"))

            # Save config
            with open(os.path.join(ckpt_dir, "config.json"), "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)

            # Update latest.json
            with open(os.path.join(save_dir, "latest.json"), "w") as f:
                json.dump({"latest": os.path.basename(ckpt_dir)}, f)

            # Rotate checkpoints
            if max_to_keep and max_to_keep > 0:
                items = self._list_checkpoints_with_steps(save_dir)
                if len(items) > max_to_keep:
                    to_remove = items[: len(items) - max_to_keep]  # remove oldest checkpoints
                    for _, p, is_dir in to_remove:
                        try:
                            if is_dir:
                                shutil.rmtree(p)
                            else:
                                os.remove(p)
                            print_on_rank0(f"Removed old checkpoint: {p}")
                        except OSError:
                            pass

        if dist.is_initialized():
            dist.barrier(group)
        
        print_on_rank0(f"Checkpoint saved to {ckpt_dir} at step {step}.")


    def load_checkpoint(self, ckpt_dir_or_parent: str) -> int:
        """
        Load checkpoint for MTP model training.

        Args:
            ckpt_dir_or_parent (str): The directory to load the checkpoint from.

        Returns:
            int: The global step of the checkpoint.
        """
        
        if not ckpt_dir_or_parent:
            return 0

        group = get_dp_group()
        if dist.is_initialized():
            dist.barrier(group)

        # Resolve path
        path = ckpt_dir_or_parent
        if os.path.isdir(path):
            # If parent, select latest directory if latest.json does not exist
            model_file = os.path.join(path, "model.safetensors")
            state_file = os.path.join(path, "state.pt")
            if not os.path.exists(model_file):
                latest_json = os.path.join(path, "latest.json")
                if os.path.exists(latest_json):
                    with open(latest_json) as f:
                        j = json.load(f)
                    path = os.path.join(path, j["latest"])
                else:
                    cand = self._list_checkpoints_with_steps(path)
                    if not cand:
                        return 0
                    path = cand[-1][1]

        # Load weights (trainable-only → strict=False)
        model_path = os.path.join(path, "model.safetensors")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"model.safetensors not found in {path}")
        mstate = safetensors_load_file(model_path, device="cpu")
        _unwrap_model(self.model).load_state_dict(mstate, strict=False)

        # Load state (if exists)
        step = 0
        state_path = os.path.join(path, "state.pt")
        if os.path.exists(state_path):
            st = torch.load(state_path, map_location="cpu")
            step = int(st.get("step", 0))
            if "optimizer" in st:
                self.optimizer.load_state_dict(st["optimizer"])
            if "scheduler" in st:
                self.scheduler.load_state_dict(st["scheduler"])
            if "rng_state_cpu" in st:
                torch.set_rng_state(st["rng_state_cpu"])
            if (("sampler" in st) or ("distributed_length_sampler" in st)) and self.distributed_length_sampler is not None and hasattr(self.distributed_length_sampler, "load_state_dict"):
                if "sampler" in st:
                    self.distributed_length_sampler.load_state_dict(st["sampler"])
                if "distributed_length_sampler" in st:
                    self.distributed_length_sampler.load_state_dict(st["distributed_length_sampler"])
            if torch.cuda.is_available() and "rng_state_cuda" in st and st["rng_state_cuda"]:
                try:
                    torch.cuda.set_rng_state_all(st["rng_state_cuda"])
                except Exception:
                    pass

        if dist.is_initialized():
            dist.barrier(group)
        
        print_on_rank0(f"Checkpoint loaded from {path} at step {step}. Starting from step {step+1}.")
        return step+1
