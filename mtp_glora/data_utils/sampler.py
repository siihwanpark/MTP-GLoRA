from __future__ import annotations
from typing import Iterator, List, Dict

import torch
import torch.distributed as dist
from torch.utils.data import Sampler


def _get_world_and_rank(process_group=None):
    if dist.is_initialized():
        return dist.get_world_size(process_group), dist.get_rank(process_group)
    else:
        return 1, 0


class DistributedLengthGroupedBatchSampler(Sampler[List[int]]):
    """
    Length-grouped, globally-batched, DDP-aware batch sampler with deterministic resume.
        - Length-based sortish + random window shuffle
        - Global batch (= dp_size * per_device_batch_size) first, then split per rank
        Split per rank by per_device_batch size
        - set_epoch/ state_dict/ load_state_dict supported (exact resume)
    """

    def __init__(
        self,
        lengths: List[int],
        per_device_batch_size: int,
        process_group=None,
        *,
        seed: int = 42,
        group_size_multiplier: int = 50,  # one window size = global_batch_size * this value
        drop_last: bool = True,
    ):
        self.lengths = lengths
        self.process_group = process_group
        self.world_size, self.rank = _get_world_and_rank(process_group)
        self.per_device_batch_size = int(per_device_batch_size)
        self.global_batch_size = self.world_size * self.per_device_batch_size

        self.seed = int(seed)
        self.group_size_multiplier = int(group_size_multiplier)
        self.drop_last = bool(drop_last)

        self.epoch = 0
        self.cursor = 0  # how many global batches have been consumed in this epoch

        self._n = len(lengths)
        assert self.global_batch_size > 0


    def state_dict(self) -> Dict[str, int]:
        return {"epoch": self.epoch, "cursor": self.cursor}


    def load_state_dict(self, state: Dict[str, int]):
        self.epoch = int(state.get("epoch", 0))
        self.cursor = int(state.get("cursor", 0))


    def _generator_for_epoch(self) -> torch.Generator:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        return g


    def _build_epoch_order(self) -> List[int]:
        # Shuffle all samples (epoch seed), then sort by length in window units (sortish)
        g = self._generator_for_epoch()
        perm = torch.randperm(self._n, generator=g).tolist()

        window = self.global_batch_size * self.group_size_multiplier
        if window <= 0:
            window = self.global_batch_size * 50

        ordered: List[int] = []
        for i in range(0, self._n, window):
            block = perm[i:i+window]
            block.sort(key=lambda idx: self.lengths[idx], reverse=True)  # sort by length to roughly group long ones
            ordered.extend(block)
        return ordered


    def _as_global_batches(self, ordered: List[int]) -> List[List[int]]:
        # Slice by consecutive global_batch_size
        n_full = len(ordered) // self.global_batch_size
        usable = n_full * self.global_batch_size
        sliced = ordered[:usable]
        batches = [sliced[i:i+self.global_batch_size] for i in range(0, usable, self.global_batch_size)]
        return batches


    def __iter__(self) -> Iterator[List[int]]:
        ordered = self._build_epoch_order()
        global_batches = self._as_global_batches(ordered)

        # continue from cursor (resume)
        for gb_idx in range(self.cursor, len(global_batches)):
            g = global_batches[gb_idx]
            # This rank's per-device shard
            start = self.rank * self.per_device_batch_size
            end = start + self.per_device_batch_size
            shard = g[start:end]
            if len(shard) < self.per_device_batch_size and self.drop_last:
                # Drop_last, exclude the last incomplete batch
                continue
            # immediately update cursor for next resume (checkpoint can be in the middle of a batch)
            self.cursor = gb_idx + 1
            yield shard

        self.epoch += 1
        self.cursor = 0


    def __len__(self) -> int:
        # Based on the number of global batches available in this epoch (assuming drop_last)
        n_full = self._n // self.global_batch_size
        return n_full
