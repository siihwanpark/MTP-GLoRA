from typing import Optional

import torch
import torch.distributed as dist
from datasets import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import PreTrainedTokenizer

from .collator import MTPChunkedDataCollator
from .sampler import DistributedLengthGroupedBatchSampler


def prepare_mtp_dataloader(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 1,
    draft_length: int = 4,
    chunk_size: int = 2048,
    min_chunk_size: int = 1024,
    num_workers: int = 4,
    shuffle: Optional[bool] = False,
    process_group: Optional[dist.ProcessGroup] = None,
    pin_memory: Optional[bool] = False,
    group_by_length: bool = True,
    seed: int = 42,
    drop_last: bool = True,
    **dataloader_kwargs
):
    """
    Prepare MTP dataloader with DDP support.

    Args:
        dataset: Dataset
        tokenizer: Tokenizer
        batch_size: Batch size
        draft_length: Draft length
        chunk_size: Chunk size
        min_chunk_size: Minimum chunk size
        num_workers: Number of workers
        shuffle: Shuffle
        process_group: Process group
        pin_memory: Pin memory
        group_by_length: Group by length
        seed: Sampler Seed
        drop_last: Drop last to avoid DDP deadlock
        **dataloader_kwargs: Additional dataloader kwargs

    Returns:
        dataloader: DataLoader
        sampler: DistributedLengthGroupedBatchSampler
    """
    collator = MTPChunkedDataCollator(
        tokenizer=tokenizer,
        draft_length=draft_length,
        chunk_size=chunk_size,
        min_chunk_size=min_chunk_size,
        mask_token_id=tokenizer.convert_tokens_to_ids('<mask>'),
    )

    if process_group is not None and group_by_length:
        lengths = dataset["total_len"]
        if hasattr(lengths, "tolist"):
            lengths = lengths.tolist()
        
        sampler = DistributedLengthGroupedBatchSampler(
            lengths=lengths,
            per_device_batch_size=batch_size,
            process_group=process_group,
            seed=seed,
            group_size_multiplier=50,
            drop_last=drop_last,
        )
        dataloader = DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=collator,
            num_workers=num_workers,
            pin_memory=pin_memory,
            **dataloader_kwargs
        )
        return dataloader, sampler

    # Fallback: original DistributedSampler + batch_size
    if process_group is not None:
        world_size = dist.get_world_size(process_group)
        rank = dist.get_rank(process_group)
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, drop_last=drop_last)
        shuffle = False
    else:
        sampler = None

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=shuffle,
        drop_last=drop_last,
        **dataloader_kwargs
    )
    return dataloader, sampler