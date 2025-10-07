from datetime import timedelta
from contextlib import contextmanager

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

_TP_GROUP = None
_DP_GROUP = None


def get_tp_group():
    global _TP_GROUP
    return _TP_GROUP


def get_dp_group():
    global _DP_GROUP
    return _DP_GROUP


def is_dist_main(pg=None) -> bool:
    if not dist.is_initialized(): return True
    return dist.get_rank(pg) == 0


def barrier(pg=None):
    if dist.is_initialized():
        dist.barrier(pg)


def init_distributed(timeout: int = 10, tp_size: int = 1):
    """Initialize distributed training.

    Args:
        timeout(int): Timeout for collective communication in minutes
        tp_size(int): The degree of tensor parallelism
    """
    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=timeout))
    local_rank = dist.get_rank() % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    print_with_rank(f"Bind to device {local_rank}")

    # initialize sub groups
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    dp_size = world_size // tp_size
    assert world_size == tp_size * dp_size, "world size must be divisible by tp size"
    global _TP_GROUP, _DP_GROUP

    # create tp group
    tp_ranks = [list(range(i * tp_size, (i + 1) * tp_size)) for i in range(dp_size)]
    for ranks in tp_ranks:
        tp_group = dist.new_group(ranks=ranks)
        if rank in ranks:
            _TP_GROUP = tp_group

    # create dp group
    dp_ranks = [list(range(i, world_size, tp_size)) for i in range(tp_size)]
    for ranks in dp_ranks:
        dp_group = dist.new_group(ranks=ranks)
        if rank in ranks:
            _DP_GROUP = dp_group


def destroy_distributed():
    global _TP_GROUP, _DP_GROUP
    dist.destroy_process_group(_TP_GROUP)
    dist.destroy_process_group(_DP_GROUP)
    dist.destroy_process_group()


@contextmanager
def rank0_priority():
    global _DP_GROUP
    if _DP_GROUP is None:
        yield
    else:
        rank = dist.get_rank(_DP_GROUP)
        if rank == 0:
            yield
            dist.barrier(_DP_GROUP)
        else:
            dist.barrier(_DP_GROUP)
            yield


def print_with_rank(message):
    global _DP_GROUP
    if _DP_GROUP is None:
        print(message)
    else:
        print(f"[RANK {dist.get_rank(_DP_GROUP)}] {message}")


def print_on_rank0(message):
    global _DP_GROUP
    if _DP_GROUP is None:
        print(message)
    else:
        if dist.get_rank(_DP_GROUP) == 0:
            print(message)


@contextmanager
def sync_on_last_step(ddp_model, enable_sync: bool):
    """
    Context manager to synchronize on the last step.

    Args:
        ddp_model: DDP wrapped model
        enable_sync: If True, synchronize on the last step, otherwise skip synchronization
    """
    if isinstance(ddp_model, DDP) and not enable_sync:
        with ddp_model.no_sync():
            yield
    else:
        yield