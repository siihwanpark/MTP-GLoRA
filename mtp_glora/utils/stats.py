import re
from typing import Optional, Dict, List
from dataclasses import dataclass

import torch
import torch.distributed as dist

@dataclass
class StepStatistics:
    ce_sum: float = 0.0
    chunks_used: int = 0

    batches_used: int = 0
    batches_for_acc: int = 0

    batch_acc_sum: Optional[torch.Tensor] = None

    _batch_correct: Optional[torch.Tensor] = None
    _batch_num_tokens: Optional[torch.Tensor] = None
    _batch_had_valid_chunk: bool = False

    def start_batch(self):
        """call at the start of a micro-batch."""
        self._batch_correct = None
        self._batch_num_tokens = None
        self._batch_had_valid_chunk = False

    def update_from_chunk(self, out: Dict, *, is_valid: bool, loss_scale: float = 1.0):
        """
        call after processing one chunk. if is_valid=False (e.g. gate_mask=0), do not update statistics.
        loss_scale is usually 1/grad_accum_steps.
        """
        if not is_valid:
            return

        # Loss(chunk-wise) accumulation
        # out[...] is a tensor, .item() to scalar (float accumulation)
        self.ce_sum += float(out["loss"].item()) * loss_scale
        self.chunks_used += 1

        # Temporary accumulation: numerator/denominator
        def _acc(dst: Optional[torch.Tensor], src: torch.Tensor) -> torch.Tensor:
            src = src.detach()
            return src if dst is None else (dst + src)

        self._batch_correct = _acc(self._batch_correct, out["correct"])
        self._batch_num_tokens = _acc(self._batch_num_tokens, out["num_regular_tokens"])

        self._batch_had_valid_chunk = True

    def end_batch(self):
        """
        call at the end of a micro-batch.
        - Loss denominator: batches_used is incremented by 1 for each batch
        - acc is only included in the average for batches with valid tokens (batches_for_acc)
        """
        self.batches_used += 1

        if self._batch_had_valid_chunk and (self._batch_num_tokens is not None):
            eps = 1e-8
            # Self._batch_base_correct: [bsz, draft_length]
            # self._batch_num_tokens  : scalar tensor (or [bsz], see below)
            # batch dimension(bsz) average -> [draft_length]
            acc_pct = (self._batch_correct / (self._batch_num_tokens + eps)).mean(dim=0) * 100.0

            if self.batch_acc_sum is None:
                self.batch_acc_sum = acc_pct.detach()
            else:
                self.batch_acc_sum += acc_pct.detach()

            self.batches_for_acc += 1

    # Utility for step-wise/rank-wise aggregation

    def merge_(self, other: "StepStatistics"):
        """use when merging statistics from multiple parts within the same step (if needed)."""
        self.ce_sum += other.ce_sum
        self.chunks_used += other.chunks_used
        self.batches_used += other.batches_used
        self.batches_for_acc += other.batches_for_acc

        def _acc(dst: Optional[torch.Tensor], src: Optional[torch.Tensor]):
            if src is None: return dst
            return src if dst is None else (dst + src)

        self.batch_acc_sum = _acc(self.batch_acc_sum, other.batch_acc_sum)
        return self

    def reduce_dp_(self, group=None):
        """DDP all-reduce before logging. call once before logging."""
        if not dist.is_initialized():
            return self
        group = group or dist.group.WORLD

        # scalar sum
        scalars = torch.tensor(
            [self.ce_sum,
             float(self.chunks_used), float(self.batches_used), float(self.batches_for_acc)],
            device=torch.cuda.current_device(), dtype=torch.float32
        )
        dist.all_reduce(scalars, op=dist.ReduceOp.SUM, group=group)
        self.ce_sum, chunks, batches, batches_acc = scalars.tolist()
        self.chunks_used     = int(chunks)
        self.batches_used    = int(batches)
        self.batches_for_acc = int(batches_acc)

        # vector sum
        def _all_reduce_(t: Optional[torch.Tensor]):
            if t is None: return
            dist.all_reduce(t, op=dist.ReduceOp.SUM, group=group)

        _all_reduce_(self.batch_acc_sum)
        return self

    def to_logdict(self, *, loss_avg_by: str = "chunk") -> Dict[str, float | List[float]]:
        """
        loss_avg_by:
          - 'chunk' : Average by global valid chunk number
          - 'batch' : Average by global batch number
          - 'none'  : sum of loss
        acc is always 'batch average'(= arithmetic mean of valid batches, no token weighting)
        """
        if loss_avg_by == "chunk":
            denom = max(1, self.chunks_used)
        elif loss_avg_by == "batch":
            denom = max(1, self.batches_used)
        else:
            denom = 1

        ce = self.ce_sum / denom

        if self.batches_for_acc > 0 and self.batch_acc_sum is not None:
            acc = (self.batch_acc_sum / self.batches_for_acc).tolist()
        else:
            acc = [], []

        return {
            "loss": ce,
            "acc": acc,                 # [%] List (draft pos-wise)
            "chunks_used": self.chunks_used,
            "batches_used": self.batches_used,
            "batches_for_acc": self.batches_for_acc,
        }

def format_metrics_line(metrics: dict, step: int) -> str:
    parts = [f"Step {step}"]

    if 'learning_rate' in metrics:
        lr = metrics['learning_rate']
        parts.append(f"LR: {lr:.1e}")
    
    parts.append(f"Loss: {metrics['train/loss']:.4f}")

    temp_acc = {}
    for key, value in metrics.items():
        if 'acc' in key:
            num = re.search(r'_(\d+)$', key)
            if num:
                temp_acc[int(num.group(1))] = value
    
    accs = [v for k, v in sorted(temp_acc.items())]
    if accs:
        acc_str = ", ".join([f"{acc:.2f}%" for acc in accs])
        parts.append(f"Acc: {acc_str}")

    if 'train/chunks_used' in metrics:
        parts.append(f"Num. Processed Chunks: {int(metrics['train/chunks_used'])}")


    return " | ".join(parts)
