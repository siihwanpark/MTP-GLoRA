import torch
import torch.nn as nn
import triton
import triton.language as tl
from triton import Config, autotune


def _compute_loss_prob(logits, target_p):
    """Reference CE using soft targets p (no mask); returns SUM over all elements."""
    logits = logits.float()
    out_logp = nn.LogSoftmax(dim=2)(logits)
    return -(target_p * out_logp).sum(dim=2).sum()

def _compute_loss_logits_no_prob(student_logits, teacher_logits):
    """
    Stable CE without materializing probabilities. Returns SUM over rows.
    CE = -(sum_j e^{t_j-mt} x_j / sum_j e^{t_j-mt}) + mx + log sum_j e^{x_j-mx}
    Shapes: [B, T, V] each.
    """
    x = student_logits.float()
    t = teacher_logits.float()
    mx = x.amax(dim=-1, keepdim=True)
    dx = (x - mx).exp().sum(dim=-1, keepdim=True).clamp_min_(1e-20)
    mt = t.amax(dim=-1, keepdim=True)
    dt = (t - mt).exp().sum(dim=-1, keepdim=True).clamp_min_(1e-20)
    S  = ((t - mt).exp() * x).sum(dim=-1, keepdim=True)
    ce = -(S / dt) + mx + dx.log()
    return ce.sum()


def _calculate_settings(n: int):
    """
    Big-V oriented fallback:
    - Prefer 32K tile (robust against reg-spill), high warps for latency hiding.
    """
    MAX_TILE = 32768
    tile_base = min(n, MAX_TILE)
    BLOCK_SIZE = 1 << ((tile_base - 1).bit_length())
    BLOCK_SIZE = min(BLOCK_SIZE, MAX_TILE)
    if BLOCK_SIZE >= 32768:
        num_warps = 32
    elif BLOCK_SIZE >= 16384:
        num_warps = 16
    else:
        num_warps = 8
    return BLOCK_SIZE, num_warps


AUTOTUNE_CONFIGS = [
    # Most aggressive (Big-V main)
    Config({'BLOCK_SIZE': 65536}, num_warps=32, num_stages=3),
    Config({'BLOCK_SIZE': 65536}, num_warps=32, num_stages=2),

    # Alternative for 64K to mitigate register pressure
    Config({'BLOCK_SIZE': 65536}, num_warps=16, num_stages=2),

    # 32K strong backup (stable/compatible)
    Config({'BLOCK_SIZE': 32768}, num_warps=32, num_stages=2),
    Config({'BLOCK_SIZE': 32768}, num_warps=16, num_stages=2),
]


@autotune(configs=AUTOTUNE_CONFIGS, key=['n_cols'], warmup=10, rep=50, cache_results=True)
@triton.jit
def stable_soft_ce_forward_kernel_auto(
    stud_ptr, stud_stride,     # READ: student logits [V]
    teach_ptr, teach_stride,   # READ: teacher logits [V]
    loss_ptr, loss_stride,     # WRITE: row loss [1]
    ms_ptr, ds_ptr,            # WRITE: student m_x, d_x
    mt_ptr, dt_ptr,            # WRITE: teacher m_t, d_t
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    x_row = stud_ptr  + pid * stud_stride
    t_row = teach_ptr + pid * teach_stride

    # Alignment / coalescing hints
    tl.max_contiguous(tl.arange(0, BLOCK_SIZE), BLOCK_SIZE)
    tl.multiple_of(tl.arange(0, BLOCK_SIZE), 128)

    # 1) Student LSE (mx, dx)
    mx = -float("inf")
    dx = 0.0
    for i in range(0, n_cols, BLOCK_SIZE):
        offs = i + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_cols
        xv = tl.load(x_row + offs, mask=mask, other=-float("inf"), cache_modifier=".cg").to(tl.float32)
        block_max = tl.max(tl.where(mask, xv, -float("inf")))
        mx_new = tl.maximum(mx, block_max)
        dx = dx * tl.exp(mx - mx_new) + tl.sum(tl.where(mask, tl.exp(xv - mx_new), 0.0))
        mx = mx_new
    dx = tl.maximum(dx, 1e-20)
    log_dx = tl.log(dx)

    # 2) Teacher LSE (mt, dt) + weighted sum S (fused into one streaming pass)
    mt = -float("inf")
    dt = 0.0
    S  = 0.0
    for i in range(0, n_cols, BLOCK_SIZE):
        offs = i + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_cols
        tv = tl.load(t_row + offs, mask=mask, other=-float("inf"), cache_modifier=".cg").to(tl.float32)
        xv = tl.load(x_row + offs, mask=mask, other=0.0, cache_modifier=".cg").to(tl.float32)

        block_max = tl.max(tl.where(mask, tv, -float("inf")))
        mt_new = tl.maximum(mt, block_max)
        scale = tl.exp(mt - mt_new)   # rescale prior partials to new base
        dt = dt * scale
        S  = S  * scale

        et = tl.where(mask, tl.exp(tv - mt_new), 0.0)
        dt += tl.sum(et)
        S  += tl.sum(et * xv)

        mt = mt_new

    dt = tl.maximum(dt, 1e-20)
    ce = -(S / dt) + mx + log_dx

    tl.store(loss_ptr + pid * loss_stride, ce.to(tl.float32))
    tl.store(ms_ptr + pid, mx.to(tl.float32))
    tl.store(ds_ptr + pid, dx.to(tl.float32))
    tl.store(mt_ptr + pid, mt.to(tl.float32))
    tl.store(dt_ptr + pid, dt.to(tl.float32))


@triton.jit
def stable_soft_ce_forward_kernel_manual(
    stud_ptr, stud_stride,
    teach_ptr, teach_stride,
    loss_ptr, loss_stride,
    ms_ptr, ds_ptr, mt_ptr, dt_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    x_row = stud_ptr  + pid * stud_stride
    t_row = teach_ptr + pid * teach_stride

    tl.max_contiguous(tl.arange(0, BLOCK_SIZE), BLOCK_SIZE)
    tl.multiple_of(tl.arange(0, BLOCK_SIZE), 128)

    mx = -float("inf")
    dx = 0.0
    for i in range(0, n_cols, BLOCK_SIZE):
        offs = i + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_cols
        xv = tl.load(x_row + offs, mask=mask, other=-float("inf"), cache_modifier=".cg").to(tl.float32)
        block_max = tl.max(tl.where(mask, xv, -float("inf")))
        mx_new = tl.maximum(mx, block_max)
        dx = dx * tl.exp(mx - mx_new) + tl.sum(tl.where(mask, tl.exp(xv - mx_new), 0.0))
        mx = mx_new
    dx = tl.maximum(dx, 1e-20)
    log_dx = tl.log(dx)

    mt = -float("inf")
    dt = 0.0
    S  = 0.0
    for i in range(0, n_cols, BLOCK_SIZE):
        offs = i + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_cols
        tv = tl.load(t_row + offs, mask=mask, other=-float("inf"), cache_modifier=".cg").to(tl.float32)
        xv = tl.load(x_row + offs, mask=mask, other=0.0, cache_modifier=".cg").to(tl.float32)

        block_max = tl.max(tl.where(mask, tv, -float("inf")))
        mt_new = tl.maximum(mt, block_max)
        scale = tl.exp(mt - mt_new)
        dt = dt * scale
        S  = S  * scale

        et = tl.where(mask, tl.exp(tv - mt_new), 0.0)
        dt += tl.sum(et)
        S  += tl.sum(et * xv)

        mt = mt_new

    dt = tl.maximum(dt, 1e-20)
    ce = -(S / dt) + mx + log_dx

    tl.store(loss_ptr + pid * loss_stride, ce.to(tl.float32))
    tl.store(ms_ptr + pid, mx.to(tl.float32))
    tl.store(ds_ptr + pid, dx.to(tl.float32))
    tl.store(mt_ptr + pid, mt.to(tl.float32))
    tl.store(dt_ptr + pid, dt.to(tl.float32))


@autotune(configs=AUTOTUNE_CONFIGS, key=['n_cols'], warmup=10, rep=50, cache_results=True)
@triton.jit
def stable_soft_ce_backward_kernel_auto(
    stud_ptr, stud_stride,     # READ: student logits [V]
    teach_ptr, teach_stride,   # READ: teacher logits [V]
    grad_ptr, grad_stride,     # WRITE: grad wrt student logits [V]
    grad_out_ptr,              # READ: upstream scalar grad
    ms_ptr, ds_ptr, mt_ptr, dt_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    x_row = stud_ptr  + pid * stud_stride
    t_row = teach_ptr + pid * teach_stride
    g_row = grad_ptr  + pid * grad_stride

    tl.max_contiguous(tl.arange(0, BLOCK_SIZE), BLOCK_SIZE)
    tl.multiple_of(tl.arange(0, BLOCK_SIZE), 128)

    mx = tl.load(ms_ptr + pid).to(tl.float32)
    dx = tl.maximum(tl.load(ds_ptr + pid).to(tl.float32), 1e-20)
    mt = tl.load(mt_ptr + pid).to(tl.float32)
    dt = tl.maximum(tl.load(dt_ptr + pid).to(tl.float32), 1e-20)
    go = tl.load(grad_out_ptr).to(tl.float32)

    for i in range(0, n_cols, BLOCK_SIZE):
        offs = i + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_cols
        xv = tl.load(x_row + offs, mask=mask, other=0.0, cache_modifier=".cg").to(tl.float32)
        tv = tl.load(t_row + offs, mask=mask, other=0.0, cache_modifier=".cg").to(tl.float32)
        ps = tl.exp(xv - mx) / dx
        pt = tl.exp(tv - mt) / dt
        grad = go * (ps - pt)
        tl.store(g_row + offs, grad.to(tl.float32), mask=mask)


@triton.jit
def stable_soft_ce_backward_kernel_manual(
    stud_ptr, stud_stride,
    teach_ptr, teach_stride,
    grad_ptr, grad_stride,
    grad_out_ptr,
    ms_ptr, ds_ptr, mt_ptr, dt_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    x_row = stud_ptr  + pid * stud_stride
    t_row = teach_ptr + pid * teach_stride
    g_row = grad_ptr  + pid * grad_stride

    tl.max_contiguous(tl.arange(0, BLOCK_SIZE), BLOCK_SIZE)
    tl.multiple_of(tl.arange(0, BLOCK_SIZE), 128)

    mx = tl.load(ms_ptr + pid).to(tl.float32)
    dx = tl.maximum(tl.load(ds_ptr + pid).to(tl.float32), 1e-20)
    mt = tl.load(mt_ptr + pid).to(tl.float32)
    dt = tl.maximum(tl.load(dt_ptr + pid).to(tl.float32), 1e-20)
    go = tl.load(grad_out_ptr).to(tl.float32)

    for i in range(0, n_cols, BLOCK_SIZE):
        offs = i + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_cols
        xv = tl.load(x_row + offs, mask=mask, other=0.0, cache_modifier=".cg").to(tl.float32)
        tv = tl.load(t_row + offs, mask=mask, other=0.0, cache_modifier=".cg").to(tl.float32)
        ps = tl.exp(xv - mx) / dx
        pt = tl.exp(tv - mt) / dt
        grad = go * (ps - pt)
        tl.store(g_row + offs, grad.to(tl.float32), mask=mask)


class StableSoftCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, student_logits: torch.Tensor, teacher_logits: torch.Tensor):
        assert student_logits.ndim == 3 and teacher_logits.ndim == 3
        assert student_logits.shape == teacher_logits.shape
        B, T, V = student_logits.shape

        xs = student_logits.contiguous().view(B*T, V)
        ts = teacher_logits.contiguous().view(B*T, V)

        loss_rows = torch.empty((B*T, 1), device=xs.device, dtype=torch.float32)
        ms = torch.empty((B*T,), device=xs.device, dtype=torch.float32)
        ds = torch.empty((B*T,), device=xs.device, dtype=torch.float32)
        mt = torch.empty((B*T,), device=xs.device, dtype=torch.float32)
        dt = torch.empty((B*T,), device=xs.device, dtype=torch.float32)

        grid = (B*T,)
        try:
            # AUTO path: let autotuner pick and cache BLOCK_SIZE/warps/stages
            stable_soft_ce_forward_kernel_auto[grid](
                xs, xs.stride(0),
                ts, ts.stride(0),
                loss_rows, loss_rows.stride(0),
                ms, ds, mt, dt,
                V,
            )
        except Exception:
            # MANUAL fallback: explicitly specify meta-parameters
            BLOCK_SIZE, num_warps = _calculate_settings(V)
            stable_soft_ce_forward_kernel_manual[grid](
                xs, xs.stride(0),
                ts, ts.stride(0),
                loss_rows, loss_rows.stride(0),
                ms, ds, mt, dt,
                V,
                BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps,
            )

        ctx.save_for_backward(student_logits.detach(), teacher_logits.detach(), ms, ds, mt, dt)
        ctx.student_dtype = student_logits.dtype
        return loss_rows.squeeze(1).sum()

    @staticmethod
    def backward(ctx, grad_output):
        x_saved, t_saved, ms, ds, mt, dt = ctx.saved_tensors
        B, T, V = x_saved.shape
        xs = x_saved.contiguous().view(B*T, V)
        ts = t_saved.contiguous().view(B*T, V)

        grad_x = xs.new_empty((B*T, V), dtype=torch.float32)
        grid = (B*T,)
        try:
            stable_soft_ce_backward_kernel_auto[grid](
                xs, xs.stride(0),
                ts, ts.stride(0),
                grad_x, grad_x.stride(0),
                grad_output.contiguous(),
                ms, ds, mt, dt,
                V,
            )
        except Exception:
            BLOCK_SIZE, num_warps = _calculate_settings(V)
            stable_soft_ce_backward_kernel_manual[grid](
                xs, xs.stride(0),
                ts, ts.stride(0),
                grad_x, grad_x.stride(0),
                grad_output.contiguous(),
                ms, ds, mt, dt,
                V,
                BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps,
            )
        return grad_x.view(B, T, V).to(ctx.student_dtype), None


# ===================== Minimal tests =====================

if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda"

    # Small test
    B, T, V = 4, 4, 4096
    student = torch.randn(B, T, V, device=device, requires_grad=True)
    teacher = torch.randn(B, T, V, device=device)

    out_triton = StableSoftCrossEntropy.apply(student, teacher)
    target_p = torch.softmax(teacher, dim=-1)

    student_ref = student.clone().detach().requires_grad_(True)
    out_ref_prob = _compute_loss_prob(student_ref, target_p)

    student_ref2 = student.clone().detach().requires_grad_(True)
    out_ref_tlog = _compute_loss_logits_no_prob(student_ref2, teacher)

    torch.testing.assert_close(out_triton, out_ref_prob, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(out_triton, out_ref_tlog, rtol=2e-4, atol=2e-4)

    out_triton.backward()
    out_ref_prob.backward()
    out_ref_tlog.backward()

    torch.testing.assert_close(student.grad, student_ref.grad, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(student.grad, student_ref2.grad, rtol=2e-4, atol=2e-4)
    print("small passed")

    # Big-V smoke (light verification; recommend tuning/performance check in actual training environment)
    B2, T2, V2 = 8, 2, 151_936
    student2 = torch.randn(B2, T2, V2, device=device, requires_grad=True)
    teacher2 = torch.randn(B2, T2, V2, device=device)

    out_t = StableSoftCrossEntropy.apply(student2, teacher2)
    target2 = torch.softmax(teacher2, dim=-1)
    student2_ref = student2.clone().detach().requires_grad_(True)
    out_r = _compute_loss_prob(student2_ref, target2)

    student2_ref2 = student2.clone().detach().requires_grad_(True)
    out_r2 = _compute_loss_logits_no_prob(student2_ref2, teacher2)

    torch.testing.assert_close(out_t, out_r,  rtol=3e-4, atol=3e-4)
    torch.testing.assert_close(out_t, out_r2, rtol=3e-4, atol=3e-4)

    out_t.backward()
    out_r.backward()
    out_r2.backward()

    torch.testing.assert_close(student2.grad, student2_ref.grad,  rtol=3e-4, atol=3e-4)
    torch.testing.assert_close(student2.grad, student2_ref2.grad, rtol=3e-4, atol=3e-4)
    print("big-V passed")