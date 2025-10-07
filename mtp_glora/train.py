from __future__ import annotations

from tqdm import tqdm

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from mtp_glora.args import parse_args
from mtp_glora.core import MTPModel
from mtp_glora.models.layers import RMSNorm
from mtp_glora.trainer import (
    MTPTrainer,
    setup_distributed_training,
    load_tokenizer,
    load_model_and_apply_lora,
    get_train_dataloader,
    accumulate_batches,
)
from mtp_glora.utils import (
    create_tracker,
    get_dp_group,
    destroy_distributed,
    print_on_rank0,
    format_metrics_line,
    set_seed,
)


def main():
    args = parse_args()
    set_seed(args.seed)

    # Init distributed
    is_dist, local_rank = setup_distributed_training(args)

    # Tracker
    tracker = create_tracker(args)

    # Tokenizer/Model
    tokenizer = load_tokenizer(args, append_mask_token=True)
    base_model = load_model_and_apply_lora(args, tokenizer, fuse_weights=args.fuse_weights)

    model = MTPModel(
        model=base_model,
        draft_length=args.draft_length,
    )

    if is_dist:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            process_group=get_dp_group(),
        )
        print_on_rank0("Wrapped model with DDP")
    
    # DataLoader
    train_loader, distributed_length_sampler = get_train_dataloader(args, tokenizer)
    if is_dist: dist.barrier(get_dp_group())

    # Optimizer/Scheduler
    rms_params = [m.weight for m in model.module.sampler.modules() if isinstance(m, RMSNorm)]
    other_params = [p for n,p in model.module.named_parameters() if p.requires_grad and (not n.endswith("norm.weight"))]
    optimizer = torch.optim.AdamW([
        {'params': other_params, 'lr': args.lr, 'weight_decay': 0.01},
        {'params': rms_params, 'lr': args.lr, 'weight_decay': 0.0},
    ], fused=True)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=args.warmup_steps)

    # Trainer
    trainer = MTPTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        grad_accum_steps=args.grad_accumulation_steps,
        draft_length=args.draft_length,
        is_distributed=is_dist,
        local_rank=local_rank,
        distributed_length_sampler=distributed_length_sampler,
    )

    # Step counter
    global_step = 0
    if args.resume and args.checkpoint_dir is not None:
        global_step = trainer.load_checkpoint(args.checkpoint_dir)
        if global_step == 0: print_on_rank0(f"No checkpoint found at {args.checkpoint_dir}. Starting from scratch.")
    
    # Progress bar
    if not is_dist or local_rank == 0:
        pbar_total = args.max_steps
        pbar = tqdm(total=pbar_total, desc="Training", initial=global_step)
    else: pbar = None

    model.train()
    train_iter = accumulate_batches(train_loader, args.grad_accumulation_steps, distributed_length_sampler)
    while global_step < args.max_steps:
        metrics = trainer.training_step(next(train_iter))
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
        optimizer.step()
        scheduler.step()

        if global_step % args.logging_steps == 0:
            logdict = {"learning_rate": optimizer.param_groups[0]["lr"]}
            for k, v in metrics.items():
                if k in ["acc"]:
                    for i, acc in enumerate(v, start=1):
                        logdict[f"train/{k}_{i}"] = acc
                else: logdict[f"train/{k}"] = v
            tracker.log(logdict, step=global_step)
            if not is_dist or local_rank == 0: print_on_rank0(format_metrics_line(logdict, step=global_step))

        if global_step > 0 and global_step % args.save_steps == 0 and args.save_dir:
            trainer.save_checkpoint(args.save_dir, dict(vars(args)), global_step, max_to_keep=args.save_limit)

        global_step += 1
        if pbar is not None: pbar.update(1)
        if global_step >= args.max_steps: break

    # Save final checkpoint
    if args.save_dir:
        trainer.save_checkpoint(args.save_dir, dict(vars(args)), args.max_steps, max_to_keep=args.save_limit)

    tracker.close()
    if is_dist: destroy_distributed()

if __name__ == "__main__":
    main()