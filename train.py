# SPDX-License-Identifier: LicenseRef-LMU-CompVis-NC-Research-1.0
# SPDX-FileCopyrightText: Copyright 2026 Ulrich Prestel, Stefan Baumann et al., CompVis @ LMU Munich

import argparse
import json
import logging
import math
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.nn as dist_nn
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, LinearLR, SequentialLR
from tqdm.auto import tqdm



DEFAULT_RUN_NAME = "test-dl3dv-1fps-128-6-frames-ar-all-tricks-full-jitter"
DEFAULT_DATA_PATH = "/export/koala/uco3d_shards/dl3dv_360p_6fps_resharded_1gb/"


# ---------------------------------------------------------------------------------------------------------------------
# Small Utilities
# ---------------------------------------------------------------------------------------------------------------------

def endless_iter(iterable):
    while True:
        yield from iterable


def make_scheduler(optimizer, lr, warmup_steps, max_steps, scheduler_type="linear"):
    """Build the warmup/decay schedule used by the minimal training loop."""
    has_warmup = warmup_steps > 0
    has_decay = max_steps is not None

    if scheduler_type == "cosine":
        if has_warmup and has_decay:
            warmup = LinearLR(optimizer, start_factor=1e-8 / lr, end_factor=1.0, total_iters=warmup_steps)
            decay = CosineAnnealingLR(optimizer, T_max=max(1, max_steps - warmup_steps), eta_min=1e-8)
            return SequentialLR(optimizer, schedulers=[warmup, decay], milestones=[warmup_steps])
        if has_warmup:
            return LinearLR(optimizer, start_factor=1e-8 / lr, end_factor=1.0, total_iters=warmup_steps)
        if has_decay:
            return CosineAnnealingLR(optimizer, T_max=max_steps, eta_min=1e-8)
        return LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

    if has_warmup and has_decay:
        warmup = LinearLR(optimizer, start_factor=1e-8 / lr, end_factor=1.0, total_iters=warmup_steps)
        decay = LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=max(1, max_steps - warmup_steps))
        return SequentialLR(optimizer, schedulers=[warmup, decay], milestones=[warmup_steps])
    if has_warmup:
        return LinearLR(optimizer, start_factor=1e-8 / lr, end_factor=1.0, total_iters=warmup_steps)
    if has_decay:
        return LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=max_steps)
    return LambdaLR(optimizer, lr_lambda=lambda _: 1.0)


def parse_frame_index_jitter(value: str) -> bool | str:
    if value.lower() in {"false", "0", "no"}:
        return False
    if value.lower() in {"true", "1", "yes"}:
        return True
    if value == "fully_random":
        return "fully_random"
    raise argparse.ArgumentTypeError("Expected one of: false, true, fully_random")


# ---------------------------------------------------------------------------------------------------------------------
# Training Protocol
# - This wrapper defines the minimal self-supervised training objective used here:
#   1. Estimate cameras and dynamic states for all frames in a sampled clip.
#   2. Reconstruct target frames from earlier input frames using the predicted cameras.
#   3. Optimize RGB MSE in [-1, 1] image space.
# - target_mode="next" follows the AR spirit of train_rayzer.sh: frames 0..T-2 condition reconstructions of 1..T-1.
#   target_mode="heldout" is a simpler split where the first N frames reconstruct the remaining frames.
# ---------------------------------------------------------------------------------------------------------------------

class RayDerTrainingWrapper(nn.Module):
    """Small train-only wrapper around the inference-oriented RayDer module."""

    def __init__(self, model: nn.Module, target_mode: str = "next") -> None:
        super().__init__()
        self.model = model
        self.target_mode = target_mode

    def split_batch(self, x: torch.Tensor, n_in: int | None):
        # The core model consumes channels-last video tensors: [B, T, H, W, C].
        if self.target_mode == "next":
            if x.size(1) < 2:
                raise ValueError("target_mode='next' requires at least two frames")
            return x[:, :-1], x[:, 1:], slice(0, -1), slice(1, None)

        if n_in is None:
            n_in = x.size(1) - 1
        if not 1 <= n_in < x.size(1):
            raise ValueError(f"num_input_views must be in [1, {x.size(1) - 1}], got {n_in}")
        return x[:, :n_in], x[:, n_in:], slice(0, n_in), slice(n_in, None)

    def forward(
        self,
        x: torch.Tensor,
        block_mask: Any,
        n_in: int | None = None,
        drop_state: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x_in, x_target, idx_in, idx_target = self.split_batch(x, n_in)

        # Camera estimation is run on the full clip so the target cameras/states are produced by the model itself.
        cameras, state = self.model._estimate_cameras(x)
        x_pred = self.model._reconstruct(
            x_in=x_in,
            camera_in=cameras[:, idx_in],
            camera_target=cameras[:, idx_target],
            state_target=state[:, idx_target],
            block_mask=block_mask,
            drop_state=drop_state,
        )

        loss = F.mse_loss(x_pred, x_target)
        with torch.no_grad():
            # Images live in [-1, 1], so MSE/4 puts PSNR on the conventional [0, 1] intensity scale.
            mse = F.mse_loss(x_pred.float(), x_target.float())
            psnr = -10 * torch.log10((mse / 4).clamp_min(1e-12))
            metrics = {
                "mse": mse.detach(),
                "psnr": psnr.detach(),
                "focal_length_mean": cameras.f.float().mean().detach(),
                "translation_norm": cameras.t.float().norm(dim=-1).mean().detach(),
            }
        return loss, metrics


# ---------------------------------------------------------------------------------------------------------------------
# Distributed & Metric Helpers
# ---------------------------------------------------------------------------------------------------------------------

def setup_distributed(logger: logging.Logger):
    """Initialize torch.distributed when launched with multiple workers, otherwise choose the best local device."""
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed = world_size > 1
    if is_distributed:
        dist.init_process_group()
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        logger.info(f"Running distributed. rank={rank}, local_rank={local_rank}, world_size={world_size}")
    else:
        rank = 0
        local_rank = 0
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch, "mps") and torch.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        logger.info(f"Running non-distributed on {device}")
    return is_distributed, rank, local_rank, world_size, device


def reduce_metrics(metrics: dict[str, torch.Tensor], is_distributed: bool, world_size: int) -> dict[str, float]:
    """Average scalar metrics across ranks and convert them to Python floats for logging."""
    reduced = {}
    for key, value in metrics.items():
        value = value.detach()
        if is_distributed:
            value = dist_nn.all_reduce(value, op=dist.ReduceOp.SUM) / world_size
        reduced[key] = value.item()
    return reduced


# ---------------------------------------------------------------------------------------------------------------------
# Model/Data Factories
# ---------------------------------------------------------------------------------------------------------------------

def make_block_mask_cache(wrapper: RayDerTrainingWrapper, device: torch.device):
    """Cache flex-attention block masks by view split and spatial token count.

    Mask construction is not free, and the sampled batches have fixed shapes in normal training, so caching keeps the
    hot loop focused on model compute.
    """
    cache = {}

    def get(x: torch.Tensor, n_in_arg: int | None):
        _, T, _, H, W = x.shape
        if wrapper.target_mode == "next":
            n_in, n_t = T - 1, T - 1
        else:
            n_in = n_in_arg if n_in_arg is not None else T - 1
            n_t = T - n_in
        n_tokens = (H // wrapper.model.total_spatial_downsample) * (W // wrapper.model.total_spatial_downsample)
        key = (n_in, n_t, n_tokens, str(device))
        if key not in cache:
            cache[key] = wrapper.model._train_block_mask(n_in, n_t, n_tokens, device)
        return cache[key]

    return get


def build_model(preset: str, dynamic_state_dropout: float) -> nn.Module:
    """Construct one of the public RayDer model presets."""
    import rayder.model as rayder_model

    constructors = {
        "XS": rayder_model.RayDer_XS,
        "S": rayder_model.RayDer_S,
        "B": rayder_model.RayDer_B,
        "L": rayder_model.RayDer_L,
    }
    return constructors[preset](dynamic_state_dropout=dynamic_state_dropout)


def build_loader(args, *, validation: bool = False):
    """Build the WebDataset/PyAV video loader with the requested train or validation settings."""
    from rayder.data import SmolVideoLoader, get_minimal_video_transform

    return SmolVideoLoader(
        data_paths=args.val_data_path if validation else args.data_path,
        clip_length=args.num_frames,
        transform=get_minimal_video_transform(size=args.size),
        fps=args.fps,
        batch_size=args.val_batch_size if validation else args.batch_size,
        shuffle=0 if validation else args.shuffle,
        num_workers=args.num_workers,
        frame_index_jitter=False if validation else args.frame_index_jitter,
        video_backend="av",
    )


# ---------------------------------------------------------------------------------------------------------------------
# Validation & Checkpointing
# ---------------------------------------------------------------------------------------------------------------------

@torch.no_grad()
def validate(model, wrapper, val_loader, args, device, is_distributed, world_size):
    """Run a short validation pass over `max_val_steps` batches."""
    model.eval()
    metric_sums: dict[str, float] = {}
    n_steps = 0
    get_block_mask = make_block_mask_cache(wrapper, device)
    for batch in val_loader:
        x = batch["x"].to(device, non_blocking=True)
        block_mask = get_block_mask(x, args.num_input_views)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            loss, metrics = model(x, block_mask=block_mask, n_in=args.num_input_views, drop_state=args.drop_state)
        metrics = {"loss": loss.detach(), **metrics}
        metrics_f = reduce_metrics(metrics, is_distributed, world_size)
        for key, value in metrics_f.items():
            metric_sums[key] = metric_sums.get(key, 0.0) + value
        n_steps += 1
        if n_steps >= args.max_val_steps:
            break
    model.train()
    return {f"val/{key}": value / max(1, n_steps) for key, value in metric_sums.items()}


def save_checkpoint(path, model, optimizer, scheduler, step, args):
    """Save a simple restartable PyTorch checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
            "args": vars(args),
        },
        path,
    )


# ---------------------------------------------------------------------------------------------------------------------
# Main Training Loop
# ---------------------------------------------------------------------------------------------------------------------

def train(args):
    # Output directories are timestamped so repeated local experiments do not overwrite each other.
    run_id = args.run_name
    if run_id is None:
        slurm_id = os.environ.get("SLURM_JOB_ID")
        run_id = slurm_id if slurm_id is not None else DEFAULT_RUN_NAME
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = Path(args.out_dir) / run_id / timestamp
    out_path.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(out_path / "train.log")],
    )
    logger = logging.getLogger("rayder.train")

    is_distributed, rank, local_rank, world_size, device = setup_distributed(logger)
    rank0logger = logger if rank == 0 else logging.getLogger("rayder.train.disabled")
    rank0logger.disabled = rank != 0

    if rank == 0:
        with open(out_path / "args.json", "w") as f:
            json.dump(vars(args), f, indent=2, default=str)
        rank0logger.info(f"Output directory: {out_path}")

    # Rank-dependent seeding avoids identical dataloader/model randomness across distributed workers.
    seed = args.seed + rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    base_model = build_model(args.model, args.dynamic_state_dropout).to(device)
    real_wrapper = RayDerTrainingWrapper(base_model, target_mode=args.target_mode).to(device)

    optimizer = AdamW(real_wrapper.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    scheduler = make_scheduler(optimizer, args.lr, args.warmup_steps, args.max_steps, args.scheduler)

    start_step = 0
    if args.load_checkpoint is not None:
        checkpoint = torch.load(args.load_checkpoint, map_location=device, weights_only=False)
        real_wrapper.load_state_dict(checkpoint["model"])
        if args.ckpt_load_optim:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if args.ckpt_load_scheduler:
            scheduler.load_state_dict(checkpoint["scheduler"])
        start_step = int(checkpoint.get("step", 0))
        rank0logger.info(f"Loaded checkpoint {args.load_checkpoint} at step {start_step}")

    # Keep `real_wrapper` as the plain module for checkpoints/mask metadata; `train_module` may be compiled or DDP-wrapped.
    train_module = real_wrapper
    if args.compile:
        train_module = torch.compile(train_module, fullgraph=False, mode="max-autotune" if args.autotune else "default")

    if is_distributed:
        train_module = DDP(
            train_module,
            device_ids=[local_rank],
            find_unused_parameters=args.find_unused_parameters,
        )

    get_block_mask = make_block_mask_cache(real_wrapper, device)

    train_loader = build_loader(args, validation=False)
    val_loader = build_loader(args, validation=True) if args.val_freq > 0 and args.max_val_steps > 0 else None

    if args.wandb and rank == 0:
        import wandb

        wandb.init(project=args.wandb_project, name=run_id, dir=out_path, config=vars(args))
    else:
        wandb = None

    rank0logger.info(
        f"Model RayDer-{args.model}: {sum(p.numel() for p in real_wrapper.parameters()) / 1e6:.2f}M params"
    )
    train_module.train()

    pbar = tqdm(
        endless_iter(train_loader),
        desc="Training",
        disable=rank != 0,
        initial=start_step,
        total=args.max_steps,
    )
    step = start_step
    for batch in pbar:
        if args.max_steps is not None and step >= args.max_steps:
            break

        # SmolVideoLoader yields [B, T, C, H, W] tensors in [-1, 1].
        x = batch["x"].to(device, non_blocking=True)
        block_mask = get_block_mask(x, args.num_input_views)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            loss, metrics = train_module(x, block_mask=block_mask, n_in=args.num_input_views, drop_state=args.drop_state)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(real_wrapper.parameters(), args.clip_grad_norm)
        optimizer.step()
        scheduler.step()

        step += 1
        metrics = {"loss": loss.detach(), "grad_norm": grad_norm.detach(), **metrics}
        train_metrics = reduce_metrics(metrics, is_distributed, world_size)
        train_metrics["lr"] = scheduler.get_last_lr()[0]
        pbar.set_postfix({k: f"{v:.4g}" for k, v in train_metrics.items()})

        if wandb is not None and rank == 0:
            wandb.log({f"train/{k}": v for k, v in train_metrics.items()}, step=step)

        if val_loader is not None and step % args.val_freq == 0:
            val_metrics = validate(train_module, real_wrapper, val_loader, args, device, is_distributed, world_size)
            if rank == 0:
                rank0logger.info("Validation " + ", ".join(f"{k}={v:.5f}" for k, v in val_metrics.items()))
                if wandb is not None:
                    wandb.log(val_metrics, step=step)

        if rank == 0 and args.checkpoint_freq > 0 and step % args.checkpoint_freq == 0:
            save_checkpoint(out_path / "checkpoints" / f"checkpoint_{step:07d}.pt", real_wrapper, optimizer, scheduler, step, args)
            save_checkpoint(out_path / "checkpoints" / "latest.pt", real_wrapper, optimizer, scheduler, step, args)

    if rank == 0:
        save_checkpoint(out_path / "checkpoints" / "latest.pt", real_wrapper, optimizer, scheduler, step, args)
        rank0logger.info(f"Training stopped at step {step}")


# ---------------------------------------------------------------------------------------------------------------------
# Command Line Interface
# ---------------------------------------------------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Minimal RayDer training")

    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--data-path", nargs="+", default=[DEFAULT_DATA_PATH])
    parser.add_argument("--val-data-path", nargs="+", default=[DEFAULT_DATA_PATH])
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--num-frames", type=int, default=6)
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--frame-index-jitter", type=parse_frame_index_jitter, default="fully_random")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--val-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--shuffle", type=int, default=100)

    parser.add_argument("--model", choices=["XS", "S", "B", "L"], default="B")
    parser.add_argument("--dynamic-state-dropout", type=float, default=0.5)
    parser.add_argument("--target-mode", choices=["next", "heldout"], default="next")
    parser.add_argument("--num-input-views", type=int, default=None)
    parser.add_argument("--drop-state", action="store_true")

    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--val-freq", type=int, default=2000)
    parser.add_argument("--max-val-steps", type=int, default=4)
    parser.add_argument("--checkpoint-freq", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--scheduler", choices=["linear", "cosine"], default="linear")
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)

    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--autotune", action="store_true")
    parser.add_argument("--find-unused-parameters", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load-checkpoint", type=Path, default=None)
    parser.add_argument("--ckpt-load-optim", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ckpt-load-scheduler", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="rayder")

    return parser.parse_args()


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    if hasattr(torch, "_dynamo"):
        torch._dynamo.config.cache_size_limit = max(64, torch._dynamo.config.cache_size_limit)

    try:
        train(parse_args())
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
