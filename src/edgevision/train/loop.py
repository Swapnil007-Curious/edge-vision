"""Training: seeding, the epoch loop with early stopping, and the run directory's outputs.

Train and val arrays live on the device as whole float32 tensors and batches are drawn by a
seeded index permutation; there are no DataLoader workers.
"""

import contextlib
import math
import os
import shutil
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from torch import nn

from edgevision.data.io import write_json
from edgevision.evaluation.metrics import macro_f1, top_k_accuracy
from edgevision.models.config import MODEL_NAMES, ModelName
from edgevision.models.features import SignClassifier, compute_feature_stats
from edgevision.models.registry import build_model, count_parameters, save_checkpoint
from edgevision.runs import CHECKPOINT_FILE, SUMMARY_FILE, is_complete, run_dir
from edgevision.train.augment import augment_batch
from edgevision.train.config import AugmentConfig, TrainConfig, dump_train_config
from edgevision.train.data import (
    label_map_from_manifest,
    load_split_arrays,
    prepare_splits,
    read_manifest,
)
from edgevision.utils.runinfo import collect_run_info
from edgevision.utils.seed import set_seed

SplitMode = Literal["signer_independent", "signer_dependent"]
DeviceChoice = Literal["auto", "cpu", "cuda"]

DETERMINISM_NOTE = (
    "Seeded, cudnn.benchmark off, deterministic algorithms requested with warn_only=True. "
    "CPU runs repeat exactly for a seed; GPU runs may still be non-bitwise-deterministic."
)
EVAL_BATCH = 1024
METRICS_COLUMNS = (
    "epoch",
    "train_loss",
    "train_top1",
    "val_loss",
    "val_top1",
    "val_top5",
    "lr",
    "epoch_seconds",
)


class RunExistsError(FileExistsError):
    """The run directory already holds a completed run."""


def seed_everything(seed: int) -> None:
    """Seed every generator and request deterministic kernels; call before any CUDA work."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # Read when cuBLAS initialises.
    set_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(choice: DeviceChoice) -> torch.device:
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device cuda was requested but CUDA is not available")
    return torch.device(choice)


def warmup_cosine(total_steps: int, warmup_fraction: float) -> Callable[[int], float]:
    """LR factor per optimizer step: linear warmup, then cosine decay towards zero."""
    warmup = max(1, round(total_steps * warmup_fraction)) if warmup_fraction > 0.0 else 0

    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return factor


def _autocast(enabled: bool) -> contextlib.AbstractContextManager[Any]:
    if enabled:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


@dataclass
class EpochContext:
    """Everything an epoch needs besides the model and data."""

    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    criterion: nn.Module
    scaler: torch.amp.GradScaler
    batch_size: int
    grad_clip: float
    use_amp: bool
    augment: AugmentConfig | None
    order_generator: torch.Generator
    augment_generator: torch.Generator


def train_one_epoch(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor, ctx: EpochContext
) -> tuple[float, float, float]:
    """One pass over (x, y) in a seeded random order.

    Returns (mean loss, top-1 accuracy, learning rate of the last step). Accuracy is measured
    on the augmented batches, in train mode.
    """
    model.train()
    n = x.shape[0]
    order = torch.randperm(n, generator=ctx.order_generator).to(x.device)
    loss_sum = torch.zeros((), device=x.device)
    correct = torch.zeros((), device=x.device, dtype=torch.long)
    last_lr = 0.0
    for start in range(0, n, ctx.batch_size):
        index = order[start : start + ctx.batch_size]
        batch, target = x[index], y[index]
        if ctx.augment is not None:
            batch = augment_batch(batch, ctx.augment, ctx.augment_generator)
        with _autocast(ctx.use_amp):
            logits = model(batch)
            loss = ctx.criterion(logits, target)
        ctx.optimizer.zero_grad(set_to_none=True)
        ctx.scaler.scale(loss).backward()
        ctx.scaler.unscale_(ctx.optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), ctx.grad_clip)
        ctx.scaler.step(ctx.optimizer)
        ctx.scaler.update()
        last_lr = float(ctx.optimizer.param_groups[0]["lr"])
        ctx.scheduler.step()
        loss_sum += loss.detach().float() * index.numel()
        correct += (logits.argmax(dim=-1) == target).sum()
    return loss_sum.item() / n, correct.item() / n, last_lr


@torch.no_grad()
def predict_logits(model: nn.Module, x: torch.Tensor, batch_size: int = EVAL_BATCH) -> torch.Tensor:
    """Float32 logits in eval mode; evaluation never uses autocast."""
    model.eval()
    return torch.cat(
        [model(x[start : start + batch_size]).float() for start in range(0, len(x), batch_size)]
    )


def evaluate(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor, criterion: nn.Module
) -> tuple[float, npt.NDArray[np.float32]]:
    """(loss with the training criterion, logits as numpy)."""
    logits = predict_logits(model, x)
    loss = float(criterion(logits, y).item())
    return loss, logits.cpu().numpy()


def _device_name(device: torch.device, run_info: dict[str, Any]) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return str(run_info["cpu_model"])


def run_training(
    cfg: TrainConfig,
    *,
    seed: int,
    device: DeviceChoice = "auto",
    overwrite: bool = False,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Train one model and write its run directory; returns the summary."""
    directory = run_dir(cfg.output_root, cfg.model.name, cfg.split.mode, seed)
    if is_complete(directory) and not overwrite:
        raise RunExistsError(f"{directory} holds a completed run; pass --overwrite to replace it")
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)

    seed_everything(seed)
    torch_device = resolve_device(device)
    if torch_device.type == "cpu":
        torch.set_num_threads(cfg.num_threads)
    use_amp = torch_device.type == "cuda"

    splits_path = prepare_splits(cfg.processed_dir, cfg.split, directory)
    arrays = load_split_arrays(cfg.processed_dir, splits_path, ("train", "val"))
    label_map = label_map_from_manifest(read_manifest(cfg.processed_dir))
    num_classes = len(label_map)
    (directory / "config.yaml").write_text(dump_train_config(cfg), encoding="utf-8")
    write_json(
        directory / "label_map.json", {str(index): sign for index, sign in label_map.items()}
    )

    train, val = arrays["train"], arrays["val"]
    x_train = torch.from_numpy(train.X).to(torch_device)
    y_train = torch.from_numpy(train.y).to(torch_device)
    x_val = torch.from_numpy(val.X).to(torch_device)
    y_val = torch.from_numpy(val.y).to(torch_device)

    model: SignClassifier = build_model(cfg.model.name, num_classes, cfg.model).to(torch_device)
    compute_feature_stats(model.features, x_train)
    params = count_parameters(model)
    log(
        f"{cfg.model.name} ({cfg.split.mode}, seed {seed}): {params:,} parameters, "
        f"{num_classes} classes, train {len(train.y)} / val {len(val.y)} sequences, "
        f"device {torch_device.type}, amp {use_amp}"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = cfg.epochs * math.ceil(len(train.y) / cfg.batch_size)
    order_seed, augment_seed = (
        int(child.generate_state(1)[0]) for child in np.random.SeedSequence(seed).spawn(2)
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    ctx = EpochContext(
        optimizer=optimizer,
        scheduler=torch.optim.lr_scheduler.LambdaLR(
            optimizer, warmup_cosine(total_steps, cfg.warmup_fraction)
        ),
        criterion=criterion,
        scaler=torch.amp.GradScaler("cuda", enabled=use_amp),
        batch_size=cfg.batch_size,
        grad_clip=cfg.grad_clip,
        use_amp=use_amp,
        augment=cfg.augment if cfg.augment.enabled else None,
        order_generator=torch.Generator().manual_seed(order_seed),
        augment_generator=torch.Generator(device=torch_device).manual_seed(augment_seed),
    )

    rows: list[dict[str, float]] = []
    best_top1, best_top5, best_epoch, stale = -1.0, 0.0, 0, 0
    best_predictions = np.empty(0, dtype=np.int64)
    started = time.perf_counter()
    for epoch in range(1, cfg.epochs + 1):
        epoch_started = time.perf_counter()
        train_loss, train_top1, lr = train_one_epoch(model, x_train, y_train, ctx)
        val_loss, val_logits = evaluate(model, x_val, y_val, criterion)
        val_top1 = top_k_accuracy(val_logits, val.y, 1)
        val_top5 = top_k_accuracy(val_logits, val.y, 5)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_top1": train_top1,
            "val_loss": val_loss,
            "val_top1": val_top1,
            "val_top5": val_top5,
            "lr": lr,  # Used by the epoch's last step.
            "epoch_seconds": time.perf_counter() - epoch_started,
        }
        rows.append(row)
        pd.DataFrame(rows, columns=list(METRICS_COLUMNS)).to_csv(
            directory / "metrics.csv", index=False
        )
        log(
            f"epoch {epoch:3d}/{cfg.epochs} train_loss {train_loss:.4f} "
            f"train_top1 {train_top1:.4f} val_loss {val_loss:.4f} "
            f"val_top1 {val_top1:.4f} val_top5 {val_top5:.4f} "
            f"lr {row['lr']:.2e} {row['epoch_seconds']:.1f}s"
        )
        if val_top1 > best_top1:
            best_top1, best_top5, best_epoch, stale = val_top1, val_top5, epoch, 0
            best_predictions = val_logits.argmax(axis=1)
            save_checkpoint(
                directory / CHECKPOINT_FILE,
                model,
                cfg.model,
                num_classes=num_classes,
                label_map=label_map,
            )
        else:
            stale += 1
            if stale >= cfg.patience:
                log(f"early stop: no val top-1 gain for {cfg.patience} epochs")
                break

    run_info = collect_run_info()
    summary: dict[str, Any] = {
        "model": cfg.model.name,
        "params": params,
        "seed": seed,
        "split_mode": cfg.split.mode,
        "num_classes": num_classes,
        "train_sequences": len(train.y),
        "val_sequences": len(val.y),
        "best_epoch": best_epoch,
        "best_val_top1": best_top1,
        "best_val_top5": best_top5,
        "best_val_macro_f1": macro_f1(val.y, best_predictions),
        "epochs_run": len(rows),
        "total_train_seconds": time.perf_counter() - started,
        "device": torch_device.type,
        "device_name": _device_name(torch_device, run_info),
        "amp": use_amp,
        "torch_version": torch.__version__,
        "determinism": DETERMINISM_NOTE,
        "processed_dir": str(cfg.processed_dir),
        "splits_file": splits_path.name,
        "run_info": run_info,
    }
    write_json(directory / SUMMARY_FILE, summary)
    return summary


def grid_plan(seeds: Sequence[int]) -> list[tuple[ModelName, SplitMode, int]]:
    """Signer-independent runs for every model and seed, seed-major so an interrupted grid
    still has every model at the first seed; then signer-dependent runs at the first seed."""
    if not seeds:
        raise ValueError("at least one seed is required")
    plan: list[tuple[ModelName, SplitMode, int]] = [
        (model, "signer_independent", seed) for seed in seeds for model in MODEL_NAMES
    ]
    plan += [(model, "signer_dependent", seeds[0]) for model in MODEL_NAMES]
    return plan
