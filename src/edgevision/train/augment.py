"""Training-batch augmentation in pure torch, vectorised over the batch on its own device.

Every transform keeps missing points (x == y == 0.0) at exactly 0.0, and only the temporal
transform touches the presence channels. All randomness comes from the explicit generator,
which must live on the batch's device, so a seed reproduces an augmented batch exactly.
There is no horizontal flip: preprocessing already put the dominant hand on the right.
"""

import math

import torch

from edgevision.layout import COORDS_PER_LANDMARK, NUM_COORDS, NUM_LANDMARKS
from edgevision.train.config import AugmentConfig


def _uniform(
    shape: tuple[int, ...], low: float, high: float, generator: torch.Generator, like: torch.Tensor
) -> torch.Tensor:
    unit = torch.rand(shape, generator=generator, device=like.device, dtype=like.dtype)
    return low + (high - low) * unit


def _chosen(p: float, batch: int, generator: torch.Generator, like: torch.Tensor) -> torch.Tensor:
    """Per-sample Bernoulli(p) mask of shape (B,)."""
    return torch.rand((batch,), generator=generator, device=like.device) < p


def affine(
    points: torch.Tensor, missing: torch.Tensor, cfg: AugmentConfig, generator: torch.Generator
) -> torch.Tensor:
    """Rotate, scale isotropically and translate each chosen sample's (B, T, L, 2) points."""
    batch = points.shape[0]
    chosen = _chosen(cfg.affine_p, batch, generator, points)
    angle = _uniform((batch,), -cfg.rotation_deg, cfg.rotation_deg, generator, points)
    angle = angle * (math.pi / 180.0)
    scale = _uniform((batch,), cfg.scale_min, cfg.scale_max, generator, points)
    shift = _uniform((batch, 2), -cfg.translate, cfg.translate, generator, points)

    cos, sin = torch.cos(angle) * scale, torch.sin(angle) * scale
    # Row-vector convention: [x, y] @ [[cos, sin], [-sin, cos]] rotates counter-clockwise.
    matrix = torch.stack([torch.stack([cos, sin], -1), torch.stack([-sin, cos], -1)], -2)
    moved = torch.einsum("btlc,bcd->btld", points, matrix) + shift[:, None, None, :]
    moved = torch.where(chosen[:, None, None, None], moved, points)
    return moved.masked_fill(missing.unsqueeze(-1), 0.0)


def temporal(
    points: torch.Tensor,
    presence: torch.Tensor,
    missing: torch.Tensor,
    cfg: AugmentConfig,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Crop a window of U(window_min, 1) of the sequence and resample it back to T frames.

    Coordinates are linearly interpolated; a blend with a missing endpoint is missing instead.
    Presence uses the nearest frame.
    """
    batch, frames = points.shape[:2]
    chosen = _chosen(cfg.temporal_p, batch, generator, points)
    last = float(frames - 1)
    length = _uniform((batch,), cfg.window_min, 1.0, generator, points) * last
    offset = _uniform((batch,), 0.0, 1.0, generator, points) * (last - length)
    steps = torch.arange(frames, device=points.device, dtype=points.dtype)
    positions = (offset[:, None] + steps[None, :] * (length / last)[:, None]).clamp(0.0, last)

    lower = positions.floor()
    weight = positions - lower
    low = lower.long()
    # An endpoint with zero weight is not part of the blend.
    high = torch.where(weight > 0.0, low + 1, low).clamp(max=frames - 1)

    def gather_frames(values: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        shaped = index.reshape(batch, frames, *([1] * (values.dim() - 2)))
        return values.gather(1, shaped.expand_as(values))

    start, end = gather_frames(points, low), gather_frames(points, high)
    resampled_missing = gather_frames(missing, low) | gather_frames(missing, high)
    blended = start + (end - start) * weight[:, :, None, None]
    blended = blended.masked_fill(resampled_missing.unsqueeze(-1), 0.0)
    nearest = gather_frames(presence, (positions + 0.5).floor().long().clamp(max=frames - 1))

    return (
        torch.where(chosen[:, None, None, None], blended, points),
        torch.where(chosen[:, None, None], nearest, presence),
        torch.where(chosen[:, None, None], resampled_missing, missing),
    )


def jitter(
    points: torch.Tensor, missing: torch.Tensor, cfg: AugmentConfig, generator: torch.Generator
) -> torch.Tensor:
    """Add N(0, jitter_std) noise to the non-missing coordinates of each chosen sample."""
    chosen = _chosen(cfg.jitter_p, points.shape[0], generator, points)
    noise = torch.randn(points.shape, generator=generator, device=points.device, dtype=points.dtype)
    keep = chosen[:, None, None, None] & ~missing.unsqueeze(-1)
    return torch.where(keep, points + cfg.jitter_std * noise, points)


def augment_batch(x: torch.Tensor, cfg: AugmentConfig, generator: torch.Generator) -> torch.Tensor:
    """Apply affine, temporal and jitter transforms to a (B, T, 100) batch.

    Returns a new tensor, or ``x`` itself when every probability is 0.
    """
    if not cfg.enabled:
        return x
    batch, frames = x.shape[:2]
    points = x[..., :NUM_COORDS].reshape(batch, frames, NUM_LANDMARKS, COORDS_PER_LANDMARK)
    presence = x[..., NUM_COORDS:]
    missing = (points == 0.0).all(dim=-1)

    if cfg.affine_p > 0.0:
        points = affine(points, missing, cfg, generator)
    if cfg.temporal_p > 0.0:
        points, presence, missing = temporal(points, presence, missing, cfg, generator)
    if cfg.jitter_p > 0.0:
        points = jitter(points, missing, cfg, generator)
    return torch.cat([points.reshape(batch, frames, NUM_COORDS), presence], dim=-1)
