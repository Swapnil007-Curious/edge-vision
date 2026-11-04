"""Training-batch augmentation in pure torch, vectorised over the batch on its own device.

Every transform keeps missing points (x == y == 0.0) at exactly 0.0. Only the temporal
transforms (temporal crop, frame dropout) and time masking touch the presence channels. All
randomness comes from the explicit generator, which must live on the batch's device, so a seed
reproduces an augmented batch exactly. A transform whose probability is 0 draws no random
numbers, so the v2 transforms leave v1 batches bit-for-bit unchanged. There is no horizontal
flip: preprocessing already put the dominant hand on the right.

Any landmark count works: L comes from the frame width (2L + 2).
"""

import math

import torch

from edgevision.data.landmarks import LEFT_HAND, N_HAND, RIGHT_HAND
from edgevision.layout import COORDS_PER_LANDMARK
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


def _gather_frames(values: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """values[b, index[b, t]] for (B, T, ...) values and a (B, T) frame index."""
    shaped = index.reshape(*index.shape, *([1] * (values.dim() - 2)))
    return values.gather(1, shaped.expand_as(values))


def hand_rotation(
    points: torch.Tensor, missing: torch.Tensor, cfg: AugmentConfig, generator: torch.Generator
) -> torch.Tensor:
    """Rotate each hand's 21 points about its own wrist by an independent U(-deg, deg) angle
    per chosen sample. A frame whose wrist is missing keeps that hand as it is."""
    batch = points.shape[0]
    chosen = _chosen(cfg.hand_rotation_p, batch, generator, points)
    angle = _uniform((batch, 2), -cfg.hand_rotation_deg, cfg.hand_rotation_deg, generator, points)
    angle = angle * (math.pi / 180.0)
    rotated = points.clone()
    for hand, block in enumerate((LEFT_HAND, RIGHT_HAND)):
        cos = torch.cos(angle[:, hand])[:, None, None]
        sin = torch.sin(angle[:, hand])[:, None, None]
        hand_points = points[:, :, block]
        wrist = hand_points[:, :, :1]
        offset = hand_points - wrist
        turned = wrist + torch.stack(
            [
                offset[..., 0] * cos - offset[..., 1] * sin,
                offset[..., 0] * sin + offset[..., 1] * cos,
            ],
            dim=-1,
        )
        hand_missing = missing[:, :, block]
        keep = chosen[:, None, None] & ~hand_missing & ~hand_missing[:, :, :1]
        rotated[:, :, block] = torch.where(keep.unsqueeze(-1), turned, hand_points)
    return rotated


def frame_dropout(
    points: torch.Tensor,
    presence: torch.Tensor,
    missing: torch.Tensor,
    cfg: AugmentConfig,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Drop floor(U(0, frame_dropout_max) * T) random frames of each chosen sample, then
    resample the remaining frames back to T: coordinates linearly (a blend with a missing
    endpoint is missing, so missing endpoints stay missing), presence from the nearest frame."""
    batch, frames = points.shape[:2]
    chosen = _chosen(cfg.frame_dropout_p, batch, generator, points)
    fraction = _uniform((batch,), 0.0, cfg.frame_dropout_max, generator, points)
    n_drop = (fraction * frames).floor().long().clamp(max=frames - 2)
    scores = torch.rand((batch, frames), generator=generator, device=points.device)
    rank = scores.argsort(dim=1, stable=True).argsort(dim=1, stable=True)
    steps = torch.arange(frames, device=points.device)
    # Kept frames first, in time order; dropped frames sort after them.
    kept_order = torch.where(rank < n_drop[:, None], steps + frames, steps).argsort(dim=1)
    last_kept = (frames - 1 - n_drop)[:, None]
    positions = steps.to(points.dtype)[None, :] * (last_kept.to(points.dtype) / (frames - 1))

    lower = positions.floor()
    weight = positions - lower
    low = lower.long()
    high = torch.minimum(torch.where(weight > 0.0, low + 1, low), last_kept)
    nearest = torch.minimum((positions + 0.5).floor().long(), last_kept)
    source_low, source_high = kept_order.gather(1, low), kept_order.gather(1, high)

    start, end = _gather_frames(points, source_low), _gather_frames(points, source_high)
    resampled_missing = _gather_frames(missing, source_low) | _gather_frames(missing, source_high)
    blended = start + (end - start) * weight[:, :, None, None]
    blended = blended.masked_fill(resampled_missing.unsqueeze(-1), 0.0)
    resampled_presence = _gather_frames(presence, kept_order.gather(1, nearest))
    return (
        torch.where(chosen[:, None, None, None], blended, points),
        torch.where(chosen[:, None, None], resampled_presence, presence),
        torch.where(chosen[:, None, None], resampled_missing, missing),
    )


def time_mask(
    points: torch.Tensor,
    presence: torch.Tensor,
    missing: torch.Tensor,
    cfg: AugmentConfig,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Blank 1 to time_mask_windows windows of 1 to time_mask_max_frames frames in each chosen
    sample: every coordinate becomes 0.0 (missing) and both presence channels 0."""
    batch, frames = points.shape[:2]
    device = points.device
    windows = cfg.time_mask_windows
    chosen = _chosen(cfg.time_mask_p, batch, generator, points)
    count = torch.randint(1, windows + 1, (batch,), generator=generator, device=device)
    longest = min(cfg.time_mask_max_frames, frames)
    lengths = torch.randint(1, longest + 1, (batch, windows), generator=generator, device=device)
    unit = torch.rand((batch, windows), generator=generator, device=device)
    starts = (unit * (frames - lengths + 1)).floor().long()
    steps = torch.arange(frames, device=device)[None, None, :]
    used = torch.arange(windows, device=device)[None, :, None] < count[:, None, None]
    inside = (steps >= starts[..., None]) & (steps < (starts + lengths)[..., None]) & used
    blanked = inside.any(dim=1) & chosen[:, None]
    return (
        points.masked_fill(blanked[:, :, None, None], 0.0),
        presence.masked_fill(blanked[:, :, None], 0.0),
        missing | blanked[:, :, None],
    )


def landmark_mask(
    points: torch.Tensor, missing: torch.Tensor, cfg: AugmentConfig, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """Make 1 to landmark_mask_max distinct landmarks missing for a chosen sample's whole
    sequence. The two hand wrists are never chosen; presence channels are left alone."""
    batch, _, landmarks = points.shape[:3]
    device = points.device
    chosen = _chosen(cfg.landmark_mask_p, batch, generator, points)
    count = torch.randint(
        1, cfg.landmark_mask_max + 1, (batch,), generator=generator, device=device
    )
    eligible = torch.ones(landmarks, dtype=torch.bool, device=device)
    eligible[[LEFT_HAND.start, RIGHT_HAND.start]] = False
    scores = torch.rand((batch, landmarks), generator=generator, device=device)
    scores = scores.masked_fill(~eligible, 2.0)  # Above every uniform draw: never among the first.
    rank = scores.argsort(dim=1, stable=True).argsort(dim=1, stable=True)
    masked = (rank < count[:, None]) & chosen[:, None]
    return points.masked_fill(masked[:, None, :, None], 0.0), missing | masked[:, None, :]


def augment_batch(x: torch.Tensor, cfg: AugmentConfig, generator: torch.Generator) -> torch.Tensor:
    """Apply the enabled transforms to a (B, T, 2L + 2) batch, in this order: affine, hand
    rotation, temporal crop, frame dropout, jitter, time masking, landmark masking.

    Returns a new tensor, or ``x`` itself when every probability is 0.
    """
    if not cfg.enabled:
        return x
    batch, frames = x.shape[:2]
    num_coords = x.shape[-1] - 2
    points = x[..., :num_coords].reshape(
        batch, frames, num_coords // COORDS_PER_LANDMARK, COORDS_PER_LANDMARK
    )
    if points.shape[2] < 2 * N_HAND:
        raise ValueError(f"expected at least {2 * N_HAND} landmarks, got {points.shape[2]}")
    presence = x[..., num_coords:]
    missing = (points == 0.0).all(dim=-1)

    if cfg.affine_p > 0.0:
        points = affine(points, missing, cfg, generator)
    if cfg.hand_rotation_p > 0.0:
        points = hand_rotation(points, missing, cfg, generator)
    if cfg.temporal_p > 0.0:
        points, presence, missing = temporal(points, presence, missing, cfg, generator)
    if cfg.frame_dropout_p > 0.0:
        points, presence, missing = frame_dropout(points, presence, missing, cfg, generator)
    if cfg.jitter_p > 0.0:
        points = jitter(points, missing, cfg, generator)
    if cfg.time_mask_p > 0.0:
        points, presence, missing = time_mask(points, presence, missing, cfg, generator)
    if cfg.landmark_mask_p > 0.0:
        points, missing = landmark_mask(points, missing, cfg, generator)
    return torch.cat([points.reshape(batch, frames, num_coords), presence], dim=-1)
