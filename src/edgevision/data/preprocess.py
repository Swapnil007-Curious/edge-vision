"""Pure per-sequence transforms; ``preprocess_sequence`` applies them in order.

Internal coordinate arrays have shape (T, L, C) with C = 3 until normalization drops z when it
is not used. Hand presence has shape (T, 2) with columns (left, right).
"""

from collections.abc import Sequence
from typing import NamedTuple, TypedDict

import numpy as np
import numpy.typing as npt
import pandas as pd

from edgevision.data.landmarks import (
    DEFAULT_POSE_INDICES,
    FIRST_POSE_SLOT,
    LEFT_HAND,
    N_ALL_LANDMARKS,
    RIGHT_HAND,
    TYPE_CODES,
    TYPE_OFFSETS,
    TYPE_SIZES,
    mirror_permutation,
    num_selected,
    selection_table,
    shoulder_slots,
)

FloatArray = npt.NDArray[np.float64]
Float32Array = npt.NDArray[np.float32]
BoolArray = npt.NDArray[np.bool_]

MIN_SCALE = 1e-6

# x -> 1 - x cannot be an exact involution on arbitrary floats: there are far more floats in
# (0, 0.5) than in (0.5, 1), so some must collide. On this dyadic grid every value and its
# mirror are exactly representable for |x| < 2**20, so x is snapped here before any mirroring.
# The grid step (2.3e-10) is well below the float32 resolution of the outputs.
X_GRID = 2.0**32


class SequenceInfo(TypedDict):
    t_raw: int
    t_kept: int
    mirrored: bool
    norm_method: str | None
    excluded: bool
    reason: str | None


class PreprocessResult(NamedTuple):
    """Features and presence are None when ``info["excluded"]`` is True."""

    features: Float32Array | None
    hand_presence: Float32Array | None
    info: SequenceInfo


def snap_x(x: FloatArray) -> FloatArray:
    """Round to the dyadic grid; adding 0.0 turns -0.0 into 0.0 so mirroring keeps every bit."""
    return np.rint(x * X_GRID) / X_GRID + 0.0


def pivot_sequence(
    df: pd.DataFrame, pose_indices: Sequence[int] = DEFAULT_POSE_INDICES
) -> FloatArray:
    """Pivot long-format rows to a dense (T_raw, L, 3) array sorted by frame.

    Missing rows become NaN. Duplicates, unknown types, out-of-range landmark indices and
    infinite coordinates break the dataset contract and raise ValueError.
    """
    type_codes = df["type"].map(TYPE_CODES)
    if type_codes.isna().any():
        unknown = sorted(set(df.loc[type_codes.isna(), "type"].astype(str)))
        raise ValueError(f"unknown landmark types: {unknown}")
    codes = type_codes.to_numpy(dtype=np.intp)
    landmarks = df["landmark_index"].to_numpy(dtype=np.intp)
    if ((landmarks < 0) | (landmarks >= TYPE_SIZES[codes])).any():
        raise ValueError("landmark_index outside the valid range for its type")
    flat_ids = TYPE_OFFSETS[codes] + landmarks

    frames, frame_rows = np.unique(df["frame"].to_numpy(dtype=np.int64), return_inverse=True)
    if np.unique(frame_rows * N_ALL_LANDMARKS + flat_ids).size != flat_ids.size:
        raise ValueError("duplicate (frame, type, landmark_index) rows")

    xyz = df[["x", "y", "z"]].to_numpy(dtype=np.float64)
    if np.isinf(xyz).any():
        raise ValueError("infinite coordinates")

    slots = selection_table(pose_indices)[flat_ids]
    selected = slots >= 0
    coords = np.full((frames.size, num_selected(pose_indices), 3), np.nan)
    coords[frame_rows[selected], slots[selected]] = xyz[selected]
    coords[..., 0] = snap_x(coords[..., 0])
    return coords


def hand_presence(coords: FloatArray) -> BoolArray:
    """A hand is present in a frame when all 21 of its x, y values are finite."""
    finite = np.isfinite(coords[..., :2])
    left = finite[:, LEFT_HAND].all(axis=(1, 2))
    right = finite[:, RIGHT_HAND].all(axis=(1, 2))
    return np.stack([left, right], axis=1)


def left_is_dominant(presence: BoolArray) -> bool:
    """True when the left hand is present in strictly more frames; ties go to the right."""
    left_frames, right_frames = presence.sum(axis=0)
    return bool(left_frames > right_frames)


def mirror_sequence(
    coords: FloatArray, pose_indices: Sequence[int] = DEFAULT_POSE_INDICES
) -> FloatArray:
    """Flip x and swap left/right landmarks. Exactly self-inverse on snapped coordinates."""
    mirrored = coords[:, mirror_permutation(pose_indices)]
    mirrored[..., 0] = 1.0 - mirrored[..., 0]
    return mirrored


def normalization_params(
    coords: FloatArray, pose_indices: Sequence[int] = DEFAULT_POSE_INDICES
) -> tuple[FloatArray, float, str]:
    """Return (center_xy, scale, method) from the shoulders, or the hands as a fallback."""
    slots = shoulder_slots(pose_indices)
    if slots is not None:
        left, right = coords[:, slots[0], :2], coords[:, slots[1], :2]
        if (np.isfinite(left).all(axis=1) & np.isfinite(right).all(axis=1)).any():
            center = np.nanmedian(0.5 * (left + right), axis=0)
            distance = np.hypot(left[:, 0] - right[:, 0], left[:, 1] - right[:, 1])
            return center, float(np.nanmedian(distance)), "shoulders"

    hands = coords[:, :FIRST_POSE_SLOT, :2].reshape(-1, 2)
    center = np.nanmean(hands, axis=0)
    spread = np.nanmax(hands, axis=0) - np.nanmin(hands, axis=0)
    return center, float(spread.max()), "hand_fallback"


def normalize_coords(
    coords: FloatArray, center: FloatArray, scale: float, *, use_z: bool
) -> FloatArray:
    """Apply (p - center) / scale to x, y and scale z when it is kept; z is not centered."""
    xy = (coords[..., :2] - center) / scale
    if not use_z:
        return xy
    return np.concatenate([xy, coords[..., 2:3] / scale], axis=-1)


def resample_coords(coords: FloatArray, num_frames: int) -> FloatArray:
    """Linearly resample over the kept-frame index, per coordinate, between finite values only.

    Positions before the first or after the last finite value of a coordinate stay NaN rather
    than holding the edge value, as do coordinates that are never finite.
    """
    t_kept, n_points, n_coords = coords.shape
    source = np.arange(t_kept, dtype=np.float64)
    positions = np.linspace(0.0, t_kept - 1, num_frames)
    resampled = np.full((num_frames, n_points, n_coords), np.nan)
    for point in range(n_points):
        for axis in range(n_coords):
            series = coords[:, point, axis]
            finite = np.isfinite(series)
            if finite.any():
                resampled[:, point, axis] = np.interp(
                    positions, source[finite], series[finite], left=np.nan, right=np.nan
                )
    return resampled


def resample_presence(presence: BoolArray, num_frames: int) -> BoolArray:
    """Nearest-neighbour resampling over the kept-frame index."""
    positions = np.linspace(0.0, presence.shape[0] - 1, num_frames)
    return presence[np.floor(positions + 0.5).astype(np.intp)]


def fill_missing(coords: FloatArray) -> Float32Array:
    """Replace NaN with 0.0 and convert to float32, refusing any non-finite result."""
    features = np.where(np.isnan(coords), 0.0, coords).astype(np.float32)
    if not np.isfinite(features).all():
        raise AssertionError("non-finite values remain after NaN replacement")
    return features


def preprocess_sequence(
    df: pd.DataFrame,
    *,
    num_frames: int = 32,
    use_z: bool = False,
    pose_indices: Sequence[int] = DEFAULT_POSE_INDICES,
    min_hand_frames: int = 3,
) -> PreprocessResult:
    """Turn one long-format sequence into fixed-length features, presence and an info dict."""
    coords = pivot_sequence(df, pose_indices)
    presence = hand_presence(coords)
    any_hand = presence.any(axis=1)
    coords, presence = coords[any_hand], presence[any_hand]

    info = SequenceInfo(
        t_raw=int(any_hand.size),
        t_kept=int(any_hand.sum()),
        mirrored=False,
        norm_method=None,
        excluded=False,
        reason=None,
    )
    if info["t_kept"] < min_hand_frames:
        return _excluded(info, "insufficient_hand_frames")

    if left_is_dominant(presence):
        coords = mirror_sequence(coords, pose_indices)
        # The presence columns must follow the swapped hand blocks.
        presence = presence[:, ::-1]
        info["mirrored"] = True

    center, scale, info["norm_method"] = normalization_params(coords, pose_indices)
    if not np.isfinite(scale) or scale < MIN_SCALE:
        return _excluded(info, "degenerate_scale")

    normalized = normalize_coords(coords, center, scale, use_z=use_z)
    features = fill_missing(resample_coords(normalized, num_frames))
    resampled_presence = resample_presence(presence, num_frames).astype(np.float32)
    return PreprocessResult(features, resampled_presence, info)


def _excluded(info: SequenceInfo, reason: str) -> PreprocessResult:
    info["excluded"] = True
    info["reason"] = reason
    return PreprocessResult(None, None, info)
