"""The v2 augmentations: the missing-point rule, determinism, and no effect at probability 0."""

import math

import pytest
import torch

from edgevision.train.augment import (
    augment_batch,
    frame_dropout,
    hand_rotation,
    landmark_mask,
    time_mask,
)
from edgevision.train.config import AugmentConfig

V1_ONLY = AugmentConfig()  # The v1 probabilities: affine 0.8, temporal 0.5, jitter 0.5.
NONE = AugmentConfig(affine_p=0.0, temporal_p=0.0, jitter_p=0.0)
ONLY = {
    "frame_dropout": NONE.model_copy(update={"frame_dropout_p": 1.0}),
    "time_mask": NONE.model_copy(update={"time_mask_p": 1.0}),
    "landmark_mask": NONE.model_copy(update={"landmark_mask_p": 1.0}),
    "hand_rotation": NONE.model_copy(update={"hand_rotation_p": 1.0}),
}
V2_RECIPE = AugmentConfig(
    frame_dropout_p=0.5, time_mask_p=0.5, landmark_mask_p=0.3, hand_rotation_p=0.5
)


def make_batch(batch=16, landmarks=59, seed=0):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, 32, 2 * landmarks + 2, generator=generator)
    points = x[..., : 2 * landmarks].view(batch, 32, landmarks, 2)
    missing = torch.rand(batch, 32, landmarks, generator=generator) < 0.15
    missing[:, :, :21] = True  # Left hand absent throughout.
    missing[0, 10:20] = True  # Whole frames missing.
    points[missing] = 0.0
    x[..., 2 * landmarks :] = (torch.rand(batch, 32, 2, generator=generator) < 0.7).float()
    return x


def split(x):
    landmarks = (x.shape[-1] - 2) // 2
    points = x[..., : 2 * landmarks].reshape(x.shape[0], 32, landmarks, 2)
    return points, x[..., 2 * landmarks :], (points == 0.0).all(dim=-1)


def run(x, cfg, seed=1):
    return augment_batch(x, cfg, torch.Generator().manual_seed(seed))


@pytest.mark.parametrize("landmarks", [49, 59])
@pytest.mark.parametrize("name", [*ONLY, "v2_recipe"])
def test_points_missing_throughout_stay_missing(name, landmarks):
    cfg = V2_RECIPE if name == "v2_recipe" else ONLY[name]
    out = run(make_batch(landmarks=landmarks), cfg)
    points, _, _ = split(out)
    assert out.shape == (16, 32, 2 * landmarks + 2)
    assert (points[:, :, :21] == 0.0).all()


@pytest.mark.parametrize("name", [*ONLY, "v2_recipe"])
def test_same_seed_repeats_and_another_seed_differs(name):
    cfg = V2_RECIPE if name == "v2_recipe" else ONLY[name]
    x = make_batch()
    assert torch.equal(run(x, cfg, seed=3), run(x, cfg, seed=3))
    assert not torch.equal(run(x, cfg, seed=3), run(x, cfg, seed=4))


def test_probability_zero_is_the_identity_and_draws_no_random_numbers():
    x = make_batch()
    assert run(x, NONE) is x
    changed_magnitudes = V1_ONLY.model_copy(
        update={
            "frame_dropout_max": 0.5,
            "time_mask_windows": 5,
            "landmark_mask_max": 9,
            "hand_rotation_deg": 45.0,
        }
    )
    # The v2 transforms at probability 0 leave the v1 transforms' random stream untouched.
    assert torch.equal(run(x, changed_magnitudes, seed=7), run(x, V1_ONLY, seed=7))


def test_hand_rotation_is_rigid_about_each_wrist_and_keeps_missing_points():
    x = make_batch()
    points, _, missing = split(x)
    generator = torch.Generator().manual_seed(2)
    rotated = hand_rotation(points, missing, ONLY["hand_rotation"], generator)
    assert torch.equal((rotated == 0.0).all(dim=-1), missing)
    right = slice(21, 42)
    wrist_present = ~missing[:, :, 21]
    before = points[:, :, right] - points[:, :, 21:22]
    after = rotated[:, :, right] - rotated[:, :, 21:22]
    both = wrist_present[..., None] & ~missing[:, :, right]
    torch.testing.assert_close(before.norm(dim=-1)[both], after.norm(dim=-1)[both])
    assert torch.equal(rotated[:, :, 21], points[:, :, 21])  # Wrists stay put.
    assert torch.equal(rotated[:, :, 42:], points[:, :, 42:])  # Pose points are untouched.
    both[:, :, 0] = False  # The wrist itself has no direction to turn.
    cosine = (before * after).sum(-1)[both] / before.norm(dim=-1)[both].square()
    assert cosine.min() >= math.cos(math.radians(10.0)) - 1e-5  # At most 10 degrees.
    assert cosine.max() < 1.0  # Something did turn.


def test_frame_dropout_keeps_the_missing_rule_and_resamples_presence():
    x = make_batch()
    points, presence, missing = split(x)
    generator = torch.Generator().manual_seed(5)
    out_points, out_presence, out_missing = frame_dropout(
        points, presence, missing, ONLY["frame_dropout"], generator
    )
    assert torch.equal((out_points == 0.0).all(dim=-1), out_missing)
    assert set(out_presence.unique().tolist()) <= set(presence.unique().tolist())


def test_frame_dropout_resamples_the_kept_frames_in_order():
    """Every point holds its frame index + 1, so the output shows which frames were kept."""
    t = torch.arange(1.0, 33.0)
    points = t[None, :, None, None].expand(64, 32, 59, 2).clone()
    presence = torch.ones(64, 32, 2)
    missing = torch.zeros(64, 32, 59, dtype=torch.bool)
    cfg = ONLY["frame_dropout"]
    out, _, out_missing = frame_dropout(
        points, presence, missing, cfg, torch.Generator().manual_seed(0)
    )
    values = out[:, :, 0, 0]
    assert not out_missing.any()
    assert (values[:, 1:] >= values[:, :-1]).all()  # Time order is kept.
    assert (values >= 1.0).all() and (values <= 32.0).all()
    dropped_somewhere = (values[:, 0] > 1.0) | (values[:, -1] < 32.0)
    assert dropped_somewhere.any()
    none_dropped = cfg.model_copy(update={"frame_dropout_max": 0.01})  # floor(0.01 * 32) = 0.
    same, _, _ = frame_dropout(points, presence, missing, none_dropped, torch.Generator())
    assert torch.equal(same, points)


def test_time_mask_blanks_whole_frames_within_the_limits():
    x = make_batch(batch=64)
    points, presence, missing = split(x)
    cfg = ONLY["time_mask"]
    out_points, out_presence, out_missing = time_mask(
        points, presence, missing, cfg, torch.Generator().manual_seed(1)
    )
    blanked = out_missing.all(dim=-1) & ~missing.all(dim=-1)
    assert blanked.any(dim=1).all()  # Probability 1: every sample loses frames.
    assert (blanked.sum(dim=1) <= cfg.time_mask_windows * cfg.time_mask_max_frames).all()
    assert (out_points[blanked] == 0.0).all()
    assert (out_presence[blanked] == 0.0).all()
    kept = ~blanked
    assert torch.equal(out_points[kept], points[kept])
    assert torch.equal(out_presence[kept], presence[kept])


def test_landmark_mask_removes_whole_landmarks_but_never_the_hand_wrists():
    generator = torch.Generator().manual_seed(0)
    points = torch.rand(64, 32, 59, 2, generator=generator) + 0.5  # Every point present.
    missing = torch.zeros(64, 32, 59, dtype=torch.bool)
    cfg = ONLY["landmark_mask"]
    out, out_missing = landmark_mask(points, missing, cfg, torch.Generator().manual_seed(2))
    masked = out_missing.any(dim=1)  # (B, L): the landmarks chosen per sample.
    assert torch.equal(out_missing, masked[:, None, :].expand_as(out_missing))  # Whole sequence.
    counts = masked.sum(dim=1)
    assert ((counts >= 1) & (counts <= cfg.landmark_mask_max)).all()
    assert len(set(counts.tolist())) > 1
    assert not masked[:, [0, 21]].any()  # Never a hand wrist.
    assert (out[out_missing] == 0.0).all()
    assert torch.equal(out[~out_missing], points[~out_missing])
