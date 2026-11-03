import pytest
import torch

from edgevision.train.augment import augment_batch, temporal
from edgevision.train.config import AugmentConfig

OFF = AugmentConfig(affine_p=0.0, temporal_p=0.0, jitter_p=0.0)
AFFINE = AugmentConfig(affine_p=1.0, temporal_p=0.0, jitter_p=0.0)
TEMPORAL = AugmentConfig(affine_p=0.0, temporal_p=1.0, jitter_p=0.0)
JITTER = AugmentConfig(affine_p=0.0, temporal_p=0.0, jitter_p=1.0)
ALL_ON = AugmentConfig(affine_p=1.0, temporal_p=1.0, jitter_p=1.0)
CONFIGS = {"default": AugmentConfig(), "all": ALL_ON, "affine": AFFINE, "temporal": TEMPORAL}
CONFIGS["jitter"] = JITTER


def make_batch(batch=16, seed=0):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, 32, 100, generator=generator)
    points = x[..., :98].view(batch, 32, 49, 2)
    missing = torch.rand(batch, 32, 49, generator=generator) < 0.15
    missing[:, :, :21] = True  # Left hand absent throughout.
    missing[0, 10:20] = True  # Whole frames missing.
    points[missing] = 0.0
    x[..., 98:] = (torch.rand(batch, 32, 2, generator=generator) < 0.7).float()
    return x


def points_of(x):
    return x[..., :98].reshape(x.shape[0], 32, 49, 2)


def missing_of(x):
    return (points_of(x) == 0.0).all(dim=-1)


def run(x, cfg, seed=1):
    return augment_batch(x, cfg, torch.Generator().manual_seed(seed))


@pytest.mark.parametrize("name", list(CONFIGS))
def test_shapes_and_dtype_are_unchanged(name):
    x = make_batch()
    out = run(x, CONFIGS[name])
    assert out.shape == x.shape
    assert out.dtype == x.dtype


@pytest.mark.parametrize("name", ["affine", "jitter"])
def test_spatial_transforms_keep_missing_points_exactly_zero(name):
    x = make_batch()
    out = run(x, CONFIGS[name])
    missing = missing_of(x)
    assert (points_of(out)[missing] == 0.0).all()
    assert torch.equal(missing_of(out), missing)


@pytest.mark.parametrize("name", list(CONFIGS))
def test_points_missing_throughout_stay_zero(name):
    out = run(make_batch(), CONFIGS[name])
    assert (points_of(out)[:, :, :21] == 0.0).all()


def test_temporal_never_blends_with_a_missing_endpoint():
    x = torch.zeros(64, 32, 100)
    points = points_of(x)
    points[:, 0::2, 21] = 1.0  # Present on even frames, missing on odd frames.
    points[:, :, 22] = 1.0  # Present everywhere.
    out = points_of(run(x, TEMPORAL))
    values = set(out[:, :, 21].unique().tolist())
    assert values <= {0.0, 1.0}  # A blend would give values in between.
    assert 0.0 in values
    assert (out[:, :, 22] == 1.0).all()


def test_full_window_temporal_is_the_identity():
    x = make_batch()
    cfg = TEMPORAL.model_copy(update={"window_min": 1.0})
    out = run(x, cfg)
    assert torch.equal(out, x)


def test_temporal_resamples_presence_by_nearest_frame():
    x = make_batch()
    x[..., 98] = torch.arange(32.0) % 2  # Alternating presence.
    out = run(x, TEMPORAL)
    assert set(out[..., 98:].unique().tolist()) <= {0.0, 1.0}
    assert not torch.equal(out[..., 98], x[..., 98])


@pytest.mark.parametrize("name", ["affine", "jitter"])
def test_only_the_temporal_transform_touches_presence(name):
    x = make_batch()
    assert torch.equal(run(x, CONFIGS[name])[..., 98:], x[..., 98:])


def test_affine_rotates_and_scales_about_the_origin():
    x = make_batch()
    cfg = AFFINE.model_copy(update={"scale_min": 1.1, "scale_max": 1.1, "translate": 0.0})
    out = run(x, cfg)
    present = ~missing_of(x)
    before = points_of(x)[present].norm(dim=-1)
    after = points_of(out)[present].norm(dim=-1)
    torch.testing.assert_close(after, 1.1 * before, rtol=1e-5, atol=1e-6)
    assert not torch.allclose(points_of(out)[present], 1.1 * points_of(x)[present])


def test_jitter_noise_has_the_configured_scale():
    x = make_batch(batch=256)
    present = ~missing_of(x)
    noise = (points_of(run(x, JITTER)) - points_of(x))[present]
    assert noise.std().item() == pytest.approx(0.01, rel=0.05)


def test_each_sample_is_chosen_with_the_configured_probability():
    x = make_batch(batch=400)
    out = run(x, AFFINE.model_copy(update={"affine_p": 0.5}))
    changed = (out != x).flatten(1).any(dim=1).float().mean().item()
    assert 0.4 < changed < 0.6


def test_same_seed_gives_identical_output_and_another_seed_does_not():
    x = make_batch()
    first, second = run(x, ALL_ON, seed=3), run(x, ALL_ON, seed=3)
    assert torch.equal(first, second)
    assert not torch.equal(first, run(x, ALL_ON, seed=4))


def test_all_probabilities_zero_is_the_identity():
    x = make_batch()
    assert torch.equal(run(x, OFF), x)


def test_input_batch_is_not_modified():
    x = make_batch()
    original = x.clone()
    run(x, ALL_ON)
    assert torch.equal(x, original)


def test_temporal_returns_the_resampled_missing_mask():
    x = make_batch()
    points, presence = points_of(x), x[..., 98:]
    moved, _, missing = temporal(
        points, presence, missing_of(x), TEMPORAL, torch.Generator().manual_seed(0)
    )
    assert torch.equal((moved == 0.0).all(dim=-1), missing)
