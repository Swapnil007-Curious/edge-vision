import numpy as np
import pandas as pd
import pytest

from conftest import landmark_blocks, long_frame
from edgevision.data.landmarks import (
    DEFAULT_POSE_INDICES,
    LEFT_HAND,
    POSE,
    RIGHT_HAND,
    mirror_permutation,
    num_selected,
    pose_slot,
    shoulder_slots,
)
from edgevision.data.preprocess import (
    fill_missing,
    hand_presence,
    left_is_dominant,
    mirror_sequence,
    normalization_params,
    pivot_sequence,
    preprocess_sequence,
    resample_coords,
    resample_presence,
    snap_x,
)

L = 49


def test_selected_landmark_layout():
    assert num_selected() == L
    assert (LEFT_HAND.start, LEFT_HAND.stop) == (0, 21)
    assert (RIGHT_HAND.start, RIGHT_HAND.stop) == (21, 42)
    assert POSE.start == 42
    assert pose_slot(DEFAULT_POSE_INDICES, 11) == 43
    assert shoulder_slots(DEFAULT_POSE_INDICES) == (43, 44)


def test_pivot_orders_by_frame_then_selected_landmark():
    df = pd.DataFrame(
        {
            "frame": [5, 5, 2, 5],
            "row_id": ["a", "b", "c", "d"],
            "type": ["pose", "right_hand", "left_hand", "face"],
            "landmark_index": [12, 3, 20, 0],
            "x": [0.1, 0.2, 0.3, 0.4],
            "y": [0.5, 0.6, 0.7, 0.8],
            "z": [0.0, 0.0, 0.0, 0.0],
        }
    )
    coords = pivot_sequence(df)
    assert coords.shape == (2, L, 3)
    assert coords[0, 20, 1] == 0.7  # Frame 2 sorts first; left_hand 20 is slot 20.
    assert coords[1, 24, 1] == 0.6  # right_hand 3 is slot 21 + 3.
    assert coords[1, 44, 1] == 0.5  # pose 12 is the third selected pose point.
    assert np.isfinite(coords).sum() == 9  # Three points, nothing from the face row.


def test_pivot_missing_rows_become_nan(rng):
    blocks = landmark_blocks(12, rng)
    del blocks["pose"]
    coords = pivot_sequence(long_frame(blocks))
    assert coords.shape == (12, L, 3)
    assert np.isnan(coords[:, POSE]).all()
    assert np.isfinite(coords[:, LEFT_HAND]).all()


def test_pivot_duplicate_rows_raise(rng):
    df = long_frame(landmark_blocks(10, rng))
    with pytest.raises(ValueError, match="duplicate"):
        pivot_sequence(pd.concat([df, df.iloc[[0]]], ignore_index=True))


def test_pivot_duplicate_face_rows_also_raise(rng):
    df = long_frame(landmark_blocks(10, rng))
    face_row = df.loc[df["type"] == "face"].iloc[[0]]
    with pytest.raises(ValueError, match="duplicate"):
        pivot_sequence(pd.concat([df, face_row], ignore_index=True))


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("type", "torso", "unknown landmark types"),
        ("landmark_index", 21, "valid range"),
        ("x", np.inf, "infinite"),
    ],
)
def test_pivot_rejects_contract_violations(rng, column, value, message):
    df = long_frame(landmark_blocks(10, rng))
    df[column] = df[column].astype(object)
    target = df.index[df["type"] == "left_hand"][0]
    df.loc[target, column] = value
    df["landmark_index"] = df["landmark_index"].astype(int)
    df["x"] = df["x"].astype(float)
    with pytest.raises(ValueError, match=message):
        pivot_sequence(df)


def test_hand_presence_needs_every_point_finite():
    coords = np.full((3, L, 3), 0.5)
    coords[0, 5, 0] = np.nan  # One left-hand x missing.
    coords[1, 30, 1] = np.nan  # One right-hand y missing.
    coords[2, 5, 2] = np.nan  # z does not count.
    presence = hand_presence(coords)
    assert presence.tolist() == [[False, True], [True, False], [True, True]]


@pytest.mark.parametrize(
    ("presence", "expected"),
    [
        ([[True, False], [True, False], [False, True]], True),
        ([[True, False], [False, True], [False, True]], False),
        ([[True, False], [False, True]], False),  # Ties go to the right hand.
        ([[True, True], [True, True]], False),
    ],
)
def test_left_is_dominant(presence, expected):
    assert left_is_dominant(np.array(presence)) is expected


def test_mirror_twice_is_exactly_identity(rng):
    coords = rng.uniform(-3.0, 4.0, (40, L, 3))
    coords[rng.random(coords.shape) < 0.1] = np.nan
    coords[0, 0, 0] = 1e-12  # Tiny values are where a plain 1 - x loses bits.
    coords[..., 0] = snap_x(coords[..., 0])

    twice = mirror_sequence(mirror_sequence(coords))

    assert twice.tobytes() == coords.tobytes()


def test_unsnapped_mirror_would_not_round_trip():
    """Documents why pivot_sequence snaps x: this is the failure it prevents."""
    x = 0.1
    assert 1.0 - (1.0 - x) != x
    snapped = float(snap_x(np.array(x)))
    assert 1.0 - (1.0 - snapped) == snapped


def test_pivoted_real_path_mirror_round_trips(rng):
    coords = pivot_sequence(long_frame(landmark_blocks(25, rng)))
    assert mirror_sequence(mirror_sequence(coords)).tobytes() == coords.tobytes()


def test_mirror_swaps_blocks_and_pose_pairs():
    coords = np.arange(3.0 * L).reshape(1, L, 3) / (3.0 * L)
    mirrored = mirror_sequence(coords)
    np.testing.assert_array_equal(mirrored[:, RIGHT_HAND, 1], coords[:, LEFT_HAND, 1])
    np.testing.assert_array_equal(mirrored[:, LEFT_HAND, 1], coords[:, RIGHT_HAND, 1])
    nose, left_shoulder, right_shoulder = 42, 43, 44
    assert mirrored[0, nose, 1] == coords[0, nose, 1]
    assert mirrored[0, left_shoulder, 1] == coords[0, right_shoulder, 1]
    assert mirrored[0, 45, 1] == coords[0, 46, 1]  # Elbows.
    assert mirrored[0, 47, 1] == coords[0, 48, 1]  # Wrists.
    assert mirrored[0, 0, 0] == 1.0 - coords[0, 21, 0]


def test_mirror_permutation_rejects_half_a_pair():
    with pytest.raises(ValueError, match="both or neither"):
        mirror_permutation((0, 11, 12, 13))


@pytest.mark.parametrize("use_z", [False, True])
@pytest.mark.parametrize("n_frames", [10, 32, 60])
def test_output_shapes_dtypes_and_finiteness(rng, use_z, n_frames):
    df = long_frame(landmark_blocks(n_frames, rng))
    features, presence, info = preprocess_sequence(df, use_z=use_z)
    assert features.shape == (32, L, 3 if use_z else 2)
    assert features.dtype == np.float32
    assert presence.shape == (32, 2)
    assert presence.dtype == np.float32
    assert np.isfinite(features).all()
    assert set(info) == {"t_raw", "t_kept", "mirrored", "norm_method", "excluded", "reason"}
    assert (info["t_raw"], info["t_kept"], info["excluded"]) == (n_frames, n_frames, False)


def test_input_frame_is_not_mutated(rng):
    df = long_frame(landmark_blocks(15, rng, "left_hand_only"))
    before = df.copy(deep=True)
    preprocess_sequence(df)
    pd.testing.assert_frame_equal(df, before)


def test_left_dominant_signal_ends_up_in_right_block(rng):
    df = long_frame(landmark_blocks(20, rng, "left_hand_only"))
    features, presence, info = preprocess_sequence(df)
    assert info["mirrored"] is True
    assert (features[:, LEFT_HAND] == 0.0).all()
    assert np.abs(features[:, RIGHT_HAND]).sum() > 0.0
    # The presence columns must follow the swapped blocks.
    assert (presence[:, 0] == 0.0).all()
    assert (presence[:, 1] == 1.0).all()


def test_right_dominant_is_left_alone(rng):
    df = long_frame(landmark_blocks(20, rng, "right_hand_only"))
    features, presence, info = preprocess_sequence(df)
    assert info["mirrored"] is False
    assert np.abs(features[:, RIGHT_HAND]).sum() > 0.0
    assert (presence[:, 1] == 1.0).all()


def test_shoulder_normalization_at_median_frame(rng):
    n_frames = 41
    blocks = landmark_blocks(n_frames, rng)
    # Skewed monotonic drift and widening: the middle frame is the median but not the mean.
    ramp = np.linspace(0.0, 1.0, n_frames) ** 3
    drift, widening = 0.08 * ramp, 0.05 * ramp
    blocks["pose"][:, 11, 0] = 0.70 + drift + widening
    blocks["pose"][:, 11, 1] = 0.52 + drift
    blocks["pose"][:, 12, 0] = 0.30 + drift - widening
    blocks["pose"][:, 12, 1] = 0.52 + drift

    features, _, info = preprocess_sequence(long_frame(blocks), num_frames=n_frames)

    assert info["norm_method"] == "shoulders"
    left, right = features[n_frames // 2, 43], features[n_frames // 2, 44]
    np.testing.assert_allclose((left + right) / 2.0, (0.0, 0.0), atol=1e-4)
    assert np.hypot(*(left - right)) == pytest.approx(1.0, abs=1e-4)


def test_shoulder_normalization_after_mirroring(rng):
    n_frames = 21
    blocks = landmark_blocks(n_frames, rng, "left_hand_only")
    features, _, info = preprocess_sequence(long_frame(blocks), num_frames=n_frames)
    assert info["mirrored"] is True
    assert info["norm_method"] == "shoulders"
    left, right = features[n_frames // 2, 43], features[n_frames // 2, 44]
    np.testing.assert_allclose((left + right) / 2.0, (0.0, 0.0), atol=1e-4)
    assert np.hypot(*(left - right)) == pytest.approx(1.0, abs=1e-4)


def test_shoulders_finite_in_a_single_frame_are_enough(rng):
    blocks = landmark_blocks(15, rng)
    blocks["pose"][1:, [11, 12]] = np.nan
    _, _, info = preprocess_sequence(long_frame(blocks))
    assert info["norm_method"] == "shoulders"


@pytest.mark.parametrize("pose_rows", ["nan", "absent"])
def test_fallback_normalization_when_pose_missing(rng, pose_rows):
    blocks = landmark_blocks(20, rng, "pose_missing")
    if pose_rows == "absent":
        del blocks["pose"]
    features, _, info = preprocess_sequence(long_frame(blocks), num_frames=20)

    assert info["norm_method"] == "hand_fallback"
    hand_xy = features[:, :42].reshape(-1, 2)
    np.testing.assert_allclose(hand_xy.mean(axis=0), (0.0, 0.0), atol=1e-5)
    spread = hand_xy.max(axis=0) - hand_xy.min(axis=0)
    assert spread.max() == pytest.approx(1.0, abs=1e-5)


def test_fallback_when_shoulders_are_not_selected(rng):
    blocks = landmark_blocks(20, rng)
    pose_indices = (0, 13, 14)
    features, _, info = preprocess_sequence(long_frame(blocks), pose_indices=pose_indices)
    assert info["norm_method"] == "hand_fallback"
    assert features.shape == (32, 45, 2)


def test_z_is_scaled_but_not_centered(rng):
    n_frames = 21
    blocks = landmark_blocks(n_frames, rng)
    df = long_frame(blocks)
    features, _, info = preprocess_sequence(df, num_frames=n_frames, use_z=True)
    assert info["norm_method"] == "shoulders"
    shoulder_distance = 0.30  # Set by the fixture.
    expected = blocks["right_hand"][..., 2] / shoulder_distance
    np.testing.assert_allclose(features[:, RIGHT_HAND, 2], expected, atol=1e-5)


def test_normalization_params_fallback_values():
    coords = np.full((2, L, 3), np.nan)
    coords[0, :21, 0] = 0.2
    coords[0, :21, 1] = 0.4
    coords[1, :21, 0] = 0.6
    coords[1, :21, 1] = 0.5
    center, scale, method = normalization_params(coords)
    assert method == "hand_fallback"
    np.testing.assert_allclose(center, (0.4, 0.45))
    assert scale == pytest.approx(0.4)


@pytest.mark.parametrize("num_frames", [8, 32, 90])
@pytest.mark.parametrize("n_frames", [10, 45])
def test_resampling_yields_exact_length(rng, num_frames, n_frames):
    df = long_frame(landmark_blocks(n_frames, rng))
    features, presence, _ = preprocess_sequence(df, num_frames=num_frames)
    assert features.shape[0] == num_frames
    assert presence.shape[0] == num_frames


@pytest.mark.parametrize("num_frames", [8, 32, 90])
def test_linear_ramp_stays_linear(num_frames):
    ramp = np.linspace(2.0, 7.0, 17)
    coords = ramp[:, None, None]
    resampled = resample_coords(coords, num_frames)
    np.testing.assert_allclose(resampled[:, 0, 0], np.linspace(2.0, 7.0, num_frames))


def test_resampling_bridges_gaps_but_never_extrapolates():
    series = np.array([np.nan, 1.0, np.nan, 3.0, np.nan])
    resampled = resample_coords(series[:, None, None], 5)[:, 0, 0]
    np.testing.assert_array_equal(resampled[1:4], [1.0, 2.0, 3.0])
    assert np.isnan(resampled[0])
    assert np.isnan(resampled[4])


def test_resampling_keeps_never_finite_points_nan():
    coords = np.full((6, 2, 2), np.nan)
    coords[:, 0, :] = 1.0
    resampled = resample_coords(coords, 4)
    assert np.isnan(resampled[:, 1]).all()
    assert (resampled[:, 0] == 1.0).all()


def test_resampling_handles_a_single_kept_frame():
    resampled = resample_coords(np.full((1, 3, 2), 0.25), 5)
    assert resampled.shape == (5, 3, 2)
    assert (resampled == 0.25).all()


def test_presence_is_resampled_with_nearest_neighbour():
    presence = np.array([[True, False], [False, True], [False, False]])
    resampled = resample_presence(presence, 5)
    assert resampled.tolist() == [
        [True, False],
        [False, True],
        [False, True],
        [False, False],
        [False, False],
    ]


def test_fill_missing_replaces_nan_and_refuses_non_finite():
    filled = fill_missing(np.array([[np.nan, 1.5]]))
    assert filled.dtype == np.float32
    assert filled.tolist() == [[0.0, 1.5]]
    with np.errstate(over="ignore"), pytest.raises(AssertionError, match="non-finite"):
        fill_missing(np.array([np.inf]))


def test_partial_hands_keep_only_frames_with_a_hand(rng):
    df = long_frame(landmark_blocks(30, rng, "partial_hands"))
    _, presence, info = preprocess_sequence(df)
    assert info["t_raw"] == 30
    assert info["t_kept"] == 20  # The first third has no hand at all.
    assert (presence[:, 1] == 1.0).all()


def test_exclusion_insufficient_hand_frames(rng):
    features, presence, info = preprocess_sequence(long_frame(landmark_blocks(20, rng, "no_hands")))
    assert features is None
    assert presence is None
    assert info["excluded"] is True
    assert info["reason"] == "insufficient_hand_frames"
    assert (info["t_raw"], info["t_kept"]) == (20, 0)


def test_min_hand_frames_boundary(rng):
    blocks = landmark_blocks(10, rng, "right_hand_only")
    blocks["right_hand"][3:] = np.nan
    df = long_frame(blocks)
    assert preprocess_sequence(df, min_hand_frames=3).info["excluded"] is False
    assert preprocess_sequence(df, min_hand_frames=4).info["excluded"] is True


def test_exclusion_degenerate_scale(rng):
    df = long_frame(landmark_blocks(20, rng, "coincident_shoulders"))
    features, _, info = preprocess_sequence(df)
    assert features is None
    assert info["excluded"] is True
    assert info["reason"] == "degenerate_scale"
    assert info["norm_method"] == "shoulders"


def test_exclusion_degenerate_scale_with_hand_fallback(rng):
    blocks = landmark_blocks(10, rng, "right_hand_only")
    blocks["pose"][:] = np.nan
    blocks["right_hand"][:] = 0.5
    _, _, info = preprocess_sequence(long_frame(blocks))
    assert info["reason"] == "degenerate_scale"
    assert info["norm_method"] == "hand_fallback"


def test_empty_sequence_is_excluded_not_a_crash(rng):
    df = long_frame(landmark_blocks(10, rng)).iloc[:0]
    assert preprocess_sequence(df).info["reason"] == "insufficient_hand_frames"
