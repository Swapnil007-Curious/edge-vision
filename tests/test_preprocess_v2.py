"""v2 landmark selection: derived swap pairs, mirroring, the recorded selection, aspect stats and
splits identical to v1's."""

import json

import numpy as np
import pandas as pd
import pytest

from conftest import landmark_blocks, long_frame
from edgevision.config import load_config
from edgevision.data.io import aspect_stats
from edgevision.data.landmarks import (
    DEFAULT_POSE_INDICES,
    POSE_SWAP_PAIRS,
    mirror_permutation,
    num_selected,
    pose_slot,
    pose_swap_pairs,
)
from edgevision.data.preprocess import (
    AspectMeasurement,
    aspect_measurement,
    mirror_sequence,
    pivot_sequence,
    preprocess_sequence_measured,
    snap_x,
)
from training_support import EXPERIMENTS
from v1_baseline import run_v1_preprocessing

V2 = tuple(range(17))
V2_PAIRS = ((1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16))
# The v1 permutation as it was hard-coded before the pairs were derived: hands swapped, then
# shoulders (43, 44), elbows (45, 46) and wrists (47, 48).
V1_PERMUTATION = [*range(21, 42), *range(21), 42, 44, 43, 46, 45, 48, 47]


def test_preprocess_v2_config_is_the_default_with_pose_0_to_16():
    default = load_config(EXPERIMENTS / "preprocess_default.yaml")
    v2 = load_config(EXPERIMENTS / "preprocess_v2.yaml")
    assert v2.preprocess.pose_indices == list(V2)
    assert num_selected(v2.preprocess.pose_indices) == 59
    assert str(v2.preprocess.output_dir).replace("\\", "/") == "/kaggle/working/processed_v2"
    assert v2.split == default.split
    unchanged = default.preprocess.model_dump(exclude={"pose_indices", "output_dir"})
    assert v2.preprocess.model_dump(exclude={"pose_indices", "output_dir"}) == unchanged


def test_swap_pairs_are_derived_from_the_selection():
    assert POSE_SWAP_PAIRS == V2_PAIRS
    assert pose_swap_pairs(DEFAULT_POSE_INDICES) == ((11, 12), (13, 14), (15, 16))
    assert pose_swap_pairs(V2) == V2_PAIRS
    assert pose_swap_pairs((0, 9, 10, 11, 12)) == ((9, 10), (11, 12))
    assert mirror_permutation(DEFAULT_POSE_INDICES).tolist() == V1_PERMUTATION


@pytest.mark.parametrize("half", [(0, 1, 11, 12), (0, 9, 11, 12), (0, 2, 5, 3, 11, 12)])
def test_a_selection_with_half_a_pair_is_refused(half):
    with pytest.raises(ValueError, match="both or neither"):
        mirror_permutation(half)


def test_v2_mirror_swaps_every_selected_pair():
    count = num_selected(V2)
    coords = np.arange(count * 3.0).reshape(1, count, 3) / (3.0 * count)
    mirrored = mirror_sequence(coords, V2)
    for first, second in V2_PAIRS:
        a, b = pose_slot(V2, first), pose_slot(V2, second)
        assert mirrored[0, a, 1] == coords[0, b, 1]
        assert mirrored[0, b, 1] == coords[0, a, 1]
    nose = pose_slot(V2, 0)
    assert mirrored[0, nose, 1] == coords[0, nose, 1]
    assert mirrored[0, nose, 0] == 1.0 - coords[0, nose, 0]


def test_v2_mirror_twice_is_exactly_identity(rng):
    coords = rng.uniform(-3.0, 4.0, (40, num_selected(V2), 3))
    coords[rng.random(coords.shape) < 0.1] = np.nan
    coords[..., 0] = snap_x(coords[..., 0])
    assert mirror_sequence(mirror_sequence(coords, V2), V2).tobytes() == coords.tobytes()

    pivoted = pivot_sequence(long_frame(landmark_blocks(25, rng)), V2)
    assert mirror_sequence(mirror_sequence(pivoted, V2), V2).tobytes() == pivoted.tobytes()


def crafted_sequence(rng):
    """Six frames with known geometry: shoulders 0.2 apart at y = 0.5, the nose above by
    0.1 * (frame + 1), so r = 0.5, 1.0, ... Frame 3 loses the nose, frame 4 a shoulder, and
    frame 5 both hands (so it is not a kept frame)."""
    blocks = landmark_blocks(6, rng)
    pose = blocks["pose"]
    pose[:, 11, :2] = (0.6, 0.5)
    pose[:, 12, :2] = (0.4, 0.5)
    pose[:, 0, 0] = 0.5
    pose[:, 0, 1] = 0.5 - 0.1 * np.arange(1, 7)
    pose[3, 0] = np.nan
    pose[4, 12] = np.nan
    blocks["left_hand"][5] = np.nan
    blocks["right_hand"][5] = np.nan
    return long_frame(blocks)


@pytest.mark.parametrize("pose_indices", [DEFAULT_POSE_INDICES, V2])
def test_aspect_measurement_on_a_crafted_sequence(rng, pose_indices):
    _, measured = preprocess_sequence_measured(crafted_sequence(rng), pose_indices=pose_indices)
    np.testing.assert_allclose(measured.ratios, [0.5, 1.0, 1.5], rtol=1e-9)
    assert (measured.shoulder_frames, measured.kept_frames) == (4, 5)

    stats = aspect_stats(measured)
    for name, percent in (("p10", 10), ("median", 50), ("p90", 90)):
        assert stats[name] == pytest.approx(np.percentile([0.5, 1.0, 1.5], percent))
    assert stats["median"] == pytest.approx(1.0)
    assert stats["frames_measured"] == 3
    assert stats["shoulders_finite_fraction"] == pytest.approx(4 / 5)


def test_aspect_measurement_needs_the_nose_and_shoulders(rng):
    coords = pivot_sequence(crafted_sequence(rng), (11, 12))
    no_nose = aspect_measurement(coords[:5], (11, 12))
    assert (no_nose.ratios.size, no_nose.shoulder_frames) == (0, 4)
    stats = aspect_stats(AspectMeasurement(np.empty(0), 0, 0))
    assert (stats["median"], stats["shoulders_finite_fraction"]) == (None, None)


@pytest.fixture(scope="module")
def preprocessed(tmp_path_factory):
    """The default synthetic dataset through scripts/preprocess.py with the v1 and v2 settings."""
    v1 = run_v1_preprocessing(tmp_path_factory.mktemp("v1"))
    root = tmp_path_factory.mktemp("v2")
    data_root = v1.parent / "asl-signs"
    v2_config = (EXPERIMENTS / "preprocess_v2.yaml").read_text(encoding="utf-8")
    v2_config = v2_config.replace("/kaggle/input/asl-signs", data_root.as_posix())
    v2_config = v2_config.replace("/kaggle/working/processed_v2", (root / "processed").as_posix())
    v2_config = v2_config.replace("num_workers: 4", "num_workers: 1")
    v2 = run_v1_preprocessing(root, config_text=v2_config)
    return v1, v2


def read(directory, name):
    return json.loads((directory / name).read_text(encoding="utf-8"))


def test_v2_report_records_the_selection_and_aspect_stats(preprocessed):
    _, v2 = preprocessed
    report = read(v2, "preprocess_report.json")
    assert report["landmarks"] == {
        "num_landmarks": 59,
        "landmark_order": ["left_hand_0-20", "right_hand_0-20", "pose_indices"],
        "pose_indices": list(V2),
        "pose_swap_pairs": [list(pair) for pair in V2_PAIRS],
    }
    assert report["aspect_stats"]["frames_measured"] > 0
    with np.load(v2 / "features.npz") as data:
        assert data["X"].shape[1:] == (32, 59, 2)


def test_v2_split_holds_the_same_participants_and_sequences_as_v1(preprocessed):
    v1, v2 = preprocessed
    splits_v1 = read(v1, "splits_signer_independent.json")
    splits_v2 = read(v2, "splits_signer_independent.json")
    assert splits_v2["participants"] == splits_v1["participants"]
    for name in ("train", "val", "test"):
        assert splits_v2[name] == splits_v1[name]
    assert read(v2, "preprocess_report.json")["aspect_stats"] == pytest.approx(
        read(v1, "preprocess_report.json")["aspect_stats"]
    )


def test_v2_features_match_v1_on_the_landmarks_both_select(preprocessed):
    """Mirroring and normalisation treat the shared landmarks identically, so v2 holds v1's
    features plus the new pose points."""
    v1, v2 = preprocessed
    pd.testing.assert_frame_equal(
        pd.read_csv(v1 / "manifest.csv", keep_default_na=False),
        pd.read_csv(v2 / "manifest.csv", keep_default_na=False),
    )
    shared = list(range(42)) + [pose_slot(V2, index) for index in DEFAULT_POSE_INDICES]
    with np.load(v1 / "features.npz") as old, np.load(v2 / "features.npz") as new:
        np.testing.assert_array_equal(new["X"][:, :, shared], old["X"])
        np.testing.assert_array_equal(new["presence"], old["presence"])
