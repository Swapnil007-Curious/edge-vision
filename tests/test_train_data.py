import ast
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import SIGNS
from edgevision.config import SplitConfig
from edgevision.layout import (
    FRAME_DIM,
    NUM_COORDS,
    NUM_FRAMES,
    PRESENCE_LEFT,
    PRESENCE_RIGHT,
    to_model_input,
)
from edgevision.train.data import (
    TEST_ACCESS_MESSAGE,
    label_map_from_manifest,
    load_split_arrays,
    prepare_splits,
    read_manifest,
)
from training_support import REPO_ROOT, build_processed_dir, snapshot

SPLITS_FILE = "splits_signer_independent.json"


@pytest.fixture(scope="module")
def processed(tmp_path_factory):
    return build_processed_dir(tmp_path_factory.mktemp("processed"))


def owners(manifest, sequence_ids):
    return set(manifest.set_index("sequence_id").loc[sequence_ids, "participant_id"].tolist())


def test_layout_constants():
    assert (NUM_FRAMES, NUM_COORDS, FRAME_DIM) == (32, 98, 100)
    assert (PRESENCE_LEFT, PRESENCE_RIGHT) == (98, 99)


def test_model_input_is_landmark_major_then_presence():
    coords = np.zeros((1, 32, 49, 2), dtype=np.float16)
    coords[0, 3] = np.arange(98, dtype=np.float16).reshape(49, 2)
    presence = np.zeros((1, 32, 2), dtype=np.uint8)
    presence[0, 3] = (1, 0)
    packed = to_model_input(coords, presence)
    assert packed.dtype == np.float32
    assert packed.shape == (1, 32, 100)
    # [l0x, l0y, l1x, l1y, ...] then presence_left, presence_right.
    assert packed[0, 3, :4].tolist() == [0.0, 1.0, 2.0, 3.0]
    assert packed[0, 3, 96:].tolist() == [96.0, 97.0, 1.0, 0.0]


def test_model_input_rejects_other_shapes():
    with pytest.raises(ValueError, match="expected coordinates"):
        to_model_input(np.zeros((2, 32, 49, 3), np.float16), np.zeros((2, 32, 2), np.uint8))


def test_test_split_is_refused_before_anything_is_read():
    with pytest.raises(ValueError, match=re.escape(TEST_ACCESS_MESSAGE)):
        load_split_arrays(Path("missing-dir"), Path("missing.json"), ["val", "test"])


def unlocks_test_split(path):
    """True if the file contains a call passing allow_test=True."""
    return any(
        isinstance(node, ast.Call)
        and any(
            keyword.arg == "allow_test"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is True
            for keyword in node.keywords
        )
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
    )


def test_only_final_eval_unlocks_the_test_split():
    offenders = sorted(
        path.relative_to(REPO_ROOT).as_posix()
        for folder in ("src", "scripts")
        for path in (REPO_ROOT / folder).rglob("*.py")
        if unlocks_test_split(path)
    )
    assert offenders == ["scripts/final_eval.py"]


def test_unknown_split_name_raises(processed):
    with pytest.raises(ValueError, match="unknown split names"):
        load_split_arrays(processed.root, processed.root / SPLITS_FILE, ["holdout"])


def test_arrays_follow_the_split_file_and_align_with_features(processed):
    payload = json.loads((processed.root / SPLITS_FILE).read_text(encoding="utf-8"))
    arrays = load_split_arrays(processed.root, processed.root / SPLITS_FILE, ["train", "val"])
    row_of = {int(sid): row for row, sid in enumerate(processed.features["sequence_id"])}
    for name in ("train", "val"):
        split = arrays[name]
        rows = [row_of[sid] for sid in payload[name]]
        assert split.sequence_id.tolist() == payload[name]
        assert split.X.shape == (len(rows), 32, 100)
        assert split.X.dtype == np.float32
        assert split.y.dtype == np.int64
        expected_coords = processed.features["X"][rows].reshape(len(rows), 32, 98)
        np.testing.assert_array_equal(split.X[..., :98], expected_coords.astype(np.float32))
        np.testing.assert_array_equal(split.X[..., 98:], processed.features["presence"][rows])
        np.testing.assert_array_equal(split.y, processed.features["y"][rows])
        np.testing.assert_array_equal(
            split.participant_id, processed.features["participant_id"][rows]
        )
    assert not owners(processed.manifest, payload["train"]) & owners(
        processed.manifest, payload["val"]
    )


def test_missing_split_ids_are_counted(processed, tmp_path):
    payload = json.loads((processed.root / SPLITS_FILE).read_text(encoding="utf-8"))
    payload["val"] += [1, 2, 3]
    path = tmp_path / "splits.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=r"3 of \d+ val sequence ids .* missing"):
        load_split_arrays(processed.root, path, ["val"])


def test_features_with_z_are_rejected(tmp_path):
    data = build_processed_dir(tmp_path / "processed", per_cell=1)
    features = {**data.features, "X": np.zeros((*data.features["X"].shape[:3], 3), np.float16)}
    np.savez_compressed(data.root / "features.npz", **features)
    with pytest.raises(ValueError, match=r"X must be float16 \(\d+, 32, 49, 2\)"):
        load_split_arrays(data.root, data.root / SPLITS_FILE, ["train"])


def test_missing_split_file_is_built_in_the_run_dir_not_processed_dir(tmp_path):
    data = build_processed_dir(tmp_path / "processed", write_splits=False)
    before = snapshot(data.root)
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    path = prepare_splits(data.root, SplitConfig(), run_dir)

    assert path == run_dir / SPLITS_FILE
    assert snapshot(data.root) == before
    payload = json.loads(path.read_text(encoding="utf-8"))
    signers = [owners(data.manifest, payload[name]) for name in ("train", "val", "test")]
    assert [len(group) for group in signers] == [1, 3, 4]
    assert not (signers[0] & signers[1] or signers[0] & signers[2] or signers[1] & signers[2])
    kept = set(data.manifest.loc[~data.manifest["excluded"], "sequence_id"])
    assert set(payload["train"]) | set(payload["val"]) | set(payload["test"]) == kept


def test_signer_dependent_split_is_built_from_the_manifest(processed, tmp_path):
    before = snapshot(processed.root)
    path = prepare_splits(processed.root, SplitConfig(mode="signer_dependent"), tmp_path)
    assert path == tmp_path / "splits_signer_dependent.json"
    assert snapshot(processed.root) == before
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["mode"] == "signer_dependent"
    kept = set(processed.manifest.loc[~processed.manifest["excluded"], "sequence_id"])
    assert set(payload["train"]) | set(payload["val"]) | set(payload["test"]) == kept


def test_existing_split_file_is_copied_unchanged(processed, tmp_path):
    path = prepare_splits(processed.root, SplitConfig(), tmp_path)
    assert path.read_bytes() == (processed.root / SPLITS_FILE).read_bytes()


@pytest.mark.parametrize(
    ("change", "message"),
    [({"seed": 7}, "seed 42 != 7"), ({"n_test_signers": 2}, "4 test signers != 2")],
)
def test_existing_split_made_with_other_settings_is_refused(processed, tmp_path, change, message):
    with pytest.raises(ValueError, match=message):
        prepare_splits(processed.root, SplitConfig(**change), tmp_path)


def test_label_map_comes_from_the_manifest(processed):
    assert label_map_from_manifest(read_manifest(processed.root)) == dict(enumerate(SIGNS))


@pytest.mark.parametrize(
    ("labels", "signs", "message"),
    [([0, 2], ["a", "c"], r"exactly 0\.\.1"), ([0, 0], ["a", "b"], "one to one")],
)
def test_label_map_rejects_inconsistent_labels(labels, signs, message):
    with pytest.raises(ValueError, match=message):
        label_map_from_manifest(pd.DataFrame({"label": labels, "sign": signs}))
