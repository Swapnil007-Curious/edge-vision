"""Load preprocessed arrays for training and evaluation, and resolve the split file.

The processed directory is only ever read: on Kaggle it is a read-only input.
"""

import json
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt
import pandas as pd

from edgevision.config import SplitConfig
from edgevision.data.io import LABEL_DTYPE, PARTICIPANT_DTYPE, SEQUENCE_DTYPE, write_json
from edgevision.data.splits import (
    SPLIT_NAMES,
    Splits,
    assert_signer_disjoint,
    make_splits,
    splits_to_json,
    summarize_splits,
)
from edgevision.layout import COORDS_PER_LANDMARK, NUM_FRAMES, NUM_LANDMARKS, to_model_input

FEATURES_NPZ = "features.npz"
MANIFEST_CSV = "manifest.csv"
FEATURE_KEYS = ("X", "presence", "y", "participant_id", "sequence_id")
MANIFEST_REQUIRED = ("sequence_id", "participant_id", "sign", "label", "excluded")
TEST_ACCESS_MESSAGE = "test split is only accessible from scripts/final_eval.py"


class SplitArrays(NamedTuple):
    X: npt.NDArray[np.float32]  # (n, 32, 100), see edgevision.layout.
    y: npt.NDArray[np.int64]
    participant_id: npt.NDArray[np.int64]
    sequence_id: npt.NDArray[np.int64]


def splits_filename(mode: str) -> str:
    return f"splits_{mode}.json"


def read_manifest(processed_dir: Path) -> pd.DataFrame:
    """NA parsing stays off, as in preprocessing, so a sign spelled "null" remains a string."""
    path = processed_dir / MANIFEST_CSV
    manifest = pd.read_csv(path, keep_default_na=False)
    missing = [column for column in MANIFEST_REQUIRED if column not in manifest.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    if not pd.api.types.is_bool_dtype(manifest["excluded"]):
        raise ValueError(
            f"{path}: column excluded must be boolean, got {manifest['excluded'].dtype}"
        )
    return manifest


def label_map_from_manifest(manifest: pd.DataFrame) -> dict[int, str]:
    """Label index -> sign, requiring a one-to-one mapping over exactly 0..K-1."""
    pairs = manifest[["label", "sign"]].drop_duplicates()
    if pairs["label"].duplicated().any() or pairs["sign"].duplicated().any():
        raise ValueError("manifest labels and signs must map one to one")
    labels = sorted(int(label) for label in pairs["label"])
    if labels != list(range(len(labels))):
        raise ValueError(
            f"manifest labels must be exactly 0..{len(labels) - 1}, found {len(labels)} distinct "
            f"labels spanning {labels[0]}..{labels[-1]}"
        )
    pairs = pairs.sort_values("label")
    return dict(zip(pairs["label"].astype(int), pairs["sign"].astype(str), strict=True))


def read_splits(path: Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not all(
        isinstance(payload.get(name), list) for name in SPLIT_NAMES
    ):
        raise ValueError(f"{path}: expected an object with train, val and test id lists")
    return payload


def _check_split_settings(payload: dict[str, Any], cfg: SplitConfig, path: Path) -> None:
    mismatches = [
        f"{key} {payload.get(key)!r} != {expected!r}"
        for key, expected in (("mode", cfg.mode), ("seed", cfg.seed))
        if payload.get(key) != expected
    ]
    if cfg.mode == "signer_independent":
        participants = payload.get("participants") or {}
        for name, expected in (("test", cfg.n_test_signers), ("val", cfg.n_val_signers)):
            found = len(participants.get(name) or [])
            if found != expected:
                mismatches.append(f"{found} {name} signers != {expected}")
    if mismatches:
        raise ValueError(
            f"{path} was made with other split settings ({'; '.join(mismatches)}); "
            "make the config's split section match the preprocessing config"
        )


def _as_splits(payload: dict[str, Any]) -> Splits:
    return Splits(
        train=np.asarray(payload["train"], dtype=np.int64),
        val=np.asarray(payload["val"], dtype=np.int64),
        test=np.asarray(payload["test"], dtype=np.int64),
    )


def prepare_splits(processed_dir: Path, cfg: SplitConfig, run_dir: Path) -> Path:
    """Place the split file for ``cfg.mode`` in ``run_dir`` and return its path.

    An existing processed_dir/splits_<mode>.json is checked against ``cfg`` and copied, so every
    run records the split it used. Otherwise the split is built from manifest.csv. Signer
    disjointness is verified for signer_independent either way.
    """
    target = run_dir / splits_filename(cfg.mode)
    source = processed_dir / splits_filename(cfg.mode)
    manifest = read_manifest(processed_dir)
    if source.exists():
        payload = read_splits(source)
        _check_split_settings(payload, cfg, source)
        if cfg.mode == "signer_independent":
            assert_signer_disjoint(_as_splits(payload), manifest)
        shutil.copyfile(source, target)
        return target

    splits = make_splits(manifest, cfg)
    if cfg.mode == "signer_independent":
        assert_signer_disjoint(splits, manifest)
    write_json(target, splits_to_json(splits, summarize_splits(splits, manifest), cfg))
    return target


def _validate_features(arrays: dict[str, npt.NDArray[Any]], path: Path) -> None:
    n = arrays["X"].shape[0]
    expected = {
        "X": ((n, NUM_FRAMES, NUM_LANDMARKS, COORDS_PER_LANDMARK), np.float16),
        "presence": ((n, NUM_FRAMES, 2), np.uint8),
        "y": ((n,), LABEL_DTYPE),
        "participant_id": ((n,), PARTICIPANT_DTYPE),
        "sequence_id": ((n,), SEQUENCE_DTYPE),
    }
    for key, (shape, dtype) in expected.items():
        array = arrays[key]
        if array.shape != shape or array.dtype != dtype:
            raise ValueError(
                f"{path}: {key} must be {np.dtype(dtype).name} {shape}, "
                f"got {array.dtype.name} {array.shape}"
            )
    if np.unique(arrays["sequence_id"]).size != n:
        raise ValueError(f"{path}: sequence_id values are not unique")


def load_split_arrays(
    processed_dir: Path,
    splits_path: Path,
    split_names: Sequence[str],
    allow_test: bool = False,
) -> dict[str, SplitArrays]:
    """Arrays for each requested split, in the split file's sequence order.

    Only scripts/final_eval.py passes ``allow_test=True``.
    """
    unknown = sorted(set(split_names) - set(SPLIT_NAMES))
    if unknown:
        raise ValueError(f"unknown split names {unknown}; expected some of {SPLIT_NAMES}")
    if "test" in split_names and not allow_test:
        raise ValueError(TEST_ACCESS_MESSAGE)

    payload = read_splits(splits_path)
    path = processed_dir / FEATURES_NPZ
    with np.load(path, allow_pickle=False) as data:
        missing_keys = sorted(set(FEATURE_KEYS) - set(data.files))
        if missing_keys:
            raise ValueError(f"{path}: missing arrays {missing_keys}")
        arrays = {key: data[key] for key in FEATURE_KEYS}
    _validate_features(arrays, path)

    index = pd.Index(arrays["sequence_id"])
    result: dict[str, SplitArrays] = {}
    for name in split_names:
        wanted = np.asarray(payload[name], dtype=np.int64)
        rows = index.get_indexer(wanted)
        n_missing = int((rows < 0).sum())
        if n_missing:
            raise ValueError(
                f"{n_missing} of {wanted.size} {name} sequence ids from {splits_path} are missing "
                f"from {path}"
            )
        result[name] = SplitArrays(
            X=to_model_input(arrays["X"][rows], arrays["presence"][rows]),
            y=arrays["y"][rows].astype(np.int64),
            participant_id=arrays["participant_id"][rows].astype(np.int64),
            sequence_id=arrays["sequence_id"][rows].astype(np.int64),
        )
    return result
