"""Reference outputs of the v1 pipeline, recorded before the v2 feature set was added.

tests/baselines/v1_reference.json holds what these helpers produced on the synthetic fixtures
with the code as it stood before v2: a digest of the preprocessing output and the metrics.csv of
a short v1 training run. test_v1_unchanged.py recomputes both with the current code.
"""

import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from conftest import build_fake_dataset, load_cli_module
from edgevision.config import SplitConfig
from edgevision.train.config import load_train_config
from edgevision.train.loop import run_training
from training_support import EXPERIMENTS, build_processed_dir

REFERENCE_PATH = Path(__file__).resolve().parent / "baselines" / "v1_reference.json"
FEATURE_ARRAYS = ("X", "presence", "y", "participant_id", "sequence_id")
# Columns of metrics.csv that a seeded CPU run determines; epoch_seconds is wall time.
METRIC_VALUES = ("epoch", "train_loss", "train_top1", "val_loss", "val_top1", "val_top5", "lr")
METRICS_SPLIT = SplitConfig(n_test_signers=2, n_val_signers=3)
METRICS_EPOCHS = 3
METRICS_BATCH = 64


def _text_bytes(path: Path) -> bytes:
    """File contents with Windows line endings normalised, so the digest is portable."""
    return path.read_bytes().replace(b"\r\n", b"\n")


def preprocess_digest(output_dir: Path) -> str:
    """sha256 over features.npz arrays, manifest.csv and the split file.

    preprocess_report.json is left out: it holds wall time, run info and new fields. A read
    error's reason names a pyarrow exception class, which is normalised so the digest does not
    depend on the pyarrow version.
    """
    digest = hashlib.sha256()
    with np.load(output_dir / "features.npz", allow_pickle=False) as data:
        for key in FEATURE_ARRAYS:
            array = np.ascontiguousarray(data[key])
            digest.update(f"{key}:{array.dtype.str}:{array.shape}".encode())
            digest.update(array.tobytes())
    manifest = pd.read_csv(output_dir / "manifest.csv", keep_default_na=False)
    reason = manifest["reason"].astype(str)
    manifest["reason"] = reason.where(~reason.str.startswith("read_error: "), "read_error")
    digest.update(manifest.to_csv(index=False, lineterminator="\n").encode())
    digest.update(_text_bytes(output_dir / "splits_signer_independent.json"))
    return digest.hexdigest()


def run_v1_preprocessing(root: Path, config_text: str | None = None) -> Path:
    """The default synthetic dataset through scripts/preprocess.py with one worker, no cache."""
    dataset = build_fake_dataset(root / "asl-signs")
    output_dir = root / "processed"
    config = root / "preprocess.yaml"
    config.write_text(
        config_text
        or (
            "preprocess:\n"
            f"  data_root: {dataset.root.as_posix()}\n"
            f"  output_dir: {output_dir.as_posix()}\n"
            "  num_workers: 1\n"
            "split: {}\n"
        ),
        encoding="utf-8",
    )
    assert load_cli_module().main(["--config", str(config), "--no-cache"]) == 0
    return output_dir


def run_v1_training(root: Path) -> pd.DataFrame:
    """experiments/train_cnn.yaml (augmentation on) for three short CPU epochs."""
    processed = build_processed_dir(root / "processed", write_splits=False)
    cfg = load_train_config(
        EXPERIMENTS / "train_cnn.yaml",
        processed_dir=processed.root,
        output_root=root / "runs",
        epochs=METRICS_EPOCHS,
    ).model_copy(update={"batch_size": METRICS_BATCH, "split": METRICS_SPLIT})
    run_training(cfg, seed=0, device="cpu", log=lambda _: None)
    return pd.read_csv(root / "runs" / "cnn_signer_independent" / "seed_0" / "metrics.csv")


def platform_key() -> str:
    """Where a reference was recorded; floats are compared exactly only on the same key."""
    return f"{platform.system()}-{platform.machine()}-torch{torch.__version__}"


def metric_columns(metrics: pd.DataFrame) -> dict[str, list[Any]]:
    """The deterministic columns as plain lists; JSON keeps every float64 exactly."""
    return {column: metrics[column].tolist() for column in METRIC_VALUES}


def load_reference() -> dict[str, Any]:
    reference: dict[str, Any] = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    return reference
