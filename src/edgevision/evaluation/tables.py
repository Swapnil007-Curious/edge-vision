"""Result tables over completed runs: validation aggregation and test-split reports.

``aggregate_val`` reads only summary.json files. The test helpers work on logits and arrays that
scripts/final_eval.py produced; nothing in this module opens the test split itself.
Standard deviations are sample standard deviations over seeds (ddof=1), empty for one seed.
"""

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from edgevision.evaluation.metrics import error_pairs, macro_f1, per_group_accuracy, top_k_accuracy
from edgevision.runs import CompletedRun
from edgevision.utils.runinfo import collect_run_info

PRIMARY_SPLIT_MODE = "signer_independent"
TOP_CONFUSIONS = 30
VAL_SUMMARY_COLUMNS = (
    "model",
    "split_mode",
    "n_seeds",
    "val_top1_mean",
    "val_top1_std",
    "val_top5_mean",
    "val_top5_std",
    "val_macro_f1_mean",
    "val_macro_f1_std",
    "params",
    "train_seconds_mean",
)
TEST_SUMMARY_COLUMNS = (
    "model",
    "split_mode",
    "n_seeds",
    "test_top1_mean",
    "test_top1_std",
    "test_top5_mean",
    "test_top5_std",
    "test_macro_f1_mean",
    "test_macro_f1_std",
)
PER_PARTICIPANT_COLUMNS = (
    "model",
    "split_mode",
    "participant_id",
    "n_sequences",
    "n_seeds",
    "test_top1_mean",
)
CONFUSION_COLUMNS = (
    "rank",
    "true_label",
    "true_sign",
    "predicted_label",
    "predicted_sign",
    "count",
)


def _mean_std(frame: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    named: dict[str, tuple[str, str]] = {"n_seeds": ("seed", "nunique")}
    for metric in metrics:
        named[f"{metric}_mean"] = (metric, "mean")
        named[f"{metric}_std"] = (metric, "std")
    return frame.groupby(["model", "split_mode"], sort=True).agg(**named).reset_index()


def aggregate_val(runs: Sequence[CompletedRun]) -> pd.DataFrame:
    """One row per (model, split mode) with validation metrics at each run's best epoch."""
    if not runs:
        raise ValueError("no completed runs to aggregate")
    frame = pd.DataFrame(
        [
            {
                "model": run.summary["model"],
                "split_mode": run.summary["split_mode"],
                "seed": run.summary["seed"],
                "val_top1": run.summary["best_val_top1"],
                "val_top5": run.summary["best_val_top5"],
                "val_macro_f1": run.summary["best_val_macro_f1"],
                "params": run.summary["params"],
                "train_seconds": run.summary["total_train_seconds"],
            }
            for run in runs
        ]
    )
    table = _mean_std(frame, ("val_top1", "val_top5", "val_macro_f1"))
    extras = frame.groupby(["model", "split_mode"], sort=True).agg(
        params=("params", "max"), train_seconds_mean=("train_seconds", "mean")
    )
    table = table.join(extras, on=["model", "split_mode"])
    return table.loc[:, list(VAL_SUMMARY_COLUMNS)]


def select_best_model(val_summary: pd.DataFrame) -> tuple[str, str]:
    """(model, split mode) with the best mean val top-1, among signer-independent rows if any."""
    primary = val_summary.loc[val_summary["split_mode"] == PRIMARY_SPLIT_MODE]
    candidates = primary if not primary.empty else val_summary
    if candidates.empty:
        raise ValueError("val summary has no rows")
    best = candidates.sort_values(["val_top1_mean", "model"], ascending=[False, True]).iloc[0]
    return str(best["model"]), str(best["split_mode"])


def score_test_split(
    logits: npt.NDArray[np.float32],
    labels: npt.NDArray[np.int64],
    participants: npt.NDArray[np.int64],
) -> dict[str, Any]:
    """Top-1, top-5, macro-F1 and per-participant top-1 (not named test_* so pytest skips it)."""
    predictions = logits.argmax(axis=1)
    per_participant = per_group_accuracy(labels, predictions, participants)
    return {
        "n_sequences": int(labels.size),
        "test_top1": top_k_accuracy(logits, labels, 1),
        "test_top5": top_k_accuracy(logits, labels, 5),
        "test_macro_f1": macro_f1(labels, predictions),
        "per_participant": {
            str(participant): {"n_sequences": count, "top1": accuracy}
            for participant, (count, accuracy) in per_participant.items()
        },
    }


def summarize_test(records: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """``records`` hold model, split_mode, seed and the score_test_split() keys."""
    frame = pd.DataFrame(records)
    table = _mean_std(frame, ("test_top1", "test_top5", "test_macro_f1"))
    return table.loc[:, list(TEST_SUMMARY_COLUMNS)]


def per_participant_table(records: Sequence[dict[str, Any]]) -> pd.DataFrame:
    rows = [
        {
            "model": record["model"],
            "split_mode": record["split_mode"],
            "seed": record["seed"],
            "participant_id": int(participant),
            "n_sequences": values["n_sequences"],
            "top1": values["top1"],
        }
        for record in records
        for participant, values in record["per_participant"].items()
    ]
    table = (
        pd.DataFrame(rows)
        .groupby(["model", "split_mode", "participant_id"], sort=True)
        .agg(
            n_sequences=("n_sequences", "max"),
            n_seeds=("seed", "nunique"),
            test_top1_mean=("top1", "mean"),
        )
        .reset_index()
    )
    return table.loc[:, list(PER_PARTICIPANT_COLUMNS)]


def top_confusions(
    predictions: Iterable[tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]],
    label_map: dict[int, str],
    limit: int = TOP_CONFUSIONS,
) -> pd.DataFrame:
    """Most frequent (true, predicted) error pairs, summed over (labels, predictions) per seed."""
    totals: dict[tuple[int, int], int] = {}
    for labels, predicted in predictions:
        for pair, count in error_pairs(labels, predicted).items():
            totals[pair] = totals.get(pair, 0) + count
    ranked = sorted(totals.items(), key=lambda item: (-item[1], item[0]))[:limit]
    rows = [
        {
            "rank": rank,
            "true_label": true,
            "true_sign": label_map[true],
            "predicted_label": predicted,
            "predicted_sign": label_map[predicted],
            "count": count,
        }
        for rank, ((true, predicted), count) in enumerate(ranked, start=1)
    ]
    return pd.DataFrame(rows, columns=list(CONFUSION_COLUMNS))


def append_access_log(path: Path, runs_evaluated: Sequence[str]) -> dict[str, Any]:
    """Append one entry per test-split access; the log is a JSON list, rewritten atomically."""
    entries: list[dict[str, Any]] = []
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            loaded = json.load(handle)
        if not isinstance(loaded, list):
            raise ValueError(f"{path}: expected a JSON list")
        entries = loaded
    info = collect_run_info()
    entry = {
        "timestamp_utc": info["timestamp_utc"],
        "git_commit": info["git_commit"],
        "git_dirty": info["git_dirty"],
        "runs_evaluated": list(runs_evaluated),
    }
    entries.append(entry)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return entry
