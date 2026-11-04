"""Result tables over completed runs: validation aggregation and test-split reports.

``aggregate_val`` reads only summary.json files. The test helpers work on logits and arrays that
scripts/final_eval.py produced; nothing in this module opens the test split itself.
Standard deviations are sample standard deviations over seeds (ddof=1), empty for one seed.

Every table starts with an "experiment" column. Records without one come from v1 experiments,
named <model>_<split_mode> (see edgevision.runs.experiment_of), so v1 rows keep their values.
"""

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from edgevision.evaluation.metrics import error_pairs, macro_f1, per_group_accuracy, top_k_accuracy
from edgevision.runs import PRIMARY_SPLIT_MODE, QUANT_VARIANTS, CompletedRun, experiment_of
from edgevision.utils.runinfo import collect_run_info

__all__ = ["PRIMARY_SPLIT_MODE"]  # Re-exported: scripts import it from here.

TOP_CONFUSIONS = 30
GROUP_KEYS = ["model", "split_mode", "experiment"]  # v1 experiments sort as models did.
VAL_SUMMARY_COLUMNS = (
    "experiment",
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
    "experiment",
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
    "experiment",
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
QUANT_TEST_SUMMARY_COLUMNS = (
    "experiment",
    "model",
    "variant",
    "n_seeds",
    "top1_mean",
    "top1_std",
    "delta_vs_fp32_mean",
    "delta_vs_fp32_std",
    "top5_mean",
    "macro_f1_mean",
    "agreement_mean",
    "size_mb",
)
ENSEMBLE_KEYS = ("experiment", "model", "split_mode", "feature_set", "recipe", "n_seeds", "seeds")
VAL_ENSEMBLE_COLUMNS = (
    *ENSEMBLE_KEYS,
    "val_top1",
    "val_top5",
    "val_macro_f1",
    "val_top1_seed_mean",
)
TEST_ENSEMBLE_COLUMNS = (
    *ENSEMBLE_KEYS,
    "test_top1",
    "test_top5",
    "test_macro_f1",
    "test_top1_seed_mean",
)


def with_experiment(frame: pd.DataFrame) -> pd.DataFrame:
    """``frame`` with an experiment column, filled for rows from records that have none."""
    if "experiment" in frame.columns and frame["experiment"].notna().all():
        return frame
    records = frame.to_dict("records")
    names = [experiment_of({str(key): value for key, value in row.items()}) for row in records]
    return frame.assign(experiment=names)


def _mean_std(frame: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    named: dict[str, tuple[str, str]] = {"n_seeds": ("seed", "nunique")}
    for metric in metrics:
        named[f"{metric}_mean"] = (metric, "mean")
        named[f"{metric}_std"] = (metric, "std")
    return with_experiment(frame).groupby(GROUP_KEYS, sort=True).agg(**named).reset_index()


def aggregate_val(runs: Sequence[CompletedRun]) -> pd.DataFrame:
    """One row per experiment with validation metrics at each run's best epoch."""
    if not runs:
        raise ValueError("no completed runs to aggregate")
    frame = pd.DataFrame(
        [
            {
                "experiment": run.experiment,
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
    extras = frame.groupby(GROUP_KEYS, sort=True).agg(
        params=("params", "max"), train_seconds_mean=("train_seconds", "mean")
    )
    table = table.join(extras, on=GROUP_KEYS)
    return table.loc[:, list(VAL_SUMMARY_COLUMNS)]


def _best_row(val_summary: pd.DataFrame) -> "pd.Series[Any]":
    primary = val_summary.loc[val_summary["split_mode"] == PRIMARY_SPLIT_MODE]
    candidates = with_experiment(primary if not primary.empty else val_summary)
    if candidates.empty:
        raise ValueError("val summary has no rows")
    order = ["val_top1_mean", "model", "experiment"]
    return candidates.sort_values(order, ascending=[False, True, True]).iloc[0]


def select_best_model(val_summary: pd.DataFrame) -> tuple[str, str]:
    """(model, split mode) with the best mean val top-1, among signer-independent rows if any."""
    best = _best_row(val_summary)
    return str(best["model"]), str(best["split_mode"])


def select_best_experiment(val_summary: pd.DataFrame) -> str:
    """The experiment with the best mean val top-1, among signer-independent rows if any; ties
    go to the model name, then the experiment name. For v1 rows it picks select_best_model's."""
    return str(_best_row(val_summary)["experiment"])


def ensemble_scores(
    member_logits: Sequence[npt.NDArray[np.float32]], labels: npt.NDArray[np.int64]
) -> dict[str, float]:
    """Top-1, top-5 and macro-F1 of the mean of the members' logits, plus the members' mean
    single-model top-1."""
    if len(member_logits) < 2:
        raise ValueError("an ensemble needs at least two members")
    averaged = np.mean(np.stack(member_logits), axis=0)
    return {
        "top1": top_k_accuracy(averaged, labels, 1),
        "top5": top_k_accuracy(averaged, labels, 5),
        "macro_f1": macro_f1(labels, averaged.argmax(axis=1)),
        "top1_seed_mean": float(
            np.mean([top_k_accuracy(logits, labels, 1) for logits in member_logits])
        ),
    }


def ensemble_row(
    runs: Sequence[CompletedRun], scores: dict[str, float], split: str
) -> dict[str, Any]:
    """One row of val_ensemble.csv or test_ensemble.csv (``split`` is "val" or "test")."""
    first = runs[0]
    return {
        "experiment": first.experiment,
        "model": first.model,
        "split_mode": first.split_mode,
        "feature_set": first.feature_set,
        "recipe": first.recipe,
        "n_seeds": len(runs),
        "seeds": " ".join(str(run.seed) for run in runs),
        **{f"{split}_{name}": value for name, value in scores.items()},
    }


def ensemble_table(rows: Sequence[dict[str, Any]], split: str) -> pd.DataFrame:
    columns = list(VAL_ENSEMBLE_COLUMNS if split == "val" else TEST_ENSEMBLE_COLUMNS)
    if not rows:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(rows).sort_values(GROUP_KEYS, kind="stable")
    return frame.loc[:, columns].reset_index(drop=True)


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
    """``records`` hold model, split_mode, seed, the score_test_split() keys and, optionally,
    experiment."""
    frame = pd.DataFrame(records)
    table = _mean_std(frame, ("test_top1", "test_top5", "test_macro_f1"))
    return table.loc[:, list(TEST_SUMMARY_COLUMNS)]


def summarize_quant_test(records: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """One row per (experiment, variant) of ONNX variants on test.

    ``records`` hold model, seed, variant, size_bytes, agreement_with_fp32, the
    score_test_split() keys and, optionally, experiment. Deltas are paired by seed against that
    seed's fp32 variant; size_mb is in units of 10^6 bytes.
    """
    frame = with_experiment(pd.DataFrame(records))
    fp32 = frame.loc[frame["variant"] == "fp32", ["experiment", "seed", "test_top1"]]
    frame = frame.merge(
        fp32.rename(columns={"test_top1": "fp32_top1"}), on=["experiment", "seed"], how="left"
    )
    frame["delta"] = frame["test_top1"] - frame["fp32_top1"]
    frame["size_mb"] = frame["size_bytes"] / 1e6
    frame["variant"] = pd.Categorical(frame["variant"], categories=QUANT_VARIANTS, ordered=True)
    table = (
        frame.groupby(["model", "experiment", "variant"], sort=True, observed=True)
        .agg(
            n_seeds=("seed", "nunique"),
            top1_mean=("test_top1", "mean"),
            top1_std=("test_top1", "std"),
            delta_vs_fp32_mean=("delta", "mean"),
            delta_vs_fp32_std=("delta", "std"),
            top5_mean=("test_top5", "mean"),
            macro_f1_mean=("test_macro_f1", "mean"),
            agreement_mean=("agreement_with_fp32", "mean"),
            size_mb=("size_mb", "mean"),
        )
        .reset_index()
    )
    table["variant"] = table["variant"].astype(str)
    return table.loc[:, list(QUANT_TEST_SUMMARY_COLUMNS)]


def per_participant_table(records: Sequence[dict[str, Any]]) -> pd.DataFrame:
    rows = [
        {
            "experiment": experiment_of(record),
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
        .groupby([*GROUP_KEYS, "participant_id"], sort=True)
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
