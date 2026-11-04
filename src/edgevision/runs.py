"""Run-directory conventions: <output_root>/<experiment>/seed_<seed>/.

A v1 experiment is named <model>_<split_mode> (e.g. gru_signer_independent); v2 experiments
name themselves in their config (e.g. c_cnn_v2feat_v2recipe). Discovery reads each run's
summary.json fields, never the directory name pattern. A run is complete once summary.json
exists; it is the last file a training run writes. Its quantised variants use the same layout
under the quant root: <quant_root>/<experiment>/seed_<seed>/<variant>.onnx.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SUMMARY_FILE = "summary.json"
CHECKPOINT_FILE = "best.pt"
QUANT_REPORT_FILE = "quant_report.json"
PRIMARY_SPLIT_MODE = "signer_independent"
# int8_static_full is an ablation: int8_static without the FeatureBlock excluded from quantisation.
QUANT_VARIANTS = ("fp32", "fp16", "int8_dynamic", "int8_static", "int8_static_full")


def variant_file(variant: str) -> str:
    return f"{variant}.onnx"


def run_name(model: str, split_mode: str) -> str:
    """The v1 experiment name of a model and split mode."""
    return f"{model}_{split_mode}"


def experiment_run_dir(output_root: Path, experiment: str, seed: int) -> Path:
    return output_root / experiment / f"seed_{seed}"


def run_dir(output_root: Path, model: str, split_mode: str, seed: int) -> Path:
    """The run directory of a v1 experiment."""
    return experiment_run_dir(output_root, run_name(model, split_mode), seed)


def experiment_of(record: Mapping[str, Any]) -> str:
    """The experiment of a summary.json or result record. Records written before experiments
    were named belong to v1 experiments, <model>_<split_mode> (signer-independent if absent)."""
    named = record.get("experiment")
    if named:
        return str(named)
    return run_name(str(record["model"]), str(record.get("split_mode") or PRIMARY_SPLIT_MODE))


def is_complete(directory: Path) -> bool:
    return (directory / SUMMARY_FILE).is_file()


@dataclass(frozen=True)
class CompletedRun:
    directory: Path
    summary: dict[str, Any]

    @property
    def key(self) -> str:
        """Path relative to the runs root, e.g. "gru_signer_independent/seed_0"."""
        return f"{self.directory.parent.name}/{self.directory.name}"

    @property
    def experiment(self) -> str:
        return experiment_of(self.summary)

    @property
    def model(self) -> str:
        return str(self.summary["model"])

    @property
    def split_mode(self) -> str:
        return str(self.summary["split_mode"])

    @property
    def seed(self) -> int:
        return int(self.summary["seed"])

    @property
    def feature_set(self) -> str:
        """ "v1" for runs trained before feature sets were recorded."""
        return str(self.summary.get("feature_set", "v1"))

    @property
    def recipe(self) -> str:
        """ "v1" for runs trained before recipes were recorded."""
        return str(self.summary.get("recipe", "v1"))

    @property
    def results_key(self) -> str:
        """Directory name for this run's per-variant test results under final/quant/: the model
        name for v1 experiments, as before experiments were named, else the experiment."""
        if self.experiment == run_name(self.model, self.split_mode):
            return self.model
        return self.experiment


def find_completed_runs(output_root: Path) -> list[CompletedRun]:
    """Every completed run under ``output_root``, in sorted path order."""
    runs = []
    for summary_path in sorted(output_root.glob(f"*/seed_*/{SUMMARY_FILE}")):
        with summary_path.open(encoding="utf-8") as handle:
            runs.append(CompletedRun(summary_path.parent, json.load(handle)))
    return runs


def runs_by_experiment(runs: list[CompletedRun]) -> dict[str, list[CompletedRun]]:
    """Completed runs grouped by experiment, each group sorted by seed."""
    groups: dict[str, list[CompletedRun]] = {}
    for run in runs:
        groups.setdefault(run.experiment, []).append(run)
    return {name: sorted(group, key=lambda run: run.seed) for name, group in sorted(groups.items())}
