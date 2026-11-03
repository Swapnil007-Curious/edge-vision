"""Run-directory conventions: <output_root>/<model>_<split_mode>/seed_<seed>/.

A run is complete once summary.json exists; it is the last file a training run writes. Its
quantised variants use the same layout under the quant root: <quant_root>/<run key>/<variant>.onnx.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SUMMARY_FILE = "summary.json"
CHECKPOINT_FILE = "best.pt"
QUANT_REPORT_FILE = "quant_report.json"
QUANT_VARIANTS = ("fp32", "fp16", "int8_dynamic", "int8_static")


def variant_file(variant: str) -> str:
    return f"{variant}.onnx"


def run_name(model: str, split_mode: str) -> str:
    return f"{model}_{split_mode}"


def run_dir(output_root: Path, model: str, split_mode: str, seed: int) -> Path:
    return output_root / run_name(model, split_mode) / f"seed_{seed}"


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


def find_completed_runs(output_root: Path) -> list[CompletedRun]:
    """Every completed run under ``output_root``, in sorted path order."""
    runs = []
    for summary_path in sorted(output_root.glob(f"*/seed_*/{SUMMARY_FILE}")):
        with summary_path.open(encoding="utf-8") as handle:
            runs.append(CompletedRun(summary_path.parent, json.load(handle)))
    return runs
