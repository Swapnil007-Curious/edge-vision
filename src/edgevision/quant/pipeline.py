"""Quantise one completed run: build its ONNX variants, check them on val, write quant_report.json.

Val is read here (allowed); the test split is read only by scripts/final_eval.py. A variant that
fails is recorded in the report and the remaining variants still run.
"""

import hashlib
import json
import logging
import platform
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import onnx
import pandas as pd
import torch

from edgevision.data.io import write_json
from edgevision.evaluation.evaluators import OnnxEvaluator
from edgevision.evaluation.metrics import top_k_accuracy
from edgevision.evaluation.tables import PRIMARY_SPLIT_MODE
from edgevision.export.onnx_export import DEFAULT_OPSET
from edgevision.layout import INPUT_NAME
from edgevision.quant.calibration import CALIBRATION_SAMPLES, TrainCalibrationReader
from edgevision.quant.graph_stats import graph_stats, quantizers_of
from edgevision.quant.variants import (
    DYNAMIC_SETTINGS,
    FP16_SETTINGS,
    STATIC_SETTINGS,
    VARIANT_SCHEMA_VERSIONS,
    convert_fp16,
    describe,
    export_fp32,
    quantize_int8_dynamic,
    quantize_int8_static,
    read_metadata,
    save_with_metadata,
    schema_version,
    scratch_path,
)
from edgevision.runs import (
    CHECKPOINT_FILE,
    QUANT_REPORT_FILE,
    QUANT_VARIANTS,
    CompletedRun,
    experiment_of,
    variant_file,
)
from edgevision.runtime import (
    META_SCHEMA_VERSION,
    META_SOURCE_SHA256,
    META_VARIANT,
    onnxruntime_version,
)
from edgevision.train.data import SplitArrays, load_split_arrays
from edgevision.utils.runinfo import collect_run_info

FloatArray = npt.NDArray[np.float32]

FP16_MIN_AGREEMENT = 0.99
VAL_DIFF_SAMPLES = 512
STATIC_VARIANTS = ("int8_static", "int8_static_full")
META_FP16_FEATURE_BLOCK = "edgevision.fp16_feature_block_fp32"
META_FP16_FALLBACK = "edgevision.fp16_fallback_reason"
META_CALIBRATION = "edgevision.calibration"
META_EXCLUDED_NODES = "edgevision.excluded_nodes"
QUANT_OPS_COLUMNS = (
    "experiment",
    "model",
    "variant",
    "size_mb",
    "size_ratio_vs_fp32",
    "fraction_compute_ops_quantised",
    "gru_float_remaining",
    "failures",
)


class StaleVariantError(RuntimeError):
    """An existing variant file was built from a different checkpoint."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def describe_error(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"


def agreement(logits: FloatArray, reference: FloatArray) -> float:
    """Share of rows whose top-1 class matches the reference's."""
    return float((logits.argmax(axis=1) == reference.argmax(axis=1)).mean())


class _MessageCounter(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.counts: Counter[str] = Counter()

    def emit(self, record: logging.LogRecord) -> None:
        self.counts[record.getMessage()] += 1


@contextmanager
def collect_tool_messages() -> Iterator[Counter[str]]:
    """Count the warnings onnxruntime's tools log through the logging module, so they land in
    the report instead of the console."""
    handler = _MessageCounter()
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        yield handler.counts
    finally:
        root.removeHandler(handler)


def tool_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "onnx": onnx.__version__,
        "onnxruntime": onnxruntime_version(),
    }


class _RunQuantizer:
    """Builds and inspects the variants of one run; see quantize_run."""

    def __init__(
        self,
        run: CompletedRun,
        processed_dir: Path,
        out_dir: Path,
        overwrite: bool,
        log: Callable[[str], None],
    ):
        self.run = run
        self.processed_dir = processed_dir
        self.out_dir = out_dir
        self.overwrite = overwrite
        self.log = log
        self.checkpoint = run.directory / CHECKPOINT_FILE
        self.source_sha256 = sha256_file(self.checkpoint)
        self.splits_path = run.directory / run.summary["splits_file"]
        self.val: SplitArrays = load_split_arrays(processed_dir, self.splits_path, ["val"])["val"]
        self.failed: set[str] = set()
        self._fp32_logits: FloatArray | None = None
        self._calibration_reader: TrainCalibrationReader | None = None

    def path(self, variant: str) -> Path:
        return self.out_dir / variant_file(variant)

    def fp32_val_logits(self) -> FloatArray:
        if self._fp32_logits is None:
            if "fp32" in self.failed or not self.path("fp32").exists():
                raise RuntimeError("the fp32 variant is unavailable")
            self._fp32_logits = OnnxEvaluator(self.path("fp32")).predict(self.val.X)
        return self._fp32_logits

    def calibration_reader(self) -> TrainCalibrationReader:
        """One reader for every static variant, rewound, so they calibrate on the same data."""
        if self._calibration_reader is None:
            self._calibration_reader = TrainCalibrationReader(self.processed_dir, self.splits_path)
        self._calibration_reader.rewind()
        return self._calibration_reader

    def derived_metadata(self, variant: str, extra: dict[str, str] | None = None) -> dict[str, str]:
        """The fp32 file's metadata with this variant's name, schema version and ``extra``."""
        return {
            **read_metadata(self.path("fp32")),
            META_VARIANT: variant,
            META_SCHEMA_VERSION: str(VARIANT_SCHEMA_VERSIONS[variant]),
            **(extra or {}),
        }

    def build(self, variant: str) -> dict[str, Any]:
        """Build ``variant`` unless a file from this checkpoint and the variant's current schema
        version exists and overwrite is off.

        A file from this checkpoint but another schema version is deleted and rebuilt, so a
        failed rebuild cannot leave it behind for evaluation or benchmarks.
        """
        path = self.path(variant)
        rebuild_reason = None
        if path.exists() and not self.overwrite:
            metadata = read_metadata(path)
            if metadata.get(META_SOURCE_SHA256) != self.source_sha256:
                raise StaleVariantError(
                    f"{path} was not built from the current {CHECKPOINT_FILE}; pass --overwrite"
                )
            found, current = schema_version(metadata), VARIANT_SCHEMA_VERSIONS[variant]
            if found == current:
                details = self.recorded_details(variant, metadata)
                return {"status": "existing", **with_exclusions(details)}
            rebuild_reason = f"built with variant schema {found}; current is {current}"
            path.unlink()
        with collect_tool_messages() as messages:
            details = self.builders()[variant](path)
        entry = {"status": "built", **with_exclusions(details), "tool_messages": dict(messages)}
        if rebuild_reason is not None:
            entry["rebuild_reason"] = rebuild_reason
        return entry

    def builders(self) -> dict[str, Callable[[Path], dict[str, Any]]]:
        return {
            "fp32": self.build_fp32,
            "fp16": self.build_fp16,
            "int8_dynamic": self.build_int8_dynamic,
            "int8_static": self.build_int8_static,
            "int8_static_full": self.build_int8_static_full,
        }

    def recorded_details(self, variant: str, metadata: dict[str, str]) -> dict[str, Any]:
        """What an earlier build recorded in the file, for a variant that is reused."""
        settings = {
            "fp32": {"opset": DEFAULT_OPSET},
            "fp16": describe(FP16_SETTINGS),
            "int8_dynamic": describe(DYNAMIC_SETTINGS),
            "int8_static": describe(STATIC_SETTINGS),
            "int8_static_full": describe(STATIC_SETTINGS),
        }
        details: dict[str, Any] = {"settings": settings[variant]}
        if variant == "fp16":
            details["feature_block_fp32"] = json.loads(metadata[META_FP16_FEATURE_BLOCK])
            details["fallback_reason"] = metadata[META_FP16_FALLBACK] or None
        if variant in STATIC_VARIANTS:
            details["calibration"] = json.loads(metadata[META_CALIBRATION])
            details["excluded_node_names"] = json.loads(metadata[META_EXCLUDED_NODES])
        return details

    def build_fp32(self, path: Path) -> dict[str, Any]:
        export_fp32(self.checkpoint, path, self.source_sha256)
        return {"settings": {"opset": DEFAULT_OPSET}}

    def build_fp16(self, path: Path) -> dict[str, Any]:
        """Plain conversion first; if any val output is non-finite or val top-1 agreement with
        fp32 is below FP16_MIN_AGREEMENT, convert again with the FeatureBlock kept in float32."""
        reference = self.fp32_val_logits()
        attempts: list[dict[str, Any]] = []
        for keep_feature_block in (False, True):
            candidate = convert_fp16(self.path("fp32"), keep_feature_block_fp32=keep_feature_block)
            scratch = scratch_path(path, "candidate")
            try:
                onnx.save(candidate, str(scratch))
                logits = OnnxEvaluator(scratch).predict(self.val.X)
            finally:
                scratch.unlink(missing_ok=True)
            finite = bool(np.isfinite(logits).all())
            matched = agreement(logits, reference)
            attempts.append(
                {
                    "feature_block_fp32": keep_feature_block,
                    "all_finite": finite,
                    "val_agreement_with_fp32": matched,
                }
            )
            if finite and matched >= FP16_MIN_AGREEMENT:
                break
        reason = None
        if len(attempts) > 1:
            first = attempts[0]
            reason = (
                f"plain conversion: all finite {first['all_finite']}, val top-1 agreement "
                f"{first['val_agreement_with_fp32']:.4f} (needs >= {FP16_MIN_AGREEMENT})"
            )
        extra = {
            META_FP16_FEATURE_BLOCK: json.dumps(keep_feature_block),
            META_FP16_FALLBACK: reason or "",
        }
        save_with_metadata(candidate, path, self.derived_metadata("fp16", extra))
        return {
            "settings": describe(FP16_SETTINGS),
            "feature_block_fp32": keep_feature_block,
            "fallback_reason": reason,
            "attempts": attempts,
        }

    def build_int8_dynamic(self, path: Path) -> dict[str, Any]:
        model = quantize_int8_dynamic(self.path("fp32"), path)
        save_with_metadata(model, path, self.derived_metadata("int8_dynamic"))
        return {"settings": describe(DYNAMIC_SETTINGS)}

    def build_int8_static(self, path: Path) -> dict[str, Any]:
        """Input and FeatureBlock stay float32; quantisation starts at the projection."""
        return self.build_static(path, "int8_static", exclude_feature_block=True)

    def build_int8_static_full(self, path: Path) -> dict[str, Any]:
        """Ablation: the earlier int8_static, with nothing excluded from quantisation."""
        return self.build_static(path, "int8_static_full", exclude_feature_block=False)

    def build_static(
        self, path: Path, variant: str, *, exclude_feature_block: bool
    ) -> dict[str, Any]:
        reader = self.calibration_reader()
        calibration = {
            "split": "train",
            "requested_samples": CALIBRATION_SAMPLES,
            "samples": len(reader),
            "train_sequences": reader.train_sequences,
            "seed": reader.seed,
            "batch_size": reader.batch_size,
        }
        model, excluded = quantize_int8_static(
            self.path("fp32"), path, reader, exclude_feature_block=exclude_feature_block
        )
        extra = {
            META_CALIBRATION: json.dumps(calibration),
            META_EXCLUDED_NODES: json.dumps(excluded),
        }
        save_with_metadata(model, path, self.derived_metadata(variant, extra))
        return {
            "settings": describe(STATIC_SETTINGS),
            "calibration": calibration,
            "excluded_node_names": excluded,
        }

    def inspect(self, variant: str) -> dict[str, Any]:
        """Size, graph statistics, checker, onnxruntime load and val metrics of a built file."""
        path = self.path(variant)
        model = onnx.load(str(path))
        details: dict[str, Any] = {
            "size_bytes": path.stat().st_size,
            "graph": graph_stats(model),
            "input_quantised": bool(quantizers_of(model, {INPUT_NAME})),
        }
        try:
            onnx.checker.check_model(str(path))
            details["onnx_checker"] = {"passed": True, "error": None}
        except Exception as error:
            details["onnx_checker"] = {"passed": False, "error": describe_error(error)}
        try:
            evaluator = OnnxEvaluator(path)
        except Exception as error:
            details["ort_load"] = {"passed": False, "error": describe_error(error)}
            return details
        details["ort_load"] = {"passed": True, "error": None}
        logits = self.fp32_val_logits() if variant == "fp32" else evaluator.predict(self.val.X)
        details["val"] = self.val_metrics(logits)
        return details

    def val_metrics(self, logits: FloatArray) -> dict[str, Any]:
        metrics: dict[str, Any] = {
            "top1": top_k_accuracy(logits, self.val.y, 1),
            "all_finite": bool(np.isfinite(logits).all()),
            "agreement_with_fp32": None,
            "max_abs_logit_diff_vs_fp32": None,
            "diff_samples": min(VAL_DIFF_SAMPLES, len(logits)),
        }
        try:
            reference = self.fp32_val_logits()
        except Exception:
            return metrics  # Without an fp32 reference there is nothing to compare against.
        head = slice(0, VAL_DIFF_SAMPLES)
        metrics["agreement_with_fp32"] = agreement(logits, reference)
        metrics["max_abs_logit_diff_vs_fp32"] = float(np.abs(logits[head] - reference[head]).max())
        return metrics

    def quantize(self, variants: Sequence[str]) -> dict[str, dict[str, Any]]:
        entries: dict[str, dict[str, Any]] = {}
        for variant in variants:
            entry: dict[str, Any] = {"file": variant_file(variant), "error": None}
            try:
                entry.update(self.build(variant))
            except Exception as error:
                self.failed.add(variant)
                entry.update(status="failed", error=describe_error(error))
            entries[variant] = entry
        for variant, entry in entries.items():
            if variant not in self.failed:
                try:
                    entry.update(self.inspect(variant))
                except Exception as error:
                    self.failed.add(variant)
                    entry.update(status="failed", error=describe_error(error))
            self.log(f"{self.run.key} {variant}: {_one_line(entry)}")
        return entries


def with_exclusions(details: dict[str, Any]) -> dict[str, Any]:
    """``details`` with excluded_node_count and excluded_node_names, the nodes passed to the
    quantiser's nodes_to_exclude. Only int8_static excludes any; other variants report none."""
    rest = dict(details)
    names = list(rest.pop("excluded_node_names", []))
    return {**rest, "excluded_node_count": len(names), "excluded_node_names": names}


def _one_line(entry: dict[str, Any]) -> str:
    if entry["error"]:
        return f"FAILED {entry['error']}"
    val = entry.get("val") or {}
    parts = [str(entry["status"]), f"{entry['size_bytes'] / 1e6:.2f} MB"]
    if "top1" in val:
        parts.append(f"val top-1 {val['top1']:.4f}")
    if val.get("agreement_with_fp32") is not None:
        parts.append(f"agreement {val['agreement_with_fp32']:.4f}")
    if entry.get("feature_block_fp32"):
        parts.append("FeatureBlock kept in fp32")
    if entry.get("excluded_node_count"):
        parts.append(f"{entry['excluded_node_count']} FeatureBlock nodes excluded")
    if entry.get("input_quantised"):
        parts.append("input quantised")
    if entry.get("rebuild_reason"):
        parts.append(f"rebuilt: {entry['rebuild_reason']}")
    return ", ".join(parts)


def quantize_run(
    run: CompletedRun,
    processed_dir: Path,
    out_dir: Path,
    *,
    overwrite: bool = False,
    variants: Sequence[str] = QUANT_VARIANTS,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Build ``variants`` of ``run`` into ``out_dir`` and write quant_report.json there.

    Existing variant files built from the same best.pt at their variant's current schema version
    are reused unless ``overwrite``; one from an older schema version is rebuilt, and one built
    from another checkpoint is reported as stale. Failures are recorded, never raised.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "run": run.key,
        "model": run.summary["model"],
        "seed": run.summary["seed"],
        "split_mode": run.summary["split_mode"],
        "checkpoint": str(run.directory / CHECKPOINT_FILE),
        "checkpoint_val_top1": run.summary["best_val_top1"],
        "processed_dir": str(processed_dir),
        "error": None,
    }
    try:
        quantizer = _RunQuantizer(run, processed_dir, out_dir, overwrite, log)
    except Exception as error:
        report.update(error=describe_error(error), variants={}, failures=list(variants))
        log(f"{run.key}: FAILED {report['error']}")
    else:
        report.update(
            checkpoint_sha256=quantizer.source_sha256,
            splits_file=str(quantizer.splits_path),
            val_sequences=len(quantizer.val.y),
            fp16_min_agreement=FP16_MIN_AGREEMENT,
        )
        entries = quantizer.quantize(variants)
        report.update(
            calibration=entries.get("int8_static", {}).get("calibration"),
            variants=entries,
            failures=sorted(quantizer.failed, key=list(variants).index),
        )
    report.update(tool_versions=tool_versions(), run_info=collect_run_info())
    write_json(out_dir / QUANT_REPORT_FILE, report)
    return report


def find_quant_reports(quant_root: Path) -> list[dict[str, Any]]:
    """Every signer-independent quant_report.json under ``quant_root``, in path order."""
    reports = []
    for path in sorted(quant_root.glob(f"*/seed_*/{QUANT_REPORT_FILE}")):
        with path.open(encoding="utf-8") as handle:
            report = json.load(handle)
        if report.get("split_mode", PRIMARY_SPLIT_MODE) == PRIMARY_SPLIT_MODE:
            reports.append(report)
    return reports


def report_experiment(report: dict[str, Any]) -> str:
    """The experiment of a quant report: the first part of its run key, else the v1 name."""
    run = report.get("run")
    return str(run).split("/")[0] if run else experiment_of(report)


def quant_ops_summary(reports: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """One row per (experiment, variant): means over seeds; failures counts seeds without the
    file."""
    rows = []
    for report in reports:
        fp32_size = report["variants"].get("fp32", {}).get("size_bytes")
        for variant in QUANT_VARIANTS:
            entry = report["variants"].get(variant, {})
            ok = not entry.get("error") and "size_bytes" in entry
            size = entry["size_bytes"] if ok else None
            graph = entry.get("graph", {}) if ok else {}
            rows.append(
                {
                    "experiment": report_experiment(report),
                    "model": report["model"],
                    "variant": variant,
                    "size_mb": size / 1e6 if size else None,
                    "size_ratio_vs_fp32": size / fp32_size if size and fp32_size else None,
                    "fraction_compute_ops_quantised": graph.get("fraction_compute_ops_quantised"),
                    "gru_float_remaining": graph.get("gru_float_remaining"),
                    "failed": not ok,
                }
            )
    if not rows:
        return pd.DataFrame(columns=list(QUANT_OPS_COLUMNS))
    frame = pd.DataFrame(rows)
    frame["variant"] = pd.Categorical(frame["variant"], categories=QUANT_VARIANTS, ordered=True)
    table = (
        frame.groupby(["model", "experiment", "variant"], sort=True, observed=True)
        .agg(
            size_mb=("size_mb", "mean"),
            size_ratio_vs_fp32=("size_ratio_vs_fp32", "mean"),
            fraction_compute_ops_quantised=("fraction_compute_ops_quantised", "mean"),
            gru_float_remaining=("gru_float_remaining", "max"),
            failures=("failed", "sum"),
        )
        .reset_index()
    )
    table["variant"] = table["variant"].astype(str)
    table["gru_float_remaining"] = table["gru_float_remaining"].astype("Int64")
    return table.loc[:, list(QUANT_OPS_COLUMNS)]
