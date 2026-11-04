"""Export one trained model for the browser demo: web/public/model/{model.onnx, meta.json} and the
golden vectors in web/tests/golden/.

Variant rule: use int8_static if its val top-1 (from quant_report.json) is within 1.0 percentage
point of fp32 val top-1, otherwise fp32. This script prints the comparison; --variant decides.
The confidence threshold is calibrated on val; the test split is never read.
"""

import argparse
import importlib.util
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType

from edgevision.web_export import (
    ASL_SIGNS_CLASSES,
    DATA_SOURCES,
    DEFAULT_TARGET_PRECISION,
    WEB_VARIANTS,
    export_web,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "tests" / "conftest.py"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="run dir, e.g. .../seed_0")
    parser.add_argument("--variant", choices=WEB_VARIANTS, required=True)
    parser.add_argument("--quant-root", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--target-precision", type=float, default=DEFAULT_TARGET_PRECISION)
    parser.add_argument("--out", type=Path, default=Path("web/public/model"))
    parser.add_argument("--golden-dir", type=Path, default=Path("web/tests/golden"))
    parser.add_argument(
        "--data-source",
        choices=DATA_SOURCES,
        default="asl-signs",
        help="recorded in meta.json and model_cases.json; pass synthetic for fixture data",
    )
    return parser.parse_args(argv)


def load_fixtures() -> ModuleType:
    """tests/conftest.py, whose synthetic sequence builders generate the preprocessing cases."""
    spec = importlib.util.spec_from_file_location("edgevision_test_fixtures", FIXTURES)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {FIXTURES}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def percent(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.2f}%"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    fixtures = load_fixtures()
    result = export_web(
        args.run,
        args.variant,
        args.quant_root,
        args.processed_dir,
        out_dir=args.out,
        golden_dir=args.golden_dir,
        landmark_blocks=fixtures.landmark_blocks,
        long_frame=fixtures.long_frame,
        target_precision=args.target_precision,
        data_source=args.data_source,
    )
    meta = result.meta
    selection = meta.variant_selection
    drop = "n/a" if selection.drop_pp is None else f"{selection.drop_pp:+.2f} pp"
    print(f"run {meta.run} ({meta.model_name}), data source: {meta.data_source}")
    print(
        f"variant rule: fp32 val top-1 {percent(selection.fp32_val_top1)}, int8_static val top-1 "
        f"{percent(selection.int8_static_val_top1)}, drop {drop} (limit "
        f"{selection.max_drop_pp:.1f} pp) -> recommended {selection.recommended}"
    )
    if not selection.follows_rule:
        print(f"note: exporting {selection.chosen}, not the recommended {selection.recommended}")
    threshold = meta.threshold
    print(
        f"threshold {threshold.threshold:.6f} ({threshold.method}): coverage "
        f"{percent(threshold.coverage)}, accuracy above threshold "
        f"{percent(threshold.accuracy_above_threshold)}, overall val top-1 "
        f"{percent(threshold.overall_val_top1)} on {threshold.val_sequences} val sequences"
    )
    if threshold.warning:
        print(f"warning: {threshold.warning}")
    if meta.data_source == "asl-signs" and meta.num_classes != ASL_SIGNS_CLASSES:
        print(
            f"warning: {meta.num_classes} classes, but asl-signs has {ASL_SIGNS_CLASSES}; "
            "pass --data-source synthetic for fixture data"
        )
    if meta.data_source == "synthetic":
        print("warning: synthetic data; the exported model does not recognise real signs")
    print(f"wrote {result.model_path} ({meta.model_bytes} bytes, sha256 {meta.model_sha256})")
    print(f"wrote {result.meta_path}")
    print(f"wrote {result.preprocess_golden_path} and {result.model_golden_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
