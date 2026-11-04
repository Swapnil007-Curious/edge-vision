"""Export one trained model for the browser demo in web/.

Writes ``<out>/model.onnx`` (a copy of one quantisation variant), ``<out>/meta.json`` (labels,
input layout, preprocessing settings, hashes and the confidence threshold) and the golden vectors
that the web test suite compares the browser against. Only the val split is read; asking for any
other split raises.
"""

import hashlib
import json
import platform
import shutil
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, NamedTuple, Self

import numpy as np
import numpy.typing as npt
import onnx
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from edgevision.config import PreprocessConfig
from edgevision.data.io import to_storage
from edgevision.data.landmarks import N_HAND
from edgevision.data.preprocess import MIN_SCALE, X_GRID, preprocess_sequence, snap_x
from edgevision.layout import (
    FRAME_DIM,
    NUM_COORDS,
    NUM_FRAMES,
    POSE_INDICES,
    layout_dict,
    to_model_input,
)
from edgevision.runs import (
    CHECKPOINT_FILE,
    QUANT_REPORT_FILE,
    SUMMARY_FILE,
    variant_file,
)
from edgevision.runtime import (
    META_SOURCE_SHA256,
    META_VARIANT,
    OrtModel,
    label_map_from_metadata,
    onnxruntime_version,
)
from edgevision.train.data import SplitArrays, load_split_arrays

FloatArray = npt.NDArray[np.float64]
Float32Array = npt.NDArray[np.float32]

WebVariant = Literal["fp32", "fp16", "int8_dynamic", "int8_static"]
WEB_VARIANTS: tuple[WebVariant, ...] = ("fp32", "fp16", "int8_dynamic", "int8_static")
DataSource = Literal["asl-signs", "synthetic"]
DATA_SOURCES: tuple[DataSource, ...] = ("asl-signs", "synthetic")
ASL_SIGNS_CLASSES = 250

EVAL_SPLIT = "val"
DEFAULT_TARGET_PRECISION = 0.90
FALLBACK_MIN_COVERAGE = 0.2
VARIANT_RULE_MAX_DROP_PP = 1.0
# Absorbs float noise in (fp32 - int8) * 100, e.g. 0.85 - 0.84 = 0.0100000000000000089.
_PP_EPSILON = 1e-9
MODEL_CASES = 16
META_SCHEMA_VERSION: Literal[1] = 1
GOLDEN_SCHEMA_VERSION: Literal[1] = 1
INPUT_DECIMALS = 6
GOLDEN_SEED = 2026

MODEL_FILE = "model.onnx"
META_FILE = "meta.json"
PREPROCESS_CASES_FILE = "preprocess_cases.json"
MODEL_CASES_FILE = "model_cases.json"
PREPROCESS_REPORT_FILE = "preprocess_report.json"
SHA256_PATTERN = r"^[0-9a-f]{64}$"

VARIANT_RULE = (
    f"use int8_static if its val top-1 is within {VARIANT_RULE_MAX_DROP_PP:.1f} percentage point "
    "of fp32 val top-1 (from quant_report.json), otherwise fp32"
)
FLOAT_ENCODING = (
    "float32 arrays are written as the shortest decimal that rounds to the same float32; "
    "raw input coordinates are exact float64 values (rounded to 6 decimals when generated); "
    "null is NaN"
)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# ---------------------------------------------------------------------------------------------
# Split access


def load_eval_split(processed_dir: Path, splits_path: Path, split: str = EVAL_SPLIT) -> SplitArrays:
    """The val split; any other split name raises, so the export cannot read test."""
    if split != EVAL_SPLIT:
        raise ValueError(f"web export reads only the {EVAL_SPLIT} split, not {split!r}")
    return load_split_arrays(processed_dir, splits_path, [EVAL_SPLIT])[EVAL_SPLIT]


# ---------------------------------------------------------------------------------------------
# Variant rule


class VariantSelection(_StrictModel):
    rule: str
    fp32_val_top1: float
    int8_static_val_top1: float | None
    drop_pp: float | None
    max_drop_pp: float
    recommended: Literal["fp32", "int8_static"]
    chosen: WebVariant
    follows_rule: bool


def _val_top1(variants: Mapping[str, Any], variant: str) -> float | None:
    entry = variants.get(variant) or {}
    if entry.get("error"):
        return None
    top1 = (entry.get("val") or {}).get("top1")
    return None if top1 is None else float(top1)


def select_variant(report: Mapping[str, Any], chosen: WebVariant) -> VariantSelection:
    """Apply VARIANT_RULE to a quant_report.json; ``chosen`` is what the caller exports."""
    variants = report.get("variants") or {}
    fp32 = _val_top1(variants, "fp32")
    if fp32 is None:
        raise ValueError("quant_report.json has no fp32 val top-1; rerun scripts/quantize.py")
    int8 = _val_top1(variants, "int8_static")
    drop = None if int8 is None else (fp32 - int8) * 100.0
    within = drop is not None and drop <= VARIANT_RULE_MAX_DROP_PP + _PP_EPSILON
    recommended: Literal["fp32", "int8_static"] = "int8_static" if within else "fp32"
    return VariantSelection(
        rule=VARIANT_RULE,
        fp32_val_top1=fp32,
        int8_static_val_top1=int8,
        drop_pp=drop,
        max_drop_pp=VARIANT_RULE_MAX_DROP_PP,
        recommended=recommended,
        chosen=chosen,
        follows_rule=chosen == recommended,
    )


# ---------------------------------------------------------------------------------------------
# Confidence threshold


class ThresholdBlock(_StrictModel):
    """Accept a prediction when its softmax max-probability is >= ``threshold``."""

    computed_on: Literal["val"]
    target_precision: float = Field(gt=0.0, le=1.0)
    threshold: float = Field(ge=0.0, le=1.0)
    coverage: float = Field(ge=0.0, le=1.0)
    accuracy_above_threshold: float = Field(ge=0.0, le=1.0)
    overall_val_top1: float = Field(ge=0.0, le=1.0)
    val_sequences: int = Field(ge=1)
    method: Literal["target_precision", "fallback_max_accuracy"]
    fallback_min_coverage: float
    warning: str | None


def softmax(logits: npt.NDArray[np.floating[Any]]) -> FloatArray:
    """Row-wise softmax in float64."""
    values = np.asarray(logits, dtype=np.float64)
    exp = np.exp(values - values.max(axis=1, keepdims=True))
    result: FloatArray = exp / exp.sum(axis=1, keepdims=True)
    return result


def select_threshold(
    max_prob: npt.NDArray[np.floating[Any]],
    correct: npt.NDArray[np.bool_],
    target_precision: float = DEFAULT_TARGET_PRECISION,
    min_coverage: float = FALLBACK_MIN_COVERAGE,
) -> ThresholdBlock:
    """The smallest t with top-1 accuracy >= ``target_precision`` among samples with p >= t.

    Candidates are the distinct max-probabilities, so accuracy above t is exactly what was
    measured. If no candidate reaches the target, the candidate with the highest accuracy among
    those accepting at least ``min_coverage`` of the samples is used (the smallest such t on
    ties) and a warning is recorded.
    """
    p = np.asarray(max_prob, dtype=np.float64)
    ok = np.asarray(correct, dtype=bool)
    if p.ndim != 1 or p.shape != ok.shape or p.size == 0:
        raise ValueError(f"expected two equal non-empty 1-D arrays, got {p.shape} and {ok.shape}")
    if not 0.0 < target_precision <= 1.0:
        raise ValueError(f"target_precision must be in (0, 1], got {target_precision}")
    order = np.argsort(p, kind="stable")
    p_sorted, ok_sorted = p[order], ok[order]
    candidates, first = np.unique(p_sorted, return_index=True)
    n = p.size
    accepted = n - first
    correct_from = np.cumsum(ok_sorted[::-1])[::-1]
    accuracy = correct_from[first] / accepted
    coverage = accepted / n

    reaching = np.flatnonzero(accuracy >= target_precision)
    warning = None
    method: Literal["target_precision", "fallback_max_accuracy"] = "target_precision"
    if reaching.size:
        index = int(reaching[0])
    else:
        eligible = np.flatnonzero(coverage >= min_coverage)
        index = int(eligible[np.argmax(accuracy[eligible])])
        method = "fallback_max_accuracy"
        warning = (
            f"no threshold reaches top-1 accuracy {target_precision:.2f} on val; using the "
            f"threshold with the highest accuracy among those accepting at least "
            f"{min_coverage:.0%} of val sequences"
        )
    return ThresholdBlock(
        computed_on="val",
        target_precision=target_precision,
        threshold=float(candidates[index]),
        coverage=float(coverage[index]),
        accuracy_above_threshold=float(accuracy[index]),
        overall_val_top1=float(ok.mean()),
        val_sequences=n,
        method=method,
        fallback_min_coverage=min_coverage,
        warning=warning,
    )


def threshold_from_logits(
    logits: Float32Array,
    labels: npt.NDArray[np.int64],
    target_precision: float = DEFAULT_TARGET_PRECISION,
) -> ThresholdBlock:
    probabilities = softmax(logits)
    correct = probabilities.argmax(axis=1) == labels
    return select_threshold(probabilities.max(axis=1), correct, target_precision)


# ---------------------------------------------------------------------------------------------
# meta.json


class InputLayout(_StrictModel):
    """Mirrors edgevision.layout.layout_dict(); extra or missing keys fail validation."""

    num_frames: int
    num_landmarks: int
    coords_per_landmark: int
    frame_dim: int
    coordinate_order: Literal["landmark_major_xy"]
    landmark_order: list[str]
    pose_indices: list[int]
    presence_left_index: int
    presence_right_index: int
    missing_point: str
    input_name: str
    output_name: str


class PreprocessingSettings(_StrictModel):
    num_frames: int = Field(ge=1)
    use_z: Literal[False]
    pose_indices: list[int]
    min_hand_frames: int = Field(ge=1)
    min_scale: float
    x_grid: float
    storage_dtype: Literal["float16"]
    source: str


class WebMeta(_StrictModel):
    schema_version: Literal[1]
    model_name: str
    variant: WebVariant
    data_source: DataSource
    run: str
    seed: int
    num_classes: int = Field(ge=2)
    labels: list[str]
    num_frames: int
    landmark_order: list[str]
    pose_indices: list[int]
    input_layout: InputLayout
    preprocessing: PreprocessingSettings
    opset: int
    model_file: str
    model_bytes: int = Field(ge=1)
    model_sha256: str = Field(pattern=SHA256_PATTERN)
    source_checkpoint_sha256: str = Field(pattern=SHA256_PATTERN)
    threshold: ThresholdBlock
    variant_selection: VariantSelection
    tool_versions: dict[str, str]

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        problems = []
        if len(self.labels) != self.num_classes:
            problems.append(f"{len(self.labels)} labels for {self.num_classes} classes")
        if len(set(self.labels)) != len(self.labels):
            problems.append("labels are not unique")
        frames = {self.num_frames, self.input_layout.num_frames, self.preprocessing.num_frames}
        if len(frames) != 1:
            problems.append("num_frames differs between fields")
        if len(self.landmark_order) != self.input_layout.num_landmarks:
            problems.append("landmark_order length differs from num_landmarks")
        pose = (self.pose_indices, self.input_layout.pose_indices, self.preprocessing.pose_indices)
        if pose[0] != pose[1] or pose[0] != pose[2]:
            problems.append("pose_indices differ between fields")
        if self.variant != self.variant_selection.chosen:
            problems.append("variant differs from variant_selection.chosen")
        if problems:
            raise ValueError("; ".join(problems))
        return self


def landmark_names(pose_indices: Sequence[int] = POSE_INDICES) -> list[str]:
    """Selected landmark order: left_hand_0..20, right_hand_0..20, pose_<index>..."""
    hands = [f"{side}_{index}" for side in ("left_hand", "right_hand") for index in range(N_HAND)]
    return hands + [f"pose_{index}" for index in pose_indices]


def preprocessing_settings(processed_dir: Path) -> PreprocessingSettings:
    """Settings from the processed dir's preprocess_report.json, else PreprocessConfig defaults.

    Refuses settings the exported input layout cannot represent.
    """
    report_path = processed_dir / PREPROCESS_REPORT_FILE
    if report_path.is_file():
        with report_path.open(encoding="utf-8") as handle:
            report = json.load(handle)
        values = dict(report["config"]["preprocess"])
        if "landmarks" in report:  # The effective selection, recorded since v2.
            values["pose_indices"] = report["landmarks"]["pose_indices"]
        source = PREPROCESS_REPORT_FILE
    else:
        fields = PreprocessConfig.model_fields
        values = {
            "num_frames": fields["num_frames"].default,
            "use_z": fields["use_z"].default,
            "pose_indices": list(POSE_INDICES),
            "min_hand_frames": fields["min_hand_frames"].default,
        }
        source = f"PreprocessConfig defaults (no {PREPROCESS_REPORT_FILE} in the processed dir)"
    if (
        values["num_frames"] != NUM_FRAMES
        or list(values["pose_indices"]) != list(POSE_INDICES)
        or values["use_z"]
    ):
        raise ValueError(
            f"preprocessing settings {values} do not match the model input layout "
            f"(num_frames {NUM_FRAMES}, pose_indices {list(POSE_INDICES)}, use_z False)"
        )
    return PreprocessingSettings(
        num_frames=int(values["num_frames"]),
        use_z=False,
        pose_indices=[int(index) for index in values["pose_indices"]],
        min_hand_frames=int(values["min_hand_frames"]),
        min_scale=MIN_SCALE,
        x_grid=X_GRID,
        storage_dtype="float16",
        source=source,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def onnx_opset(path: Path) -> int:
    model = onnx.load(str(path), load_external_data=False)
    versions = [entry.version for entry in model.opset_import if entry.domain in ("", "ai.onnx")]
    if not versions:
        raise ValueError(f"{path}: no default-domain opset")
    return int(versions[0])


def tool_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "onnx": onnx.__version__,
        "onnxruntime": onnxruntime_version(),
    }


# ---------------------------------------------------------------------------------------------
# Golden vectors


class GoldenRows(_StrictModel):
    """Long-format input rows, column-wise."""

    frame: list[int]
    type: list[str]
    landmark_index: list[int]
    x: list[float | None]
    y: list[float | None]
    z: list[float | None]

    @model_validator(mode="after")
    def _equal_lengths(self) -> Self:
        lengths = {len(getattr(self, name)) for name in type(self).model_fields}
        if len(lengths) != 1:
            raise ValueError(f"row columns differ in length: {sorted(lengths)}")
        return self


class GoldenInfo(_StrictModel):
    t_raw: int
    t_kept: int
    mirrored: bool
    norm_method: Literal["shoulders", "hand_fallback"] | None
    excluded: bool
    reason: Literal["insufficient_hand_frames", "degenerate_scale"] | None


def _check_matrix(name: str, value: list[list[float]] | None, columns: int) -> None:
    if value is None:
        return
    if len(value) != NUM_FRAMES or any(len(row) != columns for row in value):
        raise ValueError(f"{name} must be {NUM_FRAMES} x {columns}")


class PreprocessCase(_StrictModel):
    """One sequence: features (num_frames x FRAME_DIM, float32) are the preprocessed coordinates
    then presence; model_input is the same after the float16 storage step training data took."""

    name: str
    description: str
    rows: GoldenRows
    info: GoldenInfo
    features: list[list[float]] | None
    presence: list[list[float]] | None
    model_input: list[list[float]] | None

    @model_validator(mode="after")
    def _shapes(self) -> Self:
        arrays = (self.features, self.presence, self.model_input)
        if self.info.excluded != all(array is None for array in arrays):
            raise ValueError("excluded cases have no arrays; kept cases have all three")
        if self.info.excluded != (self.info.reason is not None):
            raise ValueError("reason must be set exactly when the case is excluded")
        _check_matrix("features", self.features, FRAME_DIM)
        _check_matrix("presence", self.presence, 2)
        _check_matrix("model_input", self.model_input, FRAME_DIM)
        return self


class GoldenPreprocessConfig(_StrictModel):
    num_frames: int
    use_z: Literal[False]
    pose_indices: list[int]
    min_hand_frames: int


class SnapCase(_StrictModel):
    x: float
    snapped: float


class PreprocessGolden(_StrictModel):
    schema_version: Literal[1]
    description: str
    float_encoding: str
    config: GoldenPreprocessConfig
    snap_x_cases: list[SnapCase]
    cases: list[PreprocessCase] = Field(min_length=12)


class ModelCase(_StrictModel):
    sequence_id: int
    label: int
    input: list[list[float]]
    logits: list[float]

    @model_validator(mode="after")
    def _shapes(self) -> Self:
        _check_matrix("input", self.input, FRAME_DIM)
        return self


class ModelGolden(_StrictModel):
    schema_version: Literal[1]
    description: str
    float_encoding: str
    source: DataSource
    split: Literal["val"]
    variant: WebVariant
    model_sha256: str = Field(pattern=SHA256_PATTERN)
    num_classes: int
    cases: list[ModelCase] = Field(min_length=MODEL_CASES, max_length=MODEL_CASES)

    @model_validator(mode="after")
    def _logit_widths(self) -> Self:
        if any(len(case.logits) != self.num_classes for case in self.cases):
            raise ValueError(f"every case needs {self.num_classes} logits")
        return self


def float32_lists(array: npt.ArrayLike) -> Any:
    """Nested lists of the shortest decimals that round back to the same float32 values."""
    values = np.asarray(array, dtype=np.float32)
    short = np.array([float(str(value)) for value in values.ravel()], dtype=np.float64)
    # Decimal -> float64 -> float32 can in principle round differently from decimal -> float32;
    # any such value is written as its exact float64 instead.
    mismatch = short.astype(np.float32) != values.ravel()
    short[mismatch] = values.ravel()[mismatch].astype(np.float64)
    return short.reshape(values.shape).tolist()


def _nullable(values: npt.NDArray[np.float64]) -> list[float | None]:
    return [None if np.isnan(value) else float(value) for value in values.tolist()]


def rows_to_json(df: pd.DataFrame) -> GoldenRows:
    return GoldenRows(
        frame=[int(value) for value in df["frame"]],
        type=[str(value) for value in df["type"]],
        landmark_index=[int(value) for value in df["landmark_index"]],
        x=_nullable(df["x"].to_numpy(dtype=np.float64)),
        y=_nullable(df["y"].to_numpy(dtype=np.float64)),
        z=_nullable(df["z"].to_numpy(dtype=np.float64)),
    )


def rows_from_json(rows: GoldenRows) -> pd.DataFrame:
    def column(values: list[float | None]) -> FloatArray:
        return np.array([np.nan if value is None else value for value in values], dtype=np.float64)

    return pd.DataFrame(
        {
            "frame": np.asarray(rows.frame, dtype=np.int64),
            "type": rows.type,
            "landmark_index": np.asarray(rows.landmark_index, dtype=np.int64),
            "x": column(rows.x),
            "y": column(rows.y),
            "z": column(rows.z),
        }
    )


BlockBuilder = Callable[[int, np.random.Generator, str], dict[str, FloatArray]]
FrameBuilder = Callable[[Mapping[str, FloatArray]], pd.DataFrame]


class RawCase(NamedTuple):
    name: str
    description: str
    rows: pd.DataFrame


def synthetic_cases(
    landmark_blocks: BlockBuilder, long_frame: FrameBuilder, seed: int = GOLDEN_SEED
) -> list[RawCase]:
    """Long-format sequences covering every preprocessing branch.

    ``landmark_blocks`` and ``long_frame`` are the test suite's synthetic fixture builders
    (tests/conftest.py). Face rows are dropped except in one case, to keep the file small;
    coordinates are rounded to INPUT_DECIMALS so the JSON stores them exactly.
    """
    rng = np.random.default_rng(seed)
    nan = np.nan

    def blocks(n_frames: int, case: str = "both_hands", face: bool = False) -> dict[str, Any]:
        built = dict(landmark_blocks(n_frames, rng, case))
        if not face:
            del built["face"]
        return built

    cases: list[tuple[str, str, Mapping[str, FloatArray]]] = []

    b = blocks(45, "right_hand_only")
    cases.append(("right_only_longer", "Right hand only for 45 frames, downsampled to 32.", b))

    b = blocks(12, "both_hands")
    cases.append(
        (
            "both_hands_tie_shorter",
            "Both hands in all 12 frames: a presence tie, so no mirroring (ties go to the right "
            "hand); upsampled to 32.",
            b,
        )
    )

    b = blocks(30, "left_hand_only")
    cases.append(("left_only_mirrored", "Left hand only: mirrored to the right.", b))

    b = blocks(36, "both_hands")
    b["right_hand"][5:17] = nan
    cases.append(
        (
            "left_dominant_both_hands",
            "Both hands, right hand missing in frames 5-16 (left 36 vs right 24 frames): mirrored, "
            "hand blocks, pose pairs and presence columns swapped.",
            b,
        )
    )

    b = blocks(30, "both_hands")
    b["left_hand"][20:] = nan
    b["right_hand"][:10] = nan
    cases.append(
        (
            "tie_with_edge_gaps",
            "Left hand in frames 0-19, right hand in frames 10-29: a 20-20 tie, not mirrored; "
            "each hand is NaN (then 0) outside its own span, without extrapolation.",
            b,
        )
    )

    b = blocks(25, "pose_missing")
    cases.append(
        (
            "pose_nan_hand_fallback",
            "Pose NaN in every frame: no shoulders, so the hands' bounding box normalises.",
            b,
        )
    )

    b = blocks(20, "left_hand_only")
    del b["pose"]
    cases.append(
        (
            "pose_rows_absent_mirrored_fallback",
            "No pose rows at all and only the left hand: mirrored, hand_fallback normalisation.",
            b,
        )
    )

    b = blocks(40, "right_hand_only")
    b["right_hand"][12:19] = nan
    cases.append(
        (
            "hand_less_frames_dropped",
            "Right hand missing in frames 12-18: those frames are dropped before resampling "
            "(t_raw 40, t_kept 33).",
            b,
        )
    )

    b = blocks(28, "both_hands")
    b["right_hand"][8:14, :5] = nan
    b["left_hand"][20:, 10] = nan
    cases.append(
        (
            "partial_hand_landmarks",
            "Right hand has 5 NaN landmarks in frames 8-13 and the left hand 1 in frames 20-27: "
            "those hands count as absent there, but their finite points are kept and resampled.",
            b,
        )
    )

    b = blocks(33, "partial_hands")
    cases.append(
        (
            "fixture_partial_hands",
            "The fixture's partial_hands case: no left hand, right hand missing in the first "
            "third of the frames.",
            b,
        )
    )

    b = blocks(30, "right_hand_only")
    b["pose"][:6, 11] = nan
    b["pose"][24:, 12] = nan
    b["pose"][10:12, [11, 12], 1] = nan
    cases.append(
        (
            "shoulders_partly_missing",
            "Left shoulder NaN in frames 0-5, right shoulder in frames 24-29 and both shoulder y "
            "values in frames 10-11: medians use only finite values, per axis.",
            b,
        )
    )

    b = blocks(22, "right_hand_only")
    b["pose"][:, 12] = nan
    cases.append(
        (
            "one_shoulder_never_visible",
            "Right shoulder NaN in every frame: no frame has both shoulders, so hand_fallback.",
            b,
        )
    )

    b = blocks(18, "right_hand_only")
    keep = np.zeros(18, dtype=bool)
    keep[[2, 9, 15]] = True
    b["right_hand"][~keep] = nan
    cases.append(
        (
            "exactly_min_hand_frames",
            "A hand in exactly 3 frames (the minimum): kept and upsampled from 3 to 32 frames.",
            b,
        )
    )

    b = blocks(20, "right_hand_only")
    keep = np.zeros(20, dtype=bool)
    keep[[4, 11]] = True
    b["right_hand"][~keep] = nan
    cases.append(
        (
            "excluded_insufficient_hand_frames",
            "A hand in only 2 frames: excluded as insufficient_hand_frames.",
            b,
        )
    )

    b = blocks(16, "coincident_shoulders")
    cases.append(
        (
            "excluded_degenerate_shoulders",
            "Both shoulders at the same point: shoulder distance 0, excluded as degenerate_scale.",
            b,
        )
    )

    b = blocks(10, "right_hand_only")
    del b["pose"]
    b["right_hand"][..., 0] = 0.4
    b["right_hand"][..., 1] = 0.5
    cases.append(
        (
            "excluded_degenerate_hand_fallback",
            "No pose and every hand point at one position: zero hand spread, excluded as "
            "degenerate_scale under hand_fallback.",
            b,
        )
    )

    b = blocks(8, "both_hands", face=True)
    b["right_hand"][[1, 6]] = nan
    cases.append(
        (
            "face_rows_shuffled_frame_gaps",
            "Includes face rows (dropped), rows in random order and non-consecutive frame "
            "numbers; left 8 vs right 6 frames, so mirrored.",
            b,
        )
    )

    b = blocks(24, "right_hand_only")
    b["right_hand"][..., 0] += 0.75
    b["pose"][:, [15, 16], 1] -= 0.8
    cases.append(
        (
            "coordinates_outside_unit_square",
            "Hand x beyond 1.0 and wrist y below 0.0, as when a hand leaves the frame: such "
            "values are used as they are.",
            b,
        )
    )

    raw = []
    for name, description, case_blocks in cases:
        df = long_frame(case_blocks).loc[:, ["frame", "type", "landmark_index", "x", "y", "z"]]
        if name == "face_rows_shuffled_frame_gaps":
            frames = np.sort(df["frame"].unique())
            remap = dict(zip(frames, 100 + np.cumsum(rng.integers(1, 5, frames.size)), strict=True))
            df = df.assign(frame=df["frame"].map(remap)).sample(frac=1.0, random_state=seed)
        df = df.assign(
            frame=df["frame"].astype(np.int64),
            landmark_index=df["landmark_index"].astype(np.int64),
            x=df["x"].round(INPUT_DECIMALS),
            y=df["y"].round(INPUT_DECIMALS),
            z=df["z"].round(INPUT_DECIMALS),
        ).reset_index(drop=True)
        raw.append(RawCase(name, description, df))
    return raw


def pack_features(features: Float32Array, presence: Float32Array) -> Float32Array:
    """(T, L, 2) coordinates and (T, 2) presence -> (T, FRAME_DIM) in the model layout."""
    packed = np.empty((features.shape[0], FRAME_DIM), dtype=np.float32)
    packed[:, :NUM_COORDS] = features.reshape(features.shape[0], NUM_COORDS)
    packed[:, NUM_COORDS:] = presence
    return packed


def golden_case(case: RawCase, settings: GoldenPreprocessConfig) -> PreprocessCase:
    result = preprocess_sequence(
        case.rows,
        num_frames=settings.num_frames,
        use_z=False,
        pose_indices=settings.pose_indices,
        min_hand_frames=settings.min_hand_frames,
    )
    info = GoldenInfo.model_validate(dict(result.info))
    if result.features is None or result.hand_presence is None:
        return PreprocessCase(
            name=case.name,
            description=case.description,
            rows=rows_to_json(case.rows),
            info=info,
            features=None,
            presence=None,
            model_input=None,
        )
    stored = to_storage(0, result)
    if stored.features is None or stored.presence is None:
        raise ValueError(f"{case.name}: excluded by the float16 storage step")
    model_input = to_model_input(stored.features[None], stored.presence[None])[0]
    return PreprocessCase(
        name=case.name,
        description=case.description,
        rows=rows_to_json(case.rows),
        info=info,
        features=float32_lists(pack_features(result.features, result.hand_presence)),
        presence=float32_lists(result.hand_presence),
        model_input=float32_lists(model_input),
    )


SNAP_X_INPUTS: tuple[float, ...] = (
    0.5 + 0.5 / X_GRID,  # exactly half-way: rounds to even (down)
    0.25 + 1.5 / X_GRID,  # exactly half-way: rounds to even (up)
    0.7 + 2.5 / X_GRID,
    0.123456789,
    1.0 - 2.0**-40,
    1e-12,
    -0.0,
    -0.3 + 0.5 / X_GRID,
    1.25,
)


def build_preprocess_golden(
    landmark_blocks: BlockBuilder, long_frame: FrameBuilder, min_hand_frames: int
) -> PreprocessGolden:
    settings = GoldenPreprocessConfig(
        num_frames=NUM_FRAMES,
        use_z=False,
        pose_indices=list(POSE_INDICES),
        min_hand_frames=min_hand_frames,
    )
    raw = synthetic_cases(landmark_blocks, long_frame)
    cases = [golden_case(case, settings) for case in raw]
    for case, built in zip(raw, cases, strict=True):
        if not rows_from_json(built.rows).equals(case.rows):
            raise AssertionError(f"{case.name}: rows do not survive the JSON encoding exactly")
    snapped = snap_x(np.asarray(SNAP_X_INPUTS, dtype=np.float64))
    return PreprocessGolden(
        schema_version=GOLDEN_SCHEMA_VERSION,
        description=(
            "Python preprocess_sequence outputs for synthetic long-format sequences built with "
            "the test suite's fixture logic; written by scripts/export_web.py."
        ),
        float_encoding=FLOAT_ENCODING,
        config=settings,
        snap_x_cases=[
            SnapCase(x=float(x), snapped=float(s))
            for x, s in zip(SNAP_X_INPUTS, snapped, strict=True)
        ],
        cases=cases,
    )


def model_case_rows(n_val: int, n_cases: int = MODEL_CASES) -> npt.NDArray[np.intp]:
    """Evenly spaced val rows, so the cases span the split rather than its first sequences."""
    if n_val < n_cases:
        raise ValueError(f"val has {n_val} sequences, {n_cases} model cases are needed")
    return np.unique(np.linspace(0, n_val - 1, n_cases).round().astype(np.intp))


def build_model_golden(
    val: SplitArrays,
    logits: Float32Array,
    *,
    source: DataSource,
    variant: WebVariant,
    model_sha256: str,
) -> ModelGolden:
    rows = model_case_rows(len(val.y))
    return ModelGolden(
        schema_version=GOLDEN_SCHEMA_VERSION,
        description=(
            "Processed val inputs (num_frames x FRAME_DIM) and the logits Python onnxruntime "
            "gives for them with the exported model.onnx; written by scripts/export_web.py."
        ),
        float_encoding=FLOAT_ENCODING,
        source=source,
        split="val",
        variant=variant,
        model_sha256=model_sha256,
        num_classes=int(logits.shape[1]),
        cases=[
            ModelCase(
                sequence_id=int(val.sequence_id[row]),
                label=int(val.y[row]),
                input=float32_lists(val.X[row]),
                logits=float32_lists(logits[row]),
            )
            for row in rows
        ],
    )


def write_golden(path: Path, golden: BaseModel) -> None:
    """Compact JSON with one case per line, written through a temporary name."""
    payload = golden.model_dump(mode="json")
    cases = payload.pop("cases")
    header = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    lines = [json.dumps(case, separators=(",", ":"), allow_nan=False) for case in cases]
    text = header[:-1] + ',"cases":[\n' + ",\n".join(lines) + "\n]}\n"
    _write_text(path, text)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


# ---------------------------------------------------------------------------------------------
# Export


class ExportResult(NamedTuple):
    meta: WebMeta
    model_path: Path
    meta_path: Path
    preprocess_golden_path: Path
    model_golden_path: Path


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload: dict[str, Any] = json.load(handle)
    return payload


def export_web(
    run_directory: Path,
    variant: WebVariant,
    quant_root: Path,
    processed_dir: Path,
    *,
    out_dir: Path,
    golden_dir: Path,
    landmark_blocks: BlockBuilder,
    long_frame: FrameBuilder,
    target_precision: float = DEFAULT_TARGET_PRECISION,
    data_source: DataSource = "asl-signs",
) -> ExportResult:
    """Copy one variant to ``out_dir``, write meta.json there and the golden vectors to
    ``golden_dir``. The variant file must have been built from the run's current best.pt."""
    if variant not in WEB_VARIANTS:
        raise ValueError(f"variant must be one of {WEB_VARIANTS}, got {variant!r}")
    summary = read_json(run_directory / SUMMARY_FILE)
    model_name, seed = summary["model"], int(summary["seed"])
    if summary.get("feature_set", "v1") != "v1":
        raise ValueError(
            f"{run_directory} uses feature set {summary['feature_set']}; the web demo implements "
            "v1 preprocessing only"
        )
    run_key = f"{run_directory.parent.name}/{run_directory.name}"
    quant_dir = quant_root / run_key  # <quant root>/<experiment>/seed_<seed>.
    selection = select_variant(read_json(quant_dir / QUANT_REPORT_FILE), variant)

    source = quant_dir / variant_file(variant)
    checkpoint_sha256 = sha256_file(run_directory / CHECKPOINT_FILE)
    probe = OrtModel(source, intra_op_threads=1)
    metadata = probe.metadata
    if metadata.get(META_SOURCE_SHA256) != checkpoint_sha256:
        raise ValueError(f"{source} was not built from {run_directory / CHECKPOINT_FILE}")
    if metadata.get(META_VARIANT) != variant:
        raise ValueError(f"{source} records variant {metadata.get(META_VARIANT)!r}")
    label_map = label_map_from_metadata(metadata, source)
    run_labels = {int(k): v for k, v in read_json(run_directory / "label_map.json").items()}
    if label_map != run_labels:
        raise ValueError(f"{source} label map differs from the run's label_map.json")
    labels = [label_map[index] for index in range(len(label_map))]

    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / MODEL_FILE
    temporary = model_path.with_name(f"{model_path.name}.tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(model_path)
    model_sha256 = sha256_file(model_path)

    val = load_eval_split(processed_dir, run_directory / summary["splits_file"])
    logits = OrtModel(model_path, intra_op_threads=1).run(val.X)
    threshold = threshold_from_logits(logits, val.y, target_precision)
    settings = preprocessing_settings(processed_dir)

    meta = WebMeta(
        schema_version=META_SCHEMA_VERSION,
        model_name=model_name,
        variant=variant,
        data_source=data_source,
        run=run_key,
        seed=seed,
        num_classes=len(labels),
        labels=labels,
        num_frames=NUM_FRAMES,
        landmark_order=landmark_names(),
        pose_indices=list(POSE_INDICES),
        input_layout=InputLayout.model_validate(layout_dict()),
        preprocessing=settings,
        opset=onnx_opset(model_path),
        model_file=MODEL_FILE,
        model_bytes=model_path.stat().st_size,
        model_sha256=model_sha256,
        source_checkpoint_sha256=checkpoint_sha256,
        threshold=threshold,
        variant_selection=selection,
        tool_versions=tool_versions(),
    )
    meta_path = out_dir / META_FILE
    _write_text(meta_path, json.dumps(meta.model_dump(mode="json"), indent=2) + "\n")

    preprocess_golden = build_preprocess_golden(
        landmark_blocks, long_frame, settings.min_hand_frames
    )
    preprocess_path = golden_dir / PREPROCESS_CASES_FILE
    write_golden(preprocess_path, preprocess_golden)
    model_golden = build_model_golden(
        val, logits, source=data_source, variant=variant, model_sha256=model_sha256
    )
    model_golden_path = golden_dir / MODEL_CASES_FILE
    write_golden(model_golden_path, model_golden)
    return ExportResult(meta, model_path, meta_path, preprocess_path, model_golden_path)
