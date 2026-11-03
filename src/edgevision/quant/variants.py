"""Builders for the ONNX variants of one checkpoint: fp32, fp16, int8 dynamic and int8 static.

Each builder returns or writes an onnx.ModelProto; save_with_metadata writes it through a
temporary name with edgevision metadata (variant, label map, input layout, checkpoint sha256).
"""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import onnx

# onnxruntime ships no py.typed marker or stubs.
from onnxruntime.quantization import (  # type: ignore[import-untyped]
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_dynamic,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import (  # type: ignore[import-untyped]
    quant_pre_process,
)
from onnxruntime.transformers.float16 import (  # type: ignore[import-untyped]
    convert_float_to_float16,
)
from onnxruntime.transformers.onnx_model import OnnxModel  # type: ignore[import-untyped]

from edgevision.export.onnx_export import export_onnx
from edgevision.layout import layout_dict
from edgevision.models.registry import load_checkpoint
from edgevision.quant.calibration import TrainCalibrationReader
from edgevision.runtime import META_INPUT_LAYOUT, META_LABEL_MAP, META_SOURCE_SHA256, META_VARIANT

# Exporter scope of SignClassifier.features, which computes velocity and standardisation.
FEATURE_SCOPE = "/features/"

FP16_SETTINGS: dict[str, Any] = {"keep_io_types": True}
DYNAMIC_SETTINGS: dict[str, Any] = {"weight_type": QuantType.QInt8, "per_channel": True}
STATIC_SETTINGS: dict[str, Any] = {
    "quant_format": QuantFormat.QDQ,
    "activation_type": QuantType.QUInt8,
    "weight_type": QuantType.QInt8,
    "per_channel": True,
    "reduce_range": False,
    "calibrate_method": CalibrationMethod.MinMax,
}


def describe(settings: Mapping[str, Any]) -> dict[str, Any]:
    """Settings in JSON-ready form, enums by name."""
    return {key: getattr(value, "name", value) for key, value in settings.items()}


def scratch_path(path: Path, tag: str) -> Path:
    """A hidden working file beside ``path``; callers delete it."""
    return path.with_name(f".{path.stem}.{tag}.onnx")


def read_metadata(path: Path) -> dict[str, str]:
    model = onnx.load(str(path), load_external_data=False)
    return {entry.key: entry.value for entry in model.metadata_props}


def save_with_metadata(model: onnx.ModelProto, path: Path, metadata: Mapping[str, str]) -> None:
    """Merge ``metadata`` into the model's properties and write it atomically to ``path``."""
    properties = {entry.key: entry.value for entry in model.metadata_props}
    properties.update(metadata)
    onnx.helper.set_model_props(model, properties)
    temporary = path.with_name(f"{path.name}.tmp")
    onnx.save(model, str(temporary))
    temporary.replace(path)


def feature_block_nodes(model: onnx.ModelProto) -> list[str]:
    names = [node.name for node in model.graph.node if node.name.startswith(FEATURE_SCOPE)]
    if not names:
        raise ValueError(f"no nodes under {FEATURE_SCOPE!r}; was the model written by export_onnx?")
    return names


def export_fp32(checkpoint: Path, path: Path, source_sha256: str) -> None:
    """Export best.pt with export_onnx (opset 17) and attach the metadata every variant copies."""
    loaded = load_checkpoint(checkpoint)
    scratch = scratch_path(path, "export")
    try:
        export_onnx(loaded.model, scratch)
        metadata = {
            META_VARIANT: "fp32",
            META_SOURCE_SHA256: source_sha256,
            META_LABEL_MAP: json.dumps(
                {str(index): sign for index, sign in loaded.label_map.items()}
            ),
            META_INPUT_LAYOUT: json.dumps(layout_dict()),
        }
        save_with_metadata(onnx.load(str(scratch)), path, metadata)
    finally:
        scratch.unlink(missing_ok=True)


def convert_fp16(fp32_path: Path, *, keep_feature_block_fp32: bool) -> onnx.ModelProto:
    """float16 weights and compute with float32 inputs and outputs; optionally the FeatureBlock
    nodes stay float32."""
    model = onnx.load(str(fp32_path))
    block = feature_block_nodes(model) if keep_feature_block_fp32 else None
    converted: onnx.ModelProto = convert_float_to_float16(
        model, node_block_list=block, **FP16_SETTINGS
    )
    drop_identical_nodes(converted.graph)
    # keep_io_types appends the input Cast after the nodes that read it. onnxruntime sorts the
    # graph on load but onnx.checker requires topological order, so restore it (stable sort).
    OnnxModel.graph_topological_sort(converted.graph)
    return converted


def drop_identical_nodes(graph: onnx.GraphProto) -> None:
    """Keep one copy of byte-identical nodes.

    With a node_block_list, convert_float_to_float16 inserts the same float32 Cast (same name,
    input and output) once per blocked consumer of a float16 tensor, which onnxruntime rejects as
    duplicate node names. Identical copies compute the same tensor, so dropping them is exact.
    """
    seen: set[bytes] = set()
    kept = []
    for node in graph.node:
        key = node.SerializeToString(deterministic=True)
        if key not in seen:
            seen.add(key)
            kept.append(node)
    graph.ClearField("node")
    graph.node.extend(kept)


def quantize_int8_dynamic(fp32_path: Path, path: Path) -> onnx.ModelProto:
    scratch = scratch_path(path, "dynamic")
    try:
        quantize_dynamic(fp32_path, scratch, **DYNAMIC_SETTINGS)
        return onnx.load(str(scratch))
    finally:
        scratch.unlink(missing_ok=True)


def quantize_int8_static(
    fp32_path: Path, path: Path, reader: TrainCalibrationReader
) -> onnx.ModelProto:
    """quant_pre_process, then QDQ static quantisation calibrated on ``reader``."""
    preprocessed, scratch = scratch_path(path, "preprocessed"), scratch_path(path, "static")
    try:
        quant_pre_process(fp32_path, preprocessed)
        quantize_static(preprocessed, scratch, reader, **STATIC_SETTINGS)
        return onnx.load(str(scratch))
    finally:
        preprocessed.unlink(missing_ok=True)
        scratch.unlink(missing_ok=True)
