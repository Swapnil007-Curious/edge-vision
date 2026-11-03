"""onnxruntime sessions with explicit CPU settings, shared by evaluation, quantisation and
benchmarks, plus the metadata keys this project writes into its ONNX files.

onnxruntime ships no type information, so it is imported only here and wrapped in OrtModel.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

# onnxruntime ships no py.typed marker or stubs.
import onnxruntime as ort  # type: ignore[import-untyped]

from edgevision.layout import INPUT_NAME, OUTPUT_NAME, layout_dict

PROVIDERS = ("CPUExecutionProvider",)

META_VARIANT = "edgevision.variant"
META_SOURCE_SHA256 = "edgevision.source_checkpoint_sha256"
META_LABEL_MAP = "edgevision.label_map"
META_INPUT_LAYOUT = "edgevision.input_layout"


def onnxruntime_version() -> str:
    return str(ort.__version__)


def available_providers() -> list[str]:
    return [str(provider) for provider in ort.get_available_providers()]


def session_settings(intra_op_threads: int | None) -> dict[str, Any]:
    """What OrtModel configures, in JSON-ready form; None where threads vary per session."""
    return {
        "providers": list(PROVIDERS),
        "graph_optimization_level": "ORT_ENABLE_ALL",
        "execution_mode": "ORT_SEQUENTIAL",
        "intra_op_num_threads": intra_op_threads,
        "inter_op_num_threads": 1,
    }


class OrtModel:
    """One ONNX file in a CPU InferenceSession: ORT_ENABLE_ALL, sequential execution,
    ``intra_op_threads`` intra-op threads and one inter-op thread."""

    def __init__(self, path: Path, intra_op_threads: int = 4):
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.intra_op_num_threads = intra_op_threads
        options.inter_op_num_threads = 1
        self.path = Path(path)
        self.session = ort.InferenceSession(
            str(path), sess_options=options, providers=list(PROVIDERS)
        )

    @property
    def metadata(self) -> dict[str, str]:
        custom = self.session.get_modelmeta().custom_metadata_map
        return {str(key): str(value) for key, value in custom.items()}

    @property
    def num_outputs(self) -> int:
        """Size of the logits axis, which export_onnx leaves static."""
        return int(self.session.get_outputs()[0].shape[1])

    def run(self, x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """(n, 32, 100) float32 in, (n, num_classes) float32 logits out."""
        (logits,) = self.session.run([OUTPUT_NAME], {INPUT_NAME: x})
        return np.asarray(logits, dtype=np.float32)


def label_map_from_metadata(metadata: dict[str, str], path: Path) -> dict[int, str]:
    """The label map an ONNX file carries; refuses files made for another input layout."""
    if META_LABEL_MAP not in metadata or META_INPUT_LAYOUT not in metadata:
        raise ValueError(f"{path}: no edgevision metadata; build it with scripts/quantize.py")
    if json.loads(metadata[META_INPUT_LAYOUT]) != layout_dict():
        raise ValueError(f"{path}: model input layout differs from edgevision.layout")
    return {int(index): str(sign) for index, sign in json.loads(metadata[META_LABEL_MAP]).items()}
