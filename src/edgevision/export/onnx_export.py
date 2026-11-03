"""ONNX export of a full model, FeatureBlock included."""

import warnings
from pathlib import Path

import onnx
import torch

from edgevision.layout import FRAME_DIM, INPUT_NAME, NUM_FRAMES, OUTPUT_NAME

DEFAULT_OPSET = 17
# A batch of 1 could let the tracer fold the batch size into a constant; 2 cannot.
_TRACE_BATCH = 2

# Known warnings of the TorchScript exporter, each matched narrowly and judged harmless here.
_EXPECTED_WARNINGS: tuple[tuple[type[Warning], str, str], ...] = (
    # The legacy exporter is chosen on purpose; see export_onnx.
    (DeprecationWarning, r"You are using the legacy TorchScript-based ONNX export", ""),
    # torch's own exporter calls one of its deprecated helpers.
    (DeprecationWarning, r"The feature will be removed", r"torch\.onnx\."),
    # A torch.jit.TracerWarning (not re-exported in torch's stubs, hence Warning here): nn.GRU
    # validates input and hidden sizes on traced shapes, which are fixed for these models.
    (Warning, r"Converting a tensor to a Python boolean", r"torch\.nn\.modules\.rnn"),
    # Emitted for every RNN export; the parity tests run batch sizes 1 and 7 to rule it out.
    (UserWarning, r"Exporting a model to ONNX with a batch_size other than 1", r"torch\.onnx\."),
)


def export_onnx(model: torch.nn.Module, path: Path, opset: int = DEFAULT_OPSET) -> Path:
    """Export with input "landmarks" (batch, 32, 100) and output "logits", batch axis dynamic.

    Uses the TorchScript exporter (dynamo=False): the dynamo exporter needs onnxscript, which
    this project does not depend on. The model is put in eval mode and left there.
    """
    model.eval()
    device = next(model.parameters()).device
    example = torch.zeros(_TRACE_BATCH, NUM_FRAMES, FRAME_DIM, device=device)
    path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings(), torch.no_grad():
        for category, message, module in _EXPECTED_WARNINGS:
            warnings.filterwarnings("ignore", message=message, category=category, module=module)
        torch.onnx.export(
            model,
            (example,),
            str(path),
            input_names=[INPUT_NAME],
            output_names=[OUTPUT_NAME],
            dynamic_axes={INPUT_NAME: {0: "batch"}, OUTPUT_NAME: {0: "batch"}},
            opset_version=opset,
            dynamo=False,
        )
    onnx.checker.check_model(str(path))
    return path
