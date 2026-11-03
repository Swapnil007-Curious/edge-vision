import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
from torch import nn

from edgevision.export.onnx_export import export_onnx
from edgevision.layout import INPUT_NAME, OUTPUT_NAME
from edgevision.models.config import MODEL_NAMES
from edgevision.models.registry import build_model

NUM_CLASSES = 250


def landmark_batch(batch, seed):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, 32, 100, generator=generator)
    points = x[..., :98].view(batch, 32, 49, 2)
    points[torch.rand(batch, 32, 49, generator=generator) < 0.25] = 0.0
    x[..., 98:] = (torch.rand(batch, 32, 2, generator=generator) < 0.6).float()
    return x


def prepared_model(name):
    """Random weights, with non-trivial feature statistics and BatchNorm running statistics."""
    torch.manual_seed(0)
    model = build_model(name, NUM_CLASSES)
    generator = torch.Generator().manual_seed(1)
    with torch.no_grad():
        model.features.mean.copy_(0.5 * torch.randn(198, generator=generator))
        model.features.std.copy_(0.2 + 1.5 * torch.rand(198, generator=generator))
        for module in model.modules():
            if isinstance(module, nn.BatchNorm1d):
                module.running_mean.copy_(
                    0.1 * torch.randn(module.num_features, generator=generator)
                )
                module.running_var.copy_(0.5 + torch.rand(module.num_features, generator=generator))
    return model


@pytest.fixture(scope="module", params=MODEL_NAMES)
def exported(request, tmp_path_factory):
    model = prepared_model(request.param)
    path = export_onnx(model, tmp_path_factory.mktemp("onnx") / f"{request.param}.onnx")
    return request.param, model, path


def test_export_passes_the_checker_with_the_expected_signature(exported):
    _, model, path = exported
    proto = onnx.load(str(path))
    onnx.checker.check_model(proto)
    assert not model.training
    (graph_input,) = proto.graph.input
    (graph_output,) = proto.graph.output
    assert graph_input.name == INPUT_NAME
    assert graph_output.name == OUTPUT_NAME
    input_dims = graph_input.type.tensor_type.shape.dim
    assert input_dims[0].dim_param  # Symbolic batch axis.
    assert [dim.dim_value for dim in input_dims[1:]] == [32, 100]
    assert [dim.dim_value for dim in graph_output.type.tensor_type.shape.dim[1:]] == [NUM_CLASSES]


@pytest.mark.parametrize("batch", [1, 7])
def test_onnxruntime_matches_torch(exported, batch):
    name, model, path = exported
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    x = landmark_batch(batch, seed=batch)
    with torch.no_grad():
        expected = model(x).numpy()
    (actual,) = session.run([OUTPUT_NAME], {INPUT_NAME: x.numpy()})
    assert actual.shape == (batch, NUM_CLASSES)
    difference = float(np.abs(actual - expected).max())
    print(f"{name} batch {batch}: max abs diff {difference:.2e}")
    assert difference < 1e-4


def test_transformer_attention_exports_as_plain_matmul(tmp_path):
    path = export_onnx(prepared_model("transformer"), tmp_path / "transformer.onnx")
    op_types = {node.op_type for node in onnx.load(str(path)).graph.node}
    assert {"MatMul", "Softmax"} <= op_types
    assert not {"Attention", "MultiHeadAttention"} & op_types
