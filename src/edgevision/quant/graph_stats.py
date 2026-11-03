"""Node statistics of an ONNX graph: what a quantiser converted and what stayed float."""

from collections import Counter
from typing import Any

import onnx

COMPUTE_OPS = ("MatMul", "Gemm", "Conv", "GRU")
INTEGER_COMPUTE_OPS = (
    "MatMulInteger",
    "ConvInteger",
    "DynamicQuantizeMatMul",
    "QLinearMatMul",
    "QLinearConv",
)
QDQ_OPS = ("QuantizeLinear", "DequantizeLinear")


def graph_stats(model: onnx.ModelProto) -> dict[str, Any]:
    """Op counts and the quantised share of compute nodes, counted on the saved graph.

    A compute node counts as quantised if it is an integer op, or (QDQ format) a float op whose
    data and weight inputs both come from DequantizeLinear nodes, which onnxruntime fuses into
    an integer kernel. Every other MatMul, Gemm, Conv or GRU node counts as float.
    """
    nodes = list(model.graph.node)
    op_counts = Counter(node.op_type for node in nodes)
    dequantized = {
        output for node in nodes if node.op_type == "DequantizeLinear" for output in node.output
    }
    float_nodes: Counter[str] = Counter()
    qdq_nodes: Counter[str] = Counter()
    for node in nodes:
        if node.op_type not in COMPUTE_OPS:
            continue
        operands = list(node.input[:2])
        if len(operands) == 2 and all(name in dequantized for name in operands):
            qdq_nodes[node.op_type] += 1
        else:
            float_nodes[node.op_type] += 1
    quantised = sum(op_counts[op] for op in INTEGER_COMPUTE_OPS) + sum(qdq_nodes.values())
    total = quantised + sum(float_nodes.values())
    return {
        "node_count": len(nodes),
        "op_counts": dict(sorted(op_counts.items())),
        "float_compute_nodes": {op: float_nodes[op] for op in COMPUTE_OPS},
        "qdq_compute_nodes": {op: qdq_nodes[op] for op in COMPUTE_OPS},
        "integer_and_qdq_nodes": {op: op_counts[op] for op in INTEGER_COMPUTE_OPS + QDQ_OPS},
        "fraction_compute_ops_quantised": quantised / total if total else 0.0,
        "gru_float_remaining": float_nodes["GRU"],
        "any_gru_float": float_nodes["GRU"] > 0,
    }
