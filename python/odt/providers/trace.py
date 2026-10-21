"""Trace a PyTorch model into a shaped creator IR at batch size 1."""

import operator

import torch
from torch import fx, nn

from elasticai.creator import ir
from elasticai.creator.experimental.ir.shape_inference import (
    ShapedDatagraph,
    get_default_shape_inference,
)
from elasticai.creator.torch2ir import get_default_converter

from .errors import UnsupportedLayerError

SUPPORTED_MODULES: tuple[type[nn.Module], ...] = (
    nn.Conv1d,
    nn.ReLU,
    nn.MaxPool1d,
    nn.AvgPool1d,
    nn.Flatten,
    nn.Linear,
)

# The residual add has a shape rule and is rejected later as a non-chain,
# with the node named.
SUPPORTED_FUNCTIONS: tuple[object, ...] = (operator.add, torch.add)


def _check_input_shape(input_shape: tuple[int, int, int]) -> None:
    if len(input_shape) != 3:
        raise ValueError(f"input shape must have rank 3 (N, C, L), got {input_shape}")
    if not all(isinstance(d, int) and not isinstance(d, bool) for d in input_shape):
        raise ValueError(f"input shape must contain ints, got {input_shape}")
    if input_shape[0] != 1:
        raise ValueError(f"batch dimension must be 1, got {input_shape[0]}")
    if any(d <= 0 for d in input_shape):
        raise ValueError(f"input shape must be positive, got {input_shape}")


def _check_supported(model: nn.Module) -> None:
    """Reject every op the converter cannot handle, before converting.

    Only the ops ``forward`` actually calls are checked, so a module that is
    defined but never called is ignored.
    """
    for node in fx.symbolic_trace(model).graph.nodes:
        if node.op == "call_method":
            raise UnsupportedLayerError(node.target, node.name)
        if node.op == "call_function" and node.target not in SUPPORTED_FUNCTIONS:
            name = getattr(node.target, "__name__", str(node.target))
            raise UnsupportedLayerError(name, node.name)
        if node.op == "call_module":
            module = model.get_submodule(node.target)
            if not isinstance(module, SUPPORTED_MODULES):
                raise UnsupportedLayerError(type(module).__name__, node.target)


def to_shaped_ir(
    model: nn.Module, input_shape: tuple[int, int, int]
) -> tuple[ShapedDatagraph, ir.Registry]:
    """Convert ``model`` to creator IR and annotate every edge with its shape."""
    _check_input_shape(input_shape)
    _check_supported(model)
    root, reg = get_default_converter()(model)
    (input_node,) = [n.name for n in root.nodes.values() if n.type == "input"]
    infer = get_default_shape_inference()
    return infer(root, reg, {input_node: tuple(input_shape)}), reg
