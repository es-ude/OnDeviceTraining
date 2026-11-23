"""Trace a PyTorch model into a shaped creator IR at batch size 1."""

import operator
from collections import Counter

import torch
from torch import fx, nn

from elasticai.creator import ir
from elasticai.creator.experimental.ir.shape_inference import (
    ShapedDatagraph,
    get_default_shape_inference,
)
from elasticai.creator.torch2ir import get_default_converter

from .errors import UnsupportedGraphError, UnsupportedLayerError

SUPPORTED_MODULES: tuple[type[nn.Module], ...] = (
    nn.Conv1d,
    nn.ReLU,
    nn.MaxPool1d,
    nn.AvgPool1d,
    nn.Flatten,
    nn.Linear,
)

# operator.add and torch.add become the ADD merge node; annotate refuses a
# constant operand or keyword arguments.
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
    graph = fx.symbolic_trace(model).graph
    for node in graph.nodes:
        if node.op == "call_method":
            raise UnsupportedLayerError(node.target, node.name)
        if node.op == "call_function" and node.target not in SUPPORTED_FUNCTIONS:
            name = getattr(node.target, "__name__", str(node.target))
            raise UnsupportedLayerError(name, node.name)
        if node.op == "call_module":
            module = model.get_submodule(node.target)
            if not isinstance(module, SUPPORTED_MODULES):
                raise UnsupportedLayerError(type(module).__name__, node.target)
            if node.kwargs:
                raise UnsupportedLayerError(
                    type(module).__name__,
                    node.target,
                    "modules must be called with positional tensor arguments",
                )
    _check_no_shared_state(model, graph)
    if not _is_chain(graph):
        _check_no_inplace_relu(model, graph)


def _check_no_shared_state(model: nn.Module, graph: fx.Graph) -> None:
    """Each C node owns its parameters and buffers (frozen ones too), so one tensor
    may reach only one call site: a module holding any is called once, and two
    modules never hold the same tensor (tied weights). A stateless module becomes
    one node per call."""
    calls = Counter(
        id(model.get_submodule(n.target)) for n in graph.nodes if n.op == "call_module"
    )
    owner: dict[int, str] = {}
    for node in graph.nodes:
        if node.op != "call_module":
            continue
        module = model.get_submodule(node.target)
        tensors = {id(t) for t in [*module.parameters(), *module.buffers()]}
        if tensors and calls[id(module)] > 1:
            raise UnsupportedGraphError(
                node.target,
                f"module {node.target!r} is called {calls[id(module)]} times; a module "
                "with parameters must be called once (weight sharing is not supported)",
            )
        for tensor in tensors:
            first = owner.setdefault(tensor, node.target)
            if first != node.target:
                raise UnsupportedGraphError(
                    node.target,
                    f"module {node.target!r} shares a parameter or buffer with {first!r} "
                    "(weight sharing is not supported)",
                )


def _is_chain(graph: fx.Graph) -> bool:
    """problem.py's chain test on the fx program: every call reads exactly the value
    produced just before it."""
    previous = None
    for node in graph.nodes:
        if node.op == "placeholder":
            previous = node
        elif node.op in ("call_module", "call_function"):
            if node.args != (previous,):
                return False
            previous = node
    return True


def _check_no_inplace_relu(model: nn.Module, graph: fx.Graph) -> None:
    """Off a chain an in-place ReLU can rewrite a value another layer still reads,
    directly or through a view (a Flatten output aliases its input), which the C
    model's separate buffers would not reproduce."""
    for node in graph.nodes:
        if node.op != "call_module":
            continue
        module = model.get_submodule(node.target)
        if isinstance(module, nn.ReLU) and module.inplace:
            raise UnsupportedGraphError(
                node.target,
                "in-place ReLU in a branching model is not supported (it would "
                "mutate a value another layer reads)",
            )


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
