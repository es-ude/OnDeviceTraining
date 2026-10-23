"""Write a ``training`` metadata block onto every node of a shaped creator IR."""

from collections.abc import Callable
from dataclasses import dataclass, replace

import torch
from torch import nn

from elasticai.creator import ir
from elasticai.creator.experimental.ir.shape_inference import ShapedDatagraph

from .errors import NotAChainError, UnsupportedDtypeError, UnsupportedLayerError
from .profile import FLOAT32_UNIFORM, QuantProfile


@dataclass(frozen=True)
class Annotated:
    root: ShapedDatagraph
    registry: ir.Registry
    order: tuple[str, ...]  # annotated layer names in execution order, no input/output


@dataclass(frozen=True)
class _Acc:
    root: ShapedDatagraph  # annotated so far
    shaped: ShapedDatagraph  # the unannotated input, read for out-edge shapes
    registry: ir.Registry
    model: nn.Module
    profile: QuantProfile
    order: tuple[str, ...] = ()


# --- helpers ---------------------------------------------------------------


def _unwrap1(value: int | tuple[int]) -> int:
    """torch2ir stores some 1d sizes as 1-tuples and others as plain ints."""
    if isinstance(value, tuple):
        (value,) = value
    return int(value)


def _out_shape(acc: _Acc, node: ir.Node) -> tuple[int, ...]:
    (succ,) = acc.shaped.successors[node.name]
    return acc.shaped.edges[(node.name, succ)].shape


def _impl_attrs(acc: _Acc, node: ir.Node) -> ir.AttributeMapping:
    return acc.registry[node.attributes["implementation"]].attributes


def _frozen(acc: _Acc, node: ir.Node) -> bool:
    """True iff the module has parameters and none of them requires grad."""
    module = acc.model.get_submodule(node.attributes["implementation"])
    params = list(module.parameters())
    return bool(params) and not any(p.requires_grad for p in params)


def _write(acc: _Acc, node: ir.Node, block: dict[str, object]) -> _Acc:
    attrs = node.attributes | ir.attribute(training=block)
    return replace(acc, root=acc.root.add_node(node.name, attrs))


def _layer(
    acc: _Acc, node: ir.Node, odt_type: str, cost: dict[str, int] | None = None
) -> _Acc:
    out_shape = _out_shape(acc, node)
    if any(d <= 0 for d in out_shape):
        # e.g. a pool whose window is longer than its input
        raise ValueError(
            f"layer {node.name!r} has non-positive output shape {tuple(out_shape)}"
        )
    block: dict[str, object] = {
        "odt_type": odt_type,
        "frozen": _frozen(acc, node),
        "out_shape": out_shape,
        "output_q": acc.profile.output,
        "prop_loss_q": acc.profile.prop_loss,
    }
    if cost is not None:
        block["cost"] = cost
    acc = _write(acc, node, block)
    return replace(acc, order=acc.order + (node.name,))


def _pool_cost(acc: _Acc, node: ir.Node) -> dict[str, int]:
    return {"kernel": _unwrap1(_impl_attrs(acc, node)["kernel_size"])}


# --- handlers, one per IR node type ---------------------------------------


def _input(node: ir.Node, acc: _Acc) -> _Acc:
    block = {
        "odt_type": "INPUT",
        "frozen": False,
        "out_shape": _out_shape(acc, node),
        "output_q": acc.profile.input,
    }
    return _write(acc, node, block)


def _conv1d(node: ir.Node, acc: _Acc) -> _Acc:
    a = _impl_attrs(acc, node)
    mac = (a["in_channels"] // a["groups"]) * _unwrap1(a["kernel_size"])
    return _layer(acc, node, "CONV1D", {"macPerElement": mac})


def _relu(node: ir.Node, acc: _Acc) -> _Acc:
    return _layer(acc, node, "RELU")


def _maxpool1d(node: ir.Node, acc: _Acc) -> _Acc:
    return _layer(acc, node, "MAXPOOL1D", _pool_cost(acc, node))


def _avgpool1d(node: ir.Node, acc: _Acc) -> _Acc:
    return _layer(acc, node, "AVGPOOL1D", _pool_cost(acc, node))


def _flatten(node: ir.Node, acc: _Acc) -> _Acc:
    # the C layer always produces [batch, product of the remaining dims]
    a = _impl_attrs(acc, node)
    start, end = a["start_dim"], a["end_dim"]
    if (start, end) != (1, -1):
        raise UnsupportedLayerError(
            f"FLATTEN(start_dim={start}, end_dim={end})", node.name
        )
    return _layer(acc, node, "FLATTEN")


def _linear(node: ir.Node, acc: _Acc) -> _Acc:
    # shape inference keeps only (N, out_features) and drops middle dims
    (pred,) = acc.shaped.predecessors[node.name]
    rank = len(acc.shaped.edges[(pred, node.name)].shape)
    if rank != 2:
        raise UnsupportedLayerError(f"LINEAR on rank-{rank} input", node.name)
    in_features = _impl_attrs(acc, node)["in_features"]
    return _layer(acc, node, "LINEAR", {"macPerElement": in_features})


def _output(node: ir.Node, acc: _Acc) -> _Acc:
    """Write nothing; check that every layer was visited before the output.

    A part of the graph not reachable from the input (e.g. a disconnected
    cycle) passes the degree check but is never handed to a handler.
    """
    visited = set(acc.order)
    for n in acc.shaped.nodes.values():
        if n.type not in ("input", "output") and n.name not in visited:
            raise NotAChainError(n.name)
    return acc


_HANDLERS: dict[str, Callable[[ir.Node, _Acc], _Acc]] = {
    "input": _input,
    "conv1d": _conv1d,
    "relu": _relu,
    "maxpool1d": _maxpool1d,
    "avgpool1d": _avgpool1d,
    "flatten": _flatten,
    "linear": _linear,
    "output": _output,
}


# --- checks ------------------------------------------------------------------


def _check_float32(model: nn.Module) -> None:
    """Every parameter and every floating-point buffer must be float32.

    Integer buffers (e.g. a batch counter) are allowed.
    """
    for name, p in model.named_parameters():
        if p.dtype != torch.float32:
            raise UnsupportedDtypeError(
                f"parameter {name!r} has dtype {p.dtype}, expected torch.float32"
            )
    for name, b in model.named_buffers():
        if b.is_floating_point() and b.dtype != torch.float32:
            raise UnsupportedDtypeError(
                f"buffer {name!r} has dtype {b.dtype}, expected torch.float32"
            )


def _check_chain(root: ShapedDatagraph) -> None:
    """Exactly one input and one output, and a single path between them.

    The input is checked first, so a branch right after it is reported there.
    """
    nodes = list(root.nodes.values())
    inputs = [n.name for n in nodes if n.type == "input"]
    outputs = [n.name for n in nodes if n.type == "output"]
    if len(inputs) != 1:
        raise NotAChainError(inputs[1] if inputs else "<no input>")
    if len(outputs) != 1:
        raise NotAChainError(outputs[1] if outputs else "<no output>")

    def degree(name: str) -> tuple[int, int]:
        return len(root.predecessors[name]), len(root.successors[name])

    if degree(inputs[0]) != (0, 1):
        raise NotAChainError(inputs[0])
    for n in nodes:
        if n.type not in ("input", "output") and degree(n.name) != (1, 1):
            raise NotAChainError(n.name)
    if degree(outputs[0]) != (1, 0):
        raise NotAChainError(outputs[0])


# --- entry point -------------------------------------------------------------


def annotate(
    root: ShapedDatagraph,
    registry: ir.Registry,
    model: nn.Module,
    profile: QuantProfile = FLOAT32_UNIFORM,
) -> Annotated:
    """Annotate each node of a chain-shaped IR with its ``training`` block."""
    _check_float32(model)
    _check_chain(root)

    reducer = ir.ExecutionOrderGraphReducer()
    for node_type, handler in _HANDLERS.items():
        reducer.register(node_type, handler)

    start = _Acc(
        root=root, shaped=root, registry=registry, model=model, profile=profile
    )
    try:
        acc = reducer(root, start)
    except KeyError as err:
        # the dispatcher raises a bare KeyError for a type with no handler
        unknown = next(
            (n for n in root.nodes.values() if n.type not in _HANDLERS), None
        )
        if unknown is None:
            raise
        # a function node names its op (e.g. "add") in its implementation
        layer_type = (
            unknown.attributes.get("implementation", unknown.type)
            if unknown.type == "function"
            else unknown.type
        )
        raise UnsupportedLayerError(layer_type, unknown.name) from err
    if not acc.order:
        raise ValueError("model has no layers")
    return Annotated(root=acc.root, registry=registry, order=acc.order)
