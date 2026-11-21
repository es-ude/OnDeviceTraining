"""Write a ``training`` metadata block onto every node of a shaped creator IR."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace

import torch
from torch import fx, nn

from elasticai.creator import ir
from elasticai.creator.experimental.ir.shape_inference import ShapedDatagraph

from .errors import UnsupportedDtypeError, UnsupportedGraphError, UnsupportedLayerError
from .profile import FLOAT32_UNIFORM, QuantProfile


@dataclass(frozen=True)
class Annotated:
    root: ShapedDatagraph
    registry: ir.Registry
    # annotated layer names in C node order (fx program order), no input/output
    order: tuple[str, ...]


@dataclass(frozen=True)
class _Acc:
    root: ShapedDatagraph  # annotated so far
    shaped: ShapedDatagraph  # the unannotated input, read for out-edge shapes
    registry: ir.Registry
    model: nn.Module
    profile: QuantProfile
    fx: Mapping[str, fx.Node]
    buffer: Mapping[str, int]  # value name -> C buffer id (0 = input, j + 1 = layer j)
    order: tuple[str, ...] = ()  # visit order of the reducer, not C node order


# --- helpers ---------------------------------------------------------------


def _unwrap1(value: int | tuple[int]) -> int:
    """torch2ir stores some 1d sizes as 1-tuples and others as plain ints."""
    if isinstance(value, tuple):
        (value,) = value
    return int(value)


def _fx_program(model: nn.Module) -> tuple[Mapping[str, fx.Node], Mapping[str, int]]:
    """fx program order is the C node order; IR node names are fx node names."""
    graph = fx.symbolic_trace(model).graph
    (placeholder,) = [n for n in graph.nodes if n.op == "placeholder"]
    layers = (n for n in graph.nodes if n.op in ("call_module", "call_function"))
    buffer = {placeholder.name: 0} | {n.name: j + 1 for j, n in enumerate(layers)}
    return {n.name: n for n in graph.nodes}, buffer


def _out_shape(acc: _Acc, node: ir.Node) -> tuple[int, ...]:
    """Every out-edge of a value carries its shape; a fanned-out value has several."""
    succ = next(iter(acc.shaped.successors[node.name]))
    return acc.shaped.edges[(node.name, succ)].shape


def _impl_attrs(acc: _Acc, node: ir.Node) -> ir.AttributeMapping:
    return acc.registry[node.attributes["implementation"]].attributes


def _frozen(acc: _Acc, node: ir.Node) -> bool:
    """True iff the module has parameters and none of them requires grad."""
    if node.type == "function":
        return False  # an ADD has no module and no parameters
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
    block["inputs"] = tuple(acc.buffer[a.name] for a in acc.fx[node.name].args)
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


def _add(node: ir.Node, acc: _Acc) -> _Acc:
    """Trace admits only operator.add and torch.add as calls, so every function
    node is an ADD; operand order is the fx argument order (the IR keeps one
    edge per producer, so ``y + y`` and the order of operands are lost there)."""
    fx_node = acc.fx[node.name]
    if (
        fx_node.kwargs
        or len(fx_node.args) != 2
        or not all(isinstance(a, fx.Node) for a in fx_node.args)
    ):
        raise UnsupportedLayerError("add", node.name)
    # the C ADD does not broadcast
    shapes = [acc.shaped.edges[(a.name, node.name)].shape for a in fx_node.args]
    if shapes[0] != shapes[1]:
        raise UnsupportedLayerError(
            "add", node.name, f"operand shapes {shapes[0]} and {shapes[1]} differ"
        )
    return _layer(acc, node, "ADD")


def _output(node: ir.Node, acc: _Acc) -> _Acc:
    # The reachability check runs after the reducer: in a DAG the reducer can
    # reach the output before a layer on a longer branch.
    return acc


_HANDLERS: dict[str, Callable[[ir.Node, _Acc], _Acc]] = {
    "input": _input,
    "conv1d": _conv1d,
    "relu": _relu,
    "maxpool1d": _maxpool1d,
    "avgpool1d": _avgpool1d,
    "flatten": _flatten,
    "linear": _linear,
    "function": _add,
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


def _check_graph(root: ShapedDatagraph) -> None:
    """One input, one output fed by one tensor, and every value read."""
    nodes = list(root.nodes.values())
    inputs = [n.name for n in nodes if n.type == "input"]
    outputs = [n.name for n in nodes if n.type == "output"]
    if len(inputs) != 1:
        raise UnsupportedGraphError(
            inputs[1] if inputs else "<no input>",
            "the model must have exactly one input",
        )
    if len(outputs) != 1:
        raise UnsupportedGraphError(
            outputs[1] if outputs else "<no output>",
            "the model must have exactly one output",
        )
    if len(root.predecessors[outputs[0]]) != 1:
        raise UnsupportedGraphError(
            outputs[0], "the model must return exactly one tensor"
        )
    for n in nodes:
        if n.type != "output" and not root.successors[n.name]:
            raise UnsupportedGraphError(n.name, "its output is never read")


def _check_reached(acc: _Acc) -> None:
    """A part of the graph not reachable from the input (e.g. a disconnected
    cycle) passes the graph check but is never handed to a handler."""
    visited = set(acc.order)
    for n in acc.shaped.nodes.values():
        if n.type not in ("input", "output") and n.name not in visited:
            raise UnsupportedGraphError(n.name, "not reachable from the input")


# --- entry point -------------------------------------------------------------


def annotate(
    root: ShapedDatagraph,
    registry: ir.Registry,
    model: nn.Module,
    profile: QuantProfile = FLOAT32_UNIFORM,
) -> Annotated:
    """Annotate each node of a DAG-shaped IR with its ``training`` block.

    Handlers never read another node's block: the reducer may visit a merge
    before its producers.
    """
    _check_float32(model)
    _check_graph(root)
    fx_nodes, buffer = _fx_program(model)

    reducer = ir.ExecutionOrderGraphReducer()
    for node_type, handler in _HANDLERS.items():
        reducer.register(node_type, handler)

    start = _Acc(
        root=root,
        shaped=root,
        registry=registry,
        model=model,
        profile=profile,
        fx=fx_nodes,
        buffer=buffer,
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
        raise UnsupportedLayerError(unknown.type, unknown.name) from err
    _check_reached(acc)
    if not acc.order:
        raise ValueError("model has no layers")
    order = tuple(sorted(acc.order, key=acc.buffer.__getitem__))
    return Annotated(root=acc.root, registry=registry, order=order)
