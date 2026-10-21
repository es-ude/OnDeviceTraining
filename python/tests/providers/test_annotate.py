import pytest
import torch
from torch import nn

from elasticai.creator import ir
from odt.providers.annotate import Annotated, annotate
from odt.providers.errors import (
    NotAChainError,
    UnsupportedDtypeError,
    UnsupportedLayerError,
)
from odt.providers.trace import to_shaped_ir

SHAPE = (1, 2, 16)


def _tiny() -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(2, 4, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool1d(2),  # kernel stored as int
        nn.AvgPool1d(4),  # kernel stored as 1-tuple
        nn.Flatten(),
        nn.Linear(8, 3),
    )


def _run(model: nn.Module, shape: tuple[int, int, int] = SHAPE) -> Annotated:
    return annotate(*to_shaped_ir(model, shape), model)


def _block(result: Annotated, name: str):
    return result.root.nodes[name].attributes["training"]


def _input_name(result: Annotated) -> str:
    (name,) = [n.name for n in result.root.nodes.values() if n.type == "input"]
    return name


def test_annotate_writes_training_block_per_layer() -> None:
    result = _run(_tiny())
    blocks = [_block(result, name) for name in result.order]

    assert [b["odt_type"] for b in blocks] == [
        "CONV1D",
        "RELU",
        "MAXPOOL1D",
        "AVGPOOL1D",
        "FLATTEN",
        "LINEAR",
    ]

    for name, block in zip(result.order, blocks):
        (succ,) = result.root.successors[name]
        assert block["out_shape"] == result.root.edges[(name, succ)].shape
    assert [b["out_shape"] for b in blocks] == [
        (1, 4, 16),
        (1, 4, 16),
        (1, 4, 8),
        (1, 4, 2),
        (1, 8),
        (1, 3),
    ]

    conv, relu, maxpool, avgpool, flatten, linear = blocks
    assert dict(conv["cost"]) == {"macPerOutput": 2 * 3}
    assert dict(linear["cost"]) == {"macPerOutput": 8}
    assert dict(maxpool["cost"]) == {"kernel": 2}
    assert dict(avgpool["cost"]) == {"kernel": 4}
    assert type(maxpool["cost"]["kernel"]) is int
    assert type(avgpool["cost"]["kernel"]) is int
    assert type(conv["cost"]["macPerOutput"]) is int
    assert "cost" not in relu
    assert "cost" not in flatten

    for block in blocks:
        assert dict(block["output_q"]) == {"kind": "FLOAT32"}
        assert dict(block["prop_loss_q"]) == {"kind": "FLOAT32"}


def test_annotate_input_node_carries_input_dtype() -> None:
    result = _run(_tiny())
    block = _block(result, _input_name(result))

    assert block["odt_type"] == "INPUT"
    assert dict(block["output_q"]) == {"kind": "FLOAT32"}
    assert "prop_loss_q" not in block
    assert block["out_shape"] == SHAPE


def test_frozen_only_when_all_params_frozen() -> None:
    partly = _tiny()
    partly[0].weight.requires_grad_(False)  # bias stays trainable
    result = _run(partly)
    conv, relu = result.order[0], result.order[1]
    assert _block(result, conv)["frozen"] is False
    assert _block(result, relu)["frozen"] is False

    fully = _tiny()
    fully[0].requires_grad_(False)
    assert _block(_run(fully), conv)["frozen"] is True


class _Residual(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv1d(2, 2, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv(x)


def test_rejects_non_chain() -> None:
    model = _Residual()
    root, registry = to_shaped_ir(model, (1, 2, 8))
    (input_name,) = [n.name for n in root.nodes.values() if n.type == "input"]

    with pytest.raises(NotAChainError, match=repr(input_name)) as info:
        annotate(root, registry, model)
    assert info.value.node == input_name


def test_rejects_nodes_unreachable_from_input() -> None:
    model = _tiny()
    root, registry = to_shaped_ir(model, SHAPE)
    # a disconnected 2-node cycle: every node still has one pred and one succ
    relu = ir.attribute(type="relu", implementation="1")
    root = root.add_nodes(("loop_a", relu), ("loop_b", relu))
    root = root.add_edges(
        ("loop_a", "loop_b", ir.attribute(shape=(1, 4, 16))),
        ("loop_b", "loop_a", ir.attribute(shape=(1, 4, 16))),
    )

    with pytest.raises(NotAChainError) as info:
        annotate(root, registry, model)
    assert info.value.node == "loop_a"


def test_rejects_non_float32_params() -> None:
    with pytest.raises(UnsupportedDtypeError):
        _run(_tiny().double())


def test_unregistered_node_type_raises_unsupported_layer() -> None:
    model = _tiny()
    root, registry = to_shaped_ir(model, SHAPE)
    # a chain node whose type has no handler, as if it had slipped past trace
    relu = next(n for n in root.nodes.values() if n.type == "relu")
    root = root.add_node(relu.name, relu.attributes | ir.attribute(type="softmax"))

    with pytest.raises(UnsupportedLayerError, match="softmax") as info:
        annotate(root, registry, model)
    assert info.value.where == relu.name


def test_annotation_is_serializable() -> None:
    result = _run(_tiny())
    ir.IrSerializer().serialize(result.root)


def test_rejects_non_float32_integer_param() -> None:
    model = _tiny()
    model[0].register_parameter(
        "scale",
        nn.Parameter(torch.ones(1, dtype=torch.int32), requires_grad=False),
    )
    with pytest.raises(UnsupportedDtypeError, match=r"'0\.scale'"):
        _run(model)


def test_rejects_float64_buffer() -> None:
    model = _tiny()
    model[0].register_buffer("stat", torch.zeros(4, dtype=torch.float64))
    with pytest.raises(UnsupportedDtypeError, match=r"'0\.stat'"):
        _run(model)


def test_accepts_integer_buffer() -> None:
    model = _tiny()
    model[0].register_buffer("count", torch.zeros(1, dtype=torch.int64))
    assert len(_run(model).order) == 6


class _AddConst(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv1d(2, 2, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) + 1


def test_function_node_reported_by_implementation() -> None:
    with pytest.raises(UnsupportedLayerError) as info:
        _run(_AddConst())
    assert info.value.layer_type == "add"


def test_rejects_non_positive_out_shape() -> None:
    model = nn.Sequential(nn.Conv1d(2, 2, 1), nn.MaxPool1d(2))
    root, registry = to_shaped_ir(model, (1, 2, 1))
    (pool,) = [n.name for n in root.nodes.values() if n.type == "maxpool1d"]

    with pytest.raises(ValueError, match=repr(pool)):
        annotate(root, registry, model)


def test_rejects_linear_on_rank3_input() -> None:
    model = nn.Sequential(nn.Conv1d(2, 4, 3, padding=1), nn.Linear(16, 3))
    root, registry = to_shaped_ir(model, SHAPE)
    (linear,) = [n.name for n in root.nodes.values() if n.type == "linear"]

    with pytest.raises(UnsupportedLayerError) as info:
        annotate(root, registry, model)
    assert info.value.layer_type == "LINEAR on rank-3 input"
    assert info.value.where == linear


@pytest.mark.parametrize(
    "flatten, label",
    [
        (nn.Flatten(start_dim=0), "FLATTEN(start_dim=0, end_dim=-1)"),
        (nn.Flatten(start_dim=1, end_dim=1), "FLATTEN(start_dim=1, end_dim=1)"),
    ],
)
def test_rejects_non_default_flatten(flatten: nn.Flatten, label: str) -> None:
    model = nn.Sequential(nn.Conv1d(2, 4, 3, padding=1), flatten)
    root, registry = to_shaped_ir(model, SHAPE)
    (name,) = [n.name for n in root.nodes.values() if n.type == "flatten"]

    with pytest.raises(UnsupportedLayerError) as info:
        annotate(root, registry, model)
    assert (info.value.layer_type, info.value.where) == (label, name)


def test_accepts_default_flatten() -> None:
    model = nn.Sequential(nn.Conv1d(2, 4, 3, padding=1), nn.Flatten())
    result = _run(model)
    assert _block(result, result.order[-1])["out_shape"] == (1, 64)


def test_rejects_empty_model() -> None:
    with pytest.raises(ValueError, match="model has no layers"):
        _run(nn.Sequential())
