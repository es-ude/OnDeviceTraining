import pytest
import torch
from torch import nn

from odt.providers.errors import UnsupportedLayerError
from odt.providers.trace import to_shaped_ir


def _tiny() -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(2, 4, 3, padding=1),
        nn.ReLU(),
        nn.AvgPool1d(4),
        nn.Flatten(),
        nn.Linear(16, 3),
    )


def test_to_shaped_ir_puts_shapes_on_edges() -> None:
    model = _tiny()
    root, _ = to_shaped_ir(model, (1, 2, 16))

    expected = [(1, 2, 16), (1, 4, 16), (1, 4, 16), (1, 4, 4), (1, 16), (1, 3)]

    x = torch.zeros(1, 2, 16)
    from_torch = [tuple(x.shape)]
    for layer in model:
        x = layer(x)
        from_torch.append(tuple(x.shape))
    assert from_torch == expected

    # edges are stored in chain order
    assert [tuple(e.shape) for e in root.edges.values()] == expected


@pytest.mark.parametrize(
    "shape", [(2, 2, 16), (1, 2), (1, 2, 16, 1), (1, 0, 16), (1, 2, -1)]
)
def test_to_shaped_ir_rejects_bad_input_shape(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError):
        to_shaped_ir(_tiny(), shape)  # type: ignore[arg-type]


def test_to_shaped_ir_rejects_unsupported_module() -> None:
    model = nn.Sequential(nn.Conv1d(2, 2, 3), nn.Softmax(dim=1))
    with pytest.raises(UnsupportedLayerError, match="Softmax"):
        to_shaped_ir(model, (1, 2, 16))


def test_to_shaped_ir_rejects_functional_call() -> None:
    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv1d(2, 2, 3)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.relu(self.conv(x))

    with pytest.raises(UnsupportedLayerError, match="relu"):
        to_shaped_ir(Net(), (1, 2, 16))


class _FlattenMethod(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv1d(2, 2, 3)
        self.fc = nn.Linear(28, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x).flatten(1))


def test_to_shaped_ir_rejects_method_call_before_conversion() -> None:
    with pytest.raises(UnsupportedLayerError, match="flatten") as info:
        to_shaped_ir(_FlattenMethod(), (1, 2, 16))
    assert info.value.layer_type == "flatten"


def test_to_shaped_ir_rejects_view_call() -> None:
    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv1d(2, 2, 3)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.conv(x).view(1, -1)

    with pytest.raises(UnsupportedLayerError, match="view"):
        to_shaped_ir(Net(), (1, 2, 16))


def test_to_shaped_ir_ignores_unused_module() -> None:
    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv1d(2, 2, 3)
            self.dropout = nn.Dropout()  # defined but never called

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.conv(x)

    root, _ = to_shaped_ir(Net(), (1, 2, 16))
    assert [tuple(e.shape) for e in root.edges.values()] == [(1, 2, 16), (1, 2, 14)]


def test_to_shaped_ir_rejects_unsupported_module_by_target() -> None:
    model = nn.Sequential(nn.Conv1d(2, 2, 3), nn.Softmax(dim=1))
    with pytest.raises(UnsupportedLayerError) as info:
        to_shaped_ir(model, (1, 2, 16))
    assert (info.value.layer_type, info.value.where) == ("Softmax", "1")
