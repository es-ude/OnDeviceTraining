import json
import re
from collections import OrderedDict

import pytest
import torch
from torch import nn

from odt.planner import UnsupportedProblem, parse_problem, require_supported
from odt.providers import build_problem, problem_bytes
from tests.planner_oracle import residual_doc

SHAPE = (1, 2, 16)
F32 = {"kind": "FLOAT32"}


def _tiny() -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(2, 4, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool1d(2),
        nn.AvgPool1d(4),
        nn.Flatten(),
        nn.Linear(8, 3),
    )


def test_problem_v0_fields() -> None:
    p = build_problem(_tiny(), SHAPE, "CROSS_ENTROPY")

    assert p["schema_version"] == 0
    assert p["input"] == {"shape": [1, 2, 16], "dtype": F32}
    base = {
        "type",
        "frozen",
        "trackRunningStats",
        "outShape",
        "outputDtype",
        "propLossDtype",
    }
    for layer in p["layers"]:
        has_cost = layer["type"] in {"CONV1D", "LINEAR", "MAXPOOL1D", "AVGPOOL1D"}
        assert set(layer) == (base | {"cost"} if has_cost else base)
        assert layer["trackRunningStats"] is False
        assert isinstance(layer["outShape"], list)


def test_cross_entropy_appends_softmax() -> None:
    p = build_problem(_tiny(), SHAPE, "CROSS_ENTROPY")

    assert len(p["layers"]) == 7
    assert p["layers"][-1] == {
        "type": "SOFTMAX",
        "frozen": False,
        "trackRunningStats": False,
        "outShape": p["layers"][-2]["outShape"],
        "outputDtype": F32,
        "propLossDtype": F32,
    }


def test_mse_does_not_append_softmax() -> None:
    p = build_problem(_tiny(), SHAPE, "MSE")

    assert len(p["layers"]) == 6
    assert p["layers"][-1]["type"] == "LINEAR"


def test_rejects_unknown_loss() -> None:
    with pytest.raises(ValueError):
        build_problem(_tiny(), SHAPE, "NLL")


def test_problem_is_plain_json() -> None:
    p = build_problem(_tiny(), SHAPE, "CROSS_ENTROPY")

    assert json.loads(json.dumps(p)) == p


def test_reused_module_yields_two_layers() -> None:
    r = nn.ReLU()
    model = nn.Sequential(OrderedDict(c=nn.Conv1d(2, 2, 3, padding=1), a=r, b=r))

    p = build_problem(model, SHAPE, "MSE")

    assert [layer["type"] for layer in p["layers"]] == ["CONV1D", "RELU", "RELU"]


def test_softmax_does_not_alias_last_layer() -> None:
    p = build_problem(_tiny(), SHAPE, "CROSS_ENTROPY")
    softmax, last = p["layers"][-1], p["layers"][-2]

    softmax["outShape"].append(99)
    softmax["outputDtype"]["kind"] = "INT8"
    softmax["propLossDtype"]["kind"] = "INT8"

    assert last["outShape"] == [1, 3]
    assert last["outputDtype"] == F32
    assert last["propLossDtype"] == F32


class _ResidualHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv1d(2, 2, 3, padding=1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.flatten(self.relu(self.conv(x)) + x))


def test_a_residual_problem_carries_inputs_as_every_layers_last_key() -> None:
    p = build_problem(_ResidualHead(), (1, 2, 8), "CROSS_ENTROPY")
    # byte equality pins the key order the C dump must write too
    assert problem_bytes(p) == problem_bytes(residual_doc())


def test_a_residual_problem_loads_and_the_planner_refuses_it() -> None:
    problem = parse_problem(
        problem_bytes(build_problem(_ResidualHead(), (1, 2, 8), "CROSS_ENTROPY"))
    )
    assert [layer.inputs for layer in problem.layers] == [
        (0,),
        (1,),
        (2, 0),
        (3,),
        (4,),
        (5,),
    ]
    with pytest.raises(
        UnsupportedProblem,
        match=re.escape("layers[2].inputs: [2, 0] has no C reference run"),
    ):
        require_supported(problem, 1)
