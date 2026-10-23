"""Golden memory problems for the HAR and mnist_cnn twins.

The expected types, shapes and costs are the layer chains the C examples build
for the same models; the JSON fixtures pin the full problems.
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import pytest

from examples.har_classifier.train_pytorch import HarModel
from examples.mnist_cnn.train_pytorch import MnistCnn
from odt.providers import build_problem

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _har() -> dict:
    return build_problem(HarModel(), (1, 9, 128), "CROSS_ENTROPY")


def _mnist_cnn() -> dict:
    return build_problem(MnistCnn(), (1, 1, 784), "CROSS_ENTROPY")


def _costs(p: dict, layer_type: str, key: str) -> list[int]:
    return [l["cost"][key] for l in p["layers"] if l["type"] == layer_type]


def _pool_kernels(p: dict) -> list[int]:
    return [
        l["cost"]["kernel"]
        for l in p["layers"]
        if l["type"] in ("MAXPOOL1D", "AVGPOOL1D")
    ]


def test_har_problem_matches_c_layout() -> None:
    p = _har()

    assert [l["type"] for l in p["layers"]] == [
        "CONV1D", "RELU", "MAXPOOL1D",
        "CONV1D", "RELU", "MAXPOOL1D",
        "CONV1D", "RELU", "AVGPOOL1D",
        "FLATTEN", "LINEAR", "SOFTMAX",
    ]  # fmt: skip
    assert [l["outShape"] for l in p["layers"]] == [
        [1, 16, 128], [1, 16, 128], [1, 16, 64],
        [1, 32, 64], [1, 32, 64], [1, 32, 32],
        [1, 64, 32], [1, 64, 32], [1, 64, 1],
        [1, 64], [1, 6], [1, 6],
    ]  # fmt: skip
    assert _costs(p, "CONV1D", "macPerElement") == [63, 80, 96]
    assert _pool_kernels(p) == [2, 2, 32]
    assert _costs(p, "LINEAR", "macPerElement") == [64]


def test_mnist_cnn_problem_matches_c_layout() -> None:
    p = _mnist_cnn()

    assert [l["type"] for l in p["layers"]] == [
        "CONV1D", "RELU", "MAXPOOL1D",
        "CONV1D", "RELU", "MAXPOOL1D",
        "AVGPOOL1D", "FLATTEN", "LINEAR", "SOFTMAX",
    ]  # fmt: skip
    assert [l["outShape"] for l in p["layers"]] == [
        [1, 8, 784], [1, 8, 784], [1, 8, 392],
        [1, 16, 392], [1, 16, 392], [1, 16, 196],
        [1, 16, 1], [1, 16], [1, 10], [1, 10],
    ]  # fmt: skip
    assert _costs(p, "CONV1D", "macPerElement") == [3, 24]
    assert _pool_kernels(p) == [2, 2, 196]
    assert _costs(p, "LINEAR", "macPerElement") == [16]


@pytest.mark.parametrize(
    ("build", "fixture"),
    [(_har, "har_problem_v0.json"), (_mnist_cnn, "mnist_cnn_problem_v0.json")],
    ids=["har", "mnist_cnn"],
)
def test_problems_equal_fixtures(build, fixture: str) -> None:
    expected = json.loads((FIXTURES / fixture).read_text())

    assert build() == expected
