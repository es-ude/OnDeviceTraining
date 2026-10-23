"""Golden memory problems for the HAR and mnist_cnn twins.

The expected types, shapes and costs are the layer chains the C examples build
for the same models. The full problems are pinned against the C trainers' own
DUMP_PROBLEM=1 output (python/tests/planner_goldens): the IR adapter and the C
dump must agree byte for byte, so a change on either side fails here.
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import pytest

from examples.har_classifier.train_pytorch import HarModel
from examples.mnist_cnn.train_pytorch import MnistCnn
from odt.planner import problem_hash
from odt.providers import build_problem

C_DUMPS = REPO_ROOT / "python/tests/planner_goldens"


def _har() -> dict:
    return build_problem(HarModel(), (1, 9, 128), "CROSS_ENTROPY")


def _har_finetune() -> dict:
    # the C finetune trainer freezes the three convs and trains the head
    model = HarModel()
    for conv in (model.conv1, model.conv2, model.conv3):
        conv.requires_grad_(False)
    return build_problem(model, (1, 9, 128), "CROSS_ENTROPY")


def _mnist_cnn() -> dict:
    return build_problem(MnistCnn(), (1, 1, 784), "CROSS_ENTROPY")


def _as_c_dump(problem: dict) -> bytes:
    """Serialize as C does: one compact line, keys in emission order, then a newline."""
    return (json.dumps(problem, separators=(",", ":")) + "\n").encode()


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
    ("build", "model"),
    [(_har, "har"), (_har_finetune, "har_finetune"), (_mnist_cnn, "mnist_cnn")],
    ids=["har", "har_finetune", "mnist_cnn"],
)
def test_problem_bytes_equal_c_dump(build, model: str) -> None:
    ir_bytes = _as_c_dump(build())
    c_bytes = (C_DUMPS / f"{model}.problem.json").read_bytes()
    c_hash_line = (C_DUMPS / f"{model}.problem.hash").read_text()

    # parsed first, so a mismatch shows the differing field, not two byte strings
    assert json.loads(ir_bytes) == json.loads(c_bytes)
    assert ir_bytes == c_bytes
    assert f"PROBLEM_HASH fnv1a64={problem_hash(ir_bytes)}\n" == c_hash_line
