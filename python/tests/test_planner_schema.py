"""The planner's problem schema v0: the loader, its refusals and the problem hash."""
import json
import re
from pathlib import Path

import pytest

from odt.planner import (Dtype, Problem, ProblemError, load_problem, parse_problem, problem_bytes,
                         problem_hash)

GOLDENS = Path(__file__).resolve().parent / "planner_goldens"
MODELS = ("har", "har_finetune", "mnist_cnn")


def golden_hash_line(model: str) -> str:
    return (GOLDENS / f"{model}.problem.hash").read_text()


@pytest.mark.parametrize("model", MODELS)
def test_every_c_dump_loads_and_hashes_to_its_c_line(model):
    path = GOLDENS / f"{model}.problem.json"
    problem = load_problem(path)
    assert isinstance(problem, Problem)
    assert golden_hash_line(model) == f"PROBLEM_HASH fnv1a64={problem.hash}\n"
    assert problem.hash == problem_hash(path.read_bytes())


def test_the_har_dump_reads_as_its_model():
    p = load_problem(GOLDENS / "har.problem.json")
    assert p.schema_version == 0
    assert p.input.shape == (1, 9, 128)
    assert p.input.dtype == Dtype("FLOAT32")
    assert p.loss == "CROSS_ENTROPY"
    assert [layer.type for layer in p.layers] == [
        "CONV1D", "RELU", "MAXPOOL1D", "CONV1D", "RELU", "MAXPOOL1D",
        "CONV1D", "RELU", "AVGPOOL1D", "FLATTEN", "LINEAR", "SOFTMAX",
    ]
    assert [layer.mac_per_element for layer in p.layers] == [
        63, None, None, 80, None, None, 96, None, None, None, 64, None,
    ]
    assert [layer.kernel for layer in p.layers] == [
        None, None, 2, None, None, 2, None, None, 32, None, None, None,
    ]
    assert p.layers[9].out_shape == (1, 64)
    assert not any(layer.frozen or layer.track_running_stats for layer in p.layers)


def test_the_finetune_dump_marks_exactly_its_three_convs_frozen():
    p = load_problem(GOLDENS / "har_finetune.problem.json")
    assert [j for j, layer in enumerate(p.layers) if layer.frozen] == [0, 3, 6]


def test_the_problem_hash_is_fnv1a_64():
    # Published FNV-1a 64 vectors: the empty string is the offset basis; "a" and "foobar".
    assert problem_hash(b"") == "cbf29ce484222325"
    assert problem_hash(b"a") == "af63dc4c8601ec8c"
    assert problem_hash(b"foobar") == "85944171f73967e8"


def test_problem_bytes_is_one_compact_line_in_key_order():
    # odt.providers imports torch, so only this test pays for it
    import odt.planner
    import odt.providers

    assert problem_bytes({"b": 1, "a": [1, 2]}) == b'{"b":1,"a":[1,2]}\n'
    assert odt.providers.problem_bytes is odt.planner.problem_bytes


def base() -> dict:
    return json.loads((GOLDENS / "har.problem.json").read_bytes())


def with_change(path, value):
    data = base()
    node = data
    for key in path[:-1]:
        node = node[key]
    if value is DELETE:
        del node[path[-1]]
    else:
        node[path[-1]] = value
    return json.dumps(data).encode()


DELETE = object()
BAD = [
    ("not json", b"{", "problem: not a JSON document"),
    ("not an object", b"[]", "problem: expected an object"),
    ("version 1", with_change(("schema_version",), 1),
     "schema_version: 1 is not a schema version this planner reads (expected 0)"),
    ("version true", with_change(("schema_version",), True),
     "schema_version: expected an integer"),
    ("unknown key", with_change(("budget",), 4), "problem: unknown key 'budget'"),
    ("missing loss", with_change(("loss",), DELETE), "problem: missing key 'loss'"),
    ("loss", with_change(("loss",), "NLL"),
     "loss: 'NLL' is not one of CROSS_ENTROPY, MSE"),
    ("batch 2", with_change(("input", "shape"), [2, 9, 128]),
     "input.shape: the first dimension is the batch and must be 1, got 2"),
    ("empty shape", with_change(("input", "shape"), []),
     "input.shape: expected a non-empty list of positive integers"),
    ("zero dim", with_change(("layers", 0, "outShape"), [1, 0, 128]),
     "layers[0].outShape: expected a non-empty list of positive integers"),
    ("kind", with_change(("input", "dtype"), {"kind": "FLOAT16"}),
     "input.dtype.kind: 'FLOAT16' is not one of INT32, FLOAT32, SYM_INT32, SYM, ASYM, BOOL, BFP"),
    ("float extra", with_change(("layers", 2, "outputDtype"), {"kind": "FLOAT32", "qBits": 8}),
     "layers[2].outputDtype: FLOAT32 takes no field 'qBits'"),
    ("sym without bits", with_change(("layers", 2, "outputDtype"), {"kind": "SYM"}),
     "layers[2].outputDtype: SYM needs the integer field 'qBits'"),
    ("bfp without group", with_change(
        ("layers", 2, "propLossDtype"), {"kind": "BFP", "mantissaBits": 8, "exponentBits": 8}),
     "layers[2].propLossDtype: BFP needs the integer field 'groupSize'"),
    ("type", with_change(("layers", 1, "type"), "GELU"),
     "layers[1].type: 'GELU' is not a layer type this schema knows"),
    ("frozen", with_change(("layers", 1, "frozen"), 0), "layers[1].frozen: expected true or false"),
    ("stats missing", with_change(("layers", 1, "trackRunningStats"), DELETE),
     "layers[1]: missing key 'trackRunningStats'"),
    ("conv cost missing", with_change(("layers", 0, "cost"), DELETE),
     'layers[0].cost: CONV1D needs {"macPerElement": <positive integer>}'),
    ("pool cost key", with_change(("layers", 2, "cost"), {"macPerElement": 2}),
     'layers[2].cost: MAXPOOL1D needs {"kernel": <positive integer>}'),
    ("relu cost", with_change(("layers", 1, "cost"), {"kernel": 2}),
     "layers[1].cost: RELU takes no cost operand"),
    ("no layers", with_change(("layers",), []), "layers: expected a non-empty list"),
    ("sym zero bits", with_change(("layers", 2, "outputDtype"), {"kind": "SYM", "qBits": 0}),
     "layers[2].outputDtype: SYM qBits must be at least 1, got 0"),
    ("asym 17 bits", with_change(("layers", 2, "outputDtype"), {"kind": "ASYM", "qBits": 17}),
     "layers[2].outputDtype: ASYM qBits 17 is outside C's range 1 to 16"),
    ("asym 0 bits", with_change(("layers", 2, "outputDtype"), {"kind": "ASYM", "qBits": 0}),
     "layers[2].outputDtype: ASYM qBits 0 is outside C's range 1 to 16"),
    ("bfp mantissa 1", with_change(
        ("layers", 2, "propLossDtype"),
        {"kind": "BFP", "mantissaBits": 1, "exponentBits": 8, "groupSize": 0}),
     "layers[2].propLossDtype: BFP mantissaBits 1 is outside C's range 2 to 16"),
    ("bfp exponent 9", with_change(
        ("layers", 2, "propLossDtype"),
        {"kind": "BFP", "mantissaBits": 8, "exponentBits": 9, "groupSize": 0}),
     "layers[2].propLossDtype: BFP exponentBits 9 is outside C's range 2 to 8"),
    ("relu null cost", with_change(("layers", 1, "cost"), None),
     "layers[1].cost: RELU takes no cost operand"),
    ("dtype not object", with_change(("input", "dtype"), "FLOAT32"),
     "input.dtype: expected an object"),
    ("stats not bool", with_change(("layers", 1, "trackRunningStats"), 1),
     "layers[1].trackRunningStats: expected true or false"),
    ("bool cost", with_change(("layers", 0, "cost"), {"macPerElement": True}),
     'layers[0].cost: CONV1D needs {"macPerElement": <positive integer>}'),
    ("layers not list", with_change(("layers",), {"0": {}}), "layers: expected a non-empty list"),
]


@pytest.mark.parametrize("data, message", [(d, m) for _, d, m in BAD], ids=[i for i, _, _ in BAD])
def test_a_malformed_problem_is_refused_naming_its_field(data, message):
    with pytest.raises(ProblemError, match=f"^{re.escape(message)}$"):
        parse_problem(data)


def test_the_schema_holds_more_than_the_planner_accepts():
    data = base()
    data["loss"] = "MSE"
    data["input"]["dtype"] = {"kind": "BFP", "mantissaBits": 8, "exponentBits": 8, "groupSize": 16}
    data["layers"][1] = dict(data["layers"][1], type="BATCHNORM1D", trackRunningStats=True)
    data["layers"][2]["outputDtype"] = {"kind": "SYM", "qBits": 4}
    p = parse_problem(json.dumps(data).encode())
    assert p.loss == "MSE"
    assert p.input.dtype == Dtype("BFP", (("exponentBits", 8), ("groupSize", 16),
                                          ("mantissaBits", 8)))
    assert p.layers[1].track_running_stats
    assert p.layers[2].output_dtype.params == (("qBits", 4),)


def test_bfp_per_tensor_group_size_0_is_valid_schema():
    data = base()
    data["layers"][2]["propLossDtype"] = {"kind": "BFP", "mantissaBits": 8, "exponentBits": 8,
                                          "groupSize": 0}
    p = parse_problem(json.dumps(data).encode())
    assert p.layers[2].prop_loss_dtype.params == (("exponentBits", 8), ("groupSize", 0),
                                                  ("mantissaBits", 8))
