"""Problem schema v0: the model data a planner solves for, as the C harness writes it.

One JSON object (C's DUMP_PROBLEM=1 writes it as one line):

    {"schema_version": 0,
     "input": {"shape": [1, ...], "dtype": {"kind": ...}},
     "loss": "CROSS_ENTROPY" | "MSE",
     "layers": [{"type", "frozen", "trackRunningStats", "outShape", "outputDtype",
                 "propLossDtype", "cost"?}, ...]}

Shapes carry the leading batch dimension, which is 1: the micro-batch is a planner input.
Dtype widths are checked against C's ranges (WIDTH_RANGE), not just for being integers.
`layers` is positional, one entry per C layer (Softmax included under CROSS_ENTROPY).
`cost` holds the operand C's forward-cost count needs beyond element counts: CONV1D and
LINEAR `{"macPerElement"}` per output element, CONV1D_TRANSPOSED per input element,
MAXPOOL1D and AVGPOOL1D `{"kernel"}`; every other type has none. The schema is wider than
what the planner accepts (see accept.py), so widening the planner needs no schema change.
"""
import functools
import json
from dataclasses import dataclass
from pathlib import Path

SCHEMA_VERSION = 0
KINDS = ("INT32", "FLOAT32", "SYM_INT32", "SYM", "ASYM", "BOOL", "BFP")
KIND_FIELDS = {"SYM": ("qBits",), "ASYM": ("qBits",),
               "BFP": ("exponentBits", "groupSize", "mantissaBits")}
# C validates these (validateAsymQBits; the BFP init); SYM qBits has no C validator, so only 0
# (the sub-byte packer underflows) is refused.
WIDTH_RANGE = {("ASYM", "qBits"): (1, 16), ("BFP", "mantissaBits"): (2, 16),
               ("BFP", "exponentBits"): (2, 8)}
LOSSES = ("CROSS_ENTROPY", "MSE")
LAYER_TYPES = ("LINEAR", "RELU", "CONV1D", "CONV1D_TRANSPOSED", "MAXPOOL1D", "AVGPOOL1D",
               "SOFTMAX", "FLATTEN", "QUANTIZATION", "ADAPTIVE_AVGPOOL1D", "DROPOUT",
               "LAYERNORM", "GROUPNORM", "BATCHNORM1D")
COST_KEY = {"LINEAR": "macPerElement", "CONV1D": "macPerElement",
            "CONV1D_TRANSPOSED": "macPerElement", "MAXPOOL1D": "kernel", "AVGPOOL1D": "kernel"}
TOP_KEYS = ("schema_version", "input", "loss", "layers")
INPUT_KEYS = ("shape", "dtype")
LAYER_KEYS = ("type", "frozen", "trackRunningStats", "outShape", "outputDtype",
              "propLossDtype")
FNV_BASIS = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3


class ProblemError(ValueError):
    """The problem file is not schema v0; the message names the field."""


@dataclass(frozen=True)
class Dtype:
    kind: str
    params: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True)
class InputSpec:
    shape: tuple[int, ...]
    dtype: Dtype


@dataclass(frozen=True)
class LayerSpec:
    type: str
    frozen: bool
    track_running_stats: bool
    out_shape: tuple[int, ...]
    output_dtype: Dtype
    prop_loss_dtype: Dtype
    mac_per_element: int | None = None
    kernel: int | None = None


@dataclass(frozen=True)
class Problem:
    schema_version: int
    input: InputSpec
    loss: str
    layers: tuple[LayerSpec, ...]
    hash: str


def problem_bytes(problem: dict) -> bytes:
    """The bytes C's DUMP_PROBLEM=1 writes for the same problem: one compact line, keys in
    the dict's order, then a newline. Write a problem file with these, so its hash is C's."""
    return (json.dumps(problem, separators=(",", ":")) + "\n").encode()


def problem_hash(data: bytes) -> str:
    """FNV-1a 64 over the exact bytes, as 16 lowercase hex digits (C prints the same)."""
    return format(functools.reduce(lambda h, b: ((h ^ b) * FNV_PRIME) % 2**64, data, FNV_BASIS),
                  "016x")


def load_problem(path: str | Path) -> Problem:
    return parse_problem(Path(path).read_bytes())


def parse_problem(data: bytes) -> Problem:
    try:
        doc = json.loads(data)
    except (ValueError, UnicodeDecodeError) as e:
        raise ProblemError("problem: not a JSON document") from e
    _keys(doc, "problem", TOP_KEYS, ())
    version = doc["schema_version"]
    if not is_int(version):
        raise ProblemError("schema_version: expected an integer")
    if version != SCHEMA_VERSION:
        raise ProblemError(f"schema_version: {version} is not a schema version this planner "
                           f"reads (expected {SCHEMA_VERSION})")
    _keys(doc["input"], "input", INPUT_KEYS, ())
    spec = InputSpec(_shape(doc["input"]["shape"], "input.shape"),
                     _dtype(doc["input"]["dtype"], "input.dtype"))
    loss = doc["loss"]
    if loss not in LOSSES:
        raise ProblemError(f"loss: {loss!r} is not one of {', '.join(LOSSES)}")
    layers = doc["layers"]
    if not isinstance(layers, list) or not layers:
        raise ProblemError("layers: expected a non-empty list")
    return Problem(version, spec, loss, tuple(_layer(x, f"layers[{j}]")
                                              for j, x in enumerate(layers)), problem_hash(data))


def is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _keys(node, where: str, required: tuple[str, ...], optional: tuple[str, ...]) -> None:
    if not isinstance(node, dict):
        raise ProblemError(f"{where}: expected an object")
    for key in node:
        if key not in required and key not in optional:
            raise ProblemError(f"{where}: unknown key {key!r}")
    for key in required:
        if key not in node:
            raise ProblemError(f"{where}: missing key {key!r}")


def _shape(v, where: str) -> tuple[int, ...]:
    if not isinstance(v, list) or not v or not all(is_int(d) and d >= 1 for d in v):
        raise ProblemError(f"{where}: expected a non-empty list of positive integers")
    if v[0] != 1:
        raise ProblemError(f"{where}: the first dimension is the batch and must be 1, got {v[0]}")
    return tuple(v)


def _dtype(v, where: str) -> Dtype:
    if not isinstance(v, dict):
        raise ProblemError(f"{where}: expected an object")
    kind = v.get("kind")
    if kind not in KINDS:
        raise ProblemError(f"{where}.kind: {kind!r} is not one of {', '.join(KINDS)}")
    fields = KIND_FIELDS.get(kind, ())
    for key in v:
        if key != "kind" and key not in fields:
            raise ProblemError(f"{where}: {kind} takes no field {key!r}")
    for key in fields:
        if not is_int(v.get(key)) or v[key] < 0:
            raise ProblemError(f"{where}: {kind} needs the integer field {key!r}")
        if kind == "SYM" and v[key] == 0:
            raise ProblemError(f"{where}: SYM qBits must be at least 1, got 0")
        lo, hi = WIDTH_RANGE.get((kind, key), (None, None))
        if lo is not None and not lo <= v[key] <= hi:
            raise ProblemError(f"{where}: {kind} {key} {v[key]} is outside C's range {lo} to {hi}")
    return Dtype(kind, tuple((key, v[key]) for key in fields))


def _flag(v, where: str) -> bool:
    if not isinstance(v, bool):
        raise ProblemError(f"{where}: expected true or false")
    return v


def _layer(v, where: str) -> LayerSpec:
    _keys(v, where, LAYER_KEYS, ("cost",))
    kind = v["type"]
    if kind not in LAYER_TYPES:
        raise ProblemError(f"{where}.type: {kind!r} is not a layer type this schema knows")
    operand = COST_KEY.get(kind)
    cost = v.get("cost")
    if operand is None and "cost" in v:
        raise ProblemError(f"{where}.cost: {kind} takes no cost operand")
    if operand is not None and (not isinstance(cost, dict) or list(cost) != [operand]
                                or not is_int(cost[operand]) or cost[operand] < 1):
        raise ProblemError(f'{where}.cost: {kind} needs {{"{operand}": <positive integer>}}')
    value = None if operand is None else cost[operand]
    return LayerSpec(kind, _flag(v["frozen"], f"{where}.frozen"),
                     _flag(v["trackRunningStats"], f"{where}.trackRunningStats"),
                     _shape(v["outShape"], f"{where}.outShape"),
                     _dtype(v["outputDtype"], f"{where}.outputDtype"),
                     _dtype(v["propLossDtype"], f"{where}.propLossDtype"),
                     mac_per_element=value if operand == "macPerElement" else None,
                     kernel=value if operand == "kernel" else None)
