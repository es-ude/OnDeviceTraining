#!/usr/bin/env python3
"""Run every differential cell: build operands, drive the C, run both legs, report.

    uv run test/differential/run_cells.py [--build-dir build/differential]
                                          [--cells 'test/differential/cells/*.json']
                                          [--out test/differential/out] [--gold auto|skip|PATH]

For each cell JSON and each of its tiers:
  1. build the operand CODES + float32 scales (and float32 VALUES for a
     FLOAT32-stored operand) from the cell's `fixture`;
  2. write them (.npy + spec.json) and run `odt_diffdrive`;
  3. Leg A (ref/sym_ref.py, docs-only) recomputes every produced output; the
     bias seed is resolved per output channel by the seed policy
     (ref/census.py::seed_policy) and the census classifies every remaining
     mismatch -- exactly on the kernel tier, +-1 at restore ties on the
     layer tier; a FLOAT32 grad is compared to the float64 value within the
     float32 roundings its production can take; a reduction whose exact
     accumulator does not fit int32 is classified MATCH / ABORT / WRAP;
  4. Leg B (ref/exact_ref.py) computes the float64 exact result on the
     dequantized operands and checks the analytic bound;
  5. optionally cross-checks a Conv1d layer-tier output against the existing
     PyTorch gold header (calibration aid for the cell-0 fixtures).
Writes out/report.md and out/report.json. Exit status 1 if any row FAILs.

Op families: `conv1d` (forward), `linear` (forward, dx, weightGrad) and
`convert` (encode, requant_dynamic, requant_fixed). A Linear backward cell
runs ONE backward on the layer tier; the direction names the output the row
evaluates (dx -> propLoss, weightGrad -> weightGrad + biasGrad). Cells with a
`family` of `bfp_pack` / `bfp_gemm` / `bfp_fq` (design step 3, BFP) are
dispatched to bfp_rows.py, which follows the same row shape.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))

import bfp_rows  # noqa: E402
from ref import bfp_ref, census, exact_ref, sym_ref  # noqa: E402
from ref.census import SEED_REF, ChannelSeed  # noqa: E402
from ref.sym_ref import ConvGeometry  # noqa: E402

F32_U = 2.0**-24


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #


def f32(x: float) -> float:
    return float(np.float32(x))


def hex32(x: float) -> str:
    return f32(x).hex()


def is_power_of_two(x: float) -> bool:
    m, _e = np.frexp(np.float64(x))
    return float(m) == 0.5


@dataclass
class Operand:
    """One operand of a cell. `storage` is how the LAYER tier stores it
    (SYM_INT32 codes, packed SYM codes, or FLOAT32 values); the kernel tier
    always receives the codes as SYM_INT32 at `bits`. A FLOAT32 operand is
    grid-exact: values == codes * scale in float32, so both tiers see the
    same integer image and Leg A's staging assumption is checked without a
    float32 encode tie ever entering a MAC."""

    storage: str
    codes: np.ndarray  # int64
    scale: float  # float32 value as a Python float
    bits: int
    values: np.ndarray | None = None  # float32 values (FLOAT32 storage only)


class Fixture:
    def __init__(self):
        self.ops: dict[str, Operand] = {}
        self.notes: list[str] = []

    def add_codes(self, name: str, codes: np.ndarray, scale: float, bits: int, storage: str = "SYM_INT32") -> None:
        codes = np.asarray(codes, dtype=np.int64)
        q = sym_ref.qmax(bits)
        if codes.size and (np.abs(codes) > q).any():
            raise ValueError(f"{name}: codes exceed the {bits}-bit grid")
        values = None
        if storage == "FLOAT32":
            values = (codes.astype(np.float32) * np.float32(scale)).astype(np.float32)
            back = values.astype(np.float64) / f32(scale)
            if not np.array_equal(np.rint(back), codes) or not np.all(back == np.rint(back)):
                raise ValueError(f"{name}: FLOAT32 operand is not grid-exact")
        self.ops[name] = Operand(storage, codes, f32(scale), bits, values)

    def add_encoded(self, name: str, values: np.ndarray, bits: int) -> None:
        enc = sym_ref.encode_sym(np.asarray(values, dtype=np.float64), bits)
        self.add_codes(name, enc.codes, f32(enc.scale), bits)

    def has(self, name: str) -> bool:
        return name in self.ops

    def codes(self, name: str) -> np.ndarray:
        return self.ops[name].codes

    def scale(self, name: str) -> float:
        return self.ops[name].scale

    def deq(self, name: str) -> np.ndarray:
        """float64 value image: the float32 values of a FLOAT32 operand, else
        codes * scale (identical for grid-exact operands)."""
        op = self.ops[name]
        if op.values is not None:
            return op.values.astype(np.float64)
        return exact_ref.dequant(op.codes, op.scale)


DEFAULT_SCALE_RANGES = {  # value absmax ranges, divided by qMax below
    "input": (0.5, 4.0),
    "weight": (0.05, 0.5),
    "bias": (0.02, 0.3),
    "loss": (0.01, 1.0),
}


def random_scale(rng: np.random.Generator, lo: float, hi: float, qmax: int) -> float:
    absmax = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    scale = f32(absmax / qmax)
    if is_power_of_two(scale):
        scale = f32(scale * 1.0000001)
    return scale


def grid_exact_scale(rng: np.random.Generator, lo: float, hi: float, qmax: int) -> float:
    """A non-power-of-two float32 with at most 13 significant bits, so that
    code * scale is exact in float32 for |code| <= 2047 (11 bits) and the
    staged absmax / qMax reproduces the scale exactly (an exact product
    divided by its own integer factor is exact)."""
    absmax = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    s = absmax / qmax
    e = int(np.floor(np.log2(s)))
    unit = 2.0 ** (e - 12)
    m = int(round(s / unit))  # 13-bit significand (4096 <= m <= 8192)
    if m % 2 == 0:
        m += 1  # odd: never a power of two, keeps 13 bits
    if m >= 8192:
        m = 8191
    return f32(m * unit)


def codes_with_extremes(rng: np.random.Generator, shape: tuple, qmax: int, rows_full: bool) -> np.ndarray:
    """Random codes in [-qmax, qmax] with +-qmax present (absmax maps onto
    qMax); with rows_full every row of a 2-D array carries a +-qmax so a
    per-sample and a per-tensor staging derive the same scale."""
    codes = rng.integers(-qmax, qmax + 1, size=shape, dtype=np.int64)
    flat = codes.reshape(-1)
    if rows_full and codes.ndim == 2:
        for r in range(codes.shape[0]):
            j = int(rng.integers(0, codes.shape[1]))
            codes[r, j] = qmax if rng.integers(0, 2) else -qmax
    elif flat.size:
        j = int(rng.integers(0, flat.size))
        flat[j] = qmax if rng.integers(0, 2) else -qmax
    return codes


def build_conv_fixture(cell: dict) -> Fixture:
    fx_spec = cell["fixture"]
    bits = int(cell.get("operandBits", 12))
    fx = Fixture()
    kind = fx_spec["type"]
    if kind == "float_values":
        for name in ("input", "weight", "bias"):
            if fx_spec.get(name) is not None:
                fx.add_encoded(name, np.asarray(fx_spec[name], dtype=np.float64), bits)
    elif kind == "torch_randn":
        import torch  # fixture reproduction only; Leg A never sees torch values

        torch.manual_seed(int(fx_spec["seed"]))
        for name, shape in fx_spec["draws"]:
            fx.add_encoded(name, torch.randn(*shape).numpy().astype(np.float64), bits)
    elif kind == "random_codes":
        rng = np.random.default_rng(int(fx_spec["seed"]))
        qmax = sym_ref.qmax(bits)
        ranges = dict(DEFAULT_SCALE_RANGES)
        ranges.update({k: tuple(v) for k, v in fx_spec.get("absmaxRanges", {}).items()})
        for name in ("input", "weight", "bias"):
            shape = fx_spec.get(f"{name}_shape")
            if shape is None:
                continue
            codes = rng.integers(-qmax, qmax + 1, size=shape, dtype=np.int64)
            lo, hi = ranges[name]
            fx.add_codes(name, codes, random_scale(rng, lo, hi, qmax), bits)
    else:
        raise ValueError(f"{cell['name']}: unknown fixture type {kind!r}")

    for name in ("input", "weight"):
        if not fx.has(name):
            raise ValueError(f"{cell['name']}: fixture lacks operand {name}")
    for name, op in fx.ops.items():
        if is_power_of_two(op.scale):
            fx.notes.append(f"{name} scale {op.scale!r} is a power of two")
    geom = ConvGeometry.from_spec(cell["geometry"])
    c_in = fx.codes("input").shape[1]
    reduction = (c_in // geom.groups) * geom.kernel_size
    if reduction > 511:
        raise ValueError(
            f"{cell['name']}: reduction length {reduction} > 511 (int12 soundness bound)"
        )
    return fx


def build_linear_fixture(cell: dict) -> Fixture:
    """Linear operands. `linear_grid`: random codes with +-qMax present (per
    row for the input, so per-sample and per-tensor staging coincide --
    `rowAbsmax: "first_only"` deliberately breaks that to probe the
    granularity), non-power-of-two scales, a grid-exact FLOAT32 input when
    `inputStorage` is FLOAT32. `full_scale`: every input and weight code is
    +qMax, so every product is the largest the grids allow and the exact
    accumulator is N * qMax_in * qMax_w (the F9 hazard probe)."""
    fx_spec = cell["fixture"]
    fx = Fixture()
    rng = np.random.default_rng(int(fx_spec["seed"]))
    batch, in_f, out_f = int(fx_spec["batch"]), int(fx_spec["inF"]), int(fx_spec["outF"])
    in_bits = int(fx_spec.get("inputBits", 12))
    w_bits = int(fx_spec.get("weightBits", 8))
    b_bits = int(fx_spec.get("biasBits", w_bits))
    l_bits = int(fx_spec.get("lossBits", 12))
    in_storage = fx_spec.get("inputStorage", "FLOAT32")
    w_storage = fx_spec.get("weightStorage", "SYM")
    b_storage = fx_spec.get("biasStorage", w_storage)
    ranges = dict(DEFAULT_SCALE_RANGES)
    ranges.update({k: tuple(v) for k, v in fx_spec.get("absmaxRanges", {}).items()})
    kind = fx_spec["type"]
    q_in, q_w, q_b, q_l = (sym_ref.qmax(b) for b in (in_bits, w_bits, b_bits, l_bits))

    if kind == "linear_grid":
        rows_full = fx_spec.get("rowAbsmax", "every_row") == "every_row"
        x = codes_with_extremes(rng, (batch, in_f), q_in, rows_full=True)
        if not rows_full:
            x[1:] = np.clip(x[1:], -q_in // 2, q_in // 2)  # only row 0 reaches the grid edge
        w = codes_with_extremes(rng, (out_f, in_f), q_w, rows_full=False)
    elif kind == "full_scale":
        x = np.full((batch, in_f), q_in, dtype=np.int64)
        w = np.full((out_f, in_f), q_w, dtype=np.int64)
    else:
        raise ValueError(f"{cell['name']}: unknown fixture type {kind!r}")

    s_x = grid_exact_scale(rng, *ranges["input"], q_in) if in_storage == "FLOAT32" else random_scale(rng, *ranges["input"], q_in)
    fx.add_codes("input", x, s_x, in_bits, in_storage)
    fx.add_codes("weight", w, random_scale(rng, *ranges["weight"], q_w), w_bits, w_storage)
    if fx_spec.get("bias", True):
        b = codes_with_extremes(rng, (out_f,), q_b, rows_full=False)
        fx.add_codes("bias", b, random_scale(rng, *ranges["bias"], q_b), b_bits, b_storage)
    if cell["direction"] in ("dx", "weightGrad"):
        l = codes_with_extremes(rng, (batch, out_f), q_l, rows_full=False)
        if kind == "full_scale":
            l = np.full((batch, out_f), q_l, dtype=np.int64)
        fx.add_codes("loss", l, random_scale(rng, *ranges["loss"], q_l), l_bits)
    for name, op in fx.ops.items():
        if is_power_of_two(op.scale):
            fx.notes.append(f"{name} scale {op.scale!r} is a power of two")
    return fx


def tie_values(rng: np.random.Generator, n: int, qmax: int, scale: float, k: int) -> np.ndarray:
    """float32 values whose real-valued code x / scale sits within k ulp32 of a
    .5 boundary: (m + 0.5) * scale nudged by j ulp32(m + 0.5), j in [-k, k]."""
    m = rng.integers(0, qmax - 1, size=n)
    sign = np.where(rng.integers(0, 2, size=n) == 1, 1.0, -1.0)
    j = rng.integers(-k, k + 1, size=n)
    pre = (m + 0.5) + j * np.spacing((m + 0.5).astype(np.float32)).astype(np.float64)
    return (sign * pre * scale).astype(np.float32)


def tie_codes(m_max: int, q_out: int, count: int, k: int, exclude: set[int]) -> list[int]:
    """Input codes m (|m| <= m_max) whose exact requant ratio m * q_out / m_max
    lies closest to a .5 boundary -- exact ties where the ratio allows them,
    else the nearest the integer grid offers (reported, since a near-miss
    outside k ulp32 is a legitimate non-tie)."""
    cands = []
    for m in range(1, m_max):
        if m in exclude:
            continue
        v = m * q_out / m_max
        frac = v - np.floor(v)
        cands.append((abs(frac - 0.5), m))
    cands.sort()
    return [m for _d, m in cands[:count]]


def build_convert_fixture(cell: dict, k: int) -> Fixture:
    """S11 operands. encode: float32 values with explicit tie elements and an
    absmax that pins the scale exactly. requant_dynamic: codes at s_in whose
    absmax code M makes m * qMax_out / M an exact .5 for chosen m.
    requant_fixed: codes whose target-grid images include interior ties,
    the +-(qMax + 0.5) clamp boundary and values far outside the grid."""
    fx_spec = cell["fixture"]
    fx = Fixture()
    rng = np.random.default_rng(int(fx_spec["seed"]))
    n = int(fx_spec.get("n", 64))
    in_bits = int(fx_spec.get("inputBits", 12))
    out_bits = int(cell.get("outputBits", 12))
    q_in, q_out = sym_ref.qmax(in_bits), sym_ref.qmax(out_bits)
    direction = cell["direction"]
    if direction == "encode":
        scale = grid_exact_scale(rng, 0.5, 4.0, q_out)
        n_tie = int(fx_spec.get("ties", 16))
        vals = (rng.uniform(-1.0, 1.0, size=n) * q_out * scale).astype(np.float32)
        vals[:n_tie] = tie_values(rng, n_tie, q_out, scale, k)
        vals[n_tie] = np.float32(q_out * scale)  # absmax exactly qMax * scale
        vals[n_tie + 1] = np.float32((q_out - 0.5) * scale)  # tie right under the top code
        fx.ops["input"] = Operand("FLOAT32", np.zeros(n, dtype=np.int64), scale, out_bits, vals)
        fx.notes.append(f"{n_tie + 1} tie elements, absmax = qMax * scale (scale 13-bit exact)")
    elif direction == "requant_dynamic":
        m_max = int(fx_spec.get("absmaxCode", q_in))
        n_tie = int(fx_spec.get("ties", 16))
        codes = rng.integers(-m_max, m_max + 1, size=n, dtype=np.int64)
        ties = tie_codes(m_max, q_out, n_tie, k, exclude=set())
        codes[: len(ties)] = ties
        codes[len(ties)] = m_max
        codes[len(ties) + 1] = -m_max
        s_in = random_scale(rng, 0.5, 4.0, m_max)
        fx.add_codes("input", codes, s_in, in_bits)
        worst = max(abs((m * q_out / m_max) % 1.0 - 0.5) for m in ties) if ties else 0.0
        fx.notes.append(f"absmax code {m_max}, {len(ties)} tie codes, worst |frac-.5| = {worst:.2e}")
    elif direction == "requant_fixed":
        ratio_num, ratio_den = fx_spec.get("ratio", [3, 4])  # s_in / s_target as a rational
        s_t = random_scale(rng, 0.5, 4.0, q_out)
        s_in = f32(s_t * ratio_num / ratio_den)
        r = s_in / s_t  # float64 ratio of the two float32 scales
        codes = rng.integers(-q_in, q_in + 1, size=n, dtype=np.int64)
        # interior ties: m * r = integer + 0.5 when m*ratio_num is an odd multiple of ratio_den/2
        ties = [m for m in range(1, q_in) if abs((m * ratio_num / ratio_den) % 1.0 - 0.5) < 1e-12][:12]
        codes[: len(ties)] = ties
        i = len(ties)
        edge = int(round((q_out + 0.5) * ratio_den / ratio_num))  # m * r ~ qMax + 0.5
        for v in (edge, -edge, edge + 1, -(edge + 1), edge - 1, -(edge - 1)):
            if abs(v) <= q_in:
                codes[i] = v
                i += 1
        codes[i] = q_in  # far outside the target grid when r * q_in > q_out
        codes[i + 1] = -q_in
        fx.add_codes("input", codes, s_in, in_bits)
        fx.ops["target_scale"] = Operand("SCALE", np.zeros(0, dtype=np.int64), s_t, out_bits)
        fx.notes.append(
            f"ratio s_in/s_t = {r!r} (~{ratio_num}/{ratio_den}), {len(ties)} interior ties, "
            f"clamp-edge codes around {edge}"
        )
    else:
        raise ValueError(f"{cell['name']}: unknown convert direction {direction!r}")
    return fx


# --------------------------------------------------------------------------- #
# Driver I/O                                                                  #
# --------------------------------------------------------------------------- #


def operand_spec(op: Operand, name: str, tier: str, in_dir: Path, rounding: str) -> dict:
    """Writes the operand's .npy for `tier` and returns its spec entry. The
    kernel tier always gets codes (SYM_INT32); the layer tier gets the
    declared storage: float32 values, packed SYM codes, or SYM_INT32 codes."""
    storage = op.storage if tier != "kernel" else "SYM_INT32"
    if storage == "FLOAT32":
        np.save(in_dir / f"{name}.npy", np.ascontiguousarray(op.values, dtype=np.float32))
        return {"file": f"{name}.npy", "dtype": "FLOAT32"}
    np.save(in_dir / f"{name}.npy", np.ascontiguousarray(op.codes, dtype=np.int32))
    entry = {"file": f"{name}.npy", "dtype": storage, "scale": hex32(op.scale), "rounding": rounding}
    entry["qBits" if storage == "SYM" else "qMaxBits"] = op.bits
    return entry


def write_inputs(in_dir: Path, fx: Fixture, cell: dict, tier: str) -> Path:
    in_dir.mkdir(parents=True, exist_ok=True)
    rounding = cell["arith"].get("rounding", "HALF_AWAY")
    operands = {
        name: operand_spec(op, name, tier, in_dir, rounding)
        for name, op in fx.ops.items()
        if op.storage != "SCALE"
    }
    output = {
        "dtype": "SYM_INT32",
        "qMaxBits": int(cell.get("outputBits", 12)),
        "rounding": rounding,
    }
    if fx.has("target_scale"):
        output["scale"] = hex32(fx.scale("target_scale"))
    spec = {
        "op": cell["op"],
        "tier": tier,
        "direction": cell["direction"],
        "arith": cell["arith"],
        "operands": operands,
        "output": output,
    }
    if "geometry" in cell:
        spec["geometry"] = cell["geometry"]
    spec_path = in_dir / "spec.json"
    spec_path.write_text(json.dumps(spec, indent=1))
    return spec_path


@dataclass
class DriverResult:
    exit: int
    outputs: dict[str, dict] = field(default_factory=dict)  # name -> {codes|values, scale, meta}
    stdout: str = ""
    stderr: str = ""

    def out(self, name: str) -> dict:
        if name not in self.outputs:
            raise RuntimeError(f"driver produced no output {name!r} (has {sorted(self.outputs)})")
        return self.outputs[name]


def run_driver(driver: Path, spec_path: Path, in_dir: Path, out_dir: Path) -> DriverResult:
    """Runs the driver. Exit 2 (a spec error) raises -- that is a harness bug.
    Exit 1 (a library fail-fast, PRINT_ERROR + exit(1)) is returned as an
    ABORT outcome for the caller to classify; the overflow cells expect it
    as one legitimate answer."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("*"):
        stale.unlink()
    proc = subprocess.run(
        [str(driver), str(spec_path), str(in_dir), str(out_dir)],
        capture_output=True,
        text=True,
    )
    if proc.returncode not in (0, 1):
        raise RuntimeError(
            f"driver exit {proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    res = DriverResult(exit=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)
    if proc.returncode != 0:
        return res
    result = json.loads((out_dir / "result.json").read_text())
    for name, meta in result["outputs"].items():
        arr = np.load(out_dir / meta["file"])
        entry = {"meta": meta, "dtype": meta["dtype"]}
        if meta["dtype"] == "FLOAT32":
            entry["values"] = arr.astype(np.float32)
            entry["scale"] = None
        else:
            entry["codes"] = arr.astype(np.int64)
            entry["scale"] = float.fromhex(meta["scale_hex"]) if "scale_hex" in meta else None
        res.outputs[name] = entry
    return res


# --------------------------------------------------------------------------- #
# Leg A: raw wire, seed policy, per-channel resolution                       #
# --------------------------------------------------------------------------- #


@dataclass
class LegAInputs:
    raw: sym_ref.RawForward  # mac [B, C, L], s_raw, real-valued seed (or None)
    policy: list[ChannelSeed] | None  # per output channel; None without bias
    f32_seed: np.ndarray | None  # the float32 left-to-right reading (NaN = undefined)

    @property
    def allowed(self) -> list[list[tuple[int, str]]]:
        """Acceptable (seed, kind) pairs per output channel; a single
        (0, "none") without a bias."""
        c_out = self.raw.mac.shape[1]
        if self.policy is None:
            return [[(0, "none")] for _ in range(c_out)]
        return [cs.allowed for cs in self.policy]

    @property
    def seed_abs_max(self) -> int:
        if self.policy is None:
            return 0
        return max((max(abs(s) for s, _k in cs.allowed) for cs in self.policy), default=0)


def with_seed_policy(raw: sym_ref.RawForward, fx: Fixture, k: int) -> LegAInputs:
    """Attaches the per-channel seed policy when the raw wire carries a bias
    seed (the seed's float32 reading needs the bias, input and weight scales)."""
    policy = None
    f32_seed = None
    if raw.seed is not None:
        f32_seed = sym_ref.bias_seed_f32_literal(
            fx.codes("bias"), fx.scale("bias"), fx.scale("input"), fx.scale("weight")
        )
        policy = census.seed_policy(raw.seed.pre, f32_seed, k)
    return LegAInputs(raw=raw, policy=policy, f32_seed=f32_seed)


def seed_summary(inp: LegAInputs, chosen: list[tuple[int, str]], extra: list[dict] | None = None) -> dict | None:
    """Per-channel record of what the seed policy allowed and what was used,
    plus the counts the report prints. ``extra`` adds per-channel fields
    (kernel tier: residual class)."""
    if inp.policy is None:
        return None
    channels = []
    n_alt = n_f32 = n_f32_differs = 0
    max_dev = 0
    f32_agree = 0
    regimes: dict[str, int] = {}
    for c, cs in enumerate(inp.policy):
        seed, kind = chosen[c]
        dev = int(seed - cs.ref)
        max_dev = max(max_dev, abs(dev))
        regimes[cs.regime] = regimes.get(cs.regime, 0) + 1
        if kind == census.SEED_ALT:
            n_alt += 1
        elif kind == census.SEED_F32:
            n_f32 += 1
            if seed != cs.ref:
                n_f32_differs += 1
        if cs.f32 is not None and cs.f32 == cs.ref:
            f32_agree += 1
        entry = {
            "channel": c,
            "pre": cs.pre,
            "ref": cs.ref,
            "alt": cs.alt,
            "f32": cs.f32,
            "regime": cs.regime,
            "granted_lsb": cs.granted_lsb,
            "used": int(seed),
            "kind": kind,
            "deviation_from_ref": dev,
        }
        if extra is not None:
            entry.update(extra[c])
        channels.append(entry)
    return {
        "n_channels": len(channels),
        "regimes": regimes,
        "n_alt": n_alt,
        "n_f32_gated": n_f32,
        "n_f32_gated_differs_from_ref": n_f32_differs,
        "max_deviation_lsb": max_dev,
        "f32_agrees_with_ref": {"channels_agree": f32_agree, "channels": len(channels)},
        "channels": channels,
    }


def deviates(inp: LegAInputs, chosen: list[tuple[int, str]]) -> bool:
    """True when some channel's accepted seed is not the real-valued
    rounding (an alt flip, or a float32-gated seed that differs)."""
    if inp.policy is None:
        return False
    return any(seed != cs.ref for cs, (seed, _k) in zip(inp.policy, chosen))


def resolve_leg_a_kernel(c_codes: np.ndarray, c_scale: float, inp: LegAInputs, k: int) -> dict:
    """Kernel tier: raw int32 must match EXACTLY. The seed decouples per
    channel (s_raw does not depend on it), so each channel's residual
    C - mac is examined on its own: a constant residual that is an allowed
    seed is the channel's seed; a constant residual that is not is a
    seed-class mismatch; a non-constant residual is a MAC-class mismatch."""
    mac = inp.raw.mac
    if c_codes.shape != mac.shape:
        raise RuntimeError(f"shape mismatch C {c_codes.shape} vs ref {mac.shape}")
    allowed = inp.allowed
    chosen: list[tuple[int, str]] = []
    extra: list[dict] = []
    classes: set[str] = set()
    for c in range(mac.shape[1]):
        resid = c_codes[:, c, :] - mac[:, c, :]
        constant = resid.size == 0 or bool((resid == resid.flat[0]).all())
        implied = int(resid.flat[0]) if resid.size else 0
        match = next(((s, kind) for s, kind in allowed[c] if constant and s == implied), None)
        if match is None:
            match = min(allowed[c], key=lambda sk: int((resid != sk[0]).sum()))
            if resid.size:
                classes.add("seed" if constant else "mac")
        chosen.append(match)
        extra.append({"residual_constant": constant, "implied_seed": implied if constant else None})
    seeds = np.array([s for s, _ in chosen], dtype=np.int64)
    ref_codes = mac + seeds.reshape(1, -1, 1)
    res = census.census(c_codes, ref_codes, None, k)  # exact mode
    taxonomy = res.taxonomy
    if taxonomy == "EXACT" and deviates(inp, chosen):
        taxonomy = "TIE_ONLY"
    return {
        "taxonomy": taxonomy,
        "census": res.to_json(),
        "mismatch_class": "+".join(sorted(classes)) if classes else "none",
        "seed": seed_summary(inp, chosen, extra),
        "scale": census.scale_check(c_scale, inp.raw.s_raw),
        "ref_scale": inp.raw.s_raw,
    }


def resolve_leg_a_layer(
    c_codes: np.ndarray, c_scale: float, inp: LegAInputs, out_bits: int, k: int, scale_rel_tol: float
) -> dict:
    """Layer tier: the restore couples the channels only through the fresh
    scale absmax(raw * s_raw) / qMax. Every assignment of allowed seeds has
    an absmax that is some channel's maximum under one of its allowed seeds,
    so the realizable absmax values are enumerated (at most 2 * Cout), the
    observed C scale selects those within the derived scale tolerance (none:
    scale MISMATCH, the closest is still evaluated for the report), and under
    each selected scale the channels are resolved independently -- linear in
    Cout instead of 2^(tie channels)."""
    mac = inp.raw.mac
    s_raw = inp.raw.s_raw
    if c_codes.shape != mac.shape:
        raise RuntimeError(f"shape mismatch C {c_codes.shape} vs ref {mac.shape}")
    q_max = sym_ref.qmax(out_bits)
    q_min = sym_ref.qmin(out_bits)
    c_out = mac.shape[1]
    allowed = inp.allowed
    chmax = [
        [int(np.abs(mac[:, c, :] + s).max()) if mac[:, c, :].size else 0 for s, _ in allowed[c]]
        for c in range(c_out)
    ]
    lower = max((min(row) for row in chmax), default=0)
    cands = sorted({v for row in chmax for v in row if v >= lower}) or [0]

    def scale_of(v: int) -> float:
        return 1.0 if v == 0 else v * s_raw / q_max

    rels = {v: abs(c_scale - scale_of(v)) / scale_of(v) for v in cands}
    matching = [v for v in cands if rels[v] <= scale_rel_tol]
    evaluate = matching if matching else [min(cands, key=lambda v: rels[v])]

    best = None
    for v in evaluate:
        scale = scale_of(v)
        per_channel: list[dict[int, tuple]] = []
        chosen_idx: list[int] = []
        for c in range(c_out):
            opts: dict[int, tuple] = {}
            for j, (s, kind) in enumerate(allowed[c]):
                if chmax[c][j] > v:
                    continue  # would exceed the candidate absmax
                pre = (mac[:, c, :] + s) * s_raw / scale
                codes = np.clip(sym_ref.round_half_away(pre), q_min, q_max)
                res = census.census(c_codes[:, c, :], codes, pre, k)
                opts[j] = (
                    res.n_mismatch - res.n_tie,
                    res.n_mismatch,
                    0 if chmax[c][j] == v else 1,  # prefer realizing the candidate
                    0 if kind in (SEED_REF, "none") else 1,  # then the real-valued rounding
                )
            per_channel.append(opts)
            chosen_idx.append(min(opts, key=lambda j: opts[j]))
        if not any(chmax[c][chosen_idx[c]] == v for c in range(c_out)):
            # The best per-channel choices do not reach this absmax: force the
            # cheapest realizer so the evaluated assignment is a legitimate
            # Leg A outcome (its own absmax IS v).
            realizers = [(c, j) for c in range(c_out) for j in per_channel[c] if chmax[c][j] == v]
            c_force, j_force = min(realizers, key=lambda cj: per_channel[cj[0]][cj[1]])
            chosen_idx[c_force] = j_force
        chosen = [allowed[c][chosen_idx[c]] for c in range(c_out)]
        seeds = np.array([s for s, _ in chosen], dtype=np.int64)
        restored = sym_ref.restore_sym(mac + seeds.reshape(1, -1, 1), s_raw, out_bits)
        if abs(restored.scale - scale) > 1e-9 * scale:
            raise AssertionError("candidate absmax not realized by the chosen seeds")
        res = census.census(c_codes, restored.codes, restored.pre, k)
        n_dev = sum(seed != cs.ref for cs, (seed, _k) in zip(inp.policy or [], chosen))
        key = (res.n_mismatch - res.n_tie, res.n_mismatch, n_dev)
        if best is None or key < best[0]:
            best = (key, v, chosen, res, restored.scale)
    _key, v_best, chosen, res, ref_scale = best
    taxonomy = res.taxonomy
    if taxonomy == "EXACT" and deviates(inp, chosen):
        taxonomy = "TIE_ONLY"
    scale = census.scale_check(c_scale, ref_scale, scale_rel_tol)
    scale["ok"] = bool(matching)
    return {
        "taxonomy": taxonomy,
        "census": res.to_json(),
        "seed": seed_summary(inp, chosen),
        "scale": scale,
        "ref_scale": ref_scale,
        "scale_candidates": {"realizable": len(cands), "matching": len(matching), "used_absmax": v_best},
    }


def resolve_leg_a_float(c_values: np.ndarray, ref: np.ndarray, n_roundings: int) -> dict:
    """A FLOAT32-stored output against its float64 reference: EXACT when every
    element equals the float32 rounding of the reference, TIE_ONLY when the
    difference stays within the ``n_roundings`` float32 roundings the
    documented production can take (the float32 expression order is not
    documented), MISMATCH beyond that. The scale field is kept for the
    report's shape (a FLOAT32 output has none)."""
    c = np.asarray(c_values, dtype=np.float64).ravel()
    r = np.asarray(ref, dtype=np.float64).ravel()
    if c.shape != r.shape:
        raise RuntimeError(f"shape mismatch C {c.shape} vs ref {r.shape}")
    ref32 = r.astype(np.float32).astype(np.float64)
    diff = np.abs(c - r)
    tol = ((1.0 + F32_U) ** n_roundings - 1.0) * np.abs(r) + np.finfo(np.float32).tiny
    bad = np.flatnonzero(diff > tol)
    n_mismatch = int((c != ref32).sum())
    ulp = census.ulp32(np.maximum(np.abs(r), np.finfo(np.float32).tiny))
    max_ulp = float((diff / ulp).max()) if diff.size else 0.0
    taxonomy = "EXACT" if n_mismatch == 0 else ("TIE_ONLY" if bad.size == 0 else "MISMATCH")
    worst = [
        {"index": int(i), "c": float(c[i]), "ref": float(r[i]), "diff_ulp32": float(diff[i] / ulp[i])}
        for i in bad[np.argsort(-diff[bad])][:8]
    ]
    return {
        "taxonomy": taxonomy,
        "census": {
            "taxonomy": taxonomy,
            "mode": "float32",
            "n_elements": int(c.size),
            "n_mismatch": n_mismatch,
            "n_tie": n_mismatch - int(bad.size),
            "max_abs_diff": max_ulp,
            "n_roundings": n_roundings,
            "worst": worst,
        },
        "mismatch_class": "none" if bad.size == 0 else "float",
        "seed": None,
        "scale": {"c": None, "ref": None, "rel": 0.0, "ok": True},
        "ref_scale": None,
        "max_err_ulp32": max_ulp,
    }


def classify_overflow(drv: DriverResult, out_name: str, raw_exact: np.ndarray, restore) -> dict:
    """F9: the exact accumulator does not fit int32. The C outcome is one of
    ABORT (library fail-fast, exit 1), MATCH (the codes equal what a widening
    accumulator would produce -- `restore(raw_exact)`), WRAP (the codes equal
    `restore(wrap_int32(raw_exact))`, a silent two's-complement wrap: the
    finding F9 predicts), SATURATE (equal `restore(clip)`), else MISMATCH.
    `restore` maps an int64 raw wire to the compared codes (identity on the
    kernel tier, the width restore on the layer tier)."""
    if drv.exit != 0:
        return {"kind": "ABORT", "detail": drv.stderr.strip().splitlines()[-1:] or ["(no stderr)"]}
    c = drv.out(out_name)["codes"].ravel()  # flat: the wire's [B, outF] vs the ref's [B, C, L]
    options = {
        "MATCH": restore(raw_exact),
        "WRAP": restore(sym_ref.wrap_int32(raw_exact)),
        "SATURATE": restore(np.clip(raw_exact, sym_ref.INT32_MIN, sym_ref.INT32_MAX)),
    }
    for kind, ref in options.items():
        r = np.asarray(ref).ravel()
        if c.size == r.size and np.array_equal(c, r):
            return {"kind": kind, "detail": []}
    return {"kind": "MISMATCH", "detail": [f"codes match none of MATCH/WRAP/SATURATE (max |C| = {int(np.abs(c).max())})"]}


# --------------------------------------------------------------------------- #
# Scale tolerances (derived per path from the float32 rounding count)         #
# --------------------------------------------------------------------------- #


def scale_tol(n_roundings: int) -> float:
    """Relative tolerance for a scale produced by `n_roundings` float32
    roundings: (1+u)^n - 1, u = 2^-24; the 2^-22 cell-0 figure is the 4-rounding
    case rounded up. Stated per path BEFORE the run, never widened after."""
    return (1.0 + F32_U) ** n_roundings - 1.0


# kernel tier: s_raw = s_a * s_b, one float32 product (two units kept, as in
# cell 0: the scale check is rel 2^-22 there)
SCALE_TOL_KERNEL = census.SCALE_REL_TOL
# layer tier, SYM_INT32 operands: restore scale = absmax((float)raw * s_raw)
# / qMax -- (float)raw (1, only above 2^24), s_raw product (1), multiply (1),
# divide (1): 4 roundings
SCALE_TOL_LAYER = max(census.SCALE_REL_TOL, scale_tol(4))
# layer tier with a staged FLOAT32 input: the staging scale absmax / qMax (1,
# exact for grid-exact fixtures but counted) multiplies into s_raw: 5
SCALE_TOL_LAYER_STAGED = max(census.SCALE_REL_TOL, scale_tol(5))


# --------------------------------------------------------------------------- #
# Conv1d rows                                                                 #
# --------------------------------------------------------------------------- #


def conv_leg_a_inputs(fx: Fixture, geom: ConvGeometry, k: int) -> LegAInputs:
    raw = sym_ref.conv1d_sym_forward_raw(
        fx.codes("input"),
        fx.scale("input"),
        fx.codes("weight"),
        fx.scale("weight"),
        fx.codes("bias") if fx.has("bias") else None,
        fx.scale("bias") if fx.has("bias") else None,
        geom,
    )
    return with_seed_policy(raw, fx, k)


def conv_leg_b(fx: Fixture, geom: ConvGeometry, tier: str, c_codes, c_scale, inp: LegAInputs, out_bits: int, k: int) -> dict:
    length = fx.codes("input").shape[2]
    pad_left, pad_right = geom.pads(length)
    exact = exact_ref.conv1d_exact(
        fx.deq("input"),
        fx.deq("weight"),
        fx.deq("bias") if fx.has("bias") else None,
        geom.stride,
        geom.dilation,
        geom.groups,
        pad_left,
        pad_right,
    )
    return gemm_leg_b(exact, tier, c_codes, c_scale, inp, out_bits, k, staging=0.0)


def gemm_leg_b(exact, tier, c_codes, c_scale, inp: LegAInputs, out_bits: int, k: int, staging: float) -> dict:
    if exact.shape != c_codes.shape:
        raise RuntimeError(f"Leg B shape {exact.shape} vs C {c_codes.shape}")
    deq_c = exact_ref.dequant(c_codes, c_scale)
    err_max = float(np.max(np.abs(deq_c - exact))) if exact.size else 0.0
    y_absmax = float(np.max(np.abs(exact))) if exact.size else 0.0
    seed_granted = census.seed_granted_lsb(inp.policy)
    bound = exact_ref.bound_gemm_raw(
        tier,
        c_scale if tier == "layer" else None,
        inp.raw.s_raw,
        seed_granted,
        y_absmax,
        sym_ref.qmax(out_bits),
        k,
        staging,
    )
    out = census.leg_b_check(err_max, bound)
    out["y_absmax"] = y_absmax
    out["s_out"] = c_scale if tier == "layer" else None
    out["s_raw"] = inp.raw.s_raw
    out["seed_granted_lsb"] = seed_granted
    out["staging_term"] = staging
    out["restore_slack_codes"] = exact_ref.restore_slack_codes(sym_ref.qmax(out_bits), k) if tier == "layer" else None
    return out


def run_conv_row(cell: dict, tier: str, fx: Fixture, drv: DriverResult, k: int, gold_header) -> dict:
    geom = ConvGeometry.from_spec(cell["geometry"])
    out_bits = int(cell.get("outputBits", 12))
    if drv.exit != 0:
        raise RuntimeError(f"driver exit {drv.exit}\n{drv.stderr}")
    o = drv.out("output")
    inp = conv_leg_a_inputs(fx, geom, k)
    if tier == "kernel":
        a = resolve_leg_a_kernel(o["codes"], o["scale"], inp, k)
    else:
        a = resolve_leg_a_layer(o["codes"], o["scale"], inp, out_bits, k, SCALE_TOL_LAYER)
    b = conv_leg_b(fx, geom, tier, o["codes"], o["scale"], inp, out_bits, k)
    gold = gold_check(gold_header, cell.get("gold"), o["codes"], o["scale"]) if tier == "layer" else None
    return {"output": "output", "legA": a, "legB": b, "gold": gold, "c_scale": o["scale"], "n_elements": int(o["codes"].size)}


# --------------------------------------------------------------------------- #
# Linear rows                                                                 #
# --------------------------------------------------------------------------- #


def linear_raw(fx: Fixture, direction: str) -> sym_ref.RawForward:
    if direction == "forward":
        return sym_ref.linear_sym_forward_raw(
            fx.codes("input"),
            fx.scale("input"),
            fx.codes("weight"),
            fx.scale("weight"),
            fx.codes("bias") if fx.has("bias") else None,
            fx.scale("bias") if fx.has("bias") else None,
        )
    if direction == "dx":
        return sym_ref.linear_sym_dx_raw(fx.codes("loss"), fx.scale("loss"), fx.codes("weight"), fx.scale("weight"))
    if direction == "weightGrad":
        return sym_ref.linear_sym_weight_grad_raw(fx.codes("loss"), fx.scale("loss"), fx.codes("input"), fx.scale("input"))
    raise ValueError(direction)


def linear_exact(fx: Fixture, direction: str) -> np.ndarray:
    if direction == "forward":
        return exact_ref.linear_exact(fx.deq("input"), fx.deq("weight"), fx.deq("bias") if fx.has("bias") else None)
    if direction == "dx":
        return exact_ref.linear_dx_exact(fx.deq("loss"), fx.deq("weight"))
    if direction == "weightGrad":
        return exact_ref.linear_weight_grad_exact(fx.deq("loss"), fx.deq("input"))
    raise ValueError(direction)


def linear_staging_term(fx: Fixture, direction: str, tier: str, k: int) -> float:
    """Value-unit error the funnel's int12 staging of a FLOAT32 input can add
    (layer tier only; zero for a SYM_INT32-stored input)."""
    if tier != "layer" or fx.ops["input"].storage != "FLOAT32":
        return 0.0
    s_x = fx.scale("input")
    q_in = sym_ref.qmax(fx.ops["input"].bits)
    if direction == "forward":
        other = float(np.abs(fx.deq("weight")).sum(axis=1).max())  # max over outF of sum_i |W|
    elif direction == "weightGrad":
        other = float(np.abs(fx.deq("loss")).sum(axis=0).max())  # max over outF of sum_b |L|
    else:
        return 0.0  # dx does not touch the input
    return exact_ref.staging_term(s_x, q_in, other, k)


def linear_reduction(fx: Fixture, direction: str) -> tuple[int, int, int]:
    """(N, bits_a, bits_b) of the integer reduction: forward over inF
    (input x weight), dx over outF (loss x weight), weightGrad over the
    batch (loss x input)."""
    if direction == "forward":
        return fx.codes("input").shape[1], fx.ops["input"].bits, fx.ops["weight"].bits
    if direction == "dx":
        return fx.codes("loss").shape[1], fx.ops["loss"].bits, fx.ops["weight"].bits
    return fx.codes("loss").shape[0], fx.ops["loss"].bits, fx.ops["input"].bits


def run_linear_row(cell: dict, tier: str, fx: Fixture, drv: DriverResult, k: int) -> dict:
    direction = cell["direction"]
    out_bits = int(cell.get("outputBits", 12))
    raw = linear_raw(fx, direction)
    inp = with_seed_policy(raw, fx, k)
    n_red, bits_a, bits_b = linear_reduction(fx, direction)
    ov = sym_ref.overflow_report(inp.raw.raw, n_red, bits_a, bits_b, inp.seed_abs_max)
    out_name = {"forward": "output", "dx": "propLoss", "weightGrad": "weightGrad"}[direction]
    row: dict = {"output": out_name, "int32": ov.to_json()}
    exact = linear_exact(fx, direction)
    staging = linear_staging_term(fx, direction, tier, k)
    batch_shape = inp.raw.mac.shape

    if not ov.fits:
        # F9: classify instead of census; Leg B still reported.
        if tier == "kernel":
            restore = lambda r: r.reshape(batch_shape)  # noqa: E731
        elif direction == "weightGrad":
            restore = lambda r: sym_ref.dequant(r, inp.raw.s_raw).astype(np.float32).astype(np.float64)  # noqa: E731
        else:
            restore = lambda r: sym_ref.restore_sym(r, inp.raw.s_raw, out_bits).codes  # noqa: E731
        raw_exact = inp.raw.raw
        if direction == "weightGrad" and tier == "layer":
            cls = classify_overflow_float(drv, out_name, raw_exact, restore)
        else:
            cls = classify_overflow(drv, out_name, raw_exact, restore)
        row["overflow"] = cls
        row["legA"] = {"taxonomy": cls["kind"], "census": None, "mismatch_class": "-", "seed": None,
                       "scale": {"c": None, "ref": inp.raw.s_raw, "rel": 0.0, "ok": True}, "ref_scale": inp.raw.s_raw}
        if drv.exit == 0:
            o = drv.out(out_name)
            if o["dtype"] == "FLOAT32":
                deq_c = o["values"].astype(np.float64).reshape(exact.shape)
                err = float(np.max(np.abs(deq_c - exact)))
                row["legB"] = census.leg_b_check(err, exact_ref.bound_float32_grad(float(np.abs(exact).max()), 3, staging))
            else:
                row["legB"] = gemm_leg_b(exact, tier, o["codes"].reshape(exact.shape), o["scale"], inp, out_bits, k, staging)
            row["c_scale"] = o["scale"]
            row["n_elements"] = int(exact.size)
        else:
            row["legB"] = {"taxonomy": "n/a", "err_max": float("nan"), "bound": float("nan"), "ratio": float("nan")}
            row["c_scale"] = None
            row["n_elements"] = int(exact.size)
        return row

    if drv.exit != 0:
        raise RuntimeError(f"driver exit {drv.exit}\n{drv.stderr}")
    o = drv.out(out_name)
    if tier == "kernel":
        c = o["codes"].reshape(batch_shape)
        a = resolve_leg_a_kernel(c, o["scale"], inp, k)
        b = gemm_leg_b(exact, tier, c.reshape(exact.shape), o["scale"], inp, out_bits, k, staging)
    elif direction == "weightGrad":
        # FLOAT32 grad: the first store into a zeroed grad is the increment's
        # float32 value (sym_ref.DOC_GAPS[11]: <= 3 roundings from raw * s_raw).
        ref = sym_ref.dequant(inp.raw.raw.reshape(exact.shape), inp.raw.s_raw)
        a = resolve_leg_a_float(o["values"].reshape(exact.shape), ref, 3)
        deq_c = o["values"].astype(np.float64).reshape(exact.shape)
        err = float(np.max(np.abs(deq_c - exact)))
        b = census.leg_b_check(err, exact_ref.bound_float32_grad(float(np.abs(exact).max()), 3, staging))
        b["staging_term"] = staging
    else:
        tol = SCALE_TOL_LAYER_STAGED if fx.ops["input"].storage == "FLOAT32" and direction == "forward" else SCALE_TOL_LAYER
        c = o["codes"].reshape(batch_shape)
        a = resolve_leg_a_layer(c, o["scale"], inp, out_bits, k, tol)
        b = gemm_leg_b(exact, tier, c.reshape(exact.shape), o["scale"], inp, out_bits, k, staging)
    row.update({"legA": a, "legB": b, "c_scale": o["scale"], "n_elements": int(exact.size)})

    if direction == "weightGrad" and fx.has("bias"):
        row["biasGrad"] = linear_bias_grad_row(fx, tier, drv)
    return row


def classify_overflow_float(drv: DriverResult, out_name: str, raw_exact: np.ndarray, restore) -> dict:
    """``classify_overflow`` for a FLOAT32 output: equality within 3 float32
    roundings of the candidate value instead of code equality."""
    if drv.exit != 0:
        return {"kind": "ABORT", "detail": drv.stderr.strip().splitlines()[-1:] or ["(no stderr)"]}
    c = drv.out(out_name)["values"].astype(np.float64).ravel()
    tol_rel = (1.0 + F32_U) ** 3 - 1.0
    options = {
        "MATCH": restore(raw_exact),
        "WRAP": restore(sym_ref.wrap_int32(raw_exact)),
        "SATURATE": restore(np.clip(raw_exact, sym_ref.INT32_MIN, sym_ref.INT32_MAX)),
    }
    for kind, ref in options.items():
        r = np.asarray(ref, dtype=np.float64).ravel()
        if c.shape == r.shape and np.all(np.abs(c - r) <= tol_rel * np.abs(r) + np.finfo(np.float32).tiny):
            return {"kind": kind, "detail": []}
    return {"kind": "MISMATCH", "detail": ["values match none of MATCH/WRAP/SATURATE"]}


def linear_bias_grad_row(fx: Fixture, tier: str, drv: DriverResult) -> dict:
    """The bias grad that rides along with the weightGrad (kernel: the raw
    int32 value sum at s_loss, exact; layer: the FLOAT32 grad, <= 2 float32
    roundings from sum * s_loss)."""
    o = drv.out("biasGrad")
    ref_sum = sym_ref.linear_sym_bias_grad_raw(fx.codes("loss"))
    exact = exact_ref.bias_grad_exact(fx.deq("loss"))
    g_absmax = float(np.abs(exact).max()) if exact.size else 0.0
    if tier == "kernel":
        res = census.census(o["codes"], ref_sum, None)
        scale = census.scale_check(o["scale"], fx.scale("loss"), 0.0)  # copied verbatim
        a = {"taxonomy": res.taxonomy, "census": res.to_json(), "scale": scale, "ref_scale": fx.scale("loss")}
        deq_c = exact_ref.dequant(o["codes"], o["scale"])
        err = float(np.max(np.abs(deq_c - exact))) if exact.size else 0.0
        b = census.leg_b_check(err, g_absmax * F32_U + np.finfo(np.float32).tiny)
    else:
        ref = sym_ref.dequant(ref_sum, fx.scale("loss"))
        a = resolve_leg_a_float(o["values"], ref, 2)
        err = float(np.max(np.abs(o["values"].astype(np.float64) - exact))) if exact.size else 0.0
        b = census.leg_b_check(err, exact_ref.bound_float32_grad(g_absmax, 2))
    return {"output": "biasGrad", "legA": a, "legB": b, "n_elements": int(exact.size)}


# --------------------------------------------------------------------------- #
# Convert rows (S11)                                                          #
# --------------------------------------------------------------------------- #


def run_convert_row(cell: dict, fx: Fixture, drv: DriverResult, k: int) -> dict:
    direction = cell["direction"]
    out_bits = int(cell.get("outputBits", 12))
    if drv.exit != 0:
        raise RuntimeError(f"driver exit {drv.exit}\n{drv.stderr}")
    o = drv.out("output")
    q_out = sym_ref.qmax(out_bits)
    op = fx.ops["input"]
    if direction == "encode":
        values = op.values.astype(np.float64)
        ref = sym_ref.encode_sym(values, out_bits)
        scale_ok_tol = scale_tol(1)  # absmax / qMax, one float32 division
    elif direction == "requant_dynamic":
        values = exact_ref.dequant(op.codes, op.scale)
        ref = sym_ref.restore_sym(op.codes, op.scale, out_bits)
        scale_ok_tol = scale_tol(3)  # (float)m * s_in (1 each: product, absmax exact), / qMax
    else:
        values = exact_ref.dequant(op.codes, op.scale)
        ref = sym_ref.requant_fixed(op.codes, op.scale, fx.scale("target_scale"), out_bits)
        scale_ok_tol = 0.0  # preset, never modified
    res = census.census(o["codes"], ref.codes, ref.pre, k)
    scale = census.scale_check(o["scale"], ref.scale, scale_ok_tol)
    clamped = np.abs(ref.pre) > q_out + 0.5
    at_edge = (o["codes"] == q_out) | (o["codes"] == sym_ref.qmin(out_bits))
    a = {
        "taxonomy": res.taxonomy,
        "census": res.to_json(),
        "mismatch_class": "none" if res.taxonomy != "MISMATCH" else "code",
        "seed": None,
        "scale": scale,
        "ref_scale": ref.scale,
        "n_clamped_predicted": int(clamped.sum()),
        "c_clamped_to_grid_end": int((at_edge & clamped).sum()),
        "c_codes_at_qmin": int((o["codes"] == sym_ref.qmin(out_bits)).sum()),
    }
    deq_c = exact_ref.dequant(o["codes"], o["scale"])
    err_all = np.abs(deq_c - values)
    err = float(err_all[~clamped].max()) if (~clamped).any() else 0.0
    b = census.leg_b_check(err, exact_ref.bound_requant_roundtrip(o["scale"], q_out, k))
    b["n_clamped_excluded"] = int(clamped.sum())
    b["err_max_clamped"] = float(err_all[clamped].max()) if clamped.any() else 0.0
    return {"output": "output", "legA": a, "legB": b, "c_scale": o["scale"], "n_elements": int(o["codes"].size)}


# --------------------------------------------------------------------------- #
# Gold cross-check (calibration aid; parses the generated header as text)    #
# --------------------------------------------------------------------------- #


def obtain_gold_header(mode: str, out_root: Path) -> str | None:
    if mode == "skip":
        return None
    if mode != "auto":
        return Path(mode).read_text()
    # Not named *.h: the clang-format gate finds every .c/.h under test/, and
    # a generated header in a dirty tree would trip the local `ci` script.
    target = out_root / "gold" / "expected_conv1d.gold.txt"
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        gen = REPO / "test" / "unit" / "layer" / "generate_expected_conv1d.py"
        proc = subprocess.run(
            ["uv", "run", str(gen), "--out", str(target)],
            cwd=str(REPO),
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            print(f"gold: generator failed, skipping cross-check:\n{proc.stderr}", file=sys.stderr)
            return None
    return target.read_text()


def _gold_array(header: str, name: str) -> np.ndarray | None:
    m = re.search(rf"\b{re.escape(name)}\[\]\s*=\s*\{{([^}}]*)\}}", header)
    if not m:
        return None
    body = m.group(1).strip()
    if not body:
        return np.zeros(0)
    return np.array([float(tok.strip().rstrip("fF")) for tok in body.split(",") if tok.strip()])


def _gold_scalar(header: str, name: str) -> float | None:
    m = re.search(rf"\b{re.escape(name)}\s*=\s*([-+0-9.eE]+)f?\s*;", header)
    return float(m.group(1)) if m else None


def gold_check(header: str | None, gold_name: str | None, c_codes, c_scale) -> dict | None:
    if header is None or gold_name is None:
        return None
    pre = f"conv1dSym_{gold_name}"
    exp = _gold_array(header, f"expectedForward_{pre}")
    exp_scale = _gold_scalar(header, f"expectedForwardScale_{pre}")
    mtol = _gold_scalar(header, f"forwardMantissaTol_{pre}")
    stol = _gold_scalar(header, f"forwardScaleTol_{pre}")
    if exp is None or exp_scale is None or mtol is None or stol is None:
        return {"status": "n/a", "note": f"gold symbols for {pre} not found"}
    exp = exp.astype(np.int64)
    if exp.size != c_codes.size:
        return {"status": "mismatch", "note": f"gold has {exp.size} codes, C {c_codes.size}"}
    diff = np.abs(c_codes.ravel() - exp)
    scale_rel = abs(c_scale - exp_scale) / abs(exp_scale)
    ok = bool(diff.max() <= mtol and scale_rel <= stol)
    return {
        "status": "match" if ok else "mismatch",
        "max_code_diff": int(diff.max()),
        "mantissa_tol": int(mtol),
        "scale_rel": float(scale_rel),
        "scale_tol": float(stol),
    }


# --------------------------------------------------------------------------- #
# Orchestration                                                               #
# --------------------------------------------------------------------------- #


def build_fixture(cell: dict, k: int) -> Fixture:
    op = cell["op"]
    if op == "conv1d":
        return build_conv_fixture(cell)
    if op == "linear":
        return build_linear_fixture(cell)
    if op == "convert":
        return build_convert_fixture(cell, k)
    raise ValueError(f"{cell['name']}: unknown op {op!r}")


def verdict_of(row: dict, expect: dict) -> str:
    a, b = row["legA"], row["legB"]
    if "overflow" in row:
        kind = row["overflow"]["kind"]
        allowed = expect.get("overflow", ["MATCH", "ABORT"])
        return "PASS" if kind in allowed else "FAIL"
    v = census.combine(a["taxonomy"], a["scale"]["ok"], b["taxonomy"])
    if expect:
        if a["taxonomy"] not in expect.get("legA", [a["taxonomy"]]):
            v = "FAIL"
        if b["taxonomy"] != expect.get("legB", b["taxonomy"]):
            v = "FAIL"
    return v


def run_row(cell: dict, tier: str, driver: Path, out_root: Path, k: int, gold_header) -> list[dict]:
    """One cell x tier -> one or more rows (a weightGrad cell adds its
    biasGrad row). Every row lands in the report, errors included."""
    name = cell["name"]
    base = {"cell": name, "tier": tier, "op": cell["op"], "direction": cell["direction"]}
    try:
        fx = build_fixture(cell, k)
        work = out_root / "runs" / f"{name}__{tier}"
        spec_path = write_inputs(work / "in", fx, cell, tier)
        drv = run_driver(driver, spec_path, work / "in", work / "out")
        if cell["op"] == "conv1d":
            res = run_conv_row(cell, tier, fx, drv, k, gold_header)
        elif cell["op"] == "linear":
            res = run_linear_row(cell, tier, fx, drv, k)
        else:
            res = run_convert_row(cell, fx, drv, k)
        fixture_info = {
            "type": cell["fixture"]["type"],
            "shapes": {n: list(op.codes.shape) if op.values is None else list(op.values.shape) for n, op in fx.ops.items()},
            "storage": {n: op.storage for n, op in fx.ops.items()},
            "scales": {n: op.scale for n, op in fx.ops.items()},
            "notes": fx.notes,
        }
        expect = cell.get("expect", {})
        rows = []
        extra = res.pop("biasGrad", None)
        row = dict(base)
        row.update(res)
        row["fixture"] = fixture_info
        row["verdict"] = verdict_of(row, expect)
        rows.append(row)
        if extra is not None:
            r2 = dict(base)
            r2.update(extra)
            r2["fixture"] = fixture_info
            r2["verdict"] = verdict_of(r2, expect.get("biasGrad", {}))
            rows.append(r2)
        return rows
    except Exception as exc:  # noqa: BLE001 -- every row must land in the report
        row = dict(base)
        row.update({"verdict": "ERROR", "error": str(exc)})
        return [row]


def fmt_row(r: dict) -> str:
    if r["verdict"] == "ERROR":
        return f"| {r['cell']} | {r['tier']} | {r['direction']} | - | ERROR | | | | | | | | | | | | {r['error'].splitlines()[0]} |"
    a, b = r["legA"], r["legB"]
    c = a.get("census")
    gold = r.get("gold")
    gold_s = "-" if gold is None else gold["status"]
    seed = a.get("seed")
    if seed is None:
        seed_s, dev_s = "-", "-"
    else:
        n = seed["n_channels"]
        n_ref = n - seed["n_alt"] - seed["n_f32_gated"]
        seed_s = f"{n_ref}/{seed['n_alt']}/{seed['n_f32_gated']}"
        dev_s = str(seed["max_deviation_lsb"])
    cls = a.get("mismatch_class", "-")
    ov = r.get("int32")
    if ov is None:
        ov_s = r.get("bfp", {}).get("summary", "-")
    else:
        ov_s = f"{'fits' if ov['fits'] else 'OVERFLOW'} N={ov['reduction']} wrap@{ov['n_wrap_full_scale']}"
    if "overflow" in r:
        ov_s += f" -> {r['overflow']['kind']}"
    if c is None:
        mm_s, tie_s, diff_s = "-", "-", "-"
    else:
        mm_s = f"{c['n_mismatch']}/{c['n_elements']}"
        tie_s = str(c["n_tie"])
        diff_s = f"{c['max_abs_diff']:.3g}" if isinstance(c["max_abs_diff"], float) else str(c["max_abs_diff"])
    rel = a["scale"]["rel"]
    rel_s = "-" if rel is None else f"{rel:.2e}"
    bt = b["taxonomy"]
    if bt == "n/a":
        b_s = "n/a | - | -"
    else:
        b_s = f"{bt} | {b['err_max']:.3e} | {b['bound']:.3e} ({b['ratio']:.2f})"
    return (
        f"| {r['cell']} | {r['tier']} | {r['direction']} | {r.get('output', '-')} | {a['taxonomy']} | "
        f"{mm_s} | {tie_s} | {cls} | {seed_s} | {dev_s} | {diff_s} | {rel_s} | {ov_s} | "
        f"{b_s} | {gold_s} | {r['verdict']} |"
    )


HEADER = (
    "| cell | tier | dir | output | Leg A | mismatches | ties | class | seed ref/alt/f32 | seed dev | "
    "max diff | scale rel | int32 | Leg B | max err | bound (ratio) | gold | verdict |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
)

FOOTER = (
    "Leg A taxonomy: EXACT | TIE_ONLY | MISMATCH, or for a reduction whose exact accumulator does "
    "not fit int32: MATCH | ABORT | WRAP | SATURATE (F9 classification; WRAP = the codes equal the "
    "two's-complement wrap of the exact accumulator, a silent overflow); Leg B: BOUNDED | EXCESS. "
    "`mismatches` = differing codes / total after the seed policy (FLOAT32 outputs: elements not "
    "equal to the float32 rounding of the reference); `ties` = those within k ulp32 of a .5 boundary "
    "(layer tier) or within the allowed float32 roundings (FLOAT32 outputs); `class` (kernel tier) "
    "= seed / mac / none, `float` for a FLOAT32 output beyond its roundings; `seed ref/alt/f32` = bias "
    "channels resolved to the real-valued rounding / the other side of a float32-tied .5 boundary / "
    "the float32 reading; `seed dev` = largest |used - real-valued rounding| in raw LSB; `max diff` = "
    "largest code difference, or ulp32 error for FLOAT32 outputs; `scale rel` = relative scale "
    "disagreement (tolerance derived per path from its float32 rounding count); `int32` = whether the "
    "exact accumulator fits int32, the reduction length N and the first N at which sign-aligned "
    "full-scale operands of this cell's widths would wrap."
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build-dir", default=str(REPO / "build" / "differential"))
    ap.add_argument("--cells", default=str(HERE / "cells" / "*.json"))
    ap.add_argument("--out", default=str(HERE / "out"))
    ap.add_argument("--gold", default="auto", help="auto | skip | path to expected_conv1d.h")
    ap.add_argument("--k", type=int, default=census.K_DEFAULT, help="tie window in ulp32")
    args = ap.parse_args()

    driver = Path(args.build_dir) / "test" / "differential" / "odt_diffdrive"
    if not driver.exists():
        print(
            f"driver not found at {driver}\n"
            "build it with: cmake --preset differential && cmake --build --preset differential",
            file=sys.stderr,
        )
        return 2
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    cell_files = sorted(glob.glob(args.cells))
    if not cell_files:
        print(f"no cells match {args.cells}", file=sys.stderr)
        return 2
    gold_header = obtain_gold_header(args.gold, out_root)

    rows = []
    for path in cell_files:
        cell = json.loads(Path(path).read_text())
        cell.setdefault("name", Path(path).stem)
        if cell.get("family", "").startswith("bfp"):
            rows.extend(bfp_rows.run_cell(cell, driver, out_root, args.k, run_driver))
            continue
        for tier in cell.get("tiers", ["kernel", "layer"]):
            rows.extend(run_row(cell, tier, driver, out_root, args.k, gold_header))

    doc_gaps = {"sym_ref": sym_ref.DOC_GAPS, "bfp_ref": bfp_ref.DOC_GAPS}
    summary = {
        "n_rows": len(rows),
        "n_pass": sum(r["verdict"] == "PASS" for r in rows),
        "n_fail": sum(r["verdict"] == "FAIL" for r in rows),
        "n_error": sum(r["verdict"] == "ERROR" for r in rows),
        "k_ulp32": args.k,
    }
    (out_root / "report.json").write_text(
        json.dumps({"summary": summary, "rows": rows, "doc_gaps": doc_gaps}, indent=1, default=str)
    )
    md = ["# Differential numerics report", "", HEADER]
    md += [fmt_row(r) for r in rows]
    md += [
        "",
        f"rows: {summary['n_rows']}  PASS: {summary['n_pass']}  FAIL: {summary['n_fail']}  "
        f"ERROR: {summary['n_error']}  (tie window k = {args.k} ulp32)",
        "",
        FOOTER,
        "",
        "## Documentation gaps hit by Leg A (ref/sym_ref.py DOC_GAPS)",
        "",
    ]
    md += [f"- {g}" for g in sym_ref.DOC_GAPS]
    md += ["", "## Documentation gaps hit by Leg A (ref/bfp_ref.py DOC_GAPS)", ""]
    md += [f"- {g}" for g in bfp_ref.DOC_GAPS]
    (out_root / "report.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[:3 + len(rows) + 2]))
    return 0 if summary["n_fail"] == 0 and summary["n_error"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
