#!/usr/bin/env python3
"""Run every differential cell: build operands, drive the C, run both legs, report.

    uv run test/differential/run_cells.py [--build-dir build/differential]
                                          [--cells 'test/differential/cells/*.json']
                                          [--out test/differential/out] [--gold auto|skip|PATH]

For each cell JSON and each of its tiers:
  1. build the operand CODES + float32 scales from the cell's `fixture`
     (explicit floats or torch draws encoded with the documented rule, or
     random codes with random non-power-of-two scales);
  2. write them (.npy + spec.json) and run `odt_diffdrive`;
  3. Leg A (ref/sym_ref.py, docs-only) recomputes the codes + scale; the
     bias seed is resolved per output channel by the seed policy
     (ref/census.py::seed_policy) and the census classifies every remaining
     mismatch -- exactly on the kernel tier, +-1 at restore ties on the
     layer tier;
  4. Leg B (ref/exact_ref.py) computes the float64 exact result on the
     dequantized operands and checks the analytic bound;
  5. optionally cross-checks the layer-tier output against the existing
     PyTorch gold header (calibration aid for the cell-0 fixtures).
Writes out/report.md and out/report.json. Exit status 1 if any row FAILs.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))

from ref import census, exact_ref, sym_ref  # noqa: E402
from ref.census import SEED_REF, ChannelSeed  # noqa: E402
from ref.sym_ref import ConvGeometry  # noqa: E402


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


class Fixture:
    def __init__(self, bits: int):
        self.bits = bits
        self.codes: dict[str, np.ndarray] = {}
        self.scales: dict[str, float] = {}  # float32 values as Python floats
        self.notes: list[str] = []

    def add_encoded(self, name: str, values: np.ndarray) -> None:
        enc = sym_ref.encode_sym(np.asarray(values, dtype=np.float64), self.bits)
        self.codes[name] = enc.codes
        self.scales[name] = f32(enc.scale)

    def add_codes(self, name: str, codes: np.ndarray, scale: float) -> None:
        self.codes[name] = np.asarray(codes, dtype=np.int64)
        self.scales[name] = f32(scale)

    @property
    def has_bias(self) -> bool:
        return "bias" in self.codes


DEFAULT_SCALE_RANGES = {  # value absmax ranges, divided by qMax below
    "input": (0.5, 4.0),
    "weight": (0.05, 0.5),
    "bias": (0.02, 0.3),
}


def build_fixture(cell: dict) -> Fixture:
    fx_spec = cell["fixture"]
    bits = int(cell.get("operandBits", 12))
    fx = Fixture(bits)
    kind = fx_spec["type"]
    if kind == "float_values":
        for name in ("input", "weight", "bias"):
            if fx_spec.get(name) is not None:
                fx.add_encoded(name, np.asarray(fx_spec[name], dtype=np.float64))
    elif kind == "torch_randn":
        import torch  # fixture reproduction only; Leg A never sees torch values

        torch.manual_seed(int(fx_spec["seed"]))
        for name, shape in fx_spec["draws"]:
            fx.add_encoded(name, torch.randn(*shape).numpy().astype(np.float64))
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
            absmax = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
            scale = f32(absmax / qmax)
            if is_power_of_two(scale):
                scale = f32(scale * 1.0000001)
            fx.add_codes(name, codes, scale)
    else:
        raise ValueError(f"{cell['name']}: unknown fixture type {kind!r}")

    for name in ("input", "weight"):
        if name not in fx.codes:
            raise ValueError(f"{cell['name']}: fixture lacks operand {name}")
    for name, scale in fx.scales.items():
        if is_power_of_two(scale):
            fx.notes.append(f"{name} scale {scale!r} is a power of two")
    geom = ConvGeometry.from_spec(cell["geometry"])
    c_in = fx.codes["input"].shape[1]
    reduction = (c_in // geom.groups) * geom.kernel_size
    if reduction > 511:
        raise ValueError(
            f"{cell['name']}: reduction length {reduction} > 511 (int12 soundness bound)"
        )
    return fx


# --------------------------------------------------------------------------- #
# Driver I/O                                                                  #
# --------------------------------------------------------------------------- #


def write_inputs(in_dir: Path, fx: Fixture, cell: dict, tier: str) -> Path:
    in_dir.mkdir(parents=True, exist_ok=True)
    operands = {}
    for name, codes in fx.codes.items():
        np.save(in_dir / f"{name}.npy", np.ascontiguousarray(codes, dtype=np.int32))
        operands[name] = {
            "file": f"{name}.npy",
            "dtype": "SYM_INT32",
            "scale": hex32(fx.scales[name]),
            "qMaxBits": fx.bits,
            "rounding": cell["arith"].get("rounding", "HALF_AWAY"),
        }
    spec = {
        "op": cell["op"],
        "tier": tier,
        "direction": cell["direction"],
        "arith": cell["arith"],
        "geometry": cell["geometry"],
        "operands": operands,
        "output": {
            "dtype": "SYM_INT32",
            "qMaxBits": int(cell.get("outputBits", 12)),
            "rounding": cell["arith"].get("rounding", "HALF_AWAY"),
        },
    }
    spec_path = in_dir / "spec.json"
    spec_path.write_text(json.dumps(spec, indent=1))
    return spec_path


def run_driver(driver: Path, spec_path: Path, in_dir: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [str(driver), str(spec_path), str(in_dir), str(out_dir)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"driver exit {proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    result = json.loads((out_dir / "result.json").read_text())
    out = result["outputs"]["output"]
    codes = np.load(out_dir / out["file"]).astype(np.int64)
    scale = float.fromhex(out["scale_hex"]) if "scale_hex" in out else None
    return {"codes": codes, "scale": scale, "meta": out, "stdout": proc.stdout}


# --------------------------------------------------------------------------- #
# Leg A: raw wire, seed policy, per-channel resolution                       #
# --------------------------------------------------------------------------- #


@dataclass
class LegAInputs:
    raw: sym_ref.RawForward  # mac, s_raw, real-valued seed (or None)
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


def leg_a_inputs(fx: Fixture, geom: ConvGeometry, k: int) -> LegAInputs:
    raw = sym_ref.conv1d_sym_forward_raw(
        fx.codes["input"],
        fx.scales["input"],
        fx.codes["weight"],
        fx.scales["weight"],
        fx.codes.get("bias"),
        fx.scales.get("bias"),
        geom,
    )
    policy = None
    f32_seed = None
    if raw.seed is not None:
        f32_seed = sym_ref.bias_seed_f32_literal(
            fx.codes["bias"], fx.scales["bias"], fx.scales["input"], fx.scales["weight"]
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
    c_codes: np.ndarray, c_scale: float, inp: LegAInputs, out_bits: int, k: int
) -> dict:
    """Layer tier: the restore couples the channels only through the fresh
    scale absmax(raw * s_raw) / qMax. Every assignment of allowed seeds has
    an absmax that is some channel's maximum under one of its allowed seeds,
    so the realizable absmax values are enumerated (at most 2 * Cout), the
    observed C scale selects those within 2^-22 relative (none: scale
    MISMATCH, the closest is still evaluated for the report), and under each
    selected scale the channels are resolved independently -- linear in
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
    matching = [v for v in cands if rels[v] <= census.SCALE_REL_TOL]
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
    scale = census.scale_check(c_scale, ref_scale)
    scale["ok"] = bool(matching)
    return {
        "taxonomy": taxonomy,
        "census": res.to_json(),
        "seed": seed_summary(inp, chosen),
        "scale": scale,
        "ref_scale": ref_scale,
        "scale_candidates": {"realizable": len(cands), "matching": len(matching), "used_absmax": v_best},
    }


# --------------------------------------------------------------------------- #
# Leg B                                                                       #
# --------------------------------------------------------------------------- #


def evaluate_leg_b(
    fx: Fixture,
    geom: ConvGeometry,
    tier: str,
    c_codes: np.ndarray,
    c_scale: float,
    inp: LegAInputs,
    out_bits: int,
    k: int,
) -> dict:
    length = fx.codes["input"].shape[2]
    pad_left, pad_right = geom.pads(length)
    exact = exact_ref.conv1d_exact(
        exact_ref.dequant(fx.codes["input"], fx.scales["input"]),
        exact_ref.dequant(fx.codes["weight"], fx.scales["weight"]),
        None if not fx.has_bias else exact_ref.dequant(fx.codes["bias"], fx.scales["bias"]),
        geom.stride,
        geom.dilation,
        geom.groups,
        pad_left,
        pad_right,
    )
    if exact.shape != c_codes.shape:
        raise RuntimeError(f"Leg B shape {exact.shape} vs C {c_codes.shape}")
    deq_c = exact_ref.dequant(c_codes, c_scale)
    err_max = float(np.max(np.abs(deq_c - exact))) if exact.size else 0.0
    y_absmax = float(np.max(np.abs(exact))) if exact.size else 0.0
    seed_granted = census.seed_granted_lsb(inp.policy)
    bound = exact_ref.bound_conv1d_forward(
        tier,
        c_scale if tier == "layer" else None,
        inp.raw.s_raw,
        seed_granted,
        y_absmax,
        sym_ref.qmax(out_bits),
        k,
    )
    out = census.leg_b_check(err_max, bound)
    out["y_absmax"] = y_absmax
    out["s_out"] = c_scale if tier == "layer" else None
    out["s_raw"] = inp.raw.s_raw
    out["seed_granted_lsb"] = seed_granted
    out["restore_slack_codes"] = exact_ref.restore_slack_codes(sym_ref.qmax(out_bits), k) if tier == "layer" else None
    return out


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


def run_row(cell: dict, tier: str, driver: Path, out_root: Path, k: int, gold_header) -> dict:
    name = cell["name"]
    row = {"cell": name, "tier": tier, "op": cell["op"], "direction": cell["direction"]}
    geom = ConvGeometry.from_spec(cell["geometry"])
    out_bits = int(cell.get("outputBits", 12))
    try:
        fx = build_fixture(cell)
        work = out_root / "runs" / f"{name}__{tier}"
        spec_path = write_inputs(work / "in", fx, cell, tier)
        drv = run_driver(driver, spec_path, work / "in", work / "out")
        inp = leg_a_inputs(fx, geom, k)
        if tier == "kernel":
            a = resolve_leg_a_kernel(drv["codes"], drv["scale"], inp, k)
        elif tier == "layer":
            a = resolve_leg_a_layer(drv["codes"], drv["scale"], inp, out_bits, k)
        else:
            raise ValueError(f"unknown tier {tier!r}")
        b = evaluate_leg_b(fx, geom, tier, drv["codes"], drv["scale"], inp, out_bits, k)
        verdict = census.combine(a["taxonomy"], a["scale"]["ok"], b["taxonomy"])
        expect = cell.get("expect", {})
        if expect:
            if a["taxonomy"] not in expect.get("legA", [a["taxonomy"]]):
                verdict = "FAIL"
            if b["taxonomy"] != expect.get("legB", b["taxonomy"]):
                verdict = "FAIL"
        gold = None
        if tier == "layer":
            gold = gold_check(gold_header, cell.get("gold"), drv["codes"], drv["scale"])
        row.update(
            {
                "verdict": verdict,
                "legA": a,
                "legB": b,
                "gold": gold,
                "fixture": {
                    "type": cell["fixture"]["type"],
                    "shapes": {n: list(c.shape) for n, c in fx.codes.items()},
                    "scales": fx.scales,
                    "notes": fx.notes,
                },
                "c_scale": drv["scale"],
                "n_elements": int(drv["codes"].size),
            }
        )
    except Exception as exc:  # noqa: BLE001 -- every row must land in the report
        row.update({"verdict": "ERROR", "error": str(exc)})
    return row


def fmt_row(r: dict) -> str:
    if r["verdict"] == "ERROR":
        return f"| {r['cell']} | {r['tier']} | ERROR | | | | | | | | | | | | {r['error'].splitlines()[0]} |"
    a, b = r["legA"], r["legB"]
    c = a["census"]
    gold = r.get("gold")
    gold_s = "-" if gold is None else gold["status"]
    seed = a.get("seed")
    if seed is None:
        seed_s, dev_s, f32_s = "-", "-", "-"
    else:
        n = seed["n_channels"]
        n_ref = n - seed["n_alt"] - seed["n_f32_gated"]
        seed_s = f"{n_ref}/{seed['n_alt']}/{seed['n_f32_gated']}"
        dev_s = str(seed["max_deviation_lsb"])
        ag = seed["f32_agrees_with_ref"]
        f32_s = f"{ag['channels_agree']}/{ag['channels']}"
    cls = a.get("mismatch_class", "-")
    return (
        f"| {r['cell']} | {r['tier']} | {a['taxonomy']} | {c['n_mismatch']}/{c['n_elements']} | "
        f"{c['n_tie']} | {cls} | {seed_s} | {dev_s} | {f32_s} | "
        f"{c['max_abs_diff']} | {a['scale']['rel']:.2e} | {b['taxonomy']} | "
        f"{b['err_max']:.3e} | {b['bound']:.3e} ({b['ratio']:.2f}) | {gold_s} | {r['verdict']} |"
    )


HEADER = (
    "| cell | tier | Leg A | mismatches | ties (restore) | class | seed ref/alt/f32 | seed dev | "
    "f32=real | max code diff | scale rel | Leg B | max err | bound (ratio) | gold | verdict |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
)

FOOTER = (
    "Leg A taxonomy: EXACT | TIE_ONLY | MISMATCH; Leg B: BOUNDED | EXCESS. "
    "`mismatches` = differing codes / total after the seed policy; `ties (restore)` = those "
    "within k ulp32 of a .5 boundary of the restore (layer tier only; the kernel tier compares "
    "raw int32 exactly); `class` (kernel tier) = seed (a constant per-channel residual that is "
    "not an allowed seed) / mac (a non-constant residual) / none; `seed ref/alt/f32` = bias "
    "channels resolved to the real-valued rounding / the other side of a float32-tied .5 "
    "boundary / the float32 left-to-right reading (gating only where k*ulp32(|seed|) >= 0.5, "
    "i.e. |seed| >= 2^20 at k = 4 -- an interpretation, see README); `seed dev` = largest "
    "|used - real-valued rounding| in raw LSB (TIE_ONLY on an f32-gated channel can exceed 1); "
    "`f32=real` (informational) = channels whose float32 reading equals the real-valued "
    "rounding / channels."
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
        for tier in cell.get("tiers", ["kernel", "layer"]):
            rows.append(run_row(cell, tier, driver, out_root, args.k, gold_header))

    doc_gaps = {"sym_ref": sym_ref.DOC_GAPS}
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
    (out_root / "report.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[:3 + len(rows) + 2]))
    return 0 if summary["n_fail"] == 0 and summary["n_error"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
