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
     census (ref/census.py) classifies every mismatch (tie or not);
  4. Leg B (ref/exact_ref.py) computes the float64 exact result on the
     dequantized operands and checks the analytic bound;
  5. optionally cross-checks the layer-tier output against the existing
     PyTorch gold header (calibration aid for the cell-0 fixtures).
Writes out/report.md and out/report.json. Exit status 1 if any row FAILs.
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))

from ref import census, exact_ref, sym_ref  # noqa: E402
from ref.sym_ref import ConvGeometry  # noqa: E402

MAX_SEED_TIE_CHANNELS = 12  # 2^12 Leg A candidates at most per row


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
# Leg A with bias-seed tie candidates                                        #
# --------------------------------------------------------------------------- #


def leg_a_candidates(fx: Fixture, geom: ConvGeometry, tier: str, out_bits: int, k: int) -> dict:
    """Leg A result plus the alternatives a bias-seed float32 tie allows.

    The seed is rounded once per output channel in C; when its pre-rounding
    value is within k ulp32 of a .5 boundary, C may legitimately have rounded
    the other way, shifting that whole channel by one raw LSB. Candidates
    enumerate those per-channel alternatives (base candidate first)."""
    raw = sym_ref.conv1d_sym_forward_raw(
        fx.codes["input"],
        fx.scales["input"],
        fx.codes["weight"],
        fx.scales["weight"],
        fx.codes.get("bias"),
        fx.scales.get("bias"),
        geom,
    )
    tie_channels: list[int] = []
    alt_seed = None
    if raw.seed is not None:
        ties = census.tie_mask(raw.seed.pre, k)
        alt_seed = census.alt_rounding(raw.seed.pre, raw.seed.seed)
        tie_channels = [int(i) for i in np.flatnonzero(ties)]
        if len(tie_channels) > MAX_SEED_TIE_CHANNELS:
            raise RuntimeError(
                f"{len(tie_channels)} bias-seed tie channels; narrow the fixture "
                f"(cap {MAX_SEED_TIE_CHANNELS})"
            )

    candidates = []
    for flips in itertools.product((False, True), repeat=len(tie_channels)):
        seed = None if raw.seed is None else raw.seed.seed.copy()
        flipped = [ch for ch, f in zip(tie_channels, flips) if f]
        if seed is not None:
            for ch in flipped:
                seed[ch] = alt_seed[ch]
        raw_codes = raw.mac if seed is None else raw.mac + seed.reshape(1, -1, 1)
        if tier == "kernel":
            cand = {
                "codes": raw_codes,
                "pre": raw_codes.astype(np.float64),  # integers: no restore ties
                "scale": raw.s_raw,
            }
        else:
            restored = sym_ref.restore_sym(raw_codes, raw.s_raw, out_bits)
            cand = {"codes": restored.codes, "pre": restored.pre, "scale": restored.scale}
        cand["seed_flips"] = flipped
        cand["seed"] = seed
        candidates.append(cand)
    seed_f32 = None
    if raw.seed is not None and tier == "kernel":  # the seed is observable only on the raw wire
        seed_f32 = sym_ref.bias_seed_f32_literal(
            fx.codes["bias"], fx.scales["bias"], fx.scales["input"], fx.scales["weight"]
        )
    return {
        "candidates": candidates,
        "s_raw": raw.s_raw,
        "tie_channels": tie_channels,
        "seed_pre": None if raw.seed is None else raw.seed.pre.tolist(),
        "seed_f32_literal": seed_f32,
    }


def evaluate_leg_a(c_codes: np.ndarray, c_scale: float, leg_a: dict, k: int) -> dict:
    best = None
    for cand in leg_a["candidates"]:
        if cand["codes"].shape != c_codes.shape:
            raise RuntimeError(f"shape mismatch C {c_codes.shape} vs ref {cand['codes'].shape}")
        res = census.census(c_codes, cand["codes"], cand["pre"], k)
        non_tie = res.n_mismatch - res.n_tie
        key = (non_tie, res.n_mismatch, len(cand["seed_flips"]))
        if best is None or key < best[0]:
            best = (key, cand, res)
    _key, cand, res = best
    taxonomy = res.taxonomy
    if cand["seed_flips"] and taxonomy in ("EXACT", "TIE_ONLY"):
        taxonomy = "TIE_ONLY"  # exact only up to a bias-seed tie
    scale = census.scale_check(c_scale, cand["scale"])
    # Informational (never gating): does the seed the best candidate needed
    # equal the doc expression evaluated left-to-right in float32?
    seed_f32 = None
    if cand.get("seed") is not None and leg_a["seed_f32_literal"] is not None:
        agree = cand["seed"] == leg_a["seed_f32_literal"]
        seed_f32 = {"channels_agree": int(agree.sum()), "channels": int(agree.size)}
    return {
        "taxonomy": taxonomy,
        "census": res.to_json(),
        "seed_flips_used": cand["seed_flips"],
        "seed_tie_channels": leg_a["tie_channels"],
        "seed_f32_literal": seed_f32,
        "scale": scale,
        "ref_scale": cand["scale"],
    }


# --------------------------------------------------------------------------- #
# Leg B                                                                       #
# --------------------------------------------------------------------------- #


def evaluate_leg_b(
    fx: Fixture, geom: ConvGeometry, tier: str, c_codes: np.ndarray, c_scale: float, s_raw: float
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
    bound = exact_ref.bound_conv1d_forward(
        tier, c_scale if tier == "layer" else None, s_raw, fx.has_bias, y_absmax
    )
    out = census.leg_b_check(err_max, bound)
    out["y_absmax"] = y_absmax
    out["s_out"] = c_scale if tier == "layer" else None
    out["s_raw"] = s_raw
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
    try:
        fx = build_fixture(cell)
        work = out_root / "runs" / f"{name}__{tier}"
        spec_path = write_inputs(work / "in", fx, cell, tier)
        drv = run_driver(driver, spec_path, work / "in", work / "out")
        leg_a = leg_a_candidates(fx, geom, tier, int(cell.get("outputBits", 12)), k)
        a = evaluate_leg_a(drv["codes"], drv["scale"], leg_a, k)
        b = evaluate_leg_b(fx, geom, tier, drv["codes"], drv["scale"], leg_a["s_raw"])
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
        return f"| {r['cell']} | {r['tier']} | ERROR | | | | | | | | | | | {r['error'].splitlines()[0]} |"
    a, b = r["legA"], r["legB"]
    c = a["census"]
    gold = r.get("gold")
    gold_s = "-" if gold is None else gold["status"]
    sf = a.get("seed_f32_literal")
    sf_s = "-" if sf is None else f"{sf['channels_agree']}/{sf['channels']}"
    return (
        f"| {r['cell']} | {r['tier']} | {a['taxonomy']} | {c['n_mismatch']}/{c['n_elements']} | "
        f"{c['n_tie']} | {len(a['seed_flips_used'])}/{len(a['seed_tie_channels'])} | {sf_s} | "
        f"{c['max_abs_diff']} | {a['scale']['rel']:.2e} | {b['taxonomy']} | "
        f"{b['err_max']:.3e} | {b['bound']:.3e} ({b['ratio']:.2f}) | {gold_s} | {r['verdict']} |"
    )


HEADER = (
    "| cell | tier | Leg A | mismatches | ties (restore) | seed flips/ties | seed f32-literal | "
    "max code diff | scale rel | Leg B | max err | bound (ratio) | gold | verdict |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
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
        "Leg A taxonomy: EXACT | TIE_ONLY | MISMATCH; Leg B: BOUNDED | EXCESS. "
        "`mismatches` = differing codes / total; `ties (restore)` = those within k ulp32 of a .5 "
        "boundary of the restore; `seed flips/ties` = bias-seed channels where the other "
        "rounding of a float32-tied seed was needed / number of such channels; "
        "`seed f32-literal` (informational, never gating) = channels whose seed equals the doc "
        "expression evaluated left-to-right in float32 / channels. Note: at |seed| >= 2^21 one "
        "ulp32 is >= 0.25, so every seed counts as a tie and only this column discriminates.",
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
