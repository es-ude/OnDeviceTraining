"""BFP cell families of the differential harness (design step 3): fixtures,
driver I/O and row evaluation for

- ``bfp_pack``  B1: FLOAT32 -> BFP pack on the convert tier (exponent rule,
                D6 edges, F2 boundary census);
- ``bfp_gemm``  B2/B3/B4: Linear / Conv1d / ConvT1d forward, dx, weightGrad
                (+ biasGrad) under ARITH_BFP, kernel and layer tier, raw
                FLOAT32 output or a packed BFP wire;
- ``bfp_fq``    B5: ARITH_FLOAT32 over BFP storage against the FLOAT32 layer
                run on the dequantized operands, bit for bit.

Leg A is ``ref/bfp_ref.py`` (docs only). Leg B is ``ref/exact_ref.py``
(float64 on the dequantized operands) with the design's walk-order-free fold
bound N * 2^-24 * (sum|a_i b_i| + |bias|). Rows use the common taxonomy
(README): EXACT | TIE_ONLY | MISMATCH (+ ABORT for a predicted headroom
fail-fast) and BOUNDED | EXCESS.

Single-segment reductions (every output element folds ONE same-exponent
segment) are predicted to the bit by the documented fold and gate EXACT /
MISMATCH; multi-segment reductions have no documented fold order (F10,
``bfp_ref.DOC_GAPS[4]``) -- the documented algorithm along the assumed walk
is evaluated and reported (bit-exact or not, as `seq_fold`), the row gates on
Leg B only, and a difference inside the bound is TIE_ONLY (class
`fold_order`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ref import bfp_ref, census, exact_ref
from ref.sym_ref import ConvGeometry

F32_U = 2.0**-24


# --------------------------------------------------------------------------- #
# Operands                                                                    #
# --------------------------------------------------------------------------- #


@dataclass
class BfpOperand:
    """One BFP-gridded operand: integer codes on a per-group 2^E grid. The
    layer tier stores it as declared (`storage` BFP = packed, FLOAT32 = the
    exact dequantized float32 values, grid-exact by construction so the
    funnel's staging has a deterministic answer); the kernel tier always
    receives the codes + exponents (the unpacked scratch form)."""

    storage: str  # "BFP" | "FLOAT32"
    codes: np.ndarray  # int64, shaped
    stored: np.ndarray  # int64 [numGroups]
    m: int
    e: int
    group_size: int

    @property
    def n(self) -> int:
        return int(self.codes.size)

    @property
    def bits(self) -> tuple[int, int, int]:
        return (self.m, self.e, self.group_size)

    def deq(self) -> np.ndarray:
        return bfp_ref.dequant(self.codes, self.stored, self.e, self.group_size).reshape(self.codes.shape)

    def values32(self) -> np.ndarray:
        v = self.deq().astype(np.float32)
        if not np.array_equal(v.astype(np.float64), self.deq()):
            raise ValueError("BFP operand is not exactly representable in float32")
        return v

    def spec(self, name: str, in_dir: Path, rounding: str, storage: str | None = None) -> dict:
        st = storage or self.storage
        if st == "FLOAT32":
            np.save(in_dir / f"{name}.npy", np.ascontiguousarray(self.values32(), dtype=np.float32))
            return {"file": f"{name}.npy", "dtype": "FLOAT32"}
        np.save(in_dir / f"{name}.npy", np.ascontiguousarray(self.codes, dtype=np.int32))
        return {
            "file": f"{name}.npy",
            "dtype": "BFP",
            "mantissaBits": self.m,
            "exponentBits": self.e,
            "groupSize": self.group_size,
            "exponents": [int(s) for s in self.stored],
            "rounding": rounding,
        }

    def describe(self) -> dict:
        return {
            "storage": self.storage,
            "shape": list(self.codes.shape),
            "m": self.m,
            "e": self.e,
            "groupSize": self.group_size,
            "numGroups": int(self.stored.size),
            "E_range": [int(self.stored.min() - bfp_ref.exponent_bias(self.e)), int(self.stored.max() - bfp_ref.exponent_bias(self.e))],
        }


def random_bfp_operand(rng: np.random.Generator, shape: tuple, spec: dict) -> BfpOperand:
    """Random codes on the grid with +-qMax present in every group (the
    absmax maps onto qMax, as a real pack would leave it), stored exponents
    around the one the exact rule gives for `absmax` (value units), spread
    by +-`expSpread` per group and clamped into the D6 range. `codes:
    "full_scale"` sets every code to +qMax (the headroom / section-7
    probes); `codes: "alternating"` to +-qMax."""
    m, e, gs = int(spec["m"]), int(spec["e"]), int(spec.get("groupSize", 0))
    n = int(np.prod(shape))
    q = bfp_ref.qmax(m)
    ng = bfp_ref.num_groups(n, gs)
    gs_eff = gs if (gs and gs < n) else 0
    kind = spec.get("codes", "random")
    if kind == "full_scale":
        codes = np.full(n, q, dtype=np.int64)
    elif kind == "alternating":
        codes = np.where(np.arange(n) % 2 == 0, q, -q).astype(np.int64)
    else:
        codes = rng.integers(-q, q + 1, size=n, dtype=np.int64)
        g_of = bfp_ref.group_ids(n, gs_eff)
        for g in range(ng):
            idx = np.flatnonzero(g_of == g)
            j = idx[int(rng.integers(0, idx.size))]
            codes[j] = q if rng.integers(0, 2) else -q
    e_center, _ = bfp_ref.exponent_rule_exact(float(spec["absmax"]), m)
    spread = int(spec.get("expSpread", 0))
    bias = bfp_ref.exponent_bias(e)
    cap = bfp_ref.stored_exponent_cap(e)
    stored = bias + e_center + rng.integers(-spread, spread + 1, size=ng)
    stored = np.clip(stored, 0, cap).astype(np.int64)
    return BfpOperand(spec.get("storage", "BFP"), codes.reshape(shape), stored, m, e, gs_eff)


# --------------------------------------------------------------------------- #
# Driver I/O                                                                  #
# --------------------------------------------------------------------------- #


def write_spec(in_dir: Path, cell: dict, tier: str, operands: dict[str, dict], output: dict, extra: dict | None = None) -> Path:
    in_dir.mkdir(parents=True, exist_ok=True)
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
    if extra:
        spec.update(extra)
    spec_path = in_dir / "spec.json"
    spec_path.write_text(json.dumps(spec, indent=1))
    return spec_path


def output_spec(cell: dict) -> dict:
    """spec.output from the cell: FLOAT32 (raw) or a BFP wire."""
    o = cell.get("output", {"dtype": "FLOAT32"})
    if o["dtype"] == "FLOAT32":
        return {"dtype": "FLOAT32"}
    return {
        "dtype": "BFP",
        "mantissaBits": int(o["m"]),
        "exponentBits": int(o["e"]),
        "groupSize": int(o.get("groupSize", 0)),
        "rounding": cell["arith"].get("rounding", "HALF_AWAY"),
    }


def c_output_image(o: dict) -> tuple[np.ndarray, dict]:
    """The C output as float64 values plus its grid (empty for FLOAT32)."""
    if o["dtype"] == "FLOAT32":
        return o["values"].astype(np.float64), {}
    meta = o["meta"]
    stored = np.asarray(meta["exponents"], dtype=np.int64)
    gs = int(meta["groupSize"])
    e = int(meta["exponentBits"])
    deq = bfp_ref.dequant(o["codes"], stored, e, gs).reshape(o["codes"].shape)
    return deq, {"stored": stored, "m": int(meta["mantissaBits"]), "e": e, "groupSize": gs, "codes": o["codes"]}


# --------------------------------------------------------------------------- #
# Shared evaluators                                                           #
# --------------------------------------------------------------------------- #


def float_bitexact(c: np.ndarray, pred: np.ndarray) -> dict:
    """Bit comparison of a C float32 output against a float32 prediction."""
    c32 = np.asarray(c, dtype=np.float64).ravel()
    p32 = np.asarray(pred, dtype=np.float64).ravel()
    diff = c32 - p32
    bad = np.flatnonzero(diff != 0)
    ulp = census.ulp32(np.maximum(np.abs(p32), np.finfo(np.float32).tiny))
    return {
        "n_elements": int(c32.size),
        "n_mismatch": int(bad.size),
        "max_abs_diff": float(np.abs(diff).max()) if diff.size else 0.0,
        "max_ulp32": float((np.abs(diff) / ulp).max()) if diff.size else 0.0,
        "worst": [
            {"index": int(i), "c": float(c32[i]), "pred": float(p32[i]), "diff_ulp32": float(abs(diff[i]) / ulp[i])}
            for i in bad[np.argsort(-np.abs(diff[bad]))][:8]
        ],
    }


def compare_packed(
    values: np.ndarray, c_codes: np.ndarray, c_stored: np.ndarray, m: int, e: int, gs: int, k: int, code_slack: np.ndarray | None = None
) -> dict:
    """Leg A for a pack of `values` (float32 image) onto a (m, e, gs) grid
    against the C codes + exponents. Per group: E must equal the exact
    section-3 rule; an E off by one is accepted only in the F2 window (the
    exact quotient absmax / qMax within k ulp32 of the power of two) and
    only if the codes are then self-consistent under the C's E. Codes: equal
    under the (accepted) E, else MISMATCH -- except the documented clamp
    inconsistency (C -2^(m-1) vs Leg A -qMax below -(qMax + 0.5), class
    `clamp_floor`) and, when `code_slack` (code units per element) is given,
    a +-1 whose pre-rounding value lies within that slack of a .5 boundary
    (class `fold_order`: the raw being packed was itself only bounded)."""
    ref = bfp_ref.pack_bfp(values, m, e, gs)
    c_codes = np.asarray(c_codes, dtype=np.int64).ravel()
    c_stored = np.asarray(c_stored, dtype=np.int64)
    q = bfp_ref.qmax(m)
    floor_code = bfp_ref.code_floor_twos_complement(m)
    if c_stored.shape != ref.stored.shape:
        raise RuntimeError(f"numGroups: C {c_stored.size} vs ref {ref.stored.size}")
    classes: set[str] = set()
    e_boundary = []
    e_bad = []
    regimes: dict[str, int] = {}
    for g, ge in enumerate(ref.groups):
        regimes[ge.regime] = regimes.get(ge.regime, 0) + 1
        if c_stored[g] == ref.stored[g]:
            continue
        legit = False
        if ge.regime == "normal" and ge.boundary_rel is not None:
            win = bfp_ref.boundary_window_rel(ge.absmax, k)
            # boundary_rel == 0 (absmax IS qMax * 2^E) is pinned by the doc's own
            # frexpf wording (frac == 0.5 -> E = e - 1): no window there.
            if c_stored[g] == ref.stored[g] + 1 and -win < ge.boundary_rel < 0.0:
                legit = True  # quotient just below 2^E: a float32 quotient can land on/above it
            if c_stored[g] == ref.stored[g] - 1 and ge.boundary_rel < -0.5 + win / 2:
                legit = True  # quotient just above 2^(E-1): a float32 quotient can land exactly on it
        entry = {"group": g, "c_E": int(c_stored[g]), "ref_E": int(ref.stored[g]), "absmax": ge.absmax, "boundary_rel": ge.boundary_rel, "regime": ge.regime}
        (e_boundary if legit else e_bad).append(entry)
    if e_bad:
        classes.add("E")
    if e_boundary:
        classes.add("E_boundary")
    # codes under the C's exponents (equal to ref where E agrees)
    under_c = bfp_ref.repack_under(values, c_stored, m, e, gs)
    pre = under_c.pre
    diff = c_codes - under_c.codes
    mism = np.flatnonzero(diff != 0)
    n_clamp_floor = 0
    n_fold = 0
    worst = []
    for i in mism:
        if c_codes[i] == floor_code and under_c.codes[i] == -q and pre[i] < -(q + 0.5):
            n_clamp_floor += 1
            continue
        if code_slack is not None and abs(diff[i]) == 1:
            frac = abs(pre[i]) - np.floor(abs(pre[i]))
            if abs(frac - 0.5) <= code_slack[i] and c_codes[i] == bfp_ref.round_half_away(np.array([pre[i]]))[0] + np.sign(diff[i]):
                n_fold += 1
                continue
        worst.append({"index": int(i), "c": int(c_codes[i]), "ref": int(under_c.codes[i]), "pre": float(pre[i])})
    worst.sort(key=lambda w: -abs(w["c"] - w["ref"]))
    if n_clamp_floor:
        classes.add("clamp_floor")
    if n_fold:
        classes.add("fold_order")
    if worst:
        classes.add("code")
    hard = bool(e_bad) or bool(worst)
    soft = bool(e_boundary) or n_clamp_floor > 0 or n_fold > 0
    taxonomy = "MISMATCH" if hard else ("TIE_ONLY" if soft else "EXACT")
    return {
        "taxonomy": taxonomy,
        "census": {
            "taxonomy": taxonomy,
            "mode": "bfp_pack",
            "n_elements": int(c_codes.size),
            "n_mismatch": int(mism.size),
            "n_tie": n_clamp_floor + n_fold,
            "max_abs_diff": int(np.abs(diff).max()) if diff.size else 0,
            "n_groups": int(ref.stored.size),
            "n_E_differs": len(e_boundary) + len(e_bad),
            "E_boundary": e_boundary[:8],
            "E_bad": e_bad[:8],
            "n_clamp_floor": n_clamp_floor,
            "n_clamped_predicted": int(ref.n_clamped),
            "regimes": regimes,
            "worst": worst[:8],
        },
        "mismatch_class": "+".join(sorted(classes)) if classes else "none",
        "seed": None,
        "scale": {"c": None, "ref": None, "rel": None, "ok": True},
        "ref_scale": None,
        "ref": ref,
        "under_c": under_c,
    }


def pack_roundtrip_leg_b(values: np.ndarray, c_codes: np.ndarray, c_stored: np.ndarray, m: int, e: int, gs: int) -> dict:
    """|x - deq(C)| <= 0.5 * 2^(E_C) for every element the grid can hold
    (|pre| <= qMax + 0.5); clamped elements are outside any round-trip bound
    by construction and reported separately."""
    x = np.asarray(values, dtype=np.float32).ravel().astype(np.float64)
    deq = bfp_ref.dequant(c_codes, c_stored, e, gs)
    bias = bfp_ref.exponent_bias(e)
    scale = 2.0 ** (np.asarray(c_stored, dtype=np.int64)[bfp_ref.group_ids(x.size, gs)] - bias)
    pre = x / scale
    clamped = np.abs(pre) > bfp_ref.qmax(m) + 0.5
    err = np.abs(x - deq)
    bound = 0.5 * scale
    out = elementwise_leg_b(err, bound, ~clamped)
    out["n_clamped_excluded"] = int(clamped.sum())
    out["err_max_clamped"] = float(err[clamped].max()) if clamped.any() else 0.0
    return out


def elementwise_leg_b(err: np.ndarray, bound: np.ndarray, mask: np.ndarray | None = None) -> dict:
    """Leg B with a per-element bound: BOUNDED iff err <= bound everywhere
    (inside `mask`); `err_max` / `bound` report the element with the largest
    err / bound ratio, so the two numbers printed belong together."""
    err = np.asarray(err, dtype=np.float64).ravel()
    bound = np.asarray(bound, dtype=np.float64).ravel()
    sel = np.ones(err.size, dtype=bool) if mask is None else np.asarray(mask, dtype=bool).ravel()
    if not sel.any():
        return {"err_max": 0.0, "bound": float(bound.min()) if bound.size else 0.0, "ratio": 0.0, "taxonomy": "BOUNDED"}
    ratios = np.where(sel, err / bound, -1.0)
    i = int(np.argmax(ratios))
    return {
        "err_max": float(err[i]),
        "bound": float(bound[i]),
        "ratio": float(ratios[i]),
        "taxonomy": "BOUNDED" if ratios[i] <= 1.0 else "EXCESS",
        "err_abs_max": float(err[sel].max()),
        "worst_index": i,
    }


# --------------------------------------------------------------------------- #
# B1: pack                                                                    #
# --------------------------------------------------------------------------- #


def build_pack_values(cell: dict, k: int) -> tuple[np.ndarray, list[str]]:
    """FLOAT32 values for a pack cell, group by group. Regimes (fixture
    `groups`, one entry per group, or `regime` for all): `random` (one
    log-uniform binade per group, a tie element, absmax away from any
    boundary), `boundary:j` (absmax = qMax * 2^k nudged by j ulp32, k random
    -- the F2 census), `ties` (every value an exact half-integer multiple of
    the group scale, absmax a .5 itself: HALF_AWAY vs half-even), `zero`,
    `saturate` (absmax beyond the D6 cap), `flush` (absmax below the D6 floor
    for this exponentBits), `negfloor` (the most negative value far below
    -qMax at the cap: the clamp-floor inconsistency)."""
    fx = cell["fixture"]
    rng = np.random.default_rng(int(fx["seed"]))
    m, e = int(cell["output"]["m"]), int(cell["output"]["e"])
    gs = int(cell["output"].get("groupSize", 0))
    n = int(fx["n"])
    gs_eff = gs if (gs and gs < n) else 0
    ng = bfp_ref.num_groups(n, gs_eff)
    size = n if gs_eff == 0 else gs_eff
    regimes = fx.get("groups") or [fx.get("regime", "random")] * ng
    if len(regimes) != ng:
        raise ValueError(f"{cell['name']}: {len(regimes)} group regimes for {ng} groups")
    q = bfp_ref.qmax(m)
    bias = bfp_ref.exponent_bias(e)
    cap = bfp_ref.stored_exponent_cap(e)
    e_lo, e_hi = -bias, cap - bias  # representable unbiased exponents
    vals = np.zeros(n, dtype=np.float32)
    notes = []
    for g, reg in enumerate(regimes):
        lo = g * size
        sl = slice(lo, lo + size)
        if reg == "zero":
            vals[sl] = 0.0
            continue
        fmax = float(np.finfo(np.float32).max)
        if reg == "saturate":
            # Beyond the D6 cap when float32 can hold it (exponentBits < 8;
            # at e = 8 the cap is 2^127 and every finite float32 fits for
            # m > 2, so the group degenerates to a top-binade normal group).
            top = min(q * 2.0 ** (e_hi + 2), 0.75 * fmax)
            x = rng.uniform(-1.0, 1.0, size=size) * top
            x[0] = top
            x[1 % size] = -top
            vals[sl] = x.astype(np.float32)
            continue
        if reg == "negfloor":
            top = min((q + 2) * 2.0 ** (e_hi + 1), 0.75 * fmax)
            x = rng.uniform(-1.0, 1.0, size=size) * min(q * 2.0**e_hi, 0.5 * fmax)
            x[0] = -top
            vals[sl] = x.astype(np.float32)
            continue
        if reg == "flush":
            ek = e_lo - 3
            x = rng.uniform(-1.0, 1.0, size=size) * q * 2.0**ek
            x[0] = 0.75 * 2.0**e_lo  # rounds to code 1 at the floor scale
            x[1 % size] = 0.4 * 2.0**e_lo  # rounds to 0 (flushed)
            vals[sl] = x.astype(np.float32)
            continue
        ek = int(rng.integers(max(e_lo + 1, -20), min(e_hi - 1, 20) + 1))
        scale = 2.0**ek
        if reg == "ties":
            halves = rng.integers(-q, q, size=size) + 0.5  # exact .5 multiples
            x = halves * scale
            x[0] = (q - 0.5) * scale  # absmax: E stays ek (q - 0.5 < q), rounds to q
            vals[sl] = x.astype(np.float32)
            continue
        if reg.startswith("boundary"):
            j = int(reg.split(":")[1]) if ":" in reg else 0
            a = np.float32(q * scale)
            for _ in range(abs(j)):
                a = np.nextafter(a, np.float32(np.inf) if j > 0 else np.float32(0.0))
            x = rng.uniform(-0.9, 0.9, size=size) * q * scale
            x[0] = float(a)
            vals[sl] = x.astype(np.float32)
            if g < 8:
                notes.append(f"group {g}: absmax = qMax*2^{ek} {j:+d} ulp32")
            continue
        # random: absmax strictly inside the binade (no boundary), one tie element
        x = rng.uniform(-0.95, 0.95, size=size) * q * scale
        x[0] = rng.uniform(0.6, 0.95) * q * scale * (1 if rng.integers(0, 2) else -1)
        x[1 % size] = (int(rng.integers(0, max(q // 2, 1))) + 0.5) * scale * 0.5  # a .5 at the next-finer scale: NOT a tie at E
        vals[sl] = x.astype(np.float32)
    return vals, notes


def run_pack_cell(cell: dict, driver: Path, out_root: Path, k: int, run_driver) -> list[dict]:
    name = cell["name"]
    base = {"cell": name, "tier": "convert", "op": cell["op"], "direction": cell["direction"]}
    try:
        vals, notes = build_pack_values(cell, k)
        m, e = int(cell["output"]["m"]), int(cell["output"]["e"])
        gs = int(cell["output"].get("groupSize", 0))
        gs_eff = gs if (gs and gs < vals.size) else 0
        work = out_root / "runs" / f"{name}__convert"
        in_dir = work / "in"
        in_dir.mkdir(parents=True, exist_ok=True)
        np.save(in_dir / "input.npy", np.ascontiguousarray(vals, dtype=np.float32))
        spec_path = write_spec(in_dir, cell, "convert", {"input": {"file": "input.npy", "dtype": "FLOAT32"}}, output_spec(cell))
        drv = run_driver(driver, spec_path, in_dir, work / "out")
        if drv.exit != 0:
            raise RuntimeError(f"driver exit {drv.exit}\n{drv.stderr}")
        o = drv.out("output")
        c_stored = np.asarray(o["meta"]["exponents"], dtype=np.int64)
        a = compare_packed(vals, o["codes"], c_stored, m, e, gs_eff, k)
        b = pack_roundtrip_leg_b(vals, o["codes"], c_stored, m, e, gs_eff)
        row = dict(base)
        row.update(
            {
                "output": "output",
                "legA": {kk: v for kk, v in a.items() if kk not in ("ref", "under_c")},
                "legB": b,
                "c_scale": None,
                "n_elements": int(vals.size),
                "bfp": {"summary": f"m{m} e{e} gs{gs_eff} groups={a['census']['n_groups']} E!={a['census']['n_E_differs']} regimes={a['census']['regimes']}"},
                "fixture": {"type": cell["fixture"]["type"], "shapes": {"input": [int(vals.size)]}, "storage": {"input": "FLOAT32"}, "scales": {}, "notes": notes},
            }
        )
        row["verdict"] = verdict(row, cell.get("expect", {}))
        return [row]
    except Exception as exc:  # noqa: BLE001
        row = dict(base)
        row.update({"verdict": "ERROR", "error": str(exc)})
        return [row]


# --------------------------------------------------------------------------- #
# B2/B3/B4: GEMM family under ARITH_BFP                                       #
# --------------------------------------------------------------------------- #


@dataclass
class GemmFixture:
    op: str
    ops: dict[str, BfpOperand]  # input, weight, [bias], [loss]
    geom: bfp_ref.ConvGeom | None
    geom_dict: dict
    notes: list[str]

    @property
    def shapes(self) -> dict:
        x = self.ops["input"].codes.shape
        w = self.ops["weight"].codes.shape
        return {"x": x, "w": w}


def conv_geoms(cell: dict, length: int) -> tuple[bfp_ref.ConvGeom, dict]:
    g = cell["geometry"]
    if cell["op"] == "conv1d":
        cg = ConvGeometry.from_spec(g)
        pl, pr = cg.pads(length)
        geom = bfp_ref.ConvGeom(cg.kernel_size, cg.stride, cg.dilation, cg.groups, pl, pr, 0)
        return geom, {"stride": cg.stride, "dilation": cg.dilation, "groups": cg.groups, "pad_left": pl, "pad_right": pr}
    geom = bfp_ref.ConvGeom(int(g["kernelSize"]), int(g.get("stride", 1)), int(g.get("dilation", 1)), int(g.get("groups", 1)), 0, 0, int(g.get("outputPadding", 0)))
    return geom, {"stride": geom.stride, "dilation": geom.dilation, "groups": geom.groups, "output_padding": geom.output_padding}


def build_gemm_fixture(cell: dict) -> GemmFixture:
    fx = cell["fixture"]
    rng = np.random.default_rng(int(fx["seed"]))
    op = cell["op"]
    ops: dict[str, BfpOperand] = {}
    notes: list[str] = []
    if op == "linear":
        b, in_f, out_f = int(fx["batch"]), int(fx["inF"]), int(fx["outF"])
        ops["input"] = random_bfp_operand(rng, (b, in_f), fx["input"])
        ops["weight"] = random_bfp_operand(rng, (out_f, in_f), fx["weight"])
        if fx.get("bias"):
            ops["bias"] = random_bfp_operand(rng, (out_f,), fx["bias"])
        if cell["direction"] != "forward":
            ops["loss"] = random_bfp_operand(rng, (b, out_f), fx["loss"])
        geom, gd = None, {}
    else:
        b, c_in, length, c_out, ksz = (int(fx[kk]) for kk in ("batch", "cin", "length", "cout", "kernelSize"))
        geom, gd = conv_geoms(cell, length)
        ops["input"] = random_bfp_operand(rng, (b, c_in, length), fx["input"])
        if op == "conv1d":
            ops["weight"] = random_bfp_operand(rng, (c_out, c_in // geom.groups, ksz), fx["weight"])
            l_out = geom.conv_out_len(length)
        else:
            ops["weight"] = random_bfp_operand(rng, (c_in, c_out // geom.groups, ksz), fx["weight"])
            l_out = geom.convt_out_len(length)
        if fx.get("bias"):
            ops["bias"] = random_bfp_operand(rng, (c_out,), fx["bias"])
        if cell["direction"] != "forward":
            ops["loss"] = random_bfp_operand(rng, (b, c_out, l_out), fx["loss"])
    for nm, o in ops.items():
        notes.append(f"{nm}: m{o.m} e{o.e} gs{o.group_size} E in {o.describe()['E_range']}")
    return GemmFixture(op, ops, geom, gd, notes)


def gemm_walk(fx: GemmFixture, direction: str) -> bfp_ref.Walk:
    x = fx.ops["input"].codes.shape
    w = fx.ops["weight"].codes.shape
    with_bias = "bias" in fx.ops
    if fx.op == "linear":
        return bfp_ref.linear_walk(x[0], x[1], w[0], direction, with_bias)
    c_out = w[0] if fx.op == "conv1d" else w[1] * fx.geom.groups
    fn = bfp_ref.conv1d_walk if fx.op == "conv1d" else bfp_ref.convt1d_walk
    return fn(x[0], x[1], x[2], c_out, fx.geom, direction, with_bias)


def gemm_operands(fx: GemmFixture, direction: str) -> tuple[BfpOperand, BfpOperand | None]:
    """(a, b) of the documented reduction per direction: forward (input,
    weight), dx (loss, weight), weightGrad (loss, input), biasGrad (loss, -)."""
    if direction == "forward":
        return fx.ops["input"], fx.ops["weight"]
    if direction == "dx":
        return fx.ops["loss"], fx.ops["weight"]
    if direction == "weightGrad":
        return fx.ops["loss"], fx.ops["input"]
    return fx.ops["loss"], None


def gemm_exact_and_abs(fx: GemmFixture, direction: str) -> tuple[np.ndarray, np.ndarray]:
    """Leg B float64 result and the absolute-sum oracle S = sum|a_i b_i| (+
    |bias|) for the direction, both shaped like the output."""
    x, w = fx.ops["input"].deq(), fx.ops["weight"].deq()
    b = fx.ops["bias"].deq() if "bias" in fx.ops else None
    if direction == "forward":
        exact = exact_ref.gemm_family_exact(fx.op, x, w, b, fx.geom_dict)
        s = exact_ref.gemm_family_exact(fx.op, np.abs(x), np.abs(w), None if b is None else np.abs(b), fx.geom_dict)
        return exact, s
    l = fx.ops["loss"].deq()
    g = exact_ref.gemm_family_backward_exact(fx.op, x, w, "bias" in fx.ops, l, fx.geom_dict)
    ga = exact_ref.gemm_family_backward_exact(fx.op, np.abs(x), np.abs(w), "bias" in fx.ops, np.abs(l), fx.geom_dict)
    return g[direction], ga[direction]


def gemm_headroom(fx: GemmFixture, direction: str, walk: bfp_ref.Walk) -> dict:
    a, b = gemm_operands(fx, direction)
    n = walk.reduction
    if b is None:
        ok = bfp_ref.sum_headroom_ok(a.group_size, n, a.m)
        return {"ok": ok, "kind": "sum", "limit": bfp_ref.sum_segment_limit(a.m), "max_segment": bfp_ref.max_segment(a.group_size, None, n), "reduction": n, "widths": [a.m]}
    ok = bfp_ref.headroom_ok(a.group_size, b.group_size, n, a.m, b.m)
    return {"ok": ok, "kind": "product", "limit": bfp_ref.segment_limit(a.m, b.m), "max_segment": bfp_ref.max_segment(a.group_size, b.group_size, n), "reduction": n, "widths": [a.m, b.m]}


def gemm_leg_a_fold(fx: GemmFixture, direction: str, walk: bfp_ref.Walk) -> bfp_ref.WalkFold:
    a, b = gemm_operands(fx, direction)
    bias_vals = None
    if direction == "forward" and "bias" in fx.ops:
        bias_vals = fx.ops["bias"].deq().ravel()[walk.bias_index]
    if b is None:
        return bfp_ref.fold_walk(a.codes, a.stored, a.bits, walk.ia, None, None, None, None, None)
    return bfp_ref.fold_walk(a.codes, a.stored, a.bits, walk.ia, b.codes, b.stored, b.bits, walk.ib, bias_vals)


def evaluate_gemm_output(fx: GemmFixture, direction: str, drv, out_name: str, k: int, with_bias_in_bound: bool) -> dict:
    """One output (forward output / propLoss / weightGrad / biasGrad): Leg A
    = headroom prediction + documented fold (bit prediction for single
    segments, assumed walk otherwise), Leg B = float64 within the design
    bound; a BFP wire output is additionally judged as a pack of the
    predicted raw (compare_packed with the fold slack)."""
    walk = gemm_walk(fx, direction)
    hr = gemm_headroom(fx, direction, walk)
    exact, s_abs = gemm_exact_and_abs(fx, direction)
    n_red = walk.reduction
    bound = exact_ref.bound_bfp_fold(n_red, s_abs)
    bound_rig = exact_ref.bound_bfp_fold_rigorous(n_red, s_abs, direction == "forward" and "bias" in fx.ops)
    row: dict = {"output": out_name, "n_elements": int(exact.size), "headroom": hr}
    if drv.exit != 0:
        kind = "ABORT" if not hr["ok"] else "MISMATCH"
        row["legA"] = {
            "taxonomy": kind,
            "census": None,
            "mismatch_class": "-" if kind == "ABORT" else "abort",
            "seed": None,
            "scale": {"c": None, "ref": None, "rel": None, "ok": kind == "ABORT"},
            "ref_scale": None,
        }
        row["legB"] = {"taxonomy": "n/a", "err_max": float("nan"), "bound": float("nan"), "ratio": float("nan")}
        row["c_scale"] = None
        row["bfp"] = {"summary": f"N={n_red} seg<={hr['max_segment']} limit={hr['limit']} -> {'ABORT (predicted)' if kind == 'ABORT' else 'ABORT (NOT predicted)'}"}
        return row
    fold = gemm_leg_a_fold(fx, direction, walk)
    pred = fold.predicted.reshape(exact.shape)
    if not np.allclose(fold.exact.reshape(exact.shape), exact, rtol=1e-12, atol=1e-30):
        raise RuntimeError("harness: the walk's exact sum disagrees with the float64 reference")
    o = drv.out(out_name)
    c_vals, grid = c_output_image(o)
    c_vals = c_vals.reshape(exact.shape)
    st = fold.stats
    if not hr["ok"]:
        # The guard should have fired (DOC_GAPS[6]); the run completed anyway.
        guard_class = "guard"
    else:
        guard_class = None
    if not grid:
        # Raw FLOAT32 output: bit comparison against the documented fold.
        bit = float_bitexact(c_vals, pred)
        within = np.abs(c_vals - pred) <= bound
        if bit["n_mismatch"] == 0:
            taxonomy = "EXACT"
            cls = "none"
        elif st.single_segment:
            taxonomy = "MISMATCH"
            cls = "fold"
        elif bool(within.all()):
            taxonomy = "TIE_ONLY"
            cls = "fold_order"
        else:
            taxonomy = "MISMATCH"
            cls = "fold"
        if guard_class:
            taxonomy = "MISMATCH"
            cls = cls + "+" + guard_class if cls != "none" else guard_class
        a = {
            "taxonomy": taxonomy,
            "census": {"taxonomy": taxonomy, "mode": "float32_bits", "n_elements": bit["n_elements"], "n_mismatch": bit["n_mismatch"], "n_tie": bit["n_mismatch"] if taxonomy == "TIE_ONLY" else 0, "max_abs_diff": bit["max_ulp32"], "worst": bit["worst"]},
            "mismatch_class": cls,
            "seed": None,
            "scale": {"c": None, "ref": None, "rel": None, "ok": True},
            "ref_scale": None,
            "seq_fold_bitexact": bit["n_mismatch"] == 0,
            "fold_stats": st.to_json(),
        }
        err = np.abs(c_vals - exact)
        b = elementwise_leg_b(err, bound)
        b["ratio_rigorous"] = float((err / bound_rig).max()) if err.size else 0.0
        b["bound_kind"] = "N*u*S (design)"
        row["c_scale"] = None
    else:
        # BFP wire: the OUT_WRITE pack of the raw; the raw is the predicted
        # fold (bit-exact for single segments, bounded otherwise -> slack).
        m, e, gs = grid["m"], grid["e"], grid["groupSize"]
        pred32 = pred.astype(np.float32)
        slack_codes = None
        if not st.single_segment:
            bias_e = bfp_ref.exponent_bias(e)
            scale = 2.0 ** (grid["stored"][bfp_ref.group_ids(pred.size, gs)] - bias_e)
            slack_codes = (bound.ravel() / scale) + k * census.ulp32(np.abs(pred.ravel()) / scale)
        a = compare_packed(pred32.ravel(), grid["codes"], grid["stored"], m, e, gs, k, slack_codes)
        a = {kk: v for kk, v in a.items() if kk not in ("ref", "under_c")}
        a["seq_fold_single_segment"] = st.single_segment
        a["fold_stats"] = st.to_json()
        if guard_class:
            a["taxonomy"] = "MISMATCH"
            a["mismatch_class"] = (a["mismatch_class"] + "+" + guard_class) if a["mismatch_class"] != "none" else guard_class
        # Leg B: fold bound + the pack rounding (half a code at the C's scale)
        bias_e = bfp_ref.exponent_bias(e)
        scale = 2.0 ** (grid["stored"][bfp_ref.group_ids(pred.size, gs)] - bias_e)
        pack_bound = (0.5 * scale).reshape(exact.shape)
        err = np.abs(c_vals - exact)
        total = bound + pack_bound
        clamped = (np.abs(pred.ravel() / scale) > bfp_ref.qmax(m) + 0.5).reshape(exact.shape)
        b = elementwise_leg_b(err, total, ~clamped)
        b["bound_kind"] = "N*u*S + 0.5*2^E_C (design fold + pack)"
        b["n_clamped_excluded"] = int(clamped.sum())
        row["c_scale"] = None
    row["legA"] = a
    row["legB"] = b
    row["bfp"] = {
        "summary": (
            f"N={n_red} segs<={st.n_segments_max} |partial|<={st.partial_abs_max} >2^24:{st.n_partials_gt_2p24} "
            f"cast_loss={st.cast_loss_max_value:.2e} seq_fold={'bitexact' if a.get('seq_fold_bitexact', a['taxonomy'] == 'EXACT') else 'differs'}"
        ),
        "stats": st.to_json(),
        "ratio_rigorous": b.get("ratio_rigorous"),
    }
    return row


def run_gemm_cell(cell: dict, driver: Path, out_root: Path, k: int, run_driver) -> list[dict]:
    name = cell["name"]
    rows = []
    for tier in cell.get("tiers", ["kernel", "layer"]):
        base = {"cell": name, "tier": tier, "op": cell["op"], "direction": cell["direction"]}
        try:
            fx = build_gemm_fixture(cell)
            work = out_root / "runs" / f"{name}__{tier}"
            in_dir = work / "in"
            in_dir.mkdir(parents=True, exist_ok=True)
            rounding = cell["arith"].get("rounding", "HALF_AWAY")
            operands = {nm: o.spec(nm, in_dir, rounding, "BFP" if tier == "kernel" else None) for nm, o in fx.ops.items()}
            out = output_spec(cell) if tier == "layer" else {"dtype": "FLOAT32"}
            spec_path = write_spec(in_dir, cell, tier, operands, out)
            drv = run_driver(driver, spec_path, in_dir, work / "out")
            direction = cell["direction"]
            out_name = {"forward": "output", "dx": "propLoss", "weightGrad": "weightGrad"}[direction]
            fixture_info = {
                "type": cell["fixture"]["type"],
                "shapes": {nm: list(o.codes.shape) for nm, o in fx.ops.items()},
                "storage": {nm: ("BFP" if tier == "kernel" else o.storage) for nm, o in fx.ops.items()},
                "scales": {},
                "notes": fx.notes,
            }
            expect = cell.get("expect", {})
            res = evaluate_gemm_output(fx, direction, drv, out_name, k, True)
            row = dict(base)
            row.update(res)
            row["fixture"] = fixture_info
            row["verdict"] = verdict(row, expect)
            rows.append(row)
            if direction == "weightGrad" and "bias" in fx.ops and (drv.exit == 0 or not res["headroom"]["ok"]):
                res2 = evaluate_gemm_output(fx, "biasGrad", drv, "biasGrad", k, False)
                r2 = dict(base)
                r2.update(res2)
                r2["fixture"] = fixture_info
                r2["verdict"] = verdict(r2, expect.get("biasGrad", {}))
                rows.append(r2)
        except Exception as exc:  # noqa: BLE001
            row = dict(base)
            row.update({"verdict": "ERROR", "error": str(exc)})
            rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# B5: fake-quant twin                                                          #
# --------------------------------------------------------------------------- #


def run_fq_cell(cell: dict, driver: Path, out_root: Path, k: int, run_driver) -> list[dict]:
    """Two layer-tier runs under ARITH_FLOAT32: (bfp) the operands as stored
    (BFP weights/bias, the input as declared), (f32) every BFP operand
    replaced by its exact float32 dequantization. Every produced output must
    be bit-identical (Leg A); Leg B is float64 within the float32 kernel
    bound (2N+1 roundings)."""
    name = cell["name"]
    base = {"cell": name, "tier": "layer", "op": cell["op"], "direction": cell["direction"]}
    rows = []
    try:
        fx = build_gemm_fixture(cell)
        rounding = cell["arith"].get("rounding", "HALF_AWAY")
        results = {}
        for variant in ("bfp", "f32"):
            work = out_root / "runs" / f"{name}__fq_{variant}"
            in_dir = work / "in"
            in_dir.mkdir(parents=True, exist_ok=True)
            operands = {nm: o.spec(nm, in_dir, rounding, None if variant == "bfp" else "FLOAT32") for nm, o in fx.ops.items()}
            spec_path = write_spec(in_dir, cell, "layer", operands, {"dtype": "FLOAT32"})
            drv = run_driver(driver, spec_path, in_dir, work / "out")
            if drv.exit != 0:
                raise RuntimeError(f"driver exit {drv.exit} on the {variant} twin\n{drv.stderr}")
            results[variant] = drv
        direction = cell["direction"]
        outs = ["output"] if direction == "forward" else (["propLoss"] if direction == "dx" else ["weightGrad"] + (["biasGrad"] if "bias" in fx.ops else []))
        fixture_info = {
            "type": cell["fixture"]["type"],
            "shapes": {nm: list(o.codes.shape) for nm, o in fx.ops.items()},
            "storage": {nm: o.storage for nm, o in fx.ops.items()},
            "scales": {},
            "notes": fx.notes + ["twin f32: every BFP operand replaced by its exact float32 dequantization"],
        }
        for out_name in outs:
            d = "biasGrad" if out_name == "biasGrad" else direction
            walk = gemm_walk(fx, d)
            exact, s_abs = gemm_exact_and_abs(fx, d)
            c_bfp = results["bfp"].out(out_name)["values"].astype(np.float64).reshape(exact.shape)
            c_f32 = results["f32"].out(out_name)["values"].astype(np.float64).reshape(exact.shape)
            bit = float_bitexact(c_bfp, c_f32)
            taxonomy = "EXACT" if bit["n_mismatch"] == 0 else "MISMATCH"
            a = {
                "taxonomy": taxonomy,
                "census": {"taxonomy": taxonomy, "mode": "fq_twin_bits", "n_elements": bit["n_elements"], "n_mismatch": bit["n_mismatch"], "n_tie": 0, "max_abs_diff": bit["max_ulp32"], "worst": bit["worst"]},
                "mismatch_class": "none" if taxonomy == "EXACT" else "fq",
                "seed": None,
                "scale": {"c": None, "ref": None, "rel": None, "ok": True},
                "ref_scale": None,
            }
            bound = exact_ref.bound_float32_kernel(walk.reduction, s_abs, d == "forward" and "bias" in fx.ops)
            err = np.abs(c_bfp - exact)
            ratio = float((err / bound).max()) if err.size else 0.0
            b = census.leg_b_check(float(err.max()) if err.size else 0.0, float(bound.min()))
            b["ratio"] = ratio
            b["taxonomy"] = "BOUNDED" if ratio <= 1.0 else "EXCESS"
            b["bound_kind"] = "(2N+1)*u*S float32 kernel"
            row = dict(base)
            row.update({"output": out_name, "legA": a, "legB": b, "c_scale": None, "n_elements": int(exact.size), "fixture": fixture_info})
            row["bfp"] = {"summary": f"fq twin N={walk.reduction} bits_equal={taxonomy == 'EXACT'} f32_vs_f64_ulp<= {float((err / census.ulp32(np.maximum(np.abs(exact), np.finfo(np.float32).tiny))).max()) if err.size else 0.0:.2f}"}
            row["verdict"] = verdict(row, cell.get("expect", {}) if out_name != "biasGrad" else cell.get("expect", {}).get("biasGrad", {}))
            rows.append(row)
        return rows
    except Exception as exc:  # noqa: BLE001
        row = dict(base)
        row.update({"verdict": "ERROR", "error": str(exc)})
        return [row]


# --------------------------------------------------------------------------- #
# Verdict and dispatch                                                        #
# --------------------------------------------------------------------------- #


def verdict(row: dict, expect: dict) -> str:
    a, b = row["legA"], row["legB"]
    if a["taxonomy"] == "ABORT":
        return "PASS" if expect.get("abort", True) is not False else "FAIL"
    v = census.combine(a["taxonomy"], a["scale"]["ok"], b["taxonomy"] if b["taxonomy"] != "n/a" else "BOUNDED")
    if expect:
        if a["taxonomy"] not in expect.get("legA", [a["taxonomy"]]):
            v = "FAIL"
        if b["taxonomy"] != expect.get("legB", b["taxonomy"]):
            v = "FAIL"
        if expect.get("abort") is True and a["taxonomy"] != "ABORT":
            v = "FAIL"
    return v


def run_cell(cell: dict, driver: Path, out_root: Path, k: int, run_driver) -> list[dict]:
    fam = cell["family"]
    if fam == "bfp_pack":
        return run_pack_cell(cell, driver, out_root, k, run_driver)
    if fam == "bfp_gemm":
        return run_gemm_cell(cell, driver, out_root, k, run_driver)
    if fam == "bfp_fq":
        return run_fq_cell(cell, driver, out_root, k, run_driver)
    raise ValueError(f"{cell['name']}: unknown BFP family {fam!r}")
