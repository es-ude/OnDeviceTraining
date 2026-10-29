"""Mismatch census and the result taxonomy.

For every element where the C codes and the Leg A codes differ, decide whether
the difference is a float32 rounding tie: |delta| == 1 and the reference's
pre-rounding value lies within k * ulp32(|v|) of a .5 boundary (k = 4), with
the C code on the other side of that boundary. Everything else is a real
mismatch.

Taxonomy (per leg, then combined):
  EXACT     bit-equal codes
  TIE_ONLY  every mismatch is a tie (spec underspecified in float32, not a bug)
  MISMATCH  Leg A disagrees away from a tie, or the scale is off by more than
            2^-22 relative (implementation finding)
  BOUNDED   Leg B inside its analytic bound
  EXCESS    Leg B outside the bound (scheme finding)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

K_DEFAULT = 4
SCALE_REL_TOL = 2.0**-22


def ulp32(v: np.ndarray) -> np.ndarray:
    """Spacing of float32 at |v| (float64 result); the spacing at the smallest
    normal float32 for |v| == 0 so a zero never divides anything."""
    a = np.abs(np.asarray(v, dtype=np.float64))
    a32 = np.maximum(a, np.finfo(np.float32).tiny).astype(np.float32)
    return np.spacing(a32).astype(np.float64)


def tie_mask(pre: np.ndarray, k: int = K_DEFAULT) -> np.ndarray:
    """True where the pre-rounding value sits within k ulp32 of a .5 boundary."""
    a = np.abs(np.asarray(pre, dtype=np.float64))
    frac = a - np.floor(a)
    return np.abs(frac - 0.5) <= k * ulp32(a)


def alt_rounding(pre: np.ndarray, ref_codes: np.ndarray) -> np.ndarray:
    """The code on the other side of the nearest .5 boundary: for |v| with
    frac < 0.5 the reference rounded toward zero, so the alternative is one
    step away from zero; frac >= 0.5 (incl. the exact tie, which half-away
    sends away from zero) the alternative is one step toward zero."""
    v = np.asarray(pre, dtype=np.float64)
    a = np.abs(v)
    frac = a - np.floor(a)
    sign = np.where(v >= 0.0, 1, -1)
    step = np.where(frac < 0.5, 1, -1)
    return np.asarray(ref_codes, dtype=np.int64) + sign * step


@dataclass
class CensusResult:
    taxonomy: str  # EXACT | TIE_ONLY | MISMATCH
    n_elements: int
    n_mismatch: int
    n_tie: int
    max_abs_diff: int
    worst: list[dict] = field(default_factory=list)  # up to 8 non-tie mismatches
    k: int = K_DEFAULT

    def to_json(self) -> dict:
        return {
            "taxonomy": self.taxonomy,
            "n_elements": self.n_elements,
            "n_mismatch": self.n_mismatch,
            "n_tie": self.n_tie,
            "max_abs_diff": self.max_abs_diff,
            "k_ulp32": self.k,
            "worst": self.worst,
        }


def census(
    c_codes: np.ndarray, ref_codes: np.ndarray, pre: np.ndarray, k: int = K_DEFAULT
) -> CensusResult:
    """Classify the C-vs-reference code mismatches (see module docstring)."""
    c = np.asarray(c_codes, dtype=np.int64).ravel()
    r = np.asarray(ref_codes, dtype=np.int64).ravel()
    v = np.asarray(pre, dtype=np.float64).ravel()
    if c.shape != r.shape or v.shape != r.shape:
        raise ValueError(f"shape mismatch: C {c.shape}, ref {r.shape}, pre {v.shape}")
    diff = c - r
    mismatch = diff != 0
    n_mismatch = int(mismatch.sum())
    max_abs = int(np.abs(diff).max()) if diff.size else 0
    if n_mismatch == 0:
        return CensusResult("EXACT", int(c.size), 0, 0, 0, [], k)
    ties = mismatch & tie_mask(v, k) & (c == alt_rounding(v, r))
    n_tie = int(ties.sum())
    bad = np.flatnonzero(mismatch & ~ties)
    worst = []
    order = bad[np.argsort(-np.abs(diff[bad]))][:8]
    for i in order:
        worst.append({"index": int(i), "c": int(c[i]), "ref": int(r[i]), "pre": float(v[i])})
    taxonomy = "TIE_ONLY" if bad.size == 0 else "MISMATCH"
    return CensusResult(taxonomy, int(c.size), n_mismatch, n_tie, max_abs, worst, k)


def scale_check(c_scale: float, ref_scale: float, rel_tol: float = SCALE_REL_TOL) -> dict:
    """Relative scale agreement; a miss is a MISMATCH-class finding."""
    rel = abs(float(c_scale) - float(ref_scale)) / abs(float(ref_scale))
    return {"c": float(c_scale), "ref": float(ref_scale), "rel": rel, "ok": rel <= rel_tol}


def leg_b_check(err_max: float, bound: float) -> dict:
    return {
        "err_max": float(err_max),
        "bound": float(bound),
        "ratio": float(err_max / bound) if bound > 0 else float("inf"),
        "taxonomy": "BOUNDED" if err_max <= bound else "EXCESS",
    }


def combine(leg_a_taxonomy: str, scale_ok: bool, leg_b_taxonomy: str) -> str:
    """Overall cell verdict: PASS unless Leg A is a MISMATCH (incl. a scale
    miss) or Leg B is in EXCESS."""
    a = leg_a_taxonomy if scale_ok else "MISMATCH"
    if a == "MISMATCH" or leg_b_taxonomy == "EXCESS":
        return "FAIL"
    return "PASS"
