"""Mismatch census, the bias-seed policy and the result taxonomy.

Two comparison modes:

- exact (``pre=None``): every differing code is a mismatch. The kernel tier
  uses it: a raw int32 accumulator has no rounding point of its own, so
  nothing there may be waived (integers are never "near a .5 boundary").
- tie (``pre`` given): a differing code is a float32 rounding tie when
  |delta| == 1 and the reference's pre-rounding value lies within
  k * ulp32(|v|) of a .5 boundary (k = 4) with the C code on the other side.
  The layer tier uses it for the restored wire, whose pre-rounding values
  are |v| <= qMax so the window is always much narrower than 1.

The bias seed is handled per output channel by ``seed_policy`` (below), on
the seed's own pre-rounding value, never on the accumulator it is added to.

Taxonomy (per leg, then combined):
  EXACT     bit-equal codes, every seed channel resolved to the real-valued
            rounding
  TIE_ONLY  every mismatch is a restore tie, or a seed channel needed the
            alternative rounding / the float32 reading (spec underspecified
            in float32, not a bug) -- see ``seed_policy`` for the deviation
            such a channel may carry
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
F32_U = 2.0**-24  # float32 unit roundoff (half an ulp, relative)

# The documented seed expression ``param_q * param_scale / accumulator_scale``
# read left-to-right in float32 is three roundings (q*s_b, s_in*s_w, the
# quotient); the relative error of such a chain is at most
# (1+u)^2 / (1-u) - 1 (two factors rounded up, one divisor rounded down).
SEED_F32_REL = (1.0 + F32_U) ** 2 / (1.0 - F32_U) - 1.0


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


def window_discriminates(pre: np.ndarray, k: int = K_DEFAULT) -> np.ndarray:
    """True where the k-ulp32 tie window is narrower than the unit interval,
    i.e. k * ulp32(|v|) < 0.5. Where it is not, every value passes
    ``tie_mask`` and the window says nothing (for k = 4 that is |v| >= 2^20:
    ulp32 = 2^-3 there, so k * ulp32 = 0.5)."""
    return k * ulp32(pre) < 0.5


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


# --------------------------------------------------------------------------- #
# Bias-seed policy (per output channel, on the seed's own value)             #
# --------------------------------------------------------------------------- #

SEED_REF = "ref"  # the real-valued rounding of the documented rule
SEED_ALT = "alt"  # the other side of a float32-tied .5 boundary (window)
SEED_F32 = "f32"  # the documented expression read left-to-right in float32


@dataclass
class ChannelSeed:
    """What Leg A accepts as the bias seed of one output channel.

    ``allowed`` lists (seed, kind) pairs, the real-valued rounding first when
    it is allowed. ``granted_lsb`` is the deviation from the real-valued
    pre-rounding value that the policy grants beyond the 0.5 of the rounding
    itself (Leg B's bound must cover it, else Leg A and Leg B contradict):
      pinned      0
      window      k * ulp32(|pre|): the alternative is at most that far
                  past the boundary
      f32-gated   SEED_F32_REL * |pre|: the float32 reading of the expression
                  is that far from the real value, which can move the rounding
    """

    pre: float  # real-valued pre-rounding value b_q * s_b / s_raw
    ref: int
    alt: int
    f32: int | None  # None when the float32 reading is undefined (non-finite)
    regime: str  # "pinned" | "window" | "f32" | "f32-undefined"
    allowed: list[tuple[int, str]]
    granted_lsb: float


def seed_policy(pre: np.ndarray, f32_seed: np.ndarray | None, k: int = K_DEFAULT) -> list[ChannelSeed]:
    """Per-channel acceptance set for the forward bias seed.

    Judged on the seed's own real-valued pre-rounding value ``pre``:
      - window discriminates (k * ulp32(|pre|) < 0.5) and |pre| within it of
        a .5 boundary: the real-valued rounding or the other side (a float32
        expression-order tie the doc does not fix);
      - window discriminates, not in it: the real-valued rounding only;
      - window does not discriminate (|pre| >= 2^20 at k = 4): the documented
        expression read left-to-right in float32 is the gate -- an
        INTERPRETATION recorded in README.md and DOC_GAPS, since the doc
        states the rule in reals and float32 cannot represent the .5
        boundary there; when that reading is undefined (non-finite) both
        roundings stay acceptable and the full window is granted.
    """
    pre = np.asarray(pre, dtype=np.float64)
    ref = np.where(pre >= 0.0, np.floor(pre + 0.5), np.ceil(pre - 0.5)).astype(np.int64)
    alt = alt_rounding(pre, ref)
    disc = window_discriminates(pre, k)
    in_window = tie_mask(pre, k)
    out: list[ChannelSeed] = []
    for c in range(pre.size):
        p = float(pre[c])
        r = int(ref[c])
        a = int(alt[c])
        f = None
        if f32_seed is not None and np.isfinite(f32_seed[c]):
            f = int(f32_seed[c])
        if disc[c]:
            if in_window[c]:
                out.append(ChannelSeed(p, r, a, f, "window", [(r, SEED_REF), (a, SEED_ALT)], float(k * ulp32(p))))
            else:
                out.append(ChannelSeed(p, r, a, f, "pinned", [(r, SEED_REF)], 0.0))
        elif f is not None:
            out.append(ChannelSeed(p, r, a, f, "f32", [(f, SEED_F32)], SEED_F32_REL * abs(p)))
        else:
            out.append(
                ChannelSeed(p, r, a, f, "f32-undefined", [(r, SEED_REF), (a, SEED_ALT)], float(k * ulp32(p)))
            )
    return out


def seed_granted_lsb(policy: list[ChannelSeed] | None) -> float | None:
    """The largest deviation (raw LSB, beyond the 0.5 of the rounding) the
    policy grants on any channel; None without a bias."""
    if policy is None:
        return None
    return max((cs.granted_lsb for cs in policy), default=0.0)


# --------------------------------------------------------------------------- #
# Census                                                                     #
# --------------------------------------------------------------------------- #


@dataclass
class CensusResult:
    taxonomy: str  # EXACT | TIE_ONLY | MISMATCH
    n_elements: int
    n_mismatch: int
    n_tie: int
    max_abs_diff: int
    worst: list[dict] = field(default_factory=list)  # up to 8 non-tie mismatches
    k: int = K_DEFAULT
    mode: str = "tie"  # "exact" | "tie"

    def to_json(self) -> dict:
        return {
            "taxonomy": self.taxonomy,
            "mode": self.mode,
            "n_elements": self.n_elements,
            "n_mismatch": self.n_mismatch,
            "n_tie": self.n_tie,
            "max_abs_diff": self.max_abs_diff,
            "k_ulp32": self.k,
            "worst": self.worst,
        }


def census(
    c_codes: np.ndarray,
    ref_codes: np.ndarray,
    pre: np.ndarray | None = None,
    k: int = K_DEFAULT,
) -> CensusResult:
    """Classify the C-vs-reference code mismatches (see module docstring).

    ``pre=None`` is the exact mode: no tie waiver at all."""
    c = np.asarray(c_codes, dtype=np.int64).ravel()
    r = np.asarray(ref_codes, dtype=np.int64).ravel()
    if c.shape != r.shape:
        raise ValueError(f"shape mismatch: C {c.shape}, ref {r.shape}")
    mode = "exact" if pre is None else "tie"
    diff = c - r
    mismatch = diff != 0
    n_mismatch = int(mismatch.sum())
    max_abs = int(np.abs(diff).max()) if diff.size else 0
    if n_mismatch == 0:
        return CensusResult("EXACT", int(c.size), 0, 0, 0, [], k, mode)
    if pre is None:
        ties = np.zeros_like(mismatch)
        v = None
    else:
        v = np.asarray(pre, dtype=np.float64).ravel()
        if v.shape != r.shape:
            raise ValueError(f"shape mismatch: ref {r.shape}, pre {v.shape}")
        ties = mismatch & tie_mask(v, k) & (c == alt_rounding(v, r))
    n_tie = int(ties.sum())
    bad = np.flatnonzero(mismatch & ~ties)
    worst = []
    order = bad[np.argsort(-np.abs(diff[bad]))][:8]
    for i in order:
        entry = {"index": int(i), "c": int(c[i]), "ref": int(r[i])}
        if v is not None:
            entry["pre"] = float(v[i])
        worst.append(entry)
    taxonomy = "TIE_ONLY" if bad.size == 0 else "MISMATCH"
    return CensusResult(taxonomy, int(c.size), n_mismatch, n_tie, max_abs, worst, k, mode)


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
