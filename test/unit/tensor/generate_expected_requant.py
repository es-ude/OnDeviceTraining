#!/usr/bin/env python3
"""Generate expected_requant.h for UnitTestTensorConversion (#192 PR-1, spec D1/D7).

Emulates the two C SYM_INT32 -> SYM_INT32 requant kernels BIT-EXACTLY:
the C kernels do all per-element arithmetic in float32 (int->float cast,
multiply by inScale, divide by scale, clamp), so the emulator mirrors every
op in torch.float32 and applies the half-away rounding on the float64-exact
image of the float32 quotient (C round() operates on the double-promoted
float — identical). Expected MANTISSAS are therefore exact integers;
expected SCALES carry a 2-ulp analytic tolerance (>= 2 ulp32 for any
magnitude: |s| * 2^-22).

Fixtures (qMaxBits = 16, qMax = 32767, qMin = -32768 throughout):
  f1AccumRange  dynamic; +-2e9 accumulator-range mantissas, inScale 3.05e-5;
                pins scale-blindness (pass A must dequantize) and the exact
                absmax -> qMax mapping (max|out| == 32767).
  f2AbsmaxZero  dynamic; all-zero mantissas -> zero mantissas, scale 1.0.
  f3Rescale     dynamic; same mantissas at inScale s and 10*s; the C test
                reuses ONE output qConfig across both calls -> pins
                freeze-the-scale (scale must track the input rescale ~10x).
  f4Tie         dynamic; scale lands on exactly 4.0, quotients exactly
                k + 0.5 -> pins half-away-from-zero vs half-to-even AND vs
                floor/trunc casts (positive and negative ties).
  f5ToScaleFit  fixed-scale; target > absmax/qMax -> no clamping; kernel
                must NOT touch the pre-set scale.
  f6ToScaleSat  fixed-scale; target too small -> mantissas CLAMP at
                qMin/qMax (saturation is the documented Deutel-Eq.4
                semantics, NOT an error).

Self-checks: see asserts inline; the script aborts instead of emitting a
header that contradicts its own emulation. Run via `uv run` (CMake wires
this automatically).
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "goldgen"))

from sym_gold import (QMAX, QMIN, emit_float_scalar, emit_int32_array,
                      emit_int32_scalar, round_half_away)

QMAX_BITS = 16


def _f32(x) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32)


def scale_tol(scale: float) -> float:
    """Analytic scale tolerance: >= 2 ulp32 for any magnitude (s in
    [2^e, 2^(e+1)) has ulp32 = 2^(e-23) <= s * 2^-23)."""
    return abs(scale) * 2.0 ** -22


def emulate_requant_dynamic(mantissas: torch.Tensor, in_scale: float):
    """requantSymInt32Tensor mirrored in float32: pass A absmax over
    |(float)m_i * inScale| (reads only), scale = absmax/qMax (1.0 if
    absmax == 0), pass B round-clamp of ((float)m_i * inScale)/scale."""
    deq = mantissas.to(torch.float32) * _f32(in_scale)
    absmax = deq.abs().max()
    if absmax.item() == 0.0:
        scale = _f32(1.0)
    else:
        scale = absmax / _f32(QMAX)
    quot = torch.clamp(deq / scale, QMIN, QMAX).to(torch.float64)
    out = round_half_away(quot).to(torch.int32)
    return out, float(scale.item())


def emulate_requant_to_scale(mantissas: torch.Tensor, in_scale: float,
                             target_scale: float):
    """requantSymInt32TensorToScale mirrored in float32: round-clamp of
    ((float)m_i * inScale)/targetScale; saturates at qMin/qMax; the target
    scale is pre-set on the output qConfig and never modified."""
    assert target_scale > 0.0, "fixture bug: guard regime is not unit-testable"
    deq = mantissas.to(torch.float32) * _f32(in_scale)
    quot = torch.clamp(deq / _f32(target_scale), QMIN, QMAX).to(torch.float64)
    return round_half_away(quot).to(torch.int32), deq


def _roundtrip_check(name, out, scale, mantissas, in_scale, max_abs_mantissa):
    """|out_i*scale - m_i*inScale| <= 0.5 LSB (rounding) + ~0.002 LSB
    (1 ulp32(qMax) float32 division drift) + int->float32 cast error
    (<= 0.5 ulp of the largest input mantissa, in value space)."""
    v64 = mantissas.to(torch.float64) * float(_f32(in_scale).item())
    cast_err = (2.0 ** -24) * 2.0 * max_abs_mantissa * float(_f32(in_scale).item())
    err = (out.to(torch.float64) * scale - v64).abs().max().item()
    bound = 0.502 * scale + cast_err
    assert err <= bound, f"{name}: round-trip err {err} > {bound}"


def fixture_f1_accum_range():
    m = torch.tensor([1999999999, -2000000000, 123456789, -987654321,
                      31415926, -27182818, 500000000, 0], dtype=torch.int32)
    in_scale = 3.05e-5
    out, scale = emulate_requant_dynamic(m, in_scale)
    assert out.abs().max().item() == int(QMAX), (
        f"f1AccumRange: absmax must map exactly to qMax, got "
        f"{out.abs().max().item()}")
    _roundtrip_check("f1AccumRange", out, scale, m, in_scale, 2.0e9)
    return {"name": "f1AccumRange", "input": m, "inputScale": in_scale,
            "expected": out, "expectedScale": scale}


def fixture_f2_absmax_zero():
    m = torch.zeros(6, dtype=torch.int32)
    in_scale = 0.25
    out, scale = emulate_requant_dynamic(m, in_scale)
    assert torch.equal(out, torch.zeros(6, dtype=torch.int32)), "f2: out != 0"
    assert scale == 1.0, f"f2AbsmaxZero: scale must be exactly 1.0, got {scale}"
    return {"name": "f2AbsmaxZero", "input": m, "inputScale": in_scale,
            "expected": out, "expectedScale": scale}


def fixture_f3_rescale():
    m = torch.tensor([1000, -2000, 30, 4500, -16000, 12345], dtype=torch.int32)
    in_scale_a = 2.0 ** -8     # 0.00390625, exact in float32
    in_scale_b = 10.0 * in_scale_a  # 0.0390625 = 5*2^-7, ALSO exact in float32
    out_a, scale_a = emulate_requant_dynamic(m, in_scale_a)
    out_b, scale_b = emulate_requant_dynamic(m, in_scale_b)
    # Same mantissas + exactly-10x input scale => identical quotients up to
    # float32 ulps (no quotient sits near a tie) => identical mantissas.
    assert torch.equal(out_a, out_b), "f3Rescale: mantissas must be invariant"
    assert abs(scale_b / scale_a - 10.0) < 1e-6, (
        f"f3Rescale: scale ratio {scale_b / scale_a} != 10")
    assert out_a.abs().max().item() == int(QMAX), "f3Rescale: absmax->qMax"
    _roundtrip_check("f3Rescale/A", out_a, scale_a, m, in_scale_a, 16000.0)
    _roundtrip_check("f3Rescale/B", out_b, scale_b, m, in_scale_b, 16000.0)
    return {"name": "f3Rescale", "input": m,
            "inputScaleA": in_scale_a, "inputScaleB": in_scale_b,
            "expectedA": out_a, "expectedScaleA": scale_a,
            "expectedB": out_b, "expectedScaleB": scale_b}


def fixture_f4_tie():
    # absmax mantissa 131068 = 4*32767 with inScale 1.0 => scale exactly 4.0f
    # => quotients m/4: 10 -> 2.5, 18 -> 4.5, -10 -> -2.5, -26 -> -6.5 are
    # EXACT float32 ties; 7 -> 1.75 is a non-tie control value.
    m = torch.tensor([131068, 10, 18, -10, -26, 7], dtype=torch.int32)
    in_scale = 1.0
    out, scale = emulate_requant_dynamic(m, in_scale)
    assert scale == 4.0, f"f4Tie: scale must be exactly 4.0, got {scale}"
    quot = (m.to(torch.float32) * _f32(in_scale)) / _f32(scale)
    quot64 = torch.clamp(quot, QMIN, QMAX).to(torch.float64)
    # the fixture must DIFFER under half-to-even (torch.round) ...
    assert not torch.equal(out, torch.round(quot64).to(torch.int32)), (
        "f4Tie: vacuous against half-to-even")
    # ... and under truncation toward zero ((int32_t) cast in C)
    assert not torch.equal(out, quot64.trunc().to(torch.int32)), (
        "f4Tie: vacuous against trunc-cast")
    expected = torch.tensor([32767, 3, 5, -3, -7, 2], dtype=torch.int32)
    assert torch.equal(out, expected), f"f4Tie: got {out.tolist()}"
    return {"name": "f4Tie", "input": m, "inputScale": in_scale,
            "expected": out, "expectedScale": scale}


def fixture_f5_to_scale_fit():
    # values m * 2^-10, target 2^-4: quotients m/64 stay well inside
    # [qMin, qMax] => no clamping; all arithmetic float32-exact.
    m = torch.tensor([12000, -32000, 500, -1, 25000, 0], dtype=torch.int32)
    in_scale = 2.0 ** -10
    target = 0.0625  # 2^-4, exact; > absmax/qMax = 31.25/32767 ~ 9.5e-4
    out, deq = emulate_requant_to_scale(m, in_scale, target)
    quot = deq.to(torch.float64) / target
    assert (quot.abs() < QMAX).all(), "f5ToScaleFit: must not saturate"
    expected = torch.tensor([188, -500, 8, 0, 391, 0], dtype=torch.int32)
    assert torch.equal(out, expected), f"f5ToScaleFit: got {out.tolist()}"
    return {"name": "f5ToScaleFit", "input": m, "inputScale": in_scale,
            "targetScale": target, "expected": out}


def fixture_f6_to_scale_sat():
    # values m * 0.5, target 2^-7: quotients m*64 overshoot BOTH bounds
    # (600*64 = 38400 > qMax, -600*64 < qMin, 512*64 = 32768 = qMax+1)
    # => pins saturation semantics at qMin AND qMax.
    m = torch.tensor([600, -600, 20, -40, 512, -512], dtype=torch.int32)
    in_scale = 0.5
    target = 0.0078125  # 2^-7, exact
    out, deq = emulate_requant_to_scale(m, in_scale, target)
    quot = deq.to(torch.float64) / target
    assert (quot > QMAX).any() and (quot < QMIN).any(), (
        "f6ToScaleSat: vacuous, no saturation on one of the bounds")
    expected = torch.tensor([32767, -32768, 1280, -2560, 32767, -32768],
                            dtype=torch.int32)
    assert torch.equal(out, expected), f"f6ToScaleSat: got {out.tolist()}"
    assert out.max().item() == int(QMAX) and out.min().item() == int(QMIN)
    return {"name": "f6ToScaleSat", "input": m, "inputScale": in_scale,
            "targetScale": target, "expected": out}


def emit_dynamic_fixture(parts, fx):
    pre = f"requant_{fx['name']}"
    parts.append(emit_int32_array(f"input_{pre}", fx["input"]))
    parts.append(emit_float_scalar(f"inputScale_{pre}", fx["inputScale"]))
    parts.append(emit_int32_array(f"expected_{pre}", fx["expected"]))
    parts.append(emit_float_scalar(f"expectedScale_{pre}", fx["expectedScale"]))
    parts.append(emit_float_scalar(f"scaleTol_{pre}", scale_tol(fx["expectedScale"])))


def emit_rescale_fixture(parts, fx):
    pre = f"requant_{fx['name']}"
    parts.append(emit_int32_array(f"input_{pre}", fx["input"]))
    parts.append(emit_float_scalar(f"inputScaleA_{pre}", fx["inputScaleA"]))
    parts.append(emit_float_scalar(f"inputScaleB_{pre}", fx["inputScaleB"]))
    parts.append(emit_int32_array(f"expectedA_{pre}", fx["expectedA"]))
    parts.append(emit_float_scalar(f"expectedScaleA_{pre}", fx["expectedScaleA"]))
    parts.append(emit_float_scalar(f"scaleTolA_{pre}", scale_tol(fx["expectedScaleA"])))
    parts.append(emit_int32_array(f"expectedB_{pre}", fx["expectedB"]))
    parts.append(emit_float_scalar(f"expectedScaleB_{pre}", fx["expectedScaleB"]))
    parts.append(emit_float_scalar(f"scaleTolB_{pre}", scale_tol(fx["expectedScaleB"])))


def emit_to_scale_fixture(parts, fx):
    pre = f"requant_{fx['name']}"
    parts.append(emit_int32_array(f"input_{pre}", fx["input"]))
    parts.append(emit_float_scalar(f"inputScale_{pre}", fx["inputScale"]))
    parts.append(emit_float_scalar(f"targetScale_{pre}", fx["targetScale"]))
    parts.append(emit_int32_array(f"expected_{pre}", fx["expected"]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    parts = [
        "// AUTOGENERATED by generate_expected_requant.py — DO NOT EDIT\n",
        "#ifndef ODT_EXPECTED_REQUANT_H\n",
        "#define ODT_EXPECTED_REQUANT_H\n",
        "#include <stdint.h>\n",
        "#include <stdlib.h>\n\n",
    ]
    parts.append(emit_int32_scalar("qMaxBits_requant", QMAX_BITS))

    emit_dynamic_fixture(parts, fixture_f1_accum_range())
    emit_dynamic_fixture(parts, fixture_f2_absmax_zero())
    emit_rescale_fixture(parts, fixture_f3_rescale())
    emit_dynamic_fixture(parts, fixture_f4_tie())
    emit_to_scale_fixture(parts, fixture_f5_to_scale_fit())
    emit_to_scale_fixture(parts, fixture_f6_to_scale_sat())

    parts.append("\n#endif // ODT_EXPECTED_REQUANT_H\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(parts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
