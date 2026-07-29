#!/usr/bin/env python3
"""Generate expected_asym_nudged.h for UnitTestTensorConversion (group-quant
PR4 Task 1, spec D6).

Emulates the NUDGED CODE-DOMAIN ASYM quantizer (deriveAsymGridFromMinMax +
emitAsymChunk after the PR4 rewrite) BIT-EXACTLY in float32 -- see
sym_gold.quantize_asym_nudged for the math and its inline self-checks
(zp in [0, 2^b-1], exact-zero decode, 0.5*scale round-trip bound).

This is a DELIBERATE numerics change (D6): existing ASYM pins re-derive
through the new math, they are NOT sign-flips of the old codes. Every fixture
below therefore also runs the OLD value-domain emulation
(quantize_asym_old_value_domain) and asserts old != new on at least one of
{codes, zp} -- a generator run that cannot distinguish the two grids aborts
instead of emitting vacuous pins.

Fixtures:
  f1SpanZero   {1,2,3,4,-1,-2} @ qBits=5 -- the canonical 6-element pin
               (testConversionFloatAsym / IntAsym / SymInt32Asym). Band spans
               zero, nudge is a no-op; zp flips -10 -> +10 and the codes
               happen to coincide with the old grid (verified, not assumed).
  f2NegBand16  {-10,-1,-5.5,-2.5,-9.25,-3.75} @ qBits=16 -- all-negative
               band; the nudge extends it to [-10, 0] so zp lands exactly on
               the uint16 ceiling 2^16-1 = 65535 (the D6 boundary pin that
               replaces the old -72817 int32-width pin).
  f3NegFar     {-5000000, -4999999.5} @ qBits=8 -- the old
               ZeroPointBeyondInt32 DEATH data: un-nudged zpReal ~ -2.55e9
               was an int32-overflow abort; nudged it derives a VALID grid
               (zp = 255). Pins the death-test obsolescence.
  f4ClampTie   {-1.5, 5.5, 0.0, 2.0} @ qBits=3, scale exactly 1.0 -- zpReal
               = 1.5 is an exact tie (zp -> 2) AND v=5.5 rounds up too, so
               the un-clamped top code is 8 > qMax=7: the ONE fixture whose
               emitted code needs the encode clamp (mutation (ii) guard).
               Also contains an exact 0.0 (zero-representability pin).
  f5SymDeq     {5,-4,2,-1,3,-5} @ qBits=5 -- the SYM->ASYM re-pin
               (testConversionSymToAsymRoundTrips' dequantized mantissas).
  f6AccumRef   {3, 2.5, 6, 5.25} @ qBits=5 -- the ASYM accumulate re-pin
               (testAccumulateAsymRescaleMatchesFloatReference's
               decode+increment reference); all-positive band, so the nudge
               is LIVE here: scale 3.5/31 -> 6/31, zp 22 -> 0.

Grouped fixtures (group-quant PR4 Task 2 -- quantizeFloatToAsym's grouped
path + convertAsymTensorToFloatTensor's grouped dequant):
  AsymGrouped  12 floats @ qBits=4, groupSize=4 (3 groups): one all-positive
               group (zp 0), one span-zero group (mid zp), one all-negative
               group (zp at the code ceiling 15). Self-checks (collapse
               discriminators): per-group zps are PAIRWISE DISTINCT (so a
               dequant that reads zeroPoints[0] for every group diverges on
               groups 1..2 -- mutation (ii) guard) and every per-group scale
               differs from the WHOLE-TENSOR derivation's scale (so a phase-1
               min/max over the whole tensor instead of the group diverges on
               every group -- mutation (i) guard). Also emits the round-trip
               dequant array (dequant_asym_grouped over the pinned codes),
               the expected output of ASYM(grouped) -> FLOAT32.
  AsymTorchXCheck
               8x4 row-major weight (32 floats) @ qBits=8, groupSize=4 -- one
               group per output row (per-out-channel, axis=0), the layout a
               GEMM weight groups over. Cross-checked in-generator against
               torch.quantize_per_channel(w, scales, zps, axis=0,
               dtype=torch.quint8).int_repr(). ZP-CONVENTION MAPPING: torch's
               quint8 zero_point is already CODE-domain (q = clamp(round(v/s)
               + zp, 0, 255), dequant = (q - zp)*s), the same affine we use --
               BUT torch does NOT derive the grid with our zero-inclusion
               nudge (torch observers have their own reduce_range/eps rules),
               so the grid is derived with OUR nudged rule and passed TO
               torch, which takes explicit scales/zero_points and derives
               nothing. The remaining divergence is rounding only: torch
               rounds half-to-even, we round half-away-from-zero, so a code
               may differ by exactly 1 on an exact x.5 quotient (genuine tie);
               any other mismatch aborts generation.

Run via `uv run` (CMake wires this automatically).
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "goldgen"))

from sym_gold import (dequant_asym_grouped, emit_float_array, emit_float_scalar,
                      emit_int32_array, emit_int32_scalar, quantize_asym_grouped,
                      quantize_asym_nudged, quantize_asym_old_value_domain)

FIXTURES = [
    ("f1SpanZero", [1.0, 2.0, 3.0, 4.0, -1.0, -2.0], 5),
    ("f2NegBand16", [-10.0, -1.0, -5.5, -2.5, -9.25, -3.75], 16),
    ("f3NegFar", [-5000000.0, -4999999.5], 8),
    ("f4ClampTie", [-1.5, 5.5, 0.0, 2.0], 3),
    ("f5SymDeq", [5.0, -4.0, 2.0, -1.0, 3.0, -5.0], 5),
    ("f6AccumRef", [3.0, 2.5, 6.0, 5.25], 5),
]


def emit_fixture(parts, name, values, q_bits):
    codes, scale, zp = quantize_asym_nudged(values, q_bits)
    old_codes, _old_scale, old_zp = quantize_asym_old_value_domain(values, q_bits)
    assert (codes != old_codes) or (zp != old_zp), (
        f"{name}: nudged grid is indistinguishable from the old value-domain "
        f"grid -- pin would be vacuous (codes {codes}, zp {zp})")
    pre = f"asym_nudged_{name}"
    parts.append(emit_float_array(f"input_{pre}", __import__("torch").tensor(values)))
    parts.append(emit_int32_array(f"codes_{pre}", __import__("torch").tensor(codes, dtype=__import__("torch").int32)))
    parts.append(emit_float_scalar(f"scale_{pre}", scale))
    parts.append(emit_int32_scalar(f"zp_{pre}", zp))
    parts.append(emit_int32_scalar(f"qBits_{pre}", q_bits))
    return codes, scale, zp


def fixture_asym_grouped():
    # 3 groups of 4 @ qBits=4 (qMax 15), bands chosen so the zps are pairwise
    # distinct AND every group scale differs from the whole-tensor derivation:
    #   g0 all-positive  [0, 3]   -> scale 3/15  = 0.2f, zp 0
    #   g1 span-zero     [-1, 2]  -> scale 3/15  = 0.2f, zp 5
    #   g2 all-negative  [-6, 0]  -> scale 6/15  = 0.4f, zp 15 (code ceiling)
    #   whole tensor     [-6, 3]  -> scale 9/15  = 0.6f, zp 10
    raw = [0.5, 1.5, 3.0, 2.0,
          -1.0, 2.0, 0.5, -0.5,
          -6.0, -1.5, -3.0, -0.75]
    q_bits, group_size = 4, 4
    codes, scales, zps = quantize_asym_grouped(raw, q_bits, group_size)

    # Collapse discriminators (see module docstring):
    assert len(set(zps)) == len(zps), (
        f"AsymGrouped: per-group zps {zps} are not pairwise distinct -- the "
        f"zeroPoints[0]-lookup mutation (ii) would not be discriminated")
    _wc, wscales, wzps = quantize_asym_grouped(raw, q_bits, len(raw))
    for g, s in enumerate(scales):
        assert s != wscales[0], (
            f"AsymGrouped: group {g} scale {s} equals the whole-tensor scale "
            f"{wscales[0]} -- the whole-tensor-min/max mutation (i) would not "
            f"be discriminated on this group")

    dequant = dequant_asym_grouped(codes, scales, zps, group_size)
    return {"input": raw, "codes": codes, "scales": scales, "zps": zps,
           "dequant": dequant, "qBits": q_bits, "groupSize": group_size,
           "numGroups": len(scales)}


def fixture_asym_torch_cross_check():
    # 8x4 row-major weight, one group per row (per-out-channel axis=0),
    # qBits=8 (quint8 code domain [0, 255]). Rows mix all-positive (zp 0),
    # all-negative (zp 255) and span-zero bands.
    raw = [0.3, -0.7, 1.1, -1.3,
          2.0, 0.1, 0.05, 0.9,
          -3.3, -3.1, -0.15, -0.02,
          0.6, -0.6, 0.61, -0.59,
          4.4, -1.1, 0.02, 0.03,
          1.9, -1.85, 0.5001, -0.4999,
          0.11, 0.12, 0.13, 0.14,
          -5.5, 5.6, 0.001, -0.001]
    q_bits, group_size, num_groups = 8, 4, 8
    codes, scales, zps = quantize_asym_grouped(raw, q_bits, group_size)
    assert 0 in zps and (2 ** q_bits - 1) in zps, (
        f"AsymTorchXCheck: zps {zps} miss the {0}/{2 ** q_bits - 1} band "
        f"edges -- fixture no longer exercises the nudge at both ends")

    # torch cross-check: OUR nudged grid handed TO torch (it derives nothing;
    # see the module docstring for the zp-convention mapping).
    w = torch.tensor(raw, dtype=torch.float32).reshape(num_groups, group_size)
    torch_q = torch.quantize_per_channel(
        w, torch.tensor(scales, dtype=torch.float64),
        torch.tensor(zps, dtype=torch.int64), axis=0, dtype=torch.quint8)
    torch_codes = torch_q.int_repr().flatten().tolist()
    for i, (c, t) in enumerate(zip(codes, torch_codes)):
        if c == t:
            continue
        # Only an exact half-integer quotient can legitimately disagree
        # between round-half-away-from-zero (ours) and round-half-to-even
        # (torch) -- and only by exactly 1 code.
        g = i // group_size
        quotient = (torch.tensor(raw[i], dtype=torch.float32) /
                    torch.tensor(scales[g], dtype=torch.float32)).item()
        is_tie = abs(abs(quotient) - (abs(quotient) // 1) - 0.5) < 1e-9
        assert is_tie and abs(c - t) == 1, (
            f"AsymTorchXCheck: element {i} mismatches torch.quantize_per_channel "
            f"beyond a rounding-tie boundary (ours={c}, torch's={t}, "
            f"quotient={quotient}) -- investigate, do not loosen this assert")

    dequant = dequant_asym_grouped(codes, scales, zps, group_size)
    return {"input": raw, "codes": codes, "scales": scales, "zps": zps,
           "dequant": dequant, "qBits": q_bits, "groupSize": group_size,
           "numGroups": num_groups}


def emit_grouped_fixture(parts, prefix, fx):
    parts.append(emit_float_array(f"k{prefix}Input", torch.tensor(fx["input"], dtype=torch.float32)))
    parts.append(emit_int32_array(f"k{prefix}Codes", torch.tensor(fx["codes"], dtype=torch.int32)))
    parts.append(emit_float_array(f"k{prefix}Scales", torch.tensor(fx["scales"], dtype=torch.float32)))
    parts.append(emit_int32_array(f"k{prefix}Zps", torch.tensor(fx["zps"], dtype=torch.int32)))
    parts.append(emit_float_array(f"k{prefix}Dequant", torch.tensor(fx["dequant"], dtype=torch.float32)))
    parts.append(emit_int32_scalar(f"k{prefix}QBits", fx["qBits"]))
    parts.append(emit_int32_scalar(f"k{prefix}GroupSize", fx["groupSize"]))
    parts.append(emit_int32_scalar(f"k{prefix}NumGroups", fx["numGroups"]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    parts = [
        "// AUTOGENERATED by generate_expected_asym_nudged.py — DO NOT EDIT\n",
        "#ifndef ODT_EXPECTED_ASYM_NUDGED_H\n",
        "#define ODT_EXPECTED_ASYM_NUDGED_H\n",
        "#include <stdint.h>\n",
        "#include <stdlib.h>\n\n",
    ]

    results = {}
    for name, values, q_bits in FIXTURES:
        results[name] = emit_fixture(parts, name, values, q_bits)

    # -- cross-fixture self-checks (see module docstring) --
    codes, scale, zp = results["f1SpanZero"]
    assert zp == 10 and codes == [15, 20, 26, 31, 5, 0], (
        f"f1SpanZero drifted from the hand derivation: codes {codes}, zp {zp}")
    _codes2, _scale2, zp2 = results["f2NegBand16"]
    assert zp2 == 2 ** 16 - 1, f"f2NegBand16: zp {zp2} != uint16 ceiling 65535"
    assert _codes2[0] == 0, f"f2NegBand16: min value must land on code 0, got {_codes2[0]}"
    codes3, _scale3, zp3 = results["f3NegFar"]
    assert zp3 == 255, f"f3NegFar: zp {zp3} != 255"
    codes4, scale4, zp4 = results["f4ClampTie"]
    assert scale4 == 1.0 and zp4 == 2, f"f4ClampTie: scale {scale4}, zp {zp4}"
    # the clamp must actually FIRE for v=5.5: un-clamped code is 6 + 2 = 8 > 7
    assert codes4[1] == 7, f"f4ClampTie: clamped top code {codes4[1]} != 7"
    from sym_gold import round_half_away
    import torch
    unclamped = int(round_half_away(torch.tensor(5.5 / scale4, dtype=torch.float32)).item()) + zp4
    assert unclamped == 8, (
        f"f4ClampTie: un-clamped top code {unclamped} != 8 -- fixture no "
        f"longer exercises the encode clamp (mutation (ii) guard vacuous)")
    assert codes4[2] == zp4, "f4ClampTie: 0.0 must encode to code == zp"

    emit_grouped_fixture(parts, "AsymGrouped", fixture_asym_grouped())
    emit_grouped_fixture(parts, "AsymTorchXCheck", fixture_asym_torch_cross_check())

    parts.append("\n#endif // ODT_EXPECTED_ASYM_NUDGED_H\n")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(parts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
