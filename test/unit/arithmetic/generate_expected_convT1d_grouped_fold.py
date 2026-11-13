#!/usr/bin/env python3
"""Generate expected_convT1d_grouped_fold.h for UnitTestConvTranspose1dKernel's
grouped-scatter fold tests (linq fold T3).

Every gold comes from sym_gold.convt_grouped_fold_ref, which computes the fold
in Python ints from the canonical (m, e) of each scale -- independent of the C
kernel. HALF_AWAY only: an SR gold needs the C RNG stream and is recorded from
the kernel instead (golden_convT1d_scatter.h).

Fixtures:
  ctGold    the kernel test's LCG fixture (ctGoldRunGrouped): batch 2, Cin 2,
            Cout 7, K 5, conv groups 1, 10 quantization groups of 7 (a group
            straddles the kernel rows), EXPLICIT padding, three geometries.
  channel   batch 2, Cin 2, Cout 3, K 5, Lin 4, stride 2, dilation 2, VALID,
            groupSize 15 = Cout*K: each input channel's weight block is one
            group (the kernel's cursor-free path).
  group3    the same operands with groupSize 3: groups straddle the K = 5
            rows (the kernel's segment walk).
  cgAligned conv groups 2: batch 2, Cin 4, Cout 6 (outChPerGroup 3), K 5,
            Lin 4, stride 2, dilation 2, VALID; groupSize 15 = outChPerGroup*K
            (cursor-free path across both conv-group slabs).
  cgCursor  the same operands with groupSize 4 (segment walk, groups
            straddle rows and input-channel blocks).

Each fixture asserts it is not vacuous: several distinct group exponents, a
fold that actually rounds, a zero input code, and a gold that differs from the
legacy float path's (sym_gold.convT1d_grouped_ref) where that ref applies.

Run via `uv run` (CMake wires this automatically).
"""
import argparse
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "goldgen"))

from dyadic_scale import canon, decompose  # noqa: E402
from sym_gold import (convt_grouped_fold_ref, convT1d_grouped_ref, emit_float_array,  # noqa: E402
                      emit_float_scalar, emit_int32_array, emit_int32_scalar,
                      fold_result_bits, mul_shift_round_half_away, rescale_f32)

def rescaled_bias_seeds(bias, bias_scale, acc_scale):
    return [rescale_f32(b, bias_scale, acc_scale) for b in bias]


def folds_round(x, w_scales, frac):
    """True when some (code, group) fold drops a nonzero fraction."""
    ws = [decompose(s) for s in w_scales]
    e_max = max(e for _, e in ws)
    return any(v != 0 and (v * m) % (1 << (e_max - e + 15 - frac)) != 0
               for v in x for m, e in ws)


def assert_fold_rounds(x, w_scales, frac):
    """At least one fold drops a nonzero fraction (the rounding is exercised)."""
    assert folds_round(x, w_scales, frac), "no fold rounds: the fixture cannot see a rounding bug"


# ---- ctGold: the C test's LCG fixture, reproduced bit for bit ----

def lcg(state):
    state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
    return state, state >> 8


def gold_code(state, r):
    state, v = lcg(state)
    v = (v % (2 * r)) - r
    return state, (v + 1 if v >= 0 else v)


CT_GEOMS = [(5, 12, 2, 2, 3), (2, 5, 2, 2, 3), (4, 11, 3, 2, 4)]  # lin, lout, stride, dil, pad
CT_B, CT_CIN, CT_COUT, CT_K, CT_NUM_GROUPS, CT_GROUP_SIZE = 2, 2, 7, 5, 10, 7
CT_X_SCALE, CT_X_BITS, CT_W_BITS = 0.25, 8, 8
CT_BIAS_SCALE = canon(0.3)


def ct_gold_fixture(lin):
    s = 7717 + lin
    x, w, bias = [], [], []
    for _ in range(CT_B * CT_CIN * lin):
        s, v = gold_code(s, 100)
        x.append(v)
    for _ in range(CT_CIN * CT_COUT * CT_K):
        s, v = gold_code(s, 100)
        w.append(v)
    for _ in range(CT_COUT):
        s, v = gold_code(s, 200)
        bias.append(v)
    return x, w, bias


def ct_gold_scales():
    return [canon((5 + 3 * ((g * 7) % CT_NUM_GROUPS)) / 64.0) for g in range(CT_NUM_GROUPS)]


def ct_gold():
    scales = ct_gold_scales()
    w_min, w_max = -(1 << (CT_W_BITS - 1)), (1 << (CT_W_BITS - 1)) - 1
    outs, bits, ref0 = [], [], None
    for lin, lout, stride, dil, pad in CT_GEOMS:
        x, w, bias = ct_gold_fixture(lin)
        r = convt_grouped_fold_ref(x, CT_X_SCALE, CT_X_BITS, w, scales, CT_GROUP_SIZE, w_min,
                                   w_max, CT_B, CT_CIN, CT_COUT, CT_K, lin, stride, dil,
                                   pad_left=pad, out_len=lout, bias_mantissas=bias,
                                   bias_scale=CT_BIAS_SCALE)
        # k/64 scales carry >= 12 trailing zero mantissa bits, so every fold is
        # exact here (rounding is the channel/group3 fixtures' job; the SR gold
        # recorded from the kernel relies on it)
        assert not folds_round(x, scales, r["frac_bits"]), "ctGold: a fold rounds"
        seeds = rescaled_bias_seeds(bias, CT_BIAS_SCALE, r["acc_scale"])
        bits.append(fold_result_bits(CT_X_BITS, r["frac_bits"], CT_CIN * CT_K, w_min, w_max,
                                     seeds))
        outs.append(r["out"])
        if ref0 is None:
            ref0 = r
        assert (r["acc_m"], r["acc_e"], r["frac_bits"]) == (ref0["acc_m"], ref0["acc_e"],
                                                            ref0["frac_bits"])
    return outs, ref0, bits


# ---- channel / group3: VALID stride 2 dilation 2, seeded operands ----

FX_B, FX_CIN, FX_COUT, FX_K, FX_LIN, FX_STRIDE, FX_DIL = 2, 2, 3, 5, 4, 2, 2
FX_X_BITS, FX_W_BITS = 8, 8
FX_X_SCALE = canon(0.37)
FX_BIAS_SCALE = canon(0.3)
FX_CHANNEL_SCALES = [canon(0.021), canon(0.093)]
FX_GROUP3_SCALES = [canon(v) for v in (0.021, 0.093, 0.0031, 0.05, 0.17, 0.011, 0.0007, 0.066,
                                       0.12, 0.04)]


def fx_operands():
    rng = random.Random(20261111)
    x = [rng.randint(-120, 120) for _ in range(FX_B * FX_CIN * FX_LIN)]
    x[3] = 0  # the zero rule: never folded, contributes nothing
    x[9] = 0
    w = [rng.randint(-127, 127) for _ in range(FX_CIN * FX_COUT * FX_K)]
    bias = [rng.randint(-300, 300) for _ in range(FX_COUT)]
    return x, w, bias


def fx_fixture(name, scales, group_size):
    x, w, bias = fx_operands()
    assert len(w) % group_size == 0 and len(w) // group_size == len(scales)
    w_min, w_max = -(1 << (FX_W_BITS - 1)), (1 << (FX_W_BITS - 1)) - 1
    r = convt_grouped_fold_ref(x, FX_X_SCALE, FX_X_BITS, w, scales, group_size, w_min, w_max,
                               FX_B, FX_CIN, FX_COUT, FX_K, FX_LIN, FX_STRIDE, FX_DIL,
                               bias_mantissas=bias, bias_scale=FX_BIAS_SCALE)
    assert len({decompose(s)[1] for s in scales}) > 1, f"{name}: one group exponent only"
    assert_fold_rounds(x, scales, r["frac_bits"])
    legacy, _, _ = convT1d_grouped_ref(x, FX_X_SCALE, w, scales, group_size, FX_B, FX_CIN,
                                       FX_COUT, FX_K, FX_LIN, FX_STRIDE, FX_DIL, 0,
                                       bias_mantissas=bias, bias_scale=FX_BIAS_SCALE)
    assert legacy != r["out"], f"{name}: fold gold equals the legacy float path's (vacuous)"
    seeds = rescaled_bias_seeds(bias, FX_BIAS_SCALE, r["acc_scale"])
    r["bits"] = fold_result_bits(FX_X_BITS, r["frac_bits"], FX_CIN * FX_K, w_min, w_max, seeds)
    r.update({"x": x, "w": w, "bias": bias, "scales": scales, "group_size": group_size})
    return r


CG_B, CG_CIN, CG_COUT, CG_GROUPS = 2, 4, 6, 2
CG_ALIGNED_SCALES = [canon(v) for v in (0.021, 0.093, 0.0031, 0.05)]
CG_CURSOR_SCALES = [canon(v) for v in (0.021, 0.093, 0.0031, 0.05, 0.17, 0.011, 0.0007, 0.066,
                                       0.12, 0.04, 0.029, 0.0045, 0.081, 0.013, 0.055)]


def cg_operands():
    rng = random.Random(20261112)
    x = [rng.randint(-120, 120) for _ in range(CG_B * CG_CIN * FX_LIN)]
    x[5] = 0
    w = [rng.randint(-127, 127) for _ in range(CG_CIN * (CG_COUT // CG_GROUPS) * FX_K)]
    bias = [rng.randint(-300, 300) for _ in range(CG_COUT)]
    return x, w, bias


def cg_fixture(name, scales, group_size):
    x, w, bias = cg_operands()
    assert len(w) % group_size == 0 and len(w) // group_size == len(scales)
    w_min, w_max = -(1 << (FX_W_BITS - 1)), (1 << (FX_W_BITS - 1)) - 1
    r = convt_grouped_fold_ref(x, FX_X_SCALE, FX_X_BITS, w, scales, group_size, w_min, w_max,
                               CG_B, CG_CIN, CG_COUT, FX_K, FX_LIN, FX_STRIDE, FX_DIL,
                               bias_mantissas=bias, bias_scale=FX_BIAS_SCALE,
                               conv_groups=CG_GROUPS)
    assert_fold_rounds(x, scales, r["frac_bits"])
    in_per = CG_CIN // CG_GROUPS
    seeds = rescaled_bias_seeds(bias, FX_BIAS_SCALE, r["acc_scale"])
    r["bits"] = fold_result_bits(FX_X_BITS, r["frac_bits"], in_per * FX_K, w_min, w_max, seeds)
    r.update({"x": x, "w": w, "bias": bias, "scales": scales, "group_size": group_size})
    return r


def assert_group3_straddles():
    """groupSize 3 against K = 5: some group spans two tap rows."""
    rows = {g: {i // FX_K for i in range(g * 3, g * 3 + 3)} for g in range(10)}
    assert any(len(r) > 1 for r in rows.values()), "group3: no group straddles a row"


def emit_fx(parts, prefix, r):
    parts.append(emit_float_array(f"k{prefix}WScales", torch.tensor(r["scales"],
                                                                   dtype=torch.float64)))
    parts.append(emit_int32_scalar(f"k{prefix}GroupSize", r["group_size"]))
    parts.append(emit_int32_scalar(f"k{prefix}NumGroups", len(r["scales"])))
    parts.append(emit_int32_array(f"k{prefix}Out", torch.tensor(r["out"])))
    parts.append(emit_int32_scalar(f"k{prefix}AccM", r["acc_m"]))
    parts.append(emit_int32_scalar(f"k{prefix}AccE", r["acc_e"]))
    parts.append(emit_int32_scalar(f"k{prefix}FracBits", r["frac_bits"]))
    parts.append(emit_int32_scalar(f"k{prefix}Bits", r["bits"]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    # the half-away helper itself: ties go away from zero on both signs
    assert mul_shift_round_half_away(3, 1, 1) == 2 and mul_shift_round_half_away(-3, 1, 1) == -2
    assert mul_shift_round_half_away(5, 3, 2) == 4 and mul_shift_round_half_away(-5, 3, 2) == -4

    ct_outs, ct_ref, ct_bits = ct_gold()
    channel = fx_fixture("channel", FX_CHANNEL_SCALES, FX_COUT * FX_K)
    group3 = fx_fixture("group3", FX_GROUP3_SCALES, 3)
    assert_group3_straddles()
    assert channel["out"] != group3["out"], "channel and group3 golds coincide (vacuous)"
    cg_aligned = cg_fixture("cgAligned", CG_ALIGNED_SCALES, (CG_COUT // CG_GROUPS) * FX_K)
    cg_cursor = cg_fixture("cgCursor", CG_CURSOR_SCALES, 4)
    assert cg_aligned["out"] != cg_cursor["out"], "conv-groups golds coincide (vacuous)"
    # the kernel takes the aligned loop iff groupSize % (outChPerGroup * K) == 0
    assert 4 % ((CG_COUT // CG_GROUPS) * FX_K) != 0, "cgCursor must not be channel-aligned"

    x, w, bias = fx_operands()
    parts = [
        "// AUTOGENERATED by generate_expected_convT1d_grouped_fold.py -- DO NOT EDIT\n",
        "#ifndef ODT_EXPECTED_CONVT1D_GROUPED_FOLD_H\n",
        "#define ODT_EXPECTED_CONVT1D_GROUPED_FOLD_H\n",
        "#include <stddef.h>\n",
        "#include <stdint.h>\n\n",
        "/* ctGold (ctGoldRunGrouped's LCG fixture), fold path, HALF_AWAY */\n",
    ]
    for suffix, out in zip("ABC", ct_outs):
        parts.append(emit_int32_array(f"kCtFoldHa{suffix}", torch.tensor(out)))
    parts.append(emit_int32_scalar("kCtFoldAccM", ct_ref["acc_m"]))
    parts.append(emit_int32_scalar("kCtFoldAccE", ct_ref["acc_e"]))
    parts.append(emit_int32_scalar("kCtFoldFracBits", ct_ref["frac_bits"]))
    parts.append(emit_int32_array("kCtFoldBits", torch.tensor(ct_bits)))
    parts.append("\n/* channel / group3: shared operands, VALID stride 2 dilation 2 */\n")
    parts.append(emit_int32_array("kConvTFoldX", torch.tensor(x)))
    parts.append(emit_int32_array("kConvTFoldW", torch.tensor(w)))
    parts.append(emit_int32_array("kConvTFoldBias", torch.tensor(bias)))
    parts.append(emit_float_scalar("kConvTFoldXScale", FX_X_SCALE))
    parts.append(emit_float_scalar("kConvTFoldBiasScale", FX_BIAS_SCALE))
    parts.append(emit_int32_scalar("kConvTFoldOutLen", channel["out_len"]))
    emit_fx(parts, "ConvTFoldChannel", channel)
    emit_fx(parts, "ConvTFoldGroup3", group3)
    cx, cw, cb = cg_operands()
    parts.append("\n/* conv groups 2: Cin 4, Cout 6, same geometry */\n")
    parts.append(emit_int32_array("kConvTFoldCgX", torch.tensor(cx)))
    parts.append(emit_int32_array("kConvTFoldCgW", torch.tensor(cw)))
    parts.append(emit_int32_array("kConvTFoldCgBias", torch.tensor(cb)))
    emit_fx(parts, "ConvTFoldCgAligned", cg_aligned)
    emit_fx(parts, "ConvTFoldCgCursor", cg_cursor)
    parts.append("\n#endif // ODT_EXPECTED_CONVT1D_GROUPED_FOLD_H\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(parts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
