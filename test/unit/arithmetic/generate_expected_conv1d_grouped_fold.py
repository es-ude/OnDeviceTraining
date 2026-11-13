#!/usr/bin/env python3
"""Generate expected_conv1d_grouped_fold.h for UnitTestConv1dKernel's grouped
gather fold tests (linq fold T4).

Every gold comes from sym_gold.conv1d_grouped_fold_ref, which computes the
fold in Python ints from the canonical (m, e) of each scale -- independent of
the C kernel. HALF_AWAY only: an SR gold needs the C RNG stream and is recorded
from the kernel instead (golden_conv1d_grouped.h).

Fixtures:
  cgGold    the kernel test's LCG fixture (cgGoldRun), reproduced bit for bit:
            batch 2, Cin 2, K 5, L 8, stride 2, dilation 2, conv groups 1,
            twelve (padding, groupSize, outC) cases.
  cg2       cg2GoldRun's LCG fixture: Cin 4, Cout 4, conv groups 2, EXPLICIT 3;
            group sizes 4, 8 (segment walk) and 10, 20 (channel-aligned loop).
  channel   batch 2, Cin 2, Cout 3, K 5, L 8, stride 2, dilation 2, EXPLICIT 9
            (output 0 has no valid tap), groupSize 10 = Cin*K: one group per
            output channel (the cursor-free loop). Non-dyadic group scales, so
            the combines round.
  group2    the same operands with groupSize 2: groups straddle the rows and
            every window combines several runs (the segment walk).
  cgAligned conv groups 2: Cin 4, Cout 4, EXPLICIT 3, groupSize 10 (aligned).
  cgCursor  the same operands with groupSize 4 (segment walk).

Each new fixture asserts it is not vacuous: several group exponents, a
combine that rounds (a truncating combine changes the gold), a zero input
code, and for channel/group2 a window with no valid tap whose output is the
bias seed alone.

Run via `uv run` (CMake wires this automatically).
"""
import argparse
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "goldgen"))

import sym_gold  # noqa: E402
from dyadic_scale import canon, decompose  # noqa: E402
from sym_gold import (conv1d_grouped_fold_ref, emit_float_array, emit_float_scalar,  # noqa: E402
                      emit_int32_array, emit_int32_scalar, fold_result_bits,
                      window_geometry_1d, window_slice_1d_full)

K, STRIDE, DIL, L = 5, 2, 2, 8
X_BITS, W_BITS = 8, 8
W_MIN, W_MAX = -(1 << (W_BITS - 1)), (1 << (W_BITS - 1)) - 1


def lcg(state):
    state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
    return state, state >> 8


def gold_code(state, r):
    state, v = lcg(state)
    v = (v % (2 * r)) - r
    return state, (v + 1 if v >= 0 else v)


def gold_scales(num_groups):
    return [canon((5 + 3 * ((g * 7) % num_groups)) / 64.0) for g in range(num_groups)]


def run_ref(x, x_scale, w, scales, group_size, batch, cin, cout, pad_type, padding, bias,
            bias_scale, conv_groups=1, self_check=True):
    r = conv1d_grouped_fold_ref(x, x_scale, X_BITS, w, scales, group_size, W_MIN, W_MAX, batch,
                                cin, cout, K, L, STRIDE, DIL, pad_type, padding,
                                bias_mantissas=bias, bias_scale=bias_scale,
                                conv_groups=conv_groups, self_check=self_check)
    r["bits"] = fold_result_bits(X_BITS, r["frac_bits"], (cin // conv_groups) * K, W_MIN, W_MAX,
                                 r["seeds"])
    return r


def truncation_replay(x, x_scale, w, scales, group_size, batch, cin, cout, pad_type, padding,
                      conv_groups=1):
    """(half-away gold, a replay whose combine truncates toward zero, whether
    any combine dropped a nonzero fraction), bias-free."""
    def ref():
        return conv1d_grouped_fold_ref(x, x_scale, X_BITS, w, scales, group_size, W_MIN, W_MAX,
                                       batch, cin, cout, K, L, STRIDE, DIL, pad_type, padding,
                                       conv_groups=conv_groups, self_check=False)["out"]

    dropped = []

    def trunc(v, m, s):
        if abs(v * m) & ((1 << s) - 1):
            dropped.append(True)
        q = abs(v * m) >> s
        return q if v * m >= 0 else -q

    half_away = ref()
    saved = sym_gold.mul_shift_round_half_away
    sym_gold.mul_shift_round_half_away = trunc
    try:
        truncated = ref()
    finally:
        sym_gold.mul_shift_round_half_away = saved
    return half_away, truncated, bool(dropped)


def combines_exact(*args, **kwargs):
    """True when no combine drops a nonzero fraction."""
    return not truncation_replay(*args, **kwargs)[2]


# ---- cgGold: cgGoldRun's LCG fixture ----

CG_B, CG_CIN = 2, 2
CG_X_SCALE = 0.25
CG_BIAS_SCALE = canon(0.3)
CG_CASES = [("SAME", 0, 3, 3, 4), ("SAME", 0, 4, 2, 4), ("SAME", 0, 7, 7, 4), ("SAME", 0, 10, 3, 4),
            ("EXPLICIT", 3, 3, 3, 3), ("EXPLICIT", 3, 4, 2, 3), ("EXPLICIT", 3, 7, 7, 3),
            ("EXPLICIT", 3, 10, 3, 3), ("EXPLICIT", 9, 3, 3, 9), ("EXPLICIT", 9, 4, 2, 9),
            ("EXPLICIT", 9, 7, 7, 9), ("EXPLICIT", 9, 10, 3, 9)]


def cg_gold_case(pad_type, padding, group_size, out_c, out_len):
    s = 4049 + group_size * 16 + padding
    x, w, bias = [], [], []
    for _ in range(CG_B * CG_CIN * L):
        s, v = gold_code(s, 100)
        x.append(v)
    w_elems = out_c * CG_CIN * K
    for _ in range(w_elems):
        s, v = gold_code(s, 100)
        w.append(v)
    for _ in range(out_c):
        s, v = gold_code(s, 200)
        bias.append(v)
    scales = gold_scales(w_elems // group_size)
    r = run_ref(x, CG_X_SCALE, w, scales, group_size, CG_B, CG_CIN, out_c, pad_type, padding,
                bias, CG_BIAS_SCALE)
    assert r["out_len"] == out_len
    # k/64 scales carry >= 12 trailing zero mantissa bits: every combine is exact
    # here (the SR gold recorded from the kernel relies on it; rounding is the
    # channel/group2 fixtures' job)
    assert combines_exact(x, CG_X_SCALE, w, scales, group_size, CG_B, CG_CIN, out_c, pad_type, padding)
    return r


# ---- cg2: cg2GoldRun's LCG fixture (conv groups 2) ----

CG2_B, CG2_CIN, CG2_COUT, CG2_GROUPS, CG2_PAD = 2, 4, 4, 2, 3


def cg2_gold_case(group_size):
    s = 7121 + group_size
    x, w, bias = [], [], []
    for _ in range(CG2_B * CG2_CIN * L):
        s, v = gold_code(s, 100)
        x.append(v)
    w_elems = CG2_COUT * (CG2_CIN // CG2_GROUPS) * K
    for _ in range(w_elems):
        s, v = gold_code(s, 100)
        w.append(v)
    for _ in range(CG2_COUT):
        s, v = gold_code(s, 200)
        bias.append(v)
    scales = gold_scales(w_elems // group_size)
    r = run_ref(x, CG_X_SCALE, w, scales, group_size, CG2_B, CG2_CIN, CG2_COUT, "EXPLICIT",
                CG2_PAD, bias, CG_BIAS_SCALE, conv_groups=CG2_GROUPS)
    assert r["out_len"] == 3
    assert combines_exact(x, CG_X_SCALE, w, scales, group_size, CG2_B, CG2_CIN, CG2_COUT,
                          "EXPLICIT", CG2_PAD, conv_groups=CG2_GROUPS)
    return r


# ---- channel / group2: seeded operands, non-dyadic scales ----

FX_B, FX_CIN, FX_COUT, FX_PAD = 2, 2, 3, 9
FX_X_SCALE = canon(0.37)
FX_BIAS_SCALE = canon(0.3)
FX_CHANNEL_SCALES = [canon(v) for v in (0.021, 0.093, 0.0031)]
FX_GROUP2_SCALES = [canon(v) for v in (0.021, 0.093, 0.0031, 0.05, 0.17, 0.011, 0.0007, 0.066,
                                       0.12, 0.04, 0.029, 0.0045, 0.081, 0.013, 0.055)]


def fx_operands():
    rng = random.Random(20261113)
    x = [rng.randint(-120, 120) for _ in range(FX_B * FX_CIN * L)]
    x[3] = 0
    x[17] = 0
    w = [rng.randint(-127, 127) for _ in range(FX_COUT * FX_CIN * K)]
    bias = [rng.randint(-300, 300) for _ in range(FX_COUT)]
    return x, w, bias


def assert_rounds(x, w, scales, group_size, batch, cin, cout, pad_type, padding, conv_groups=1):
    half_away, truncated, _ = truncation_replay(x, FX_X_SCALE, w, scales, group_size, batch, cin,
                                                cout, pad_type, padding, conv_groups)
    assert truncated != half_away, (
        "no combine rounds up: the fixture cannot see a rounding bug")


def fx_fixture(name, scales, group_size):
    x, w, bias = fx_operands()
    assert len(w) == group_size * len(scales)
    r = run_ref(x, FX_X_SCALE, w, scales, group_size, FX_B, FX_CIN, FX_COUT, "EXPLICIT", FX_PAD,
                bias, FX_BIAS_SCALE)
    assert len({decompose(s)[1] for s in scales}) > 1, f"{name}: one group exponent only"
    assert_rounds(x, w, scales, group_size, FX_B, FX_CIN, FX_COUT, "EXPLICIT", FX_PAD)
    geom = window_geometry_1d(L, K, STRIDE, DIL, "EXPLICIT", FX_PAD)
    empty = [p for p in range(r["out_len"]) if window_slice_1d_full(geom, p)[2] == 0]
    assert empty, f"{name}: no window without a valid tap"
    for b in range(FX_B):
        for oc in range(FX_COUT):
            for p in empty:
                flat = (b * FX_COUT + oc) * r["out_len"] + p
                assert r["out"][flat] == r["seeds"][oc] and r["combines"][flat] == 0
    nobias = run_ref(x, FX_X_SCALE, w, scales, group_size, FX_B, FX_CIN, FX_COUT, "EXPLICIT",
                     FX_PAD, None, None)
    r["combines_total"] = sum(nobias["combines"])
    r["bits_nobias"] = nobias["bits"]
    r.update({"scales": scales, "group_size": group_size})
    return r


CGF_B, CGF_CIN, CGF_COUT, CGF_GROUPS, CGF_PAD = 2, 4, 4, 2, 3
CGF_ALIGNED_SCALES = [canon(v) for v in (0.021, 0.093, 0.0031, 0.05)]
CGF_CURSOR_SCALES = [canon(v) for v in (0.021, 0.093, 0.0031, 0.05, 0.17, 0.011, 0.0007, 0.066,
                                        0.12, 0.04)]


def cgf_operands():
    rng = random.Random(20261114)
    x = [rng.randint(-120, 120) for _ in range(CGF_B * CGF_CIN * L)]
    x[5] = 0
    w = [rng.randint(-127, 127) for _ in range(CGF_COUT * (CGF_CIN // CGF_GROUPS) * K)]
    bias = [rng.randint(-300, 300) for _ in range(CGF_COUT)]
    return x, w, bias


def cgf_fixture(scales, group_size):
    x, w, bias = cgf_operands()
    assert len(w) == group_size * len(scales)
    r = run_ref(x, FX_X_SCALE, w, scales, group_size, CGF_B, CGF_CIN, CGF_COUT, "EXPLICIT",
                CGF_PAD, bias, FX_BIAS_SCALE, conv_groups=CGF_GROUPS)
    assert_rounds(x, w, scales, group_size, CGF_B, CGF_CIN, CGF_COUT, "EXPLICIT", CGF_PAD,
                  CGF_GROUPS)
    r.update({"scales": scales, "group_size": group_size})
    return r


def emit_fx(parts, prefix, r):
    parts.append(emit_float_array(f"k{prefix}WScales", torch.tensor(r["scales"],
                                                                   dtype=torch.float64)))
    parts.append(emit_int32_scalar(f"k{prefix}GroupSize", r["group_size"]))
    parts.append(emit_int32_scalar(f"k{prefix}NumGroups", len(r["scales"])))
    parts.append(emit_int32_array(f"k{prefix}Out", torch.tensor(r["out"])))
    parts.append(emit_int32_array(f"k{prefix}Seeds", torch.tensor(r["seeds"])))
    parts.append(emit_int32_scalar(f"k{prefix}AccM", r["acc_m"]))
    parts.append(emit_int32_scalar(f"k{prefix}AccE", r["acc_e"]))
    parts.append(emit_int32_scalar(f"k{prefix}FracBits", r["frac_bits"]))
    parts.append(emit_int32_scalar(f"k{prefix}Bits", r["bits"]))


def emit_cases(parts, prefix, refs):
    for i, r in enumerate(refs):
        parts.append(emit_int32_array(f"k{prefix}Ha{i}", torch.tensor(r["out"])))
    parts.append(emit_int32_array(f"k{prefix}AccM", torch.tensor([r["acc_m"] for r in refs])))
    parts.append(emit_int32_array(f"k{prefix}AccE", torch.tensor([r["acc_e"] for r in refs])))
    parts.append(emit_int32_array(f"k{prefix}Bits", torch.tensor([r["bits"] for r in refs])))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    cg = [cg_gold_case(*c) for c in CG_CASES]
    cg2 = [cg2_gold_case(gs) for gs in (4, 8)]
    cg2_aligned = [cg2_gold_case(gs) for gs in (10, 20)]
    channel = fx_fixture("channel", FX_CHANNEL_SCALES, FX_CIN * K)
    group2 = fx_fixture("group2", FX_GROUP2_SCALES, 2)
    assert channel["out"] != group2["out"], "channel and group2 golds coincide (vacuous)"
    cg_aligned = cgf_fixture(CGF_ALIGNED_SCALES, (CGF_CIN // CGF_GROUPS) * K)
    cg_cursor = cgf_fixture(CGF_CURSOR_SCALES, 4)
    assert cg_aligned["out"] != cg_cursor["out"], "conv-groups golds coincide (vacuous)"
    # the kernel takes the aligned loop iff groupSize % rowLength == 0
    assert 4 % ((CGF_CIN // CGF_GROUPS) * K) != 0, "cgCursor must not be channel-aligned"

    x, w, bias = fx_operands()
    cx, cw, cb = cgf_operands()
    parts = [
        "// AUTOGENERATED by generate_expected_conv1d_grouped_fold.py -- DO NOT EDIT\n",
        "#ifndef ODT_EXPECTED_CONV1D_GROUPED_FOLD_H\n",
        "#define ODT_EXPECTED_CONV1D_GROUPED_FOLD_H\n",
        "#include <stddef.h>\n",
        "#include <stdint.h>\n\n",
        "/* cgGold (cgGoldRun's LCG fixture), fold path, HALF_AWAY; index = kCgGoldCases */\n",
    ]
    emit_cases(parts, "CgFold", cg)
    parts.append("\n/* cg2 (cg2GoldRun, conv groups 2): group sizes 4, 8 | aligned 10, 20 */\n")
    emit_cases(parts, "Cg2Fold", cg2)
    emit_cases(parts, "Cg2AlignedFold", cg2_aligned)
    parts.append("\n/* channel / group2: shared operands, EXPLICIT 9 stride 2 dilation 2 */\n")
    parts.append(emit_int32_array("kConv1dFoldX", torch.tensor(x)))
    parts.append(emit_int32_array("kConv1dFoldW", torch.tensor(w)))
    parts.append(emit_int32_array("kConv1dFoldBias", torch.tensor(bias)))
    parts.append(emit_float_scalar("kConv1dFoldXScale", FX_X_SCALE))
    parts.append(emit_float_scalar("kConv1dFoldBiasScale", FX_BIAS_SCALE))
    parts.append(emit_int32_scalar("kConv1dFoldOutLen", channel["out_len"]))
    emit_fx(parts, "Conv1dFoldChannel", channel)
    parts.append(emit_int32_scalar("kConv1dFoldChannelCombines", channel["combines_total"]))
    parts.append(emit_int32_scalar("kConv1dFoldChannelBitsNoBias", channel["bits_nobias"]))
    emit_fx(parts, "Conv1dFoldGroup2", group2)
    parts.append(emit_int32_scalar("kConv1dFoldGroup2Combines", group2["combines_total"]))
    parts.append("\n/* conv groups 2: Cin 4, Cout 4, EXPLICIT 3 */\n")
    parts.append(emit_int32_array("kConv1dFoldCgX", torch.tensor(cx)))
    parts.append(emit_int32_array("kConv1dFoldCgW", torch.tensor(cw)))
    parts.append(emit_int32_array("kConv1dFoldCgBias", torch.tensor(cb)))
    parts.append(emit_int32_scalar("kConv1dFoldCgOutLen", cg_aligned["out_len"]))
    emit_fx(parts, "Conv1dFoldCgAligned", cg_aligned)
    emit_fx(parts, "Conv1dFoldCgCursor", cg_cursor)
    parts.append("\n#endif // ODT_EXPECTED_CONV1D_GROUPED_FOLD_H\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(parts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
