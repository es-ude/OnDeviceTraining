#!/usr/bin/env python3
"""Generate expected_conv1d_grouped.h for UnitTestConv1d (group-quant PR2,
Task 4 -- spec docs/superpowers/specs/2026-07-28-group-quantization-design.md).

Pins conv1dKernelSymInt32Grouped's GGUF-style running group-partial
rescale-combine (mirrors Task 3's matmulIntCoreGrouped exactly, just walking
the conv gather core's (icOffset, kernelIdx) reduction via sliding-window
geometry instead of a flat dot product): int MACs per group (exact), a
rescale-combine at every group boundary AND at the end of each
(batch, outChannel, outPos) reduction via rescaleIntoAccumulatorScale
(HALF_AWAY), s_acc = inScale * max_g(wScales[g]) (never scales[0]). See
sym_gold.conv1d_grouped_ref for the exact float32-mirrored emulation.

Fixture: IC=2, OC=3, K=3, L=6, B=1, VALID padding, stride=1, dilation=1
(outLen = 6-3+1 = 4). Both fixtures share the SAME weight/input/bias
mantissas (int12 codes) and bias; only the GROUP SHAPE (and therefore the
per-group scales) differs:

  perChannel  groupSize=6 (== inChannels*kernelSize, the full reduction
              length per output channel, numGroups=3): each output channel
              is exactly ONE group, so the running-partial loop crosses a
              group boundary zero times per (b, oc, outPos) -- the ONLY
              weight-side combine is the post-loop tail combine (plus one
              bias-seed combine).
  general     groupSize=3 (numGroups=6, two groups per output channel's own
              weight block -- the boundary falls INSIDE that output
              channel's reduction, between its two input channels' taps):
              one mid-loop boundary combine + one tail combine per output
              element (plus the bias-seed combine).

Self-checks (mutation-discriminating fixture properties, asserted here so a
broken fixture aborts generation rather than silently passing a vacuous
test) mirror generate_expected_group_matmul.py's Task-3 discipline exactly:
  (i)   general fixture's scales[0] != max(scales).
  (ii)  the LAST group's raw contribution is nonzero for both fixtures (a
        dropped tail combine would otherwise zero it out silently).
  (iii) at least one combine in the general fixture has a float32 quotient
        whose |fractional part| >= 0.5 (round-half-away vs
        truncate-toward-zero divergence point) -- the rounding-mode
        discriminator. SR_HALF_AWAY vs HALF_AWAY divergence is NOT emulated
        here for the same reason Task 3 didn't (needs the C-side seeded RNG
        stream); that mutation direction is instead covered by a dedicated
        C test running the real kernel under SR_HALF_AWAY with two RNG
        seeds and asserting the outputs differ.

Group-quant PR3 (Task 3) additions:
  dx        Conv1d backward propLoss with the grouped weight -- the adjoint
            SCATTER (convTranspose1dKernelSymInt32Grouped in the adjoint
            role, per-PRODUCT rescale into s_acc = lossScale*max(scales)).
            Gold via sym_gold.conv1d_dx_grouped_ref (the Task-2 scatter
            emulation under the Conv1d-adjoint parameter remapping); the
            emulation is cross-checked against a PyTorch float autograd
            reference (y.backward(lossGrad), the scalar-generator discipline)
            within the per-element bound 0.5*C_i*s_acc (C_i = contributing
            products, scatter error model).
  samePad   Conv1d FORWARD grouped under SAME padding -- PR2's disclosed
            coverage gap (a): the per-element group lookup's raison d'être.
            The generator PROVES the discriminating geometry: at least one
            padding-CLIPPED window exists AND a quantization-group boundary
            falls between two VISITED taps of one (oc, ic) row inside such a
            window, and a replay computing the group once per (oc, ic) row
            (the run-based assumption, C mutation (ii)) diverges from the
            gold.
  cg        conv-groups=2 x quant-groups -- PR2's disclosed coverage gap (b):
            weight layout [oc][ic/convGroups][K]; the generator asserts the
            two group systems disagree on at least one boundary (a quant
            boundary strictly inside a channel row, and a channel-row start
            that is no quant boundary).

Run via `uv run` (CMake wires this automatically, see CMakeLists.txt).
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "goldgen"))

from sym_gold import (combine_quotient_f32, conv1d_dx_grouped_ref, conv1d_grouped_ref,
                      emit_float_array, emit_float_scalar, emit_int32_array, emit_int32_scalar,
                      rescale_f32, window_geometry_1d, window_slice_1d_full)

BATCH = 1
IN_CHANNELS = 2
OUT_CHANNELS = 3
KERNEL_SIZE = 3
INPUT_LENGTH = 6

# weight: [out_channels=3, in_channels=2, kernel_size=3] row-major, one
# 6-element block per output channel (reused from Task 3's Matmul fixture --
# same numbers, purely for cross-review convenience; the actual gather-core
# computation is genuinely different, see conv1d_grouped_ref).
W_MANTISSAS = [4, -3, 2, -1, 5, -2,
              1, 2, -4, 3, -1, 2,
              -2, 3, 1, -5, 4, -3]

# input: [batch=1, in_channels=2, input_length=6] row-major.
X_MANTISSAS = [1, -2, 3, -1, 2, -3,
              2, 1, -1, 3, -2, 1]
X_SCALE = 0.5

BIAS_MANTISSAS = [10, -5, 3]
BIAS_SCALE = 0.1


def fixture_per_channel():
    group_size, num_groups = 6, 3
    w_scales = [0.02, 0.05, 0.01]
    assert len(w_scales) == num_groups

    out, s_acc, out_len = conv1d_grouped_ref(
        X_MANTISSAS, X_SCALE, W_MANTISSAS, w_scales, group_size,
        BATCH, IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, INPUT_LENGTH,
        bias_mantissas=BIAS_MANTISSAS, bias_scale=BIAS_SCALE)

    # Mutation (ii): the last group (output channel 2, its only group) must
    # have a nonzero raw contribution -- else dropping the tail combine would
    # be invisible. Check at out_pos=0.
    last_group_partial = sum(
        X_MANTISSAS[ic * INPUT_LENGTH + k] * W_MANTISSAS[2 * IN_CHANNELS * KERNEL_SIZE +
                                                          ic * KERNEL_SIZE + k]
        for ic in range(IN_CHANNELS) for k in range(KERNEL_SIZE))
    assert last_group_partial != 0, "perChannel: last group's contribution is vacuously zero"

    return {"wMantissas": W_MANTISSAS, "wScales": w_scales, "groupSize": group_size,
           "numGroups": num_groups, "outMantissas": out, "outScale": s_acc, "outLen": out_len}


def fixture_general_groups():
    group_size, num_groups = 3, 6
    # Deliberately scales[0] != max(scales) (mutation (i)'s pin).
    w_scales = [0.02, 0.05, 0.01, 0.08, 0.03, 0.06]
    assert w_scales[0] != max(w_scales), "general: scales[0] must differ from max (mutation i)"

    out, s_acc, out_len = conv1d_grouped_ref(
        X_MANTISSAS, X_SCALE, W_MANTISSAS, w_scales, group_size,
        BATCH, IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, INPUT_LENGTH,
        bias_mantissas=BIAS_MANTISSAS, bias_scale=BIAS_SCALE)

    # Mutation (ii): the LAST group (output channel 2's second group,
    # ic=1's taps) must have a nonzero raw contribution at out_pos=0.
    oc = 2
    ic = 1
    last_group_partial = sum(
        X_MANTISSAS[ic * INPUT_LENGTH + k] *
        W_MANTISSAS[oc * IN_CHANNELS * KERNEL_SIZE + ic * KERNEL_SIZE + k]
        for k in range(KERNEL_SIZE))
    assert last_group_partial != 0, "general: last group's contribution is vacuously zero"

    # Mutation (iii): at least one combine across the whole fixture must have
    # a float32 quotient with |fractional part| >= 0.5 (round-half-away vs
    # truncate-toward-zero divergence point). Recompute every weight-group
    # combine's quotient the same way conv1d_grouped_ref does internally.
    geom = window_geometry_1d(INPUT_LENGTH, KERNEL_SIZE, 1, 1, "VALID", 0)
    found_divergent = False
    for b in range(BATCH):
        for oc in range(OUT_CHANNELS):
            w_base = oc * IN_CHANNELS * KERNEL_SIZE
            for out_pos in range(geom["out_len"]):
                first_valid_idx, first_valid_k, valid_count = window_slice_1d_full(geom, out_pos)
                partial, current_group = 0, None
                for icc in range(IN_CHANNELS):
                    for i in range(valid_count):
                        kernel_idx = first_valid_k + i
                        w_idx = w_base + icc * KERNEL_SIZE + kernel_idx
                        g = w_idx // group_size
                        if g != current_group:
                            if current_group is not None:
                                param_scale = (torch.tensor(X_SCALE, dtype=torch.float32) *
                                              torch.tensor(w_scales[current_group],
                                                           dtype=torch.float32)).item()
                                q = combine_quotient_f32(partial, param_scale, s_acc)
                                if abs(abs(q) - int(abs(q)) - 0.5) < 1e-4:
                                    found_divergent = True
                            partial = 0
                            current_group = g
                        input_idx = first_valid_idx + i
                        partial += (X_MANTISSAS[(b * IN_CHANNELS + icc) * INPUT_LENGTH + input_idx] *
                                   W_MANTISSAS[w_idx])
                param_scale = (torch.tensor(X_SCALE, dtype=torch.float32) *
                              torch.tensor(w_scales[current_group], dtype=torch.float32)).item()
                q = combine_quotient_f32(partial, param_scale, s_acc)
                if abs(abs(q) - int(abs(q)) - 0.5) < 1e-4:
                    found_divergent = True
    assert found_divergent, (
        "general: no combine has a round-vs-truncate-divergent quotient "
        "(fixture is vacuous against a truncation-instead-of-rounding mutation)")

    return {"wMantissas": W_MANTISSAS, "wScales": w_scales, "groupSize": group_size,
           "numGroups": num_groups, "outMantissas": out, "outScale": s_acc, "outLen": out_len}


# ---- PR3 Task 3: Conv1d dx (adjoint scatter) fixture. Seeded random lossGrad
# (the vacuity lesson: a uniform lossGrad cannot discriminate geometry
# mutations); int12-safe magnitudes; power-of-two loss scale (the equal-scales
# twin's exactness argument needs it). Reuses the forward fixtures' weight
# mantissas and group shapes (perChannel/general). ----

FWD_OUT_LEN = INPUT_LENGTH - KERNEL_SIZE + 1  # VALID, stride=1, dilation=1

torch.manual_seed(20260730)
DX_LOSS_MANTISSAS = [int(v) for v in torch.randint(
    -40, 41, (BATCH * OUT_CHANNELS * FWD_OUT_LEN,)).tolist()]
DX_LOSS_SCALE = 0.5


def dequant_grouped(mantissas, scales, group_size):
    return [float(m) * scales[i // group_size] for i, m in enumerate(mantissas)]


def dx_product_counts():
    """Per-dx-element contributing-product counts C_i (the scatter error
    model's C): dx[(ic, l)] receives one product per (oc, outPos, k) with
    outPos + k == l (stride=1, dilation=1, VALID). Flat [IC*L] list."""
    counts = [0] * (IN_CHANNELS * INPUT_LENGTH)
    for ic in range(IN_CHANNELS):
        for _ in range(OUT_CHANNELS):
            for out_pos in range(FWD_OUT_LEN):
                for k in range(KERNEL_SIZE):
                    counts[ic * INPUT_LENGTH + out_pos + k] += 1
    return counts


def fixture_dx(name, w_scales, group_size):
    out, s_acc = conv1d_dx_grouped_ref(
        DX_LOSS_MANTISSAS, DX_LOSS_SCALE, W_MANTISSAS, w_scales, group_size,
        BATCH, IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, INPUT_LENGTH)
    assert any(v != 0 for v in out), f"dx {name}: gold is vacuously all-zero"

    # Cross-check the integer emulation against the PyTorch float autograd
    # reference (the scalar dx generators' discipline): same dequantized
    # operands, so the only divergence is the emulation's per-product rounding
    # -- |gold*s_acc - x.grad| <= 0.5*C_i*s_acc per element (+ float noise).
    w_deq = torch.tensor(dequant_grouped(W_MANTISSAS, w_scales, group_size),
                         dtype=torch.float32).reshape(OUT_CHANNELS, IN_CHANNELS, KERNEL_SIZE)
    gy = (torch.tensor(DX_LOSS_MANTISSAS, dtype=torch.float32) * DX_LOSS_SCALE).reshape(
        BATCH, OUT_CHANNELS, FWD_OUT_LEN)
    x = torch.zeros(BATCH, IN_CHANNELS, INPUT_LENGTH, dtype=torch.float32, requires_grad=True)
    y = F.conv1d(x, w_deq)
    y.backward(gy)
    ref = x.grad.flatten().tolist()
    counts = dx_product_counts()
    for i, (g, r) in enumerate(zip(out, ref)):
        bound = 0.5 * counts[i] * s_acc + 1e-4
        assert abs(g * s_acc - r) <= bound, (
            f"dx {name}: emulation deviates from torch autograd beyond the scatter "
            f"bound at {i}: |{g * s_acc} - {r}| > {bound}")
    return {"outMantissas": out, "outScale": s_acc}


# ---- PR3 Task 3, coverage gap (a): Conv1d FORWARD grouped under SAME
# padding. groupSize=2 puts a quantization-group boundary INSIDE the
# padding-clipped windows' visited tap ranges (proved by
# assert_same_padding_geometry below) -- the per-element group lookup's
# raison d'être. Shares W/X/bias mantissas with the VALID fixtures. ----

SAME_GROUP_SIZE = 2
SAME_NUM_GROUPS = 9
SAME_W_SCALES = [0.02, 0.05, 0.01, 0.08, 0.03, 0.06, 0.04, 0.07, 0.02]


def same_geom():
    return window_geometry_1d(INPUT_LENGTH, KERNEL_SIZE, 1, 1, "SAME", 0)


def assert_same_padding_geometry():
    """The padding fixture's discriminating geometry, PROVED not assumed:
    (1) at least one window is padding-clipped (validCount < K), and
    (2) inside such a clipped window, a quantization-group boundary falls
        between two VISITED taps of one (oc, ic) row -- so a group lookup
        that is not per-element misattributes at least one visited tap."""
    geom = same_geom()
    clipped = [p for p in range(geom["out_len"])
               if window_slice_1d_full(geom, p)[2] < KERNEL_SIZE]
    assert clipped, "samePad: no padding-clipped window -- fixture is vacuous"
    boundary_in_clipped = False
    for p in clipped:
        _, fvk, vc = window_slice_1d_full(geom, p)
        for oc in range(OUT_CHANNELS):
            for ic in range(IN_CHANNELS):
                row = [(oc * IN_CHANNELS + ic) * KERNEL_SIZE + fvk + i for i in range(vc)]
                groups = [w // SAME_GROUP_SIZE for w in row]
                if any(groups[j] != groups[j + 1] for j in range(len(groups) - 1)):
                    boundary_in_clipped = True
    assert boundary_in_clipped, (
        "samePad: no quant-group boundary inside a padding-clipped window -- "
        "the per-element lookup is not exercised where it matters")


def ref_same_group_once_per_row():
    """C mutation (ii) replay: the group id computed ONCE per (oc, ic) row
    (from the row's first VISITED element) and held for the whole row -- the
    run-based assumption the per-element division exists to prevent. Same
    float32-mirrored arithmetic as conv1d_grouped_ref otherwise."""
    geom = same_geom()
    s_acc = (torch.tensor(X_SCALE, dtype=torch.float32) *
            torch.tensor(max(SAME_W_SCALES), dtype=torch.float32)).item()
    out = []
    for b in range(BATCH):
        for oc in range(OUT_CHANNELS):
            w_base = oc * IN_CHANNELS * KERNEL_SIZE
            seed = rescale_f32(BIAS_MANTISSAS[oc], BIAS_SCALE, s_acc)
            for out_pos in range(geom["out_len"]):
                fvi, fvk, vc = window_slice_1d_full(geom, out_pos)
                acc = seed
                partial = 0
                current_group = None
                for ic in range(IN_CHANNELS):
                    row_group = (w_base + ic * KERNEL_SIZE + fvk) // SAME_GROUP_SIZE
                    for i in range(vc):
                        if row_group != current_group:
                            if current_group is not None:
                                ps = (torch.tensor(X_SCALE, dtype=torch.float32) *
                                      torch.tensor(SAME_W_SCALES[current_group],
                                                   dtype=torch.float32)).item()
                                acc += rescale_f32(partial, ps, s_acc)
                            partial = 0
                            current_group = row_group
                        w_idx = w_base + ic * KERNEL_SIZE + fvk + i
                        x_val = X_MANTISSAS[(b * IN_CHANNELS + ic) * INPUT_LENGTH + fvi + i]
                        partial += x_val * W_MANTISSAS[w_idx]
                if current_group is not None:
                    ps = (torch.tensor(X_SCALE, dtype=torch.float32) *
                          torch.tensor(SAME_W_SCALES[current_group],
                                       dtype=torch.float32)).item()
                    acc += rescale_f32(partial, ps, s_acc)
                out.append(acc)
    return out


def gather_combine_counts(group_size, conv_groups, in_channels, out_channels, kernel_size,
                          input_length, padding_type):
    """Per-(oc, outPos) combine counts for the gather core (= number of group
    RUNS in the monotonically-visited weight sequence; each run folds into the
    accumulator scale exactly once -- mid-loop boundary or tail)."""
    geom = window_geometry_1d(input_length, kernel_size, 1, 1, padding_type, 0)
    icpg = in_channels // conv_groups
    counts = []
    for oc in range(out_channels):
        w_base = oc * icpg * kernel_size
        for out_pos in range(geom["out_len"]):
            _, fvk, vc = window_slice_1d_full(geom, out_pos)
            visited = [w_base + ic_off * kernel_size + fvk + i
                       for ic_off in range(icpg) for i in range(vc)]
            combines = 0
            cur = None
            for g in (w // group_size for w in visited):
                if g != cur:
                    combines += 1
                    cur = g
            counts.append(combines)
    return counts


def torch_forward_check(name, gold, s_acc, x_mant, x_scale, w_mant, w_scales, group_size,
                        bias_mant, bias_scale, in_channels, out_channels, kernel_size,
                        input_length, conv_groups, torch_padding):
    """Cross-check the forward emulation against F.conv1d on the SAME
    dequantized operands: per element |gold*s_acc - ref| <= 0.5*(combines_i +
    1)*s_acc (weight-run combines + the bias seed, each one HALF_AWAY
    rounding) + float-noise headroom."""
    w_deq = torch.tensor(dequant_grouped(w_mant, w_scales, group_size),
                         dtype=torch.float32).reshape(out_channels,
                                                      in_channels // conv_groups, kernel_size)
    x = (torch.tensor(x_mant, dtype=torch.float32) * x_scale).reshape(
        BATCH, in_channels, input_length)
    b = torch.tensor(bias_mant, dtype=torch.float32) * bias_scale
    ref = F.conv1d(x, w_deq, b, padding=torch_padding, groups=conv_groups).flatten().tolist()
    counts = gather_combine_counts(group_size, conv_groups, in_channels, out_channels,
                                   kernel_size, input_length,
                                   "SAME" if torch_padding != 0 else "VALID")
    assert len(ref) == len(gold) == len(counts)
    for i, (g, r) in enumerate(zip(gold, ref)):
        bound = 0.5 * (counts[i] + 1) * s_acc + 1e-4
        assert abs(g * s_acc - r) <= bound, (
            f"{name}: emulation deviates from torch beyond the combine bound at "
            f"{i}: |{g * s_acc} - {r}| > {bound}")


def fixture_same_padding():
    assert SAME_W_SCALES[0] != max(SAME_W_SCALES), "samePad: scales[0] must differ from max"
    assert_same_padding_geometry()
    geom = same_geom()
    # The torch cross-check maps our SAME to F.conv1d(padding=1) -- valid only
    # while the minimal SAME pad is symmetric (total=2, padLeft=1 here).
    assert geom["pad_left"] == 1 and geom["out_len"] == INPUT_LENGTH

    out, s_acc, out_len = conv1d_grouped_ref(
        X_MANTISSAS, X_SCALE, W_MANTISSAS, SAME_W_SCALES, SAME_GROUP_SIZE,
        BATCH, IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, INPUT_LENGTH,
        padding_type="SAME", bias_mantissas=BIAS_MANTISSAS, bias_scale=BIAS_SCALE)
    assert out_len == INPUT_LENGTH

    wrong = ref_same_group_once_per_row()
    assert wrong != out, (
        "samePad: the once-per-row group lookup reproduces the gold -- "
        "mutation (ii) would be vacuous")

    torch_forward_check("samePad", out, s_acc, X_MANTISSAS, X_SCALE, W_MANTISSAS,
                        SAME_W_SCALES, SAME_GROUP_SIZE, BIAS_MANTISSAS, BIAS_SCALE,
                        IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, INPUT_LENGTH, 1, 1)
    return {"wMantissas": W_MANTISSAS, "wScales": SAME_W_SCALES,
            "groupSize": SAME_GROUP_SIZE, "numGroups": SAME_NUM_GROUPS,
            "outMantissas": out, "outScale": s_acc, "outLen": out_len}


# ---- PR3 Task 3, coverage gap (b): conv-groups=2 x quant-groups. Weight
# layout [oc][ic/convGroups][K] (4x2x3 = 24 elements); quant groupSize=4 puts
# quant boundaries at {4,8,12,16,20} vs channel-row starts {6,12,18} -- the
# two group systems disagree (asserted), so a kernel that derived the quant
# group from channel structure instead of flat storage would diverge. ----

CG_IN_CHANNELS = 4
CG_OUT_CHANNELS = 4
CG_CONV_GROUPS = 2
CG_KERNEL_SIZE = 3
CG_INPUT_LENGTH = 6
CG_OUT_LEN = CG_INPUT_LENGTH - CG_KERNEL_SIZE + 1
CG_GROUP_SIZE = 4
CG_NUM_GROUPS = 6
CG_W_SCALES = [0.02, 0.05, 0.01, 0.08, 0.03, 0.06]
CG_X_SCALE = 0.5
CG_BIAS_MANTISSAS = [10, -5, 3, -7]
CG_BIAS_SCALE = 0.1

torch.manual_seed(20260731)
CG_W_MANTISSAS = [int(v) for v in torch.randint(
    -60, 61, (CG_OUT_CHANNELS * (CG_IN_CHANNELS // CG_CONV_GROUPS) * CG_KERNEL_SIZE,)).tolist()]
CG_X_MANTISSAS = [int(v) for v in torch.randint(
    -40, 41, (BATCH * CG_IN_CHANNELS * CG_INPUT_LENGTH,)).tolist()]


def fixture_conv_groups():
    row_len = (CG_IN_CHANNELS // CG_CONV_GROUPS) * CG_KERNEL_SIZE
    n = CG_OUT_CHANNELS * row_len
    assert CG_GROUP_SIZE * CG_NUM_GROUPS == n
    quant_starts = set(range(0, n, CG_GROUP_SIZE))
    row_starts = set(range(0, n, row_len))
    assert any(q not in row_starts for q in quant_starts), (
        "cg: every quant boundary aligns with a channel row -- the two group "
        "systems never disagree (fixture vacuous)")
    assert any(r not in quant_starts for r in row_starts), (
        "cg: every channel-row start is a quant boundary -- the two group "
        "systems never disagree (fixture vacuous)")
    assert CG_W_SCALES[0] != max(CG_W_SCALES)

    out, s_acc, out_len = conv1d_grouped_ref(
        CG_X_MANTISSAS, CG_X_SCALE, CG_W_MANTISSAS, CG_W_SCALES, CG_GROUP_SIZE,
        BATCH, CG_IN_CHANNELS, CG_OUT_CHANNELS, CG_KERNEL_SIZE, CG_INPUT_LENGTH,
        bias_mantissas=CG_BIAS_MANTISSAS, bias_scale=CG_BIAS_SCALE,
        conv_groups=CG_CONV_GROUPS)
    assert out_len == CG_OUT_LEN
    assert any(v != 0 for v in out), "cg: gold is vacuously all-zero"

    torch_forward_check("cg", out, s_acc, CG_X_MANTISSAS, CG_X_SCALE, CG_W_MANTISSAS,
                        CG_W_SCALES, CG_GROUP_SIZE, CG_BIAS_MANTISSAS, CG_BIAS_SCALE,
                        CG_IN_CHANNELS, CG_OUT_CHANNELS, CG_KERNEL_SIZE, CG_INPUT_LENGTH,
                        CG_CONV_GROUPS, 0)
    return {"outMantissas": out, "outScale": s_acc}


def emit_fixture(parts, prefix, fx):
    parts.append(emit_int32_array(f"k{prefix}WMantissas", torch.tensor(fx["wMantissas"])))
    parts.append(emit_float_array(f"k{prefix}WScales", torch.tensor(fx["wScales"])))
    parts.append(emit_int32_scalar(f"k{prefix}GroupSize", fx["groupSize"]))
    parts.append(emit_int32_scalar(f"k{prefix}NumGroups", fx["numGroups"]))
    parts.append(emit_int32_array(f"k{prefix}OutMantissas", torch.tensor(fx["outMantissas"])))
    parts.append(emit_float_scalar(f"k{prefix}OutScale", fx["outScale"]))
    parts.append(emit_int32_scalar(f"k{prefix}OutLen", fx["outLen"]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    parts = [
        "// AUTOGENERATED by generate_expected_conv1d_grouped.py — DO NOT EDIT\n",
        "#ifndef ODT_EXPECTED_CONV1D_GROUPED_H\n",
        "#define ODT_EXPECTED_CONV1D_GROUPED_H\n",
        "#include <stdint.h>\n",
        "#include <stdlib.h>\n\n",
        f"static const int32_t kConv1dGroupedWMantissas[] = "
        f"{{ {', '.join(str(v) for v in W_MANTISSAS)} }};\n",
        f"static const int32_t kConv1dGroupedXMantissas[] = "
        f"{{ {', '.join(str(v) for v in X_MANTISSAS)} }};\n",
        f"static const float kConv1dGroupedXScale = {X_SCALE}f;\n",
        f"static const int32_t kConv1dGroupedBiasMantissas[] = "
        f"{{ {', '.join(str(v) for v in BIAS_MANTISSAS)} }};\n",
        f"static const float kConv1dGroupedBiasScale = {BIAS_SCALE}f;\n",
        f"static const int32_t kConv1dGroupedBatch = {BATCH};\n",
        f"static const int32_t kConv1dGroupedInChannels = {IN_CHANNELS};\n",
        f"static const int32_t kConv1dGroupedOutChannels = {OUT_CHANNELS};\n",
        f"static const int32_t kConv1dGroupedKernelSize = {KERNEL_SIZE};\n",
        f"static const int32_t kConv1dGroupedInputLength = {INPUT_LENGTH};\n",
        "\n",
    ]

    emit_fixture(parts, "PerChannel", fixture_per_channel())
    emit_fixture(parts, "General", fixture_general_groups())

    # PR3 Task 3: dx fixtures reuse the forward fixtures' group shapes/scales;
    # only the lossGrad and the adjoint-scatter golds are new.
    dx_pc = fixture_dx("perChannel", [0.02, 0.05, 0.01], 6)
    dx_gen = fixture_dx("general", [0.02, 0.05, 0.01, 0.08, 0.03, 0.06], 3)
    assert ([v * dx_pc["outScale"] for v in dx_pc["outMantissas"]] !=
            [v * dx_gen["outScale"] for v in dx_gen["outMantissas"]]), (
        "dx: perChannel and general golds dequantize identically -- the group "
        "shape does not reach the dx path (fixture vacuous)")
    counts = dx_product_counts()
    assert max(counts) == OUT_CHANNELS * KERNEL_SIZE, (
        f"dx: max products per element is {max(counts)}, expected "
        f"{OUT_CHANNELS * KERNEL_SIZE} -- re-derive the C float-path tolerance")

    parts.append(emit_int32_array("kConv1dDxLossMantissas", torch.tensor(DX_LOSS_MANTISSAS)))
    parts.append(emit_float_scalar("kConv1dDxLossScale", DX_LOSS_SCALE))
    parts.append(emit_int32_scalar("kConv1dDxFwdOutLen", FWD_OUT_LEN))
    parts.append(emit_int32_scalar("kConv1dDxMaxProductsPerOut", max(counts)))
    parts.append(emit_int32_array("kDxPerChannelOutMantissas",
                                  torch.tensor(dx_pc["outMantissas"])))
    parts.append(emit_float_scalar("kDxPerChannelOutScale", dx_pc["outScale"]))
    parts.append(emit_int32_array("kDxGeneralOutMantissas", torch.tensor(dx_gen["outMantissas"])))
    parts.append(emit_float_scalar("kDxGeneralOutScale", dx_gen["outScale"]))
    parts.append("\n")

    emit_fixture(parts, "SamePad", fixture_same_padding())

    cg = fixture_conv_groups()
    parts.append(emit_int32_array("kCgWMantissas", torch.tensor(CG_W_MANTISSAS)))
    parts.append(emit_int32_array("kCgXMantissas", torch.tensor(CG_X_MANTISSAS)))
    parts.append(emit_float_scalar("kCgXScale", CG_X_SCALE))
    parts.append(emit_int32_array("kCgBiasMantissas", torch.tensor(CG_BIAS_MANTISSAS)))
    parts.append(emit_float_scalar("kCgBiasScale", CG_BIAS_SCALE))
    parts.append(emit_float_array("kCgWScales", torch.tensor(CG_W_SCALES)))
    parts.append(emit_int32_scalar("kCgGroupSize", CG_GROUP_SIZE))
    parts.append(emit_int32_scalar("kCgNumGroups", CG_NUM_GROUPS))
    parts.append(emit_int32_scalar("kCgConvGroups", CG_CONV_GROUPS))
    parts.append(emit_int32_scalar("kCgInChannels", CG_IN_CHANNELS))
    parts.append(emit_int32_scalar("kCgOutChannels", CG_OUT_CHANNELS))
    parts.append(emit_int32_scalar("kCgKernelSize", CG_KERNEL_SIZE))
    parts.append(emit_int32_scalar("kCgInputLength", CG_INPUT_LENGTH))
    parts.append(emit_int32_scalar("kCgOutLen", CG_OUT_LEN))
    parts.append(emit_int32_array("kCgOutMantissas", torch.tensor(cg["outMantissas"])))
    parts.append(emit_float_scalar("kCgOutScale", cg["outScale"]))

    parts.append("\n#endif // ODT_EXPECTED_CONV1D_GROUPED_H\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(parts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
