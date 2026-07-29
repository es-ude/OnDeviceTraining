#!/usr/bin/env python3
"""Generate expected_convT1d_grouped.h for UnitTestConv1dTransposed (group-quant
PR3, Task 2 -- spec docs/superpowers/specs/2026-07-28-group-quantization-design.md).

Pins convTranspose1dKernelSymInt32Grouped's SCATTER core with PER-PRODUCT
rescale-combine: a scatter's consecutive products land in DIFFERENT output
elements, so no per-(target, group) running raw partial exists -- each
product is folded into s_acc = inScale*max_g(scales[g]) immediately
(rescaleIntoAccumulatorScale, HALF_AWAY), bias AFTER the scatter at s_acc.
See sym_gold.convT1d_grouped_ref for the exact float32-mirrored emulation.

Fixture: IC=2, OC=3, K=3, L=4, B=1, VALID, stride=1, dilation=1,
outputPadding=0 (outLen = (4-1)*1 + 1*(3-1) + 0 + 1 = 6). Seeded random
int12-range mantissas. Both grouped fixtures share the SAME weight/input/bias
mantissas; only the GROUP SHAPE (and therefore the per-group scales) differs.

GROUP-SHAPE MEANING FOR THIS LAYOUT (ConvT1d weight storage
[Cin, Cout, K] with conv-groups==1, flat index (ic*Cout + oc)*K + k):
storage-contiguous groups interleave OUTPUT channels inside each
INPUT-channel slab, so:

  perChannel  groupSize=9 (== Cout*K, one INPUT-channel slab, numGroups=2):
              "per-channel" here means per-INPUT-channel -- the only channel
              axis expressible as contiguous storage groups in this layout
              (a per-OUTPUT-channel grouping is NOT contiguous here, unlike
              Conv1d's [Cout, Cin, K] layout where it is). Every product of
              one ic's scatter shares one group.
  general     groupSize=3 (== K, one (ic, oc) tap-row per group,
              numGroups=6): group boundaries fall INSIDE each ic-slab,
              between output channels' tap-rows.

Self-checks (mutation-discriminating fixture properties, asserted here so a
broken fixture aborts generation rather than silently passing a vacuous
test):
  (i)   scales[0] != max(scales) for BOTH fixtures, and replaying the
        perChannel fixture with the group taken from the OUTPUT flat index
        instead of the weight flat index changes the output (mutation (i)).
  (ii)  the per-product rescale sequence differs from the
        sum-then-rescale-once-per-(target, ic, group)-run composition (the
        running-partial idiom wrongly ported to the scatter core) on BOTH
        fixtures (mutation (ii)).
  (iii) the bias seed at s_acc differs from a seed at inScale*scales[0] for
        at least one output channel, BOTH fixtures (mutation (iii)).
  (iv)  replaying the general fixture with truncation-toward-zero instead of
        HALF_AWAY per-product rounding changes the output -- the
        rounding-mode discriminator. SR_HALF_AWAY vs HALF_AWAY is NOT
        emulated (needs the C-side seeded RNG stream, same scope note as the
        Conv1d/Matmul grouped generators).
  (v)   the equal-scales fixture (all scales 0.25, inScale 0.5, both powers
        of two) reproduces the SCALAR kernel emulation (raw int scatter at
        s_in*s_w, bias seed at that same scale) EXACTLY: every per-product
        rescale is paramScale/s_acc == 1.0 bit-identically, and multiplying/
        dividing a float by the same power of two is exact, so
        round_half_away(p * 1.0) == p -- the bit-identity the C twin test
        (testConvT1dForwardGroupedEqualScalesBitIdenticalToScalar) relies on.
  (vi)  the max number of contributing products per output element is
        exactly the value emitted as kConvTGroupedMaxProductsPerOut -- the C
        float-path test derives its tolerance bound 0.5*(C+1)*s_acc from it.

Group-quant PR3 (Task 3) addition -- ConvT1d dx (adjoint GATHER): backward
propLoss consumes the grouped weight through conv1dKernelSymInt32Grouped in
the adjoint role (running group-partial, one rescale-combine per group RUN of
the reduction, s_acc = lossScale*max(scales)). Gold via
sym_gold.convT1d_dx_grouped_ref (the PR2 gather emulation under the
ConvT1d-adjoint parameter remapping); cross-checked against a PyTorch float
autograd reference (y.backward(lossGrad)) within the per-element bound
0.5*C_i*s_acc, C_i = combines (group runs) per output reduction -- the gather
error model, vs the scatter's per-product model above. The per-fixture max
C_i is emitted (kConvTDx*MaxCombines) for the C float-path tolerance.

Run via `uv run` (CMake wires this automatically, see CMakeLists.txt).
"""
import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "goldgen"))

from sym_gold import (assert_rounding_canary, combine_quotient_f32, convT1d_dx_grouped_ref,
                      convT1d_grouped_ref, emit_float_array, emit_float_scalar, emit_int32_array,
                      emit_int32_scalar, rescale_f32, window_geometry_1d, window_slice_1d_full)

BATCH = 1
IN_CHANNELS = 2
OUT_CHANNELS = 3
KERNEL_SIZE = 3
INPUT_LENGTH = 4
STRIDE = 1
DILATION = 1
OUTPUT_PADDING = 0
OUT_LEN = (INPUT_LENGTH - 1) * STRIDE + DILATION * (KERNEL_SIZE - 1) + OUTPUT_PADDING + 1

# Seeded random mantissas (int12-safe magnitudes; products <= 60*40 = 2400
# stay far inside exact-float32 integer range).
torch.manual_seed(20260729)
W_MANTISSAS = [int(v) for v in torch.randint(-60, 61,
                                             (IN_CHANNELS * OUT_CHANNELS * KERNEL_SIZE,)).tolist()]
X_MANTISSAS = [int(v) for v in torch.randint(-40, 41,
                                             (BATCH * IN_CHANNELS * INPUT_LENGTH,)).tolist()]
X_SCALE = 0.5  # power of two -- the equal-scales twin's exactness argument needs it

BIAS_MANTISSAS = [17, -9, 23]
BIAS_SCALE = 0.1


def param_scale_f32(w_scale):
    return (torch.tensor(X_SCALE, dtype=torch.float32) *
           torch.tensor(w_scale, dtype=torch.float32)).item()


def s_acc_f32(w_scales):
    return param_scale_f32(max(w_scales))


def scatter_targets():
    """Yield (out_flat_idx, [(ic, product, w_idx)]) plus bias channel, i.e. the
    per-output-element contributing-product structure the self-checks below
    reason over. Batch is 1."""
    contrib = {}
    for ic in range(IN_CHANNELS):
        for in_pos in range(INPUT_LENGTH):
            x_val = X_MANTISSAS[ic * INPUT_LENGTH + in_pos]
            for oc in range(OUT_CHANNELS):
                for k in range(KERNEL_SIZE):
                    out_idx = in_pos * STRIDE + k * DILATION
                    w_idx = (ic * OUT_CHANNELS + oc) * KERNEL_SIZE + k
                    flat = oc * OUT_LEN + out_idx
                    contrib.setdefault(flat, []).append((ic, x_val * W_MANTISSAS[w_idx], w_idx))
    return contrib


def ref_wrong_group_from_output_index(w_scales, group_size):
    """Mutation (i) replay: g = OUTPUT flat index // groupSize (both fixtures'
    group counts stay in range for the 18-element output, so the mutated C
    kernel reads a VALID-but-wrong scale instead of crashing)."""
    s_acc = s_acc_f32(w_scales)
    out = [0] * (BATCH * OUT_CHANNELS * OUT_LEN)
    for flat, products in scatter_targets().items():
        g = flat // group_size
        assert g < len(w_scales)
        for _, p, _ in products:
            out[flat] += rescale_f32(p, param_scale_f32(w_scales[g]), s_acc)
    for oc in range(OUT_CHANNELS):
        seed = rescale_f32(BIAS_MANTISSAS[oc], BIAS_SCALE, s_acc)
        for l in range(OUT_LEN):
            out[oc * OUT_LEN + l] += seed
    return out


def ref_sum_then_rescale_per_run(w_scales, group_size):
    """Mutation (ii) replay: the running-partial idiom wrongly ported to the
    scatter core -- per output element, raw-int-sum each (ic, group) run of
    contributing products, ONE rescale per run (instead of one per product)."""
    s_acc = s_acc_f32(w_scales)
    out = [0] * (BATCH * OUT_CHANNELS * OUT_LEN)
    for flat, products in scatter_targets().items():
        runs = {}
        for ic, p, w_idx in products:
            g = w_idx // group_size
            runs[(ic, g)] = runs.get((ic, g), 0) + p
        for (_, g), partial in runs.items():
            out[flat] += rescale_f32(partial, param_scale_f32(w_scales[g]), s_acc)
    for oc in range(OUT_CHANNELS):
        seed = rescale_f32(BIAS_MANTISSAS[oc], BIAS_SCALE, s_acc)
        for l in range(OUT_LEN):
            out[oc * OUT_LEN + l] += seed
    return out


def ref_truncating(w_scales, group_size):
    """Mutation (iv) replay: per-product truncation-toward-zero instead of
    HALF_AWAY (same float32 quotient, different rounding)."""
    s_acc = s_acc_f32(w_scales)
    out = [0] * (BATCH * OUT_CHANNELS * OUT_LEN)
    for flat, products in scatter_targets().items():
        for _, p, w_idx in products:
            g = w_idx // group_size
            out[flat] += math.trunc(combine_quotient_f32(p, param_scale_f32(w_scales[g]), s_acc))
    for oc in range(OUT_CHANNELS):
        seed = math.trunc(combine_quotient_f32(BIAS_MANTISSAS[oc], BIAS_SCALE, s_acc))
        for l in range(OUT_LEN):
            out[oc * OUT_LEN + l] += seed
    return out


def check_fixture(name, w_scales, group_size, out):
    # (i) scales[0] != max, and output-index group lookup diverges.
    assert w_scales[0] != max(w_scales), f"{name}: scales[0] must differ from max (mutation i)"
    wrong_g = ref_wrong_group_from_output_index(w_scales, group_size)
    assert wrong_g != out, (
        f"{name}: output-index group lookup reproduces the gold (mutation (i) vacuous)")
    # (ii) sum-then-rescale-once per run diverges.
    wrong_run = ref_sum_then_rescale_per_run(w_scales, group_size)
    assert wrong_run != out, (
        f"{name}: sum-then-rescale-once per run reproduces the gold (mutation (ii) vacuous)")
    # (iii) bias seed at inScale*scales[0] diverges for at least one channel.
    s_acc = s_acc_f32(w_scales)
    seeds_right = [rescale_f32(BIAS_MANTISSAS[oc], BIAS_SCALE, s_acc)
                   for oc in range(OUT_CHANNELS)]
    seeds_wrong = [rescale_f32(BIAS_MANTISSAS[oc], BIAS_SCALE, param_scale_f32(w_scales[0]))
                   for oc in range(OUT_CHANNELS)]
    assert seeds_right != seeds_wrong, (
        f"{name}: bias seed at inScale*scales[0] equals the s_acc seed (mutation (iii) vacuous)")


def fixture(name, w_scales, group_size, num_groups):
    assert len(w_scales) == num_groups
    assert group_size * num_groups == len(W_MANTISSAS)
    out, s_acc, out_len = convT1d_grouped_ref(
        X_MANTISSAS, X_SCALE, W_MANTISSAS, w_scales, group_size,
        BATCH, IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, INPUT_LENGTH,
        STRIDE, DILATION, OUTPUT_PADDING, bias_mantissas=BIAS_MANTISSAS, bias_scale=BIAS_SCALE)
    assert out_len == OUT_LEN
    check_fixture(name, w_scales, group_size, out)
    return {"wScales": w_scales, "groupSize": group_size, "numGroups": num_groups,
            "outMantissas": out, "outScale": s_acc}


def check_equal_scales_reproduces_scalar():
    """Self-check (v): all scales == 0.25 (power of two, like X_SCALE) makes
    every per-product rescale the exact identity, so the grouped emulation
    must equal the scalar kernel's raw int scatter + bias seed bit-for-bit."""
    common = 0.25
    w_scales = [common, common]
    out, s_acc, _ = convT1d_grouped_ref(
        X_MANTISSAS, X_SCALE, W_MANTISSAS, w_scales, 9,
        BATCH, IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, INPUT_LENGTH,
        STRIDE, DILATION, OUTPUT_PADDING, bias_mantissas=BIAS_MANTISSAS, bias_scale=BIAS_SCALE)
    scalar = [0] * (BATCH * OUT_CHANNELS * OUT_LEN)
    for flat, products in scatter_targets().items():
        for _, p, _ in products:
            scalar[flat] += p
    output_scale = param_scale_f32(common)
    assert output_scale == s_acc, "equal-scales: s_acc must be bit-identical to s_in*s_w"
    for oc in range(OUT_CHANNELS):
        seed = rescale_f32(BIAS_MANTISSAS[oc], BIAS_SCALE, output_scale)
        for l in range(OUT_LEN):
            scalar[oc * OUT_LEN + l] += seed
    assert out == scalar, (
        "equal-scales grouped emulation does not reproduce the scalar scatter exactly")


def max_products_per_out():
    return max(len(v) for v in scatter_targets().values())


# ---- PR3 Task 3: ConvT1d dx (adjoint gather) fixture. Seeded random
# lossGrad [B, Cout, Lout] (vacuity lesson), power-of-two loss scale (the
# equal-scales twin's exactness argument). Reuses the forward fixtures'
# weight mantissas and group shapes. ----

torch.manual_seed(20260732)
DX_LOSS_MANTISSAS = [int(v) for v in torch.randint(
    -40, 41, (BATCH * OUT_CHANNELS * OUT_LEN,)).tolist()]
DX_LOSS_SCALE = 0.5


def dx_combine_counts(group_size):
    """Per-dx-element combine counts C_i for the adjoint gather: the
    reduction for dx[(ic, inPos)] walks the weight slab
    (ic*Cout + oc)*K + k monotonically (oc rows ascending, visited taps
    ascending); each group RUN in that visited sequence folds into s_acc
    exactly once. VALID geometry on Lout (fully-valid windows here). Flat
    [Cin*Lin] list."""
    geom = window_geometry_1d(OUT_LEN, KERNEL_SIZE, STRIDE, DILATION, "VALID", 0)
    assert geom["out_len"] == INPUT_LENGTH
    counts = []
    for ic in range(IN_CHANNELS):
        for in_pos in range(INPUT_LENGTH):
            _, fvk, vc = window_slice_1d_full(geom, in_pos)
            visited = [(ic * OUT_CHANNELS + oc) * KERNEL_SIZE + fvk + i
                       for oc in range(OUT_CHANNELS) for i in range(vc)]
            combines = 0
            cur = None
            for g in (w // group_size for w in visited):
                if g != cur:
                    combines += 1
                    cur = g
            counts.append(combines)
    return counts


def fixture_dx(name, w_scales, group_size):
    out, s_acc = convT1d_dx_grouped_ref(
        DX_LOSS_MANTISSAS, DX_LOSS_SCALE, W_MANTISSAS, w_scales, group_size,
        BATCH, IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, INPUT_LENGTH,
        STRIDE, DILATION, OUTPUT_PADDING)
    assert any(v != 0 for v in out), f"dx {name}: gold is vacuously all-zero"

    # Cross-check vs the PyTorch float autograd reference on the SAME
    # dequantized operands: the only divergence is the emulation's
    # once-per-group-run combine rounding, <= 0.5*C_i*s_acc per element.
    w_deq = torch.tensor(
        [float(m) * w_scales[i // group_size] for i, m in enumerate(W_MANTISSAS)],
        dtype=torch.float32).reshape(IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE)
    gy = (torch.tensor(DX_LOSS_MANTISSAS, dtype=torch.float32) * DX_LOSS_SCALE).reshape(
        BATCH, OUT_CHANNELS, OUT_LEN)
    x = torch.zeros(BATCH, IN_CHANNELS, INPUT_LENGTH, dtype=torch.float32, requires_grad=True)
    y = F.conv_transpose1d(x, w_deq, stride=STRIDE, dilation=DILATION,
                           output_padding=OUTPUT_PADDING)
    y.backward(gy)
    ref = x.grad.flatten().tolist()
    counts = dx_combine_counts(group_size)
    assert len(ref) == len(out) == len(counts)
    for i, (g, r) in enumerate(zip(out, ref)):
        bound = 0.5 * counts[i] * s_acc + 1e-4
        assert abs(g * s_acc - r) <= bound, (
            f"dx {name}: emulation deviates from torch autograd beyond the gather "
            f"bound at {i}: |{g * s_acc} - {r}| > {bound}")
    return {"outMantissas": out, "outScale": s_acc, "maxCombines": max(counts)}


def emit_fixture(parts, prefix, fx):
    parts.append(emit_float_array(f"k{prefix}WScales", torch.tensor(fx["wScales"])))
    parts.append(emit_int32_scalar(f"k{prefix}GroupSize", fx["groupSize"]))
    parts.append(emit_int32_scalar(f"k{prefix}NumGroups", fx["numGroups"]))
    parts.append(emit_int32_array(f"k{prefix}OutMantissas", torch.tensor(fx["outMantissas"])))
    parts.append(emit_float_scalar(f"k{prefix}OutScale", fx["outScale"]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    assert_rounding_canary()

    per_channel = fixture("perChannel", [0.02, 0.05], 9, 2)
    general = fixture("general", [0.02, 0.05, 0.01, 0.08, 0.03, 0.06], 3, 6)

    # (iv) rounding discriminator on the general fixture.
    assert ref_truncating(general["wScales"], general["groupSize"]) != general["outMantissas"], (
        "general: truncation reproduces the gold (rounding-mode mutation vacuous)")

    check_equal_scales_reproduces_scalar()

    c_max = max_products_per_out()
    assert c_max == IN_CHANNELS * KERNEL_SIZE, (
        f"max products per output element is {c_max}, expected "
        f"{IN_CHANNELS * KERNEL_SIZE} -- the C float-path tolerance comment "
        "counts Cin*K fully-overlapped taps; re-derive it for this geometry")

    parts = [
        "// AUTOGENERATED by generate_expected_convT1d_grouped.py — DO NOT EDIT\n",
        "#ifndef ODT_EXPECTED_CONVT1D_GROUPED_H\n",
        "#define ODT_EXPECTED_CONVT1D_GROUPED_H\n",
        "#include <stdint.h>\n",
        "#include <stdlib.h>\n\n",
        f"static const int32_t kConvTGroupedWMantissas[] = "
        f"{{ {', '.join(str(v) for v in W_MANTISSAS)} }};\n",
        f"static const int32_t kConvTGroupedXMantissas[] = "
        f"{{ {', '.join(str(v) for v in X_MANTISSAS)} }};\n",
        f"static const float kConvTGroupedXScale = {X_SCALE}f;\n",
        f"static const int32_t kConvTGroupedBiasMantissas[] = "
        f"{{ {', '.join(str(v) for v in BIAS_MANTISSAS)} }};\n",
        f"static const float kConvTGroupedBiasScale = {BIAS_SCALE}f;\n",
        f"static const int32_t kConvTGroupedBatch = {BATCH};\n",
        f"static const int32_t kConvTGroupedInChannels = {IN_CHANNELS};\n",
        f"static const int32_t kConvTGroupedOutChannels = {OUT_CHANNELS};\n",
        f"static const int32_t kConvTGroupedKernelSize = {KERNEL_SIZE};\n",
        f"static const int32_t kConvTGroupedInputLength = {INPUT_LENGTH};\n",
        f"static const int32_t kConvTGroupedOutLen = {OUT_LEN};\n",
        f"static const int32_t kConvTGroupedMaxProductsPerOut = {c_max};\n",
        "\n",
    ]

    emit_fixture(parts, "ConvTPerChannel", per_channel)
    emit_fixture(parts, "ConvTGeneral", general)

    # PR3 Task 3: dx fixtures reuse the forward fixtures' group shapes/scales.
    dx_pc = fixture_dx("perChannel", per_channel["wScales"], per_channel["groupSize"])
    dx_gen = fixture_dx("general", general["wScales"], general["groupSize"])
    assert ([v * dx_pc["outScale"] for v in dx_pc["outMantissas"]] !=
            [v * dx_gen["outScale"] for v in dx_gen["outMantissas"]]), (
        "dx: perChannel and general golds dequantize identically -- the group "
        "shape does not reach the dx path (fixture vacuous)")
    # Concrete combine counts for the C float-path tolerance derivation:
    # perChannel groups are whole Cin slabs, so every reduction stays in ONE
    # group (C=1); general groups are single (ic, oc) tap-rows, so every
    # reduction crosses Cout=3 groups (C=3).
    assert dx_pc["maxCombines"] == 1, "dx perChannel: expected exactly 1 combine per element"
    assert dx_gen["maxCombines"] == OUT_CHANNELS, (
        f"dx general: expected {OUT_CHANNELS} combines per element")

    parts.append(emit_int32_array("kConvTDxLossMantissas", torch.tensor(DX_LOSS_MANTISSAS)))
    parts.append(emit_float_scalar("kConvTDxLossScale", DX_LOSS_SCALE))
    parts.append(emit_int32_array("kConvTDxPerChannelOutMantissas",
                                  torch.tensor(dx_pc["outMantissas"])))
    parts.append(emit_float_scalar("kConvTDxPerChannelOutScale", dx_pc["outScale"]))
    parts.append(emit_int32_scalar("kConvTDxPerChannelMaxCombines", dx_pc["maxCombines"]))
    parts.append(emit_int32_array("kConvTDxGeneralOutMantissas",
                                  torch.tensor(dx_gen["outMantissas"])))
    parts.append(emit_float_scalar("kConvTDxGeneralOutScale", dx_gen["outScale"]))
    parts.append(emit_int32_scalar("kConvTDxGeneralMaxCombines", dx_gen["maxCombines"]))

    parts.append("\n#endif // ODT_EXPECTED_CONVT1D_GROUPED_H\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(parts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
