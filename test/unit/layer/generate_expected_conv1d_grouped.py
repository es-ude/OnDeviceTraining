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

Run via `uv run` (CMake wires this automatically, see CMakeLists.txt).
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "goldgen"))

from sym_gold import (combine_quotient_f32, conv1d_grouped_ref, emit_float_array,
                      emit_float_scalar, emit_int32_array, emit_int32_scalar,
                      window_geometry_1d, window_slice_1d_full)

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

    parts.append("\n#endif // ODT_EXPECTED_CONV1D_GROUPED_H\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(parts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
