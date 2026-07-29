#!/usr/bin/env python3
"""Generate expected_group_matmul.h for UnitTestMatmul (group-quant PR2, Task 3
-- spec docs/superpowers/specs/2026-07-28-group-quantization-design.md).

Pins matmulSymInt32TensorsGroupedWeight's GGUF-style running group-partial
rescale-combine (matmulIntCoreGrouped): int MACs per group (exact), a
rescale-combine at every group boundary AND at the reduction's end via
rescaleIntoAccumulatorScale(HALF_AWAY), s_acc = a_scale * max_g(w_scales[g])
(never scales[0] -- that would grow some group's rescale factor past 1 and
overflow accumulator headroom). See sym_gold.matmul_grouped_ref for the exact
float32-mirrored emulation.

Both fixtures share the SAME `a` (2x6, batch=2/inFeatures=6) and the SAME
raw weight mantissas (3x6 row-major, one row per output channel -- the
GEMM-weight storage order Linear.c's transposeTensor(w,0,1) always exposes as
storage-contiguous in the reduction axis) and bias; only the GROUP SHAPE
(and therefore the per-group scales) differs:

  perChannel  groupSize=6 (== the full reduction length per output channel,
              numGroups=3): each output channel is exactly ONE group, so the
              running-partial loop crosses a group boundary zero times --
              the ONLY combine is the post-loop tail combine (per-channel
              case, spec "exactly ONE combine per output element").
  general     groupSize=3 (numGroups=6, two groups per output channel): one
              mid-loop boundary combine + one tail combine per output
              element (two combines/row).

Self-checks (mutation-discriminating fixture properties, asserted here so a
broken fixture aborts generation rather than silently passing a vacuous
test):
  (i)   general fixture's scales[0] != max(scales) -- a `sAcc` bug that reads
        scales[0] instead of the max would silently use the wrong accumulator
        scale and produce a different result.
  (ii)  the LAST group's raw contribution is nonzero for both fixtures -- a
        bug that drops the post-loop tail combine would zero out that
        group's contribution entirely.
  (iii) at least one combine in the general fixture has a float32 quotient
        whose |fractional part| >= 0.5 -- round-half-away and truncate-
        toward-zero disagree there, so a combine that truncates instead of
        rounding is caught. (SR_HALF_AWAY vs HALF_AWAY divergence is NOT
        emulated here -- SR needs the C-side seeded RNG stream, which no
        existing goldgen script emulates; that mutation direction is instead
        covered by testMatmulGroupedHonorsOpRoundingMode in UnitTestMatmul.c,
        which runs the real C kernel under SR_HALF_AWAY with two RNG seeds
        and asserts the outputs differ -- proving the combine's rounding
        mode is not hardcoded. Disclosed substitution, see task-3-report.md.)

Group-quant PR3 Task 1 adds the dx (Linear propLoss) orientation fixtures
(sym_gold.matmul_grouped_dx_ref): out[r][k] = sum_o loss[r][o] * W[o][k]
with W in its RAW [outFeatures, inFeatures] storage order -- the reduction
axis is storage-STRIDED (stride = inFeatures), so consecutive reduction
steps hop groups and the per-element group binding (`g = w_idx //
group_size`, the unified matmulIntCoreGrouped walk) combines on EVERY
visited-group change. Same weight mantissas/scales as the forward fixtures;
the lossGrad operand is seeded pseudo-random with row-distinct values
(uniform-lossGrad vacuity lesson). Self-checks (asserted, abort generation):
  (a) >=2 distinct group scales and scales[0] != max(scales);
  (b) at least one combine quotient where HALF_AWAY and truncation disagree
      (|frac| >= 0.5) -- pins a truncation mutation;
  (c) the emulated dx dequant agrees with the exact float64 reference within
      0.5*C*s_acc per element (C = that element's combine count) AND its
      mantissas differ from a scales[0]-everywhere collapse emulation --
      pins the s_acc-from-scales[0] mutation;
  (d) the last visited group's raw partial is nonzero somewhere -- pins a
      dropped tail combine (both dx fixtures);
  (e) lossGrad rows are pairwise distinct.

test/unit/layer/UnitTestLinear.c hand-duplicates this SAME a/weight/bias/
output data as its own file-local literals (kGroupedWMantissas,
kGroupedWScales, kGroupedBiasMantissas, kGroupedOutMantissas, and the PR3
dx literals kGroupedDxLossMantissas/kGroupedDxOutMantissas/
kGroupedDxOutScale -- see that file's comment for why it isn't shared
across test binaries). If you regenerate this fixture, update those
literals by hand to match -- they do NOT update automatically.

Run via `uv run` (CMake wires this automatically, see CMakeLists.txt).
"""
import math
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "goldgen"))

from sym_gold import (combine_quotient_f32, emit_float_array, emit_float_scalar,
                      emit_int32_array, emit_int32_scalar, matmul_grouped_dx_ref,
                      matmul_grouped_ref)

OUT_ROWS = 2
OUT_COLS = 3
REDUCE_LEN = 6

# a: [out_rows=2, reduce_len=6], row-major.
A_MANTISSAS = [1, -2, 3, -1, 2, -3,
              2, 1, -1, 3, -2, 1]
A_SCALE = 0.5

# b (weight): [out_cols=3, reduce_len=6] STORAGE order, one row per output
# channel (row c = c*REDUCE_LEN .. c*REDUCE_LEN+5).
W_MANTISSAS = [4, -3, 2, -1, 5, -2,
              1, 2, -4, 3, -1, 2,
              -2, 3, 1, -5, 4, -3]

BIAS_MANTISSAS = [10, -5, 3]
BIAS_SCALE = 0.1

# ---- PR3 Task 1: dx (Linear propLoss) orientation --------------------------
# out[r][k] = sum_o loss[r][o] * W[o][k]; W stays in its RAW [outFeatures=3,
# inFeatures=6] storage order (the SAME W_MANTISSAS flat array as above), so
# the dx reduction (over o) strides weight storage by 6.
DX_OUT_ROWS = 2     # batch
DX_OUT_COLS = 6     # inFeatures (dx output width)
DX_REDUCE_LEN = 3   # outFeatures (dx reduction length)

# Seeded pseudo-random lossGrad mantissas, row-distinct (vacuity lesson: a
# uniform lossGrad hides orientation/indexing bugs). int12-operand safe.
LOSS_MANTISSAS = torch.randint(
    -40, 41, (DX_OUT_ROWS * DX_REDUCE_LEN,),
    generator=torch.Generator().manual_seed(20260729)).tolist()
LOSS_SCALE = 0.5


def dx_walk_stats(w_scales, group_size):
    """Mirrors matmul_grouped_dx_ref's walk, collecting per-element combine
    stats for the self-checks: (quotients: every combine's float32 quotient,
    last_partials: the tail combine's raw partial per element, combine_counts:
    combines per element)."""
    max_scale = max(w_scales)
    s_acc = (torch.tensor(LOSS_SCALE, dtype=torch.float32) *
            torch.tensor(max_scale, dtype=torch.float32)).item()
    quotients, last_partials, combine_counts = [], [], []
    for r in range(DX_OUT_ROWS):
        for k in range(DX_OUT_COLS):
            partial, current_group, combines = 0, None, 0
            for o in range(DX_REDUCE_LEN):
                w_idx = o * DX_OUT_COLS + k
                g = w_idx // group_size
                if g != current_group:
                    if current_group is not None:
                        param_scale = (torch.tensor(LOSS_SCALE, dtype=torch.float32) *
                                      torch.tensor(w_scales[current_group],
                                                   dtype=torch.float32)).item()
                        quotients.append(combine_quotient_f32(partial, param_scale, s_acc))
                        combines += 1
                    partial = 0
                    current_group = g
                partial += LOSS_MANTISSAS[r * DX_REDUCE_LEN + o] * W_MANTISSAS[w_idx]
            param_scale = (torch.tensor(LOSS_SCALE, dtype=torch.float32) *
                          torch.tensor(w_scales[current_group], dtype=torch.float32)).item()
            quotients.append(combine_quotient_f32(partial, param_scale, s_acc))
            combines += 1
            last_partials.append(partial)
            combine_counts.append(combines)
    return quotients, last_partials, combine_counts


def check_dx_fixture(name, w_scales, group_size, num_groups, out, s_acc):
    """Self-checks (a)-(e) from the module docstring; abort on any failure."""
    # (a) scale-shape discriminability.
    assert len(set(w_scales)) >= 2, f"{name}: needs >=2 distinct group scales"
    assert w_scales[0] != max(w_scales), f"{name}: scales[0] must differ from max"

    quotients, last_partials, combine_counts = dx_walk_stats(w_scales, group_size)

    # (b) at least one combine where HALF_AWAY and truncate-toward-zero
    # disagree: |frac| >= 0.5 (magnitude argument holds for either sign).
    assert any(math.floor(abs(q) + 0.5) != math.trunc(abs(q)) for q in quotients), (
        f"{name}: no combine quotient with |frac| >= 0.5 -- fixture is vacuous "
        "against a truncation-instead-of-rounding mutation")

    # (c1) emulated dx vs exact float64 reference: per-element bound
    # 0.5 * C * s_acc (C combines, each one HALF_AWAY rounding of <= 0.5
    # accumulator-quanta error; float32 rescale noise is orders below).
    for r in range(DX_OUT_ROWS):
        for k in range(DX_OUT_COLS):
            ref = 0.0
            for o in range(DX_REDUCE_LEN):
                w_idx = o * DX_OUT_COLS + k
                g = w_idx // group_size
                ref += (LOSS_MANTISSAS[r * DX_REDUCE_LEN + o] * LOSS_SCALE *
                        W_MANTISSAS[w_idx] * w_scales[g])
            i = r * DX_OUT_COLS + k
            bound = 0.5 * combine_counts[i] * s_acc
            assert abs(out[i] * s_acc - ref) <= bound, (
                f"{name}: element {i} emulation {out[i] * s_acc} vs float ref {ref} "
                f"exceeds 0.5*C*s_acc = {bound}")

    # (c2) collapse discriminability: a scales[0]-everywhere emulation (the
    # s_acc-from-scales[0] mutation collapses every rescale factor to 1) must
    # produce DIFFERENT mantissas.
    out_mut, _ = matmul_grouped_dx_ref(LOSS_MANTISSAS, LOSS_SCALE, W_MANTISSAS,
                                       [w_scales[0]] * num_groups, group_size,
                                       DX_OUT_ROWS, DX_OUT_COLS, DX_REDUCE_LEN)
    assert out_mut != out, (
        f"{name}: scales[0]-everywhere emulation is indistinguishable from gold "
        "(fixture is vacuous against the s_acc collapse mutation)")

    # (d) dropped-tail-combine discriminability: the tail partial must be
    # nonzero somewhere.
    assert any(p != 0 for p in last_partials), (
        f"{name}: every tail-combine partial is zero -- dropping the tail "
        "combine would be invisible")

    # (e) row-distinct lossGrad.
    rows = [tuple(LOSS_MANTISSAS[r * DX_REDUCE_LEN:(r + 1) * DX_REDUCE_LEN])
            for r in range(DX_OUT_ROWS)]
    assert len(set(rows)) == DX_OUT_ROWS, "lossGrad rows must be pairwise distinct"


def fixture_dx(name, group_size, num_groups, w_scales):
    assert len(w_scales) == num_groups
    out, s_acc = matmul_grouped_dx_ref(LOSS_MANTISSAS, LOSS_SCALE, W_MANTISSAS, w_scales,
                                       group_size, DX_OUT_ROWS, DX_OUT_COLS, DX_REDUCE_LEN)
    check_dx_fixture(name, w_scales, group_size, num_groups, out, s_acc)
    return {"lossMantissas": LOSS_MANTISSAS, "lossScale": LOSS_SCALE,
           "wMantissas": W_MANTISSAS, "wScales": w_scales, "groupSize": group_size,
           "numGroups": num_groups, "outRows": DX_OUT_ROWS, "outCols": DX_OUT_COLS,
           "reduceLen": DX_REDUCE_LEN, "outMantissas": out, "outScale": s_acc}


def fixture_per_channel():
    group_size, num_groups = 6, 3
    w_scales = [0.02, 0.05, 0.01]
    assert len(w_scales) == num_groups

    out, s_acc = matmul_grouped_ref(A_MANTISSAS, A_SCALE, W_MANTISSAS, w_scales, group_size,
                                    OUT_ROWS, OUT_COLS, REDUCE_LEN, BIAS_MANTISSAS, BIAS_SCALE)

    # Mutation (ii): last group's (group 2, output column 2) raw contribution
    # must be nonzero -- else dropping the tail combine would be invisible.
    last_group_partial = sum(a * w for a, w in zip(A_MANTISSAS[0:REDUCE_LEN],
                                                    W_MANTISSAS[2 * REDUCE_LEN:3 * REDUCE_LEN]))
    assert last_group_partial != 0, "perChannel: last group's contribution is vacuously zero"

    return {"aMantissas": A_MANTISSAS, "aScale": A_SCALE, "wMantissas": W_MANTISSAS,
           "wScales": w_scales, "groupSize": group_size, "numGroups": num_groups,
           "biasMantissas": BIAS_MANTISSAS, "biasScale": BIAS_SCALE,
           "outRows": OUT_ROWS, "outCols": OUT_COLS, "reduceLen": REDUCE_LEN,
           "outMantissas": out, "outScale": s_acc}


def fixture_general_groups():
    group_size, num_groups = 3, 6
    # Deliberately scales[0] != max(scales) (mutation (i)'s pin) and a
    # non-uniform spread so every group differs from its row-mate.
    w_scales = [0.02, 0.05, 0.01, 0.08, 0.03, 0.06]
    assert w_scales[0] != max(w_scales), "general: scales[0] must differ from max (mutation i)"

    out, s_acc = matmul_grouped_ref(A_MANTISSAS, A_SCALE, W_MANTISSAS, w_scales, group_size,
                                    OUT_ROWS, OUT_COLS, REDUCE_LEN, BIAS_MANTISSAS, BIAS_SCALE)

    # Mutation (ii): last group (group 5, output column 2, k=3..5)'s raw
    # contribution must be nonzero.
    last_group_partial = sum(a * w for a, w in zip(A_MANTISSAS[0:3],
                                                    W_MANTISSAS[2 * REDUCE_LEN + 3:3 * REDUCE_LEN]))
    assert last_group_partial != 0, "general: last group's contribution is vacuously zero"

    # Mutation (iii) substitute: at least one combine's float32 quotient must
    # have |fractional part| >= 0.5 (round-half-away vs truncate-toward-zero
    # divergence point). Recompute every combine's quotient the same way
    # matmul_grouped_ref does internally.
    found_divergent = False
    for r in range(OUT_ROWS):
        for c in range(OUT_COLS):
            partial, current_group = 0, None
            for k in range(REDUCE_LEN):
                w_idx = c * REDUCE_LEN + k
                g = w_idx // group_size
                if g != current_group:
                    if current_group is not None:
                        param_scale = (torch.tensor(A_SCALE, dtype=torch.float32) *
                                      torch.tensor(w_scales[current_group],
                                                   dtype=torch.float32)).item()
                        q = combine_quotient_f32(partial, param_scale, s_acc)
                        if abs(abs(q) - int(abs(q)) - 0.5) < 1e-4:
                            found_divergent = True
                    partial = 0
                    current_group = g
                partial += A_MANTISSAS[r * REDUCE_LEN + k] * W_MANTISSAS[w_idx]
            param_scale = (torch.tensor(A_SCALE, dtype=torch.float32) *
                          torch.tensor(w_scales[current_group], dtype=torch.float32)).item()
            q = combine_quotient_f32(partial, param_scale, s_acc)
            if abs(abs(q) - int(abs(q)) - 0.5) < 1e-4:
                found_divergent = True
    assert found_divergent, (
        "general: no combine has a round-vs-truncate-divergent quotient "
        "(fixture is vacuous against a truncation-instead-of-rounding mutation)")

    return {"aMantissas": A_MANTISSAS, "aScale": A_SCALE, "wMantissas": W_MANTISSAS,
           "wScales": w_scales, "groupSize": group_size, "numGroups": num_groups,
           "biasMantissas": BIAS_MANTISSAS, "biasScale": BIAS_SCALE,
           "outRows": OUT_ROWS, "outCols": OUT_COLS, "reduceLen": REDUCE_LEN,
           "outMantissas": out, "outScale": s_acc}


def emit_fixture(parts, prefix, fx):
    parts.append(emit_int32_array(f"k{prefix}AMantissas", torch.tensor(fx["aMantissas"])))
    parts.append(emit_float_scalar(f"k{prefix}AScale", fx["aScale"]))
    parts.append(emit_int32_array(f"k{prefix}WMantissas", torch.tensor(fx["wMantissas"])))
    parts.append(emit_float_array(f"k{prefix}WScales", torch.tensor(fx["wScales"])))
    parts.append(emit_int32_scalar(f"k{prefix}GroupSize", fx["groupSize"]))
    parts.append(emit_int32_scalar(f"k{prefix}NumGroups", fx["numGroups"]))
    parts.append(emit_int32_array(f"k{prefix}BiasMantissas", torch.tensor(fx["biasMantissas"])))
    parts.append(emit_float_scalar(f"k{prefix}BiasScale", fx["biasScale"]))
    parts.append(emit_int32_scalar(f"k{prefix}OutRows", fx["outRows"]))
    parts.append(emit_int32_scalar(f"k{prefix}OutCols", fx["outCols"]))
    parts.append(emit_int32_scalar(f"k{prefix}ReduceLen", fx["reduceLen"]))
    parts.append(emit_int32_array(f"k{prefix}OutMantissas", torch.tensor(fx["outMantissas"])))
    parts.append(emit_float_scalar(f"k{prefix}OutScale", fx["outScale"]))


def emit_dx_fixture(parts, prefix, fx):
    parts.append(emit_int32_array(f"k{prefix}LossMantissas", torch.tensor(fx["lossMantissas"])))
    parts.append(emit_float_scalar(f"k{prefix}LossScale", fx["lossScale"]))
    parts.append(emit_int32_array(f"k{prefix}WMantissas", torch.tensor(fx["wMantissas"])))
    parts.append(emit_float_array(f"k{prefix}WScales", torch.tensor(fx["wScales"])))
    parts.append(emit_int32_scalar(f"k{prefix}GroupSize", fx["groupSize"]))
    parts.append(emit_int32_scalar(f"k{prefix}NumGroups", fx["numGroups"]))
    parts.append(emit_int32_scalar(f"k{prefix}OutRows", fx["outRows"]))
    parts.append(emit_int32_scalar(f"k{prefix}OutCols", fx["outCols"]))
    parts.append(emit_int32_scalar(f"k{prefix}ReduceLen", fx["reduceLen"]))
    parts.append(emit_int32_array(f"k{prefix}OutMantissas", torch.tensor(fx["outMantissas"])))
    parts.append(emit_float_scalar(f"k{prefix}OutScale", fx["outScale"]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    parts = [
        "// AUTOGENERATED by generate_expected_group_matmul.py — DO NOT EDIT\n",
        "#ifndef ODT_EXPECTED_GROUP_MATMUL_H\n",
        "#define ODT_EXPECTED_GROUP_MATMUL_H\n",
        "#include <stdint.h>\n",
        "#include <stdlib.h>\n\n",
    ]

    emit_fixture(parts, "PerChannel", fixture_per_channel())
    emit_fixture(parts, "General", fixture_general_groups())
    # dx fixtures reuse the forward weight mantissas/scale shapes: per-channel
    # (groupSize=6, one group per weight row) and general (groupSize=3).
    emit_dx_fixture(parts, "DxPerChannel",
                    fixture_dx("dxPerChannel", 6, 3, [0.02, 0.05, 0.01]))
    emit_dx_fixture(parts, "DxGeneral",
                    fixture_dx("dxGeneral", 3, 6, [0.02, 0.05, 0.01, 0.08, 0.03, 0.06]))

    parts.append("\n#endif // ODT_EXPECTED_GROUP_MATMUL_H\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(parts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
