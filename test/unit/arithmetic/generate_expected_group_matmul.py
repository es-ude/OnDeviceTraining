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

Run via `uv run` (CMake wires this automatically, see CMakeLists.txt).
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "goldgen"))

from sym_gold import (combine_quotient_f32, emit_float_array, emit_float_scalar,
                      emit_int32_array, emit_int32_scalar, matmul_grouped_ref)

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

    parts.append("\n#endif // ODT_EXPECTED_GROUP_MATMUL_H\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(parts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
