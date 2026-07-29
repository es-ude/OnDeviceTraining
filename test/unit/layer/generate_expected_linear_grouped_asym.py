#!/usr/bin/env python3
"""Generate expected_linear_grouped_asym.h for UnitTestLinear (group-quant PR4
Task 3 -- spec docs/superpowers/specs/2026-07-28-group-quantization-design.md).

Pins Linear's grouped-ASYM weight forward AND dx through the funnel's ASYM
grouped-unpack arm (ExecuteOp.c): the prologue zero-extends the packed codes
and shifts each element by ITS group's code-domain zeroPoint (code - zp[g])
into the SAME signed-mantissa image the SYM arm produces; the kernels then
read per-group scales via the layer's symQConfig-shaped VIEW of the asym
config. THAT IS D5 MADE EXECUTABLE: the grouped ASYM forward/dx is EXACTLY
the grouped SYM forward/dx on shifted mantissas -- so the golds here are the
EXISTING symmetric references (sym_gold.matmul_grouped_ref /
matmul_grouped_dx_ref) fed with mantissas = codes - zps[g] and the asym
scales; no new reference arithmetic exists for ASYM at all.

Fixture (mirrors generate_expected_group_matmul.py's perChannel shape):
a 2x6 SYM_INT32 input, a 3x6 grouped-ASYM weight (qBits=8, numGroups=3,
groupSize=6 == one group per output channel/weight row), SYM_INT32 bias.
The weight grid comes from quantize_asym_grouped (the Task-2 nudged
code-domain quantizer emulation) over hand-picked float values whose per-group
bands straddle 0 with distinct spans.

Self-checks (abort generation on failure):
  (i)   zps pairwise distinct AND strictly interior (0 < zp < 2^8-1) -- a
        zp[g] -> zp[0] unpack-shift mutation must change the mantissas, and
        interior zps guarantee mantissas of BOTH signs (sign-handling
        coverage in the zero-extend+shift path).
  (ii)  scales pairwise distinct and scales[0] != max(scales) (s_acc
        discriminability, generate_expected_group_matmul.py precedent).
  (iii) D5 equivalence, asserted EXACTLY: the per-element affine dequant
        (code - zp[g])*scale[g] (dequant_asym_grouped, the C grouped
        ASYM->FLOAT32 cell's float32 mirror) equals the symmetric dequant
        mantissa*scale[g] (dequant_sym_grouped_f32) bit-for-bit -- the
        shifted-mantissa identity the whole Task-3 design rests on.
  (iv)  zp[0]-collapse discriminability: shifting EVERY group by zps[0]
        (the unpack mutation) yields DIFFERENT forward out mantissas.
  (v)   mantissas within the int12 operand contract (|m| <= 2047).
  (vi)  dx lossGrad rows pairwise distinct (uniform-lossGrad vacuity lesson).

Run via `uv run` (CMake wires this automatically, see CMakeLists.txt).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "goldgen"))

from sym_gold import (dequant_asym_grouped, dequant_sym_grouped_f32, emit_float_array,
                      emit_float_scalar, emit_int32_array, emit_int32_scalar,
                      matmul_grouped_dx_ref, matmul_grouped_ref, quantize_asym_grouped)

Q_BITS = 8
OUT_ROWS = 2
OUT_COLS = 3
REDUCE_LEN = 6
GROUP_SIZE = 6   # per output channel (one group per weight row)
NUM_GROUPS = 3
N_WEIGHTS = OUT_COLS * REDUCE_LEN  # 18

# Weight float values, [out_cols=3, reduce_len=6] storage order (the
# GEMM-weight layout Linear.c's transposeTensor(w,0,1) exposes contiguously).
# Per-group bands straddle 0 with distinct spans -> distinct scales AND
# distinct interior zps after the nudged grid derivation.
W_FLOATS = [0.8, -0.6, 0.4, -0.2, 1.0, -0.4,
            0.3, 0.6, -1.2, 0.9, -0.3, 0.6,
            -0.5, 0.75, 0.25, -1.25, 1.0, -0.75]

# a: [out_rows=2, reduce_len=6] SYM_INT32 mantissas (int12-safe), row-major.
A_MANTISSAS = [1, -2, 3, -1, 2, -3,
               2, 1, -1, 3, -2, 1]
A_SCALE = 0.5

BIAS_MANTISSAS = [10, -5, 3]
BIAS_SCALE = 0.1

# dx orientation: out[r][k] = sum_o loss[r][o] * W[o][k], W strided by
# REDUCE_LEN (its raw [outFeatures=3, inFeatures=6] storage).
DX_OUT_ROWS = 2     # batch
DX_OUT_COLS = 6     # inFeatures
DX_REDUCE_LEN = 3   # outFeatures

LOSS_MANTISSAS = torch.randint(
    -40, 41, (DX_OUT_ROWS * DX_REDUCE_LEN,),
    generator=torch.Generator().manual_seed(20260730)).tolist()
LOSS_SCALE = 0.5


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    codes, scales, zps = quantize_asym_grouped(W_FLOATS, Q_BITS, GROUP_SIZE)

    # (i) zp distinctness + interiority.
    assert len(set(zps)) == len(zps), "zps must be pairwise distinct (mutation ii pin)"
    assert all(0 < z < 2 ** Q_BITS - 1 for z in zps), (
        "zps must be strictly interior so mantissas carry both signs")

    # (ii) scale discriminability.
    assert len(set(scales)) == len(scales), "scales must be pairwise distinct"
    assert scales[0] != max(scales), "scales[0] must differ from max (s_acc pin)"

    # The shifted-mantissa image the funnel's ASYM grouped-unpack arm produces.
    mantissas = [c - zps[i // GROUP_SIZE] for i, c in enumerate(codes)]

    # (iii) D5 equivalence, exact.
    affine = dequant_asym_grouped(codes, scales, zps, GROUP_SIZE)
    shifted = dequant_sym_grouped_f32(mantissas, scales, GROUP_SIZE).tolist()
    assert affine == shifted, (
        "D5 violated: affine dequant != symmetric dequant of shifted mantissas")

    # (v) operand contract.
    assert all(abs(m) <= 2047 for m in mantissas), "mantissas exceed int12 operand bound"
    assert any(m < 0 for m in mantissas) and any(m > 0 for m in mantissas), (
        "mantissas must carry both signs")

    # Forward gold: the EXISTING symmetric grouped reference on the shifted
    # mantissas (D5) -- python-int MACs, rescale_f32(HALF_AWAY) combines.
    out, s_acc = matmul_grouped_ref(A_MANTISSAS, A_SCALE, mantissas, scales, GROUP_SIZE,
                                    OUT_ROWS, OUT_COLS, REDUCE_LEN, BIAS_MANTISSAS, BIAS_SCALE)

    # (iv) zp[0]-collapse discriminability (the unpack-shift mutation).
    mantissas_zp0 = [c - zps[0] for c in codes]
    out_mut, _ = matmul_grouped_ref(A_MANTISSAS, A_SCALE, mantissas_zp0, scales, GROUP_SIZE,
                                    OUT_ROWS, OUT_COLS, REDUCE_LEN, BIAS_MANTISSAS, BIAS_SCALE)
    assert out_mut != out, (
        "zp[0]-everywhere shift is indistinguishable from gold -- fixture is "
        "vacuous against the zp[g] -> zp[0] unpack mutation")

    # dx gold: same shifted mantissas through the strided grouped dx reference.
    dx_out, dx_s_acc = matmul_grouped_dx_ref(LOSS_MANTISSAS, LOSS_SCALE, mantissas, scales,
                                             GROUP_SIZE, DX_OUT_ROWS, DX_OUT_COLS, DX_REDUCE_LEN)

    # (vi) row-distinct lossGrad.
    rows = [tuple(LOSS_MANTISSAS[r * DX_REDUCE_LEN:(r + 1) * DX_REDUCE_LEN])
            for r in range(DX_OUT_ROWS)]
    assert len(set(rows)) == DX_OUT_ROWS, "lossGrad rows must be pairwise distinct"

    parts = [
        "// AUTOGENERATED by generate_expected_linear_grouped_asym.py - DO NOT EDIT\n",
        "#ifndef ODT_EXPECTED_LINEAR_GROUPED_ASYM_H\n",
        "#define ODT_EXPECTED_LINEAR_GROUPED_ASYM_H\n",
        "#include <stdint.h>\n",
        "#include <stdlib.h>\n\n",
    ]
    parts.append(emit_int32_scalar("kLinAsymQBits", Q_BITS))
    parts.append(emit_int32_scalar("kLinAsymNumGroups", NUM_GROUPS))
    parts.append(emit_int32_scalar("kLinAsymGroupSize", GROUP_SIZE))
    parts.append(emit_int32_array("kLinAsymWCodes", torch.tensor(codes)))
    parts.append(emit_float_array("kLinAsymWScales", torch.tensor(scales)))
    parts.append(emit_int32_array("kLinAsymWZps", torch.tensor(zps)))
    parts.append(emit_int32_array("kLinAsymWMantissas", torch.tensor(mantissas)))
    parts.append(emit_int32_array("kLinAsymAMantissas", torch.tensor(A_MANTISSAS)))
    parts.append(emit_float_scalar("kLinAsymAScale", A_SCALE))
    parts.append(emit_int32_array("kLinAsymBiasMantissas", torch.tensor(BIAS_MANTISSAS)))
    parts.append(emit_float_scalar("kLinAsymBiasScale", BIAS_SCALE))
    parts.append(emit_int32_array("kLinAsymOutMantissas", torch.tensor(out)))
    parts.append(emit_float_scalar("kLinAsymOutScale", s_acc))
    parts.append(emit_int32_array("kLinAsymDxLossMantissas", torch.tensor(LOSS_MANTISSAS)))
    parts.append(emit_float_scalar("kLinAsymDxLossScale", LOSS_SCALE))
    parts.append(emit_int32_array("kLinAsymDxOutMantissas", torch.tensor(dx_out)))
    parts.append(emit_float_scalar("kLinAsymDxOutScale", dx_s_acc))
    parts.append("\n#endif // ODT_EXPECTED_LINEAR_GROUPED_ASYM_H\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(parts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
