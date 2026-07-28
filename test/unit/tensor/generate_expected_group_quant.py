#!/usr/bin/env python3
"""Generate expected_group_quant.h for UnitTestTensorConversion (group-quant PR2,
Task 2 -- spec docs/superpowers/specs/2026-07-28-group-quantization-design.md).

Pins the per-GROUP (storage-order, group of element i = i // groupSize) absmax
symmetric quantization that packFloatBufferAsSym / convertSymTensorToFloat32Tensor
implement for numGroups > 1: scale_g = absMax_g == 0 ? 1.0 : absMax_g / qMax,
round-half-away-from-zero per element (matches the existing per-tensor formula,
applied per group). NEVER torch.round (half-to-even, silently diverges on ties);
see sym_gold.round_half_away.

Fixtures:
  groupQuant   12 floats, qBits=4, groupSize=4 (3 groups). The EMITTED input is
               the dequantization-ROUND-TRIP-STABLE float32 image of a hand-picked
               12-value fixture (stable_dequant_grouped) so the C side's
               tensorFillFromFloatBuffer lands on exactly the pinned codes; also
               serves as the expected output of the SYM->FLOAT32 grouped dequant
               (round trip). Self-check: the per-group scales must differ from
               the WHOLE-TENSOR (groupSize=12) reference scale on at least one
               group -- the mutation-discriminating property for "Phase-1 absmax
               over the whole tensor instead of the group" (a collapsed absmax
               would make every group's scale equal the whole-tensor one).
  torchXCheck  8x4 row-major weight (32 floats), qBits=8, groupSize=4 -- one
               group per output row (per-out-channel), the layout a GEMM weight
               groups over. Cross-checked in-generator against
               torch.quantize_per_channel(w, scales, zero_points=0, axis=0,
               dtype=torch.qint8).int_repr(): asserts exact code equality (a
               tie-only ±1 divergence would be tolerated if torch's
               round-half-to-even ever disagreed with our round-half-away on an
               exact x.5 quotient; any other mismatch aborts generation).

Self-checks abort generation (assert) instead of emitting a header that
contradicts its own emulation. Run via `uv run` (CMake wires this
automatically, see CMakeLists.txt).
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "goldgen"))

from sym_gold import (emit_float_array, emit_float_scalar, emit_int32_array,
                      emit_int32_scalar, quantize_sym_grouped,
                      stable_dequant_grouped)


def fixture_group_quant():
    # 3 groups of 4: absmax per group is 2.0 / 5.0 / 0.2 -- deliberately
    # distinct from each other AND from the whole-tensor absmax (5.0, from
    # group 1) so groups 0 and 2 pin the per-group-vs-whole-tensor divergence.
    raw = [1.5, -2.0, 0.3, -0.7,
          5.0, -4.5, 1.0, -0.25,
          0.1, -0.05, 0.2, -0.15]
    q_bits, group_size = 4, 4
    codes, scales, deq = stable_dequant_grouped(raw, q_bits, group_size)

    # Mutation-discriminating self-check: per-group scales must NOT all equal
    # the whole-tensor (groupSize=12) reference scale -- a Phase-1 bug that
    # takes absmax over the WHOLE tensor instead of the group would collapse
    # every group's scale to this single value.
    _, whole_tensor_scales = quantize_sym_grouped(deq, q_bits, 12)
    whole_tensor_scale = whole_tensor_scales[0]
    assert scales[0] != whole_tensor_scale, (
        "groupQuant: group 0 scale must differ from the whole-tensor scale "
        "(fixture is vacuous against the whole-tensor-absmax mutation)")
    assert scales[2] != whole_tensor_scale, (
        "groupQuant: group 2 scale must differ from the whole-tensor scale "
        "(fixture is vacuous against the whole-tensor-absmax mutation)")
    assert len(scales) == 3 and len(codes) == 12

    return {"input": deq, "codes": codes, "scales": scales,
           "qBits": q_bits, "groupSize": group_size, "numGroups": 3}


def fixture_torch_cross_check():
    # 8x4 row-major weight: one group per row (per-out-channel), qBits=8
    # (qint8 range [-128, 127] matches qMax = 2^7-1 = 127 exactly).
    raw = [0.3, -0.7, 1.1, -1.3,
          2.0, -2.0, 0.05, 0.9,
          -3.3, 3.1, 0.001, -0.002,
          0.6, -0.6, 0.61, -0.59,
          4.4, -4.1, 0.02, 0.03,
          1.9, -1.85, 0.5001, -0.4999,
          0.11, -0.12, 0.13, -0.14,
          5.5, -5.6, 0.001, -0.001]
    q_bits, group_size, num_groups = 8, 4, 8
    codes, scales, deq = stable_dequant_grouped(raw, q_bits, group_size)

    w = torch.tensor(deq, dtype=torch.float32).reshape(num_groups, group_size)
    scales_f32 = torch.tensor(scales, dtype=torch.float32)
    zero_points = torch.zeros(num_groups, dtype=torch.int64)
    torch_q = torch.quantize_per_channel(w, scales_f32, zero_points, axis=0,
                                        dtype=torch.qint8)
    torch_codes = torch_q.int_repr().flatten().tolist()

    q_max = 2.0 ** (q_bits - 1) - 1
    for i, (c, t) in enumerate(zip(codes, torch_codes)):
        if c == t:
            continue
        # Only an exact half-integer quotient can legitimately disagree
        # between round-half-away-from-zero (ours) and round-half-to-even
        # (torch's default quantizer) -- and only by exactly 1 code.
        scale = scales[i // group_size]
        quotient = deq[i] / scale
        is_tie = abs(abs(quotient) - (abs(quotient) // 1) - 0.5) < 1e-9
        assert is_tie and abs(c - t) == 1, (
            f"torchXCheck: element {i} mismatches torch.quantize_per_channel "
            f"beyond a rounding-tie boundary (ours={c}, torch's={t}, "
            f"quotient={quotient}) -- investigate, do not loosen this assert")

    return {"input": deq, "codes": codes, "scales": scales,
           "qBits": q_bits, "groupSize": group_size, "numGroups": num_groups}


def emit_fixture(parts, prefix, fx):
    parts.append(emit_float_array(f"k{prefix}Input", torch.tensor(fx["input"])))
    parts.append(emit_int32_array(f"k{prefix}Codes", torch.tensor(fx["codes"])))
    parts.append(emit_float_array(f"k{prefix}Scales", torch.tensor(fx["scales"])))
    parts.append(emit_int32_scalar(f"k{prefix}QBits", fx["qBits"]))
    parts.append(emit_int32_scalar(f"k{prefix}GroupSize", fx["groupSize"]))
    parts.append(emit_int32_scalar(f"k{prefix}NumGroups", fx["numGroups"]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    parts = [
        "// AUTOGENERATED by generate_expected_group_quant.py — DO NOT EDIT\n",
        "#ifndef ODT_EXPECTED_GROUP_QUANT_H\n",
        "#define ODT_EXPECTED_GROUP_QUANT_H\n",
        "#include <stdint.h>\n",
        "#include <stdlib.h>\n\n",
    ]

    emit_fixture(parts, "GroupQuant", fixture_group_quant())
    emit_fixture(parts, "TorchXCheck", fixture_torch_cross_check())

    parts.append("\n#endif // ODT_EXPECTED_GROUP_QUANT_H\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(parts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
