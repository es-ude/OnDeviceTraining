#!/usr/bin/env python3
"""Generate expected_max_pool_1d.h for UnitTestMaxPool1d (Layer-3 tests).

Produces a C header with PyTorch-derived ground-truth values for each
test fixture: forward output, expected argmax indices (int32_t),
propLoss (dL/dx). For most fixtures lossGrad = torch.ones_like(y);
the `withStrideAndDilation` fixture uses torch.randn_like(y) (and
emits the lossGrad as a gold array) so that index-permuting mutations
on the stride/dilation backward path are non-vacuous (per
codebase_uniform_lossgrad_mutation_vacuity).

Run via `uv run` (CMake wires this automatically).
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "goldgen"))
from sym_gold import (
    assert_rounding_canary,
    emit_float_scalar,
    emit_int32_scalar,
    emit_uint8_array,
    f32_scale_i12,
    maxpool1d_bfp_backward_ref,
    maxpool1d_bfp_forward_ref,
    requant_absmax_i12_f32,
    stable_dequant_i12,
    window_geometry_1d,
    window_slice_1d,
)


def _format_float_literal(v: float) -> str:
    """Format a Python float as a valid C float literal.

    Python's f"{v:.9g}" strips the trailing ".0" from whole-number
    floats (10.0 -> "10"), which combined with the f suffix produces
    invalid C (10f is rejected by gcc as an integer suffix). repr
    always contains "." or "e" for finite floats, both legal in C
    with the f suffix.
    """
    s = repr(v)
    if s in ("inf", "-inf", "nan"):
        raise ValueError(f"non-finite gold value: {v!r}")
    return s + "f"


def emit_float_array(name: str, tensor: torch.Tensor) -> str:
    flat = tensor.detach().flatten().tolist()
    body = ", ".join(_format_float_literal(v) for v in flat)
    return (
        f"static const float {name}[] = {{ {body} }};\n"
        f"static const size_t {name}_len = {len(flat)};\n"
    )


def emit_int32_array(name: str, tensor: torch.Tensor) -> str:
    flat = tensor.detach().to(torch.int32).flatten().tolist()
    body = ", ".join(str(v) for v in flat)
    return (
        f"static const int32_t {name}[] = {{ {body} }};\n"
        f"static const size_t {name}_len = {len(flat)};\n"
    )


def _hand_check_max_forward(x, kernel_size, stride, padding, dilation):
    """Numpy-level sanity check: for each output position, max over the
    valid window. Used as a self-check against PyTorch's MaxPool1d output."""
    import math
    B, C, L = x.shape
    in_padded_L = L + 2 * padding
    eff_k = (kernel_size - 1) * dilation + 1
    out_L = (in_padded_L - eff_k) // stride + 1
    out = torch.full((B, C, out_L), -math.inf)
    for b in range(B):
        for c in range(C):
            for o in range(out_L):
                start = o * stride - padding
                vals = []
                for k in range(kernel_size):
                    idx = start + k * dilation
                    if 0 <= idx < L:
                        vals.append(x[b, c, idx].item())
                if vals:
                    out[b, c, o] = max(vals)
                else:
                    out[b, c, o] = 0.0
    return out


def _run_max_fixture(name, x, *, kernel_size, stride, padding, dilation,
                     loss_grad_kind="ones"):
    """Run forward + autograd-backward on a MaxPool1d fixture; return all arrays.

    loss_grad_kind: "ones" -> lossGrad = torch.ones_like(y) (default);
                    "randn" -> lossGrad = torch.randn_like(y), reproducible via seed.
    """
    x_in = x.clone().detach().requires_grad_(True)

    # PyTorch's nn.functional.max_pool1d returns (output) by default; with
    # return_indices=True it returns (output, indices). The indices are 1D
    # input positions per output position (no batch/channel offset), matching
    # our argmax convention exactly.
    y, idx = F.max_pool1d(
        x_in, kernel_size=kernel_size, stride=stride,
        padding=padding, dilation=dilation, return_indices=True,
    )

    # Self-check forward against hand-computed reference.
    expected_y = _hand_check_max_forward(
        x_in.detach(), kernel_size, stride, padding, dilation,
    )
    assert torch.allclose(y.detach(), expected_y, atol=1e-5), (
        f"{name}: PyTorch max_pool1d forward disagrees with hand-derived reference"
    )

    if loss_grad_kind == "ones":
        gy = torch.ones_like(y)
    elif loss_grad_kind == "randn":
        torch.manual_seed(hash(name) & 0xFFFF)
        gy = torch.randn_like(y)
    else:
        raise ValueError(loss_grad_kind)

    y.backward(gy)

    return {
        "name": name,
        "x": x_in.detach(),
        "y": y.detach(),
        "argmax": idx.detach(),    # int64 in PyTorch; we cast to int32 in emit_int32_array
        "dx": x_in.grad.detach(),
        "gy": gy.detach(),
        "loss_grad_kind": loss_grad_kind,
    }


def fixture_basic():
    # B=1, C=1, L=4, K=2, stride=1, VALID
    x = torch.tensor([[[1.0, 4.0, 2.0, 3.0]]])
    return _run_max_fixture("basic", x, kernel_size=2, stride=1, padding=0, dilation=1)


def fixture_multi_channel():
    # B=1, C=3, L=5, K=2, stride=1, VALID
    torch.manual_seed(101)
    x = torch.randn(1, 3, 5)
    return _run_max_fixture("multiChannel", x, kernel_size=2, stride=1,
                            padding=0, dilation=1)


def fixture_multi_batch():
    # B=4, C=2, L=4, K=2, stride=1, VALID
    torch.manual_seed(102)
    x = torch.randn(4, 2, 4)
    return _run_max_fixture("multiBatch", x, kernel_size=2, stride=1,
                            padding=0, dilation=1)


def fixture_with_stride_and_dilation():
    # B=1, C=1, L=9, K=2, stride=3, dilation=2, VALID
    # Effective kernel span = (2-1)*2+1 = 3; output length = (9-3)/3+1 = 3.
    # Using random lossGrad (Errata 3) so positional mutations are non-vacuous.
    torch.manual_seed(103)
    x = torch.randn(1, 1, 9)
    return _run_max_fixture("withStrideAndDilation", x,
                            kernel_size=2, stride=3, padding=0, dilation=2,
                            loss_grad_kind="randn")


def fixture_with_same_padding():
    # B=1, C=1, L=5, K=3, stride=1, dilation=1.
    # SAME for K=3, S=1, D=1 -> total pad = 2; padLeft=1, padRight=1 (symmetric).
    # Output length = 5. Edge windows at outPos=0 (validCount=2, drops left)
    # and outPos=4 (validCount=2, drops right) — exercises Errata 4.
    x = torch.tensor([[[5.0, 1.0, 4.0, 2.0, 6.0]]])
    return _run_max_fixture("withSamePadding", x, kernel_size=3, stride=1,
                            padding=1, dilation=1)


def fixture_edge_cases():
    # B=1, C=1, L=4, K=1, stride=1, VALID. K=1 -> identity pool;
    # output equals input, argmax equals position index, propLoss = lossGrad.
    # Ensures the validCount-loop with N=1 doesn't have an off-by-one
    # (e.g., loop bound `i < validCount - 1` would compute -INFINITY here).
    x = torch.tensor([[[7.0, -2.0, 3.0, 0.5]]])
    return _run_max_fixture("edgeCases", x, kernel_size=1, stride=1,
                            padding=0, dilation=1)


def emit_fixture(parts, fx):
    pre = f"maxPool1d_{fx['name']}"
    parts.append(emit_float_array(f"input_{pre}", fx["x"]))
    parts.append(emit_float_array(f"expectedForward_{pre}", fx["y"]))
    parts.append(emit_int32_array(f"expectedArgmax_{pre}", fx["argmax"]))
    parts.append(emit_float_array(f"expectedPropLoss_{pre}", fx["dx"]))
    if fx["loss_grad_kind"] != "ones":
        parts.append(emit_float_array(f"lossGrad_{pre}", fx["gy"]))


# ---------------- SYM_INT32 fixtures (#205) ----------------
#
# The SYM forward/backward kernels are integer-EXACT in the mantissa domain
# (pure select + scatter, no products, no divisions), so this emulation
# reproduces the raw kernel output bit-for-bit. The only C-vs-emulation gap is
# the executeOp OUT_WRITE restore (requantSymInt32Tensor): C evaluates
# m * (inScale / scale) where the emulation evaluates (m * inScale) / scale —
# a 1-ULP expression-order difference that can flip one rounding boundary.
# Hence MANTISSA_TOL = 1 and the Conv1d-precedent 1e-4 relative scale floor.
MANTISSA_TOL_SYM = 1
SCALE_REL_TOL_SYM = 1e-4


def _max_sym_forward(xq, geom):
    """Mirror maxPool1dForwardKernelSymInt32: per-window argmax over int32
    mantissas with strict > (first occurrence wins — same tie-break as the
    FLOAT32 arm); empty window -> (0, -1) sentinel. Quantized-domain argmax is
    the documented semantics: inputs that quantize to the same mantissa tie
    HERE even when their float values differ."""
    batch, channels, _ = xq.shape
    out_len = geom["out_len"]
    y = torch.zeros((batch, channels, out_len), dtype=torch.int32)
    am = torch.zeros((batch, channels, out_len), dtype=torch.int32)
    for b in range(batch):
        for c in range(channels):
            for o in range(out_len):
                first, count = window_slice_1d(geom, o)
                best = None
                best_idx = -1
                for i in range(count):
                    idx = first + i * geom["dilation"]
                    v = int(xq[b, c, idx])
                    if best is None or v > best:
                        best, best_idx = v, idx
                y[b, c, o] = best if count > 0 else 0
                am[b, c, o] = best_idx
    return y, am


def _run_max_sym_fixture(name, x, *, kernel_size, stride, dilation, padding_type,
                         loss_grad_kind="ones"):
    assert_rounding_canary()
    x = x.to(torch.float64)
    xq, _, x_deq = stable_dequant_i12(x)
    s_in = f32_scale_i12(x_deq)
    geom = window_geometry_1d(x.shape[2], kernel_size, stride, dilation, padding_type)

    y_raw, argmax = _max_sym_forward(xq, geom)
    yq_r, s_y = requant_absmax_i12_f32(y_raw, s_in)

    # C-semantics float64 reference: the selected mantissas at the input scale
    # (select is exact — the restore is the only rounding on this path).
    y_ref = y_raw.to(torch.float64) * s_in
    y_dequant_tol = (MANTISSA_TOL_SYM + 0.5) * s_y * 1.5
    err = (yq_r.to(torch.float64) * s_y - y_ref).abs().max().item()
    assert err <= y_dequant_tol, f"{name}: restored fwd dequant {err} > {y_dequant_tol}"

    if loss_grad_kind == "ones":
        gy = torch.ones((x.shape[0], x.shape[1], geom["out_len"]), dtype=torch.float64)
    elif loss_grad_kind == "randn":
        torch.manual_seed(hash(name) & 0xFFFF)
        gy = torch.randn(x.shape[0], x.shape[1], geom["out_len"], dtype=torch.float64)
    else:
        raise ValueError(loss_grad_kind)
    gyq, _, gy_deq = stable_dequant_i12(gy)
    s_gy = f32_scale_i12(gy_deq)

    # Mirror maxPool1dBackwardKernelSymInt32: zero + scatter gy mantissas to the
    # argmax positions (raw scale = s_gy), restored by the same OUT_WRITE diagonal.
    dx_raw = torch.zeros_like(xq)
    for b in range(x.shape[0]):
        for c in range(x.shape[1]):
            for o in range(geom["out_len"]):
                idx = int(argmax[b, c, o])
                if idx >= 0:
                    dx_raw[b, c, idx] += gyq[b, c, o]
    dxq_r, s_dx = requant_absmax_i12_f32(dx_raw, s_gy)

    dx_ref = dx_raw.to(torch.float64) * s_gy
    dx_dequant_tol = (MANTISSA_TOL_SYM + 0.5) * s_dx * 1.5
    err = (dxq_r.to(torch.float64) * s_dx - dx_ref).abs().max().item()
    assert err <= dx_dequant_tol, f"{name}: restored bwd dequant {err} > {dx_dequant_tol}"

    return {
        "name": name,
        "x": x_deq,
        "y_mantissas": yq_r, "y_scale": s_y,
        "y_dequant": y_ref.to(torch.float32), "y_dequant_tol": y_dequant_tol,
        "argmax": argmax,
        "gy": gy_deq,
        "dx_mantissas": dxq_r, "dx_scale": s_dx,
        "dx_dequant": dx_ref.to(torch.float32), "dx_dequant_tol": dx_dequant_tol,
    }


def fixture_sym_basic():
    # Same values as the FLOAT basic fixture: K=2, S=1, D=1, VALID.
    x = torch.tensor([[[1.0, 4.0, 2.0, 3.0]]])
    return _run_max_sym_fixture("symBasic", x, kernel_size=2, stride=1, dilation=1,
                                padding_type="VALID")


def fixture_sym_stride_dilation():
    # K=2, S=3, D=2, VALID on L=9 (outLen 3). Random lossGrad so positional
    # mutations on the SYM scatter path are non-vacuous.
    torch.manual_seed(205)
    x = torch.randn(1, 1, 9)
    return _run_max_sym_fixture("symStrideDilation", x, kernel_size=2, stride=3,
                                dilation=2, padding_type="VALID",
                                loss_grad_kind="randn")


def fixture_sym_tie_same_padding():
    # K=3, S=1, SAME on L=5 (padLeft=1): edge windows exercise validCount < K,
    # and x[0] == x[1] quantize to the SAME mantissa — the argmax gold pins the
    # first-occurrence tie-break (a `>` -> `>=` mutation flips it).
    x = torch.tensor([[[3.0, 3.0, 1.0, -2.0, 2.0]]])
    return _run_max_sym_fixture("symTieSamePadding", x, kernel_size=3, stride=1,
                                dilation=1, padding_type="SAME")


def emit_sym_fixture(parts, fx):
    pre = f"maxPool1dSym_{fx['name']}"
    parts.append(emit_float_array(f"input_{pre}", fx["x"]))
    parts.append(emit_int32_array(f"expectedForwardMantissas_{pre}", fx["y_mantissas"]))
    parts.append(emit_float_scalar(f"expectedForwardScale_{pre}", fx["y_scale"]))
    parts.append(emit_int32_array(f"expectedArgmax_{pre}", fx["argmax"]))
    parts.append(emit_float_array(f"expectedForwardDequant_{pre}", fx["y_dequant"]))
    parts.append(emit_float_scalar(f"forwardDequantTol_{pre}", fx["y_dequant_tol"]))
    parts.append(emit_float_array(f"lossGrad_{pre}", fx["gy"]))
    parts.append(emit_int32_array(f"expectedPropLossMantissas_{pre}", fx["dx_mantissas"]))
    parts.append(emit_float_scalar(f"expectedPropLossScale_{pre}", fx["dx_scale"]))
    parts.append(emit_float_array(f"expectedPropLossDequant_{pre}", fx["dx_dequant"]))
    parts.append(emit_float_scalar(f"propLossDequantTol_{pre}", fx["dx_dequant_tol"]))
    parts.append(emit_int32_scalar(f"mantissaTol_{pre}", MANTISSA_TOL_SYM))
    parts.append(emit_float_scalar(f"scaleTol_{pre}", SCALE_REL_TOL_SYM))


# ---- BFP epic PR4 (R-P4): MaxPool1d ARITH_BFP forward --------------------
# Input [1, 2, 8], grid {4 groups x 4}. Group 0 carries a MUCH larger stored
# exponent (133 -> scale 2^6) than group 1 (125 -> scale 2^-2), so window 0
# of channel 0 ({0, 2, 4} under dilation 2) has its true maximum at a SMALL
# code in group 0 while the largest raw mantissa sits in group 1: a kernel
# that compares mantissas (the SYM arm's free ride) picks the wrong element
# AND the wrong argmax. Asserted by the ref's self-check.
BFP_MAX_BATCH = 1
BFP_MAX_CHANNELS = 2
BFP_MAX_INPUT_LENGTH = 8
BFP_MAX_KERNEL_SIZE = 3
BFP_MAX_STRIDE = 1
BFP_MAX_DILATION = 2
BFP_MAX_MANTISSA_BITS = 6
BFP_MAX_EXPONENT_BITS = 8
BFP_MAX_IN_NUM_GROUPS = 4
BFP_MAX_IN_GROUP_SIZE = 4
BFP_MAX_IN_CODES = [12, -7, 3, 31, 30, 5, -1, 20,
                    8, 0, -30, 6, 22, -1, 14, -14]
BFP_MAX_IN_EXPS = [133, 125, 127, 130]


def emit_bfp_fixtures(parts):
    geom = window_geometry_1d(BFP_MAX_INPUT_LENGTH, BFP_MAX_KERNEL_SIZE, BFP_MAX_STRIDE,
                              BFP_MAX_DILATION, "VALID")
    assert geom["out_len"] == 4, geom
    qc = {"mantissa_bits": BFP_MAX_MANTISSA_BITS, "exponent_bits": BFP_MAX_EXPONENT_BITS,
          "group_size": BFP_MAX_IN_GROUP_SIZE}
    values, argmax = maxpool1d_bfp_forward_ref(BFP_MAX_IN_CODES, BFP_MAX_IN_EXPS, qc,
                                               BFP_MAX_BATCH, BFP_MAX_CHANNELS, geom)

    # The C tests build their wires from these, so a fixture-shape change
    # propagates instead of drifting away from hardcoded literals.
    parts.append(emit_int32_scalar("kBfpMaxPoolMantissaBits", BFP_MAX_MANTISSA_BITS))
    parts.append(emit_int32_scalar("kBfpMaxPoolExponentBits", BFP_MAX_EXPONENT_BITS))
    parts.append(emit_int32_scalar("kBfpMaxPoolInNumGroups", BFP_MAX_IN_NUM_GROUPS))
    parts.append(emit_int32_scalar("kBfpMaxPoolInGroupSize", BFP_MAX_IN_GROUP_SIZE))
    parts.append(emit_int32_scalar("kBfpMaxPoolKernelSize", BFP_MAX_KERNEL_SIZE))
    parts.append(emit_int32_scalar("kBfpMaxPoolStride", BFP_MAX_STRIDE))
    parts.append(emit_int32_scalar("kBfpMaxPoolDilation", BFP_MAX_DILATION))

    parts.append(emit_int32_array("kBfpMaxPoolInCodes", torch.tensor(BFP_MAX_IN_CODES)))
    parts.append(emit_uint8_array("kBfpMaxPoolInExps", BFP_MAX_IN_EXPS))
    parts.append(emit_float_array("kBfpMaxPoolExpectedForward",
                                  torch.tensor(values, dtype=torch.float32)))
    parts.append(emit_int32_array("kBfpMaxPoolExpectedArgmax", torch.tensor(argmax)))

    # BACKWARD: gy [1, 2, 4] = 8 elements, grid {2 groups x 4}. The argmax
    # array is the FORWARD gold's, so the two fixtures cannot drift.
    gy_codes = [9, -21, 4, 30, -12, 7, 25, -3]
    gy_exps = [129, 124]
    gy_group_size = 4
    gy_qc = {"mantissa_bits": BFP_MAX_MANTISSA_BITS, "exponent_bits": BFP_MAX_EXPONENT_BITS,
             "group_size": gy_group_size}
    bwd = maxpool1d_bfp_backward_ref(gy_codes, gy_exps, gy_qc, argmax, BFP_MAX_BATCH,
                                     BFP_MAX_CHANNELS, BFP_MAX_INPUT_LENGTH,
                                     geom["out_len"])
    parts.append(emit_int32_scalar("kBfpMaxPoolGyNumGroups", len(gy_exps)))
    parts.append(emit_int32_scalar("kBfpMaxPoolGyGroupSize", gy_group_size))
    parts.append(emit_int32_array("kBfpMaxPoolGyCodes", torch.tensor(gy_codes)))
    parts.append(emit_uint8_array("kBfpMaxPoolGyExps", gy_exps))
    parts.append(emit_float_array("kBfpMaxPoolExpectedBackward",
                                  torch.tensor(bwd, dtype=torch.float32)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    parts = [
        "// AUTOGENERATED by generate_expected_max_pool_1d.py — DO NOT EDIT\n",
        "#ifndef ODT_EXPECTED_MAX_POOL_1D_H\n",
        "#define ODT_EXPECTED_MAX_POOL_1D_H\n",
        "#include <stdint.h>\n",
        "#include <stdlib.h>\n\n",
    ]

    fixtures = [
        fixture_basic(),
        fixture_multi_channel(),
        fixture_multi_batch(),
        fixture_with_stride_and_dilation(),
        fixture_with_same_padding(),
        fixture_edge_cases(),
    ]
    for fx in fixtures:
        emit_fixture(parts, fx)

    for fx in [fixture_sym_basic(), fixture_sym_stride_dilation(),
               fixture_sym_tie_same_padding()]:
        emit_sym_fixture(parts, fx)

    emit_bfp_fixtures(parts)

    parts.append("\n#endif // ODT_EXPECTED_MAX_POOL_1D_H\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(parts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
