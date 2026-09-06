#!/usr/bin/env python3
"""Generate expected_avg_pool_1d.h for UnitTestAvgPool1d (Layer-3 tests).

Produces a C header with PyTorch-derived ground-truth values for each
fixture: forward output, propLoss (dL/dx). For most fixtures
lossGrad = torch.ones_like(y); withStrideAndDilation uses
torch.randn_like(y) (and emits the lossGrad as a gold array) so that
positional mutations on the stride/dilation backward path are
non-vacuous (per codebase_uniform_lossgrad_mutation_vacuity).

Divisor is `count_include_pad=True` (PyTorch default — A1 semantics
per spec §6.4 / §11). The hand-derived self-check uses kernel_size
as divisor for ALL output positions, including SAME edges where
validCount < kernel_size — same convention as the C impl.

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
    avgpool1d_bfp_backward_ref,
    avgpool1d_bfp_forward_ref,
    bfp_quantize_grouped,
    emit_float_scalar,
    emit_int32_array,
    emit_int32_scalar,
    emit_uint8_array,
    f32_scale_i12,
    requant_absmax_i12_f32,
    stable_dequant_i12,
    window_geometry_1d,
    window_slice_1d,
)


def _format_float_literal(v: float) -> str:
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


def _hand_check_avg_forward(x, kernel_size, stride, padding, dilation):
    """count_include_pad=True semantics: divisor = kernel_size always.
    Padded positions are treated as zero and INCLUDED in the divisor count."""
    B, C, L = x.shape
    in_padded_L = L + 2 * padding
    eff_k = (kernel_size - 1) * dilation + 1
    out_L = (in_padded_L - eff_k) // stride + 1
    out = torch.zeros((B, C, out_L))
    for b in range(B):
        for c in range(C):
            for o in range(out_L):
                start = o * stride - padding
                accum = 0.0
                for k in range(kernel_size):
                    idx = start + k * dilation
                    if 0 <= idx < L:
                        accum += x[b, c, idx].item()
                    # else: padded position contributes 0, INCLUDED in divisor
                out[b, c, o] = accum / kernel_size
    return out


def _run_avg_fixture(name, x, *, kernel_size, stride, padding, dilation,
                     loss_grad_kind="ones"):
    x_in = x.clone().detach().requires_grad_(True)

    # PyTorch's F.avg_pool1d defaults to count_include_pad=True.
    # Note: F.avg_pool1d does NOT support `dilation` — for the
    # withStrideAndDilation fixture we manually pad+stride+dilate via a
    # hand-built reference (verified internally below).
    if dilation == 1:
        y = F.avg_pool1d(x_in, kernel_size=kernel_size, stride=stride,
                         padding=padding, count_include_pad=True)
    else:
        # F.avg_pool1d has no dilation arg; build the reference manually,
        # using count_include_pad=True semantics, then verify against numpy.
        y = _hand_check_avg_forward(x_in.detach(), kernel_size, stride,
                                     padding, dilation).clone()
        # Re-attach to autograd by recomputing in a way that keeps the
        # graph: use a manual gather + mean-by-divisor.
        y = _autograd_avg_with_dilation(x_in, kernel_size, stride, padding, dilation)

    expected_y = _hand_check_avg_forward(
        x_in.detach(), kernel_size, stride, padding, dilation,
    )
    assert torch.allclose(y.detach(), expected_y, atol=1e-5), (
        f"{name}: avg_pool1d forward disagrees with hand-derived (count_include_pad=True)"
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
        "dx": x_in.grad.detach(),
        "gy": gy.detach(),
        "loss_grad_kind": loss_grad_kind,
    }


def _autograd_avg_with_dilation(x_in, kernel_size, stride, padding, dilation):
    """Recompute avg pool with dilation via index_select chain so autograd
    can track gradients. count_include_pad=True semantics: divisor = kernel_size.

    Uses torch.zeros for padding (which keeps the autograd graph since
    out-of-range positions get 0 directly without indexing past x_in's bounds).
    """
    B, C, L = x_in.shape
    in_padded_L = L + 2 * padding
    eff_k = (kernel_size - 1) * dilation + 1
    out_L = (in_padded_L - eff_k) // stride + 1

    # Pad x_in with zeros on left/right.
    x_pad = F.pad(x_in, (padding, padding))  # shape: [B, C, L + 2*padding]

    # Build output by gathering windows.
    outputs = []
    for o in range(out_L):
        start = o * stride
        # Gather kernel_size positions, each at start + k*dilation in x_pad.
        col = torch.zeros((B, C), dtype=x_in.dtype)
        for k in range(kernel_size):
            idx = start + k * dilation
            col = col + x_pad[:, :, idx]
        outputs.append(col / kernel_size)
    y = torch.stack(outputs, dim=2)
    return y


def fixture_basic():
    x = torch.tensor([[[1.0, 4.0, 2.0, 3.0]]])
    return _run_avg_fixture("basic", x, kernel_size=2, stride=1, padding=0, dilation=1)


def fixture_multi_channel():
    torch.manual_seed(201)
    x = torch.randn(1, 3, 5)
    return _run_avg_fixture("multiChannel", x, kernel_size=2, stride=1,
                            padding=0, dilation=1)


def fixture_multi_batch():
    torch.manual_seed(202)
    x = torch.randn(4, 2, 4)
    return _run_avg_fixture("multiBatch", x, kernel_size=2, stride=1,
                            padding=0, dilation=1)


def fixture_with_stride_and_dilation():
    # Random lossGrad — Errata 3: stride/dilation backward needs non-uniform gy.
    torch.manual_seed(203)
    x = torch.randn(1, 1, 9)
    return _run_avg_fixture("withStrideAndDilation", x,
                            kernel_size=2, stride=3, padding=0, dilation=2,
                            loss_grad_kind="randn")


def fixture_with_same_padding():
    # K=3, S=1, D=1, padLeft=padRight=1 -> outLen = inLen = 5.
    # Edge windows: outPos=0 (validCount=2, padded left counts in divisor=3),
    # outPos=4 (validCount=2, padded right counts in divisor=3).
    x = torch.tensor([[[2.0, 4.0, 6.0, 8.0, 10.0]]])
    return _run_avg_fixture("withSamePadding", x, kernel_size=3, stride=1,
                            padding=1, dilation=1)


def fixture_edge_cases():
    # K=L=4, full-input window. outLen=1; output is x.mean() per channel.
    # validCount == kernel_size == 4. Tests the divisor == kernel_size path
    # with no truncation.
    x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    return _run_avg_fixture("edgeCases", x, kernel_size=4, stride=1,
                            padding=0, dilation=1)


def emit_fixture(parts, fx):
    pre = f"avgPool1d_{fx['name']}"
    parts.append(emit_float_array(f"input_{pre}", fx["x"]))
    parts.append(emit_float_array(f"expectedForward_{pre}", fx["y"]))
    parts.append(emit_float_array(f"expectedPropLoss_{pre}", fx["dx"]))
    if fx["loss_grad_kind"] != "ones":
        parts.append(emit_float_array(f"lossGrad_{pre}", fx["gy"]))


# ---------------- SYM_INT32 fixtures (#205) ----------------
#
# The SYM kernels are integer-EXACT: the mantissa window sum carries no
# rounding, and the /K division folds EXACTLY into the scale (s_out = s_in/K,
# Dropout idiom) — count_include_pad=True falls out of the fold itself: padded
# positions contribute 0 to the sum, the divisor stays K. The only
# C-vs-emulation gap is the OUT_WRITE restore's float32 expression-order ULP
# (see generate_expected_max_pool_1d.py) -> mantissa +-1, scale rel 1e-4.
MANTISSA_TOL_SYM = 1
SCALE_REL_TOL_SYM = 1e-4


def _f32_div(a: float, b: float) -> float:
    """One IEEE-754 float32 divide (matches the C `inScale / (float)K`)."""
    return (torch.tensor(a, dtype=torch.float32) / torch.tensor(b, dtype=torch.float32)).item()


def _run_avg_sym_fixture(name, x, *, kernel_size, stride, dilation, padding_type,
                         loss_grad_kind="ones"):
    assert_rounding_canary()
    x = x.to(torch.float64)
    xq, _, x_deq = stable_dequant_i12(x)
    s_in = f32_scale_i12(x_deq)
    geom = window_geometry_1d(x.shape[2], kernel_size, stride, dilation, padding_type)
    batch, channels, _ = xq.shape
    out_len = geom["out_len"]

    # Mirror avgPool1dForwardKernelSymInt32: exact int32 window sum of the
    # valid mantissas, raw scale = s_in / K in float32.
    y_raw = torch.zeros((batch, channels, out_len), dtype=torch.int32)
    for b in range(batch):
        for c in range(channels):
            for o in range(out_len):
                first, count = window_slice_1d(geom, o)
                acc = 0
                for i in range(count):
                    acc += int(xq[b, c, first + i * geom["dilation"]])
                y_raw[b, c, o] = acc
    s_fold = _f32_div(s_in, float(kernel_size))
    yq_r, s_y = requant_absmax_i12_f32(y_raw, s_fold)

    y_ref = y_raw.to(torch.float64) * s_fold
    y_dequant_tol = (MANTISSA_TOL_SYM + 0.5) * s_y * 1.5
    err = (yq_r.to(torch.float64) * s_y - y_ref).abs().max().item()
    assert err <= y_dequant_tol, f"{name}: restored fwd dequant {err} > {y_dequant_tol}"

    if loss_grad_kind == "ones":
        gy = torch.ones((batch, channels, out_len), dtype=torch.float64)
    elif loss_grad_kind == "randn":
        torch.manual_seed(hash(name) & 0xFFFF)
        gy = torch.randn(batch, channels, out_len, dtype=torch.float64)
    else:
        raise ValueError(loss_grad_kind)
    gyq, _, gy_deq = stable_dequant_i12(gy)
    s_gy = f32_scale_i12(gy_deq)

    # Mirror avgPool1dBackwardKernelSymInt32: zero + scatter gy mantissas into
    # every valid window member, raw scale = s_gy / K in float32.
    dx_raw = torch.zeros_like(xq)
    for b in range(batch):
        for c in range(channels):
            for o in range(out_len):
                first, count = window_slice_1d(geom, o)
                for i in range(count):
                    dx_raw[b, c, first + i * geom["dilation"]] += gyq[b, c, o]
    s_dx_fold = _f32_div(s_gy, float(kernel_size))
    dxq_r, s_dx = requant_absmax_i12_f32(dx_raw, s_dx_fold)

    dx_ref = dx_raw.to(torch.float64) * s_dx_fold
    dx_dequant_tol = (MANTISSA_TOL_SYM + 0.5) * s_dx * 1.5
    err = (dxq_r.to(torch.float64) * s_dx - dx_ref).abs().max().item()
    assert err <= dx_dequant_tol, f"{name}: restored bwd dequant {err} > {dx_dequant_tol}"

    return {
        "name": name,
        "x": x_deq,
        "y_mantissas": yq_r, "y_scale": s_y,
        "y_dequant": y_ref.to(torch.float32), "y_dequant_tol": y_dequant_tol,
        "gy": gy_deq,
        "dx_mantissas": dxq_r, "dx_scale": s_dx,
        "dx_dequant": dx_ref.to(torch.float32), "dx_dequant_tol": dx_dequant_tol,
    }


def fixture_sym_basic():
    # Same values as the FLOAT basic fixture: K=2, S=1, D=1, VALID.
    x = torch.tensor([[[1.0, 4.0, 2.0, 3.0]]])
    return _run_avg_sym_fixture("symBasic", x, kernel_size=2, stride=1, dilation=1,
                                padding_type="VALID")


def fixture_sym_stride_dilation():
    # K=2, S=3, D=2, VALID on L=9. Random lossGrad so positional mutations on
    # the SYM scatter path are non-vacuous.
    torch.manual_seed(206)
    x = torch.randn(1, 1, 9)
    return _run_avg_sym_fixture("symStrideDilation", x, kernel_size=2, stride=3,
                                dilation=2, padding_type="VALID",
                                loss_grad_kind="randn")


def fixture_sym_same_padding():
    # K=3, S=1, SAME on L=5: edge windows have validCount=2 but the scale fold
    # keeps the divisor at K=3 — pins count_include_pad=True on the SYM path
    # (a validCount-based divisor mutation shifts edge outputs by 3/2).
    x = torch.tensor([[[2.0, 4.0, 6.0, 8.0, 10.0]]])
    return _run_avg_sym_fixture("symSamePadding", x, kernel_size=3, stride=1,
                                dilation=1, padding_type="SAME")


def emit_sym_fixture(parts, fx):
    pre = f"avgPool1dSym_{fx['name']}"
    parts.append(emit_float_array(f"input_{pre}", fx["x"]))
    parts.append(emit_int32_array(f"expectedForwardMantissas_{pre}", fx["y_mantissas"]))
    parts.append(emit_float_scalar(f"expectedForwardScale_{pre}", fx["y_scale"]))
    parts.append(emit_float_array(f"expectedForwardDequant_{pre}", fx["y_dequant"]))
    parts.append(emit_float_scalar(f"forwardDequantTol_{pre}", fx["y_dequant_tol"]))
    parts.append(emit_float_array(f"lossGrad_{pre}", fx["gy"]))
    parts.append(emit_int32_array(f"expectedPropLossMantissas_{pre}", fx["dx_mantissas"]))
    parts.append(emit_float_scalar(f"expectedPropLossScale_{pre}", fx["dx_scale"]))
    parts.append(emit_float_array(f"expectedPropLossDequant_{pre}", fx["dx_dequant"]))
    parts.append(emit_float_scalar(f"propLossDequantTol_{pre}", fx["dx_dequant_tol"]))
    parts.append(emit_int32_scalar(f"mantissaTol_{pre}", MANTISSA_TOL_SYM))
    parts.append(emit_float_scalar(f"scaleTol_{pre}", SCALE_REL_TOL_SYM))


# ---- BFP epic PR4 (R-P4): AvgPool1d ARITH_BFP forward ------------------
# Geometry: input [1, 2, 8], kernel 3, VALID, stride 1, DILATION 2 ->
# effective kernel 5, out_len 4. Dilation is load-bearing: every window
# visits non-consecutive storage indices, so a run precompute instead of the
# per-element bfpGroupOf lookup binds the wrong exponent. The input grid is
# {4 groups x 4} over the 16 storage elements, so channel 0 spans groups
# 0/1 and channel 1 spans groups 2/3 -- every window crosses a boundary.
# K = 3 is NOT a power of two, which is exactly why the /K divide lives in
# the FLOAT32 raw (the SYM s/K scale fold has no BFP analog).
BFP_AVG_BATCH = 1
BFP_AVG_CHANNELS = 2
BFP_AVG_INPUT_LENGTH = 8
BFP_AVG_KERNEL_SIZE = 3
BFP_AVG_STRIDE = 1
BFP_AVG_DILATION = 2
BFP_AVG_MANTISSA_BITS = 6
BFP_AVG_EXPONENT_BITS = 8
BFP_AVG_IN_NUM_GROUPS = 4
BFP_AVG_IN_GROUP_SIZE = 4
BFP_AVG_IN_CODES = [12, -7, 0, 31, -32, 5, -1, 20,
                    3, -14, 22, -1, 8, 0, -30, 6]
BFP_AVG_IN_EXPS = [130, 127, 133, 125]
# Output wire for the packed-write fixture: 8 elements -> {2 groups x 4}.
BFP_AVG_OUT_NUM_GROUPS = 2
BFP_AVG_OUT_GROUP_SIZE = 4
# Staged-FLOAT32 fixture: values deliberately LOSSY at m=6 so a kernel that
# stages at a hardcoded m=8 produces observably different outputs.
BFP_AVG_STAGE_INPUT = [1.3, -2.6, 3.3, -1.55, 10.0, -6.1, 4.2, 7.9,
                       0.55, -0.35, 0.85, 1.15, 4.1, 2.3, -1.05, 0.65]


def emit_bfp_fixtures(parts):
    geom = window_geometry_1d(BFP_AVG_INPUT_LENGTH, BFP_AVG_KERNEL_SIZE, BFP_AVG_STRIDE,
                              BFP_AVG_DILATION, "VALID")
    assert geom["out_len"] == 4, geom
    in_qc = {"mantissa_bits": BFP_AVG_MANTISSA_BITS,
             "exponent_bits": BFP_AVG_EXPONENT_BITS,
             "group_size": BFP_AVG_IN_GROUP_SIZE}

    # The C tests build their wires from these, so a fixture-shape change
    # propagates instead of drifting away from hardcoded literals.
    parts.append(emit_int32_scalar("kBfpAvgPoolMantissaBits", BFP_AVG_MANTISSA_BITS))
    parts.append(emit_int32_scalar("kBfpAvgPoolExponentBits", BFP_AVG_EXPONENT_BITS))
    parts.append(emit_int32_scalar("kBfpAvgPoolInNumGroups", BFP_AVG_IN_NUM_GROUPS))
    parts.append(emit_int32_scalar("kBfpAvgPoolInGroupSize", BFP_AVG_IN_GROUP_SIZE))
    parts.append(emit_int32_scalar("kBfpAvgPoolOutNumGroups", BFP_AVG_OUT_NUM_GROUPS))
    parts.append(emit_int32_scalar("kBfpAvgPoolOutGroupSize", BFP_AVG_OUT_GROUP_SIZE))
    parts.append(emit_int32_scalar("kBfpAvgPoolKernelSize", BFP_AVG_KERNEL_SIZE))
    parts.append(emit_int32_scalar("kBfpAvgPoolStride", BFP_AVG_STRIDE))
    parts.append(emit_int32_scalar("kBfpAvgPoolDilation", BFP_AVG_DILATION))

    # (1) BFP-STORED grouped input, borrowed zero-copy (D8): full self-checks.
    fwd = avgpool1d_bfp_forward_ref(BFP_AVG_IN_CODES, BFP_AVG_IN_EXPS, in_qc,
                                    BFP_AVG_BATCH, BFP_AVG_CHANNELS, geom)
    parts.append(emit_int32_array("kBfpAvgPoolInCodes", torch.tensor(BFP_AVG_IN_CODES)))
    parts.append(emit_uint8_array("kBfpAvgPoolInExps", BFP_AVG_IN_EXPS))
    parts.append(emit_float_array("kBfpAvgPoolExpectedForward",
                                  torch.tensor(fwd, dtype=torch.float32)))

    # (2) FLOAT32-stored input staged PER-TENSOR at the produced wire's widths
    # (R-P1: pools have no weight operand, so outputQ is the anchor). The
    # stage template is always {1, 0} -- the anchor's GROUPING is irrelevant,
    # only its mantissa/exponent widths are read.
    st_codes, st_exps = bfp_quantize_grouped(BFP_AVG_STAGE_INPUT, BFP_AVG_MANTISSA_BITS,
                                             BFP_AVG_EXPONENT_BITS, 0)
    st_qc = {"mantissa_bits": BFP_AVG_MANTISSA_BITS,
             "exponent_bits": BFP_AVG_EXPONENT_BITS, "group_size": 0}
    staged = avgpool1d_bfp_forward_ref(st_codes, st_exps, st_qc, BFP_AVG_BATCH,
                                       BFP_AVG_CHANNELS, geom, self_check=False)
    wide_codes, wide_exps = bfp_quantize_grouped(BFP_AVG_STAGE_INPUT, 8,
                                                 BFP_AVG_EXPONENT_BITS, 0)
    wide = avgpool1d_bfp_forward_ref(wide_codes, wide_exps,
                                     {**st_qc, "mantissa_bits": 8}, BFP_AVG_BATCH,
                                     BFP_AVG_CHANNELS, geom, self_check=False)
    assert wide != staged, (
        "staging at m=8 gives the same output as m=6 -- the rule-1 width-anchor "
        "mutation would be unobservable; pick lossier input values")
    assert any(v != 0.0 for v in staged), "staged AvgPool BFP fixture is all-zero"
    parts.append(emit_float_array("kBfpAvgPoolStageInput",
                                  torch.tensor(BFP_AVG_STAGE_INPUT, dtype=torch.float32)))
    parts.append(emit_float_array("kBfpAvgPoolExpectedStaged",
                                  torch.tensor(staged, dtype=torch.float32)))

    # (3) OUT_WRITE onto a BFP output wire: the epilogue packs the FLOAT32 raw
    # through packFloatBufferAsBfp (fresh per-group exponents, op rounding =
    # HALF_AWAY), which bfp_quantize_grouped mirrors bit-for-bit.
    pk_codes, pk_exps = bfp_quantize_grouped(torch.tensor(fwd, dtype=torch.float32),
                                             BFP_AVG_MANTISSA_BITS, BFP_AVG_EXPONENT_BITS,
                                             BFP_AVG_OUT_GROUP_SIZE)
    assert len(set(pk_exps)) > 1, (
        "packed AvgPool BFP output has one exponent for both groups -- a per-tensor "
        "epilogue would be indistinguishable")
    parts.append(emit_int32_array("kBfpAvgPoolPackedCodes", torch.tensor(pk_codes)))
    parts.append(emit_uint8_array("kBfpAvgPoolPackedExps", pk_exps))

    # (4) BACKWARD: gy [1, 2, 4] = 8 elements, grid {2 groups x 4}. Same
    # dilated geometry, so windows overlap (index 2 is covered by outputs 0
    # and 2) and the memset + accumulating scatter is load-bearing.
    gy_codes = [9, -21, 4, 30, -12, 7, 25, -3]
    gy_exps = [129, 124]
    gy_qc = {"mantissa_bits": BFP_AVG_MANTISSA_BITS,
             "exponent_bits": BFP_AVG_EXPONENT_BITS,
             "group_size": BFP_AVG_OUT_GROUP_SIZE}
    bwd = avgpool1d_bfp_backward_ref(gy_codes, gy_exps, gy_qc, BFP_AVG_BATCH,
                                     BFP_AVG_CHANNELS, geom)
    parts.append(emit_int32_array("kBfpAvgPoolGyCodes", torch.tensor(gy_codes)))
    parts.append(emit_uint8_array("kBfpAvgPoolGyExps", gy_exps))
    parts.append(emit_float_array("kBfpAvgPoolExpectedBackward",
                                  torch.tensor(bwd, dtype=torch.float32)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    parts = [
        "// AUTOGENERATED by generate_expected_avg_pool_1d.py — DO NOT EDIT\n",
        "#ifndef ODT_EXPECTED_AVG_POOL_1D_H\n",
        "#define ODT_EXPECTED_AVG_POOL_1D_H\n",
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
               fixture_sym_same_padding()]:
        emit_sym_fixture(parts, fx)

    emit_bfp_fixtures(parts)

    parts.append("\n#endif // ODT_EXPECTED_AVG_POOL_1D_H\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(parts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
