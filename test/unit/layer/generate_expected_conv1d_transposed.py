#!/usr/bin/env python3
"""Generate expected_conv1d_transposed.h for UnitTestConv1dTransposed.

Produces a C header with PyTorch-derived forward outputs and
autograd-derived dL/dx, dL/dW, dL/db for each fixture. lossGrad is
torch.ones_like(y) for every FLOAT fixture.

ConvTranspose1d weight shape: [Cin, Cout/groups, K] — note the order
vs. Conv1d's [Cout, Cin/groups, K].
"""
import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "goldgen"))
import sym_gold
from sym_gold import (
    round_half_away,
    emit_int32_array,
    emit_float_scalar,
    emit_int32_scalar,
)

QMAX_F32 = torch.tensor(32767.0, dtype=torch.float32)
QMIN_F32 = torch.tensor(-32768.0, dtype=torch.float32)

QMAX_I12 = torch.tensor(2047.0, dtype=torch.float32)
QMIN_I12 = torch.tensor(-2048.0, dtype=torch.float32)


def quantize_sym_i12(x):
    """int12 operand quantize (qMaxBits=12): absmax->scale=absmax/2047, round-clamp half-away."""
    absmax = x.abs().max().item()
    scale = 1.0 if absmax == 0.0 else absmax / 2047.0
    q = round_half_away(torch.clamp(x / scale, -2048.0, 2047.0))
    return q.to(torch.int32), scale


def stable_dequant_i12(x):
    q, s = quantize_sym_i12(x.to(torch.float64))
    deq32 = (q.to(torch.float64) * s).to(torch.float32)
    q2, _ = quantize_sym_i12(deq32.to(torch.float64))
    assert torch.equal(q, q2), "int12 fixture not dequant round-trip stable"
    return q, s, deq32


def f32_scale_i12(deq32: torch.Tensor) -> float:
    """int12 scale the C runtime derives from the emitted fixture: absmax_f32/2047 in float32."""
    absmax = deq32.abs().max().to(torch.float32)
    if absmax.item() == 0.0:
        return 1.0
    return (absmax / QMAX_I12).item()


def f32_mul(a: float, b: float) -> float:
    """One IEEE-754 float32 multiply (matches the C `aScale * bScale`)."""
    return (torch.tensor(a, dtype=torch.float32) * torch.tensor(b, dtype=torch.float32)).item()


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


def _run_fixture(name, x, w, b, *, stride, padding, output_padding,
                 dilation, groups):
    x = x.clone().detach().requires_grad_(True)
    w = w.clone().detach().requires_grad_(True)
    if b is not None:
        b = b.clone().detach().requires_grad_(True)
    y = F.conv_transpose1d(x, w, bias=b, stride=stride, padding=padding,
                           output_padding=output_padding,
                           dilation=dilation, groups=groups)
    loss = y.sum()
    loss.backward()
    return {
        "name": name,
        "x": x.detach(),
        "w": w.detach(),
        "b": b.detach() if b is not None else None,
        "y": y.detach(),
        "dx": x.grad.detach(),
        "dw": w.grad.detach(),
        "db": b.grad.detach() if b is not None else None,
        "_params": {"stride": stride, "padding": padding, "output_padding": output_padding,
                    "dilation": dilation, "groups": groups, "has_bias": b is not None},
    }


def fixture_single_channel_single_batch():
    # B=1, Cin=1, Cout=1, L=3, K=2, stride=1, no bias
    x = torch.tensor([[[1.0, 2.0, 3.0]]])
    w = torch.tensor([[[2.0, 4.0]]])
    return _run_fixture("singleChannelSingleBatch", x, w, None,
                        stride=1, padding=0, output_padding=0,
                        dilation=1, groups=1)


def fixture_single_channel_with_bias():
    x = torch.tensor([[[1.0, 2.0, 3.0]]])
    w = torch.tensor([[[2.0, 4.0]]])
    b = torch.tensor([0.5])
    return _run_fixture("singleChannelWithBias", x, w, b,
                        stride=1, padding=0, output_padding=0,
                        dilation=1, groups=1)


def fixture_multi_channel_with_bias():
    torch.manual_seed(10)
    x = torch.arange(12.0).reshape(1, 3, 4)
    w = torch.arange(12.0).reshape(3, 2, 2) * 0.1
    b = torch.tensor([0.5, -0.5])
    return _run_fixture("multiChannelWithBias", x, w, b,
                        stride=1, padding=0, output_padding=0,
                        dilation=1, groups=1)


def fixture_multi_batch():
    torch.manual_seed(11)
    x = torch.randn(3, 2, 4)
    w = torch.randn(2, 2, 2)
    return _run_fixture("multiBatch", x, w, None,
                        stride=1, padding=0, output_padding=0,
                        dilation=1, groups=1)


def fixture_groups_depthwise():
    # Cin=4, Cout=4, groups=4 (depthwise transpose). Weight: [Cin, Cout/g=1, K]
    torch.manual_seed(12)
    x = torch.randn(1, 4, 4)
    w = torch.randn(4, 1, 2)
    return _run_fixture("groupsDepthwise", x, w, None,
                        stride=1, padding=0, output_padding=0,
                        dilation=1, groups=4)


def fixture_groups_grouped():
    # Cin=4, Cout=8, groups=2. Weight: [Cin, Cout/g=4, K]
    torch.manual_seed(13)
    x = torch.randn(1, 4, 4)
    w = torch.randn(4, 4, 2)
    b = torch.randn(8)
    return _run_fixture("groupsGrouped", x, w, b,
                        stride=1, padding=0, output_padding=0,
                        dilation=1, groups=2)


def fixture_stride2():
    # The classic upsampling ConvT use case: stride=2 doubles length-1.
    x = torch.tensor([[[1.0, 2.0, 3.0]]])
    w = torch.tensor([[[2.0, 4.0]]])
    return _run_fixture("stride2", x, w, None,
                        stride=2, padding=0, output_padding=0,
                        dilation=1, groups=1)


def fixture_stride2_with_output_padding():
    # outputPadding=1 trails an extra zero (plus bias if any)
    x = torch.tensor([[[1.0, 2.0, 3.0]]])
    w = torch.tensor([[[2.0, 4.0]]])
    b = torch.tensor([1.0])
    return _run_fixture("stride2WithOutputPadding", x, w, b,
                        stride=2, padding=0, output_padding=1,
                        dilation=1, groups=1)


def fixture_dilation2():
    # dilation>1: kernel taps stretched; output_length = (Lin-1)*stride + dilation*(K-1) + 1
    x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    w = torch.tensor([[[2.0, 4.0]]])
    return _run_fixture("dilation2", x, w, None,
                        stride=1, padding=0, output_padding=0,
                        dilation=2, groups=1)


def _conv_params(fx):
    return fx["_params"]  # {stride, padding, output_padding, dilation, groups, has_bias}


def _quant_inputs(fx):
    """int12-quantize x/w/(b)/lossGrad operands; return mantissas, float32 scales, float fixtures."""
    xq, _, x_deq = stable_dequant_i12(fx["x"])
    wq, _, w_deq = stable_dequant_i12(fx["w"])
    s_in = f32_scale_i12(x_deq)
    s_w = f32_scale_i12(w_deq)
    bq = b_deq = s_b = None
    if fx["b"] is not None:
        bq, _, b_deq = stable_dequant_i12(fx["b"])
        s_b = f32_scale_i12(b_deq)
    gy = fx["lossGrad"] if fx.get("lossGrad") is not None else torch.ones_like(fx["y"])
    gyq, _, gy_deq = stable_dequant_i12(gy)
    s_loss = f32_scale_i12(gy_deq)
    return dict(xq=xq, wq=wq, bq=bq, gyq=gyq, s_in=s_in, s_w=s_w, s_b=s_b, s_loss=s_loss,
                x_deq=x_deq, w_deq=w_deq, b_deq=b_deq, gy_deq=gy_deq)


def _int_mac(q, p):
    """Exact integer ConvT forward + autograd grads via float64 conv_transpose1d of int mantissas.
    Returns (fwd_mac[B,Cout,Lout] int64, dw_mac int64 [weightGrad], dx_mac int64 [dx gather])."""
    xqf = q["xq"].to(torch.float64).requires_grad_(True)
    wqf = q["wq"].to(torch.float64).requires_grad_(True)
    y = F.conv_transpose1d(xqf, wqf, bias=None, stride=p["stride"], padding=p["padding"],
                           output_padding=p["output_padding"], dilation=p["dilation"],
                           groups=p["groups"])
    y.backward(q["gyq"].to(torch.float64))
    fwd_mac = y.detach().round().to(torch.int64)
    dw_mac = wqf.grad.round().to(torch.int64)
    dx_mac = xqf.grad.round().to(torch.int64)
    return fwd_mac, dw_mac, dx_mac


def _round_away_f32(x: torch.Tensor) -> torch.Tensor:
    return round_half_away(x.to(torch.float32))


def _requant_absmax_f32(mac_int: torch.Tensor, in_scale: float):
    """addSymInt32TensorsInplace into a zero grad: dequant (f32) -> absmax (f32) ->
    scale=absmax/32767 (f32) -> round_half_away(clamp(.../scale)). Returns (q int32, scale)."""
    deq = mac_int.to(torch.float32) * torch.tensor(in_scale, dtype=torch.float32)
    absmax = deq.abs().max().to(torch.float32)
    if absmax.item() == 0.0:
        return torch.zeros_like(mac_int, dtype=torch.int32), 1.0
    scale = (absmax / QMAX_F32)
    q = round_half_away(torch.clamp(deq / scale, QMIN_F32, QMAX_F32))
    return q.to(torch.int32), scale.item()


def _requant_absmax_i12_f32(mac_int: torch.Tensor, in_scale: float):
    """PR1b.2 D3: propLoss now routes through executeOp's OUT_WRITE
    epilogue; for a SYM_INT32 target this hits the conversionMatrix diagonal
    (requantSymInt32Tensor, TensorConversion.c) instead of a raw direct write.
    Mirrors _requant_absmax_f32 above but at propLossQ's declared qMaxBits=12
    (int12) instead of the grad contract's 16 — same dequant(f32) -> absmax(f32)
    -> scale=absmax/2047(f32) -> round_half_away(clamp(...)) sequence."""
    deq = mac_int.to(torch.float32) * torch.tensor(in_scale, dtype=torch.float32)
    absmax = deq.abs().max().to(torch.float32)
    if absmax.item() == 0.0:
        return torch.zeros_like(mac_int, dtype=torch.int32), 1.0
    scale = (absmax / QMAX_I12)
    q = round_half_away(torch.clamp(deq / scale, QMIN_I12, QMAX_I12))
    return q.to(torch.int32), scale.item()


def _restore_tolerances(raw_q, raw_scale, mtol_before, base_q, base_scale):
    """Empirically DERIVE (not hand-guess) the restored (mantissa,
    scale-relative) tolerances for a raw producer wire PR1b.2 now restores via
    _requant_absmax_i12_f32 — mirrors generate_expected_conv1d.py's
    identically-named helper (same forward-restoration hazard applies to
    ConvT1d) and the LayerNorm forward precedent
    (generate_expected_layernorm_sym.py::_restore_tolerances, PR1b.2 Task 3).
    Perturb the RAW pre-restoration mantissas by their own worst-case
    C-vs-emulation noise band (+-mtol_before — the bias-seed rounding-
    boundary ambiguity the forward fixture already carries pre-restoration),
    applied elementwise AND concentrated at the argmax element (which drives
    the requant's absmax renormalization), and measure how far the restored
    mantissas/scale actually move under the SAME transform. A fixed 1.5x
    safety margin is applied to the observed worst case."""
    # Floor at 1e-4 relative: the pre-existing (pre-PR1b.2) convention for
    # every SYM scale assertion in this file, kept as a defensive margin
    # against ordinary float32 noise even when no bias-seed ambiguity exists
    # (mtol_before == 0) to perturb against.
    floor = 1e-4
    if mtol_before == 0:
        return 0, floor
    idx = int(raw_q.abs().reshape(-1).argmax().item())
    variants = [raw_q + mtol_before, raw_q - mtol_before]
    for delta in (mtol_before, -mtol_before):
        v = raw_q.clone()
        v.reshape(-1)[idx] += delta
        variants.append(v)
    max_mdiff = 0
    max_sdiff = 0.0
    for variant in variants:
        vq, vs = _requant_absmax_i12_f32(variant, raw_scale)
        mdiff = int((vq.to(torch.int64) - base_q.to(torch.int64)).abs().max().item())
        sdiff = abs(vs - base_scale) / base_scale
        max_mdiff = max(max_mdiff, mdiff)
        max_sdiff = max(max_sdiff, sdiff)
    margin = 1.5
    mantissa_tol_r = int(math.ceil(max_mdiff * margin)) + 1
    scale_rel_tol_r = max(max_sdiff * margin, floor)
    return mantissa_tol_r, scale_rel_tol_r


def _autograd_ref_f64(q, p):
    """float64 torch-autograd on the EXACT dequantized inputs (q*s in f64) — the true grads."""
    x64 = (q["xq"].to(torch.float64) * q["s_in"]).requires_grad_(True)
    w64 = (q["wq"].to(torch.float64) * q["s_w"]).requires_grad_(True)
    b64 = None
    if q["bq"] is not None:
        b64 = (q["bq"].to(torch.float64) * q["s_b"]).requires_grad_(True)
    y = F.conv_transpose1d(x64, w64, bias=b64, stride=p["stride"], padding=p["padding"],
                           output_padding=p["output_padding"], dilation=p["dilation"],
                           groups=p["groups"])
    y.backward(q["gyq"].to(torch.float64) * q["s_loss"])
    return (y.detach(), x64.grad, w64.grad, b64.grad if b64 is not None else None)


def emulate_sym_convT(fx):
    """Full SYM emulation of conv1dTransposedForward + conv1dTransposedBackward for one fixture."""
    # Rounding-mode canary: half-AWAY must round 0.5 -> 1 and -0.5 -> -1 (NOT half-to-even's 0).
    _canary = _round_away_f32(torch.tensor([0.5, -0.5], dtype=torch.float32))
    assert _canary.tolist() == [1.0, -1.0], \
        "round_half_away is not half-away-from-zero — rounding mode wrong"
    p = _conv_params(fx)
    q = _quant_inputs(fx)
    fwd_mac, dw_mac, dx_mac = _int_mac(q, p)
    out_scale = f32_mul(q["s_in"], q["s_w"])          # RAW forward accumulator scale (s_in*s_w)
    dx_scale_raw = f32_mul(q["s_loss"], q["s_w"])     # propLoss RAW gather-adjoint scale (pre-PR1b.2)
    inter_scale = f32_mul(q["s_loss"], q["s_in"])     # weightGrad intermediate scale

    # forward: int MAC + per-channel bias seed (float32 refold), at the RAW
    # s_in*s_w accumulator scale — matches convTranspose1dKernelSymInt32's own
    # direct output today (pre-PR1b.2: this raw wire is what a downstream
    # Quantization layer used to restore).
    out_channels = fwd_mac.shape[1]
    if q["bq"] is not None:
        seed = _round_away_f32(q["bq"].to(torch.float32)
                               * torch.tensor(q["s_b"], dtype=torch.float32)
                               / torch.tensor(out_scale, dtype=torch.float32)).to(torch.int64)
        fwd_raw_q = (fwd_mac + seed.reshape(1, out_channels, 1)).to(torch.int32)
        fwd_mtol_raw = 1
    else:
        fwd_raw_q = fwd_mac.to(torch.int32)
        fwd_mtol_raw = 0
    # PR1b.2 D3: conv1dTransposedForward now routes through
    # executeOp's OUT_WRITE epilogue; a SYM_INT32 target hits the
    # conversionMatrix diagonal (requantSymInt32Tensor) instead of the raw
    # direct kernel write above (pre-PR1b.2 behavior) — the same restoration
    # propLoss already got in Task 2, at the same declared qMaxBits=12
    # (_requant_absmax_i12_f32).
    fwd_q, fwd_scale = _requant_absmax_i12_f32(fwd_raw_q, out_scale)
    fwd_deq = fwd_q.to(torch.float32) * torch.tensor(fwd_scale, dtype=torch.float32)
    # The restored scale is now itself derived from an absmax over the
    # (float32, C-vs-Python-noisy) mantissas — sensitive to the same
    # bias-seed rounding-boundary noise the mantissas are, unlike the old raw
    # out_scale (a direct product of two independently-computed scales,
    # insensitive to that noise). Empirically derive both tolerances instead
    # of trusting the pre-restoration fwd_mtol_raw/hardcoded 1e-4 relative.
    fwd_mtol, fwd_scale_tol = _restore_tolerances(fwd_raw_q, out_scale, fwd_mtol_raw, fwd_q,
                                                  fwd_scale)

    # dx (propLoss): PR1b.2 D3 routes propLoss through executeOp's
    # OUT_WRITE epilogue; for a SYM_INT32 target this requants through the
    # conversionMatrix diagonal (requantSymInt32Tensor) instead of writing the
    # raw, unrestored accumulator-range gather-adjoint directly (pre-PR1b.2
    # behavior). Mirrors weightGrad's Strategy-A requant (_requant_absmax_f32)
    # but at propLossQ's declared qMaxBits=12 (int12), not the grad qMaxBits=16.
    dx_q, dx_scale = _requant_absmax_i12_f32(dx_mac, dx_scale_raw)
    dx_deq = dx_q.to(torch.float32) * torch.tensor(dx_scale, dtype=torch.float32)

    # weightGrad: strategy A requant of the int scatter-gather from a zero grad
    dw_q, dw_scale = _requant_absmax_f32(dw_mac, inter_scale)

    # biasGrad: int sum over (batch, outLen) per oc, fixed-scale rescale (s_bg = 1.0, zero-init grad).
    db_q = db_scale = None
    if q["bq"] is not None:
        sum_int = q["gyq"].sum(dim=(0, 2))            # exact int per output channel
        db_q = _round_away_f32(sum_int.to(torch.float32)
                               * torch.tensor(q["s_loss"], dtype=torch.float32)
                               / torch.tensor(1.0, dtype=torch.float32)).to(torch.int32)
        db_scale = 1.0

    ref_y, ref_dx, ref_dw, ref_db = _autograd_ref_f64(q, p)

    # ---- self-checks (fail at generation time, not in C) ----
    _eps32 = float(torch.finfo(torch.float32).eps)
    fwd_f32_prec = fwd_q.abs().max().item() * fwd_scale * _eps32
    dx_f32_prec = dx_q.abs().max().item() * dx_scale * _eps32

    fwd_err = (fwd_deq.to(torch.float64) - ref_y).abs().max().item()
    fwd_tol = 8.0 * fwd_scale + fwd_f32_prec + 1e-9
    assert fwd_err <= fwd_tol, f"{fx['name']}: fwd emulation err {fwd_err} > tol {fwd_tol}"
    dx_err = (dx_deq.to(torch.float64) - ref_dx).abs().max().item()
    dx_tol = 8.0 * dx_scale + dx_f32_prec + 1e-9
    assert dx_err <= dx_tol, f"{fx['name']}: dx emulation err {dx_err} > tol {dx_tol}"
    dw_err = (dw_q.to(torch.float64) * dw_scale - ref_dw.reshape(-1).reshape(dw_q.shape)
              ).abs().max().item()
    assert dw_err <= (0.5 * inter_scale + 2.5 * dw_scale) * 1.5 + 1e-9, \
        f"{fx['name']}: dw emulation err {dw_err}"
    if db_q is not None:
        db_err = (db_q.to(torch.float64) * db_scale - ref_db.reshape(-1)).abs().max().item()
        assert db_err <= (0.5 * q["s_loss"] + 2.5 * 1.0) * 1.5 + 1e-9, \
            f"{fx['name']}: db emulation err {db_err}"

    return dict(
        name=fx["name"], has_bias=q["bq"] is not None,
        x_deq=q["x_deq"], w_deq=q["w_deq"], b_deq=q["b_deq"], gy_deq=q["gy_deq"],
        fwd_q=fwd_q, fwd_scale=fwd_scale, fwd_deq=ref_y.to(torch.float32),
        fwd_mtol=fwd_mtol, fwd_dtol=8.0 * fwd_scale + fwd_f32_prec + 1e-6,
        fwd_scale_tol=fwd_scale_tol,
        dx_q=dx_q, dx_scale=dx_scale, dx_deq=ref_dx.to(torch.float32),
        dx_mtol=0, dx_dtol=8.0 * dx_scale + dx_f32_prec + 1e-6,
        dw_q=dw_q, dw_scale=dw_scale, dw_deq=ref_dw.to(torch.float32),
        dw_mtol=2, dw_dtol=(0.5 * inter_scale + 2.5 * dw_scale) * 1.5 + 1e-6,
        db_q=db_q, db_scale=db_scale,
        db_deq=ref_db.to(torch.float32) if db_q is not None else None,
        db_mtol=1, db_dtol=(0.5 * q["s_loss"] + 2.5 * 1.0) * 1.5 + 1e-6,
    )


def _run_fixture_sym(name, x, w, b, lossGrad, *, stride, padding, output_padding, dilation, groups):
    x = x.clone().detach().requires_grad_(True)
    w = w.clone().detach().requires_grad_(True)
    if b is not None:
        b = b.clone().detach().requires_grad_(True)
    y = F.conv_transpose1d(x, w, bias=b, stride=stride, padding=padding,
                           output_padding=output_padding, dilation=dilation, groups=groups)
    y.backward(lossGrad)
    return {
        "name": name, "x": x.detach(), "w": w.detach(),
        "b": b.detach() if b is not None else None, "y": y.detach(),
        "dx": x.grad.detach(), "dw": w.grad.detach(),
        "db": b.grad.detach() if b is not None else None, "lossGrad": lossGrad.detach(),
        "_params": {"stride": stride, "padding": padding, "output_padding": output_padding,
                    "dilation": dilation, "groups": groups, "has_bias": b is not None},
    }


def fixture_multi_channel_with_bias_sym():
    torch.manual_seed(20)
    x = torch.randn(1, 3, 4)
    w = torch.randn(3, 2, 2)          # [Cin=3, Cout/groups=2, K=2]
    b = torch.randn(2)
    lossGrad = torch.randn(1, 2, 5)   # Lout=(4-1)*1+1*(2-1)+0+1=5
    return _run_fixture_sym("multiChannelWithBiasSym", x, w, b, lossGrad,
                            stride=1, padding=0, output_padding=0, dilation=1, groups=1)


def fixture_groups_grouped_sym():
    torch.manual_seed(21)
    x = torch.randn(1, 4, 4)
    w = torch.randn(4, 4, 2)          # [Cin=4, Cout/groups=4, K=2], groups=2 -> Cout=8
    b = torch.randn(8)
    lossGrad = torch.randn(1, 8, 5)
    return _run_fixture_sym("groupsGroupedSym", x, w, b, lossGrad,
                            stride=1, padding=0, output_padding=0, dilation=1, groups=2)


def fixture_stride2_sym():
    torch.manual_seed(22)
    x = torch.randn(1, 1, 3)
    w = torch.randn(1, 1, 2)
    lossGrad = torch.randn(1, 1, 6)   # Lout=(3-1)*2+1*(2-1)+0+1=6
    return _run_fixture_sym("stride2Sym", x, w, None, lossGrad,
                            stride=2, padding=0, output_padding=0, dilation=1, groups=1)


def fixture_dilation2_sym():
    torch.manual_seed(23)
    x = torch.randn(1, 1, 4)
    w = torch.randn(1, 1, 2)
    lossGrad = torch.randn(1, 1, 6)   # Lout=(4-1)*1+2*(2-1)+0+1=6
    return _run_fixture_sym("dilation2Sym", x, w, None, lossGrad,
                            stride=1, padding=0, output_padding=0, dilation=2, groups=1)


def fixture_stride2_output_padding_sym():
    torch.manual_seed(24)
    x = torch.randn(1, 1, 3)
    w = torch.randn(1, 1, 2)
    b = torch.randn(1)
    lossGrad = torch.randn(1, 1, 7)   # Lout=(3-1)*2+1*(2-1)+1+1=7
    return _run_fixture_sym("stride2OutputPaddingSym", x, w, b, lossGrad,
                            stride=2, padding=0, output_padding=1, dilation=1, groups=1)


SYM_FIXTURES = [
    fixture_single_channel_single_batch(),   # reused FLOAT: VALID, no bias, ones (allowed)
    fixture_single_channel_with_bias(),      # reused FLOAT: VALID, bias, ones (allowed)
    fixture_multi_channel_with_bias_sym(),   # channel mixing, bias, RANDOM lossGrad
    fixture_groups_grouped_sym(),            # groups=2, bias, RANDOM lossGrad
    fixture_stride2_sym(),                   # stride=2 upsample, no bias, RANDOM lossGrad
    fixture_dilation2_sym(),                 # dilation=2, no bias, RANDOM lossGrad
    fixture_stride2_output_padding_sym(),    # outputPadding=1, bias, RANDOM lossGrad
]


def emit_fixture(parts, fx):
    pre = f"convT1d_{fx['name']}"
    parts.append(emit_float_array(f"input_{pre}", fx["x"]))
    parts.append(emit_float_array(f"weight_{pre}", fx["w"]))
    if fx["b"] is not None:
        parts.append(emit_float_array(f"bias_{pre}", fx["b"]))
    parts.append(emit_float_array(f"expectedForward_{pre}", fx["y"]))
    parts.append(emit_float_array(f"expectedPropLoss_{pre}", fx["dx"]))
    parts.append(emit_float_array(f"expectedWeightGrad_{pre}", fx["dw"]))
    if fx["db"] is not None:
        parts.append(emit_float_array(f"expectedBiasGrad_{pre}", fx["db"]))


def emit_sym_fixture(parts, fx):
    g = emulate_sym_convT(fx)
    pre = f"convT1dSym_{g['name']}"
    parts.append(emit_float_array(f"input_{pre}", g["x_deq"]))
    parts.append(emit_float_array(f"weight_{pre}", g["w_deq"]))
    if g["has_bias"]:
        parts.append(emit_float_array(f"bias_{pre}", g["b_deq"]))
    parts.append(emit_float_array(f"lossGrad_{pre}", g["gy_deq"]))
    # forward (width-restored, as conv1dTransposedForward now writes it — PR1b.2 D3)
    parts.append(emit_int32_array(f"expectedForward_{pre}", g["fwd_q"]))
    parts.append(emit_float_scalar(f"expectedForwardScale_{pre}", g["fwd_scale"]))
    parts.append(emit_int32_scalar(f"forwardMantissaTol_{pre}", g["fwd_mtol"]))
    parts.append(emit_float_array(f"expectedForwardDequant_{pre}", g["fwd_deq"]))
    parts.append(emit_float_scalar(f"forwardDequantTol_{pre}", g["fwd_dtol"]))
    parts.append(emit_float_scalar(f"forwardScaleTol_{pre}", g["fwd_scale_tol"]))
    # dx / propLoss
    parts.append(emit_int32_array(f"expectedPropLoss_{pre}", g["dx_q"]))
    parts.append(emit_float_scalar(f"expectedPropLossScale_{pre}", g["dx_scale"]))
    parts.append(emit_int32_scalar(f"propLossMantissaTol_{pre}", g["dx_mtol"]))
    parts.append(emit_float_array(f"expectedPropLossDequant_{pre}", g["dx_deq"]))
    parts.append(emit_float_scalar(f"propLossDequantTol_{pre}", g["dx_dtol"]))
    # weightGrad
    parts.append(emit_int32_array(f"expectedWeightGrad_{pre}", g["dw_q"]))
    parts.append(emit_float_scalar(f"expectedWeightGradScale_{pre}", g["dw_scale"]))
    parts.append(emit_int32_scalar(f"weightGradMantissaTol_{pre}", g["dw_mtol"]))
    parts.append(emit_float_array(f"expectedWeightGradDequant_{pre}", g["dw_deq"]))
    parts.append(emit_float_scalar(f"weightGradDequantTol_{pre}", g["dw_dtol"]))
    # biasGrad
    if g["has_bias"]:
        parts.append(emit_int32_array(f"expectedBiasGrad_{pre}", g["db_q"]))
        parts.append(emit_float_scalar(f"expectedBiasGradScale_{pre}", g["db_scale"]))
        parts.append(emit_int32_scalar(f"biasGradMantissaTol_{pre}", g["db_mtol"]))
        parts.append(emit_float_array(f"expectedBiasGradDequant_{pre}", g["db_deq"]))
        parts.append(emit_float_scalar(f"biasGradDequantTol_{pre}", g["db_dtol"]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    parts = [
        "// AUTOGENERATED by generate_expected_conv1d_transposed.py — DO NOT EDIT\n",
        "#ifndef ODT_EXPECTED_CONV1D_TRANSPOSED_H\n",
        "#define ODT_EXPECTED_CONV1D_TRANSPOSED_H\n",
        "#include <stdlib.h>\n\n",
    ]

    fixtures = [
        fixture_single_channel_single_batch(),
        fixture_single_channel_with_bias(),
        fixture_multi_channel_with_bias(),
        fixture_multi_batch(),
        fixture_groups_depthwise(),
        fixture_groups_grouped(),
        fixture_stride2(),
        fixture_stride2_with_output_padding(),
        fixture_dilation2(),
    ]
    for fx in fixtures:
        emit_fixture(parts, fx)

    for fx in SYM_FIXTURES:
        emit_sym_fixture(parts, fx)

    parts.append("\n#endif // ODT_EXPECTED_CONV1D_TRANSPOSED_H\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(parts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
