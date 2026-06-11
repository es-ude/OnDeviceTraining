#!/usr/bin/env python3
"""Generate expected_layernorm_sym_bwd.h for UnitTestLayerNorm (SYM_INT32 bwd, #148 PR-3).

Emulates the C SYM_INT32 LayerNorm backward end-to-end in float64:
  quantize x/gamma/dy -> recompute group stats from x mantissas (int sum, then
  scale; biased /N; eps INSIDE sqrt) -> per-element dy dequant -> dgamma/dbeta
  increments summed over groups -> dx reduction -> whole-tensor absmax requant
  (absmax==0 -> zeros, scale 1.0) -> grad accumulation exactly as the C does it
  (quantize increment, then addSymInt32TensorsInplace: dequant both, float add,
  requant to a fresh absmax/QMAX scale).

Self-checks:
 - emitted float fixtures are dequantization-STABLE (re-quantizing reproduces
   the mantissas bit-exactly), pinning the C input mantissas to this emulation;
 - every dy fixture is per-group NON-uniform (with the default constant gamma,
   uniform dy degenerates dx to float rounding noise — vacuous for
   scatter/reduction mutations; enforced unconditionally as the stronger
   fixture property — codebase memory);
 - emulated dequantized dx/dgamma/dbeta match float64 torch-autograd on the
   EXACT float64 dequants (q*s; NOT the float32-rounded emitted fixtures —
   those differ by ~0.5 ulp_f32 and would trip the 1e-9 increment bound)
   within analytic quantization bounds;
 - the var~eps fixture really sits in the eps-visible regime (0.5 < var/eps < 2).

Rounding: the framework's roundByMode(HALF_AWAY) is C round() =
half-AWAY-from-zero (Rounding.c, roundHalfAway; renamed in #188) — emulate
with sign(x)*floor(|x|+0.5), NEVER torch.round (true half-to-even, silently
diverges on ties).
Run via `uv run` (CMake wires this automatically).
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

QMAX = 32767.0
QMIN = -32768.0


def round_half_away(x: torch.Tensor) -> torch.Tensor:
    """Match the C kernel: roundByMode(HALF_AWAY) is C round() =
    half-away-from-zero (Rounding.c, roundHalfAway; renamed in #188)."""
    return torch.sign(x) * torch.floor(torch.abs(x) + 0.5)


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


def emit_int32_array(name: str, tensor: torch.Tensor) -> str:
    flat = [int(v) for v in tensor.detach().flatten().tolist()]
    body = ", ".join(str(v) for v in flat)
    return (
        f"static const int32_t {name}[] = {{ {body} }};\n"
        f"static const size_t {name}_len = {len(flat)};\n"
    )


def emit_float_scalar(name: str, v: float) -> str:
    return f"static const float {name} = {_format_float_literal(float(v))};\n"


def emit_int32_scalar(name: str, v: int) -> str:
    return f"static const int32_t {name} = {int(v)};\n"


def quantize_sym(x: torch.Tensor):
    """convertFloatTensorToSymInt32Tensor: absmax -> scale (1.0 if absmax==0),
    round-clamp with the C rounding (half-away-from-zero)."""
    absmax = x.abs().max().item()
    scale = 1.0 if absmax == 0.0 else absmax / QMAX
    q = round_half_away(torch.clamp(x / scale, QMIN, QMAX))
    return q.to(torch.int32), scale


def stable_dequant(x: torch.Tensor):
    """Quantize once; return (mantissas, scale, dequantized float32 fixture).
    The EMITTED float32 fixture is asserted ROUND-TRIP STABLE so the C side's
    tensorFillFromFloatBuffer lands on exactly these mantissas."""
    q, s = quantize_sym(x.to(torch.float64))
    deq32 = (q.to(torch.float64) * s).to(torch.float32)
    q2, _ = quantize_sym(deq32.to(torch.float64))
    assert torch.equal(q, q2), "fixture is not dequantization round-trip stable"
    return q, s, deq32


def emulate_group_stats(xq, sx, n_inner, eps):
    """layerNormGroupStatsSymInt32 in float64: int mantissa sum then scale;
    biased variance /N; eps INSIDE sqrt. Returns (n, inv_sigma) per group."""
    xqr = xq.reshape(-1, n_inner).to(torch.float64)
    mean = sx * xqr.sum(dim=1, keepdim=True) / n_inner
    yhat = xqr * sx - mean
    var = (yhat ** 2).mean(dim=1, keepdim=True)
    inv_sigma = 1.0 / torch.sqrt(var + eps)
    return yhat * inv_sigma, inv_sigma, var


def emulate_sym_backward(xq, sx, gq, sg, dyq, sdy, n_inner, eps):
    """layerNormBackwardSymInt32's math in float64. Returns
    (dx_mantissas int32 [G,N], s_dx, dgamma_inc float64 [N], dbeta_inc float64 [N])."""
    n, inv_sigma, _ = emulate_group_stats(xq, sx, n_inner, eps)
    dy = dyq.reshape(-1, n_inner).to(torch.float64) * sdy
    g = (gq.to(torch.float64) * sg).reshape(1, n_inner)
    dbeta_inc = dy.sum(dim=0)
    dgamma_inc = (dy * n).sum(dim=0)
    dn = dy * g
    mean_dn = dn.mean(dim=1, keepdim=True)
    mean_dnn = (dn * n).mean(dim=1, keepdim=True)
    dx = inv_sigma * (dn - mean_dn - n * mean_dnn)
    absmax = dx.abs().max().item()
    if absmax == 0.0:
        s_dx = 1.0
        dxq = torch.zeros_like(dx)
    else:
        s_dx = absmax / QMAX
        dxq = round_half_away(torch.clamp(dx / s_dx, QMIN, QMAX))
    return dxq.to(torch.int32), s_dx, dgamma_inc, dbeta_inc


def emulate_accumulate(grad_q, grad_s, inc):
    """Strategy-A grad accumulation exactly as the C does it:
    (1) convertFloatTensorToSymInt32Tensor on the float increment (fresh
        absmax scale, absmax==0 -> 1.0, round-clamp);
    (2) addSymInt32TensorsInplace: dequantize BOTH with their own scales,
        float-add, requantize the sum with a fresh absmax/QMAX scale.
    Returns ((mantissas, scale) after accumulation, increment scale)."""
    inc_q, inc_s = quantize_sym(inc)
    total = grad_q.to(torch.float64) * grad_s + inc_q.to(torch.float64) * inc_s
    out_q, out_s = quantize_sym(total)
    return (out_q, out_s), inc_s


def autograd_reference(x64, g64, dy64, normalized_shape, eps):
    """float64 torch-autograd on the EXACT float64 dequants (q*s in float64) —
    the SAME inputs the emulator uses, so the increment self-checks hold to
    ~machine eps. Do NOT feed the emitted float32 fixtures here: they are
    rounded by ~0.5 ulp_f32 per element, which shifts dgamma by ~1e-8 (and
    ~6e-7 in the var~eps fixture, where cancellation amplifies it) and trips
    the 1e-9 bound. The C kernel sees identical MANTISSAS either way, so the
    emitted golds are unaffected by this choice."""
    x_in = x64.clone().detach().requires_grad_(True)
    g_in = g64.clone().detach().requires_grad_(True)
    b_in = torch.zeros(normalized_shape, dtype=torch.float64, requires_grad=True)
    y = F.layer_norm(x_in, normalized_shape, weight=g_in, bias=b_in, eps=eps)
    y.backward(dy64)
    return x_in.grad, g_in.grad, b_in.grad


def _check_dy_nonuniform(name, dy_deq, n_inner):
    per_group = dy_deq.reshape(-1, n_inner).to(torch.float64)
    spread = per_group.max(dim=1).values - per_group.min(dim=1).values
    assert (spread > 1e-6).all(), (
        f"{name}: a dy group is uniform — degenerate/vacuous fixture (see docstring)"
    )


def _run_bwd_fixture(name, x, gamma, dy, normalized_shape, *, eps=1e-5,
                     dy2=None, var_near_eps=False):
    n_inner = 1
    for d in normalized_shape:
        n_inner *= d

    xq, sx, x_deq = stable_dequant(x)
    gq, sg, g_deq = stable_dequant(gamma)
    dyq, sdy, dy_deq = stable_dequant(dy)
    _check_dy_nonuniform(name, dy_deq, n_inner)

    if var_near_eps:
        _, _, var = emulate_group_stats(xq, sx, n_inner, eps)
        ratio = (var / eps).flatten()
        assert ((ratio > 0.5) & (ratio < 2.0)).all(), (
            f"{name}: var/eps = {ratio.tolist()} outside the eps-visible regime"
        )

    dx_q, s_dx, dg_inc, db_inc = emulate_sym_backward(xq, sx, gq, sg, dyq, sdy,
                                                      n_inner, eps)
    # EXACT float64 dequants for the autograd reference (see autograd_reference).
    x64 = xq.to(torch.float64) * sx
    g64 = gq.to(torch.float64) * sg
    dy64 = dyq.to(torch.float64) * sdy
    ref_dx, ref_dg, ref_db = autograd_reference(x64, g64, dy64, normalized_shape, eps)

    # Self-check: emulated dequant dx vs autograd. The emulation IS the math
    # (sympy-verified equal to the Jacobian) in float64; the only difference is
    # the dx output quantization (<= 0.5 LSB). 1.5x margin.
    err = (dx_q.to(torch.float64).reshape(-1) * s_dx
           - ref_dx.reshape(-1)).abs().max().item()
    assert err <= 0.5 * s_dx * 1.5 + 1e-12, f"{name}: dx emulation err {err}"
    # Increments are pure float64 math — must match autograd to ~machine eps.
    assert (dg_inc - ref_dg.reshape(-1)).abs().max().item() <= 1e-9, name
    assert (db_inc - ref_db.reshape(-1)).abs().max().item() <= 1e-9, name

    # Grad accumulation from FRESH zero grads (scale 1.0, sgdZeroGrad state).
    zeros = torch.zeros(n_inner, dtype=torch.int32)
    (dg_q, dg_s), dg_inc_s = emulate_accumulate(zeros, 1.0, dg_inc)
    (db_q, db_s), db_inc_s = emulate_accumulate(zeros, 1.0, db_inc)

    # Analytic dequant tolerance for the C-vs-autograd grad assert:
    # increment quant (0.5*s_inc) + running-sum requant (0.5*s_final)
    # + +-2 LSB float32-vs-float64 mantissa drift (2*s_final); 1.5x margin.
    dg_tol = (0.5 * dg_inc_s + 2.5 * dg_s) * 1.5
    db_tol = (0.5 * db_inc_s + 2.5 * db_s) * 1.5
    err_g = (dg_q.to(torch.float64) * dg_s - ref_dg.reshape(-1)).abs().max().item()
    err_b = (db_q.to(torch.float64) * db_s - ref_db.reshape(-1)).abs().max().item()
    assert err_g <= dg_tol, f"{name}: dgamma accum err {err_g} > {dg_tol}"
    assert err_b <= db_tol, f"{name}: dbeta accum err {err_b} > {db_tol}"

    fx = {
        "name": name, "x": x_deq, "gamma": g_deq, "dy": dy_deq,
        "dx_q": dx_q, "s_dx": s_dx,
        "dx_dequant": ref_dx.to(torch.float32),
        "dx_dequant_tol": 3.0 * s_dx,  # +-2 LSB drift + 0.5 LSB rounding + margin
        "dg_q": dg_q, "dg_s": dg_s, "db_q": db_q, "db_s": db_s,
        "dg_dequant": ref_dg.to(torch.float32),
        "db_dequant": ref_db.to(torch.float32),
        "grad_dequant_tol": max(dg_tol, db_tol),
    }

    if dy2 is not None:
        dy2q, sdy2, dy2_deq = stable_dequant(dy2)
        _check_dy_nonuniform(name + "/dy2", dy2_deq, n_inner)
        dx2_q, s_dx2, dg_inc2, db_inc2 = emulate_sym_backward(
            xq, sx, gq, sg, dy2q, sdy2, n_inner, eps)
        dy2_64 = dy2q.to(torch.float64) * sdy2
        ref_dx2, ref_dg2, ref_db2 = autograd_reference(
            x64, g64, dy2_64, normalized_shape, eps)
        (dg2_q, dg2_s), dg2_inc_s = emulate_accumulate(dg_q, dg_s, dg_inc2)
        (db2_q, db2_s), db2_inc_s = emulate_accumulate(db_q, db_s, db_inc2)
        ref_dg_sum = (ref_dg + ref_dg2).reshape(-1)
        ref_db_sum = (ref_db + ref_db2).reshape(-1)
        dg2_tol = (0.5 * dg_inc_s + 0.5 * dg_s + 0.5 * dg2_inc_s + 2.5 * dg2_s) * 1.5
        db2_tol = (0.5 * db_inc_s + 0.5 * db_s + 0.5 * db2_inc_s + 2.5 * db2_s) * 1.5
        err_g2 = (dg2_q.to(torch.float64) * dg2_s - ref_dg_sum).abs().max().item()
        err_b2 = (db2_q.to(torch.float64) * db2_s - ref_db_sum).abs().max().item()
        assert err_g2 <= dg2_tol, f"{name}: 2-call dgamma err {err_g2} > {dg2_tol}"
        assert err_b2 <= db2_tol, f"{name}: 2-call dbeta err {err_b2} > {db2_tol}"
        fx.update({
            "dy2": dy2_deq, "dx2_q": dx2_q, "s_dx2": s_dx2,
            "dg2_q": dg2_q, "dg2_s": dg2_s, "db2_q": db2_q, "db2_s": db2_s,
            "dg2_dequant": ref_dg_sum.to(torch.float32),
            "db2_dequant": ref_db_sum.to(torch.float32),
            "grad_dequant_tol2": max(dg2_tol, db2_tol),
        })
    return fx


def fixture_bwd_base():
    # [3,4], D=1, G=3, N=4 (N>=3). Per-group variance ratio ~445x (row1) pins
    # per-group invSigma in dx. Non-uniform dy with sign/magnitude structure
    # per group; gamma non-trivial with s_gamma = 2/32767 (scale-blindness
    # mutations must show up in the dequant/scale asserts).
    x = torch.tensor([[0.1, -0.4, 0.7, 1.2],
                      [20.0, 10.0, -15.0, 5.0],
                      [-0.3, -0.8, 0.2, 0.9]], dtype=torch.float64)
    gamma = torch.tensor([1.5, -0.5, 2.0, 0.25], dtype=torch.float64)
    dy = torch.tensor([[1.0, -2.0, 0.5, 1.5],
                       [-0.25, 0.75, 1.25, -1.0],
                       [2.0, 0.5, -1.5, -0.75]], dtype=torch.float64)
    return _run_bwd_fixture("bwdBase", x, gamma, dy, [4])


def fixture_bwd_var_eps():
    # var == eps == 1e-5 (d = sqrt(1.5e-5), var = 2d^2/3): the ONLY regime
    # where eps placement in the backward is visible (codebase memory:
    # O(1)-variance fixtures are blind). N=3, G=1, gamma=ones.
    d = 0.0038729833
    x = torch.tensor([1.0 - d, 1.0, 1.0 + d], dtype=torch.float64)
    gamma = torch.ones(3, dtype=torch.float64)
    dy = torch.tensor([0.5, -1.0, 0.25], dtype=torch.float64)
    return _run_bwd_fixture("bwdVarEps", x, gamma, dy, [3], var_near_eps=True)


def fixture_bwd_two_calls():
    # [2,4]; dy2 = 10x dy1 -> s_dx must refresh ~10x on call 2 (freeze-scale
    # test) and grads must accumulate per strategy A across calls.
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0],
                      [10.0, 20.0, 30.0, 40.0]], dtype=torch.float64)
    gamma = torch.tensor([2.0, 1.0, -1.0, 0.5], dtype=torch.float64)
    dy1 = torch.tensor([[0.5, -1.0, 0.25, 0.75],
                        [-0.5, 1.5, -0.25, 1.0]], dtype=torch.float64)
    dy2 = torch.tensor([[5.0, -10.0, 2.5, 7.5],
                        [-5.0, 15.0, -2.5, 10.0]], dtype=torch.float64)
    return _run_bwd_fixture("bwdTwoCalls", x, gamma, dy1, [4], dy2=dy2)


def emit_fixture(parts, fx):
    pre = f"layerNormSymBwd_{fx['name']}"
    parts.append(emit_float_array(f"input_{pre}", fx["x"]))
    parts.append(emit_float_array(f"gamma_{pre}", fx["gamma"]))
    parts.append(emit_float_array(f"lossGrad_{pre}", fx["dy"]))
    parts.append(emit_int32_array(f"expectedDx_{pre}", fx["dx_q"]))
    parts.append(emit_float_scalar(f"expectedDxScale_{pre}", fx["s_dx"]))
    parts.append(emit_int32_scalar(f"dxMantissaTol_{pre}", 2))
    parts.append(emit_float_array(f"expectedDxDequant_{pre}", fx["dx_dequant"]))
    parts.append(emit_float_scalar(f"dxDequantTol_{pre}", fx["dx_dequant_tol"]))
    parts.append(emit_int32_array(f"expectedDgamma_{pre}", fx["dg_q"]))
    parts.append(emit_float_scalar(f"expectedDgammaScale_{pre}", fx["dg_s"]))
    parts.append(emit_int32_array(f"expectedDbeta_{pre}", fx["db_q"]))
    parts.append(emit_float_scalar(f"expectedDbetaScale_{pre}", fx["db_s"]))
    parts.append(emit_int32_scalar(f"gradMantissaTol_{pre}", 2))
    parts.append(emit_float_array(f"expectedDgammaDequant_{pre}", fx["dg_dequant"]))
    parts.append(emit_float_array(f"expectedDbetaDequant_{pre}", fx["db_dequant"]))
    parts.append(emit_float_scalar(f"gradDequantTol_{pre}", fx["grad_dequant_tol"]))
    if "dy2" in fx:
        parts.append(emit_float_array(f"lossGrad2_{pre}", fx["dy2"]))
        parts.append(emit_int32_array(f"expectedDx2_{pre}", fx["dx2_q"]))
        parts.append(emit_float_scalar(f"expectedDxScale2_{pre}", fx["s_dx2"]))
        parts.append(emit_int32_array(f"expectedDgammaAfter2_{pre}", fx["dg2_q"]))
        parts.append(emit_float_scalar(f"expectedDgammaScaleAfter2_{pre}", fx["dg2_s"]))
        parts.append(emit_int32_array(f"expectedDbetaAfter2_{pre}", fx["db2_q"]))
        parts.append(emit_float_scalar(f"expectedDbetaScaleAfter2_{pre}", fx["db2_s"]))
        parts.append(emit_float_array(f"expectedDgammaDequant2_{pre}", fx["dg2_dequant"]))
        parts.append(emit_float_array(f"expectedDbetaDequant2_{pre}", fx["db2_dequant"]))
        parts.append(emit_float_scalar(f"gradDequantTol2_{pre}", fx["grad_dequant_tol2"]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    parts = [
        "// AUTOGENERATED by generate_expected_layernorm_sym_bwd.py — DO NOT EDIT\n",
        "#ifndef ODT_EXPECTED_LAYERNORM_SYM_BWD_H\n",
        "#define ODT_EXPECTED_LAYERNORM_SYM_BWD_H\n",
        "#include <stdint.h>\n",
        "#include <stdlib.h>\n\n",
    ]

    for fx in [fixture_bwd_base(), fixture_bwd_var_eps(), fixture_bwd_two_calls()]:
        emit_fixture(parts, fx)

    parts.append("\n#endif // ODT_EXPECTED_LAYERNORM_SYM_BWD_H\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(parts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
