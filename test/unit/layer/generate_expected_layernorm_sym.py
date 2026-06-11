#!/usr/bin/env python3
"""Generate expected_layernorm_sym.h for UnitTestLayerNorm (SYM_INT32 fwd, #148 PR-2).

Emulates the C SYM_INT32 LayerNorm forward end-to-end in float64 (quantize
inputs -> per-group stats from mantissas -> global absmax stretch -> separate
affine with beta rescale) and emits the expected output mantissas + scale,
plus the true PyTorch xhat for dequant-domain parity, per fixture.

Self-checks:
 - emitted float fixtures are dequantization-STABLE: re-quantizing them
   reproduces the mantissas bit-exactly (so the C side's
   tensorFillFromFloatBuffer lands on the same input mantissas);
 - the emulated dequantized output matches F.layer_norm on the dequantized
   inputs within the analytic quantization bound;
 - parity fixtures keep absmax(xhat) <= 1.9, which guarantees the documented
   3e-5 C-side xhat tolerance (s_norm/2 <= 1.9/65534 ~ 2.9e-5).

Biased variance (/N) via explicit emulation + F.layer_norm; NEVER torch.var
(ddof=1 default). Rounding: the framework's roundByMode(HALF_AWAY) is C
round() = half-AWAY-from-zero (Rounding.c, roundHalfAway; renamed in #188),
so all rounding here uses round_half_away — NEVER torch.round (true
half-to-even, which diverges on ties like 16382.5).
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
    half-away-from-zero (Rounding.c, roundHalfAway; renamed in #188).
    torch.round would be true half-to-even and silently diverge on .5 ties."""
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
    The EMITTED float32 fixture is asserted ROUND-TRIP STABLE: re-quantizing
    it reproduces the same mantissas, so the C side's tensorFillFromFloatBuffer
    lands on exactly these mantissas."""
    q, s = quantize_sym(x.to(torch.float64))
    deq32 = (q.to(torch.float64) * s).to(torch.float32)
    q2, _ = quantize_sym(deq32.to(torch.float64))
    assert torch.equal(q, q2), "fixture is not dequantization round-trip stable"
    return q, s, deq32


def emulate_sym_forward(xq, sx, gq, sg, bq, sb, normalized_shape, eps):
    """Float64 emulation of layerNormForwardSymInt32 + layerNormAffineSymInt32."""
    n_inner = 1
    for d in normalized_shape:
        n_inner *= d
    groups = xq.numel() // n_inner
    xqr = xq.reshape(groups, n_inner).to(torch.float64)
    mean = sx * xqr.sum(dim=1, keepdim=True) / n_inner  # int sum, then scale
    yhat = xqr * sx - mean
    var = (yhat ** 2).mean(dim=1, keepdim=True)  # biased /N
    inv_sigma = 1.0 / torch.sqrt(var + eps)  # eps INSIDE sqrt
    n = yhat * inv_sigma
    absmax = n.abs().max().item()
    if absmax == 0.0:
        s_norm = 1.0
        q = torch.zeros_like(n)
    else:
        k = QMAX / absmax
        s_norm = 1.0 / k
        q = round_half_away(torch.clamp(n * k, QMIN, QMAX))
    s_y = s_norm * sg
    seed = round_half_away(bq.to(torch.float64) * (sb / s_y)).reshape(1, n_inner)
    yq = q * gq.reshape(1, n_inner).to(torch.float64) + seed
    var_ratio = (var / (var + eps)).flatten()
    return yq.to(torch.int32).reshape(xq.shape), s_y, s_norm, var_ratio


def _run_fixture(name, x, gamma, beta, normalized_shape, *, eps=1e-5,
                 require_parity=False):
    xq, sx, x_deq = stable_dequant(x)
    gq, sg, g_deq = stable_dequant(gamma)
    bq, sb, b_deq = stable_dequant(beta)

    yq, s_y, s_norm, var_ratio = emulate_sym_forward(
        xq, sx, gq, sg, bq, sb, normalized_shape, eps)

    # True PyTorch reference ON THE DEQUANTIZED inputs (removes input-quant
    # error from the comparison entirely).
    ref = F.layer_norm(x_deq.to(torch.float64), normalized_shape,
                       weight=g_deq.to(torch.float64),
                       bias=b_deq.to(torch.float64), eps=eps)
    xhat = F.layer_norm(x_deq.to(torch.float64), normalized_shape, eps=eps)

    # Self-check: emulated dequant vs F.layer_norm within the analytic bound
    # (n rounding <= 0.5 LSB * |gamma|, seed rounding <= 0.5 LSB; 1.5x margin).
    gmax = g_deq.abs().max().item()
    bound = (0.5 * s_norm * gmax + 0.5 * s_y) * 1.5 + 1e-7
    err = (yq.to(torch.float64) * s_y - ref).abs().max().item()
    assert err <= bound, f"{name}: emulation vs F.layer_norm {err} > {bound}"

    if require_parity:
        a = xhat.abs().max().item()
        assert a <= 1.9, f"{name}: absmax(xhat)={a} > 1.9 breaks the 3e-5 tol"

    # C-vs-emulation: float32 stats noise can flip a rounding boundary by
    # +-2 LSB of q, scaled by gamma_q, +-1 on the seed.
    mantissa_tol = 2 * int(gq.abs().max().item()) + 1
    dequant_tol = bound + mantissa_tol * s_y

    return {
        "name": name,
        "x": x_deq, "gamma": g_deq, "beta": b_deq,
        "mantissas": yq, "scale": s_y,
        "mantissa_tol": mantissa_tol, "dequant_tol": dequant_tol,
        "dequant": ref.to(torch.float32),
        "xhat": xhat.to(torch.float32) if require_parity else None,
        "var_ratio": var_ratio.to(torch.float32),
    }


def fixture_sym_parity():
    # [3,4], D=1, G=3, gamma=1/beta=0: pure normalization. N=4 (N>=3: avoids
    # the N=2 saturation trap). Hand values keep absmax(xhat) <= 1.9.
    x = torch.tensor([[0.1, -0.4, 0.7, 1.2],
                      [2.0, 1.0, -1.5, 0.5],
                      [-0.3, -0.8, 0.2, 0.9]], dtype=torch.float64)
    return _run_fixture("symParity", x, torch.ones(4, dtype=torch.float64),
                        torch.zeros(4, dtype=torch.float64), [4],
                        require_parity=True)


def fixture_sym_affine():
    # [2,4], non-trivial gamma/beta with deliberately different scales:
    # s_gamma = 2/32767, s_beta = 0.05/32767 — the beta rescale MUST matter.
    x = torch.tensor([[0.2, -1.0, 0.6, 1.4],
                      [1.0, 3.0, -2.0, 0.0]], dtype=torch.float64)
    gamma = torch.tensor([1.5, -0.5, 2.0, 0.25], dtype=torch.float64)
    beta = torch.tensor([0.05, -0.0125, 0.025, -0.05], dtype=torch.float64)
    return _run_fixture("symAffine", x, gamma, beta, [4])


def fixture_sym_sigma_ratio():
    # [2,4], per-group sigma differs ~22x (0.354 vs 7.9): pins that the
    # per-group 1/sigma_g hits the DATA — a single folded sigma is ~20x off
    # for one of the groups. Values are x10 the obvious choice so the low-var
    # group keeps var = 0.125 >> eps: eps mutations stay INVISIBLE here
    # (division of labor — Task 6 owns eps visibility). gamma=1/beta=0.
    x = torch.tensor([[4.5, 5.0, 5.5, 5.0],
                      [-10.0, 10.0, 5.0, -5.0]], dtype=torch.float64)
    return _run_fixture("symSigmaRatio", x, torch.ones(4, dtype=torch.float64),
                        torch.zeros(4, dtype=torch.float64), [4],
                        require_parity=True)


def emit_fixture(parts, fx):
    pre = f"layerNormSym_{fx['name']}"
    parts.append(emit_float_array(f"input_{pre}", fx["x"]))
    parts.append(emit_float_array(f"gamma_{pre}", fx["gamma"]))
    parts.append(emit_float_array(f"beta_{pre}", fx["beta"]))
    parts.append(emit_int32_array(f"expectedMantissas_{pre}", fx["mantissas"]))
    parts.append(emit_float_scalar(f"expectedScale_{pre}", fx["scale"]))
    parts.append(emit_int32_scalar(f"mantissaTol_{pre}", fx["mantissa_tol"]))
    parts.append(emit_float_array(f"expectedDequant_{pre}", fx["dequant"]))
    parts.append(emit_float_scalar(f"dequantTol_{pre}", fx["dequant_tol"]))
    if fx["xhat"] is not None:
        parts.append(emit_float_array(f"expectedXhat_{pre}", fx["xhat"]))
    parts.append(emit_float_array(f"expectedGroupVarRatio_{pre}", fx["var_ratio"]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    parts = [
        "// AUTOGENERATED by generate_expected_layernorm_sym.py — DO NOT EDIT\n",
        "#ifndef ODT_EXPECTED_LAYERNORM_SYM_H\n",
        "#define ODT_EXPECTED_LAYERNORM_SYM_H\n",
        "#include <stdint.h>\n",
        "#include <stdlib.h>\n\n",
    ]

    for fx in [fixture_sym_parity(), fixture_sym_affine(), fixture_sym_sigma_ratio()]:
        emit_fixture(parts, fx)

    parts.append("\n#endif // ODT_EXPECTED_LAYERNORM_SYM_H\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(parts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
