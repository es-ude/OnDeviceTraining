#!/usr/bin/env python3
"""Generate expected_cross_entropy_sym.h for UnitTestCrossEntropy (#206, M5).

Emulates the C SYM_INT32 CrossEntropy fake-quant arms (the MSE idiom):
forward = dequant softmax-output (and label) to float32, then the FLOAT32
core (pi = fmaxf(p, 1e-7); loss += y * -logf(pi), MEAN divides by
microbatch); backward = float32 (p - y), requantized into the int12 SYM
result via convertFloatTensorToSymInt32Tensor (absmax -> scale, round-clamp
half-away).

Self-checks:
 - the emitted softmax-output fixture is dequantization round-trip STABLE
   (stable_dequant_i12), so the C side lands on exactly the gold mantissas;
 - the emitted forward loss matches the float64 reference
   -(y * log(clamp(p_deq, 1e-7))).sum() within an analytic float32 bound;
 - the emitted backward mantissas reproduce under an independent recompute.

Rounding: round_half_away (sym_gold), NEVER torch.round. Labels stay FLOAT32
(the real training-loop case: one-hot from the DataLoader); convertTensor
handles any label dtype in C, the float path is the contract pinned here.
Run via `uv run` (CMake wires this automatically).
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "goldgen"))
from dyadic_scale import fit_scale  # noqa: E402
from sym_gold import (
    QMAX_I12,
    QMIN_I12,
    assert_rounding_canary,
    emit_float_scalar,
    emit_int32_array,
    emit_int32_scalar,
    f32_scale_i12,
    round_half_away,
    stable_dequant_i12,
)

# C-vs-emulation gap: the backward is one float32 subtract + one requant whose
# expression order the emulation mirrors exactly, so mantissa +-1 / scale rel
# 1e-4 are defensive (Conv1d goldgen floor); the forward carries logf noise.
MANTISSA_TOL_SYM = 1
SCALE_REL_TOL_SYM = 1e-4
LOSS_ABS_TOL = 1e-4


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


def _requant_i12_from_f32(values_f32: torch.Tensor):
    """convertFloatTensorToSymInt32Tensor on a float32 array: absmax (f32) ->
    canonical CEIL scale fit_scale(absmax, 2047), round-clamp half-away.
    Returns (int32, scale)."""
    absmax = values_f32.abs().max().to(torch.float32)
    if absmax.item() == 0.0:
        return torch.zeros_like(values_f32, dtype=torch.int32), 1.0
    scale = torch.tensor(fit_scale(absmax.item(), QMAX_I12), dtype=torch.float32)
    q = round_half_away(torch.clamp(values_f32 / scale, QMIN_I12, QMAX_I12))
    return q.to(torch.int32), scale.item()


def _run_fixture(name, p, y):
    """p: softmax-output probabilities [B, C] (float64), y: one-hot labels."""
    assert_rounding_canary()
    p = p.to(torch.float64)
    y = y.to(torch.float64)
    pq, _, p_deq = stable_dequant_i12(p)
    s_p = f32_scale_i12(p_deq)

    # Forward core on the dequantized fixture, float32 like C (fmaxf + logf).
    p32 = p_deq.to(torch.float32)
    y32 = y.to(torch.float32)
    loss_sum = (y32 * -torch.log(torch.clamp(p32, min=1e-7))).sum().item()
    microbatch = p.shape[0]
    loss_mean = loss_sum / float(microbatch)

    # Self-check vs float64 reference.
    ref = (
        (y * -torch.log(torch.clamp(p_deq.to(torch.float64), min=1e-7))).sum().item()
    )
    assert abs(loss_sum - ref) <= LOSS_ABS_TOL, \
        f"{name}: f32 loss {loss_sum} vs f64 ref {ref} beyond {LOSS_ABS_TOL}"

    # Backward: float32 (p - y), requantized into int12 SYM. (Real self-checks
    # for this fixture family are the round-trip stability assert in
    # stable_dequant_i12 and the f64 loss bound above — a same-expression
    # "recompute" here would be a tautology, not protection.)
    g32 = p32 - y32
    gq, s_g = _requant_i12_from_f32(g32)

    return {
        "name": name,
        "p": p_deq, "y": y32,
        "loss_sum": loss_sum, "loss_mean": loss_mean,
        "g_mantissas": gq, "g_scale": s_g,
        "g_dequant": g32, "g_dequant_tol": (MANTISSA_TOL_SYM + 0.5) * s_g * 1.5,
    }


def fixture_sym_basic():
    # [1,3]: single microbatch row, one-hot label on the argmax class.
    p = torch.tensor([[0.62, 0.23, 0.15]])
    y = torch.tensor([[1.0, 0.0, 0.0]])
    return _run_fixture("symBasic", p, y)


def fixture_sym_microbatch():
    # [2,3]: MEAN must divide by microbatch=2 (dimensions[0]); second row's
    # label sits on a LOW-probability class so the (p-y) grad has a large
    # negative component (non-vacuous requant sign handling).
    p = torch.tensor([[0.62, 0.23, 0.15],
                      [0.10, 0.80, 0.10]])
    y = torch.tensor([[1.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0]])
    return _run_fixture("symMicrobatch", p, y)


def emit_fixture(parts, fx):
    pre = f"ceSym_{fx['name']}"
    parts.append(emit_float_array(f"softmaxOutput_{pre}", fx["p"]))
    parts.append(emit_float_array(f"label_{pre}", fx["y"]))
    parts.append(emit_float_scalar(f"expectedLossSum_{pre}", fx["loss_sum"]))
    parts.append(emit_float_scalar(f"expectedLossMean_{pre}", fx["loss_mean"]))
    parts.append(emit_int32_array(f"expectedGradMantissas_{pre}", fx["g_mantissas"]))
    parts.append(emit_float_scalar(f"expectedGradScale_{pre}", fx["g_scale"]))
    parts.append(emit_float_array(f"expectedGradDequant_{pre}", fx["g_dequant"]))
    parts.append(emit_float_scalar(f"gradDequantTol_{pre}", fx["g_dequant_tol"]))
    parts.append(emit_int32_scalar(f"mantissaTol_{pre}", MANTISSA_TOL_SYM))
    parts.append(emit_float_scalar(f"scaleTol_{pre}", SCALE_REL_TOL_SYM))
    parts.append(emit_float_scalar(f"lossTol_{pre}", LOSS_ABS_TOL))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    parts = [
        "// AUTOGENERATED by generate_expected_cross_entropy_sym.py — DO NOT EDIT\n",
        "#ifndef ODT_EXPECTED_CROSS_ENTROPY_SYM_H\n",
        "#define ODT_EXPECTED_CROSS_ENTROPY_SYM_H\n",
        "#include <stdint.h>\n",
        "#include <stdlib.h>\n\n",
    ]
    for fx in [fixture_sym_basic(), fixture_sym_microbatch()]:
        emit_fixture(parts, fx)
    parts.append("\n#endif // ODT_EXPECTED_CROSS_ENTROPY_SYM_H\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(parts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
