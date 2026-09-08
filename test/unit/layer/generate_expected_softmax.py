#!/usr/bin/env python3
"""Generate expected_softmax.h for UnitTestSoftmax + UnitTestMultiLayerTraining
(BFP epic PR6 Task 2, ruling P6-1).

Root-cause bug: the training loop hands every layer backward the layer's
INPUT (layerOutputs[i]); softmax's backward arms treated that as the softmax
OUTPUT `s` and applied the Jacobian formula to raw logits. The fix recomputes
`s` from the logits inside the backward, in ALL arms. This script's fixtures
are therefore LOGITS (pre-fix fixtures fed probabilities directly and
happened to "work" only because CrossEntropy's combined-gradient shortcut
skips softmax backward entirely -- backwardIndex -= 1 -- so nothing noticed).

Sections:
  1. Float backward fixture (UnitTestSoftmax): X (logits, rank-2 [2,3], ONE
     distribution over all 6 elements -- the layer normalizes the WHOLE
     tensor, not per row) and DLDS (upstream grad); S = softmax(X.flatten())
     and EXPECTED_DX = S * (DLDS - dot(S, DLDS)), self-checked against
     torch.autograd on softmax(x) with upstream grad DLDS (this pins the
     FORMULA, not just the arithmetic).
  2. SYM_INT32 section (UnitTestSoftmax): X quantized/dequantized at the
     int12 per-tensor absmax grid (qMaxBits = ODT_SYM_OPERAND_QMAXBITS = 12)
     -- the SAME grid tensorFillFromFloatBuffer derives when filling a
     SymInt32 tensor from the literal X array (convertFloatTensorToSymInt32-
     Tensor, TensorConversion.c) -- then the SAME float Jacobian formula on
     the dequantized X. DLDS is used RAW (not requantized): the SYM test's
     existing (loose) tolerance already absorbs that secondary quantization
     noise, same as the pre-fix fixture relying on the tolerance to absorb
     all three wires' quantization noise.
  3. Loop-contract e2e section (UnitTestMultiLayerTraining): Linear(3->3,
     ramp weights, buildRampParam2D convention) -> Softmax, MSE loss
     (reduction=SUM, matching the raw per-element 2(o-l) backward convention
     -- docs/conventions/loss.md: the backward emits the raw per-element
     gradient regardless of the reduction the caller reports), ONE
     forward+backward through the WHOLE chain via torch.autograd. Emits the
     LINEAR layer's weight grad -- the strongest pin that P6-1's fix is wired
     correctly end to end: a wrong dx through softmax poisons the upstream
     linear weight grad.

Literals: repr(v)+"f". Run via `uv run` (CMake wires this automatically).
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "goldgen"))
from sym_gold import stable_dequant_i12  # noqa: E402


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


def softmax_values(x: torch.Tensor) -> torch.Tensor:
    """Mirror softmaxValuesFloat EXACTLY: max by strict > first-wins, expf,
    float sum in index order, divide -- the same formula the C forward
    kernel (and, after this task, the backward's recompute) implements."""
    m = x[0].clone()
    for i in range(1, x.numel()):
        if x[i] > m:
            m = x[i].clone()
    e = torch.exp(x - m)
    s = e.sum()
    return e / s


def jacobian_vjp(s: torch.Tensor, dLds: torch.Tensor) -> torch.Tensor:
    dot = (s * dLds).sum()
    return s * (dLds - dot)


# ---------------------------------------------------------------------------
# Section 1: float backward fixture (Steps 1-4).
# ---------------------------------------------------------------------------


def fixture_float_backward():
    x = torch.tensor([0.9, -0.3, 1.7, -1.2, 0.4, 2.1], dtype=torch.float64)
    dlds = torch.tensor([0.0, 2.0, -4.0, 6.0, 3.0, 2.0], dtype=torch.float64)

    s = softmax_values(x)
    dx = jacobian_vjp(s, dlds)

    # Self-check against torch.autograd (pins the FORMULA, not just the
    # arithmetic): softmax(x) with upstream grad dlds must reproduce dx.
    x_ag = x.clone().requires_grad_(True)
    s_ag = torch.softmax(x_ag, dim=0)
    s_ag.backward(dlds)
    assert torch.allclose(s_ag.detach(), s, atol=1e-12), (
        "fixture_float_backward: hand softmax_values disagrees with torch.softmax")
    assert torch.allclose(x_ag.grad, dx, atol=1e-10), (
        "fixture_float_backward: hand Jacobian-VJP disagrees with torch.autograd")

    return {
        "x": x.to(torch.float32),
        "dlds": dlds.to(torch.float32),
        "dx": dx.to(torch.float32),
    }


# ---------------------------------------------------------------------------
# Section 2: SYM_INT32 backward.
# ---------------------------------------------------------------------------


def fixture_sym_backward(float_fixture):
    x64 = float_fixture["x"].to(torch.float64)
    dlds64 = float_fixture["dlds"].to(torch.float64)

    _, _, x_deq = stable_dequant_i12(x64)  # float32, round-trip stable at int12

    s = softmax_values(x_deq.to(torch.float64))
    dx = jacobian_vjp(s, dlds64)

    return {"dx": dx.to(torch.float32)}


# ---------------------------------------------------------------------------
# Section 3: loop-contract e2e (UnitTestMultiLayerTraining Step 6).
# Linear(3->3, ramp weights) -> Softmax, MSE(reduction=SUM, matching the raw
# per-element 2(o-l) backward convention).
# ---------------------------------------------------------------------------


def _ramp_2d(base: float, step: float, rows: int, cols: int) -> torch.Tensor:
    """Mirror buildRampParam2D EXACTLY (C: values[i] = base + step*(float)i,
    row-major, float32 arithmetic throughout)."""
    n = rows * cols
    base32 = torch.tensor(base, dtype=torch.float32)
    step32 = torch.tensor(step, dtype=torch.float32)
    idx = torch.arange(n, dtype=torch.float32)
    flat = base32 + step32 * idx
    return flat.reshape(rows, cols)


def fixture_e2e_linear_softmax_mse():
    w0 = _ramp_2d(0.1, 0.05, 3, 3)  # float32, bit-matches the C ramp
    b0 = _ramp_2d(0.0, 0.0, 1, 3).flatten()  # float32, all zero

    w = w0.to(torch.float64).clone().requires_grad_(True)
    b = b0.to(torch.float64).clone().requires_grad_(True)
    x = torch.tensor([[1.0, -0.5, 2.0]], dtype=torch.float64)
    label = torch.tensor([[0.2, 0.5, 0.3]], dtype=torch.float64)

    y = F.linear(x, w, b)
    s = torch.softmax(y.flatten(), dim=0).reshape(y.shape)
    loss = F.mse_loss(s, label, reduction="sum")
    loss.backward()

    # Self-check: reconstruct the SAME chain via the repo's raw-per-element
    # backward convention (docs/conventions/loss.md: MSE backward emits
    # 2(o-l) regardless of reduction) -- torch.autograd's softmax VJP (the
    # correct math P6-1 makes the C code compute) must reproduce the SAME
    # weight grad as loss.backward() under reduction='sum'.
    w2 = w.detach().clone().requires_grad_(True)
    b2 = b.detach().clone().requires_grad_(True)
    y2 = F.linear(x, w2, b2)
    s2 = torch.softmax(y2.flatten(), dim=0).reshape(y2.shape)
    dLds2 = 2.0 * (s2.detach() - label)
    s2.backward(dLds2)
    assert torch.allclose(w2.grad, w.grad, atol=1e-9), (
        "fixture_e2e: raw-2(o-l) backward disagrees with reduction=sum autograd")

    return {"weight_grad": w.grad.detach().to(torch.float32)}


def emit_fixture_float(parts, fx):
    parts.append(emit_float_array("softmaxBackwardX", fx["x"]))
    parts.append(emit_float_array("softmaxBackwardDLds", fx["dlds"]))
    parts.append(emit_float_array("softmaxBackwardExpectedDx", fx["dx"]))


def emit_fixture_sym(parts, fx):
    parts.append(emit_float_array("softmaxBackwardSymExpectedDx", fx["dx"]))


def emit_fixture_e2e(parts, fx):
    parts.append(emit_float_array("softmaxMseE2eExpectedWeightGrad", fx["weight_grad"]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    parts = [
        "// AUTOGENERATED by generate_expected_softmax.py — DO NOT EDIT\n",
        "#ifndef ODT_EXPECTED_SOFTMAX_H\n",
        "#define ODT_EXPECTED_SOFTMAX_H\n",
        "#include <stdlib.h>\n\n",
    ]

    float_fx = fixture_float_backward()
    sym_fx = fixture_sym_backward(float_fx)
    e2e_fx = fixture_e2e_linear_softmax_mse()

    emit_fixture_float(parts, float_fx)
    emit_fixture_sym(parts, sym_fx)
    emit_fixture_e2e(parts, e2e_fx)

    parts.append("\n#endif // ODT_EXPECTED_SOFTMAX_H\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(parts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
