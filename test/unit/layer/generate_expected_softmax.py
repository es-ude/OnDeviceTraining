#!/usr/bin/env python3
"""Generate expected_softmax.h for UnitTestSoftmax + UnitTestMultiLayerTraining
(BFP epic PR6 Task 2, ruling P6-1; per-row semantic #152).

Semantic (#152): the layer normalizes PER ROW. Row = axis 0; every row
normalizes over ALL elements after axis 0 (count / dims[0] of them); a rank-1
tensor is one row -- crossEntropyForwardFloat's MEAN rule. Every fixture
below is therefore softmax(x.reshape(rows, -1), dim=-1). For rank >= 3
that is NOT PyTorch's last-axis softmax (section 4 asserts the difference).

Root-cause bug (P6-1): the training loop hands every layer backward the
layer's INPUT (layerOutputs[i]); softmax's backward arms treated that as the
softmax OUTPUT `s` and applied the Jacobian formula to raw logits. The fix
recomputes `s` from the logits inside the backward, in ALL arms. This
script's fixtures are therefore LOGITS.

Sections:
  0. Forward fixture (UnitTestSoftmax): X [2,3] logits, per-row S; plus the
     rank-1 fixture -- the SAME six logits as ONE row (the whole-vector
     softmax, i.e. what the layer computed for every rank before #152).
  1. Float backward fixture (UnitTestSoftmax): X [2,3] logits and DLDS; per
     row S and EXPECTED_DX = S * (DLDS - rowdot(S, DLDS)), self-checked
     against torch.autograd (pins the FORMULA, not just the arithmetic);
     plus the rank-1 dx over the same six values as ONE row.
  2. SYM_INT32 section (UnitTestSoftmax): X quantized/dequantized at the
     int12 per-tensor absmax grid (qMaxBits = ODT_SYM_OPERAND_QMAXBITS = 12)
     -- the SAME grid tensorFillFromFloatBuffer derives when filling a
     SymInt32 tensor from the literal X array -- then the SAME per-row float
     Jacobian formula on the dequantized X. DLDS is used RAW (not
     requantized): the SYM test's (loose) tolerance absorbs that noise.
  3. Loop-contract e2e (UnitTestMultiLayerTraining): Linear(3->3, ramp
     weights) -> Softmax, MSE (reduction=SUM, the raw per-element 2(o-l)
     backward convention, docs/conventions/loss.md), ONE row [1,3]. Emits the
     LINEAR layer's weight grad. B=1: byte-identical to the pre-#152 gold.
  4. Rank-3 fixture [2,2,3] (UnitTestSoftmax): each row spans its 6 trailing
     elements; forward S and backward dx. Asserts the result differs from a
     last-axis softmax, so the fixture discriminates the row geometry.
  5. Mixed-scale rows [2,3] (UnitTestSoftmax): row 0 logits ~120, row 1 ~0.
     Asserts that a SHARED global max underflows row 1 to 0/0 in float32, so
     the fixture pins the per-row max subtraction.
  6. CE e2e [2,3] (UnitTestMultiLayerTraining): Linear(3->3, ramp weights +
     ramp bias) -> Softmax, CrossEntropy. Emits the REDUCTION_MEAN loss
     (sum / rows, crossEntropyForwardFloat's MEAN rule) and the raw
     weight/bias grads (the fused (p - y) backward is the sum-reduction
     gradient). Asserts a whole-tensor softmax would move the grads, so the
     fixture discriminates.
  7. MSE e2e [2,3] (UnitTestMultiLayerTraining): same Linear -> Softmax, MSE.
     Emits the REDUCTION_MEAN loss (sum / numel, MSE.c) and the raw
     weight/bias grads; the only loop path that reaches the softmax
     BACKWARD at rows > 1 (CE skips the layer).

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


def emit_float_scalar(name: str, v: float) -> str:
    return f"static const float {name} = {_format_float_literal(float(v))};\n"


def softmax_values(x: torch.Tensor) -> torch.Tensor:
    """Mirror softmaxValuesFloat EXACTLY on ONE row: max by strict > first-
    wins, exp, sum in index order, divide."""
    m = x[0].clone()
    for i in range(1, x.numel()):
        if x[i] > m:
            m = x[i].clone()
    e = torch.exp(x - m)
    s = e.sum()
    return e / s


def row_view(x: torch.Tensor) -> torch.Tensor:
    """The #152 row partition (softmaxRowGeometry): [rows, rowLen] with
    rows = dims[0] for rank >= 2, one row for rank 1."""
    rows = x.shape[0] if x.dim() >= 2 else 1
    return x.reshape(rows, -1)


def softmax_rows(x: torch.Tensor) -> torch.Tensor:
    return torch.stack([softmax_values(r) for r in row_view(x)]).reshape(x.shape)


def jacobian_vjp_rows(s: torch.Tensor, dlds: torch.Tensor) -> torch.Tensor:
    s2 = row_view(s)
    d2 = row_view(dlds)
    dot = (s2 * d2).sum(dim=1, keepdim=True)
    return (s2 * (d2 - dot)).reshape(s.shape)


def torch_softmax_rows(x: torch.Tensor) -> torch.Tensor:
    """torch's own softmax over the same row partition (the autograd oracle)."""
    return torch.softmax(row_view(x), dim=-1).reshape(x.shape)


def check_against_autograd(name: str, x: torch.Tensor, dlds: torch.Tensor,
                           s: torch.Tensor, dx: torch.Tensor) -> None:
    x_ag = x.clone().requires_grad_(True)
    s_ag = torch_softmax_rows(x_ag)
    s_ag.backward(dlds)
    assert torch.allclose(s_ag.detach(), s, atol=1e-12), (
        f"{name}: hand softmax_rows disagrees with torch.softmax")
    assert torch.allclose(x_ag.grad, dx, atol=1e-10), (
        f"{name}: hand per-row Jacobian-VJP disagrees with torch.autograd")


def _ramp_2d(base: float, step: float, rows: int, cols: int) -> torch.Tensor:
    """Mirror buildRampParam2D EXACTLY (C: values[i] = base + step*(float)i,
    row-major, float32 arithmetic throughout)."""
    n = rows * cols
    base32 = torch.tensor(base, dtype=torch.float32)
    step32 = torch.tensor(step, dtype=torch.float32)
    idx = torch.arange(n, dtype=torch.float32)
    flat = base32 + step32 * idx
    return flat.reshape(rows, cols)


# ---------------------------------------------------------------------------
# Section 0: forward fixture.
# ---------------------------------------------------------------------------


def fixture_forward():
    x = torch.tensor([[-1.0, 0.0, 1.0], [2.0, 5.0, -6.0]], dtype=torch.float64)
    s = softmax_rows(x)
    assert torch.allclose(s, torch_softmax_rows(x), atol=1e-12)
    assert torch.allclose(row_view(s).sum(dim=1), torch.ones(2, dtype=torch.float64))

    x_rank1 = x.flatten()
    s_rank1 = softmax_rows(x_rank1)
    assert torch.allclose(s_rank1, torch.softmax(x_rank1, dim=0), atol=1e-12)
    assert not torch.allclose(s.flatten(), s_rank1, atol=1e-3), (
        "fixture_forward: per-row and one-row gold must differ (discriminator)")
    return {
        "x": x.to(torch.float32),
        "s": s.to(torch.float32),
        "s_rank1": s_rank1.to(torch.float32),
    }


# ---------------------------------------------------------------------------
# Section 1: float backward fixture.
# ---------------------------------------------------------------------------


def fixture_float_backward():
    x = torch.tensor([[0.9, -0.3, 1.7], [-1.2, 0.4, 2.1]], dtype=torch.float64)
    dlds = torch.tensor([[0.0, 2.0, -4.0], [6.0, 3.0, 2.0]], dtype=torch.float64)

    s = softmax_rows(x)
    dx = jacobian_vjp_rows(s, dlds)
    check_against_autograd("fixture_float_backward", x, dlds, s, dx)

    x_rank1 = x.flatten()
    dlds_rank1 = dlds.flatten()
    s_rank1 = softmax_rows(x_rank1)
    dx_rank1 = jacobian_vjp_rows(s_rank1, dlds_rank1)
    check_against_autograd("fixture_float_backward_rank1", x_rank1, dlds_rank1, s_rank1,
                           dx_rank1)
    assert not torch.allclose(dx.flatten(), dx_rank1, atol=1e-3), (
        "fixture_float_backward: per-row and one-row dx must differ (discriminator)")

    return {
        "x": x.to(torch.float32),
        "dlds": dlds.to(torch.float32),
        "dx": dx.to(torch.float32),
        "dx_rank1": dx_rank1.to(torch.float32),
    }


# ---------------------------------------------------------------------------
# Section 2: SYM_INT32 backward.
# ---------------------------------------------------------------------------


def fixture_sym_backward(float_fixture):
    x64 = float_fixture["x"].to(torch.float64)
    dlds64 = float_fixture["dlds"].to(torch.float64)

    _, _, x_deq = stable_dequant_i12(x64)  # float32, round-trip stable at int12

    s = softmax_rows(x_deq.to(torch.float64))
    dx = jacobian_vjp_rows(s, dlds64)

    return {"dx": dx.to(torch.float32)}


# ---------------------------------------------------------------------------
# Section 3: loop-contract e2e, MSE, ONE row (UnitTestMultiLayerTraining).
# ---------------------------------------------------------------------------


def fixture_e2e_linear_softmax_mse():
    w0 = _ramp_2d(0.1, 0.05, 3, 3)  # float32, bit-matches the C ramp
    b0 = _ramp_2d(0.0, 0.0, 1, 3).flatten()  # float32, all zero

    w = w0.to(torch.float64).clone().requires_grad_(True)
    b = b0.to(torch.float64).clone().requires_grad_(True)
    x = torch.tensor([[1.0, -0.5, 2.0]], dtype=torch.float64)
    label = torch.tensor([[0.2, 0.5, 0.3]], dtype=torch.float64)

    y = F.linear(x, w, b)
    s = torch.softmax(y, dim=-1)
    loss = F.mse_loss(s, label, reduction="sum")
    loss.backward()

    # Self-check: the repo's raw-per-element backward convention
    # (docs/conventions/loss.md: MSE backward emits 2(o-l) regardless of
    # reduction) through torch.autograd's softmax VJP must reproduce the SAME
    # weight grad as loss.backward() under reduction='sum'.
    w2 = w.detach().clone().requires_grad_(True)
    b2 = b.detach().clone().requires_grad_(True)
    y2 = F.linear(x, w2, b2)
    s2 = torch.softmax(y2, dim=-1)
    dLds2 = 2.0 * (s2.detach() - label)
    s2.backward(dLds2)
    assert torch.allclose(w2.grad, w.grad, atol=1e-9), (
        "fixture_e2e: raw-2(o-l) backward disagrees with reduction=sum autograd")

    return {"weight_grad": w.grad.detach().to(torch.float32)}


# ---------------------------------------------------------------------------
# Section 4: rank-3 fixture [2,2,3].
# ---------------------------------------------------------------------------


def fixture_rank3():
    x = torch.tensor([[[0.3, -1.1, 2.2], [0.8, -0.4, 1.5]],
                      [[-2.0, 0.6, 0.1], [1.9, -0.7, -1.3]]], dtype=torch.float64)
    dlds = torch.tensor([[[1.0, -2.0, 0.5], [3.0, -1.0, 2.0]],
                         [[-0.5, 4.0, 1.0], [2.0, 0.0, -3.0]]], dtype=torch.float64)

    s = softmax_rows(x)
    dx = jacobian_vjp_rows(s, dlds)
    check_against_autograd("fixture_rank3", x, dlds, s, dx)

    s_last_axis = torch.softmax(x, dim=-1)
    assert not torch.allclose(s, s_last_axis, atol=1e-3), (
        "fixture_rank3: per-row (axis 0) and last-axis softmax must differ (discriminator)")
    return {
        "x": x.to(torch.float32),
        "dlds": dlds.to(torch.float32),
        "s": s.to(torch.float32),
        "dx": dx.to(torch.float32),
    }


# ---------------------------------------------------------------------------
# Section 5: mixed-scale rows.
# ---------------------------------------------------------------------------


def fixture_mixed_scale():
    x = torch.tensor([[120.0, 119.0, 1.0], [-1.0, 0.0, 1.0]], dtype=torch.float64)
    s = softmax_rows(x)
    assert torch.allclose(s, torch_softmax_rows(x), atol=1e-12)

    # Discriminator: one SHARED max (120) makes every exp of row 1 underflow
    # in float32, so its sum is exactly 0 and the divide yields NaN.
    x32 = x.to(torch.float32)
    row1_shared = torch.exp(x32[1] - x32.max())
    assert row1_shared.sum().item() == 0.0, (
        "fixture_mixed_scale: row 1 must underflow under a shared max (discriminator)")
    return {"x": x.to(torch.float32), "s": s.to(torch.float32)}


# ---------------------------------------------------------------------------
# Sections 6/7: loop e2e at two rows (UnitTestMultiLayerTraining).
# ---------------------------------------------------------------------------

E2E_TWO_ROWS_X = [[1.0, -0.5, 2.0], [-1.5, 0.5, 0.25]]


def _e2e_linear_params():
    w0 = _ramp_2d(0.1, 0.05, 3, 3)  # float32, bit-matches the C ramp
    b0 = _ramp_2d(0.05, -0.1, 1, 3).flatten()  # float32, bit-matches the C ramp
    return w0.to(torch.float64), b0.to(torch.float64)


def fixture_e2e_ce_two_rows():
    w0, b0 = _e2e_linear_params()
    x = torch.tensor(E2E_TWO_ROWS_X, dtype=torch.float64)
    label = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)

    w = w0.clone().requires_grad_(True)
    b = b0.clone().requires_grad_(True)
    logits = F.linear(x, w, b)
    loss_sum = F.cross_entropy(logits, label, reduction="sum")
    loss_sum.backward()
    loss_mean = F.cross_entropy(logits.detach(), label, reduction="mean").item()

    # Self-check: the fused (p - y) chain the C loop runs (softmax per row,
    # CE backward = p - y, softmax layer skipped) must give the SAME grads.
    p = torch.softmax(logits.detach(), dim=-1)
    g = p - label
    assert torch.allclose(g.t() @ x, w.grad, atol=1e-12)
    assert torch.allclose(g.sum(dim=0), b.grad, atol=1e-12)
    assert abs(loss_mean - loss_sum.item() / 2.0) < 1e-12

    # Discriminator: a whole-tensor softmax over both rows moves the grads.
    p_whole = torch.softmax(logits.detach().flatten(), dim=0).reshape(2, 3)
    assert not torch.allclose((p_whole - label).t() @ x, w.grad, atol=1e-3)

    return {
        "x": x.to(torch.float32),
        "label": label.to(torch.float32),
        "loss_mean": loss_mean,
        "weight_grad": w.grad.detach().to(torch.float32),
        "bias_grad": b.grad.detach().to(torch.float32),
    }


def fixture_e2e_mse_two_rows():
    w0, b0 = _e2e_linear_params()
    x = torch.tensor(E2E_TWO_ROWS_X, dtype=torch.float64)
    label = torch.tensor([[0.2, 0.5, 0.3], [0.6, 0.1, 0.3]], dtype=torch.float64)

    w = w0.clone().requires_grad_(True)
    b = b0.clone().requires_grad_(True)
    s = torch.softmax(F.linear(x, w, b), dim=-1)
    loss_sum = F.mse_loss(s, label, reduction="sum")
    loss_sum.backward()
    loss_mean = F.mse_loss(s.detach(), label, reduction="mean").item()

    # Self-check: the raw 2(o-l) seed through the per-row softmax VJP.
    w2 = w0.clone().requires_grad_(True)
    b2 = b0.clone().requires_grad_(True)
    s2 = torch.softmax(F.linear(x, w2, b2), dim=-1)
    s2.backward(2.0 * (s2.detach() - label))
    assert torch.allclose(w2.grad, w.grad, atol=1e-12)
    assert torch.allclose(b2.grad, b.grad, atol=1e-12)

    return {
        "label": label.to(torch.float32),
        "loss_mean": loss_mean,
        "weight_grad": w.grad.detach().to(torch.float32),
        "bias_grad": b.grad.detach().to(torch.float32),
    }


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

    fwd_fx = fixture_forward()
    float_fx = fixture_float_backward()
    sym_fx = fixture_sym_backward(float_fx)
    e2e_fx = fixture_e2e_linear_softmax_mse()
    rank3_fx = fixture_rank3()
    mixed_fx = fixture_mixed_scale()
    ce2_fx = fixture_e2e_ce_two_rows()
    mse2_fx = fixture_e2e_mse_two_rows()

    parts.append(emit_float_array("softmaxForwardX", fwd_fx["x"]))
    parts.append(emit_float_array("softmaxForwardExpected", fwd_fx["s"]))
    parts.append(emit_float_array("softmaxForwardRank1Expected", fwd_fx["s_rank1"]))

    parts.append(emit_float_array("softmaxBackwardX", float_fx["x"]))
    parts.append(emit_float_array("softmaxBackwardDLds", float_fx["dlds"]))
    parts.append(emit_float_array("softmaxBackwardExpectedDx", float_fx["dx"]))
    parts.append(emit_float_array("softmaxBackwardRank1ExpectedDx", float_fx["dx_rank1"]))
    parts.append(emit_float_array("softmaxBackwardSymExpectedDx", sym_fx["dx"]))

    parts.append(emit_float_array("softmaxMseE2eExpectedWeightGrad", e2e_fx["weight_grad"]))

    parts.append(emit_float_array("softmaxRank3X", rank3_fx["x"]))
    parts.append(emit_float_array("softmaxRank3DLds", rank3_fx["dlds"]))
    parts.append(emit_float_array("softmaxRank3ExpectedS", rank3_fx["s"]))
    parts.append(emit_float_array("softmaxRank3ExpectedDx", rank3_fx["dx"]))

    parts.append(emit_float_array("softmaxMixedScaleX", mixed_fx["x"]))
    parts.append(emit_float_array("softmaxMixedScaleExpected", mixed_fx["s"]))

    parts.append(emit_float_array("softmaxTwoRowsE2eX", ce2_fx["x"]))
    parts.append(emit_float_array("softmaxCeTwoRowsE2eLabel", ce2_fx["label"]))
    parts.append(emit_float_scalar("softmaxCeTwoRowsE2eExpectedLossMean", ce2_fx["loss_mean"]))
    parts.append(emit_float_array("softmaxCeTwoRowsE2eExpectedWeightGrad", ce2_fx["weight_grad"]))
    parts.append(emit_float_array("softmaxCeTwoRowsE2eExpectedBiasGrad", ce2_fx["bias_grad"]))

    parts.append(emit_float_array("softmaxMseTwoRowsE2eLabel", mse2_fx["label"]))
    parts.append(emit_float_scalar("softmaxMseTwoRowsE2eExpectedLossMean", mse2_fx["loss_mean"]))
    parts.append(emit_float_array("softmaxMseTwoRowsE2eExpectedWeightGrad",
                                  mse2_fx["weight_grad"]))
    parts.append(emit_float_array("softmaxMseTwoRowsE2eExpectedBiasGrad", mse2_fx["bias_grad"]))

    parts.append("\n#endif // ODT_EXPECTED_SOFTMAX_H\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(parts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
