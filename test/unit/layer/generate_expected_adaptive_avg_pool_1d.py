#!/usr/bin/env python3
"""Generate expected_adaptive_avg_pool_1d.h for UnitTestAdaptiveAvgPool1d.

PyTorch torch.nn.functional.adaptive_avg_pool1d ground truth: forward output and
propLoss (dL/dx). Overlap/upsample fixtures use torch.randn_like(y) lossGrad (and
emit it as a gold array) so backward-path positional mutations are non-vacuous
(per codebase_uniform_lossgrad_mutation_vacuity). Divisor is the actual window
size; adaptive pooling has no padding.

Run via `uv run` (CMake wires this automatically).
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "goldgen"))
from sym_gold import (
    adaptiveavgpool1d_bfp_backward_ref,
    adaptiveavgpool1d_bfp_forward_ref,
    assert_rounding_canary,
    emit_float_scalar,
    emit_int32_array,
    emit_int32_scalar,
    emit_uint8_array,
    f32_scale_i12,
    requant_absmax_i12_f32,
    stable_dequant_i12,
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


def _hand_adaptive_forward(x, output_size):
    """floor/ceil reference (spec §2): start=floor(o*L/Lout), end=ceil((o+1)*L/Lout)."""
    B, C, L = x.shape
    out = torch.zeros((B, C, output_size))
    for b in range(B):
        for c in range(C):
            for o in range(output_size):
                start = (o * L) // output_size
                end = ((o + 1) * L + output_size - 1) // output_size
                out[b, c, o] = x[b, c, start:end].sum() / (end - start)
    return out


def _run_fixture(name, x, *, output_size, loss_grad_kind="ones", loss_seed=None):
    x_in = x.clone().detach().requires_grad_(True)
    y = F.adaptive_avg_pool1d(x_in, output_size)

    expected_y = _hand_adaptive_forward(x_in.detach(), output_size)
    assert torch.allclose(y.detach(), expected_y, atol=1e-5), (
        f"{name}: adaptive_avg_pool1d forward disagrees with hand-derived reference"
    )

    if loss_grad_kind == "ones":
        gy = torch.ones_like(y)
    elif loss_grad_kind == "randn":
        # Explicit literal seed (not hash(name), which is non-reproducible across
        # runs under PYTHONHASHSEED randomization) so the backward gold is stable
        # every regeneration — Task 3's backward tests depend on these arrays.
        if loss_seed is None:
            raise ValueError(f"{name}: randn lossGrad requires an explicit loss_seed")
        torch.manual_seed(loss_seed)
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


def fixture_basic():
    x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    return _run_fixture("basic", x, output_size=2)


def fixture_multi_channel():
    torch.manual_seed(301)
    x = torch.randn(1, 3, 5)
    return _run_fixture("multiChannel", x, output_size=2, loss_grad_kind="randn", loss_seed=401)


def fixture_multi_batch():
    torch.manual_seed(302)
    x = torch.randn(4, 2, 6)
    return _run_fixture("multiBatch", x, output_size=4, loss_grad_kind="randn", loss_seed=402)


def fixture_global():
    torch.manual_seed(303)
    x = torch.randn(1, 2, 7)
    return _run_fixture("global", x, output_size=1)


def fixture_identity():
    x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    return _run_fixture("identity", x, output_size=4)


def fixture_upsample():
    torch.manual_seed(305)
    x = torch.randn(1, 1, 3)
    return _run_fixture("upsample", x, output_size=5, loss_grad_kind="randn", loss_seed=403)


def emit_fixture(parts, fx):
    pre = f"adaptiveAvgPool1d_{fx['name']}"
    parts.append(emit_float_array(f"input_{pre}", fx["x"]))
    parts.append(emit_float_array(f"expectedForward_{pre}", fx["y"]))
    parts.append(emit_float_array(f"expectedPropLoss_{pre}", fx["dx"]))
    if fx["loss_grad_kind"] != "ones":
        parts.append(emit_float_array(f"lossGrad_{pre}", fx["gy"]))


# ---------------- SYM_INT32 fixtures (#205) ----------------
#
# Per-window element count varies, so the division cannot fold into ONE scale
# (unlike AvgPool1d). Decided mechanics: rounded INTEGER division of the
# mantissa sum (half-away-from-zero, roundByMode-consistent; standard
# fixed-point practice), scale unchanged — at most 0.5 LSB rounding error per
# element, and the quotient stays in operand range (|mean| <= |max|), so the
# raw wire is NOT accumulator-range here. The restore is the usual OUT_WRITE
# renormalization. C-vs-emulation gap: restore ULP only (mantissa +-1, scale
# rel 1e-4 — see generate_expected_max_pool_1d.py).
MANTISSA_TOL_SYM = 1
SCALE_REL_TOL_SYM = 1e-4


def _round_div_half_away(s: int, k: int) -> int:
    """Rounded integer division, half away from zero — mirrors the C kernel's
    (|s| + k/2) / k with truncating division (exact for k odd, tie-away for k
    even; no float involved)."""
    mag = abs(s)
    q = (mag + k // 2) // k
    return q if s >= 0 else -q


def _adaptive_window(L, O, o):
    start = (o * L) // O
    end = ((o + 1) * L + O - 1) // O
    return start, end - start


def _adaptive_sym_forward(xq, output_size, div):
    batch, channels, L = xq.shape
    y = torch.zeros((batch, channels, output_size), dtype=torch.int32)
    for b in range(batch):
        for c in range(channels):
            for o in range(output_size):
                start, count = _adaptive_window(L, output_size, o)
                acc = sum(int(xq[b, c, start + i]) for i in range(count))
                y[b, c, o] = div(acc, count)
    return y


def _run_adaptive_sym_fixture(name, x, *, output_size, loss_grad_kind="ones",
                              loss_seed=None):
    assert_rounding_canary()
    x = x.to(torch.float64)
    xq, _, x_deq = stable_dequant_i12(x)
    s_in = f32_scale_i12(x_deq)
    batch, channels, L = xq.shape

    y_raw = _adaptive_sym_forward(xq, output_size, _round_div_half_away)
    # Rounding-mutation canaries for task-7: record whether truncating or
    # half-even division would change THIS fixture's gold values anywhere.
    y_trunc = _adaptive_sym_forward(
        xq, output_size,
        lambda s, k: (abs(s) // k) if s >= 0 else -(abs(s) // k))
    y_even = _adaptive_sym_forward(
        xq, output_size,
        lambda s, k: int(round(s / k)))  # Python banker's rounding = half-even
    yq_r, s_y = requant_absmax_i12_f32(y_raw, s_in)

    # C-semantics reference: rounded quotient at the input scale. Sanity-check
    # it stays within 0.5 LSB of the true mean of the dequantized inputs.
    y_ref = y_raw.to(torch.float64) * s_in
    true_mean = torch.zeros_like(y_ref)
    for b in range(batch):
        for c in range(channels):
            for o in range(output_size):
                start, count = _adaptive_window(L, output_size, o)
                true_mean[b, c, o] = x_deq.to(torch.float64)[b, c, start:start + count].mean()
    # Relative epsilon: an exact .5-tie sits ON the 0.5 LSB bound, and the
    # float32 rounding of the emitted fixture values (2^-24 rel of values up
    # to qMax LSBs ~ 2.4e-4 of the half-LSB) must not tip it over. 1e-3 keeps
    # this a real sanity check — a wrong divisor overshoots by orders of
    # magnitude.
    assert (y_ref - true_mean).abs().max().item() <= 0.5 * s_in * (1.0 + 1e-3), \
        f"{name}: rounded-div forward leaves the 0.5 LSB band"

    y_dequant_tol = (MANTISSA_TOL_SYM + 0.5) * s_y * 1.5
    err = (yq_r.to(torch.float64) * s_y - y_ref).abs().max().item()
    assert err <= y_dequant_tol, f"{name}: restored fwd dequant {err} > {y_dequant_tol}"

    if loss_grad_kind == "ones":
        gy = torch.ones((batch, channels, output_size), dtype=torch.float64)
    elif loss_grad_kind == "randn":
        if loss_seed is None:
            raise ValueError(f"{name}: randn lossGrad requires an explicit loss_seed")
        torch.manual_seed(loss_seed)
        gy = torch.randn(batch, channels, output_size, dtype=torch.float64)
    else:
        raise ValueError(loss_grad_kind)
    gyq, _, gy_deq = stable_dequant_i12(gy)
    s_gy = f32_scale_i12(gy_deq)

    # Mirror the backward kernel: per-window rounded quotient of the loss-grad
    # mantissa, scattered += into every window member; scale unchanged.
    dx_raw = torch.zeros_like(xq)
    for b in range(batch):
        for c in range(channels):
            for o in range(output_size):
                start, count = _adaptive_window(L, output_size, o)
                contribution = _round_div_half_away(int(gyq[b, c, o]), count)
                for i in range(count):
                    dx_raw[b, c, start + i] += contribution
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
        "gy": gy_deq,
        "dx_mantissas": dxq_r, "dx_scale": s_dx,
        "dx_dequant": dx_ref.to(torch.float32), "dx_dequant_tol": dx_dequant_tol,
        "_diff_trunc": not torch.equal(y_raw, y_trunc),
        "_diff_even": not torch.equal(y_raw, y_even),
    }


def fixture_sym_uneven():
    # L=5 -> O=3: window counts 2/3/2 with overlaps at indices 1 and 3 — the
    # core adaptive case (varying divisor + overlapping backward scatter).
    torch.manual_seed(207)
    x = torch.randn(1, 1, 5)
    return _run_adaptive_sym_fixture("symUneven", x, output_size=3,
                                     loss_grad_kind="randn", loss_seed=404)


def fixture_sym_global():
    # L=6 -> O=1: global average, the largest divisor (count = L).
    torch.manual_seed(208)
    x = torch.randn(1, 2, 6)
    return _run_adaptive_sym_fixture("symGlobal", x, output_size=1)


def emit_sym_fixture(parts, fx):
    pre = f"adaptiveAvgPool1dSym_{fx['name']}"
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


# ---- BFP epic PR4 (R-P4): AdaptiveAvgPool1d ARITH_BFP fwd + dx -----------
# L = 8, O = 3 -> window counts 3 / 4 / 3 (UNEVEN, so the per-window divide
# cannot be mistaken for a constant K) and windows {0,1,2} {2,3,4,5}
# {5,6,7} overlap at 2 and 5 (so the dx memset + accumulating scatter is
# load-bearing). Input grid {4 groups x 4} over 16 elements -> window
# {2,3,4,5} crosses a group boundary; channel 1 lives in groups 2/3, so a
# kernel that looks the group up by the WITHIN-CHANNEL index binds the wrong
# exponents.
BFP_AD_BATCH = 1
BFP_AD_CHANNELS = 2
BFP_AD_INPUT_LENGTH = 8
BFP_AD_OUTPUT_SIZE = 3
BFP_AD_MANTISSA_BITS = 6
BFP_AD_EXPONENT_BITS = 8
BFP_AD_IN_NUM_GROUPS = 4
BFP_AD_IN_GROUP_SIZE = 4
BFP_AD_IN_CODES = [12, -7, 0, 31, -32, 5, -1, 20,
                   3, -14, 22, -1, 8, 0, -30, 6]
BFP_AD_IN_EXPS = [130, 127, 133, 125]
BFP_AD_GY_NUM_GROUPS = 2
BFP_AD_GY_GROUP_SIZE = 3
BFP_AD_GY_CODES = [9, -21, 4, 30, -12, 7]
BFP_AD_GY_EXPS = [129, 124]


def emit_bfp_fixtures(parts):
    # The C tests build their wires from these, so a fixture-shape change
    # propagates instead of drifting away from hardcoded literals.
    parts.append(emit_int32_scalar("kBfpAdaptiveMantissaBits", BFP_AD_MANTISSA_BITS))
    parts.append(emit_int32_scalar("kBfpAdaptiveExponentBits", BFP_AD_EXPONENT_BITS))
    parts.append(emit_int32_scalar("kBfpAdaptiveInNumGroups", BFP_AD_IN_NUM_GROUPS))
    parts.append(emit_int32_scalar("kBfpAdaptiveInGroupSize", BFP_AD_IN_GROUP_SIZE))
    parts.append(emit_int32_scalar("kBfpAdaptiveGyNumGroups", BFP_AD_GY_NUM_GROUPS))
    parts.append(emit_int32_scalar("kBfpAdaptiveGyGroupSize", BFP_AD_GY_GROUP_SIZE))
    parts.append(emit_int32_scalar("kBfpAdaptiveOutputSize", BFP_AD_OUTPUT_SIZE))

    in_qc = {"mantissa_bits": BFP_AD_MANTISSA_BITS, "exponent_bits": BFP_AD_EXPONENT_BITS,
             "group_size": BFP_AD_IN_GROUP_SIZE}
    fwd = adaptiveavgpool1d_bfp_forward_ref(BFP_AD_IN_CODES, BFP_AD_IN_EXPS, in_qc,
                                            BFP_AD_BATCH, BFP_AD_CHANNELS,
                                            BFP_AD_INPUT_LENGTH, BFP_AD_OUTPUT_SIZE)
    parts.append(emit_int32_array("kBfpAdaptiveInCodes", torch.tensor(BFP_AD_IN_CODES)))
    parts.append(emit_uint8_array("kBfpAdaptiveInExps", BFP_AD_IN_EXPS))
    parts.append(emit_float_array("kBfpAdaptiveExpectedForward",
                                  torch.tensor(fwd, dtype=torch.float32)))

    gy_qc = {"mantissa_bits": BFP_AD_MANTISSA_BITS, "exponent_bits": BFP_AD_EXPONENT_BITS,
             "group_size": BFP_AD_GY_GROUP_SIZE}
    bwd = adaptiveavgpool1d_bfp_backward_ref(BFP_AD_GY_CODES, BFP_AD_GY_EXPS, gy_qc,
                                             BFP_AD_BATCH, BFP_AD_CHANNELS,
                                             BFP_AD_INPUT_LENGTH, BFP_AD_OUTPUT_SIZE)
    parts.append(emit_int32_array("kBfpAdaptiveGyCodes", torch.tensor(BFP_AD_GY_CODES)))
    parts.append(emit_uint8_array("kBfpAdaptiveGyExps", BFP_AD_GY_EXPS))
    parts.append(emit_float_array("kBfpAdaptiveExpectedBackward",
                                  torch.tensor(bwd, dtype=torch.float32)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    parts = [
        "// AUTOGENERATED by generate_expected_adaptive_avg_pool_1d.py — DO NOT EDIT\n",
        "#ifndef ODT_EXPECTED_ADAPTIVE_AVG_POOL_1D_H\n",
        "#define ODT_EXPECTED_ADAPTIVE_AVG_POOL_1D_H\n",
        "#include <stdlib.h>\n\n",
    ]
    for fx in [
        fixture_basic(),
        fixture_multi_channel(),
        fixture_multi_batch(),
        fixture_global(),
        fixture_identity(),
        fixture_upsample(),
    ]:
        emit_fixture(parts, fx)

    sym_fixtures = [fixture_sym_uneven(), fixture_sym_global()]
    # Divisor-rounding mutation canary: somewhere in the SYM fixture set the
    # half-away gold values must differ from BOTH truncating and half-even
    # division, else a rounding mutation in the C kernel would be untestable.
    assert any(fx["_diff_trunc"] for fx in sym_fixtures), \
        "no SYM fixture distinguishes half-away from truncating division"
    assert any(fx["_diff_even"] for fx in sym_fixtures), \
        "no SYM fixture distinguishes half-away from half-even division"
    for fx in sym_fixtures:
        emit_sym_fixture(parts, fx)

    emit_bfp_fixtures(parts)

    parts.append("\n#endif // ODT_EXPECTED_ADAPTIVE_AVG_POOL_1D_H\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(parts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
