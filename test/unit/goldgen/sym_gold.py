#!/usr/bin/env python3
"""Shared SYM_INT32 gold-value helpers for unit-test gold generators (spec D7, #192).

Extracted verbatim from test/unit/layer/generate_expected_layernorm_sym_bwd.py
(the LayerNorm generators keep their private copies for now and migrate
opportunistically — do NOT import this module from them yet).

Rounding: the framework's roundByMode(HALF_AWAY) is C round() =
half-away-from-zero — emulate with sign(x)*floor(|x|+0.5), NEVER torch.round
(true half-to-even, silently diverges on ties).

Generators import via a sys.path bootstrap relative to the script:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "goldgen"))
"""
import torch

QMAX = 32767.0
QMIN = -32768.0


def round_half_away(x: torch.Tensor) -> torch.Tensor:
    """Match the C kernel: roundByMode(HALF_AWAY) is C round() = half-away-from-zero
    (Rounding.c)."""
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
