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
import math

import numpy as np
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


def emit_uint8_array(name: str, values) -> str:
    vals = [int(v) for v in values]
    assert all(0 <= v <= 255 for v in vals), f"{name}: value outside uint8 range"
    body = ", ".join(str(v) for v in vals)
    return (
        f"static const uint8_t {name}[] = {{ {body} }};\n"
        f"static const size_t {name}_len = {len(vals)};\n"
    )


def check_exact_roundtrip(name, values, codes, exps, qc):
    """Exact-float-regime pin: code * 2^(stored - bias) must reproduce the
    input float bit-for-bit (float32 multiply by a power of two is exact)."""
    bias = 2 ** (qc["exponent_bits"] - 1) - 1
    gsz = len(values) if qc["group_size"] == 0 else qc["group_size"]
    for i, v in enumerate(values):
        scale = np.float32(np.ldexp(np.float32(1.0), np.int32(exps[i // gsz] - bias)))
        deq = float(np.float32(np.float32(codes[i]) * scale))
        assert deq == v, (
            f"{name}: element {i} dequantizes to {deq}, not {v} -- fixture left "
            "the exact float regime; pick grid-exact values")


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


# ---- Group-quant PR2: per-group (storage-order) symmetric quantization,
# mirroring packFloatBufferAsSym's grouped path (group of element i =
# i // group_size, groups are group_size consecutive elements). ----


def quantize_sym_grouped(values, q_bits: int, group_size: int):
    """Per-group absmax symmetric quantization, storage-order groups.
    Mirrors packFloatBufferAsSym's grouped path: absMax_g -> scale_g =
    absMax_g/qMax (1.0 if absMax_g == 0), round-half-away, per group.
    `values` is any array-like (list or tensor); returns (codes, scales) as
    plain Python lists (codes: int, scales: float)."""
    x = torch.as_tensor(values, dtype=torch.float64).flatten()
    n = x.numel()
    assert n % group_size == 0, (
        f"quantize_sym_grouped: n={n} not divisible by group_size={group_size}")
    q_max = 2.0 ** (q_bits - 1) - 1
    codes, scales = [], []
    for g0 in range(0, n, group_size):
        grp = x[g0:g0 + group_size]
        abs_max = grp.abs().max().item()
        scale = 1.0 if abs_max == 0.0 else abs_max / q_max
        scales.append(scale)
        q = round_half_away(grp / scale)
        codes.extend(int(v) for v in q.tolist())
    assert all(abs(c) <= q_max for c in codes), (
        "quantize_sym_grouped: code out of range")
    return codes, scales


def stable_dequant_grouped(values, q_bits: int, group_size: int):
    """Grouped variant of stable_dequant: quantize per-group once, return
    (codes, scales, dequantized float64 list) with a round-trip stability
    assertion so the C side's tensorFillFromFloatBuffer lands on exactly
    these codes under the SAME per-group grid it was quantized from."""
    codes, scales = quantize_sym_grouped(values, q_bits, group_size)
    n = len(codes)
    deq = [0.0] * n
    for gi, g0 in enumerate(range(0, n, group_size)):
        for i in range(group_size):
            deq[g0 + i] = float(codes[g0 + i]) * scales[gi]
    codes2, scales2 = quantize_sym_grouped(deq, q_bits, group_size)
    assert codes2 == codes, "grouped fixture is not dequantization round-trip stable (codes)"
    assert scales2 == scales, (
        "grouped fixture is not dequantization round-trip stable (scales)")
    return codes, scales, deq


# ---- Group-quant PR2 Task 3: grouped-weight GEMM reference (GGUF rescale-
# combine pattern), matmulSymInt32TensorsGroupedWeight's kernel emulation. ----


def rescale_f32(param_q: int, param_scale: float, accumulator_scale: float) -> int:
    """Mirrors rescaleIntoAccumulatorScale(..., HALF_AWAY) (Rounding.c):
    float rescaled = (float)paramQ * paramScale / accumulatorScale;
    return roundByMode(rescaled, HALF_AWAY); -- every intermediate stays in
    float32 (torch.float32 tensors), matching the C `float` arithmetic
    left-to-right (multiply THEN divide), not a float64 shortcut."""
    pq = torch.tensor(float(param_q), dtype=torch.float32)
    ps = torch.tensor(param_scale, dtype=torch.float32)
    acc_s = torch.tensor(accumulator_scale, dtype=torch.float32)
    rescaled = pq * ps / acc_s
    return int(round_half_away(rescaled).item())


def combine_quotient_f32(param_q: int, param_scale: float, accumulator_scale: float) -> float:
    """The float32 quotient BEFORE rounding (rescale_f32's `rescaled`) -- used
    by generators to self-check that a fixture actually exercises a rounding
    decision (|frac| >= 0.5, where round-half-away and truncate-toward-zero
    diverge), not just to compute the final int."""
    pq = torch.tensor(float(param_q), dtype=torch.float32)
    ps = torch.tensor(param_scale, dtype=torch.float32)
    acc_s = torch.tensor(accumulator_scale, dtype=torch.float32)
    return (pq * ps / acc_s).item()


def matmul_grouped_ref(a_mantissas, a_scale: float, w_mantissas, w_scales, group_size: int,
                       out_rows: int, out_cols: int, reduce_len: int,
                       bias_mantissas=None, bias_scale=None, bias_rounding_mode="HALF_AWAY"):
    """Emulates matmulSymInt32TensorsGroupedWeight/matmulIntCoreGrouped exactly:
    python-int MACs per group (exact, arbitrary precision), a rescale-combine
    at every group boundary AND at the end of the reduction (`rescale_f32`),
    bias seeded via the same rescale primitive. `w_mantissas` is `b`'s
    STORAGE-order flat array (row `c` of length `reduce_len` = output channel
    c's weights, contiguous -- the GEMM-weight wiring Linear.c always
    produces via transposeTensor(w,0,1)); `w_idx = c * reduce_len + k` is
    therefore the SAME physical index matmulIntCoreGrouped derives via
    calcElementIndexByIndices. `a_mantissas` is `a`'s row-major [out_rows,
    reduce_len] flat array. Only HALF_AWAY is emulated (bias_rounding_mode is
    accepted for signature symmetry with the C bias seed's own roundingMode
    field but must be "HALF_AWAY" -- SR_HALF_AWAY needs the C RNG stream and
    is exercised in C directly, see testMatmulGroupedHonorsOpRoundingMode).
    Returns (out_mantissas flat list [out_rows*out_cols], s_acc)."""
    assert bias_rounding_mode == "HALF_AWAY", "matmul_grouped_ref only emulates HALF_AWAY"
    max_scale = max(w_scales)
    s_acc = (torch.tensor(a_scale, dtype=torch.float32) *
            torch.tensor(max_scale, dtype=torch.float32)).item()

    out = []
    for r in range(out_rows):
        for c in range(out_cols):
            acc = 0
            if bias_mantissas is not None:
                acc = rescale_f32(bias_mantissas[c], bias_scale, s_acc)
            partial = 0
            current_group = None
            for k in range(reduce_len):
                w_idx = c * reduce_len + k
                g = w_idx // group_size
                if g != current_group:
                    if current_group is not None:
                        param_scale = (torch.tensor(a_scale, dtype=torch.float32) *
                                      torch.tensor(w_scales[current_group],
                                                   dtype=torch.float32)).item()
                        acc += rescale_f32(partial, param_scale, s_acc)
                    partial = 0
                    current_group = g
                a_val = a_mantissas[r * reduce_len + k]
                w_val = w_mantissas[w_idx]
                partial += a_val * w_val
            param_scale = (torch.tensor(a_scale, dtype=torch.float32) *
                          torch.tensor(w_scales[current_group], dtype=torch.float32)).item()
            acc += rescale_f32(partial, param_scale, s_acc)
            out.append(acc)
    return out, s_acc


def matmul_grouped_dx_ref(a_mantissas, a_scale: float, w_mantissas, w_scales, group_size: int,
                          out_rows: int, out_cols: int, reduce_len: int):
    """Group-quant PR3 Task 1: emulates matmulSymInt32TensorsGroupedWeight on
    the Linear dx (propLoss) orientation: out[r][k] = sum_o a[r][o] * W[o][k]
    with `a` the loss ([out_rows(=batch), reduce_len(=outFeatures)] row-major
    flat) and W passed in its RAW [reduce_len(=outFeatures),
    out_cols(=inFeatures)] storage order -- Linear dx hands the weight over
    UNtransposed, so the visited weight storage index at reduction step o for
    output column k is `w_idx = o * out_cols + k`: STRIDED by out_cols, not
    contiguous like matmul_grouped_ref's forward orientation. Groups still
    bind to flat storage (`g = w_idx // group_size`), so consecutive
    reduction steps hop groups; the emulation folds the running int partial
    via rescale_f32 EVERY time the visited element's group changes (for
    per-channel weights that is EVERY step -- one rescale per term) plus the
    final tail combine: the exact sequence the unified per-element C core
    (matmulIntCoreGrouped) produces. dx has no bias operand. Only HALF_AWAY
    is emulated (see matmul_grouped_ref's docstring for why SR is excluded).
    Returns (out_mantissas flat list [out_rows*out_cols], s_acc)."""
    max_scale = max(w_scales)
    s_acc = (torch.tensor(a_scale, dtype=torch.float32) *
            torch.tensor(max_scale, dtype=torch.float32)).item()

    out = []
    for r in range(out_rows):
        for k in range(out_cols):
            acc = 0
            partial = 0
            current_group = None
            for o in range(reduce_len):
                w_idx = o * out_cols + k
                g = w_idx // group_size
                if g != current_group:
                    if current_group is not None:
                        param_scale = (torch.tensor(a_scale, dtype=torch.float32) *
                                      torch.tensor(w_scales[current_group],
                                                   dtype=torch.float32)).item()
                        acc += rescale_f32(partial, param_scale, s_acc)
                    partial = 0
                    current_group = g
                a_val = a_mantissas[r * reduce_len + o]
                w_val = w_mantissas[w_idx]
                partial += a_val * w_val
            param_scale = (torch.tensor(a_scale, dtype=torch.float32) *
                          torch.tensor(w_scales[current_group], dtype=torch.float32)).item()
            acc += rescale_f32(partial, param_scale, s_acc)
            out.append(acc)
    return out, s_acc


# ---- int12 operand helpers (#227 operand flip; default quantizationInitSymInt32
# tensors carry qMaxBits = ODT_SYM_OPERAND_QMAXBITS = 12). Additive: existing
# int16-era users above keep their semantics; pool/conv generators use these. ----

QMAX_I12 = 2047.0
QMIN_I12 = -2048.0


def quantize_sym_i12(x: torch.Tensor):
    """convertFloatTensorToSymInt32Tensor into an int12 config (qMaxBits=12):
    absmax -> scale=absmax/2047 (1.0 if absmax==0), round-clamp half-away."""
    absmax = x.abs().max().item()
    scale = 1.0 if absmax == 0.0 else absmax / QMAX_I12
    q = round_half_away(torch.clamp(x / scale, QMIN_I12, QMAX_I12))
    return q.to(torch.int32), scale


def stable_dequant_i12(x: torch.Tensor):
    """int12 variant of stable_dequant: quantize once, assert the emitted
    float32 fixture is dequantization round-trip stable."""
    q, s = quantize_sym_i12(x.to(torch.float64))
    deq32 = (q.to(torch.float64) * s).to(torch.float32)
    q2, _ = quantize_sym_i12(deq32.to(torch.float64))
    assert torch.equal(q, q2), "int12 fixture not dequant round-trip stable"
    return q, s, deq32


def f32_scale_i12(deq32: torch.Tensor) -> float:
    """int12 scale the C runtime derives from the EMITTED float32 fixture:
    float32(absmax)/2047 computed in float32 (convertFloatTensorToSymInt32Tensor)."""
    absmax = deq32.abs().max().to(torch.float32)
    if absmax.item() == 0.0:
        return 1.0
    return (absmax / torch.tensor(QMAX_I12, dtype=torch.float32)).item()


def requant_absmax_i12_f32(mantissas: torch.Tensor, in_scale: float):
    """executeOp OUT_WRITE epilogue for a SYM_INT32 target: the conversionMatrix
    diagonal (requantSymInt32Tensor, TensorConversion.c) at the target's declared
    qMaxBits=12. Emulates the same float32 sequence: dequant (f32) -> absmax (f32)
    -> scale=absmax/2047 (f32) -> round_half_away(clamp(...)). Returns
    (restored int32 mantissas, restored scale)."""
    deq = mantissas.to(torch.float32) * torch.tensor(in_scale, dtype=torch.float32)
    absmax = deq.abs().max().to(torch.float32)
    if absmax.item() == 0.0:
        return torch.zeros_like(mantissas, dtype=torch.int32), 1.0
    scale = absmax / torch.tensor(QMAX_I12, dtype=torch.float32)
    q = round_half_away(torch.clamp(deq / scale, QMIN_I12, QMAX_I12))
    return q.to(torch.int32), scale.item()


def assert_rounding_canary():
    """Fires if round_half_away is ever a half-even rounder — per-fixture
    tolerances cannot catch this on non-tie fixtures (Conv1d goldgen precedent)."""
    c = round_half_away(torch.tensor([0.5, -0.5], dtype=torch.float32))
    assert c.tolist() == [1.0, -1.0], \
        "round_half_away is not half-away-from-zero — rounding mode wrong"


# ---- 1D sliding-window geometry emulation (SlidingWindow1d.c) for pool/conv
# generators that emulate C kernels in the mantissa domain. All divisions below
# operate on non-negative operands, so Python floor-division matches C
# truncation exactly. ----


def window_geometry_1d(input_length, kernel_size, stride, dilation, padding_type,
                       padding=0):
    """Mirror windowGeometry1dCalc: returns dict with pad_left and out_len."""
    eff_k = dilation * (kernel_size - 1) + 1
    if padding_type == "VALID":
        pad_left = 0
        out_len = (input_length - eff_k) // stride + 1 if input_length >= eff_k else 0
    elif padding_type == "SAME":
        out_len = (input_length + stride - 1) // stride
        needed = eff_k + (out_len - 1) * stride
        total = needed - input_length if needed > input_length else 0
        pad_left = total // 2
    elif padding_type == "EXPLICIT":
        pad_left = padding
        padded = input_length + 2 * padding
        out_len = (padded - eff_k) // stride + 1 if padded >= eff_k else 0
    else:
        raise ValueError(padding_type)
    return dict(input_length=input_length, kernel_size=kernel_size, stride=stride,
                dilation=dilation, pad_left=pad_left, out_len=out_len)


def window_slice_1d(geom, out_pos):
    """Mirror windowSlice1dAt: returns (first_valid_input_idx, valid_count)."""
    input_start = out_pos * geom["stride"] - geom["pad_left"]
    length = geom["input_length"]
    if input_start >= length:
        return 0, 0
    d = geom["dilation"]
    first_k = 0 if input_start >= 0 else (-input_start + d - 1) // d
    last_k = geom["kernel_size"] - 1
    max_k = (length - 1 - input_start) // d
    if max_k < last_k:
        last_k = max_k
    if first_k > last_k:
        return 0, 0
    return input_start + first_k * d, last_k - first_k + 1


def window_slice_1d_full(geom, out_pos):
    """Full windowSlice1dAt mirror: returns (first_valid_input_idx,
    first_valid_kernel_offset, valid_count) -- the same three fields as the C
    windowSlice1d_t struct. window_slice_1d (above) only returns the first two
    of these collapsed into one value, which is enough for pool/conv
    generators that only ever read AT the returned input index; a grouped-
    weight reduction needs the actual kernel tap index (firstValidKernelOffset
    + i) to bind each visited weight element to its STORAGE index, so this
    variant exposes it directly instead of assuming it is always 0 (true only
    for VALID padding, where every window is fully in-bounds)."""
    input_start = out_pos * geom["stride"] - geom["pad_left"]
    length = geom["input_length"]
    if input_start >= length:
        return 0, 0, 0
    d = geom["dilation"]
    first_k = 0 if input_start >= 0 else (-input_start + d - 1) // d
    last_k = geom["kernel_size"] - 1
    max_k = (length - 1 - input_start) // d
    if max_k < last_k:
        last_k = max_k
    if first_k > last_k:
        return 0, 0, 0
    return input_start + first_k * d, first_k, last_k - first_k + 1


# ---- Group-quant PR2 Task 4: grouped-weight Conv1d gather-core reference
# (GGUF rescale-combine pattern), conv1dKernelSymInt32Grouped's kernel
# emulation. Mirrors matmul_grouped_ref's running-partial idiom exactly, but
# walks the (icOffset, kernelIdx) reduction via sliding-window geometry
# instead of a flat dot product. ----


def conv1d_grouped_ref(x_mantissas, in_scale: float, w_mantissas, w_scales, group_size: int,
                       batch: int, in_channels: int, out_channels: int, kernel_size: int,
                       input_length: int, stride: int = 1, dilation: int = 1,
                       padding_type: str = "VALID", padding: int = 0,
                       bias_mantissas=None, bias_scale=None, conv_groups: int = 1):
    """Emulates conv1dKernelSymInt32Grouped exactly: python-int MACs per group
    (exact), a rescale-combine (rescale_f32, HALF_AWAY) at every group
    boundary AND at the end of EACH (batch, outChannel, outPos) reduction.
    `w_mantissas` is [out_channels, in_channels/conv_groups, kernel_size]
    row-major flat (weight storage index for (oc, icOffset, k) is
    (oc*inChPerGroup + icOffset)*kernel_size + k -- the SAME index
    conv1dKernelSymInt32Grouped computes for its weight reads, since the
    (icOffset, kernelIdx) nested-loop order visits it monotonically
    increasing). `x_mantissas` is [batch, in_channels, input_length]
    row-major flat.

    PR3 Task 3 extensions (closing PR2's two disclosed emulation gaps):
    - Partial windows (SAME/EXPLICIT padding, or VALID with a too-short tail)
      are emulated exactly like the C kernel: the (first_valid_k, valid_count)
      walk from window_slice_1d_full skips out-of-bounds taps, so visited
      weight storage indices can have GAPS -- the per-element group division
      below (g = w_idx // group_size, never a precomputed run) binds each
      visited element to its true group across those gaps, mirroring the C
      kernel's gap-robust per-element division.
    - conv_groups > 1: channel grouping (the kernel's `groups` param),
      INDEPENDENT of the quantization groups. oc's conv group is
      oc // (out_channels/conv_groups); its reduction covers icOffset in
      [0, in_channels/conv_groups) at actual input channel
      convG*inChPerGroup + icOffset -- iteration (global oc ascending) visits
      the same (b, oc, outPos) cells in the same flat order as the C kernel's
      (g, ocOffset) nesting, and each cell's reduction order is identical.
    Returns (out_mantissas flat [batch*out_channels*out_len], s_acc, out_len).
    """
    assert in_channels % conv_groups == 0 and out_channels % conv_groups == 0, (
        "conv1d_grouped_ref: conv_groups must divide in_channels and out_channels")
    in_ch_per_group = in_channels // conv_groups
    out_ch_per_group = out_channels // conv_groups
    geom = window_geometry_1d(input_length, kernel_size, stride, dilation, padding_type, padding)
    out_len = geom["out_len"]
    max_scale = max(w_scales)
    s_acc = (torch.tensor(in_scale, dtype=torch.float32) *
            torch.tensor(max_scale, dtype=torch.float32)).item()

    out = []
    for b in range(batch):
        for oc in range(out_channels):
            conv_g = oc // out_ch_per_group
            in_lo = conv_g * in_ch_per_group
            w_base = oc * in_ch_per_group * kernel_size
            seed = 0
            if bias_mantissas is not None:
                seed = rescale_f32(bias_mantissas[oc], bias_scale, s_acc)
            for out_pos in range(out_len):
                first_valid_idx, first_valid_k, valid_count = window_slice_1d_full(geom, out_pos)
                acc = seed
                partial = 0
                current_group = None
                for ic_offset in range(in_ch_per_group):
                    ic = in_lo + ic_offset
                    for i in range(valid_count):
                        kernel_idx = first_valid_k + i
                        w_idx = w_base + ic_offset * kernel_size + kernel_idx
                        g = w_idx // group_size
                        if g != current_group:
                            if current_group is not None:
                                param_scale = (torch.tensor(in_scale, dtype=torch.float32) *
                                              torch.tensor(w_scales[current_group],
                                                           dtype=torch.float32)).item()
                                acc += rescale_f32(partial, param_scale, s_acc)
                            partial = 0
                            current_group = g
                        input_idx = first_valid_idx + i * dilation
                        x_val = x_mantissas[(b * in_channels + ic) * input_length + input_idx]
                        w_val = w_mantissas[w_idx]
                        partial += x_val * w_val
                if current_group is not None:
                    param_scale = (torch.tensor(in_scale, dtype=torch.float32) *
                                  torch.tensor(w_scales[current_group],
                                               dtype=torch.float32)).item()
                    acc += rescale_f32(partial, param_scale, s_acc)
                out.append(acc)
    return out, s_acc, out_len


# ---- Group-quant PR3 Task 2: grouped-weight ConvT1d SCATTER-core reference
# (per-PRODUCT rescale-combine), convTranspose1dKernelSymInt32Grouped's kernel
# emulation. Deliberately NOT the running-partial idiom of matmul_grouped_ref/
# conv1d_grouped_ref: a scatter's consecutive products land in DIFFERENT
# output elements (outIdx = inPos*stride + k*dilation moves with k), so there
# is no per-(target, group) run across which a raw int partial could be
# carried -- each product is folded into the accumulator scale immediately.
# Error consequence: every contributing product rounds independently (<= 0.5
# quanta of s_acc each), so an output element that C products scatter into
# carries |err| <= 0.5*C*s_acc worst case vs exact arithmetic. ----


def convT1d_grouped_ref(x_mantissas, in_scale: float, w_mantissas, w_scales, group_size: int,
                        batch: int, in_channels: int, out_channels: int, kernel_size: int,
                        input_length: int, stride: int = 1, dilation: int = 1,
                        output_padding: int = 0, bias_mantissas=None, bias_scale=None):
    """Emulates convTranspose1dKernelSymInt32Grouped exactly: python-int
    products (exact), ONE rescale_f32 (HALF_AWAY) per product into s_acc =
    float32(in_scale) * float32(max(w_scales)), accumulated (exact int adds)
    into the scattered output element; bias AFTER the scatter, per the C pass
    order (rescale_f32(bias[oc], bias_scale, s_acc) added to every (b, oc, l)).

    VALID-only geometry (Phase-1 ConvT1d forward contract, and the only
    padding these fixtures use): pad_left = 0, out_len = (input_length-1)*
    stride + dilation*(kernel_size-1) + output_padding + 1.

    `w_mantissas` is ConvT1d's [in_channels, out_channels, kernel_size]
    row-major flat storage (conv-groups == 1 here -- quantization groups
    only, mirroring conv1d_grouped_ref's scope): the weight index for
    (ic, oc, k) is (ic*out_channels + oc)*kernel_size + k -- the SAME flat
    index convTranspose1dKernelSymInt32Grouped reads at (:235's wArr[...]),
    so the per-product group is g = w_idx // group_size directly.

    NOTE on "per-channel" in THIS layout: a contiguous storage group only
    ever spans consecutive flat indices, and ConvT1d storage interleaves
    output channels INSIDE each input-channel slab -- so groupSize =
    out_channels*kernel_size ("per-channel" fixture) means one group per
    INPUT channel (one ic-slab), NOT per output channel. A per-OUTPUT-channel
    grouping is not expressible as contiguous groups in this layout at all.

    `x_mantissas` is [batch, in_channels, input_length] row-major flat.
    Returns (out flat [batch*out_channels*out_len], s_acc, out_len)."""
    out_len = (input_length - 1) * stride + dilation * (kernel_size - 1) + output_padding + 1
    max_scale = max(w_scales)
    s_acc = (torch.tensor(in_scale, dtype=torch.float32) *
            torch.tensor(max_scale, dtype=torch.float32)).item()
    # Per-group product scale, float32-mirrored exactly like the C kernel's
    # inScale * weightGroups->scales[g] (both float32 operands, float32 mul).
    param_scales = [(torch.tensor(in_scale, dtype=torch.float32) *
                    torch.tensor(s, dtype=torch.float32)).item() for s in w_scales]

    out = [0] * (batch * out_channels * out_len)
    for b in range(batch):
        for ic in range(in_channels):
            for in_pos in range(input_length):
                x_val = x_mantissas[(b * in_channels + ic) * input_length + in_pos]
                for oc in range(out_channels):
                    for k in range(kernel_size):
                        out_idx = in_pos * stride + k * dilation
                        w_idx = (ic * out_channels + oc) * kernel_size + k
                        g = w_idx // group_size
                        out[(b * out_channels + oc) * out_len + out_idx] += rescale_f32(
                            x_val * w_mantissas[w_idx], param_scales[g], s_acc)

    if bias_mantissas is not None:
        for oc in range(out_channels):
            seed = rescale_f32(bias_mantissas[oc], bias_scale, s_acc)
            for b in range(batch):
                for l in range(out_len):
                    out[(b * out_channels + oc) * out_len + l] += seed
    return out, s_acc, out_len


# ---- Group-quant PR3 Task 3: dx-orientation references for the two conv
# adjoints. Both are pure PARAMETER REMAPPINGS of the forward emulations above
# -- the flat weight STORAGE index each core computes in the adjoint role is
# identical to the layer's own storage index, so the group binding
# (g = w_idx // group_size) carries over unchanged. VALID-only (the dx
# fixtures' scope; the C kernels additionally handle the SAME/EXPLICIT
# adjoint, exercised by the FORWARD SAME fixture instead). ----


def conv1d_dx_grouped_ref(loss_mantissas, loss_scale: float, w_mantissas, w_scales,
                          group_size: int, batch: int, in_channels: int, out_channels: int,
                          kernel_size: int, input_length: int, stride: int = 1,
                          dilation: int = 1):
    """Conv1d dx (propLoss) with a grouped weight: the adjoint of a VALID
    forward Conv1d is a SCATTER of lossGrad through the SAME weight --
    conv1dBackward routes it to convTranspose1dKernelSymInt32Grouped
    (per-product rescale), so this ref delegates to convT1d_grouped_ref with
    the roles swapped: the scatter's "input" is lossGrad [batch, out_channels,
    forward_out_len] and its "output" is dx [batch, in_channels,
    input_length]. Parameters are named from the FORWARD Conv1d's perspective
    (in_channels/out_channels/input_length = x's channels/L; loss length is
    derived).

    Weight-index identity (why no re-layout is needed): convT1d_grouped_ref
    reads w at (ic_ref*outC_ref + oc_ref)*K + k with ic_ref over ITS
    in-channels (= Conv1d's out_channels) and oc_ref over ITS out-channels
    (= Conv1d's in_channels) -- i.e. (oc_conv*in_channels + ic_conv)*K + k,
    exactly Conv1d's [out_channels, in_channels, K] flat storage index (the
    same index ConvTranspose1dKernel.c computes at its wArr read in the
    adjoint role). `w_mantissas` is therefore passed in Conv1d's OWN storage
    order, and g = w_idx // group_size binds to the stored weight unchanged.
    conv-groups==1 only (the dx fixtures' scope). No bias (dx never has one).
    Returns (out flat [batch*in_channels*input_length], s_acc)."""
    eff_k = dilation * (kernel_size - 1) + 1
    assert input_length >= eff_k, "conv1d_dx_grouped_ref: forward geometry is empty"
    forward_out_len = (input_length - eff_k) // stride + 1
    assert (forward_out_len - 1) * stride + eff_k == input_length, (
        "conv1d_dx_grouped_ref: VALID forward geometry does not invert exactly "
        "(stride leaves a remainder) -- pick L with (L - effK) % stride == 0")
    out, s_acc, out_len = convT1d_grouped_ref(
        loss_mantissas, loss_scale, w_mantissas, w_scales, group_size,
        batch, out_channels, in_channels, kernel_size, forward_out_len,
        stride, dilation, 0)
    assert out_len == input_length
    return out, s_acc


def convT1d_dx_grouped_ref(loss_mantissas, loss_scale: float, w_mantissas, w_scales,
                           group_size: int, batch: int, in_channels: int, out_channels: int,
                           kernel_size: int, input_length: int, stride: int = 1,
                           dilation: int = 1, output_padding: int = 0):
    """ConvT1d dx (propLoss) with a grouped weight: the adjoint of a VALID
    forward ConvT1d is a GATHER (correlation) of lossGrad with the SAME
    weight -- conv1dTransposedBackward routes it to conv1dKernelSymInt32Grouped
    (running group-partial), so this ref delegates to conv1d_grouped_ref with
    the roles swapped: the gather's "input" is lossGrad [batch, out_channels,
    out_len] and its "output" is dx [batch, in_channels, input_length].
    Parameters are named from the FORWARD ConvT1d's perspective
    (in_channels/out_channels/input_length = x's channels/Lin).

    Weight-index identity: conv1d_grouped_ref reads w at
    (oc_ref*inC_ref + ic_ref)*K + k with oc_ref over ITS out-channels
    (= ConvT1d's in_channels) and ic_ref over ITS in-channels (= ConvT1d's
    out_channels) -- i.e. (ic_convT*out_channels + oc_convT)*K + k, exactly
    ConvT1d's [in_channels, out_channels, K] flat storage index (the same
    index Conv1dKernel.c computes at its wArr read in the adjoint role), so
    `w_mantissas` is passed in ConvT1d's OWN storage order and the group
    binding carries over unchanged. outputPadding only pads trailing zeros of
    the forward output; the adjoint gather walks the padded length, whose
    tail windows the geometry then clips -- fixtures here use 0.
    conv-groups==1 only. No bias. Returns (out flat
    [batch*in_channels*input_length], s_acc)."""
    out_len = ((input_length - 1) * stride + dilation * (kernel_size - 1) +
               output_padding + 1)
    out, s_acc, dx_len = conv1d_grouped_ref(
        loss_mantissas, loss_scale, w_mantissas, w_scales, group_size,
        batch, out_channels, in_channels, kernel_size, out_len,
        stride, dilation, "VALID", 0)
    assert dx_len == input_length, (
        f"convT1d_dx_grouped_ref: adjoint gather length {dx_len} != forward "
        f"input length {input_length}")
    return out, s_acc


# ---- Group-quant PR3 Task 4: optimizer enablement on grouped params. The
# update opSpecs declare the param's groupedSymOperandPos, so the existing
# funnel machinery does the requant -- no new C-side conversion code, just
# these two float32-precise primitives (mirroring the FLOAT32-arithmetic
# prologue's dequant and the OUT_WRITE epilogue's requant bit-for-bit) plus
# the SGD momentum==0 update they compose into. ----


def dequant_sym_grouped_f32(mantissas, scales, group_size: int) -> torch.Tensor:
    """float32 mirror of convertSymTensorToFloat32Tensor's grouped path
    (TensorConversion.c): out[i] = (float)mant[i] * scales[g], g = i //
    group_size -- EVERY intermediate stays float32 (never float64, unlike
    stable_dequant_grouped's fixture-construction helper above), matching the
    C `float` arithmetic exactly. Returns a torch.float32 tensor."""
    mant = torch.as_tensor(mantissas, dtype=torch.int32).flatten()
    n = mant.numel()
    assert n % group_size == 0, (
        f"dequant_sym_grouped_f32: n={n} not divisible by group_size={group_size}")
    scales_t = torch.as_tensor(scales, dtype=torch.float32)
    num_groups = n // group_size
    assert scales_t.numel() == num_groups, (
        f"dequant_sym_grouped_f32: {scales_t.numel()} scales for {num_groups} groups")
    per_elem_scale = scales_t.repeat_interleave(group_size)
    return mant.to(torch.float32) * per_elem_scale


def requant_absmax_grouped_f32(values, q_bits: int, group_size: int):
    """float32 mirror of packFloatBufferAsSym's grouped path
    (TensorConversion.c): per-group absMax -> scale = absMax/qMax (1.0 if
    absMax == 0), codes = round_half_away(clamp(value/scale)) -- EVERY
    intermediate stays float32 (the C kernel's `float` arithmetic; never
    float64). group_size == len(values) emulates the whole-tensor (numGroups
    == 1) requant -- used by generators as the "group collapse" mutation
    reference, not a real per-tensor path. Returns (codes: list[int],
    scales: list[float])."""
    x = torch.as_tensor(values, dtype=torch.float32).flatten()
    n = x.numel()
    assert n % group_size == 0, (
        f"requant_absmax_grouped_f32: n={n} not divisible by group_size={group_size}")
    num_groups = n // group_size
    q_max = torch.tensor(2.0 ** (q_bits - 1) - 1, dtype=torch.float32)
    q_min = torch.tensor(-(2.0 ** (q_bits - 1)), dtype=torch.float32)
    codes = []
    scales = []
    for g in range(num_groups):
        grp = x[g * group_size:(g + 1) * group_size]
        abs_max = grp.abs().max()
        if abs_max.item() == 0.0:
            scale = torch.tensor(1.0, dtype=torch.float32)
        else:
            scale = abs_max / q_max
        scales.append(scale.item())
        q = round_half_away(grp / scale)
        q = torch.clamp(q, q_min, q_max)
        codes.extend(int(v) for v in q.tolist())
    return codes, scales


# ---- Group-quant PR4 Task 1: nudged code-domain ASYM affine (TFLite-standard),
# deriveAsymGridFromMinMax + emitAsymChunk's per-tensor emulation. DELIBERATE
# numerics change vs the old value-domain int32-zeroPoint grid (spec D6):
# the band is nudged to include 0 (mn=min(mn,0), mx=max(mx,0)), which (a)
# makes 0.0 exactly representable (code == zp decodes to exactly 0.0) and
# (b) bounds zpReal into [0, 2^b-1] BY CONSTRUCTION, so the code-domain
# zeroPoint fits uint16 for qBits <= 16. Encode rounds the value quotient
# FIRST and adds the integer zp AFTER (round(v/scale) + zp), unlike the old
# single-round round(v/scale - zp_old) -- ties on negative values land
# differently (HALF_AWAY's "away" flips with the shift), so old pins
# re-derive through here, never by sign-flipping the old codes. ----


def quantize_asym_nudged(values, q_bits: int):
    """float32 mirror of the nudged code-domain ASYM quantizer
    (deriveAsymGridFromMinMax + emitAsymChunk, TensorConversion.c): EVERY
    intermediate stays float32, matching the C `float` arithmetic exactly.
      mn = min(min(values), 0); mx = max(max(values), 0)
      scale = (mx - mn)/(2^b - 1)   (mn == mx only for the all-zero buffer
                                     -> scale = 1.0, the adapted constant-band
                                     fallback)
      zp    = clamp(round_half_away(-mn/scale), 0, 2^b - 1)   [uint16 domain]
      code  = clamp(round_half_away(v/scale) + zp, 0, 2^b - 1)
      deq   = (code - zp) * scale
    Self-checks: zp in [0, 2^b-1]; any 0.0 input decodes to EXACTLY 0.0;
    every in-band value round-trips within 0.5*scale (+1 ulp headroom).
    Returns (codes: list[int], scale: float, zp: int)."""
    assert 1 <= q_bits <= 16, f"quantize_asym_nudged: qBits {q_bits} outside [1, 16] (D6)"
    x = torch.as_tensor(values, dtype=torch.float32).flatten()
    assert x.numel() > 0, "quantize_asym_nudged: empty buffer has no grid (n==0 is a C no-op)"
    q_max = 2 ** q_bits - 1
    zero = torch.tensor(0.0, dtype=torch.float32)
    mn = torch.minimum(x.min(), zero)
    mx = torch.maximum(x.max(), zero)
    if mn.item() == mx.item():
        # post-nudge mn == mx only when the whole buffer is 0.0
        assert mn.item() == 0.0
        scale = torch.tensor(1.0, dtype=torch.float32)
    else:
        scale = (mx - mn) / torch.tensor(float(q_max), dtype=torch.float32)
    zp_real = (-mn) / scale
    zp = int(round_half_away(zp_real).item())
    zp = max(0, min(q_max, zp))
    codes = []
    for v in x.tolist():
        q = int(round_half_away(torch.as_tensor(v, dtype=torch.float32) / scale).item()) + zp
        codes.append(max(0, min(q_max, q)))
    # -- self-checks (generator aborts rather than emit a self-contradiction) --
    assert 0 <= zp <= q_max, f"quantize_asym_nudged: zp {zp} outside [0, {q_max}]"
    s = scale.item()
    for c, v in zip(codes, x.tolist()):
        deq = float((torch.tensor(float(c - zp), dtype=torch.float32) *
                     scale).item())
        if v == 0.0:
            assert deq == 0.0, (
                f"quantize_asym_nudged: 0.0 decodes to {deq}, not exactly 0.0")
        # 0.5*scale is the exact rounding bound; the additive term covers the
        # float32 v/scale division drift (<= ~1 ulp of the quotient, i.e.
        # ~|v|*2^-23 in value space, doubled for headroom) which matters when
        # a band-edge quotient crosses a tie and the encode clamp folds the
        # overshoot back (e.g. v == mx rounding up).
        assert abs(deq - v) <= 0.5 * s + (abs(v) + s) * 2.0 ** -20, (
            f"quantize_asym_nudged: value {v} round-trips to {deq} "
            f"(err {abs(deq - v)} > scale/2 = {0.5 * s} + drift)")
    return codes, s, zp


def quantize_asym_old_value_domain(values, q_bits: int):
    """The PRE-PR4 value-domain grid (un-nudged, int32 zeroPoint), kept ONLY
    as the generators' old!=new self-check reference: scale = (mx-mn)/(2^b-1)
    on the RAW band, zp_old = round_half_away(mn/scale), code =
    clamp(round_half_away(v/scale - zp_old), 0, 2^b-1) -- the single-round
    encode the old emitAsymChunk used. Returns (codes, scale, zp_old)."""
    x = torch.as_tensor(values, dtype=torch.float32).flatten()
    q_max = 2 ** q_bits - 1
    mn = x.min()
    mx = x.max()
    if mn.item() == mx.item():
        scale = torch.tensor(1.0 if mn.item() == 0.0 else abs(mn.item()),
                             dtype=torch.float32)
    else:
        scale = (mx - mn) / torch.tensor(float(q_max), dtype=torch.float32)
    zp_old = int(round_half_away(mn / scale).item())
    codes = []
    for v in x.tolist():
        q = int(round_half_away(
            torch.as_tensor(v, dtype=torch.float32) / scale -
            torch.tensor(float(zp_old), dtype=torch.float32)).item())
        codes.append(max(0, min(q_max, q)))
    return codes, scale.item(), zp_old


# ---- Group-quant PR4 Task 2: per-group (storage-order) nudged code-domain
# ASYM quantization, quantizeFloatToAsym's grouped path (group of element i =
# i // group_size, exactly the SYM grouped grammar) -- each group derives its
# OWN nudged grid via quantize_asym_nudged over the group slice, so all of
# that helper's self-checks (zp bounds, exact-zero decode, round-trip bound)
# run per group. ----


def quantize_asym_grouped(values, q_bits: int, group_size: int):
    """Per-group nudged code-domain ASYM quantization, storage-order groups.
    Mirrors quantizeFloatToAsym's grouped path (TensorConversion.c): phase-1
    per-group min/max + nudge -> scales[g]/zps[g], phase-2 encode each element
    against ITS group's grid (round_half_away(v/scale) + zp, clamped to
    [0, 2^b-1]) -- float32 intermediates throughout (via quantize_asym_nudged).
    Returns (codes: list[int], scales: list[float], zps: list[int])."""
    x = torch.as_tensor(values, dtype=torch.float32).flatten()
    n = x.numel()
    assert n % group_size == 0, (
        f"quantize_asym_grouped: n={n} not divisible by group_size={group_size}")
    codes, scales, zps = [], [], []
    for g0 in range(0, n, group_size):
        c, s, z = quantize_asym_nudged(x[g0:g0 + group_size].tolist(), q_bits)
        codes.extend(c)
        scales.append(s)
        zps.append(z)
    return codes, scales, zps


def dequant_asym_grouped(codes, scales, zps, group_size: int):
    """float32 mirror of convertAsymTensorToFloatTensor's grouped path
    (TensorConversion.c): out[i] = (float)(code[i] - zps[g]) * scales[g],
    g = i // group_size -- the integer subtract is exact (both operands
    <= 2^16-1), the multiply is a single float32 op, matching the C `float`
    arithmetic exactly. Returns a plain list of floats."""
    n = len(codes)
    assert n % group_size == 0, (
        f"dequant_asym_grouped: n={n} not divisible by group_size={group_size}")
    out = []
    for i, c in enumerate(codes):
        g = i // group_size
        out.append(float((torch.tensor(float(c - zps[g]), dtype=torch.float32) *
                          torch.tensor(scales[g], dtype=torch.float32)).item()))
    return out


def sgd_grouped_step_ref(param_mantissas, param_scales, group_size: int, q_bits: int, grad,
                         lr: float, weight_decay: float = 0.0):
    """Group-quant PR3 Task 4: SGD momentum==0 update on a grouped-SYM param
    (sgdStepM's single-op fast path, sgdUpdateKernel {param, grad} -> the
    funnel's declared groupedSymOperandPos==1). Mirrors the executeOp funnel
    exactly:
      prologue: paramDeq = dequant_sym_grouped_f32(param) (float32, per-group)
      kernel:   g = grad + wd*paramDeq;  new = paramDeq - lr*g  (float32,
                same left-to-right op order as sgdUpdateKernel, Sgd.c)
      epilogue: per-group absmax requant (packFloatBufferAsSym's grouped
                path), HALF_AWAY -- holds only if the optimizer's
                writeBackRounding is HALF_AWAY (the fixture must call
                optimizerSetWriteBackRounding(optim, HALF_AWAY); factories
                default to seeded SR_HALF_AWAY, #279).
    `grad` is per-tensor FLOAT32 (the gradInit default -- no dequant/prologue
    conversion needed for it). Returns (new_mantissas: list[int],
    new_scales: list[float])."""
    param_deq = dequant_sym_grouped_f32(param_mantissas, param_scales, group_size)
    g = torch.as_tensor(grad, dtype=torch.float32)
    lr_t = torch.tensor(lr, dtype=torch.float32)
    wd_t = torch.tensor(weight_decay, dtype=torch.float32)
    combined = g + wd_t * param_deq
    new_param = param_deq - lr_t * combined
    return requant_absmax_grouped_f32(new_param, q_bits, group_size)


# ---- BFP epic PR2 Task 3: block-floating-point emulation, matmulBfpTensors'
# kernel reference (spec docs/superpowers/specs/2026-07-29-block-floating-point-design.md).
# The fold arithmetic mirrors the C kernel in np.float32 (never float64): one
# int partial per (a-group, b-group) segment, folded via float32 ldexp into a
# float32 accumulator whenever EITHER operand's group changes, plus a tail
# fold. The int partial itself is exact Python arithmetic guarded by an
# INT32-range assert -- the C kernel guarantees that bound via
# bfpValidateBlockHeadroom, never via int64 (int32 partials only). ----

_INT32_MAX = 2 ** 31 - 1


def bfp_derive_stored_exponent(abs_max, q_max, bias, max_stored):
    """Mirror C deriveBfpStoredExponent (TensorConversion.c): frexp snap-up,
    clamp [0, min(max_stored, bias + 127)] -- the high cap keeps the scale a
    FINITE float32 power of two (2^127); only exponentBits=8 can reach it.
    The quotient is computed in float32 first (the C divides absMax / qMax in
    float32 before frexpf); frexp of that exact float32 value in double
    preserves frac/exponent bit-for-bit."""
    if abs_max == 0.0:
        return bias
    frac, e = math.frexp(float(np.float32(abs_max) / np.float32(q_max)))
    E = e - 1 if frac == 0.5 else e
    return max(0, min(min(max_stored, bias + 127), E + bias))


def bfp_quantize_grouped(values, mantissa_bits, exponent_bits, group_size):
    """HALF_AWAY only (SR is exercised in C). Mirrors
    quantizeFloatBufferToBfpCodes (TensorConversion.c): per-group float32
    absmax -> bfp_derive_stored_exponent -> round_half_away(v / scale), clamp
    to [-2^(m-1), 2^(m-1)-1]; scale = 2^(stored - bias) exactly. group_size
    == 0 is the per-tensor sentinel (one group spanning all n). Returns
    (codes int32 list, stored exponents list)."""
    x = torch.as_tensor(values, dtype=torch.float32).flatten()
    n = x.numel()
    gsz = n if group_size == 0 else group_size
    assert n % gsz == 0, f"bfp_quantize_grouped: n={n} not divisible by group_size={gsz}"
    q_max = 2 ** (mantissa_bits - 1) - 1
    q_min = -(2 ** (mantissa_bits - 1))
    bias = 2 ** (exponent_bits - 1) - 1
    max_stored = 2 ** exponent_bits - 1
    codes, exps = [], []
    for g0 in range(0, n, gsz):
        grp = x[g0:g0 + gsz]
        abs_max = grp.abs().max().item()
        stored = bfp_derive_stored_exponent(abs_max, float(q_max), bias, max_stored)
        exps.append(stored)
        scale = torch.tensor(math.ldexp(1.0, stored - bias), dtype=torch.float32)
        q = torch.clamp(round_half_away(grp / scale), float(q_min), float(q_max))
        codes.extend(int(v) for v in q.tolist())
    return codes, exps


def _bfp_group_of(idx, group_size):
    """bfpGroupOf's twin (BfpKernelSupport.h): 0 for the per-tensor sentinel."""
    return 0 if group_size == 0 else idx // group_size


def matmul_bfp_ref(a_codes, a_exp, a_qc, b_codes, b_exp, b_qc, bias_codes, bias_exp, bias_qc,
                   rows, cols, K, b_transposed, a_transposed=False, self_check=True):
    """Mirror the C fold order exactly: int partial (assert |partial| <=
    2**31-1 -- the C kernel guarantees this via bfpValidateBlockHeadroom),
    np.float32 acc, np.ldexp folds, bias seeds acc first. Each *_qc is a dict
    with keys mantissa_bits / exponent_bits / group_size; codes are storage-
    order flat, exponents are stored (biased) per-group bytes. b's storage is
    [cols, K] when b_transposed (the GEMM-weight bOrder {1,0} view: b_idx =
    c*K + k), else [K, cols] (b_idx = k*cols + c). a's storage is [K, rows]
    when a_transposed (the loss^T view Linear weightGrad uses: a_idx =
    k*rows + r), else [rows, K] (a_idx = r*K + k). Self-checks (skipped on
    the collapse rerun): (i) >= 2 groups crossed on EACH operand somewhere,
    (ii) >= 1 fold with a NONZERO partial whose float conversion is exact
    (regression anchor), (iii) result differs from an all-per-tensor run
    (group structure matters), (iv) >= 1 reduction step where a's group
    changes while b's does NOT (pins the EITHER-operand fold clause: a
    fixture whose a-boundaries all coincide with b-boundaries cannot tell a
    b-only fold condition from the correct either-operand one). Returns the
    float32 outputs as Python floats, row-major [rows*cols]."""
    a_bias = 2 ** (a_qc["exponent_bits"] - 1) - 1
    b_bias = 2 ** (b_qc["exponent_bits"] - 1) - 1
    out = []
    fold_partials = []
    max_a_groups_crossed = 0
    max_b_groups_crossed = 0
    a_only_boundaries = 0
    for r in range(rows):
        for c in range(cols):
            acc = np.float32(0.0)
            if bias_codes is not None:
                bg = _bfp_group_of(c, bias_qc["group_size"])
                bias_bias = 2 ** (bias_qc["exponent_bits"] - 1) - 1
                scale = np.float32(math.ldexp(1.0, bias_exp[bg] - bias_bias))
                acc = np.float32(np.float32(bias_codes[c]) * scale)
            partial = 0
            cur_ga, cur_gb = 0, 0
            a_groups_seen, b_groups_seen = set(), set()
            for k in range(K):
                # a_transposed: logical a[r][k] reads storage [K, rows] row-
                # major -- the loss^T view Linear weightGrad uses. Storage is
                # loss [batch, outF] and the logical matrix is [outF, batch],
                # so rows == outF, K == batch: a_idx = k*outF + r == k*rows + r.
                a_idx = (k * rows + r) if a_transposed else (r * K + k)
                b_idx = c * K + k if b_transposed else k * cols + c
                ga = _bfp_group_of(a_idx, a_qc["group_size"])
                gb = _bfp_group_of(b_idx, b_qc["group_size"])
                a_groups_seen.add(ga)
                b_groups_seen.add(gb)
                if k == 0:
                    cur_ga, cur_gb = ga, gb
                elif ga != cur_ga or gb != cur_gb:
                    if ga != cur_ga and gb == cur_gb:
                        a_only_boundaries += 1
                    shift = (a_exp[cur_ga] - a_bias) + (b_exp[cur_gb] - b_bias)
                    fold_partials.append(partial)
                    acc = np.float32(acc + np.ldexp(np.float32(partial), np.int32(shift)))
                    partial = 0
                    cur_ga, cur_gb = ga, gb
                partial += a_codes[a_idx] * b_codes[b_idx]
                assert abs(partial) <= _INT32_MAX, (
                    f"matmul_bfp_ref: partial {partial} exceeds int32 -- fixture violates "
                    "the bfpValidateBlockHeadroom bound the C kernel enforces")
            if K > 0:
                shift = (a_exp[cur_ga] - a_bias) + (b_exp[cur_gb] - b_bias)
                fold_partials.append(partial)
                acc = np.float32(acc + np.ldexp(np.float32(partial), np.int32(shift)))
            max_a_groups_crossed = max(max_a_groups_crossed, len(a_groups_seen))
            max_b_groups_crossed = max(max_b_groups_crossed, len(b_groups_seen))
            out.append(float(acc))

    if self_check:
        # (i) group tracking is exercised on BOTH operands.
        assert max_a_groups_crossed >= 2, (
            "matmul_bfp_ref: no reduction crosses >= 2 a-groups -- a's group "
            "tracking is unexercised")
        assert max_b_groups_crossed >= 2, (
            "matmul_bfp_ref: no reduction crosses >= 2 b-groups -- b's group "
            "tracking is unexercised")
        # (ii) regression anchor: >= 1 fold whose (float)partial conversion is
        # exact AND nonzero (a zero partial is vacuously exact).
        assert any(p != 0 and float(np.float32(p)) == float(p) for p in fold_partials), (
            "matmul_bfp_ref: no fold has a nonzero exactly-float-convertible "
            "partial -- fixture lost its exact-regime anchor")
        # (iv) EITHER-operand fold clause: >= 1 step where a's group changes
        # while b's does not -- a-boundaries must not all hide behind
        # b-boundaries, or a b-only fold condition is indistinguishable.
        assert a_only_boundaries >= 1, (
            "matmul_bfp_ref: every a-group boundary coincides with a b-group "
            "boundary -- the either-operand fold clause is unexercised")
        # (iii) group structure matters: collapsing both operands to per-tensor
        # (exponents[0] everywhere) must change the result.
        collapsed = matmul_bfp_ref(
            a_codes, [a_exp[0]], {**a_qc, "group_size": 0},
            b_codes, [b_exp[0]], {**b_qc, "group_size": 0},
            bias_codes, bias_exp, bias_qc, rows, cols, K, b_transposed,
            a_transposed=a_transposed, self_check=False)
        assert collapsed != out, (
            "matmul_bfp_ref: per-tensor collapse is indistinguishable from the "
            "grouped run -- fixture is vacuous against group-structure bugs")
    return out


def matmul_bfp_bias_grad_ref(codes, exps, qc, rows, cols, self_check=True):
    """BFP epic PR3: bias-grad reference -- db[f] = sum_n loss[n][f] on BFP loss
    codes (int mantissas, per-element group lookup). Fold rule: int partial per
    same-group VISITED segment (the walk strides by `cols`, so groups can change
    every step), fold acc += ldexp(float32(partial), E) on group change + tail.
    Sum headroom: |partial| <= segment_len * (2^(m-1)) asserted <= 2^31-1."""
    bias = 2 ** (qc["exponent_bits"] - 1) - 1
    gsz = qc["group_size"] if qc["group_size"] else rows * cols
    out = []
    crossings = 0
    for f in range(cols):
        acc = np.float32(0.0)
        partial = 0
        cur_g = None
        for n in range(rows):
            idx = n * cols + f
            g = idx // gsz
            if cur_g is None:
                cur_g = g
            elif g != cur_g:
                assert abs(partial) <= _INT32_MAX
                acc = np.float32(acc + np.ldexp(np.float32(partial),
                                                np.int32(exps[cur_g] - bias)))
                partial = 0
                cur_g = g
                crossings += 1
            partial += codes[idx]
        if cur_g is not None:
            assert abs(partial) <= _INT32_MAX
            acc = np.float32(acc + np.ldexp(np.float32(partial),
                                            np.int32(exps[cur_g] - bias)))
        out.append(float(acc))
    if self_check:
        assert crossings >= 1, "bias-grad fixture never crosses a loss group -- vacuous"
        collapse = matmul_bfp_bias_grad_ref(
            codes, [exps[0]] * len(exps), qc, rows, cols, self_check=False)
        assert collapse != out, "per-tensor collapse identical -- exponent binding unobservable"
    return out


def conv1d_bfp_ref(x_codes, x_exp, x_qc, w_codes, w_exp, w_qc, bias_codes, bias_exp, bias_qc,
                   batch, in_channels, out_channels, kernel_size, input_length,
                   stride=1, dilation=1, padding_type="VALID", padding=0, conv_groups=1,
                   self_check=True):
    """BFP epic PR2 Task 4: conv1dKernelBfp's kernel emulation. The fold rule
    is matmul_bfp_ref's, transplanted onto conv1d_grouped_ref's sliding-window
    walk: per (b, oc, out_pos) ONE int partial (assert |partial| <= 2**31-1 --
    the C kernel guarantees this via bfpValidateBlockHeadroom); each visited
    tap maps BOTH operands' STORAGE indices to group ids (_bfp_group_of --
    per-element division, gap-robust across the index gaps that clipped
    windows create); when EITHER id changes, the finished segment folds via
    np.float32 acc += np.ldexp((float32)partial, Ein + Ew - biasIn - biasW)
    and the partial resets; tail fold after the walk (guarded on >= 1 visited
    tap, mirroring the C kernel's empty-window branch). Bias is a value-seed
    dequantized to float32 BEFORE the reduction. `w_codes` is
    [out_channels, in_channels/conv_groups, kernel_size] row-major flat,
    `x_codes` is [batch, in_channels, input_length] row-major flat.

    Self-checks (skipped on the collapse rerun):
      (i)  >= 2 groups crossed on EACH operand within a single reduction;
      (ii) >= 1 fold with a NONZERO exactly-float-convertible partial;
      (iii) result differs from an all-per-tensor (exponents[0]) collapse;
      (iv) >= 1 output element whose tap window is CLIPPED (0 < valid_count
           < kernel_size -- pins the gap-robust per-element group lookup);
      plus the disjoint-boundary pins (Task 3 review lesson, both directions):
      >= 1 step where ONLY the input's group changes and >= 1 step where ONLY
      the weight's group changes -- a fixture whose boundaries always
      coincide cannot tell a one-operand fold condition from the correct
      either-operand one.
    Returns the float32 outputs as Python floats, row-major
    [batch*out_channels*out_len]."""
    assert in_channels % conv_groups == 0 and out_channels % conv_groups == 0, (
        "conv1d_bfp_ref: conv_groups must divide in_channels and out_channels")
    in_ch_per_group = in_channels // conv_groups
    out_ch_per_group = out_channels // conv_groups
    geom = window_geometry_1d(input_length, kernel_size, stride, dilation, padding_type, padding)
    out_len = geom["out_len"]
    x_bias = 2 ** (x_qc["exponent_bits"] - 1) - 1
    w_bias = 2 ** (w_qc["exponent_bits"] - 1) - 1

    out = []
    fold_partials = []
    max_x_groups_crossed = 0
    max_w_groups_crossed = 0
    x_only_boundaries = 0
    w_only_boundaries = 0
    clipped_windows = 0
    for b in range(batch):
        for oc in range(out_channels):
            conv_g = oc // out_ch_per_group
            in_lo = conv_g * in_ch_per_group
            w_base = oc * in_ch_per_group * kernel_size
            for out_pos in range(out_len):
                first_valid_idx, first_valid_k, valid_count = window_slice_1d_full(geom, out_pos)
                if 0 < valid_count < kernel_size:
                    clipped_windows += 1
                acc = np.float32(0.0)
                if bias_codes is not None:
                    bg = _bfp_group_of(oc, bias_qc["group_size"])
                    bias_bias = 2 ** (bias_qc["exponent_bits"] - 1) - 1
                    scale = np.float32(math.ldexp(1.0, bias_exp[bg] - bias_bias))
                    acc = np.float32(np.float32(bias_codes[oc]) * scale)
                partial = 0
                cur_gx, cur_gw = None, None
                x_groups_seen, w_groups_seen = set(), set()
                for ic_offset in range(in_ch_per_group):
                    ic = in_lo + ic_offset
                    for i in range(valid_count):
                        kernel_idx = first_valid_k + i
                        w_idx = w_base + ic_offset * kernel_size + kernel_idx
                        input_idx = first_valid_idx + i * dilation
                        x_idx = (b * in_channels + ic) * input_length + input_idx
                        gx = _bfp_group_of(x_idx, x_qc["group_size"])
                        gw = _bfp_group_of(w_idx, w_qc["group_size"])
                        x_groups_seen.add(gx)
                        w_groups_seen.add(gw)
                        if cur_gw is None:
                            cur_gx, cur_gw = gx, gw
                        elif gx != cur_gx or gw != cur_gw:
                            if gx != cur_gx and gw == cur_gw:
                                x_only_boundaries += 1
                            if gw != cur_gw and gx == cur_gx:
                                w_only_boundaries += 1
                            shift = (x_exp[cur_gx] - x_bias) + (w_exp[cur_gw] - w_bias)
                            fold_partials.append(partial)
                            acc = np.float32(acc + np.ldexp(np.float32(partial), np.int32(shift)))
                            partial = 0
                            cur_gx, cur_gw = gx, gw
                        partial += x_codes[x_idx] * w_codes[w_idx]
                        assert abs(partial) <= _INT32_MAX, (
                            f"conv1d_bfp_ref: partial {partial} exceeds int32 -- fixture "
                            "violates the bfpValidateBlockHeadroom bound the C kernel enforces")
                if cur_gw is not None:
                    shift = (x_exp[cur_gx] - x_bias) + (w_exp[cur_gw] - w_bias)
                    fold_partials.append(partial)
                    acc = np.float32(acc + np.ldexp(np.float32(partial), np.int32(shift)))
                max_x_groups_crossed = max(max_x_groups_crossed, len(x_groups_seen))
                max_w_groups_crossed = max(max_w_groups_crossed, len(w_groups_seen))
                out.append(float(acc))

    if self_check:
        # (i) group tracking is exercised on BOTH operands.
        assert max_x_groups_crossed >= 2, (
            "conv1d_bfp_ref: no reduction crosses >= 2 input groups -- the "
            "input's group tracking is unexercised")
        assert max_w_groups_crossed >= 2, (
            "conv1d_bfp_ref: no reduction crosses >= 2 weight groups -- the "
            "weight's group tracking is unexercised")
        # (ii) regression anchor: >= 1 fold whose (float)partial conversion is
        # exact AND nonzero (a zero partial is vacuously exact).
        assert any(p != 0 and float(np.float32(p)) == float(p) for p in fold_partials), (
            "conv1d_bfp_ref: no fold has a nonzero exactly-float-convertible "
            "partial -- fixture lost its exact-regime anchor")
        # (iv) clipped-window pin: the gap-robust per-element lookup is only
        # under test if some window actually skips taps.
        assert clipped_windows >= 1, (
            "conv1d_bfp_ref: no output element has a clipped tap window -- "
            "the gap-robust group lookup is unexercised")
        # Disjoint-boundary pins (both directions, Task 3 review lesson).
        assert x_only_boundaries >= 1, (
            "conv1d_bfp_ref: every input-group boundary coincides with a "
            "weight-group boundary -- the either-operand fold clause is "
            "unexercised on the input side")
        assert w_only_boundaries >= 1, (
            "conv1d_bfp_ref: every weight-group boundary coincides with an "
            "input-group boundary -- the either-operand fold clause is "
            "unexercised on the weight side")
        # (iii) group structure matters: collapsing both operands to
        # per-tensor (exponents[0] everywhere) must change the result.
        collapsed = conv1d_bfp_ref(
            x_codes, [x_exp[0]], {**x_qc, "group_size": 0},
            w_codes, [w_exp[0]], {**w_qc, "group_size": 0},
            bias_codes, bias_exp, bias_qc,
            batch, in_channels, out_channels, kernel_size, input_length,
            stride, dilation, padding_type, padding, conv_groups,
            self_check=False)
        assert collapsed != out, (
            "conv1d_bfp_ref: per-tensor collapse is indistinguishable from the "
            "grouped run -- fixture is vacuous against group-structure bugs")
    return out


# ---- BFP epic PR2 Task 5: gather-formulated ConvT1d BFP reference (D9),
# convTranspose1dKernelBfpGather's kernel emulation. Output-centric: every
# output element is ONE dot product over its contributors (convT1d_taps_at),
# restoring the int32 block-partial contract the scatter formulation cannot
# offer (a scatter's consecutive products land in DIFFERENT output elements).
# The SYM scatter core stays untouched -- this ref pins the NEW gather walk. ----


def convT1d_taps_at(out_pos, input_length, kernel_size, stride, dilation, pad_left):
    """Mirror convTranspose1dTapsAt (SlidingWindow1d.c): the contributors of
    ConvT1d output position out_pos -- kernel taps k with
    (out_pos + pad_left - k*dilation) % stride == 0 and
    in_pos = (out_pos + pad_left - k*dilation) // stride in [0, input_length),
    emitted in ascending k order. Returns a list of (in_pos, k) pairs."""
    taps = []
    p = out_pos + pad_left
    for k in range(kernel_size):
        kd = k * dilation
        if kd > p:
            break  # k*dilation grows monotonically; later taps reach even further left
        rem = p - kd
        if rem % stride != 0:
            continue
        in_pos = rem // stride
        if in_pos >= input_length:
            continue
        taps.append((in_pos, k))
    return taps


def convT1d_bfp_gather_ref(x_codes, x_exp, x_qc, w_codes, w_exp, w_qc,
                           bias_codes, bias_exp, bias_qc,
                           batch, in_channels, out_channels, kernel_size, input_length,
                           stride=1, dilation=1, output_padding=0, conv_groups=1,
                           self_check=True):
    """Mirror the C gather kernel's fold order exactly: per (b, conv-group, oc,
    out_pos) ONE int partial (assert |partial| <= 2**31-1 -- the C kernel
    guarantees this via bfpValidateBlockHeadroom over inChPerGroup*kernelSize);
    the reduction walks taps OUTER, ic_offset INNER (the brief's normative
    order -- NOT conv1d_bfp_ref's ic-outer order); each visited step maps BOTH
    operands' storage indices to group ids (_bfp_group_of, per-element -- tap
    hops make both index sequences non-contiguous); when EITHER id changes the
    finished segment folds via np.float32 acc += np.ldexp((float32)partial,
    Ein + Ew - biasIn - biasW) and resets; tail fold after the walk (guarded
    on >= 1 visited step -- outputPadding tail positions have ZERO taps and
    stay at the bias seed). Bias is a value-seed dequantized to float32 BEFORE
    the reduction. VALID-only geometry (pad_left = 0; the C kernel's
    SAME/EXPLICIT adjoint branch is exercised in C directly via the geometry
    parity test): out_len = (input_length-1)*stride + dilation*(kernel_size-1)
    + output_padding + 1.

    `w_codes` is ConvT1d's [in_channels, out_channels/conv_groups, kernel_size]
    row-major flat storage: the weight index for (ic, oc_offset, k) is
    (ic*out_ch_per_group + oc_offset)*kernel_size + k -- the SAME flat index
    every ConvTranspose1dKernel.c core reads. `x_codes` is
    [batch, in_channels, input_length] row-major flat.

    Self-checks (skipped on the collapse rerun):
      (i)   >= 2 groups crossed on EACH operand within a single reduction;
      (ii)  >= 1 fold with a NONZERO exactly-float-convertible partial;
      (iii) result differs from an all-per-tensor (exponents[0]) collapse;
      plus the disjoint-boundary pins (both directions): >= 1 step where ONLY
      the input's group changes and >= 1 step where ONLY the weight's group
      changes; >= 1 (b, oc, out_pos) with ZERO taps (the outputPadding tail --
      pins the bias-seed-only path); and the SCATTER CROSS-CHECK: a float32
      scatter reference (convTranspose1dKernelFloat32's loop structure) on the
      DEQUANTIZED values must equal the gather output bit-for-bit -- valid
      because the fixture lives in the exact float regime, where add order
      cannot matter.
    Returns the float32 outputs as Python floats, row-major
    [batch*out_channels*out_len]."""
    assert in_channels % conv_groups == 0 and out_channels % conv_groups == 0, (
        "convT1d_bfp_gather_ref: conv_groups must divide in_channels and out_channels")
    in_ch_per_group = in_channels // conv_groups
    out_ch_per_group = out_channels // conv_groups
    out_len = (input_length - 1) * stride + dilation * (kernel_size - 1) + output_padding + 1
    x_bias = 2 ** (x_qc["exponent_bits"] - 1) - 1
    w_bias = 2 ** (w_qc["exponent_bits"] - 1) - 1

    out = []
    fold_partials = []
    max_x_groups_crossed = 0
    max_w_groups_crossed = 0
    x_only_boundaries = 0
    w_only_boundaries = 0
    tap_free_positions = 0
    for b in range(batch):
        for oc in range(out_channels):
            conv_g = oc // out_ch_per_group
            in_lo = conv_g * in_ch_per_group
            oc_offset = oc % out_ch_per_group
            for out_pos in range(out_len):
                taps = convT1d_taps_at(out_pos, input_length, kernel_size, stride, dilation, 0)
                if not taps:
                    tap_free_positions += 1
                acc = np.float32(0.0)
                if bias_codes is not None:
                    bg = _bfp_group_of(oc, bias_qc["group_size"])
                    bias_bias = 2 ** (bias_qc["exponent_bits"] - 1) - 1
                    scale = np.float32(math.ldexp(1.0, bias_exp[bg] - bias_bias))
                    acc = np.float32(np.float32(bias_codes[oc]) * scale)
                partial = 0
                cur_gx, cur_gw = None, None
                x_groups_seen, w_groups_seen = set(), set()
                for in_pos, k in taps:
                    for ic_offset in range(in_ch_per_group):
                        ic = in_lo + ic_offset
                        x_idx = (b * in_channels + ic) * input_length + in_pos
                        w_idx = (ic * out_ch_per_group + oc_offset) * kernel_size + k
                        gx = _bfp_group_of(x_idx, x_qc["group_size"])
                        gw = _bfp_group_of(w_idx, w_qc["group_size"])
                        x_groups_seen.add(gx)
                        w_groups_seen.add(gw)
                        if cur_gw is None:
                            cur_gx, cur_gw = gx, gw
                        elif gx != cur_gx or gw != cur_gw:
                            if gx != cur_gx and gw == cur_gw:
                                x_only_boundaries += 1
                            if gw != cur_gw and gx == cur_gx:
                                w_only_boundaries += 1
                            shift = (x_exp[cur_gx] - x_bias) + (w_exp[cur_gw] - w_bias)
                            fold_partials.append(partial)
                            acc = np.float32(acc + np.ldexp(np.float32(partial), np.int32(shift)))
                            partial = 0
                            cur_gx, cur_gw = gx, gw
                        partial += x_codes[x_idx] * w_codes[w_idx]
                        assert abs(partial) <= _INT32_MAX, (
                            f"convT1d_bfp_gather_ref: partial {partial} exceeds int32 -- fixture "
                            "violates the bfpValidateBlockHeadroom bound the C kernel enforces")
                if cur_gw is not None:
                    shift = (x_exp[cur_gx] - x_bias) + (w_exp[cur_gw] - w_bias)
                    fold_partials.append(partial)
                    acc = np.float32(acc + np.ldexp(np.float32(partial), np.int32(shift)))
                max_x_groups_crossed = max(max_x_groups_crossed, len(x_groups_seen))
                max_w_groups_crossed = max(max_w_groups_crossed, len(w_groups_seen))
                out.append(float(acc))

    if self_check:
        # (i) group tracking is exercised on BOTH operands.
        assert max_x_groups_crossed >= 2, (
            "convT1d_bfp_gather_ref: no reduction crosses >= 2 input groups -- "
            "the input's group tracking is unexercised")
        assert max_w_groups_crossed >= 2, (
            "convT1d_bfp_gather_ref: no reduction crosses >= 2 weight groups -- "
            "the weight's group tracking is unexercised")
        # (ii) regression anchor: >= 1 fold whose (float)partial conversion is
        # exact AND nonzero (a zero partial is vacuously exact).
        assert any(p != 0 and float(np.float32(p)) == float(p) for p in fold_partials), (
            "convT1d_bfp_gather_ref: no fold has a nonzero exactly-float-"
            "convertible partial -- fixture lost its exact-regime anchor")
        # Disjoint-boundary pins (both directions, Task 3 review lesson).
        assert x_only_boundaries >= 1, (
            "convT1d_bfp_gather_ref: every input-group boundary coincides with "
            "a weight-group boundary -- the either-operand fold clause is "
            "unexercised on the input side")
        assert w_only_boundaries >= 1, (
            "convT1d_bfp_gather_ref: every weight-group boundary coincides "
            "with an input-group boundary -- the either-operand fold clause is "
            "unexercised on the weight side")
        # outputPadding tail pin: >= 1 output position with ZERO taps, whose
        # value is the bias seed alone (the gather's empty-tap branch).
        assert tap_free_positions >= 1, (
            "convT1d_bfp_gather_ref: no output position is tap-free -- the "
            "outputPadding/bias-seed-only branch is unexercised")
        # Scatter cross-check (D9): a float32 scatter on the DEQUANTIZED values
        # must reproduce the gather bit-for-bit in the exact regime -- pins the
        # gather's tap set AND index mapping against the shipped scatter form.
        def _deq(codes, exps, qc, n):
            bias_ = 2 ** (qc["exponent_bits"] - 1) - 1
            return [np.float32(np.float32(codes[i]) *
                               np.float32(math.ldexp(1.0, exps[_bfp_group_of(
                                   i, qc["group_size"])] - bias_)))
                    for i in range(n)]
        deq_x = _deq(x_codes, x_exp, x_qc, batch * in_channels * input_length)
        deq_w = _deq(w_codes, w_exp, w_qc,
                     in_channels * out_ch_per_group * kernel_size)
        scatter = [np.float32(0.0)] * (batch * out_channels * out_len)
        for b in range(batch):
            for conv_g in range(conv_groups):
                for ic_offset in range(in_ch_per_group):
                    ic = conv_g * in_ch_per_group + ic_offset
                    for in_pos in range(input_length):
                        xv = deq_x[(b * in_channels + ic) * input_length + in_pos]
                        for oc_offset in range(out_ch_per_group):
                            oc = conv_g * out_ch_per_group + oc_offset
                            for k in range(kernel_size):
                                out_idx = in_pos * stride + k * dilation
                                if out_idx >= out_len:
                                    continue
                                wv = deq_w[(ic * out_ch_per_group + oc_offset) *
                                           kernel_size + k]
                                flat = (b * out_channels + oc) * out_len + out_idx
                                scatter[flat] = np.float32(scatter[flat] +
                                                           np.float32(xv * wv))
        if bias_codes is not None:
            deq_b = _deq(bias_codes, bias_exp, bias_qc, out_channels)
            for b in range(batch):
                for oc in range(out_channels):
                    for l in range(out_len):
                        flat = (b * out_channels + oc) * out_len + l
                        scatter[flat] = np.float32(scatter[flat] + deq_b[oc])
        assert [float(v) for v in scatter] == out, (
            "convT1d_bfp_gather_ref: gather output diverges from the float "
            "scatter reference on dequantized values -- either the tap "
            "enumeration mirror or the fixture's exact-regime claim is broken")
        # (iii) group structure matters: collapsing both operands to
        # per-tensor (exponents[0] everywhere) must change the result.
        collapsed = convT1d_bfp_gather_ref(
            x_codes, [x_exp[0]], {**x_qc, "group_size": 0},
            w_codes, [w_exp[0]], {**w_qc, "group_size": 0},
            bias_codes, bias_exp, bias_qc,
            batch, in_channels, out_channels, kernel_size, input_length,
            stride, dilation, output_padding, conv_groups,
            self_check=False)
        assert collapsed != out, (
            "convT1d_bfp_gather_ref: per-tensor collapse is indistinguishable "
            "from the grouped run -- fixture is vacuous against group-structure "
            "bugs")
    return out


# ---- BFP epic PR3 Task 3: Conv1d backward references. weightGrad is a NEW
# output-centric core (one gw element per reduction, contributors walked
# b outer / outPos inner -- the NORMATIVE order the C kernel mirrors);
# biasGrad is the single-operand segment fold shared by Conv1d and ConvT1d;
# dx delegates to the D9 gather ref with the adjoint role swap. ----


def conv1d_bfp_weight_grad_ref(x_codes, x_exp, x_qc, gy_codes, gy_exp, gy_qc,
                               batch, in_channels, out_channels, kernel_size, input_length,
                               stride=1, dilation=1, padding_type="VALID", padding=0,
                               conv_groups=1, group_offset_shift=0, self_check=True):
    """Conv1d weight grad on BFP operands, output-centric: per gw element
    (oc, ic_offset, k) ONE int partial over its contributors -- the
    (b, out_pos) pairs whose window visits tap k -- walked b OUTER, out_pos
    INNER (the normative order the C kernel mirrors). Per contributor
    window_slice_1d_full(geom, out_pos) -> (first_in, first_k, valid);
    contribute iff first_k <= k < first_k + valid with in_idx_local =
    first_in + (k - first_k) * dilation; storage indices
    x_idx = (b*in_channels + ic)*input_length + in_idx_local and
    gy_idx = (b*out_channels + oc)*output_length + out_pos map to group ids
    per step (_bfp_group_of, per-element); when EITHER id changes the
    finished segment folds via np.float32 acc += np.ldexp((float32)partial,
    Ex + Egy - biasX - biasGy) and resets; tail fold guarded on >= 1 visited
    contributor (a (oc, k) whose windows never reach tap k -- extreme
    padding -- emits 0.0).

    CONV GROUPS (#420 C1): the outer nest mirrors the C kernel
    operation-for-operation -- `for g` derives `in_lo = g*in_ch_per_group`
    and `out_lo = g*out_ch_per_group`, `oc = out_lo + oc_offset`,
    `ic = in_lo + ic_offset`, and the result cell is the WEIGHT storage
    index `(oc*in_ch_per_group + ic_offset)*kernel_size + k` (weight shape
    [Cout, Cin/groups, K]). Each cell is assigned exactly once (asserted) --
    the C twin has no memset for the same reason. `group_offset_shift` is a
    SELF-CHECK knob, not a modelling parameter: a nonzero value rotates the
    input-channel base by that many groups (`in_lo` only), reproducing the
    "group arithmetic is inert / off by one group" mutant so the generator
    can prove the fixture observes it. Leave it 0 for gold emission.

    Self-checks (skipped on the collapse/shift reruns, mirroring
    conv1d_bfp_ref's):
      (i)   >= 2 groups crossed on EACH operand within a single reduction;
      (ii)  >= 1 fold with a NONZERO exactly-float-convertible partial;
      (iii) result differs from an all-per-tensor (exponents[0]) collapse;
      plus the disjoint-boundary pins (both directions): >= 1 step where
      ONLY x's group changes and >= 1 step where ONLY gy's group changes.
    Grouped/padded fixtures (#420 C1) additionally pin, whenever the
    geometry offers them, that the group arithmetic and both padding
    branches are load-bearing:
      (iv)  conv_groups > 1: shifting `in_lo` by one group changes >= 1 cell
            (a kernel whose `in_lo`/`out_lo` derivation is inert would be
            indistinguishable);
      (v)   padded geometries: >= 1 (out_pos, k) pair takes the
            tap-membership skip on a tap that DOES contribute elsewhere
            (the partially-clipped window the `continue` exists for);
      (vi)  padded geometries: >= 1 cell has NO contributor at all and
            therefore lands on exactly 0.0 through the unvisited-contributor
            branch (the guarded tail fold).
    (v)/(vi) are asserted only when `padding` is nonzero -- a VALID fixture
    cannot clip a window at all, which is why the pre-#420 fixtures leave
    both branches dead.

    Returns the float32 grads as Python floats, row-major
    [out_channels*(in_channels//conv_groups)*kernel_size]."""
    assert in_channels % conv_groups == 0 and out_channels % conv_groups == 0, (
        "conv1d_bfp_weight_grad_ref: conv_groups must divide both channel counts")
    in_ch_per_group = in_channels // conv_groups
    out_ch_per_group = out_channels // conv_groups
    geom = window_geometry_1d(input_length, kernel_size, stride, dilation, padding_type, padding)
    output_length = geom["out_len"]
    x_bias = 2 ** (x_qc["exponent_bits"] - 1) - 1
    gy_bias = 2 ** (gy_qc["exponent_bits"] - 1) - 1

    out = [None] * (out_channels * in_ch_per_group * kernel_size)
    fold_partials = []
    max_x_groups_crossed = 0
    max_gy_groups_crossed = 0
    x_only_boundaries = 0
    gy_only_boundaries = 0
    clipped_skips = 0      # (v): skip on a tap that contributes somewhere else
    unvisited_cells = 0    # (vi): cell with no contributor at all
    for g in range(conv_groups):
        in_lo = ((g + group_offset_shift) % conv_groups) * in_ch_per_group
        out_lo = g * out_ch_per_group
        for oc_offset in range(out_ch_per_group):
            oc = out_lo + oc_offset
            for ic_offset in range(in_ch_per_group):
                ic = in_lo + ic_offset
                for k in range(kernel_size):
                    acc = np.float32(0.0)
                    partial = 0
                    cur_gx, cur_ggy = None, None
                    x_groups_seen, gy_groups_seen = set(), set()
                    skipped_here = 0
                    for b in range(batch):
                        for out_pos in range(output_length):
                            first_in, first_k, valid = window_slice_1d_full(geom, out_pos)
                            if not (first_k <= k < first_k + valid):
                                skipped_here += 1
                                continue
                            in_idx_local = first_in + (k - first_k) * dilation
                            x_idx = (b * in_channels + ic) * input_length + in_idx_local
                            gy_idx = (b * out_channels + oc) * output_length + out_pos
                            gx = _bfp_group_of(x_idx, x_qc["group_size"])
                            ggy = _bfp_group_of(gy_idx, gy_qc["group_size"])
                            x_groups_seen.add(gx)
                            gy_groups_seen.add(ggy)
                            if cur_ggy is None:
                                cur_gx, cur_ggy = gx, ggy
                            elif gx != cur_gx or ggy != cur_ggy:
                                if gx != cur_gx and ggy == cur_ggy:
                                    x_only_boundaries += 1
                                if ggy != cur_ggy and gx == cur_gx:
                                    gy_only_boundaries += 1
                                shift = (x_exp[cur_gx] - x_bias) + (gy_exp[cur_ggy] - gy_bias)
                                fold_partials.append(partial)
                                acc = np.float32(
                                    acc + np.ldexp(np.float32(partial), np.int32(shift)))
                                partial = 0
                                cur_gx, cur_ggy = gx, ggy
                            partial += x_codes[x_idx] * gy_codes[gy_idx]
                            assert abs(partial) <= _INT32_MAX, (
                                f"conv1d_bfp_weight_grad_ref: partial {partial} exceeds int32 -- "
                                "fixture violates the bfpValidateBlockHeadroom bound the C kernel "
                                "enforces")
                    if cur_ggy is not None:
                        shift = (x_exp[cur_gx] - x_bias) + (gy_exp[cur_ggy] - gy_bias)
                        fold_partials.append(partial)
                        acc = np.float32(acc + np.ldexp(np.float32(partial), np.int32(shift)))
                        if skipped_here:
                            clipped_skips += skipped_here
                    else:
                        unvisited_cells += 1
                    max_x_groups_crossed = max(max_x_groups_crossed, len(x_groups_seen))
                    max_gy_groups_crossed = max(max_gy_groups_crossed, len(gy_groups_seen))
                    cell = (oc * in_ch_per_group + ic_offset) * kernel_size + k
                    assert out[cell] is None, (
                        "conv1d_bfp_weight_grad_ref: weight-grad cell written twice -- the "
                        "output-centric walk must visit each cell exactly once")
                    out[cell] = float(acc)
    assert all(v is not None for v in out), (
        "conv1d_bfp_weight_grad_ref: weight-grad cell never written -- the group nest does "
        "not cover the weight tensor")

    if self_check:
        assert max_x_groups_crossed >= 2, (
            "conv1d_bfp_weight_grad_ref: no reduction crosses >= 2 input groups "
            "-- the input's group tracking is unexercised")
        assert max_gy_groups_crossed >= 2, (
            "conv1d_bfp_weight_grad_ref: no reduction crosses >= 2 loss groups "
            "-- the loss's group tracking is unexercised")
        assert any(p != 0 and float(np.float32(p)) == float(p) for p in fold_partials), (
            "conv1d_bfp_weight_grad_ref: no fold has a nonzero exactly-float-"
            "convertible partial -- fixture lost its exact-regime anchor")
        assert x_only_boundaries >= 1, (
            "conv1d_bfp_weight_grad_ref: every input-group boundary coincides "
            "with a loss-group boundary -- the either-operand fold clause is "
            "unexercised on the input side")
        assert gy_only_boundaries >= 1, (
            "conv1d_bfp_weight_grad_ref: every loss-group boundary coincides "
            "with an input-group boundary -- the either-operand fold clause is "
            "unexercised on the loss side")
        collapsed = conv1d_bfp_weight_grad_ref(
            x_codes, [x_exp[0]], {**x_qc, "group_size": 0},
            gy_codes, [gy_exp[0]], {**gy_qc, "group_size": 0},
            batch, in_channels, out_channels, kernel_size, input_length,
            stride, dilation, padding_type, padding, conv_groups, self_check=False)
        assert collapsed != out, (
            "conv1d_bfp_weight_grad_ref: per-tensor collapse is "
            "indistinguishable from the grouped run -- fixture is vacuous "
            "against group-structure bugs")
        if conv_groups > 1:
            shifted = conv1d_bfp_weight_grad_ref(
                x_codes, x_exp, x_qc, gy_codes, gy_exp, gy_qc,
                batch, in_channels, out_channels, kernel_size, input_length,
                stride, dilation, padding_type, padding, conv_groups,
                group_offset_shift=1, self_check=False)
            assert shifted != out, (
                "conv1d_bfp_weight_grad_ref: rotating the input-channel group base by one "
                "group leaves every cell unchanged -- the kernel's in_lo/out_lo derivation "
                "is unobservable; pick channel data that differs across groups")
        if padding:
            assert clipped_skips >= 1, (
                "conv1d_bfp_weight_grad_ref: no contributing tap is ever clipped away at "
                "some out_pos -- the tap-membership skip is dead code under this geometry")
            assert unvisited_cells >= 1, (
                "conv1d_bfp_weight_grad_ref: every weight cell has a contributor -- the "
                "unvisited-contributor 0.0 branch is dead code under this geometry")
    return out


def conv_bfp_bias_grad_ref(gy_codes, gy_exp, gy_qc, batch, out_channels, output_length,
                           self_check=True):
    """Conv-family bias grad on BFP loss codes -- db[oc] = sum over (b,
    out_pos) of gy[b][oc][out_pos], walked b OUTER / out_pos INNER over
    gy_idx = (b*out_channels + oc)*output_length + out_pos. Fold rule and
    self-checks are matmul_bfp_bias_grad_ref's, on the conv walk (the walk
    hops by output_length at each b step, so groups can change every step).
    Shared by Conv1d and ConvT1d (identical [B, C, L] loss layout), hence
    the generic name."""
    bias = 2 ** (gy_qc["exponent_bits"] - 1) - 1
    out = []
    crossings = 0
    for oc in range(out_channels):
        acc = np.float32(0.0)
        partial = 0
        cur_g = None
        for b in range(batch):
            for out_pos in range(output_length):
                idx = (b * out_channels + oc) * output_length + out_pos
                g = _bfp_group_of(idx, gy_qc["group_size"])
                if cur_g is None:
                    cur_g = g
                elif g != cur_g:
                    assert abs(partial) <= _INT32_MAX
                    acc = np.float32(acc + np.ldexp(np.float32(partial),
                                                    np.int32(gy_exp[cur_g] - bias)))
                    partial = 0
                    cur_g = g
                    crossings += 1
                partial += gy_codes[idx]
        if cur_g is not None:
            assert abs(partial) <= _INT32_MAX
            acc = np.float32(acc + np.ldexp(np.float32(partial),
                                            np.int32(gy_exp[cur_g] - bias)))
        out.append(float(acc))
    if self_check:
        assert crossings >= 1, (
            "conv_bfp_bias_grad_ref: bias-grad fixture never crosses a loss "
            "group -- vacuous")
        collapse = conv_bfp_bias_grad_ref(
            gy_codes, [gy_exp[0]] * len(gy_exp), gy_qc, batch, out_channels,
            output_length, self_check=False)
        assert collapse != out, (
            "conv_bfp_bias_grad_ref: per-tensor collapse identical -- exponent "
            "binding unobservable")
    return out


def conv1d_bfp_dx_ref(loss_codes, loss_exp, loss_qc, w_codes, w_exp, w_qc,
                      batch, in_channels, out_channels, kernel_size, input_length,
                      stride=1, dilation=1, conv_groups=1, self_check=True):
    """Conv1d dx (propLoss) on BFP operands: the adjoint of a VALID forward
    Conv1d, computed GATHER-formulated (D9) -- conv1dBackward routes it to
    convTranspose1dKernelBfpGather, so this ref delegates to
    convT1d_bfp_gather_ref with the roles swapped: the gather's "input" is
    lossGrad [batch, out_channels, forward_out_len] and its "output" is dx
    [batch, in_channels, input_length]. Parameters are named from the FORWARD
    Conv1d's perspective (in_channels/out_channels/input_length = x's
    channels/L; loss length is derived). bias is None (dx never has one),
    output_padding 0.

    Weight-index identity (why no re-layout is needed): convT1d_bfp_gather_ref
    reads w at (ic_ref*outChPerGroup_ref + oc_offset_ref)*K + k with ic_ref
    over ITS in-channels (= Conv1d's out_channels) and oc_offset_ref over ITS
    per-group out-channels (= Conv1d's per-group in_channels) -- i.e.
    (oc_conv*inChPerGroup + ic_offset_conv)*K + k, exactly Conv1d's
    [out_channels, in_channels/conv_groups, K] flat storage index (the same
    index ConvTranspose1dKernel.c computes at its wArr read in the adjoint
    role). `w_codes` is therefore passed in Conv1d's OWN storage order, and
    the per-element group binding carries over unchanged. conv_groups passes
    through (#416: the gather's conv-group of its oc IS the conv's group of
    that in-channel). `self_check` passes through to the gather ref's
    built-ins; layer-wiring fixtures whose 2-channel gather walk cannot
    satisfy the disjoint-boundary pins structurally call with False and pin
    layer-relevant properties in the generator instead. Returns the float32
    dx as Python floats, row-major [batch*in_channels*input_length]."""
    eff_k = dilation * (kernel_size - 1) + 1
    assert input_length >= eff_k, "conv1d_bfp_dx_ref: forward geometry is empty"
    forward_out_len = (input_length - eff_k) // stride + 1
    assert (forward_out_len - 1) * stride + eff_k == input_length, (
        "conv1d_bfp_dx_ref: VALID forward geometry does not invert exactly "
        "(stride leaves a remainder) -- pick L with (L - effK) % stride == 0")
    out = convT1d_bfp_gather_ref(
        loss_codes, loss_exp, loss_qc, w_codes, w_exp, w_qc,
        None, None, None, batch, out_channels, in_channels, kernel_size,
        forward_out_len, stride, dilation, 0, conv_groups, self_check=self_check)
    assert len(out) == batch * in_channels * input_length
    return out


# ---- BFP epic PR3 Task 4: ConvT1d backward references. weightGrad is the
# ConvT twin of conv1d_bfp_weight_grad_ref -- output-centric with the ConvT
# AFFINE contributor map (out_idx = in_pos*stride + k*dilation; no window
# geometry); biasGrad reuses conv_bfp_bias_grad_ref (identical [B, C, L]
# loss walk, see its docstring); dx delegates to conv1d_bfp_ref with the
# adjoint role swap (the SYM convT1d_dx_grouped_ref pattern). ----


def convT1d_bfp_weight_grad_ref(x_codes, x_exp, x_qc, gy_codes, gy_exp, gy_qc,
                                batch, in_channels, out_channels, kernel_size, input_length,
                                stride=1, dilation=1, output_padding=0, conv_groups=1,
                                group_offset_shift=0, self_check=True):
    """ConvT1d weight grad on BFP operands, output-centric: per gw element
    (ic, oc_offset, k) -- ConvT weight [Cin, Cout/groups, K] storage order --
    ONE int partial over its contributors, walked b OUTER, in_pos INNER (the
    normative order the C kernel mirrors). Per contributor
    out_idx = in_pos * stride + k * dilation; contribute iff
    out_idx < output_length with output_length = (input_length-1)*stride +
    dilation*(kernel_size-1) + output_padding + 1 (the clip is DEFENSIVE
    under this forward-shaped geometry: max out_idx = output_length -
    output_padding - 1, so every (b, in_pos) pair contributes -- outputPadding
    tail positions of gy are simply never read). Storage indices
    x_idx = (b*in_channels + ic)*input_length + in_pos and
    gy_idx = (b*out_channels + oc)*output_length + out_idx map to group ids
    per step (_bfp_group_of, per-element); when EITHER id changes the
    finished segment folds via np.float32 acc += np.ldexp((float32)partial,
    Ex + Egy - biasX - biasGy) and resets; tail fold guarded on >= 1 visited
    contributor.

    CONV GROUPS (#420 C1): the outer nest mirrors the C kernel
    operation-for-operation -- `for g` derives `in_lo = g*in_ch_per_group`
    and `out_lo = g*out_ch_per_group`, with ic_offset OUTER and oc_offset
    INNER (the reverse of Conv1d's nest, matching Conv1dTransposed.c), and
    the result cell is the WEIGHT storage index
    `(ic*out_ch_per_group + oc_offset)*kernel_size + k`. Each cell is
    assigned exactly once (asserted) -- the C twin has no memset for the
    same reason. `group_offset_shift` is a SELF-CHECK knob, not a modelling
    parameter: a nonzero value rotates the OUTPUT-channel base by that many
    groups (`out_lo` only -- the ConvT write index is keyed off the global
    `ic`, so rotating `in_lo` would only permute cells rather than mispair
    operands), reproducing the "group arithmetic is inert / off by one
    group" mutant. Leave it 0 for gold emission.

    UNLIKE Conv1d, this kernel has NO reachable tap-membership skip and no
    reachable unvisited-contributor branch: Conv1dTransposed rejects any
    paddingType other than VALID at layer init (Phase-1 contract,
    Conv1dTransposed.c), so the contributor map is the unconditional affine
    out_idx = in_pos*stride + k*dilation whose maximum is
    output_length - output_padding - 1. `output_padding` is therefore the
    ConvT analogue of Conv1d's padding for fixture purposes: it lengthens gy
    (shifting every gy group binding) and leaves tail positions that only
    biasGrad reads.

    Self-checks (skipped on the collapse/shift reruns,
    conv1d_bfp_weight_grad_ref's suite):
      (i)   >= 2 groups crossed on EACH operand within a single reduction;
      (ii)  >= 1 fold with a NONZERO exactly-float-convertible partial;
      (iii) result differs from an all-per-tensor (exponents[0]) collapse;
      plus the disjoint-boundary pins (both directions): >= 1 step where
      ONLY x's group changes and >= 1 step where ONLY gy's group changes;
      plus, for conv_groups > 1, (iv) rotating the output-channel group base
      by one group changes >= 1 cell.
    Returns the float32 grads as Python floats, row-major
    [in_channels*(out_channels//conv_groups)*kernel_size]."""
    assert in_channels % conv_groups == 0 and out_channels % conv_groups == 0, (
        "convT1d_bfp_weight_grad_ref: conv_groups must divide both channel counts")
    in_ch_per_group = in_channels // conv_groups
    out_ch_per_group = out_channels // conv_groups
    output_length = (input_length - 1) * stride + dilation * (kernel_size - 1) \
        + output_padding + 1
    x_bias = 2 ** (x_qc["exponent_bits"] - 1) - 1
    gy_bias = 2 ** (gy_qc["exponent_bits"] - 1) - 1

    out = [None] * (in_channels * out_ch_per_group * kernel_size)
    fold_partials = []
    max_x_groups_crossed = 0
    max_gy_groups_crossed = 0
    x_only_boundaries = 0
    gy_only_boundaries = 0
    for g in range(conv_groups):
        in_lo = g * in_ch_per_group
        out_lo = ((g + group_offset_shift) % conv_groups) * out_ch_per_group
        for ic_offset in range(in_ch_per_group):
            ic = in_lo + ic_offset
            for oc_offset in range(out_ch_per_group):
                oc = out_lo + oc_offset
                for k in range(kernel_size):
                    acc = np.float32(0.0)
                    partial = 0
                    cur_gx, cur_ggy = None, None
                    x_groups_seen, gy_groups_seen = set(), set()
                    for b in range(batch):
                        for in_pos in range(input_length):
                            out_idx = in_pos * stride + k * dilation
                            if out_idx >= output_length:
                                continue
                            x_idx = (b * in_channels + ic) * input_length + in_pos
                            gy_idx = (b * out_channels + oc) * output_length + out_idx
                            gx = _bfp_group_of(x_idx, x_qc["group_size"])
                            ggy = _bfp_group_of(gy_idx, gy_qc["group_size"])
                            x_groups_seen.add(gx)
                            gy_groups_seen.add(ggy)
                            if cur_ggy is None:
                                cur_gx, cur_ggy = gx, ggy
                            elif gx != cur_gx or ggy != cur_ggy:
                                if gx != cur_gx and ggy == cur_ggy:
                                    x_only_boundaries += 1
                                if ggy != cur_ggy and gx == cur_gx:
                                    gy_only_boundaries += 1
                                shift = (x_exp[cur_gx] - x_bias) + (gy_exp[cur_ggy] - gy_bias)
                                fold_partials.append(partial)
                                acc = np.float32(
                                    acc + np.ldexp(np.float32(partial), np.int32(shift)))
                                partial = 0
                                cur_gx, cur_ggy = gx, ggy
                            partial += x_codes[x_idx] * gy_codes[gy_idx]
                            assert abs(partial) <= _INT32_MAX, (
                                f"convT1d_bfp_weight_grad_ref: partial {partial} exceeds int32 "
                                "-- fixture violates the bfpValidateBlockHeadroom bound the C "
                                "kernel enforces")
                    if cur_ggy is not None:
                        shift = (x_exp[cur_gx] - x_bias) + (gy_exp[cur_ggy] - gy_bias)
                        fold_partials.append(partial)
                        acc = np.float32(acc + np.ldexp(np.float32(partial), np.int32(shift)))
                    max_x_groups_crossed = max(max_x_groups_crossed, len(x_groups_seen))
                    max_gy_groups_crossed = max(max_gy_groups_crossed, len(gy_groups_seen))
                    cell = (ic * out_ch_per_group + oc_offset) * kernel_size + k
                    assert out[cell] is None, (
                        "convT1d_bfp_weight_grad_ref: weight-grad cell written twice -- the "
                        "output-centric walk must visit each cell exactly once")
                    out[cell] = float(acc)
    assert all(v is not None for v in out), (
        "convT1d_bfp_weight_grad_ref: weight-grad cell never written -- the group nest does "
        "not cover the weight tensor")

    if self_check:
        assert max_x_groups_crossed >= 2, (
            "convT1d_bfp_weight_grad_ref: no reduction crosses >= 2 input groups "
            "-- the input's group tracking is unexercised")
        assert max_gy_groups_crossed >= 2, (
            "convT1d_bfp_weight_grad_ref: no reduction crosses >= 2 loss groups "
            "-- the loss's group tracking is unexercised")
        assert any(p != 0 and float(np.float32(p)) == float(p) for p in fold_partials), (
            "convT1d_bfp_weight_grad_ref: no fold has a nonzero exactly-float-"
            "convertible partial -- fixture lost its exact-regime anchor")
        assert x_only_boundaries >= 1, (
            "convT1d_bfp_weight_grad_ref: every input-group boundary coincides "
            "with a loss-group boundary -- the either-operand fold clause is "
            "unexercised on the input side")
        assert gy_only_boundaries >= 1, (
            "convT1d_bfp_weight_grad_ref: every loss-group boundary coincides "
            "with an input-group boundary -- the either-operand fold clause is "
            "unexercised on the loss side")
        collapsed = convT1d_bfp_weight_grad_ref(
            x_codes, [x_exp[0]], {**x_qc, "group_size": 0},
            gy_codes, [gy_exp[0]], {**gy_qc, "group_size": 0},
            batch, in_channels, out_channels, kernel_size, input_length,
            stride, dilation, output_padding, conv_groups, self_check=False)
        assert collapsed != out, (
            "convT1d_bfp_weight_grad_ref: per-tensor collapse is "
            "indistinguishable from the grouped run -- fixture is vacuous "
            "against group-structure bugs")
        if conv_groups > 1:
            shifted = convT1d_bfp_weight_grad_ref(
                x_codes, x_exp, x_qc, gy_codes, gy_exp, gy_qc,
                batch, in_channels, out_channels, kernel_size, input_length,
                stride, dilation, output_padding, conv_groups,
                group_offset_shift=1, self_check=False)
            assert shifted != out, (
                "convT1d_bfp_weight_grad_ref: rotating the output-channel group base by one "
                "group leaves every cell unchanged -- the kernel's in_lo/out_lo derivation "
                "is unobservable; pick channel data that differs across groups")
    return out


def convT1d_bfp_dx_ref(loss_codes, loss_exp, loss_qc, w_codes, w_exp, w_qc,
                       batch, in_channels, out_channels, kernel_size, input_length,
                       stride=1, dilation=1, output_padding=0, conv_groups=1,
                       self_check=True):
    """ConvT1d dx (propLoss) on BFP operands: the adjoint of a VALID forward
    ConvT1d is a GATHER (correlation) of lossGrad with the SAME weight --
    conv1dTransposedBackward routes it to conv1dKernelBfp, so this ref
    delegates to conv1d_bfp_ref with the roles swapped (the SYM
    convT1d_dx_grouped_ref pattern): the gather's "input" is lossGrad
    [batch, out_channels, out_len] and its "output" is dx
    [batch, in_channels, input_length]. Parameters are named from the FORWARD
    ConvT1d's perspective (in_channels/out_channels/input_length = x's
    channels/Lin). VALID-only, no bias (dx never has one); outputPadding only
    pads trailing zeros of the forward output, whose tail positions the
    adjoint gather's VALID windows never reach.

    Weight-index identity (why no re-layout is needed): conv1d_bfp_ref reads
    w at (oc_ref*inChPerGroup_ref + ic_offset_ref)*K + k with oc_ref over ITS
    out-channels (= ConvT1d's in_channels) and ic_offset_ref over ITS
    per-group in-channels (= ConvT1d's per-group out_channels) -- i.e.
    (ic_convT*outChPerGroup + oc_offset_convT)*K + k, exactly ConvT1d's
    [in_channels, out_channels/conv_groups, K] flat storage index (the same
    index Conv1dKernel.c computes at its wArr read in the adjoint role), so
    `w_codes` is passed in ConvT1d's OWN storage order and the per-element
    group binding carries over unchanged. conv_groups passes through (#416:
    the gather's conv-group of its oc IS the ConvT group of that in-channel).

    `self_check` passes through to the delegate's built-ins. Every dx
    delegation here must run with False -- NOT because of the
    disjoint-boundary pins (the delegate walks ic outer / taps inner with
    unit weight-index steps, so quiet steps exist on both operands), but
    because conv1d_bfp_ref's clipped-window pin (iv) requires a window with
    0 < valid_count < kernel_size and the ConvT dx adjoint is VALID-only
    (Phase-1 contract), where no window is ever clipped. Generators pin
    layer-relevant replacements instead (per-operand collapse-differs,
    >= 1 nonzero). Returns the float32 dx as Python floats, row-major
    [batch*in_channels*input_length]."""
    out_len = (input_length - 1) * stride + dilation * (kernel_size - 1) \
        + output_padding + 1
    out = conv1d_bfp_ref(
        loss_codes, loss_exp, loss_qc, w_codes, w_exp, w_qc,
        None, None, None, batch, out_channels, in_channels, kernel_size,
        out_len, stride, dilation, "VALID", 0, conv_groups, self_check=self_check)
    assert len(out) == batch * in_channels * input_length, (
        f"convT1d_bfp_dx_ref: adjoint gather length {len(out) // (batch * in_channels)} != "
        f"forward input length {input_length} -- outputPadding >= stride does not invert")
    return out


# ---- BFP epic PR3 Task 7: SGD momentum step with BFP param AND BFP momentum
# state (per-tensor storage; the optimizer machinery itself is unchanged --
# this mirrors the funnel's dequant/kernel/repack sequence around the two
# Sgd.c momentum ops). ----


def bfp_dequant_f32(codes, exps, qc):
    """Exact float32 dequant, dequantChunkToFloat's BFP arm: code *
    2^(stored - bias) -- a float32 multiply by a power of two is exact.
    Returns a torch.float32 tensor."""
    bias = 2 ** (qc["exponent_bits"] - 1) - 1
    gsz = qc["group_size"]
    vals = []
    for i, c in enumerate(codes):
        g = _bfp_group_of(i, gsz)
        scale = np.float32(math.ldexp(1.0, exps[g] - bias))
        vals.append(float(np.float32(np.float32(c) * scale)))
    return torch.tensor(vals, dtype=torch.float32)


def sgd_bfp_step_ref(param_codes, param_exps, param_qc, grad, lr, momentum,
                     state_codes, state_exps, state_qc, weight_decay=0.0):
    """One sgdStepM momentum step (momentumFactor > 0, Sgd.c:147-185) with a
    per-tensor BFP param AND a per-tensor BFP momentum state, mirroring the
    executeOp funnel bit-for-bit:

      op1 sgdMStateKernel {state, grad, param}: the FLOAT32 prologue dequants
          state and param EXACTLY (code * 2^(E-bias)); float32 kernel
          g = grad + wd*paramDeq; newState = momentum*stateDeq + g (same
          left-to-right op order as the C); the OUT_WRITE epilogue re-packs
          the state tensor per-tensor BFP (packFloatBufferAsBfp: fresh absmax
          exponent + HALF_AWAY codes -- holds only under the #279
          writeBackRounding opt-out, optimizerSetWriteBackRounding /
          .writeBackRounding = HALF_AWAY).
      op2 sgdMParamKernel {param, state}: paramDeq is RE-derived from the
          UNTOUCHED param codes (op1 never wrote param); the state operand is
          dequanted from its FRESHLY REQUANTIZED codes -- NOT op1's raw float
          result (this ordering is what makes the state repack load-bearing);
          newParam = paramDeq - lr*stateReq; per-tensor BFP repack of param.

    Self-checks (abort rather than emit a vacuous fixture):
      (i)   canonical inputs: requantizing the exact dequant of each input
            reproduces its codes AND exponents bit-for-bit (a non-canonical
            fixture would leave the exact-float regime and hide
            exponent-derivation bugs);
      (ii)  the param exponent MOVES across the step (an implementation that
            forgets to re-derive the shared exponent on write-back is
            observable);
      (iii) BOTH repacks change values (the BFP quantization is load-bearing
            on each write-back -- the raw float update alone cannot
            reproduce the gold).

    Codes/exps are lists (exps: stored/biased, per-group; per-tensor = one
    entry); each qc is the mantissa_bits/exponent_bits/group_size dict the
    other BFP refs use. Returns (new_param_codes, new_param_exps,
    new_state_codes, new_state_exps)."""
    param_deq = bfp_dequant_f32(param_codes, param_exps, param_qc)
    state_deq = bfp_dequant_f32(state_codes, state_exps, state_qc)

    # (i) canonical-input roundtrip.
    for name, deq, codes, exps, qc in (
            ("param", param_deq, param_codes, param_exps, param_qc),
            ("state", state_deq, state_codes, state_exps, state_qc)):
        rq_codes, rq_exps = bfp_quantize_grouped(deq, qc["mantissa_bits"],
                                                 qc["exponent_bits"], qc["group_size"])
        assert rq_codes == list(codes) and rq_exps == list(exps), (
            f"sgd_bfp_step_ref: {name} input is not canonical -- requantizing its exact "
            f"dequant gives codes {rq_codes} exps {rq_exps}, not the fixture's; pick "
            "grid-exact values whose absmax code sits in (qMax/2, qMax]")

    g = torch.as_tensor(grad, dtype=torch.float32)
    wd_t = torch.tensor(weight_decay, dtype=torch.float32)
    momentum_t = torch.tensor(momentum, dtype=torch.float32)
    lr_t = torch.tensor(lr, dtype=torch.float32)

    # op1 sgdMStateKernel: g = grad + wd*paramDeq; newState = momentum*stateDeq + g.
    combined = g + wd_t * param_deq
    new_state_float = momentum_t * state_deq + combined
    new_state_codes, new_state_exps = bfp_quantize_grouped(
        new_state_float, state_qc["mantissa_bits"], state_qc["exponent_bits"],
        state_qc["group_size"])

    # op2 sgdMParamKernel reads the REQUANTIZED state, never op1's raw floats.
    state_req_deq = bfp_dequant_f32(new_state_codes, new_state_exps, state_qc)
    new_param_float = param_deq - lr_t * state_req_deq
    new_param_codes, new_param_exps = bfp_quantize_grouped(
        new_param_float, param_qc["mantissa_bits"], param_qc["exponent_bits"],
        param_qc["group_size"])

    # (ii) param exponent moves.
    assert new_param_exps != list(param_exps), (
        "sgd_bfp_step_ref: param exponent did not move -- fixture cannot observe a "
        "write-back that forgets to re-derive the shared exponent; scale the update "
        "so the param absmax crosses a binade")
    # (iii) both repacks change values.
    assert not torch.equal(state_req_deq, new_state_float), (
        "sgd_bfp_step_ref: state repack is value-neutral -- fixture cannot tell the "
        "requantized state from op1's raw float result")
    param_req_deq = bfp_dequant_f32(new_param_codes, new_param_exps, param_qc)
    assert not torch.equal(param_req_deq, new_param_float), (
        "sgd_bfp_step_ref: param repack is value-neutral -- the BFP quantization is "
        "not load-bearing on the param write-back")

    return new_param_codes, new_param_exps, new_state_codes, new_state_exps


# ---- BFP epic PR4 (R-P3): Dropout's float bridge on packed BFP storage. ----


def bfp_mask_scale_repack_ref(codes, exps, qc, keep_mask, factor, self_check=True):
    """Mirror the C two-pass walk (Dropout.c dropoutMaskScaleBfp, which is
    scaleBfpTensorInPlace's skeleton with the BOOL mask fused in): dequantize
    exactly (code * 2^(E-bias)), apply the keep mask and the 1/(1-p) factor in
    float32 (C order: (float)mant * srcScale * factor), then re-derive EVERY
    group's exponent from the NEW absmax and requantize HALF_AWAY.

    Re-deriving here is NOT double quantization: the multiply changed the
    values, so the fresh exponents quantize NEW numbers (spec D8 forbids
    re-blocking UNCHANGED values).

    Self-checks (abort rather than emit a vacuous fixture):
      (i)   the mask both keeps and drops (it is load-bearing);
      (ii)  >= 1 group's exponent MOVES, so a Relu-style verbatim exponent
            copy is observable;
      (iii) the codes differ from the verbatim-exponent mutant (same values
            requantized onto the OLD grid) -- the fresh derive is observable
            element-wise, not only in the exponent bytes.
    Returns (codes, stored exponents) for the destination wire."""
    deq = bfp_dequant_f32(codes, exps, qc)
    f = np.float32(factor)
    vals = torch.tensor(
        [float(np.float32(np.float32(deq[i].item()) * f)) if keep_mask[i] else 0.0
         for i in range(len(codes))], dtype=torch.float32)
    new_codes, new_exps = bfp_quantize_grouped(
        vals, qc["mantissa_bits"], qc["exponent_bits"], qc["group_size"])
    if self_check:
        assert any(keep_mask) and not all(keep_mask), (
            "bfp_mask_scale_repack_ref: mask keeps everything or nothing -- vacuous")
        assert new_exps != list(exps), (
            "bfp_mask_scale_repack_ref: no group exponent moved -- a verbatim exponent "
            "copy would be indistinguishable; pick values whose masked absmax crosses "
            "a binade")
        gsz = len(codes) if qc["group_size"] == 0 else qc["group_size"]
        bias = 2 ** (qc["exponent_bits"] - 1) - 1
        q_max = 2 ** (qc["mantissa_bits"] - 1) - 1
        q_min = -(2 ** (qc["mantissa_bits"] - 1))
        mutant = []
        for i in range(len(codes)):
            scale = torch.tensor(math.ldexp(1.0, exps[i // gsz] - bias), dtype=torch.float32)
            v = torch.tensor([float(vals[i])], dtype=torch.float32)
            mutant.append(int(torch.clamp(round_half_away(v / scale),
                                          float(q_min), float(q_max))[0]))
        assert mutant != new_codes, (
            "bfp_mask_scale_repack_ref: verbatim-exponent codes equal the fresh-derive "
            "codes -- the re-derivation is unobservable in this fixture")
    return new_codes, new_exps


# ---- BFP epic PR4 (R-P4): pooling kernels. ----


def bfp_window_sum_ref(codes, exps, qc, indices):
    """Shared BFP pooling window-sum core (AvgPool1d / AdaptiveAvgPool1d
    forward kernels): ONE int32 partial per same-group visited segment of the
    STORAGE indices `indices`, folded into a float32 accumulator with np.ldexp
    on every group change and at the tail -- the C kernels' bfpGroupOf/ldexpf
    walk, element for element. Per-ELEMENT group lookup, never a run
    precompute: dilated/strided windows skip storage indices, so a run-based
    shortcut would bind the wrong exponent (the PR3 precedent). Asserts the
    int32 bound bfpValidateSumHeadroom guarantees in C.
    Returns (np.float32 accumulator, number of group crossings)."""
    bias = 2 ** (qc["exponent_bits"] - 1) - 1
    acc = np.float32(0.0)
    partial = 0
    cur_g = None
    crossings = 0
    for j, idx in enumerate(indices):
        g = _bfp_group_of(idx, qc["group_size"])
        if j == 0:
            cur_g = g
        elif g != cur_g:
            assert abs(partial) <= _INT32_MAX, (
                "bfp_window_sum_ref: partial exceeds int32 -- fixture violates the "
                "bfpValidateSumHeadroom bound the C kernel enforces")
            acc = np.float32(acc + np.ldexp(np.float32(partial), np.int32(exps[cur_g] - bias)))
            partial = 0
            cur_g = g
            crossings += 1
        partial += codes[idx]
    if cur_g is not None:
        assert abs(partial) <= _INT32_MAX
        acc = np.float32(acc + np.ldexp(np.float32(partial), np.int32(exps[cur_g] - bias)))
    return acc, crossings


def avgpool1d_bfp_forward_ref(codes, exps, qc, batch, channels, geom, self_check=True):
    """AvgPool1d ARITH_BFP forward reference: bfp_window_sum_ref per window,
    then a float32 divide by K. count_include_pad=True, so the divisor is
    ALWAYS kernel_size (padded positions simply are not visited) -- and the
    SYM arm's exact s/K scale fold has NO BFP analog, because a BFP scale is
    2^E and K is not a power of two in general. `geom` is a window_geometry_1d
    dict. Returns row-major [batch*channels*out_len] float32 as Python floats.

    Self-checks (abort rather than emit a vacuous fixture):
      (i)   >= 1 window crosses a group boundary (the fold clause runs);
      (ii)  the geometry is strided or dilated (per-element bfpGroupOf is
            load-bearing -- a consecutive walk cannot tell it from a run
            precompute);
      (iii) collapsing the input to per-tensor changes the result (the group
            structure is observable);
      (iv)  >= 1 nonzero output."""
    out = []
    crossings_total = 0
    k = np.float32(geom["kernel_size"])
    for b in range(batch):
        for c in range(channels):
            base = (b * channels + c) * geom["input_length"]
            for o in range(geom["out_len"]):
                first, count = window_slice_1d(geom, o)
                idxs = [base + first + i * geom["dilation"] for i in range(count)]
                acc, crossings = bfp_window_sum_ref(codes, exps, qc, idxs)
                crossings_total += crossings
                out.append(float(np.float32(acc / k)))
    if self_check:
        assert crossings_total >= 1, (
            "avgpool1d_bfp_forward_ref: no window crosses a group boundary -- the "
            "ldexpf fold clause is unexercised")
        assert geom["dilation"] > 1 or geom["stride"] > 1, (
            "avgpool1d_bfp_forward_ref: consecutive windows cannot distinguish a "
            "per-element bfpGroupOf from a run precompute -- use stride or dilation")
        collapsed = avgpool1d_bfp_forward_ref(codes, [exps[0]], {**qc, "group_size": 0},
                                              batch, channels, geom, self_check=False)
        assert collapsed != out, (
            "avgpool1d_bfp_forward_ref: per-tensor collapse is indistinguishable -- "
            "fixture is vacuous against group-structure bugs")
        assert any(v != 0.0 for v in out), "avgpool1d_bfp_forward_ref: all-zero output"
    return out
