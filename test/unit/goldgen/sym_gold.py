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
