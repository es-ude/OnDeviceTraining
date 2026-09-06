#define SOURCE_FILE "TENSOR_CONVERSION"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "Common.h"
#include "DTypes.h"
#include "MinMax.h"
#include "Tensor.h"
#include "TensorConversion.h"
#include "math.h"

static void packFloatBufferAsSym(const float *values, size_t n, symQConfig_t *outQC, uint8_t *dst,
                                 const char *what);
static void packFloatBufferAsBfp(const float *values, size_t n, bfpQConfig_t *outQC, uint8_t *dst,
                                 const char *what);
/* Chunk reader signature for the streaming BFP pack -- dequantChunkToFloat's
 * shape, so that helper binds directly; per-dtype readers cover what it
 * denies (grouped SYM, grouped ASYM). */
typedef void (*bfpSrcChunkReader_t)(const tensor_t *src, size_t elemOffset, size_t count,
                                    float *out);
static void packStreamAsBfp(const tensor_t *src, bfpSrcChunkReader_t readChunk, size_t n,
                            bfpQConfig_t *outQC, uint8_t *dst, const char *what);
static void packDequantStreamAsBfp(const tensor_t *src, size_t n, bfpQConfig_t *outQC, uint8_t *dst,
                                   const char *what);
static void dequantGroupedSymChunkToFloat(const tensor_t *src, size_t elemOffset, size_t count,
                                          float *out);
static void dequantGroupedAsymChunkToFloat(const tensor_t *src, size_t elemOffset, size_t count,
                                           float *out);
static void quantizeFloatToAsym(const float *values, size_t n, asymQConfig_t *outQC, uint8_t *dst);

_Static_assert(ODT_CONVERSION_CHUNK_ELEMS % 8 == 0,
               "chunk starts must stay byte-aligned for every packed qBits");
/* elemIndex*bits/8; caller guarantees elemIndex%8==0 (chunk starts and the
 * ODT_CONVERSION_CHUNK_ELEMS stride are byte-aligned for every packed qBits,
 * per the _Static_assert above). */
static size_t packedByteOffset(size_t elemIndex, size_t bits);
static void unpackSignExtendChunk(const uint8_t *srcBase, size_t srcBits, size_t elemOffset,
                                  size_t count, int32_t *dst);
/* ASYM codes: byteConversion only (no sign bit to restore). */
static void unpackZeroExtendChunk(const uint8_t *srcBase, size_t srcBits, size_t elemOffset,
                                  size_t count, int32_t *dst);
static void packChunkGuarded(const int32_t *codes, size_t count, uint8_t *dstBase, size_t dstBits,
                             size_t elemOffset, const char *what);
/* Factored out of quantizeFloatToAsym verbatim, incl. the single zeroPoint
 * roundByMode draw -- called BEFORE any per-element round (bit-identity
 * invariant: exactly one roundByMode call per element, in element order). */
static void deriveAsymGridFromMinMax(float mn, float mx, asymQConfig_t *outQC);
/* Group-quant PR4 Task 2: the nudge+derive core, writing group g's grid
 * (scales[g]/zeroPoints[g]). deriveAsymGridFromMinMax is the per-tensor
 * wrapper (qBits + per-tensor gates, then g=0); quantizeFloatToAsym's
 * grouped phase 1 calls this once per group. */
static void deriveAsymGridForGroup(float mn, float mx, asymQConfig_t *outQC, size_t g);
static void emitAsymChunk(const float *vals, size_t count, const asymQConfig_t *qc,
                          uint8_t *dstBase, size_t elemOffset);
/* Group-quant PR4 Task 2: the two PRIMARY ASYM cells -- quantizeFloatToAsym
 * (FLOAT32->ASYM) and convertAsymTensorToFloatTensor (ASYM->FLOAT32) -- are
 * group-aware; every OTHER ASYM converter/dequant/accumulate cell reads
 * scales[0]/zeroPoints[0] only and calls this per-tensor choke point first
 * (mirrors the SYM scalar-cell gates). */
static void requirePerTensorAsym(const asymQConfig_t *qc, const char *what);

void zeroTensorData(tensor_t *tensor) {
    size_t numberOfElements = calcNumberOfElementsByTensor(tensor);
    memset(tensor->data, 0, calcNumberOfBytesForData(tensor->quantization, numberOfElements));
}

void convertInt32TensorToFloatTensor(tensor_t *inputTensor, tensor_t *outputTensor) {
    size_t n = calcNumberOfElementsByTensor(inputTensor);
    float *out = (float *)outputTensor->data;
    int32_t inBuf[ODT_CONVERSION_CHUNK_ELEMS];
    for (size_t off = 0; off < n; off += ODT_CONVERSION_CHUNK_ELEMS) {
        size_t count = n - off < ODT_CONVERSION_CHUNK_ELEMS ? n - off : ODT_CONVERSION_CHUNK_ELEMS;
        /* alignment-safe staging, like readBytesAsInt32Array's whole-buffer memcpy */
        memcpy(inBuf, (const int32_t *)inputTensor->data + off, count * sizeof(int32_t));
        for (size_t i = 0; i < count; i++) {
            out[off + i] = (float)inBuf[i];
        }
    }
}

void convertInt32TensorToSymInt32Tensor(tensor_t *inputTensor, tensor_t *outputTensor) {
    size_t numberOfElements = calcNumberOfElementsByTensor(inputTensor);

    symInt32QConfig_t *outputSymInt32QConfig = outputTensor->quantization->qConfig;
    outputSymInt32QConfig->scale = 1;

    memcpy(outputTensor->data, inputTensor->data, numberOfElements * sizeof(int32_t));
}

static size_t packedByteOffset(size_t elemIndex, size_t bits) {
    /* Callers iterate in ODT_CONVERSION_CHUNK_ELEMS strides (multiple of 8),
     * so elemIndex*bits is always a whole number of bytes. */
    return elemIndex * bits / 8;
}

static void unpackSignExtendChunk(const uint8_t *srcBase, size_t srcBits, size_t elemOffset,
                                  size_t count, int32_t *dst) {
    unpackSignExtend(srcBase + packedByteOffset(elemOffset, srcBits), srcBits, 0, dst, count);
}

static void unpackZeroExtendChunk(const uint8_t *srcBase, size_t srcBits, size_t elemOffset,
                                  size_t count, int32_t *dst) {
    byteConversion((uint8_t *)(srcBase + packedByteOffset(elemOffset, srcBits)), srcBits,
                   (uint8_t *)dst, 32, count);
}

static void packChunkGuarded(const int32_t *codes, size_t count, uint8_t *dstBase, size_t dstBits,
                             size_t elemOffset, const char *what) {
    if (dstBits == 0 || dstBits > 31) {
        PRINT_ERROR("%s: dstBits (%u) must be in [1, 31]", what, (unsigned)dstBits);
        exit(1);
    }
    const int32_t hi = ((int32_t)1 << (dstBits - 1)) - 1;
    const int32_t lo = -((int32_t)1 << (dstBits - 1));
    for (size_t i = 0; i < count; i++) {
        if (codes[i] < lo || codes[i] > hi) {
            /* abort-on-overflow, process-fatal (#227 discipline; spec §3.2 —
             * earlier chunks of dst may already be written, which is fine
             * because exit(1) is not recoverable) */
            PRINT_ERROR("%s: value %d does not fit %u-bit SYM range [%d, %d] (#227)", what,
                        codes[i], (unsigned)dstBits, lo, hi);
            exit(1);
        }
    }
    byteConversion((uint8_t *)codes, 32, dstBase + packedByteOffset(elemOffset, dstBits), dstBits,
                   count);
}

void dequantChunkToFloat(const tensor_t *src, size_t elemOffset, size_t count, float *out) {
    if (count > ODT_CONVERSION_CHUNK_ELEMS || elemOffset % 8 != 0) {
        PRINT_ERROR("dequantChunkToFloat: count %zu > chunk (%d) or unaligned offset %zu", count,
                    ODT_CONVERSION_CHUNK_ELEMS, elemOffset);
        exit(1);
    }
    size_t srcElems = calcNumberOfElementsByTensor((tensor_t *)src);
    if (elemOffset > srcElems || count > srcElems - elemOffset) {
        PRINT_ERROR("dequantChunkToFloat: range [%zu, %zu) exceeds source tensor (%zu elements)",
                    elemOffset, elemOffset + count, srcElems);
        exit(1);
    }
    switch (src->quantization->type) {
    case FLOAT32:
        memcpy(out, (const float *)src->data + elemOffset, count * sizeof(float));
        return;
    case SYM_INT32: {
        float scale = ((symInt32QConfig_t *)src->quantization->qConfig)->scale;
        const int32_t *in = (const int32_t *)src->data + elemOffset;
        for (size_t i = 0; i < count; i++) {
            out[i] = (float)in[i] * scale;
        }
        return;
    }
    case SYM: {
        symQConfig_t *qc = src->quantization->qConfig;
        if (qc->numGroups > 1) {
            /* Group-quant PR2: this helper's only callers are the grad-accumulate
             * engines (accumulateIntoSym*Engine via incSrcChunk, and
             * accumulateTensorIntoFloat32Inplace), all of which operate on grad
             * tensors -- gradInit fail-fasts on a grouped SYM template (#300 axis),
             * so a grouped source here would indicate a caller contract violation,
             * not a reachable production path. Dequantize via
             * convertSymTensorToFloat32Tensor first if a grouped SYM source ever
             * needs to feed this helper. */
            PRINT_ERROR("dequantChunkToFloat: grouped SYM (numGroups=%zu) has no per-tensor "
                        "dequant image here; grad-accumulate paths only ever see per-tensor "
                        "SYM (gradInit rejects grouped templates) -- convert to FLOAT32 first "
                        "for any other route",
                        qc->numGroups);
            exit(1);
        }
        float scale = qc->scales[0]; /* hoisted: no per-element array indexing */
        int32_t mant[ODT_CONVERSION_CHUNK_ELEMS];
        unpackSignExtendChunk(src->data, qc->qBits, elemOffset, count, mant);
        for (size_t i = 0; i < count; i++) {
            out[i] = (float)mant[i] * scale;
        }
        return;
    }
    case ASYM: {
        asymQConfig_t *qc = src->quantization->qConfig;
        requirePerTensorAsym(qc, "dequantChunkToFloat");
        float scale = qc->scales[0]; /* hoisted like the SYM arm above */
        int32_t zp = (int32_t)qc->zeroPoints[0];
        int32_t codes[ODT_CONVERSION_CHUNK_ELEMS];
        unpackZeroExtendChunk(src->data, qc->qBits, elemOffset, count, codes);
        for (size_t i = 0; i < count; i++) {
            out[i] = (float)(codes[i] - zp) * scale;
        }
        return;
    }
    case BFP: {
        /* Grouped-capable by construction -- unlike the grouped-SYM deny
         * above, whose only callers are per-tensor grad paths. Native hot
         * path since epic PR2 (OUT_WRITE re-packs a BFP wire every producer
         * step), so the 2^E scale is derived once per group RUN
         * (dequantGroupedSymChunkToFloat's run-walking shape) instead of
         * ldexpf+bias per element -- same multiply per element in element
         * order, bit-identical. */
        bfpQConfig_t *qc = src->quantization->qConfig;
        int32_t mant[ODT_CONVERSION_CHUNK_ELEMS];
        unpackSignExtendChunk(src->data, qc->mantissaBits, elemOffset, count, mant);
        const size_t groupSize = qc->groupSize;
        if (groupSize == 0) {
            const float scale = bfpGroupScale(qc, 0);
            for (size_t i = 0; i < count; i++) {
                out[i] = (float)mant[i] * scale;
            }
            return;
        }
        size_t chunkEnd = elemOffset + count;
        size_t idx = elemOffset;
        while (idx < chunkEnd) {
            size_t g = idx / groupSize;
            size_t groupEnd = (g + 1) * groupSize;
            size_t runEnd = groupEnd < chunkEnd ? groupEnd : chunkEnd;
            size_t runLen = runEnd - idx;
            const float scale = bfpGroupScale(qc, g);
            for (size_t i = 0; i < runLen; i++) {
                out[idx - elemOffset + i] = (float)mant[idx - elemOffset + i] * scale;
            }
            idx = runEnd;
        }
        return;
    }
    default:
        PRINT_ERROR("dequantChunkToFloat: dtype %d not supported", (int)src->quantization->type);
        exit(1);
    }
}

static void requirePerTensorAsym(const asymQConfig_t *qc, const char *what) {
    if (qc->numGroups > 1) {
        PRINT_ERROR("%s: grouped ASYM (numGroups=%zu) has no compute image in this cell -- only "
                    "FLOAT32->ASYM and ASYM->FLOAT32 are group-aware; route through FLOAT32 "
                    "instead",
                    what, qc->numGroups);
        exit(1);
    }
}

/* Funnel re-check of the initAsymQConfig ceiling for field-assigned configs:
 * the code-domain zeroPoint is uint16, so 17+ has no zp representation (D6;
 * supersedes the [1, 30] #246 ceiling). */
static void requireAsymComputeQBits(const asymQConfig_t *qc, const char *what) {
    if (qc->qBits == 0 || qc->qBits > 16) {
        PRINT_ERROR("%s: qBits (%u) outside the ASYM range [1, 16] (D6)", what,
                    (unsigned)qc->qBits);
        exit(1);
    }
}

static void deriveAsymGridFromMinMax(float mn, float mx, asymQConfig_t *outQC) {
    requireAsymComputeQBits(outQC, "deriveAsymGridFromMinMax");
    requirePerTensorAsym(outQC, "deriveAsymGridFromMinMax");
    deriveAsymGridForGroup(mn, mx, outQC, 0);
}

static void deriveAsymGridForGroup(float mn, float mx, asymQConfig_t *outQC, size_t g) {
    const float qMax = powf(2, (float)outQC->qBits) - 1;
    /* Zero-inclusion nudge (D6, TFLite-standard): extend the band to contain
     * 0 so (a) 0.0 is exactly representable (code == zp decodes to exactly
     * 0.0f) and (b) zpReal = -mn/scale is bounded into [0, qMax] BY
     * CONSTRUCTION (mn <= 0 gives zpReal >= 0; mx >= 0 gives -mn <= mx - mn
     * so zpReal <= qMax). */
    mn = fminf(mn, 0.f);
    mx = fmaxf(mx, 0.f);
    float scale;
    if (mn == mx) {
        /* post-nudge mn == mx only for the all-zero buffer (mn >= 0 and
         * mx <= 0 force both to 0) -- keep a nonzero scale; zp derives to 0
         * below, so code 0 decodes to exactly 0.0f. */
        scale = 1.f;
    } else {
        scale = (mx - mn) / qMax;
    }
    float zpReal = -mn / scale;
    /* Belt-and-suspenders: the nudge bounds zpReal into [0, qMax] (proof
     * above), so this clamp only ever trims rounding at the band edges
     * (e.g. an exact .5 tie at qMax rounding up) -- it cannot mask a
     * genuinely out-of-band zpReal on any reachable path. */
    outQC->scales[g] = scale;
    outQC->zeroPoints[g] =
        (uint16_t)clampInt32(roundByMode(zpReal, outQC->roundingMode), 0, (int32_t)qMax);
}

static void emitAsymChunk(const float *vals, size_t count, const asymQConfig_t *qc,
                          uint8_t *dstBase, size_t elemOffset) {
    const float qMax = powf(2, (float)qc->qBits) - 1;
    const float scale = qc->scales[0];
    const int32_t zp = (int32_t)qc->zeroPoints[0];
    int32_t codes[ODT_CONVERSION_CHUNK_ELEMS];
    for (size_t i = 0; i < count; i++) {
        /* Code-domain encode (D6): round the VALUE quotient first, add the
         * integer zp after -- NOT the old single-round round(v/scale - zp),
         * whose HALF_AWAY ties land differently under the shift. The clamp
         * is load-bearing at the band edges: a .5 tie at both the zp
         * derivation and the max value (e.g. scale 1.0, band [-1.5, 5.5]@3)
         * rounds both up, overshooting qMax by 1. */
        codes[i] =
            clampInt32(roundByMode(vals[i] / scale, qc->roundingMode) + zp, 0, (int32_t)qMax);
    }
    byteConversion((uint8_t *)codes, 32, dstBase + packedByteOffset(elemOffset, qc->qBits),
                   qc->qBits, count);
}

/* Nudged code-domain asymmetric quantization (group-quant PR4, D6; replaces
 * the #243 value-domain grid): band nudged to include 0, scale =
 * (max-min)/(2^qBits-1), zeroPoint = clamp(round(-min/scale), 0, 2^qBits-1)
 * [uint16 code domain], code = clamp(round(v/scale) + zp, 0, 2^qBits-1).
 * Dequant (elsewhere) is (code - zp)*scale. Constant tensor: after the nudge
 * min==max only for the all-zero buffer -> scale 1.f, zp 0. Grid derivation
 * scans the WHOLE buffer once (min/max, no rounding); emission then streams
 * in ODT_CONVERSION_CHUNK_ELEMS chunks so no VLA/heap scratch scales with n
 * (#296 Stage 2). numGroups > 1 (PR4 Task 2, FLOAT32 source only -- the
 * other three *ToAsymTensor converters stay per-tensor-gated): the same
 * affine applied per storage-order group (group of element i = i/groupSize),
 * packFloatBufferAsSym's grouped shape -- phase-1 per-group min/max ->
 * per-group nudged grid, phase-2 run-based sequential emit. */
static void quantizeFloatToAsym(const float *values, size_t n, asymQConfig_t *outQC, uint8_t *dst) {
    if (n == 0) {
        /* n == 0: no grid can be derived from an empty payload -- leave the
         * caller's qConfig untouched, write nothing (same no-op contract as
         * the sibling n==0 guards; the old code read values[0] here, UB). */
        return;
    }
    if (outQC->numGroups == 1) {
        /* Per-tensor fast path -- byte-identical to the pre-group-quant code
         * (regression gate: the Task-1 re-pinned suite pins this verbatim). */
        float mn = findMinFloat((uint8_t *)values, n);
        float mx = findMaxFloat((uint8_t *)values, n);
        deriveAsymGridFromMinMax(mn, mx, outQC);
        for (size_t off = 0; off < n; off += ODT_CONVERSION_CHUNK_ELEMS) {
            size_t count =
                n - off < ODT_CONVERSION_CHUNK_ELEMS ? n - off : ODT_CONVERSION_CHUNK_ELEMS;
            emitAsymChunk(values + off, count, outQC, dst, off);
        }
        return;
    }

    /* Grouped path (#groups>1). Phase 1: per-group min/max -> per-group
     * nudged grid (never a single whole-tensor pass, or every group would
     * collapse onto one grid). qBits funnel re-check hoisted out of the loop
     * (deriveAsymGridForGroup trusts it). */
    requireAsymComputeQBits(outQC, "quantizeFloatToAsym");
    const size_t groupSize = outQC->groupSize;
    for (size_t g = 0; g < outQC->numGroups; g++) {
        float mn = findMinFloat((uint8_t *)(values + g * groupSize), groupSize);
        float mx = findMaxFloat((uint8_t *)(values + g * groupSize), groupSize);
        deriveAsymGridForGroup(mn, mx, outQC, g);
    }
    /* Phase 2: sequential encode+pack, chunked exactly like the per-tensor
     * path above. WITHIN a chunk, walk per-RUN -- a run is the span from the
     * current index to min(chunkEnd, groupEnd) -- so the grid is fetched
     * once per run (one `idx / groupSize` division), never per element
     * (packFloatBufferAsSym's grouped shape; encode math is emitAsymChunk's
     * verbatim, with scales[g]/zeroPoints[g] instead of element 0). */
    const float qMax = powf(2, (float)outQC->qBits) - 1;
    int32_t codes[ODT_CONVERSION_CHUNK_ELEMS];
    for (size_t off = 0; off < n; off += ODT_CONVERSION_CHUNK_ELEMS) {
        size_t count = n - off < ODT_CONVERSION_CHUNK_ELEMS ? n - off : ODT_CONVERSION_CHUNK_ELEMS;
        size_t chunkEnd = off + count;
        size_t idx = off;
        while (idx < chunkEnd) {
            size_t g = idx / groupSize;
            size_t groupEnd = (g + 1) * groupSize;
            size_t runEnd = groupEnd < chunkEnd ? groupEnd : chunkEnd;
            size_t runLen = runEnd - idx;
            const float scale = outQC->scales[g];
            const int32_t zp = (int32_t)outQC->zeroPoints[g];
            for (size_t i = 0; i < runLen; i++) {
                codes[idx - off + i] =
                    clampInt32(roundByMode(values[idx + i] / scale, outQC->roundingMode) + zp, 0,
                               (int32_t)qMax);
            }
            idx = runEnd;
        }
        byteConversion((uint8_t *)codes, 32, dst + packedByteOffset(off, outQC->qBits),
                       outQC->qBits, count);
    }
}

void convertInt32TensorToAsymTensor(tensor_t *inputTensor, tensor_t *outputTensor) {
    size_t n = calcNumberOfElementsByTensor(inputTensor);
    asymQConfig_t *outQC = outputTensor->quantization->qConfig;
    if (n == 0) {
        /* No first element to seed mn/mx from; old code's vals[0] VLA read (via
         * quantizeFloatToAsym -> findMinFloat/findMaxFloat) was UB. New code
         * no-ops (matches the SYM_INT32/SYM -> ASYM siblings' n=0 guard; #296
         * Stage 2). */
        return;
    }
    const int32_t *in = (const int32_t *)inputTensor->data;
    /* pass 1: min/max over (float)in[i], direct loop -- input is already a flat
     * int32 array, no unpack staging needed; no rounding in this pass. */
    float mn = (float)in[0];
    float mx = mn;
    for (size_t i = 1; i < n; i++) {
        float v = (float)in[i];
        if (v < mn) {
            mn = v;
        }
        if (v > mx) {
            mx = v;
        }
    }
    deriveAsymGridFromMinMax(mn, mx, outQC);
    /* pass 2: chunked emit -- one roundByMode per element (inside emitAsymChunk),
     * element order */
    float vals[ODT_CONVERSION_CHUNK_ELEMS];
    for (size_t off = 0; off < n; off += ODT_CONVERSION_CHUNK_ELEMS) {
        size_t count = n - off < ODT_CONVERSION_CHUNK_ELEMS ? n - off : ODT_CONVERSION_CHUNK_ELEMS;
        for (size_t i = 0; i < count; i++) {
            vals[i] = (float)in[off + i];
        }
        emitAsymChunk(vals, count, outQC, outputTensor->data, off);
    }
}

void convertFloatTensorToInt32Tensor(tensor_t *inputTensor, tensor_t *outputTensor) {
    size_t n = calcNumberOfElementsByTensor(inputTensor);
    int32_t *out = (int32_t *)outputTensor->data;
    float inBuf[ODT_CONVERSION_CHUNK_ELEMS];
    for (size_t off = 0; off < n; off += ODT_CONVERSION_CHUNK_ELEMS) {
        size_t count = n - off < ODT_CONVERSION_CHUNK_ELEMS ? n - off : ODT_CONVERSION_CHUNK_ELEMS;
        /* alignment-safe staging, like readBytesAsFloatArray's whole-buffer memcpy */
        memcpy(inBuf, (const float *)inputTensor->data + off, count * sizeof(float));
        for (size_t i = 0; i < count; i++) {
            out[off + i] = (int32_t)inBuf[i]; /* cast semantics preserved verbatim */
        }
    }
}

void convertFloatTensorToSymInt32Tensor(tensor_t *inputTensor, tensor_t *outputTensor) {
    size_t numberOfElements = calcNumberOfElementsByTensor(inputTensor);

    float absMax = findAbsMaxFloat(inputTensor->data, numberOfElements);

    symInt32QConfig_t *symInt32QC = outputTensor->quantization->qConfig;
    uint8_t qMaxBits = symInt32QC->qMaxBits;

    const float qMax = powf(2, (float)qMaxBits - 1) - 1;
    const float qMin = -powf(2, (float)qMaxBits - 1);

    float scale;
    if (absMax == 0.f) {
        scale = 1.f;
    } else {
        scale = absMax / qMax;
    }

    symInt32QConfig_t *outputSymInt32QC = outputTensor->quantization->qConfig;
    outputSymInt32QC->scale = scale;

    int32_t *outputInt32 = (int32_t *)outputTensor->data;
    float *inputFloat = (float *)inputTensor->data;

    for (size_t i = 0; i < numberOfElements; i++) {
        outputInt32[i] =
            clampInt32(roundByMode(inputFloat[i] / scale, outputSymInt32QC->roundingMode),
                       (int32_t)qMin, (int32_t)qMax);
    }
}

void convertInt32TensorToSymTensor(tensor_t *inputTensor, tensor_t *outputTensor) {
    size_t n = calcNumberOfElementsByTensor(inputTensor);
    symQConfig_t *outQC = outputTensor->quantization->qConfig;
    if (outQC->numGroups > 1) {
        PRINT_ERROR("convertInt32TensorToSymTensor: grouped SYM target (numGroups=%zu) has no "
                    "scalar compute image here; target a per-tensor SYM config, or convert "
                    "INT32->FLOAT32 then FLOAT32->SYM(grouped) instead",
                    outQC->numGroups);
        exit(1);
    }
    outQC->scales[0] = 1.f;
    int32_t codes[ODT_CONVERSION_CHUNK_ELEMS];
    for (size_t off = 0; off < n; off += ODT_CONVERSION_CHUNK_ELEMS) {
        size_t count = n - off < ODT_CONVERSION_CHUNK_ELEMS ? n - off : ODT_CONVERSION_CHUNK_ELEMS;
        /* alignment-safe staging, like readBytesAsInt32Array's whole-buffer memcpy */
        memcpy(codes, (const int32_t *)inputTensor->data + off, count * sizeof(int32_t));
        packChunkGuarded(codes, count, outputTensor->data, outQC->qBits, off,
                         "convertInt32TensorToSymTensor");
    }
}

void convertFloatTensorToSymTensor(tensor_t *inputTensor, tensor_t *outputTensor) {
    size_t n = calcNumberOfElementsByTensor(inputTensor);
    symQConfig_t *outQC = outputTensor->quantization->qConfig;
    packFloatBufferAsSym((float *)inputTensor->data, n, outQC, outputTensor->data,
                         "convertFloatTensorToSymTensor");
}

void convertFloatTensorToAsymTensor(tensor_t *inputTensor, tensor_t *outputTensor) {
    size_t numberOfElements = calcNumberOfElementsByTensor(inputTensor);
    asymQConfig_t *asymQConfig = outputTensor->quantization->qConfig;
    quantizeFloatToAsym((float *)inputTensor->data, numberOfElements, asymQConfig,
                        outputTensor->data);
}

// Important: Scale is ignored!
void extractInt32TensorFromSymInt32Tensor(tensor_t *inputTensor, tensor_t *outputTensor) {
    size_t n = calcNumberOfElementsByTensor(inputTensor);
    memcpy(outputTensor->data, inputTensor->data, n * sizeof(int32_t));
}

void convertSymInt32TensorToFloat32Tensor(tensor_t *inputTensor, tensor_t *outputTensor) {
    size_t n = calcNumberOfElementsByTensor(inputTensor);
    const int32_t *in = (const int32_t *)inputTensor->data;
    float *out = (float *)outputTensor->data;
    float scale = ((symInt32QConfig_t *)inputTensor->quantization->qConfig)->scale;
    /* same-index read-then-write: safe for the in-place (shared-buffer) case */
    for (size_t i = 0; i < n; i++) {
        out[i] = (float)in[i] * scale;
    }
}

void requantSymInt32Tensor(tensor_t *inputTensor, tensor_t *outputTensor) {
    size_t numberOfElements = calcNumberOfElementsByTensor(inputTensor);

    symInt32QConfig_t *inputSymInt32QC = inputTensor->quantization->qConfig;
    symInt32QConfig_t *outputSymInt32QC = outputTensor->quantization->qConfig;
    /* latch BEFORE writing outputSymInt32QC->scale: when called in-place
     * (inputTensor == outputTensor) both pointers alias the same config */
    float inScale = inputSymInt32QC->scale;

    const float qMax = powf(2, (float)outputSymInt32QC->qMaxBits - 1) - 1;
    const float qMin = -powf(2, (float)outputSymInt32QC->qMaxBits - 1);

    int32_t *inputInt32 = (int32_t *)inputTensor->data;
    int32_t *outputInt32 = (int32_t *)outputTensor->data;

    /* pass A: absmax over dequantized values — reads only (alias-safe) */
    float absMax = 0.f;
    for (size_t i = 0; i < numberOfElements; i++) {
        float dequant = fabsf((float)inputInt32[i] * inScale);
        if (dequant > absMax) {
            absMax = dequant;
        }
    }

    float scale;
    if (absMax == 0.f) {
        scale = 1.f;
    } else {
        scale = absMax / qMax;
    }
    outputSymInt32QC->scale = scale;

    /* pass B: same-index read-then-write — in-place safe (int32 both sides) */
    for (size_t i = 0; i < numberOfElements; i++) {
        outputInt32[i] = clampInt32(
            roundByMode((float)inputInt32[i] * (inScale / scale), outputSymInt32QC->roundingMode),
            (int32_t)qMin, (int32_t)qMax);
    }
}

void requantSymInt32TensorToScale(tensor_t *inputTensor, tensor_t *outputTensor) {
    size_t numberOfElements = calcNumberOfElementsByTensor(inputTensor);

    symInt32QConfig_t *inputSymInt32QC = inputTensor->quantization->qConfig;
    symInt32QConfig_t *outputSymInt32QC = outputTensor->quantization->qConfig;
    float inScale = inputSymInt32QC->scale;
    float targetScale = outputSymInt32QC->scale;

    /* NaN-robust: !(x > 0.f) is also true for NaN, unlike (x <= 0.f) */
    if (!(targetScale > 0.f)) {
        PRINT_ERROR("requantSymInt32TensorToScale: target scale must be pre-set and > 0 on "
                    "the output qConfig, got %f",
                    targetScale);
        exit(1);
    }

    const float qMax = powf(2, (float)outputSymInt32QC->qMaxBits - 1) - 1;
    const float qMin = -powf(2, (float)outputSymInt32QC->qMaxBits - 1);

    int32_t *inputInt32 = (int32_t *)inputTensor->data;
    int32_t *outputInt32 = (int32_t *)outputTensor->data;

    /* single same-index read-then-write pass — shared-buffer in-place safe;
     * clamp saturates at qMin/qMax BY DESIGN (Deutel Eq. 4 analog) */
    for (size_t i = 0; i < numberOfElements; i++) {
        outputInt32[i] =
            roundByMode(clamp(((float)inputInt32[i] * inScale) / targetScale, qMin, qMax),
                        outputSymInt32QC->roundingMode);
    }
}

void convertSymInt32TensorToAsymTensor(tensor_t *inputTensor, tensor_t *outputTensor) {
    size_t n = calcNumberOfElementsByTensor(inputTensor);
    symInt32QConfig_t *inQC = inputTensor->quantization->qConfig;
    asymQConfig_t *outQC = outputTensor->quantization->qConfig;
    if (n == 0) {
        /* No first element to seed mn/mx from; old code's inputAsFloat[0] VLA
         * read (via quantizeFloatToAsym -> findMinFloat/findMaxFloat) was UB.
         * New code no-ops (matches packFloatBufferAsSym's n=0 no-op; #296 Stage 2). */
        return;
    }
    float scale = inQC->scale;
    const int32_t *in = (const int32_t *)inputTensor->data;
    /* pass 1: min/max over dequantized values, direct loop -- input is already a
     * flat int32 array, no unpack staging needed */
    float mn = (float)in[0] * scale;
    float mx = mn;
    for (size_t i = 1; i < n; i++) {
        float v = (float)in[i] * scale;
        if (v < mn) {
            mn = v;
        }
        if (v > mx) {
            mx = v;
        }
    }
    deriveAsymGridFromMinMax(mn, mx, outQC);
    /* pass 2: chunked emit -- one roundByMode per element (inside emitAsymChunk),
     * element order */
    float vals[ODT_CONVERSION_CHUNK_ELEMS];
    for (size_t off = 0; off < n; off += ODT_CONVERSION_CHUNK_ELEMS) {
        size_t count = n - off < ODT_CONVERSION_CHUNK_ELEMS ? n - off : ODT_CONVERSION_CHUNK_ELEMS;
        for (size_t i = 0; i < count; i++) {
            vals[i] = (float)in[off + i] * scale;
        }
        emitAsymChunk(vals, count, outQC, outputTensor->data, off);
    }
}

void convertAsymTensorToInt32Tensor(tensor_t *inputTensor, tensor_t *outputTensor) {
    size_t n = calcNumberOfElementsByTensor(inputTensor);
    asymQConfig_t *inQC = inputTensor->quantization->qConfig;
    requirePerTensorAsym(inQC, "convertAsymTensorToInt32Tensor");
    const int32_t zp = (int32_t)inQC->zeroPoints[0];
    int32_t *out = (int32_t *)outputTensor->data;
    int32_t codes[ODT_CONVERSION_CHUNK_ELEMS];
    for (size_t off = 0; off < n; off += ODT_CONVERSION_CHUNK_ELEMS) {
        size_t count = n - off < ODT_CONVERSION_CHUNK_ELEMS ? n - off : ODT_CONVERSION_CHUNK_ELEMS;
        unpackZeroExtendChunk(inputTensor->data, inQC->qBits, off, count, codes);
        for (size_t i = 0; i < count; i++) {
            /* code-domain mantissa image (D6): code - zp (was code + zp
             * under the old value-domain signed zeroPoint) */
            out[off + i] = codes[i] - zp;
        }
    }
}

void convertAsymTensorToFloatTensor(tensor_t *inputTensor, tensor_t *outputTensor) {
    size_t n = calcNumberOfElementsByTensor(inputTensor);
    asymQConfig_t *inQC = inputTensor->quantization->qConfig;
    float *out = (float *)outputTensor->data;
    int32_t codes[ODT_CONVERSION_CHUNK_ELEMS];

    if (inQC->numGroups == 1) {
        /* Per-tensor fast path -- byte-identical to the pre-group-quant code. */
        const float scale = inQC->scales[0];
        const int32_t zp = (int32_t)inQC->zeroPoints[0];
        for (size_t off = 0; off < n; off += ODT_CONVERSION_CHUNK_ELEMS) {
            size_t count =
                n - off < ODT_CONVERSION_CHUNK_ELEMS ? n - off : ODT_CONVERSION_CHUNK_ELEMS;
            unpackZeroExtendChunk(inputTensor->data, inQC->qBits, off, count, codes);
            for (size_t i = 0; i < count; i++) {
                /* code-domain decode (D6): (code - zp)*scale, integer subtract
                 * exact (both operands <= 2^16-1) */
                out[off + i] = (float)(codes[i] - zp) * scale;
            }
        }
        return;
    }

    /* Grouped path (PR4 Task 2): same run-walking pattern as the grouped
     * convertSymTensorToFloat32Tensor, with the affine decode per run --
     * out[i] = (code - zeroPoints[g]) * scales[g], grid fetched once per run. */
    const size_t groupSize = inQC->groupSize;
    for (size_t off = 0; off < n; off += ODT_CONVERSION_CHUNK_ELEMS) {
        size_t count = n - off < ODT_CONVERSION_CHUNK_ELEMS ? n - off : ODT_CONVERSION_CHUNK_ELEMS;
        unpackZeroExtendChunk(inputTensor->data, inQC->qBits, off, count, codes);
        size_t chunkEnd = off + count;
        size_t idx = off;
        while (idx < chunkEnd) {
            size_t g = idx / groupSize;
            size_t groupEnd = (g + 1) * groupSize;
            size_t runEnd = groupEnd < chunkEnd ? groupEnd : chunkEnd;
            size_t runLen = runEnd - idx;
            const float scale = inQC->scales[g];
            const int32_t zp = (int32_t)inQC->zeroPoints[g];
            for (size_t i = 0; i < runLen; i++) {
                out[idx + i] = (float)(codes[idx - off + i] - zp) * scale;
            }
            idx = runEnd;
        }
    }
}

void convertAsymTensorToSymInt32Tensor(tensor_t *inputTensor, tensor_t *outputTensor) {
    size_t n = calcNumberOfElementsByTensor(inputTensor);
    asymQConfig_t *inQC = inputTensor->quantization->qConfig;
    requirePerTensorAsym(inQC, "convertAsymTensorToSymInt32Tensor");
    const int32_t zp = (int32_t)inQC->zeroPoints[0];
    symInt32QConfig_t *outQC = outputTensor->quantization->qConfig;
    int32_t *out = (int32_t *)outputTensor->data;
    int32_t codes[ODT_CONVERSION_CHUNK_ELEMS];
    for (size_t off = 0; off < n; off += ODT_CONVERSION_CHUNK_ELEMS) {
        size_t count = n - off < ODT_CONVERSION_CHUNK_ELEMS ? n - off : ODT_CONVERSION_CHUNK_ELEMS;
        unpackZeroExtendChunk(inputTensor->data, inQC->qBits, off, count, codes);
        for (size_t i = 0; i < count; i++) {
            /* code-domain mantissa image (D6): code - zp */
            out[off + i] = codes[i] - zp;
        }
    }
    outQC->scale = inQC->scales[0]; /* scale copy unchanged */
}

void unpackSignExtend(const uint8_t *src, size_t srcBits, size_t srcStartBit, int32_t *dst,
                      size_t n) {
    if (srcBits == 0) {
        /* 1 << (srcBits - 1) below underflows size_t to SIZE_MAX -> UB shift (#247). */
        PRINT_ERROR("unpackSignExtend: srcBits must be > 0");
        exit(1);
    }
    /* clear-then-set writeByte actively zero-fills the high bits on widen,
     * so no memset of dst is needed. */
    byteConversionAppend((uint8_t *)src, srcBits, (uint8_t *)dst, 32, n, 0, srcStartBit);
    if (srcBits >= 32) {
        return;
    }
    const int32_t signBit = (int32_t)1 << (srcBits - 1);
    const int32_t mask = (int32_t)(((uint32_t)1 << srcBits) - 1u);
    for (size_t i = 0; i < n; i++) {
        int32_t v = dst[i] & mask;
        dst[i] = (v ^ signBit) - signBit; /* sign-extend from srcBits */
    }
}

// Important: Scale is ignored! Emits sign-extended integer codes (int_repr).
void convertSymTensorToInt32Tensor(tensor_t *inputTensor, tensor_t *outputTensor) {
    size_t n = calcNumberOfElementsByTensor(inputTensor);
    symQConfig_t *inQC = inputTensor->quantization->qConfig;
    unpackSignExtend(inputTensor->data, inQC->qBits, 0, (int32_t *)outputTensor->data, n);
}

void convertSymTensorToFloat32Tensor(tensor_t *inputTensor, tensor_t *outputTensor) {
    size_t n = calcNumberOfElementsByTensor(inputTensor);
    symQConfig_t *inQC = inputTensor->quantization->qConfig;
    float *out = (float *)outputTensor->data;
    int32_t mant[ODT_CONVERSION_CHUNK_ELEMS];

    if (inQC->numGroups == 1) {
        /* Per-tensor fast path -- byte-identical to the pre-group-quant code. */
        float scale = inQC->scales[0]; /* hoisted: no per-element array indexing */
        for (size_t off = 0; off < n; off += ODT_CONVERSION_CHUNK_ELEMS) {
            size_t count =
                n - off < ODT_CONVERSION_CHUNK_ELEMS ? n - off : ODT_CONVERSION_CHUNK_ELEMS;
            unpackSignExtendChunk(inputTensor->data, inQC->qBits, off, count, mant);
            for (size_t i = 0; i < count; i++) {
                out[off + i] = (float)mant[i] * scale;
            }
        }
        return;
    }

    /* Grouped path: same run-walking pattern as packFloatBufferAsSym's
     * grouped path, mirrored for dequant (out[i] = mant[i] * scales[g]). */
    const size_t groupSize = inQC->groupSize;
    for (size_t off = 0; off < n; off += ODT_CONVERSION_CHUNK_ELEMS) {
        size_t count = n - off < ODT_CONVERSION_CHUNK_ELEMS ? n - off : ODT_CONVERSION_CHUNK_ELEMS;
        unpackSignExtendChunk(inputTensor->data, inQC->qBits, off, count, mant);
        size_t chunkEnd = off + count;
        size_t idx = off;
        while (idx < chunkEnd) {
            size_t g = idx / groupSize;
            size_t groupEnd = (g + 1) * groupSize;
            size_t runEnd = groupEnd < chunkEnd ? groupEnd : chunkEnd;
            size_t runLen = runEnd - idx;
            const float scale = inQC->scales[g];
            for (size_t i = 0; i < runLen; i++) {
                out[idx + i] = (float)mant[idx - off + i] * scale;
            }
            idx = runEnd;
        }
    }
}

void convertSymTensorToSymInt32Tensor(tensor_t *inputTensor, tensor_t *outputTensor) {
    size_t n = calcNumberOfElementsByTensor(inputTensor);
    symQConfig_t *inQC = inputTensor->quantization->qConfig;
    symInt32QConfig_t *outQC = outputTensor->quantization->qConfig;
    if (inQC->numGroups > 1) {
        PRINT_ERROR("convertSymTensorToSymInt32Tensor: grouped SYM (numGroups=%zu) has no "
                    "scalar compute image; group-aware ops consume it via the executeOp "
                    "grouped-operand path (later PR of this epic), dequant to FLOAT32 otherwise",
                    inQC->numGroups);
        exit(1);
    }

    unpackSignExtend(inputTensor->data, inQC->qBits, 0, (int32_t *)outputTensor->data, n);
    outQC->scale = inQC->scales[0];
    outQC->qMaxBits = inQC->qBits;
}

void convertSymTensorToAsymTensor(tensor_t *inputTensor, tensor_t *outputTensor) {
    size_t n = calcNumberOfElementsByTensor(inputTensor);
    symQConfig_t *inQC = inputTensor->quantization->qConfig;
    asymQConfig_t *outQC = outputTensor->quantization->qConfig;
    if (inQC->numGroups > 1) {
        PRINT_ERROR("convertSymTensorToAsymTensor: grouped SYM (numGroups=%zu) has no scalar "
                    "compute image; dequantize to FLOAT32 first (convertSymTensorToFloat32Tensor), "
                    "then FLOAT32->ASYM",
                    inQC->numGroups);
        exit(1);
    }
    if (n == 0) {
        /* No first element to seed mn/mx from; old code's unpackSignExtend +
         * deq[0] VLA read (via quantizeFloatToAsym) was UB. New code no-ops
         * (matches packFloatBufferAsSym's n=0 no-op; #296 Stage 2). */
        return;
    }
    float scale = inQC->scales[0]; /* hoisted: no per-element array indexing */
    int32_t mant[ODT_CONVERSION_CHUNK_ELEMS];
    /* pass 1: min/max over dequantized values, chunked unpack (no O(n) scratch) */
    float mn = 0.f, mx = 0.f;
    bool seeded = false;
    for (size_t off = 0; off < n; off += ODT_CONVERSION_CHUNK_ELEMS) {
        size_t count = n - off < ODT_CONVERSION_CHUNK_ELEMS ? n - off : ODT_CONVERSION_CHUNK_ELEMS;
        unpackSignExtendChunk(inputTensor->data, inQC->qBits, off, count, mant);
        for (size_t i = 0; i < count; i++) {
            float v = (float)mant[i] * scale;
            if (!seeded) {
                mn = v;
                mx = v;
                seeded = true;
            } else {
                if (v < mn) {
                    mn = v;
                }
                if (v > mx) {
                    mx = v;
                }
            }
        }
    }
    deriveAsymGridFromMinMax(mn, mx, outQC);
    /* pass 2: chunked unpack + emit -- one roundByMode per element (inside
     * emitAsymChunk), element order */
    float vals[ODT_CONVERSION_CHUNK_ELEMS];
    for (size_t off = 0; off < n; off += ODT_CONVERSION_CHUNK_ELEMS) {
        size_t count = n - off < ODT_CONVERSION_CHUNK_ELEMS ? n - off : ODT_CONVERSION_CHUNK_ELEMS;
        unpackSignExtendChunk(inputTensor->data, inQC->qBits, off, count, mant);
        for (size_t i = 0; i < count; i++) {
            vals[i] = (float)mant[i] * scale;
        }
        emitAsymChunk(vals, count, outQC, outputTensor->data, off);
    }
}

void convertAsymTensorToSymTensor(tensor_t *inputTensor, tensor_t *outputTensor) {
    size_t n = calcNumberOfElementsByTensor(inputTensor);
    asymQConfig_t *inQC = inputTensor->quantization->qConfig;
    requirePerTensorAsym(inQC, "convertAsymTensorToSymTensor");
    const float inScale = inQC->scales[0];
    const int32_t inZp = (int32_t)inQC->zeroPoints[0];
    size_t inBits = calcBitsPerElement(inputTensor->quantization);
    symQConfig_t *outQC = outputTensor->quantization->qConfig;
    if (outQC->numGroups > 1) {
        PRINT_ERROR("convertAsymTensorToSymTensor: grouped SYM target (numGroups=%zu) has no "
                    "scalar compute image; target a per-tensor SYM config, or dequantize "
                    "ASYM->FLOAT32 first then FLOAT32->SYM(grouped)",
                    outQC->numGroups);
        exit(1);
    }
    int32_t codes[ODT_CONVERSION_CHUNK_ELEMS];
    /* pass 1: absmax over dequantized values, chunked unpack (no O(n) scratch) */
    float absMax = 0.f;
    for (size_t off = 0; off < n; off += ODT_CONVERSION_CHUNK_ELEMS) {
        size_t count = n - off < ODT_CONVERSION_CHUNK_ELEMS ? n - off : ODT_CONVERSION_CHUNK_ELEMS;
        unpackZeroExtendChunk(inputTensor->data, inBits, off, count, codes);
        for (size_t i = 0; i < count; i++) {
            float v = (float)(codes[i] - inZp) * inScale;
            if (fabsf(v) > absMax) {
                absMax = fabsf(v);
            }
        }
    }
    const float qMax = powf(2, (float)outQC->qBits - 1) - 1;
    const float qMin = -powf(2, (float)outQC->qBits - 1);
    float outScale = (absMax == 0.f) ? 1.f : absMax / qMax;
    outQC->scales[0] = outScale;
    /* pass 2: chunked unpack + emit -- one roundByMode per element, element order */
    for (size_t off = 0; off < n; off += ODT_CONVERSION_CHUNK_ELEMS) {
        size_t count = n - off < ODT_CONVERSION_CHUNK_ELEMS ? n - off : ODT_CONVERSION_CHUNK_ELEMS;
        unpackZeroExtendChunk(inputTensor->data, inBits, off, count, codes);
        for (size_t i = 0; i < count; i++) {
            float v = (float)(codes[i] - inZp) * inScale;
            codes[i] = clampInt32(roundByMode(v / outScale, outQC->roundingMode), (int32_t)qMin,
                                  (int32_t)qMax);
        }
        packChunkGuarded(codes, count, outputTensor->data, outQC->qBits, off,
                         "convertAsymTensorToSymTensor");
    }
}

char *quantTypeToString(qtype_t t) {
    switch (t) {
    case INT32:
        return "INT32";
    case FLOAT32:
        return "FLOAT32";
    case SYM_INT32:
        return "SYMINT32";
    case SYM:
        return "SYM";
    case ASYM:
        return "ASYM";
    case BOOL:
        return "BOOL";
    case BFP:
        return "BFP";
    default:
        return "UNKNOWN";
    }
}

void unsupportedConversionTypes(tensor_t *inputTensor, tensor_t *outputTensor) {
    qtype_t inputQType = inputTensor->quantization->type;
    qtype_t outputQType = outputTensor->quantization->type;

    PRINT_ERROR("Conversion from %s to %s is not supported", quantTypeToString(inputQType),
                quantTypeToString(outputQType));
    exit(1);
}

static void packFloatBufferAsSym(const float *values, size_t n, symQConfig_t *outQC, uint8_t *dst,
                                 const char *what) {
    const float qMax = powf(2, (float)outQC->qBits - 1) - 1;
    const float qMin = -powf(2, (float)outQC->qBits - 1);
    int32_t codes[ODT_CONVERSION_CHUNK_ELEMS];

    if (outQC->numGroups == 1) {
        /* Per-tensor fast path -- byte-identical to the pre-group-quant code
         * (regression gate: the existing suite pins this verbatim). */
        float absMax = findAbsMaxFloat((uint8_t *)values, n);
        float scale = (absMax == 0.f) ? 1.f : absMax / qMax;
        outQC->scales[0] = scale;
        for (size_t off = 0; off < n; off += ODT_CONVERSION_CHUNK_ELEMS) {
            size_t count =
                n - off < ODT_CONVERSION_CHUNK_ELEMS ? n - off : ODT_CONVERSION_CHUNK_ELEMS;
            for (size_t i = 0; i < count; i++) {
                codes[i] = clampInt32(roundByMode(values[off + i] / scale, outQC->roundingMode),
                                      (int32_t)qMin, (int32_t)qMax);
            }
            packChunkGuarded(codes, count, dst, outQC->qBits, off, what);
        }
        return;
    }

    /* Grouped path (#groups>1): groups are groupSize consecutive elements in
     * STORAGE order (group of element i = i / groupSize). Phase 1: per-group
     * absmax -> scales[g], one findAbsMaxFloat call per group (never a single
     * whole-tensor pass, or every group would collapse onto one scale). */
    const size_t groupSize = outQC->groupSize;
    for (size_t g = 0; g < outQC->numGroups; g++) {
        float absMax = findAbsMaxFloat((uint8_t *)(values + g * groupSize), groupSize);
        outQC->scales[g] = (absMax == 0.f) ? 1.f : absMax / qMax;
    }
    /* Phase 2: sequential quantize+pack, chunked exactly like the per-tensor
     * path above (packChunkGuarded keeps its byte-alignment contract: `off`
     * is always a multiple of ODT_CONVERSION_CHUNK_ELEMS, itself a multiple
     * of 8). WITHIN a chunk, walk per-RUN -- a run is the span from the
     * current index to min(chunkEnd, groupEnd) -- so the scale is derived
     * once per run (one `idx / groupSize` division), never per element. */
    for (size_t off = 0; off < n; off += ODT_CONVERSION_CHUNK_ELEMS) {
        size_t count = n - off < ODT_CONVERSION_CHUNK_ELEMS ? n - off : ODT_CONVERSION_CHUNK_ELEMS;
        size_t chunkEnd = off + count;
        size_t idx = off;
        while (idx < chunkEnd) {
            size_t g = idx / groupSize;
            size_t groupEnd = (g + 1) * groupSize;
            size_t runEnd = groupEnd < chunkEnd ? groupEnd : chunkEnd;
            size_t runLen = runEnd - idx;
            const float scale = outQC->scales[g];
            for (size_t i = 0; i < runLen; i++) {
                codes[idx - off + i] =
                    clampInt32(roundByMode(values[idx + i] / scale, outQC->roundingMode),
                               (int32_t)qMin, (int32_t)qMax);
            }
            idx = runEnd;
        }
        packChunkGuarded(codes, count, dst, outQC->qBits, off, what);
    }
}

/* Smallest E with absMax / 2^E <= qMax, derived via frexpf so exact powers of
 * two land on the boundary with no log2f rounding surprises: frexpf returns
 * ratio = frac * 2^e with frac in [0.5, 1), so ceil(log2(ratio)) is e except
 * when ratio is itself a power of two (frac == 0.5), where it is e - 1. The
 * stored-range clamp IS the D6 saturation: stored > max -> the emit pass
 * clamps mantissas to +-qMax (high regime); stored < 0 -> quotients round to
 * 0 (flush-toward-zero regime). The high clamp additionally never exceeds
 * bias + 127 (only reachable at exponentBits=8, where the natural top 255
 * means E=128): 2^127 is the largest FINITE float32 power of two --
 * ldexpf(1, 128) is +inf, which would quantize every code to 0 and dequantize
 * the whole group to NaN (0 * inf) instead of saturating. Public since epic
 * PR2 (TensorConversion.h): the funnel's staging quantizer and (since PR2)
 * wire OUT_WRITE epilogues derive exponents through this single authority;
 * epic PR3 added the grad-accumulate engines and the scale arm, and extended
 * OUT_WRITE's reach to the backward's dx wire -- op-local re-blocking never
 * happens (the D8 amendment, docs/conventions/arithmetic-bfp.md §9). */
void deriveBfpStoredExponent(float absMax, float qMax, int32_t bias, uint8_t maxStored,
                             uint8_t *storedOut) {
    if (absMax == 0.f) {
        *storedOut = (uint8_t)bias; /* zero-state parity: E = 0 */
        return;
    }
    int cap = (int)maxStored;
    if (cap > (int)bias + 127) {
        cap = (int)bias + 127; /* largest stored exponent with a finite scale */
    }
    if (!isfinite(absMax)) {
        /* D6's mantissa-saturation regime taken to its limit. frexpf(inf|NaN)
         * has an unspecified result AND an unspecified *exp (C17 7.12.6.4), so
         * deriving would emit an arbitrary exponent and leave the emit pass
         * rounding a non-finite quotient (undefined). A magnitude too large
         * for ANY grid saturates at the largest finite one -- mantissas then
         * clamp to the code range, exactly as a merely-too-large FINITE absmax
         * already does below. */
        *storedOut = (uint8_t)cap;
        return;
    }
    int e;
    float frac = frexpf(absMax / qMax, &e);
    int E = (frac == 0.5f) ? e - 1 : e; /* smallest E with absMax/2^E <= qMax */
    int stored = E + (int)bias;
    if (stored < 0) {
        stored = 0; /* D6: flush-toward-zero regime */
    }
    if (stored > cap) {
        stored = cap; /* D6: mantissa-saturation regime */
    }
    *storedOut = (uint8_t)stored;
}

/* Packed sibling of quantizeFloatBufferToBfpCodes below: same two-pass
 * value-domain quantization, but chunked and emitting a PACKED payload. */
static void packFloatBufferAsBfp(const float *values, size_t n, bfpQConfig_t *outQC, uint8_t *dst,
                                 const char *what) {
    /* #421 U9: the fifth exponents[g] indexer, same exact-division invariant
     * (numGroups * groupSize == n) the engines and the streaming twin already
     * validate. Pass 2 derives g = idx / gsz with no bound against numGroups,
     * so a violating geometry reads exponents[] OUT OF BOUNDS (< n) or
     * silently mis-blocks (> n) -- and the route is PUBLIC:
     * conversionMatrix[FLOAT32][BFP] takes the geometry from the TARGET while
     * n comes from the SOURCE, so a shape-agnostic template produces the
     * mismatch with neither tensor malformed on its own (#420 G2's rationale,
     * verbatim). */
    validateBfpQConfigShape(outQC, n);
    const float qMax = powf(2, (float)outQC->mantissaBits - 1) - 1;
    const float qMin = -powf(2, (float)outQC->mantissaBits - 1);
    const int32_t bias = bfpExponentBias(outQC);
    const uint8_t maxStored = (uint8_t)((1u << outQC->exponentBits) - 1u);
    const size_t gsz = outQC->groupSize == 0 ? n : outQC->groupSize;
    /* pass 1: sequential groups (contiguous runs), running absmax scalar */
    for (size_t g = 0; g < outQC->numGroups; g++) {
        float absMax = 0.f;
        size_t start = g * gsz;
        size_t end = start + gsz > n ? n : start + gsz;
        for (size_t i = start; i < end; i++) {
            float v = fabsf(values[i]);
            if (v > absMax) {
                absMax = v;
            }
        }
        deriveBfpStoredExponent(absMax, qMax, bias, maxStored, &outQC->exponents[g]);
    }
    /* pass 2: chunked quantize + pack; saturation via clamp BEFORE the guard
     * (value-domain quantization saturates by design -- D6; raw code packing
     * elsewhere keeps the #227 abort). WITHIN a chunk, walk per-RUN -- span
     * to min(chunkEnd, groupEnd) -- so the ldexpf+bias scale derivation
     * leaves the inner loop (packFloatBufferAsSym's grouped phase-2 shape;
     * native hot path since epic PR2). gsz = n for per-tensor keeps g = 0.
     * One roundByMode per element in element order -- bit-identical.
     * #421 U2: that clamp runs in the FLOAT domain first. In the D6 cap
     * regime (a narrow exponentBits pins the scale far below what the data
     * needs) values / scale leaves int32 range for entirely FINITE inputs,
     * and (int32_t)round(...) is undefined there (C17 6.3.1.4). Identical to
     * clamping after the round wherever the round is defined -- in range the
     * float clamp is the identity, outside it roundByMode is monotone -- and
     * the clampInt32 behind it stays load-bearing for SR_HALF_AWAY, whose
     * pre-round dither can still push a clamped qMin one step past it. */
    int32_t codes[ODT_CONVERSION_CHUNK_ELEMS];
    for (size_t off = 0; off < n; off += ODT_CONVERSION_CHUNK_ELEMS) {
        size_t count = n - off < ODT_CONVERSION_CHUNK_ELEMS ? n - off : ODT_CONVERSION_CHUNK_ELEMS;
        size_t chunkEnd = off + count;
        size_t idx = off;
        while (idx < chunkEnd) {
            size_t g = idx / gsz;
            size_t groupEnd = (g + 1) * gsz;
            size_t runEnd = groupEnd < chunkEnd ? groupEnd : chunkEnd;
            size_t runLen = runEnd - idx;
            const float scale = bfpGroupScale(outQC, g);
            for (size_t i = 0; i < runLen; i++) {
                float q = clamp(values[idx + i] / scale, qMin, qMax);
                codes[idx - off + i] =
                    clampInt32(roundByMode(q, outQC->roundingMode), (int32_t)qMin, (int32_t)qMax);
            }
            idx = runEnd;
        }
        packChunkGuarded(codes, count, dst, outQC->mantissaBits, off, what);
    }
}

/* Unpacked sibling of packFloatBufferAsBfp above (same two-pass value-domain
 * quantization, but caller-buffer codes instead of a packed payload): the
 * executeOp funnel stages FLOAT32 operands as unpacked int32 codes, so
 * pack-then-unpack would be wasted work. Writes into the CALLER's codesOut
 * (n entries) -- exempt from the no-O(n)-internal-scratch converter contract
 * by design; allocates nothing. Contract details in TensorConversion.h. */
void quantizeFloatBufferToBfpCodes(const float *values, size_t n, bfpQConfig_t *outQC,
                                   int32_t *codesOut) {
    /* #421 U9: the packed sibling's guard, for the same invariant. This form
     * never reads out of bounds (its per-group loops clip `end` to n), but a
     * numGroups * groupSize < n geometry leaves the codesOut TAIL
     * UNINITIALIZED -- the funnel would then hand a kernel uninitialized
     * mantissa scratch. Its one production caller is funnel-internal staging
     * (ExecuteOp.c), whose template is per-tensor {1,0} today. */
    validateBfpQConfigShape(outQC, n);
    float qMax = (float)((1 << (outQC->mantissaBits - 1)) - 1);
    int32_t qMin = -(1 << (outQC->mantissaBits - 1));
    int32_t bias = bfpExponentBias(outQC);
    uint8_t maxStored = (uint8_t)((1u << outQC->exponentBits) - 1u);
    size_t gsz = outQC->groupSize == 0 ? (n > 0 ? n : 1) : outQC->groupSize;
    for (size_t g = 0; g < outQC->numGroups; g++) {
        size_t start = g * gsz;
        size_t end = (start + gsz > n) ? n : start + gsz;
        float absMax = 0.f;
        for (size_t i = start; i < end; i++) {
            float a = fabsf(values[i]);
            if (a > absMax) {
                absMax = a;
            }
        }
        deriveBfpStoredExponent(absMax, qMax, bias, maxStored, &outQC->exponents[g]);
        float scale = bfpGroupScale(outQC, g);
        for (size_t i = start; i < end; i++) {
            /* #421 U2: float-domain clamp before the round -- see the packed
             * sibling's pass-2 comment for the cap-regime rationale. */
            float q = clamp(values[i] / scale, (float)qMin, qMax);
            codesOut[i] = clampInt32(roundByMode(q, outQC->roundingMode), qMin, (int32_t)qMax);
        }
    }
}

/* packFloatBufferAsBfp's streaming twin: identical two-pass value-domain BFP
 * pack, but the SOURCE is read chunk-wise through `readChunk` instead of a
 * contiguous float buffer -- so no O(n) dequant scratch is ever allocated.
 * packDequantStreamAsBfp below binds it to dequantChunkToFloat (any dtype
 * that helper dispatches on) for the SYM_INT32/per-tensor-SYM/per-tensor-ASYM
 * -> BFP cells; the grouped-SYM and grouped-ASYM sources bind
 * dequantGroupedSymChunkToFloat / dequantGroupedAsymChunkToFloat instead,
 * because dequantChunkToFloat denies grouped SYM and grouped ASYM by design. */
static void packStreamAsBfp(const tensor_t *src, bfpSrcChunkReader_t readChunk, size_t n,
                            bfpQConfig_t *outQC, uint8_t *dst, const char *what) {
    /* #420 G2: the accumulate/scale engines' guard extended to the fourth
     * exponents[g] indexer. Pass 1 derives exponents by index arithmetic and
     * pass 2 reads bfpGroupScale(outQC, g), both under the exact-division
     * invariant (numGroups * groupSize == n): a field-assigned config
     * violating it OOB-writes the exponent array (< n) or silently
     * mis-blocks (> n). Unlike the engines, the geometry here belongs to the
     * TARGET while n comes from the SOURCE, so a shape-agnostic template can
     * produce the mismatch with neither tensor malformed on its own. */
    validateBfpQConfigShape(outQC, n);
    const float qMax = powf(2, (float)outQC->mantissaBits - 1) - 1;
    const float qMin = -powf(2, (float)outQC->mantissaBits - 1);
    const int32_t bias = bfpExponentBias(outQC);
    const uint8_t maxStored = (uint8_t)((1u << outQC->exponentBits) - 1u);
    if (n == 0) {
        /* No values to derive exponents from: zero-state outcome (stored =
         * bias), parity with packFloatBufferAsBfp's empty-group absMax==0. */
        for (size_t g = 0; g < outQC->numGroups; g++) {
            deriveBfpStoredExponent(0.f, qMax, bias, maxStored, &outQC->exponents[g]);
        }
        return;
    }
    const size_t gsz = outQC->groupSize == 0 ? n : outQC->groupSize;
    float buf[ODT_CONVERSION_CHUNK_ELEMS];
    /* pass 1: running per-group absmax over dequanted chunks -- target groups
     * are contiguous element runs, so a group closes exactly when the running
     * index reaches a gsz multiple (groups may straddle chunk boundaries,
     * hence boundary detection by index arithmetic, not per-chunk resets). */
    float absMax = 0.f;
    size_t g = 0;
    for (size_t off = 0; off < n; off += ODT_CONVERSION_CHUNK_ELEMS) {
        size_t count = n - off < ODT_CONVERSION_CHUNK_ELEMS ? n - off : ODT_CONVERSION_CHUNK_ELEMS;
        readChunk(src, off, count, buf);
        for (size_t i = 0; i < count; i++) {
            float v = fabsf(buf[i]);
            if (v > absMax) {
                absMax = v;
            }
            if ((off + i + 1) % gsz == 0) {
                deriveBfpStoredExponent(absMax, qMax, bias, maxStored, &outQC->exponents[g]);
                g++;
                absMax = 0.f;
            }
        }
    }
    /* pass 2: chunked quantize + pack; saturation via clamp BEFORE the guard
     * (value-domain quantization saturates by design -- D6), one roundByMode
     * per element in element order -- bit-identical under the run-walk.
     * WITHIN a chunk, walk per-RUN so the ldexpf+bias scale derivation
     * leaves the inner loop (packFloatBufferAsBfp's pass-2 shape; native hot
     * path since epic PR2). gsz = n for per-tensor keeps g = 0. */
    int32_t codes[ODT_CONVERSION_CHUNK_ELEMS];
    for (size_t off = 0; off < n; off += ODT_CONVERSION_CHUNK_ELEMS) {
        size_t count = n - off < ODT_CONVERSION_CHUNK_ELEMS ? n - off : ODT_CONVERSION_CHUNK_ELEMS;
        readChunk(src, off, count, buf);
        size_t chunkEnd = off + count;
        size_t idx = off;
        while (idx < chunkEnd) {
            size_t g = idx / gsz;
            size_t groupEnd = (g + 1) * gsz;
            size_t runEnd = groupEnd < chunkEnd ? groupEnd : chunkEnd;
            size_t runLen = runEnd - idx;
            const float scale = bfpGroupScale(outQC, g);
            for (size_t i = 0; i < runLen; i++) {
                /* #421 U2: float-domain clamp before the round -- see
                 * packFloatBufferAsBfp's pass-2 comment. */
                float q = clamp(buf[idx - off + i] / scale, qMin, qMax);
                codes[idx - off + i] =
                    clampInt32(roundByMode(q, outQC->roundingMode), (int32_t)qMin, (int32_t)qMax);
            }
            idx = runEnd;
        }
        packChunkGuarded(codes, count, dst, outQC->mantissaBits, off, what);
    }
}

static void packDequantStreamAsBfp(const tensor_t *src, size_t n, bfpQConfig_t *outQC, uint8_t *dst,
                                   const char *what) {
    packStreamAsBfp(src, dequantChunkToFloat, n, outQC, dst, what);
}

/* Grouped-capable SYM chunk reader for the BFP pack stream: dequantChunkToFloat
 * DENIES grouped SYM by design (its other callers are per-tensor grad paths),
 * so the SYM -> BFP cell brings its own read applying each element's own group
 * scale -- convertSymTensorToFloat32Tensor's run-walking, mirrored per chunk.
 * Only bound for numGroups > 1 (groupSize > 0, no division sentinel). */
static void dequantGroupedSymChunkToFloat(const tensor_t *src, size_t elemOffset, size_t count,
                                          float *out) {
    symQConfig_t *qc = src->quantization->qConfig;
    int32_t mant[ODT_CONVERSION_CHUNK_ELEMS];
    unpackSignExtendChunk(src->data, qc->qBits, elemOffset, count, mant);
    const size_t groupSize = qc->groupSize;
    size_t chunkEnd = elemOffset + count;
    size_t idx = elemOffset;
    while (idx < chunkEnd) {
        size_t g = idx / groupSize;
        size_t groupEnd = (g + 1) * groupSize;
        size_t runEnd = groupEnd < chunkEnd ? groupEnd : chunkEnd;
        size_t runLen = runEnd - idx;
        const float scale = qc->scales[g];
        for (size_t i = 0; i < runLen; i++) {
            out[idx - elemOffset + i] = (float)mant[idx - elemOffset + i] * scale;
        }
        idx = runEnd;
    }
}

/* Grouped-capable ASYM chunk reader for the BFP pack stream (SYM twin above):
 * dequantChunkToFloat DENIES grouped ASYM by design (requirePerTensorAsym --
 * its other callers are per-tensor compute paths), so the ASYM -> BFP cell
 * brings its own read applying each element's own group grid -- the affine
 * code-domain decode (code - zeroPoints[g]) * scales[g] of
 * convertAsymTensorToFloatTensor's grouped path, run-walked per chunk.
 * Only bound for numGroups > 1 (groupSize > 0, no division sentinel). */
static void dequantGroupedAsymChunkToFloat(const tensor_t *src, size_t elemOffset, size_t count,
                                           float *out) {
    asymQConfig_t *qc = src->quantization->qConfig;
    int32_t codes[ODT_CONVERSION_CHUNK_ELEMS];
    unpackZeroExtendChunk(src->data, qc->qBits, elemOffset, count, codes);
    const size_t groupSize = qc->groupSize;
    size_t chunkEnd = elemOffset + count;
    size_t idx = elemOffset;
    while (idx < chunkEnd) {
        size_t g = idx / groupSize;
        size_t groupEnd = (g + 1) * groupSize;
        size_t runEnd = groupEnd < chunkEnd ? groupEnd : chunkEnd;
        size_t runLen = runEnd - idx;
        const float scale = qc->scales[g];
        const int32_t zp = (int32_t)qc->zeroPoints[g];
        for (size_t i = 0; i < runLen; i++) {
            out[idx - elemOffset + i] = (float)(codes[idx - elemOffset + i] - zp) * scale;
        }
        idx = runEnd;
    }
}

void convertFloatTensorToBfpTensor(tensor_t *inputTensor, tensor_t *outputTensor) {
    size_t n = calcNumberOfElementsByTensor(inputTensor);
    packFloatBufferAsBfp((float *)inputTensor->data, n, outputTensor->quantization->qConfig,
                         outputTensor->data, "convertFloatTensorToBfpTensor");
}

void convertBfpTensorToFloat32Tensor(tensor_t *inputTensor, tensor_t *outputTensor) {
    size_t n = calcNumberOfElementsByTensor(inputTensor);
    float *out = (float *)outputTensor->data;
    /* no bounce buffer: each chunk dequants straight into its own span of the
     * caller's float storage -- the staging memcpy added nothing (native hot
     * path since epic PR2) */
    for (size_t off = 0; off < n; off += ODT_CONVERSION_CHUNK_ELEMS) {
        size_t count = n - off < ODT_CONVERSION_CHUNK_ELEMS ? n - off : ODT_CONVERSION_CHUNK_ELEMS;
        dequantChunkToFloat(inputTensor, off, count, out + off);
    }
}

/* INT32 -> BFP is codes-in (#227 code domain): mantissas verbatim, ABORT on
 * overflow via packChunkGuarded (D6 saturation is value-domain only). Every
 * group exponent is reset to the zero state (= bias, E=0, scale 1) so a
 * reused destination cannot carry a stale grid -- the BFP image of
 * convertInt32TensorToSymTensor's scales[0] = 1.f reset. */
void convertInt32TensorToBfpTensor(tensor_t *inputTensor, tensor_t *outputTensor) {
    size_t n = calcNumberOfElementsByTensor(inputTensor);
    bfpQConfig_t *outQC = outputTensor->quantization->qConfig;
    const int32_t bias = bfpExponentBias(outQC);
    for (size_t g = 0; g < outQC->numGroups; g++) {
        outQC->exponents[g] = (uint8_t)bias;
    }
    int32_t codes[ODT_CONVERSION_CHUNK_ELEMS];
    for (size_t off = 0; off < n; off += ODT_CONVERSION_CHUNK_ELEMS) {
        size_t count = n - off < ODT_CONVERSION_CHUNK_ELEMS ? n - off : ODT_CONVERSION_CHUNK_ELEMS;
        /* alignment-safe staging, like readBytesAsInt32Array's whole-buffer memcpy */
        memcpy(codes, (const int32_t *)inputTensor->data + off, count * sizeof(int32_t));
        packChunkGuarded(codes, count, outputTensor->data, outQC->mantissaBits, off,
                         "convertInt32TensorToBfpTensor");
    }
}

// Important: Exponents are ignored! Emits sign-extended mantissa codes (int_repr).
void convertBfpTensorToInt32Tensor(tensor_t *inputTensor, tensor_t *outputTensor) {
    size_t n = calcNumberOfElementsByTensor(inputTensor);
    bfpQConfig_t *inQC = inputTensor->quantization->qConfig;
    unpackSignExtend(inputTensor->data, inQC->mantissaBits, 0, (int32_t *)outputTensor->data, n);
}

void convertSymInt32TensorToBfpTensor(tensor_t *inputTensor, tensor_t *outputTensor) {
    size_t n = calcNumberOfElementsByTensor(inputTensor);
    packDequantStreamAsBfp(inputTensor, n, outputTensor->quantization->qConfig, outputTensor->data,
                           "convertSymInt32TensorToBfpTensor");
}

/* BFP -> SYM_INT32 is value-preserving onto a FRESH scalar absmax grid
 * (convertAsymTensorToSymTensor's two-pass shape); the target payload is
 * unpacked int32, so plain stores -- no pack guard. Never saturates by
 * construction (absMax maps exactly to +-qMax); the clamp mirrors the
 * fresh-grid sibling cells. */
void convertBfpTensorToSymInt32Tensor(tensor_t *inputTensor, tensor_t *outputTensor) {
    size_t n = calcNumberOfElementsByTensor(inputTensor);
    symInt32QConfig_t *outQC = outputTensor->quantization->qConfig;
    int32_t *out = (int32_t *)outputTensor->data;
    float buf[ODT_CONVERSION_CHUNK_ELEMS];
    /* pass 1: absmax over dequantized values, chunked (grouped-capable via
     * dequantChunkToFloat's per-element group lookup) */
    float absMax = 0.f;
    for (size_t off = 0; off < n; off += ODT_CONVERSION_CHUNK_ELEMS) {
        size_t count = n - off < ODT_CONVERSION_CHUNK_ELEMS ? n - off : ODT_CONVERSION_CHUNK_ELEMS;
        dequantChunkToFloat(inputTensor, off, count, buf);
        for (size_t i = 0; i < count; i++) {
            float v = fabsf(buf[i]);
            if (v > absMax) {
                absMax = v;
            }
        }
    }
    const float qMax = powf(2, (float)outQC->qMaxBits - 1) - 1;
    const float qMin = -powf(2, (float)outQC->qMaxBits - 1);
    float scale = (absMax == 0.f) ? 1.f : absMax / qMax;
    outQC->scale = scale;
    /* pass 2: chunked emit -- one roundByMode per element (TARGET config's
     * roundingMode), element order */
    for (size_t off = 0; off < n; off += ODT_CONVERSION_CHUNK_ELEMS) {
        size_t count = n - off < ODT_CONVERSION_CHUNK_ELEMS ? n - off : ODT_CONVERSION_CHUNK_ELEMS;
        dequantChunkToFloat(inputTensor, off, count, buf);
        for (size_t i = 0; i < count; i++) {
            out[off + i] = clampInt32(roundByMode(buf[i] / scale, outQC->roundingMode),
                                      (int32_t)qMin, (int32_t)qMax);
        }
    }
}

/* SYM -> BFP is value-preserving (fresh 2^E grid over the dequantized
 * absmax); grouped SYM sources of ANY shape convert -- the grouped reader
 * applies scales[i/groupSize] per element on BOTH pack passes. */
void convertSymTensorToBfpTensor(tensor_t *inputTensor, tensor_t *outputTensor) {
    size_t n = calcNumberOfElementsByTensor(inputTensor);
    symQConfig_t *inQC = inputTensor->quantization->qConfig;
    if (inQC->numGroups > 1) {
        packStreamAsBfp(inputTensor, dequantGroupedSymChunkToFloat, n,
                        outputTensor->quantization->qConfig, outputTensor->data,
                        "convertSymTensorToBfpTensor");
        return;
    }
    packDequantStreamAsBfp(inputTensor, n, outputTensor->quantization->qConfig, outputTensor->data,
                           "convertSymTensorToBfpTensor");
}

/* ASYM -> BFP is value-preserving (fresh 2^E grid over the dequantized
 * absmax); grouped ASYM sources of ANY shape convert -- the grouped reader
 * applies each element's own group grid ((code - zeroPoints[g]) * scales[g],
 * group-quant PR4 code-domain decode) on BOTH pack passes, mirroring the
 * grouped-SYM arm above and convertAsymTensorToFloatTensor's group scope. */
void convertAsymTensorToBfpTensor(tensor_t *inputTensor, tensor_t *outputTensor) {
    size_t n = calcNumberOfElementsByTensor(inputTensor);
    asymQConfig_t *inQC = inputTensor->quantization->qConfig;
    if (inQC->numGroups > 1) {
        packStreamAsBfp(inputTensor, dequantGroupedAsymChunkToFloat, n,
                        outputTensor->quantization->qConfig, outputTensor->data,
                        "convertAsymTensorToBfpTensor");
        return;
    }
    packDequantStreamAsBfp(inputTensor, n, outputTensor->quantization->qConfig, outputTensor->data,
                           "convertAsymTensorToBfpTensor");
}

/* BFP -> SYM is value-preserving onto a FRESH scalar absmax grid
 * (convertAsymTensorToSymTensor's two-pass shape, packed-target emit);
 * grouped BFP sources dequant per group via dequantChunkToFloat's
 * per-element group lookup. */
void convertBfpTensorToSymTensor(tensor_t *inputTensor, tensor_t *outputTensor) {
    size_t n = calcNumberOfElementsByTensor(inputTensor);
    symQConfig_t *outQC = outputTensor->quantization->qConfig;
    if (outQC->numGroups > 1) {
        PRINT_ERROR("convertBfpTensorToSymTensor: grouped SYM target (numGroups=%zu) has no "
                    "scalar compute image; target a per-tensor SYM config, or dequantize "
                    "BFP->FLOAT32 first then FLOAT32->SYM(grouped)",
                    outQC->numGroups);
        exit(1);
    }
    float buf[ODT_CONVERSION_CHUNK_ELEMS];
    /* pass 1: absmax over dequantized values, chunked (no O(n) scratch) */
    float absMax = 0.f;
    for (size_t off = 0; off < n; off += ODT_CONVERSION_CHUNK_ELEMS) {
        size_t count = n - off < ODT_CONVERSION_CHUNK_ELEMS ? n - off : ODT_CONVERSION_CHUNK_ELEMS;
        dequantChunkToFloat(inputTensor, off, count, buf);
        for (size_t i = 0; i < count; i++) {
            float v = fabsf(buf[i]);
            if (v > absMax) {
                absMax = v;
            }
        }
    }
    const float qMax = powf(2, (float)outQC->qBits - 1) - 1;
    const float qMin = -powf(2, (float)outQC->qBits - 1);
    float outScale = (absMax == 0.f) ? 1.f : absMax / qMax;
    outQC->scales[0] = outScale;
    /* pass 2: chunked emit -- one roundByMode per element (TARGET config's
     * roundingMode), element order */
    int32_t codes[ODT_CONVERSION_CHUNK_ELEMS];
    for (size_t off = 0; off < n; off += ODT_CONVERSION_CHUNK_ELEMS) {
        size_t count = n - off < ODT_CONVERSION_CHUNK_ELEMS ? n - off : ODT_CONVERSION_CHUNK_ELEMS;
        dequantChunkToFloat(inputTensor, off, count, buf);
        for (size_t i = 0; i < count; i++) {
            codes[i] = clampInt32(roundByMode(buf[i] / outScale, outQC->roundingMode),
                                  (int32_t)qMin, (int32_t)qMax);
        }
        packChunkGuarded(codes, count, outputTensor->data, outQC->qBits, off,
                         "convertBfpTensorToSymTensor");
    }
}

/* BFP -> BFP requant onto the TARGET's geometry/widths: per-group exponents
 * derived FRESH from the source's dequantized VALUES (never copied) -- this
 * one cell is simultaneously the OUT_WRITE width-restore, the mantissa- and
 * exponent-width change, and the re-block point (spec §3/§5). The source is
 * read per-element with its OWN group scales (dequantChunkToFloat's
 * grouped-capable BFP arm), so any source geometry converts to any target
 * geometry. n == 0 falls through to packStreamAsBfp's zero-state arm.
 * NOT in-place capable (unlike requantSymInt32Tensor): pass 2 re-reads the
 * source under its original exponents, which an aliased pass-1 write would
 * already have clobbered; the funnel always passes distinct tensors. */
void requantBfpTensor(tensor_t *inputTensor, tensor_t *outputTensor) {
    if (inputTensor->data == outputTensor->data) {
        PRINT_ERROR("requantBfpTensor: in-place requant is unsupported -- pass 2 re-reads the "
                    "source under its original exponents (see TensorConversion.h)");
        exit(1);
    }
    size_t n = calcNumberOfElementsByTensor(inputTensor);
    packDequantStreamAsBfp(inputTensor, n, outputTensor->quantization->qConfig, outputTensor->data,
                           "requantBfpTensor");
}

/* BFP -> ASYM derives the affine grid via the canonical
 * deriveAsymGridFromMinMax (group-quant PR4 D6: nudged CODE-domain grid,
 * single grid source) -- convertSymInt32TensorToAsymTensor's two-pass shape
 * with the source read through dequantChunkToFloat's grouped-capable BFP
 * arm. Grouped ASYM targets fail-fast inside the grid helper
 * (requirePerTensorAsym), exactly like the sibling *ToAsymTensor cells. */
void convertBfpTensorToAsymTensor(tensor_t *inputTensor, tensor_t *outputTensor) {
    size_t n = calcNumberOfElementsByTensor(inputTensor);
    asymQConfig_t *outQC = outputTensor->quantization->qConfig;
    if (n == 0) {
        /* No first element to seed mn/mx from -- no-op, matching the sibling
         * *ToAsymTensor n==0 guards (#296 Stage 2). */
        return;
    }
    float buf[ODT_CONVERSION_CHUNK_ELEMS];
    /* pass 1: min/max over dequantized values, chunked (no O(n) scratch) */
    float mn = 0.f, mx = 0.f;
    bool seeded = false;
    for (size_t off = 0; off < n; off += ODT_CONVERSION_CHUNK_ELEMS) {
        size_t count = n - off < ODT_CONVERSION_CHUNK_ELEMS ? n - off : ODT_CONVERSION_CHUNK_ELEMS;
        dequantChunkToFloat(inputTensor, off, count, buf);
        for (size_t i = 0; i < count; i++) {
            float v = buf[i];
            if (!seeded) {
                mn = v;
                mx = v;
                seeded = true;
            } else {
                if (v < mn) {
                    mn = v;
                }
                if (v > mx) {
                    mx = v;
                }
            }
        }
    }
    deriveAsymGridFromMinMax(mn, mx, outQC);
    /* pass 2: chunked emit -- one roundByMode per element (inside
     * emitAsymChunk, TARGET config's roundingMode), element order */
    for (size_t off = 0; off < n; off += ODT_CONVERSION_CHUNK_ELEMS) {
        size_t count = n - off < ODT_CONVERSION_CHUNK_ELEMS ? n - off : ODT_CONVERSION_CHUNK_ELEMS;
        dequantChunkToFloat(inputTensor, off, count, buf);
        emitAsymChunk(buf, count, outQC, outputTensor->data, off);
    }
}

void convertSymInt32TensorToSymTensor(tensor_t *inputTensor, tensor_t *outputTensor) {
    size_t n = calcNumberOfElementsByTensor(inputTensor);
    symInt32QConfig_t *inQC = inputTensor->quantization->qConfig;
    symQConfig_t *outQC = outputTensor->quantization->qConfig;
    if (outQC->numGroups > 1) {
        PRINT_ERROR("convertSymInt32TensorToSymTensor: grouped SYM target (numGroups=%zu) has no "
                    "scalar compute image; target a per-tensor SYM config, or dequantize "
                    "SYM_INT32->FLOAT32 first then FLOAT32->SYM(grouped)",
                    outQC->numGroups);
        exit(1);
    }
    float inScale = inQC->scale;
    const int32_t *in = (const int32_t *)inputTensor->data;
    /* pass 1: absmax over dequantized values (requantSymInt32Tensor precedent) */
    float absMax = 0.f;
    for (size_t i = 0; i < n; i++) {
        float v = fabsf((float)in[i] * inScale);
        if (v > absMax) {
            absMax = v;
        }
    }
    const float qMax = powf(2, (float)outQC->qBits - 1) - 1;
    const float qMin = -powf(2, (float)outQC->qBits - 1);
    float outScale = (absMax == 0.f) ? 1.f : absMax / qMax;
    outQC->scales[0] = outScale;
    /* pass 2: chunked emit -- one roundByMode per element, element order */
    int32_t codes[ODT_CONVERSION_CHUNK_ELEMS];
    for (size_t off = 0; off < n; off += ODT_CONVERSION_CHUNK_ELEMS) {
        size_t count = n - off < ODT_CONVERSION_CHUNK_ELEMS ? n - off : ODT_CONVERSION_CHUNK_ELEMS;
        for (size_t i = 0; i < count; i++) {
            float v = (float)in[off + i] * inScale;
            codes[i] = clampInt32(roundByMode(v / outScale, outQC->roundingMode), (int32_t)qMin,
                                  (int32_t)qMax);
        }
        packChunkGuarded(codes, count, outputTensor->data, outQC->qBits, off,
                         "convertSymInt32TensorToSymTensor");
    }
}

void repackSymInt32ToSymNoRescale(tensor_t *inputTensor, tensor_t *outputTensor) {
    size_t n = calcNumberOfElementsByTensor(inputTensor);
    symInt32QConfig_t *inQC = inputTensor->quantization->qConfig;
    symQConfig_t *outQC = outputTensor->quantization->qConfig;
    if (outQC->numGroups > 1) {
        PRINT_ERROR("repackSymInt32ToSymNoRescale: grouped SYM target (numGroups=%zu) has no "
                    "no-rescale repack image (a single carried scale cannot fan out to per-group "
                    "scales); target a per-tensor SYM config, or use the rescale route "
                    "(convertSymInt32TensorToSymTensor) into a per-tensor target",
                    outQC->numGroups);
        exit(1);
    }
    outQC->scales[0] = inQC->scale;
    const int32_t *in = (const int32_t *)inputTensor->data;
    for (size_t off = 0; off < n; off += ODT_CONVERSION_CHUNK_ELEMS) {
        size_t count = n - off < ODT_CONVERSION_CHUNK_ELEMS ? n - off : ODT_CONVERSION_CHUNK_ELEMS;
        packChunkGuarded(in + off, count, outputTensor->data, outQC->qBits, off,
                         "repackSymInt32ToSymNoRescale");
    }
}

/* Grad-accumulate primitives (PR3, #261; streamed #296 Stage 2) -- see header
 * doc comments for the when-to-use contract. Increment source: exactly one of
 * flat / tens is non-NULL -- the float* primitives feed a flat buffer, the
 * tensor-typed entry points feed a source tensor dequantized chunk-wise, so
 * neither an O(n) increment copy nor an O(n) packed-target unpack VLA is ever
 * needed regardless of which side (target or increment) is packed/sub-byte. */
typedef struct {
    const float *flat;
    const tensor_t *tens;
} incSrc_t;

static void rejectAliasedIncrement(const tensor_t *target, const tensor_t *increment,
                                   const char *what) {
    /* Self-aliasing is rejected: the rescale engines rewrite the target's
     * grid between phase A and phase B, so an aliased increment would be
     * decoded against the wrong grid mid-stream (release-review finding,
     * PR #324). The funnel epilogue always passes a distinct intermediate. */
    if (increment->data == target->data) {
        PRINT_ERROR("%s: increment must not alias the target", what);
        exit(1);
    }
}

static void incSrcChunk(const incSrc_t *src, size_t off, size_t count, float *out) {
    if (src->flat != NULL) {
        memcpy(out, src->flat + off, count * sizeof(float));
        return;
    }
    dequantChunkToFloat(src->tens, off, count, out);
}

/* #421 U4 (ruling R7): the framework's "no data seen yet" BFP state, shared
 * by every empty-target path. An n == 0 tensor that kept its previous
 * exponents would carry a scale describing data it does not have,
 * indistinguishable downstream from a derived one; the canonical zero state
 * (stored = bias, scale 1.0) is what the exponent authority's absMax == 0
 * arm, optimizerZeroGrad's BFP arm and getQLike's per-tensor clone all use.
 * Past validateBfpQConfigShape, n == 0 implies the per-tensor {1,0} geometry
 * (a grouped config needs numGroups > 1 AND groupSize > 0, whose product is
 * never 0), but the loop stays shape-agnostic anyway. */
static void resetBfpGridToZeroState(bfpQConfig_t *qc) {
    const uint8_t zeroStateExponent = (uint8_t)bfpExponentBias(qc);
    for (size_t g = 0; g < qc->numGroups; g++) {
        qc->exponents[g] = zeroStateExponent;
    }
}

/* #421 U3 (ruling R6): a non-finite value in a BFP accumulate's INCREMENT is
 * a fail-fast, the increment-side twin of the scale path's non-finite-factor
 * guard. A BFP grid is (int32 mantissa, shared power-of-two exponent) and has
 * no NaN/inf code, so there is nothing to propagate into: mapping the value
 * to 0 would silently drop the caller's data, saturating it to +-qMax would
 * invent data. Unguarded it is not even defined -- pass 1's `v > absMax` is
 * false for NaN, so the fresh exponent ignores it, and the emit pass's
 * comparison-based float clamp cannot map NaN either, so roundByMode gets it
 * and (int32_t)round(NaN) is undefined (C17 6.3.1.4).
 * Deliberate contract asymmetry, mirroring the factor guard's: the FLOAT32
 * grad-storage path keeps propagating a NaN gradient loudly, because FLOAT32
 * CAN represent it -- only the BFP arms refuse.
 * Non-finite INTERMEDIATES are a different question and are NOT rejected: a
 * finite increment on a grid already at the exponent cap can still overflow
 * mant*oldScale*factor + inc to +-inf, and that saturates through the emit
 * clamp as D6 value-domain behaviour. Only the INPUT is checked.
 * Checked in the pass that first reads each increment element, so a
 * violation in the first chunk is caught before any write; a later one
 * aborts after earlier chunks are already written, exactly as
 * packChunkGuarded's own mid-walk exit(1) does (the process is not
 * recoverable either way). */
static void rejectNonFiniteIncrement(const float *values, size_t count, size_t baseIndex,
                                     const char *what) {
    for (size_t i = 0; i < count; i++) {
        if (!isfinite(values[i])) {
            PRINT_ERROR("%s: non-finite increment value %f at element %zu -- BFP cannot "
                        "represent non-finite values (no NaN/inf code to propagate into); "
                        "refusing to drop it or saturate it into invented data",
                        what, (double)values[i], baseIndex + i);
            exit(1);
        }
    }
}

/* Both FixedGrid engines' "fresh vs. already-gridded" decision: a codes-only
 * all-zero scan over the WHOLE packed payload (post-initTensor zero-fill or
 * post-optimizerZeroGrad memset -> fresh, derive the grid from the increment;
 * anything else -> carry the existing grid verbatim). Codes-only by design:
 * a zeroed payload decodes to 0.0f under ANY scale/exponent, so the scan must
 * not consult the grid it is about to decide. Reads only, and bails at the
 * first nonzero code.
 * #421 U8: "mantissa 0 decodes to exactly 0.0f" presumes a FINITE scale --
 * 0 * inf is NaN, not 0. That holds for every in-contract config: derived
 * exponents never exceed deriveBfpStoredExponent's bias + 127 cap (the
 * largest finite float32 power of two), and since #420 G5 the deserializer
 * rejects stored exponents above it too, so a non-finite-scale grid can no
 * longer be ingested from an untrusted record -- only hand-built by a caller
 * assigning qConfig fields directly. */
static bool packedPayloadIsAllZero(const uint8_t *data, size_t bits, size_t n) {
    int32_t mant[ODT_CONVERSION_CHUNK_ELEMS];
    for (size_t off = 0; off < n; off += ODT_CONVERSION_CHUNK_ELEMS) {
        size_t count = n - off < ODT_CONVERSION_CHUNK_ELEMS ? n - off : ODT_CONVERSION_CHUNK_ELEMS;
        unpackSignExtendChunk(data, bits, off, count, mant);
        for (size_t i = 0; i < count; i++) {
            if (mant[i] != 0) {
                return false;
            }
        }
    }
    return true;
}

static void accumulateIntoSymFixedGridEngine(tensor_t *target, const incSrc_t *inc, size_t n) {
    symQConfig_t *qc = target->quantization->qConfig;
    int32_t mant[ODT_CONVERSION_CHUNK_ELEMS];
    float incBuf[ODT_CONVERSION_CHUNK_ELEMS];
    int32_t codes[ODT_CONVERSION_CHUNK_ELEMS];

    /* phase A: all-zero scan of the packed accumulator (reads only) */
    if (packedPayloadIsAllZero(target->data, qc->qBits, n)) {
        /* Fresh accumulator (post-initTensor zero-fill or post-optimizerZeroGrad
         * memset): derive the grid from the increment (absmax/qMax; absmax
         * 0 -> scale 1.f, packFloatBufferAsSym convention). */
        float absMax = 0.f;
        if (inc->flat != NULL) {
            absMax = findAbsMaxFloat((uint8_t *)inc->flat, n);
        } else {
            for (size_t off = 0; off < n; off += ODT_CONVERSION_CHUNK_ELEMS) {
                size_t count =
                    n - off < ODT_CONVERSION_CHUNK_ELEMS ? n - off : ODT_CONVERSION_CHUNK_ELEMS;
                incSrcChunk(inc, off, count, incBuf);
                for (size_t i = 0; i < count; i++) {
                    float v = fabsf(incBuf[i]);
                    if (v > absMax) {
                        absMax = v;
                    }
                }
            }
        }
        const float qMax = powf(2, (float)qc->qBits - 1) - 1;
        qc->scales[0] = (absMax == 0.f) ? 1.f : absMax / qMax;
    }
    /* else: carry the grid verbatim -- no re-derivation, no renorm (D1/D2). */

    /* phase B: chunked read-modify-write, one roundByMode per element in
     * element order (SR stream identical to the old whole-tensor pass);
     * in-place safe: chunk k is fully read before chunk k is rewritten and
     * the code width is unchanged, so offsets never shift. */
    float scale = qc->scales[0];
    for (size_t off = 0; off < n; off += ODT_CONVERSION_CHUNK_ELEMS) {
        size_t count = n - off < ODT_CONVERSION_CHUNK_ELEMS ? n - off : ODT_CONVERSION_CHUNK_ELEMS;
        unpackSignExtendChunk(target->data, qc->qBits, off, count, mant);
        incSrcChunk(inc, off, count, incBuf);
        for (size_t i = 0; i < count; i++) {
            codes[i] = roundByMode(((float)mant[i] * scale + incBuf[i]) / scale, qc->roundingMode);
        }
        /* No clamp: packChunkGuarded aborts on overflow (D2, #227 discipline). */
        packChunkGuarded(codes, count, target->data, qc->qBits, off,
                         "accumulateFloatIntoSymTensorFixedGrid");
    }
}

void accumulateFloatIntoSymTensorFixedGrid(tensor_t *target, const float *inc, size_t n) {
    incSrc_t src = {.flat = inc, .tens = NULL};
    accumulateIntoSymFixedGridEngine(target, &src, n);
}

void accumulateTensorIntoSymFixedGrid(tensor_t *target, const tensor_t *increment) {
    size_t n = calcNumberOfElementsByTensor(target);
    if (calcNumberOfElementsByTensor((tensor_t *)increment) != n) {
        PRINT_ERROR("accumulateTensorIntoSymFixedGrid: element-count mismatch");
        exit(1);
    }
    rejectAliasedIncrement(target, increment, "accumulateTensorIntoSymFixedGrid");
    incSrc_t src = {.flat = NULL, .tens = increment};
    accumulateIntoSymFixedGridEngine(target, &src, n);
}

static void accumulateIntoSymRescaleEngine(tensor_t *target, const incSrc_t *inc, size_t n) {
    symQConfig_t *qc = target->quantization->qConfig;
    int32_t mant[ODT_CONVERSION_CHUNK_ELEMS];
    float incBuf[ODT_CONVERSION_CHUNK_ELEMS];
    int32_t codes[ODT_CONVERSION_CHUNK_ELEMS];

    /* latch the OLD scale before phase B overwrites qc->scales[0] below --
     * dequanting the packed accumulator always uses the grid it was stored
     * under, never the freshly derived one. */
    float oldScale = qc->scales[0];

    /* phase A: chunked absmax of (mant*oldScale + inc), no rounding, no
     * writes -- fresh absmax every call, no carried grid (unlike the
     * FixedGrid twin). */
    float absMax = 0.f;
    for (size_t off = 0; off < n; off += ODT_CONVERSION_CHUNK_ELEMS) {
        size_t count = n - off < ODT_CONVERSION_CHUNK_ELEMS ? n - off : ODT_CONVERSION_CHUNK_ELEMS;
        unpackSignExtendChunk(target->data, qc->qBits, off, count, mant);
        incSrcChunk(inc, off, count, incBuf);
        for (size_t i = 0; i < count; i++) {
            float v = fabsf((float)mant[i] * oldScale + incBuf[i]);
            if (v > absMax) {
                absMax = v;
            }
        }
    }
    const float qMax = powf(2, (float)qc->qBits - 1) - 1;
    const float qMin = -powf(2, (float)qc->qBits - 1);
    float scale = (absMax == 0.f) ? 1.f : absMax / qMax;
    qc->scales[0] = scale;

    /* phase B: chunked read-modify-write, one roundByMode per element in
     * element order (replicates packFloatBufferAsSym's clamp+round); chunk k
     * is fully read (both target and increment) before chunk k is
     * rewritten, so this is in-place safe. */
    for (size_t off = 0; off < n; off += ODT_CONVERSION_CHUNK_ELEMS) {
        size_t count = n - off < ODT_CONVERSION_CHUNK_ELEMS ? n - off : ODT_CONVERSION_CHUNK_ELEMS;
        unpackSignExtendChunk(target->data, qc->qBits, off, count, mant);
        incSrcChunk(inc, off, count, incBuf);
        for (size_t i = 0; i < count; i++) {
            float v = (float)mant[i] * oldScale + incBuf[i];
            codes[i] =
                clampInt32(roundByMode(v / scale, qc->roundingMode), (int32_t)qMin, (int32_t)qMax);
        }
        packChunkGuarded(codes, count, target->data, qc->qBits, off,
                         "accumulateFloatIntoSymTensorRescale");
    }
}

void accumulateFloatIntoSymTensorRescale(tensor_t *target, const float *inc, size_t n) {
    incSrc_t src = {.flat = inc, .tens = NULL};
    accumulateIntoSymRescaleEngine(target, &src, n);
}

void accumulateTensorIntoSymRescale(tensor_t *target, const tensor_t *increment) {
    size_t n = calcNumberOfElementsByTensor(target);
    if (calcNumberOfElementsByTensor((tensor_t *)increment) != n) {
        PRINT_ERROR("accumulateTensorIntoSymRescale: element-count mismatch");
        exit(1);
    }
    rejectAliasedIncrement(target, increment, "accumulateTensorIntoSymRescale");
    incSrc_t src = {.flat = NULL, .tens = increment};
    accumulateIntoSymRescaleEngine(target, &src, n);
}

static void accumulateIntoAsymRescaleEngine(tensor_t *target, const incSrc_t *inc, size_t n) {
    asymQConfig_t *qc = target->quantization->qConfig;
    /* grads are per-tensor unconditionally (gradInit's carrier gate) -- a
     * grouped target here is a caller contract violation, same rationale as
     * dequantChunkToFloat's gate. */
    requirePerTensorAsym(qc, "accumulateIntoAsymRescaleEngine");
    int32_t codes[ODT_CONVERSION_CHUNK_ELEMS];
    float incBuf[ODT_CONVERSION_CHUNK_ELEMS];
    float vals[ODT_CONVERSION_CHUNK_ELEMS];

    /* latch the OLD grid (element 0: per-tensor by the gate above) before
     * deriveAsymGridFromMinMax overwrites it below. */
    float oldScale = qc->scales[0];
    int32_t oldZeroPoint = (int32_t)qc->zeroPoints[0];

    /* phase A: chunked min/max of the decoded-plus-increment values (no
     * rounding, no writes) -- fresh affine grid every call (D4: no
     * fit-preserving ASYM pack exists). */
    float mn = 0.f, mx = 0.f;
    bool seeded = false;
    for (size_t off = 0; off < n; off += ODT_CONVERSION_CHUNK_ELEMS) {
        size_t count = n - off < ODT_CONVERSION_CHUNK_ELEMS ? n - off : ODT_CONVERSION_CHUNK_ELEMS;
        unpackZeroExtendChunk(target->data, qc->qBits, off, count, codes);
        incSrcChunk(inc, off, count, incBuf);
        for (size_t i = 0; i < count; i++) {
            float v = (float)(codes[i] - oldZeroPoint) * oldScale + incBuf[i];
            if (!seeded) {
                mn = v;
                mx = v;
                seeded = true;
            } else {
                if (v < mn) {
                    mn = v;
                }
                if (v > mx) {
                    mx = v;
                }
            }
        }
    }
    deriveAsymGridFromMinMax(mn, mx, qc);

    /* phase B: chunked recompute + emit -- one roundByMode per element
     * (inside emitAsymChunk), element order; chunk k is fully read (target
     * unpack + increment) before emitAsymChunk rewrites it, and the code
     * width is unchanged, so this is in-place safe. */
    for (size_t off = 0; off < n; off += ODT_CONVERSION_CHUNK_ELEMS) {
        size_t count = n - off < ODT_CONVERSION_CHUNK_ELEMS ? n - off : ODT_CONVERSION_CHUNK_ELEMS;
        unpackZeroExtendChunk(target->data, qc->qBits, off, count, codes);
        incSrcChunk(inc, off, count, incBuf);
        for (size_t i = 0; i < count; i++) {
            vals[i] = (float)(codes[i] - oldZeroPoint) * oldScale + incBuf[i];
        }
        emitAsymChunk(vals, count, qc, target->data, off);
    }
}

void accumulateFloatIntoAsymTensorRescale(tensor_t *target, const float *inc, size_t n) {
    incSrc_t src = {.flat = inc, .tens = NULL};
    accumulateIntoAsymRescaleEngine(target, &src, n);
}

void accumulateTensorIntoAsymRescale(tensor_t *target, const tensor_t *increment) {
    size_t n = calcNumberOfElementsByTensor(target);
    if (calcNumberOfElementsByTensor((tensor_t *)increment) != n) {
        PRINT_ERROR("accumulateTensorIntoAsymRescale: element-count mismatch");
        exit(1);
    }
    rejectAliasedIncrement(target, increment, "accumulateTensorIntoAsymRescale");
    incSrc_t src = {.flat = NULL, .tens = increment};
    accumulateIntoAsymRescaleEngine(target, &src, n);
}

/* BFP grad-accumulate engines (epic PR3): the SYM engines' shape plus
 * per-group grids. The outer walk stays chunk-aligned (the unpack/pack/
 * dequant helpers' byte-alignment contract) and group boundaries are handled
 * by the run-walk inside each chunk (dequantChunkToFloat's BFP shape) -- a
 * literal group-sequential walk would start chunks at group boundaries,
 * whose bit offsets (g*groupSize*mantissaBits) are not byte-aligned for
 * arbitrary geometries. No width guard here or in the funnel arm: BFP
 * mantissaBits is capped to [2,16] at construction (initBfpQConfigGrouped),
 * so an ODT_SYM_GRAD_QMAXBITS-style re-check would be dead code. */
static void accumulateIntoBfpFixedGridEngine(tensor_t *target, const incSrc_t *inc, size_t n,
                                             const char *what) {
    bfpQConfig_t *qc = target->quantization->qConfig;
    /* Both engines and the scale twin index exponents[g] under the
     * exact-division invariant (numGroups * groupSize == n); a field-assigned
     * config violating it would OOB-write the exponent array (< n) or
     * silently mis-block (> n). Engine entry covers both wrappers once --
     * the float* wrappers' n is caller-supplied, so it is validated here
     * too, not just the tensor-derived one. */
    validateBfpQConfigShape(qc, n);
    if (n == 0) {
        /* R7: the scale arm's semantic, now shared. The flat wrapper reached
         * it by accident (an empty group's absmax is 0, which the exponent
         * authority maps to the zero state); the streamed one skipped its
         * derive loop entirely and kept the stale grid. */
        resetBfpGridToZeroState(qc);
        return;
    }
    const size_t gsz = qc->groupSize == 0 ? n : qc->groupSize;
    int32_t mant[ODT_CONVERSION_CHUNK_ELEMS];
    float incBuf[ODT_CONVERSION_CHUNK_ELEMS];
    int32_t codes[ODT_CONVERSION_CHUNK_ELEMS];

    /* phase A: all-zero scan of the packed accumulator (reads only) */
    if (packedPayloadIsAllZero(target->data, qc->mantissaBits, n)) {
        /* Fresh accumulator (post-initTensor zero-fill or post-optimizerZeroGrad
         * memset): derive the per-group grid from the increment alone through
         * the exponent authority (absMax 0 -> stored = bias, the zero-state
         * convention). */
        const float qMax = powf(2, (float)qc->mantissaBits - 1) - 1;
        const int32_t bias = bfpExponentBias(qc);
        const uint8_t maxStored = (uint8_t)((1u << qc->exponentBits) - 1u);
        if (inc->flat != NULL) {
            /* contiguous float groups: packFloatBufferAsBfp's pass-1 shape */
            for (size_t g = 0; g < qc->numGroups; g++) {
                size_t start = g * gsz;
                size_t len = start + gsz > n ? n - start : gsz;
                float absMax = findAbsMaxFloat((uint8_t *)(inc->flat + start), len);
                deriveBfpStoredExponent(absMax, qMax, bias, maxStored, &qc->exponents[g]);
            }
        } else {
            /* streamed increment: running absmax, groups close exactly at gsz
             * multiples (packStreamAsBfp's pass-1 idiom) */
            float absMax = 0.f;
            size_t g = 0;
            for (size_t off = 0; off < n; off += ODT_CONVERSION_CHUNK_ELEMS) {
                size_t count =
                    n - off < ODT_CONVERSION_CHUNK_ELEMS ? n - off : ODT_CONVERSION_CHUNK_ELEMS;
                incSrcChunk(inc, off, count, incBuf);
                for (size_t i = 0; i < count; i++) {
                    float v = fabsf(incBuf[i]);
                    if (v > absMax) {
                        absMax = v;
                    }
                    if ((off + i + 1) % gsz == 0) {
                        deriveBfpStoredExponent(absMax, qMax, bias, maxStored, &qc->exponents[g]);
                        g++;
                        absMax = 0.f;
                    }
                }
            }
        }
    }
    /* else: carry the grid verbatim -- no re-derivation, no renorm (the SYM
     * engine's D1/D2 analog, fit-preserving). */

    /* phase B: chunked read-modify-write, one roundByMode per element in
     * element order; the run-walk hoists each group's 2^E scale out of the
     * inner loop. NO saturation: packChunkGuarded aborts on overflow (#227
     * code-domain discipline; D6 saturation is value-domain only). In-place
     * safe: chunk k is fully read before chunk k is rewritten and the code
     * width is unchanged.
     * #421 U2: this site's pre-clamp therefore deliberately does NOT use
     * [qMin, qMax] like the value-domain emit passes -- pulling an
     * overflowing accumulator back onto the grid would turn the #227 abort
     * into silent saturation. It clamps to a band TWO codes wider on each
     * side, which is enough to make the conversion defined (a group at the
     * exponent cap can drive the quotient past int32 range, or to +-inf, for
     * entirely finite increments, and (int32_t)round(...) is undefined there
     * -- C17 6.3.1.4) while leaving every overflowing value out of range for
     * packChunkGuarded, which still aborts. Two codes rather than one is
     * conservative headroom, not a requirement: SR_HALF_AWAY dithers by
     * [-0.5, 0.5) before rounding, but C's round() is half-AWAY, so a value
     * clamped to qMax+1 dithers into [qMax+0.5, qMax+1.5) and rounds to
     * qMax+1 at worst -- it can never land back on qMax, and a +-1 band would
     * already preserve the abort. The extra code buys margin against any
     * future rounding mode with a wider dither. */
    const float qMax = powf(2, (float)qc->mantissaBits - 1) - 1;
    const float qMin = -powf(2, (float)qc->mantissaBits - 1);
    for (size_t off = 0; off < n; off += ODT_CONVERSION_CHUNK_ELEMS) {
        size_t count = n - off < ODT_CONVERSION_CHUNK_ELEMS ? n - off : ODT_CONVERSION_CHUNK_ELEMS;
        unpackSignExtendChunk(target->data, qc->mantissaBits, off, count, mant);
        incSrcChunk(inc, off, count, incBuf);
        /* Phase B is the only pass that reads EVERY increment element in both
         * branches (phase A reads it only for a fresh, all-zero target), so
         * the R6 check lives here. In the fresh branch phase A has already
         * absorbed the value into an absmax first: NaN is ignored by the
         * comparison and inf takes the exponent authority's non-finite arm
         * (the cap), both defined, and the abort follows immediately here. */
        rejectNonFiniteIncrement(incBuf, count, off, what);
        size_t chunkEnd = off + count;
        size_t idx = off;
        while (idx < chunkEnd) {
            size_t g = idx / gsz;
            size_t groupEnd = (g + 1) * gsz;
            size_t runEnd = groupEnd < chunkEnd ? groupEnd : chunkEnd;
            const float scale = bfpGroupScale(qc, g);
            for (size_t i = idx; i < runEnd; i++) {
                float q = clamp(((float)mant[i - off] * scale + incBuf[i - off]) / scale,
                                qMin - 2.f, qMax + 2.f);
                codes[i - off] = roundByMode(q, qc->roundingMode);
            }
            idx = runEnd;
        }
        packChunkGuarded(codes, count, target->data, qc->mantissaBits, off, what);
    }
}

/* #420 G3: the FLOAT-pointer wrappers' documented `n == target element count`
 * contract, enforced. It cannot live in the engines: the tensor-typed twins
 * DERIVE n from the target (calcNumberOfElementsByTensor below), so the
 * engines only ever see a caller-supplied n through these two entry points.
 * A GROUPED target already dies in the engine's validateBfpQConfigShape, but
 * the per-tensor sentinel {1, 0} accepts any n -- gsz collapses to n, so a
 * short n silently partial-updates the accumulator and leaves the tail stale
 * under a grid derived from the prefix alone. */
static void requireBfpFloatIncrementCoversTarget(const tensor_t *target, size_t n,
                                                 const char *what) {
    size_t expected = calcNumberOfElementsByTensor((tensor_t *)target);
    if (n != expected) {
        PRINT_ERROR("%s: n (%zu) must equal the target's element count (%zu) -- a partial "
                    "increment would leave the tail stale under a prefix-derived grid",
                    what, n, expected);
        exit(1);
    }
}

void accumulateFloatIntoBfpTensorFixedGrid(tensor_t *target, const float *inc, size_t n) {
    requireBfpFloatIncrementCoversTarget(target, n, "accumulateFloatIntoBfpTensorFixedGrid");
    incSrc_t src = {.flat = inc, .tens = NULL};
    accumulateIntoBfpFixedGridEngine(target, &src, n, "accumulateFloatIntoBfpTensorFixedGrid");
}

void accumulateTensorIntoBfpFixedGrid(tensor_t *target, const tensor_t *increment) {
    size_t n = calcNumberOfElementsByTensor(target);
    if (calcNumberOfElementsByTensor((tensor_t *)increment) != n) {
        PRINT_ERROR("accumulateTensorIntoBfpFixedGrid: element-count mismatch");
        exit(1);
    }
    rejectAliasedIncrement(target, increment, "accumulateTensorIntoBfpFixedGrid");
    incSrc_t src = {.flat = NULL, .tens = increment};
    accumulateIntoBfpFixedGridEngine(target, &src, n, "accumulateTensorIntoBfpFixedGrid");
}

/* Fresh-exponent ring capacity for the rescale walker (#421 U6): the live set
 * at any pass-2 chunk spans at most one chunk of groups -- at most
 * ODT_CONVERSION_CHUNK_ELEMS of them, at groupSize 1 -- so that many slots
 * are already exactly enough to keep them distinct. The extra slot is slack,
 * not a requirement. */
#define ODT_RESCALE_FRESH_WINDOW (ODT_CONVERSION_CHUNK_ELEMS + 1)

/* The ONE BFP value-domain rescale walker (#421 U1): both the grad-accumulate
 * DYNAMIC_RESCALE arm and the optimizer's in-place scale arm are the same
 * two-pass repack over v = mant * oldScale * factor + inc, differing only in
 * which of the two knobs is inert --
 *
 *   accumulate: inc = the increment source, factor = 1.f
 *   scale:      inc = NULL (a zero increment), factor = the scale factor
 *
 * so they are one function with two callers instead of two near-identical
 * walkers. Both degenerate cases are EXACT, not approximate: (float)mant *
 * oldScale is an integer times a power of two, so multiplying it by 1.f is
 * the identity, and x + 0.f is x for every finite x and for +-inf alike (the
 * -0.0f -> +0.0f flip an added zero can cause is invisible downstream --
 * fabsf in pass 1, and a quotient of +-0 rounds to the code 0 either way).
 * The same holds if the compiler contracts the multiply-add into an fma: the
 * mant * oldScale product is exact, so fma and mul-then-add round identically
 * here. `inc == NULL` therefore reproduces the old scale path bit for bit,
 * and `factor == 1.f` the old accumulate path.
 *
 * The SYM/ASYM engines keep their own walkers: their grids are scalars, so
 * they have neither the per-group latch nor the run-walk this shape exists
 * for. The BFP FixedGrid engine also stays separate -- its emit differs in
 * KIND, not in formula (a #227 code-domain abort on overflow instead of D6
 * value-domain saturation), and it carries the target's grid instead of
 * re-deriving one. */
static void bfpRescaleWalk(tensor_t *target, const incSrc_t *inc, float factor, size_t n,
                           const char *what) {
    bfpQConfig_t *qc = target->quantization->qConfig;
    /* Exact-division invariant -- see the FixedGrid twin's guard comment. Must
     * run before the walk below, whose group indexing assumes it. */
    validateBfpQConfigShape(qc, n);
    if (n == 0) {
        /* R7: both passes below are element loops, so an empty target would
         * keep whatever grid it carried. Shared with the FixedGrid twin and
         * with the scale arm, which reaches this through its own entry. */
        resetBfpGridToZeroState(qc);
        return;
    }
    const float qMax = powf(2, (float)qc->mantissaBits - 1) - 1;
    const float qMin = -powf(2, (float)qc->mantissaBits - 1);
    const int32_t bias = bfpExponentBias(qc);
    const uint8_t maxStored = (uint8_t)((1u << qc->exponentBits) - 1u);
    const size_t gsz = qc->groupSize == 0 ? n : qc->groupSize;
    int32_t mant[ODT_CONVERSION_CHUNK_ELEMS];
    float incBuf[ODT_CONVERSION_CHUNK_ELEMS];
    int32_t codes[ODT_CONVERSION_CHUNK_ELEMS];
    if (inc == NULL) {
        /* Scale path: incBuf is never refilled, so one zero fill here makes
         * the unified formula's `+ inc` term inert for the whole walk. */
        memset(incBuf, 0, sizeof(incBuf));
    }

    /* #421 U6: the two passes are INTERLEAVED, with pass 1 running exactly
     * one group-boundary ahead of pass 2, so the only exponents that must be
     * held outside qc->exponents are the FRESH ones of the groups the
     * current pass-2 chunk still touches -- at most one chunk's worth plus
     * the group straddling its end. That is a fixed
     * ODT_CONVERSION_CHUNK_ELEMS + 1 byte window (257 B), replacing the
     * O(numGroups) latch VLA the whole-tensor two-pass needed (n bytes at
     * groupSize 1); this is what closes the unbounded-stack item. The frame
     * is deterministic rather than smaller: sharing one walker costs the
     * SCALE arm an incBuf it never had, so at numGroups == 1 that arm's frame
     * grows by ~1280 B (1 KB incBuf + 257 B window - the 1-byte VLA), from
     * ~2 KB to ~3.3 KB. Accepted deliberately (a scratch split stays
     * mechanical if MCU stack budgets ever complain).
     *
     * The direction is inverted relative to that latch, which is what makes
     * the window small: qc->exponents keeps holding the OLD grid, and pass 2
     * publishes each group's fresh exponent into it only after consuming the
     * group's LAST element. Both passes can therefore read the old exponent
     * straight out of qc->exponents -- pass 1 only ever looks at groups pass
     * 2 has not finished, and pass 2 finishes a group exactly when it
     * publishes it.
     *
     * Ring indexing is g % ODT_RESCALE_FRESH_WINDOW. The live set at pass-2
     * chunk [off, chunkEnd) is exactly the groups [off / gsz,
     * (chunkEnd - 1) / gsz], whose span is at most
     * (ODT_CONVERSION_CHUNK_ELEMS - 1) / gsz -- at most
     * ODT_CONVERSION_CHUNK_ELEMS distinct groups, which the ring already
     * covers exactly at gsz == 1, with one slot of slack. */
    uint8_t fresh[ODT_RESCALE_FRESH_WINDOW];
    size_t scanned = 0; /* pass-1 frontier: elements [0, scanned) are absorbed */
    float absMax = 0.f;

    for (size_t off = 0; off < n; off += ODT_CONVERSION_CHUNK_ELEMS) {
        size_t count = n - off < ODT_CONVERSION_CHUNK_ELEMS ? n - off : ODT_CONVERSION_CHUNK_ELEMS;
        size_t chunkEnd = off + count;

        /* pass 1, advanced just far enough: absmax of
         * (mant*oldScale*factor + inc) per group, deriving each group's fresh
         * exponent into the window when the group closes. It stops at the END
         * of the last group this chunk touches -- never a whole chunk beyond,
         * which is what keeps the window one chunk wide. Reads restart at the
         * 8-aligned offset below the frontier (the unpack/dequant helpers'
         * alignment contract); the <= 7 re-read elements are re-READ, never
         * re-absorbed, so each element still contributes to its group's
         * absmax exactly once. */
        size_t needed = ((chunkEnd - 1) / gsz + 1) * gsz;
        if (needed > n) {
            needed = n;
        }
        while (scanned < needed) {
            size_t start = scanned & ~(size_t)7;
            size_t take = needed - start < ODT_CONVERSION_CHUNK_ELEMS ? needed - start
                                                                      : ODT_CONVERSION_CHUNK_ELEMS;
            unpackSignExtendChunk(target->data, qc->mantissaBits, start, take, mant);
            if (inc != NULL) {
                incSrcChunk(inc, start, take, incBuf);
                rejectNonFiniteIncrement(incBuf + (scanned - start), start + take - scanned,
                                         scanned, what);
            }
            size_t segEnd = start + take;
            size_t idx = scanned;
            while (idx < segEnd) {
                size_t g = idx / gsz;
                size_t groupEnd = (g + 1) * gsz;
                size_t runEnd = groupEnd < segEnd ? groupEnd : segEnd;
                const float oldScale = ldexpf(1.f, (int32_t)qc->exponents[g] - bias);
                for (size_t i = idx; i < runEnd; i++) {
                    float v = fabsf((float)mant[i - start] * oldScale * factor + incBuf[i - start]);
                    if (v > absMax) {
                        absMax = v;
                    }
                }
                if (runEnd == groupEnd) {
                    deriveBfpStoredExponent(absMax, qMax, bias, maxStored,
                                            &fresh[g % ODT_RESCALE_FRESH_WINDOW]);
                    absMax = 0.f;
                }
                idx = runEnd;
            }
            scanned = segEnd;
        }

        /* pass 2: chunked read-modify-write -- decode at the still-unpublished
         * OLD scale, requantize at the windowed fresh one, then publish the
         * fresh exponent as each group closes; one roundByMode per element in
         * element order; clamp before the pack guard (value-domain
         * saturation, D6 -- unlike the FixedGrid twin's abort). In-place safe:
         * chunk k is fully read before chunk k is rewritten and the code width
         * is unchanged.
         * The saturation clamp runs in the FLOAT domain FIRST (Rounding.h's
         * clamp): a group already sitting at the exponent cap has no headroom,
         * so even entirely finite inputs push v / scale past int32 range or
         * all the way to +-inf, and (int32_t)round(...) is undefined there
         * (C17 6.3.1.4). Behaviour-identical to clamping after the round for
         * every DEFINED case -- in range the float clamp is the identity, and
         * outside it roundByMode is monotone, so both orders land on the same
         * boundary code. The clampInt32 stays behind it and is load-bearing,
         * not decoration: SR_HALF_AWAY dithers by [-0.5, 0.5) BEFORE rounding,
         * so a value already clamped to qMin can still round one step past it
         * (round(-128.5) = -129, half away from zero).
         * NaN never reaches this clamp -- which matters because a
         * comparison-based clamp cannot map it. Every input is finite by
         * construction: the factor is checked at scaleBfpTensorInPlace's
         * entry (R1) and the increment element-wise in pass 1 (R6), and
         * mantissa * oldScale * factor + inc over finite operands is at worst
         * +-inf, never NaN, given a FINITE oldScale -- guaranteed for
         * in-contract configs, since deriveBfpStoredExponent caps derived
         * exponents at bias + 127 and #420 G5 rejects deserialized records
         * above that cap. A mantissa of 0 then decodes to exactly 0, not
         * 0 * inf. */
        unpackSignExtendChunk(target->data, qc->mantissaBits, off, count, mant);
        if (inc != NULL) {
            incSrcChunk(inc, off, count, incBuf);
        }
        size_t idx = off;
        while (idx < chunkEnd) {
            size_t g = idx / gsz;
            size_t groupEnd = (g + 1) * gsz;
            size_t runEnd = groupEnd < chunkEnd ? groupEnd : chunkEnd;
            const float oldScale = ldexpf(1.f, (int32_t)qc->exponents[g] - bias);
            const uint8_t freshStored = fresh[g % ODT_RESCALE_FRESH_WINDOW];
            const float scale = ldexpf(1.f, (int32_t)freshStored - bias);
            for (size_t i = idx; i < runEnd; i++) {
                float v = (float)mant[i - off] * oldScale * factor + incBuf[i - off];
                float q = clamp(v / scale, qMin, qMax);
                codes[i - off] =
                    clampInt32(roundByMode(q, qc->roundingMode), (int32_t)qMin, (int32_t)qMax);
            }
            if (runEnd == groupEnd) {
                qc->exponents[g] = freshStored;
            }
            idx = runEnd;
        }
        packChunkGuarded(codes, count, target->data, qc->mantissaBits, off, what);
    }
}

void accumulateFloatIntoBfpTensorRescale(tensor_t *target, const float *inc, size_t n) {
    requireBfpFloatIncrementCoversTarget(target, n, "accumulateFloatIntoBfpTensorRescale");
    incSrc_t src = {.flat = inc, .tens = NULL};
    bfpRescaleWalk(target, &src, 1.f, n, "accumulateFloatIntoBfpTensorRescale");
}

void accumulateTensorIntoBfpRescale(tensor_t *target, const tensor_t *increment) {
    size_t n = calcNumberOfElementsByTensor(target);
    if (calcNumberOfElementsByTensor((tensor_t *)increment) != n) {
        PRINT_ERROR("accumulateTensorIntoBfpRescale: element-count mismatch");
        exit(1);
    }
    rejectAliasedIncrement(target, increment, "accumulateTensorIntoBfpRescale");
    incSrc_t src = {.flat = NULL, .tens = increment};
    bfpRescaleWalk(target, &src, 1.f, n, "accumulateTensorIntoBfpRescale");
}

/* The optimizer's in-place scale arm: the SAME value-domain rescale walk the
 * DYNAMIC_RESCALE accumulate arm runs, with a NULL increment source instead
 * of one and a real factor instead of 1.f (#421 U1 -- bfpRescaleWalk's doc
 * block above carries the shared contract and the exactness argument for
 * both degenerate knobs). Only the two entry guards are the scale path's own:
 * `factor` must be finite, and the ROUNDING is the config's own STORAGE
 * roundingMode by construction -- scaling is storage requantization, not an
 * op (#282 target-owned convention), unlike the accumulate arms whose mode
 * reaches the target config via the funnel's op arithmetic. */
void scaleBfpTensorInPlace(tensor_t *t, float factor) {
    /* A BFP grid is (int32 mantissa, shared power-of-two exponent) -- it has
     * no NaN/inf code, so there is no propagation path a non-finite factor
     * could take. Unguarded the walker's two passes would DISAGREE about it:
     * pass 1's `v > absMax` is false for NaN, so the fresh exponent silently
     * ignores it, while pass 2's float clamp passes NaN through (a
     * comparison-based clamp cannot map it) into roundByMode, and
     * (int32_t)round(NaN) is undefined behavior (C17 6.3.1.4). Fail fast
     * instead of dropping the factor (masks the caller's bug) or saturating
     * (invents data). */
    if (!isfinite(factor)) {
        PRINT_ERROR("scaleBfpTensorInPlace: non-finite factor %f -- BFP cannot represent "
                    "non-finite values (no NaN/inf propagation path); refusing to silently "
                    "drop or corrupt the tensor",
                    (double)factor);
        exit(1);
    }
    /* Everything past the factor guard -- the geometry check, the n == 0
     * canonical zero state (R4, unified as R7 across all three arms) and the
     * walk itself -- lives in the shared walker. */
    size_t n = calcNumberOfElementsByTensor(t);
    bfpRescaleWalk(t, NULL, factor, n, "scaleBfpTensorInPlace");
}

/* SYM_INT32 -> SYM_INT32 grad accumulate: reproduces addSymInt32TensorsInplace's
 * Strategy-A semantics (dequant both -> float add -> fresh-absmax requant with
 * the TARGET's roundingMode) directly over the flat int32 mantissa arrays --
 * SYM_INT32 storage is never packed/sub-byte, so no chunk buffer is needed to
 * keep this O(1) extra memory; Add.c stays untouched (#296 Stage 2). */
void accumulateSymInt32IntoSymInt32Rescale(tensor_t *target, const tensor_t *increment) {
    size_t n = calcNumberOfElementsByTensor(target);
    if (calcNumberOfElementsByTensor((tensor_t *)increment) != n) {
        PRINT_ERROR("accumulateSymInt32IntoSymInt32Rescale: element-count mismatch");
        exit(1);
    }
    symInt32QConfig_t *tQC = target->quantization->qConfig;
    symInt32QConfig_t *iQC = increment->quantization->qConfig;
    float tScale = tQC->scale;
    float iScale = iQC->scale;
    int32_t *tg = (int32_t *)target->data;
    const int32_t *in = (const int32_t *)increment->data;
    /* pass 1: absmax of the float sums (no rounding, no writes) */
    float absMax = 0.f;
    for (size_t i = 0; i < n; i++) {
        float v = fabsf((float)tg[i] * tScale + (float)in[i] * iScale);
        if (v > absMax) {
            absMax = v;
        }
    }
    const float qMax = powf(2, (float)tQC->qMaxBits - 1) - 1;
    const float qMin = -powf(2, (float)tQC->qMaxBits - 1);
    float scale = (absMax == 0.f) ? 1.f : absMax / qMax;
    tQC->scale = scale;
    /* pass 2: same-index read-then-write, one round per element in order */
    for (size_t i = 0; i < n; i++) {
        float v = (float)tg[i] * tScale + (float)in[i] * iScale;
        tg[i] = clampInt32(roundByMode(v / scale, tQC->roundingMode), (int32_t)qMin, (int32_t)qMax);
    }
}

/* FLOAT32 grad accumulate: the FLOAT32-increment fast path is unchanged
 * (addFloat32TensorsInplace, VLA-free already); a non-FLOAT32 increment is
 * dequantized chunk-wise so no O(n) scratch is ever allocated. */
void accumulateTensorIntoFloat32Inplace(tensor_t *target, const tensor_t *increment) {
    size_t n = calcNumberOfElementsByTensor(target);
    if (calcNumberOfElementsByTensor((tensor_t *)increment) != n) {
        PRINT_ERROR("accumulateTensorIntoFloat32Inplace: element-count mismatch");
        exit(1);
    }
    if (increment->quantization->type == FLOAT32) {
        /* Flat same-index add — epilogue targets are identity-ordered
         * grad/param tensors, so flat indexing is exact and matches the
         * dequant branch below (severs the TensorConversion->Add cycle). */
        float *tg = (float *)target->data;
        const float *in = (const float *)increment->data;
        for (size_t i = 0; i < n; i++) {
            tg[i] += in[i];
        }
        return;
    }
    float *out = (float *)target->data;
    float buf[ODT_CONVERSION_CHUNK_ELEMS];
    for (size_t off = 0; off < n; off += ODT_CONVERSION_CHUNK_ELEMS) {
        size_t count = n - off < ODT_CONVERSION_CHUNK_ELEMS ? n - off : ODT_CONVERSION_CHUNK_ELEMS;
        dequantChunkToFloat(increment, off, count, buf);
        for (size_t i = 0; i < count; i++) {
            out[off + i] += buf[i];
        }
    }
}

_Static_assert(BFP + 1 == 7, "extend conversionMatrix when adding qtype_t entries");

conversionFunction_t conversionMatrix[7][7] = {
    [INT32] = {[INT32] = NULL,
               [FLOAT32] = convertInt32TensorToFloatTensor,
               [SYM_INT32] = convertInt32TensorToSymInt32Tensor,
               [SYM] = convertInt32TensorToSymTensor,
               [ASYM] = convertInt32TensorToAsymTensor,
               [BOOL] = unsupportedConversionTypes,
               [BFP] = convertInt32TensorToBfpTensor},
    [FLOAT32] = {[INT32] = convertFloatTensorToInt32Tensor,
                 [FLOAT32] = NULL,
                 [SYM_INT32] = convertFloatTensorToSymInt32Tensor,
                 [SYM] = convertFloatTensorToSymTensor,
                 [ASYM] = convertFloatTensorToAsymTensor,
                 [BOOL] = unsupportedConversionTypes,
                 [BFP] = convertFloatTensorToBfpTensor},
    [SYM_INT32] = {[INT32] = extractInt32TensorFromSymInt32Tensor,
                   [FLOAT32] = convertSymInt32TensorToFloat32Tensor,
                   [SYM_INT32] = requantSymInt32Tensor,
                   [SYM] = convertSymInt32TensorToSymTensor,
                   [ASYM] = convertSymInt32TensorToAsymTensor,
                   [BOOL] = unsupportedConversionTypes,
                   [BFP] = convertSymInt32TensorToBfpTensor},
    [SYM] = {[INT32] = convertSymTensorToInt32Tensor,
             [FLOAT32] = convertSymTensorToFloat32Tensor,
             [SYM_INT32] = convertSymTensorToSymInt32Tensor,
             [SYM] = NULL,
             [ASYM] = convertSymTensorToAsymTensor,
             [BOOL] = unsupportedConversionTypes,
             [BFP] = convertSymTensorToBfpTensor},
    [ASYM] = {[INT32] = convertAsymTensorToInt32Tensor,
              [FLOAT32] = convertAsymTensorToFloatTensor,
              [SYM_INT32] = convertAsymTensorToSymInt32Tensor,
              [SYM] = convertAsymTensorToSymTensor,
              [ASYM] = NULL,
              [BOOL] = unsupportedConversionTypes,
              [BFP] = convertAsymTensorToBfpTensor},
    [BOOL] = {[INT32] = unsupportedConversionTypes,
              [FLOAT32] = unsupportedConversionTypes,
              [SYM_INT32] = unsupportedConversionTypes,
              [SYM] = unsupportedConversionTypes,
              [ASYM] = unsupportedConversionTypes,
              [BOOL] = NULL,
              [BFP] = unsupportedConversionTypes},
    [BFP] = {[INT32] = convertBfpTensorToInt32Tensor,
             [FLOAT32] = convertBfpTensorToFloat32Tensor,
             [SYM_INT32] = convertBfpTensorToSymInt32Tensor,
             [SYM] = convertBfpTensorToSymTensor,
             [ASYM] = convertBfpTensorToAsymTensor,
             [BOOL] = unsupportedConversionTypes,
             [BFP] = requantBfpTensor}};

static void convertTensorsWithSameType(tensor_t *inputTensor, tensor_t *outputTensor,
                                       qtype_t qType) {
    size_t inputBits = calcBitsPerElement(inputTensor->quantization);
    size_t outputBits = calcBitsPerElement(outputTensor->quantization);
    if (inputBits != outputBits) {
        /* Same-type conversion is a verbatim packed-byte copy; differing widths
         * would reinterpret the packing (and overflow the output for wider inputs).
         * Width-changing SYM/ASYM rewrites are real conversions (repack policy:
         * PR3, #261). */
        PRINT_ERROR("Same-type conversion requires equal element widths (%zu vs %zu bits)",
                    inputBits, outputBits);
        exit(1);
    }
    size_t numberOfElements = calcNumberOfElementsByTensor(inputTensor);
    size_t numberOfBytes = calcNumberOfBytesForData(inputTensor->quantization, numberOfElements);

    memmove(outputTensor->data, inputTensor->data, numberOfBytes);

    switch (qType) {
    case SYM_INT32: {
        symInt32QConfig_t *inputSymIntQC = inputTensor->quantization->qConfig;
        symInt32QConfig_t *outputSymIntQC = outputTensor->quantization->qConfig;
        outputSymIntQC->scale = inputSymIntQC->scale;
        break;
    }
    case SYM: {
        symQConfig_t *inputSymQC = inputTensor->quantization->qConfig;
        symQConfig_t *outputSymQC = outputTensor->quantization->qConfig;
        /* Group-quant PR2: dest->scales is a fixed-size (numGroups-element) heap
         * array -- a numGroups mismatch means src's group shape doesn't fit
         * dest's array, so this must fail fast rather than over/under-copy
         * (mirrors copySymQConfigInto, Tensor.c ~:442-459). groupSize is
         * adopted from src unconditionally, same precedent. */
        if (outputSymQC->numGroups != inputSymQC->numGroups) {
            PRINT_ERROR("convertTensorsWithSameType: SYM group-shape mismatch (dest "
                        "numGroups=%zu groupSize=%zu, src numGroups=%zu groupSize=%zu)",
                        outputSymQC->numGroups, outputSymQC->groupSize, inputSymQC->numGroups,
                        inputSymQC->groupSize);
            exit(1);
        }
        memcpy(outputSymQC->scales, inputSymQC->scales, inputSymQC->numGroups * sizeof(float));
        outputSymQC->groupSize = inputSymQC->groupSize;
        break;
    }
    case ASYM: {
        asymQConfig_t *inputAsymQC = inputTensor->quantization->qConfig;
        asymQConfig_t *outputAsymQC = outputTensor->quantization->qConfig;
        /* Group-quant PR4: group-faithful, mirroring the SYM arm above (and
         * copyAsymQConfigInto, Tensor.c) -- dest's arrays are fixed-size
         * numGroups blocks, so a shape mismatch must fail fast. */
        if (outputAsymQC->numGroups != inputAsymQC->numGroups) {
            PRINT_ERROR("convertTensorsWithSameType: ASYM group-shape mismatch (dest "
                        "numGroups=%zu groupSize=%zu, src numGroups=%zu groupSize=%zu)",
                        outputAsymQC->numGroups, outputAsymQC->groupSize, inputAsymQC->numGroups,
                        inputAsymQC->groupSize);
            exit(1);
        }
        memcpy(outputAsymQC->scales, inputAsymQC->scales, inputAsymQC->numGroups * sizeof(float));
        memcpy(outputAsymQC->zeroPoints, inputAsymQC->zeroPoints,
               inputAsymQC->numGroups * sizeof(uint16_t));
        outputAsymQC->groupSize = inputAsymQC->groupSize;
        break;
    }
    case BFP: {
        bfpQConfig_t *inputBfpQC = inputTensor->quantization->qConfig;
        bfpQConfig_t *outputBfpQC = outputTensor->quantization->qConfig;
        /* The width guard above only sees mantissaBits; a same-type copy is
         * only meaningful between IDENTICAL BFP configs -- a numGroups/
         * groupSize mismatch re-blocks the exponents[] indexing (and can
         * over/under-run dest's fixed-size array, the SYM precedent above)
         * and an exponentBits mismatch shifts the bias, silently re-gridding
         * every copied exponent. Geometry/width changes are real conversions:
         * route them through a Quantization layer (the conversionMatrix
         * [BFP][BFP] requant). */
        if (outputBfpQC->numGroups != inputBfpQC->numGroups ||
            outputBfpQC->groupSize != inputBfpQC->groupSize ||
            outputBfpQC->mantissaBits != inputBfpQC->mantissaBits ||
            outputBfpQC->exponentBits != inputBfpQC->exponentBits) {
            PRINT_ERROR("convertTensorsWithSameType: BFP config mismatch (dest numGroups=%zu "
                        "groupSize=%zu m=%u e=%u, src numGroups=%zu groupSize=%zu m=%u e=%u) "
                        "-- re-block/width changes go through conversionMatrix[BFP][BFP] "
                        "(Quantization layer)",
                        outputBfpQC->numGroups, outputBfpQC->groupSize,
                        (unsigned)outputBfpQC->mantissaBits, (unsigned)outputBfpQC->exponentBits,
                        inputBfpQC->numGroups, inputBfpQC->groupSize,
                        (unsigned)inputBfpQC->mantissaBits, (unsigned)inputBfpQC->exponentBits);
            exit(1);
        }
        memcpy(outputBfpQC->exponents, inputBfpQC->exponents,
               inputBfpQC->numGroups * sizeof(uint8_t));
        break;
    }
    default:
        break;
    }
}

void convertTensor(tensor_t *inputTensor, tensor_t *outputTensor) {
    qtype_t inputDType = inputTensor->quantization->type;
    qtype_t outputDType = outputTensor->quantization->type;

    if (inputDType == outputDType) {
        convertTensorsWithSameType(inputTensor, outputTensor, inputDType);
    } else {
        conversionFunction_t conversionFn = conversionMatrix[inputDType][outputDType];
        if (conversionFn == NULL) {
            PRINT_ERROR("No conversion function registered for %s to %s",
                        quantTypeToString(inputDType), quantTypeToString(outputDType));
            exit(1);
        }
        conversionFn(inputTensor, outputTensor);
    }
}
