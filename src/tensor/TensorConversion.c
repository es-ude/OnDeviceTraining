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

static void accumulateIntoSymFixedGridEngine(tensor_t *target, const incSrc_t *inc, size_t n) {
    symQConfig_t *qc = target->quantization->qConfig;
    int32_t mant[ODT_CONVERSION_CHUNK_ELEMS];
    float incBuf[ODT_CONVERSION_CHUNK_ELEMS];
    int32_t codes[ODT_CONVERSION_CHUNK_ELEMS];

    /* phase A: all-zero scan of the packed accumulator (reads only) */
    bool allZero = true;
    for (size_t off = 0; off < n && allZero; off += ODT_CONVERSION_CHUNK_ELEMS) {
        size_t count = n - off < ODT_CONVERSION_CHUNK_ELEMS ? n - off : ODT_CONVERSION_CHUNK_ELEMS;
        unpackSignExtendChunk(target->data, qc->qBits, off, count, mant);
        for (size_t i = 0; i < count; i++) {
            if (mant[i] != 0) {
                allZero = false;
                break;
            }
        }
    }
    if (allZero) {
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

_Static_assert(BOOL + 1 == 6, "extend conversionMatrix when adding qtype_t entries");

conversionFunction_t conversionMatrix[6][6] = {
    [INT32] = {[INT32] = NULL,
               [FLOAT32] = convertInt32TensorToFloatTensor,
               [SYM_INT32] = convertInt32TensorToSymInt32Tensor,
               [SYM] = convertInt32TensorToSymTensor,
               [ASYM] = convertInt32TensorToAsymTensor,
               [BOOL] = unsupportedConversionTypes},
    [FLOAT32] = {[INT32] = convertFloatTensorToInt32Tensor,
                 [FLOAT32] = NULL,
                 [SYM_INT32] = convertFloatTensorToSymInt32Tensor,
                 [SYM] = convertFloatTensorToSymTensor,
                 [ASYM] = convertFloatTensorToAsymTensor,
                 [BOOL] = unsupportedConversionTypes},
    [SYM_INT32] = {[INT32] = extractInt32TensorFromSymInt32Tensor,
                   [FLOAT32] = convertSymInt32TensorToFloat32Tensor,
                   [SYM_INT32] = requantSymInt32Tensor,
                   [SYM] = convertSymInt32TensorToSymTensor,
                   [ASYM] = convertSymInt32TensorToAsymTensor,
                   [BOOL] = unsupportedConversionTypes},
    [SYM] = {[INT32] = convertSymTensorToInt32Tensor,
             [FLOAT32] = convertSymTensorToFloat32Tensor,
             [SYM_INT32] = convertSymTensorToSymInt32Tensor,
             [SYM] = NULL,
             [ASYM] = convertSymTensorToAsymTensor,
             [BOOL] = unsupportedConversionTypes},
    [ASYM] = {[INT32] = convertAsymTensorToInt32Tensor,
              [FLOAT32] = convertAsymTensorToFloatTensor,
              [SYM_INT32] = convertAsymTensorToSymInt32Tensor,
              [SYM] = convertAsymTensorToSymTensor,
              [ASYM] = NULL,
              [BOOL] = unsupportedConversionTypes},
    [BOOL] = {[INT32] = unsupportedConversionTypes,
              [FLOAT32] = unsupportedConversionTypes,
              [SYM_INT32] = unsupportedConversionTypes,
              [SYM] = unsupportedConversionTypes,
              [ASYM] = unsupportedConversionTypes,
              [BOOL] = NULL}};

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
