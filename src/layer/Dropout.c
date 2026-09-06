#define SOURCE_FILE "DROPOUT"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "Dropout.h"

#include "ArithmeticType.h"
#include "Bernoulli.h"
#include "BfpKernelSupport.h"
#include "Common.h"
#include "Layer.h"
#include "Quantization.h"
#include "Rounding.h"
#include "Tensor.h"
#include "TensorConversion.h"

static float dropoutScale(float p) {
    return 1.0f / (1.0f - p);
}

void initDropoutConfig(dropoutConfig_t *cfg, float p, tensor_t *mask, quantization_t *forwardQ,
                       quantization_t *backwardQ) {
    if (!(p >= 0.0f && p < 1.0f)) {
        PRINT_ERROR("Dropout: p must be in [0, 1), got %f", (double)p);
        exit(1);
    }
    if (mask == NULL) {
        PRINT_ERROR("Dropout: mask must not be NULL — caller must pre-allocate a BOOL tensor");
        exit(1);
    }
    cfg->p = p;
    cfg->training = false;
    cfg->mask = mask;
    cfg->forwardMath = arithmeticFromQuantizationOrDefault(forwardQ);
    cfg->propLossMath = arithmeticFromQuantizationOrDefault(backwardQ);
    cfg->outputQ = forwardQ;
    cfg->propLossQ = backwardQ;
    cfg->ownsQuantizations = false;
}

/* BFP epic PR4 (R-P3): Dropout still runs OUTSIDE the executeOp funnel — each
 * arm raw-views ->data in ITS OWN storage format, so the guard stays keyed on
 * the wire's STORAGE dtype and is only NARROWED per arm: the FLOAT32/SYM_INT32
 * arms keep rejecting a packed BFP wire, while the ARITH_BFP arm requires both
 * wires BFP-stored. */
static void requireNoBfpWire(const tensor_t *t, const char *what) {
    if (t->quantization->type == BFP) {
        PRINT_ERROR("%s: this arm raw-views the wire in its own storage format and cannot read "
                    "packed BFP mantissas -- derive ARITH_BFP from a BFP wire config, or keep "
                    "BFP off this wire",
                    what);
        exit(1);
    }
}

static void requireBfpWire(const tensor_t *t, const char *what) {
    if (t->quantization->type != BFP) {
        PRINT_ERROR("%s: the ARITH_BFP arm requires BFP-stored wires -- got dtype %d; see "
                    "docs/conventions/arithmetic-bfp.md §5.7",
                    what, (int)t->quantization->type);
        exit(1);
    }
}

static void dropoutForwardFloat(dropoutConfig_t *cfg, tensor_t *input, tensor_t *output) {
    size_t numberOfElements = calcNumberOfElementsByTensor(input);
    float *in = (float *)input->data;
    float *out = (float *)output->data;

    if (!cfg->training) {
        for (size_t i = 0; i < numberOfElements; i++) {
            out[i] = in[i];
        }
        return;
    }

    float scale = dropoutScale(cfg->p);
    for (size_t i = 0; i < numberOfElements; i++) {
        out[i] = tensorBoolGet(cfg->mask, i) ? in[i] * scale : 0.0f;
    }
}

static void dropoutForwardSymInt32(dropoutConfig_t *cfg, tensor_t *input, tensor_t *output) {
    size_t numberOfElements = calcNumberOfElementsByTensor(input);
    int32_t *in = (int32_t *)input->data;
    int32_t *out = (int32_t *)output->data;
    symInt32QConfig_t *inQC = input->quantization->qConfig;
    symInt32QConfig_t *outQC = output->quantization->qConfig;

    if (!cfg->training) {
        for (size_t i = 0; i < numberOfElements; i++) {
            out[i] = in[i];
        }
        outQC->scale = inQC->scale;
        return;
    }

    float scale = dropoutScale(cfg->p);
    for (size_t i = 0; i < numberOfElements; i++) {
        out[i] = tensorBoolGet(cfg->mask, i) ? in[i] : 0;
    }
    outQC->scale = inQC->scale * scale; // scale-fold: ints copied unchanged, the 1/(1-p) factor
                                        // goes into the quant scale
}

/* Eval-mode / mask-free carry: identical element count and geometry mean
 * identical packed byte counts, so the payload moves verbatim and the group
 * exponents are memcpy'd — the R-P2 transparency argument, unchanged values.
 * The byte count is sized off SRC and written into DST, so the gate runs here
 * too rather than being left to the caller (PR4 idiom).
 *
 * Unlike the bridge below, aliasing is ACCEPTED, not rejected: this is a
 * one-pass verbatim copy, so copying a wire onto itself is the identity —
 * exactly what eval mode promises — and requireBfpPairForArm admits src == dst
 * (one wire passed as both input and output satisfies every count/grid/width
 * check). It must not reach the memcpys, though: their parameters are
 * restrict-qualified, so memcpy(p, p, n) is formally undefined even where every
 * real libc is benign. Each copy is skipped on ITS OWN alias rather than
 * returning on the payload's, so a pair that aliases in one block but not the
 * other still carries the block it does not alias. */
static void dropoutCopyBfpVerbatim(tensor_t *src, tensor_t *dst) {
    size_t numberOfElements = calcNumberOfElementsByTensor(src);
    const bfpQConfig_t *srcQC = src->quantization->qConfig;
    bfpQConfig_t *dstQC = dst->quantization->qConfig;
    bfpRequireSameGeometry(srcQC, numberOfElements, dstQC, calcNumberOfElementsByTensor(dst),
                           "Dropout BFP verbatim carry");
    if (src->data != dst->data) {
        memcpy(dst->data, src->data, calcNumberOfBytesForData(src->quantization, numberOfElements));
    }
    if (srcQC->exponents != dstQC->exponents) {
        memcpy(dstQC->exponents, srcQC->exponents, srcQC->numGroups);
    }
}

/* BFP epic PR4 (R-P3, spec D4 + deviations register 5): Dropout is non-native
 * BY DECISION — 1/(1-p) is not a power of two, so unlike SYM there is no single
 * scale to fold it into and no exponent shift that expresses it. One float
 * bridge, walked in TWO passes over the packed payload: pass 1 derives every
 * group's FRESH exponent from the masked-and-scaled absmax, pass 2 decodes at
 * the source grid, applies mask+factor and requantizes onto the fresh grid.
 * The skeleton is scaleBfpTensorInPlace's (TensorConversion.c) with the mask
 * fused in — cite, do NOT call: that primitive is in-place on ONE tensor and
 * cannot fuse a mask.
 *
 * Re-deriving here is NOT the double quantization spec §9 / D8 forbid: the
 * multiply CHANGES the values, so the fresh exponents quantize NEW numbers.
 * Rounding is the DESTINATION config's own storage roundingMode (#282
 * target-owned convention, scaleBfpTensorInPlace precedent) — Dropout runs
 * outside the funnel and has no op-owned rounding slot to swap in.
 * p = 0.5 is exact and falls out of the float path; no power-of-two
 * special-casing. The BOOL mask is dtype-agnostic and needs no BFP handling. */
static void dropoutMaskScaleBfp(dropoutConfig_t *cfg, tensor_t *src, tensor_t *dst) {
    size_t n = calcNumberOfElementsByTensor(src);
    const bfpQConfig_t *srcQC = src->quantization->qConfig;
    bfpQConfig_t *dstQC = dst->quantization->qConfig;
    bfpRequireSameGeometry(srcQC, n, dstQC, calcNumberOfElementsByTensor(dst),
                           "Dropout BFP bridge");
    /* TWO aliases to reject, not one. The payload alias is the obvious half.
     * The EXPONENT-ARRAY alias is the subtle one: pass 1 writes fresh stored
     * exponents into dstQC->exponents, and pass 2 then reads srcQC->exponents
     * to decode at the ORIGINAL grid — if the two qConfigs share one exponent
     * array (or are literally the same qConfig object, which a caller that
     * reuses one quantization_t for both wires produces), pass 1 has already
     * destroyed the grid pass 2 depends on and every decoded value is wrong.
     * Same precedent as the accumulate engines' rejectAliasedIncrement: a
     * two-pass walk fails fast on aliasing instead of silently miscomputing.
     *
     * The exponents term is unreachable through today's API: every
     * quantizationInitBfp / …Grouped allocates its own exponents block, so two
     * DISTINCT qConfigs cannot share one and the only constructible alias is
     * the shared quantization_t caught by srcQC == dstQC. It stays as defence
     * for a future borrowed / non-owning qConfig, not as dead code. */
    if (src->data == dst->data || srcQC == dstQC || srcQC->exponents == dstQC->exponents) {
        PRINT_ERROR("Dropout BFP bridge: source and destination wires must not alias in EITHER "
                    "their packed payload or their exponent array -- pass 2 re-reads the source "
                    "under its ORIGINAL exponents, which pass 1 has already overwritten on the "
                    "destination");
        exit(1);
    }

    const float factor = dropoutScale(cfg->p);
    const float qMax = powf(2, (float)dstQC->mantissaBits - 1) - 1;
    const float qMin = -powf(2, (float)dstQC->mantissaBits - 1);
    const int32_t bias = bfpExponentBias(srcQC); /* widths gated identical */
    const uint8_t maxStored = (uint8_t)((1u << dstQC->exponentBits) - 1u);
    const size_t gsz = srcQC->groupSize == 0 ? n : srcQC->groupSize;
    int32_t mant[ODT_CONVERSION_CHUNK_ELEMS];
    int32_t codes[ODT_CONVERSION_CHUNK_ELEMS];

    /* pass 1: per-group absmax of the masked, scaled values -> fresh exponent
     * through deriveBfpStoredExponent (the single exponent authority). An
     * all-dropped group's absmax is 0 and re-derives the zero state. */
    float absMax = 0.f;
    for (size_t off = 0; off < n; off += ODT_CONVERSION_CHUNK_ELEMS) {
        size_t count = n - off < ODT_CONVERSION_CHUNK_ELEMS ? n - off : ODT_CONVERSION_CHUNK_ELEMS;
        unpackSignExtend((const uint8_t *)src->data + off * srcQC->mantissaBits / 8,
                         srcQC->mantissaBits, 0, mant, count);
        size_t chunkEnd = off + count;
        size_t idx = off;
        while (idx < chunkEnd) {
            size_t g = idx / gsz;
            size_t groupEnd = (g + 1) * gsz;
            size_t runEnd = groupEnd < chunkEnd ? groupEnd : chunkEnd;
            const float srcScale = ldexpf(1.f, (int32_t)srcQC->exponents[g] - bias);
            for (size_t i = idx; i < runEnd; i++) {
                float v = tensorBoolGet(cfg->mask, i)
                              ? fabsf((float)mant[i - off] * srcScale * factor)
                              : 0.f;
                if (v > absMax) {
                    absMax = v;
                }
            }
            if (runEnd == groupEnd) {
                deriveBfpStoredExponent(absMax, qMax, bias, maxStored, &dstQC->exponents[g]);
                absMax = 0.f;
            }
            idx = runEnd;
        }
    }

    /* pass 2: decode at the source grid, mask+scale, requantize at the fresh
     * one; clamp before the write (value-domain saturation, D6). */
    for (size_t off = 0; off < n; off += ODT_CONVERSION_CHUNK_ELEMS) {
        size_t count = n - off < ODT_CONVERSION_CHUNK_ELEMS ? n - off : ODT_CONVERSION_CHUNK_ELEMS;
        unpackSignExtend((const uint8_t *)src->data + off * srcQC->mantissaBits / 8,
                         srcQC->mantissaBits, 0, mant, count);
        size_t chunkEnd = off + count;
        size_t idx = off;
        while (idx < chunkEnd) {
            size_t g = idx / gsz;
            size_t groupEnd = (g + 1) * gsz;
            size_t runEnd = groupEnd < chunkEnd ? groupEnd : chunkEnd;
            const float srcScale = ldexpf(1.f, (int32_t)srcQC->exponents[g] - bias);
            const float dstScale = bfpGroupScale(dstQC, g);
            for (size_t i = idx; i < runEnd; i++) {
                float v =
                    tensorBoolGet(cfg->mask, i) ? (float)mant[i - off] * srcScale * factor : 0.f;
                codes[i - off] = clampInt32(roundByMode(v / dstScale, dstQC->roundingMode),
                                            (int32_t)qMin, (int32_t)qMax);
            }
            idx = runEnd;
        }
        byteConversion((uint8_t *)codes, 32, (uint8_t *)dst->data + off * dstQC->mantissaBits / 8,
                       dstQC->mantissaBits, count);
    }
}

/* The full ARITH_BFP admission check, as ONE callable unit: dtype, element
 * counts, grid, widths. It is a pure predicate, so the forward can run it
 * BEFORE bernoulliFillMask (RNG neutrality, below) while the workers still
 * gate themselves — no ordering constraint is expressed by where it sits. */
static void requireBfpPairForArm(tensor_t *a, tensor_t *b, const char *whatA, const char *whatB,
                                 const char *whatPair) {
    requireBfpWire(a, whatA);
    requireBfpWire(b, whatB);
    bfpRequireSameGeometry(a->quantization->qConfig, calcNumberOfElementsByTensor(a),
                           b->quantization->qConfig, calcNumberOfElementsByTensor(b), whatPair);
}

static void dropoutForwardBfp(dropoutConfig_t *cfg, tensor_t *input, tensor_t *output) {
    if (!cfg->training) {
        dropoutCopyBfpVerbatim(input, output);
        return;
    }
    dropoutMaskScaleBfp(cfg, input, output);
}

static void dropoutBackwardBfp(dropoutConfig_t *cfg, tensor_t *loss, tensor_t *propLoss) {
    dropoutMaskScaleBfp(cfg, loss, propLoss);
}

void dropoutForward(layer_t *dropoutLayer, tensor_t *input, tensor_t *output) {
    dropoutConfig_t *cfg = dropoutLayer->config->dropout;
    /* Before bernoulliFillMask: a rejected call must not consume RNG draws, so
     * the ARITH_BFP case validates EVERYTHING here — dtype, element counts,
     * grid, widths — and not just the dtype. Leaving the geometry half inside
     * dropoutForwardBfp (which runs after the fill) would make a
     * geometry-rejected call burn a mask draw and desync every later layer's
     * dropout pattern from the reference run, which is precisely the property
     * this ordering exists to protect. R-P3: keyed on the wire's STORAGE
     * dtype, narrowed per declared arm. */
    switch (cfg->forwardMath.type) {
    case ARITH_FLOAT32:
    case ARITH_SYM_INT32:
        requireNoBfpWire(input, "Dropout forward (input)");
        requireNoBfpWire(output, "Dropout forward (output)");
        break;
    case ARITH_BFP:
        requireBfpPairForArm(input, output, "Dropout forward (input)", "Dropout forward (output)",
                             "Dropout forward BFP (input -> output)");
        break;
    default:
        PRINT_ERROR("Dropout forward: quantization type not implemented");
        exit(1);
    }
    if (cfg->training) {
        size_t maskElements = calcNumberOfElementsByTensor(cfg->mask);
        size_t inputElements = calcNumberOfElementsByTensor(input);
        if (maskElements != inputElements) {
            PRINT_ERROR("Dropout forward: mask element count (%zu) does not match input (%zu)",
                        maskElements, inputElements);
            exit(1);
        }
        bernoulliFillMask(cfg->mask, 1.0f - cfg->p); // §6.0.5: fill once before dtype apply
    }
    /* The first switch has already rejected everything but the three known
     * arms, so this default IS the BFP arm — no unreachable duplicate error. */
    switch (cfg->forwardMath.type) {
    case ARITH_FLOAT32:
        dropoutForwardFloat(cfg, input, output);
        break;
    case ARITH_SYM_INT32:
        dropoutForwardSymInt32(cfg, input, output);
        break;
    default:
        dropoutForwardBfp(cfg, input, output);
        break;
    }
}

static void dropoutBackwardFloat(dropoutConfig_t *cfg, tensor_t *loss, tensor_t *propLoss) {
    size_t numberOfElements = calcNumberOfElementsByTensor(loss);
    float *gradOut = (float *)loss->data;
    float *gradIn = (float *)propLoss->data;
    float scale = dropoutScale(cfg->p);

    for (size_t i = 0; i < numberOfElements; i++) {
        gradIn[i] = tensorBoolGet(cfg->mask, i) ? gradOut[i] * scale : 0.0f;
    }
}

static void dropoutBackwardSymInt32(dropoutConfig_t *cfg, tensor_t *loss, tensor_t *propLoss) {
    size_t numberOfElements = calcNumberOfElementsByTensor(loss);
    int32_t *gradOut = (int32_t *)loss->data;
    int32_t *gradIn = (int32_t *)propLoss->data;
    symInt32QConfig_t *lossQC = loss->quantization->qConfig;
    symInt32QConfig_t *propLossQC = propLoss->quantization->qConfig;
    float scale = dropoutScale(cfg->p);

    for (size_t i = 0; i < numberOfElements; i++) {
        gradIn[i] = tensorBoolGet(cfg->mask, i) ? gradOut[i] : 0;
    }
    propLossQC->scale = lossQC->scale * scale; // scale-fold: ints copied unchanged, the 1/(1-p)
                                               // factor goes into the quant scale
}

void dropoutBackward(layer_t *dropoutLayer, tensor_t *forwardInput, tensor_t *loss,
                     tensor_t *propLoss) {
    (void)forwardInput; // not needed: the stored mask + p fully determine the gradient.
    dropoutConfig_t *cfg = dropoutLayer->config->dropout;
    /* forwardInput is deliberately NOT guarded — it is never dereferenced here
     * (callers legitimately pass NULL). */
    size_t maskElements = calcNumberOfElementsByTensor(cfg->mask);
    size_t lossElements = calcNumberOfElementsByTensor(loss);
    if (maskElements != lossElements) {
        PRINT_ERROR("Dropout backward: mask element count (%zu) does not match loss (%zu)",
                    maskElements, lossElements);
        exit(1);
    }
    switch (cfg->propLossMath.type) {
    case ARITH_FLOAT32:
        /* Dropout backward bypasses the executeOp funnel and raw-casts loss/propLoss
         * to float* (forwardInput is unused — the mask + p fully determine dx). Fed a
         * SYM_INT32 wire, the FLOAT32 arm reads int mantissa codes as floats — silent
         * garbage grads. Guard the dereferenced wire dtypes and fail fast, mirroring
         * the LayerNorm/GroupNorm backward guards (#315, #261). */
        if (loss->quantization->type != FLOAT32 || propLoss->quantization->type != FLOAT32) {
            PRINT_ERROR("Dropout backward: FLOAT32 arm requires FLOAT32 wires — got loss %d, "
                        "propLoss %d",
                        (int)loss->quantization->type, (int)propLoss->quantization->type);
            exit(1);
        }
        dropoutBackwardFloat(cfg, loss, propLoss);
        break;
    case ARITH_SYM_INT32:
        /* The SYM_INT32 arm raw-casts to int32* and derefs loss/propLoss->qConfig;
         * a FLOAT32 wire carries qConfig == NULL, so the mismatch is a NULL deref. */
        if (loss->quantization->type != SYM_INT32 || propLoss->quantization->type != SYM_INT32) {
            PRINT_ERROR("Dropout backward: SYM_INT32 arm requires SYM_INT32 wires — got loss %d, "
                        "propLoss %d",
                        (int)loss->quantization->type, (int)propLoss->quantization->type);
            exit(1);
        }
        dropoutBackwardSymInt32(cfg, loss, propLoss);
        break;
    case ARITH_BFP:
        requireBfpPairForArm(loss, propLoss, "Dropout backward (loss)",
                             "Dropout backward (propLoss)",
                             "Dropout backward BFP (loss -> propLoss)");
        dropoutBackwardBfp(cfg, loss, propLoss);
        break;
    default:
        PRINT_ERROR("Dropout backward: quantization type not implemented");
        exit(1);
    }
}

void dropoutCalcOutputShape(layer_t *dropoutLayer, shape_t *inputShape, shape_t *outputShape) {
    (void)dropoutLayer;
    memcpy(outputShape->dimensions, inputShape->dimensions,
           inputShape->numberOfDimensions * sizeof(size_t));
    memcpy(outputShape->orderOfDimensions, inputShape->orderOfDimensions,
           inputShape->numberOfDimensions * sizeof(size_t));
    outputShape->numberOfDimensions = inputShape->numberOfDimensions;
}
