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
#include "LayerExec.h"
#include "Quantization.h"
#include "Rounding.h"
#include "ScaleCanon.h"
#include "Tensor.h"
#include "TensorConversion.h"

static float dropoutScale(float p) {
    return 1.0f / (1.0f - p);
}

void dropoutValidateConfig(const dropoutConfig_t *cfg, const char *who) {
    if (!(cfg->p >= 0.0f && cfg->p < 1.0f)) {
        PRINT_ERROR("%s: p must be in [0, 1), got %f", who, (double)cfg->p);
        exit(1);
    }
}

void initDropoutConfig(dropoutConfig_t *cfg, float p, tensor_t *mask, quantization_t *forwardQ,
                       quantization_t *backwardQ) {
    cfg->p = p;
    dropoutValidateConfig(cfg, "Dropout");
    if (mask == NULL || mask->quantization->type != BOOL) {
        PRINT_ERROR("Dropout: mask must be a caller-allocated BOOL tensor");
        exit(1);
    }
    cfg->training = false;
    cfg->mask = mask;
    cfg->forwardMath = arithmeticFromQuantizationOrDefault(forwardQ);
    cfg->propLossMath = arithmeticFromQuantizationOrDefault(backwardQ);
    cfg->outputQ = forwardQ;
    cfg->propLossQ = backwardQ;
    cfg->ownsQuantizations = false;
}

/* This path operates directly on storage; admit compatible wire dtypes before applying it. Each
 * arithmetic reads its own storage dtype, and the BFP arms copy codes between equal geometries. */
static void dropoutRequireWires(tensor_t *src, tensor_t *dst, arithmeticType_t math,
                                const char *where) {
    static const qtype_t wireType[] = {
        [ARITH_FLOAT32] = FLOAT32, [ARITH_SYM_INT32] = SYM_INT32, [ARITH_BFP] = BFP};
    if (src->quantization->type != wireType[math] || dst->quantization->type != wireType[math]) {
        PRINT_ERROR("%s: arithmetic %d requires dtype %d wires, got %d and %d", where, (int)math,
                    (int)wireType[math], (int)src->quantization->type,
                    (int)dst->quantization->type);
        exit(1);
    }
    if (math == ARITH_BFP) {
        bfpRequireSameGeometry(src->quantization->qConfig, calcNumberOfElementsByTensor(src),
                               dst->quantization->qConfig, calcNumberOfElementsByTensor(dst),
                               where);
    }
}

static void dropoutRequireMaskCount(const dropoutConfig_t *cfg, tensor_t *t, const char *where) {
    size_t maskElements = calcNumberOfElementsByTensor(cfg->mask);
    size_t elements = calcNumberOfElementsByTensor(t);
    if (maskElements != elements) {
        PRINT_ERROR("%s: mask element count (%zu) does not match the wire (%zu)", where,
                    maskElements, elements);
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
        storeCanonScale(&outQC->scale, loadCanonScale(&inQC->scale));
        return;
    }

    float scale = dropoutScale(cfg->p);
    for (size_t i = 0; i < numberOfElements; i++) {
        out[i] = tensorBoolGet(cfg->mask, i) ? in[i] : 0;
    }
    /* scale-fold: ints copied unchanged, the 1/(1-p) factor goes into the quant
     * scale -- canonicalized NEAREST, folded with odtScaleMul (canonCoeffScale) */
    storeCanonScale(&outQC->scale, canonCoeffScale(loadCanonScale(&inQC->scale), scale));
}

/* Eval preserves codes and exponents. Skip each copy independently when its storage aliases
 * (memcpy onto itself is undefined). */
static void dropoutCopyBfpVerbatim(tensor_t *src, tensor_t *dst) {
    size_t numberOfElements = calcNumberOfElementsByTensor(src);
    const bfpQConfig_t *srcQC = src->quantization->qConfig;
    bfpQConfig_t *dstQC = dst->quantization->qConfig;
    if (src->data != dst->data) {
        memcpy(dst->data, src->data, calcNumberOfBytesForData(src->quantization, numberOfElements));
    }
    if (srcQC->exponents != dstQC->exponents) {
        memcpy(dstQC->exponents, srcQC->exponents, srcQC->numGroups);
    }
}

/* Requires disjoint payloads and exponent arrays: pass 2 re-reads the source under the exponents
 * pass 1 overwrote on the destination. Derive destination exponents before requantizing masked,
 * scaled values; use destination rounding. A float bridge because 1/(1-p) is no power of two
 * (docs/conventions/arithmetic-bfp.md §12); the fresh exponents quantize new values, so this is
 * no double quantization. */
static void dropoutMaskScaleBfp(dropoutConfig_t *cfg, tensor_t *src, tensor_t *dst,
                                rngStream_t *bridge) {
    size_t n = calcNumberOfElementsByTensor(src);
    const bfpQConfig_t *srcQC = src->quantization->qConfig;
    bfpQConfig_t *dstQC = dst->quantization->qConfig;
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

    /* pass 2: decode at the source grid, mask+scale, requantize at the fresh one. Clamp in float
     * before rounding: exponent saturation can put finite inputs outside int32 range. The bounds
     * are exact integers (mantissa width <= 16), so the rounded code stays inside them. */
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
                float q = clamp(v / dstScale, qMin, qMax);
                codes[i - off] = roundByMode(q, dstQC->roundingMode, bridge);
            }
            idx = runEnd;
        }
        byteConversion((uint8_t *)codes, 32, (uint8_t *)dst->data + off * dstQC->mantissaBits / 8,
                       dstQC->mantissaBits, count);
    }
}

static void dropoutForwardBfp(dropoutConfig_t *cfg, tensor_t *input, tensor_t *output,
                              rngStream_t *bridge) {
    if (!cfg->training) {
        dropoutCopyBfpVerbatim(input, output);
        return;
    }
    dropoutMaskScaleBfp(cfg, input, output, bridge);
}

void dropoutForward(layer_t *dropoutLayer, tensor_t *input, tensor_t *output,
                    const layerExec_t *exec) {
    dropoutConfig_t *cfg = dropoutLayer->config->dropout;
    /* Reject incompatible wires before filling the mask. */
    dropoutRequireWires(input, output, cfg->forwardMath.type, "Dropout forward");
    if (cfg->training) {
        dropoutRequireMaskCount(cfg, input, "Dropout forward");
        rngStream_t maskStorage;
        bernoulliFillMask(cfg->mask, 1.0f - cfg->p,
                          layerExecPurposeStream(exec, RNG_TAG_FORWARD, RNG_TAG_DROPOUT_MASK,
                                                 &maskStorage)); // fill once before dtype apply
    }
    switch (cfg->forwardMath.type) {
    case ARITH_FLOAT32:
        dropoutForwardFloat(cfg, input, output);
        break;
    case ARITH_SYM_INT32:
        dropoutForwardSymInt32(cfg, input, output);
        break;
    case ARITH_BFP: {
        rngStream_t bridgeStorage;
        dropoutForwardBfp(
            cfg, input, output,
            layerExecPurposeStream(exec, RNG_TAG_FORWARD, RNG_TAG_DROPOUT_BRIDGE, &bridgeStorage));
        break;
    }
    }
}

/* Alias-capable: propLoss may share loss's bytes (same-index read-then-write),
 * as an in-place dx plan binds them. */
static void dropoutBackwardFloat(dropoutConfig_t *cfg, tensor_t *loss, tensor_t *propLoss) {
    size_t numberOfElements = calcNumberOfElementsByTensor(loss);
    float *gradOut = (float *)loss->data;
    float *gradIn = (float *)propLoss->data;
    float scale = dropoutScale(cfg->p);

    for (size_t i = 0; i < numberOfElements; i++) {
        gradIn[i] = tensorBoolGet(cfg->mask, i) ? gradOut[i] * scale : 0.0f;
    }
}

/* Alias-capable: propLoss may share loss's bytes (same-index read-then-write),
 * as an in-place dx plan binds them; the two wires keep their own qConfigs, so
 * the scale fold below writes propLoss's config only. */
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
    /* scale-fold: ints copied unchanged, the 1/(1-p) factor goes into the quant
     * scale -- canonicalized NEAREST, folded with odtScaleMul (canonCoeffScale) */
    storeCanonScale(&propLossQC->scale, canonCoeffScale(loadCanonScale(&lossQC->scale), scale));
}

void dropoutBackward(layer_t *dropoutLayer, tensor_t *forwardInput, tensor_t *loss,
                     tensor_t *propLoss, const layerExec_t *exec) {
    (void)forwardInput; // not needed: the stored mask + p fully determine the gradient.
    dropoutConfig_t *cfg = dropoutLayer->config->dropout;
    dropoutRequireMaskCount(cfg, loss, "Dropout backward");
    dropoutRequireWires(loss, propLoss, cfg->propLossMath.type, "Dropout backward");
    switch (cfg->propLossMath.type) {
    case ARITH_FLOAT32:
        dropoutBackwardFloat(cfg, loss, propLoss);
        break;
    case ARITH_SYM_INT32:
        dropoutBackwardSymInt32(cfg, loss, propLoss);
        break;
    case ARITH_BFP: {
        rngStream_t bridgeStorage;
        dropoutMaskScaleBfp(
            cfg, loss, propLoss,
            layerExecPurposeStream(exec, RNG_TAG_BACKWARD, RNG_TAG_DROPOUT_BRIDGE, &bridgeStorage));
        break;
    }
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
