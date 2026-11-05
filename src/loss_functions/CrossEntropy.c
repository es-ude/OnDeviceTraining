#define SOURCE_FILE "CROSS_ENTROPY"

#include <limits.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "BfpKernelSupport.h"
#include "Common.h"
#include "CrossEntropy.h"
#include "ExecuteOp.h"
#include "LayerExec.h"
#include "Log.h"
#include "LossFunctionInternal.h"
#include "TensorConversion.h"

#include <math.h>

float crossEntropyForwardFloat(tensor_t *softmaxOutput, tensor_t *distribution,
                               reduction_t reduction) {
    size_t n = calcNumberOfElementsByTensor(softmaxOutput);
    float *p = (float *)softmaxOutput->data;
    float *y = (float *)distribution->data;

    float loss = 0.f;
    for (size_t i = 0; i < n; i++) {
        float pi = p[i];

        if (!isfinite(pi)) {
            /* #446: a diverged model (NaN/Inf logits) must not take the
             * process down. Return NaN so the training loop can see the
             * divergence (trainingRunOptions_t.stopOnNonFiniteLoss) and the
             * harness can log it. Printed once per process, not per call:
             * at batch 1 this runs thousands of times per epoch. */
            static int reported = 0;
            if (!reported) {
                PRINT_ERROR("crossEntropyForward: non-finite softmax output at element %zu "
                            "(%f); returning NaN loss (#446)",
                            i, (double)pi);
                reported = 1;
            }
            return NAN;
        }

        pi = fmaxf(pi, 1e-7f);
        loss += y[i] * -logf(pi);
    }

    /* Rank < 2 only via a direct arm call (the dispatchers reject it, #153): no
     * batch axis, so MEAN is the raw sum. */
    if (reduction == REDUCTION_MEAN && softmaxOutput->shape->numberOfDimensions >= 2) {
        size_t microbatch = softmaxOutput->shape->dimensions[0];
        return loss / (float)microbatch;
    }
    return loss;
}

/* Fake-quant forward (#206, M5 — the MSE idiom): dequantize both operands
 * into FLOAT32 stack scratch (convertTensor handles any source dtype; a
 * FLOAT32 label passes through as a copy) and run the float core on the
 * scratch views. The scratch tensors share softmaxOutput's shape/sparsity
 * pointers, so the core's microbatch MEAN branch sees the real
 * dimensions[0].
 * Since BFP epic PR4 (R-P6) the same body also serves BFP: convertTensor hits
 * conversionMatrix[BFP][FLOAT32] on the read side and, in the backward,
 * conversionMatrix[FLOAT32][BFP] (fresh per-group exponents) on the write
 * side. */
static float crossEntropyForwardFakeQuant(tensor_t *softmaxOutput, tensor_t *distribution,
                                          reduction_t reduction) {
    size_t inputSize = calcNumberOfElementsByTensor(softmaxOutput);

    tensor_t softmaxOutputFloat;
    quantization_t softmaxOutputFloatQ;
    initFloat32Quantization(&softmaxOutputFloatQ);
    uint8_t softmaxOutputFloatData[inputSize * sizeof(float)];
    setTensorValuesForConversion(softmaxOutputFloatData, &softmaxOutputFloatQ, softmaxOutput,
                                 &softmaxOutputFloat);
    convertTensor(softmaxOutput, &softmaxOutputFloat, NULL);

    tensor_t distributionFloat;
    quantization_t distributionFloatQ;
    initFloat32Quantization(&distributionFloatQ);
    uint8_t distributionFloatData[inputSize * sizeof(float)];
    setTensorValuesForConversion(distributionFloatData, &distributionFloatQ, distribution,
                                 &distributionFloat);
    convertTensor(distribution, &distributionFloat, NULL);

    return crossEntropyForwardFloat(&softmaxOutputFloat, &distributionFloat, reduction);
}

/* The loss VALUE is monitoring only (logged, the divergence check): no
 * gradient is computed from it, so it stays float on every head -- also for
 * the integer head, whose gradient is crossEntropySoftmaxBackwardKernelBfp. */
float crossEntropyForward(tensor_t *softmaxOutput, tensor_t *distribution, reduction_t reduction,
                          const layerExec_t *exec) {
    requireOperandMatchesOutput(softmaxOutput, distribution, "distribution", "crossEntropyForward");
    switch (softmaxOutput->quantization->type) {
    case FLOAT32:
        /* The float arm raw-casts BOTH operands: a quantized label next to a
         * FLOAT32 output goes through the dequantizing arm instead. */
        if (distribution->quantization->type == FLOAT32) {
            return crossEntropyForwardFloat(softmaxOutput, distribution, reduction);
        }
        return crossEntropyForwardFakeQuant(softmaxOutput, distribution, reduction);
    case SYM_INT32:
    case BFP:
        return crossEntropyForwardFakeQuant(softmaxOutput, distribution, reduction);
    default:
        PRINT_ERROR("crossEntropyForward: model-output dtype %d not supported "
                    "(FLOAT32/SYM_INT32/BFP)",
                    (int)softmaxOutput->quantization->type);
        exit(1);
    }
}

/* Out of line (BFP_WALK_NOINLINE) so the dispatcher's own body holds no
 * float: the bfp-census checks it as a `self` root and this arm as the float
 * head's island. */
BFP_WALK_NOINLINE void crossEntropySoftmaxBackwardFloat(tensor_t *softmaxOutput,
                                                        tensor_t *distribution, tensor_t *loss) {
    size_t totalInputSize = calcNumberOfElementsByTensor(softmaxOutput);

    float *softmaxOutputFloat = (float *)softmaxOutput->data;
    float *distributionFloat = (float *)distribution->data;
    float *lossFloat = (float *)loss->data;

    for (size_t i = 0; i < totalInputSize; i++) {
        lossFloat[i] = softmaxOutputFloat[i] - distributionFloat[i];
    }
}

/* Fake-quant backward — dtype-generic: every operand goes through
 * convertTensor, so the same body serves ASYM (its original home) and
 * SYM_INT32 (#206). The final convertTensor requantizes the raw (p-y) into
 * the result's own quantized config (fresh absmax scale for SYM — dynamic
 * activation scaling, like every quantized wire producer), drawing from rs. */
static void crossEntropySoftmaxBackwardFakeQuant(tensor_t *softmaxOutput, tensor_t *distribution,
                                                 tensor_t *loss, rngStream_t *rs) {
    size_t inputSize = calcNumberOfElementsByTensor(softmaxOutput);

    tensor_t softmaxOutputFloat;
    quantization_t softmaxOutputFloatQ;
    initFloat32Quantization(&softmaxOutputFloatQ);
    uint8_t softmaxOutputFloatData[inputSize * sizeof(float)];
    setTensorValuesForConversion(softmaxOutputFloatData, &softmaxOutputFloatQ, softmaxOutput,
                                 &softmaxOutputFloat);
    convertTensor(softmaxOutput, &softmaxOutputFloat, NULL);

    tensor_t distributionFloat;
    quantization_t distributionFloatQ;
    initFloat32Quantization(&distributionFloatQ);
    uint8_t distributionFloatData[inputSize * sizeof(float)];
    setTensorValuesForConversion(distributionFloatData, &distributionFloatQ, distribution,
                                 &distributionFloat);
    convertTensor(distribution, &distributionFloat, NULL);

    /* Write-only scratch: the loop below overwrites every element, so the
     * pre-call contents of `loss` are never dequantized (MSE.c precedent —
     * mseLossBackwardFakeQuant skips the same wasted convert). */
    tensor_t lossFloat;
    quantization_t lossFloatQ;
    initFloat32Quantization(&lossFloatQ);
    uint8_t lossFloatData[inputSize * sizeof(float)];
    setTensorValuesForConversion(lossFloatData, &lossFloatQ, loss, &lossFloat);

    float *softmaxOutputFloatArr = (float *)softmaxOutputFloat.data;
    float *distributionFloatArr = (float *)distributionFloat.data;
    float *lossFloatArr = (float *)lossFloat.data;

    for (size_t i = 0; i < inputSize; i++) {
        lossFloatArr[i] = softmaxOutputFloatArr[i] - distributionFloatArr[i];
    }

    convertTensor(&lossFloat, loss, rs);
}

/* OP(k) of the funnel op of the loss boundary, for layerExecOpKey (the
 * context is the layer position after the last layer). Part of the value
 * contract: append, never renumber. */
enum {
    CROSS_ENTROPY_OP_GRAD = 0,
};

/* A FLOAT32 one-hot label element by its bit pattern -- 1.0f is the class,
 * +-0.0f is not, anything else fails fast -- so the integer head needs no
 * float compare. */
static bool ceLabelIsOne(const float *y, size_t i) {
    /* the bits through a union (defined in C; a plain word load, no float op
     * and no memcpy call on the M0+ build) */
    union {
        float f;
        uint32_t u;
    } label = {.f = y[i]};
    uint32_t bits = label.u;
    if (bits == 0x3F800000u) {
        return true;
    }
    if ((bits & 0x7FFFFFFFu) == 0u) {
        return false;
    }
    PRINT_ERROR("crossEntropySoftmaxBackward: the integer head's label element %zu (bits 0x%08lx) "
                "is not one-hot (0 or 1): soft labels need the float head",
                i, (unsigned long)bits);
    exit(1);
}

/* The integer term of element i: p's code c on its group grid E_g, minus 1.0
 * at the label. The label's grid is E = clamp(E_g, -30, 0) so that 1.0 =
 * 2^-E is an int32: exact for E_g in [-30, 0] (also when 1.0 is code
 * 2^(m-1), one beyond the m-bit maximum); a p group below 2^-30 rounds its
 * code half away onto 2^-30 first (< 2^-31 absolute, next to a value near
 * -1); E_g > 0 is the zero-code case only (a nonzero code there is p > 1).
 * |term| <= 2^30 + 2^15 < 2^31. */
static inline int32_t ceTermOf(int32_t c, int eG, bool one, size_t i, int *e) {
    if (!one) {
        *e = eG;
        return c;
    }
    if (eG > 0) {
        if (c != 0) {
            PRINT_ERROR("crossEntropySoftmaxBackward: the integer head's p at the label element "
                        "%zu is a probability above 1 (code %ld on 2^%d)",
                        i, (long)c, eG);
            exit(1);
        }
        *e = 0;
        return -1;
    }
    if (eG < -30) {
        *e = -30;
        return bfpShrHalfAway(c, (unsigned)(-30 - eG)) - ((int32_t)1 << 30);
    }
    *e = eG;
    return c - ((int32_t)1 << (unsigned)(-eG));
}

void crossEntropySoftmaxBackwardKernelBfp(tensor_t **ops, size_t n, tensor_t *rawOut,
                                          tensor_t *auxOut, const void *ctx, rngStream_t *rs) {
    (void)n;
    (void)auxOut;
    (void)rs;
    const char *what = "crossEntropySoftmaxBackward (integer head)";
    tensor_t *p = ops[0];
    const tensor_t *label = ctx;
    size_t count = calcNumberOfElementsByTensor(p);
    if (p->quantization->type != BFP || rawOut->quantization->type != BFP_RAW ||
        calcNumberOfElementsByTensor(rawOut) != count) {
        PRINT_ERROR("%s: needs a BFP p (unpacked scratch) and a BFP_RAW output of the same "
                    "element count, got dtypes %d -> %d",
                    what, (int)p->quantization->type, (int)rawOut->quantization->type);
        exit(1);
    }
    if (label->quantization->type != FLOAT32 ||
        calcNumberOfElementsByTensor((tensor_t *)label) != count) {
        PRINT_ERROR("%s: the integer head needs a FLOAT32 one-hot label of p's element count, got "
                    "dtype %d",
                    what, (int)label->quantization->type);
        exit(1);
    }
    const bfpQConfig_t *qC = p->quantization->qConfig;
    validateBfpQConfigShape(qC, count);
    bfpRawQConfig_t *rawQC = rawOut->quantization->qConfig;
    size_t gsz = bfpRawRequireOutput(rawQC, count, what);
    if (count == 0) {
        return;
    }
    const int32_t expBias = bfpExponentBias(qC);
    const int32_t *codes = (const int32_t *)p->data;
    const float *y = (const float *)label->data;
    int32_t *mant = (int32_t *)rawOut->data;
    int16_t *rawE = rawQC->exponents;
    const size_t pGroupSize = qC->groupSize == 0 ? count : qC->groupSize;

    /* Per raw target group, two passes over its terms (recomputed, no
     * table): the data rule E_t = max over its nonzero terms of E +
     * bitlen(|term|), minus 31 (bfpRawEmitFromRows' rule), then every term
     * onto 2^E_t -- left exact, right half away only for a term more than 31
     * binades below the group's top. p's group by cursor: no division. */
    size_t pGroup = 0;
    size_t pGroupEnd = pGroupSize;
    for (size_t t = 0, start = 0; start < count; t++, start += gsz) {
        size_t stop = start + gsz;
        size_t g0 = pGroup;
        size_t g0End = pGroupEnd;
        for (int pass = 0; pass < 2; pass++) {
            pGroup = g0;
            pGroupEnd = g0End;
            int top = INT_MIN;
            for (size_t i = start; i < stop; i++) {
                if (i == pGroupEnd) {
                    pGroup++;
                    pGroupEnd += pGroupSize;
                }
                int e;
                int32_t v = ceTermOf(codes[i], (int)qC->exponents[pGroup] - expBias,
                                     ceLabelIsOne(y, i), i, &e);
                if (pass == 0) {
                    if (v != 0) {
                        uint32_t mag = v < 0 ? 0u - (uint32_t)v : (uint32_t)v;
                        int eTop = e + (int)bfpBitLength32(mag);
                        top = eTop > top ? eTop : top;
                    }
                } else {
                    mant[i] = v == 0 ? 0 : bfpAlignPartial(v, (int)rawE[t] - e);
                }
            }
            if (pass == 0) {
                rawE[t] = (int16_t)(top == INT_MIN ? 0 : top - 31);
            }
        }
    }
}

/* The integer head (ruling R2): p - y with no float. One funnel op: p is
 * borrowed as unpacked codes, the label rides in the kernel context (no
 * staging), the raw is BFP_RAW_BACKWARD_DX on the loss-grad wire's geometry,
 * and the OUT_WRITE epilogue re-blocks it by the WIRE's rounding mode (the
 * template of the Softmax's propLossQ: SR under SR), drawing from this op's
 * OUT stream -- the training-side seam rule of every dx wire. */
static void crossEntropySoftmaxBackwardBfp(tensor_t *softmaxOutput, tensor_t *distribution,
                                           tensor_t *loss, const layerExec_t *exec) {
    if (loss->quantization->type != BFP) {
        PRINT_ERROR("crossEntropySoftmaxBackward: a BFP softmax output selects the integer head, "
                    "and the integer head needs a BFP loss-grad wire, got dtype %d (a float head "
                    "has a FLOAT32 softmax output)",
                    (int)loss->quantization->type);
        exit(1);
    }
    if (distribution->quantization->type != FLOAT32) {
        PRINT_ERROR("crossEntropySoftmaxBackward: the integer head needs a FLOAT32 one-hot label, "
                    "got dtype %d",
                    (int)distribution->quantization->type);
        exit(1);
    }
    const bfpQConfig_t *target = loss->quantization->qConfig;
    rngKey_t opKey;
    executeOp(
        &(opSpec_t){
            .kernel = crossEntropySoftmaxBackwardKernelBfp,
            .ctx = distribution,
            .inputs = (tensor_t *[]){softmaxOutput},
            .nInputs = 1,
            .arithmetic = {.type = ARITH_BFP, .roundingMode = target->roundingMode},
            .key = layerExecOpKey(exec, RNG_TAG_BACKWARD, CROSS_ENTROPY_OP_GRAD, &opKey),
            .mode = OUT_WRITE,
            .bfpRawOut = BFP_RAW_BACKWARD_DX,
        },
        loss);
}

/* The combined Softmax + CrossEntropy gradient p - y: the training loop skips
 * the model's last layer (which must be the Softmax, rematBackwardRange) and
 * seeds the backward with this, on the Softmax's dx wire (its propLossQ).
 * The model's HEAD -- the Softmax forward plus this gradient -- is declared
 * by the Softmax's config and validated at init (requireCrossEntropyHead,
 * arithmetic-bfp.md section 14); the arm here follows p's dtype:
 *   - BFP p (head bfp): the integer p - y above, no float.
 *   - FLOAT32 p (head float, and every FLOAT32 model): the float island --
 *     p - y in float, then ONE conversion into the grad wire's own dtype and
 *     config (convertTensor, drawing from the RNG_TAG_LOSS stream under SR,
 *     rounding by the wire template's mode). The float arm raw-casts all
 *     three operands, so it runs only when all three are FLOAT32; a
 *     quantized label or grad wire takes the converting arm, so a float can
 *     never be written into a packed payload.
 *   - SYM_INT32 / ASYM p: the dequantizing island (fake-quant), unchanged. */
void crossEntropySoftmaxBackward(tensor_t *softmaxOutput, tensor_t *distribution, tensor_t *loss,
                                 const layerExec_t *exec) {
    requireOperandMatchesOutput(softmaxOutput, distribution, "distribution",
                                "crossEntropySoftmaxBackward");
    requireOperandMatchesOutput(softmaxOutput, loss, "loss (grad wire)",
                                "crossEntropySoftmaxBackward");
    rngStream_t lossStorage;
    switch (softmaxOutput->quantization->type) {
    case FLOAT32:
        if (distribution->quantization->type == FLOAT32 && loss->quantization->type == FLOAT32) {
            crossEntropySoftmaxBackwardFloat(softmaxOutput, distribution, loss);
            break;
        }
        crossEntropySoftmaxBackwardFakeQuant(
            softmaxOutput, distribution, loss,
            layerExecPurposeStream(exec, RNG_TAG_BACKWARD, RNG_TAG_LOSS, &lossStorage));
        break;
    case BFP:
        crossEntropySoftmaxBackwardBfp(softmaxOutput, distribution, loss, exec);
        break;
    case SYM_INT32:
    case ASYM:
        crossEntropySoftmaxBackwardFakeQuant(
            softmaxOutput, distribution, loss,
            layerExecPurposeStream(exec, RNG_TAG_BACKWARD, RNG_TAG_LOSS, &lossStorage));
        break;
    default:
        PRINT_ERROR("crossEntropySoftmaxBackward: model-output dtype %d not supported "
                    "(FLOAT32/SYM_INT32/ASYM/BFP)",
                    (int)softmaxOutput->quantization->type);
        exit(1);
    }
}

float computeMeanScaleCE(size_t totalSamples, tensor_t *modelOutput) {
    (void)modelOutput;
    return 1.0f / (float)totalSamples;
}

size_t computeMeanDivisorCE(size_t totalSamples, tensor_t *modelOutput) {
    (void)modelOutput;
    return totalSamples;
}
