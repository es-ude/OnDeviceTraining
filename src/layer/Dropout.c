#define SOURCE_FILE "DROPOUT"

#include <stdlib.h>
#include <string.h>

#include "Dropout.h"

#include "ArithmeticType.h"
#include "Bernoulli.h"
#include "Common.h"
#include "Layer.h"
#include "Quantization.h"
#include "Tensor.h"

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

/* BFP epic PR2 Task 8: Dropout runs OUTSIDE the executeOp funnel — every arm
 * below raw-casts ->data and, for SYM_INT32, folds 1/(1-p) into the wire's
 * scale. A BFP wire stores PACKED mantissa codes under a per-GROUP exponent, so
 * such a view reads packed bytes as wide scalars and leaves the destination's
 * exponents stale (there is no single scale to fold the keep-probability into
 * either): silent corruption, never a crash. Keyed on the wire's STORAGE dtype,
 * not the declared arithmetic. */
static void requireNoBfpWire(const tensor_t *t, const char *what) {
    if (t->quantization->type == BFP) {
        PRINT_ERROR("%s: BFP Dropout semantics arrive with epic PR4 -- keep BFP off this wire or "
                    "use FLOAT32 wires",
                    what);
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

void dropoutForward(layer_t *dropoutLayer, tensor_t *input, tensor_t *output) {
    dropoutConfig_t *cfg = dropoutLayer->config->dropout;
    /* Before bernoulliFillMask: a rejected call must not consume RNG draws. */
    requireNoBfpWire(input, "Dropout forward (input)");
    requireNoBfpWire(output, "Dropout forward (output)");
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
    switch (cfg->forwardMath.type) {
    case ARITH_FLOAT32:
        dropoutForwardFloat(cfg, input, output);
        break;
    case ARITH_SYM_INT32:
        dropoutForwardSymInt32(cfg, input, output);
        break;
    default:
        PRINT_ERROR("Dropout forward: quantization type not implemented");
        exit(1);
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
    requireNoBfpWire(loss, "Dropout backward (loss)");
    requireNoBfpWire(propLoss, "Dropout backward (propLoss)");
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
