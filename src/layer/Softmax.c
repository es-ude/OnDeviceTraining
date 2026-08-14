#define SOURCE_FILE "SOFTMAX"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ArithmeticType.h"
#include "Common.h"
#include "ExecuteOp.h"
#include "Softmax.h"
#include "TensorConversion.h"

void softmaxInitConfig(softmaxConfig_t *softmaxConfig, quantization_t *forwardQ,
                       quantization_t *backwardQ) {
    softmaxConfig->forwardMath = arithmeticFromQuantizationOrDefault(forwardQ);
    softmaxConfig->propLossMath = arithmeticFromQuantizationOrDefault(backwardQ);
    softmaxConfig->outputQ = forwardQ;
    softmaxConfig->propLossQ = backwardQ;
}

void softmaxInitLayer(layerConfig_t *softmaxConfig, layer_t *softmaxLayer) {
    softmaxLayer->type = SOFTMAX;
    softmaxLayer->config = softmaxConfig;
}

/* Softmax's real compute is always float (numerically stable max-shifted exp);
 * SYM_INT32 forwardMath only ever meant "convert in, compute in float, convert
 * out" (never native SYM arithmetic like Linear/Conv), so the funnel's
 * prologue/epilogue perform that conversion automatically. Arithmetic is
 * hardcoded ARITH_FLOAT32 here on purpose — forwardMath no longer selects a
 * compute path, it only declares the layer's storage dtype. */
static void softmaxForwardKernel(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                                 const void *ctx) {
    (void)n;
    (void)auxOut;
    (void)ctx;
    tensor_t *input = ops[0];
    size_t count = calcNumberOfElementsByTensor(input);

    float *x = (float *)input->data;
    float *y = (float *)rawOut->data;

    // 1. find max
    float max = x[0];
    for (size_t i = 1; i < count; i++) {
        if (x[i] > max) {
            max = x[i];
        }
    }

    // 2. exp and sum
    float sum = 0.f;
    for (size_t i = 0; i < count; i++) {
        float e = expf(x[i] - max);
        y[i] = e;
        sum += e;
    }

    // 3. normalize
    for (size_t i = 0; i < count; i++) {
        y[i] /= sum;
    }
}

void softmaxForward(layer_t *softmaxLayer, tensor_t *input, tensor_t *output) {
    (void)softmaxLayer;
    executeOp(
        &(opSpec_t){
            .kernel = softmaxForwardKernel,
            .inputs = (tensor_t *[]){input},
            .nInputs = 1,
            .arithmetic = (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY},
            .mode = OUT_WRITE,
        },
        output);
}

/* BFP epic PR2 Task 8: softmaxBackward is the fourth outside-funnel site (after
 * Relu/Dropout/Flatten). Its ARITH_FLOAT32 arm raw-casts all three wires to
 * float* with no dtype check at all, and it is selected by the layer's DECLARED
 * propLossMath -- so a BFP dx wire whose math slot is pinned to (or, before the
 * Task 9 flip, derived as) ARITH_FLOAT32 lands straight in the raw casts: a 4x
 * heap over-read on input/loss and an over-write into the packed propLoss buffer.
 * That became reachable only with this task's initGradTensor BFP arm (before it,
 * a BFP propLossQ died in the allocator's default arm).
 *
 * Forward needs no guard: it runs inside executeOp, whose prologue/epilogue
 * convert both ways. PR6, not PR4 — softmax BFP semantics belong to research
 * package II. */
static void requireNoBfpWire(const tensor_t *t, const char *what) {
    if (t->quantization->type == BFP) {
        PRINT_ERROR("%s: BFP Softmax semantics arrive with epic PR6 -- keep BFP off this wire or "
                    "use FLOAT32 wires",
                    what);
        exit(1);
    }
}

static void softmaxBackwardFloat(tensor_t *input, tensor_t *loss, tensor_t *propLoss) {
    size_t n = calcNumberOfElementsByTensor(input);

    float *s = (float *)input->data;
    float *dLds = (float *)loss->data;
    float *dLdx = (float *)propLoss->data;

    float dot = 0.0f;
    for (size_t i = 0; i < n; i++) {
        dot += s[i] * dLds[i];
    }

    for (size_t i = 0; i < n; i++) {
        dLdx[i] = s[i] * (dLds[i] - dot);
    }
}

static void softmaxBackwardSymInt32(tensor_t *input, tensor_t *loss, tensor_t *propLoss) {
    size_t inputSize = calcNumberOfElementsByTensor(input);

    tensor_t inputFloat;
    quantization_t inputFloatQ;
    initFloat32Quantization(&inputFloatQ);
    uint8_t inputFloatData[inputSize * sizeof(float)];
    setTensorValuesForConversion(inputFloatData, &inputFloatQ, input, &inputFloat);
    convertTensor(input, &inputFloat);

    tensor_t lossFloat;
    quantization_t lossFloatQ;
    initFloat32Quantization(&lossFloatQ);
    uint8_t lossFloatData[inputSize * sizeof(float)];
    setTensorValuesForConversion(lossFloatData, &lossFloatQ, loss, &lossFloat);
    convertTensor(loss, &lossFloat);

    tensor_t propLossFloat;
    quantization_t propLossFloatQ;
    initFloat32Quantization(&propLossFloatQ);
    uint8_t propLossFloatData[inputSize * sizeof(float)];
    setTensorValuesForConversion(propLossFloatData, &propLossFloatQ, propLoss, &propLossFloat);
    convertTensor(propLoss, &propLossFloat);

    float *s = (float *)inputFloat.data;
    float *dLds = (float *)lossFloat.data;
    float *dLdx = (float *)propLossFloat.data;

    float dot = 0.0f;
    for (size_t i = 0; i < inputSize; i++) {
        dot += s[i] * dLds[i];
    }

    for (size_t i = 0; i < inputSize; i++) {
        dLdx[i] = s[i] * (dLds[i] - dot);
    }

    convertTensor(&propLossFloat, propLoss);
}

void softmaxBackward(layer_t *softmaxLayer, tensor_t *input, tensor_t *loss, tensor_t *propLoss) {
    /* Before the dispatch (the Relu placement): all three wires are dereferenced
     * by whichever arm runs, and the check is on STORAGE dtype, not the declared
     * arithmetic that selects the arm. */
    requireNoBfpWire(input, "Softmax backward (input)");
    requireNoBfpWire(loss, "Softmax backward (loss)");
    requireNoBfpWire(propLoss, "Softmax backward (propLoss)");

    switch (softmaxLayer->config->softmax->propLossMath.type) {
    case ARITH_FLOAT32:
        softmaxBackwardFloat(input, loss, propLoss);
        break;
    case ARITH_SYM_INT32:
        softmaxBackwardSymInt32(input, loss, propLoss);
        break;
    default:
        PRINT_ERROR("Softmax backward: declared propLossMath %d not implemented "
                    "(FLOAT32/SYM_INT32 only) -- native BFP softmax arrives with epic PR6",
                    (int)softmaxLayer->config->softmax->propLossMath.type);
        exit(1);
    }
}

void softmaxCalcOutputShape(layer_t *softmaxLayer, shape_t *inputShape, shape_t *outputShape) {
    memcpy(outputShape->dimensions, inputShape->dimensions,
           inputShape->numberOfDimensions * sizeof(size_t));
    memcpy(outputShape->orderOfDimensions, inputShape->orderOfDimensions,
           inputShape->numberOfDimensions * sizeof(size_t));
    outputShape->numberOfDimensions = inputShape->numberOfDimensions;
}
