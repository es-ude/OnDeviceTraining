#define SOURCE_FILE "CROSS_ENTROPY"

#include <stdlib.h>

#include "Common.h"
#include "CrossEntropy.h"
#include "Log.h"
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
            printf("NaN/Inf softmax at %zu\n", i);
            printf("%f\n", pi);
            abort();
        }

        pi = fmaxf(pi, 1e-7f);
        loss += y[i] * -logf(pi);
    }

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
 * dimensions[0]. */
static float crossEntropyForwardFakeQuant(tensor_t *softmaxOutput, tensor_t *distribution,
                                          reduction_t reduction) {
    size_t inputSize = calcNumberOfElementsByTensor(softmaxOutput);

    tensor_t softmaxOutputFloat;
    quantization_t softmaxOutputFloatQ;
    initFloat32Quantization(&softmaxOutputFloatQ);
    uint8_t softmaxOutputFloatData[inputSize * sizeof(float)];
    setTensorValuesForConversion(softmaxOutputFloatData, &softmaxOutputFloatQ, softmaxOutput,
                                 &softmaxOutputFloat);
    convertTensor(softmaxOutput, &softmaxOutputFloat);

    tensor_t distributionFloat;
    quantization_t distributionFloatQ;
    initFloat32Quantization(&distributionFloatQ);
    uint8_t distributionFloatData[inputSize * sizeof(float)];
    setTensorValuesForConversion(distributionFloatData, &distributionFloatQ, distribution,
                                 &distributionFloat);
    convertTensor(distribution, &distributionFloat);

    return crossEntropyForwardFloat(&softmaxOutputFloat, &distributionFloat, reduction);
}

float crossEntropyForward(tensor_t *softmaxOutput, tensor_t *distribution, reduction_t reduction) {
    switch (softmaxOutput->quantization->type) {
    case FLOAT32:
        return crossEntropyForwardFloat(softmaxOutput, distribution, reduction);
    case SYM_INT32:
        return crossEntropyForwardFakeQuant(softmaxOutput, distribution, reduction);
    default:
        PRINT_ERROR("crossEntropyForward: model-output dtype %d not supported (FLOAT32/SYM_INT32) "
                    "-- BFP loss arms arrive with epic PR4; keep the loss-facing wire FLOAT32 (see "
                    "docs/conventions/arithmetic-bfp.md)",
                    (int)softmaxOutput->quantization->type);
        exit(1);
    }
}

static void crossEntropySoftmaxBackwardFloat(tensor_t *softmaxOutput, tensor_t *distribution,
                                             tensor_t *loss) {
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
 * activation scaling, like every quantized wire producer). */
static void crossEntropySoftmaxBackwardFakeQuant(tensor_t *softmaxOutput, tensor_t *distribution,
                                                 tensor_t *loss) {
    size_t inputSize = calcNumberOfElementsByTensor(softmaxOutput);

    tensor_t softmaxOutputFloat;
    quantization_t softmaxOutputFloatQ;
    initFloat32Quantization(&softmaxOutputFloatQ);
    uint8_t softmaxOutputFloatData[inputSize * sizeof(float)];
    setTensorValuesForConversion(softmaxOutputFloatData, &softmaxOutputFloatQ, softmaxOutput,
                                 &softmaxOutputFloat);
    convertTensor(softmaxOutput, &softmaxOutputFloat);

    tensor_t distributionFloat;
    quantization_t distributionFloatQ;
    initFloat32Quantization(&distributionFloatQ);
    uint8_t distributionFloatData[inputSize * sizeof(float)];
    setTensorValuesForConversion(distributionFloatData, &distributionFloatQ, distribution,
                                 &distributionFloat);
    convertTensor(distribution, &distributionFloat);

    /* Write-only scratch: the loop below overwrites every element, so the
     * pre-call contents of `loss` are never dequantized (MSE.c precedent —
     * mseLossBackwardSymInt32 skips the same wasted convert). */
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

    convertTensor(&lossFloat, loss);
}

// IMPORTANT: This implementation already takes the softmax backward into account
void crossEntropySoftmaxBackward(tensor_t *softmaxOutput, tensor_t *distribution, tensor_t *loss) {
    switch (softmaxOutput->quantization->type) {
    case FLOAT32:
        crossEntropySoftmaxBackwardFloat(softmaxOutput, distribution, loss);
        break;
    case SYM_INT32:
    case ASYM:
        crossEntropySoftmaxBackwardFakeQuant(softmaxOutput, distribution, loss);
        break;
    default:
        PRINT_ERROR("crossEntropySoftmaxBackward: model-output dtype %d not supported "
                    "(FLOAT32/SYM_INT32/ASYM) -- BFP loss arms arrive with epic PR4; keep the "
                    "loss-facing wire FLOAT32 (see docs/conventions/arithmetic-bfp.md)",
                    (int)softmaxOutput->quantization->type);
        exit(1);
    }
}

float computeMeanScaleCE(size_t totalSamples, tensor_t *modelOutput) {
    (void)modelOutput;
    return 1.0f / (float)totalSamples;
}
