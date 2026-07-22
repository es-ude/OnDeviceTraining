#define SOURCE_FILE "SERIALIZE"

#include <stdlib.h>

#include "AdaptiveAvgPool1d.h"
#include "ArithmeticType.h"
#include "AvgPool1d.h"
#include "Common.h"
#include "Conv1d.h"
#include "Conv1dTransposed.h"
#include "Dropout.h"
#include "GroupNorm.h"
#include "Kernel.h"
#include "Layer.h"
#include "LayerNorm.h"
#include "Linear.h"
#include "MaxPool1d.h"
#include "QuantizationLayer.h"
#include "Relu.h"
#include "Rounding.h"
#include "SerialWire.h"
#include "Serialize.h"
#include "SerializeInternal.h"
#include "Softmax.h"
#include "Tensor.h"

/* Locked format v2 (#370): magic "ODTS" + u32 version + u32 layerCount, then
 * one Record per layer (u8 tag + payload). Every count/dim/kernel field is u32
 * little-endian and every scalar goes through the checked SerialWire
 * primitives, so a model written on a 64-bit host loads bit-identically on
 * 32-bit MCU targets. ASYM zeroPoint is i32 LE on the wire, matching the
 * int32 in-memory field (#246).
 * v3: parameter records carry a grad-presence byte (#380). Frozen layers
 * (parameter->grad == NULL, layer freezing epic) write hasGrad=0 and skip the
 * grad tensor entirely; deserialize is TOLERANT of a presence/skeleton
 * mismatch (#380 PR3) -- grads are skipped into frozen skeletons or left
 * zeroed when absent, never NULL-dereferenced; see Deserialize.c. */
#define SERIALIZE_MAGIC "ODTS"
#define SERIALIZE_FORMAT_VERSION 3u

void serializeTensor(tensor_t *tensor, FILE *f) {
    size_t numberOfValues = calcNumberOfElementsByTensor(tensor);
    /* Data payload = packed size (calcNumberOfBytesForData) -- byte-identical to
     * N * elementSize for byte-aligned dtypes, packed-tight for sub-byte (#172). */
    size_t dataBytes = calcNumberOfBytesForData(tensor->quantization, numberOfValues);

    serializeShape(tensor->shape, f);
    serializeQuantization(tensor->quantization, f);
    serialWriteBytes(tensor->data, dataBytes, f);
    serializeSparsity();
}

void serializeParameter(parameter_t *parameter, FILE *f) {
    /* v3 (#380): frozen layers carry no grad tensor - presence byte first. */
    uint8_t hasGrad = parameter->grad != NULL ? 1u : 0u;
    serialWriteU8(hasGrad, f);
    serializeTensor(parameter->param, f);
    if (hasGrad) {
        serializeTensor(parameter->grad, f);
    }
}

void serializeModel(layer_t **model, size_t sizeModel, FILE *f) {
    serialWriteBytes(SERIALIZE_MAGIC, 4, f);
    serialWriteU32LE(SERIALIZE_FORMAT_VERSION, f);
    serialWriteSizeAsU32LE(sizeModel, f);

    for (size_t i = 0; i < sizeModel; i++) {
        /* The tag byte is the layerType_t enum position -- append-only wire
         * contract, pinned in test/unit/serial/UnitTestSerialize.c. */
        serialWriteU8((uint8_t)model[i]->type, f);
        serializeLayer(model[i], f);
    }
}

// Helper Functions

static void serializeShape(shape_t *shape, FILE *f) {
    serialWriteSizeAsU32LE(shape->numberOfDimensions, f);
    for (size_t d = 0; d < shape->numberOfDimensions; d++) {
        serialWriteSizeAsU32LE(shape->dimensions[d], f);
    }
    for (size_t d = 0; d < shape->numberOfDimensions; d++) {
        serialWriteSizeAsU32LE(shape->orderOfDimensions[d], f);
    }
}

static void serializeQuantization(quantization_t *q, FILE *f) {
    serialWriteU8((uint8_t)q->type, f);
    serializeQConfig(q, f);
}

static void serializeArithmetic(arithmetic_t *arithmetic, FILE *f) {
    serialWriteU8((uint8_t)arithmetic->type, f);
    serialWriteU8((uint8_t)arithmetic->roundingMode, f);
}

static void serializeKernel(kernel_t *kernel, FILE *f) {
    serialWriteSizeAsU32LE(kernel->size, f);
    serialWriteU8((uint8_t)kernel->paddingType, f);
    serialWriteSizeAsU32LE(kernel->stride, f);
    serialWriteSizeAsU32LE(kernel->dilation, f);
    serialWriteSizeAsU32LE(kernel->padding, f);
}

static void serializeQConfig(quantization_t *q, FILE *f) {
    switch (q->type) {
    case INT32:
    case FLOAT32:
    case BOOL:
        break;
    case SYM_INT32: {
        symInt32QConfig_t *symIntQC = q->qConfig;
        serialWriteF32LE(symIntQC->scale, f);
        serialWriteU8((uint8_t)symIntQC->roundingMode, f);
        serialWriteU8(symIntQC->qMaxBits, f);
        break;
    }
    case SYM: {
        symQConfig_t *symQC = q->qConfig;
        serialWriteF32LE(symQC->scale, f);
        serialWriteU8(symQC->qBits, f);
        serialWriteU8((uint8_t)symQC->roundingMode, f);
        break;
    }
    case ASYM: {
        asymQConfig_t *asymQC = q->qConfig;
        serialWriteF32LE(asymQC->scale, f);
        serialWriteU8(asymQC->qBits, f);
        serialWriteU8((uint8_t)asymQC->roundingMode, f);
        serialWriteI32LE(asymQC->zeroPoint, f);
        break;
    }
    default:
        PRINT_ERROR("Unknown qType!");
        exit(1);
    }
}

// TODO
static void serializeSparsity() {}

static void serializeLayer(layer_t *layer, FILE *f) {
    switch (layer->type) {
    case LINEAR: {
        linearConfig_t *linearConfig = layer->config->linear;
        serializeParameter(linearConfig->weights, f);
        serializeParameter(linearConfig->bias, f);
        serializeArithmetic(&linearConfig->forwardMath, f);
        serializeArithmetic(&linearConfig->weightGradMath, f);
        serializeArithmetic(&linearConfig->biasGradMath, f);
        serializeArithmetic(&linearConfig->propLossMath, f);
        serializeQuantization(linearConfig->outputQ, f);
        serializeQuantization(linearConfig->propLossQ, f);
        break;
    }
    case RELU: {
        reluConfig_t *reluConfig = layer->config->relu;
        serializeArithmetic(&reluConfig->forwardMath, f);
        serializeArithmetic(&reluConfig->propLossMath, f);
        serializeQuantization(reluConfig->outputQ, f);
        serializeQuantization(reluConfig->propLossQ, f);
        break;
    }
    case CONV1D: {
        conv1dConfig_t *conv1dConfig = layer->config->conv1d;
        serializeKernel(conv1dConfig->kernel, f);
        serialWriteSizeAsU32LE(conv1dConfig->groups, f);
        serializeParameter(conv1dConfig->weights, f);
        uint8_t conv1dHasBias = conv1dConfig->bias != NULL ? 1 : 0;
        serialWriteU8(conv1dHasBias, f);
        if (conv1dHasBias) {
            serializeParameter(conv1dConfig->bias, f);
        }
        serializeArithmetic(&conv1dConfig->forwardMath, f);
        serializeArithmetic(&conv1dConfig->weightGradMath, f);
        serializeArithmetic(&conv1dConfig->biasGradMath, f);
        serializeArithmetic(&conv1dConfig->propLossMath, f);
        serializeQuantization(conv1dConfig->outputQ, f);
        serializeQuantization(conv1dConfig->propLossQ, f);
        break;
    }
    case CONV1D_TRANSPOSED: {
        conv1dTransposedConfig_t *conv1dTransposedConfig = layer->config->conv1dTransposed;
        serializeKernel(conv1dTransposedConfig->kernel, f);
        serialWriteSizeAsU32LE(conv1dTransposedConfig->groups, f);
        serialWriteSizeAsU32LE(conv1dTransposedConfig->outputPadding, f);
        serializeParameter(conv1dTransposedConfig->weights, f);
        uint8_t conv1dTransposedHasBias = conv1dTransposedConfig->bias != NULL ? 1 : 0;
        serialWriteU8(conv1dTransposedHasBias, f);
        if (conv1dTransposedHasBias) {
            serializeParameter(conv1dTransposedConfig->bias, f);
        }
        serializeArithmetic(&conv1dTransposedConfig->forwardMath, f);
        serializeArithmetic(&conv1dTransposedConfig->weightGradMath, f);
        serializeArithmetic(&conv1dTransposedConfig->biasGradMath, f);
        serializeArithmetic(&conv1dTransposedConfig->propLossMath, f);
        serializeQuantization(conv1dTransposedConfig->outputQ, f);
        serializeQuantization(conv1dTransposedConfig->propLossQ, f);
        break;
    }
    case MAXPOOL1D: {
        maxPool1dConfig_t *maxPool1dConfig = layer->config->maxPool1d;
        serializeKernel(maxPool1dConfig->kernel, f);
        serializeArithmetic(&maxPool1dConfig->forwardMath, f);
        serializeArithmetic(&maxPool1dConfig->propLossMath, f);
        serializeQuantization(maxPool1dConfig->outputQ, f);
        serializeQuantization(maxPool1dConfig->propLossQ, f);
        break;
    }
    case AVGPOOL1D: {
        avgPool1dConfig_t *avgPool1dConfig = layer->config->avgPool1d;
        serializeKernel(avgPool1dConfig->kernel, f);
        serializeArithmetic(&avgPool1dConfig->forwardMath, f);
        serializeArithmetic(&avgPool1dConfig->propLossMath, f);
        serializeQuantization(avgPool1dConfig->outputQ, f);
        serializeQuantization(avgPool1dConfig->propLossQ, f);
        break;
    }
    case SOFTMAX: {
        softmaxConfig_t *softmaxConfig = layer->config->softmax;
        serializeArithmetic(&softmaxConfig->forwardMath, f);
        serializeArithmetic(&softmaxConfig->propLossMath, f);
        serializeQuantization(softmaxConfig->outputQ, f);
        serializeQuantization(softmaxConfig->propLossQ, f);
        break;
    }
    case FLATTEN:
        // Flatten carries no state (no parameters, no quantization).
        break;
    case QUANTIZATION: {
        quantizationConfig_t *quantizationConfig = layer->config->quantization;
        serializeQuantization(quantizationConfig->outputQ, f);
        serializeQuantization(quantizationConfig->propLossQ, f);
        break;
    }
    case ADAPTIVE_AVGPOOL1D: {
        adaptiveAvgPool1dConfig_t *adaptiveAvgPool1dConfig = layer->config->adaptiveAvgPool1d;
        serialWriteSizeAsU32LE(adaptiveAvgPool1dConfig->outputSize, f);
        serializeArithmetic(&adaptiveAvgPool1dConfig->forwardMath, f);
        serializeArithmetic(&adaptiveAvgPool1dConfig->propLossMath, f);
        serializeQuantization(adaptiveAvgPool1dConfig->outputQ, f);
        serializeQuantization(adaptiveAvgPool1dConfig->propLossQ, f);
        break;
    }
    case DROPOUT: {
        dropoutConfig_t *dropoutConfig = layer->config->dropout;
        serialWriteF32LE(dropoutConfig->p, f);
        serializeArithmetic(&dropoutConfig->forwardMath, f);
        serializeArithmetic(&dropoutConfig->propLossMath, f);
        serializeQuantization(dropoutConfig->outputQ, f);
        serializeQuantization(dropoutConfig->propLossQ, f);
        break;
    }
    case LAYERNORM: {
        layerNormConfig_t *layerNormConfig = layer->config->layerNorm;
        serialWriteSizeAsU32LE(layerNormConfig->numNormDims, f);
        for (size_t d = 0; d < layerNormConfig->numNormDims; d++) {
            serialWriteSizeAsU32LE(layerNormConfig->normalizedShape[d], f);
        }
        serialWriteF32LE(layerNormConfig->eps, f);
        serializeParameter(layerNormConfig->gamma, f);
        serializeParameter(layerNormConfig->beta, f);
        serializeArithmetic(&layerNormConfig->forwardMath, f);
        serializeArithmetic(&layerNormConfig->propLossMath, f);
        serializeQuantization(layerNormConfig->outputQ, f);
        serializeQuantization(layerNormConfig->propLossQ, f);
        break;
    }
    case GROUPNORM: {
        groupNormConfig_t *groupNormConfig = layer->config->groupNorm;
        serialWriteSizeAsU32LE(groupNormConfig->numGroups, f);
        serialWriteSizeAsU32LE(groupNormConfig->numChannels, f);
        serialWriteF32LE(groupNormConfig->eps, f);
        serializeParameter(groupNormConfig->gamma, f);
        serializeParameter(groupNormConfig->beta, f);
        serializeArithmetic(&groupNormConfig->forwardMath, f);
        serializeArithmetic(&groupNormConfig->propLossMath, f);
        serializeQuantization(groupNormConfig->outputQ, f);
        serializeQuantization(groupNormConfig->propLossQ, f);
        break;
    }
    default:
        PRINT_ERROR("Unsupported layer type!\n");
        exit(1);
    }
}
