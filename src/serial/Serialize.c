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
#include "Serialize.h"
#include "SerializeInternal.h"
#include "Softmax.h"
#include "Tensor.h"

/* Locked format v1 (docs/superpowers/plans/2026-07-02-arithmetic-type-split.md,
 * Task 11): magic "ODTS" + u32 version + u32 layerCount, then one Record per
 * layer (u8 tag + payload). Task 12 adds more record types; these primitives
 * do not change. */
#define SERIALIZE_MAGIC "ODTS"
#define SERIALIZE_FORMAT_VERSION 1u

void serializeTensor(tensor_t *tensor, FILE *f) {
    size_t numberOfValues = calcNumberOfElementsByTensor(tensor);
    /* Data payload = packed size (calcNumberOfBytesForData) -- byte-identical to
     * N * elementSize for byte-aligned dtypes, packed-tight for sub-byte (#172). */
    size_t dataBytes = calcNumberOfBytesForData(tensor->quantization, numberOfValues);

    serializeShape(tensor->shape, f);
    serializeQuantization(tensor->quantization, f);
    serializeData(tensor->data, dataBytes, 1, f);
    serializeSparsity();
}

void serializeParameter(parameter_t *parameter, FILE *f) {
    serializeTensor(parameter->param, f);
    serializeTensor(parameter->grad, f);
}

void serializeModel(layer_t **model, size_t sizeModel, FILE *f) {
    serialize((void *)SERIALIZE_MAGIC, 4, sizeof(char), f);

    uint32_t version = SERIALIZE_FORMAT_VERSION;
    serialize(&version, 1, sizeof(uint32_t), f);

    uint32_t layerCount = (uint32_t)sizeModel;
    serialize(&layerCount, 1, sizeof(uint32_t), f);

    for (size_t i = 0; i < sizeModel; i++) {
        uint8_t tag = (uint8_t)model[i]->type;
        serialize(&tag, 1, sizeof(uint8_t), f);
        serializeLayer(model[i], f);
    }
}

// Helper Functions

static void serialize(void *values, size_t numberOfElements, size_t sizeOfElement, FILE *f) {
    fwrite(values, numberOfElements, sizeOfElement, f);
}

static void serializeShape(shape_t *shape, FILE *f) {
    serialize(&shape->numberOfDimensions, 1, sizeof(size_t), f);
    serialize(shape->dimensions, shape->numberOfDimensions, sizeof(size_t), f);
    serialize(shape->orderOfDimensions, shape->numberOfDimensions, sizeof(size_t), f);
}

static void serializeQuantization(quantization_t *q, FILE *f) {
    uint8_t type = (uint8_t)q->type;
    serialize(&type, 1, sizeof(uint8_t), f);
    serializeQConfig(q, f);
}

static void serializeArithmetic(arithmetic_t *arithmetic, FILE *f) {
    uint8_t type = (uint8_t)arithmetic->type;
    uint8_t roundingMode = (uint8_t)arithmetic->roundingMode;
    serialize(&type, 1, sizeof(uint8_t), f);
    serialize(&roundingMode, 1, sizeof(uint8_t), f);
}

static void serializeData(uint8_t *data, size_t numberOfValues, size_t bytesPerValue, FILE *f) {
    serialize(data, numberOfValues, bytesPerValue, f);
}

static void serializeKernel(kernel_t *kernel, FILE *f) {
    serialize(&kernel->size, 1, sizeof(size_t), f);
    uint8_t paddingType = (uint8_t)kernel->paddingType;
    serialize(&paddingType, 1, sizeof(uint8_t), f);
    serialize(&kernel->stride, 1, sizeof(size_t), f);
    serialize(&kernel->dilation, 1, sizeof(size_t), f);
    serialize(&kernel->padding, 1, sizeof(size_t), f);
}

static void serializeQConfig(quantization_t *q, FILE *f) {
    switch (q->type) {
    case INT32:
    case FLOAT32:
    case BOOL:
        break;
    case SYM_INT32:
        symInt32QConfig_t *symIntQC = q->qConfig;
        serialize(&symIntQC->scale, 1, sizeof(float), f);
        serialize(&symIntQC->roundingMode, 1, sizeof(roundingMode_t), f);
        serialize(&symIntQC->qMaxBits, 1, sizeof(uint8_t), f);
        break;
    case SYM:
        symQConfig_t *symQC = q->qConfig;
        serialize(&symQC->scale, 1, sizeof(float), f);
        serialize(&symQC->qBits, 1, sizeof(uint8_t), f);
        serialize(&symQC->roundingMode, 1, sizeof(roundingMode_t), f);
        break;
    case ASYM:
        asymQConfig_t *asymQC = q->qConfig;
        serialize(&asymQC->scale, 1, sizeof(float), f);
        serialize(&asymQC->qBits, 1, sizeof(uint8_t), f);
        serialize(&asymQC->roundingMode, 1, sizeof(roundingMode_t), f);
        serialize(&asymQC->zeroPoint, 1, sizeof(int16_t), f);
        break;
    default:
        PRINT_ERROR("Unknown qType!");
        exit(1);
    }
}

// TODO
static void serializeSparsity() {}

static void serializeLayer(layer_t *layer, FILE *f) {
    switch (layer->type) {
    case LINEAR:
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
    case RELU:
        reluConfig_t *reluConfig = layer->config->relu;
        serializeArithmetic(&reluConfig->forwardMath, f);
        serializeArithmetic(&reluConfig->propLossMath, f);
        serializeQuantization(reluConfig->outputQ, f);
        serializeQuantization(reluConfig->propLossQ, f);
        break;
    case CONV1D:
        conv1dConfig_t *conv1dConfig = layer->config->conv1d;
        serializeKernel(conv1dConfig->kernel, f);
        uint32_t conv1dGroups = (uint32_t)conv1dConfig->groups;
        serialize(&conv1dGroups, 1, sizeof(uint32_t), f);
        serializeParameter(conv1dConfig->weights, f);
        uint8_t conv1dHasBias = conv1dConfig->bias != NULL ? 1 : 0;
        serialize(&conv1dHasBias, 1, sizeof(uint8_t), f);
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
    case CONV1D_TRANSPOSED:
        conv1dTransposedConfig_t *conv1dTransposedConfig = layer->config->conv1dTransposed;
        serializeKernel(conv1dTransposedConfig->kernel, f);
        uint32_t conv1dTransposedGroups = (uint32_t)conv1dTransposedConfig->groups;
        serialize(&conv1dTransposedGroups, 1, sizeof(uint32_t), f);
        uint32_t conv1dTransposedOutputPadding = (uint32_t)conv1dTransposedConfig->outputPadding;
        serialize(&conv1dTransposedOutputPadding, 1, sizeof(uint32_t), f);
        serializeParameter(conv1dTransposedConfig->weights, f);
        uint8_t conv1dTransposedHasBias = conv1dTransposedConfig->bias != NULL ? 1 : 0;
        serialize(&conv1dTransposedHasBias, 1, sizeof(uint8_t), f);
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
    case MAXPOOL1D:
        maxPool1dConfig_t *maxPool1dConfig = layer->config->maxPool1d;
        serializeKernel(maxPool1dConfig->kernel, f);
        serializeArithmetic(&maxPool1dConfig->forwardMath, f);
        serializeArithmetic(&maxPool1dConfig->propLossMath, f);
        serializeQuantization(maxPool1dConfig->outputQ, f);
        serializeQuantization(maxPool1dConfig->propLossQ, f);
        break;
    case AVGPOOL1D:
        avgPool1dConfig_t *avgPool1dConfig = layer->config->avgPool1d;
        serializeKernel(avgPool1dConfig->kernel, f);
        serializeArithmetic(&avgPool1dConfig->forwardMath, f);
        serializeArithmetic(&avgPool1dConfig->propLossMath, f);
        serializeQuantization(avgPool1dConfig->outputQ, f);
        serializeQuantization(avgPool1dConfig->propLossQ, f);
        break;
    case SOFTMAX:
        softmaxConfig_t *softmaxConfig = layer->config->softmax;
        serializeArithmetic(&softmaxConfig->forwardMath, f);
        serializeArithmetic(&softmaxConfig->propLossMath, f);
        serializeQuantization(softmaxConfig->outputQ, f);
        serializeQuantization(softmaxConfig->propLossQ, f);
        break;
    case FLATTEN:
        // Flatten carries no state (no parameters, no quantization).
        break;
    case QUANTIZATION:
        quantizationConfig_t *quantizationConfig = layer->config->quantization;
        serializeQuantization(quantizationConfig->outputQ, f);
        serializeQuantization(quantizationConfig->propLossQ, f);
        break;
    case ADAPTIVE_AVGPOOL1D:
        adaptiveAvgPool1dConfig_t *adaptiveAvgPool1dConfig = layer->config->adaptiveAvgPool1d;
        uint32_t adaptiveAvgPool1dOutputSize = (uint32_t)adaptiveAvgPool1dConfig->outputSize;
        serialize(&adaptiveAvgPool1dOutputSize, 1, sizeof(uint32_t), f);
        serializeArithmetic(&adaptiveAvgPool1dConfig->forwardMath, f);
        serializeArithmetic(&adaptiveAvgPool1dConfig->propLossMath, f);
        serializeQuantization(adaptiveAvgPool1dConfig->outputQ, f);
        serializeQuantization(adaptiveAvgPool1dConfig->propLossQ, f);
        break;
    case DROPOUT:
        dropoutConfig_t *dropoutConfig = layer->config->dropout;
        serialize(&dropoutConfig->p, 1, sizeof(float), f);
        serializeArithmetic(&dropoutConfig->forwardMath, f);
        serializeArithmetic(&dropoutConfig->propLossMath, f);
        serializeQuantization(dropoutConfig->outputQ, f);
        serializeQuantization(dropoutConfig->propLossQ, f);
        break;
    case LAYERNORM:
        layerNormConfig_t *layerNormConfig = layer->config->layerNorm;
        uint32_t layerNormNumNormDims = (uint32_t)layerNormConfig->numNormDims;
        serialize(&layerNormNumNormDims, 1, sizeof(uint32_t), f);
        for (size_t d = 0; d < layerNormConfig->numNormDims; d++) {
            serialize(&layerNormConfig->normalizedShape[d], 1, sizeof(size_t), f);
        }
        serialize(&layerNormConfig->eps, 1, sizeof(float), f);
        serializeParameter(layerNormConfig->gamma, f);
        serializeParameter(layerNormConfig->beta, f);
        serializeArithmetic(&layerNormConfig->forwardMath, f);
        serializeArithmetic(&layerNormConfig->propLossMath, f);
        serializeQuantization(layerNormConfig->outputQ, f);
        serializeQuantization(layerNormConfig->propLossQ, f);
        break;
    case GROUPNORM:
        groupNormConfig_t *groupNormConfig = layer->config->groupNorm;
        uint32_t groupNormNumGroups = (uint32_t)groupNormConfig->numGroups;
        serialize(&groupNormNumGroups, 1, sizeof(uint32_t), f);
        uint32_t groupNormNumChannels = (uint32_t)groupNormConfig->numChannels;
        serialize(&groupNormNumChannels, 1, sizeof(uint32_t), f);
        serialize(&groupNormConfig->eps, 1, sizeof(float), f);
        serializeParameter(groupNormConfig->gamma, f);
        serializeParameter(groupNormConfig->beta, f);
        serializeArithmetic(&groupNormConfig->forwardMath, f);
        serializeArithmetic(&groupNormConfig->propLossMath, f);
        serializeQuantization(groupNormConfig->outputQ, f);
        serializeQuantization(groupNormConfig->propLossQ, f);
        break;
    default:
        PRINT_ERROR("Unsupported layer type!\n");
        exit(1);
    }
}
