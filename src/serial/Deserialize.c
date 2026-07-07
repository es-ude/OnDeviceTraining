#define SOURCE_FILE "DESERIALIZE"

#include <stdlib.h>
#include <string.h>

#include "AdaptiveAvgPool1d.h"
#include "ArithmeticType.h"
#include "AvgPool1d.h"
#include "Common.h"
#include "Conv1d.h"
#include "Conv1dTransposed.h"
#include "Deserialize.h"
#include "DeserializeInternal.h"
#include "Dropout.h"
#include "GroupNorm.h"
#include "Kernel.h"
#include "LayerNorm.h"
#include "Linear.h"
#include "MaxPool1d.h"
#include "QuantizationLayer.h"
#include "Relu.h"
#include "Rounding.h"
#include "Softmax.h"
#include "Tensor.h"

/* Mirrors Serialize.c's locked format v1 constants. */
#define SERIALIZE_MAGIC "ODTS"
#define SERIALIZE_FORMAT_VERSION 1u

void deserializeTensor(tensor_t *tensor, FILE *f) {
    deserializeShape(tensor->shape, f);
    deserializeQuantization(tensor->quantization, f);

    size_t numberOfValues = calcNumberOfElementsByShape(tensor->shape);
    /* Mirrors Serialize.c: payload length is the packed size. */
    size_t dataBytes = calcNumberOfBytesForData(tensor->quantization, numberOfValues);

    deserializeData(tensor->data, dataBytes, 1, f);
    deserializeSparsity();
}

void deserializeParameter(parameter_t *parameter, FILE *f) {
    deserializeTensor(parameter->param, f);
    deserializeTensor(parameter->grad, f);
}

void deserializeModel(layer_t **model, size_t sizeModel, FILE *f) {
    char magic[4];
    deserialize(magic, 4, sizeof(char), f);
    if (memcmp(magic, SERIALIZE_MAGIC, 4) != 0) {
        PRINT_ERROR("deserializeModel: bad magic bytes (expected \"ODTS\")");
        exit(1);
    }

    uint32_t version;
    deserialize(&version, 1, sizeof(uint32_t), f);
    if (version != SERIALIZE_FORMAT_VERSION) {
        PRINT_ERROR("deserializeModel: unsupported format version %u (expected %u)",
                    (unsigned)version, (unsigned)SERIALIZE_FORMAT_VERSION);
        exit(1);
    }

    uint32_t layerCount;
    deserialize(&layerCount, 1, sizeof(uint32_t), f);
    if (layerCount != (uint32_t)sizeModel) {
        PRINT_ERROR("deserializeModel: layerCount mismatch (file has %u, caller expects %zu)",
                    (unsigned)layerCount, sizeModel);
        exit(1);
    }

    for (size_t i = 0; i < sizeModel; i++) {
        uint8_t tag;
        deserialize(&tag, 1, sizeof(uint8_t), f);
        if (tag != (uint8_t)model[i]->type) {
            PRINT_ERROR("deserializeModel: record tag %u does not match expected layer type "
                        "%u at index %zu",
                        (unsigned)tag, (unsigned)model[i]->type, i);
            exit(1);
        }
        deserializeLayer(model[i], f);
    }
}

// Helper Functions

static void deserialize(void *values, size_t numberOfElements, size_t sizeOfElement, FILE *f) {
    fread(values, numberOfElements, sizeOfElement, f);
}

static void deserializeShape(shape_t *shape, FILE *f) {
    deserialize(&shape->numberOfDimensions, 1, sizeof(size_t), f);
    deserialize(shape->dimensions, shape->numberOfDimensions, sizeof(size_t), f);
    deserialize(shape->orderOfDimensions, shape->numberOfDimensions, sizeof(size_t), f);
}

static void deserializeQuantization(quantization_t *q, FILE *f) {
    uint8_t type;
    deserialize(&type, 1, sizeof(uint8_t), f);
    q->type = (qtype_t)type;
    deserializeQConfig(q, f);
}

static void deserializeArithmetic(arithmetic_t *arithmetic, FILE *f) {
    uint8_t type;
    uint8_t roundingMode;
    deserialize(&type, 1, sizeof(uint8_t), f);
    deserialize(&roundingMode, 1, sizeof(uint8_t), f);
    arithmetic->type = (arithmeticType_t)type;
    arithmetic->roundingMode = (roundingMode_t)roundingMode;
}

static void deserializeData(uint8_t *data, size_t numberOfValues, size_t bytesPerValue, FILE *f) {
    deserialize(data, numberOfValues, bytesPerValue, f);
}

static void deserializeKernel(kernel_t *kernel, FILE *f) {
    deserialize(&kernel->size, 1, sizeof(size_t), f);
    uint8_t paddingType;
    deserialize(&paddingType, 1, sizeof(uint8_t), f);
    kernel->paddingType = (paddingType_t)paddingType;
    deserialize(&kernel->stride, 1, sizeof(size_t), f);
    deserialize(&kernel->dilation, 1, sizeof(size_t), f);
    deserialize(&kernel->padding, 1, sizeof(size_t), f);
}

static void deserializeQConfig(quantization_t *q, FILE *f) {
    switch (q->type) {
    case INT32:
    case FLOAT32:
    case BOOL:
        break;
    case SYM_INT32:
        symInt32QConfig_t *symIntQC = q->qConfig;
        deserialize(&symIntQC->scale, 1, sizeof(float), f);
        deserialize(&symIntQC->roundingMode, 1, sizeof(roundingMode_t), f);
        deserialize(&symIntQC->qMaxBits, 1, sizeof(uint8_t), f);
        break;
    case SYM:
        symQConfig_t *symQC = q->qConfig;
        deserialize(&symQC->scale, 1, sizeof(float), f);
        deserialize(&symQC->qBits, 1, sizeof(uint8_t), f);
        deserialize(&symQC->roundingMode, 1, sizeof(roundingMode_t), f);
        break;
    case ASYM:
        asymQConfig_t *asymQC = q->qConfig;
        deserialize(&asymQC->scale, 1, sizeof(float), f);
        deserialize(&asymQC->qBits, 1, sizeof(uint8_t), f);
        deserialize(&asymQC->roundingMode, 1, sizeof(roundingMode_t), f);
        deserialize(&asymQC->zeroPoint, 1, sizeof(int16_t), f);
        break;
    default:
        PRINT_ERROR("Unknown qType!");
        exit(1);
    }
}

// TODO
static void deserializeSparsity() {}

static void deserializeLayer(layer_t *layer, FILE *f) {
    switch (layer->type) {
    case LINEAR:
        linearConfig_t *linearConfig = layer->config->linear;
        deserializeParameter(linearConfig->weights, f);
        deserializeParameter(linearConfig->bias, f);
        deserializeArithmetic(&linearConfig->forwardMath, f);
        deserializeArithmetic(&linearConfig->weightGradMath, f);
        deserializeArithmetic(&linearConfig->biasGradMath, f);
        deserializeArithmetic(&linearConfig->propLossMath, f);
        deserializeQuantization(linearConfig->outputQ, f);
        deserializeQuantization(linearConfig->propLossQ, f);
        break;
    case RELU:
        reluConfig_t *reluConfig = layer->config->relu;
        deserializeArithmetic(&reluConfig->forwardMath, f);
        deserializeArithmetic(&reluConfig->propLossMath, f);
        deserializeQuantization(reluConfig->outputQ, f);
        deserializeQuantization(reluConfig->propLossQ, f);
        break;
    case CONV1D:
        conv1dConfig_t *conv1dConfig = layer->config->conv1d;
        deserializeKernel(conv1dConfig->kernel, f);
        uint32_t conv1dGroups;
        deserialize(&conv1dGroups, 1, sizeof(uint32_t), f);
        conv1dConfig->groups = (size_t)conv1dGroups;
        deserializeParameter(conv1dConfig->weights, f);
        uint8_t conv1dHasBias;
        deserialize(&conv1dHasBias, 1, sizeof(uint8_t), f);
        if (conv1dHasBias) {
            deserializeParameter(conv1dConfig->bias, f);
        }
        deserializeArithmetic(&conv1dConfig->forwardMath, f);
        deserializeArithmetic(&conv1dConfig->weightGradMath, f);
        deserializeArithmetic(&conv1dConfig->biasGradMath, f);
        deserializeArithmetic(&conv1dConfig->propLossMath, f);
        deserializeQuantization(conv1dConfig->outputQ, f);
        deserializeQuantization(conv1dConfig->propLossQ, f);
        break;
    case CONV1D_TRANSPOSED:
        conv1dTransposedConfig_t *conv1dTransposedConfig = layer->config->conv1dTransposed;
        deserializeKernel(conv1dTransposedConfig->kernel, f);
        uint32_t conv1dTransposedGroups;
        deserialize(&conv1dTransposedGroups, 1, sizeof(uint32_t), f);
        conv1dTransposedConfig->groups = (size_t)conv1dTransposedGroups;
        uint32_t conv1dTransposedOutputPadding;
        deserialize(&conv1dTransposedOutputPadding, 1, sizeof(uint32_t), f);
        conv1dTransposedConfig->outputPadding = (size_t)conv1dTransposedOutputPadding;
        deserializeParameter(conv1dTransposedConfig->weights, f);
        uint8_t conv1dTransposedHasBias;
        deserialize(&conv1dTransposedHasBias, 1, sizeof(uint8_t), f);
        if (conv1dTransposedHasBias) {
            deserializeParameter(conv1dTransposedConfig->bias, f);
        }
        deserializeArithmetic(&conv1dTransposedConfig->forwardMath, f);
        deserializeArithmetic(&conv1dTransposedConfig->weightGradMath, f);
        deserializeArithmetic(&conv1dTransposedConfig->biasGradMath, f);
        deserializeArithmetic(&conv1dTransposedConfig->propLossMath, f);
        deserializeQuantization(conv1dTransposedConfig->outputQ, f);
        deserializeQuantization(conv1dTransposedConfig->propLossQ, f);
        break;
    case MAXPOOL1D:
        maxPool1dConfig_t *maxPool1dConfig = layer->config->maxPool1d;
        deserializeKernel(maxPool1dConfig->kernel, f);
        deserializeArithmetic(&maxPool1dConfig->forwardMath, f);
        deserializeArithmetic(&maxPool1dConfig->propLossMath, f);
        deserializeQuantization(maxPool1dConfig->outputQ, f);
        deserializeQuantization(maxPool1dConfig->propLossQ, f);
        break;
    case AVGPOOL1D:
        avgPool1dConfig_t *avgPool1dConfig = layer->config->avgPool1d;
        deserializeKernel(avgPool1dConfig->kernel, f);
        deserializeArithmetic(&avgPool1dConfig->forwardMath, f);
        deserializeArithmetic(&avgPool1dConfig->propLossMath, f);
        deserializeQuantization(avgPool1dConfig->outputQ, f);
        deserializeQuantization(avgPool1dConfig->propLossQ, f);
        break;
    case SOFTMAX:
        softmaxConfig_t *softmaxConfig = layer->config->softmax;
        deserializeArithmetic(&softmaxConfig->forwardMath, f);
        deserializeArithmetic(&softmaxConfig->propLossMath, f);
        deserializeQuantization(softmaxConfig->outputQ, f);
        deserializeQuantization(softmaxConfig->propLossQ, f);
        break;
    case FLATTEN:
        // Flatten carries no state (no parameters, no quantization).
        break;
    case QUANTIZATION:
        quantizationConfig_t *quantizationConfig = layer->config->quantization;
        deserializeQuantization(quantizationConfig->outputQ, f);
        deserializeQuantization(quantizationConfig->propLossQ, f);
        break;
    case ADAPTIVE_AVGPOOL1D:
        adaptiveAvgPool1dConfig_t *adaptiveAvgPool1dConfig = layer->config->adaptiveAvgPool1d;
        uint32_t adaptiveAvgPool1dOutputSize;
        deserialize(&adaptiveAvgPool1dOutputSize, 1, sizeof(uint32_t), f);
        adaptiveAvgPool1dConfig->outputSize = (size_t)adaptiveAvgPool1dOutputSize;
        deserializeArithmetic(&adaptiveAvgPool1dConfig->forwardMath, f);
        deserializeArithmetic(&adaptiveAvgPool1dConfig->propLossMath, f);
        deserializeQuantization(adaptiveAvgPool1dConfig->outputQ, f);
        deserializeQuantization(adaptiveAvgPool1dConfig->propLossQ, f);
        break;
    case DROPOUT:
        dropoutConfig_t *dropoutConfig = layer->config->dropout;
        deserialize(&dropoutConfig->p, 1, sizeof(float), f);
        deserializeArithmetic(&dropoutConfig->forwardMath, f);
        deserializeArithmetic(&dropoutConfig->propLossMath, f);
        deserializeQuantization(dropoutConfig->outputQ, f);
        deserializeQuantization(dropoutConfig->propLossQ, f);
        break;
    case LAYERNORM:
        layerNormConfig_t *layerNormConfig = layer->config->layerNorm;
        uint32_t layerNormNumNormDims;
        deserialize(&layerNormNumNormDims, 1, sizeof(uint32_t), f);
        layerNormConfig->numNormDims = (size_t)layerNormNumNormDims;
        for (size_t d = 0; d < layerNormConfig->numNormDims; d++) {
            deserialize(&layerNormConfig->normalizedShape[d], 1, sizeof(size_t), f);
        }
        deserialize(&layerNormConfig->eps, 1, sizeof(float), f);
        deserializeParameter(layerNormConfig->gamma, f);
        deserializeParameter(layerNormConfig->beta, f);
        deserializeArithmetic(&layerNormConfig->forwardMath, f);
        deserializeArithmetic(&layerNormConfig->propLossMath, f);
        deserializeQuantization(layerNormConfig->outputQ, f);
        deserializeQuantization(layerNormConfig->propLossQ, f);
        break;
    case GROUPNORM:
        groupNormConfig_t *groupNormConfig = layer->config->groupNorm;
        uint32_t groupNormNumGroups;
        deserialize(&groupNormNumGroups, 1, sizeof(uint32_t), f);
        groupNormConfig->numGroups = (size_t)groupNormNumGroups;
        uint32_t groupNormNumChannels;
        deserialize(&groupNormNumChannels, 1, sizeof(uint32_t), f);
        groupNormConfig->numChannels = (size_t)groupNormNumChannels;
        deserialize(&groupNormConfig->eps, 1, sizeof(float), f);
        deserializeParameter(groupNormConfig->gamma, f);
        deserializeParameter(groupNormConfig->beta, f);
        deserializeArithmetic(&groupNormConfig->forwardMath, f);
        deserializeArithmetic(&groupNormConfig->propLossMath, f);
        deserializeQuantization(groupNormConfig->outputQ, f);
        deserializeQuantization(groupNormConfig->propLossQ, f);
        break;
    default:
        PRINT_ERROR("Unsupported layer type!\n");
        exit(1);
    }
}
