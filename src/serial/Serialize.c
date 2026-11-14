#define SOURCE_FILE "SERIALIZE"

#include "Serialize.h"
#include "AdaptiveAvgPool1d.h"
#include "AddLayer.h"
#include "ArithmeticType.h"
#include "AvgPool1d.h"
#include "BatchNorm1d.h"
#include "Common.h"
#include "Conv1d.h"
#include "Conv1dTransposed.h"
#include "Dropout.h"
#include "DyadicScale.h"
#include "GroupNorm.h"
#include "Kernel.h"
#include "Layer.h"
#include "LayerNorm.h"
#include "Linear.h"
#include "MaxPool1d.h"
#include "Model.h"
#include "QuantizationLayer.h"
#include "Relu.h"
#include "Rounding.h"
#include "SerialWire.h"
#include "SerializeInternal.h"
#include "Softmax.h"
#include "Tensor.h"
#include "TensorRecord.h"

/* ODTS layout, version SERIALIZE_FORMAT_VERSION (TensorRecord.h):
 *   file:         "ODTS", u32 version, u32 layerCount, u8 wiringKind, then per
 *                 layer a u8 tag (layerType_t position, append-only) and that
 *                 layer's record
 *   wiringKind:   0 = chain (node j reads node j - 1; every model modelIsChain
 *                 accepts), 1 = graph, followed by the model's CSR:
 *                 u16 inputStart[layerCount + 1], u16 inputIds[inputStart[layerCount]];
 *                 the reader requires kind and CSR to equal the skeleton's
 *   tensor:       u32 rank, u32 dimensions[rank], u32 orderOfDimensions[rank],
 *                 quantization, packed payload (calcNumberOfBytesForData bytes)
 *   parameter:    u8 hasGrad, param tensor, grad tensor iff hasGrad
 *   scale:        u16 m, i16 e (value m * 2^e, DyadicScale.h); a non-canonical
 *                 in-memory scale fails fast at the writer
 *   quantization: u8 dtype, then by dtype
 *     SYM_INT32   scale, u8 rounding, u8 qMaxBits
 *     SYM         u32 numGroups, u32 groupSize, scale scales[numGroups], u8 qBits, u8 rounding
 *     ASYM        u32 numGroups, u32 groupSize, scale scales[numGroups],
 *                 u16 zeroPoints[numGroups], u8 qBits, u8 rounding
 *     BFP         u32 numGroups, u32 groupSize, u8 exponents[numGroups],
 *                 u8 mantissaBits, u8 exponentBits, u8 rounding
 *   arithmetic:   u8 type, u8 rounding
 * Layer records follow serializeLayer's field order; a layer with an optional
 * bias writes u8 hasBias before it. Scalars are little-endian through the
 * checked SerialWire primitives, so a file written on a 64-bit host loads
 * bit-identically on a 32-bit MCU. */
#define SERIALIZE_MAGIC "ODTS"

void serializeTensor(tensor_t *tensor, FILE *f) {
    size_t numberOfValues = calcNumberOfElementsByTensor(tensor);
    size_t dataBytes = calcNumberOfBytesForData(tensor->quantization, numberOfValues);

    serializeShape(tensor->shape, f);
    serializeQuantization(tensor->quantization, f);
    serialWriteBytes(tensor->data, dataBytes, f);
}

void serializeParameter(parameter_t *parameter, FILE *f) {
    /* Frozen layers carry no grad tensor. */
    uint8_t hasGrad = parameter->grad != NULL ? 1u : 0u;
    serialWriteU8(hasGrad, f);
    serializeTensor(parameter->param, f);
    if (hasGrad) {
        serializeTensor(parameter->grad, f);
    }
}

void serializeModel(const model_t *m, FILE *f) {
    modelRequireValid(m, "serializeModel");
    serialWriteBytes(SERIALIZE_MAGIC, 4, f);
    serialWriteU32LE(SERIALIZE_FORMAT_VERSION, f);
    serialWriteSizeAsU32LE(m->numLayers, f);
    if (modelIsChain(m)) {
        serialWriteU8(0u, f);
    } else {
        serialWriteU8(1u, f);
        for (size_t j = 0; j <= m->numLayers; j++) {
            serialWriteU16LE(m->inputStart[j], f);
        }
        for (size_t e = 0; e < m->inputStart[m->numLayers]; e++) {
            serialWriteU16LE(m->inputIds[e], f);
        }
    }

    for (size_t i = 0; i < m->numLayers; i++) {
        serialWriteU8((uint8_t)m->layers[i]->type, f);
        serializeLayer(m->layers[i], f);
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

static void serializeScale(odtScale_t s, FILE *f) {
    if (s.m < ODT_SCALE_M_MIN || s.m > ODT_SCALE_M_MAX || s.e < ODT_SCALE_E_MIN ||
        s.e > ODT_SCALE_E_MAX) {
        PRINT_ERROR("serializeQConfig: scale {m=%u, e=%d} is not canonical", (unsigned)s.m,
                    (int)s.e);
        exit(1);
    }
    serialWriteU16LE(s.m, f);
    serialWriteI16LE(s.e, f);
}

static void serializeQConfig(quantization_t *q, FILE *f) {
    switch (q->type) {
    case INT32:
    case FLOAT32:
    case BOOL:
        break;
    case SYM_INT32: {
        symInt32QConfig_t *symIntQC = q->qConfig;
        serializeScale(odtScaleLoad(&symIntQC->scale), f);
        serialWriteU8((uint8_t)symIntQC->roundingMode, f);
        serialWriteU8(symIntQC->qMaxBits, f);
        break;
    }
    case SYM: {
        symQConfig_t *symQC = q->qConfig;
        serialWriteSizeAsU32LE(symQC->numGroups, f);
        serialWriteSizeAsU32LE(symQC->groupSize, f);
        for (size_t g = 0; g < symQC->numGroups; g++) {
            serializeScale(odtScaleLoad(&symQC->scales[g]), f);
        }
        serialWriteU8(symQC->qBits, f);
        serialWriteU8((uint8_t)symQC->roundingMode, f);
        break;
    }
    case ASYM: {
        asymQConfig_t *asymQC = q->qConfig;
        serialWriteSizeAsU32LE(asymQC->numGroups, f);
        serialWriteSizeAsU32LE(asymQC->groupSize, f);
        for (size_t g = 0; g < asymQC->numGroups; g++) {
            serializeScale(odtScaleLoad(&asymQC->scales[g]), f);
        }
        for (size_t g = 0; g < asymQC->numGroups; g++) {
            serialWriteU16LE(asymQC->zeroPoints[g], f);
        }
        serialWriteU8(asymQC->qBits, f);
        serialWriteU8((uint8_t)asymQC->roundingMode, f);
        break;
    }
    case BFP: {
        bfpQConfig_t *bfpQC = q->qConfig;
        serialWriteSizeAsU32LE(bfpQC->numGroups, f);
        serialWriteSizeAsU32LE(bfpQC->groupSize, f);
        serialWriteBytes(bfpQC->exponents, bfpQC->numGroups, f);
        serialWriteU8(bfpQC->mantissaBits, f);
        serialWriteU8(bfpQC->exponentBits, f);
        serialWriteU8((uint8_t)bfpQC->roundingMode, f);
        break;
    }
    }
}

/* BIAS_FALSE layers carry bias == NULL: presence byte first, bias record
 * only when present. */
static void serializeOptionalBias(parameter_t *bias, FILE *f) {
    uint8_t hasBias = bias != NULL ? 1u : 0u;
    serialWriteU8(hasBias, f);
    if (hasBias) {
        serializeParameter(bias, f);
    }
}

/* The record tail most layers share, in wire order. */
static void serializeLayerTail(arithmetic_t *forwardMath, arithmetic_t *propLossMath,
                               quantization_t *outputQ, quantization_t *propLossQ, FILE *f) {
    serializeArithmetic(forwardMath, f);
    serializeArithmetic(propLossMath, f);
    serializeQuantization(outputQ, f);
    serializeQuantization(propLossQ, f);
}

static void serializeLayer(layer_t *layer, FILE *f) {
    switch (layer->type) {
    case LINEAR: {
        linearConfig_t *linearConfig = layer->config->linear;
        serializeParameter(linearConfig->weights, f);
        serializeOptionalBias(linearConfig->bias, f);
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
        serializeLayerTail(&reluConfig->forwardMath, &reluConfig->propLossMath, reluConfig->outputQ,
                           reluConfig->propLossQ, f);
        break;
    }
    case ADD: {
        addConfig_t *addConfig = layer->config->add;
        serializeArithmetic(&addConfig->forwardMath, f);
        serializeArithmetic(&addConfig->propLossMath, f);
        serializeQuantization(addConfig->outputQ, f);
        serializeQuantization(addConfig->propLossQ, f);
        break;
    }
    case CONV1D: {
        conv1dConfig_t *conv1dConfig = layer->config->conv1d;
        serializeKernel(conv1dConfig->kernel, f);
        serialWriteSizeAsU32LE(conv1dConfig->groups, f);
        serializeParameter(conv1dConfig->weights, f);
        serializeOptionalBias(conv1dConfig->bias, f);
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
        serializeOptionalBias(conv1dTransposedConfig->bias, f);
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
        serializeLayerTail(&maxPool1dConfig->forwardMath, &maxPool1dConfig->propLossMath,
                           maxPool1dConfig->outputQ, maxPool1dConfig->propLossQ, f);
        break;
    }
    case AVGPOOL1D: {
        avgPool1dConfig_t *avgPool1dConfig = layer->config->avgPool1d;
        serializeKernel(avgPool1dConfig->kernel, f);
        serializeLayerTail(&avgPool1dConfig->forwardMath, &avgPool1dConfig->propLossMath,
                           avgPool1dConfig->outputQ, avgPool1dConfig->propLossQ, f);
        break;
    }
    case SOFTMAX: {
        softmaxConfig_t *softmaxConfig = layer->config->softmax;
        serializeLayerTail(&softmaxConfig->forwardMath, &softmaxConfig->propLossMath,
                           softmaxConfig->outputQ, softmaxConfig->propLossQ, f);
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
        serializeLayerTail(&adaptiveAvgPool1dConfig->forwardMath,
                           &adaptiveAvgPool1dConfig->propLossMath, adaptiveAvgPool1dConfig->outputQ,
                           adaptiveAvgPool1dConfig->propLossQ, f);
        break;
    }
    case DROPOUT: {
        dropoutConfig_t *dropoutConfig = layer->config->dropout;
        serialWriteF32LE(dropoutConfig->p, f);
        serializeLayerTail(&dropoutConfig->forwardMath, &dropoutConfig->propLossMath,
                           dropoutConfig->outputQ, dropoutConfig->propLossQ, f);
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
        serializeLayerTail(&layerNormConfig->forwardMath, &layerNormConfig->propLossMath,
                           layerNormConfig->outputQ, layerNormConfig->propLossQ, f);
        break;
    }
    case GROUPNORM: {
        groupNormConfig_t *groupNormConfig = layer->config->groupNorm;
        serialWriteSizeAsU32LE(groupNormConfig->numGroups, f);
        serialWriteSizeAsU32LE(groupNormConfig->numChannels, f);
        serialWriteF32LE(groupNormConfig->eps, f);
        serializeParameter(groupNormConfig->gamma, f);
        serializeParameter(groupNormConfig->beta, f);
        serializeLayerTail(&groupNormConfig->forwardMath, &groupNormConfig->propLossMath,
                           groupNormConfig->outputQ, groupNormConfig->propLossQ, f);
        break;
    }
    case BATCHNORM1D: {
        batchNorm1dConfig_t *c = layer->config->batchNorm1d;
        serialWriteSizeAsU32LE(c->numChannels, f);
        serialWriteF32LE(c->eps, f);
        serialWriteU8((uint8_t)c->momentumMode, f);
        serialWriteF32LE(c->momentum, f);
        serialWriteU8(c->affine ? 1u : 0u, f);
        serialWriteU8(c->trackRunningStats ? 1u : 0u, f);
        if (c->affine) {
            serializeParameter(c->gamma, f);
            serializeParameter(c->beta, f);
        }
        if (c->trackRunningStats) {
            const float *rm = (const float *)c->runningMean->data;
            const float *rv = (const float *)c->runningVar->data;
            for (size_t ch = 0; ch < c->numChannels; ch++) {
                serialWriteF32LE(rm[ch], f);
            }
            for (size_t ch = 0; ch < c->numChannels; ch++) {
                serialWriteF32LE(rv[ch], f);
            }
            serialWriteU64LE(c->numBatchesTracked, f);
        }
        serializeLayerTail(&c->forwardMath, &c->propLossMath, c->outputQ, c->propLossQ, f);
        break;
    }
    }
}
