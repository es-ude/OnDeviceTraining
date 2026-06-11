#define SOURCE_FILE "INFERENCE_Api"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include "AdaptiveAvgPool1d.h"
#include "AvgPool1d.h"
#include "Common.h"
#include "Conv1d.h"
#include "Conv1dTransposed.h"
#include "Dropout.h"
#include "InferenceApi.h"
#include "Layer.h"
#include "LayerNorm.h"
#include "Linear.h"
#include "MaxPool1d.h"
#include "QuantizationLayer.h"
#include "Relu.h"
#include "Softmax.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "TensorConversion.h"

// Initializes buffer to match output
static void initBufferOutput(tensor_t *buffer, layer_t *currentLayer, shape_t *inputShape,
                             sparsity_t *inputSparsity, quantization_t *inputQ) {
    layerType_t currentLayerType = currentLayer->type;
    quantization_t *currentQ = NULL;

    switch (currentLayerType) {
    case LINEAR:
        currentQ = currentLayer->config->linear->forwardQ;
        break;
    case RELU:
        currentQ = currentLayer->config->relu->forwardQ;
        break;
    case SOFTMAX:
        currentQ = currentLayer->config->softmax->forwardQ;
        break;
    case FLATTEN:
        // Flatten has no per-layer quantization; output dtype equals input dtype.
        currentQ = inputQ;
        break;
    case CONV1D:
        currentQ = currentLayer->config->conv1d->forwardQ;
        break;
    case CONV1D_TRANSPOSED:
        currentQ = currentLayer->config->conv1dTransposed->forwardQ;
        break;
    case MAXPOOL1D:
        currentQ = currentLayer->config->maxPool1d->forwardQ;
        break;
    case AVGPOOL1D:
        currentQ = currentLayer->config->avgPool1d->forwardQ;
        break;
    case ADAPTIVE_AVGPOOL1D:
        currentQ = currentLayer->config->adaptiveAvgPool1d->forwardQ;
        break;
    case DROPOUT:
        currentQ = currentLayer->config->dropout->forwardQ;
        break;
    case LAYERNORM:
        currentQ = currentLayer->config->layerNorm->forwardQ;
        break;
    case QUANTIZATION:
        currentQ = currentLayer->config->quantization->forwardQ;
        break;
    default:
        PRINT_ERROR("Unknown Layer Type!");
        exit(1);
    }

    size_t sizeDims = inputShape->numberOfDimensions;

    shape_t *outShape = reserveMemory(sizeof(shape_t));
    size_t *outDims = reserveMemory(sizeDims * sizeof(size_t));
    size_t *outOrder = reserveMemory(sizeDims * sizeof(size_t));

    outShape->dimensions = outDims;
    outShape->numberOfDimensions = sizeDims;
    outShape->orderOfDimensions = outOrder;

    calcOutputShapeFn_t calcOutputShape = layerFunctions[currentLayerType].calcOutputShape;
    calcOutputShape(currentLayer, inputShape, outShape);

    size_t numValues = calcNumberOfElementsByShape(outShape);
    size_t sizeData = calcNumberOfBytesForData(currentQ, numValues);
    uint8_t *data = reserveMemory(sizeData);

    quantization_t *q = reserveMemory(sizeof(quantization_t));
    switch (currentQ->type) {
    case FLOAT32:
        initFloat32Quantization(q);
        break;
    case SYM_INT32:
        symInt32QConfig_t *currentQC = currentQ->qConfig;
        symInt32QConfig_t *symInt32QC = reserveMemory(sizeof(symInt32QConfig_t));

        initSymInt32QConfig(currentQC->roundingMode, symInt32QC);
        initSymInt32Quantization(symInt32QC, q);
        break;
    default:
        PRINT_ERROR("Unknown QType!");
        exit(1);
    }

    setTensorValues(buffer, data, outShape, q, inputSparsity);
}

// Initializes buffer to match given input
static void initBufferInput(tensor_t *input, tensor_t *buffer) {
    quantization_t *currentQ = input->quantization;

    size_t sizeDims = input->shape->numberOfDimensions;

    shape_t *outShape = reserveMemory(sizeof(shape_t));
    size_t *outDims = reserveMemory(sizeDims * sizeof(size_t));
    size_t *outOrder = reserveMemory(sizeDims * sizeof(size_t));

    outShape->dimensions = outDims;
    outShape->numberOfDimensions = sizeDims;
    outShape->orderOfDimensions = outOrder;

    size_t numValues = calcNumberOfElementsByTensor(input);
    size_t sizeData = calcNumberOfBytesForData(currentQ, numValues);
    uint8_t *data = reserveMemory(sizeData);

    quantization_t *q = reserveMemory(sizeof(quantization_t));
    switch (currentQ->type) {
    case FLOAT32:
        q->type = FLOAT32;
        q->qConfig = NULL;
        break;
    case SYM_INT32:
        q->type = SYM_INT32;
        symInt32QConfig_t *currentQC = currentQ->qConfig;
        symInt32QConfig_t *symInt32QC = reserveMemory(sizeof(symInt32QConfig_t));
        symInt32QC->roundingMode = currentQC->roundingMode;
        q->qConfig = symInt32QC;
        break;
    default:
        PRINT_ERROR("Unknown QType!");
        exit(1);
    }

    setTensorValues(buffer, data, outShape, q, input->sparsity);

    copyTensor(buffer, input);
}

static void deInitBuffer(tensor_t *buffer) {
    freeData(buffer);
    freeShape(buffer->shape);
    freeQuantization(buffer->quantization);
}

tensor_t *inference(layer_t **model, size_t numberOfLayers, tensor_t *input) {
    tensor_t outputNext;

    initBufferInput(input, &outputNext);

    for (size_t i = 0; i < numberOfLayers; i++) {
        layer_t *currentLayer = model[i];
        layerType_t currentLayerType = currentLayer->type;
        forwardFn_t forward = layerFunctions[currentLayerType].forward;

        tensor_t outputCurr;
        initBufferOutput(&outputCurr, currentLayer, outputNext.shape, outputNext.sparsity,
                         outputNext.quantization);
        forward(currentLayer, &outputNext, &outputCurr);

        deInitBuffer(&outputNext);
        outputNext = outputCurr;
    }

    tensor_t *output = getTensorLike(&outputNext);
    convertTensor(&outputNext, output);
    deInitBuffer(&outputNext);
    return output;
}

tensor_t **inferenceBatched(layer_t **model, size_t numberOfLayers, batch_t *batch) {
    tensor_t **tensorArr = reserveMemory(batch->size * sizeof(tensor_t));

    for (size_t i = 0; i < batch->size; i++) {
        tensorArr[i] = inference(model, numberOfLayers, batch->samples[i]->item);
    }

    return tensorArr;
}

inferenceStats_t *reserveInferenceStats(tensor_t *label) {
    inferenceStats_t *inferenceStats = reserveMemory(sizeof(inferenceStats_t));

    shape_t *outputShape = getShapeLike(label->shape);
    quantization_t *outputQ = getQLike(label->quantization);
    inferenceStats->output = initTensor(outputShape, outputQ, NULL);

    return inferenceStats;
}

void freeInferenceStats(inferenceStats_t *inferenceStats) {
    freeTensor(inferenceStats->output);
    freeReservedMemory(inferenceStats);
}

inferenceStats_t *inferenceWithLoss(layer_t **model, size_t numberOfLayers, tensor_t *input,
                                    tensor_t *label, lossFuncType_t funcType,
                                    reduction_t forwardReduction) {
    tensor_t outputNext;
    initBufferInput(input, &outputNext);

    for (size_t i = 0; i < numberOfLayers; i++) {
        layer_t *currentLayer = model[i];
        layerType_t currentLayerType = currentLayer->type;
        forwardFn_t forward = layerFunctions[currentLayerType].forward;

        tensor_t outputCurr;
        initBufferOutput(&outputCurr, currentLayer, outputNext.shape, outputNext.sparsity,
                         outputNext.quantization);
        forward(currentLayer, &outputNext, &outputCurr);
        deInitBuffer(&outputNext);
        outputNext = outputCurr;
    }

    inferenceStats_t *inferenceStats = reserveInferenceStats(label);
    copyTensor(inferenceStats->output, &outputNext);

    lossFunctions_t lossFns = lossFunctions[funcType];
    float loss = lossFns.forward(&outputNext, label, forwardReduction);
    inferenceStats->loss = loss;

    deInitBuffer(&outputNext);
    return inferenceStats;
}
