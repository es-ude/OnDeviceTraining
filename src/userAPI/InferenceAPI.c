#include <stdio.h>
#include <stdbool.h>

#include "Tensor.h"
#include "Layer.h"
#include "InferenceAPI.h"
#include "TensorAPI.h"

#include "StorageAPI.h"
#include "Linear.h"
#include "Relu.h"
#include "TensorConversion.h"

// Initializes ping pong buffer to match output
static void initPingPongBufferOutput(tensor_t *buffer, layer_t *currentLayer, shape_t *inputShape,
                                     sparsity_t *inputSparsity) {
    layerType_t currentLayerType = currentLayer->type;
    quantization_t *currentQ = NULL;

    switch (currentLayerType) {
    case LINEAR:
        currentQ = currentLayer->config->linear->forwardQ;
        break;
    case RELU:
        currentQ = currentLayer->config->relu->forwardQ;
        break;
    default:
        break;
    }

    size_t outputNumberOfDims = inputShape->numberOfDimensions;
    size_t sizeDims = outputNumberOfDims;

    shape_t *outShape = *reserveMemory(sizeof(shape_t));
    size_t *outDims = *reserveMemory(sizeDims * sizeof(size_t));
    size_t *outOrder = *reserveMemory(sizeDims * sizeof(size_t));

    outShape->dimensions = outDims;
    outShape->numberOfDimensions = outputNumberOfDims;
    outShape->orderOfDimensions = outOrder;

    calcOutputShapeFn_t calcOutputShape = layerFunctions[currentLayerType].calcOutputShape;
    calcOutputShape(currentLayer, inputShape, outShape);

    size_t numValues =
        calcNumberOfElementsByShape(outShape);
    size_t sizeData = calcNumberOfBytesForData(currentQ, numValues);
    uint8_t *data = *reserveMemory(sizeData);

    quantization_t *q = *reserveMemory(sizeof(quantization_t));
    switch (currentQ->type) {
    case FLOAT32:
        q->type = FLOAT32;
        q->qConfig = NULL;
        break;
    case SYM_INT32:
        q->type = SYM_INT32;
        symInt32QConfig_t *currentQC = currentQ->qConfig;
        symInt32QConfig_t *symInt32QC = *reserveMemory(sizeof(symInt32QConfig_t));
        symInt32QC->roundingMode = currentQC->roundingMode;
        q->qConfig = symInt32QC;
        break;
    default:
        break;
    }

    setTensorValues(buffer, data, outShape, q, inputSparsity);
}

// Initializes ping pong buffer to match given input
static void initPingPongBufferInput(tensor_t *input, tensor_t *buffer) {
    quantization_t *currentQ = input->quantization;

    size_t sizeDims = input->shape->numberOfDimensions;

    shape_t *outShape = *reserveMemory(sizeof(shape_t));
    size_t *outDims = *reserveMemory(sizeDims * sizeof(size_t));
    size_t *outOrder = *reserveMemory(sizeDims * sizeof(size_t));

    outShape->dimensions = outDims;
    outShape->numberOfDimensions = sizeDims;
    outShape->orderOfDimensions = outOrder;

    size_t numValues = calcNumberOfElementsByTensor(input);
    size_t sizeData = calcNumberOfBytesForData(currentQ, numValues);
    uint8_t *data = *reserveMemory(sizeData);

    quantization_t *q = *reserveMemory(sizeof(quantization_t));
    switch (currentQ->type) {
    case FLOAT32:
        q->type = FLOAT32;
        q->qConfig = NULL;
        break;
    case SYM_INT32:
        q->type = SYM_INT32;
        symInt32QConfig_t *currentQC = currentQ->qConfig;
        symInt32QConfig_t *symInt32QC = *reserveMemory(sizeof(symInt32QConfig_t));
        symInt32QC->roundingMode = currentQC->roundingMode;
        q->qConfig = symInt32QC;
        break;
    default:
        break;
    }

    setTensorValues(buffer, data, outShape, q, input->sparsity);

    copyTensor(buffer, input);
}

static void deInitPingPongBuffer(tensor_t *buffer) {
    freeData(buffer);
    freeShape(buffer->shape);
    freeQuantization(buffer->quantization);
}

tensor_t *inference(layer_t **model, size_t numberOfLayers, tensor_t *input) {
    tensor_t ping;
    tensor_t pong;
    bool toggle = false;

    initPingPongBufferInput(input, &ping);

    for (size_t i = 0; i < numberOfLayers; i++) {
        layer_t *currentLayer = model[i];
        layerType_t currentLayerType = currentLayer->type;
        forwardFn_t forward = layerFunctions[currentLayerType].forward;

        if (!toggle) {
            initPingPongBufferOutput(&pong, currentLayer, input->shape, input->sparsity);
            forward(currentLayer, &ping, &pong);
            deInitPingPongBuffer(&ping);
            toggle = true;
        } else {
            initPingPongBufferOutput(&ping, currentLayer, input->shape, input->sparsity);
            forward(currentLayer, &pong, &ping);
            deInitPingPongBuffer(&pong);
            toggle = false;
        }
    }

    if (!toggle) {
        tensor_t *output = getTensorLike(&ping);
        convertTensor(&ping, output);
        return output;
    }

    tensor_t *output = getTensorLike(&pong);
    convertTensor(&pong, output);
    return output;
}

inferenceStats_t *reserveInferenceStats(tensor_t *label) {
    inferenceStats_t *inferenceStats = *reserveMemory(sizeof(inferenceStats_t));

    size_t sizeOutput = calcNumberOfElementsByTensor(label);

    float *outputData = *reserveMemory(sizeOutput * sizeof(float));
    size_t outputNumberOfDims = label->shape->numberOfDimensions;
    size_t *outputDims = *reserveMemory(outputNumberOfDims * sizeof(size_t));
    quantization_t *outputQ = getQLike(label->quantization);
    tensor_t *output = tensorInit(outputData, outputDims, outputNumberOfDims, outputQ, NULL);

    inferenceStats->output = output;

    return inferenceStats;
}

void freeInferenceStats(inferenceStats_t *inferenceStats) {
    freeTensor(inferenceStats->output);
    freeReservedMemory(inferenceStats);
}

inferenceStats_t *inferenceWithLoss(layer_t **model, size_t numberOfLayers, tensor_t *input,
                                    tensor_t *label, lossType_t lossType) {
    for (size_t i = 0; i < numberOfLayers; i++) {
        layer_t *currentLayer = model[i];
        layerType_t currentLayerType = currentLayer->type;
        quantization_t *currentQ = NULL;

        switch (currentLayerType) {
        case LINEAR:
            currentQ = currentLayer->config->linear->forwardQ;
            break;
        case RELU:
            currentQ = currentLayer->config->relu->forwardQ;
            break;
        default:
            break;
        }

        size_t outputNumberOfDims = input->shape->numberOfDimensions;
        size_t sizeDims = outputNumberOfDims;
        size_t outDims[sizeDims];
        size_t outOrder[sizeDims];

        shape_t outShape;
        outShape.dimensions = outDims;
        outShape.numberOfDimensions = outputNumberOfDims;
        outShape.orderOfDimensions = outOrder;

        calcOutputShapeFn_t calcOutputShape = layerFunctions[currentLayerType].calcOutputShape;
        calcOutputShape(currentLayer, input->shape, &outShape);

        size_t numValues =
            calcNumberOfElementsByShape(&outShape);

        size_t sizeData = calcNumberOfBytesForData(currentQ, numValues);
        uint8_t data[sizeData];

        quantization_t q;
        symInt32QConfig_t symInt32QConfig;
        switch (currentQ->type) {
        case FLOAT32:
            q.type = FLOAT32;
            q.qConfig = NULL;
            break;
        case ASYM:
            q.type = SYM_INT32;
            symInt32QConfig_t *currentQC = currentQ->qConfig;
            symInt32QConfig.roundingMode = currentQC->roundingMode;
            q.qConfig = &symInt32QConfig;
            break;
        default:
            break;
        }

        tensor_t intermediateOutput;
        setTensorValues(&intermediateOutput, data, &outShape, &q, input->sparsity);

        forwardFn_t forward = layerFunctions[currentLayerType].forward;
        forward(currentLayer, input, &intermediateOutput);

        if (i == numberOfLayers - 1) {
            inferenceStats_t *inferenceStats = reserveInferenceStats(label);
            copyTensor(inferenceStats->output, &intermediateOutput);

            lossFunctions_t lossFns = lossFunctions[lossType];
            float loss = lossFns.forward(&intermediateOutput, label);
            inferenceStats->loss = loss;

            return inferenceStats;
        }

        copyTensor(input, &intermediateOutput);
    }
    return NULL;
}
