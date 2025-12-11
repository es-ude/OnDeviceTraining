#include <stdio.h>

#include "Tensor.h"
#include "Layer.h"
#include "InferenceAPI.h"
#include "TensorAPI.h"
#include "StorageAPI.h"

tensor_t *inference(layer_t **model, size_t numberOfLayers, tensor_t *input) {
    for (size_t i = 0; i < numberOfLayers; i++) {
        layer_t *currentLayer = model[i];
        layerType_t currentLayerType = currentLayer->type;
        quantization_t *currentQ = currentLayer->outputQ;

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

        size_t sizeData = calcBytesOutputData(currentLayer->outputQ, numValues);
        uint8_t data[sizeData];

        quantization_t q;
        asymQConfig_t asymQConfig;
        switch (currentQ->type) {
        case FLOAT32:
            q.type = FLOAT32;
            q.qConfig = NULL;
            break;
        case ASYM:
            q.type = ASYM;
            asymQConfig_t *currentQC = currentQ->qConfig;
            asymQConfig.qBits = currentQC->qBits;
            asymQConfig.roundingMode = currentQC->roundingMode;
            q.qConfig = &asymQConfig;
            break;
        default:
            break;
        }

        tensor_t intermediateOutput;
        setTensorValues(&intermediateOutput, data, &outShape, &q, input->sparsity);

        forwardFn_t forward = layerFunctions[currentLayerType].forward;
        forward(currentLayer, input, &intermediateOutput);

        if (i == numberOfLayers - 1) {
            tensor_t *output = getTensorLike(&intermediateOutput);
            copyTensor(output, &intermediateOutput);
            return output;
        }

        copyTensor(input, &intermediateOutput);
    }
    return NULL;
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
        quantization_t *currentQ = currentLayer->outputQ;

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

        size_t sizeData = calcBytesOutputData(currentLayer->outputQ, numValues);
        uint8_t data[sizeData];

        quantization_t q;
        asymQConfig_t asymQConfig;
        switch (currentQ->type) {
        case FLOAT32:
            q.type = FLOAT32;
            q.qConfig = NULL;
            break;
        case ASYM:
            q.type = ASYM;
            asymQConfig_t *currentQC = currentQ->qConfig;
            asymQConfig.qBits = currentQC->qBits;
            asymQConfig.roundingMode = currentQC->roundingMode;
            q.qConfig = &asymQConfig;
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
