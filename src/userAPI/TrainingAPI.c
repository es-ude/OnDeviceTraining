#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "Layer.h"
#include "MSE.h"
#include "TrainingAPI.h"
#include "TensorAPI.h"
#include "StorageAPI.h"

void deInitGradTensor(tensor_t *tensor) {
    freeData(tensor);
    freeShape(tensor->shape);
    freeQuantization(tensor->quantization);
}

static void deInitTensorPtrArray(tensor_t **tensorPtrArray, size_t sizeNetwork, size_t startIndex) {
    for (size_t i = startIndex; i <= sizeNetwork; i++) {
        freeTensor(tensorPtrArray[i]);
    }
}

static void getLossFunctionByType(lossFunctionType_t lossType, lossFn_t *lossFunction) {
    switch (lossType) {
    case MSE:
        *lossFunction = MSELossBackward;
        break;
    case CROSS_ENTROPY:
        // lossFunction = crossEntropySoftmaxBackward;
        break;
    default:
        printf("Loss type not found");
        break;
    }
}

static void initLayerOutputs(tensor_t **layerOutputs, layer_t **model, size_t sizeNetwork) {
    for (size_t i = 0; i < sizeNetwork; i++) {
        layer_t *currentLayer = model[i];
        quantization_t *currentQ = currentLayer->outputQ;
        layerType_t currentLayerType = currentLayer->type;
        calcOutputShapeFn_t calcOutputShape = layerFunctions[currentLayerType].calcOutputShape;
        size_t numberOfDims = layerOutputs[i]->shape->numberOfDimensions;

        size_t *dims = *reserveMemory(numberOfDims * sizeof(size_t));
        size_t *order = *reserveMemory(numberOfDims * sizeof(size_t));
        shape_t *outShape = *reserveMemory(sizeof(shape_t));

        outShape->dimensions = dims;
        outShape->numberOfDimensions = numberOfDims;
        outShape->orderOfDimensions = order;

        calcOutputShape(currentLayer, layerOutputs[i]->shape, outShape);

        size_t numberOfValues = calcNumberOfElementsByShape(outShape);
        size_t sizeData = calcBytesOutputData(currentLayer->outputQ, numberOfValues);
        uint8_t *data = *reserveMemory(sizeData);
        uint8_t *sparsityBitmask = *reserveMemory(numberOfValues);

        quantization_t *q = *reserveMemory(sizeof(quantization_t));
        switch (currentQ->type) {
        case FLOAT32:
            q->type = FLOAT32;
            q->qConfig = NULL;
            break;
        case ASYM:
            q->type = ASYM;
            asymQConfig_t *currentQC = currentQ->qConfig;
            asymQConfig_t *qC = *reserveMemory(sizeof(asymQConfig_t));
            qC->scale = currentQC->scale;
            qC->qBits = currentQC->qBits;
            qC->roundingMode = currentQC->roundingMode;
            qC->zeroPoint = currentQC->zeroPoint;
            q->qConfig = qC;
            break;
        default:
            break;
        }

        tensor_t *tensor = *reserveMemory(sizeof(tensor_t));
        tensor->data = data;
        tensor->quantization = q;
        tensor->shape = outShape;
        tensor->sparsityBitmask = sparsityBitmask;

        layerOutputs[i + 1] = tensor;
    }
}

static void initGradTensor(tensor_t *grad, tensor_t *layerOutput, layer_t *layer) {
    shape_t *currentShape = layerOutput->shape;
    quantization_t *currentQ = layerOutput->quantization;

    size_t *dims = *reserveMemory(currentShape->numberOfDimensions * sizeof(size_t));
    size_t *order = *reserveMemory(currentShape->numberOfDimensions * sizeof(size_t));
    shape_t *inShape = *reserveMemory(sizeof(shape_t));

    inShape->dimensions = dims;
    inShape->numberOfDimensions = currentShape->numberOfDimensions;
    inShape->orderOfDimensions = order;

    memcpy(inShape->dimensions, currentShape->dimensions,
           currentShape->numberOfDimensions * sizeof(size_t));
    memcpy(inShape->orderOfDimensions, currentShape->orderOfDimensions,
           currentShape->numberOfDimensions * sizeof(size_t));

    setOrderOfDimsForNewTensor(inShape->numberOfDimensions, inShape->orderOfDimensions);

    size_t numberOfValues = calcNumberOfElementsByShape(currentShape);
    size_t sizeData = calcBytesOutputData(currentQ, numberOfValues);
    uint8_t *data = *reserveMemory(sizeData);
    uint8_t *sparsityBitmask = *reserveMemory(numberOfValues);

    quantization_t *q = *reserveMemory(sizeof(quantization_t));
    switch (currentQ->type) {
    case FLOAT32:
        q->type = FLOAT32;
        q->qConfig = NULL;
        break;
    case ASYM:
        q->type = ASYM;
        asymQConfig_t *currentQC = currentQ->qConfig;
        asymQConfig_t *qC = *reserveMemory(sizeof(asymQConfig_t));
        qC->scale = currentQC->scale;
        qC->qBits = currentQC->qBits;
        qC->roundingMode = currentQC->roundingMode;
        qC->zeroPoint = currentQC->zeroPoint;
        q->qConfig = qC;
        break;
    default:
        break;
    }

    grad->data = data;
    grad->quantization = q;
    grad->shape = inShape;
    grad->sparsityBitmask = sparsityBitmask;
}

static quantization_t *getLikeQuantization(quantization_t *quantization) {
    quantization_t *likeQ = calloc(1, sizeof(quantization_t));
    likeQ->type = quantization->type;

    switch(quantization->type) {
    case FLOAT32:
        likeQ->qConfig = NULL;
        break;
    case ASYM:
        asymQConfig_t *asymQC = quantization->qConfig;
        asymQConfig_t *likeAsymQC = calloc(1, sizeof(asymQConfig_t));
        likeAsymQC->qBits = asymQC->qBits;
        likeAsymQC->roundingMode = asymQC->roundingMode;
        likeQ->qConfig = likeAsymQC;
    default:
        break;
    }
    return likeQ;
}

trainingStats_t * initTrainingStats(tensor_t *label) {


    trainingStats_t *trainingStats = calloc(1, sizeof(trainingStats_t));

    size_t sizeOutput = calcNumberOfElementsByTensor(label);
    size_t sizeTrainingStatsEntryData = calcBytesOutputData(label->quantization, sizeOutput);

    uint8_t *outputData = calloc(1, sizeTrainingStatsEntryData);
    size_t outputNumberOfDims = label->shape->numberOfDimensions;
    size_t *outputDims = calloc(outputNumberOfDims, sizeof(size_t));
    quantization_t *outputQ = getLikeQuantization(label->quantization);
    tensor_t *output = tensorInit(outputData, outputDims, outputNumberOfDims, false);

    // TODO implement tensorInit by quantization

    uint8_t *lossData = calloc(1, sizeTrainingStatsEntryData);
    size_t lossNumberOfDims = label->shape->numberOfDimensions;
    size_t *lossDims = calloc(lossNumberOfDims, sizeof(size_t));
    quantization_t *lossQ = getLikeQuantization(label->quantization);
    tensor_t *loss = tensorInit(lossData, lossDims, lossNumberOfDims, false);

    trainingStats->loss = loss;
    trainingStats->output = output;
}


/*! IMPORTANT: We assume, that if you use Cross Entropy as your loss function,
 * you also use Softmax with it. We introduce Softmax as a dedicated Layer,
 * but in the backward pass it is ignored. We do this, because the Cross Entropy Backward
 * already takes the Softmax Backward into account.
 */
trainingStats_t *calculateGrads(layer_t **model, size_t sizeNetwork,
                    lossFunctionType_t lossFunctionType, tensor_t *input, tensor_t *label) {

    tensor_t *layerOutputs[sizeNetwork + 1];
    layerOutputs[0] = input;
    initLayerOutputs(layerOutputs, model, sizeNetwork);

    // Forward pass
    for (size_t i = 0; i < sizeNetwork; i++) {
        layer_t *currentLayer = model[i];
        layerType_t currentLayerType = currentLayer->type;
        forwardFn_t forward = layerFunctions[currentLayerType].forward;
        forward(currentLayer, layerOutputs[i], layerOutputs[i + 1]);
    }

    copyTensor(trainingStats->output, layerOutputs[sizeNetwork]);

    // LOSS
    lossFn_t lossFn;
    getLossFunctionByType(lossFunctionType, &lossFn);

    tensor_t ping;
    initGradTensor(&ping, layerOutputs[sizeNetwork], model[sizeNetwork - 1]);
    tensor_t pong;
    bool toggle = 0;

    lossFn(layerOutputs[sizeNetwork], label, &ping);
    copyTensor(trainingStats->loss, &ping);

    // Backward pass
    size_t backwardIndex = sizeNetwork - 1;
    if (lossFunctionType == CROSS_ENTROPY) {
        backwardIndex -= 1;
    }

    for (int i = (int)backwardIndex; i >= 0; i--) {
        if (!toggle) {
            initGradTensor(&pong, layerOutputs[backwardIndex], model[backwardIndex]);
            layerType_t layerType = model[i]->type;
            backwardFn_t backward = layerFunctions[layerType].backward;
            backward(model[i], layerOutputs[i], &ping, &pong);
            deInitGradTensor(&ping);
            toggle = true;
        } else {
            initGradTensor(&ping, layerOutputs[backwardIndex], model[backwardIndex]);
            layerType_t layerType = model[i]->type;
            backwardFn_t backward = layerFunctions[layerType].backward;
            backward(model[i], layerOutputs[i], &pong, &ping);
            deInitGradTensor(&pong);
            toggle = false;
        }
    }
    deInitTensorPtrArray(layerOutputs, sizeNetwork, 1);
    freeTensor(&ping);
    freeTensor(&pong);

    return trainingStats;
}

void freeTrainingStats(trainingStats_t *trainingStats) {
    freeTensor(trainingStats->loss);
    freeTensor(trainingStats->output);
    freeReservedMemory(trainingStats);
}