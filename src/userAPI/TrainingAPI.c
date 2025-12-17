#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#include "Layer.h"
#include "LossFunction.h"
#include "Optimizer.h"
#include "TensorAPI.h"
#include "StorageAPI.h"
#include "TrainingAPI.h"
#include "Linear.h"
#include "Relu.h"

void deInitGradTensor(tensor_t *tensor) {
    freeData(tensor);
    freeShape(tensor->shape);
    freeQuantization(tensor->quantization);
}

static void deInitLayerOutputs(tensor_t **layerOutputs, size_t sizeNetwork) {
    for (size_t i = 1; i <= sizeNetwork; i++) {
        freeTensor(layerOutputs[i]);
    }
}

static void initLayerOutputs(tensor_t **layerOutputs, layer_t **model, size_t sizeNetwork) {
    for (size_t i = 0; i < sizeNetwork; i++) {
        layer_t *currentLayer = model[i];
        quantization_t *currentQ = NULL;

        switch (currentLayer->type) {
        case LINEAR:
            linearConfig_t *linearConfig = currentLayer->config->linear;
            currentQ = linearConfig->propLossQ;
            break;
        case RELU:
            reluConfig_t *reluConfig = currentLayer->config->relu;
            currentQ = reluConfig->backwardQ;
            break;
        default:
            break;
        }


        calcOutputShapeFn_t calcOutputShape = layerFunctions[currentLayer->type].calcOutputShape;
        size_t numberOfDims = layerOutputs[i]->shape->numberOfDimensions;

        size_t *dims = *reserveMemory(numberOfDims * sizeof(size_t));
        size_t *order = *reserveMemory(numberOfDims * sizeof(size_t));
        shape_t *outShape = *reserveMemory(sizeof(shape_t));

        outShape->dimensions = dims;
        outShape->numberOfDimensions = numberOfDims;
        outShape->orderOfDimensions = order;

        calcOutputShape(currentLayer, layerOutputs[i]->shape, outShape);

        size_t numberOfValues = calcNumberOfElementsByShape(outShape);
        size_t sizeData = calcNumberOfBytesForData(currentQ, numberOfValues);
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
            symInt32QConfig_t *qC = *reserveMemory(sizeof(symInt32QConfig_t));
            qC->scale = currentQC->scale;
            qC->roundingMode = currentQC->roundingMode;
            q->qConfig = qC;
            break;
        default:
            break;
        }

        tensor_t *tensor = *reserveMemory(sizeof(tensor_t));
        tensor->data = data;
        tensor->quantization = q;
        tensor->shape = outShape;

        tensor->sparsity = NULL;
        if (layerOutputs[i]->sparsity != NULL) {
            sparsity_t *sparsity = *reserveMemory(sizeof(sparsity_t));
            tensor->sparsity = sparsity;
        }

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
    size_t sizeData = calcNumberOfBytesForData(currentQ, numberOfValues);
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
        symInt32QConfig_t *qC = *reserveMemory(sizeof(symInt32QConfig_t));
        qC->scale = currentQC->scale;
        qC->roundingMode = currentQC->roundingMode;
        q->qConfig = qC;
        break;
    default:
        break;
    }

    grad->data = data;
    grad->quantization = q;
    grad->shape = inShape;

    grad->sparsity = NULL;
    if (layerOutput->sparsity != NULL) {
        sparsity_t *sparsity = *reserveMemory(sizeof(sparsity_t));
        grad->sparsity = sparsity;
    }
}

trainingStats_t *initTrainingStats(tensor_t *label) {
    trainingStats_t *trainingStats = *reserveMemory(sizeof(trainingStats_t));

    size_t sizeOutput = calcNumberOfElementsByTensor(label);

    float *outputData = *reserveMemory(sizeOutput * sizeof(float));
    size_t outputNumberOfDims = label->shape->numberOfDimensions;
    size_t *outputDims = *reserveMemory(outputNumberOfDims * sizeof(size_t));
    quantization_t *outputQ = getQLike(label->quantization);
    tensor_t *output = tensorInit(outputData, outputDims, outputNumberOfDims, outputQ, NULL);

    trainingStats->output = output;

    return trainingStats;
}

void freeTrainingStats(trainingStats_t *trainingStats) {
    freeTensor(trainingStats->output);
    freeReservedMemory(trainingStats);
}


/*! IMPORTANT: We assume, that if you use Cross Entropy as your loss function,
 * you also use Softmax with it. We introduce Softmax as a dedicated Layer,
 * but in the backward pass it is ignored. We do this, because the Cross Entropy Backward
 * already takes the Softmax Backward into account.
 */
trainingStats_t *calculateGrads(layer_t **model, size_t sizeNetwork,
                                lossType_t lossType, tensor_t *input,
                                tensor_t *label) {

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

    trainingStats_t *trainingStats = initTrainingStats(label);
    copyTensor(trainingStats->output, layerOutputs[sizeNetwork]);

    // LOSS
    lossFunctions_t mseFns = lossFunctions[MSE];

    float loss = mseFns.forward(layerOutputs[sizeNetwork], label);
    trainingStats->loss = loss;

    tensor_t ping;
    initGradTensor(&ping, layerOutputs[sizeNetwork], model[sizeNetwork - 1]);
    tensor_t pong;
    bool toggle = 0;
    mseFns.backward(layerOutputs[sizeNetwork], label, &ping);

    // Backward pass
    size_t backwardIndex = sizeNetwork - 1;
    if (lossType == CROSS_ENTROPY) {
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

    deInitLayerOutputs(layerOutputs, sizeNetwork);

    if (!toggle) {
        deInitGradTensor(&ping);
    } else {
        deInitGradTensor(&pong);
    }

    return trainingStats;
}

/*! IMPORTANT: We assume, that if you use Cross Entropy as your loss function,
 * you also use Softmax with it. We introduce Softmax as a dedicated Layer,
 * but in the backward pass it is ignored. We do this, because the Cross Entropy Backward
 * already takes the Softmax Backward into account.
 */
trainingStats_t *trainingEpoch(layer_t **model, size_t sizeNetwork,
                               lossType_t lossFunctionType, tensor_t *input,
                               tensor_t *label, optimizer_t *optimizer) {

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

    trainingStats_t *trainingStats = initTrainingStats(label);
    copyTensor(trainingStats->output, layerOutputs[sizeNetwork]);

    // LOSS
    lossFunctions_t mseFns = lossFunctions[MSE];
    float loss = mseFns.forward(layerOutputs[sizeNetwork], label);
    trainingStats->loss = loss;

    tensor_t ping;
    initGradTensor(&ping, layerOutputs[sizeNetwork], model[sizeNetwork - 1]);
    tensor_t pong;
    bool toggle = 0;

    mseFns.backward(layerOutputs[sizeNetwork], label, &ping);

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

    deInitLayerOutputs(layerOutputs, sizeNetwork);

    if (!toggle) {
        deInitGradTensor(&ping);
    } else {
        deInitGradTensor(&pong);
    }

    optimizerFunctions_t optimFns = optimizerFunctions[optimizer->type];
    optimFns.step(optimizer);

    return trainingStats;
}
