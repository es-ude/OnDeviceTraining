#define SOURCE_FILE "CALCULATE_GRADS_SEQUENTIAL"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "AdaptiveAvgPool1d.h"
#include "AvgPool1d.h"
#include "CalculateGradsSequential.h"
#include "Common.h"
#include "Conv1d.h"
#include "Conv1dTransposed.h"
#include "Dropout.h"
#include "Layer.h"
#include "LayerNorm.h"
#include "Linear.h"
#include "LossFunction.h"
#include "MaxPool1d.h"
#include "QuantizationLayer.h"
#include "Relu.h"
#include "Softmax.h"
#include "StorageApi.h"
#include "TensorApi.h"
#include "TrainingLoopApiInternal.h"

static void setDropoutLayersTraining(layer_t **model, size_t modelSize, bool training) {
    for (size_t i = 0; i < modelSize; i++) {
        if (model[i]->type == DROPOUT) {
            model[i]->config->dropout->training = training;
        }
    }
}

trainingStats_t *calculateGradsSequential(layer_t **model, size_t modelSize,
                                          lossConfig_t lossConfig, reduction_t forwardReduction,
                                          tensor_t *input, tensor_t *label) {

    tensor_t *layerOutputs[modelSize + 1];
    layerOutputs[0] = input;
    setDropoutLayersTraining(model, modelSize, true);
    initLayerOutputs(layerOutputs, model, modelSize);

    // Forward pass
    for (size_t i = 0; i < modelSize; i++) {
        layer_t *currentLayer = model[i];
        layerType_t currentLayerType = currentLayer->type;
        forwardFn_t forward = layerFunctions[currentLayerType].forward;
        forward(currentLayer, layerOutputs[i], layerOutputs[i + 1]);
    }

    trainingStats_t *trainingStats = initTrainingStats(layerOutputs[modelSize]);
    copyTensor(trainingStats->output, layerOutputs[modelSize]);

    // LOSS

    lossFunctions_t lossFns = lossFunctions[lossConfig.funcType];
    float loss = lossFns.forward(layerOutputs[modelSize], label, forwardReduction);
    trainingStats->loss = loss;

    // Backward pass
    size_t backwardIndex = modelSize - 1;
    if (lossConfig.funcType == CROSS_ENTROPY) {
        backwardIndex -= 1;
    }

    tensor_t gradNext;
    initGradTensor(&gradNext, layerOutputs[modelSize]);
    lossFns.backward(layerOutputs[modelSize], label, &gradNext);

    for (int i = (int)backwardIndex; i >= 0; i--) {
        tensor_t gradCurr;
        initGradTensor(&gradCurr, layerOutputs[i]);

        layerType_t layerType = model[i]->type;
        backwardFn_t backward = layerFunctions[layerType].backward;

        backward(model[i], layerOutputs[i], &gradNext, &gradCurr);

        deInitGradTensor(&gradNext);
        gradNext = gradCurr;
    }

    deInitGradTensor(&gradNext);
    deInitLayerOutputs(layerOutputs, modelSize);

    setDropoutLayersTraining(model, modelSize, false);
    return trainingStats;
}

static void initLayerOutputs(tensor_t **layerOutputs, layer_t **model, size_t sizeNetwork) {
    for (size_t i = 0; i < sizeNetwork; i++) {
        layer_t *currentLayer = model[i];
        quantization_t *currentQ = NULL;

        switch (currentLayer->type) {
        case LINEAR:
            linearConfig_t *linearConfig = currentLayer->config->linear;
            currentQ = linearConfig->forwardQ;
            break;
        case RELU:
            reluConfig_t *reluConfig = currentLayer->config->relu;
            currentQ = reluConfig->forwardQ;
            break;
        case SOFTMAX:
            softmaxConfig_t *softmaxConfig = currentLayer->config->softmax;
            currentQ = softmaxConfig->forwardQ;
            break;
        case FLATTEN:
            // Flatten has no per-layer quantization; output dtype equals input dtype.
            currentQ = layerOutputs[i]->quantization;
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

        calcOutputShapeFn_t calcOutputShape = layerFunctions[currentLayer->type].calcOutputShape;
        size_t numberOfDims = layerOutputs[i]->shape->numberOfDimensions;
        if (currentLayer->type == FLATTEN) {
            numberOfDims = 2;
        }

        size_t *dims = reserveMemory(numberOfDims * sizeof(size_t));
        size_t *order = reserveMemory(numberOfDims * sizeof(size_t));
        shape_t *outShape = reserveMemory(sizeof(shape_t));

        outShape->dimensions = dims;
        outShape->numberOfDimensions = numberOfDims;
        outShape->orderOfDimensions = order;

        calcOutputShape(currentLayer, layerOutputs[i]->shape, outShape);

        size_t numberOfValues = calcNumberOfElementsByShape(outShape);
        size_t sizeData = calcNumberOfBytesForData(currentQ, numberOfValues);
        uint8_t *data = reserveMemory(sizeData);

        quantization_t *q = reserveMemory(sizeof(quantization_t));
        switch (currentQ->type) {
        case FLOAT32:
            initFloat32Quantization(q);
            break;
        case SYM_INT32:
            q->type = SYM_INT32;
            symInt32QConfig_t *currentQC = currentQ->qConfig;
            symInt32QConfig_t *qC = reserveMemory(sizeof(symInt32QConfig_t));
            initSymInt32QConfig(currentQC->roundingMode, qC);
            initSymInt32Quantization(qC, q);
            break;
        default:
            PRINT_ERROR("Unknown QType!");
            exit(1);
        }

        tensor_t *tensor = reserveMemory(sizeof(tensor_t));
        tensor->data = data;
        tensor->quantization = q;
        tensor->shape = outShape;

        tensor->sparsity = NULL;
        if (layerOutputs[i]->sparsity != NULL) {
            sparsity_t *sparsity = reserveMemory(sizeof(sparsity_t));
            tensor->sparsity = sparsity;
        }

        layerOutputs[i + 1] = tensor;
    }
}

static void deInitLayerOutputs(tensor_t **layerOutputs, size_t modelSize) {
    for (size_t i = 1; i <= modelSize; i++) {
        freeTensor(layerOutputs[i]);
    }
}

static void initGradTensor(tensor_t *grad, tensor_t *layerOutput) {
    shape_t *currentShape = layerOutput->shape;
    quantization_t *currentQ = layerOutput->quantization;

    size_t *dims = reserveMemory(currentShape->numberOfDimensions * sizeof(size_t));
    size_t *order = reserveMemory(currentShape->numberOfDimensions * sizeof(size_t));
    shape_t *inShape = reserveMemory(sizeof(shape_t));

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
    uint8_t *data = reserveMemory(sizeData);

    quantization_t *q = reserveMemory(sizeof(quantization_t));
    switch (currentQ->type) {
    case FLOAT32:
        initFloat32Quantization(q);
        break;
    case SYM_INT32:
        symInt32QConfig_t *currentQC = currentQ->qConfig;
        symInt32QConfig_t *qC = reserveMemory(sizeof(symInt32QConfig_t));
        initSymInt32QConfig(currentQC->roundingMode, qC);
        initSymInt32Quantization(qC, q);
        break;
    default:
        PRINT_ERROR("Unknown QType!");
        exit(1);
    }

    grad->data = data;
    grad->quantization = q;
    grad->shape = inShape;

    grad->sparsity = NULL;
    if (layerOutput->sparsity != NULL) {
        sparsity_t *sparsity = reserveMemory(sizeof(sparsity_t));
        grad->sparsity = sparsity;
    }
}

static void deInitGradTensor(tensor_t *tensor) {
    freeData(tensor);
    freeShape(tensor->shape);
    freeQuantization(tensor->quantization);
}

static trainingStats_t *initTrainingStats(tensor_t *output) {
    trainingStats_t *trainingStats = reserveMemory(sizeof(trainingStats_t));

    tensor_t *o = getTensorLike(output);
    trainingStats->output = o;

    return trainingStats;
}
