#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tgmath.h>

#include "ModelAPI.h"
#include "Layer.h"
#include "MSE.h"
#include "Linear.h"

// Important: For now the abstraction layer for memory allocation is located in ModelAPI, because it is not needed anywhere else
static void **reserveMemory(size_t numberOfBytes) {
    void *ptr = calloc(1, numberOfBytes);
    void **handle = malloc(sizeof(void *));
    *handle = ptr;
    return handle;
}

static void freeReservedMemory(void *ptr) {
    free(ptr);
}

static void freeShape(shape_t *shape) {
    freeReservedMemory(shape->dimensions);
    freeReservedMemory(shape->orderOfDimensions);
    freeReservedMemory(shape);
}

static void freeData(tensor_t *tensor) {
    freeReservedMemory(tensor->data);
    if (tensor->sparsityBitmask != NULL) {
        freeReservedMemory(tensor->sparsityBitmask);
    }
}

static void freeQuantization(quantization_t *quantization) {
    freeReservedMemory(quantization->qConfig);
    freeReservedMemory((uint8_t *)quantization);
}

static void freeTensorPointer(tensor_t *tensor) {
    freeReservedMemory((uint8_t *)tensor);
}

static void deInitTensorPtrArray(tensor_t **tensorPtrArray, size_t sizeNetwork, size_t startIndex) {
    for (size_t i = startIndex; i <= sizeNetwork; i++) {
        freeShape(tensorPtrArray[i]->shape);
        freeData(tensorPtrArray[i]);
        freeQuantization(tensorPtrArray[i]->quantization);
        freeTensorPointer(tensorPtrArray[i]);
    }
}

static void deInitGradTensor(tensor_t *tensor) {
    freeShape(tensor->shape);
    freeData(tensor);
    freeQuantization(tensor->quantization);
}


static size_t calcBytesOutputData(quantization_t *outputQ, size_t numberOfValues) {
    switch (outputQ->type) {
    case FLOAT32:
        return numberOfValues * sizeof(float);
    case ASYM:
        size_t bitsPerElement = calcBitsPerElement(outputQ);
        return ceil((bitsPerElement * numberOfValues) / 8);
    default:
        return 0;
    }
}

void inference(layer_t **model, size_t numberOfLayers, tensor_t *input, tensor_t *output) {
    for (size_t i = 0; i < numberOfLayers; i++) {
        layer_t *currentLayer = model[i];
        layerType_t currentLayerType = currentLayer->type;
        quantization_t *currentQ = currentLayer->outputQ;

        size_t outputNumberOfDims = input->shape->numberOfDimensions;
        size_t sizeDims = outputNumberOfDims * sizeof(size_t);
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
        void *maybeSparsityBitmask = NULL;
        if (input->sparsityBitmask != NULL) {
            uint8_t sparsityBitmask[sizeData];
            maybeSparsityBitmask = sparsityBitmask;
        }

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
            asymQConfig.scale = currentQC->scale;
            asymQConfig.qBits = currentQC->qBits;
            asymQConfig.roundingMode = currentQC->roundingMode;
            asymQConfig.zeroPoint = currentQC->zeroPoint;
            q.qConfig = &asymQConfig;
            break;
        default:
            break;
        }

        tensor_t intermediateOutput;
        setTensorValues(&intermediateOutput, data, &outShape, &q, maybeSparsityBitmask);

        forwardFn_t forward = layerFunctions[currentLayerType].forward;
        forward(currentLayer, input, &intermediateOutput);

        /*printf("Sequential Output [%lu]:\n", i);
        printTensor(&intermediateOutput);*/

        if (i == numberOfLayers - 1) {
            copyTensor(output, &intermediateOutput);
            break;
        }

        copyTensor(input, &intermediateOutput);
    }
}

void getLossFunctionByType(lossFunctionType_t lossType, lossFn_t *lossFunction) {
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
    quantization_t *currentQ = layer->inputQ;

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


/*! IMPORTANT: We assume, that if you use Cross Entropy as your loss function,
 * you also use Softmax with it. We introduce Softmax as a dedicated Layer,
 * but in the backward pass it is ignored. We do this, because the Cross Entropy Backward
 * already takes the Softmax Backward into account.
 */
void calculateGrads(layer_t **model, size_t sizeNetwork,
                    lossFunctionType_t lossFunctionType, tensor_t *input, tensor_t *label,
                    trainingStats_t *trainingStats) {

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
}
