#include <math.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>

#include "Rounding.h"
#include "TensorConversion.h"
#include "TensorAPI.h"
#include "StorageAPI.h"
#include "QuantizationAPI.h"

#include <stdio.h>
#include <string.h>


static tensor_t *initTensorWithQInt32(int32_t *data, size_t *dims, size_t numberOfDims,
                                      quantization_t *quantization, bool isSparse) {
    tensor_t *tensor = malloc(sizeof(tensor_t));

    tensor->data = (uint8_t *)data;

    shape_t *shape = calloc(1, sizeof(shape_t));
    size_t *order = calloc(numberOfDims, sizeof(size_t));
    setOrderOfDimsForNewTensor(numberOfDims, order);
    setShape(shape, dims, numberOfDims, order);
    tensor->shape = shape;

    tensor->quantization = quantization;

    tensor->sparsityBitmask = NULL;

    if (isSparse) {
        size_t numberOfValues = calcNumberOfElementsByShape(shape);
        uint8_t *sparsityBitmask = calloc(numberOfValues, sizeof(uint8_t));
        tensor->sparsityBitmask = sparsityBitmask;
    }

    return tensor;
}

tensor_t *tensorInitInt32(int32_t *data, size_t *dims, size_t numberOfDims, bool isSparse) {
    quantization_t *q = calloc(1, sizeof(quantization_t));
    initInt32Quantization(q);

    return initTensorWithQInt32(data, dims, numberOfDims, q, isSparse);
}


static tensor_t *initTensorWithQFloat(float *data, size_t *dims, size_t numberOfDims,
                                      quantization_t *quantization, bool isSparse) {
    tensor_t *tensor = calloc(1, sizeof(tensor_t));

    tensor->data = (uint8_t *)data;

    shape_t *shape = calloc(1, sizeof(shape_t));
    size_t *order = calloc(numberOfDims, sizeof(size_t));
    setOrderOfDimsForNewTensor(numberOfDims, order);
    setShape(shape, dims, numberOfDims, order);
    tensor->shape = shape;

    tensor->quantization = quantization;

    tensor->sparsityBitmask = NULL;

    if (isSparse) {
        size_t numberOfValues = calcNumberOfElementsByShape(shape);
        uint8_t *sparsityBitmask = calloc(numberOfValues, sizeof(uint8_t));
        tensor->sparsityBitmask = sparsityBitmask;
    }

    return tensor;
}

tensor_t *tensorInitFloat(float *data, size_t *dims, size_t numberOfDims, bool isSparse) {
    quantization_t *q = calloc(1, sizeof(quantization_t));
    initFloat32Quantization(q);

    return initTensorWithQFloat(data, dims, numberOfDims, q, isSparse);
}


static tensor_t *initTensorWithQSymInt32(float *data, size_t *dims, size_t numberOfDims,
                                         quantization_t *quantization, bool isSparse) {
    tensor_t *floatTensor = tensorInitFloat(data, dims, numberOfDims, isSparse);

    tensor_t *symInt32Tensor = calloc(1, sizeof(tensor_t));

    shape_t *shape = calloc(1, sizeof(shape_t));
    size_t *order = calloc(numberOfDims, sizeof(size_t));
    setOrderOfDimsForNewTensor(numberOfDims, order);
    setShape(shape, dims, numberOfDims, order);

    symInt32Tensor->shape = shape;
    symInt32Tensor->quantization = quantization;
    convertTensor(floatTensor, symInt32Tensor);
    symInt32Tensor->sparsityBitmask = NULL;
    if (isSparse) {
        size_t numberOfValues = calcNumberOfElementsByShape(shape);
        uint8_t *sparsityBitmask = calloc(numberOfValues, sizeof(uint8_t));
        symInt32Tensor->sparsityBitmask = sparsityBitmask;
    }

    freeTensor(floatTensor);

    return symInt32Tensor;
}

tensor_t *tensorInitSymInt32(float *data, size_t *dims, size_t numberOfDims,
                             roundingMode_t roundingMode, bool isSparse) {
    quantization_t *symInt32Q = calloc(1, sizeof(quantization_t));
    symInt32QConfig_t *symInt32QC = calloc(1, sizeof(symInt32QConfig_t));
    initSymInt32QConfig(roundingMode, symInt32QC);
    initSymInt32Quantization(symInt32QC, symInt32Q);

    return initTensorWithQSymInt32(data, dims, numberOfDims, symInt32Q, isSparse);
}


static tensor_t *initTensorWithQAsym(float *data, size_t *dims, size_t numberOfDims,
                                     quantization_t *quantization, bool isSparse) {

    tensor_t *floatTensor = tensorInitFloat(data, dims, numberOfDims, isSparse);
    tensor_t *asymTensor = calloc(1, sizeof(tensor_t));

    shape_t *shape = calloc(1, sizeof(shape_t));
    size_t *order = calloc(numberOfDims, sizeof(size_t));
    setOrderOfDimsForNewTensor(numberOfDims, order);
    setShape(shape, dims, numberOfDims, order);

    // TODO implement dedicated function for this in Tensor.c
    asymQConfig_t *asymQC = quantization->qConfig;
    size_t bitsPerElement = asymQC->qBits;
    size_t numberOfValues = calcNumberOfElementsByShape(shape);
    size_t sizeData = ceilf((float)(numberOfValues * bitsPerElement / 8));
    uint8_t *asymData = calloc(1, sizeData);

    asymTensor->data = asymData;
    asymTensor->shape = shape;
    asymTensor->quantization = quantization;

    convertTensor(floatTensor, asymTensor);

    asymTensor->sparsityBitmask = NULL;
    if (floatTensor->sparsityBitmask != NULL) {
        uint8_t *sparsityBitmask = calloc(numberOfValues, sizeof(uint8_t));
        asymTensor->sparsityBitmask = sparsityBitmask;
    }

    freeTensor(floatTensor);

    return asymTensor;
}

tensor_t *tensorInitAsym(float *data, size_t *dims, size_t numberOfDims, uint8_t qBits,
                         roundingMode_t roundingMode, bool isSparse) {
    asymQConfig_t *asymQC = calloc(1, sizeof(asymQConfig_t));
    asymQC->qBits = qBits;
    asymQC->roundingMode = roundingMode;
    quantization_t *asymQ = calloc(1, sizeof(quantization_t));
    asymQ->type = ASYM;
    asymQ->qConfig = asymQC;

    return initTensorWithQAsym(data, dims, numberOfDims, asymQ, isSparse);
}


tensor_t *tensorInit(float *data, size_t *dims, size_t numberOfDims, quantization_t *quantization,
                     bool isSparse) {
    switch (quantization->type) {
    case FLOAT32:
        return initTensorWithQFloat(data, dims, numberOfDims, quantization, isSparse);
    case INT32:
        size_t size = 0;
        for (size_t i = 0; i < numberOfDims; i++) {
            size += dims[i];
        }
        int32_t *dataInt = calloc(size, sizeof(int32_t));
        for (size_t i = 0; i < size; i++) {
            dataInt[i] = (int32_t)data[i];
        }
    case SYM_INT32:
        return initTensorWithQSymInt32(data, dims, numberOfDims, quantization, isSparse);
    case ASYM:
        return initTensorWithQAsym(data, dims, numberOfDims, quantization, isSparse);
    default:
        return NULL;
    }
}


tensor_t *gradInitInt32(tensor_t *param) {
    tensor_t *grad = calloc(1, sizeof(tensor_t));

    grad->shape = param->shape;
    quantization_t *gradQ = calloc(1, sizeof(quantization_t));
    initInt32Quantization(gradQ);
    grad->quantization = gradQ;

    size_t numberOfValues = calcNumberOfElementsByTensor(param);
    size_t bytesPerElement = sizeof(int32_t);
    uint8_t *data = calloc(numberOfValues, bytesPerElement);
    grad->data = data;

    grad->sparsityBitmask = NULL;
    if (param->sparsityBitmask != NULL) {
        uint8_t *sparsityBitmask = calloc(numberOfValues, sizeof(uint8_t));
        grad->sparsityBitmask = sparsityBitmask;
    }

    return grad;
}

tensor_t *gradInitFloat(tensor_t *param) {
    tensor_t *grad = calloc(1, sizeof(tensor_t));

    grad->shape = param->shape;
    quantization_t *gradQ = calloc(1, sizeof(quantization_t));
    initFloat32Quantization(gradQ);
    grad->quantization = gradQ;

    size_t numberOfValues = calcNumberOfElementsByTensor(param);
    size_t bytesPerElement = sizeof(float);
    uint8_t *data = calloc(numberOfValues, bytesPerElement);
    grad->data = data;

    grad->sparsityBitmask = NULL;
    if (param->sparsityBitmask != NULL) {
        uint8_t *sparsityBitmask = calloc(numberOfValues, sizeof(uint8_t));
        grad->sparsityBitmask = sparsityBitmask;
    }

    return grad;
}

tensor_t *gradInitSymInt32(tensor_t *param, roundingMode_t roundingMode) {
    tensor_t *grad = calloc(1, sizeof(tensor_t));

    grad->shape = param->shape;
    symInt32QConfig_t *gradQC = calloc(1, sizeof(symInt32QConfig_t));
    initSymInt32QConfig(roundingMode, gradQC);
    quantization_t *gradQ = calloc(1, sizeof(quantization_t));
    initSymInt32Quantization(gradQC, gradQ);
    grad->quantization = gradQ;

    size_t numberOfValues = calcNumberOfElementsByTensor(param);
    size_t bytesPerElement = sizeof(float);
    uint8_t *data = calloc(numberOfValues, bytesPerElement);
    grad->data = data;

    grad->sparsityBitmask = NULL;
    if (param->sparsityBitmask != NULL) {
        uint8_t *sparsityBitmask = calloc(numberOfValues, sizeof(uint8_t));
        grad->sparsityBitmask = sparsityBitmask;
    }

    return grad;
}

tensor_t *gradInitAsym(tensor_t *param, uint8_t qBits, roundingMode_t roundingMode) {
    tensor_t *grad = calloc(1, sizeof(tensor_t));

    grad->shape = param->shape;
    asymQConfig_t *gradQC = calloc(1, sizeof(asymQConfig_t));
    initAsymQConfig(qBits, roundingMode, gradQC);
    quantization_t *gradQ = calloc(1, sizeof(quantization_t));
    initAsymQuantization(gradQC, gradQ);
    grad->quantization = gradQ;

    size_t numberOfValues = calcNumberOfElementsByTensor(param);
    size_t bytesPerElement = sizeof(float);
    uint8_t *data = calloc(numberOfValues, bytesPerElement);
    grad->data = data;

    grad->sparsityBitmask = NULL;
    if (param->sparsityBitmask != NULL) {
        uint8_t *sparsityBitmask = calloc(numberOfValues, sizeof(uint8_t));
        grad->sparsityBitmask = sparsityBitmask;
    }

    return grad;
}

// TODO
tensor_t *gradInit(tensor_t *param, quantization_t *gradQuantization) {}


// getLike

static shape_t *getShapeLike(shape_t *shape) {
    shape_t *likeShape = calloc(1, sizeof(shape_t));

    size_t numberOfDims = shape->numberOfDimensions;

    size_t *likeDims = calloc(numberOfDims, sizeof(size_t));
    memcpy(likeDims, shape->dimensions, numberOfDims * sizeof(size_t));

    size_t *likeOrder = calloc(numberOfDims, sizeof(size_t));
    setOrderOfDimsForNewTensor(numberOfDims, likeOrder);

    setShape(likeShape, likeDims, numberOfDims, likeOrder);

    return likeShape;
}

static quantization_t *getQLike(quantization_t *quantization) {
    quantization_t *likeQ = calloc(1, sizeof(quantization_t));
    switch (quantization->type) {
    case FLOAT32:
        initFloat32Quantization(likeQ);
        break;
    case INT32:
        initInt32Quantization(likeQ);
    case SYM_INT32:
        symInt32QConfig_t *likeSymInt32QC = calloc(1, sizeof(symInt32QConfig_t));
        symInt32QConfig_t *symInt32QC = quantization->qConfig;
        likeSymInt32QC->roundingMode = symInt32QC->roundingMode;
        likeQ->qConfig = likeSymInt32QC;
        break;
    case ASYM:
        asymQConfig_t *likeAsymQC = calloc(1, sizeof(asymQConfig_t));
        asymQConfig_t *asymQC = quantization->qConfig;
        likeAsymQC->qBits = asymQC->qBits;
        likeAsymQC->roundingMode = asymQC->roundingMode;
        likeQ->qConfig = likeAsymQC;
        break;
    default:
        return NULL;
    }
    return likeQ;
}

static uint8_t *getDataLike(quantization_t *quantization, size_t numberOfValues) {
    switch (quantization->type) {
    case FLOAT32:
        return calloc(numberOfValues, sizeof(float));
    case INT32:
        return calloc(numberOfValues, sizeof(int32_t));
    case SYM_INT32:
        return calloc(numberOfValues, sizeof(int32_t));
    case ASYM:
        asymQConfig_t *asymQC = quantization->qConfig;
        size_t totalBits = numberOfValues * asymQC->qBits;
        size_t totalBytes = ceilf(totalBits / 8);
        return calloc(1, totalBytes);
    default:
        return NULL;
    }
}

static uint8_t *getSparsityBitmaskLike(size_t size) {
    return calloc(size, sizeof(uint8_t));
}

tensor_t *getTensorLike(tensor_t *tensor) {
    tensor_t *likeTensor = calloc(1, sizeof(tensor_t));
    size_t numberOfValues = calcNumberOfElementsByShape(tensor->shape);
    likeTensor->data = getDataLike(tensor->quantization, numberOfValues);
    likeTensor->quantization = getQLike(tensor->quantization);
    likeTensor->shape = getShapeLike(tensor->shape);
    likeTensor->sparsityBitmask = getSparsityBitmaskLike(numberOfValues);

    return likeTensor;
}


// Free Functions

parameter_t *parameterInit(tensor_t *param, tensor_t *grad) {
    parameter_t *parameter = calloc(1, sizeof(parameter_t));
    parameter->param = param;
    parameter->grad = grad;

    return parameter;
}

void freeData(tensor_t *tensor) {
    freeReservedMemory(tensor->data);
    if (tensor->sparsityBitmask != NULL) {
        freeReservedMemory(tensor->sparsityBitmask);
    }
}

void freeShape(shape_t *shape) {
    freeReservedMemory(shape->dimensions);
    freeReservedMemory(shape->orderOfDimensions);
    freeReservedMemory(shape);
}

void freeQuantization(quantization_t *quantization) {
    freeReservedMemory(quantization->qConfig);
    freeReservedMemory((uint8_t *)quantization);
}

static void freeTensorPointer(tensor_t *tensor) {
    freeReservedMemory((uint8_t *)tensor);
}

void freeTensor(tensor_t *tensor) {
    freeData(tensor);
    freeShape(tensor->shape);
    freeQuantization(tensor->quantization);
    freeTensorPointer(tensor);
}
