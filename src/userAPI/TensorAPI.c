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

// tensor inits

static tensor_t *initTensorWithQInt32(int32_t *data, size_t *dims, size_t numberOfDims,
                                      quantization_t *quantization, sparsity_t *sparsity) {
    tensor_t *tensor = malloc(sizeof(tensor_t));
    tensor->data = (uint8_t *)data;
    shape_t *shape = *reserveMemory(sizeof(shape_t));
    size_t *order = *reserveMemory(numberOfDims * sizeof(size_t));
    setOrderOfDimsForNewTensor(numberOfDims, order);
    setShape(shape, dims, numberOfDims, order);
    tensor->shape = shape;
    tensor->quantization = quantization;
    tensor->sparsity = sparsity;

    return tensor;
}

tensor_t *tensorInitInt32(int32_t *data, size_t *dims, size_t numberOfDims, sparsity_t *sparsity) {
    quantization_t *q = *reserveMemory(sizeof(quantization_t));
    initInt32Quantization(q);

    return initTensorWithQInt32(data, dims, numberOfDims, q, sparsity);
}


static tensor_t *initTensorWithQFloat(float *data, size_t *dims, size_t numberOfDims,
                                      quantization_t *quantization, sparsity_t *sparsity) {
    tensor_t *tensor = *reserveMemory(sizeof(tensor_t));

    tensor->data = (uint8_t *)data;

    shape_t *shape = *reserveMemory(sizeof(shape_t));
    size_t *order = *reserveMemory(numberOfDims * sizeof(size_t));
    setOrderOfDimsForNewTensor(numberOfDims, order);
    setShape(shape, dims, numberOfDims, order);
    tensor->shape = shape;
    tensor->quantization = quantization;
    tensor->sparsity = sparsity;

    return tensor;
}

tensor_t *tensorInitFloat(float *data, size_t *dims, size_t numberOfDims, sparsity_t *sparsity) {
    quantization_t *q = *reserveMemory(sizeof(quantization_t));
    initFloat32Quantization(q);

    return initTensorWithQFloat(data, dims, numberOfDims, q, sparsity);
}


static tensor_t *initTensorWithQSymInt32(float *data, size_t *dims, size_t numberOfDims,
                                         quantization_t *quantization, sparsity_t *sparsity) {
    tensor_t *floatTensor = tensorInitFloat(data, dims, numberOfDims, sparsity);

    tensor_t *symInt32Tensor = *reserveMemory(sizeof(tensor_t));

    shape_t *shape = *reserveMemory(sizeof(shape_t));
    size_t *order = *reserveMemory(numberOfDims * sizeof(size_t));
    setOrderOfDimsForNewTensor(numberOfDims, order);
    setShape(shape, dims, numberOfDims, order);

    symInt32Tensor->shape = shape;
    symInt32Tensor->quantization = quantization;
    convertTensor(floatTensor, symInt32Tensor);
    symInt32Tensor->sparsity = sparsity;

    freeTensor(floatTensor);

    return symInt32Tensor;
}

tensor_t *tensorInitSymInt32(float *data, size_t *dims, size_t numberOfDims,
                             roundingMode_t roundingMode, sparsity_t *sparsity) {
    quantization_t *symInt32Q = *reserveMemory(sizeof(quantization_t));
    symInt32QConfig_t *symInt32QC = *reserveMemory(sizeof(symInt32QConfig_t));
    initSymInt32QConfig(roundingMode, symInt32QC);
    initSymInt32Quantization(symInt32QC, symInt32Q);

    return initTensorWithQSymInt32(data, dims, numberOfDims, symInt32Q, sparsity);
}


static tensor_t *initTensorWithQAsym(float *data, size_t *dims, size_t numberOfDims,
                                     quantization_t *quantization, sparsity_t *sparsity) {

    shape_t *shape = *reserveMemory(sizeof(shape_t));
    size_t *order = *reserveMemory(numberOfDims * sizeof(size_t));
    setOrderOfDimsForNewTensor(numberOfDims, order);
    setShape(shape, dims, numberOfDims, order);

    tensor_t floatTensor;
    quantization_t floatQ;
    initFloat32Quantization(&floatQ);

    floatTensor.data = (uint8_t *)data;
    floatTensor.shape = shape;
    floatTensor.quantization = &floatQ;
    floatTensor.sparsity = sparsity;

    tensor_t *asymTensor = *reserveMemory(sizeof(tensor_t));

    asymQConfig_t *asymQC = quantization->qConfig;
    size_t bitsPerElement = asymQC->qBits;
    size_t numberOfValues = calcNumberOfElementsByShape(shape);
    size_t sizeData = ceilf((float)(numberOfValues * bitsPerElement / 8));
    uint8_t *asymData = *reserveMemory(sizeData);

    asymTensor->data = asymData;
    asymTensor->shape = shape;
    asymTensor->quantization = quantization;
    asymTensor->sparsity = sparsity;

    convertTensor(&floatTensor, asymTensor);

    return asymTensor;
}

tensor_t *tensorInitAsym(float *data, size_t *dims, size_t numberOfDims, uint8_t qBits,
                         roundingMode_t roundingMode, sparsity_t *sparsity) {
    asymQConfig_t *asymQC = *reserveMemory(sizeof(asymQConfig_t));
    asymQC->qBits = qBits;
    asymQC->roundingMode = roundingMode;
    quantization_t *asymQ = *reserveMemory(sizeof(quantization_t));
    asymQ->type = ASYM;
    asymQ->qConfig = asymQC;

    return initTensorWithQAsym(data, dims, numberOfDims, asymQ, sparsity);
}


tensor_t *tensorInit(float *data, size_t *dims, size_t numberOfDims, quantization_t *quantization, sparsity_t *sparsity) {
    switch (quantization->type) {
    case FLOAT32:
        return initTensorWithQFloat(data, dims, numberOfDims, quantization, sparsity);
    case INT32:
        size_t size = 0;
        for (size_t i = 0; i < numberOfDims; i++) {
            size += dims[i];
        }
        int32_t *dataInt = *reserveMemory(size * sizeof(int32_t));
        for (size_t i = 0; i < size; i++) {
            dataInt[i] = (int32_t)data[i];
        }
        return initTensorWithQInt32(dataInt, dims, numberOfDims, quantization, sparsity);
    case SYM_INT32:
        return initTensorWithQSymInt32(data, dims, numberOfDims, quantization, sparsity);
    case ASYM:
        return initTensorWithQAsym(data, dims, numberOfDims, quantization, sparsity);
    default:
        return NULL;
    }
}


// grad inits

tensor_t *gradInitInt32(tensor_t *param, sparsity_t *sparsity) {
    tensor_t *grad = *reserveMemory(sizeof(tensor_t));

    grad->shape = param->shape;
    quantization_t *gradQ = *reserveMemory(sizeof(quantization_t));
    initInt32Quantization(gradQ);
    grad->quantization = gradQ;

    size_t numberOfValues = calcNumberOfElementsByTensor(param);
    size_t bytesPerElement = sizeof(int32_t);
    uint8_t *data = *reserveMemory(numberOfValues * bytesPerElement);
    grad->data = data;

    grad->sparsity = sparsity;

    return grad;
}

tensor_t *gradInitFloat(tensor_t *param, sparsity_t *sparsity) {
    tensor_t *grad = *reserveMemory(sizeof(tensor_t));

    grad->shape = param->shape;
    quantization_t *gradQ = *reserveMemory(sizeof(quantization_t));
    initFloat32Quantization(gradQ);
    grad->quantization = gradQ;

    size_t numberOfValues = calcNumberOfElementsByTensor(param);
    size_t bytesPerElement = sizeof(float);
    uint8_t *data = *reserveMemory(numberOfValues * bytesPerElement);
    grad->data = data;

    grad->sparsity = sparsity;

    return grad;
}

tensor_t *gradInitSymInt32(tensor_t *param, roundingMode_t roundingMode, sparsity_t *sparsity) {
    tensor_t *grad = *reserveMemory(sizeof(tensor_t));

    grad->shape = param->shape;
    symInt32QConfig_t *gradQC = *reserveMemory(sizeof(symInt32QConfig_t));
    initSymInt32QConfig(roundingMode, gradQC);
    quantization_t *gradQ = *reserveMemory(sizeof(quantization_t));
    initSymInt32Quantization(gradQC, gradQ);
    grad->quantization = gradQ;

    size_t numberOfValues = calcNumberOfElementsByTensor(param);
    size_t bytesPerElement = sizeof(float);
    uint8_t *data = *reserveMemory(numberOfValues * bytesPerElement);
    grad->data = data;

    grad->sparsity = sparsity;

    return grad;
}

tensor_t *gradInitAsym(tensor_t *param, uint8_t qBits, roundingMode_t roundingMode, sparsity_t *sparsity) {
    tensor_t *grad = *reserveMemory(sizeof(tensor_t));

    grad->shape = param->shape;
    asymQConfig_t *gradQC = *reserveMemory(sizeof(asymQConfig_t));
    initAsymQConfig(qBits, roundingMode, gradQC);
    quantization_t *gradQ = *reserveMemory(sizeof(quantization_t));
    initAsymQuantization(gradQC, gradQ);
    grad->quantization = gradQ;

    size_t numberOfValues = calcNumberOfElementsByTensor(param);
    size_t bytesPerElement = sizeof(float);
    uint8_t *data = *reserveMemory(numberOfValues * bytesPerElement);
    grad->data = data;

    grad->sparsity = sparsity;

    return grad;
}

// getLike

static shape_t *getShapeLike(shape_t *shape) {
    shape_t *likeShape = *reserveMemory(sizeof(shape_t));

    size_t numberOfDims = shape->numberOfDimensions;

    size_t *likeDims = *reserveMemory(numberOfDims * sizeof(size_t));
    memcpy(likeDims, shape->dimensions, numberOfDims * sizeof(size_t));

    size_t *likeOrder = *reserveMemory(numberOfDims * sizeof(size_t));
    setOrderOfDimsForNewTensor(numberOfDims, likeOrder);

    setShape(likeShape, likeDims, numberOfDims, likeOrder);

    return likeShape;
}

quantization_t *getQLike(quantization_t *quantization) {
    quantization_t *likeQ = *reserveMemory(sizeof(quantization_t));
    switch (quantization->type) {
    case FLOAT32:
        initFloat32Quantization(likeQ);
        break;
    case INT32:
        initInt32Quantization(likeQ);
        break;
    case SYM_INT32:
        symInt32QConfig_t *likeSymInt32QC = *reserveMemory(sizeof(symInt32QConfig_t));
        symInt32QConfig_t *symInt32QC = quantization->qConfig;
        likeSymInt32QC->roundingMode = symInt32QC->roundingMode;
        likeQ->qConfig = likeSymInt32QC;
        break;
    case ASYM:
        asymQConfig_t *likeAsymQC = *reserveMemory(sizeof(asymQConfig_t));
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
        return *reserveMemory(numberOfValues * sizeof(float));
    case INT32:
        return *reserveMemory(numberOfValues * sizeof(int32_t));
    case SYM_INT32:
        return *reserveMemory(numberOfValues * sizeof(int32_t));
    case ASYM:
        asymQConfig_t *asymQC = quantization->qConfig;
        size_t totalBits = numberOfValues * asymQC->qBits;
        size_t totalBytes = ceilf(totalBits / 8);
        return *reserveMemory(totalBytes);
    default:
        return NULL;
    }
}

static sparsity_t *getSparsityLike(sparsity_t *sparsity) {
    if(sparsity != NULL) {
        return *reserveMemory(sizeof(sparsity_t));
    }
    return NULL;
}

tensor_t *getTensorLike(tensor_t *tensor) {
    tensor_t *likeTensor = *reserveMemory(sizeof(tensor_t));
    size_t numberOfValues = calcNumberOfElementsByShape(tensor->shape);
    likeTensor->data = getDataLike(tensor->quantization, numberOfValues);
    likeTensor->quantization = getQLike(tensor->quantization);
    likeTensor->shape = getShapeLike(tensor->shape);
    likeTensor->sparsity = getSparsityLike(tensor->sparsity);

    return likeTensor;
}

// Free Functions

parameter_t *parameterInit(tensor_t *param, tensor_t *grad) {
    parameter_t *parameter = *reserveMemory(sizeof(parameter_t));
    parameter->param = param;
    parameter->grad = grad;

    return parameter;
}

void freeData(tensor_t *tensor) {
    freeReservedMemory(tensor->data);
}

void freeSparsity(sparsity_t *sparsity) {
    if(sparsity != NULL) {
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
    freeSparsity(tensor->sparsity);
    freeTensorPointer(tensor);
}

void freeParameter(parameter_t *parameter) {
    freeTensor(parameter->param);
    freeTensor(parameter->grad);
    freeReservedMemory(parameter);
}
