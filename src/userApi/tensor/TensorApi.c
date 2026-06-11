#define SOURCE_FILE "TENSOR_API"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "Common.h"
#include "Distributions.h"
#include "QuantizationApi.h"
#include "Rounding.h"
#include "StorageApi.h"
#include "TensorApi.h"
#include "TensorApiInternal.h"
#include "TensorConversion.h"

// tensor inits

tensor_t *tensorInitInt32(int32_t *data, size_t *dims, size_t numberOfDims, sparsity_t *sparsity) {
    quantization_t *q = reserveMemory(sizeof(quantization_t));
    initInt32Quantization(q);

    return initTensorWithQInt32(data, dims, numberOfDims, q, sparsity);
}

tensor_t *tensorInitFloat(float *data, size_t *dims, size_t numberOfDims, sparsity_t *sparsity) {
    quantization_t *q = reserveMemory(sizeof(quantization_t));
    initFloat32Quantization(q);

    return initTensorWithQFloat(data, dims, numberOfDims, q, sparsity);
}

tensor_t *tensorInitSymInt32(float *data, size_t *dims, size_t numberOfDims,
                             roundingMode_t roundingMode, sparsity_t *sparsity) {
    quantization_t *symInt32Q = reserveMemory(sizeof(quantization_t));
    symInt32QConfig_t *symInt32QC = reserveMemory(sizeof(symInt32QConfig_t));
    initSymInt32QConfig(roundingMode, symInt32QC);
    initSymInt32Quantization(symInt32QC, symInt32Q);

    return initTensorWithQSymInt32(data, dims, numberOfDims, symInt32Q, sparsity);
}

tensor_t *tensorInitAsym(float *data, size_t *dims, size_t numberOfDims, uint8_t qBits,
                         roundingMode_t roundingMode, sparsity_t *sparsity) {
    asymQConfig_t *asymQC = reserveMemory(sizeof(asymQConfig_t));
    asymQC->qBits = qBits;
    asymQC->roundingMode = roundingMode;
    quantization_t *asymQ = reserveMemory(sizeof(quantization_t));
    asymQ->type = ASYM;
    asymQ->qConfig = asymQC;

    return initTensorWithQAsym(data, dims, numberOfDims, asymQ, sparsity);
}

tensor_t *tensorInit(float *data, size_t *dims, size_t numberOfDims, quantization_t *quantization,
                     sparsity_t *sparsity) {
    switch (quantization->type) {
    case FLOAT32:
        return initTensorWithQFloat(data, dims, numberOfDims, quantization, sparsity);
    case INT32:
        size_t size = 0;
        for (size_t i = 0; i < numberOfDims; i++) {
            size += dims[i];
        }
        int32_t *dataInt = reserveMemory(size * sizeof(int32_t));
        for (size_t i = 0; i < size; i++) {
            dataInt[i] = (int32_t)data[i];
        }
        return initTensorWithQInt32(dataInt, dims, numberOfDims, quantization, sparsity);
    case SYM_INT32:
        return initTensorWithQSymInt32(data, dims, numberOfDims, quantization, sparsity);
    case ASYM:
        return initTensorWithQAsym(data, dims, numberOfDims, quantization, sparsity);
    default:
        PRINT_ERROR("Unknown QType");
        exit(1);
    }
}

tensor_t *initTensor(shape_t *shape, quantization_t *quantization, sparsity_t *sparsity) {
    tensor_t *tensor = reserveMemory(sizeof(tensor_t));
    tensor->shape = shape;
    tensor->quantization = quantization;
    tensor->sparsity = sparsity;

    size_t numberOfElements = calcNumberOfElementsByShape(shape);
    size_t bytes = calcNumberOfBytesForData(quantization, numberOfElements);
    tensor->data = reserveMemory(bytes);

    return tensor;
}

void tensorFillFromFloatBuffer(tensor_t *tensor, const float *source, size_t count) {
    size_t expected = calcNumberOfElementsByTensor(tensor);
    if (count != expected) {
        PRINT_ERROR("tensorFillFromFloatBuffer count mismatch (expected vs given)");
        exit(1);
    }

    if (tensor->quantization->type == BOOL) {
        PRINT_ERROR("tensorFillFromFloatBuffer does not support BOOL tensors; "
                    "use tensorFillFromBoolBuffer instead");
        exit(1);
    }

    if (tensor->quantization->type == FLOAT32) {
        memcpy(tensor->data, source, count * sizeof(float));
        return;
    }

    /* Non-FLOAT32: route through convertTensor, mirroring the pattern in
     * initTensorWithQSymInt32. Build a temporary FLOAT32 view over `source`
     * and convert into `tensor`. The const-cast is safe per converter:
     *   - SYM_INT32: convertFloatTensorToSymInt32Tensor writes only to
     *     outputTensor->data and outputTensor->quantization->qConfig->scale.
     *   - INT32 / ASYM: convertFloatTensorTo{Int32,Asym}Tensor end with
     *     copyDimsAndSparsityToTensor(input, output), which writes
     *     outputTensor->shape = inputTensor->shape. Since we set
     *     srcView.shape = tensor->shape, that write self-assigns the same
     *     pointer back into tensor — harmless. srcView.sparsity is NULL so
     *     the sparsity memcpy branch is skipped. */
    quantization_t floatQ;
    initFloat32Quantization(&floatQ);
    tensor_t srcView;
    srcView.data = (uint8_t *)(uintptr_t)source;
    srcView.shape = tensor->shape;
    srcView.quantization = &floatQ;
    srcView.sparsity = NULL;

    convertTensor(&srcView, tensor);
}

void tensorFillFromBoolBuffer(tensor_t *tensor, const bool *source, size_t count) {
    size_t expected = calcNumberOfElementsByTensor(tensor);
    if (count != expected) {
        PRINT_ERROR("tensorFillFromBoolBuffer count mismatch (expected vs given)");
        exit(1);
    }
    if (tensor->quantization->type != BOOL) {
        PRINT_ERROR("tensorFillFromBoolBuffer requires BOOL-quantized tensor");
        exit(1);
    }
    for (size_t i = 0; i < count; i++) {
        tensorBoolSet(tensor, i, source[i]);
    }
}

void initDistribution(tensor_t *tensor, const distribution_t *distribution) {
    if (tensor->quantization->type != FLOAT32) {
        PRINT_ERROR("initDistribution only supports FLOAT32 in this iteration");
        exit(1);
    }
    float *vals = (float *)tensor->data;
    size_t n = calcNumberOfElementsByTensor(tensor);

    switch (distribution->type) {
    case ZEROS:
        memset(vals, 0, n * sizeof(float));
        break;
    case ONES:
        for (size_t i = 0; i < n; ++i) {
            vals[i] = 1.0f;
        }
        break;
    case UNIFORM:
        for (size_t i = 0; i < n; ++i) {
            vals[i] =
                randomUniform(distribution->params.uniform.min, distribution->params.uniform.max);
        }
        break;
    case NORMAL:
        for (size_t i = 0; i < n; ++i) {
            vals[i] =
                randomNormal(distribution->params.normal.mean, distribution->params.normal.stddev);
        }
        break;
    case XAVIER_UNIFORM:
        for (size_t i = 0; i < n; ++i) {
            vals[i] =
                xavierUniform(distribution->params.xavier.gain, distribution->params.xavier.fanIn,
                              distribution->params.xavier.fanOut);
        }
        break;
    case XAVIER_NORMAL:
        for (size_t i = 0; i < n; ++i) {
            vals[i] =
                xavierNormal(distribution->params.xavier.gain, distribution->params.xavier.fanIn,
                             distribution->params.xavier.fanOut);
        }
        break;
    case KAIMING_UNIFORM:
        for (size_t i = 0; i < n; ++i) {
            vals[i] = kaimingUniform(distribution->params.kaiming.gain,
                                     distribution->params.kaiming.fanMode);
        }
        break;
    case KAIMING_NORMAL:
        for (size_t i = 0; i < n; ++i) {
            vals[i] = kaimingNormal(distribution->params.kaiming.gain,
                                    distribution->params.kaiming.fanMode);
        }
        break;
    default:
        PRINT_ERROR("Unknown distribution type!");
        exit(1);
    }
}

tensor_t *tensorInitWithDistribution(distributionType_t distributionType, float *data, size_t *dims,
                                     size_t numberOfDims, quantization_t *quantization,
                                     sparsity_t *sparsity, size_t inputFeatures,
                                     size_t outputFeatures) {
    size_t numberOfValues = 1;
    for (size_t i = 0; i < numberOfDims; i++) {
        numberOfValues *= dims[i];
    }

    switch (distributionType) {
    case ZEROS:
        memset(data, 0, numberOfValues * sizeof(float));
        break;
    case ONES:
        for (size_t i = 0; i < numberOfValues; i++) {
            data[i] = 1.0f;
        }
        break;
    case NORMAL:
        for (size_t i = 0; i < numberOfValues; i++) {
            data[i] = randomNormal(0.0f, 0.01f);
        }
        break;
    case UNIFORM:
        for (size_t i = 0; i < numberOfValues; i++) {
            data[i] = randomUniform(-0.1f, 0.1f);
        }
        break;
    case XAVIER_NORMAL:
        for (size_t i = 0; i < numberOfValues; i++) {
            data[i] = xavierNormal(1.0f, inputFeatures, outputFeatures);
        }
        break;
    case XAVIER_UNIFORM:
        for (size_t i = 0; i < numberOfValues; i++) {
            data[i] = xavierUniform(1.0f, inputFeatures, outputFeatures);
        }
        break;
    case KAIMING_NORMAL:
        for (size_t i = 0; i < numberOfValues; i++) {
            data[i] = kaimingNormal(sqrtf(2.0f), inputFeatures);
        }
        break;
    case KAIMING_UNIFORM:
        for (size_t i = 0; i < numberOfValues; i++) {
            data[i] = kaimingUniform(sqrtf(2.0f), inputFeatures);
        }
        break;
    default:
        PRINT_ERROR("Unknown distribution type!");
        exit(1);
    }

    tensor_t *tensor = tensorInit(data, dims, numberOfDims, quantization, sparsity);

    return tensor;
}

// grad inits

tensor_t *gradInitInt32(tensor_t *param, sparsity_t *sparsity) {
    return initTensor(getShapeLike(param->shape), quantizationInitInt32(), sparsity);
}

tensor_t *gradInit(tensor_t *param, quantization_t *gradQ, sparsity_t *sparsity) {
    return initTensor(getShapeLike(param->shape), getQLike(gradQ), sparsity);
}

tensor_t *gradInitFloat(tensor_t *param, sparsity_t *sparsity) {
    quantization_t *floatQ = quantizationInitFloat();
    tensor_t *grad = gradInit(param, floatQ, sparsity);
    freeQuantization(floatQ);
    return grad;
}

tensor_t *gradInitSymInt32(tensor_t *param, roundingMode_t roundingMode, sparsity_t *sparsity) {
    quantization_t *symQ = quantizationInitSymInt32(roundingMode);
    tensor_t *grad = gradInit(param, symQ, sparsity);
    freeQuantization(symQ);
    return grad;
}

tensor_t *gradInitAsym(tensor_t *param, uint8_t qBits, roundingMode_t roundingMode,
                       sparsity_t *sparsity) {
    return initTensor(getShapeLike(param->shape), quantizationInitAsym(qBits, roundingMode),
                      sparsity);
}

// getLike

shape_t *getShapeLike(shape_t *shape) {
    shape_t *likeShape = reserveMemory(sizeof(shape_t));

    size_t numberOfDims = shape->numberOfDimensions;

    size_t *likeDims = reserveMemory(numberOfDims * sizeof(size_t));
    memcpy(likeDims, shape->dimensions, numberOfDims * sizeof(size_t));

    size_t *likeOrder = reserveMemory(numberOfDims * sizeof(size_t));
    setOrderOfDimsForNewTensor(numberOfDims, likeOrder);

    setShape(likeShape, likeDims, numberOfDims, likeOrder);

    return likeShape;
}

quantization_t *getQLike(quantization_t *quantization) {
    quantization_t *likeQ = reserveMemory(sizeof(quantization_t));
    switch (quantization->type) {
    case FLOAT32:
        initFloat32Quantization(likeQ);
        break;
    case INT32:
        initInt32Quantization(likeQ);
        break;
    case SYM_INT32:
        symInt32QConfig_t *likeSymInt32QC = reserveMemory(sizeof(symInt32QConfig_t));
        symInt32QConfig_t *symInt32QC = quantization->qConfig;

        initSymInt32QConfig(symInt32QC->roundingMode, likeSymInt32QC);
        initSymInt32Quantization(likeSymInt32QC, likeQ);
        break;
    case ASYM:
        asymQConfig_t *likeAsymQC = reserveMemory(sizeof(asymQConfig_t));
        asymQConfig_t *asymQC = quantization->qConfig;

        initAsymQConfig(asymQC->qBits, asymQC->roundingMode, likeAsymQC);
        initAsymQuantization(likeAsymQC, likeQ);
        break;
    default:
        PRINT_ERROR("Unknown QType");
        exit(1);
    }
    return likeQ;
}

uint8_t *getDataLike(quantization_t *quantization, size_t numberOfValues) {
    switch (quantization->type) {
    case FLOAT32:
        return reserveMemory(numberOfValues * sizeof(float));
    case INT32:
        return reserveMemory(numberOfValues * sizeof(int32_t));
    case SYM_INT32:
        return reserveMemory(numberOfValues * sizeof(int32_t));
    case ASYM:
        asymQConfig_t *asymQC = quantization->qConfig;
        size_t totalBits = numberOfValues * asymQC->qBits;
        size_t totalBytes = (totalBits + 7) / 8;
        return reserveMemory(totalBytes);
    default:
        PRINT_ERROR("Unknown QType");
        exit(1);
    }
}

sparsity_t *getSparsityLike(sparsity_t *sparsity) {
    if (sparsity != NULL) {
        return reserveMemory(sizeof(sparsity_t));
    }
    return NULL;
}

tensor_t *getTensorLike(tensor_t *tensor) {
    tensor_t *likeTensor = reserveMemory(sizeof(tensor_t));
    size_t numberOfValues = calcNumberOfElementsByShape(tensor->shape);
    likeTensor->data = getDataLike(tensor->quantization, numberOfValues);
    likeTensor->quantization = getQLike(tensor->quantization);
    likeTensor->shape = getShapeLike(tensor->shape);
    likeTensor->sparsity = getSparsityLike(tensor->sparsity);

    return likeTensor;
}

// Free Functions

parameter_t *parameterInit(tensor_t *param, tensor_t *grad) {
    parameter_t *parameter = reserveMemory(sizeof(parameter_t));
    parameter->param = param;
    parameter->grad = grad;

    return parameter;
}

void freeData(tensor_t *tensor) {
    freeReservedMemory(tensor->data);
}

void freeSparsity(sparsity_t *sparsity) {
    if (sparsity != NULL) {}
}

void freeShape(shape_t *shape) {
    freeReservedMemory(shape->dimensions);
    freeReservedMemory(shape->orderOfDimensions);
    freeReservedMemory(shape);
}

void freeQuantization(quantization_t *quantization) {
    freeReservedMemory(quantization->qConfig);
    freeReservedMemory(quantization);
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
    if (parameter->grad != NULL) {
        freeTensor(parameter->grad);
    }
    freeReservedMemory(parameter);
}

static tensor_t *initTensorWithQInt32(int32_t *data, size_t *dims, size_t numberOfDims,
                                      quantization_t *quantization, sparsity_t *sparsity) {
    tensor_t *tensor = malloc(sizeof(tensor_t));
    tensor->data = (uint8_t *)data;
    shape_t *shape = reserveMemory(sizeof(shape_t));
    size_t *order = reserveMemory(numberOfDims * sizeof(size_t));
    setOrderOfDimsForNewTensor(numberOfDims, order);
    setShape(shape, dims, numberOfDims, order);
    tensor->shape = shape;
    tensor->quantization = quantization;
    tensor->sparsity = sparsity;

    return tensor;
}

static tensor_t *initTensorWithQFloat(float *data, size_t *dims, size_t numberOfDims,
                                      quantization_t *quantization, sparsity_t *sparsity) {
    tensor_t *tensor = reserveMemory(sizeof(tensor_t));

    tensor->data = (uint8_t *)data;

    shape_t *shape = reserveMemory(sizeof(shape_t));
    size_t *order = reserveMemory(numberOfDims * sizeof(size_t));
    setOrderOfDimsForNewTensor(numberOfDims, order);
    setShape(shape, dims, numberOfDims, order);
    tensor->shape = shape;
    tensor->quantization = quantization;
    tensor->sparsity = sparsity;

    return tensor;
}

static tensor_t *initTensorWithQSymInt32(float *data, size_t *dims, size_t numberOfDims,
                                         quantization_t *quantization, sparsity_t *sparsity) {

    shape_t *shape = reserveMemory(sizeof(shape_t));
    size_t *order = reserveMemory(numberOfDims * sizeof(size_t));
    setOrderOfDimsForNewTensor(numberOfDims, order);
    setShape(shape, dims, numberOfDims, order);

    tensor_t floatTensor;
    quantization_t floatQ;
    initFloat32Quantization(&floatQ);

    floatTensor.data = (uint8_t *)data;
    floatTensor.shape = shape;
    floatTensor.quantization = &floatQ;
    floatTensor.sparsity = sparsity;

    tensor_t *symInt32Tensor = reserveMemory(sizeof(tensor_t));

    size_t numberOfValues = calcNumberOfElementsByTensor(&floatTensor);
    int32_t *symInt32Data = reserveMemory(numberOfValues * sizeof(int32_t));

    symInt32Tensor->data = (uint8_t *)symInt32Data;
    symInt32Tensor->shape = shape;
    symInt32Tensor->quantization = quantization;
    convertTensor(&floatTensor, symInt32Tensor);
    symInt32Tensor->sparsity = sparsity;

    return symInt32Tensor;
}

static tensor_t *initTensorWithQAsym(float *data, size_t *dims, size_t numberOfDims,
                                     quantization_t *quantization, sparsity_t *sparsity) {

    shape_t *shape = reserveMemory(sizeof(shape_t));
    size_t *order = reserveMemory(numberOfDims * sizeof(size_t));
    setOrderOfDimsForNewTensor(numberOfDims, order);
    setShape(shape, dims, numberOfDims, order);

    tensor_t floatTensor;
    quantization_t floatQ;
    initFloat32Quantization(&floatQ);

    floatTensor.data = (uint8_t *)data;
    floatTensor.shape = shape;
    floatTensor.quantization = &floatQ;
    floatTensor.sparsity = sparsity;

    tensor_t *asymTensor = reserveMemory(sizeof(tensor_t));

    asymQConfig_t *asymQC = quantization->qConfig;
    size_t bitsPerElement = asymQC->qBits;
    size_t numberOfValues = calcNumberOfElementsByShape(shape);
    size_t sizeData = ceilf((float)(numberOfValues * bitsPerElement / 8));
    uint8_t *asymData = reserveMemory(sizeData);

    asymTensor->data = asymData;
    asymTensor->shape = shape;
    asymTensor->quantization = quantization;
    asymTensor->sparsity = sparsity;

    convertTensor(&floatTensor, asymTensor);

    return asymTensor;
}

static void freeTensorPointer(tensor_t *tensor) {
    freeReservedMemory((uint8_t *)tensor);
}
