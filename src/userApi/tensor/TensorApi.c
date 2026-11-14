#define SOURCE_FILE "TENSOR_API"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "CheckedSize.h"
#include "Common.h"
#include "Distributions.h"
#include "QuantizationApi.h"
#include "Rounding.h"
#include "ScaleCanon.h"
#include "StorageApi.h"
#include "TensorApi.h"
#include "TensorConversion.h"

/* The quantization constructors validated the widths; only the grouping's fit
 * to this element count is new here. */
static void validateGroupingFor(const quantization_t *q, size_t numberOfElements,
                                const char *what) {
    if (q->type == SYM) {
        const symQConfig_t *qC = q->qConfig;
        validateGroupShape(qC->numGroups, qC->groupSize, numberOfElements, what);
    } else if (q->type == ASYM) {
        const asymQConfig_t *qC = q->qConfig;
        validateGroupShape(qC->numGroups, qC->groupSize, numberOfElements, what);
    } else if (q->type == BFP) {
        validateBfpQConfigShape(q->qConfig, numberOfElements);
    }
}

tensor_t *initTensor(shape_t *shape, quantization_t *quantization, sparsity_t *sparsity) {
    tensor_t *tensor = reserveMemory(sizeof(tensor_t));
    tensor->shape = shape;
    tensor->quantization = quantization;
    tensor->sparsity = sparsity;

    size_t numberOfElements = calcNumberOfElementsByShape(shape);
    validateGroupingFor(quantization, numberOfElements, "initTensor");
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

    /* Borrow source as a FLOAT32 view; converters read the source and write
     * only the destination, so the const-cast is safe. */
    quantization_t floatQ;
    initFloat32Quantization(&floatQ);
    tensor_t srcView;
    srcView.data = (uint8_t *)(uintptr_t)source;
    srcView.shape = tensor->shape;
    srcView.quantization = &floatQ;
    srcView.sparsity = NULL;

    convertTensor(&srcView, tensor, NULL);
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

/* The random kinds draw only inside their per-element loops: a missing stream
 * stops before the first element, and an empty tensor never needs one. */
static void requireStream(const rngStream_t *rs, size_t n) {
    if (rs == NULL && n > 0) {
        PRINT_ERROR("initDistribution: random distribution without a stream");
        exit(1);
    }
}

void initDistribution(tensor_t *tensor, const distribution_t *distribution, rngStream_t *rs) {
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
        requireStream(rs, n);
        for (size_t i = 0; i < n; ++i) {
            vals[i] = randomUniformCtx(rs, distribution->params.uniform.min,
                                       distribution->params.uniform.max);
        }
        break;
    case NORMAL:
        requireStream(rs, n);
        for (size_t i = 0; i < n; ++i) {
            vals[i] = randomNormalCtx(rs, distribution->params.normal.mean,
                                      distribution->params.normal.stddev);
        }
        break;
    case XAVIER_UNIFORM:
        requireStream(rs, n);
        for (size_t i = 0; i < n; ++i) {
            vals[i] =
                xavierUniform(distribution->params.xavier.gain, distribution->params.xavier.fanIn,
                              distribution->params.xavier.fanOut, rs);
        }
        break;
    case XAVIER_NORMAL:
        requireStream(rs, n);
        for (size_t i = 0; i < n; ++i) {
            vals[i] =
                xavierNormal(distribution->params.xavier.gain, distribution->params.xavier.fanIn,
                             distribution->params.xavier.fanOut, rs);
        }
        break;
    case KAIMING_UNIFORM:
        requireStream(rs, n);
        for (size_t i = 0; i < n; ++i) {
            vals[i] = kaimingUniform(distribution->params.kaiming.gain,
                                     distribution->params.kaiming.fanMode, rs);
        }
        break;
    case KAIMING_NORMAL:
        requireStream(rs, n);
        for (size_t i = 0; i < n; ++i) {
            vals[i] = kaimingNormal(distribution->params.kaiming.gain,
                                    distribution->params.kaiming.fanMode, rs);
        }
        break;
    default:
        PRINT_ERROR("Unknown distribution type!");
        exit(1);
    }
}

// grad inits

tensor_t *gradInitInt32(tensor_t *param, sparsity_t *sparsity) {
    return initTensor(getShapeLike(param->shape), quantizationInitInt32(), sparsity);
}

tensor_t *gradInit(tensor_t *param, quantization_t *gradQ, sparsity_t *sparsity) {
    /* Gradient templates must be per-tensor. */
    if (gradQ->type == SYM) {
        symQConfig_t *symQC = gradQ->qConfig;
        if (symQC->numGroups > 1) {
            PRINT_ERROR("gradInit: grouped SYM grad templates are unsupported -- "
                        "grads are per-tensor only");
            exit(1);
        }
    }
    if (gradQ->type == ASYM) {
        asymQConfig_t *asymQC = gradQ->qConfig;
        if (asymQC->numGroups > 1) {
            PRINT_ERROR("gradInit: grouped ASYM grad templates are unsupported -- "
                        "grads are per-tensor only");
            exit(1);
        }
    }
    if (gradQ->type == BFP) {
        bfpQConfig_t *bfpQC = gradQ->qConfig;
        if (bfpQC->numGroups > 1) {
            PRINT_ERROR("gradInit: grouped BFP grad templates are unsupported -- "
                        "grads are per-tensor only");
            exit(1);
        }
    }
    return initTensor(getShapeLike(param->shape), getQLike(gradQ), sparsity);
}

tensor_t *gradInitFloat(tensor_t *param, sparsity_t *sparsity) {
    quantization_t *floatQ = quantizationInitFloat();
    tensor_t *grad = gradInit(param, floatQ, sparsity);
    freeQuantization(floatQ);
    return grad;
}

tensor_t *gradInitSymInt32(tensor_t *param, roundingMode_t roundingMode, sparsity_t *sparsity) {
    quantization_t *symQ = quantizationInitSymInt32WithBits(roundingMode, ODT_SYM_GRAD_QMAXBITS);
    tensor_t *grad = gradInit(param, symQ, sparsity);
    freeQuantization(symQ);
    return grad;
}

tensor_t *gradInitAsym(tensor_t *param, uint8_t qBits, roundingMode_t roundingMode,
                       sparsity_t *sparsity) {
    return initTensor(getShapeLike(param->shape), quantizationInitAsym(qBits, roundingMode),
                      sparsity);
}

tensor_t *gradInitSym(tensor_t *param, uint8_t qBits, roundingMode_t roundingMode,
                      sparsity_t *sparsity) {
    return initTensor(getShapeLike(param->shape), quantizationInitSym(qBits, roundingMode),
                      sparsity);
}

// getLike

shape_t *buildOwnedShape(const size_t *dims, size_t numberOfDims) {
    size_t dimsBytes;
    if (!checkedMulSize(numberOfDims, sizeof(size_t), &dimsBytes)) {
        PRINT_ERROR("buildOwnedShape: %zu dimensions overflow the size_t byte count", numberOfDims);
        exit(1);
    }
    size_t *ownedDims = reserveMemory(dimsBytes);
    memcpy(ownedDims, dims, dimsBytes);
    size_t *order = reserveMemory(dimsBytes);
    setOrderOfDimsForNewTensor(numberOfDims, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, ownedDims, numberOfDims, order);
    return shape;
}

shape_t *getShapeLike(shape_t *shape) {
    return buildOwnedShape(shape->dimensions, shape->numberOfDimensions);
}

/* Grouped clones preserve geometry and grid values in independent arrays.
 * Per-tensor clones preserve widths/rounding and reset the grid. */
quantization_t *getQLike(quantization_t *quantization) {
    quantization_t *likeQ = reserveMemory(sizeof(quantization_t));
    switch (quantization->type) {
    case FLOAT32:
        initFloat32Quantization(likeQ);
        break;
    case INT32:
        initInt32Quantization(likeQ);
        break;
    case SYM_INT32: {
        symInt32QConfig_t *likeSymInt32QC = reserveMemory(sizeof(symInt32QConfig_t));
        symInt32QConfig_t *symInt32QC = quantization->qConfig;
        /* Keep the source width, not the operand default. */
        initSymInt32QConfigWithQMaxBits(symInt32QC->roundingMode, likeSymInt32QC,
                                        symInt32QC->qMaxBits);
        initSymInt32Quantization(likeSymInt32QC, likeQ);
        break;
    }
    case ASYM: {
        asymQConfig_t *likeAsymQC = reserveMemory(sizeof(asymQConfig_t));
        asymQConfig_t *asymQC = quantization->qConfig;
        if (asymQC->numGroups > 1) {
            odtScale_t *likeScales = reserveMemory(asymQC->numGroups * sizeof(*likeScales));
            copyCanonScales(likeScales, asymQC->scales, asymQC->numGroups);
            uint16_t *likeZps = reserveMemory(asymQC->numGroups * sizeof(uint16_t));
            memcpy(likeZps, asymQC->zeroPoints, asymQC->numGroups * sizeof(uint16_t));
            likeAsymQC->scales = likeScales;
            likeAsymQC->zeroPoints = likeZps;
            likeAsymQC->numGroups = asymQC->numGroups;
            likeAsymQC->groupSize = asymQC->groupSize;
            likeAsymQC->roundingMode = asymQC->roundingMode;
            likeAsymQC->qBits = asymQC->qBits;
        } else {
            initAsymQConfig(asymQC->qBits, asymQC->roundingMode, likeAsymQC);
        }
        initAsymQuantization(likeAsymQC, likeQ);
        break;
    }
    case SYM: {
        symQConfig_t *likeSymQC = reserveMemory(sizeof(symQConfig_t));
        symQConfig_t *symQC = quantization->qConfig;
        if (symQC->numGroups > 1) {
            odtScale_t *likeScales = reserveMemory(symQC->numGroups * sizeof(*likeScales));
            copyCanonScales(likeScales, symQC->scales, symQC->numGroups);
            likeSymQC->scales = likeScales;
            likeSymQC->numGroups = symQC->numGroups;
            likeSymQC->groupSize = symQC->groupSize;
            likeSymQC->roundingMode = symQC->roundingMode;
            likeSymQC->qBits = symQC->qBits;
        } else {
            initSymQConfig(symQC->qBits, symQC->roundingMode, likeSymQC);
        }
        initSymQuantization(likeSymQC, likeQ);
        break;
    }
    case BFP: {
        bfpQConfig_t *likeBfpQC = reserveMemory(sizeof(bfpQConfig_t));
        bfpQConfig_t *bfpQC = quantization->qConfig;
        if (bfpQC->numGroups > 1) {
            uint8_t *likeExponents = reserveMemory(bfpQC->numGroups * sizeof(uint8_t));
            memcpy(likeExponents, bfpQC->exponents, bfpQC->numGroups * sizeof(uint8_t));
            likeBfpQC->exponents = likeExponents;
            likeBfpQC->numGroups = bfpQC->numGroups;
            likeBfpQC->groupSize = bfpQC->groupSize;
            likeBfpQC->roundingMode = bfpQC->roundingMode;
            likeBfpQC->mantissaBits = bfpQC->mantissaBits;
            likeBfpQC->exponentBits = bfpQC->exponentBits;
        } else {
            initBfpQConfig(bfpQC->mantissaBits, bfpQC->exponentBits, bfpQC->roundingMode,
                           likeBfpQC);
        }
        initBfpQuantization(likeBfpQC, likeQ);
        break;
    }
    /* BOOL has no clone consumer: a BOOL grad/state template fails here. */
    default:
        PRINT_ERROR("Unknown QType");
        exit(1);
    }
    return likeQ;
}

uint8_t *getDataLike(quantization_t *quantization, size_t numberOfValues) {
    if (quantization->type == BOOL) {
        PRINT_ERROR("getDataLike: BOOL data has no clone consumer");
        exit(1);
    }
    return reserveMemory(calcNumberOfBytesForData(quantization, numberOfValues));
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

void requantizeTensorInPlace(tensor_t *t, quantization_t *targetQ) {
    size_t numElements = calcNumberOfElementsByTensor(t);
    quantization_t *newQ = getQLike(targetQ);
    /* Validate the target grouping against this tensor's element count before
     * constructing the conversion view. */
    validateGroupingFor(newQ, numElements, "requantizeTensorInPlace");
    uint8_t *newData = getDataLike(newQ, numElements);

    tensor_t view = {.data = newData, .shape = t->shape, .quantization = newQ, .sparsity = NULL};
    convertTensor(t, &view, NULL);

    freeData(t);
    freeQuantization(t->quantization);
    t->data = view.data;
    t->quantization = view.quantization;
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
    (void)sparsity;
}

void freeShape(shape_t *shape) {
    freeReservedMemory(shape->dimensions);
    freeReservedMemory(shape->orderOfDimensions);
    freeReservedMemory(shape);
}

void freeQuantization(quantization_t *quantization) {
    /* SYM, ASYM and BFP qConfigs own their per-group arrays. */
    if (quantization->type == SYM) {
        symQConfig_t *symQC = quantization->qConfig;
        freeReservedMemory(symQC->scales);
    }
    if (quantization->type == ASYM) {
        asymQConfig_t *asymQC = quantization->qConfig;
        freeReservedMemory(asymQC->zeroPoints);
        freeReservedMemory(asymQC->scales);
    }
    if (quantization->type == BFP) {
        bfpQConfig_t *bfpQC = quantization->qConfig;
        freeReservedMemory(bfpQC->exponents);
    }
    freeReservedMemory(quantization->qConfig);
    freeReservedMemory(quantization);
}

void freeTensor(tensor_t *tensor) {
    freeData(tensor);
    freeShape(tensor->shape);
    freeQuantization(tensor->quantization);
    freeSparsity(tensor->sparsity);
    freeReservedMemory(tensor);
}

void freeParameter(parameter_t *parameter) {
    freeTensor(parameter->param);
    if (parameter->grad != NULL) {
        freeTensor(parameter->grad);
    }
    freeReservedMemory(parameter);
}
