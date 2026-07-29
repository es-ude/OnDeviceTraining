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
#include "TensorConversion.h"

tensor_t *initTensor(shape_t *shape, quantization_t *quantization, sparsity_t *sparsity) {
    tensor_t *tensor = reserveMemory(sizeof(tensor_t));
    tensor->shape = shape;
    tensor->quantization = quantization;
    tensor->sparsity = sparsity;

    size_t numberOfElements = calcNumberOfElementsByShape(shape);
    if (quantization->type == SYM) {
        validateSymQConfigShape(quantization->qConfig, numberOfElements);
    }
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
     * and convert into `tensor`. The const-cast is safe: every converter writes
     * only outputTensor->data and outputTensor->quantization->qConfig (scale /
     * zeroPoint); none touch outputTensor->shape or ->sparsity (#247). srcView
     * still needs a valid shape because the converter reads it for the element
     * count, so it aliases tensor->shape. */
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

// grad inits

tensor_t *gradInitInt32(tensor_t *param, sparsity_t *sparsity) {
    return initTensor(getShapeLike(param->shape), quantizationInitInt32(), sparsity);
}

tensor_t *gradInit(tensor_t *param, quantization_t *gradQ, sparsity_t *sparsity) {
    /* Group-quant PR2 carrier gate: groups are legal ONLY on GEMM-family
     * weight tensors -- grads stay per-tensor unconditionally. */
    if (gradQ->type == SYM) {
        symQConfig_t *symQC = gradQ->qConfig;
        if (symQC->numGroups > 1) {
            PRINT_ERROR("gradInit: grouped SYM grad templates are unsupported -- "
                        "grouped grads are a future #300 axis (spec §3)");
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
    case SYM_INT32: {
        symInt32QConfig_t *likeSymInt32QC = reserveMemory(sizeof(symInt32QConfig_t));
        symInt32QConfig_t *symInt32QC = quantization->qConfig;
        /* preserve the source width — do NOT reset to the operand default (#227) */
        initSymInt32QConfigWithQMaxBits(symInt32QC->roundingMode, likeSymInt32QC,
                                        symInt32QC->qMaxBits);
        initSymInt32Quantization(likeSymInt32QC, likeQ);
        break;
    }
    case ASYM: {
        asymQConfig_t *likeAsymQC = reserveMemory(sizeof(asymQConfig_t));
        asymQConfig_t *asymQC = quantization->qConfig;

        initAsymQConfig(asymQC->qBits, asymQC->roundingMode, likeAsymQC);
        initAsymQuantization(likeAsymQC, likeQ);
        break;
    }
    case SYM: {
        symQConfig_t *likeSymQC = reserveMemory(sizeof(symQConfig_t));
        symQConfig_t *symQC = quantization->qConfig;
        if (symQC->numGroups > 1) {
            /* Group-quant PR2: a grouped source's group SHAPE is an
             * attach-time fact, not an ungridded zero-state -- preserve
             * numGroups/groupSize and deep-copy the scales VALUES (matches
             * deepCopyQuantization's semantics, LayerQuant.c:71-82), unlike
             * the per-tensor fresh-reset clone below. */
            float *likeScales = reserveMemory(symQC->numGroups * sizeof(float));
            memcpy(likeScales, symQC->scales, symQC->numGroups * sizeof(float));
            likeSymQC->scales = likeScales;
            likeSymQC->numGroups = symQC->numGroups;
            likeSymQC->groupSize = symQC->groupSize;
            likeSymQC->roundingMode = symQC->roundingMode;
            likeSymQC->qBits = symQC->qBits;
        } else {
            /* Precedent A clone: width + rounding preserved, scale reset — a fresh
             * clone is an ungridded zero-state (first accumulate derives the grid). */
            initSymQConfig(symQC->qBits, symQC->roundingMode, likeSymQC);
        }
        initSymQuantization(likeSymQC, likeQ);
        break;
    }
    /* BOOL deliberately unsupported here: grad/state clones must fail fast at
     * construction (see UnitTestLinear BOOL-knob death test); add an arm only
     * when a real BOOL-clone consumer appears (#269 deviation). */
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
    case SYM:
        /* Packed/sub-byte payloads size via the single ceiling authority
         * (calcNumberOfBytesForData) — never re-derive the bit-packing
         * arithmetic inline (#269). */
        return reserveMemory(calcNumberOfBytesForData(quantization, numberOfValues));
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

void requantizeTensorInPlace(tensor_t *t, quantization_t *targetQ) {
    size_t numElements = calcNumberOfElementsByTensor(t);
    quantization_t *newQ = getQLike(targetQ);
    /* Group-quant PR2 final-review Fix 1 (CRITICAL, heap-OOB): this internal
     * view is built by hand (getQLike + getDataLike), bypassing initTensor's
     * validateSymQConfigShape choke point entirely -- unlike every tensor a
     * caller builds through the public API, nothing here checks that a
     * grouped SYM target's numGroups*groupSize actually equals `t`'s element
     * count. A mismatch (e.g. a 12-element source against a {numGroups=2,
     * groupSize=4} target, which implies only 8) sizes the data buffer off
     * `numElements` while the pack/unpack path indexes scales[] by group
     * (up to numElements/groupSize - 1), reading past the target's
     * numGroups-sized scales array. Validate against the SAME numElements the
     * buffer below is sized with, before either buffer exists. */
    if (newQ->type == SYM) {
        validateSymQConfigShape(newQ->qConfig, numElements);
    }
    uint8_t *newData = getDataLike(newQ, numElements);

    tensor_t view = {.data = newData, .shape = t->shape, .quantization = newQ, .sparsity = NULL};
    convertTensor(t, &view);

    freeData(t);
    freeQuantization(t->quantization);
    t->data = view.data;
    t->quantization = view.quantization;
}

// Free Functions

static void freeTensorPointer(tensor_t *tensor);

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
    /* Group-quant PR1: SYM's qConfig owns a second heap block (the scales
     * array) beyond the qConfig struct itself -- free it first, then the
     * qConfig struct, then the wrapper (reverse-init order). */
    if (quantization->type == SYM) {
        symQConfig_t *symQC = quantization->qConfig;
        freeReservedMemory(symQC->scales);
    }
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

static void freeTensorPointer(tensor_t *tensor) {
    freeReservedMemory((uint8_t *)tensor);
}
