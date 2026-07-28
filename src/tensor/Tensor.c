#define SOURCE_FILE "TENSOR"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "Common.h"
#include "DTypes.h"
#include "MinMax.h"
#include "Quantization.h"
#include "Tensor.h"

size_t calcNumberOfElementsByShape(shape_t *shape) {
    size_t numElem = 1;
    size_t numberOfDimensions = shape->numberOfDimensions;
    size_t *dimensions = shape->dimensions;
    for (size_t i = 0; i < numberOfDimensions; i++) {
        numElem *= dimensions[i];
    }
    return numElem;
}

size_t calcNumberOfElementsByTensor(tensor_t *tensor) {
    return calcNumberOfElementsByShape(tensor->shape);
}

size_t calcNumberOfElementsByParameter(parameter_t *parameter) {
    return calcNumberOfElementsByShape(parameter->param->shape);
}

size_t calcBytesPerElement(quantization_t *quantization) {
    switch (quantization->type) {
    case INT32:
        return sizeof(int32_t);
    case FLOAT32:
        return sizeof(float);
    case SYM_INT32:
        return sizeof(int32_t);
    case SYM: {
        symQConfig_t *symQC = quantization->qConfig;
        return (size_t)ceilf((float)symQC->qBits / 8.0f);
    }
    case ASYM: {
        asymQConfig_t *asymQConfig = quantization->qConfig;
        uint32_t qBits = asymQConfig->qBits;
        return ceil((float)qBits / (float)8);
    }
    case BOOL:
        return 1;
    default:
        PRINT_ERROR("Unknown QType!");
        exit(1);
    }
}

size_t calcBitsPerElement(quantization_t *quantization) {
    switch (quantization->type) {
    case INT32:
        return sizeof(int32_t) * 8;
    case FLOAT32:
        return sizeof(float) * 8;
    case SYM_INT32:
        return sizeof(int32_t) * 8;
    case SYM: {
        symQConfig_t *symQC = quantization->qConfig;
        return symQC->qBits;
    }
    case ASYM: {
        asymQConfig_t *asymQConfig = quantization->qConfig;
        return asymQConfig->qBits;
    }
    case BOOL:
        return 1;
    default:
        PRINT_ERROR("Unknown QType!");
        exit(1);
    }
}

size_t calcBytesPerTensor(tensor_t *tensor) {
    size_t numberOfElements = calcNumberOfElementsByShape(tensor->shape);
    return calcNumberOfBytesForData(tensor->quantization, numberOfElements);
}

size_t calcNumberOfBytesForData(quantization_t *q, size_t numberOfElements) {
    switch (q->type) {
    case FLOAT32:
        return numberOfElements * sizeof(float);
    case INT32:
        return numberOfElements * sizeof(int32_t);
    case SYM_INT32:
        return numberOfElements * sizeof(int32_t);
    case SYM:
        return (calcBitsPerElement(q) * numberOfElements + 7) / 8;
    case ASYM: {
        size_t bitsPerElement = calcBitsPerElement(q);
        return (bitsPerElement * numberOfElements + 7) / 8;
    }
    case BOOL:
        return (numberOfElements + 7) / 8;
    default:
        PRINT_ERROR("Unknown QType!");
        exit(1);
    }
}

bool tensorBoolGet(tensor_t const *tensor, size_t flatIndex) {
    if (tensor->quantization->type != BOOL) {
        PRINT_ERROR("tensorBoolGet called on non-BOOL tensor");
        exit(1);
    }
    size_t byteIndex = flatIndex / 8;
    size_t bitIndex = flatIndex % 8;
    return ((tensor->data[byteIndex] >> bitIndex) & 1u) != 0;
}

void tensorBoolSet(tensor_t *tensor, size_t flatIndex, bool value) {
    if (tensor->quantization->type != BOOL) {
        PRINT_ERROR("tensorBoolSet called on non-BOOL tensor");
        exit(1);
    }
    size_t byteIndex = flatIndex / 8;
    size_t bitIndex = flatIndex % 8;
    if (value) {
        tensor->data[byteIndex] |= (uint8_t)(1u << bitIndex);
    } else {
        tensor->data[byteIndex] &= (uint8_t)~(1u << bitIndex);
    }
}

void setOrderOfDimsForNewTensor(size_t numberOfDimensions, size_t *orderOfDimensions) {
    for (size_t i = 0; i < numberOfDimensions; i++) {
        orderOfDimensions[i] = i;
    }
}

void print_binary_uint8(uint8_t x) {
    /* Show the most‑significant bit first */
    printf("Byte ");
    for (int i = 7; i >= 0; --i) {
        putchar((x >> i) & 1 ? '1' : '0');
    }
    putchar('\n'); /* newline for convenience */
}

uint32_t getBitmask(uint32_t startbit, uint32_t endbit) {
    uint32_t endbitInternal = endbit - (startbit / 8) * 8;
    uint32_t startbitInternal = startbit - (startbit / 8) * 8;
    uint32_t counter = 0;
    uint32_t value = 1;
    for (size_t i = 0; i < 8; i++) {
        if ((i >= startbitInternal) & (endbitInternal > i)) {
            counter += value;
        }
        value *= 2;
    }
    // printf("bitmask ");
    // print_binary_uint8(counter);
    return counter;
}

uint8_t readByte(uint8_t data, uint8_t startbit, uint8_t endbit) {
    uint8_t bitmask = getBitmask(startbit, endbit);
    uint8_t intermediate = data & bitmask;
    intermediate >>= startbit - (startbit / 8) * 8;
    return intermediate;
}

uint8_t writeByte(uint8_t existingData, uint8_t data, uint8_t startbit, uint8_t endbit) {
    uint8_t startbitInternal = startbit - (startbit / 8) * 8;
    uint8_t endbitInternal = endbit - (startbit / 8) * 8;
    uint8_t bitmask = getBitmask(startbitInternal, endbitInternal);
    data <<= startbitInternal;
    uint8_t intermediate = data & bitmask;
    /* Clear-then-set: the [startbit, endbit) range is fully defined by this
     * write, so callers may target buffers with stale in-range bits (bit-
     * offset appends); bits outside the mask are preserved. */
    existingData = (existingData & (uint8_t)~bitmask) | intermediate;
    return existingData;
}

int max(int a, int b) {
    return (a > b) ? a : b;
}

int min(int a, int b) {
    return (a < b) ? a : b;
}

void byteConversion(uint8_t *dataIn, size_t dataInBits, uint8_t *dataOut, size_t dataOutBits,
                    size_t numValues) {
    if (numValues == 0) {
        /* Skip the memset: N=0 tensor data may be NULL (#160). */
        return;
    }
    /* Ceiling idiom (bits+7)/8 as in calcNumberOfBytesForData: identical to
     * the previous (bits-1)/8+1 for bits > 0, but safely 0 instead of a
     * size_t underflow for dataOutBits == 0. memset also zeroes the trailing
     * pad bits of the last byte, which the append loop leaves untouched. */
    memset(dataOut, 0, (numValues * dataOutBits + 7) / 8);
    byteConversionAppend(dataIn, dataInBits, dataOut, dataOutBits, numValues, 0, 0);
}

void byteConversionAppend(uint8_t *dataIn, size_t dataInBits, uint8_t *dataOut, size_t dataOutBits,
                          size_t numValues, size_t dstStartBit, size_t srcStartBit) {
    size_t dataOutIndex = dstStartBit / 8;
    size_t dataInIndex = srcStartBit / 8;
    int dataOutStartbit = (int)(dstStartBit % 8);
    int dataInStartbit = (int)(srcStartBit % 8);
    int dataInEndbit = dataInStartbit + (int)dataInBits;
    int dataOutEndbit = dataOutStartbit + (int)dataOutBits;
    for (size_t i = 0; i < numValues; i++) {
        while ((dataInStartbit < dataInEndbit) | (dataOutStartbit < dataOutEndbit)) {
            /* Guard each side: input may exhaust before output (widening) or
             * output may fill before input (narrowing); skipping the
             * out-of-range access avoids OOB while preserving zero-fill
             * semantics. */
            uint8_t data = 0;
            if (dataInStartbit < dataInEndbit) {
                data = readByte(dataIn[dataInIndex], dataInStartbit, dataInEndbit);
            }
            if (dataOutStartbit < dataOutEndbit) {
                dataOut[dataOutIndex] =
                    writeByte(dataOut[dataOutIndex], data, dataOutStartbit, dataOutEndbit);
            }

            int valuesRead = min(dataInEndbit - dataInStartbit, 8 - dataInStartbit % 8);
            int valuesWritten = min(dataOutEndbit - dataOutStartbit, 8 - dataOutStartbit % 8);
            int minValue = min(valuesRead, valuesWritten);

            uint8_t deltaIn = minValue;
            uint8_t deltaOut = minValue;
            if (dataInStartbit == dataInEndbit) {
                dataOutStartbit += valuesWritten;
                deltaOut = valuesWritten;
            } else {
                dataOutStartbit += minValue;
            }
            if (dataOutStartbit == dataOutEndbit) {
                dataInStartbit += valuesRead;
                deltaIn = valuesRead;
            } else {
                dataInStartbit += minValue;
            }

            if (dataInStartbit / 8 > (dataInStartbit - deltaIn) / 8) {
                dataInIndex += 1;
            }
            if (dataOutStartbit / 8 > (dataOutStartbit - deltaOut) / 8) {
                dataOutIndex += 1;
            }
        }
        dataInStartbit = dataInEndbit % 8;
        dataInEndbit = dataInStartbit + (int)dataInBits;
        dataOutStartbit = dataOutEndbit % 8;
        dataOutEndbit = dataOutStartbit + (int)dataOutBits;
    }
}

tensor_t *getParamFromParameter(parameter_t *parameter) {
    return parameter->param;
}

tensor_t *getGradFromParameter(parameter_t *parameter) {
    return parameter->grad;
}

void transposeTensor(tensor_t *tensor, size_t dim0Index, size_t dim1Index) {
    if (tensor->shape->numberOfDimensions < 2) {
        return;
    }
    size_t temp = tensor->shape->orderOfDimensions[dim0Index];
    tensor->shape->orderOfDimensions[dim0Index] = tensor->shape->orderOfDimensions[dim1Index];
    tensor->shape->orderOfDimensions[dim1Index] = temp;
}

void setTensorValuesForConversion(uint8_t *data, quantization_t *q, tensor_t *originalTensor,
                                  tensor_t *outputTensor) {
    outputTensor->data = data;
    outputTensor->shape = originalTensor->shape;
    outputTensor->quantization = q;
    outputTensor->sparsity = originalTensor->sparsity;
}

void setTensorValues(tensor_t *tensor, uint8_t *data, shape_t *shape, quantization_t *quantization,
                     sparsity_t *sparsity) {
    tensor->data = data;
    tensor->shape = shape;
    tensor->quantization = quantization;
    tensor->sparsity = sparsity;
}

void setParameterValues(parameter_t *parameter, tensor_t *param, tensor_t *grad) {
    parameter->param = param;
    parameter->grad = grad;
}

void setShape(shape_t *shape, size_t *dims, size_t numberOfDims, size_t *orderOfDims) {
    shape->dimensions = dims;
    shape->numberOfDimensions = numberOfDims;
    shape->orderOfDimensions = orderOfDims;
}

void printTensor(tensor_t *t) {
    quantization_t *q = t->quantization;
    printf("TENSOR BEGIN \n");
    size_t numValues = calcNumberOfElementsByTensor(t);
    int32_t data[numValues];

    switch (q->type) {
    case INT32:
        printf("INT32Q \n");
        readBytesAsInt32Array(numValues, t->data, data);
        for (size_t i = 0; i < numValues; i++) {
            printf("%i\n", data[i]);
        }
        break;
    case FLOAT32:
        printf("FLOAT32Q \n");
        for (size_t i = 0; i < numValues; i++) {
            size_t byteIndex = i * sizeof(float);
            float currentElement = readBytesAsFloat(&t->data[byteIndex]);
            printf("%f\n", currentElement);
        }
        break;
    case SYM_INT32: {
        symInt32QConfig_t *symQC = q->qConfig;
        printf("SYM_INT32 \n");
        printf("scale=%e\n", symQC->scale);
        printf("Data \n");
        for (size_t i = 0; i < numValues; i++) {
            size_t byteIndex = i * sizeof(int32_t);
            int32_t currentElement = readBytesAsInt32(&t->data[byteIndex]);
            printf("%i\n", currentElement);
        }
        break;
    }
    case ASYM: {
        asymQConfig_t *lq = q->qConfig;
        printf("ASYM\n");
        printf("scale=%e\n", lq->scale);
        printf("offset=%i\n", lq->zeroPoint);
        printf("Data \n");
        for (size_t i = 0; i < numValues; i++) {
            printf("%i\n", t->data[i]);
        }
        break;
    }
    case BOOL:
        printf("BOOL\n");
        printf("[");
        for (size_t i = 0; i < numValues; i++) {
            printf("%d", tensorBoolGet(t, i) ? 1 : 0);
            if (i + 1 < numValues) {
                printf(", ");
            }
        }
        printf("]\n");
        break;
    default:
        printf("WTF\n");
    }

    printf("TENSOR END \n");
    printf("_____________________\n");
}

void printShape(shape_t *shape) {
    size_t numberOfDims = shape->numberOfDimensions;

    printf("NumberOfDims: %lu\n", numberOfDims);

    printf("Dims: \n");
    for (size_t i = 0; i < numberOfDims; i++) {
        printf("%lu\n", shape->dimensions[i]);
    }

    printf("OrderOfDims: \n");
    for (size_t i = 0; i < numberOfDims; i++) {
        printf("%lu\n", shape->orderOfDimensions[i]);
    }
    printf("__________\n");
}

void initOrderOfDimensions(size_t *orderOfDims, size_t numberOfDims) {
    for (size_t i = 0; i < numberOfDims; i++) {
        orderOfDims[i] = i;
    }
}

void copyData(tensor_t *dest, tensor_t *src) {
    size_t numberOfValues = calcNumberOfElementsByShape(dest->shape);
    size_t sizeData = calcNumberOfBytesForData(src->quantization, numberOfValues);

    memcpy(dest->data, src->data, sizeData);

    if (src->sparsity != NULL) {
        memcpy(dest->sparsity, src->sparsity, sizeof(sparsity_t));
    }
}

void copyShape(shape_t *dest, shape_t *src) {
    memcpy(dest->dimensions, src->dimensions, src->numberOfDimensions * sizeof(size_t));
    memcpy(dest->orderOfDimensions, src->orderOfDimensions,
           src->numberOfDimensions * sizeof(size_t));
    dest->numberOfDimensions = src->numberOfDimensions;
}

/* Clones src's quantization into dest. Config-carrying dtypes (SYM_INT32/SYM/ASYM)
 * copy INTO dest->qConfig -- the caller must have pre-allocated it for the same
 * dtype (src/tensor never allocates). NULL-config dtypes (INT32/FLOAT32/BOOL)
 * overwrite dest->qConfig with NULL; a heap config dest owned beforehand stays
 * owned by the caller. */
static void copyQConfigInto(quantization_t *dest, quantization_t *src, size_t qConfigSize,
                            const char *typeName) {
    if (dest->qConfig == NULL) {
        PRINT_ERROR("copyQuantization: dest->qConfig for %s must be caller-allocated", typeName);
        exit(1);
    }
    dest->type = src->type;
    memcpy(dest->qConfig, src->qConfig, qConfigSize);
}

/* SYM's qConfig carries a heap `scales` pointer (group-quant PR1) -- a byte
 * memcpy of the struct would overwrite dest->qConfig->scales with src's
 * pointer, aliasing the two configs and leaking dest's own array. Copy the
 * VALUES into dest's own (caller-allocated, already-owned) array instead,
 * same contract as copyQConfigInto otherwise: dest->qConfig must be
 * caller-allocated, dest keeps owning its own scales block. PR1 invariant
 * (numGroups == 1 for both sides) makes a 1-for-1 value copy exact; a
 * mismatched numGroups here would indicate a caller contract violation, not
 * something PR1's per-tensor-only paths can produce. */
static void copySymQConfigInto(quantization_t *dest, quantization_t *src) {
    if (dest->qConfig == NULL) {
        PRINT_ERROR("copyQuantization: dest->qConfig for SYM must be caller-allocated");
        exit(1);
    }
    dest->type = src->type;
    symQConfig_t *srcQC = src->qConfig;
    symQConfig_t *dstQC = dest->qConfig;
    if (dstQC->numGroups != srcQC->numGroups) {
        PRINT_ERROR("copyQuantization: SYM numGroups mismatch (dest %zu, src %zu)",
                    dstQC->numGroups, srcQC->numGroups);
        exit(1);
    }
    memcpy(dstQC->scales, srcQC->scales, srcQC->numGroups * sizeof(float));
    dstQC->groupSize = srcQC->groupSize;
    dstQC->roundingMode = srcQC->roundingMode;
    dstQC->qBits = srcQC->qBits;
}

void copyQuantization(quantization_t *dest, quantization_t *src) {
    switch (src->type) {
    case INT32:
        dest->type = INT32;
        dest->qConfig = NULL;
        break;
    case FLOAT32:
        dest->type = FLOAT32;
        dest->qConfig = NULL;
        break;
    case SYM_INT32:
        copyQConfigInto(dest, src, sizeof(symInt32QConfig_t), "SYM_INT32");
        break;
    case SYM:
        copySymQConfigInto(dest, src);
        break;
    case ASYM:
        copyQConfigInto(dest, src, sizeof(asymQConfig_t), "ASYM");
        break;
    case BOOL:
        dest->type = BOOL;
        dest->qConfig = NULL;
        break;
    default:
        PRINT_ERROR("Unknown QType!");
        exit(1);
    }
}

// TODO copy sparsity
void copyTensor(tensor_t *dest, tensor_t *src) {
    copyShape(dest->shape, src->shape);
    copyQuantization(dest->quantization, src->quantization);
    copyData(dest, src);
}
