#define SOURCE_FILE "ARITHMETIC"

#include <stdlib.h>

#include "Arithmetic.h"
#include "Common.h"
#include "DTypes.h"

size_t getDimensionsByIndex(tensor_t *tensor, size_t index) {
    size_t numberOfDims = tensor->shape->numberOfDimensions;
    for (size_t i = 0; i < numberOfDims; i++) {
        if (tensor->shape->orderOfDimensions[i] == index) {
            return tensor->shape->dimensions[i];
        }
    }
    PRINT_ERROR("Tensor doesn't have %zu dimensions!", index);
    exit(1);
}

void orderDims(tensor_t *tensor, size_t *orderedDims) {
    for (size_t i = 0; i < tensor->shape->numberOfDimensions; i++) {
        orderedDims[i] = getDimensionsByIndex(tensor, i);
    }
}

bool doDimensionsMatch(tensor_t *a, tensor_t *b) {
    size_t numberOfDims = a->shape->numberOfDimensions;
    if (numberOfDims != b->shape->numberOfDimensions) {
        return false;
    }

    size_t aOrderedDims[numberOfDims];
    size_t bOrderedDims[numberOfDims];
    orderDims(a, aOrderedDims);
    orderDims(b, bOrderedDims);

    for (size_t i = 0; i < numberOfDims; i++) {
        if (aOrderedDims[i] != bOrderedDims[i]) {
            return false;
        }
    }
    return true;
}

void calcIndicesByRawIndex(size_t numberOfDims, size_t *dims, size_t rawIndex, size_t *indices) {
    size_t offset = 1;
    for (size_t i = 0; i < numberOfDims; i++) {
        offset *= dims[i];
    }

    size_t restIndex = rawIndex;
    for (size_t i = 0; i < numberOfDims; i++) {
        offset /= dims[i];
        indices[i] = restIndex / offset;
        restIndex -= indices[i] * offset;
    }
}

size_t calcElementIndexByIndices(size_t numberOfDims, size_t *dims, size_t *indices,
                                 size_t *orderOfDimensions) {
    size_t offset = 1;
    for (size_t i = 0; i < numberOfDims; i++) {
        offset *= dims[i];
    }

    /* Physical dim p carries logical dim order[p], so the physical multi-index is
     * P[p] = indices[order[p]] (the inverse permutation, not the forward one). */
    size_t orderedIndices[numberOfDims];
    for (size_t d = 0; d < numberOfDims; d++) {
        orderedIndices[d] = indices[orderOfDimensions[d]];
    }

    size_t outputIndex = 0;
    for (size_t i = 0; i < numberOfDims; i++) {
        offset /= dims[i];
        outputIndex += orderedIndices[i] * offset;
    }
    return outputIndex;
}

void floatPointWiseArithmeticInplace(tensor_t *aTensor, tensor_t *bTensor,
                                     floatElementArithmeticFunc_t arithmeticFunc) {
    size_t numberOfElements = calcNumberOfElementsByTensor(aTensor);
    size_t numberOfDims = aTensor->shape->numberOfDimensions;
    size_t *aDims = aTensor->shape->dimensions;
    size_t *bDims = bTensor->shape->dimensions;

    size_t aOrderedDims[numberOfDims];
    size_t bOrderedDims[numberOfDims];
    orderDims(aTensor, aOrderedDims);
    orderDims(bTensor, bOrderedDims);

    for (size_t i = 0; i < numberOfElements; i++) {
        size_t aIndices[numberOfDims];
        calcIndicesByRawIndex(numberOfDims, aOrderedDims, i, aIndices);
        size_t aElementIndex = calcElementIndexByIndices(numberOfDims, aDims, aIndices,
                                                         aTensor->shape->orderOfDimensions);

        size_t bIndices[numberOfDims];
        calcIndicesByRawIndex(numberOfDims, bOrderedDims, i, bIndices);
        size_t bElementIndex = calcElementIndexByIndices(numberOfDims, bDims, bIndices,
                                                         bTensor->shape->orderOfDimensions);

        float aValue = readBytesAsFloat(&aTensor->data[aElementIndex * sizeof(float)]);
        float bValue = readBytesAsFloat(&bTensor->data[bElementIndex * sizeof(float)]);

        /* Written back to the physical slot it was read from, so a transposed aTensor
         * keeps its layout. */
        writeFloatToByteArray(arithmeticFunc(aValue, bValue),
                              &aTensor->data[aElementIndex * sizeof(float)]);
    }
}

void floatElementWithTensorArithmeticInplace(tensor_t *tensor, float x,
                                             floatElementArithmeticFunc_t arithmeticFunc) {
    size_t numberOfElements = calcNumberOfElementsByTensor(tensor);
    for (size_t i = 0; i < numberOfElements; i++) {
        size_t byteIndex = i * sizeof(float);
        float currentValue = readBytesAsFloat(&tensor->data[byteIndex]);
        writeFloatToByteArray(arithmeticFunc(currentValue, x), &tensor->data[byteIndex]);
    }
}
