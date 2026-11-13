#define SOURCE_FILE "AXPBY"

#include "Axpby.h"
#include "Arithmetic.h"
#include "Common.h"
#include "DTypes.h"
#include "Tensor.h"

void axpbyFloat32Tensors(float a, tensor_t *x, float b, tensor_t *y, tensor_t *out) {
    size_t numberOfElements = calcNumberOfElementsByTensor(out);
    size_t bytesPerElement = sizeof(float);
    size_t numberOfDims = x->shape->numberOfDimensions;
    size_t *xDims = x->shape->dimensions;
    size_t *yDims = y->shape->dimensions;

    /* Decode the logical index against the ORDERED dims, then re-encode with the
     * physical dims. Decoding against physical dims looks right whenever the
     * transposed axes share a size, and reads out of bounds otherwise. */
    size_t xOrderedDims[numberOfDims];
    size_t yOrderedDims[numberOfDims];
    orderDims(x, xOrderedDims);
    orderDims(y, yOrderedDims);

    for (size_t i = 0; i < numberOfElements; i++) {
        size_t xIndices[numberOfDims];
        calcIndicesByRawIndex(numberOfDims, xOrderedDims, i, xIndices);
        size_t xElementIndex =
            calcElementIndexByIndices(numberOfDims, xDims, xIndices, x->shape->orderOfDimensions);

        size_t yIndices[numberOfDims];
        calcIndicesByRawIndex(numberOfDims, yOrderedDims, i, yIndices);
        size_t yElementIndex =
            calcElementIndexByIndices(numberOfDims, yDims, yIndices, y->shape->orderOfDimensions);

        float xValue = readBytesAsFloat(&x->data[xElementIndex * bytesPerElement]);
        float yValue = readBytesAsFloat(&y->data[yElementIndex * bytesPerElement]);

        writeFloatToByteArray(a * xValue + b * yValue, &out->data[i * bytesPerElement]);
    }
}
