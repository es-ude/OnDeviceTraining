#define SOURCE_FILE "NPY_LOADER_API"

#include <stdio.h>
#include <stdlib.h>

#include "Common.h"
#include "Dataset.h"
#include "NPYLoader.h"
#include "NPYLoaderApiInternal.h"
#include "Quantization.h"
#include "QuantizationApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"

tensorArray_t *npyLoad(char *path) {
    FILE *f = openNPYFile(path);

    npyHeader_t header;
    npyReadHeader(f, path, &header);

    /* Order matters: (2^61, 0) passes the sample check and must still stop at the
     * element check before numberOfTensors sizes a reservation. */
    if (header.numberOfDims < 1) {
        PRINT_ERROR("npyLoad: %s has a 0-d shape; need a leading sample dimension", path);
        exit(1);
    }
    if (header.dims[0] < 1) {
        PRINT_ERROR("npyLoad: %s has zero samples", path);
        exit(1);
    }
    if (header.numberOfElements < 1) {
        PRINT_ERROR("npyLoad: %s has zero elements", path);
        exit(1);
    }

    quantization_t *q = initQByDType(header.dtype);

    shape_t totalShape;
    size_t orderOfDims[NPY_MAX_DIMS];
    setOrderOfDimsForNewTensor(header.numberOfDims, orderOfDims);
    setShape(&totalShape, header.dims, header.numberOfDims, orderOfDims);

    size_t numberOfTensors = totalShape.dimensions[0];

    shape_t rowShape;
    size_t rowNumberOfDims = totalShape.numberOfDimensions - 1;
    size_t rowDims[NPY_MAX_DIMS];
    size_t rowOrder[NPY_MAX_DIMS];
    rowShape.dimensions = rowDims;
    rowShape.numberOfDimensions = rowNumberOfDims;
    rowShape.orderOfDimensions = rowOrder;
    getRowShape(&totalShape, &rowShape);

    size_t bytesPerValue = calcBytesPerElement(q);
    size_t numberOfValuesInRow = calcNumberOfElementsByShape(&rowShape);

    tensorArray_t *tensorArr = reserveMemory(sizeof(tensorArray_t));
    /* Cannot wrap, even with a 32-bit long: elements are 4 bytes for both supported
     * dtypes and every sample holds at least one, so numberOfTensors <= numberOfElements
     * and numberOfTensors * sizeof(tensor_t *) <= 2 * payloadBytes, where payloadBytes
     * equals the file bytes after the header (an ftell difference, <= LONG_MAX). */
    tensor_t **arr = reserveMemory(numberOfTensors * sizeof(tensor_t *));
    tensorArr->array = arr;
    tensorArr->size = numberOfTensors;

    for (size_t i = 0; i < numberOfTensors; i++) {
        size_t *dims = reserveMemory(rowNumberOfDims * sizeof(size_t));
        memcpy(dims, rowDims, rowNumberOfDims * sizeof(size_t));
        size_t *order = reserveMemory(rowNumberOfDims * sizeof(size_t));
        setOrderOfDimsForNewTensor(rowNumberOfDims, order);
        shape_t *shape = reserveMemory(sizeof(shape_t));
        setShape(shape, dims, rowNumberOfDims, order);

        /* Fresh quantization clone per tensor: every array entry now owns its
         * own quantization, so freeTensor on any one doesn't double-free a
         * shared `q`. */
        quantization_t *rowQ = getQLike(q);

        tensor_t *t = initTensor(shape, rowQ, NULL);
        tensorArr->array[i] = t;

        size_t n = fread(t->data, bytesPerValue, numberOfValuesInRow, f);
        if (n != numberOfValuesInRow) {
            PRINT_ERROR("fread did not read the correct number of bytes!");
            exit(1);
        }
    }

    /* `q` was deep-copied per tensor via getQLike; free the original. */
    freeQuantization(q);

    fclose(f);

    return tensorArr;
}

tensor_t *npyLoadFlat(char *path) {
    FILE *f = openNPYFile(path);

    npyHeader_t header;
    npyReadHeader(f, path, &header);

    if (header.numberOfElements < 1) {
        PRINT_ERROR("npyLoadFlat: %s has zero elements", path);
        exit(1);
    }

    quantization_t *q = initQByDType(header.dtype);

    size_t numberOfDims = header.numberOfDims;
    size_t *fileDims = header.dims;
    shape_t fileShape;
    size_t fileOrder[NPY_MAX_DIMS];
    setOrderOfDimsForNewTensor(numberOfDims, fileOrder);
    setShape(&fileShape, fileDims, numberOfDims, fileOrder);

    /* Owned copy of the FULL on-disk shape (dim0 included) so the entire payload
     * lands in one contiguous tensor — no dim0 slicing as npyLoad() does. */
    size_t *dims = reserveMemory(numberOfDims * sizeof(size_t));
    memcpy(dims, fileDims, numberOfDims * sizeof(size_t));
    size_t *order = reserveMemory(numberOfDims * sizeof(size_t));
    setOrderOfDimsForNewTensor(numberOfDims, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, numberOfDims, order);

    /* The tensor takes ownership of `q`; freeTensor() releases it later. */
    tensor_t *t = initTensor(shape, q, NULL);

    size_t bytesPerValue = calcBytesPerElement(q);
    size_t numberOfValues = calcNumberOfElementsByShape(&fileShape);
    size_t n = fread(t->data, bytesPerValue, numberOfValues, f);
    if (n != numberOfValues) {
        PRINT_ERROR("npyLoadFlat: fread read %zu of %zu values from %s", n, numberOfValues, path);
        exit(1);
    }

    fclose(f);

    return t;
}

sample_t *npyGetSample(dataset_t *dataset, size_t index) {
    sample_t *sample = reserveMemory(sizeof(sample_t));
    sample->item = dataset->items->array[index];
    sample->label = dataset->labels->array[index];

    return sample;
}

void freeTensorArray(tensorArray_t *tensorArr) {
    /* Walk every entry and freeTensor it; freeTensor cascades to data,
     * shape (+ dims, + order), quantization, sparsity, and the tensor
     * struct. Then release the array buffer and the tensorArray_t struct
     * itself, both reservedMemory-allocated by npyLoad. */
    for (size_t i = 0; i < tensorArr->size; i++) {
        freeTensor(tensorArr->array[i]);
    }
    freeReservedMemory(tensorArr->array);
    freeReservedMemory(tensorArr);
}

static void getRowShape(shape_t *totalShape, shape_t *rowShape) {
    for (size_t i = 0; i < rowShape->numberOfDimensions; i++) {
        rowShape->dimensions[i] = totalShape->dimensions[i + 1];
        rowShape->orderOfDimensions[i] = i;
    }
}

static quantization_t *initQByDType(dtype_t dtype) {
    switch (dtype) {
    case FLOAT_32: {
        quantization_t *floatQ = quantizationInitFloat();
        return floatQ;
    }
    case INT_32: {
        quantization_t *intQ = quantizationInitInt32();
        return intQ;
    }
    default:
        PRINT_ERROR("Unsupported DType!");
        exit(1);
    }
}
