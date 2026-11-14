#define SOURCE_FILE "NPY_LOADER_API"

#include <stdio.h>
#include <stdlib.h>

#include "Common.h"
#include "Dataset.h"
#include "NPYLoader.h"
#include "Quantization.h"
#include "QuantizationApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"

/* Callers pass a dtype npyReadHeader accepted: FLOAT_32 or INT_32. */
static quantization_t *initQByDType(dtype_t dtype) {
    return dtype == FLOAT_32 ? quantizationInitFloat() : quantizationInitInt32();
}

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

    size_t numberOfTensors = header.dims[0];
    size_t sampleNumberOfDims = header.numberOfDims - 1;
    size_t bytesPerValue = calcBytesPerElement(q);
    size_t numberOfValuesInRow = header.numberOfElements / numberOfTensors;

    tensorArray_t *tensorArr = reserveMemory(sizeof(tensorArray_t));
    /* Cannot wrap, even with a 32-bit long: elements are 4 bytes for both supported
     * dtypes and every sample holds at least one, so numberOfTensors <= numberOfElements
     * and numberOfTensors * sizeof(tensor_t *) <= 2 * payloadBytes, where payloadBytes
     * equals the file bytes after the header (an ftell difference, <= LONG_MAX). */
    tensor_t **arr = reserveMemory(numberOfTensors * sizeof(tensor_t *));
    tensorArr->array = arr;
    tensorArr->size = numberOfTensors;

    for (size_t i = 0; i < numberOfTensors; i++) {
        shape_t *shape = buildOwnedShape(header.dims + 1, sampleNumberOfDims);
        /* Each tensor owns a separate quantization. */
        quantization_t *rowQ = getQLike(q);

        tensor_t *t = initTensor(shape, rowQ, NULL);
        tensorArr->array[i] = t;

        size_t n = fread(t->data, bytesPerValue, numberOfValuesInRow, f);
        if (n != numberOfValuesInRow) {
            PRINT_ERROR("fread did not read the correct number of bytes!");
            exit(1);
        }
    }

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

    tensor_t *t = initTensor(buildOwnedShape(header.dims, header.numberOfDims), q, NULL);

    size_t bytesPerValue = calcBytesPerElement(q);
    size_t numberOfValues = header.numberOfElements;
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
    for (size_t i = 0; i < tensorArr->size; i++) {
        freeTensor(tensorArr->array[i]);
    }
    freeReservedMemory(tensorArr->array);
    freeReservedMemory(tensorArr);
}
