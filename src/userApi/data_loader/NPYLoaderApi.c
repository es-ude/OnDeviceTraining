#define SOURCE_FILE "NPY_LOADER_API"

#include <stdio.h>
#include <stdlib.h>

#include "TensorApi.h"
#include "Tensor.h"
#include "Common.h"
#include "StorageApi.h"
#include "Dataset.h"
#include "NPYLoaderApiInternal.h"
#include "NPYLoader.h"
#include "QuantizationApi.h"
#include "Quantization.h"


tensorArray_t *npyLoad(char *path) {
    FILE *f = openNPYFile(path);

    checkMagic(f);

    uint32_t headerSize = readHeaderSize(f);
    char *header = *reserveMemory(headerSize + 1);
    readHeader(header, headerSize, f);

    dtype_t dtype = getDTypeFromHeader(header);
    quantization_t *q = initQByDType(dtype);

    size_t numberOfDims = getNumberOfDimsFromHeader(header);
    shape_t totalShape;
    size_t totalDims[numberOfDims];
    size_t orderOfDims[numberOfDims];
    getShapeFromHeader(&totalShape, totalDims, orderOfDims, header, numberOfDims);

    size_t numberOfTensors = totalShape.dimensions[0];

    shape_t rowShape;
    size_t rowNumberOfDims = totalShape.numberOfDimensions - 1;
    size_t rowDims[rowNumberOfDims];
    size_t rowOrder[rowNumberOfDims];
    rowShape.dimensions = rowDims;
    rowShape.numberOfDimensions = rowNumberOfDims;
    rowShape.orderOfDimensions = rowOrder;
    getRowShape(&totalShape, &rowShape);

    size_t bytesPerValue = calcBytesPerElement(q);
    size_t numberOfValuesInRow = calcNumberOfElementsByShape(&rowShape);

    tensorArray_t *tensorArr = *reserveMemory(sizeof(tensorArray_t));
    tensor_t **arr = *reserveMemory(numberOfTensors * sizeof(tensor_t *));
    tensorArr->array = arr;
    tensorArr->size = numberOfTensors;

    for (size_t i = 0; i < numberOfTensors; i++) {
        float *data = *reserveMemory(numberOfValuesInRow * bytesPerValue);
        size_t *dims = *reserveMemory(numberOfDims * sizeof(size_t));
        memcpy(dims, rowDims, numberOfDims * sizeof(size_t));

        size_t n = fread(data, bytesPerValue, numberOfValuesInRow, f);
        tensorArr->array[i] = tensorInit(data,
                                         dims,
                                         rowShape.numberOfDimensions,
                                         q,
                                         NULL);
        if (n != numberOfValuesInRow) {
            PRINT_ERROR("fread did not read the correct number of bytes!");
            exit(1);
        }
    }

    fclose(f);

    return tensorArr;
}

sample_t *npyGetSample(dataset_t *dataset, size_t index) {
    sample_t *sample = *reserveMemory(sizeof(sample_t));
    sample->item = dataset->items->array[index];
    sample->label = dataset->labels->array[index];

    return sample;
}


static void getRowShape(shape_t *totalShape, shape_t *rowShape) {
    for (size_t i = 0; i < rowShape->numberOfDimensions; i++) {
        rowShape->dimensions[i] = totalShape->dimensions[i + 1];
        rowShape->orderOfDimensions[i] = i;
    }
}

static quantization_t *initQByDType(dtype_t dtype) {
    switch (dtype) {
    case FLOAT_32:
        quantization_t *floatQ = quantizationInitFloat();
        return floatQ;
    case INT_32:
        quantization_t *intQ = quantizationInitInt32();
        return intQ;
    default:
        PRINT_ERROR("Unsupported DType!");
        exit(1);
    }
}
