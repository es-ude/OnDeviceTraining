#include "Im2Col.h"

void im2col1dFloat(tensor_t const *input, windowGeometry1d_t const *geometry, tensor_t *output) {
    float *inputData = (float *)input->data;
    float *outputData = (float *)output->data;

    size_t numberOfColumns = geometry->outputLength;
    size_t numberOfRows = geometry->kernelSize;

    for (size_t columnIndex = 0; columnIndex < numberOfColumns; columnIndex++) {
        for (size_t rowIndex = 0; rowIndex < numberOfRows; rowIndex++) {
            size_t inputIndex = columnIndex * geometry->stride + rowIndex * geometry->dilation;
            size_t outputIndex = rowIndex * geometry->outputLength + columnIndex;

            outputData[outputIndex] = inputData[inputIndex];
        }
    }
}
