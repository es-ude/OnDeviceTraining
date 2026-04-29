#define SOURCE_FILE "ODT_KERNEL"

#include <string.h>

#include "Kernel.h"

#include "Common.h"

void initKernel(kernel_t *kernel, size_t size, paddingType_t paddingType, size_t dilation,
                size_t stride) {
    kernel->dilation = dilation;
    kernel->paddingType = paddingType;
    kernel->size = size;
    kernel->stride = stride;
}

void kernelGetWindowIndices1d(kernel_t *kernel, size_t inputStartIndex, size_t *windowIndices) {
    size_t dilation = kernel->dilation;
    size_t kernelSize = kernel->size;

    for (size_t i = 0; i < kernelSize; i++) {
        windowIndices[i] = inputStartIndex + i * dilation;
    }
}

size_t kernelCalculatePaddingSize1d(size_t inputLengthPerChannel, kernel_t *kernel) {
    size_t kernelSize = kernel->size;
    size_t dilation = kernel->dilation;
    size_t stride = kernel->stride;
    paddingType_t paddingType = kernel->paddingType;

    size_t effectiveKernelSize = dilation * (kernelSize - 1) + 1;

    switch (paddingType) {
    case VALID:
        return 0;
    case SAME:
        // for the first output value, we need effectiveKernelSize-many input values
        // for every subsequent value, we need stride-many more
        // for same padding, we need inputLengthPerChannel-many output values
        size_t neededValues = effectiveKernelSize + (inputLengthPerChannel - 1) * stride;
        size_t missingValues = neededValues - inputLengthPerChannel;
        return missingValues;
    default:
        PRINT_ERROR("Unknown padding type");
        exit(1);
    }
}
