#ifndef ODT_KERNEL_H
#define ODT_KERNEL_H

#include <stdlib.h>

typedef enum { VALID, SAME } paddingType_t;

typedef struct kernel {
    size_t size;
    paddingType_t paddingType; /*! Default is 0 */
    size_t stride;             /*! Default is 1 */
    size_t dilation;           /*! Default is 1 */
} kernel_t;

void initKernel(kernel_t *kernel, size_t size, paddingType_t paddingType, size_t dilation,
                size_t stride);

/*! Calculates indices for window slice.
 *
 * @param kernel Pointer to kernel_t
 * @param inputStartIndex Pointer to first element for current window
 * @param windowIndices Buffer for indices
 */
void kernelGetWindowIndices1d(kernel_t *kernel, size_t inputStartIndex, size_t *windowIndices);

/*! Calculates 1D padding size.
 *
 * @param inputLengthPerChannel Number of inputs per channel
 * @param kernel Pointer to kernel_t
 * @returns Total size of padding needed
 */
size_t kernelCalculatePaddingSize1d(size_t inputLengthPerChannel, kernel_t *kernel);

#endif // ODT_KERNEL_H
