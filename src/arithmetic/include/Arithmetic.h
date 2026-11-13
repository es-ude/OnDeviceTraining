#ifndef ENV5_RUNTIME_TENSOR_MATH_H
#define ENV5_RUNTIME_TENSOR_MATH_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "Tensor.h"

typedef float (*floatElementArithmeticFunc_t)(float a, float b);

/*! True when both tensors have the same rank and the same logical dimensions. */
bool doDimensionsMatch(tensor_t *a, tensor_t *b);

/*! Writes the tensor's dimensions in logical order (orderOfDimensions applied). */
void orderDims(tensor_t *tensor, size_t *orderedDims);

/*! Size of logical dimension index. */
size_t getDimensionsByIndex(tensor_t *tensor, size_t index);

/*! Splits a row-major flat index over dims into one index per dimension. */
void calcIndicesByRawIndex(size_t numberOfDims, size_t *dims, size_t rawIndex, size_t *indices);

/*! Storage offset of the logical multi-index indices in a tensor with physical dims and
 * orderOfDimensions. */
size_t calcElementIndexByIndices(size_t numberOfDims, size_t *dims, size_t *indices,
                                 size_t *orderOfDimensions);

/*! a[i] = f(a[i], b[i]) for float tensors with the same logical shape; either may be
 * transposed. */
void floatPointWiseArithmeticInplace(tensor_t *aTensor, tensor_t *bTensor,
                                     floatElementArithmeticFunc_t arithmeticFunc);

/*! t[i] = f(t[i], x) over the float tensor's storage order. */
void floatElementWithTensorArithmeticInplace(tensor_t *tensor, float x,
                                             floatElementArithmeticFunc_t arithmeticFunc);

#endif // ENV5_RUNTIME_TENSOR_MATH_H
