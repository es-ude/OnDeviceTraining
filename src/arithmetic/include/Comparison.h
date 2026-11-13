#ifndef ENV5_RUNTIME_COMPARISONS_H
#define ENV5_RUNTIME_COMPARISONS_H
#include "Tensor.h"
#include <stdint.h>

/*! result[i] = a[i] < b ? altNumber : a[i] (float tensors). */
void gteFloatValue(tensor_t *a, float b, float altNumber, tensor_t *result);

/*! result[i] = a[i] < 0 ? altNumber : a[i] on SYM_INT32 codes; the caller copies the scale. */
void gteSymInt32Zero(tensor_t *a, int32_t altNumber, tensor_t *result);

/*!
 * Packed-BFP ReLU: negative codes become 0, the group exponents are copied. a and result
 * must share element count and {numGroups, groupSize, mantissaBits, exponentBits}. There
 * is no altNumber: a nonzero value-domain constant would need a different code per group.
 */
void gteBfpZero(tensor_t *a, tensor_t *result);

#endif // ENV5_RUNTIME_COMPARISONS_H
