#ifndef ENV5_RUNTIME_COMPARISONS_H
#define ENV5_RUNTIME_COMPARISONS_H
#include "Tensor.h"
#include <stdint.h>

void gteInt32Value(tensor_t *a, int32_t b, int32_t altNumber, tensor_t *result);
void gteInt32Tensor(tensor_t *a, tensor_t *b, int32_t altNumber, tensor_t *result);

void gteFloatValue(tensor_t *a, float b, float altNumber, tensor_t *result);
void gteFloatTensor(tensor_t *a, tensor_t *b, float altNumber, tensor_t *result);

void gteSymInt32Zero(tensor_t *a, int32_t altNumber, tensor_t *result);

/* BFP epic PR4 (R-P2): packed-BFP ReLU — clamp negative codes to 0, copy the
 * group exponents verbatim. No altNumber parameter: any nonzero alternative
 * would be a value-domain constant with a different code per group. Input and
 * result must hold the same number of elements and share {numGroups,
 * groupSize, mantissaBits, exponentBits}; both are enforced fail-fast. */
void gteBfpZero(tensor_t *a, tensor_t *result);

#endif // ENV5_RUNTIME_COMPARISONS_H
