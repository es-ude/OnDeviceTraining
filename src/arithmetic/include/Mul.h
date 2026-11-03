#ifndef ENV5_RUNTIME_MUL_H
#define ENV5_RUNTIME_MUL_H

#include "Tensor.h"
#include <stddef.h>
#include <stdint.h>

int32_t mulIntsWithInstructionCounter(int32_t a, int32_t b);

/* Inline so the int MAC loops can fold the multiply; TRACK_INSTRUCTIONS is
 * therefore evaluated in the including translation unit, not in Mul.c. */
static inline int32_t mulInt32s(int32_t a, int32_t b) {
#ifdef TRACK_INSTRUCTIONS
    return mulIntsWithInstructionCounter(a, b);
#else
    return a * b;
#endif
}

float mulFloat32s(float a, float b);

void mulInt32Tensors(tensor_t *a, tensor_t *b, tensor_t *outputTensor);
void mulInt32TensorsInplace(tensor_t *a, tensor_t *b);

void mulInt32ElementWithInt32Tensor(tensor_t *a, int32_t b, tensor_t *outputTensor);
void mulInt32ElementWithInt32TensorInplace(tensor_t *a, int32_t b);

void mulFloat32Tensors(tensor_t *a, tensor_t *b, tensor_t *outputTensor);
void mulFloat32TensorsInplace(tensor_t *a, tensor_t *b);

void mulFloat32ElementWithFloat32Tensor(tensor_t *a, float b, tensor_t *outputTensor);
void mulFloat32ElementWithFloat32TensorInplace(tensor_t *a, float b);

/* SYM_INT32 elementwise product: the output's (in place: aTensor's own) config
 * is overwritten -- scale = sa * sb, qMaxBits = the exact width of the product
 * range [-2^(qa-1) * (2^(qb-1) - 1), 2^(qa-1) * 2^(qb-1)] (or its mirror), up
 * to 32 (#516; Matmul.h has the requant-before-clone rule). The int32 product
 * is unchecked: with qa + qb > 32
 * it is signed-overflow undefined behaviour, documented rather than guarded. */
void mulSymInt32Tensors(tensor_t *aTensor, tensor_t *bTensor, tensor_t *outputTensor);
void mulSymInt32TensorsInplace(tensor_t *aTensor, tensor_t *bTensor);

size_t getMulInstructionCounter();

#endif // ENV5_RUNTIME_MUL_H
