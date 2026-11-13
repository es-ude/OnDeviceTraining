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

/*! Scales every element of the float tensor a by b, in place. */
void mulFloat32ElementWithFloat32TensorInplace(tensor_t *a, float b);

/* SYM_INT32 elementwise product of two equal-count tensors; the output may
 * alias an operand. The output's config is overwritten: scale = sa * sb,
 * qMaxBits = the exact width of the product range, up to 32 (#516; Matmul.h
 * has the requant-before-clone rule). The int32 product is unchecked: with
 * qa + qb > 32 it is signed-overflow undefined behaviour. */
void mulSymInt32Tensors(tensor_t *aTensor, tensor_t *bTensor, tensor_t *outputTensor);

/*! Number of multiplies counted so far; only counts when built with TRACK_INSTRUCTIONS. */
size_t getMulInstructionCounter();

#endif // ENV5_RUNTIME_MUL_H
