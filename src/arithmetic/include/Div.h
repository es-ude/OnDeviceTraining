#ifndef ENV5_RUNTIME_DIV_H
#define ENV5_RUNTIME_DIV_H

#include <stddef.h>

#include "Tensor.h"

float divFloat32s(float a, float b);

/* SYM_INT32 elementwise integer quotient of two equal-count tensors (b must
 * hold no zero code); the output's scale becomes sa / sb on the stored scale
 * grid. */
void divSymInt32Tensors(tensor_t *aTensor, tensor_t *bTensor, tensor_t *outputTensor);

/*! Number of divisions counted so far; only counts when built with TRACK_INSTRUCTIONS. */
size_t getDivInstructionCounter();

#endif // ENV5_RUNTIME_DIV_H
