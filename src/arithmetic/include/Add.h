#ifndef ENV5_RUNTIME_ADD_H
#define ENV5_RUNTIME_ADD_H

#include "Tensor.h"
#include <stddef.h>
#include <stdint.h>

/* Pointer-only use: the identical typedef lives in RNG.h (C11 allows the
 * repetition), so this header needs no RNG include path. */
typedef struct rngStream rngStream_t;

float addFloat32s(float a, float b);
int32_t addInt32s(int32_t a, int32_t b);

/*! Writes a + b into a. Both tensors must have the same logical shape (rank and dims);
 *  their axis orders may differ. */
void addFloat32TensorsInplace(tensor_t *a, tensor_t *b);

/*!
 * Writes a + b into aTensor: both are dequantized with their own scales, added in
 * float, and requantized into aTensor's grid. Both tensors must have the same logical
 * shape (rank and dims).
 */
void addSymInt32TensorsInplace(tensor_t *aTensor, tensor_t *bTensor, rngStream_t *rs);

/*! Number of adds counted so far; only counts when built with TRACK_INSTRUCTIONS. */
size_t getAddInstructionCounter();

#endif // ENV5_RUNTIME_ADD_H
