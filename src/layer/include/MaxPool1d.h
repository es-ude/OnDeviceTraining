#ifndef ODT_MAX_POOL_1D_H
#define ODT_MAX_POOL_1D_H

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#include "ArithmeticType.h"
#include "Kernel.h"
#include "Layer.h"
#include "Tensor.h"

/* argmaxIndices: INT32 [B, C, Lout] index state written by every forward and
 * read by the backward. The caller pre-allocates it (the factory sizes it for
 * B = 1); maxPool1dForward grows it on demand to the largest B seen and keeps
 * dims[0] = B, writing it only when B changes (#152 PR3b, spec §6.7), so its
 * ->data must come from reserveMemory: a growing forward frees and replaces it.
 * argmaxCapacity is the element count of the allocation argmaxCapacityData
 * points at. Capacity is trusted only while argmaxCapacityData ==
 * argmaxIndices->data; any other buffer (zero-initialised config, argmax
 * swapped in by hand) is adopted at its own element count on the next forward.
 *
 * CONCURRENCY INVARIANT: one MaxPool layer instance must never run two
 * forwards concurrently, nor a forward concurrently with its backward. This
 * already held before growth -- the forward writes the argmax CONTENTS into
 * this config -- so growing the buffer in place adds no new hazard. Concurrent
 * forwards need separate layer instances. */
typedef struct maxPool1dConfig {
    kernel_t *kernel;
    tensor_t *argmaxIndices;           // INT32 [B, C, Lout]; see the block comment above
    size_t argmaxCapacity;             // elements the argmax allocation holds
    const uint8_t *argmaxCapacityData; // the allocation argmaxCapacity describes
    arithmetic_t forwardMath;
    arithmetic_t propLossMath;
    quantization_t *outputQ;
    quantization_t *propLossQ;
    bool ownsQuantizations;
} maxPool1dConfig_t;

void initMaxPool1dConfig(maxPool1dConfig_t *cfg, kernel_t *kernel, tensor_t *argmaxIndices,
                         quantization_t *forwardQ, quantization_t *propLossQ);

void maxPool1dForward(layer_t *layer, tensor_t *input, tensor_t *output);

void maxPool1dBackward(layer_t *layer, tensor_t *forwardInput, tensor_t *lossGrad,
                       tensor_t *propLoss);
void maxPool1dBackwardFloat(layer_t *layer, tensor_t *forwardInput, tensor_t *lossGrad,
                            tensor_t *propLoss);

void maxPool1dCalcOutputShape(layer_t *layer, shape_t *inputShape, shape_t *outputShape);

#endif // ODT_MAX_POOL_1D_H
