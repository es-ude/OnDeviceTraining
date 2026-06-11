#ifndef CROSSENTROPY_H
#define CROSSENTROPY_H

#include "LossFunction.h"
#include "Tensor.h"

float crossEntropyForwardFloat(tensor_t *softmaxOutput, tensor_t *distribution,
                               reduction_t reduction);

/* Dtype dispatcher registered as lossFunctions[CROSS_ENTROPY].forward.
 * FLOAT32 -> crossEntropyForwardFloat; any other dtype fails fast — the raw
 * float impl casts the data buffer to float*, so e.g. SYM_INT32 mantissas
 * would be silently reinterpreted as float bit patterns. */
float crossEntropyForward(tensor_t *softmaxOutput, tensor_t *distribution, reduction_t reduction);

void crossEntropySoftmaxBackward(tensor_t *softmaxOutput, tensor_t *distribution, tensor_t *loss);

/* Per-loss MEAN-reduction scale factor (PyTorch parity).
 *
 * Returns 1 / totalSamples for CE. The modelOutput tensor is accepted for
 * vtable-uniformity with MSE but is ignored: the class axis is internal
 * aggregation in CE, not an independent loss-term axis.
 *
 * Caller must check backwardReduction == REDUCTION_MEAN before invoking. */
float computeMeanScaleCE(size_t totalSamples, tensor_t *modelOutput);

#endif // CROSSENTROPY_H
