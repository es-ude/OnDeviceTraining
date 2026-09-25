#ifndef MSE_H
#define MSE_H

#include "LossFunction.h"
#include "Tensor.h"

float mseLossForward(tensor_t *output, tensor_t *label, reduction_t reduction);

void mseLossBackwardFloat(tensor_t *modelOutput, tensor_t *label, tensor_t *result);

void mseLossBackward(tensor_t *modelOutput, tensor_t *label, tensor_t *result);

/* Per-loss MEAN-reduction scale factor (PyTorch parity).
 *
 * Returns 1 / (totalSamples * numFeaturesPerSample) for MSE, where
 * numFeaturesPerSample is derived from the model output's shape:
 * numElements(modelOutput) / dimensions[0]. Fails fast unless modelOutput
 * has rank >= 2 and at least one element (#153), so dimensions[0] is
 * never 0.
 *
 * Caller must check backwardReduction == REDUCTION_MEAN before invoking. */
float computeMeanScaleMSE(size_t totalSamples, tensor_t *modelOutput);

#endif // MSE_H
