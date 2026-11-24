#ifndef OPTIMIZER_API_H
#define OPTIMIZER_API_H

#include "Model.h"
#include "Optimizer.h"

/* No standalone `SGD` factory: sgdMCreateOptim() with momentumFactor == 0
 * IS plain SGD -- one stateless update op per parameter, no state buffers. */

/* Scales every tracked gradient in place by `factor` (the macro-batch mean scale or a clip
 * coefficient). A non-positive or non-finite factor only warns for FLOAT32 grads, which carry
 * it on; a SYM_INT32/SYM/ASYM scale fold fails fast (a stored scale is a positive dyadic value),
 * and so does a BFP grad (its grid has no non-finite code).
 * updateKey is the key of the update the gradients belong to (the key
 * optimizerStep gets next); parameter p's BFP rescale draws under it folded
 * with (PARAM, p), then (SCALE, 0). NULL means the caller supplies no
 * randomness. With a key, more than 2^32 parameters stop the program with a
 * named error before any gradient changes. */
void scaleOptimizerGradients(optimizer_t *optimizer, float factor, const rngKey_t *updateKey);

/* The macro-batch MEAN scale by 1 / divisor, from the INTEGER divisor (the
 * loss family's computeMeanDivisor: totalSamples, or totalSamples x F for
 * MSE). trainingEpochDefault calls it once per macro-batch, after the last
 * micro-batch's accumulation and before optimizerStep; optimizerZeroGrad
 * follows the step, so the next batch starts from zero. Non-BFP grads take
 * scaleOptimizerGradients's fold with 1.0f / (float)divisor -- bit for bit the
 * float entry with computeMeanScale. A BFP grad takes the integer path:
 * bfpScalarFromCount(divisor) once per call (exact for a power of two), then
 * scaleBfpTensorByScalarInPlace on parameter p's SCALE stream (the stream the
 * float entry draws from); a power-of-two divisor on a tight grid only lowers
 * the exponents. Fails fast: divisor 0 (or >= 2^31 with a BFP grad), more
 * than 2^32 parameters under a key, SR BFP grads without a key. */
void scaleOptimizerGradientsMean(optimizer_t *optimizer, size_t divisor, const rngKey_t *updateKey);

/* Call after accumulation and mean scaling, before stepping
 * (torch.nn.utils.clip_grad_norm_ parity). Supports FLOAT32/SYM_INT32 gradients; maxNorm must
 * be positive and finite. Returns the pre-clip global L2 norm over all gradients jointly.
 * Applies maxNorm/(norm+1e-6) only when below one; otherwise gradients stay byte-identical. */
float optimizerClipGradNorm(optimizer_t *optimizer, float maxNorm);

/* slots must hold calcTotalNumberOfStates() pointers; collection excludes frozen parameters and
 * running buffers. */
void collectTrainableParameters(const model_t *m, parameter_t **slots);

/* Tells an all-frozen model (rejected by the factories) from a parameterless one (allowed):
 * both collect zero parameters. */
bool modelHasFrozenLayer(const model_t *m);

/* Reject missing or unsupported gradient storage during optimizer construction. A tracked grad's
 * storage (dtype, grouping) is fixed once the optimizer exists; requantizing or regrouping it
 * later is out of contract. */
void validateOptimizerGradStorage(optimizer_t *optim, const char *factoryName);

/* Builds statesPerParameter zeroed state tensors per collected parameter, each at the
 * parameter's shape with its own clone of momentQuant. Moment storage must be per-tensor: a
 * grouped SYM/ASYM/BFP template fails fast, checked only when there is a parameter to build
 * a state for. Call after collectTrainableParameters. */
states_t **initMomentStates(optimizer_t *optim, size_t statesPerParameter,
                            quantization_t *momentQuant, const char *factoryName);

/* Frees one parameter's state tensors, their array and the states_t itself. */
void freeState(states_t *state);

/* Releases optimizer-owned parameters (with their gradients), state tensors and implementation
 * storage. */
void freeOptim(optimizer_t *optim);

#endif // OPTIMIZER_API_H
