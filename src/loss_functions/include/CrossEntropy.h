#ifndef CROSSENTROPY_H
#define CROSSENTROPY_H

#include "LossFunction.h"
#include "Tensor.h"

float crossEntropyForwardFloat(tensor_t *softmaxOutput, tensor_t *distribution,
                               reduction_t reduction);

/* Dtype dispatcher registered as lossFunctions[CROSS_ENTROPY].forward.
 * FLOAT32 -> crossEntropyForwardFloat (raw float* view); SYM_INT32 and BFP ->
 * the dtype-generic fake-quant arm, which dequantizes both operands through
 * convertTensor before running the float core. Every other dtype fails fast:
 * the raw float impl casts the data buffer to float*, so e.g. ASYM codes would
 * be silently reinterpreted as float bit patterns.
 * It also fails fast when softmaxOutput has rank < 2 or no elements, or the
 * distribution's shape differs from it (#153).
 * Returns NaN (and prints one PRINT_ERROR per process) when any softmax
 * element is non-finite; it no longer aborts (#446). */
float crossEntropyForward(tensor_t *softmaxOutput, tensor_t *distribution, reduction_t reduction,
                          const layerExec_t *exec);

/* The combined Softmax + CE gradient p - y into the loss-grad wire. A BFP p
 * is the integer head (needs a BFP grad wire and a FLOAT32 or BFP one-hot
 * label, fails fast otherwise); a FLOAT32 p the float island (ONE conversion into a
 * quantized grad wire); SYM_INT32 / ASYM p the dequantizing island. */
void crossEntropySoftmaxBackward(tensor_t *softmaxOutput, tensor_t *distribution, tensor_t *loss,
                                 const layerExec_t *exec);

struct rngStream;

/* The integer head's executeOp kernel (arithmetic-bfp.md section 14): ops[0]
 * = p as unpacked-BFP codes; the one-hot label is either ctx (FLOAT32, read
 * by bit pattern; n == 1) or ops[1] (BFP as unpacked codes, 1.0 exactly =
 * the class; n == 2); rawOut = BFP_RAW on the loss-grad wire's geometry. Exposed for
 * kernel-level tests; the loss calls crossEntropySoftmaxBackward. */
void crossEntropySoftmaxBackwardKernelBfp(tensor_t **ops, size_t n, tensor_t *rawOut,
                                          tensor_t *auxOut, const void *ctx, struct rngStream *rs);

/* Per-loss MEAN-reduction scale factor (PyTorch parity).
 *
 * Returns 1 / totalSamples for CE. The modelOutput tensor is accepted for
 * vtable-uniformity with MSE but is ignored: the class axis is internal
 * aggregation in CE, not an independent loss-term axis.
 *
 * Caller must check backwardReduction == REDUCTION_MEAN before invoking. */
float computeMeanScaleCE(size_t totalSamples, tensor_t *modelOutput);
/* The integer mean divisor totalSamples; computeMeanScaleCE = 1.0f / (float)it. */
size_t computeMeanDivisorCE(size_t totalSamples, tensor_t *modelOutput);

#endif // CROSSENTROPY_H
