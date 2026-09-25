#ifndef ENV5_RUNTIME_SOFTMAX_H
#define ENV5_RUNTIME_SOFTMAX_H

#include <stdbool.h>

#include "ArithmeticType.h"
#include "BfpSoftmaxExp.h"
#include "Layer.h"

typedef struct softmaxConfig {
    arithmetic_t forwardMath;
    arithmetic_t propLossMath;
    quantization_t *outputQ;
    quantization_t *propLossQ;
    bool ownsQuantizations;
    /* P6-2: rounding of the INTEGER right-shift sites inside the native BFP
     * softmax kernels (alignment shift + the >>z inside bfpIExpQ). ORTHOGONAL
     * to every roundingMode_t in the config -- it never derives from one, and
     * the OUT_WRITE pack/staging keep using the normal roundingMode_t
     * machinery. Factories default to BFP_SHIFT_TRUNC (I-BERT-faithful); set
     * via softmaxSetBfpExpShiftRounding (SoftmaxApi.h). Inert unless
     * forwardMath/propLossMath is ARITH_BFP. */
    bfpShiftRounding_t bfpExpShiftRounding;
} softmaxConfig_t;

void softmaxInitConfig(softmaxConfig_t *softmaxConfig, quantization_t *forwardQ,
                       quantization_t *backwardQ);

void softmaxInitLayer(layerConfig_t *softmaxConfig, layer_t *softmaxLayer);

/* Row semantic (#152): row = axis 0 -- each row normalizes over all elements
 * after axis 0 (count / dims[0] of them) and a rank-1 input is one row, the
 * same partition as crossEntropyForwardFloat's MEAN rule. The row count is
 * the storage dimensions[0] (the field that rule reads), not the logical
 * axis 0 of a transposeTensor view. An input with more than
 * one storage row must be identity-order (fail fast otherwise), so a logical
 * [1, N] stored as [N, 1] + transpose fails fast; one storage row takes any
 * order. The ARITH_BFP arms take a single storage row only ([N] or [1, N]);
 * more rows fail fast. */
void softmaxForward(layer_t *softmaxLayer, tensor_t *input, tensor_t *output);

/* Same row partition as softmaxForward: the Jacobian is block-diagonal over
 * rows, and each row's s is recomputed from its own logits (P6-1). */
void softmaxBackward(layer_t *softmaxLayer, tensor_t *input, tensor_t *loss, tensor_t *propLoss);

void softmaxCalcOutputShape(layer_t *softmaxLayer, shape_t *inputShape, shape_t *outputShape);

#endif // ENV5_RUNTIME_SOFTMAX_H
