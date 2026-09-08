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

void softmaxForward(layer_t *softmaxLayer, tensor_t *input, tensor_t *output);

void softmaxBackward(layer_t *softmaxLayer, tensor_t *input, tensor_t *loss, tensor_t *propLoss);

void softmaxCalcOutputShape(layer_t *softmaxLayer, shape_t *inputShape, shape_t *outputShape);

#endif // ENV5_RUNTIME_SOFTMAX_H
