#ifndef ODT_DROPOUT_H
#define ODT_DROPOUT_H

#include <stdbool.h>

#include "ArithmeticType.h"
#include "Tensor.h"

typedef struct layer layer_t;
typedef struct layerExec layerExec_t;

typedef struct dropoutConfig {
    float p;        // drop probability, must be in [0, 1)
    bool training;  // false = identity (eval); set true by the training loop
    tensor_t *mask; // BOOL, shape == input/output; pre-allocated by caller
    arithmetic_t forwardMath;
    arithmetic_t propLossMath;
    quantization_t *outputQ;
    quantization_t *propLossQ;
    bool ownsQuantizations;
} dropoutConfig_t;

void initDropoutConfig(dropoutConfig_t *cfg, float p, tensor_t *mask, quantization_t *forwardQ,
                       quantization_t *backwardQ);

void dropoutForward(layer_t *dropoutLayer, tensor_t *input, tensor_t *output,
                    const layerExec_t *exec);
void dropoutBackward(layer_t *dropoutLayer, tensor_t *forwardInput, tensor_t *loss,
                     tensor_t *propLoss, const layerExec_t *exec);
void dropoutCalcOutputShape(layer_t *dropoutLayer, shape_t *inputShape, shape_t *outputShape);

#endif // ODT_DROPOUT_H
