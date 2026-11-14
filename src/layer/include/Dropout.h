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
    tensor_t *mask; // BOOL; caller-owned, with the runtime input's element count
    arithmetic_t forwardMath;
    arithmetic_t propLossMath;
    quantization_t *outputQ;
    quantization_t *propLossQ;
    bool ownsQuantizations;
} dropoutConfig_t;

/*! Exits unless p is in [0, 1) and mask is a BOOL tensor. */
void initDropoutConfig(dropoutConfig_t *cfg, float p, tensor_t *mask, quantization_t *forwardQ,
                       quantization_t *backwardQ);

/*! The loaded-config contract forward/backward rely on: p in [0, 1) (NaN rejected). Arithmetic
 *  needs no check: the factory derives it and the deserializer bounds its tag. Call once where a
 *  config enters -- initDropoutConfig and the deserializer. `who` prefixes the message. */
void dropoutValidateConfig(const dropoutConfig_t *cfg, const char *who);

/* Each direction requires both wires stored in its arithmetic's dtype, rejected before the mask
 * is touched; ARITH_BFP also requires equal counts, grids and widths. A training-mode BFP pair
 * must not share storage; eval mode accepts aliases. */

void dropoutForward(layer_t *dropoutLayer, tensor_t *input, tensor_t *output,
                    const layerExec_t *exec);
void dropoutBackward(layer_t *dropoutLayer, tensor_t *forwardInput, tensor_t *loss,
                     tensor_t *propLoss, const layerExec_t *exec);
void dropoutCalcOutputShape(layer_t *dropoutLayer, shape_t *inputShape, shape_t *outputShape);

#endif // ODT_DROPOUT_H
