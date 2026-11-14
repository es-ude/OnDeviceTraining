#ifndef ENV5_RUNTIME_RELU_H
#define ENV5_RUNTIME_RELU_H

#include <stdbool.h>

#include "ArithmeticType.h"
#include "Tensor.h"

typedef struct layer layer_t;
typedef struct layerExec layerExec_t;

typedef struct reluConfig {
    arithmetic_t forwardMath;
    arithmetic_t propLossMath;
    quantization_t *outputQ;
    quantization_t *propLossQ;
    bool ownsQuantizations;
} reluConfig_t;

/*! Supported arithmetic in both slots. Call once where a config enters -- the factories and the
 *  deserializer. `who` prefixes the message. */
void reluValidateConfig(const reluConfig_t *cfg, const char *who);

/* Each direction requires every wire stored in its arithmetic's dtype. ARITH_BFP also requires
 * equal counts, and equal grids and widths between the copied pair; BFP wires never share
 * storage. */
void reluForward(layer_t *reluLayer, tensor_t *input, tensor_t *output, const layerExec_t *exec);
void reluBackward(layer_t *reluLayer, tensor_t *forwardInput, tensor_t *loss, tensor_t *propLoss,
                  const layerExec_t *exec);

void reluCalcOutputShape(layer_t *reluLayer, shape_t *inputShape, shape_t *outputShape);

#endif // ENV5_RUNTIME_RELU_H
