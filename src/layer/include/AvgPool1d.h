#ifndef ODT_AVG_POOL_1D_H
#define ODT_AVG_POOL_1D_H

#include <stdbool.h>
#include <stdlib.h>

#include "ArithmeticType.h"
#include "Kernel.h"
#include "Layer.h"
#include "Tensor.h"

typedef struct avgPool1dConfig {
    kernel_t *kernel;
    arithmetic_t forwardMath;
    arithmetic_t propLossMath;
    quantization_t *outputQ;
    quantization_t *propLossQ;
    bool ownsQuantizations;
} avgPool1dConfig_t;

void initAvgPool1dConfig(avgPool1dConfig_t *cfg, kernel_t *kernel, quantization_t *forwardQ,
                         quantization_t *propLossQ);

void avgPool1dForward(layer_t *layer, tensor_t *input, tensor_t *output, const layerExec_t *exec);

void avgPool1dBackward(layer_t *layer, tensor_t *forwardInput, tensor_t *lossGrad,
                       tensor_t *propLoss, const layerExec_t *exec);
void avgPool1dBackwardFloat(layer_t *layer, tensor_t *forwardInput, tensor_t *lossGrad,
                            tensor_t *propLoss);

void avgPool1dCalcOutputShape(layer_t *layer, shape_t *inputShape, shape_t *outputShape);

#endif // ODT_AVG_POOL_1D_H
