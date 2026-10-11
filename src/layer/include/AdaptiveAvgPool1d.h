#ifndef ODT_ADAPTIVE_AVG_POOL_1D_H
#define ODT_ADAPTIVE_AVG_POOL_1D_H

#include <stdbool.h>
#include <stdlib.h>

#include "ArithmeticType.h"
#include "Layer.h"
#include "Tensor.h"

typedef struct adaptiveAvgPool1dConfig {
    size_t outputSize;
    arithmetic_t forwardMath;
    arithmetic_t propLossMath;
    quantization_t *outputQ;
    quantization_t *propLossQ;
    bool ownsQuantizations;
} adaptiveAvgPool1dConfig_t;

void initAdaptiveAvgPool1dConfig(adaptiveAvgPool1dConfig_t *cfg, size_t outputSize,
                                 quantization_t *forwardQ, quantization_t *propLossQ);

void adaptiveAvgPool1dForward(layer_t *layer, tensor_t *input, tensor_t *output,
                              const layerExec_t *exec);

void adaptiveAvgPool1dBackward(layer_t *layer, tensor_t *forwardInput, tensor_t *lossGrad,
                               tensor_t *propLoss, const layerExec_t *exec);
void adaptiveAvgPool1dBackwardFloat(layer_t *layer, tensor_t *forwardInput, tensor_t *lossGrad,
                                    tensor_t *propLoss);

void adaptiveAvgPool1dCalcOutputShape(layer_t *layer, shape_t *inputShape, shape_t *outputShape);

#endif // ODT_ADAPTIVE_AVG_POOL_1D_H
