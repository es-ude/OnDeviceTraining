#ifndef ENV5_RUNTIME_RELU_H
#define ENV5_RUNTIME_RELU_H

#include <stdbool.h>

#include "ArithmeticType.h"
#include "Tensor.h"

typedef struct layer layer_t;

typedef struct reluConfig {
    arithmetic_t forwardMath;
    arithmetic_t propLossMath;
    quantization_t *outputQ;
    quantization_t *propLossQ;
    bool ownsQuantizations;
} reluConfig_t;

void reluForwardFloat(tensor_t *input, tensor_t *output);
void reluForwardBfp(tensor_t *input, tensor_t *output);
void reluForward(layer_t *reluLayer, tensor_t *input, tensor_t *output);

void reluBackwardFloat(tensor_t *forwardInput, tensor_t *loss, tensor_t *propLoss);
void reluBackwardBfp(tensor_t *forwardInput, tensor_t *loss, tensor_t *propLoss);
void reluBackward(layer_t *reluLayer, tensor_t *forwardInput, tensor_t *loss, tensor_t *propLoss);

void reluCalcOutputShape(layer_t *reluLayer, shape_t *inputShape, shape_t *outputShape);

#endif // ENV5_RUNTIME_RELU_H
