#ifndef ENV5_RUNTIME_RELU_H
#define ENV5_RUNTIME_RELU_H

#include "Tensor.h"

typedef struct layer layer_t;

typedef struct reluConfig
{
    quantization_t* forwardQ;
    quantization_t* backwardQ;
} reluConfig_t;

void reluInit(reluConfig_t* reluConfig, quantization_t* forwardQ, quantization_t* backwardQ);

void reluForwardFloat(tensor_t *input, tensor_t *output);
void reluForwardAsym(tensor_t *input, tensor_t *output);
void reluForward(layer_t* reluLayer, tensor_t* input, tensor_t* output);

void reluBackwardFloat(tensor_t *forwardInput, tensor_t *loss, tensor_t *propLoss);
void reluBackwardAsym(tensor_t *forwardInput, tensor_t *loss, tensor_t *propLoss);
void reluBackward(layer_t* reluLayer, tensor_t* forwardInput, tensor_t* loss, tensor_t* propLoss);

void reluCalcOutputShape(layer_t* reluLayer, shape_t* inputShape, shape_t* outputShape);

#endif // ENV5_RUNTIME_RELU_H
