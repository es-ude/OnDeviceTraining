#ifndef MSE_H
#define MSE_H
#include "Tensor.h"

float mseLossForward(tensor_t *output, tensor_t *label);

void mseLossBackwardFloat(tensor_t *modelOutput, tensor_t *label, tensor_t *result);

void mseLossBackwardAsym(tensor_t *modelOutput, tensor_t *label, tensor_t *result);

void mseLossBackward(tensor_t *modelOutput, tensor_t *label, tensor_t *result);


#endif //MSE_H
