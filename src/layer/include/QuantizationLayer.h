#ifndef QUANTIZATIONLAYER_H
#define QUANTIZATIONLAYER_H

#include "Tensor.h"

void quantizationForward(tensor_t *input, tensor_t *output);

void quantizationBackward(tensor_t *input, tensor_t *output);

#endif //QUANTIZATIONLAYER_H
