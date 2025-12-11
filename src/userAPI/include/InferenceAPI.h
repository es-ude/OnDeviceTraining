#ifndef INFERENCE_H
#define INFERENCE_H

#include "Tensor.h"
#include "Layer.h"

tensor_t *inference(layer_t **model, size_t numberOfLayers, tensor_t *input);

#endif //INFERENCE_H
