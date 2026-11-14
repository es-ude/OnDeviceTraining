#ifndef LAYER_WEIGHTS_API_H
#define LAYER_WEIGHTS_API_H

#include "Layer.h"

/*! Overwrite the (already-initialized) weight and bias tensors of `layer`
 *  from the provided buffers (gamma/beta for the norm layers). Dispatches by
 *  layer->type. The checks below run before the first write.
 *
 *  Errors (PRINT_ERROR + exit):
 *   - layer == NULL or weightData == NULL
 *   - layer has no parameters (ReLU/Softmax/Flatten/pooling/Dropout/
 *     Quantization, BatchNorm1d without affine)
 *   - biasData == NULL but layer has a bias or beta
 *   - biasData != NULL but layer was built with bias disabled
 *
 *  Buffer length is the responsibility of the caller — it must match the
 *  tensor allocation the factory performed for this layer. */
void layerLoadWeights(layer_t *layer, float *weightData, float *biasData);

#endif /* LAYER_WEIGHTS_API_H */
