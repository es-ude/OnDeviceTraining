#ifndef ODT_DROPOUT_API_H
#define ODT_DROPOUT_API_H

#include "Dropout.h"
#include "Layer.h"
#include "Tensor.h"

/*! Builds a Dropout layer. Borrows `forwardQ`, `backwardQ`, and `mask`
 *  (caller retains ownership; ownsQuantizations = false). `mask` must be a
 *  pre-allocated BOOL tensor holding microBatchSize x per-sample elements (the
 *  training input's count; a training forward or backward on any other count
 *  exits). Evaluation never reads the mask, so an eval chunk of any row count
 *  is fine. */
layer_t *dropoutLayerInit(float p, tensor_t *mask, quantization_t *forwardQ,
                          quantization_t *backwardQ);

/*! Frees the layer_t and its config; frees the quantizations only when the config owns them
 *  (ownsQuantizations). Never frees the mask. */
void freeDropoutLayer(layer_t *dropoutLayer);

#endif // ODT_DROPOUT_API_H
