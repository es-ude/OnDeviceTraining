#ifndef ODT_DROPOUT_API_H
#define ODT_DROPOUT_API_H

#include "Dropout.h"
#include "Layer.h"
#include "Tensor.h"

/*! Builds a Dropout layer. Borrows `forwardQ`, `backwardQ`, and `mask`
 *  (caller retains ownership; ownsQuantizations = false). `mask` must be a
 *  pre-allocated BOOL tensor whose element count equals the layer input's. */
layer_t *dropoutLayerInit(float p, tensor_t *mask, quantization_t *forwardQ,
                          quantization_t *backwardQ);

/*! Frees the layer_t + its config wrappers. Does NOT free the borrowed mask or
 *  quantizations (ownsQuantizations is false). */
void freeDropoutLayer(layer_t *dropoutLayer);

#endif // ODT_DROPOUT_API_H
