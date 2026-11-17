#ifndef ODT_DROPOUT_API_H
#define ODT_DROPOUT_API_H

#include "Dropout.h"
#include "Layer.h"
#include "Tensor.h"

/*! Builds a Dropout layer. Borrows `forwardQ`, `backwardQ`, and `mask`
 *  (caller retains ownership; ownsQuantizations = false). `mask` must be a
 *  pre-allocated BOOL tensor whose element count equals the layer input's.
 *  Dropout copies SYM_INT32 codes verbatim: declare forwardQ at least as wide
 *  as the input wire and backwardQ at least as wide as the dx it receives.
 *  Setup (scheduler init, inference()) rejects a narrower one; the framework
 *  trusts the declared widths and does not re-check a width edited after
 *  setup. */
layer_t *dropoutLayerInit(float p, tensor_t *mask, quantization_t *forwardQ,
                          quantization_t *backwardQ);

/*! Frees the layer_t + its config wrappers. Does NOT free the borrowed mask or
 *  quantizations (ownsQuantizations is false). */
void freeDropoutLayer(layer_t *dropoutLayer);

#endif // ODT_DROPOUT_API_H
