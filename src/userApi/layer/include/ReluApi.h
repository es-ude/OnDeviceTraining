#ifndef ODT_RELU_H
#define ODT_RELU_H

#include "Layer.h"
#include "LayerQuant.h"

/*! Borrowing variant: the layer keeps lq->outputQ / lq->propLossQ, which must outlive it.
 *  Use when the quantizations are shared or long-lived. Exits on unsupported arithmetic. */
layer_t *reluLayerInit(layerQuant_t *lq);

/*! Owning variant: deep-copies both quantizations, so the caller may drop them at once;
 *  freeReluLayer frees the copies. */
layer_t *reluLayerInitOwning(layerQuant_t *lq);

/*! Tears down the layer. Reads config->ownsQuantizations to decide whether
 *  to also free the two quantization_t* and their qConfigs. */
void freeReluLayer(layer_t *reluLayer);

#endif // ODT_RELU_H
