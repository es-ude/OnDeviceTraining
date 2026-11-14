#ifndef ODT_QUANT_LAYER_API_H
#define ODT_QUANT_LAYER_API_H

#include "Layer.h"
#include "LayerQuant.h"
#include "QuantizationLayer.h"

/*! Borrowing variant: stores lq->outputQ (forward output storage) and lq->propLossQ
 *  (backward output storage); they must outlive the layer, `lq` itself need not. The
 *  arithmetic, weightStorage and biasStorage slots are ignored (pure conversion node, no
 *  parameters). */
layer_t *quantLayerInit(layerQuant_t *lq);

/*! Owning variant: copies outputQ and propLossQ; caller configs may be released after
 *  return. */
layer_t *quantLayerInitOwning(layerQuant_t *lq);

/*! Frees a layer built by either factory, including owned outputQ/propLossQ. */
void freeQuantLayer(layer_t *quantLayer);

#endif // ODT_QUANT_LAYER_API_H
