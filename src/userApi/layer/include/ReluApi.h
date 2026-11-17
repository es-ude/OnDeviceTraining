#ifndef ODT_RELU_H
#define ODT_RELU_H

#include "Layer.h"
#include "LayerQuant.h"

/*! Borrowing variant — stores lq->outputQ and lq->propLossQ as the
 *  layer's outputQ / propLossQ pointers. Caller retains ownership of lq.
 *  Use when: the quantizations are shared/long-lived (e.g. reused across
 *  several layers) and the caller manages their lifetime — they must
 *  outlive the layer. */
layer_t *reluLayerInit(layerQuant_t *lq);

/*! Owning variant — deep-copies lq->outputQ and lq->propLossQ.
 *  Caller can drop lq + quantizations immediately.
 *  Use when: the quantizations are stack-locals or one-off configs and you
 *  want fire-and-forget teardown (freeReluLayer tears them down too). */
layer_t *reluLayerInitOwning(layerQuant_t *lq);

/*! Tears down the layer. Reads config->ownsQuantizations to decide whether
 *  to also free the two quantization_t* and their qConfigs. */
void freeReluLayer(layer_t *reluLayer);

#endif // ODT_RELU_H
