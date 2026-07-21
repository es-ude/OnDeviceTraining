#ifndef ODT_LINEAR_H
#define ODT_LINEAR_H

#include "Layer.h"
#include "LayerCommon.h"
#include "LayerQuant.h"

typedef struct linearInit {
    /* REQUIRED — factory aborts if 0 */
    size_t inFeatures;
    size_t outFeatures;
    /* OPTIONAL */
    bias_t bias;             /* BIAS_DEFAULT (0) → resolves to true */
    weightInit_t weightInit; /* zero-init → INIT_DEFAULT (PyTorch kaiming a=√5) */
    trainable_t trainable;   /* zero-init → TRAINABLE_DEFAULT (trainable) */
} linearInit_t;

/*! Borrowing variant — factory stores the four quantization_t* pointers from
 *  `lq` directly. Caller retains ownership of `lq` and the quantizations;
 *  `lq` itself can be a compound literal that dies after the call.
 *  Use when: the quantizations are shared/long-lived (e.g. reused across
 *  several layers) and the caller manages their lifetime — they must
 *  outlive the layer. */
layer_t *linearLayerInit(linearInit_t *init, layerQuant_t *lq);

/*! Owning variant — factory deep-copies everything reachable from `lq`.
 *  Caller can free `lq` and all four quantization_t* immediately after
 *  the factory returns. free*Layer will tear down the copies.
 *  Use when: the quantizations are stack-locals or one-off configs and you
 *  want fire-and-forget teardown (freeLinearLayer tears them down too). */
layer_t *linearLayerInitOwning(linearInit_t *init, layerQuant_t *lq);

/*! Tears down everything the factory allocated (parameters, kernel if any,
 *  internal config, layer). Reads config->ownsQuantizations to decide
 *  whether to also tear down the four quantization_t* and their qConfigs. */
void freeLinearLayer(layer_t *linearLayer);

#endif // ODT_LINEAR_H
