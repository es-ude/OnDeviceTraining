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

/*! Borrowing variant: borrows outputQ and propLossQ; they must outlive the layer. Parameter
 *  and gradient tensors own their quantization configs; `lq` itself need not outlive the
 *  call. */
layer_t *linearLayerInit(linearInit_t *init, layerQuant_t *lq);

/*! Owning variant: copies outputQ and propLossQ; caller configs may be released after
 *  return. */
layer_t *linearLayerInitOwning(linearInit_t *init, layerQuant_t *lq);

/*! Frees parameters and layer storage, including owned outputQ/propLossQ. */
void freeLinearLayer(layer_t *linearLayer);

#endif // ODT_LINEAR_H
