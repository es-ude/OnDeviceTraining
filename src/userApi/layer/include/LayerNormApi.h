#ifndef ODT_LAYERNORM_API_H
#define ODT_LAYERNORM_API_H

#include <stddef.h>

#include "Layer.h"
#include "LayerCommon.h"
#include "LayerNorm.h"
#include "LayerQuant.h"

typedef struct layerNormInit {
    size_t *normalizedShape; /* REQUIRED — last-D logical dims */
    size_t numNormDims;      /* REQUIRED — D, must be > 0 */
    float eps;               /* 0 → default 1e-5 */
    /* affine is always on: gamma and beta exist, shape normalizedShape. */
    trainable_t trainable; /* zero-init → TRAINABLE_DEFAULT (trainable) */
} layerNormInit_t;

/*! Borrowing factory: stores lq->outputQ / lq->propLossQ verbatim (the caller keeps
 *  ownership; they must outlive the layer). Allocates gamma = 1 and beta = 0 of shape
 *  normalizedShape in lq->weightStorage / lq->biasStorage (FLOAT32, SYM_INT32 or BFP; a
 *  grouped BFP template keeps its groupSize, which must divide the element count) and,
 *  unless frozen, their grads in lq->weightGradStorage / lq->biasGradStorage, FLOAT32
 *  when NULL. Copies normalizedShape, so the caller may free its array.
 *  Use when: the quantization configs are shared/long-lived. */
layer_t *layerNormLayerInit(layerNormInit_t *init, layerQuant_t *lq);

/*! Owning factory: as layerNormLayerInit, but deep-copies lq->outputQ / lq->propLossQ
 *  (freeLayerNormLayer tears the copies down). Use when: the configs are stack-locals
 *  or one-off. */
layer_t *layerNormLayerInitOwning(layerNormInit_t *init, layerQuant_t *lq);

/*! Frees a factory-built layer: parameters, copied normalizedShape and wrappers; an owning
 *  layer also frees outputQ/propLossQ. */
void freeLayerNormLayer(layer_t *layer);

#endif // ODT_LAYERNORM_API_H
