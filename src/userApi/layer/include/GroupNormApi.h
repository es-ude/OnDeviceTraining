#ifndef ODT_GROUPNORM_API_H
#define ODT_GROUPNORM_API_H

#include <stddef.h>

#include "GroupNorm.h"
#include "Layer.h"
#include "LayerCommon.h"
#include "LayerQuant.h"

typedef struct groupNormInit {
    size_t numGroups;   /* REQUIRED — G, must be > 0 and divide numChannels */
    size_t numChannels; /* REQUIRED — C, must be > 0 */
    float eps;          /* 0 -> default 1e-5 */
    /* affine is always on: gamma and beta exist, shape [C]. */
    trainable_t trainable; /* zero-init → TRAINABLE_DEFAULT (trainable) */
} groupNormInit_t;

/*! Borrowing factory: stores lq->outputQ / lq->propLossQ verbatim (the caller keeps
 *  ownership; they must outlive the layer). Allocates gamma = 1 and beta = 0 of shape
 *  [numChannels] in lq->weightStorage / lq->biasStorage (FLOAT32, SYM_INT32 or BFP; a
 *  grouped BFP template keeps its groupSize, which must divide numChannels) and, unless
 *  frozen, their grads in lq->weightGradStorage / lq->biasGradStorage, FLOAT32 when NULL.
 *  Use when: the quantization configs are shared/long-lived. */
layer_t *groupNormLayerInit(groupNormInit_t *init, layerQuant_t *lq);

/*! Owning factory: as groupNormLayerInit, but deep-copies lq->outputQ / lq->propLossQ
 *  (freeGroupNormLayer tears the copies down). Use when: the configs are stack-locals
 *  or one-off. */
layer_t *groupNormLayerInitOwning(groupNormInit_t *init, layerQuant_t *lq);

/*! Frees a factory-built layer: parameters and wrappers; an owning layer also frees
 *  outputQ/propLossQ. */
void freeGroupNormLayer(layer_t *layer);

#endif // ODT_GROUPNORM_API_H
