#ifndef ODT_GROUPNORM_API_H
#define ODT_GROUPNORM_API_H

#include <stddef.h>

#include "GroupNorm.h"
#include "Layer.h"
#include "LayerQuant.h"

typedef struct groupNormInit {
    size_t numGroups;   /* REQUIRED — G, must be > 0 and divide numChannels */
    size_t numChannels; /* REQUIRED — C, must be > 0 */
    float eps;          /* 0 -> default 1e-5 */
    /* affine is ALWAYS ON for v1 (gamma + beta present), shape [C]. */
} groupNormInit_t;

/*! Borrowing factory — stores lq->outputQ / lq->propLossQ verbatim
 *  into the config (ownsQuantizations=false; caller retains ownership of the
 *  storage quantizations). Allocates gamma (init 1) and beta (init 0) parameter_t
 *  of shape [numChannels] (storage dtype = lq->weightStorage / lq->biasStorage)
 *  and their grads via gradInit(param, lq->weightGradStorage ?: FLOAT32, NULL)
 *  (resp. biasGradStorage for beta) — the NULL-knob default is a hard-pinned
 *  FLOAT32, not lq->propLossQ.
 *  Use when: outputQ/propLossQ are shared/long-lived (e.g. reused across
 *  several layers) and the caller manages their lifetime — they must
 *  outlive the layer. */
layer_t *groupNormLayerInit(groupNormInit_t *init, layerQuant_t *lq);

/*! Owning factory — deep-copies lq->outputQ / lq->propLossQ into the
 *  config (ownsQuantizations=true; the caller may drop its outputQ/propLossQ
 *  pointers immediately). Identical gamma/beta allocation as the Borrowing
 *  variant. Mirrors layerNormLayerInitOwning.
 *  Use when: outputQ/propLossQ are stack-locals or one-off configs and you
 *  want fire-and-forget teardown (freeGroupNormLayer tears them down too). */
layer_t *groupNormLayerInitOwning(groupNormInit_t *init, layerQuant_t *lq);

/*! Frees the parameter_t (gamma/beta + grads), and the layer/config wrappers.
 *  ALWAYS frees gamma/beta. If ownsQuantizations is true (Owning factory),
 *  also frees outputQ and propLossQ (qConfig + struct, with an
 *  outputQ != propLossQ dedup guard); if false (Borrowing factory), leaves
 *  the caller-owned math quantizations untouched. */
void freeGroupNormLayer(layer_t *layer);

#endif // ODT_GROUPNORM_API_H
