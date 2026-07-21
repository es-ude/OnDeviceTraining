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
    /* affine is ALWAYS ON for v1 (gamma + beta present). */
    trainable_t trainable; /* zero-init → TRAINABLE_DEFAULT (trainable) */
} layerNormInit_t;

/*! Borrowing factory — stores lq->outputQ / lq->propLossQ verbatim
 *  into the config (ownsQuantizations=false; caller retains ownership of the
 *  storage quantizations). Allocates gamma (init 1) and beta (init 0) parameter_t
 *  (storage dtype = lq->weightStorage / lq->biasStorage) and their grads via
 *  gradInit(param, lq->weightGradStorage ?: FLOAT32, NULL) (resp.
 *  biasGradStorage for beta) — PR1c: the NULL-knob default is a hard-pinned
 *  FLOAT32, not lq->propLossQ. Copies normalizedShape into factory-owned
 *  memory.
 *  Use when: outputQ/propLossQ are shared/long-lived (e.g. reused across
 *  several layers) and the caller manages their lifetime — they must
 *  outlive the layer. */
layer_t *layerNormLayerInit(layerNormInit_t *init, layerQuant_t *lq);

/*! Owning factory — deep-copies lq->outputQ / lq->propLossQ into the
 *  config (ownsQuantizations=true; the caller may drop its outputQ/propLossQ
 *  pointers immediately). Identical gamma/beta allocation + normalizedShape copy
 *  as the Borrowing variant. Mirrors linearLayerInitOwning.
 *  Use when: outputQ/propLossQ are stack-locals or one-off configs and you
 *  want fire-and-forget teardown (freeLayerNormLayer tears them down too). */
layer_t *layerNormLayerInitOwning(layerNormInit_t *init, layerQuant_t *lq);

/*! Frees the parameter_t (gamma/beta + grads + shapes), the copied
 *  normalizedShape, and the layer/config wrappers. ALWAYS frees gamma/beta.
 *  If ownsQuantizations is true (Owning factory), also frees forwardQ and
 *  backwardQ (qConfig + struct, with a backwardQ != forwardQ dedup guard);
 *  if false (Borrowing factory), leaves the caller-owned math quantizations
 *  untouched. */
void freeLayerNormLayer(layer_t *layer);

#endif // ODT_LAYERNORM_API_H
