#ifndef ODT_PARAMETER_WALK_H
#define ODT_PARAMETER_WALK_H

#include <stddef.h>

#include "Layer.h"
#include "Model.h"
#include "Tensor.h"

typedef enum {
    PARAM_ROLE_WEIGHT,
    PARAM_ROLE_BIAS,
    PARAM_ROLE_NORM_SCALE,
    PARAM_ROLE_NORM_SHIFT
} paramRole_t;

typedef struct paramRef {
    size_t layerIndex; /* index in model->layers */
    const layer_t *layer;
    paramRole_t role;
    parameter_t *parameter;
} paramRef_t;

/* ref is valid only for the duration of the call: copy it to keep it. */
typedef void (*paramVisitFn_t)(const paramRef_t *ref, void *ctx);

/* Visits every trainable parameter in optimizer order: layers in model order, frozen layers
 * skipped, weight before bias, scale before shift; bias-less layers and non-affine BN contribute
 * what they hold. The one source of that order: anything that sizes, fills or partitions the
 * optimizer state walks it here. Exits on an unknown layer type. */
void forEachTrainableParameter(const model_t *m, paramVisitFn_t visit, void *ctx);

#endif // ODT_PARAMETER_WALK_H
