#define SOURCE_FILE "MODEL_VALIDATION_API"

#include <stdbool.h>
#include <stddef.h>

#include "Common.h"
#include "Layer.h"
#include "LayerNorm.h"
#include "Linear.h"
#include "ModelValidationApi.h"

/* Returns the forwardQ of accumulator-range producers (layers whose SYM
 * forward emits raw matmul/post-affine mantissas beyond the int16 norm),
 * NULL for every other layer type. */
static quantization_t *producerForwardQ(layer_t *layer) {
    switch (layer->type) {
    case LINEAR:
        return layer->config->linear->forwardQ;
    case LAYERNORM:
        return layer->config->layerNorm->forwardQ;
    default:
        return NULL;
    }
}

bool validateModelQuantization(layer_t **model, size_t modelSize) {
    if (model == NULL) {
        PRINT_ERROR("validateModelQuantization: model is NULL");
        return false;
    }
    bool valid = true;
    for (size_t i = 0; i < modelSize; i++) {
        if (model[i] == NULL) {
            PRINT_ERROR("validateModelQuantization: model[%zu] is NULL", i);
            valid = false;
            continue;
        }
        quantization_t *forwardQ = producerForwardQ(model[i]);
        if (forwardQ == NULL || forwardQ->type != SYM_INT32) {
            continue; /* not a SYM accumulator-range producer */
        }
        if (i + 1 == modelSize) {
            continue; /* producer in last position: loss boundary, allowed */
        }
        if (model[i + 1] == NULL || model[i + 1]->type != QUANTIZATION) {
            PRINT_ERROR("validateModelQuantization: layer %zu (type %d) emits SYM_INT32 "
                        "accumulator-range output but layer %zu (type %d) is not a "
                        "QUANTIZATION layer — insert a Quant layer to restore the int16 "
                        "inter-layer contract",
                        i, (int)model[i]->type, i + 1, model[i + 1] ? (int)model[i + 1]->type : -1);
            valid = false;
        }
    }
    return valid;
}
