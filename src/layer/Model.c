#define SOURCE_FILE "MODEL"

#include "Model.h"
#include "OpTable.h"

uint8_t modelNodeInputs(const model_t *m, size_t j, uint16_t ids[MODEL_MAX_ARITY]) {
    if (m->inputStart == NULL) {
        ids[0] = (uint16_t)j;
        return 1;
    }
    uint16_t begin = m->inputStart[j];
    uint8_t count = (uint8_t)(m->inputStart[j + 1] - begin);
    for (uint8_t k = 0; k < count; k++) {
        ids[k] = m->inputIds[begin + k];
    }
    return count;
}

bool modelIsChain(const model_t *m) {
    if (m->inputStart == NULL) {
        return true;
    }
    for (size_t j = 0; j <= m->numLayers; j++) {
        if (m->inputStart[j] != j) {
            return false;
        }
    }
    for (size_t j = 0; j < m->numLayers; j++) {
        if (m->inputIds[j] != j) {
            return false;
        }
    }
    return true;
}

void modelShapePass(const model_t *m, const shape_t *input, shape_t *shapes, size_t maxRank) {
    (void)maxRank;
    shapes[0].numberOfDimensions = input->numberOfDimensions;
    for (size_t d = 0; d < input->numberOfDimensions; d++) {
        shapes[0].dimensions[d] = input->dimensions[d];
        shapes[0].orderOfDimensions[d] = input->orderOfDimensions[d];
    }
    for (size_t j = 0; j < m->numLayers; j++) {
        uint16_t ids[MODEL_MAX_ARITY];
        uint8_t k = modelNodeInputs(m, j, ids);
        const shape_t *in[MODEL_MAX_ARITY];
        for (uint8_t i = 0; i < k; i++) {
            in[i] = &shapes[ids[i]];
        }
        layer_t *layer = m->layers[j];
        opFunctions[layer->type].shape(layer, in, k, &shapes[j + 1]);
    }
}
