#define SOURCE_FILE "MODEL"

#include "Model.h"

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
