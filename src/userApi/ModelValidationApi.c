#define SOURCE_FILE "MODEL_VALIDATION_API"

#include <stdbool.h>
#include <stddef.h>

#include "Common.h"
#include "Layer.h"
#include "ModelValidationApi.h"

bool validateModelQuantization(const model_t *m) {
    if (m == NULL || m->layers == NULL) {
        PRINT_ERROR("validateModelQuantization: model is NULL");
        return false;
    }
    bool valid = true;
    for (size_t i = 0; i < m->numLayers; i++) {
        if (m->layers[i] == NULL) {
            PRINT_ERROR("validateModelQuantization: model[%zu] is NULL", i);
            valid = false;
        }
    }
    return valid;
}
