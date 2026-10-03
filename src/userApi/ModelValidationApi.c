#define SOURCE_FILE "MODEL_VALIDATION_API"

#include <stdbool.h>
#include <stddef.h>

#include "Common.h"
#include "Layer.h"
#include "ModelValidationApi.h"

/* RETIRED (PR1b.2 D3): this validator used to reject a SYM_INT32
 * accumulator-range producer (Linear/LayerNorm/Conv1d/Conv1dTransposed) not
 * immediately followed by a QUANTIZATION layer — the forward funnel now
 * restores width at the producer's own wire, so that rule has nothing left
 * to catch (a following Quant layer is optional, not required; see
 * docs/conventions/arithmetic-sym.md). The entry point stays for future
 * model-wide rules; today it only guards the model array shape itself. */
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
        }
    }
    return valid;
}
