#define SOURCE_FILE "LAYER_EXEC"

#include <stddef.h>

#include "LayerExec.h"

const rngKey_t *layerExecPhaseKey(const layerExec_t *exec, uint32_t phaseTag, rngKey_t *storage) {
    if (exec == NULL) {
        return NULL;
    }
    *storage = rngFold(exec->key, phaseTag, 0u);
    return storage;
}
