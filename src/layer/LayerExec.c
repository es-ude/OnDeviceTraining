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

const rngKey_t *layerExecOpKey(const layerExec_t *exec, uint32_t phaseTag, uint32_t opIndex,
                               rngKey_t *storage) {
    rngKey_t phase;
    if (layerExecPhaseKey(exec, phaseTag, &phase) == NULL) {
        return NULL;
    }
    *storage = rngFold(phase, RNG_TAG_OP, opIndex);
    return storage;
}

rngStream_t *layerExecPurposeStream(const layerExec_t *exec, uint32_t phaseTag, uint32_t purposeTag,
                                    rngStream_t *storage) {
    rngKey_t phase;
    if (layerExecPhaseKey(exec, phaseTag, &phase) == NULL) {
        return NULL;
    }
    *storage = rngOpen(rngFold(phase, purposeTag, 0u));
    return storage;
}
