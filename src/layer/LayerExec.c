#define SOURCE_FILE "LAYER_EXEC"

#include <stddef.h>
#include <stdlib.h>

#include "Common.h"
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

const layerExec_t *layerExecForLayer(const rngKey_t *callKey, size_t layerIndex,
                                     layerExec_t *storage) {
    if (callKey == NULL) {
        return NULL;
    }
    /* rngFold's index is 32 bits wide: a larger index would wrap onto the key
     * of an earlier layer. */
    if (layerIndex > UINT32_MAX) {
        PRINT_ERROR("layerExecForLayer: layer index exceeds UINT32_MAX");
        exit(1);
    }
    storage->key = rngFold(*callKey, RNG_TAG_LAYER, (uint32_t)layerIndex);
    return storage;
}
