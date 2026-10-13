#define SOURCE_FILE "OPTIMIZER_KEYS"

#include <stddef.h>

#include "OptimizerKeys.h"

const rngKey_t *optimizerUpdateOpKey(const rngKey_t *updateKey, uint32_t paramIndex,
                                     uint32_t opIndex, rngKey_t *storage) {
    if (updateKey == NULL) {
        return NULL;
    }
    *storage = rngFold(rngFold(*updateKey, RNG_TAG_PARAM, paramIndex), RNG_TAG_OP, opIndex);
    return storage;
}

rngStream_t *optimizerScaleStream(const rngKey_t *updateKey, uint32_t paramIndex,
                                  rngStream_t *storage) {
    if (updateKey == NULL) {
        return NULL;
    }
    *storage = rngOpen(rngFold(rngFold(*updateKey, RNG_TAG_PARAM, paramIndex), RNG_TAG_SCALE, 0u));
    return storage;
}
