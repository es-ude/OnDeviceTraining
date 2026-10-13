#ifndef ODT_OPTIMIZER_KEYS_H
#define ODT_OPTIMIZER_KEYS_H

#include <stdint.h>

#include "RNG.h"

/* The keys of one optimizer update, folded from its update key. A NULL update
 * key means the caller supplies no randomness. paramIndex is the parameter's
 * index in optimizer->parameter[]; optimizerStep and scaleOptimizerGradients
 * stop the program before an index could exceed UINT32_MAX. */

/* The key of update op opIndex of parameter paramIndex, for opSpec_t.key:
 * NULL iff updateKey is NULL; otherwise *storage = rngFold(rngFold(*updateKey,
 * RNG_TAG_PARAM, paramIndex), RNG_TAG_OP, opIndex) and the result is storage.
 * opIndex is the optimizer's OP(k) constant. */
const rngKey_t *optimizerUpdateOpKey(const rngKey_t *updateKey, uint32_t paramIndex,
                                     uint32_t opIndex, rngKey_t *storage);

/* The stream of parameter paramIndex's gradient mean-scale: NULL iff
 * updateKey is NULL; otherwise *storage = rngOpen(rngFold(rngFold(*updateKey,
 * RNG_TAG_PARAM, paramIndex), RNG_TAG_SCALE, 0)) and the result is storage. */
rngStream_t *optimizerScaleStream(const rngKey_t *updateKey, uint32_t paramIndex,
                                  rngStream_t *storage);

#endif
