#ifndef ODT_LAYER_EXEC_H
#define ODT_LAYER_EXEC_H

#include <stdint.h>

#include "RNG.h"

/* What a layer or loss call receives besides its tensors: the layer key. The
 * callee folds its own phase from it, so a backward can reach its forward's
 * key when it replays a forward draw. A NULL context means the caller supplies
 * no randomness. Read during the call only. */
typedef struct layerExec {
    rngKey_t key;
} layerExec_t;

/* NULL iff exec is NULL; otherwise *storage = rngFold(exec->key, phaseTag, 0)
 * and the result is storage. The caller owns storage (a stack rngKey_t), so a
 * key pointer it hands on never dangles. */
const rngKey_t *layerExecPhaseKey(const layerExec_t *exec, uint32_t phaseTag, rngKey_t *storage);

#endif
