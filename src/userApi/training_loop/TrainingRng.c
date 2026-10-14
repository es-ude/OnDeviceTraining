#define SOURCE_FILE "TRAINING_RNG"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "Common.h"
#include "RNG.h"
#include "TrainingLoopApi.h"

static void requireState(const trainingRng_t *rng, const char *helper) {
    if (rng == NULL) {
        PRINT_ERROR("%s: rng is NULL", helper);
        exit(1);
    }
}

void trainingRngInit(trainingRng_t *rng, uint64_t seed) {
    requireState(rng, "trainingRngInit");
    rng->root = rngKeyFromSeed(seed);
    rng->step = 0;
    rng->epoch = 0;
}

rngKey_t trainingRngStepKey(const trainingRng_t *rng) {
    requireState(rng, "trainingRngStepKey");
    return rngFold(rng->root, RNG_TAG_STEP, rng->step);
}

rngKey_t trainingRngCallKey(rngKey_t parentKey, size_t callIndex) {
    /* rngFold's index is 32 bits wide: a larger index would wrap onto the key
     * of an earlier call. */
    if (callIndex > UINT32_MAX) {
        PRINT_ERROR("trainingRngCallKey: call index exceeds UINT32_MAX");
        exit(1);
    }
    return rngFold(parentKey, RNG_TAG_CALL, (uint32_t)callIndex);
}

rngKey_t trainingRngUpdateKey(const trainingRng_t *rng) {
    requireState(rng, "trainingRngUpdateKey");
    return rngFold(rng->root, RNG_TAG_UPDATE, rng->step);
}

rngKey_t trainingRngEvalKey(const trainingRng_t *rng) {
    requireState(rng, "trainingRngEvalKey");
    return rngFold(rng->root, RNG_TAG_EVAL, rng->epoch);
}

void trainingRngAdvance(trainingRng_t *rng) {
    requireState(rng, "trainingRngAdvance");
    if (rng->step == UINT32_MAX) {
        PRINT_ERROR("trainingRngAdvance: step counter exhausted");
        exit(1);
    }
    rng->step++;
}

void trainingRngAdvanceEpoch(trainingRng_t *rng) {
    requireState(rng, "trainingRngAdvanceEpoch");
    if (rng->epoch == UINT32_MAX) {
        PRINT_ERROR("trainingRngAdvanceEpoch: epoch counter exhausted");
        exit(1);
    }
    rng->epoch++;
}

rngKey_t trainingRngShuffleKey(const trainingRng_t *rng, bool perEpoch) {
    requireState(rng, "trainingRngShuffleKey");
    return rngFold(rng->root, RNG_TAG_SHUFFLE, perEpoch ? rng->epoch : 0u);
}

rngKey_t trainingRngShuffleEvalKey(const trainingRng_t *rng) {
    requireState(rng, "trainingRngShuffleEvalKey");
    return rngFold(rng->root, RNG_TAG_SHUFFLE_EVAL, 0u);
}

rngKey_t trainingRngInitKey(const trainingRng_t *rng, size_t layerIndex) {
    requireState(rng, "trainingRngInitKey");
    /* rngFold's index is 32 bits wide: a larger index would wrap onto the key
     * of an earlier layer. */
    if (layerIndex > UINT32_MAX) {
        PRINT_ERROR("trainingRngInitKey: layer index exceeds UINT32_MAX");
        exit(1);
    }
    return rngFold(rng->root, RNG_TAG_INIT, (uint32_t)layerIndex);
}

const trainingCall_t *trainingCallForIndex(const trainingCall_t *call, size_t callIndex,
                                           trainingCall_t *next, rngKey_t *callKey) {
    if (call == NULL) {
        return NULL;
    }
    const rngKey_t *parentKey = call->key;
    *next = *call;
    next->key = NULL;
    if (parentKey != NULL) {
        *callKey = trainingRngCallKey(*parentKey, callIndex);
        next->key = callKey;
    }
    return next;
}
