#define SOURCE_FILE "BS_SCHEDULER"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#include "BsScheduler.h"
#include "Common.h"
#include "DataLoader.h"
#include "Optimizer.h"

/* Exact (un-rounded, un-capped) target batch at lastEpoch, in double. Same
 * expression shape as LrScheduler's closed forms, with gamma DIVIDING. */
static double computeExact(const bsScheduler_t *sched) {
    switch (sched->type) {
    case STEP_BS: {
        double exponent = (double)(sched->lastEpoch / sched->params.stepBs.stepSize);
        return (double)sched->baseBs / pow((double)sched->params.stepBs.gamma, exponent);
    }
    case EXPONENTIAL_BS:
        return (double)sched->baseBs /
               pow((double)sched->params.exponentialBs.gamma, (double)sched->lastEpoch);
    }
    PRINT_ERROR("bsScheduler: unknown scheduler type %d", (int)sched->type);
    exit(1);
}

/* Shared init-time guards. `fn` is the PUBLIC init function's name so every
 * message names the function the caller actually invoked (PRINT_ERROR itself
 * only prints __FUNCTION__ of this helper). */
static void initCommon(bsScheduler_t *sched, dataLoader_t *dataLoader, optimizer_t *optimizerOrNull,
                       bsSchedulerType_t type, float gamma, size_t maxBatchSize, const char *fn) {
    if (dataLoader == NULL) {
        PRINT_ERROR("%s: dataLoader must not be NULL", fn);
        exit(1);
    }
    if (!isfinite(gamma) || gamma <= 0.0f) {
        /* gamma <= 0 would yield inf/NaN/negative targets and UB on the size_t cast. */
        PRINT_ERROR("%s: gamma must be finite and > 0", fn);
        exit(1);
    }
    if (maxBatchSize < dataLoader->batchSize) {
        PRINT_ERROR("%s: maxBatchSize must be >= the loader's initial batchSize", fn);
        exit(1);
    }
    if (maxBatchSize > UINT16_MAX) {
        PRINT_ERROR("%s: maxBatchSize must fit dataLoader_t.batchSize (uint16_t)", fn);
        exit(1);
    }
    sched->type = type;
    sched->dataLoader = dataLoader;
    sched->optimizer = optimizerOrNull;
    sched->baseBs = dataLoader->batchSize;
    sched->baseLr = (optimizerOrNull != NULL)
                        ? optimizerFunctions[optimizerOrNull->type].getLr(optimizerOrNull)
                        : 0.0f;
    sched->maxBatchSize = maxBatchSize;
    sched->lastEpoch = 0;
}

void stepBsInit(bsScheduler_t *sched, dataLoader_t *dataLoader, optimizer_t *optimizerOrNull,
                size_t stepSize, float gamma, size_t maxBatchSize) {
    if (stepSize < 1) {
        PRINT_ERROR("stepBsInit: stepSize must be >= 1");
        exit(1);
    }
    initCommon(sched, dataLoader, optimizerOrNull, STEP_BS, gamma, maxBatchSize, "stepBsInit");
    sched->params.stepBs.stepSize = stepSize;
    sched->params.stepBs.gamma = gamma;
}

void exponentialBsInit(bsScheduler_t *sched, dataLoader_t *dataLoader, optimizer_t *optimizerOrNull,
                       float gamma, size_t maxBatchSize) {
    initCommon(sched, dataLoader, optimizerOrNull, EXPONENTIAL_BS, gamma, maxBatchSize,
               "exponentialBsInit");
    sched->params.exponentialBs.gamma = gamma;
}

void bsSchedulerStep(bsScheduler_t *sched) {
    sched->lastEpoch++;
    double exact = computeExact(sched);
    double rounded = round(exact); /* half away from zero (C round), not banker's */
    size_t applied;
    if (rounded < 1.0) {
        applied = 1;
    } else if (rounded > (double)sched->maxBatchSize) {
        applied = sched->maxBatchSize;
    } else {
        applied = (size_t)rounded;
    }
    /* maxBatchSize <= UINT16_MAX (init guard) makes the narrowing safe. */
    sched->dataLoader->batchSize = (uint16_t)applied;

    if (sched->optimizer != NULL) {
        /* lr = baseLr * applied / exact keeps lr/batch = baseLr/exact, i.e. the
         * LR scheduler's ratio, regardless of rounding and the cap. Past the
         * cap this degrades into plain LR decay (baseLr * max / exact) BY
         * DESIGN — do not "fix" it. Written absolutely from baseLr (#327). */
        float lr = (float)((double)sched->baseLr * (double)applied / exact);
        optimizerFunctions[sched->optimizer->type].setLr(sched->optimizer, lr);
    }
}
