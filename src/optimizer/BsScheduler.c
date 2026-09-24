#define SOURCE_FILE "BS_SCHEDULER"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#include "BsScheduler.h"
#include "Common.h"
#include "DataLoader.h"
#include "Optimizer.h"

/* Exact (un-rounded, un-capped) target batch at `epoch` (a lastEpoch value),
 * in double. Same expression shape as LrScheduler's closed forms, with gamma
 * DIVIDING. */
static double computeExact(const bsScheduler_t *sched, size_t epoch) {
    switch (sched->type) {
    case STEP_BS: {
        double exponent = (double)(epoch / sched->params.stepBs.stepSize);
        return (double)sched->baseBs / pow((double)sched->params.stepBs.gamma, exponent);
    }
    case EXPONENTIAL_BS:
        return (double)sched->baseBs /
               pow((double)sched->params.exponentialBs.gamma, (double)epoch);
    }
    PRINT_ERROR("bsScheduler: unknown scheduler type %d", (int)sched->type);
    exit(1);
}

/* THE batch computation, shared by bsSchedulerStep (which writes it) and
 * bsSchedulerBatchSizeAt (which only reports it) so the two cannot diverge
 * (#152 D8): closed form -> non-finite guard -> round half away -> clamp to
 * [1, maxBatchSize]. `fn` names the public entry in the message; *exactOut
 * receives the un-rounded target (the LR compensation's divisor). */
static size_t appliedBatchAt(const bsScheduler_t *sched, size_t epoch, const char *fn,
                             double *exactOut) {
    double exact = computeExact(sched, epoch);
    if (!isfinite(exact) || exact <= 0.0) {
        /* gamma^epoch over- or underflowed in double: the target is 0 or inf
         * and the batch/LR trajectory is no longer defined. Fail fast instead
         * of writing a clamped batch with a 0 or inf LR. */
        PRINT_ERROR("%s: exact batch target is not finite and positive at lastEpoch %zu "
                    "(gamma^lastEpoch left the double range); shorten the run or move gamma "
                    "toward 1",
                    fn, epoch);
        exit(1);
    }
    double rounded = round(exact); /* half away from zero (C round), not banker's */
    size_t applied;
    if (rounded < 1.0) {
        applied = 1;
    } else if (rounded > (double)sched->maxBatchSize) {
        applied = sched->maxBatchSize;
    } else {
        applied = (size_t)rounded;
    }
    *exactOut = exact;
    return applied;
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
    if (maxBatchSize > dataLoader->getDatasetSize()) {
        /* A cap the loader can never serve would let the schedule write
         * batchSize > datasetSize: zero batches per epoch, 0/0 loss. */
        PRINT_ERROR("%s: maxBatchSize must be <= the loader's dataset size", fn);
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

size_t bsSchedulerBatchSizeAt(const bsScheduler_t *sched, size_t epoch) {
    double exact;
    return appliedBatchAt(sched, epoch, "bsSchedulerBatchSizeAt", &exact);
}

void bsSchedulerStep(bsScheduler_t *sched) {
    sched->lastEpoch++;
    double exact;
    size_t applied = appliedBatchAt(sched, sched->lastEpoch, "bsSchedulerStep", &exact);
    /* maxBatchSize <= UINT16_MAX (init guard) makes the narrowing safe. */
    sched->dataLoader->batchSize = (uint16_t)applied;

    if (sched->optimizer != NULL) {
        /* lr = baseLr * applied / exact keeps lr/batch = baseLr/exact, i.e. the
         * LR scheduler's ratio, regardless of rounding and the cap. Past the
         * cap this degrades into plain LR decay (baseLr * max / exact) BY
         * DESIGN — do not "fix" it. Written absolutely from baseLr (#327). */
        float lr = (float)((double)sched->baseLr * (double)applied / exact);
        if (!isfinite(lr)) {
            PRINT_ERROR("bsSchedulerStep: compensated learning rate is not finite at lastEpoch %zu "
                        "(baseLr * applied / exact overflowed float)",
                        sched->lastEpoch);
            exit(1);
        }
        optimizerFunctions[sched->optimizer->type].setLr(sched->optimizer, lr);
    }
}
