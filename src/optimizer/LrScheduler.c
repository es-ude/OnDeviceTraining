#define SOURCE_FILE "LR_SCHEDULER"

#include <math.h>
#include <stdlib.h>

#include "Common.h"
#include "LrScheduler.h"
#include "Optimizer.h"

/* Formulas mirror torch's closed-form expression order verbatim so the
 * double intermediates round identically before the float cast at setLr. */

static float stepLrComputeLr(const lrScheduler_t *sched) {
    double exponent = (double)(sched->lastEpoch / sched->params.stepLr.stepSize);
    return (float)((double)sched->baseLr * pow((double)sched->params.stepLr.gamma, exponent));
}

static float exponentialLrComputeLr(const lrScheduler_t *sched) {
    return (float)((double)sched->baseLr *
                   pow((double)sched->params.exponentialLr.gamma, (double)sched->lastEpoch));
}

static float cosineAnnealingLrComputeLr(const lrScheduler_t *sched) {
    double etaMin = (double)sched->params.cosineAnnealingLr.etaMin;
    return (float)(etaMin + ((double)sched->baseLr - etaMin) *
                                (1.0 + cos(M_PI * (double)sched->lastEpoch /
                                           (double)sched->params.cosineAnnealingLr.tMax)) /
                                2.0);
}

static float linearWarmupLrComputeLr(const lrScheduler_t *sched) {
    size_t warmupEpochs = sched->params.linearWarmupLr.warmupEpochs;
    if (sched->lastEpoch < warmupEpochs) {
        double startFactor = (double)sched->params.linearWarmupLr.startFactor;
        double fraction = (double)sched->lastEpoch / (double)warmupEpochs;
        return (float)((double)sched->baseLr * (startFactor + (1.0 - startFactor) * fraction));
    }
    const lrScheduler_t *main = sched->params.linearWarmupLr.main;
    if (main == NULL) {
        return sched->baseLr;
    }
    /* Delegate via a LOCAL COPY rather than mutating main->lastEpoch in
     * place: computeLr only takes a const pointer, and a copy means the
     * caller's main scheduler is never even transiently modified (no
     * restore-on-early-return risk, and safe if another thread were ever
     * reading main concurrently -- MCUs here are not assumed single-
     * threaded). Mirrors torch's SequentialLR handing the post-milestone
     * scheduler epoch (lastEpoch - warmupEpochs) as its OWN local_epoch. */
    lrScheduler_t localMain = *main;
    localMain.lastEpoch = sched->lastEpoch - warmupEpochs;
    return lrSchedulerFunctions[localMain.type].computeLr(&localMain);
}

lrSchedulerFunctions_t lrSchedulerFunctions[] = {
    [STEP_LR] = {.computeLr = stepLrComputeLr},
    [EXPONENTIAL_LR] = {.computeLr = exponentialLrComputeLr},
    [COSINE_ANNEALING_LR] = {.computeLr = cosineAnnealingLrComputeLr},
    [LINEAR_WARMUP_LR] = {.computeLr = linearWarmupLrComputeLr},
};

static void initCommon(lrScheduler_t *sched, optimizer_t *optimizer, lrSchedulerType_t type) {
    if (optimizer == NULL) {
        PRINT_ERROR("lrScheduler init: optimizer must not be NULL");
        exit(1);
    }
    sched->type = type;
    sched->optimizer = optimizer;
    sched->baseLr = optimizerFunctions[optimizer->type].getLr(optimizer);
    sched->lastEpoch = 0;
}

void stepLrInit(lrScheduler_t *sched, optimizer_t *optimizer, size_t stepSize, float gamma) {
    if (stepSize < 1) {
        PRINT_ERROR("stepLrInit: stepSize must be >= 1");
        exit(1);
    }
    if (!isfinite(gamma)) {
        PRINT_ERROR("stepLrInit: gamma must be finite");
        exit(1);
    }
    initCommon(sched, optimizer, STEP_LR);
    sched->params.stepLr.stepSize = stepSize;
    sched->params.stepLr.gamma = gamma;
}

void exponentialLrInit(lrScheduler_t *sched, optimizer_t *optimizer, float gamma) {
    if (!isfinite(gamma)) {
        PRINT_ERROR("exponentialLrInit: gamma must be finite");
        exit(1);
    }
    initCommon(sched, optimizer, EXPONENTIAL_LR);
    sched->params.exponentialLr.gamma = gamma;
}

void cosineAnnealingLrInit(lrScheduler_t *sched, optimizer_t *optimizer, size_t tMax,
                           float etaMin) {
    if (tMax < 1) {
        PRINT_ERROR("cosineAnnealingLrInit: tMax must be >= 1");
        exit(1);
    }
    if (!isfinite(etaMin)) {
        PRINT_ERROR("cosineAnnealingLrInit: etaMin must be finite");
        exit(1);
    }
    initCommon(sched, optimizer, COSINE_ANNEALING_LR);
    sched->params.cosineAnnealingLr.tMax = tMax;
    sched->params.cosineAnnealingLr.etaMin = etaMin;
}

void linearWarmupLrInit(lrScheduler_t *sched, optimizer_t *optimizer, size_t warmupEpochs,
                        float startFactor, lrScheduler_t *mainSched) {
    if (warmupEpochs < 1) {
        PRINT_ERROR("linearWarmupLrInit: warmupEpochs must be >= 1");
        exit(1);
    }
    if (!isfinite(startFactor) || startFactor <= 0.0f || startFactor > 1.0f) {
        PRINT_ERROR("linearWarmupLrInit: startFactor must be finite and in (0, 1]");
        exit(1);
    }
    if (mainSched != NULL && mainSched->optimizer != optimizer) {
        PRINT_ERROR("linearWarmupLrInit: mainSched is wired to a different optimizer than the "
                    "one passed to linearWarmupLrInit (#327)");
        exit(1);
    }
    initCommon(sched, optimizer, LINEAR_WARMUP_LR);
    sched->params.linearWarmupLr.warmupEpochs = warmupEpochs;
    sched->params.linearWarmupLr.startFactor = startFactor;
    sched->params.linearWarmupLr.main = mainSched;
    /* torch LinearLR applies its factor at construction (its own last_epoch
     * == 0 closed form is exactly startFactor) -- mirror that by writing
     * through setLr now, rather than waiting for the first lrSchedulerStep().
     * initCommon (above) already captured baseLr via getLr BEFORE this write,
     * so mainSched (if initialized earlier, per the ordering requirement
     * documented on this function) is unaffected. */
    float initLr = (float)((double)sched->baseLr * (double)startFactor);
    optimizerFunctions[optimizer->type].setLr(optimizer, initLr);
}

void lrSchedulerStep(lrScheduler_t *sched) {
    sched->lastEpoch++;
    float lr = lrSchedulerFunctions[sched->type].computeLr(sched);
    optimizerFunctions[sched->optimizer->type].setLr(sched->optimizer, lr);
}
