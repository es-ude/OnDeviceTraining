#ifndef BS_SCHEDULER_H
#define BS_SCHEDULER_H

#include <stddef.h>

#include "DataLoader.h"
#include "Optimizer.h"

/*! Batch-size schedulers, the batch analogue of LrScheduler (#327):
 *  where StepLR/ExponentialLR multiply the LR by gamma, these DIVIDE the
 *  batch size by gamma, so lr/batch follows the same trajectory as the LR
 *  scheduler at the same gamma. Caller-owned, zero allocation. baseBs is
 *  captured ONCE at init from dataLoader->batchSize; every bsSchedulerStep()
 *  computes the closed form FROM baseBs and writes dataLoader->batchSize
 *  absolutely. Optional LR compensation: with a non-NULL optimizer, baseLr
 *  is captured ONCE via getLr and each step also writes
 *  lr = baseLr * appliedBatch / exactBatch through setLr, which keeps
 *  lr/batch on the un-rounded, un-capped trajectory.
 *
 *  CONTRACT: step only at an epoch boundary. getBatch addresses samples as
 *  index * batchSize and trainingEpochDefault derives numberOfBatches from
 *  batchSize once per epoch; a mid-epoch change corrupts the sample walk.
 *  The tail datasetSize % batchSize is dropped every epoch (pre-existing),
 *  so a growing batch drops a growing tail (< batchSize samples).
 *
 *  Double math, float/uint16_t cast only at the write, like LrScheduler. */

typedef enum { STEP_BS, EXPONENTIAL_BS } bsSchedulerType_t;

typedef struct bsScheduler {
    bsSchedulerType_t type;
    dataLoader_t *dataLoader; /* whose batchSize is written */
    optimizer_t *optimizer;   /* NULLable: non-NULL enables LR compensation */
    size_t baseBs;            /* captured at init */
    float baseLr;             /* captured at init via getLr; 0 when optimizer == NULL */
    size_t maxBatchSize;
    size_t lastEpoch; /* counts step() calls, like torch last_epoch */
    union {
        struct {
            size_t stepSize;
            float gamma;
        } stepBs;
        struct {
            float gamma;
        } exponentialBs;
    } params;
} bsScheduler_t;

/*! batch = clamp(round(baseBs / gamma^floor(lastEpoch / stepSize)), 1, maxBatchSize) */
void stepBsInit(bsScheduler_t *sched, dataLoader_t *dataLoader, optimizer_t *optimizerOrNull,
                size_t stepSize, float gamma, size_t maxBatchSize);

/*! batch = clamp(round(baseBs / gamma^lastEpoch), 1, maxBatchSize) */
void exponentialBsInit(bsScheduler_t *sched, dataLoader_t *dataLoader, optimizer_t *optimizerOrNull,
                       float gamma, size_t maxBatchSize);

/*! lastEpoch++ -> compute -> write dataLoader->batchSize (and the LR if compensating). */
void bsSchedulerStep(bsScheduler_t *sched);

/*! The batch bsSchedulerStep writes when it advances lastEpoch to `epoch`
 *  (epoch 0 = baseBs), from the SAME computation -- closed form, rounding,
 *  clamp and non-finite fail-fast -- so a peek can never disagree with the
 *  step (#152 D8). Pure: never touches lastEpoch, the loader or the LR.
 *  trainingRun walks the whole schedule through it before epoch 0 to check
 *  every scheduled batch against microBatchSize. */
size_t bsSchedulerBatchSizeAt(const bsScheduler_t *sched, size_t epoch);

#endif // BS_SCHEDULER_H
