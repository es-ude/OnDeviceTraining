#ifndef TRAINING_BATCH_DEFAULT_H
#define TRAINING_BATCH_DEFAULT_H

#include <stddef.h>

#include "TrainingLoopApi.h"

/*! Accumulates gradients without stepping the optimizer; consumes sample
 *  wrappers. Requires whole microbatches (m = microBatchSize, 0 means 1).
 *  Stacked samples must be matching dense FLOAT32 tensors; the model's
 *  FLOAT32 eligibility at m > 1 is the caller's (trainingRun checks it).
 *  Returns row-weighted mean loss or total loss. Input views are borrowed for
 *  each callback; call keys derive from chunk indices, and a scheduler in
 *  call->remat must be keyed to [1, ...] at m == 1, [m, ...] at m > 1. */
float trainingBatchDefault(const model_t *model, lossConfig_t lossConfig, batch_t *batch,
                           calculateGradsFn_t calculateGradsFn, reduction_t forwardReduction,
                           size_t microBatchSize, const trainingCall_t *call);

#endif // TRAINING_BATCH_DEFAULT_H
