#ifndef TRAINING_EPOCH_DEFAULT_H
#define TRAINING_EPOCH_DEFAULT_H

#include "TrainingLoopApi.h"

/*! One epoch: datasetSize / batchSize macro batches, each trained by
 *  trainingBatchDefault in chunks of microBatchSize rows (0 means 1), then
 *  mean-scaled (backwardReduction == REDUCTION_MEAN), stepped via
 *  optimizerStep and zeroed. dataLoader->batchSize must be divisible by
 *  microBatchSize (trainingRun checks it); fails fast when no batch can be
 *  formed. Returns the mean of the per-batch losses. Derives each batch's
 *  step key and each optimizer update's key from call->rng (call NULLable). */
float trainingEpochDefault(const model_t *model, lossConfig_t lossConfig, dataLoader_t *dataLoader,
                           optimizer_t *optimizer, calculateGradsFn_t calculateGradsFn,
                           reduction_t forwardReduction, size_t microBatchSize,
                           const trainingCall_t *call);

#endif // TRAINING_EPOCH_DEFAULT_H
