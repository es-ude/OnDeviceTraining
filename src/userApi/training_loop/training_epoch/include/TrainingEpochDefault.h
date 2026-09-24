#ifndef TRAINING_EPOCH_DEFAULT_H
#define TRAINING_EPOCH_DEFAULT_H

#include "TrainingLoopApi.h"

/*! One epoch: datasetSize / batchSize macro batches, each trained by
 *  trainingBatchDefault in chunks of microBatchSize rows (0 means 1), then
 *  mean-scaled (backwardReduction == REDUCTION_MEAN), stepped via
 *  optimizerStep and zeroed. Fails fast unless dataLoader->batchSize is
 *  divisible by microBatchSize, and when no batch can be formed. Returns the
 *  mean of the per-batch losses. */
float trainingEpochDefault(layer_t **model, size_t modelSize, lossConfig_t lossConfig,
                           dataLoader_t *dataLoader, optimizer_t *optimizer,
                           calculateGradsFn_t calculateGradsFn, reduction_t forwardReduction,
                           size_t microBatchSize);

#endif // TRAINING_EPOCH_DEFAULT_H
