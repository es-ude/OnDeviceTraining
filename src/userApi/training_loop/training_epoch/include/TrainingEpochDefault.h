#ifndef TRAINING_EPOCH_DEFAULT_H
#define TRAINING_EPOCH_DEFAULT_H

#include "TrainingLoopApi.h"

/*! Wraps sample 0's label with batchViewOf for computeMeanScale -- see
 *  docs/conventions/data-shape.md, "Who adds the batch axis", and
 *  docs/conventions/loss.md, "Microbatch shape". */
float trainingEpochDefault(layer_t **model, size_t modelSize, lossConfig_t lossConfig,
                           dataLoader_t *dataLoader, optimizer_t *optimizer,
                           calculateGradsFn_t calculateGradsFn, reduction_t forwardReduction);

#endif // TRAINING_EPOCH_DEFAULT_H
