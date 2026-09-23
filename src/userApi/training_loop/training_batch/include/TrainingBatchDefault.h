#ifndef TRAINING_BATCH_DEFAULT_H
#define TRAINING_BATCH_DEFAULT_H

#include "TrainingLoopApi.h"

/*! Wraps each sample with batchViewOf before calling calculateGradsFn -- see
 *  docs/conventions/data-shape.md, "Who adds the batch axis". */
float trainingBatchDefault(layer_t **model, size_t modelSize, lossConfig_t lossConfig,
                           batch_t *batch, calculateGradsFn_t calculateGradsFn,
                           reduction_t forwardReduction);

#endif // TRAINING_BATCH_DEFAULT_H
