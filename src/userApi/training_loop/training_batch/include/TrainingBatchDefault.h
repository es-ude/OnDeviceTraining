#ifndef TRAINING_BATCH_DEFAULT_H
#define TRAINING_BATCH_DEFAULT_H

#include "TrainingLoopApi.h"

float trainingBatchDefault(layer_t **model, size_t modelSize, lossType_t lossType, batch_t *batch,
                           calculateGradsFn_t calculateGradsFn);

#endif // TRAINING_BATCH_DEFAULT_H
