#define SOURCE_FILE "TRAINING_EPOCH_DEFAULT"

#include "TrainingEpochDefault.h"
#include "Common.h"
#include "DataLoaderApi.h"
#include "Optimizer.h"
#include "TrainingBatchDefault.h"

float trainingEpochDefault(layer_t **model, size_t modelSize, lossType_t lossType,
                           dataLoader_t *dataLoader, optimizer_t *optimizer,
                           calculateGradsFn_t calculateGradsFn) {
    size_t datasetSize = dataLoader->getDatasetSize();
    size_t numberOfBatches = datasetSize / dataLoader->batchSize;
    optimizerFunctions_t optimFns = optimizerFunctions[optimizer->type];
    float totalLoss = 0.0f;

    for (size_t i = 0; i < numberOfBatches; i++) {
        batch_t *batch = dataLoader->getBatch(dataLoader, i);
        totalLoss += trainingBatchDefault(model, modelSize, lossType, batch, calculateGradsFn);
        optimFns.step(optimizer);
        optimFns.zero(optimizer);
        freeBatch(batch);
    }

    return totalLoss / (float)numberOfBatches;
}
