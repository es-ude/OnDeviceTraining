#define SOURCE_FILE "TRAINING_EPOCH_DEFAULT"

#include "TrainingEpochDefault.h"
#include "Common.h"
#include "DataLoaderApi.h"
#include "LossFunction.h"
#include "Optimizer.h"
#include "OptimizerApi.h"
#include "Tensor.h"
#include "TrainingBatchDefault.h"

float trainingEpochDefault(layer_t **model, size_t modelSize, lossConfig_t lossConfig,
                           dataLoader_t *dataLoader, optimizer_t *optimizer,
                           calculateGradsFn_t calculateGradsFn, reduction_t forwardReduction) {
    size_t datasetSize = dataLoader->getDatasetSize();
    size_t numberOfBatches = datasetSize / dataLoader->batchSize;
    optimizerFunctions_t optimFns = optimizerFunctions[optimizer->type];
    float totalLoss = 0.0f;

    for (size_t i = 0; i < numberOfBatches; i++) {
        batch_t *batch = dataLoader->getBatch(dataLoader, i);

        /* Capture a reference to the first sample's label BEFORE
         * trainingBatchDefault consumes the samples. freeSample only
         * releases the sample_t struct; the underlying label tensor is
         * owned by the dataset and remains alive throughout the macro
         * batch, so labelRef stays valid for computeMeanScale below. */
        tensor_t *labelRef = batch->samples[0]->label;

        totalLoss += trainingBatchDefault(model, modelSize, lossConfig, batch, calculateGradsFn,
                                          forwardReduction);

        if (lossConfig.backwardReduction == REDUCTION_MEAN) {
            /* Each loss family derives F from labelRef's shape itself. */
            float meanScale =
                lossFunctions[lossConfig.funcType].computeMeanScale(batch->size, labelRef);
            scaleOptimizerGradients(optimizer, meanScale);
        }

        optimizerStep(optimizer);
        optimFns.zero(optimizer);
        freeBatch(batch);
    }

    return totalLoss / (float)numberOfBatches;
}
