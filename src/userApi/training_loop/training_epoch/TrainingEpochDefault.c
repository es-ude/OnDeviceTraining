#define SOURCE_FILE "TRAINING_EPOCH_DEFAULT"

#include <stdlib.h>

#include "BatchView.h"
#include "Common.h"
#include "DataLoaderApi.h"
#include "LossFunction.h"
#include "Optimizer.h"
#include "OptimizerApi.h"
#include "Tensor.h"
#include "TrainingBatchDefault.h"
#include "TrainingEpochDefault.h"

float trainingEpochDefault(layer_t **model, size_t modelSize, lossConfig_t lossConfig,
                           dataLoader_t *dataLoader, optimizer_t *optimizer,
                           calculateGradsFn_t calculateGradsFn, reduction_t forwardReduction,
                           size_t microBatchSize) {
    size_t m = (microBatchSize == 0) ? 1 : microBatchSize;
    if (dataLoader->batchSize % m != 0) {
        /* Check 3 (#152 spec §6.2): covers direct callers; trainingRun checks
         * the same (and its whole batch-size schedule) before epoch 0. */
        PRINT_ERROR("trainingEpochDefault: batchSize %u is not divisible by microBatchSize %zu "
                    "(b %% m == 0 is required)",
                    (unsigned)dataLoader->batchSize, m);
        exit(1);
    }
    size_t datasetSize = dataLoader->getDatasetSize();
    size_t numberOfBatches = datasetSize / dataLoader->batchSize;
    if (numberOfBatches == 0) {
        /* batchSize > datasetSize: no batch can be formed, the loop below would
         * train nothing and the mean would be 0/0. */
        PRINT_ERROR("trainingEpochDefault: batchSize %u exceeds the dataset size %zu (zero "
                    "batches)",
                    (unsigned)dataLoader->batchSize, datasetSize);
        exit(1);
    }
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
                                          forwardReduction, m);

        if (lossConfig.backwardReduction == REDUCTION_MEAN) {
            /* Each loss family derives F from labelRef's shape itself, reading
             * dims[0] as the batch axis -- so it gets the same [1, ...] view the
             * model output has (a natural [C, L] label must give F = C*L).
             * CE ignores this tensor entirely (1/totalSamples, no F term); MSE
             * is the one that reads dims[0] -- see LossFunction.h and
             * docs/conventions/loss.md, "Microbatch shape" (#152 PR3a). */
            batchView_t labelRefView;
            float meanScale = lossFunctions[lossConfig.funcType].computeMeanScale(
                batch->size, batchViewOf(&labelRefView, labelRef));
            scaleOptimizerGradients(optimizer, meanScale);
        }

        optimizerStep(optimizer);
        optimFns.zero(optimizer);
        freeBatch(batch);
    }

    return totalLoss / (float)numberOfBatches;
}
