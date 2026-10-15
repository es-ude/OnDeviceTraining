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
                           size_t microBatchSize, const trainingCall_t *call) {
    size_t m = (microBatchSize == 0) ? 1 : microBatchSize;
    if (dataLoader->batchSize % m != 0) {
        /* Divisibility check 3 (#152): covers direct callers; trainingRun checks
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
    /* The only level that reads .rng; it never hands its own .key down. */
    trainingRng_t *rng = (call != NULL) ? call->rng : NULL;

    for (size_t i = 0; i < numberOfBatches; i++) {
        rngKey_t stepKey;
        trainingCall_t next = (call != NULL) ? *call : (trainingCall_t){0};
        next.key = NULL;
        if (rng != NULL) {
            stepKey = trainingRngStepKey(rng);
            next.key = &stepKey;
        }
        /* next.key, not &stepKey: without an rng stepKey is never set, and a
         * loader that draws must stop instead of drawing under it. */
        batch_t *batch = dataLoader->getBatch(dataLoader, i, next.key);

        /* Capture a reference to the first sample's label BEFORE
         * trainingBatchDefault consumes the samples. freeSample only
         * releases the sample_t struct; the underlying label tensor is
         * owned by the dataset and remains alive throughout the macro
         * batch, so labelRef stays valid for computeMeanScale below. */
        tensor_t *labelRef = batch->samples[0]->label;

        totalLoss += trainingBatchDefault(model, modelSize, lossConfig, batch, calculateGradsFn,
                                          forwardReduction, m, (call != NULL) ? &next : NULL);

        /* The update's key: its mean-scale and its optimizer step draw under it. */
        rngKey_t updateKeyStorage;
        const rngKey_t *updateKey = NULL;
        if (rng != NULL) {
            updateKeyStorage = trainingRngUpdateKey(rng);
            updateKey = &updateKeyStorage;
        }

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
            scaleOptimizerGradients(optimizer, meanScale, updateKey);
        }

        optimizerStep(optimizer, updateKey);
        if (rng != NULL) {
            trainingRngAdvance(rng);
        }
        optimFns.zero(optimizer);
        freeBatch(batch);
    }

    return totalLoss / (float)numberOfBatches;
}
