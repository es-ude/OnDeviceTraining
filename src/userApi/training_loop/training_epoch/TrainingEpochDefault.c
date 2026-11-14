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

float trainingEpochDefault(const model_t *model, lossConfig_t lossConfig, dataLoader_t *dataLoader,
                           optimizer_t *optimizer, calculateGradsFn_t calculateGradsFn,
                           reduction_t forwardReduction, size_t microBatchSize,
                           const trainingCall_t *call) {
    size_t m = (microBatchSize == 0) ? 1 : microBatchSize;
    size_t datasetSize = dataLoader->getDatasetSize();
    size_t numberOfBatches = datasetSize / dataLoader->batchSize;
    if (numberOfBatches == 0) {
        /* trainingRun checks the first epoch; the dataset may change between
         * epochs. The mean would be 0/0. */
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

        /* Save the dataset-owned label before consuming its sample wrapper. */
        tensor_t *labelRef = batch->samples[0]->label;

        totalLoss += trainingBatchDefault(model, lossConfig, batch, calculateGradsFn,
                                          forwardReduction, m, (call != NULL) ? &next : NULL);

        /* The update's key: its mean-scale and its optimizer step draw under it. */
        rngKey_t updateKeyStorage;
        const rngKey_t *updateKey = NULL;
        if (rng != NULL) {
            updateKeyStorage = trainingRngUpdateKey(rng);
            updateKey = &updateKeyStorage;
        }

        if (lossConfig.backwardReduction == REDUCTION_MEAN) {
            /* Mean scaling expects a batched label view so MSE counts all
             * per-sample features. */
            /* The INTEGER divisor (batch->size: the batch's actual sample
             * count), so a BFP grad's 1/divisor is derived without float;
             * the float grads get 1.0f / (float)divisor, today's scale. */
            batchView_t labelRefView;
            size_t meanDivisor = lossFunctions[lossConfig.funcType].computeMeanDivisor(
                batch->size, batchViewOf(&labelRefView, labelRef));
            scaleOptimizerGradientsMean(optimizer, meanDivisor, updateKey);
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
