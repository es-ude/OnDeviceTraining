#define SOURCE_FILE "TRAINING_BATCH_DEFAULT"

#include "TrainingBatchDefault.h"
#include "BatchView.h"
#include "Common.h"
#include "DataLoaderApi.h"

float trainingBatchDefault(layer_t **model, size_t modelSize, lossConfig_t lossConfig,
                           batch_t *batch, calculateGradsFn_t calculateGradsFn,
                           reduction_t forwardReduction) {
    float totalLoss = 0.0f;

    for (size_t i = 0; i < batch->size; i++) {
        /* Samples arrive in their natural shape; the loop owns the batch axis
         * (docs/conventions/data-shape.md) and hands the model [1, ...] views. */
        batchView_t itemView;
        batchView_t labelView;
        trainingStats_t *stats =
            calculateGradsFn(model, modelSize, lossConfig, forwardReduction,
                             batchViewOf(&itemView, batch->samples[i]->item),
                             batchViewOf(&labelView, batch->samples[i]->label));
        totalLoss += stats->loss;
        freeTrainingStats(stats);
        freeSample(batch->samples[i]);
    }

    if (forwardReduction == REDUCTION_MEAN) {
        return totalLoss / (float)batch->size;
    }
    return totalLoss;
}
