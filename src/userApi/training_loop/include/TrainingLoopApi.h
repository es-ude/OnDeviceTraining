#ifndef TRAINING_LOOP_API_H
#define TRAINING_LOOP_API_H

#include <stdbool.h>

#include "DataLoader.h"
#include "InferenceApi.h"
#include "LossFunction.h"
#include "Optimizer.h"
#include "Tensor.h"

/* #327: forward typedefs only — callers passing NULL need no scheduler
 * headers. Identical typedefs live in LrScheduler.h / BsScheduler.h (C11
 * allows the redefinition). */
typedef struct lrScheduler lrScheduler_t;
typedef struct bsScheduler bsScheduler_t;

typedef struct trainingStats {
    tensor_t *output;
    float loss;
} trainingStats_t;

/*! Aggregate evaluation metrics for a full epoch.
 *
 * Loss is averaged across batches. Accuracy is over all evaluated samples.
 * Precision, recall and F1 are macro-averaged (unweighted mean across classes).
 */
typedef struct epochStats {
    float loss;
    float accuracy;
    float precision;
    float recall;
    float f1;
} epochStats_t;

/*! Full classification report returned by evaluationEpochWithReport().
 *
 * The confusion matrix lives in a caller-provided buffer of size
 * numClasses * numClasses, indexed as cm[predicted * numClasses + actual].
 * Ownership of confusionMatrix stays with the caller.
 */
typedef struct classificationReport {
    epochStats_t stats;
    size_t *confusionMatrix;
    size_t numClasses;
} classificationReport_t;

/*! Final result of trainingRun() after the last epoch completed — either the
 * last of numberOfEpochs, or, when stopOnNonFiniteLoss ended the run early,
 * the first epoch whose train or eval loss was non-finite.
 *
 * `finalTrainLoss` and `finalEvalStats.loss` share the same unit (per-sample
 * mean of the configured loss function) but are measured over different
 * windows:
 *
 *   - `finalTrainLoss`  is a during-epoch mean of batch-means: each batch's
 *                       per-sample loss is averaged across the batch, and
 *                       those batch-means are averaged across the epoch.
 *                       Weights mutate across batches, so this number mixes
 *                       early-epoch under-trained weights with late-epoch
 *                       near-converged weights.
 *
 *   - `finalEvalStats.loss` is a post-epoch full-pass measurement on the
 *                           eval dataset with the weight state frozen at the
 *                           end of the epoch.
 *
 * As a consequence, the two values disagree — typically eval > train by a
 * factor that grows with training as the model converges. This is the
 * generalization gap × measurement-window difference, not a unit mismatch.
 */
typedef struct trainingRunResult {
    float finalTrainLoss;
    epochStats_t finalEvalStats;
    size_t epochsCompleted;      /* epochs that completed training + evaluation (== numberOfEpochs
                                    unless stopped) */
    bool stoppedOnNonFiniteLoss; /* true iff stopOnNonFiniteLoss ended the run early */
} trainingRunResult_t;

typedef trainingStats_t *(*calculateGradsFn_t)(layer_t **model, size_t modelSize,
                                               lossConfig_t lossConfig,
                                               reduction_t forwardReduction, tensor_t *input,
                                               tensor_t *label);

typedef inferenceStats_t *(*inferenceWithLossFn_t)(layer_t **model, size_t numberOfLayers,
                                                   tensor_t *input, tensor_t *label,
                                                   lossFuncType_t funcType,
                                                   reduction_t forwardReduction);

/*! Per-epoch facts handed to the epoch callback. batchSize, parameterUpdates
 * and learningRate are captured BEFORE the epoch trains — they are the values
 * the epoch actually trained with, because the schedulers only step after the
 * callback returns. */
typedef struct epochInfo {
    size_t epoch;
    float trainLoss;
    size_t batchSize;        /* trainDataLoader->batchSize this epoch trained with */
    size_t parameterUpdates; /* optimizer steps this epoch = datasetSize / batchSize */
    float learningRate;      /* getLr(optimizer) this epoch trained with */
} epochInfo_t;

/*! Invoked once per training epoch, after evaluation, before the schedulers step. */
typedef void (*epochCallbackFn_t)(epochInfo_t info, epochStats_t evalStats);

/*! Optional inputs of trainingRun(). NULL, or a zero-initialised struct, means
 * "no schedulers, no callback" — exactly the pre-port default behaviour. */
typedef struct trainingRunOptions {
    lrScheduler_t *lrScheduler; /* NULLable; stepped once per epoch after the callback (#327) */
    bsScheduler_t *bsScheduler; /* NULLable; stepped once per epoch after lrScheduler */
    epochCallbackFn_t callback; /* NULLable */
    bool stopOnNonFiniteLoss;   /* end the run after the first epoch whose train or eval loss is
                                    not finite */
} trainingRunOptions_t;

void freeTrainingStats(trainingStats_t *trainingStats);

float evaluationBatch(layer_t **model, size_t modelSize, lossFuncType_t funcType, batch_t *batch,
                      inferenceWithLossFn_t inferenceFn, reduction_t forwardReduction);

float evaluationEpoch(layer_t **model, size_t modelSize, lossFuncType_t funcType,
                      dataLoader_t *dataLoader, inferenceWithLossFn_t inferenceFn,
                      reduction_t forwardReduction);

epochStats_t evaluationEpochWithMetrics(layer_t **model, size_t modelSize, lossFuncType_t funcType,
                                        dataLoader_t *dataLoader, inferenceWithLossFn_t inferenceFn,
                                        reduction_t forwardReduction);

classificationReport_t evaluationEpochWithReport(layer_t **model, size_t modelSize,
                                                 lossFuncType_t funcType, dataLoader_t *dataLoader,
                                                 inferenceWithLossFn_t inferenceFn,
                                                 size_t *cmBuffer, size_t numClasses,
                                                 reduction_t forwardReduction);

/*! Runs numberOfEpochs of train+eval. Per epoch, in this order: reshuffle
 * the train loader (epoch > 0, #381) -> capture epochInfo_t (batch, updates,
 * LR the epoch trains with) -> trainingEpochDefault -> evaluate -> callback
 * -> lrSchedulerStep -> bsSchedulerStep (last: it only writes
 * trainDataLoader->batchSize, which the next epoch reads fresh). A callback
 * that logs the LR/batch therefore reports the values this epoch actually
 * trained with. Fails fast (before the first batch) if: the LR scheduler is
 * wired to another optimizer (#327); the batch scheduler is wired to a loader
 * other than trainDataLoader (the eval loader is never resized); its LR
 * compensation is wired to another optimizer; or an LR scheduler and a
 * compensating batch scheduler would both write the LR every epoch. */
trainingRunResult_t trainingRun(layer_t **model, size_t modelSize, lossConfig_t lossConfig,
                                dataLoader_t *trainDataLoader, dataLoader_t *evalDataLoader,
                                optimizer_t *optimizer, size_t numberOfEpochs,
                                calculateGradsFn_t calculateGradsFn,
                                inferenceWithLossFn_t inferenceFn,
                                const trainingRunOptions_t *options); /* NULL == all defaults */

#endif // TRAINING_LOOP_API_H
