#ifndef TRAINING_LOOP_API_H
#define TRAINING_LOOP_API_H

#include <stdbool.h>

#include "DataLoader.h"
#include "InferenceApi.h"
#include "LossFunction.h"
#include "Optimizer.h"
#include "Tensor.h"
#include "TrainingCall.h"

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

/*! When invoked by trainingBatchDefault, input/label are borrowed stack
 *  views, valid only for the duration of the call: at microBatchSize 1 the
 *  [1, ...] views batchViewOf built for one sample (they share its data,
 *  #152 PR3a); at m > 1 [m, ...] views over the loop's gather buffers -- a
 *  copy of the chunk's m samples, overwritten by the next chunk and freed
 *  before trainingBatchDefault returns (#152 PR3b).
 *
 *  Dropout and BatchNorm1d are in training mode only inside
 *  calculateGradsSequential / tracedGrads (they flip the per-layer
 *  `training` flag around the call); a custom function that does not route
 *  through them runs both in eval mode (#460).
 *
 *  call is the caller's, NULLable and borrowed for the call: a function that
 *  routes through calculateGradsSequential / tracedGrads forwards it, so a
 *  scheduler set in trainingRunOptions_t.remat reaches the driver. */
typedef trainingStats_t *(*calculateGradsFn_t)(layer_t **model, size_t modelSize,
                                               lossConfig_t lossConfig,
                                               reduction_t forwardReduction, tensor_t *input,
                                               tensor_t *label, const trainingCall_t *call);

/*! When invoked by the evaluation loop, input/label are borrowed stack views
 *  of shape [rows, ...sampleShape], valid only for the call: at
 *  microBatchSize 1 rows == 1 and they share the sample's data (#152 PR3a);
 *  at m > 1 they point into the loop's gather buffers, overwritten by the
 *  next chunk. The function returns an output whose leading dimension is
 *  rows and whose per-row block holds the C class scores (#468), and the
 *  loss reduced over the rows per forwardReduction. trainingRun's evaluation
 *  passes its call (see trainingRunOptions_t.remat), the public evaluation
 *  functions pass NULL; a function that routes through inferenceWithLoss
 *  forwards call. */
typedef inferenceStats_t *(*inferenceWithLossFn_t)(layer_t **model, size_t numberOfLayers,
                                                   tensor_t *input, tensor_t *label,
                                                   lossFuncType_t funcType,
                                                   reduction_t forwardReduction,
                                                   const trainingCall_t *call);

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
 * "no schedulers, no callback, per-sample training" — exactly the pre-port
 * default behaviour. */
typedef struct trainingRunOptions {
    lrScheduler_t *lrScheduler; /* NULLable; stepped once per epoch after the callback (#327) */
    bsScheduler_t *bsScheduler; /* NULLable; stepped once per epoch after lrScheduler */
    epochCallbackFn_t callback; /* NULLable */
    bool stopOnNonFiniteLoss;   /* end the run after the first epoch whose train or eval loss is
                                    not finite */
    size_t microBatchSize;      /* rows per forward/backward call inside each macro batch (#152);
                                    0 means 1. m > 1 stacks m samples into one [m, ...] call:
                                    FLOAT32 models only, every macro batch must be divisible by m.
                                    Evaluation inherits it unless evalMicroBatchSize is set.
                                    Dropout fails fast at m > 1 (its mask holds one sample). A
                                    rank-2 BatchNorm1d in training mode needs microBatchSize >= 2
                                    (batch statistics need >= 2 values per channel). */
    size_t evalMicroBatchSize;  /* rows per inferenceFn call during evaluation (#468); 0 inherits
                                    microBatchSize. m_eval > 1 stacks consecutive eval samples
                                    into [rows, ...] chunks (rows <= m_eval; the last chunk holds
                                    the N mod m_eval remainder): FLOAT32 forward only. The gather
                                    buffers hold m_eval rows even when the eval set is smaller.
                                    An untracked BatchNorm1d's eval output depends on its chunk
                                    mates (deterministic: the eval loader is never reshuffled). */
    rematScheduler_t *remat;    /* NULLable; NULL keeps a fresh scheduler per training call
                                    (remat D30). Otherwise caller-initialised (rematHeapInit /
                                    rematArenaInit) and caller-owned: trainingRun borrows it for
                                    every training call and never tears it down; the caller
                                    deinits it after the run. Its key must match the input of
                                    every training call -- [1, ...sampleShape] at microBatchSize 1,
                                    [m, ...sampleShape] at m > 1 -- so every training sample must
                                    have the one sampleShape it was keyed to: the first call
                                    whose input differs exits naming the mismatch, and a zeroed
                                    or deinitialised one exits at the first call.
                                    Evaluation runs on it too when every eval chunk has the
                                    training row count: evalMicroBatchSize equal to
                                    microBatchSize, and the eval loader's nominal count
                                    (datasetSize / batchSize * batchSize) a multiple of it.
                                    Otherwise evaluation runs without it (remat D19). On it,
                                    every eval sample must have the sampleShape too: the first
                                    eval call whose input differs exits naming the mismatch.
                                    A loader whose stream differs from its nominal count
                                    exits at its first ragged chunk. */
} trainingRunOptions_t;

void freeTrainingStats(trainingStats_t *trainingStats);

/*! Per-sample primitive over one batch_t: never stacks (the epoch entry points do, #468). */
float evaluationBatch(layer_t **model, size_t modelSize, lossFuncType_t funcType, batch_t *batch,
                      inferenceWithLossFn_t inferenceFn, reduction_t forwardReduction);

/* microBatchSize: rows per inferenceFn call; 0 means 1. m > 1 stacks consecutive samples of the
 * loader's stream into [rows, ...] chunks (rows <= m, the last chunk holds the remainder) --
 * FLOAT32 forwards only (#468). */
float evaluationEpoch(layer_t **model, size_t modelSize, lossFuncType_t funcType,
                      dataLoader_t *dataLoader, inferenceWithLossFn_t inferenceFn,
                      reduction_t forwardReduction, size_t microBatchSize);

/* microBatchSize: rows per inferenceFn call; 0 means 1. m > 1 stacks consecutive samples of the
 * loader's stream into [rows, ...] chunks (rows <= m, the last chunk holds the remainder) --
 * FLOAT32 forwards only (#468). */
epochStats_t evaluationEpochWithMetrics(layer_t **model, size_t modelSize, lossFuncType_t funcType,
                                        dataLoader_t *dataLoader, inferenceWithLossFn_t inferenceFn,
                                        reduction_t forwardReduction, size_t microBatchSize);

/* microBatchSize: rows per inferenceFn call; 0 means 1. m > 1 stacks consecutive samples of the
 * loader's stream into [rows, ...] chunks (rows <= m, the last chunk holds the remainder) --
 * FLOAT32 forwards only (#468). */
classificationReport_t evaluationEpochWithReport(layer_t **model, size_t modelSize,
                                                 lossFuncType_t funcType, dataLoader_t *dataLoader,
                                                 inferenceWithLossFn_t inferenceFn,
                                                 size_t *cmBuffer, size_t numClasses,
                                                 reduction_t forwardReduction,
                                                 size_t microBatchSize);

/*! Runs numberOfEpochs of train+eval. Per epoch, in this order: reshuffle
 * the train loader (epoch > 0, #381) -> capture epochInfo_t (batch, updates,
 * LR the epoch trains with) -> trainingEpochDefault -> evaluate -> callback
 * -> lrSchedulerStep -> bsSchedulerStep (last: it only writes
 * trainDataLoader->batchSize, which the next epoch reads fresh). A callback
 * that logs the LR/batch therefore reports the values this epoch actually
 * trained with. Fails fast (before the first batch) if: the LR scheduler is
 * wired to another optimizer (#327); the batch scheduler is wired to a loader
 * other than trainDataLoader (the eval loader is never resized); its LR
 * compensation is wired to another optimizer; an LR scheduler and a
 * compensating batch scheduler would both write the LR every epoch; the train
 * loader's batchSize is not divisible by options->microBatchSize; or, with a
 * batch scheduler and microBatchSize > 1, any batch the scheduler will set for
 * epochs 1..numberOfEpochs-1 is not divisible by it (#152); or
 * evaluation cannot run: the eval loader yields no batch; a non-FLOAT32
 * forward with an evaluation micro-batch > 1; or an untracked (noRunningStats)
 * BatchNorm1d whose evaluation chunk -- full or the N mod m tail -- has fewer
 * than 2 values per channel (#467/#468). */
trainingRunResult_t trainingRun(layer_t **model, size_t modelSize, lossConfig_t lossConfig,
                                dataLoader_t *trainDataLoader, dataLoader_t *evalDataLoader,
                                optimizer_t *optimizer, size_t numberOfEpochs,
                                calculateGradsFn_t calculateGradsFn,
                                inferenceWithLossFn_t inferenceFn,
                                const trainingRunOptions_t *options); /* NULL == all defaults */

#endif // TRAINING_LOOP_API_H
