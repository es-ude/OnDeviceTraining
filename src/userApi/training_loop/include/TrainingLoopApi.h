#ifndef TRAINING_LOOP_API_H
#define TRAINING_LOOP_API_H

#include <stdbool.h>
#include <stdio.h>

#include "DataLoader.h"
#include "InferenceApi.h"
#include "LossFunction.h"
#include "Model.h"
#include "Optimizer.h"
#include "RNG.h"
#include "Tensor.h"
#include "TrainingCall.h"

/* Forward typedefs only — callers passing NULL need no scheduler
 * headers. Identical typedefs live in LrScheduler.h / BsScheduler.h (C11
 * allows the redefinition). */
typedef struct lrScheduler lrScheduler_t;
typedef struct bsScheduler bsScheduler_t;

/*! The keyed randomness of one training stream: caller-owned, one per stream,
 * never copied. `root` names the run, `step` counts optimizer updates and
 * `epoch` completed epochs. Derive every key through the trainingRng* helpers
 * below: they are the only place the key chain is written out, so every loop
 * that uses them derives the same keys for the same update and call. */
typedef struct trainingRng {
    rngKey_t root;
    uint32_t step;
    uint32_t epoch;
} trainingRng_t;

/* Every helper that takes the state requires a non-NULL state initialised by
 * trainingRngInit; trainingRngInit, Serialize and Deserialize reject a NULL
 * one. A missing key is a NULL key pointer instead. */

/*! root = rngKeyFromSeed(seed); step = epoch = 0. */
void trainingRngInit(trainingRng_t *rng, uint64_t seed);

/*! The key of the model calls of the current update. Derive it before the
 * update's batch is fetched. */
rngKey_t trainingRngStepKey(const trainingRng_t *rng);

/*! The key of call `callIndex` under a step key or an evaluation key. An index
 * above UINT32_MAX stops the program with a named error: it would reuse the
 * key of an earlier call. */
rngKey_t trainingRngCallKey(rngKey_t parentKey, size_t callIndex);

/*! The key of the current update's optimizer step and gradient mean-scale. */
rngKey_t trainingRngUpdateKey(const trainingRng_t *rng);

/*! The key of the current epoch's evaluation. */
rngKey_t trainingRngEvalKey(const trainingRng_t *rng);

/*! Call after each optimizer update. Past UINT32_MAX updates it stops the
 * program with a named error: a wrapped counter would reuse keys. */
void trainingRngAdvance(trainingRng_t *rng);

/*! Call after each completed epoch, under the same rule as trainingRngAdvance. */
void trainingRngAdvanceEpoch(trainingRng_t *rng);

/*! The key of the training loader's sample order. perEpoch = true gives each
 * epoch its own order (the key of the current epoch); false keeps epoch 0's
 * key, so the order is the same in every epoch. */
rngKey_t trainingRngShuffleKey(const trainingRng_t *rng, bool perEpoch);

/*! The key of the evaluation loader's sample order: one key per run, so every
 * evaluation sees the same order. */
rngKey_t trainingRngShuffleEvalKey(const trainingRng_t *rng);

/*! The key of the random initial values of the layer at `layerIndex`, its
 * position in the model. Build the model with these keys before training: the
 * key does not depend on the counters. An index above UINT32_MAX stops the
 * program with a named error: it would reuse the key of an earlier layer. */
rngKey_t trainingRngInitKey(const trainingRng_t *rng, size_t layerIndex);

/*! Writes the state as one ODTK v1 record of 24 bytes, every field a
 * little-endian u32: "ODTK" | version 1 | root.w[0] | root.w[1] | step | epoch.
 * Saved with the model's ODTS file, it lets a stopped run go on with the keys
 * it would have used. It holds no position inside an epoch: save between
 * epochs. Optimizer state (SGD momentum, AdamW moments and step count) and
 * LR/batch-size scheduler state are not part of it; a scheduler keeps its
 * lastEpoch only in memory (trainingRun continues from it within one
 * process). An exact resume in a new process needs SGD without momentum and
 * no scheduler. */
void trainingRngSerialize(const trainingRng_t *rng, FILE *f);

/*! Reads one ODTK record into *rng. A record that is not ODTK, a version other
 * than 1 or a short stream stops the program with a named error. */
void trainingRngDeserialize(trainingRng_t *rng, FILE *f);

/*! The call a level hands its callee for call `callIndex`: NULL for a NULL
 * call; otherwise *next = *call with only .key replaced, by the key of that
 * call under call->key (written to *callKey), or by NULL when call->key is
 * NULL. next may be call. The result points into next and callKey: keep both
 * alive for the callee's call. */
const trainingCall_t *trainingCallForIndex(const trainingCall_t *call, size_t callIndex,
                                           trainingCall_t *next, rngKey_t *callKey);

typedef struct trainingStats {
    tensor_t *output;
    float loss;
} trainingStats_t;

/*! Aggregate evaluation metrics for a full epoch. MEAN evaluation loss is
 * divided by the streamed sample count; SUM returns the total. Accuracy is over
 * all evaluated samples. Precision, recall and F1 are unweighted class means. */
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

/*! Final result of trainingRun() after the last epoch completed: the last of
 * numberOfEpochs, or, when stopOnNonFiniteLoss ended the run early, the first
 * epoch whose train or eval loss was non-finite. Training loss averages batch
 * means while weights change; evaluation loss is measured after the epoch with
 * fixed weights. Both use per-sample mean units. */
typedef struct trainingRunResult {
    float finalTrainLoss;
    epochStats_t finalEvalStats;
    size_t epochsCompleted;      /* epochs that completed training + evaluation (== numberOfEpochs
                                    unless stopped) */
    bool stoppedOnNonFiniteLoss; /* true iff stopOnNonFiniteLoss ended the run early */
} trainingRunResult_t;

/*! When invoked by trainingBatchDefault, input/label are borrowed stack
 *  views, valid only for the duration of the call: at microBatchSize 1 the
 *  [1, ...] views batchViewOf built for one sample (they share its data);
 *  at m > 1 [m, ...] views over the loop's gather buffers -- a copy of the
 *  chunk's m samples, overwritten by the next chunk and freed before
 *  trainingBatchDefault returns.
 *
 *  Dropout and BatchNorm1d are in training mode only inside
 *  calculateGradsSequential / tracedGrads (they flip the per-layer
 *  `training` flag around the call); a custom function that does not route
 *  through them runs both in eval mode.
 *
 *  call is the caller's, NULLable and borrowed for the call: a function that
 *  routes through calculateGradsSequential / tracedGrads forwards it, so a
 *  scheduler set in trainingRunOptions_t.remat reaches the driver. */
typedef trainingStats_t *(*calculateGradsFn_t)(const model_t *model, lossConfig_t lossConfig,
                                               reduction_t forwardReduction, tensor_t *input,
                                               tensor_t *label, const trainingCall_t *call);

/*! When invoked by the evaluation loop, input/label are borrowed stack views
 *  of shape [rows, ...sampleShape], valid only for the call: at
 *  microBatchSize 1 rows == 1 and they share the sample's data;
 *  at m > 1 they point into the loop's gather buffers, overwritten by the
 *  next chunk. The function returns an output whose leading dimension is
 *  rows and whose per-row block holds the C class scores, and the
 *  loss reduced over the rows per forwardReduction. trainingRun's evaluation
 *  passes its call with this call's key (see trainingRunOptions_t.remat and
 *  .rng), the public evaluation
 *  functions pass NULL; a function that routes through inferenceWithLoss
 *  forwards call. */
typedef inferenceStats_t *(*inferenceWithLossFn_t)(const model_t *model, tensor_t *input,
                                                   tensor_t *label, lossFuncType_t funcType,
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

/*! Invoked once per training epoch, after evaluation and after options->rng
 * counts the epoch, before the schedulers step. */
typedef void (*epochCallbackFn_t)(epochInfo_t info, epochStats_t evalStats);

/*! Optional inputs of trainingRun(). NULL, or a zero-initialised struct, means
 * "no schedulers, no callback, per-sample training" — exactly the pre-port
 * default behaviour. */
typedef struct trainingRunOptions {
    lrScheduler_t *lrScheduler; /* NULLable; stepped once per epoch after the callback */
    bsScheduler_t *bsScheduler; /* NULLable; stepped once per epoch after lrScheduler */
    epochCallbackFn_t callback; /* NULLable */
    bool stopOnNonFiniteLoss;   /* end the run after the first epoch whose train or eval loss is
                                    not finite */
    size_t microBatchSize;      /* rows per forward/backward call inside each macro batch;
                                    0 means 1. m > 1 stacks m samples into one [m, ...] call:
                                    FLOAT32 models only, every macro batch must be divisible by m.
                                    Evaluation inherits it unless evalMicroBatchSize is set.
                                    Dropout fails fast at m > 1 (its mask holds one sample). A
                                    rank-2 BatchNorm1d in training mode needs microBatchSize >= 2
                                    (batch statistics need >= 2 values per channel). */
    size_t evalMicroBatchSize;  /* rows per inferenceFn call during evaluation; 0 inherits
                                    microBatchSize. m_eval > 1 stacks consecutive eval samples
                                    into [rows, ...] chunks (rows <= m_eval; the last chunk holds
                                    the N mod m_eval remainder): FLOAT32 forward only. The gather
                                    buffers hold m_eval rows even when the eval set is smaller.
                                    An untracked BatchNorm1d's eval output depends on its chunk
                                    mates (deterministic: the eval loader is never reshuffled). */
    rematScheduler_t *remat;    /* NULLable; NULL keeps a fresh scheduler per training call.
                                    Otherwise a borrowed, initialized scheduler (rematHeapInit /
                                    rematArenaInit), keyed to every training input: [1, ...] at
                                    microBatchSize 1, [m, ...] at m > 1; the caller deinits it
                                    after the run. Evaluation shares it only when its micro-batch
                                    size matches training and its nominal sample count has no
                                    tail; otherwise evaluation runs without it. Live
                                    input/key mismatches fail. */
    trainingRng_t *rng;         /* NULLable; NULL runs without keys, so a shuffle = true train
                                    or eval loader stops at its shuffle. Otherwise
                                    caller-initialised (trainingRngInit) and caller-owned: the
                                    loaders' shuffle keys, every training epoch's step and
                                    update keys and each epoch's evaluation key come from it,
                                    and the epoch counter advances after each evaluation. */
} trainingRunOptions_t;

void freeTrainingStats(trainingStats_t *trainingStats);

/* The evaluation entry points below are key-less: a model whose forward draws
 * (stochastic rounding) stops with a named error at its first draw; trainingRun
 * evaluates under each epoch's evaluation key. They never shuffle: a
 * shuffle = true loader serves its identity order until dataLoaderShuffle has
 * set its order (DataLoaderApi.h). */

/*! Per-sample primitive over one batch_t: never stacks (the epoch entry points do). */
float evaluationBatch(const model_t *model, lossFuncType_t funcType, batch_t *batch,
                      inferenceWithLossFn_t inferenceFn, reduction_t forwardReduction);

/* The three epoch entries below: microBatchSize is the rows per inferenceFn call; 0 means 1.
 * m > 1 stacks consecutive samples of the loader's stream into [rows, ...] chunks (rows <= m, the
 * last chunk holds the remainder) -- FLOAT32 forwards only. */
float evaluationEpoch(const model_t *model, lossFuncType_t funcType, dataLoader_t *dataLoader,
                      inferenceWithLossFn_t inferenceFn, reduction_t forwardReduction,
                      size_t microBatchSize);

epochStats_t evaluationEpochWithMetrics(const model_t *model, lossFuncType_t funcType,
                                        dataLoader_t *dataLoader, inferenceWithLossFn_t inferenceFn,
                                        reduction_t forwardReduction, size_t microBatchSize);

classificationReport_t
evaluationEpochWithReport(const model_t *model, lossFuncType_t funcType, dataLoader_t *dataLoader,
                          inferenceWithLossFn_t inferenceFn, size_t *cmBuffer, size_t numClasses,
                          reduction_t forwardReduction, size_t microBatchSize);

/*! Runs numberOfEpochs of train+eval. Requires distinct train/eval loaders and
 * compatible scheduler wiring. Validates microbatch schedules, the training
 * loader's first epoch, the model (FLOAT32 at microBatchSize > 1; a Softmax last
 * layer under CROSS_ENTROPY) and evaluation feasibility before training. A
 * shuffle = true eval loader is shuffled once, under trainingRngShuffleEvalKey.
 * Each epoch shuffles, trains, evaluates, advances the RNG epoch, calls back,
 * then steps LR and batch schedulers. */
trainingRunResult_t trainingRun(const model_t *model, lossConfig_t lossConfig,
                                dataLoader_t *trainDataLoader, dataLoader_t *evalDataLoader,
                                optimizer_t *optimizer, size_t numberOfEpochs,
                                calculateGradsFn_t calculateGradsFn,
                                inferenceWithLossFn_t inferenceFn,
                                const trainingRunOptions_t *options); /* NULL == all defaults */

#endif // TRAINING_LOOP_API_H
