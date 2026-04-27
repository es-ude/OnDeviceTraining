#ifndef TRAINING_LOOP_API_H
#define TRAINING_LOOP_API_H

#include "DataLoader.h"
#include "InferenceApi.h"
#include "LossFunction.h"
#include "Optimizer.h"
#include "Tensor.h"

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

/*! Final result of trainingRun() after the last epoch completed. */
typedef struct trainingRunResult {
    float finalTrainLoss;
    epochStats_t finalEvalStats;
} trainingRunResult_t;

typedef trainingStats_t *(*calculateGradsFn_t)(layer_t **model, size_t modelSize,
                                               lossType_t lossType, tensor_t *input,
                                               tensor_t *label);

typedef inferenceStats_t *(*inferenceWithLossFn_t)(layer_t **model, size_t numberOfLayers,
                                                   tensor_t *input, tensor_t *label,
                                                   lossType_t lossType);

/*! Callback invoked once per training epoch, after evaluation completes.
 *
 * \param epoch: Zero-based index of the just-finished epoch
 * \param trainLoss: Mean training loss across all batches of the epoch
 * \param evalStats: Aggregate evaluation metrics on the eval dataset
 */
typedef void (*epochCallbackFn_t)(size_t epoch, float trainLoss, epochStats_t evalStats);

void freeTrainingStats(trainingStats_t *trainingStats);

float evaluationBatch(layer_t **model, size_t modelSize, lossType_t lossType, batch_t *batch,
                      inferenceWithLossFn_t inferenceFn);

float evaluationEpoch(layer_t **model, size_t modelSize, lossType_t lossType,
                      dataLoader_t *dataLoader, inferenceWithLossFn_t inferenceFn);

/*! Evaluates model on a full epoch and returns loss and classification metrics.
 *
 * Derives the number of classes from the label tensor of the first batch.
 * Use evaluationEpochWithReport() if you need access to the confusion matrix
 * or want to provide numClasses explicitly.
 *
 * \param model: Pointer array of layer_t making up the model
 * \param modelSize: Number of layers in model
 * \param lossType: Loss function used for per-sample loss
 * \param dataLoader: Data loader over the evaluation dataset
 * \param inferenceFn: Inference function that also returns per-sample loss
 * \return Aggregate metrics for the epoch
 */
epochStats_t evaluationEpochWithMetrics(layer_t **model, size_t modelSize, lossType_t lossType,
                                        dataLoader_t *dataLoader,
                                        inferenceWithLossFn_t inferenceFn);

/*! Evaluates model on a full epoch and fills the caller's confusion matrix.
 *
 * The caller owns cmBuffer and must allocate it with exactly
 * numClasses * numClasses entries. The buffer is zeroed before accumulation.
 * Indexing convention is cm[predicted * numClasses + actual].
 *
 * \param model: Pointer array of layer_t making up the model
 * \param modelSize: Number of layers in model
 * \param lossType: Loss function used for per-sample loss
 * \param dataLoader: Data loader over the evaluation dataset
 * \param inferenceFn: Inference function that also returns per-sample loss
 * \param cmBuffer: Caller-owned confusion matrix buffer of size numClasses^2
 * \param numClasses: Number of classification classes
 * \return Report with epoch metrics and non-owning pointer to cmBuffer
 */
classificationReport_t evaluationEpochWithReport(layer_t **model, size_t modelSize,
                                                 lossType_t lossType, dataLoader_t *dataLoader,
                                                 inferenceWithLossFn_t inferenceFn,
                                                 size_t *cmBuffer, size_t numClasses);

trainingRunResult_t trainingRun(layer_t **model, size_t modelSize, lossType_t lossType,
                                dataLoader_t *trainDataLoader, dataLoader_t *evalDataLoader,
                                optimizer_t *optimizer, size_t numberOfEpochs,
                                calculateGradsFn_t calculateGradsFn,
                                inferenceWithLossFn_t inferenceFn, epochCallbackFn_t callback);

#endif // TRAINING_LOOP_API_H
