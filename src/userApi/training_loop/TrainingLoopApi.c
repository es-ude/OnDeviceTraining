#define SOURCE_FILE "TRAINING_LOOP_API"

#include <stddef.h>

#include "Common.h"
#include "DataLoaderApi.h"
#include "InferenceApi.h"
#include "Optimizer.h"
#include "StorageApi.h"
#include "TensorApi.h"
#include "TrainingEpochDefault.h"
#include "TrainingLoopApi.h"

void freeTrainingStats(trainingStats_t *trainingStats) {
    freeTensor(trainingStats->output);
    freeReservedMemory(trainingStats);
}

float evaluationBatch(layer_t **model, size_t modelSize, lossFuncType_t funcType, batch_t *batch,
                      inferenceWithLossFn_t inferenceFn) {
    float totalLoss = 0.0f;

    for (size_t i = 0; i < batch->size; i++) {
        inferenceStats_t *stats = inferenceFn(model, modelSize, batch->samples[i]->item,
                                              batch->samples[i]->label, funcType);
        totalLoss += stats->loss;
        freeInferenceStats(stats);
        freeSample(batch->samples[i]);
    }

    return totalLoss / (float)batch->size;
}

static size_t argmax(const float *data, size_t n) {
    size_t maxIdx = 0;
    float maxVal = data[0];

    for (size_t i = 1; i < n; i++) {
        if (data[i] > maxVal) {
            maxVal = data[i];
            maxIdx = i;
        }
    }
    return maxIdx;
}

float evaluationEpoch(layer_t **model, size_t modelSize, lossFuncType_t funcType,
                      dataLoader_t *dataLoader, inferenceWithLossFn_t inferenceFn) {
    size_t datasetSize = dataLoader->getDatasetSize();
    size_t numberOfBatches = datasetSize / dataLoader->batchSize;
    float totalLoss = 0.0f;

    for (size_t i = 0; i < numberOfBatches; i++) {
        batch_t *batch = dataLoader->getBatch(dataLoader, i);
        totalLoss += evaluationBatch(model, modelSize, funcType, batch, inferenceFn);
        freeBatch(batch);
    }

    return totalLoss / (float)numberOfBatches;
}

static float evaluateBatchInternal(layer_t **model, size_t modelSize, lossFuncType_t funcType,
                                   batch_t *batch, inferenceWithLossFn_t inferenceFn, size_t *tp,
                                   size_t *predCount, size_t *actualCount, size_t *confusionMatrix,
                                   size_t numClasses) {
    float totalLoss = 0.0f;

    for (size_t i = 0; i < batch->size; i++) {
        inferenceStats_t *stats = inferenceFn(model, modelSize, batch->samples[i]->item,
                                              batch->samples[i]->label, funcType);
        totalLoss += stats->loss;

        float *outputData = (float *)stats->output->data;
        float *labelData = (float *)batch->samples[i]->label->data;
        size_t predicted = argmax(outputData, numClasses);
        size_t target = argmax(labelData, numClasses);

        if (predicted == target) {
            tp[predicted]++;
        }
        predCount[predicted]++;
        actualCount[target]++;

        if (confusionMatrix != NULL) {
            confusionMatrix[predicted * numClasses + target]++;
        }

        freeInferenceStats(stats);
        freeSample(batch->samples[i]);
    }

    return totalLoss / (float)batch->size;
}

static float computeAccuracy(const size_t *tp, size_t numClasses, size_t totalSamples) {
    size_t totalCorrect = 0;
    for (size_t c = 0; c < numClasses; c++) {
        totalCorrect += tp[c];
    }
    return (float)totalCorrect / (float)totalSamples;
}

static float computeMacroPrecision(const size_t *tp, const size_t *predCount, size_t numClasses) {
    float sum = 0.0f;
    for (size_t c = 0; c < numClasses; c++) {
        if (predCount[c] > 0) {
            sum += (float)tp[c] / (float)predCount[c];
        }
    }
    return sum / (float)numClasses;
}

static float computeMacroRecall(const size_t *tp, const size_t *actualCount, size_t numClasses) {
    float sum = 0.0f;
    for (size_t c = 0; c < numClasses; c++) {
        if (actualCount[c] > 0) {
            sum += (float)tp[c] / (float)actualCount[c];
        }
    }
    return sum / (float)numClasses;
}

static float computeMacroF1(const size_t *tp, const size_t *predCount, const size_t *actualCount,
                            size_t numClasses) {
    float sum = 0.0f;
    for (size_t c = 0; c < numClasses; c++) {
        float prec = (predCount[c] > 0) ? (float)tp[c] / (float)predCount[c] : 0.0f;
        float rec = (actualCount[c] > 0) ? (float)tp[c] / (float)actualCount[c] : 0.0f;
        if (prec + rec > 0.0f) {
            sum += 2.0f * prec * rec / (prec + rec);
        }
    }
    return sum / (float)numClasses;
}

static epochStats_t evaluateEpochInternal(layer_t **model, size_t modelSize,
                                          lossFuncType_t funcType, dataLoader_t *dataLoader,
                                          inferenceWithLossFn_t inferenceFn,
                                          size_t *confusionMatrix, size_t numClasses) {
    size_t datasetSize = dataLoader->getDatasetSize();
    size_t numberOfBatches = datasetSize / dataLoader->batchSize;

    size_t *tp = reserveMemory(numClasses * sizeof(size_t));
    size_t *predCount = reserveMemory(numClasses * sizeof(size_t));
    size_t *actualCount = reserveMemory(numClasses * sizeof(size_t));

    for (size_t c = 0; c < numClasses; c++) {
        tp[c] = 0;
        predCount[c] = 0;
        actualCount[c] = 0;
    }

    float totalLoss = 0.0f;
    size_t totalSamples = 0;

    for (size_t i = 0; i < numberOfBatches; i++) {
        batch_t *batch = dataLoader->getBatch(dataLoader, i);
        totalLoss += evaluateBatchInternal(model, modelSize, funcType, batch, inferenceFn, tp,
                                           predCount, actualCount, confusionMatrix, numClasses);
        totalSamples += batch->size;
        freeBatch(batch);
    }

    epochStats_t stats;
    stats.loss = totalLoss / (float)numberOfBatches;
    stats.accuracy = computeAccuracy(tp, numClasses, totalSamples);
    stats.precision = computeMacroPrecision(tp, predCount, numClasses);
    stats.recall = computeMacroRecall(tp, actualCount, numClasses);
    stats.f1 = computeMacroF1(tp, predCount, actualCount, numClasses);

    freeReservedMemory(tp);
    freeReservedMemory(predCount);
    freeReservedMemory(actualCount);

    return stats;
}

epochStats_t evaluationEpochWithMetrics(layer_t **model, size_t modelSize, lossFuncType_t funcType,
                                        dataLoader_t *dataLoader,
                                        inferenceWithLossFn_t inferenceFn) {
    // Peek at first sample to derive numClasses from label shape
    batch_t *firstBatch = dataLoader->getBatch(dataLoader, 0);
    size_t numClasses = calcNumberOfElementsByTensor(firstBatch->samples[0]->label);
    for (size_t i = 0; i < firstBatch->size; i++) {
        freeSample(firstBatch->samples[i]);
    }
    freeBatch(firstBatch);

    return evaluateEpochInternal(model, modelSize, funcType, dataLoader, inferenceFn, NULL,
                                 numClasses);
}

classificationReport_t evaluationEpochWithReport(layer_t **model, size_t modelSize,
                                                 lossFuncType_t funcType, dataLoader_t *dataLoader,
                                                 inferenceWithLossFn_t inferenceFn,
                                                 size_t *cmBuffer, size_t numClasses) {
    // Zero the caller's CM buffer
    for (size_t i = 0; i < numClasses * numClasses; i++) {
        cmBuffer[i] = 0;
    }

    classificationReport_t report;
    report.stats = evaluateEpochInternal(model, modelSize, funcType, dataLoader, inferenceFn,
                                         cmBuffer, numClasses);
    report.confusionMatrix = cmBuffer;
    report.numClasses = numClasses;
    return report;
}

trainingRunResult_t trainingRun(layer_t **model, size_t modelSize, lossConfig_t lossConfig,
                                dataLoader_t *trainDataLoader, dataLoader_t *evalDataLoader,
                                optimizer_t *optimizer, size_t numberOfEpochs,
                                calculateGradsFn_t calculateGradsFn,
                                inferenceWithLossFn_t inferenceFn, epochCallbackFn_t callback) {
    trainingRunResult_t result = {0};

    // Derive numClasses from eval dataset labels
    batch_t *firstBatch = evalDataLoader->getBatch(evalDataLoader, 0);
    size_t numClasses = calcNumberOfElementsByTensor(firstBatch->samples[0]->label);
    for (size_t i = 0; i < firstBatch->size; i++) {
        freeSample(firstBatch->samples[i]);
    }
    freeBatch(firstBatch);

    for (size_t epoch = 0; epoch < numberOfEpochs; epoch++) {
        float trainLoss = trainingEpochDefault(model, modelSize, lossConfig, trainDataLoader,
                                               optimizer, calculateGradsFn);
        epochStats_t evalStats = evaluateEpochInternal(
            model, modelSize, lossConfig.funcType, evalDataLoader, inferenceFn, NULL, numClasses);

        if (callback != NULL) {
            callback(epoch, trainLoss, evalStats);
        }

        result.finalTrainLoss = trainLoss;
        result.finalEvalStats = evalStats;

        trainDataLoader->batchSize +=1;
    }

    return result;
}
