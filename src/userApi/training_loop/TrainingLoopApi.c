#define SOURCE_FILE "TRAINING_LOOP_API"

#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "BatchNorm1d.h"
#include "BatchView.h"
#include "BsScheduler.h"
#include "Common.h"
#include "DataLoaderApi.h"
#include "InferenceApi.h"
#include "Layer.h"
#include "LayerConfigAccess.h"
#include "LrScheduler.h"
#include "Optimizer.h"
#include "StackGather.h"
#include "StorageApi.h"
#include "TensorApi.h"
#include "TensorConversion.h"
#include "TrainingEpochDefault.h"
#include "TrainingLoopApi.h"

void freeTrainingStats(trainingStats_t *trainingStats) {
    freeTensor(trainingStats->output);
    freeReservedMemory(trainingStats);
}

float evaluationBatch(layer_t **model, size_t modelSize, lossFuncType_t funcType, batch_t *batch,
                      inferenceWithLossFn_t inferenceFn, reduction_t forwardReduction) {
    float totalLoss = 0.0f;

    for (size_t i = 0; i < batch->size; i++) {
        batchView_t itemView;
        batchView_t labelView;
        inferenceStats_t *stats = inferenceFn(
            model, modelSize, batchViewOf(&itemView, batch->samples[i]->item),
            batchViewOf(&labelView, batch->samples[i]->label), funcType, forwardReduction);
        totalLoss += stats->loss;
        freeInferenceStats(stats);
        freeSample(batch->samples[i]);
    }

    /* Pure sum — no division. Caller (evaluationEpoch) handles macro reduction. */
    return totalLoss;
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

/* Argmax over a wire tensor in ITS OWN dtype (#206 acceptance prerequisite):
 * SYM_INT32 mantissa order IS value order (scale > 0), so no dequant is
 * needed — while casting the int32 codes to float* garbles the comparison
 * (negative mantissas reinterpret as NaN bit patterns and freeze a float
 * argmax at index 0). */
static size_t argmaxByTensor(const tensor_t *t, size_t n) {
    switch (t->quantization->type) {
    case FLOAT32:
        return argmax((const float *)t->data, n);
    case SYM_INT32: {
        const int32_t *m = (const int32_t *)t->data;
        size_t maxIdx = 0;
        int32_t maxVal = m[0];
        for (size_t i = 1; i < n; i++) {
            if (m[i] > maxVal) {
                maxVal = m[i];
                maxIdx = i;
            }
        }
        return maxIdx;
    }
    case BFP: {
        /* BFP epic PR2 Task 8: mantissa order is NOT value order here — every
         * GROUP carries its own exponent, so a small value in a fine-grained
         * group can outrank a large one on the raw mantissa. Comparison must
         * happen after dequant (exact: m * 2^E, so float comparison semantics
         * are preserved). Streamed in chunks so no scratch scales with n;
         * dequantChunkToFloat requires count <= ODT_CONVERSION_CHUNK_ELEMS and
         * a byte-aligned elemOffset, both satisfied by a walk that starts at 0
         * and strides by the (multiple-of-8) chunk size. Both call sites argmax
         * a single sample's class vector from element 0 — this must not be
         * handed an arbitrary per-sample slice offset. */
        float chunk[ODT_CONVERSION_CHUNK_ELEMS];
        size_t maxIdx = 0;
        float maxVal = 0.f;
        for (size_t off = 0; off < n; off += ODT_CONVERSION_CHUNK_ELEMS) {
            size_t count =
                n - off < ODT_CONVERSION_CHUNK_ELEMS ? n - off : ODT_CONVERSION_CHUNK_ELEMS;
            dequantChunkToFloat(t, off, count, chunk);
            for (size_t i = 0; i < count; i++) {
                if (off + i == 0 || chunk[i] > maxVal) {
                    maxVal = chunk[i];
                    maxIdx = off + i;
                }
            }
        }
        return maxIdx;
    }
    default:
        PRINT_ERROR("evaluate: output-wire dtype %d not supported for argmax "
                    "(FLOAT32/SYM_INT32/BFP)",
                    (int)t->quantization->type);
        exit(1);
    }
}

/* #468 D9: computed from sizes, before any getBatch -- a dataset smaller than
 * batchSize would otherwise make getBatch overrun the loader's index table,
 * and a MEAN over zero samples is 0/0. */
static size_t requireEvalBatches(dataLoader_t *dataLoader, const char *caller) {
    size_t datasetSize = dataLoader->getDatasetSize();
    size_t numberOfBatches = datasetSize / dataLoader->batchSize;
    if (numberOfBatches == 0) {
        PRINT_ERROR("%s: the evaluation loader yields no batch (dataset size %zu < batchSize %u) "
                    "-- nothing to evaluate",
                    caller, datasetSize, (unsigned)dataLoader->batchSize);
        exit(1);
    }
    return numberOfBatches;
}

/* #468 D11: every metric indexes [0, numClasses); a caller-supplied
 * numClasses that disagrees with the label would read past each class row. */
static void requireNumClassesMatchesLabel(size_t numClasses, tensor_t *label, const char *caller) {
    size_t labelElements = calcNumberOfElementsByTensor(label);
    if (numClasses != labelElements) {
        PRINT_ERROR("%s: numClasses %zu does not match the label's %zu elements", caller,
                    numClasses, labelElements);
        exit(1);
    }
}

static size_t resolveEvalMicroBatch(size_t microBatchSize) {
    return (microBatchSize == 0) ? 1 : microBatchSize;
}

/* Per-class counters of the metric entry points; NULL for evaluationEpoch. */
typedef struct evalCounts {
    size_t *tp;
    size_t *predCount;
    size_t *actualCount;
    size_t *confusionMatrix; /* NULL: no report */
    size_t numClasses;
} evalCounts_t;

static void countPrediction(evalCounts_t *counts, size_t predicted, size_t target) {
    if (predicted == target) {
        counts->tp[predicted]++;
    }
    counts->predCount[predicted]++;
    counts->actualCount[target]++;
    if (counts->confusionMatrix != NULL) {
        counts->confusionMatrix[predicted * counts->numClasses + target]++;
    }
}

/* Runs once, on the first streamed sample of every evaluation, whatever m. */
static void evalFirstSample(layer_t **model, size_t modelSize, dataLoader_t *dataLoader,
                            sample_t *first, size_t m, size_t numClasses, const char *caller) {
    (void)model;
    (void)modelSize;
    (void)dataLoader;
    (void)m;
    if (numClasses != 0) {
        requireNumClassesMatchesLabel(numClasses, first->label, caller);
    }
}

/* Extends D9 past the size check: a custom getBatch returning only empty
 * batches would make MEAN and the metrics divide by 0. */
static void requireStreamedSamples(size_t totalSamples, const char *caller) {
    if (totalSamples == 0) {
        PRINT_ERROR("%s: the evaluation loader streamed no sample (its getBatch returned only "
                    "empty batches) -- nothing to evaluate",
                    caller);
        exit(1);
    }
}

/* A stacked chunk's output must hold `rows` contiguous rows of C FLOAT32
 * class scores -- the per-row argmax reads it at row * C. */
static void requireChunkOutput(const char *caller, inferenceStats_t *stats, size_t rows, size_t C,
                               size_t chunkFirst) {
    tensor_t *out = (stats == NULL) ? NULL : stats->output;
    if (out == NULL || out->quantization == NULL || out->shape == NULL || out->data == NULL ||
        out->shape->dimensions == NULL || out->shape->orderOfDimensions == NULL) {
        PRINT_ERROR("%s: the inference function returned no output tensor, quantization, shape or "
                    "data for the chunk starting at sample %zu",
                    caller, chunkFirst);
        exit(1);
    }
    if (out->quantization->type != FLOAT32) {
        PRINT_ERROR("%s: stacked evaluation is FLOAT32-only, but the output of the chunk starting "
                    "at sample %zu has dtype %d",
                    caller, chunkFirst, (int)out->quantization->type);
        exit(1);
    }
    const shape_t *s = out->shape;
    if (s->numberOfDimensions < 1 || s->dimensions[0] != rows) {
        PRINT_ERROR("%s: the output of the chunk starting at sample %zu must lead with its %zu "
                    "rows",
                    caller, chunkFirst, rows);
        exit(1);
    }
    size_t elements = calcNumberOfElementsByTensor(out);
    if (elements != rows * C) {
        PRINT_ERROR("%s: the output of the chunk starting at sample %zu has %zu elements, expected "
                    "%zu rows x %zu classes",
                    caller, chunkFirst, elements, rows, C);
        exit(1);
    }
    for (size_t d = 0; d < s->numberOfDimensions; d++) {
        if (s->orderOfDimensions[d] != d) {
            PRINT_ERROR("%s: the output of the chunk starting at sample %zu is not in identity "
                        "dimension order (its rows would not be contiguous)",
                        caller, chunkFirst);
            exit(1);
        }
    }
}

/* One inferenceFn call over `rows` gathered samples: stack views of the
 * reference re-pointed at the gather buffers (dims[0] = rows). Returns the
 * row-weighted loss (x rows for MEAN, plain for SUM -- spec 5.3). */
static float evaluateChunk(const char *caller, layer_t **model, size_t modelSize,
                           lossFuncType_t funcType, inferenceWithLossFn_t inferenceFn,
                           reduction_t forwardReduction, tensor_t *referenceItem,
                           tensor_t *referenceLabel, uint8_t *itemBuffer, uint8_t *labelBuffer,
                           size_t rows, size_t C, size_t chunkFirst, evalCounts_t *counts) {
    batchView_t itemView;
    batchView_t labelView;
    tensor_t *item = batchViewOf(&itemView, referenceItem);
    tensor_t *label = batchViewOf(&labelView, referenceLabel);
    item->data = itemBuffer;
    itemView.dimensions[0] = rows;
    label->data = labelBuffer;
    labelView.dimensions[0] = rows;

    inferenceStats_t *stats =
        inferenceFn(model, modelSize, item, label, funcType, forwardReduction);
    requireChunkOutput(caller, stats, rows, C, chunkFirst);
    if (counts != NULL) {
        const float *out = (const float *)stats->output->data;
        const float *target = (const float *)labelBuffer;
        for (size_t r = 0; r < rows; r++) {
            countPrediction(counts, argmax(out + r * C, C), argmax(target + r * C, C));
        }
    }
    float loss = (forwardReduction == REDUCTION_MEAN) ? stats->loss * (float)rows : stats->loss;
    freeInferenceStats(stats);
    return loss;
}

/* m > 1 (spec 5.3): gathers consecutive samples of the loader's stream --
 * ACROSS batch_t boundaries, since eval loaders typically use batchSize 1 --
 * into chunks of m rows, then runs the N mod m remainder. Counts come from
 * the stream (a replay wrapper's batches exceed batchSize, D7). The reference
 * sample's tensors are dataset-owned and outlive its sample_t (freeSample
 * frees only the wrapper). */
static float evaluateStacked(const char *caller, layer_t **model, size_t modelSize,
                             lossFuncType_t funcType, dataLoader_t *dataLoader,
                             size_t numberOfBatches, inferenceWithLossFn_t inferenceFn,
                             reduction_t forwardReduction, size_t m, evalCounts_t *counts,
                             size_t *totalSamples) {
    stackGatherRequireFloat32Model(caller, model, modelSize, m, layerForwardNonFloat32Field);
    tensor_t *referenceItem = NULL;
    tensor_t *referenceLabel = NULL;
    uint8_t *itemBuffer = NULL;
    uint8_t *labelBuffer = NULL;
    size_t itemBytes = 0;
    size_t labelBytes = 0;
    size_t C = 0;
    size_t rows = 0;
    size_t streamed = 0;
    float totalLoss = 0.0f;

    for (size_t b = 0; b < numberOfBatches; b++) {
        batch_t *batch = dataLoader->getBatch(dataLoader, b);
        for (size_t i = 0; i < batch->size; i++) {
            sample_t *sample = batch->samples[i];
            if (referenceItem == NULL) {
                evalFirstSample(model, modelSize, dataLoader, sample, m,
                                counts != NULL ? counts->numClasses : 0, caller);
                referenceItem = sample->item;
                referenceLabel = sample->label;
                /* Validates the reference's own dtype/sparsity before its
                 * bytes size the buffers. */
                stackGatherRequireStackable(caller, referenceItem, referenceItem, "item", 0, m);
                stackGatherRequireStackable(caller, referenceLabel, referenceLabel, "label", 0, m);
                itemBytes = calcBytesPerTensor(referenceItem);
                labelBytes = calcBytesPerTensor(referenceLabel);
                C = calcNumberOfElementsByTensor(referenceLabel);
                itemBuffer = stackGatherReserveBuffer(caller, m, itemBytes, "item");
                labelBuffer = stackGatherReserveBuffer(caller, m, labelBytes, "label");
            }
            stackGatherRequireStackable(caller, referenceItem, sample->item, "item", streamed, m);
            stackGatherRequireStackable(caller, referenceLabel, sample->label, "label", streamed,
                                        m);
            memcpy(itemBuffer + rows * itemBytes, sample->item->data, itemBytes);
            memcpy(labelBuffer + rows * labelBytes, sample->label->data, labelBytes);
            rows++;
            streamed++;
            freeSample(sample);
            if (rows == m) {
                totalLoss +=
                    evaluateChunk(caller, model, modelSize, funcType, inferenceFn, forwardReduction,
                                  referenceItem, referenceLabel, itemBuffer, labelBuffer, rows, C,
                                  streamed - rows, counts);
                rows = 0;
            }
        }
        freeBatch(batch);
    }
    if (rows > 0) {
        totalLoss += evaluateChunk(caller, model, modelSize, funcType, inferenceFn,
                                   forwardReduction, referenceItem, referenceLabel, itemBuffer,
                                   labelBuffer, rows, C, streamed - rows, counts);
    }
    if (labelBuffer != NULL) {
        freeReservedMemory(labelBuffer);
    }
    if (itemBuffer != NULL) {
        freeReservedMemory(itemBuffer);
    }
    *totalSamples = streamed;
    return totalLoss;
}

float evaluationEpoch(layer_t **model, size_t modelSize, lossFuncType_t funcType,
                      dataLoader_t *dataLoader, inferenceWithLossFn_t inferenceFn,
                      reduction_t forwardReduction, size_t microBatchSize) {
    size_t m = resolveEvalMicroBatch(microBatchSize);
    size_t numberOfBatches = requireEvalBatches(dataLoader, "evaluationEpoch");
    float totalLoss = 0.0f;
    size_t totalSamples = 0;

    if (m == 1) {
        for (size_t i = 0; i < numberOfBatches; i++) {
            batch_t *batch = dataLoader->getBatch(dataLoader, i);
            if (i == 0 && batch->size > 0) {
                evalFirstSample(model, modelSize, dataLoader, batch->samples[0], m, 0,
                                "evaluationEpoch");
            }
            totalLoss +=
                evaluationBatch(model, modelSize, funcType, batch, inferenceFn, forwardReduction);
            totalSamples += batch->size;
            freeBatch(batch);
        }
    } else {
        totalLoss =
            evaluateStacked("evaluationEpoch", model, modelSize, funcType, dataLoader,
                            numberOfBatches, inferenceFn, forwardReduction, m, NULL, &totalSamples);
    }
    requireStreamedSamples(totalSamples, "evaluationEpoch");

    if (forwardReduction == REDUCTION_MEAN) {
        return totalLoss / (float)totalSamples;
    }
    return totalLoss;
}

static float evaluateBatchInternal(layer_t **model, size_t modelSize, lossFuncType_t funcType,
                                   batch_t *batch, inferenceWithLossFn_t inferenceFn,
                                   evalCounts_t *counts, reduction_t forwardReduction) {
    float totalLoss = 0.0f;

    for (size_t i = 0; i < batch->size; i++) {
        batchView_t itemView;
        batchView_t labelView;
        inferenceStats_t *stats = inferenceFn(
            model, modelSize, batchViewOf(&itemView, batch->samples[i]->item),
            batchViewOf(&labelView, batch->samples[i]->label), funcType, forwardReduction);
        totalLoss += stats->loss;

        size_t predicted = argmaxByTensor(stats->output, counts->numClasses);
        /* The raw sample label IS the one row's class vector (the view shares
         * its data), so the target argmax reads it directly. */
        size_t target = argmaxByTensor(batch->samples[i]->label, counts->numClasses);
        countPrediction(counts, predicted, target);

        freeInferenceStats(stats);
        freeSample(batch->samples[i]);
    }

    /* Pure sum — caller divides by totalSamples for MEAN. */
    return totalLoss;
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
                                          size_t *confusionMatrix, size_t numClasses,
                                          reduction_t forwardReduction, size_t m,
                                          const char *caller) {
    size_t numberOfBatches = requireEvalBatches(dataLoader, caller);

    size_t *tp = reserveMemory(numClasses * sizeof(size_t));
    size_t *predCount = reserveMemory(numClasses * sizeof(size_t));
    size_t *actualCount = reserveMemory(numClasses * sizeof(size_t));

    for (size_t c = 0; c < numClasses; c++) {
        tp[c] = 0;
        predCount[c] = 0;
        actualCount[c] = 0;
    }

    evalCounts_t counts = {.tp = tp,
                           .predCount = predCount,
                           .actualCount = actualCount,
                           .confusionMatrix = confusionMatrix,
                           .numClasses = numClasses};
    float totalLoss = 0.0f;
    size_t totalSamples = 0;

    if (m == 1) {
        for (size_t i = 0; i < numberOfBatches; i++) {
            batch_t *batch = dataLoader->getBatch(dataLoader, i);
            if (i == 0 && batch->size > 0) {
                evalFirstSample(model, modelSize, dataLoader, batch->samples[0], m, numClasses,
                                caller);
            }
            totalLoss += evaluateBatchInternal(model, modelSize, funcType, batch, inferenceFn,
                                               &counts, forwardReduction);
            totalSamples += batch->size;
            freeBatch(batch);
        }
    } else {
        totalLoss = evaluateStacked(caller, model, modelSize, funcType, dataLoader, numberOfBatches,
                                    inferenceFn, forwardReduction, m, &counts, &totalSamples);
    }
    requireStreamedSamples(totalSamples, caller);

    epochStats_t stats;
    if (forwardReduction == REDUCTION_MEAN) {
        stats.loss = totalLoss / (float)totalSamples;
    } else {
        stats.loss = totalLoss;
    }
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
                                        dataLoader_t *dataLoader, inferenceWithLossFn_t inferenceFn,
                                        reduction_t forwardReduction, size_t microBatchSize) {
    size_t m = resolveEvalMicroBatch(microBatchSize);
    (void)requireEvalBatches(dataLoader, "evaluationEpochWithMetrics");
    // Peek at first sample to derive numClasses from label shape
    batch_t *firstBatch = dataLoader->getBatch(dataLoader, 0);
    size_t numClasses = calcNumberOfElementsByTensor(firstBatch->samples[0]->label);
    for (size_t i = 0; i < firstBatch->size; i++) {
        freeSample(firstBatch->samples[i]);
    }
    freeBatch(firstBatch);

    return evaluateEpochInternal(model, modelSize, funcType, dataLoader, inferenceFn, NULL,
                                 numClasses, forwardReduction, m, "evaluationEpochWithMetrics");
}

classificationReport_t evaluationEpochWithReport(layer_t **model, size_t modelSize,
                                                 lossFuncType_t funcType, dataLoader_t *dataLoader,
                                                 inferenceWithLossFn_t inferenceFn,
                                                 size_t *cmBuffer, size_t numClasses,
                                                 reduction_t forwardReduction,
                                                 size_t microBatchSize) {
    /* #468 D11: 0 is evalFirstSample's "no metrics" opt-out, so the
     * first-sample check would let it through to 0-sized counters. */
    if (numClasses == 0) {
        PRINT_ERROR("evaluationEpochWithReport: numClasses is 0 -- it must equal the label's "
                    "element count (>= 1)");
        exit(1);
    }
    size_t m = resolveEvalMicroBatch(microBatchSize);
    // Zero the caller's CM buffer
    for (size_t i = 0; i < numClasses * numClasses; i++) {
        cmBuffer[i] = 0;
    }

    classificationReport_t report;
    report.stats =
        evaluateEpochInternal(model, modelSize, funcType, dataLoader, inferenceFn, cmBuffer,
                              numClasses, forwardReduction, m, "evaluationEpochWithReport");
    report.confusionMatrix = cmBuffer;
    report.numClasses = numClasses;
    return report;
}

/* #467: an untracked BatchNorm1d normalizes with batch statistics even in
 * evaluation, which runs one sample per call -- so a [1, C] input (or
 * [1, C, 1]) can never be evaluated. Walk the first eval sample's [1, ...]
 * shape through the model and fail before epoch 0 instead of after a full
 * training epoch. Models without an untracked BN skip the walk entirely. */
static void requireUntrackedBatchNormsEvaluable(layer_t **model, size_t modelSize,
                                                const tensor_t *evalItem) {
    size_t last = modelSize;
    for (size_t i = 0; i < modelSize; i++) {
        if (model[i]->type == BATCHNORM1D && !model[i]->config->batchNorm1d->trackRunningStats) {
            last = i;
        }
    }
    if (last == modelSize) {
        return;
    }
    size_t rank = evalItem->shape->numberOfDimensions;
    if (rank >= BATCH_VIEW_MAX_RANK) {
        return; /* evaluation itself fails fast in batchViewOf */
    }
    size_t dimsA[BATCH_VIEW_MAX_RANK];
    size_t orderA[BATCH_VIEW_MAX_RANK];
    size_t dimsB[BATCH_VIEW_MAX_RANK];
    size_t orderB[BATCH_VIEW_MAX_RANK];
    shape_t a = {.dimensions = dimsA, .orderOfDimensions = orderA, .numberOfDimensions = rank + 1};
    shape_t b = {.dimensions = dimsB, .orderOfDimensions = orderB, .numberOfDimensions = 0};
    dimsA[0] = 1;
    for (size_t d = 0; d < rank; d++) {
        dimsA[d + 1] = evalItem->shape->dimensions[d];
    }
    setOrderOfDimsForNewTensor(rank + 1, orderA);
    shape_t *in = &a;
    shape_t *out = &b;
    for (size_t i = 0; i <= last; i++) {
        if (model[i]->type == BATCHNORM1D) {
            char what[64];
            snprintf(what, sizeof what, "trainingRun pre-flight (model layer %zu)", i);
            batchNorm1dRequireEvaluable(model[i], in, what);
        }
        layerFunctions[model[i]->type].calcOutputShape(model[i], in, out);
        shape_t *t = in;
        in = out;
        out = t;
    }
}

trainingRunResult_t trainingRun(layer_t **model, size_t modelSize, lossConfig_t lossConfig,
                                dataLoader_t *trainDataLoader, dataLoader_t *evalDataLoader,
                                optimizer_t *optimizer, size_t numberOfEpochs,
                                calculateGradsFn_t calculateGradsFn,
                                inferenceWithLossFn_t inferenceFn,
                                const trainingRunOptions_t *options) {
    trainingRunResult_t result = {0};
    lrScheduler_t *lrScheduler = (options != NULL) ? options->lrScheduler : NULL;
    bsScheduler_t *bsScheduler = (options != NULL) ? options->bsScheduler : NULL;
    epochCallbackFn_t callback = (options != NULL) ? options->callback : NULL;
    bool stopOnNonFiniteLoss = (options != NULL) ? options->stopOnNonFiniteLoss : false;
    /* 0 means 1 (#152 spec §6.1): zero-initialised options keep per-sample training. */
    size_t microBatchSize =
        (options != NULL && options->microBatchSize != 0) ? options->microBatchSize : 1;

    if (lrScheduler != NULL && lrScheduler->optimizer != optimizer) {
        PRINT_ERROR("trainingRun: lrScheduler is wired to a different optimizer than the one "
                    "passed to trainingRun (#327)");
        exit(1);
    }
    if (bsScheduler != NULL && bsScheduler->dataLoader != trainDataLoader) {
        PRINT_ERROR("trainingRun: bsScheduler is wired to a different data loader than the "
                    "train loader passed to trainingRun (the eval loader is never resized)");
        exit(1);
    }
    if (bsScheduler != NULL && bsScheduler->optimizer != NULL &&
        bsScheduler->optimizer != optimizer) {
        PRINT_ERROR("trainingRun: bsScheduler's LR compensation is wired to a different "
                    "optimizer than the one passed to trainingRun");
        exit(1);
    }
    if (lrScheduler != NULL && bsScheduler != NULL && bsScheduler->optimizer != NULL) {
        PRINT_ERROR("trainingRun: lrScheduler and a compensating bsScheduler would both write "
                    "the LR every epoch (last writer wins silently); use one or the other");
        exit(1);
    }

    /* #152 spec §6.2 checks 1 and 2: every macro batch this run trains must
     * split into whole chunks of microBatchSize rows (b % m == 0, no ragged
     * tails). Both fire before anything is read or trained. */
    if (trainDataLoader->batchSize % microBatchSize != 0) {
        PRINT_ERROR("trainingRun: trainDataLoader batchSize %u is not divisible by "
                    "microBatchSize %zu (b %% m == 0 is required)",
                    (unsigned)trainDataLoader->batchSize, microBatchSize);
        exit(1);
    }
    if (bsScheduler != NULL && microBatchSize > 1) {
        /* Epoch e of this run trains at the batch the scheduler writes when
         * its lastEpoch reaches lastEpoch + e (the scheduler may already have
         * been stepped); the step after the final epoch trains nothing. At
         * m == 1 every batch divides, so the walk is skipped and a default run
         * behaves exactly as before (incl. when a non-finite target fails). */
        for (size_t epoch = 1; epoch < numberOfEpochs; epoch++) {
            size_t scheduled = bsSchedulerBatchSizeAt(bsScheduler, bsScheduler->lastEpoch + epoch);
            if (scheduled % microBatchSize != 0) {
                PRINT_ERROR("trainingRun: the batch-size scheduler sets batch %zu for epoch %zu, "
                            "which is not divisible by microBatchSize %zu (the whole schedule "
                            "is checked before epoch 0)",
                            scheduled, epoch, microBatchSize);
                exit(1);
            }
        }
    }

    (void)requireEvalBatches(evalDataLoader, "trainingRun");
    batch_t *firstBatch = evalDataLoader->getBatch(evalDataLoader, 0);
    size_t numClasses = calcNumberOfElementsByTensor(firstBatch->samples[0]->label);
    requireUntrackedBatchNormsEvaluable(model, modelSize, firstBatch->samples[0]->item);
    for (size_t i = 0; i < firstBatch->size; i++) {
        freeSample(firstBatch->samples[i]);
    }
    freeBatch(firstBatch);

    /* Policy: trainingRun is the SOLE place that hardcodes forwardReduction.
     * Both train and eval use MEAN to keep the two reported losses comparable
     * (per-sample mean, same units). Direct callers of trainingEpochDefault /
     * evaluationEpoch can pick either reduction freely. */
    const reduction_t forwardReduction = REDUCTION_MEAN;

    for (size_t epoch = 0; epoch < numberOfEpochs; epoch++) {
        /* epoch 0 already got its permutation from dataLoaderInit's
         * init-shuffle; from epoch 1 on, opt in to a fresh per-epoch
         * permutation (#381). dataLoaderReshuffle self-gates on
         * shuffle/reshufflePerEpoch, so this is a no-op unless the caller
         * explicitly enabled it via dataLoaderSetReshufflePerEpoch. The eval
         * loader is NEVER reshuffled — evaluation stays order-stable. */
        if (epoch > 0) {
            dataLoaderReshuffle(trainDataLoader);
        }

        /* Captured BEFORE training: the values this epoch trains with. The
         * schedulers step only after the callback, bsScheduler last, so the
         * batch written now is what the next epoch's capture reads. Same
         * datasetSize / batchSize as trainingEpochDefault's numberOfBatches. */
        epochInfo_t info = {0};
        info.epoch = epoch;
        info.batchSize = trainDataLoader->batchSize;
        info.parameterUpdates = trainDataLoader->getDatasetSize() / info.batchSize;
        info.learningRate = optimizerFunctions[optimizer->type].getLr(optimizer);

        float trainLoss =
            trainingEpochDefault(model, modelSize, lossConfig, trainDataLoader, optimizer,
                                 calculateGradsFn, forwardReduction, microBatchSize);
        epochStats_t evalStats = evaluateEpochInternal(
            model, modelSize, lossConfig.funcType, evalDataLoader, inferenceFn, NULL, numClasses,
            forwardReduction, 1, "trainingRun");
        info.trainLoss = trainLoss;

        if (callback != NULL) {
            callback(info, evalStats);
        }
        result.epochsCompleted = epoch + 1;
        result.finalTrainLoss = trainLoss;
        result.finalEvalStats = evalStats;

        /* Opt-in divergence stop: the epoch that diverged is logged (callback
         * above) and reported as the final one; nothing steps past it. */
        if (stopOnNonFiniteLoss && (!isfinite(trainLoss) || !isfinite(evalStats.loss))) {
            result.stoppedOnNonFiniteLoss = true;
            break;
        }
        if (lrScheduler != NULL) {
            lrSchedulerStep(lrScheduler);
        }
        if (bsScheduler != NULL) {
            bsSchedulerStep(bsScheduler);
        }
    }

    return result;
}
