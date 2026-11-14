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
#include "Model.h"
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

/* Positive-scale SYM_INT32 codes preserve value ordering; compare them directly. */
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
        /* BFP groups have different exponents: compare dequantized values.
         * Chunk offsets remain byte-aligned; this helper starts at element zero. */
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

/* Computed from sizes, before any getBatch -- a dataset smaller than
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

/* Every metric indexes [0, numClasses); a caller-supplied
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

/* Per-class counters of the metric entry points; NULL for evaluationEpoch.
 * The counters are reserved at the first streamed sample, whose label also
 * gives numClasses when it is 0. */
typedef struct evalCounts {
    size_t *tp;
    size_t *predCount;
    size_t *actualCount;
    size_t *confusionMatrix; /* NULL: no report */
    size_t numClasses;
} evalCounts_t;

static void initCounts(evalCounts_t *counts, tensor_t *firstLabel, const char *caller) {
    if (counts->numClasses == 0) {
        counts->numClasses = calcNumberOfElementsByTensor(firstLabel);
    } else {
        requireNumClassesMatchesLabel(counts->numClasses, firstLabel, caller);
    }
    size_t n = counts->numClasses;
    counts->tp = reserveMemory(n * sizeof(size_t));
    counts->predCount = reserveMemory(n * sizeof(size_t));
    counts->actualCount = reserveMemory(n * sizeof(size_t));
    for (size_t c = 0; c < n; c++) {
        counts->tp[c] = 0;
        counts->predCount[c] = 0;
        counts->actualCount[c] = 0;
    }
}

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

/* An untracked BatchNorm1d normalizes with batch statistics even
 * in evaluation -- over its chunk of `rows` samples. Walk [rows, ...item]
 * through the model and fail before any forward instead of mid-epoch. Models
 * without an untracked BN skip the walk entirely. */
static void requireUntrackedBatchNormsEvaluable(const model_t *model, const tensor_t *evalItem,
                                                size_t rows, const char *caller) {
    size_t last = model->numLayers;
    for (size_t i = 0; i < model->numLayers; i++) {
        if (model->layers[i]->type == BATCHNORM1D &&
            !model->layers[i]->config->batchNorm1d->trackRunningStats) {
            last = i;
        }
    }
    if (last == model->numLayers) {
        return;
    }
    size_t rank = evalItem->shape->numberOfDimensions;
    if (rank >= BATCH_VIEW_MAX_RANK) {
        return; /* evaluation itself fails fast in batchViewOf */
    }
    size_t dimsA[BATCH_VIEW_MAX_RANK];
    size_t orderA[BATCH_VIEW_MAX_RANK];
    shape_t a = {.dimensions = dimsA, .orderOfDimensions = orderA, .numberOfDimensions = rank + 1};
    dimsA[0] = rows;
    for (size_t d = 0; d < rank; d++) {
        dimsA[d + 1] = evalItem->shape->dimensions[d];
    }
    setOrderOfDimsForNewTensor(rank + 1, orderA);

    /* A topological prefix is a valid wiring, so the shape pass stops at the last such BN. */
    model_t prefix = {model->layers, last + 1, model->inputStart, model->inputIds};
    size_t numShapes = last + 2;
    size_t maxRank = (rank + 1 > 2u) ? rank + 1 : 2u;
    /* Heap, not stack: the shapes of a deep model would cost too much MCU stack. */
    shape_t *shapes =
        reserveMemory(numShapes * sizeof(shape_t) + 2u * numShapes * maxRank * sizeof(size_t));
    if (shapes == NULL) {
        PRINT_ERROR("%s pre-flight: no memory for %zu shapes", caller, numShapes);
        exit(1);
    }
    size_t *pool = (size_t *)(shapes + numShapes);
    for (size_t s = 0; s < numShapes; s++) {
        shapes[s] = (shape_t){.dimensions = pool + 2u * s * maxRank,
                              .orderOfDimensions = pool + (2u * s + 1u) * maxRank};
    }
    modelShapePass(&prefix, &a, shapes);
    for (size_t i = 0; i <= last; i++) {
        if (model->layers[i]->type == BATCHNORM1D) {
            uint16_t ids[MODEL_MAX_ARITY];
            (void)modelNodeInputs(model, i, ids);
            char what[128];
            snprintf(what, sizeof what, "%s pre-flight (model layer %zu, %zu-row evaluation chunk)",
                     caller, i, rows);
            batchNorm1dRequireEvaluable(model->layers[i], &shapes[ids[0]], what);
        }
    }
    freeReservedMemory(shapes);
}

/* trainingRun reads numClasses and the evaluation pre-flight off the first
 * sample of getBatch(0); a custom loader may report a dataset yet return an
 * empty batch. */
static void requireNonEmptyFirstBatch(batch_t *firstBatch, const char *caller) {
    if (firstBatch->size == 0) {
        freeBatch(firstBatch);
        PRINT_ERROR("%s: the evaluation loader's first batch is empty (numClasses and the "
                    "pre-flight are derived from its first sample)",
                    caller);
        exit(1);
    }
}

/* The NOMINAL eval sample count, the samples of the loader's full batches.
 * The chunk check and trainingRun's scheduler gate must count alike. */
static size_t evalNominalCount(dataLoader_t *dataLoader) {
    return (dataLoader->getDatasetSize() / dataLoader->batchSize) * dataLoader->batchSize;
}

/* Untracked-BN evaluability over the full chunks AND the ragged tail
 * chunk, judged on the NOMINAL sample count (a loader whose batches differ
 * from batchSize is covered at run time by BatchNorm1d's own forward guard).
 * Callers run requireEvalBatches first, so batchSize and the count are >= 1. */
static void requireEvaluableChunks(const model_t *model, dataLoader_t *dataLoader,
                                   const tensor_t *firstItem, size_t m, const char *caller) {
    size_t nominal = evalNominalCount(dataLoader);
    size_t full = (m < nominal) ? m : nominal;
    requireUntrackedBatchNormsEvaluable(model, firstItem, full, caller);
    size_t tail = nominal % m;
    if (nominal > m && tail != 0) {
        requireUntrackedBatchNormsEvaluable(model, firstItem, tail, caller);
    }
}

/* Runs once, on the first streamed sample of every evaluation, whatever m. */
static void evalFirstSample(const model_t *model, dataLoader_t *dataLoader, sample_t *first,
                            size_t m, evalCounts_t *counts, const char *caller) {
    requireEvaluableChunks(model, dataLoader, first->item, m, caller);
    if (counts != NULL) {
        initCounts(counts, first->label, caller);
    }
}

/* Extends requireEvalBatches past the size check: a custom getBatch returning only empty
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
    tensor_t *out = stats->output;
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
 * row-weighted loss (x rows for MEAN, plain for SUM). callIndex numbers the
 * evaluation epoch's calls, one per chunk. */
static float evaluateChunk(const char *caller, const model_t *model, lossFuncType_t funcType,
                           inferenceWithLossFn_t inferenceFn, reduction_t forwardReduction,
                           tensor_t *referenceItem, tensor_t *referenceLabel, uint8_t *itemBuffer,
                           uint8_t *labelBuffer, size_t rows, size_t C, size_t chunkFirst,
                           size_t callIndex, evalCounts_t *counts, const trainingCall_t *call) {
    batchView_t itemView;
    batchView_t labelView;
    tensor_t *item = batchViewOf(&itemView, referenceItem);
    tensor_t *label = batchViewOf(&labelView, referenceLabel);
    item->data = itemBuffer;
    itemView.dimensions[0] = rows;
    label->data = labelBuffer;
    labelView.dimensions[0] = rows;

    trainingCall_t next;
    rngKey_t callKey;
    inferenceStats_t *stats = inferenceFn(model, item, label, funcType, forwardReduction,
                                          trainingCallForIndex(call, callIndex, &next, &callKey));
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

/* m > 1: gathers consecutive samples of the loader's stream --
 * ACROSS batch_t boundaries, since eval loaders typically use batchSize 1 --
 * into chunks of m rows, then runs the N mod m remainder. Counts come from
 * the stream (a replay wrapper's batches exceed batchSize). The reference
 * sample's tensors are dataset-owned and outlive its sample_t (freeSample
 * frees only the wrapper). */
static float evaluateStacked(const char *caller, const char *knob, const model_t *model,
                             lossFuncType_t funcType, dataLoader_t *dataLoader,
                             size_t numberOfBatches, inferenceWithLossFn_t inferenceFn,
                             reduction_t forwardReduction, size_t m, evalCounts_t *counts,
                             size_t *totalSamples, const trainingCall_t *call) {
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
        batch_t *batch = dataLoader->getBatch(dataLoader, b, NULL);
        for (size_t i = 0; i < batch->size; i++) {
            sample_t *sample = batch->samples[i];
            if (referenceItem == NULL) {
                evalFirstSample(model, dataLoader, sample, m, counts, caller);
                referenceItem = sample->item;
                referenceLabel = sample->label;
                /* Validates the reference's own dtype/sparsity before its
                 * bytes size the buffers. */
                stackGatherRequireStackable(caller, knob, referenceItem, referenceItem, "item", 0,
                                            m);
                stackGatherRequireStackable(caller, knob, referenceLabel, referenceLabel, "label",
                                            0, m);
                itemBytes = calcBytesPerTensor(referenceItem);
                labelBytes = calcBytesPerTensor(referenceLabel);
                C = calcNumberOfElementsByTensor(referenceLabel);
                itemBuffer = stackGatherReserveBuffer(caller, knob, m, itemBytes, "item");
                labelBuffer = stackGatherReserveBuffer(caller, knob, m, labelBytes, "label");
            } else {
                stackGatherRequireStackable(caller, knob, referenceItem, sample->item, "item",
                                            streamed, m);
                stackGatherRequireStackable(caller, knob, referenceLabel, sample->label, "label",
                                            streamed, m);
            }
            memcpy(itemBuffer + rows * itemBytes, sample->item->data, itemBytes);
            memcpy(labelBuffer + rows * labelBytes, sample->label->data, labelBytes);
            rows++;
            streamed++;
            freeSample(sample);
            if (rows == m) {
                totalLoss +=
                    evaluateChunk(caller, model, funcType, inferenceFn, forwardReduction,
                                  referenceItem, referenceLabel, itemBuffer, labelBuffer, rows, C,
                                  streamed - rows, (streamed - rows) / m, counts, call);
                rows = 0;
            }
        }
        freeBatch(batch);
    }
    if (rows > 0) {
        totalLoss += evaluateChunk(caller, model, funcType, inferenceFn, forwardReduction,
                                   referenceItem, referenceLabel, itemBuffer, labelBuffer, rows, C,
                                   streamed - rows, (streamed - rows) / m, counts, call);
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

/* The per-sample loop shared by evaluationBatch and every m == 1 evaluation.
 * counts NULL: no prediction counting. callBase: the evaluation epoch's call
 * index of the batch's first sample (one call per sample). Returns the plain
 * loss sum; the epoch level divides for MEAN. */
static float evaluateSamples(const model_t *model, lossFuncType_t funcType, batch_t *batch,
                             inferenceWithLossFn_t inferenceFn, reduction_t forwardReduction,
                             evalCounts_t *counts, size_t callBase, const trainingCall_t *call) {
    float totalLoss = 0.0f;

    for (size_t i = 0; i < batch->size; i++) {
        batchView_t itemView;
        batchView_t labelView;
        trainingCall_t next;
        rngKey_t callKey;
        inferenceStats_t *stats = inferenceFn(
            model, batchViewOf(&itemView, batch->samples[i]->item),
            batchViewOf(&labelView, batch->samples[i]->label), funcType, forwardReduction,
            trainingCallForIndex(call, callBase + i, &next, &callKey));
        totalLoss += stats->loss;

        if (counts != NULL) {
            size_t predicted = argmaxByTensor(stats->output, counts->numClasses);
            /* The raw sample label IS the one row's class vector (the view
             * shares its data), so the target argmax reads it directly. */
            size_t target = argmaxByTensor(batch->samples[i]->label, counts->numClasses);
            countPrediction(counts, predicted, target);
        }

        freeInferenceStats(stats);
        freeSample(batch->samples[i]);
    }
    return totalLoss;
}

float evaluationBatch(const model_t *model, lossFuncType_t funcType, batch_t *batch,
                      inferenceWithLossFn_t inferenceFn, reduction_t forwardReduction) {
    return evaluateSamples(model, funcType, batch, inferenceFn, forwardReduction, NULL, 0, NULL);
}

/* One evaluation epoch over numberOfBatches loader batches: per sample at
 * m == 1 (each batch summed on its own, then added), stacked at m > 1.
 * counts (NULLable) is filled from the first streamed sample on. Returns the
 * loss reduced per forwardReduction over the streamed count, written to
 * *totalSamples. */
static float evaluateEpochInternal(const char *caller, const char *knob, const model_t *model,
                                   lossFuncType_t funcType, dataLoader_t *dataLoader,
                                   size_t numberOfBatches, inferenceWithLossFn_t inferenceFn,
                                   reduction_t forwardReduction, size_t m, evalCounts_t *counts,
                                   size_t *totalSamples, const trainingCall_t *call) {
    float totalLoss = 0.0f;
    size_t streamed = 0;

    if (m == 1) {
        /* The first sample of the first NON-EMPTY batch, as on the m > 1 path. */
        bool firstSeen = false;
        for (size_t i = 0; i < numberOfBatches; i++) {
            batch_t *batch = dataLoader->getBatch(dataLoader, i, NULL);
            if (!firstSeen && batch->size > 0) {
                evalFirstSample(model, dataLoader, batch->samples[0], m, counts, caller);
                firstSeen = true;
            }
            totalLoss += evaluateSamples(model, funcType, batch, inferenceFn, forwardReduction,
                                         counts, streamed, call);
            streamed += batch->size;
            freeBatch(batch);
        }
    } else {
        totalLoss = evaluateStacked(caller, knob, model, funcType, dataLoader, numberOfBatches,
                                    inferenceFn, forwardReduction, m, counts, &streamed, call);
    }
    requireStreamedSamples(streamed, caller);
    *totalSamples = streamed;

    if (forwardReduction == REDUCTION_MEAN) {
        return totalLoss / (float)streamed;
    }
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

/* The metric entries' epoch: numClasses 0 takes it from the first label. */
static epochStats_t evaluateEpochWithMetrics(const char *caller, const char *knob,
                                             const model_t *model, lossFuncType_t funcType,
                                             dataLoader_t *dataLoader, size_t numberOfBatches,
                                             inferenceWithLossFn_t inferenceFn,
                                             size_t *confusionMatrix, size_t numClasses,
                                             reduction_t forwardReduction, size_t m,
                                             const trainingCall_t *call) {
    evalCounts_t counts = {.confusionMatrix = confusionMatrix, .numClasses = numClasses};
    size_t totalSamples = 0;
    epochStats_t stats;
    stats.loss =
        evaluateEpochInternal(caller, knob, model, funcType, dataLoader, numberOfBatches,
                              inferenceFn, forwardReduction, m, &counts, &totalSamples, call);
    stats.accuracy = computeAccuracy(counts.tp, counts.numClasses, totalSamples);
    stats.precision = computeMacroPrecision(counts.tp, counts.predCount, counts.numClasses);
    stats.recall = computeMacroRecall(counts.tp, counts.actualCount, counts.numClasses);
    stats.f1 = computeMacroF1(counts.tp, counts.predCount, counts.actualCount, counts.numClasses);

    freeReservedMemory(counts.tp);
    freeReservedMemory(counts.predCount);
    freeReservedMemory(counts.actualCount);
    return stats;
}

/* The standalone entries' setup, once per call: the batch count and, at
 * m > 1, the forward-only FLOAT32 gate (backward-only fields stay free). */
static size_t requireEvalEntry(const char *caller, const model_t *model, dataLoader_t *dataLoader,
                               size_t m) {
    size_t numberOfBatches = requireEvalBatches(dataLoader, caller);
    if (m > 1) {
        stackGatherRequireFloat32Model(caller, "microBatchSize", model, m,
                                       layerForwardNonFloat32Field);
    }
    return numberOfBatches;
}

float evaluationEpoch(const model_t *model, lossFuncType_t funcType, dataLoader_t *dataLoader,
                      inferenceWithLossFn_t inferenceFn, reduction_t forwardReduction,
                      size_t microBatchSize) {
    modelRequireValid(model, "evaluationEpoch");
    size_t m = resolveEvalMicroBatch(microBatchSize);
    size_t numberOfBatches = requireEvalEntry("evaluationEpoch", model, dataLoader, m);
    size_t totalSamples = 0;
    return evaluateEpochInternal("evaluationEpoch", "microBatchSize", model, funcType, dataLoader,
                                 numberOfBatches, inferenceFn, forwardReduction, m, NULL,
                                 &totalSamples, NULL);
}

epochStats_t evaluationEpochWithMetrics(const model_t *model, lossFuncType_t funcType,
                                        dataLoader_t *dataLoader, inferenceWithLossFn_t inferenceFn,
                                        reduction_t forwardReduction, size_t microBatchSize) {
    modelRequireValid(model, "evaluationEpochWithMetrics");
    size_t m = resolveEvalMicroBatch(microBatchSize);
    size_t numberOfBatches = requireEvalEntry("evaluationEpochWithMetrics", model, dataLoader, m);
    return evaluateEpochWithMetrics("evaluationEpochWithMetrics", "microBatchSize", model, funcType,
                                    dataLoader, numberOfBatches, inferenceFn, NULL, 0,
                                    forwardReduction, m, NULL);
}

classificationReport_t
evaluationEpochWithReport(const model_t *model, lossFuncType_t funcType, dataLoader_t *dataLoader,
                          inferenceWithLossFn_t inferenceFn, size_t *cmBuffer, size_t numClasses,
                          reduction_t forwardReduction, size_t microBatchSize) {
    modelRequireValid(model, "evaluationEpochWithReport");
    /* 0 internally means "take numClasses from the label". */
    if (numClasses == 0) {
        PRINT_ERROR("evaluationEpochWithReport: numClasses is 0 -- it must equal the label's "
                    "element count (>= 1)");
        exit(1);
    }
    size_t m = resolveEvalMicroBatch(microBatchSize);
    size_t numberOfBatches = requireEvalEntry("evaluationEpochWithReport", model, dataLoader, m);
    for (size_t i = 0; i < numClasses * numClasses; i++) {
        cmBuffer[i] = 0;
    }

    classificationReport_t report;
    report.stats = evaluateEpochWithMetrics("evaluationEpochWithReport", "microBatchSize", model,
                                            funcType, dataLoader, numberOfBatches, inferenceFn,
                                            cmBuffer, numClasses, forwardReduction, m, NULL);
    report.confusionMatrix = cmBuffer;
    report.numClasses = numClasses;
    return report;
}

trainingRunResult_t trainingRun(const model_t *model, lossConfig_t lossConfig,
                                dataLoader_t *trainDataLoader, dataLoader_t *evalDataLoader,
                                optimizer_t *optimizer, size_t numberOfEpochs,
                                calculateGradsFn_t calculateGradsFn,
                                inferenceWithLossFn_t inferenceFn,
                                const trainingRunOptions_t *options) {
    /* Before the evaluation pre-flight, the first reader of the wiring. */
    modelRequireValid(model, "trainingRun");
    trainingRunResult_t result = {0};
    lrScheduler_t *lrScheduler = (options != NULL) ? options->lrScheduler : NULL;
    bsScheduler_t *bsScheduler = (options != NULL) ? options->bsScheduler : NULL;
    epochCallbackFn_t callback = (options != NULL) ? options->callback : NULL;
    bool stopOnNonFiniteLoss = (options != NULL) ? options->stopOnNonFiniteLoss : false;
    /* 0 means 1: zero-initialised options keep per-sample training. */
    size_t microBatchSize =
        (options != NULL && options->microBatchSize != 0) ? options->microBatchSize : 1;
    /* Evaluation inherits the training micro-batch unless set. */
    size_t evalMicroBatchSize = (options != NULL && options->evalMicroBatchSize != 0)
                                    ? options->evalMicroBatchSize
                                    : microBatchSize;
    /* Every training call gets the options' scheduler; evaluation gets it
     * behind the entry check below. The epoch level reads rng;
     * evaluation gets one key per epoch in the loop below. */
    trainingRng_t *rng = (options != NULL) ? options->rng : NULL;
    const trainingCall_t trainCall = {.remat = (options != NULL) ? options->remat : NULL,
                                      .rng = rng};

    if (trainDataLoader == evalDataLoader) {
        PRINT_ERROR("trainingRun: trainDataLoader and evalDataLoader must be distinct loader "
                    "objects (the eval loader keeps one order per run and is never resized, "
                    "while the train loader is reshuffled each epoch and resized by the "
                    "batch-size scheduler; to evaluate on the training data, build a second "
                    "loader over the same dataset)");
        exit(1);
    }
    if (lrScheduler != NULL && lrScheduler->optimizer != optimizer) {
        PRINT_ERROR("trainingRun: lrScheduler is wired to a different optimizer than the one "
                    "passed to trainingRun");
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

    /* Validate current and scheduled macro batches before training; every
     * batch must split into whole microbatches. */
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
    /* Checked again at every epoch boundary: the dataset may change. */
    size_t trainDatasetSize = trainDataLoader->getDatasetSize();
    if (trainDatasetSize / trainDataLoader->batchSize == 0) {
        PRINT_ERROR("trainingRun: the training loader yields no batch (dataset size %zu < "
                    "batchSize %u)",
                    trainDatasetSize, (unsigned)trainDataLoader->batchSize);
        exit(1);
    }
    if (microBatchSize > 1) {
        stackGatherRequireFloat32Model("trainingRun", "microBatchSize", model, microBatchSize,
                                       layerNonFloat32Field);
    }
    /* The CrossEntropy backward writes the fused p - y gradient and skips the
     * last layer, which is right only for a Softmax. */
    const layer_t *lastLayer = model->layers[model->numLayers - 1];
    if (lossConfig.funcType == CROSS_ENTROPY && lastLayer->type != SOFTMAX) {
        PRINT_ERROR("trainingRun: a CROSS_ENTROPY loss needs a Softmax as the model's last layer "
                    "(layer %zu has layerType_t %d)",
                    model->numLayers - 1, (int)lastLayer->type);
        exit(1);
    }

    (void)requireEvalBatches(evalDataLoader, "trainingRun");
    /* A shuffle = true eval loader gets one order per run, under the run's
     * evaluation shuffle key, before the label probe below fetches from it;
     * it is never reshuffled, so every evaluation sees the same order. Without
     * options->rng it has no key and stops here. */
    if (rng != NULL) {
        rngKey_t evalShuffleKey = trainingRngShuffleEvalKey(rng);
        dataLoaderShuffle(evalDataLoader, &evalShuffleKey);
    } else {
        dataLoaderShuffle(evalDataLoader, NULL);
    }
    batch_t *firstBatch = evalDataLoader->getBatch(evalDataLoader, 0, NULL);
    requireNonEmptyFirstBatch(firstBatch, "trainingRun");
    size_t numClasses = calcNumberOfElementsByTensor(firstBatch->samples[0]->label);
    if (evalMicroBatchSize > 1) {
        stackGatherRequireFloat32Model("trainingRun", "evalMicroBatchSize", model,
                                       evalMicroBatchSize, layerForwardNonFloat32Field);
    }
    requireEvaluableChunks(model, evalDataLoader, firstBatch->samples[0]->item, evalMicroBatchSize,
                           "trainingRun");
    /* Share the training scheduler only for matching evaluation row counts
     * without a nominal tail; live shape mismatches are checked when binding. */
    size_t evalNominal = evalNominalCount(evalDataLoader);
    bool evalOnTheScheduler =
        evalMicroBatchSize == microBatchSize && evalNominal % evalMicroBatchSize == 0;
    const trainingCall_t evalCall = {.remat = evalOnTheScheduler ? trainCall.remat : NULL};
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
        /* Use the persistent RNG epoch for per-epoch shuffle; shuffle-once
         * loaders always use epoch zero's key. */
        if (rng != NULL) {
            rngKey_t shuffleKey = trainingRngShuffleKey(rng, trainDataLoader->reshufflePerEpoch);
            dataLoaderShuffle(trainDataLoader, &shuffleKey);
        } else {
            dataLoaderShuffle(trainDataLoader, NULL);
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
            trainingEpochDefault(model, lossConfig, trainDataLoader, optimizer, calculateGradsFn,
                                 forwardReduction, microBatchSize, &trainCall);
        rngKey_t evalKey;
        trainingCall_t epochEvalCall = evalCall;
        if (rng != NULL) {
            evalKey = trainingRngEvalKey(rng);
            epochEvalCall.key = &evalKey;
        }
        size_t evalBatches = requireEvalBatches(evalDataLoader, "trainingRun");
        epochStats_t evalStats = evaluateEpochWithMetrics(
            "trainingRun", "evalMicroBatchSize", model, lossConfig.funcType, evalDataLoader,
            evalBatches, inferenceFn, NULL, numClasses, forwardReduction, evalMicroBatchSize,
            &epochEvalCall);
        if (rng != NULL) {
            trainingRngAdvanceEpoch(rng);
        }
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
