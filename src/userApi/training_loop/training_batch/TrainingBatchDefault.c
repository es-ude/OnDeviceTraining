#define SOURCE_FILE "TRAINING_BATCH_DEFAULT"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "BatchView.h"
#include "Common.h"
#include "DataLoaderApi.h"
#include "LayerConfigAccess.h"
#include "StorageApi.h"
#include "TrainingBatchDefault.h"

/* m == 1: PR3a's per-sample walk, unchanged -- each sample is wrapped by
 * batchViewOf (no copy, no heap), so default runs stay bit-identical. */
static float trainingBatchPerSample(layer_t **model, size_t modelSize, lossConfig_t lossConfig,
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
    return totalLoss;
}

/* Gather buffer of m rows x perSampleBytes (spec §6.4): overflow-checked
 * multiply, NULL from reserveMemory fails fast -- never a copy through NULL. */
static uint8_t *reserveGatherBuffer(size_t m, size_t perSampleBytes, const char *what) {
    if (perSampleBytes != 0 && m > SIZE_MAX / perSampleBytes) {
        PRINT_ERROR("trainingBatchDefault: the %s gather buffer (microBatchSize %zu x %zu bytes "
                    "per sample) overflows size_t",
                    what, m, perSampleBytes);
        exit(1);
    }
    uint8_t *buffer = reserveMemory(m * perSampleBytes);
    if (buffer == NULL) {
        PRINT_ERROR("trainingBatchDefault: reserving the %s gather buffer failed (microBatchSize "
                    "%zu x %zu bytes per sample)",
                    what, m, perSampleBytes);
        exit(1);
    }
    return buffer;
}

/* FLOAT32 gate (spec §6.6, D3): evaluated over the whole model once per macro
 * batch when m > 1, before any buffer is reserved or any chunk computed. */
static void requireFloat32Model(layer_t **model, size_t modelSize, size_t m) {
    for (size_t i = 0; i < modelSize; i++) {
        if (!layerIsFloat32Only(model[i])) {
            PRINT_ERROR("trainingBatchDefault: microBatchSize %zu > 1 is FLOAT32-only, but layer "
                        "%zu (layerType_t %d) has a non-FLOAT32 %s",
                        m, i, (int)model[i]->type, layerNonFloat32Field(model[i]));
            exit(1);
        }
    }
}

/* Per-chunk validation (spec §6.3): a stacked sample must be FLOAT32, carry no
 * sparsity and match the reference in rank, dimensions and order. The
 * reference is the macro batch's sample 0 -- the sample the gather buffers
 * were sized from -- so it is also every chunk's first-sample reference. */
static void requireStackable(tensor_t *reference, tensor_t *t, const char *what, size_t sampleIndex,
                             size_t m) {
    if (t->quantization->type != FLOAT32) {
        PRINT_ERROR("trainingBatchDefault: microBatchSize %zu > 1 is FLOAT32-only, but the %s of "
                    "sample %zu has dtype %d",
                    m, what, sampleIndex, (int)t->quantization->type);
        exit(1);
    }
    if (t->sparsity != NULL) {
        PRINT_ERROR("trainingBatchDefault: microBatchSize %zu > 1 cannot stack the %s of sample "
                    "%zu: it carries sparsity",
                    m, what, sampleIndex);
        exit(1);
    }
    size_t rank = reference->shape->numberOfDimensions;
    if (t->shape->numberOfDimensions != rank) {
        PRINT_ERROR("trainingBatchDefault: the %s of sample %zu has rank %zu, sample 0 has rank "
                    "%zu -- a stacked chunk needs shape-identical samples",
                    what, sampleIndex, t->shape->numberOfDimensions, rank);
        exit(1);
    }
    for (size_t d = 0; d < rank; d++) {
        if (t->shape->dimensions[d] != reference->shape->dimensions[d] ||
            t->shape->orderOfDimensions[d] != reference->shape->orderOfDimensions[d]) {
            PRINT_ERROR("trainingBatchDefault: the %s of sample %zu differs from sample 0 in "
                        "dimension %zu (size %zu vs %zu, order %zu vs %zu) -- a stacked chunk "
                        "needs shape-identical samples",
                        what, sampleIndex, d, t->shape->dimensions[d],
                        reference->shape->dimensions[d], t->shape->orderOfDimensions[d],
                        reference->shape->orderOfDimensions[d]);
            exit(1);
        }
    }
}

/* m > 1 (spec §6.3): b/m chunks of exactly m rows. Each chunk's item and label
 * bytes are copied row after row into the two gather buffers; the stacked
 * tensors are stack-local batch views of the chunk's first sample, re-pointed
 * at the buffers with dims[0] = m (shape [m, ...sampleShape], order
 * [0, sampleOrder + 1]; quantization and sparsity stay shared with that first
 * sample). One calculateGradsFn call per chunk; the chunk's sample_t structs
 * are freed after it (the tensors stay dataset-owned). Returns the loss sum
 * weighted by rows for MEAN (Σ chunkLoss * m), plain for SUM (spec §6.5). */
static float trainingBatchStacked(layer_t **model, size_t modelSize, lossConfig_t lossConfig,
                                  batch_t *batch, calculateGradsFn_t calculateGradsFn,
                                  reduction_t forwardReduction, size_t m) {
    /* Sample 0's TENSORS (dataset-owned) stay valid for the whole call; its
     * sample_t is freed with chunk 0, so the reference is captured up front. */
    tensor_t *referenceItem = batch->samples[0]->item;
    tensor_t *referenceLabel = batch->samples[0]->label;
    size_t itemBytes = calcBytesPerTensor(referenceItem);
    size_t labelBytes = calcBytesPerTensor(referenceLabel);
    uint8_t *itemBuffer = reserveGatherBuffer(m, itemBytes, "item");
    uint8_t *labelBuffer = reserveGatherBuffer(m, labelBytes, "label");
    const float rowWeight = (forwardReduction == REDUCTION_MEAN) ? (float)m : 1.0f;
    float totalLoss = 0.0f;

    for (size_t first = 0; first < batch->size; first += m) {
        for (size_t r = 0; r < m; r++) {
            sample_t *sample = batch->samples[first + r];
            requireStackable(referenceItem, sample->item, "item", first + r, m);
            requireStackable(referenceLabel, sample->label, "label", first + r, m);
        }
        for (size_t r = 0; r < m; r++) {
            const sample_t *sample = batch->samples[first + r];
            memcpy(itemBuffer + r * itemBytes, sample->item->data, itemBytes);
            memcpy(labelBuffer + r * labelBytes, sample->label->data, labelBytes);
        }
        batchView_t itemView;
        batchView_t labelView;
        tensor_t *stackedItem = batchViewOf(&itemView, batch->samples[first]->item);
        tensor_t *stackedLabel = batchViewOf(&labelView, batch->samples[first]->label);
        stackedItem->data = itemBuffer;
        itemView.dimensions[0] = m;
        stackedLabel->data = labelBuffer;
        labelView.dimensions[0] = m;

        trainingStats_t *stats = calculateGradsFn(model, modelSize, lossConfig, forwardReduction,
                                                  stackedItem, stackedLabel);
        totalLoss += stats->loss * rowWeight;
        freeTrainingStats(stats);
        for (size_t r = 0; r < m; r++) {
            freeSample(batch->samples[first + r]);
        }
    }

    freeReservedMemory(labelBuffer);
    freeReservedMemory(itemBuffer);
    return totalLoss;
}

float trainingBatchDefault(layer_t **model, size_t modelSize, lossConfig_t lossConfig,
                           batch_t *batch, calculateGradsFn_t calculateGradsFn,
                           reduction_t forwardReduction, size_t microBatchSize) {
    size_t m = (microBatchSize == 0) ? 1 : microBatchSize;
    if (batch->size % m != 0) {
        PRINT_ERROR("trainingBatchDefault: batch size %zu is not divisible by microBatchSize %zu "
                    "(b %% m == 0 is required; a replay loader must keep its appended sample "
                    "count divisible by m)",
                    batch->size, m);
        exit(1);
    }
    if (m > 1 && batch->size == 0) {
        /* b = 0 passes b % m == 0, but the stacked path sizes its gather
         * buffers from sample 0 -- there is none. (m == 1 keeps its
         * pre-existing empty loop.) */
        PRINT_ERROR("trainingBatchDefault: batch size 0 cannot be stacked into chunks of "
                    "microBatchSize %zu (no sample 0 to size the gather buffers from)",
                    m);
        exit(1);
    }

    float totalLoss;
    if (m == 1) {
        totalLoss = trainingBatchPerSample(model, modelSize, lossConfig, batch, calculateGradsFn,
                                           forwardReduction);
    } else {
        requireFloat32Model(model, modelSize, m);
        totalLoss = trainingBatchStacked(model, modelSize, lossConfig, batch, calculateGradsFn,
                                         forwardReduction, m);
    }

    if (forwardReduction == REDUCTION_MEAN) {
        return totalLoss / (float)batch->size;
    }
    return totalLoss;
}
