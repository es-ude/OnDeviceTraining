#define SOURCE_FILE "STATE_DICT_API"

#include "StateDictApi.h"
#include "BatchNorm1d.h"
#include "Common.h"
#include "LayerWeightsApi.h"
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

static bool layerHasParameters(layer_t *layer) {
    switch (layer->type) {
    case LINEAR:
    case CONV1D:
    case CONV1D_TRANSPOSED:
    case LAYERNORM:
    case GROUPNORM:
        return true;
    case BATCHNORM1D:
        return layer->config->batchNorm1d->affine;
    case RELU:
    case SOFTMAX:
    case FLATTEN:
    case MAXPOOL1D:
    case AVGPOOL1D:
    case ADAPTIVE_AVGPOOL1D:
    case DROPOUT:
    case QUANTIZATION:
        return false;
    default:
        PRINT_ERROR("layerHasParameters: unknown layer type %d", (int)layer->type);
        exit(1);
    }
}

void modelLoadStateDict(layer_t **model, size_t numLayers, stateDictEntry_t *entries,
                        size_t numEntries) {
    /* First pass: count param layers and verify entry count matches. */
    size_t numParamLayers = 0;
    for (size_t i = 0; i < numLayers; i++) {
        if (layerHasParameters(model[i])) {
            numParamLayers++;
        }
    }
    if (numParamLayers != numEntries) {
        PRINT_ERROR("modelLoadStateDict: model has %zu param layers but %zu entries provided",
                    numParamLayers, numEntries);
        exit(1);
    }

    /* Second pass: load each param layer in order. */
    size_t entryIdx = 0;
    for (size_t i = 0; i < numLayers; i++) {
        if (!layerHasParameters(model[i])) {
            continue;
        }
        stateDictEntry_t *e = &entries[entryIdx];
        if (e->weightData == NULL) {
            if (e->name != NULL) {
                PRINT_ERROR("modelLoadStateDict: entry '%s' (#%zu): weightData is NULL", e->name,
                            entryIdx);
            } else {
                PRINT_ERROR("modelLoadStateDict: entry #%zu: weightData is NULL", entryIdx);
            }
            exit(1);
        }
        layerLoadWeights(model[i], e->weightData, e->biasData);
        entryIdx++;
    }
}

static bool layerHasRunningBuffers(layer_t *layer) {
    return layer->type == BATCHNORM1D && layer->config->batchNorm1d->trackRunningStats;
}

static void failBufferEntry(const stateDictBuffers_t *e, size_t idx, const char *why) {
    if (e->name != NULL) {
        PRINT_ERROR("modelLoadStateDictBuffers: entry '%s' (#%zu): %s", e->name, idx, why);
    } else {
        PRINT_ERROR("modelLoadStateDictBuffers: entry #%zu: %s", idx, why);
    }
    exit(1);
}

void modelLoadStateDictBuffers(layer_t **model, size_t numLayers, const stateDictBuffers_t *entries,
                               size_t numEntries) {
    /* First pass: count buffer-bearing layers and verify entry count matches. */
    size_t numBufferLayers = 0;
    for (size_t i = 0; i < numLayers; i++) {
        if (layerHasRunningBuffers(model[i])) {
            numBufferLayers++;
        }
    }
    if (numBufferLayers != numEntries) {
        PRINT_ERROR("modelLoadStateDictBuffers: model has %zu running-buffer layers but %zu "
                    "entries provided",
                    numBufferLayers, numEntries);
        exit(1);
    }

    /* Second pass: load each buffer-bearing layer in order. */
    size_t entryIdx = 0;
    for (size_t i = 0; i < numLayers; i++) {
        if (!layerHasRunningBuffers(model[i])) {
            continue;
        }
        const stateDictBuffers_t *e = &entries[entryIdx];
        batchNorm1dConfig_t *cfg = model[i]->config->batchNorm1d;
        if (e->runningMean == NULL || e->runningVar == NULL) {
            failBufferEntry(e, entryIdx, "runningMean and runningVar are required");
        }
        for (size_t c = 0; c < cfg->numChannels; c++) {
            if (!isfinite(e->runningMean[c])) {
                failBufferEntry(e, entryIdx, "runningMean holds a non-finite value");
            }
            if (!isfinite(e->runningVar[c]) || e->runningVar[c] < 0.0f) {
                failBufferEntry(e, entryIdx, "runningVar must be finite and >= 0");
            }
        }
        memcpy(cfg->runningMean->data, e->runningMean, cfg->numChannels * sizeof(float));
        memcpy(cfg->runningVar->data, e->runningVar, cfg->numChannels * sizeof(float));
        cfg->numBatchesTracked = e->numBatchesTracked;
        entryIdx++;
    }
}
