#define SOURCE_FILE "STATE_DICT_API"

#include "StateDictApi.h"
#include "BatchNorm1d.h"
#include "Common.h"
#include "LayerWeightsInternal.h"
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

static bool layerHasParameters(layer_t *layer) {
    return layerLoadTargets(layer).weights != NULL;
}

void modelLoadStateDict(const model_t *m, stateDictEntry_t *entries, size_t numEntries) {
    size_t numParamLayers = 0;
    for (size_t i = 0; i < m->numLayers; i++) {
        if (layerHasParameters(m->layers[i])) {
            numParamLayers++;
        }
    }
    if (numParamLayers != numEntries) {
        PRINT_ERROR("modelLoadStateDict: model has %zu param layers but %zu entries provided",
                    numParamLayers, numEntries);
        exit(1);
    }

    /* Every entry's buffers are checked before the first write. */
    size_t entryIdx = 0;
    for (size_t i = 0; i < m->numLayers; i++) {
        layerLoadTargets_t targets = layerLoadTargets(m->layers[i]);
        if (targets.weights == NULL) {
            continue;
        }
        const stateDictEntry_t *e = &entries[entryIdx];
        const char *problem = layerLoadTargetsProblem(targets, e->weightData, e->biasData);
        if (problem != NULL) {
            if (e->name != NULL) {
                PRINT_ERROR("modelLoadStateDict: entry '%s' (#%zu): %s", e->name, entryIdx,
                            problem);
            } else {
                PRINT_ERROR("modelLoadStateDict: entry #%zu: %s", entryIdx, problem);
            }
            exit(1);
        }
        entryIdx++;
    }

    entryIdx = 0;
    for (size_t i = 0; i < m->numLayers; i++) {
        layerLoadTargets_t targets = layerLoadTargets(m->layers[i]);
        if (targets.weights == NULL) {
            continue;
        }
        layerLoadTargetsFill(targets, entries[entryIdx].weightData, entries[entryIdx].biasData);
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

void modelLoadStateDictBuffers(const model_t *m, const stateDictBuffers_t *entries,
                               size_t numEntries) {
    size_t numBufferLayers = 0;
    for (size_t i = 0; i < m->numLayers; i++) {
        if (layerHasRunningBuffers(m->layers[i])) {
            numBufferLayers++;
        }
    }
    if (numBufferLayers != numEntries) {
        PRINT_ERROR("modelLoadStateDictBuffers: model has %zu running-buffer layers but %zu "
                    "entries provided",
                    numBufferLayers, numEntries);
        exit(1);
    }

    size_t entryIdx = 0;
    for (size_t i = 0; i < m->numLayers; i++) {
        if (!layerHasRunningBuffers(m->layers[i])) {
            continue;
        }
        const stateDictBuffers_t *e = &entries[entryIdx];
        batchNorm1dConfig_t *cfg = m->layers[i]->config->batchNorm1d;
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
