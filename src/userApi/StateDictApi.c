#define SOURCE_FILE "STATE_DICT_API"

#include "StateDictApi.h"
#include "Common.h"
#include "LayerWeightsApi.h"
#include <stdbool.h>
#include <stdlib.h>

static bool layerHasParameters(layer_t *layer) {
    switch (layer->type) {
    case LINEAR:
    case CONV1D:
    case CONV1D_TRANSPOSED:
    case LAYERNORM:
    case GROUPNORM:
        return true;
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
