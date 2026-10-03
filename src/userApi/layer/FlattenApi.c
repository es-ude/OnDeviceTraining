#define SOURCE_FILE "FLATTEN_API"

#include "FlattenApi.h"
#include "StorageApi.h"

layer_t *flattenLayerInit(void) {
    layer_t *flattenLayer = reserveMemory(sizeof(layer_t));
    flattenLayer->type = FLATTEN;
    // Load-bearing: the remat wire table's FLATTEN rule never reads config.
    flattenLayer->config = NULL;
    return flattenLayer;
}

void freeFlattenLayer(layer_t *flattenLayer) {
    freeReservedMemory(flattenLayer);
}
