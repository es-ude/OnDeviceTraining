#define SOURCE_FILE "MODEL_GRAPH_API"

#include <stdlib.h>

#include "Common.h"
#include "ModelGraphApi.h"

void graphBegin(graphBuilder_t *g, layer_t **layers, uint16_t *inputStart, uint16_t *inputIds,
                size_t nodeCapacity, size_t edgeCapacity) {
    if (edgeCapacity > UINT16_MAX) {
        PRINT_ERROR("graphBegin: edge capacity %zu exceeds the uint16_t CSR offsets", edgeCapacity);
        exit(1);
    }
    *g = (graphBuilder_t){.layers = layers,
                          .inputStart = inputStart,
                          .inputIds = inputIds,
                          .nodeCapacity = nodeCapacity,
                          .edgeCapacity = edgeCapacity};
    inputStart[0] = 0;
}

static uint16_t appendNode(graphBuilder_t *g, const char *caller, layer_t *layer,
                           const uint16_t *ids, size_t count) {
    if (g->numLayers >= g->nodeCapacity) {
        PRINT_ERROR("%s: node capacity %zu exceeded", caller, g->nodeCapacity);
        exit(1);
    }
    if (g->edgeCapacity - g->numEdges < count) {
        PRINT_ERROR("%s: edge capacity %zu exceeded", caller, g->edgeCapacity);
        exit(1);
    }
    for (size_t k = 0; k < count; k++) {
        g->inputIds[g->numEdges++] = ids[k];
    }
    g->layers[g->numLayers++] = layer;
    g->inputStart[g->numLayers] = (uint16_t)g->numEdges;
    return (uint16_t)g->numLayers;
}

uint16_t graphNode(graphBuilder_t *g, layer_t *layer, uint16_t in) {
    return appendNode(g, "graphNode", layer, &in, 1);
}

uint16_t graphMerge(graphBuilder_t *g, layer_t *add, uint16_t a, uint16_t b) {
    const uint16_t ids[2] = {a, b};
    return appendNode(g, "graphMerge", add, ids, 2);
}

model_t graphFinish(const graphBuilder_t *g) {
    model_t m = {g->layers, g->numLayers, g->inputStart, g->inputIds};
    modelRequireValid(&m, "graphFinish");
    return m;
}
