#ifndef ODT_MODEL_GRAPH_API_H
#define ODT_MODEL_GRAPH_API_H

#include <stddef.h>
#include <stdint.h>

#include "Layer.h"
#include "Model.h"

/* Buffer id of the model input; node j's output is buffer j + 1. */
#define GRAPH_INPUT ((uint16_t)0)

/* Every field is private to the builder functions; declare one and pass it to graphBegin. */
typedef struct graphBuilder {
    layer_t **layers;
    uint16_t *inputStart;
    uint16_t *inputIds;
    size_t nodeCapacity, edgeCapacity, numLayers, numEdges;
} graphBuilder_t;

/* Starts a wiring in caller storage, without allocating: layers[nodeCapacity],
 * inputStart[nodeCapacity + 1] and inputIds[edgeCapacity]. That storage must outlive every
 * model_t graphFinish returns, because the model points into it. Add nodes in execution order. */
void graphBegin(graphBuilder_t *g, layer_t **layers, uint16_t *inputStart, uint16_t *inputIds,
                size_t nodeCapacity, size_t edgeCapacity);
/* Appends a one-input node reading buffer `in` and returns its output buffer id. Ids are checked
 * by graphFinish, not here. */
uint16_t graphNode(graphBuilder_t *g, layer_t *layer, uint16_t in);
/* Appends an element-wise ADD node reading buffers a then b and returns its output buffer id. */
uint16_t graphMerge(graphBuilder_t *g, layer_t *add, uint16_t a, uint16_t b);
/* Returns the model over the builder's storage after validating the whole wiring once; exits on
 * an invalid wiring. */
model_t graphFinish(const graphBuilder_t *g);

#endif // ODT_MODEL_GRAPH_API_H
