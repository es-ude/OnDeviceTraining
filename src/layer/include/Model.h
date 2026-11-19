#ifndef ODT_MODEL_H
#define ODT_MODEL_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "Layer.h"

/* Widest op today is ADD; bump deliberately with the first wider op. */
#define MODEL_MAX_ARITY 2u

typedef struct model {
    layer_t **layers;
    size_t numLayers;
    const uint16_t *inputStart; /* numLayers + 1 entries; NULL = chain */
    const uint16_t *inputIds;   /* CSR: node j reads inputIds[inputStart[j] .. inputStart[j+1]) */
} model_t;

/* Block-lifetime compound literal: `&MODEL_CHAIN(arr, n)` is a valid argument for a call; a callee
 * that keeps the pointer past the full expression needs a named `model_t`. */
#define MODEL_CHAIN(layers, n) ((model_t){(layers), (n), NULL, NULL})

/* Writes node j's input buffer ids into ids[0..k) and returns k (chain: k = 1, ids[0] = j). Reads
 * the wiring as given; the model must already have passed validation at the public entry. */
uint8_t modelNodeInputs(const model_t *m, size_t j, uint16_t ids[MODEL_MAX_ARITY]);
/* The one canonical chain test: true for inputStart == NULL and for an explicit CSR that wires
 * every node j to buffer j. Use it wherever a chain and a DAG must be told apart. */
bool modelIsChain(const model_t *m);

#endif // ODT_MODEL_H
