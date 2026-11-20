#ifndef ODT_MODEL_H
#define ODT_MODEL_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "Layer.h"
#include "LossType.h"

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
/* Writes the shape of every buffer: shapes[0] gets a copy of input, shapes[j + 1] the output of
 * node j, each derived from the shapes of the buffers node j is wired to. Fine for any validated
 * model; the caller sizes every shape's dims and order to maxRank = max(input rank, 2). */
void modelShapePass(const model_t *m, const shape_t *input, shape_t *shapes, size_t maxRank);
/* The one producer of the needs-grad flags; anything else that needs them reads its output.
 * needsGrad[b] for b in 0..n: buffer b needs a gradient iff its producer is trainable
 * (layerParameters && !layerIsFrozen) or reads a buffer that needs one; needsGrad[0] = 0.
 * runsBackward[j] for j in 0..n-1: the same rule for node j, except that under CROSS_ENTROPY the
 * sink node never runs (the combined p - y gradient covers the Softmax). The caller sizes
 * needsGrad to n + 1 and runsBackward to n entries. */
void modelGradFlags(const model_t *m, lossFuncType_t lt, uint8_t *needsGrad, uint8_t *runsBackward);
/* Exits with an error naming caller unless m is a schedulable model: 1 <= numLayers < 0xFFFF and,
 * for an explicit CSR, well-formed offsets, exactly each op's arity of inputs per node, inputs
 * only from earlier buffers, and a consumer for every buffer but the last. Call it once at a public
 * entry; lower layers trust it and never re-check. The rules on merge inputs join with ADD. */
void modelRequireValid(const model_t *m, const char *caller);

#endif // ODT_MODEL_H
