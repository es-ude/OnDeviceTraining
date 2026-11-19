#ifndef ODT_MODEL_H
#define ODT_MODEL_H

#include <stddef.h>
#include <stdint.h>

#include "Layer.h"

typedef struct model {
    layer_t **layers;
    size_t numLayers;
    const uint16_t *inputStart; /* numLayers + 1 entries; NULL = chain */
    const uint16_t *inputIds;   /* CSR: node j reads inputIds[inputStart[j] .. inputStart[j+1]) */
} model_t;

/* Block-lifetime compound literal: `&MODEL_CHAIN(arr, n)` is a valid argument for a call; a callee
 * that keeps the pointer past the full expression needs a named `model_t`. */
#define MODEL_CHAIN(layers, n) ((model_t){(layers), (n), NULL, NULL})

#endif // ODT_MODEL_H
