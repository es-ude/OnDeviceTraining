#ifndef ODT_FLATTEN_API_H
#define ODT_FLATTEN_API_H

#include "Layer.h"

/*! Flatten has no wire config: its output and its dx take its input wire's
 *  config, and it copies SYM_INT32 codes verbatim. So the layer after it must
 *  not declare its dx (propLossQ) wider than Flatten's input wire; scheduler
 *  init rejects that, and the framework trusts the declared widths after. */
layer_t *flattenLayerInit(void);
void freeFlattenLayer(layer_t *flattenLayer);

#endif
