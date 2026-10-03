#ifndef ODT_MODEL_VALIDATION_API_H
#define ODT_MODEL_VALIDATION_API_H

#include <stdbool.h>
#include <stddef.h>

#include "Layer.h"

/*! Opt-in model-quantization validator (NOT wired into the training loop).
 *
 *  Walks the layer array and validates its shape (non-NULL model, non-NULL
 *  elements). The int16-inter-layer "SYM producer must be followed by a
 *  QUANTIZATION layer" rule this once enforced is RETIRED (PR1b.2 D3):
 *  the forward funnel now restores width at the producer's own wire, so a
 *  following Quant layer is optional, not required. This entry point is kept
 *  as the place future model-wide rules attach to.
 *
 *  Diagnostics: logs every violation (PRINT_ERROR-style, visible with
 *  DLEVEL >= 1) and returns false if at least one violation was found.
 *  NEVER exits — callers decide how to react. */
bool validateModelQuantization(layer_t **model, size_t modelSize);

#endif // ODT_MODEL_VALIDATION_API_H
