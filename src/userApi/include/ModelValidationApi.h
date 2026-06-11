#ifndef ODT_MODEL_VALIDATION_API_H
#define ODT_MODEL_VALIDATION_API_H

#include <stdbool.h>
#include <stddef.h>

#include "Layer.h"

/*! Opt-in model-quantization validator (NOT wired into the training loop).
 *
 *  Walks the layer array and checks the int16 inter-layer contract: a layer
 *  whose forwardQ is SYM_INT32 and whose type is an accumulator-range
 *  producer (LINEAR, LAYERNORM; Conv joins with #45) must be followed by a
 *  QUANTIZATION layer that requantizes the raw accumulator mantissas. A
 *  producer in the last position is allowed (loss boundary).
 *
 *  Diagnostics: logs every violation (PRINT_ERROR-style, visible with
 *  DLEVEL >= 1) and returns false if at least one violation was found.
 *  NEVER exits — callers decide how to react. */
bool validateModelQuantization(layer_t **model, size_t modelSize);

#endif // ODT_MODEL_VALIDATION_API_H
