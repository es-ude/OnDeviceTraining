#ifndef ODT_LAYER_CONFIG_ACCESS_H
#define ODT_LAYER_CONFIG_ACCESS_H

#include <stdbool.h>

#include "ArithmeticType.h"
#include "Layer.h"

/* Uniform per-layer-type accessors (design spec 2026-07-02
 * arithmetic-type-split, D3/D4). The layerType_t switch over a layer's
 * declared wire configs and arithmetic lives here and nowhere else -- every
 * consumer that needs a layer's produced-wire storage config, declared forward
 * arithmetic or its FLOAT32-only status goes through these functions instead
 * of re-deriving its own switch. */

/* Produced forward-wire storage config (dtype + qConfig for the layer's
 * output tensor). NULL for Flatten — it has no per-layer quantization;
 * callers fall back to the upstream tensor's own quantization (passthrough),
 * exactly as the pre-existing FLATTEN handling already did. */
quantization_t *layerOutputQ(layer_t *layer);

/* Producer's declared backward config for the dx wire it emits (#221). NULL
 * for Flatten -> passthrough of the upstream dtype (callers already fall
 * back to the upstream tensor's quantization, e.g. initGradTensor). */
quantization_t *backwardWireQ(layer_t *layer);

/* Declared forward compute representation. Flatten and Quantization have no
 * consumed arithmetic (D4 — Quantization is a pure conversion node) ->
 * {ARITH_FLOAT32, HALF_AWAY}, matching arithmeticFromQuantizationOrDefault(NULL). */
arithmetic_t layerForwardMath(layer_t *layer);

/* FLOAT32-only gate for stacked training (#152 PR3b, spec §6.6): true iff
 * every arithmetic the layer declares (forwardMath, propLossMath and, for the
 * GEMM family, weightGradMath/biasGradMath), both wire storage configs
 * (outputQ, propLossQ) and the param + grad storage of every parameter are
 * FLOAT32. NULL wire configs (passthrough), a NULL bias (bias-less) and NULL
 * grads (frozen, #380) count as FLOAT32; Flatten is true, Quantization false.
 * trainingBatchDefault evaluates it over the whole model when m > 1. */
bool layerIsFloat32Only(layer_t *layer);

/* The first non-FLOAT32 field layerIsFloat32Only would reject, as a static
 * string ("forwardMath", "weights.grad", "gamma.param", ...), or NULL when the
 * layer passes -- the "offending field" of the stacked-training error. */
const char *layerNonFloat32Field(layer_t *layer);

#endif // ODT_LAYER_CONFIG_ACCESS_H
