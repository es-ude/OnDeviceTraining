#ifndef ENV5_RUNTIME_SOFTMAX_H
#define ENV5_RUNTIME_SOFTMAX_H

#include <stdbool.h>

#include "ArithmeticType.h"
#include "BfpSoftmaxExp.h"
#include "Layer.h"

typedef struct softmaxConfig {
    arithmetic_t forwardMath;
    arithmetic_t propLossMath;
    quantization_t *outputQ;
    quantization_t *propLossQ;
    bool ownsQuantizations;
    /* Controls integer exp shifts independently of staging/packing rounding; defaults to TRUNC.
     * SR requires a keyed call. Inert unless a math slot is ARITH_BFP. */
    bfpShiftRounding_t bfpExpShiftRounding;
} softmaxConfig_t;

void softmaxInitConfig(softmaxConfig_t *softmaxConfig, quantization_t *forwardQ,
                       quantization_t *backwardQ);

void softmaxInitLayer(layerConfig_t *softmaxConfig, layer_t *softmaxLayer);

/*! The config contract forward/backward rely on: supported arithmetic in both slots and a
 *  BFP-typed produced-wire config for each ARITH_BFP slot. Call once where a config enters --
 *  the factories and the deserializer. `who` prefixes the message. */
void softmaxValidateConfig(const softmaxConfig_t *cfg, const char *who);

/* The integer-only ARITH_BFP forward (arithmetic-bfp.md §5.9 R-S2, §13):
 * steps 1-4 on the unpacked-BFP `input` (int32 codes under a live
 * bfpQConfig_t, mantissaBits <= 16), step 5 an integer normalisation into
 * the BFP_RAW `rawOut` (same element count; every raw group gets exponent
 * -BFP_SOFTMAX_RAW_FRAC_BITS). `mode` is the layer's bfpExpShiftRounding knob,
 * `shift` its SR stream (NULL unless mode is BFP_SHIFT_SR). The funnel's
 * forward kernel adapter calls it; exported for kernel-level tests. */
void softmaxForwardBfpRaw(const tensor_t *input, bfpShiftRounding_t mode, rngStream_t *shift,
                          tensor_t *rawOut);

/* Rank 1 is one row; otherwise storage axis 0 selects rows and trailing dimensions form each
 * row. Multiple rows require identity order. Native BFP supports at most one row. */
void softmaxForward(layer_t *softmaxLayer, tensor_t *input, tensor_t *output,
                    const layerExec_t *exec);

/* Same row partition as softmaxForward; each row's s is recomputed from its own logits. A NULL
 * propLoss is a no-op under ARITH_BFP propLossMath. */
void softmaxBackward(layer_t *softmaxLayer, tensor_t *input, tensor_t *loss, tensor_t *propLoss,
                     const layerExec_t *exec);

void softmaxCalcOutputShape(layer_t *softmaxLayer, shape_t *inputShape, shape_t *outputShape);

#endif // ENV5_RUNTIME_SOFTMAX_H
