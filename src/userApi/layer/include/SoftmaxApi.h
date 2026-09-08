#ifndef SOFTMAXAPI_H
#define SOFTMAXAPI_H

#include "BfpSoftmaxExp.h"
#include "Layer.h"
#include "LayerQuant.h"
#include "Tensor.h"

/*! Borrowing variant — stores lq->outputQ in outputQ and
 *  lq->propLossQ in propLossQ verbatim.
 *  Use when: outputQ/propLossQ are shared/long-lived (e.g. reused across
 *  several layers) and the caller manages their lifetime — they must
 *  outlive the layer. */
layer_t *softmaxLayerInit(layerQuant_t *lq);

/*! Owning variant — deep-copies outputQ + propLossQ via
 *  deepCopyQuantization.
 *  Use when: outputQ/propLossQ are stack-locals or one-off configs and you
 *  want fire-and-forget teardown (freeSoftmaxLayer tears them down too). */
layer_t *softmaxLayerInitOwning(layerQuant_t *lq);

/*! P6-2: sets the shift-rounding knob for the native BFP softmax kernels
 *  (softmaxConfig_t.bfpExpShiftRounding -- see its comment in Softmax.h).
 *  ORTHOGONAL to every roundingMode_t; factories default BFP_SHIFT_TRUNC.
 *  Use when: comparing rounding regimes of the integer exp (research knob);
 *  inert unless the layer's math slots declare ARITH_BFP. */
void softmaxSetBfpExpShiftRounding(layer_t *softmaxLayer, bfpShiftRounding_t mode);

/*! Tears down the layer. Reads config->ownsQuantizations to decide
 *  whether to also free the two quantization_t and their qConfigs. */
void freeSoftmaxLayer(layer_t *softmaxLayer);

#endif // SOFTMAXAPI_H
