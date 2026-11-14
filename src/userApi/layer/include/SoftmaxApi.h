#ifndef SOFTMAXAPI_H
#define SOFTMAXAPI_H

#include "BfpSoftmaxExp.h"
#include "Layer.h"
#include "LayerQuant.h"
#include "Tensor.h"

/*! Borrowing variant: the layer keeps lq->outputQ / lq->propLossQ, which must outlive it.
 *  Use when the quantizations are shared or long-lived. Exits on an incoherent profile
 *  (softmaxValidateConfig). */
layer_t *softmaxLayerInit(layerQuant_t *lq);

/*! Owning variant: deep-copies both quantizations, so the caller may drop them at once;
 *  freeSoftmaxLayer frees the copies. */
layer_t *softmaxLayerInitOwning(layerQuant_t *lq);

/*! Controls integer exp shifts independently of staging/packing rounding; defaults to TRUNC.
 *  SR requires a keyed call. Exits on an unknown mode. */
void softmaxSetBfpExpShiftRounding(layer_t *softmaxLayer, bfpShiftRounding_t mode);

/*! Tears down the layer. Reads config->ownsQuantizations to decide
 *  whether to also free the two quantization_t and their qConfigs. */
void freeSoftmaxLayer(layer_t *softmaxLayer);

#endif // SOFTMAXAPI_H
