#ifndef CONV1DAPI_H
#define CONV1DAPI_H

#include "Kernel.h"
#include "Layer.h"

#include "LayerCommon.h"
#include "LayerQuant.h"

_Static_assert(VALID == 0,
               "paddingType_t::VALID must be enum value 0 so .padding zero-init defaults to VALID");

/*! Conv1d factory configuration. Build via designated initializer:
 *
 *      conv1dLayerInit(&(conv1dInit_t){
 *          .inChannels = 3, .outChannels = 16, .kernelSize = 5,
 *          .padding = SAME, .stride = 1,
 *      }, lq);
 *
 *  REQUIRED fields fire PRINT_ERROR + exit(1) if zero. Defaults below are
 *  applied when the field is zero-init (compound-literal omission). */
typedef struct conv1dInit {
    /* REQUIRED */
    size_t inChannels;
    size_t outChannels;
    size_t kernelSize;
    /* OPTIONAL — zero-init defaults */
    size_t stride;           /* 0 → 1 */
    paddingType_t padding;   /* 0 → VALID (enum value 0); SAME or EXPLICIT also valid */
    size_t paddingAmount;    /* symmetric pad per side; used ONLY when padding == EXPLICIT */
    size_t dilation;         /* 0 → 1 */
    size_t groups;           /* 0 → 1 */
    bias_t bias;             /* BIAS_DEFAULT (0) → resolves to true (PyTorch parity) */
    weightInit_t weightInit; /* zero-init → INIT_DEFAULT (PyTorch kaiming a=√5) */
    trainable_t trainable;   /* zero-init → TRAINABLE_DEFAULT (trainable) */
} conv1dInit_t;

/*! Arithmetic slots are copied by value. Borrowing keeps the outputQ and
 *  propLossQ pointers; owning deep-copies those two configurations. Parameter
 *  quantizations are owned independently.
 *
 *  Borrowing variant: `lq` itself may be a compound literal, but the outputQ
 *  and propLossQ it points to must outlive the layer. Use when those
 *  configurations are shared across layers. */
layer_t *conv1dLayerInit(conv1dInit_t *init, layerQuant_t *lq);

/*! Owning variant: the caller may drop `lq` and every quantization it points
 *  to right after the call; freeConv1dLayer tears the copies down. */
layer_t *conv1dLayerInitOwning(conv1dInit_t *init, layerQuant_t *lq);

/*! Frees everything the factory allocated, plus outputQ/propLossQ when the
 *  layer owns them. */
void freeConv1dLayer(layer_t *conv1dLayer);

#endif // CONV1DAPI_H
