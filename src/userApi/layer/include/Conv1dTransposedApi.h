#ifndef CONV1D_TRANSPOSED_API_H
#define CONV1D_TRANSPOSED_API_H

#include <stddef.h>

#include "Kernel.h"
#include "Layer.h"
#include "LayerCommon.h"
#include "LayerQuant.h"

_Static_assert(VALID == 0,
               "paddingType_t::VALID must be enum value 0 so .padding zero-init defaults to VALID");

/*! Conv1dTransposed factory configuration. Mirrors conv1dInit_t plus PyTorch's
 *  outputPadding parameter. Build via designated initializer:
 *
 *      conv1dTransposedLayerInit(&(conv1dTransposedInit_t){
 *          .inChannels = 16, .outChannels = 8, .kernelSize = 5, .stride = 5,
 *      }, lq);
 *
 *  REQUIRED fields fire PRINT_ERROR + exit(1) if zero. Only VALID padding is
 *  supported; the factory and the model loader reject other values. */
typedef struct conv1dTransposedInit {
    /* REQUIRED */
    size_t inChannels;
    size_t outChannels;
    size_t kernelSize;
    /* OPTIONAL */
    size_t stride;           /* 0 → 1 */
    paddingType_t padding;   /* 0 → VALID; VALID is the only supported value */
    size_t dilation;         /* 0 → 1 */
    size_t groups;           /* 0 → 1 */
    size_t outputPadding;    /* PyTorch parity; default 0; must be < max(stride, dilation) */
    bias_t bias;             /* BIAS_DEFAULT (0) → resolves to true */
    weightInit_t weightInit; /* zero-init → INIT_DEFAULT (PyTorch kaiming a=√5) */
    trainable_t trainable;   /* zero-init → TRAINABLE_DEFAULT (trainable) */
} conv1dTransposedInit_t;

/*! Arithmetic slots are copied by value. Borrowing keeps the outputQ and
 *  propLossQ pointers; owning deep-copies those two configurations. Parameter
 *  quantizations are owned independently.
 *
 *  Borrowing variant: `lq` itself may be a compound literal, but the outputQ
 *  and propLossQ it points to must outlive the layer. Use when those
 *  configurations are shared across layers. */
layer_t *conv1dTransposedLayerInit(conv1dTransposedInit_t *init, layerQuant_t *lq);

/*! Owning variant: the caller may drop `lq` and every quantization it points
 *  to right after the call; freeConv1dTransposedLayer tears the copies down. */
layer_t *conv1dTransposedLayerInitOwning(conv1dTransposedInit_t *init, layerQuant_t *lq);

/*! Frees everything the factory allocated, plus outputQ/propLossQ when the
 *  layer owns them. */
void freeConv1dTransposedLayer(layer_t *layer);

#endif /* CONV1D_TRANSPOSED_API_H */
