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
 *  REQUIRED fields fire PRINT_ERROR + exit(1) if zero. Phase-1 contract:
 *  only VALID padding is supported (initConv1dTransposedConfigWithWeightsAndBias
 *  aborts on SAME). */
typedef struct conv1dTransposedInit {
    /* REQUIRED */
    size_t inChannels;
    size_t outChannels;
    size_t kernelSize;
    /* OPTIONAL */
    size_t stride;           /* 0 → 1 */
    paddingType_t padding;   /* 0 → VALID. SAME is rejected by the internal layer in Phase 1. */
    size_t dilation;         /* 0 → 1 */
    size_t groups;           /* 0 → 1 */
    size_t outputPadding;    /* PyTorch parity; default 0; must be < max(stride, dilation) */
    bias_t bias;             /* BIAS_DEFAULT (0) → resolves to true */
    weightInit_t weightInit; /* zero-init → INIT_DEFAULT (PyTorch kaiming a=√5) */
} conv1dTransposedInit_t;

/*! Borrowing variant — allocates kernel, weights, bias; stores the four
 *  lq math quantizations verbatim. Caller retains ownership of lq. */
layer_t *conv1dTransposedLayerInit(conv1dTransposedInit_t *init, layerQuant_t *lq);

/*! Owning variant — additionally deep-copies the four math quantizations
 *  via deepCopyQuantization. */
layer_t *conv1dTransposedLayerInitOwning(conv1dTransposedInit_t *init, layerQuant_t *lq);

/*! Tears down everything the factory allocated. Reads
 *  config->ownsQuantizations to decide whether to also free the four
 *  math quantizations and their qConfigs. */
void freeConv1dTransposedLayer(layer_t *layer);

#endif /* CONV1D_TRANSPOSED_API_H */
