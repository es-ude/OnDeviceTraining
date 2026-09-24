#ifndef POOL1D_API_H
#define POOL1D_API_H

#include <stddef.h>

#include "Kernel.h"
#include "Layer.h"
#include "LayerQuant.h"

_Static_assert(VALID == 0,
               "paddingType_t::VALID must be enum value 0 so .padding zero-init defaults to VALID");

/*! MaxPool1d factory configuration.
 *
 *  Requires input geometry (inputChannels, inputLength) because the
 *  factory pre-allocates an argmaxIndices INT32 tensor sized for the
 *  layer's output shape at batch 1; maxPool1dForward grows it on demand
 *  to the largest batch it sees (#152 PR3b: stacked micro-batches,
 *  batched inference()).
 *
 *  Usage:
 *
 *      maxPool1dLayerInit(&(maxPool1dInit_t){
 *          .kernelSize = 2, .stride = 2,
 *          .inputChannels = 16, .inputLength = 64,
 *      }, lq);
 */
typedef struct maxPool1dInit {
    /* REQUIRED */
    size_t kernelSize;
    size_t inputChannels;
    size_t inputLength;
    /* OPTIONAL — zero-init defaults */
    size_t stride;         /* 0 → kernelSize (PyTorch pool convention) */
    paddingType_t padding; /* 0 → VALID */
    size_t dilation;       /* 0 → 1 */
} maxPool1dInit_t;

/*! AvgPool1d factory configuration. No argmax tensor needed, hence no
 *  input geometry. Note: dilation field omitted because AvgPool1d
 *  arithmetic kernel does not support dilation. */
typedef struct avgPool1dInit {
    /* REQUIRED */
    size_t kernelSize;
    /* OPTIONAL */
    size_t stride;         /* 0 → kernelSize */
    paddingType_t padding; /* 0 → VALID */
} avgPool1dInit_t;

/*! Borrowing variant — allocates kernel and (for MaxPool) the argmax
 *  tensor; stores lq->outputQ in outputQ and lq->propLossQ in
 *  propLossQ verbatim.
 *  Use when: outputQ/propLossQ are shared/long-lived (e.g. reused across
 *  several layers) and the caller manages their lifetime — they must
 *  outlive the layer. */
layer_t *maxPool1dLayerInit(maxPool1dInit_t *init, layerQuant_t *lq);
layer_t *avgPool1dLayerInit(avgPool1dInit_t *init, layerQuant_t *lq);

/*! Owning variant — additionally deep-copies outputQ and
 *  propLossQ via deepCopyQuantization.
 *  Use when: outputQ/propLossQ are stack-locals or one-off configs and you
 *  want fire-and-forget teardown (free*Pool1dLayer tears them down too). */
layer_t *maxPool1dLayerInitOwning(maxPool1dInit_t *init, layerQuant_t *lq);
layer_t *avgPool1dLayerInitOwning(avgPool1dInit_t *init, layerQuant_t *lq);

/*! Tears down everything the factory allocated. For MaxPool, this
 *  includes the argmax tensor. Reads config->ownsQuantizations to
 *  decide whether to also free the two math quantizations and their
 *  qConfigs. */
void freeMaxPool1dLayer(layer_t *layer);
void freeAvgPool1dLayer(layer_t *layer);

#endif /* POOL1D_API_H */
