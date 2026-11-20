#ifndef ODT_ADD_API_H
#define ODT_ADD_API_H

#include "Layer.h"
#include "LayerQuant.h"

/*! The elementwise sum merge node of a DAG model; FLOAT32 only.
 *  lq == NULL: the layer creates and owns FLOAT32 outputQ / propLossQ.
 *  lq != NULL: it borrows lq->outputQ / lq->propLossQ, which must outlive the layer; reads
 *  forwardMath and propLossMath too. Exits unless all four are set and FLOAT32. */
layer_t *addLayerInit(layerQuant_t *lq);

/*! Frees the quantizations only when addLayerInit created them. */
void freeAddLayer(layer_t *addLayer);

#endif
