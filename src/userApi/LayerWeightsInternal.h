#ifndef LAYER_WEIGHTS_INTERNAL_H
#define LAYER_WEIGHTS_INTERNAL_H

#include "Layer.h"

/* Shared by the two public weight loaders (layerLoadWeights,
 * modelLoadStateDict): each validates its request at its own entry, then
 * loads through layerLoadTargetsFill, which trusts it. */

typedef struct {
    parameter_t *weights; /* weights, or gamma for the norm layers */
    parameter_t *bias;    /* bias or beta; NULL for a bias-free layer */
} layerLoadTargets_t;

/*! Both NULL for a layer without loadable parameters (incl. BatchNorm1d
 *  without affine). */
layerLoadTargets_t layerLoadTargets(layer_t *layer);

/*! NULL when the buffers fit the targets, else why not: weightData is
 *  required, biasData exactly when the layer has a bias/beta. */
const char *layerLoadTargetsProblem(layerLoadTargets_t targets, const float *weightData,
                                    const float *biasData);

/*! Caller guarantees targets.weights != NULL and a NULL
 *  layerLoadTargetsProblem for these buffers. */
void layerLoadTargetsFill(layerLoadTargets_t targets, const float *weightData,
                          const float *biasData);

#endif /* LAYER_WEIGHTS_INTERNAL_H */
