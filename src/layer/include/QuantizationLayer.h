#ifndef QUANTIZATIONLAYER_H
#define QUANTIZATIONLAYER_H

#include <stdbool.h>

#include "Tensor.h"

typedef struct layer layer_t;

typedef struct quantizationConfig {
    quantization_t *forwardQ;  /* forward output target: dtype + qConfig for the layer output */
    quantization_t *backwardQ; /* Intended backward propLoss target (dtype + qConfig for dy
                                * requant). NOTE: the training loop currently derives the propLoss
                                * dtype and roundingMode from the upstream layer's forward output
                                * (initGradTensor), so backwardQ is only honored when it matches
                                * that — true for uniform- quantization configs. A per-layer
                                * backward-dtype override is tracked in #221. */
    bool ownsQuantizations;    /* true → freeQuantLayer tears down forwardQ/backwardQ (Owning
                                * factory); false → caller owns them (Borrowing). */
} quantizationConfig_t;

void quantizationForward(layer_t *layer, tensor_t *inputTensor, tensor_t *outputTensor);

void quantizationBackward(layer_t *layer, tensor_t *forwardInput, tensor_t *loss,
                          tensor_t *propLoss);

void quantizationCalcOutputShape(layer_t *layer, shape_t *inputShape, shape_t *outputShape);

#endif // QUANTIZATIONLAYER_H
