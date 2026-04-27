#ifndef INFERENCE_H
#define INFERENCE_H

#include "DataLoader.h"
#include "Layer.h"
#include "LossFunction.h"
#include "Tensor.h"

typedef struct inferenceStats {
    tensor_t *output;
    float loss;
} inferenceStats_t;

void freeInferenceStats(inferenceStats_t *inferenceStats);

tensor_t *inference(layer_t **model, size_t numberOfLayers, tensor_t *input);

tensor_t **inferenceBatched(layer_t **model, size_t numberOfLayers, batch_t *batch);

inferenceStats_t *inferenceWithLoss(layer_t **model, size_t numberOfLayers, tensor_t *input,
                                    tensor_t *label, lossType_t lossType);

#endif // INFERENCE_H
