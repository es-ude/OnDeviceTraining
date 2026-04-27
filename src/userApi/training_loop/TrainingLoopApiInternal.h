#ifndef TRAINING_LOOP_API_INTERNAL_H
#define TRAINING_LOOP_API_INTERNAL_H

#include <stddef.h>

#include "Layer.h"
#include "Tensor.h"
#include "TrainingLoopApi.h"

static void initLayerOutputs(tensor_t **layerOutputs, layer_t **model, size_t sizeNetwork);

static void deInitLayerOutputs(tensor_t **layerOutputs, size_t modelSize);

static void initGradTensor(tensor_t *grad, tensor_t *layerOutput);

static void deInitGradTensor(tensor_t *tensor);

static trainingStats_t *initTrainingStats(tensor_t *output);

#endif // TRAINING_LOOP_API_INTERNAL_H
