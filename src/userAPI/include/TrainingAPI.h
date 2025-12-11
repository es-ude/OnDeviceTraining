#ifndef TRAINING_H
#define TRAINING_H

#include "Tensor.h"
#include "Optimizer.h"

typedef struct trainingStats
{
    tensor_t* output;
    float loss;
} trainingStats_t;

trainingStats_t *calculateGrads(layer_t** model, size_t sizeNetwork, lossType_t lossFunctionType,
                    tensor_t* input, tensor_t* label);

trainingStats_t *trainingEpoch(layer_t **model, size_t sizeNetwork,
                                lossType_t lossFunctionType, tensor_t *input,
                                tensor_t *label, optimizer_t *optimizer);

void freeTrainingStats(trainingStats_t *trainingStats);

#endif //TRAINING_H
