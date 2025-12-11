#ifndef TRAINING_H
#define TRAINING_H

#include "Tensor.h"

typedef struct trainingStats
{
    tensor_t* output;
    tensor_t* loss;
} trainingStats_t;

trainingStats_t *calculateGrads(layer_t** model, size_t sizeNetwork, lossFunctionType_t lossFunctionType,
                    tensor_t* input, tensor_t* label);

void freeTrainingStats(trainingStats_t *trainingStats);

#endif //TRAINING_H
