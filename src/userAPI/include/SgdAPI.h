#ifndef SGD_API_H
#define SGD_API_H

#include "Sgd.h"

optimizer_t* sgdMCreateOptim(float learningRate, float momentumFactor, float weightDecay,
                             layer_t** model, size_t sizeModel, qtype_t qType);

void freeOptimSgdM(optimizer_t* sgdM);

#endif //SGD_API_H
