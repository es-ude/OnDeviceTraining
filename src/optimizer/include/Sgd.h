#ifndef SGD_H
#define SGD_H

#include "Optimizer.h"

typedef struct sgd {
    float learningRate;
    float momentumFactor;
    float weightDecay;
}sgd_t;

void sgdInit(sgd_t *sgd, float learningRate, float momentumFactor, float weightDecay);

void sgdStep(optimizer_t *optimizer);

void sgdStepM(optimizer_t *optimizer);

void sgdZeroGrad(optimizer_t *optimizer);

#endif
