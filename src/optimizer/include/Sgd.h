#ifndef SGD_H
#define SGD_H

#include "ArithmeticType.h"
#include "Optimizer.h"

typedef struct sgd {
    float learningRate;
    float momentumFactor;
    float weightDecay;
    arithmetic_t updateMath; /* #310: arithmetic for the update ops */
} sgd_t;

void sgdInit(sgd_t *sgd, float learningRate, float momentumFactor, float weightDecay,
             arithmetic_t updateMath);

void sgdStepM(optimizer_t *optimizer, const rngKey_t *updateKey);

float sgdGetLr(optimizer_t *optimizer);

void sgdSetLr(optimizer_t *optimizer, float learningRate);

#endif
