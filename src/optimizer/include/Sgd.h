#ifndef SGD_H
#define SGD_H

#include <stdbool.h>
#include <stdint.h>

#include "ArithmeticType.h"
#include "Optimizer.h"

typedef struct sgd {
    float learningRate;
    float momentumFactor;
    float weightDecay;
    arithmetic_t updateMath; /* #310: arithmetic for the update ops */
    /* momentumFactor as the integer scalar momentumQ * 2^momentumE (bfpScalar_t,
     * BfpKernelSupport.h), encoded once by sgdInit for the integer momentum store
     * on BFP state; momentumIntValid is false when momentumFactor lies outside
     * (0, 1], which that store rejects. Creation-fixed like momentumFactor. */
    int32_t momentumQ;
    int16_t momentumE;
    bool momentumIntValid;
} sgd_t;

void sgdInit(sgd_t *sgd, float learningRate, float momentumFactor, float weightDecay,
             arithmetic_t updateMath);

void sgdStepM(optimizer_t *optimizer, const rngKey_t *updateKey);

float sgdGetLr(optimizer_t *optimizer);

void sgdSetLr(optimizer_t *optimizer, float learningRate);

#endif
