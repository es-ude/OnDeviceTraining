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
     * (0, 1], which that store rejects (momentumFactor == 0 never reaches it:
     * the stateless path has no state op). Creation-fixed like momentumFactor. */
    int32_t momentumQ;
    int16_t momentumE;
    bool momentumIntValid;
    /* learningRate as the integer scalar learningRateQ * 2^learningRateE for the
     * integer parameter update on BFP parameters, re-encoded by sgdInit AND
     * sgdSetLr (config time, outside the step); learningRateIntValid is false
     * for a non-finite learning rate, which that update rejects. Change the
     * learning rate only through sgdSetLr. */
    int32_t learningRateQ;
    int16_t learningRateE;
    bool learningRateIntValid;
    /* momentumFactor == 0 and weightDecay != 0, decided once by sgdInit (both
     * are creation-fixed): the step branches on these, not on float compares
     * (soft-float calls on a core without an FPU). */
    bool momentumIsZero;
    bool weightDecayIsNonzero;
} sgd_t;

void sgdInit(sgd_t *sgd, float learningRate, float momentumFactor, float weightDecay,
             arithmetic_t updateMath);

void sgdStepM(optimizer_t *optimizer, const rngKey_t *updateKey);

float sgdGetLr(optimizer_t *optimizer);

void sgdSetLr(optimizer_t *optimizer, float learningRate);

#endif
