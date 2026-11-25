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
    arithmetic_t updateMath; /* arithmetic for the update ops; FLOAT32-only, checked by sgdInit */
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
    /* This group's learningRate = base learningRate (group 0's) * lrMultiplier;
     * 1 for group 0. */
    float lrMultiplier;
} sgd_t;

void sgdInit(sgd_t *sgd, float learningRate, float momentumFactor, float weightDecay,
             arithmetic_t updateMath);

/* Fills a parameter group's block from group 0: a copy with lrMultiplier and
 * weightDecay replaced and the fields derived from them re-derived. Called by
 * sgdMAddGroup, which validates both values first. */
void sgdInitGroup(sgd_t *group, const sgd_t *base, float lrMultiplier, float weightDecay);

void sgdStepM(optimizer_t *optimizer, const rngKey_t *updateKey);

/* The base learning rate: group 0's. */
float sgdGetLr(optimizer_t *optimizer);

/* Sets group 0 to learningRate and every other group to learningRate * its
 * lrMultiplier. */
void sgdSetLr(optimizer_t *optimizer, float learningRate);

#endif
