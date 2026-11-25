#ifndef ADAM_W_H
#define ADAM_W_H

#include "ArithmeticType.h"
#include "Optimizer.h"

/*! Decoupled-weight-decay Adam (torch.optim.AdamW parity).
 *
 * Numerics contract: beta1/beta2/eps/weightDecay are stored DOUBLE, and every
 * per-step scalar (1-beta1, 1-beta2, the bias corrections, -(lr/bc1), 1-lr*wd)
 * is composed in double, then cast to float exactly once at the kernel-call
 * boundary -- mirroring PyTorch, which composes these in python doubles and
 * casts once inside the ATen kernel. Float betas would lose 1-beta2
 * unrecoverably. learningRate stays float: it is what the getLr/setLr vtable
 * row and the LR scheduler exchange.
 *
 * weightDecay is DECOUPLED (param *= 1 - lr*wd, torch.optim.AdamW) --
 * deliberately different from sgd_t.weightDecay's coupled L2
 * (grad += wd*param).
 *
 * lerp-parity caveat: the m update maps onto lerpFloat32TensorsInplace,
 * whose fmaf form is bit-identical to ATen only for |1-beta1| < 0.5, i.e.
 * beta1 > 0.5. Validation still admits [0, 1); beta1 <= 0.5 forgoes lerp
 * bit-parity (documented, not branched).
 *
 * stepCount is NOT persisted: StateDictApi serializes only weights/biases,
 * so a checkpoint-resumed run restarts bias correction at t=1 and
 * re-amplifies the first steps on a converged model. */
typedef struct adamW {
    float learningRate;  /* float: getLr/setLr vtable + scheduler contract */
    double beta1, beta2; /* DOUBLE, deliberately (numerics note above) */
    double eps;
    double weightDecay;      /* decoupled (torch.optim.AdamW) */
    size_t stepCount;        /* t; starts 0, ++ at each adamWStep entry */
    arithmetic_t updateMath; /* FLOAT32-only, checked by adamWInit */
    /* This group's learningRate = base learningRate (group 0's) * lrMultiplier;
     * 1 for group 0. Betas, eps and stepCount are read from group 0 only. */
    float lrMultiplier;
} adamW_t;

/*! Validates beta1/beta2 in [0,1), eps > 0, updateMath == ARITH_FLOAT32;
 *  fails fast (PRINT_ERROR + exit(1)) otherwise. */
void adamWInit(adamW_t *adamW, float learningRate, double beta1, double beta2, double eps,
               double weightDecay, arithmetic_t updateMath);

/* Fills a parameter group's block from group 0: a copy with lrMultiplier and
 * weightDecay replaced and learningRate = base->learningRate * lrMultiplier.
 * Called by adamWAddGroup, which validates both values first. */
void adamWInitGroup(adamW_t *group, const adamW_t *base, float lrMultiplier, double weightDecay);

/* The base learning rate: group 0's. */
float adamWGetLr(optimizer_t *optimizer);

/* Sets group 0 to learningRate and every other group to learningRate * its
 * lrMultiplier. */
void adamWSetLr(optimizer_t *optimizer, float learningRate);

/*! One AdamW step over every registered parameter. t = ++stepCount.
 *  Op-for-op torch.optim.AdamW single-tensor sequence via three executeOp
 *  kernels (m: lerp / v: mul+addcmul / param: mul+addcdivDenom), 5
 *  elementwise passes per parameter, no denom tensor materialized. */
void adamWStep(optimizer_t *optimizer, const rngKey_t *updateKey);

#endif // ADAM_W_H
