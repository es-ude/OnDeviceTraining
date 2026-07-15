#ifndef ADAM_W_H
#define ADAM_W_H

#include "ArithmeticType.h"
#include "Optimizer.h"

/*! Decoupled-weight-decay Adam (torch.optim.AdamW parity, #328).
 *
 * Numerics contract (spec 2026-07-10, empirical): beta1/beta2/eps/weightDecay
 * are stored DOUBLE, and every per-step scalar (1-beta1, 1-beta2, the bias
 * corrections, -(lr/bc1), 1-lr*wd) is composed in double, then cast to float
 * exactly once at the kernel-call boundary -- mirroring PyTorch, which
 * composes these in python doubles and casts once inside the ATen kernel.
 * Storing beta2=0.999 as C float loses 1-beta2 unrecoverably (~111 ulps;
 * measured: v bit-mismatch 4096/4096 after 10 steps with float betas,
 * 0/4096 with double). learningRate stays float: it is what the
 * getLr/setLr vtable row and the LR scheduler (#327) exchange.
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
 * re-amplifies the first steps on a converged model (#350). Documented
 * here, tracked there. */
typedef struct adamW {
    float learningRate;  /* float: getLr/setLr vtable + scheduler contract */
    double beta1, beta2; /* DOUBLE, deliberately (numerics note above) */
    double eps;
    double weightDecay;      /* decoupled (torch.optim.AdamW) */
    size_t stepCount;        /* t; starts 0, ++ at each adamWStep entry */
    arithmetic_t updateMath; /* FLOAT32-only, fail-fast (#310 pattern) */
} adamW_t;

/*! Validates beta1/beta2 in [0,1), eps > 0, updateMath == ARITH_FLOAT32;
 *  fails fast (PRINT_ERROR + exit(1)) otherwise. */
void adamWInit(adamW_t *adamW, float learningRate, double beta1, double beta2, double eps,
               double weightDecay, arithmetic_t updateMath);

float adamWGetLr(optimizer_t *optimizer);

void adamWSetLr(optimizer_t *optimizer, float learningRate);

/*! One AdamW step over every registered parameter. t = ++stepCount.
 *  Op-for-op torch.optim.AdamW single-tensor sequence via three executeOp
 *  kernels (m: lerp / v: mul+addcmul / param: mul+addcdivDenom), 5
 *  elementwise passes per parameter, no denom tensor materialized. */
void adamWStep(optimizer_t *optimizer);

#endif // ADAM_W_H
