#ifndef LR_SCHEDULER_H
#define LR_SCHEDULER_H

#include <stddef.h>

#include "Optimizer.h"

/*! Learning-rate schedulers with PyTorch-parity semantics (#327).
 *
 * Caller-owned struct, zero allocation. `baseLr` is captured ONCE at init
 * (via the optimizer vtable's getLr); every lrSchedulerStep() computes the
 * closed form FROM baseLr and overwrites the optimizer LR absolutely —
 * never compounding on the possibly-mutated current value.
 *
 * Stepping is boundary-agnostic: trainingRun steps once per epoch (after
 * the epoch callback); hand-rolled loops may call lrSchedulerStep() at any
 * boundary (per-batch included). `lastEpoch` counts step() calls, exactly
 * like PyTorch's last_epoch. After init (lastEpoch == 0) the LR equals
 * baseLr for STEP_LR, EXPONENTIAL_LR, and COSINE_ANNEALING_LR, matching
 * PyTorch after construction. LINEAR_WARMUP_LR (#383) is the one exception:
 * mirroring torch's LinearLR, its init writes baseLr*startFactor through
 * setLr immediately, so the LR right after linearWarmupLrInit is
 * baseLr*startFactor, not baseLr — see linearWarmupLrInit's doc comment.
 *
 * Schedule scalars are computed in double (pow/cos) and cast to float only
 * when written through setLr — mirroring PyTorch's float64 scheduler math
 * feeding float32 kernels. */

typedef enum { STEP_LR, EXPONENTIAL_LR, COSINE_ANNEALING_LR, LINEAR_WARMUP_LR } lrSchedulerType_t;

typedef struct lrScheduler {
    lrSchedulerType_t type;
    optimizer_t *optimizer;
    float baseLr;
    size_t lastEpoch;
    union {
        struct {
            size_t stepSize;
            float gamma;
        } stepLr;
        struct {
            float gamma;
        } exponentialLr;
        struct {
            size_t tMax;
            float etaMin;
        } cosineAnnealingLr;
        struct {
            size_t warmupEpochs;
            float startFactor;
            /* Caller-owned, NULLABLE. If NULL, LR holds at baseLr once
             * lastEpoch >= warmupEpochs (torch LinearLR-alone parity). Must
             * already be initialized (its own ...Init call made) BEFORE
             * linearWarmupLrInit — see that function's doc comment.
             * `struct lrScheduler *`, not `lrScheduler_t *`: the typedef
             * name isn't complete yet at this point in its own definition. */
            struct lrScheduler *main;
        } linearWarmupLr;
    } params;
} lrScheduler_t;

typedef float (*computeLrFn_t)(const lrScheduler_t *sched);

typedef struct lrSchedulerFunctions {
    computeLrFn_t computeLr;
} lrSchedulerFunctions_t;

extern lrSchedulerFunctions_t lrSchedulerFunctions[];

/*! lr = baseLr * gamma^floor(lastEpoch / stepSize)  (torch StepLR) */
void stepLrInit(lrScheduler_t *sched, optimizer_t *optimizer, size_t stepSize, float gamma);

/*! lr = baseLr * gamma^lastEpoch  (torch ExponentialLR) */
void exponentialLrInit(lrScheduler_t *sched, optimizer_t *optimizer, float gamma);

/*! lr = etaMin + (baseLr - etaMin) * (1 + cos(pi * lastEpoch / tMax)) / 2
 *  (torch CosineAnnealingLR closed form; periodic past tMax) */
void cosineAnnealingLrInit(lrScheduler_t *sched, optimizer_t *optimizer, size_t tMax, float etaMin);

/*! lr = baseLr * (startFactor + (1-startFactor) * lastEpoch / warmupEpochs)
 *  while lastEpoch < warmupEpochs; once lastEpoch >= warmupEpochs, delegates
 *  to mainSched's own computeLr evaluated at (lastEpoch - warmupEpochs), or
 *  holds at baseLr when mainSched == NULL. (torch parity:
 *  SequentialLR(opt, [LinearLR(startFactor, end_factor=1.0,
 *  total_iters=warmupEpochs), mainSched], milestones=[warmupEpochs]).)
 *
 *  Validates warmupEpochs >= 1, 0 < startFactor <= 1 (finite), and — if
 *  mainSched is non-NULL — mainSched->optimizer == optimizer (mirrors the
 *  #327 trainingRun wiring guard). Writes baseLr*startFactor through setLr
 *  immediately (torch LinearLR applies its factor at construction) — this is
 *  the one scheduler type whose LR right after init is NOT baseLr.
 *
 *  ORDERING REQUIREMENT: if mainSched is non-NULL, it MUST already be
 *  initialized (its own ...Init call already made, so it captured baseLr via
 *  getLr while the optimizer's LR was still pristine) BEFORE calling
 *  linearWarmupLrInit. This mirrors torch's SequentialLR: torch's
 *  group["initial_lr"] is set once, from the pristine LR, by whichever
 *  scheduler is EVER attached to that optimizer first, and every later
 *  scheduler on the same optimizer inherits that same value regardless of
 *  the optimizer's currently-visible LR at its own construction time. The C
 *  API has no such per-optimizer memory, so the equivalent behavior only
 *  falls out if mainSched captures baseLr strictly before this call mutates
 *  the optimizer's LR to baseLr*startFactor. Initializing mainSched AFTER
 *  linearWarmupLrInit would silently capture the wrong (already-scaled)
 *  baseLr — this is not checked at runtime, only documented here. */
void linearWarmupLrInit(lrScheduler_t *sched, optimizer_t *optimizer, size_t warmupEpochs,
                        float startFactor, lrScheduler_t *mainSched);

/*! lastEpoch++ -> computeLr -> setLr through the optimizer vtable. */
void lrSchedulerStep(lrScheduler_t *sched);

#endif // LR_SCHEDULER_H
