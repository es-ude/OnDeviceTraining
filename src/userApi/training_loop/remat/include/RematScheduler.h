#ifndef ODT_REMAT_SCHEDULER_H
#define ODT_REMAT_SCHEDULER_H

#include <stddef.h>
#include <stdint.h>

/* The remat scheduler's public header (#4, spec §5.1). In PR1a it holds only
 * the step, policy, spec and walk types the static plan needs; the scheduler
 * interface (vtable, rows, rematBegin/Next/Done/End, the report) comes with
 * the RematScheduler library. */

/* APPEND-ONLY: stored in plan tables and (PR6) caller/ir2c sequences. */
typedef enum rematStepKind {
    REMAT_STEP_FORWARD = 0,   /* layer l: ACT l -> ACT l+1 */
    REMAT_STEP_LOSS_FORWARD,  /* layer == n: output snapshot + loss of ACT n; forward->backward
                                 boundary */
    REMAT_STEP_LOSS_BACKWARD, /* layer == n: ACT n -> GRAD n (the seed) */
    REMAT_STEP_BACKWARD,      /* layer l: (ACT l, GRAD in(l)) -> GRAD l, or grads-only at deepest */
} rematStepKind_t;

/* One record, three roles: plan-table entry, the value a row's next() fills
 * in, and (PR6) the offline sequence record. No tensor pointers: the driver
 * resolves operands positionally. No explicit pad byte until a byte-image
 * consumer (#62 ir2c .rodata) needs one (spec §16.1 item 4d); uint16_t
 * alignment keeps the implicit pad at one byte on every target. */
typedef struct rematStep {
    uint8_t kind;   /* rematStepKind_t */
    uint16_t layer; /* model index; == modelSize for LOSS_*; the table's wire-count guard
                       implies modelSize < UINT16_MAX */
} rematStep_t;
_Static_assert(sizeof(rematStep_t) == 4, "fixed-width step record");

typedef enum rematPlanPolicy { REMAT_PLAN_STORE_ALL = 0, REMAT_PLAN_LIVENESS } rematPlanPolicy_t;

/* NULL, or a zero-initialised struct, means STORE_ALL (the trainingRunOptions_t
 * idiom, TrainingLoopApi.h:113-115). */
typedef struct rematPlanSpec {
    rematPlanPolicy_t policy;
} rematPlanSpec_t;

/* The static-plan cursor: step index, next range to open (begin order), next
 * range to close (endOrder). Defined here because the scheduler struct embeds
 * it. */
typedef struct rematWalk {
    size_t step, open, close;
} rematWalk_t;

#endif // ODT_REMAT_SCHEDULER_H
