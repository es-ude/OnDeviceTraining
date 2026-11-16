#ifndef ODT_REMAT_CHECK_H
#define ODT_REMAT_CHECK_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "Layer.h"
#include "Model.h"
#include "RematScheduler.h"
#include "Tensor.h"

/* The driver's half of the validating interpreter (#4): it hands every step a
 * row's next() returns to rematCheckStep before any layer runs, and executes
 * the step on the operands it resolves. The operand resolver (RematResolve.c)
 * is always compiled. The validator (RematCheck.c) is a build option,
 * ODT_REMAT_CHECK, ON by default, firmware included; ON, every violation
 * exits naming the row, the step and the rule. OFF, rematCheckInit and
 * rematCheckStep only resolve, rematCheckFinish and rematCheckReleased are
 * empty, and the driver passes no producedGen or pending. Rows never link it, and it
 * links no RNG. */

/* The operands of one step, looked up from the step's operand slice in the
 * plan; the driver executes the step on exactly these. NULL where the step has
 * no such operand: dy outside BACKWARD, out for LOSS_FORWARD and BACKWARD, and
 * dx[i] outside BACKWARD or where input i gets no gradient. */
typedef struct rematOperands {
    uint8_t k;                     /* node arity; 0 for LOSS_* and SUM */
    tensor_t *in[MODEL_MAX_ARITY]; /* forward inputs; LOSS_*: in[0] = ACT n; SUM: in[0] = PART */
    bool inMeta[MODEL_MAX_ARITY];  /* a BACKWARD input it does not read: its header may be dead */
    tensor_t *dy;                  /* BACKWARD only */
    tensor_t *out; /* FORWARD/REFORWARD: ACT j+1; LOSS_BACKWARD: the seed; SUM: the GRAD */
    tensor_t *dx[MODEL_MAX_ARITY];   /* BACKWARD: one per input, NULL = no dx */
    uint8_t dxMode[MODEL_MAX_ARITY]; /* opMode_t of dx[i] */
} rematOperands_t;

/* One call's state, on the driver's frame. The resolver reads sched, mode and
 * m; the rest is the validator's. */
typedef struct rematCheck {
    rematScheduler_t *sched; /* borrowed: the table, and the row name for messages */
    rematMode_t mode;        /* EVAL admits FORWARD and LOSS_FORWARD only */
    const model_t *m;        /* borrowed: the live model; a named model_t that outlives the call */
    bool hasBackward;        /* the keyed table's: bind compared it with the live model */
    size_t nextForward;
    /* The highest node with runsBackward whose BACKWARD has not run, -1 when
     * none: compare it only as ptrdiff_t. */
    ptrdiff_t nextBackward;
    bool lossForwardSeen, lossBackwardSeen;
    size_t stepIndex;
    size_t maxSteps;       /* 2n + 2 + n(n+1)/2: no REFORWARD at or past this step index */
    uint32_t *producedGen; /* [rematCheckNumWires(sched)], the caller's; 0 = not produced */
    /* [rematCheckNumWires(sched)], the caller's: the contributions each GRAD and PART still
     * awaits; a GRAD or PART is read only at 0 */
    uint16_t *pending;
    uint16_t lastBackward; /* the layer of the step just checked if it was a BACKWARD, else
                              REMAT_NONE: a SUM must follow its own node's */
} rematCheck_t;

/* Sizes the caller's producedGen and pending VLAs. Exits first on a scheduler
 * that is not initialised (zeroed or deinitialised), so the VLA is never sized
 * from a NULL table and never empty; rematResolveInit runs the same guard. */
size_t rematCheckNumWires(const rematScheduler_t *s);
/* The resolver's part of rematCheckInit: sched, mode and m; every other field
 * zero. */
void rematResolveInit(rematCheck_t *c, rematScheduler_t *s, const model_t *m, rematMode_t mode);
/* Looks the step's operands up from its operand slice in the plan into *ops.
 * Trusts the step: a slice outside the pool or not its node's is the
 * validator's to reject, before this runs. */
void rematResolveStep(const rematCheck_t *c, const rematStep_t *st, rematOperands_t *ops);

#ifdef ODT_REMAT_CHECK
/* Before rematBegin / rematBeginEval, in the same mode. Zeroes
 * producedGen[0..numWires) (a VLA has no initialiser) and, in TRAIN, sets
 * pending[0..numWires) from the keyed table: one per consumer edge whose node
 * runs backward into a buffer that needs a gradient, one for the loss seed,
 * one per PART; zero elsewhere and in EVAL. */
void rematCheckInit(rematCheck_t *c, rematScheduler_t *s, const model_t *m, rematMode_t mode,
                    uint32_t *producedGen, uint16_t *pending);
/* Checks one step and looks its operands up into *ops; exits on a violation,
 * before the driver runs anything. */
void rematCheckStep(rematCheck_t *c, const rematStep_t *st, rematOperands_t *ops);
/* After the last step, before rematEnd: every FORWARD, LOSS_FORWARD, and in
 * TRAIN when something trains LOSS_BACKWARD and the BACKWARD of every node
 * that runs backward ran, and no GRAD or PART awaits a contribution; else exits
 * naming the first missing step or wire. */
void rematCheckFinish(const rematCheck_t *c);
/* After rematEnd: no non-borrowed wire is bound and ACT 0 is unbound. */
void rematCheckReleased(const rematCheck_t *c);
#else
static inline void rematCheckInit(rematCheck_t *c, rematScheduler_t *s, const model_t *m,
                                  rematMode_t mode, uint32_t *producedGen, uint16_t *pending) {
    (void)producedGen;
    (void)pending;
    rematResolveInit(c, s, m, mode);
}

static inline void rematCheckStep(rematCheck_t *c, const rematStep_t *st, rematOperands_t *ops) {
    rematResolveStep(c, st, ops);
}

static inline void rematCheckFinish(const rematCheck_t *c) {
    (void)c;
}

static inline void rematCheckReleased(const rematCheck_t *c) {
    (void)c;
}
#endif

#endif // ODT_REMAT_CHECK_H
