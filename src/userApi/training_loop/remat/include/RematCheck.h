#ifndef ODT_REMAT_CHECK_H
#define ODT_REMAT_CHECK_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "Layer.h"
#include "LossFunction.h"
#include "Model.h"
#include "RematScheduler.h"
#include "Tensor.h"

/* The validating interpreter's checker (#4): the driver hands every
 * step a row's next() returns to rematCheckStep before any layer runs. Always
 * on, firmware included; every violation exits naming the row, the step
 * and the rule. Rows never link it, and it links no RNG. */

/* The operands of one step, looked up from the step's operand slice in the
 * plan; the driver executes the step on exactly these. NULL where the step has
 * no such operand: dy outside BACKWARD, out for LOSS_FORWARD and BACKWARD, and
 * dx[i] outside BACKWARD or where input i gets no gradient. */
typedef struct rematOperands {
    uint8_t k;                       /* node arity; 0 for LOSS_* */
    tensor_t *in[MODEL_MAX_ARITY];   /* forward inputs; LOSS_*: in[0] = ACT n */
    bool inMeta[MODEL_MAX_ARITY];    /* a BACKWARD input it does not read: its header may be dead */
    tensor_t *dy;                    /* BACKWARD only */
    tensor_t *out;                   /* FORWARD/REFORWARD: ACT j+1; LOSS_BACKWARD: the seed */
    tensor_t *dx[MODEL_MAX_ARITY];   /* BACKWARD: one per input, NULL = no dx */
    uint8_t dxMode[MODEL_MAX_ARITY]; /* opMode_t of dx[i] */
} rematOperands_t;

/* One call's checker state, on the driver's frame. */
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
} rematCheck_t;

/* Sizes the caller's producedGen VLA. Exits first if the table and plan were
 * never built (never initialised, or init failed before the plan), so the
 * VLA is never sized from a NULL table and never empty. A row whose own
 * block failed after that -- e.g. ARENA's data block -- is caught by
 * rematBegin instead. */
size_t rematCheckNumWires(const rematScheduler_t *s);
/* Before rematBegin / rematBeginEval, in the same mode. Zeroes
 * producedGen[0..numWires) (a VLA has no initialiser). */
void rematCheckInit(rematCheck_t *c, rematScheduler_t *s, const model_t *m, lossFuncType_t lt,
                    rematMode_t mode, uint32_t *producedGen);
/* Checks one step and looks its operands up into *ops; exits on a violation,
 * before the driver runs anything. */
void rematCheckStep(rematCheck_t *c, const rematStep_t *st, rematOperands_t *ops);
/* After the last step, before rematEnd: every FORWARD, LOSS_FORWARD, and in
 * TRAIN when something trains LOSS_BACKWARD and the BACKWARD of every node
 * that runs backward ran; else exits naming the first missing step. */
void rematCheckFinish(const rematCheck_t *c);
/* After rematEnd: no non-borrowed wire is bound and ACT 0 is unbound. */
void rematCheckReleased(const rematCheck_t *c);

#endif // ODT_REMAT_CHECK_H
