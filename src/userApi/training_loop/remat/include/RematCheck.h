#ifndef ODT_REMAT_CHECK_H
#define ODT_REMAT_CHECK_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "Layer.h"
#include "LossFunction.h"
#include "RematScheduler.h"
#include "Tensor.h"

/* The validating interpreter's checker (#4, spec §7): the driver hands every
 * step a row's next() returns to rematCheckStep before any layer runs. Always
 * on, firmware included (D28); every violation exits naming the row, the step
 * and the rule (D29). Rows never link it, and it links no RNG (D44). */

/* The operands of one step, resolved positionally from the step and the live
 * model (spec §7.4 item 3); the driver executes the step on exactly these.
 * NULL where the step has no such operand: gradIn outside BACKWARD, out for
 * LOSS_FORWARD and for the grads-only BACKWARD at deepest. */
typedef struct rematOperands {
    tensor_t *in;
    tensor_t *gradIn;
    tensor_t *out;
} rematOperands_t;

/* One call's checker state, on the driver's frame (spec §7.3). */
typedef struct rematCheck {
    rematScheduler_t *sched; /* borrowed: the table, and the row name for messages */
    layer_t **model;
    size_t n;
    size_t deepest; /* rematBackwardRange on the LIVE model */
    ptrdiff_t backwardTop;
    bool hasBackward;
    size_t nextForward;
    ptrdiff_t nextBackward; /* -1 once BACKWARD(0) ran: compare it only as ptrdiff_t */
    bool lossForwardSeen, lossBackwardSeen;
    size_t stepIndex;
    uint32_t *producedGen; /* [rematCheckNumWires(sched)], the caller's; 0 = not produced */
} rematCheck_t;

/* Sizes the caller's producedGen VLA. Exits first if the table and plan were
 * never built (never initialised, or init failed before the plan), so the
 * VLA is never sized from a NULL table and never empty. A row whose own
 * block failed after that -- e.g. ARENA's data block -- is caught by
 * rematBegin instead. */
size_t rematCheckNumWires(const rematScheduler_t *s);
/* Before rematBegin. Zeroes producedGen[0..numWires) (a VLA has no
 * initialiser). */
void rematCheckInit(rematCheck_t *c, rematScheduler_t *s, layer_t **model, size_t n,
                    lossFuncType_t lt, uint32_t *producedGen);
/* Checks one step and resolves its operands into *ops; exits on a violation,
 * before the driver runs anything. */
void rematCheckStep(rematCheck_t *c, const rematStep_t *st, rematOperands_t *ops);
/* After the last step, before rematEnd: every FORWARD, LOSS_FORWARD, and when
 * something trains LOSS_BACKWARD and every BACKWARD down to deepest ran;
 * else exits naming the first missing step. */
void rematCheckFinish(const rematCheck_t *c);
/* After rematEnd: no non-borrowed wire is bound and ACT 0 is unbound. */
void rematCheckReleased(const rematCheck_t *c);

#endif // ODT_REMAT_CHECK_H
