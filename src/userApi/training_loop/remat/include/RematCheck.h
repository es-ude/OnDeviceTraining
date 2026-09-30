#ifndef ODT_REMAT_CHECK_H
#define ODT_REMAT_CHECK_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "Layer.h"
#include "LossFunction.h"
#include "RematScheduler.h"

/* The validating interpreter's checker (#4, spec §7): the driver hands every
 * step a row's next() returns to rematCheckStep before any layer runs. Always
 * on, firmware included (D28); every violation exits naming the row, the step
 * and the rule (D29). Rows never link it, and it links no RNG (D44). */

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

#endif // ODT_REMAT_CHECK_H
