#define SOURCE_FILE "REMAT_RESOLVE"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "Common.h"
#include "Model.h"
#include "RematCheck.h"
#include "RematPlan.h"
#include "RematScheduler.h"

size_t rematCheckNumWires(const rematScheduler_t *s) {
    if (s->fns == NULL || s->wires == NULL || s->plan == NULL) {
        PRINT_ERROR("rematCheckNumWires: scheduler not initialised (never initialised, or "
                    "deinitialised)");
        exit(1);
    }
    return s->wires->numWires;
}

void rematResolveInit(rematCheck_t *c, rematScheduler_t *s, const model_t *m, rematMode_t mode) {
    (void)rematCheckNumWires(s);
    *c = (rematCheck_t){.sched = s, .mode = mode, .m = m};
}

static tensor_t *hdrOrNull(const rematWireTable_t *t, uint16_t w) {
    return w == REMAT_NONE ? NULL : rematWireHdr(t, w);
}

/* The driver's view of the slice: a BACKWARD reads dy first, then its inputs,
 * and writes one dx slot per input; every other kind reads its inputs and
 * writes out, if anything. */
static void fillOperands(const rematCheck_t *c, const rematStep_t *st, const rematOperand_t *slice,
                         rematOperands_t *ops) {
    const rematWireTable_t *t = c->sched->wires;
    bool backward = st->kind == REMAT_STEP_BACKWARD;
    *ops = (rematOperands_t){0};
    if (st->kind == REMAT_STEP_FORWARD || st->kind == REMAT_STEP_REFORWARD || backward) {
        ops->k = (uint8_t)(backward ? st->numWrites : st->numReads);
    }
    size_t first = 0;
    if (backward) {
        ops->dy = rematWireHdr(t, slice[0].wire);
        first = 1u;
    }
    for (size_t i = first; i < st->numReads; i++) {
        ops->in[i - first] = rematWireHdr(t, slice[i].wire);
        ops->inMeta[i - first] = slice[i].mode == OP_MODE_META;
    }
    const rematOperand_t *writes = slice + st->numReads;
    for (size_t i = 0; i < st->numWrites; i++) {
        if (backward) {
            ops->dx[i] = hdrOrNull(t, writes[i].wire);
            ops->dxMode[i] = writes[i].mode;
        } else {
            ops->out = rematWireHdr(t, writes[i].wire);
        }
    }
}

void rematResolveStep(const rematCheck_t *c, const rematStep_t *st, rematOperands_t *ops) {
    const rematProgram_t *p = rematPlanProgram(c->sched->plan, c->mode);
    fillOperands(c, st, &p->operands[st->operands], ops);
}
