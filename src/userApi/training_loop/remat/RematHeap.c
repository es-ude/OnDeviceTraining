#define SOURCE_FILE "REMAT_HEAP"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "Common.h"
#include "Layer.h"
#include "LossFunction.h"
#include "RematPlan.h"
#include "RematRows.h"
#include "RematScheduler.h"
#include "StorageApi.h"
#include "Tensor.h"

bool rematHeapInit(rematScheduler_t *s, layer_t **model, size_t n, lossConfig_t loss,
                   const tensor_t *inputLike, const rematPlanSpec_t *spec) {
#ifdef ODT_MEM_PROFILE
    size_t mark = memProfileCurrentBytes();
#endif
    *s = (rematScheduler_t){.type = REMAT_HEAP, .fns = &rematSchedulerFunctions[REMAT_HEAP]};
    if (!rematWireTableInit(&s->wires, model, n, loss, inputLike) ||
        !rematPlanBuild(&s->plan, s->wires, model, spec)) {
        return false;
    }
#ifdef ODT_MEM_PROFILE
    rematRequireReservedMatchesReport(s, memProfileCurrentBytes() - mark);
#endif
    return true;
}

void rematHeapBegin(rematScheduler_t *s) {
    s->walk = (rematWalk_t){0};
}

bool rematHeapNext(rematScheduler_t *s, rematStep_t *st) {
    const rematProgram_t *p = rematPlanProgram(s->plan, s->mode);
    if (s->walk.step == p->numSteps) {
        return false;
    }
    for (size_t r; (r = rematWalkOpening(p, &s->walk)) != REMAT_NONE;) {
        /* The id comes from the range, never from rematGradId. */
        uint16_t w = p->ranges[r].wire;
        size_t bytes = rematWireBytes(s->wires, w);
        uint8_t *b = reserveMemory(bytes); /* exact bytes: an exact right-boundary ASan redzone */
        if (b == NULL) {
            /* A resource failure mid-call is the row's exit; only init is
             * recoverable. */
            const rematWire_t *rec = &s->wires->wires[w];
            PRINT_ERROR("remat[heap]: reserveMemory(%zu) failed at step #%zu for wire %s %u", bytes,
                        s->walk.step, rec->kind == REMAT_WIRE_ACT ? "ACT" : "GRAD",
                        (unsigned)rec->index);
            exit(1);
        }
        rematWireBind(s->wires, w, b); /* bindGen++, accounting, VERIFY poison-at-bind */
    }
    *st = p->steps[s->walk.step];
    return true;
}

void rematHeapDone(rematScheduler_t *s, const rematStep_t *st) {
    (void)st; /* the dispatch checks that done() answers the step next() handed out */
    const rematProgram_t *p = rematPlanProgram(s->plan, s->mode);
    for (size_t r; (r = rematWalkClosing(p, &s->walk)) != REMAT_NONE;) {
        uint16_t w = p->ranges[r].wire;
        uint8_t *b = rematWireHdr(s->wires, w)->data;
        rematWireRelease(s->wires, w); /* VERIFY poison first, while still owned */
        freeReservedMemory(b);         /* then free: no write-after-free */
    }
    s->walk.step++;
}

void rematHeapEnd(rematScheduler_t *s) {
    rematRequireWalkComplete(s, "heap");
}

void rematHeapDeinit(rematScheduler_t *s) {
    (void)s;
}
