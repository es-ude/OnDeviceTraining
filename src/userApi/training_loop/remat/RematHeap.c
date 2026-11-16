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

void rematHeapInit(rematScheduler_t *s, const model_t *m, lossConfig_t loss,
                   const tensor_t *inputLike, const rematPlanSpec_t *spec) {
    *s = (rematScheduler_t){.type = REMAT_HEAP, .fns = &rematSchedulerFunctions[REMAT_HEAP]};
    rematWireTableInit(&s->wires, m, loss, inputLike);
    rematPlanBuild(&s->plan, s->wires, m, spec);
}

void rematHeapInitEval(rematScheduler_t *s, const model_t *m, const tensor_t *inputLike) {
    *s = (rematScheduler_t){.type = REMAT_HEAP, .fns = &rematSchedulerFunctions[REMAT_HEAP]};
    rematWireTableInitEval(&s->wires, m, inputLike);
    rematPlanBuildEval(&s->plan, s->wires, m);
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
        rematWireBind(s->wires, w, b);
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
        rematWireRelease(s->wires, w); /* ODT_REMAT_VERIFY poisons here, while still owned */
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
