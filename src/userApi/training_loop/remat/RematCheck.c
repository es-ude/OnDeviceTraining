#define SOURCE_FILE "REMAT_CHECK"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "Common.h"
#include "Layer.h"
#include "LossFunction.h"
#include "RematCheck.h"
#include "RematPlan.h"
#include "RematScheduler.h"

size_t rematCheckNumWires(const rematScheduler_t *s) {
    if (s->fns == NULL || s->wires == NULL || s->plan == NULL) {
        PRINT_ERROR("rematCheckNumWires: scheduler not initialised (never initialised, or its "
                    "init failed before the plan was built)");
        exit(1);
    }
    return s->wires->numWires;
}

void rematCheckInit(rematCheck_t *c, rematScheduler_t *s, layer_t **model, size_t n,
                    lossFuncType_t lt, uint32_t *producedGen) {
    size_t numWires = rematCheckNumWires(s);
    size_t deepest;
    ptrdiff_t top;
    rematBackwardRange(model, n, lt, &deepest, &top);
    *c = (rematCheck_t){.sched = s,
                        .model = model,
                        .n = n,
                        .deepest = deepest,
                        .backwardTop = top,
                        .hasBackward = deepest < n,
                        .nextBackward = top,
                        .producedGen = producedGen};
    for (size_t w = 0; w < numWires; w++) {
        producedGen[w] = 0u;
    }
}
