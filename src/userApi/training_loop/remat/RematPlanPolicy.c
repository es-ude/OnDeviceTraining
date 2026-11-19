#define SOURCE_FILE "REMAT_PLAN_POLICY"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "Layer.h"
#include "RematPlan.h"
#include "RematPlanPolicy.h"
#include "RematScheduler.h"

/* FORWARD 0..n-1, LOSS_FORWARD: the TRAIN program's first n + 1 steps. */
static size_t fillForwardSteps(size_t n, rematStep_t *steps) {
    for (size_t l = 0; l < n; l++) {
        steps[l] = (rematStep_t){.kind = REMAT_STEP_FORWARD, .layer = (uint16_t)l};
    }
    steps[n] = (rematStep_t){.kind = REMAT_STEP_LOSS_FORWARD, .layer = (uint16_t)n};
    return n + 1u;
}

void rematFillTrainSteps(const rematWireTable_t *t, rematStep_t *steps) {
    size_t n = t->modelSize;
    size_t s = fillForwardSteps(n, steps);
    if (!t->hasBackward) {
        return;
    }
    steps[s++] = (rematStep_t){.kind = REMAT_STEP_LOSS_BACKWARD, .layer = (uint16_t)n};
    for (size_t l = n; l-- > 0;) {
        if (t->runsBackward[l]) {
            steps[s++] = (rematStep_t){.kind = REMAT_STEP_BACKWARD, .layer = (uint16_t)l};
        }
    }
}

/* The seed GRAD n never is a member. Wire numbering puts the gradient BACKWARD(l) reads at
 * w - 1; the grammar checks that on the stored marks. Equal bytes is the test that matters to the
 * rows and the checker; qMaxBits is not compared, since each wire keeps its own qConfig. Reads
 * only the table, so the plan can count before it reserves. */
bool rematInPlaceMember(const rematWireTable_t *t, size_t w) {
    const rematWire_t *out = &t->wires[w];
    if (out->kind != REMAT_WIRE_GRAD || out->index == t->modelSize) {
        return false;
    }
    uint8_t type = t->layerType[out->index];
    if (type != RELU && type != DROPOUT && type != FLATTEN) {
        return false;
    }
    const rematWire_t *in = &t->wires[w - 1u];
    return out->dtype == in->dtype && (out->dtype == FLOAT32 || out->dtype == SYM_INT32) &&
           out->bytes == in->bytes;
}

size_t rematFillInPlaceMarks(const rematWireTable_t *t, uint8_t *inPlace) {
    size_t members = 0;
    for (size_t w = 0; w < t->numWires; w++) {
        bool member = rematInPlaceMember(t, w);
        if (inPlace != NULL) {
            inPlace[w] = member ? 1u : 0u;
        }
        members += member ? 1u : 0u;
    }
    return members;
}

uint16_t rematChainRoot(const uint8_t *inPlace, uint16_t wire) {
    while (inPlace != NULL && inPlace[wire] != 0u) {
        wire--;
    }
    return wire;
}

/* The generator's output cursor; with steps == NULL it only counts. */
typedef struct retainEmitter {
    rematStep_t *steps;
    size_t numSteps;
} retainEmitter_t;

static void emitStep(retainEmitter_t *e, rematStepKind_t kind, size_t layer) {
    if (e->steps != NULL) {
        e->steps[e->numSteps] = (rematStep_t){.kind = (uint8_t)kind, .layer = (uint16_t)layer};
    }
    e->numSteps++;
}

/* Makes ACT j present for the backward-phase step about to read it. ACTs
 * from *recomputedFrom up to the last need are present from earlier runs
 * (n + 1: none yet); a later need is never above an earlier one. */
static void recomputeIfDropped(const bool *retain, retainEmitter_t *e, size_t j,
                               size_t *recomputedFrom) {
    if (j == 0u || retain[j] || j >= *recomputedFrom) {
        return;
    }
    size_t k = j - 1u;
    while (k > 0u && !retain[k]) {
        k--;
    }
    for (size_t l = k; l < j; l++) {
        emitStep(e, REMAT_STEP_REFORWARD, l);
    }
    *recomputedFrom = k + 1u;
}

size_t rematFillRetainSteps(const rematWireTable_t *t, const bool *retain, rematStep_t *steps) {
    size_t n = t->modelSize;
    retainEmitter_t e = {.steps = steps, .numSteps = 0};
    for (size_t l = 0; l < n; l++) {
        emitStep(&e, REMAT_STEP_FORWARD, l);
    }
    emitStep(&e, REMAT_STEP_LOSS_FORWARD, n);
    if (t->hasBackward) {
        size_t recomputedFrom = n + 1u;
        recomputeIfDropped(retain, &e, n, &recomputedFrom);
        emitStep(&e, REMAT_STEP_LOSS_BACKWARD, n);
        for (size_t l = n; l-- > 0;) {
            if (!t->runsBackward[l]) {
                continue;
            }
            if (t->readsInput[l]) {
                recomputeIfDropped(retain, &e, l, &recomputedFrom);
            }
            emitStep(&e, REMAT_STEP_BACKWARD, l);
        }
    }
    return e.numSteps;
}
