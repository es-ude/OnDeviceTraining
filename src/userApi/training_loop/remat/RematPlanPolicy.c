#define SOURCE_FILE "REMAT_PLAN_POLICY"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "Layer.h"
#include "LayerConfigAccess.h"
#include "RematPlan.h"
#include "RematPlanPolicy.h"
#include "RematScheduler.h"

static bool hasBackwardSteps(const rematWireTable_t *t) {
    return t->hasBackward && t->backwardTop >= (ptrdiff_t)t->deepest;
}

size_t rematTrainStepCount(const rematWireTable_t *t) {
    size_t backward =
        hasBackwardSteps(t) ? (size_t)(t->backwardTop - (ptrdiff_t)t->deepest) + 1u : 0u;
    return t->modelSize + 1u + (t->hasBackward ? 1u + backward : 0u);
}

size_t rematBackwardStep(const rematWireTable_t *t, size_t l) {
    return t->modelSize + 2u + (size_t)(t->backwardTop - (ptrdiff_t)l);
}

void rematFillTrainSteps(const rematWireTable_t *t, rematStep_t *steps) {
    size_t n = t->modelSize;
    size_t s = 0;
    for (size_t l = 0; l < n; l++) {
        steps[s++] = (rematStep_t){.kind = REMAT_STEP_FORWARD, .layer = (uint16_t)l};
    }
    steps[s++] = (rematStep_t){.kind = REMAT_STEP_LOSS_FORWARD, .layer = (uint16_t)n};
    if (!t->hasBackward) {
        return;
    }
    steps[s++] = (rematStep_t){.kind = REMAT_STEP_LOSS_BACKWARD, .layer = (uint16_t)n};
    for (ptrdiff_t l = t->backwardTop; l >= (ptrdiff_t)t->deepest; l--) {
        steps[s++] = (rematStep_t){.kind = REMAT_STEP_BACKWARD, .layer = (uint16_t)l};
    }
}

/* LIVENESS: ACT j ends at its last reader. With the read-set rule (a layer
 * whose backward does not read its input leaves it dead) this frees pool / Flatten / Quantization /
 * Dropout inputs at their forward, everything below the #380 cut, the CE logits, and a frozen
 * GEMM's input, while a frozen norm keeps its input. */
static size_t actLastReader(const rematWireTable_t *t, layer_t **model, size_t j) {
    size_t n = t->modelSize;
    size_t last = (j < n) ? j : n; /* FORWARD(j), or LOSS_FORWARD for ACT n */
    if (j == n && t->hasBackward) {
        last = n + 1u; /* LOSS_BACKWARD reads the model output */
    }
    bool backwardRuns = j >= t->deepest && (ptrdiff_t)j <= t->backwardTop;
    if (backwardRuns && layerBackwardReadsInput(model[j])) {
        last = rematBackwardStep(t, j);
    }
    return last;
}

/* STORE_ALL reproduces today's lifetimes: every ACT lives until
 * deInitLayerOutputs (CalculateGradsSequential.c:116), the seed from
 * LOSS_BACKWARD to the first BACKWARD, and each dx wire from the BACKWARD that
 * writes it to the one that reads it (the gradCurr -> gradNext handover,
 * :105-111). LIVENESS differs only in where an ACT range ends. */
void rematFillTrainRanges(rematPlanPolicy_t policy, const rematWireTable_t *t, layer_t **model,
                          size_t numSteps, rematRange_t *ranges) {
    size_t n = t->modelSize;
    size_t lossBackward = n + 1u;
    for (size_t id = 1; id < t->numWires; id++) {
        const rematWire_t *w = &t->wires[id];
        size_t begin;
        size_t end;
        if (w->kind == REMAT_WIRE_ACT) {
            begin = w->index - 1u; /* FORWARD(j-1) */
            end = (policy == REMAT_PLAN_STORE_ALL) ? numSteps - 1u
                                                   : actLastReader(t, model, w->index);
        } else if (w->index == n) { /* the seed */
            begin = lossBackward;
            end = hasBackwardSteps(t) ? rematBackwardStep(t, (size_t)t->backwardTop) : lossBackward;
        } else {
            begin = rematBackwardStep(t, w->index);
            end = rematBackwardStep(t, w->index - 1u);
        }
        ranges[id - 1] =
            (rematRange_t){.wire = (uint16_t)id, .begin = (uint16_t)begin, .end = (uint16_t)end};
    }
}
