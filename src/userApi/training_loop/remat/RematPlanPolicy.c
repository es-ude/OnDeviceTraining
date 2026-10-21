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

/* FORWARD 0..n-1, LOSS_FORWARD: the whole EVAL program, and the TRAIN
 * program's first n + 1 steps. */
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

/* Wire w is an in-place member: the dx GRAD l of a ReLU, Dropout or Flatten
 * (only deepest < l <= top have one; the seed GRAD n never is), and it and
 * wire w - 1 have one dtype, FLOAT32 or SYM_INT32, and equal bytes. Wire
 * numbering puts the gradient BACKWARD(l) reads at w - 1; the grammar checks
 * that on the stored marks. Equal bytes is the test that matters to the rows
 * and the checker; qMaxBits is not compared, since each wire keeps its own
 * qConfig. Reads only the table, so the plan can count before it reserves. */
static bool inPlaceMember(const rematWireTable_t *t, bool inPlaceDx, size_t w) {
    const rematWire_t *out = &t->wires[w];
    if (!inPlaceDx || out->kind != REMAT_WIRE_GRAD || out->index == t->modelSize) {
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
        bool member = inPlaceMember(t, true, w);
        if (inPlace != NULL) {
            inPlace[w] = member ? 1u : 0u;
        }
        members += member ? 1u : 0u;
    }
    return members;
}

/* STORE_ALL reproduces the pre-remat driver's lifetimes: every ACT lives to
 * the end of the call, the seed from LOSS_BACKWARD to the first BACKWARD, and
 * each dx wire from the BACKWARD that writes it to the one that reads it.
 * LIVENESS, the plan of a NULL scheduler (remat D15), differs only in where
 * an ACT range ends. */
void rematFillTrainRanges(rematPlanPolicy_t policy, const rematWireTable_t *t, layer_t **model,
                          size_t numSteps, bool inPlaceDx, rematRange_t *ranges) {
    size_t n = t->modelSize;
    size_t lossBackward = n + 1u;
    size_t k = 0;
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
        if (inPlaceMember(t, inPlaceDx, id)) {
            /* Members follow their root in id order, and each one's reader
             * runs after the one before it: the chain's range is the last one
             * written, and it ends here. */
            ranges[k - 1u].end = (uint16_t)end;
            continue;
        }
        ranges[k++] =
            (rematRange_t){.wire = (uint16_t)id, .begin = (uint16_t)begin, .end = (uint16_t)end};
    }
}

/* Does BACKWARD(l) belong to this table's backward set top..deepest? */
static bool backwardRuns(const rematWireTable_t *t, size_t l) {
    return t->hasBackward && l >= t->deepest && (ptrdiff_t)l <= t->backwardTop;
}

/* The wire a step writes as the scan sees it: REMAT_NONE for LOSS_FORWARD,
 * the grads-only BACKWARD at deepest, an in-place member (its root's range
 * holds it), and every step the grammar rejects later, so the scan never
 * names a wire the table does not have. */
static uint16_t scanWrite(const rematWireTable_t *t, bool inPlaceDx, const rematStep_t *st) {
    size_t n = t->modelSize;
    size_t l = st->layer;
    switch (st->kind) {
    case REMAT_STEP_FORWARD:
    case REMAT_STEP_REFORWARD:
        return (l < n) ? rematActId(t, l + 1u) : REMAT_NONE;
    case REMAT_STEP_LOSS_FORWARD:
        return REMAT_NONE;
    case REMAT_STEP_LOSS_BACKWARD:
        return t->hasBackward ? rematGradId(t, n) : REMAT_NONE;
    case REMAT_STEP_BACKWARD: {
        if (!backwardRuns(t, l) || l <= t->deepest) {
            return REMAT_NONE;
        }
        uint16_t w = rematGradId(t, l);
        return inPlaceMember(t, inPlaceDx, w) ? REMAT_NONE : w;
    }
    default:
        return REMAT_NONE; /* an unknown kind: the grammar exits on it after the scan */
    }
}

static uint16_t scanRootOf(const rematWireTable_t *t, bool inPlaceDx, uint16_t w) {
    while (inPlaceMember(t, inPlaceDx, w)) {
        w--;
    }
    return w;
}

/* The wires a step reads as the scan sees it (ACT 0 included: it has no
 * range, so the scan extends nothing for it), an in-place member as its
 * chain's root; none for a rejected step. */
static size_t scanReads(const rematWireTable_t *t, layer_t **model, bool inPlaceDx,
                        const rematStep_t *st, uint16_t reads[2]) {
    size_t n = t->modelSize;
    size_t l = st->layer;
    switch (st->kind) {
    case REMAT_STEP_FORWARD:
    case REMAT_STEP_REFORWARD:
        if (l >= n) {
            return 0u;
        }
        reads[0] = rematActId(t, l);
        return 1u;
    case REMAT_STEP_LOSS_FORWARD:
    case REMAT_STEP_LOSS_BACKWARD:
        reads[0] = rematActId(t, n);
        return 1u;
    case REMAT_STEP_BACKWARD: {
        if (!backwardRuns(t, l)) {
            return 0u;
        }
        size_t k = 0;
        uint16_t gradIn =
            ((ptrdiff_t)l == t->backwardTop) ? rematGradId(t, n) : rematGradId(t, l + 1u);
        reads[k++] = scanRootOf(t, inPlaceDx, gradIn);
        if (layerBackwardReadsInput(model[l])) {
            reads[k++] = rematActId(t, l);
        }
        return k;
    }
    default:
        return 0u; /* an unknown kind: the grammar exits on it after the scan */
    }
}

size_t rematScanRangeCount(const rematWireTable_t *t, const rematStep_t *steps, size_t numSteps,
                           bool inPlaceDx) {
    size_t count = 0;
    for (size_t s = 0; s < numSteps; s++) {
        if (scanWrite(t, inPlaceDx, &steps[s]) != REMAT_NONE) {
            count++;
        }
    }
    return count;
}

/* One pass in step order. A read extends the newest range of its wire,
 * found by searching back from the last range opened: O(S * R), no scratch.
 * A read before any write of its wire extends nothing (grammar rule 4 rejects
 * it). Each step writes at most one wire, so begins strictly increase. */
void rematScanRanges(const rematWireTable_t *t, layer_t **model, const rematStep_t *steps,
                     size_t numSteps, bool inPlaceDx, rematRange_t *ranges) {
    size_t opened = 0;
    for (size_t s = 0; s < numSteps; s++) {
        uint16_t reads[2];
        size_t numReads = scanReads(t, model, inPlaceDx, &steps[s], reads);
        for (size_t i = 0; i < numReads; i++) {
            for (size_t r = opened; r > 0; r--) {
                if (ranges[r - 1u].wire == reads[i]) {
                    ranges[r - 1u].end = (uint16_t)s;
                    break;
                }
            }
        }
        uint16_t write = scanWrite(t, inPlaceDx, &steps[s]);
        if (write != REMAT_NONE) {
            ranges[opened++] =
                (rematRange_t){.wire = write, .begin = (uint16_t)s, .end = (uint16_t)s};
        }
    }
}

/* The generator's output cursor; with steps == NULL it only counts. */
typedef struct retainEmitter {
    rematStep_t *steps;
    size_t numSteps;
    size_t numWrites;
    bool inPlaceDx;
} retainEmitter_t;

static void emitStep(const rematWireTable_t *t, retainEmitter_t *e, rematStepKind_t kind,
                     size_t layer) {
    rematStep_t st = {.kind = (uint8_t)kind, .layer = (uint16_t)layer};
    if (e->steps != NULL) {
        e->steps[e->numSteps] = st;
    }
    e->numSteps++;
    if (scanWrite(t, e->inPlaceDx, &st) != REMAT_NONE) {
        e->numWrites++;
    }
}

/* Makes ACT j present for the backward-phase step about to read it. ACTs
 * from *recomputedFrom up to the last need are present from earlier runs
 * (n + 1: none yet); a later need is never above an earlier one. */
static void recomputeIfDropped(const rematWireTable_t *t, const bool *retain, retainEmitter_t *e,
                               size_t j, size_t *recomputedFrom) {
    if (j == 0u || retain[j] || j >= *recomputedFrom) {
        return;
    }
    size_t k = j - 1u;
    while (k > 0u && !retain[k]) {
        k--;
    }
    for (size_t l = k; l < j; l++) {
        emitStep(t, e, REMAT_STEP_REFORWARD, l);
    }
    *recomputedFrom = k + 1u;
}

size_t rematFillRetainSteps(const rematWireTable_t *t, layer_t **model, const bool *retain,
                            bool inPlaceDx, rematStep_t *steps, size_t *numWrites) {
    size_t n = t->modelSize;
    retainEmitter_t e = {.steps = steps, .numSteps = 0, .numWrites = 0, .inPlaceDx = inPlaceDx};
    for (size_t l = 0; l < n; l++) {
        emitStep(t, &e, REMAT_STEP_FORWARD, l);
    }
    emitStep(t, &e, REMAT_STEP_LOSS_FORWARD, n);
    if (t->hasBackward) {
        size_t recomputedFrom = n + 1u;
        recomputeIfDropped(t, retain, &e, n, &recomputedFrom);
        emitStep(t, &e, REMAT_STEP_LOSS_BACKWARD, n);
        for (ptrdiff_t l = t->backwardTop; l >= (ptrdiff_t)t->deepest; l--) {
            if (layerBackwardReadsInput(model[l])) {
                recomputeIfDropped(t, retain, &e, (size_t)l, &recomputedFrom);
            }
            emitStep(t, &e, REMAT_STEP_BACKWARD, (size_t)l);
        }
    }
    if (numWrites != NULL) {
        *numWrites = e.numWrites;
    }
    return e.numSteps;
}

/* Ping-pong lifetimes: ACT j from the FORWARD that writes it (step j - 1) to
 * the step that reads it (step j: FORWARD(j), or LOSS_FORWARD for ACT n). */
void rematFillEvalProgram(const rematWireTable_t *t, rematStep_t *steps, rematRange_t *ranges) {
    size_t n = t->modelSize;
    (void)fillForwardSteps(n, steps);
    for (size_t j = 1; j <= n; j++) {
        ranges[j - 1] = (rematRange_t){
            .wire = rematActId(t, j), .begin = (uint16_t)(j - 1u), .end = (uint16_t)j};
    }
}
