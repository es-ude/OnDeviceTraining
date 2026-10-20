#define SOURCE_FILE "REMAT_CHECK"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "Common.h"
#include "Layer.h"
#include "LayerConfigAccess.h"
#include "LossFunction.h"
#include "RematCheck.h"
#include "RematCheckedSize.h"
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
                    lossFuncType_t lt, rematMode_t mode, uint32_t *producedGen) {
    size_t numWires = rematCheckNumWires(s);
    size_t deepest;
    ptrdiff_t top;
    rematBackwardRange(model, n, lt, &deepest, &top);
    /* REFORWARD's step budget, 2n + 2 + n(n+1)/2: the minimum-memory
     * schedule's n(n+1)/2 recomputes on top of a plain stream. Computed here,
     * apart from the plan grammar's bound, so a mistake in one shows in the
     * other. */
    size_t nPlusOne = 0;
    size_t product = 0;
    size_t twoN = 0;
    size_t maxSteps = 0;
    if (!checkedAddSize(n, 1u, &nPlusOne) || !checkedMulSize(n, nPlusOne, &product) ||
        !checkedMulSize(2u, n, &twoN) || !checkedAddSize(twoN, 2u + product / 2u, &maxSteps)) {
        PRINT_ERROR("rematCheckInit: the step budget 2n + 2 + n(n+1)/2 overflows size_t (n = %zu)",
                    n);
        exit(1);
    }
    *c = (rematCheck_t){.sched = s,
                        .mode = mode,
                        .model = model,
                        .n = n,
                        .deepest = deepest,
                        .backwardTop = top,
                        .hasBackward = deepest < n,
                        .nextBackward = top,
                        .maxSteps = maxSteps,
                        .producedGen = producedGen};
    for (size_t w = 0; w < numWires; w++) {
        producedGen[w] = 0u;
    }
}

static const char *kindName(uint8_t kind) {
    switch (kind) {
    case REMAT_STEP_FORWARD:
        return "FORWARD";
    case REMAT_STEP_LOSS_FORWARD:
        return "LOSS_FORWARD";
    case REMAT_STEP_LOSS_BACKWARD:
        return "LOSS_BACKWARD";
    case REMAT_STEP_BACKWARD:
        return "BACKWARD";
    case REMAT_STEP_REFORWARD:
        return "REFORWARD";
    default:
        return "UNKNOWN";
    }
}

/* Every violation exits naming the row, the step and the rule: `violation` is a format that starts
 * with the quoted rule, so a death test can match the rule whole. No rollback exists mid-call, so
 * exiting is the only honest option. */
#define REMAT_CHECK_EXIT(c, st, violation, ...)                                                    \
    do {                                                                                           \
        PRINT_ERROR("remat[%s]: step #%zu %s(layer %u) violates " violation,                       \
                    (c)->sched->fns->name, (c)->stepIndex, kindName((st)->kind),                   \
                    (unsigned)(st)->layer, ##__VA_ARGS__);                                         \
        exit(1);                                                                                   \
    } while (false)

/* Rule 1: before any resolution, so an out-of-range layer never
 * indexes the model or the table. In EVAL the kind is judged before the
 * layer. */
static void requireKindAndLayer(const rematCheck_t *c, const rematStep_t *st) {
    if (c->mode == REMAT_MODE_EVAL &&
        (st->kind == REMAT_STEP_LOSS_BACKWARD || st->kind == REMAT_STEP_BACKWARD ||
         st->kind == REMAT_STEP_REFORWARD)) {
        REMAT_CHECK_EXIT(c, st, "'step not allowed in eval'");
    }
    switch (st->kind) {
    case REMAT_STEP_FORWARD:
    case REMAT_STEP_REFORWARD:
    case REMAT_STEP_BACKWARD:
        if (st->layer >= c->n) {
            REMAT_CHECK_EXIT(c, st, "'layer out of range' (n = %zu)", c->n);
        }
        break;
    case REMAT_STEP_LOSS_FORWARD:
    case REMAT_STEP_LOSS_BACKWARD:
        if (st->layer != c->n) {
            REMAT_CHECK_EXIT(c, st, "'layer out of range' (n = %zu)", c->n);
        }
        break;
    default:
        REMAT_CHECK_EXIT(c, st, "'unknown step kind' (kind %u)", (unsigned)st->kind);
    }
}

/* Rule 2: FORWARD strictly ascending and BACKWARD strictly
 * descending, so "exactly once, in order" is "matches the cursor"; no
 * per-layer bitset. Every cursor comparison is signed. A REFORWARD moves no
 * cursor: it is admitted inside the backward window, within the step budget,
 * and never for a layer whose training forward updates its state. */
static void requireOrder(const rematCheck_t *c, const rematStep_t *st) {
    switch (st->kind) {
    case REMAT_STEP_FORWARD:
        if (c->lossForwardSeen) {
            REMAT_CHECK_EXIT(c, st, "'forward after loss'");
        }
        if (st->layer != c->nextForward) {
            REMAT_CHECK_EXIT(c, st, "'forward order: expected FORWARD(%zu)'", c->nextForward);
        }
        break;
    case REMAT_STEP_LOSS_FORWARD:
        if (c->lossForwardSeen) {
            REMAT_CHECK_EXIT(c, st, "'duplicate loss-forward'");
        }
        if (c->nextForward != c->n) {
            REMAT_CHECK_EXIT(c, st, "'loss-forward early' (FORWARD(%zu) has not run)",
                             c->nextForward);
        }
        break;
    case REMAT_STEP_LOSS_BACKWARD:
        if (!c->lossForwardSeen) {
            REMAT_CHECK_EXIT(c, st, "'loss-backward before loss-forward'");
        }
        if (!c->hasBackward) {
            REMAT_CHECK_EXIT(c, st, "'loss-backward without trainable layer'");
        }
        if (c->lossBackwardSeen) {
            REMAT_CHECK_EXIT(c, st, "'duplicate loss-backward'");
        }
        break;
    case REMAT_STEP_REFORWARD:
        /* A pending step still reads ACT l+1. hasBackward is implied by the
         * deepest clause (without it deepest == n > top >= nextBackward); it
         * stays as the rule reads. */
        if (!c->lossForwardSeen || !c->hasBackward || c->nextBackward < (ptrdiff_t)c->deepest ||
            (ptrdiff_t)st->layer >= (c->lossBackwardSeen ? c->nextBackward : (ptrdiff_t)c->n)) {
            REMAT_CHECK_EXIT(c, st, "'reforward outside window'");
        }
        if (c->stepIndex >= c->maxSteps) {
            REMAT_CHECK_EXIT(c, st, "'step budget exceeded' (maxSteps = %zu)", c->maxSteps);
        }
        /* trackRunningStats is not in the table's key, so a layer reconfigured
         * after the plan was built gets here. The config must not change during
         * a call: this check does not make a concurrent change safe. */
        if (layerForwardUpdatesState(c->model[st->layer])) {
            REMAT_CHECK_EXIT(
                c, st,
                "'reforward of a state-updating layer' (layer %u updates its state in "
                "every training forward: a tracked, unfrozen BatchNorm1d; keep ACT %u)",
                (unsigned)st->layer, (unsigned)st->layer + 1u);
        }
        break;
    case REMAT_STEP_BACKWARD:
        if (!c->lossBackwardSeen) {
            REMAT_CHECK_EXIT(c, st, "'backward before loss-backward'");
        }
        if (c->nextBackward < (ptrdiff_t)c->deepest) {
            REMAT_CHECK_EXIT(c, st,
                             "'backward order: expected no further BACKWARD' (deepest = %zu)",
                             c->deepest);
        }
        if ((ptrdiff_t)st->layer != c->nextBackward) {
            REMAT_CHECK_EXIT(c, st, "'backward order: expected BACKWARD(%ld)'",
                             (long)c->nextBackward);
        }
        break;
    default:
        break; /* unreachable: rule 1 exits on an unknown kind first */
    }
}

/* A step's operand wire ids; REMAT_NONE where it has no such operand. */
typedef struct stepWires {
    uint16_t in, gradIn, out;
    bool readsIn; /* false only for a BACKWARD whose layer does not read its input */
} stepWires_t;

/* Rule 3: positional, from the checker's own live-model deepest
 * and backwardTop; GRAD ids come from the table's gradIdOf (rematGradId),
 * never from arithmetic on ids. */
static stepWires_t resolve(const rematCheck_t *c, const rematStep_t *st) {
    const rematWireTable_t *t = c->sched->wires;
    size_t l = st->layer;
    stepWires_t w = {REMAT_NONE, REMAT_NONE, REMAT_NONE, true};
    switch (st->kind) {
    case REMAT_STEP_FORWARD:
    case REMAT_STEP_REFORWARD:
        w.in = rematActId(t, l);
        w.out = rematActId(t, l + 1u);
        break;
    case REMAT_STEP_LOSS_FORWARD:
        w.in = rematActId(t, c->n);
        break;
    case REMAT_STEP_LOSS_BACKWARD:
        w.in = rematActId(t, c->n);
        w.out = rematGradId(t, c->n);
        break;
    case REMAT_STEP_BACKWARD:
        w.in = rematActId(t, l);
        w.readsIn = layerBackwardReadsInput(c->model[l]);
        w.gradIn = rematGradId(t, (ptrdiff_t)l == c->backwardTop ? c->n : l + 1u);
        /* REMAT_NONE at deepest (the grads-only call): the table has no GRAD
         * deepest, and the bind keeps its deepest equal to the live one. */
        w.out = rematGradId(t, l);
        break;
    default:
        break; /* unreachable: rule 1 exits on an unknown kind first */
    }
    return w;
}

static tensor_t *hdrOrNull(const rematWireTable_t *t, uint16_t w) {
    return w == REMAT_NONE ? NULL : rematWireHdr(t, w);
}

static const char *wireKindName(const rematWire_t *rec) {
    return rec->kind == REMAT_WIRE_ACT ? "ACT" : "GRAD";
}

/* Rule 4: resident, and, unless borrowed, produced under the
 * binding it has now. The bind generation is the only O(1) catch for a row
 * that re-binds a wire without its producer running again. */
static void requireReadable(const rematCheck_t *c, const rematStep_t *st, uint16_t w,
                            const char *role) {
    const rematWire_t *rec = &c->sched->wires->wires[w];
    if (rematWireHdr(c->sched->wires, w)->data == NULL) {
        REMAT_CHECK_EXIT(c, st, "'operand not resident: %s %s %u'", role, wireKindName(rec),
                         (unsigned)rec->index);
    }
    if (rec->borrowed) {
        return;
    }
    if (c->producedGen[w] == 0u) {
        REMAT_CHECK_EXIT(c, st, "'operand never produced: %s %u'", wireKindName(rec),
                         (unsigned)rec->index);
    }
    if (c->producedGen[w] != rec->bindGen) {
        REMAT_CHECK_EXIT(c, st,
                         "'operand stale: %s %u rebound since produced' (produced at bindGen %u, "
                         "bound now at bindGen %u)",
                         wireKindName(rec), (unsigned)rec->index, (unsigned)c->producedGen[w],
                         (unsigned)rec->bindGen);
    }
}

static void requireWritable(const rematCheck_t *c, const rematStep_t *st, uint16_t w) {
    const rematWire_t *rec = &c->sched->wires->wires[w];
    if (rematWireHdr(c->sched->wires, w)->data == NULL) {
        REMAT_CHECK_EXIT(c, st, "'output not resident: %s %u'", wireKindName(rec),
                         (unsigned)rec->index);
    }
}

/* A BACKWARD whose layer does not read its input may get a dead one
 * (W_dead): only its header's static metadata is used. */
static void requireResident(const rematCheck_t *c, const rematStep_t *st, const stepWires_t *w) {
    if (w->readsIn) {
        requireReadable(c, st, w->in, "in");
    }
    if (w->gradIn != REMAT_NONE) {
        requireReadable(c, st, w->gradIn, "gradIn");
    }
    if (w->out != REMAT_NONE) {
        requireWritable(c, st, w->out);
    }
}

/* Rule 5: O(1) over the exact bytes. Integer intervals, because
 * HEAP operands live in separate blocks, whose pointers C does not order. */
static void requireApart(const rematCheck_t *c, const rematStep_t *st, uint16_t a,
                         const char *roleA, uint16_t b, const char *roleB) {
    const rematWireTable_t *t = c->sched->wires;
    uintptr_t da = (uintptr_t)rematWireHdr(t, a)->data;
    uintptr_t db = (uintptr_t)rematWireHdr(t, b)->data;
    if (da < db + rematWireBytes(t, b) && db < da + rematWireBytes(t, a)) {
        const rematWire_t *ra = &t->wires[a];
        const rematWire_t *rb = &t->wires[b];
        REMAT_CHECK_EXIT(c, st, "'operands share bytes: %s/%s' (%s %u and %s %u)", roleA, roleB,
                         wireKindName(ra), (unsigned)ra->index, wireKindName(rb),
                         (unsigned)rb->index);
    }
}

/* Pairwise over every operand the step touches, not only gradIn/out: a
 * LayerNorm-style backward writes dx while it still reads x. A
 * dead input (W_dead) is not touched, so it is not compared. */
static void requireDisjoint(const rematCheck_t *c, const rematStep_t *st, const stepWires_t *w) {
    uint16_t ids[3];
    const char *roles[3];
    size_t k = 0;
    if (w->readsIn) {
        ids[k] = w->in;
        roles[k++] = "in";
    }
    if (w->gradIn != REMAT_NONE) {
        ids[k] = w->gradIn;
        roles[k++] = "gradIn";
    }
    if (w->out != REMAT_NONE) {
        ids[k] = w->out;
        roles[k++] = "out";
    }
    for (size_t i = 0; i < k; i++) {
        for (size_t j = i + 1u; j < k; j++) {
            requireApart(c, st, ids[i], roles[i], ids[j], roles[j]);
        }
    }
}

/* Rule 6: an output is produced under the binding it has now. */
static void commit(rematCheck_t *c, const rematStep_t *st, const stepWires_t *w) {
    switch (st->kind) {
    case REMAT_STEP_FORWARD:
        c->nextForward++;
        break;
    case REMAT_STEP_LOSS_FORWARD:
        c->lossForwardSeen = true;
        break;
    case REMAT_STEP_LOSS_BACKWARD:
        c->lossBackwardSeen = true;
        break;
    case REMAT_STEP_BACKWARD:
        c->nextBackward--;
        break;
    case REMAT_STEP_REFORWARD:
        break; /* only its output's bind generation, below */
    default:
        break; /* unreachable: rule 1 exits on an unknown kind first */
    }
    if (w->out != REMAT_NONE) {
        c->producedGen[w->out] = c->sched->wires->wires[w->out].bindGen;
    }
    c->stepIndex++;
}

void rematCheckStep(rematCheck_t *c, const rematStep_t *st, rematOperands_t *ops) {
    requireKindAndLayer(c, st);
    requireOrder(c, st);
    const rematWireTable_t *t = c->sched->wires;
    stepWires_t w = resolve(c, st);
    *ops = (rematOperands_t){
        .in = hdrOrNull(t, w.in), .gradIn = hdrOrNull(t, w.gradIn), .out = hdrOrNull(t, w.out)};
    requireResident(c, st, &w);
    requireDisjoint(c, st, &w);
    commit(c, st, &w);
}

#define REMAT_FINISH_EXIT(c, violation, ...)                                                       \
    do {                                                                                           \
        PRINT_ERROR("remat[%s]: stream of %zu steps violates " violation, (c)->sched->fns->name,   \
                    (c)->stepIndex, ##__VA_ARGS__);                                                \
        exit(1);                                                                                   \
    } while (false)

/* Stream completeness. Cursors, not bitsets: a stream the step rules admitted is
 * complete iff every cursor reached its end. Under CE with n = 1, top = -1 is
 * already below deepest = 0: LOSS_BACKWARD with no BACKWARD is complete. */
void rematCheckFinish(const rematCheck_t *c) {
    if (c->nextForward != c->n) {
        REMAT_FINISH_EXIT(c, "'incomplete stream: missing FORWARD(%zu)'", c->nextForward);
    }
    if (!c->lossForwardSeen) {
        REMAT_FINISH_EXIT(c, "'incomplete stream: missing LOSS_FORWARD'");
    }
    if (c->mode == REMAT_MODE_EVAL) {
        return; /* an eval stream is its forward half */
    }
    if (c->hasBackward && !c->lossBackwardSeen) {
        REMAT_FINISH_EXIT(c, "'incomplete stream: missing LOSS_BACKWARD'");
    }
    if (c->hasBackward && c->nextBackward >= (ptrdiff_t)c->deepest) {
        REMAT_FINISH_EXIT(c, "'incomplete stream: missing BACKWARD(%ld)'", (long)c->nextBackward);
    }
}

/* Release lifecycle: a row's end leaves nothing resident, and the
 * dispatch's end unbinds the input. A wire still bound here would enter the
 * next call's table bind still bound. */
void rematCheckReleased(const rematCheck_t *c) {
    const rematWireTable_t *t = c->sched->wires;
    if (rematActHdr(t, 0) != NULL) {
        PRINT_ERROR("remat[%s]: after rematEnd violates 'wire left resident after end: ACT 0' "
                    "(the input is still bound)",
                    c->sched->fns->name);
        exit(1);
    }
    for (uint16_t w = 1; w < t->numWires; w++) {
        if (rematWireHdr(t, w)->data != NULL) {
            const rematWire_t *rec = &t->wires[w];
            PRINT_ERROR("remat[%s]: after rematEnd violates 'wire left resident after end: %s %u'",
                        c->sched->fns->name, wireKindName(rec), (unsigned)rec->index);
            exit(1);
        }
    }
}
