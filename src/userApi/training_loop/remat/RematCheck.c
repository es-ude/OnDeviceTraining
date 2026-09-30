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
    default:
        return "UNKNOWN";
    }
}

/* Spec §7.7 (D29): `violation` is a format that starts with the quoted rule,
 * so a death test can match the rule whole. No rollback exists mid-call, so
 * exiting is the only honest option. */
#define REMAT_CHECK_EXIT(c, st, violation, ...)                                                    \
    do {                                                                                           \
        PRINT_ERROR("remat[%s]: step #%zu %s(layer %u) violates " violation,                       \
                    (c)->sched->fns->name, (c)->stepIndex, kindName((st)->kind),                   \
                    (unsigned)(st)->layer, ##__VA_ARGS__);                                         \
        exit(1);                                                                                   \
    } while (false)

/* Rule 1 (spec §7.4): before any resolution, so an out-of-range layer never
 * indexes the model or the table. */
static void requireKindAndLayer(const rematCheck_t *c, const rematStep_t *st) {
    switch (st->kind) {
    case REMAT_STEP_FORWARD:
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

/* Rule 2 (spec §7.5): FORWARD strictly ascending and BACKWARD strictly
 * descending, so "exactly once, in order" is "matches the cursor"; no
 * per-layer bitset. Every cursor comparison is signed (spec §7.3). */
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
    default: /* REMAT_STEP_BACKWARD */
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
    }
}

/* A step's operand wire ids; REMAT_NONE where it has no such operand. */
typedef struct stepWires {
    uint16_t in, gradIn, out;
    bool readsIn; /* false only for a BACKWARD whose layer does not read its input */
} stepWires_t;

/* Rule 3 (spec §7.4): positional, from the checker's own live-model deepest
 * and backwardTop; GRAD ids come from the table's gradIdOf (rematGradId),
 * never from arithmetic on ids. */
static stepWires_t resolve(const rematCheck_t *c, const rematStep_t *st) {
    const rematWireTable_t *t = c->sched->wires;
    size_t l = st->layer;
    stepWires_t w = {REMAT_NONE, REMAT_NONE, REMAT_NONE, true};
    switch (st->kind) {
    case REMAT_STEP_FORWARD:
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
    default: /* REMAT_STEP_BACKWARD; rule 1 admitted nothing else */
        w.in = rematActId(t, l);
        w.readsIn = layerBackwardReadsInput(c->model[l]);
        w.gradIn = rematGradId(t, (ptrdiff_t)l == c->backwardTop ? c->n : l + 1u);
        /* REMAT_NONE at deepest (the grads-only call): the table has no GRAD
         * deepest, and the bind keeps its deepest equal to the live one. */
        w.out = rematGradId(t, l);
        break;
    }
    return w;
}

static tensor_t *hdrOrNull(const rematWireTable_t *t, uint16_t w) {
    return w == REMAT_NONE ? NULL : rematWireHdr(t, w);
}

static const char *wireKindName(const rematWire_t *rec) {
    return rec->kind == REMAT_WIRE_ACT ? "ACT" : "GRAD";
}

/* Rule 4 (spec §7.3): resident, and, unless borrowed, produced under the
 * binding it has now. The bind generation is the only O(1) catch for a row
 * that re-binds a wire without its producer running again (spec §7.6). */
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

/* A BACKWARD whose layer does not read its input may get a dead one (W_dead,
 * spec §3.7): only its header's static metadata is used. */
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

/* Rule 6 (spec §7.4): an output is produced under the binding it has now. */
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
    default:
        c->nextBackward--;
        break;
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
    commit(c, st, &w);
}
