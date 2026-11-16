#define SOURCE_FILE "REMAT_CHECK"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "CheckedSize.h"
#include "Common.h"
#include "Layer.h"
#include "LayerConfigAccess.h"
#include "Model.h"
#include "OpTable.h"
#include "RematCheck.h"
#include "RematPlan.h"
#include "RematPlanPolicy.h"
#include "RematScheduler.h"

size_t rematCheckNumWires(const rematScheduler_t *s) {
    if (s->fns == NULL || s->wires == NULL || s->plan == NULL) {
        PRINT_ERROR("rematCheckNumWires: scheduler not initialised (never initialised, or "
                    "deinitialised)");
        exit(1);
    }
    return s->wires->numWires;
}

/* The highest node below `below` whose BACKWARD runs, -1 when none. Amortised
 * O(1) per BACKWARD over a call: each node is passed at most once. */
static ptrdiff_t highestRunningBelow(const rematWireTable_t *t, ptrdiff_t below) {
    for (ptrdiff_t j = below - 1; j >= 0; j--) {
        if (t->runsBackward[j] != 0u) {
            return j;
        }
    }
    return -1;
}

/* The contributions each wire expects, counted from the keyed table's flags and
 * wiring copy, not from its dx decisions, so the checker does not inherit the
 * deriver's errors: a GRAD expects one per consumer edge whose node runs
 * backward, plus the loss seed's; a PART expects its one BACKWARD. The CE
 * sink's own edge does not count: the Softmax does not run backward. */
static void countContributions(const rematWireTable_t *t, uint16_t *pending) {
    for (size_t w = 0; w < t->numWires; w++) {
        pending[w] = (t->wires[w].kind == REMAT_WIRE_PART) ? 1u : 0u;
    }
    if (t->hasBackward) {
        pending[t->gradIdOf[t->modelSize]]++;
    }
    for (size_t j = 0; j < t->modelSize; j++) {
        if (t->runsBackward[j] == 0u) {
            continue;
        }
        uint16_t ids[MODEL_MAX_ARITY];
        uint8_t k = rematNodeInputs(t, j, ids);
        for (uint8_t i = 0; i < k; i++) {
            if (t->needsGrad[ids[i]] != 0u) {
                pending[t->gradIdOf[ids[i]]]++;
            }
        }
    }
}

void rematCheckInit(rematCheck_t *c, rematScheduler_t *s, const model_t *m, rematMode_t mode,
                    uint32_t *producedGen, uint16_t *pending) {
    size_t numWires = rematCheckNumWires(s);
    size_t n = m->numLayers;
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
    /* The keyed table's flags, not the live model's: the bind compares the
     * two before the first step, and a mismatch exits there by name. */
    const rematWireTable_t *t = s->wires;
    *c = (rematCheck_t){.sched = s,
                        .mode = mode,
                        .m = m,
                        .hasBackward = t->hasBackward,
                        .nextBackward = highestRunningBelow(t, (ptrdiff_t)t->modelSize),
                        .maxSteps = maxSteps,
                        .producedGen = producedGen,
                        .pending = pending,
                        .lastBackward = REMAT_NONE};
    for (size_t w = 0; w < numWires; w++) {
        producedGen[w] = 0u;
        pending[w] = 0u;
    }
    if (mode == REMAT_MODE_TRAIN) {
        countContributions(t, pending);
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
    case REMAT_STEP_SUM:
        return "SUM";
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
         st->kind == REMAT_STEP_REFORWARD || st->kind == REMAT_STEP_SUM)) {
        REMAT_CHECK_EXIT(c, st, "'step not allowed in eval'");
    }
    switch (st->kind) {
    case REMAT_STEP_FORWARD:
    case REMAT_STEP_REFORWARD:
    case REMAT_STEP_BACKWARD:
    case REMAT_STEP_SUM:
        if (st->layer >= c->m->numLayers) {
            REMAT_CHECK_EXIT(c, st, "'layer out of range' (n = %zu)", c->m->numLayers);
        }
        break;
    case REMAT_STEP_LOSS_FORWARD:
    case REMAT_STEP_LOSS_BACKWARD:
        if (st->layer != c->m->numLayers) {
            REMAT_CHECK_EXIT(c, st, "'layer out of range' (n = %zu)", c->m->numLayers);
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
        if (c->nextForward != c->m->numLayers) {
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
         * cursor clause (with nothing running backward the cursor is -1); it
         * stays as the rule reads. */
        if (!c->lossForwardSeen || !c->hasBackward || c->nextBackward < 0 ||
            (ptrdiff_t)st->layer >=
                (c->lossBackwardSeen ? c->nextBackward : (ptrdiff_t)c->m->numLayers)) {
            REMAT_CHECK_EXIT(c, st, "'reforward outside window'");
        }
        if (c->stepIndex >= c->maxSteps) {
            REMAT_CHECK_EXIT(c, st, "'step budget exceeded' (maxSteps = %zu)", c->maxSteps);
        }
        /* trackRunningStats is not in the table's key, so a layer reconfigured
         * after the plan was built gets here. The config must not change during
         * a call: this check does not make a concurrent change safe. */
        if (layerForwardUpdatesState(c->m->layers[st->layer])) {
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
        if (c->nextBackward < 0) {
            REMAT_CHECK_EXIT(c, st, "'backward order: expected no further BACKWARD'");
        }
        if ((ptrdiff_t)st->layer != c->nextBackward) {
            REMAT_CHECK_EXIT(c, st, "'backward order: expected BACKWARD(%ld)'",
                             (long)c->nextBackward);
        }
        break;
    case REMAT_STEP_SUM:
        /* The PART it adds lives from that BACKWARD to this step only. */
        if (c->lastBackward != st->layer ||
            rematPlanProgram(c->sched->plan, c->mode)->nodeOps[st->layer].sum == REMAT_NONE) {
            REMAT_CHECK_EXIT(c, st,
                             "'sum order: not right after its node's BACKWARD writing a PART'");
        }
        break;
    default:
        break; /* unreachable: rule 1 exits on an unknown kind first */
    }
}

/* Rule 3: the step's slice is its node's canonical one, inside the pool and
 * of the kind's counts. A lookup, not a re-derivation: the grammar checked
 * every slice's content at build, so this rule only stops a step record that
 * no longer points at it. A plan block corrupted beyond its step records is
 * out of scope. Returns the slice; the rules after it read only that. */
static const rematOperand_t *requireOperands(const rematCheck_t *c, const rematStep_t *st) {
    const rematProgram_t *p = rematPlanProgram(c->sched->plan, c->mode);
    if ((size_t)st->operands + st->numReads + st->numWrites > p->numOperands) {
        REMAT_CHECK_EXIT(c, st, "'operand slice out of the pool' (operands %u + %u + %u > %zu)",
                         (unsigned)st->operands, (unsigned)st->numReads, (unsigned)st->numWrites,
                         p->numOperands);
    }
    const rematNodeOps_t *node = &p->nodeOps[st->layer];
    uint16_t canonical = node->forward;
    if (st->kind == REMAT_STEP_BACKWARD || st->kind == REMAT_STEP_LOSS_BACKWARD) {
        canonical = node->backward;
    } else if (st->kind == REMAT_STEP_SUM) {
        canonical = node->sum;
    }
    if (st->operands != canonical) {
        REMAT_CHECK_EXIT(c, st, "'operand slice is not the node's' (operands %u, canonical %u)",
                         (unsigned)st->operands, (unsigned)canonical);
    }
    unsigned k = 1u;
    if (st->kind != REMAT_STEP_LOSS_FORWARD && st->kind != REMAT_STEP_LOSS_BACKWARD &&
        st->kind != REMAT_STEP_SUM) {
        uint16_t ids[MODEL_MAX_ARITY];
        k = modelNodeInputs(c->m, st->layer, ids);
    }
    unsigned reads = k;
    unsigned writes = 1u;
    switch (st->kind) {
    case REMAT_STEP_LOSS_FORWARD:
        writes = 0u;
        break;
    case REMAT_STEP_BACKWARD:
        reads = 1u + k; /* dy, then every input */
        writes = k;     /* one dx slot per input */
        break;
    default:
        break; /* FORWARD, REFORWARD, LOSS_BACKWARD: k reads, one write; SUM: PART in, GRAD out */
    }
    if (st->numReads != reads || st->numWrites != writes) {
        REMAT_CHECK_EXIT(c, st,
                         "'operand counts differ from the kind' (numReads %u, numWrites %u, "
                         "expected %u, %u)",
                         (unsigned)st->numReads, (unsigned)st->numWrites, reads, writes);
    }
    return &p->operands[st->operands];
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

/* One operand the step touches, with the role its messages name it by. */
typedef struct touched {
    uint16_t wire;
    bool dy;
    bool write;
    const char *role;
} touched_t;

#define REMAT_CHECK_MAX_TOUCHED (2u * MODEL_MAX_ARITY + 1u)

/* The operands the step touches, inputs first, then dy, then the writes: a
 * META input (a BACKWARD's dead input, W_dead) and a write slot without dx
 * are not touched. Rule 3 bounded the counts. */
static size_t touchedOperands(const rematStep_t *st, const rematOperand_t *slice,
                              touched_t out[REMAT_CHECK_MAX_TOUCHED]) {
    size_t first = st->kind == REMAT_STEP_BACKWARD ? 1u : 0u;
    size_t num = 0;
    for (size_t i = first; i < st->numReads; i++) {
        if (slice[i].mode != OP_MODE_META) {
            out[num++] = (touched_t){slice[i].wire, false, false, "in"};
        }
    }
    if (first == 1u) {
        out[num++] = (touched_t){slice[0].wire, true, false, "gradIn"};
    }
    for (size_t i = st->numReads; i < (size_t)st->numReads + st->numWrites; i++) {
        if (slice[i].wire != REMAT_NONE) {
            out[num++] = (touched_t){slice[i].wire, false, true, "out"};
        }
    }
    return num;
}

static const char *wireKindName(const rematWire_t *rec) {
    return rematWireKindName(rec->kind);
}

/* Produced under the binding it has now: the bind generation is the only O(1)
 * catch for a row that re-binds a wire without its producer running again. An
 * ACC needs it too: it adds into the bytes its opening WRITE left. */
static void requireProducedUnderItsBinding(const rematCheck_t *c, const rematStep_t *st,
                                           uint16_t w) {
    const rematWire_t *rec = &c->sched->wires->wires[w];
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

/* Rule 4: resident, and, unless borrowed, produced under the binding it has
 * now; a GRAD or PART also complete: every contribution it expects arrived.
 * A GRAD or PART text names the buffer, then the wire id in brackets: a buffer
 * has several wires, a PART and its GRAD among them. */
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
    requireProducedUnderItsBinding(c, st, w);
    if (c->pending[w] != 0u) {
        REMAT_CHECK_EXIT(c, st, "'operand incomplete: %s %u (wire %u), %u contributions pending'",
                         wireKindName(rec), (unsigned)rec->index, (unsigned)w,
                         (unsigned)c->pending[w]);
    }
}

static void requireWritable(const rematCheck_t *c, const rematStep_t *st, uint16_t w) {
    const rematWire_t *rec = &c->sched->wires->wires[w];
    if (rematWireHdr(c->sched->wires, w)->data == NULL) {
        REMAT_CHECK_EXIT(c, st, "'output not resident: %s %u'", wireKindName(rec),
                         (unsigned)rec->index);
    }
}

/* A BACKWARD input its layer does not read (META) may be dead (W_dead): only
 * its header's static metadata is used, so it is not in the touched list. */
static void requireResident(const rematCheck_t *c, const rematStep_t *st, const touched_t *ops,
                            size_t num) {
    for (size_t i = 0; i < num; i++) {
        if (ops[i].write) {
            requireWritable(c, st, ops[i].wire);
        } else {
            requireReadable(c, st, ops[i].wire, ops[i].role);
        }
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

/* The one sharing the rules admit: a BACKWARD that writes its dx into the
 * bytes of the gradient it reads, for a pair the plan marked (the dx wire
 * right after dy), at an identical base and of equal bytes, never a partial
 * overlap. The marks come from the plan, not from the table the rows bind
 * through, so a call whose table missed them stops by name. The overwritten
 * gradient needs no stale-read guard: it is read by exactly one BACKWARD,
 * this one, and rule 2 fixes the order. */
static bool inPlacePair(const rematCheck_t *c, uint16_t dy, uint16_t dx) {
    const uint8_t *marks = c->sched->plan->inPlace;
    const rematWireTable_t *t = c->sched->wires;
    return marks != NULL && marks[dx] != 0u && dy + 1u == dx &&
           rematWireHdr(t, dy)->data == rematWireHdr(t, dx)->data &&
           rematWireBytes(t, dy) == rematWireBytes(t, dx);
}

/* Pairwise over every operand the step touches, not only dy and dx: a
 * LayerNorm-style backward writes dx while it still reads x. A dead input
 * (W_dead) is not touched, so it is not compared. One wire twice on the same
 * side is ADD(y, y): its forward reads y twice, and its backward writes the
 * dx slots one after the other, so neither pair is a hazard. */
static void requireDisjoint(const rematCheck_t *c, const rematStep_t *st, const touched_t *ops,
                            size_t num) {
    for (size_t i = 0; i < num; i++) {
        for (size_t j = i + 1u; j < num; j++) {
            if (ops[i].wire == ops[j].wire && ops[i].write == ops[j].write) {
                continue;
            }
            if (ops[i].dy && ops[j].write && inPlacePair(c, ops[i].wire, ops[j].wire)) {
                continue;
            }
            requireApart(c, st, ops[i].wire, ops[i].role, ops[j].wire, ops[j].role);
        }
    }
}

/* A contribution into a GRAD or PART, taken in slot order so that ADD(y, y)'s
 * WRITE opens the wire before its ACC adds in: a WRITE opens a wire nothing
 * has reached yet, an ACC adds into the bytes an opening WRITE left under the
 * current binding, and neither exceeds the contributions the wire expects. */
static void contribute(rematCheck_t *c, const rematStep_t *st, uint16_t w, uint8_t mode) {
    const rematWire_t *rec = &c->sched->wires->wires[w];
    if (mode == OP_MODE_WRITE) {
        if (c->producedGen[w] != 0u) {
            REMAT_CHECK_EXIT(c, st,
                             "'operand reopened: %s %u (wire %u) already holds a contribution'",
                             wireKindName(rec), (unsigned)rec->index, (unsigned)w);
        }
        c->producedGen[w] = rec->bindGen;
    } else {
        requireProducedUnderItsBinding(c, st, w);
    }
    if (c->pending[w] == 0u) {
        REMAT_CHECK_EXIT(c, st,
                         "'operand overfilled: %s %u (wire %u) expects no further contribution'",
                         wireKindName(rec), (unsigned)rec->index, (unsigned)w);
    }
    c->pending[w]--;
}

/* Rule 6: an output is produced under the binding it has now. */
static void commit(rematCheck_t *c, const rematStep_t *st, const rematOperand_t *slice) {
    const rematWireTable_t *t = c->sched->wires;
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
        c->nextBackward = highestRunningBelow(t, c->nextBackward);
        break;
    case REMAT_STEP_REFORWARD:
    case REMAT_STEP_SUM:
        break; /* only its output, below */
    default:
        break; /* unreachable: rule 1 exits on an unknown kind first */
    }
    for (size_t i = st->numReads; i < (size_t)st->numReads + st->numWrites; i++) {
        uint16_t w = slice[i].wire;
        if (w == REMAT_NONE) {
            continue;
        }
        if (t->wires[w].kind != REMAT_WIRE_ACT) {
            contribute(c, st, w, slice[i].mode);
        } else if (slice[i].mode == OP_MODE_WRITE) {
            c->producedGen[w] = t->wires[w].bindGen; /* a REFORWARD writes an ACT again */
        }
    }
    c->lastBackward = (st->kind == REMAT_STEP_BACKWARD) ? st->layer : REMAT_NONE;
    c->stepIndex++;
}

void rematCheckStep(rematCheck_t *c, const rematStep_t *st, rematOperands_t *ops) {
    requireKindAndLayer(c, st);
    requireOrder(c, st);
    const rematOperand_t *slice = requireOperands(c, st);
    fillOperands(c, st, slice, ops);
    touched_t touched[REMAT_CHECK_MAX_TOUCHED];
    size_t num = touchedOperands(st, slice, touched);
    requireResident(c, st, touched, num);
    requireDisjoint(c, st, touched, num);
    commit(c, st, slice);
}

#define REMAT_FINISH_EXIT(c, violation, ...)                                                       \
    do {                                                                                           \
        PRINT_ERROR("remat[%s]: stream of %zu steps violates " violation, (c)->sched->fns->name,   \
                    (c)->stepIndex, ##__VA_ARGS__);                                                \
        exit(1);                                                                                   \
    } while (false)

/* Stream completeness. Cursors, not bitsets: a stream the step rules admitted is
 * complete iff every cursor reached its end. With no node running backward the
 * backward cursor starts at -1: LOSS_BACKWARD with no BACKWARD is complete. */
void rematCheckFinish(const rematCheck_t *c) {
    if (c->nextForward != c->m->numLayers) {
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
    if (c->hasBackward && c->nextBackward >= 0) {
        REMAT_FINISH_EXIT(c, "'incomplete stream: missing BACKWARD(%ld)'", (long)c->nextBackward);
    }
    /* Every SUM too: a missing one leaves its GRAD a contribution short. */
    const rematWireTable_t *t = c->sched->wires;
    for (uint16_t w = 0; w < t->numWires; w++) {
        if (c->pending[w] != 0u) {
            REMAT_FINISH_EXIT(c, "'incomplete stream: %s %u (wire %u), %u contributions pending'",
                              wireKindName(&t->wires[w]), (unsigned)t->wires[w].index, (unsigned)w,
                              (unsigned)c->pending[w]);
        }
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
