#define SOURCE_FILE "REMAT_PLAN"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "Common.h"
#include "Layer.h"
#include "LayerConfigAccess.h"
#include "LossFunction.h"
#include "RematCheckedSize.h"
#include "RematPlan.h"
#include "RematPlanPolicy.h"
#include "RematScheduler.h"
#include "StorageApi.h"

void rematBackwardRange(layer_t **model, size_t n, lossFuncType_t lt, size_t *deepest,
                        ptrdiff_t *top) {
    *deepest = deepestTrainableIndex(model, n);
    *top = (ptrdiff_t)n - 1 - (lt == CROSS_ENTROPY ? 1 : 0);
}

size_t rematWalkOpening(const rematProgram_t *p, rematWalk_t *w) {
    if (w->open < p->numRanges && p->ranges[w->open].begin == w->step) {
        return w->open++;
    }
    return REMAT_NONE;
}

size_t rematWalkClosing(const rematProgram_t *p, rematWalk_t *w) {
    if (w->close < p->numRanges && p->ranges[p->endOrder[w->close]].end == w->step) {
        return p->endOrder[w->close++];
    }
    return REMAT_NONE;
}

static bool endsBefore(const rematRange_t *ranges, uint16_t a, uint16_t b) {
    return ranges[a].end < ranges[b].end ||
           (ranges[a].end == ranges[b].end && ranges[a].wire < ranges[b].wire);
}

/* In place, no scratch: O(R^2) worst case, within the O(R^2 log R) of the
 * ARENA placement that follows it at init. R is at most numWires - 1 for
 * STORE_ALL and LIVENESS, and at most ODT_REMAT_MAX_RANGES for a SEQUENCE or
 * RETAIN_LIST plan. */
static void sortEndOrder(rematProgram_t *p) {
    for (size_t i = 0; i < p->numRanges; i++) {
        p->endOrder[i] = (uint16_t)i;
    }
    for (size_t i = 1; i < p->numRanges; i++) {
        uint16_t id = p->endOrder[i];
        size_t k = i;
        while (k > 0 && endsBefore(p->ranges, id, p->endOrder[k - 1])) {
            p->endOrder[k] = p->endOrder[k - 1];
            k--;
        }
        p->endOrder[k] = id;
    }
}

/* Overflow-checked here are the plan-block size (stepsAt/rangesAt/endOrderAt/
 * blockBytes, via planAdd/planMul), the step bound, the two
 * peaks, peakLiveBytes and the activation-only peak (accumulated with planAdd
 * in peakBytesUpTo, each under its own label), the activations' element
 * counts, and the two costs, forwardCost and recomputeCost.
 * rematTrainStepCount, rematBackwardStep and numRanges (numWires - 1 less
 * the in-place members) are plain size_t arithmetic on modelSize,
 * backwardTop, deepest and numWires -- all bounded well under SIZE_MAX by the
 * table's numWires < REMAT_NONE guard, so they cannot overflow. Only the two
 * costs' exits are reachable (a layer's cost has no bound the table checked)
 * and have death tests. The others have none: a program holds at most
 * UINT16_MAX steps and no more ranges than steps, and grammar rule 6 keeps
 * the ranges of one wire apart, so a peak never exceeds the table's checked
 * total of wire bytes; an element count is a product the table checked when
 * it derived the same shape. */
static size_t planAdd(size_t a, size_t b, const char *quantity) {
    size_t out;
    if (!checkedAddSize(a, b, &out)) {
        PRINT_ERROR("rematPlanBuild: size overflow computing %s", quantity);
        exit(1);
    }
    return out;
}

static size_t planMul(size_t a, size_t b, const char *quantity) {
    size_t out;
    if (!checkedMulSize(a, b, &out)) {
        PRINT_ERROR("rematPlanBuild: size overflow computing %s", quantity);
        exit(1);
    }
    return out;
}

/* The peak over the program's steps of the live bytes of the wires whose id
 * is at most maxWire: openings count before the peak check, closings after.
 * quantity names the peak in the overflow exit. */
static size_t peakBytesUpTo(const rematProgram_t *p, const rematWireTable_t *t, size_t maxWire,
                            const char *quantity) {
    rematWalk_t walk = {0};
    size_t live = 0;
    size_t peak = 0;
    for (walk.step = 0; walk.step < p->numSteps; walk.step++) {
        for (size_t r; (r = rematWalkOpening(p, &walk)) != REMAT_NONE;) {
            if (p->ranges[r].wire <= maxWire) {
                live = planAdd(live, t->wires[p->ranges[r].wire].bytes, quantity);
            }
        }
        if (live > peak) {
            peak = live;
        }
        for (size_t r; (r = rematWalkClosing(p, &walk)) != REMAT_NONE;) {
            if (p->ranges[r].wire <= maxWire) {
                live -= t->wires[p->ranges[r].wire].bytes;
            }
        }
    }
    return peak;
}

static size_t peakLiveBytesOf(const rematProgram_t *p, const rematWireTable_t *t) {
    return peakBytesUpTo(p, t, SIZE_MAX, "peakLiveBytes");
}

/* ACT j is wire j (0..n): the ACT wires are the ids up to modelSize. */
size_t rematProgramActPeakBytes(const rematProgram_t *p, const rematWireTable_t *t) {
    return peakBytesUpTo(p, t, t->modelSize, "activationsPeakBytes");
}

static size_t roundUpTo(size_t x, size_t align) {
    return planAdd(x, align - 1u, "blockBytes") & ~(align - 1u);
}

/* One program's arrays inside the plan block, from `at` on: steps, ranges,
 * endOrder, each at its own alignment; `end` is the first byte after them. */
typedef struct programLayout {
    size_t stepsAt, rangesAt, endOrderAt, end;
} programLayout_t;

static programLayout_t layoutProgram(size_t at, size_t numSteps, size_t numRanges) {
    programLayout_t l;
    l.stepsAt = roundUpTo(at, _Alignof(rematStep_t));
    l.rangesAt = roundUpTo(
        planAdd(l.stepsAt, planMul(numSteps, sizeof(rematStep_t), "blockBytes"), "blockBytes"),
        _Alignof(rematRange_t));
    l.endOrderAt = roundUpTo(
        planAdd(l.rangesAt, planMul(numRanges, sizeof(rematRange_t), "blockBytes"), "blockBytes"),
        _Alignof(uint16_t));
    l.end = planAdd(l.endOrderAt, planMul(numRanges, sizeof(uint16_t), "blockBytes"), "blockBytes");
    return l;
}

static void placeProgram(rematProgram_t *p, uint8_t *block, const programLayout_t *l,
                         size_t numSteps, size_t numRanges) {
    p->numSteps = numSteps;
    p->steps = (rematStep_t *)(block + l->stepsAt);
    p->numRanges = numRanges;
    p->ranges = (rematRange_t *)(block + l->rangesAt);
    p->endOrder = (uint16_t *)(block + l->endOrderAt);
}

static void grammarExit(size_t s, const rematStep_t *st, const char *rule) {
    PRINT_ERROR("rematPlanBuild: step #%zu (kind %u, layer %u) violates grammar %s", s,
                (unsigned)st->kind, (unsigned)st->layer, rule);
    exit(1);
}

static void grammarEndExit(size_t numSteps, const char *rule) {
    PRINT_ERROR("rematPlanBuild: the stream of %zu steps violates grammar %s", numSteps, rule);
    exit(1);
}

/* The wire whose range holds this one's bytes: an in-place member walks down
 * to its chain's root; every other wire is its own. */
static uint16_t chainRoot(const uint8_t *inPlace, uint16_t wire) {
    while (inPlace != NULL && inPlace[wire] != 0u) {
        wire--;
    }
    return wire;
}

/* Linear in the ranges on purpose (O(S * R) over a program): independent of
 * the walk and of endOrder. The message names the operand, the search its
 * root. */
static void requireCovered(const rematProgram_t *p, size_t s, uint16_t wire, uint16_t root,
                           const char *role) {
    if (root == 0) {
        return; /* ACT 0 is borrowed: always present */
    }
    for (size_t r = 0; r < p->numRanges; r++) {
        const rematRange_t *range = &p->ranges[r];
        if (range->wire == root && range->begin <= s && s <= range->end) {
            return;
        }
    }
    const rematStep_t *st = &p->steps[s];
    PRINT_ERROR("rematPlanBuild: step #%zu (kind %u, layer %u) violates grammar rule 4: it %s wire "
                "%u outside any open range",
                s, (unsigned)st->kind, (unsigned)st->layer, role, (unsigned)wire);
    exit(1);
}

/* Rule 6 for a write: no range of the wire that opened before this step is
 * still open, so the write starts a range of its own and the ranges of one
 * wire never overlap. Linear in the ranges, like requireCovered. */
static void requireNoEarlierRangeOpen(const rematProgram_t *p, size_t s, uint16_t wire) {
    for (size_t r = 0; r < p->numRanges; r++) {
        const rematRange_t *range = &p->ranges[r];
        if (range->wire == wire && range->begin < s && s <= range->end) {
            const rematStep_t *st = &p->steps[s];
            PRINT_ERROR("rematPlanBuild: step #%zu (kind %u, layer %u) violates grammar rule 6: "
                        "it writes wire %u into a range opened earlier",
                        s, (unsigned)st->kind, (unsigned)st->layer, (unsigned)wire);
            exit(1);
        }
    }
}

/* The wires one step reads and the one it writes (REMAT_NONE: none), for a
 * step the kind rules already admitted: rule 1 and the rule-6 window bound a
 * FORWARD's and a REFORWARD's layer below n, LOSS_* steps carry layer n, and
 * rule 3 keeps a BACKWARD's layer in [deepest, top], so every id is a real
 * wire. A REFORWARD has a FORWARD's operands. */
typedef struct stepOperands {
    uint16_t reads[2];
    size_t numReads;
    uint16_t write;
} stepOperands_t;

static stepOperands_t operandsOf(const rematWireTable_t *t, layer_t **model,
                                 const rematStep_t *st) {
    size_t n = t->modelSize;
    size_t l = st->layer;
    stepOperands_t o = {.numReads = 0, .write = REMAT_NONE};
    switch (st->kind) {
    case REMAT_STEP_FORWARD:
    case REMAT_STEP_REFORWARD:
        o.reads[o.numReads++] = rematActId(t, l);
        o.write = rematActId(t, l + 1u);
        break;
    case REMAT_STEP_LOSS_FORWARD:
        o.reads[o.numReads++] = rematActId(t, n);
        break;
    case REMAT_STEP_LOSS_BACKWARD:
        o.reads[o.numReads++] = rematActId(t, n);
        o.write = rematGradId(t, n);
        break;
    case REMAT_STEP_BACKWARD:
        o.reads[o.numReads++] =
            ((ptrdiff_t)l == t->backwardTop) ? rematGradId(t, n) : rematGradId(t, l + 1u);
        if (layerBackwardReadsInput(model[l])) {
            o.reads[o.numReads++] = rematActId(t, l);
        }
        if (l > t->deepest) {
            o.write = rematGradId(t, l);
        }
        break;
    default:
        break; /* unreachable: the grammar's kind switch exits on an unknown kind first */
    }
    return o;
}

/* Rule 4 (every operand inside a range of its wire, a member inside its
 * root's) and rule 6's write rule. A member's write lands in the range its
 * root's write opened, so rule 6 skips it. Its rule-4 check cannot fail
 * first: the same step reads wire - 1 of the same chain, which resolves to
 * the same root and is checked before it. */
static void requireOperandsCovered(const rematProgram_t *p, const rematWireTable_t *t,
                                   layer_t **model, const uint8_t *inPlace, size_t s) {
    stepOperands_t o = operandsOf(t, model, &p->steps[s]);
    for (size_t i = 0; i < o.numReads; i++) {
        requireCovered(p, s, o.reads[i], chainRoot(inPlace, o.reads[i]), "reads");
    }
    if (o.write != REMAT_NONE) {
        uint16_t root = chainRoot(inPlace, o.write);
        requireCovered(p, s, o.write, root, "writes");
        if (root == o.write) {
            requireNoEarlierRangeOpen(p, s, o.write);
        }
    }
}

static bool stepReads(const rematWireTable_t *t, layer_t **model, const rematStep_t *st,
                      uint16_t wire) {
    stepOperands_t o = operandsOf(t, model, st);
    for (size_t i = 0; i < o.numReads; i++) {
        if (o.reads[i] == wire) {
            return true;
        }
    }
    return false;
}

static void rangeExit(size_t r, const rematRange_t *range, const char *rule) {
    PRINT_ERROR("rematPlanBuild: range #%zu (wire %u, steps %u..%u) violates grammar rule 6: %s", r,
                (unsigned)range->wire, (unsigned)range->begin, (unsigned)range->end, rule);
    exit(1);
}

/* Rule 6 over the ranges, once every step has passed: each range lies inside
 * the stream; the array is sorted by begin, ties by wire id (the walk opens
 * ranges in array order); each range opens at the step that writes its wire;
 * and a range a REFORWARD opens holds a read of that wire, so no recompute is
 * dead. O(R) plus the lengths of the REFORWARD ranges. */
static void requireRangesWellFormed(const rematProgram_t *p, const rematWireTable_t *t,
                                    layer_t **model) {
    for (size_t r = 0; r < p->numRanges; r++) {
        const rematRange_t *range = &p->ranges[r];
        if (range->begin > range->end || range->end >= p->numSteps) {
            rangeExit(r, range, "ranges lie inside the stream");
        }
        const rematRange_t *prev = (r > 0) ? &p->ranges[r - 1u] : NULL;
        if (prev != NULL && (prev->begin > range->begin ||
                             (prev->begin == range->begin && prev->wire >= range->wire))) {
            rangeExit(r, range, "ranges sorted by begin, ties by wire id");
        }
        const rematStep_t *opener = &p->steps[range->begin];
        if (operandsOf(t, model, opener).write != range->wire) {
            rangeExit(r, range, "a range opens where its wire is written");
        }
        if (opener->kind == REMAT_STEP_REFORWARD) {
            bool read = false;
            for (size_t s = range->begin + 1u; s <= range->end && !read; s++) {
                read = stepReads(t, model, &p->steps[s], range->wire);
            }
            if (!read) {
                rangeExit(r, range, "a REFORWARD's output is read inside its range");
            }
        }
    }
}

void rematPlanValidateGrammar(const rematProgram_t *p, const rematWireTable_t *t, layer_t **model,
                              rematMode_t mode, const uint8_t *inPlace) {
    size_t n = t->modelSize;
    size_t nextForward = 0;
    ptrdiff_t nextBackward = t->backwardTop;
    bool lossForward = false;
    bool lossBackward = false;
    /* Rule 6's step bound. The table keeps n below UINT16_MAX, so n(n + 1)
     * fits even a 32-bit size_t and the sum needs no checked arithmetic. */
    if (p->numSteps > 2u * n + 2u + n * (n + 1u) / 2u) {
        grammarEndExit(p->numSteps, "rule 6: more steps than 2n + 2 + n(n+1)/2");
    }
    for (size_t s = 0; s < p->numSteps; s++) {
        const rematStep_t *st = &p->steps[s];
        if (mode == REMAT_MODE_EVAL &&
            (st->kind == REMAT_STEP_LOSS_BACKWARD || st->kind == REMAT_STEP_BACKWARD)) {
            grammarExit(s, st, "rule 2: no LOSS_BACKWARD or BACKWARD in an EVAL program");
        }
        if (mode == REMAT_MODE_EVAL && st->kind == REMAT_STEP_REFORWARD) {
            grammarExit(s, st, "rule 6: no REFORWARD in an EVAL program");
        }
        switch (st->kind) {
        case REMAT_STEP_FORWARD:
            if (lossForward || st->layer != nextForward || st->layer >= n) {
                grammarExit(s, st, "rule 1: one FORWARD per layer, ascending, before LOSS_FORWARD");
            }
            nextForward++;
            break;
        case REMAT_STEP_LOSS_FORWARD:
            if (lossForward || nextForward != n || st->layer != n) {
                grammarExit(s, st, "rule 1: one LOSS_FORWARD, after FORWARD(n-1)");
            }
            lossForward = true;
            break;
        case REMAT_STEP_LOSS_BACKWARD:
            if (!t->hasBackward || !lossForward || lossBackward || st->layer != n) {
                grammarExit(s, st,
                            "rule 2: LOSS_BACKWARD iff hasBackward, once, after LOSS_FORWARD");
            }
            lossBackward = true;
            break;
        case REMAT_STEP_BACKWARD:
            if (!lossBackward || (ptrdiff_t)st->layer != nextBackward ||
                nextBackward < (ptrdiff_t)t->deepest) {
                grammarExit(s, st,
                            "rule 3: BACKWARD top..deepest, descending, after LOSS_BACKWARD");
            }
            nextBackward--;
            break;
        case REMAT_STEP_REFORWARD:
            /* The checker's window, coded here on its own. hasBackward needs no
             * test: without it deepest == n > top >= nextBackward. */
            if (!lossForward || nextBackward < (ptrdiff_t)t->deepest ||
                (ptrdiff_t)st->layer >= (lossBackward ? nextBackward : (ptrdiff_t)n)) {
                grammarExit(s, st,
                            "rule 6: a REFORWARD comes after LOSS_FORWARD, below a pending "
                            "backward step");
            }
            if (layerForwardUpdatesState(model[st->layer])) {
                PRINT_ERROR("rematPlanBuild: step #%zu (kind %u, layer %u) violates grammar rule "
                            "6: layer %u updates its state in every training forward (a tracked, "
                            "unfrozen BatchNorm1d), so it is never recomputed; keep ACT %u",
                            s, (unsigned)st->kind, (unsigned)st->layer, (unsigned)st->layer,
                            (unsigned)st->layer + 1u);
                exit(1);
            }
            break;
        default:
            PRINT_ERROR("rematPlanBuild: step #%zu (kind %u, layer %u) violates grammar: unknown "
                        "step kind",
                        s, (unsigned)st->kind, (unsigned)st->layer);
            exit(1);
        }
        requireOperandsCovered(p, t, model, inPlace, s);
    }
    if (!lossForward) {
        grammarEndExit(p->numSteps, "rule 1: no LOSS_FORWARD");
    }
    if (mode == REMAT_MODE_TRAIN && lossBackward != t->hasBackward) {
        grammarEndExit(p->numSteps, "rule 2: LOSS_BACKWARD missing");
    }
    if (mode == REMAT_MODE_TRAIN && t->hasBackward && nextBackward >= (ptrdiff_t)t->deepest) {
        grammarEndExit(p->numSteps, "rule 3: BACKWARD set incomplete");
    }
    requireRangesWellFormed(p, t, model);
}

/* The generator and grammar rule 4 both read the read-set off `model`, so a
 * model other than the table's yields the same wrong ranges in both and the
 * plan validates. Only the key facts are comparable: the signature has no n. */
static void requireTheTablesModel(const rematWireTable_t *t, layer_t **model) {
    for (size_t i = 0; i < t->modelSize; i++) {
        if ((uint8_t)model[i]->type != t->layerType[i]) {
            PRINT_ERROR("rematPlanBuild: the model differs from the table's key at "
                        "'layerType[%zu]': table %u, model %u",
                        i, (unsigned)t->layerType[i], (unsigned)model[i]->type);
            exit(1);
        }
        uint8_t frozen = layerIsFrozen(model[i]) ? 1u : 0u;
        if (frozen != t->frozen[i]) {
            PRINT_ERROR("rematPlanBuild: the model differs from the table's key at "
                        "'frozen[%zu]': table %u, model %u",
                        i, (unsigned)t->frozen[i], (unsigned)frozen);
            exit(1);
        }
    }
}

/* min(UINT16_MAX, 2n + 2 + n(n+1)/2): the most steps a TRAIN program of n
 * layers may hold (grammar rules 5 and 6), for the checks that must run
 * before the plan block exists. */
static size_t stepBound(size_t n) {
    size_t triangle = planMul(n, n + 1u, "the step bound") / 2u;
    size_t bound = planAdd(planAdd(planMul(2u, n, "the step bound"), 2u, "the step bound"),
                           triangle, "the step bound");
    return (bound < UINT16_MAX) ? bound : UINT16_MAX;
}

/* A field of another policy is a caller error, never ignored. */
static void requireTheFieldsOfThePolicy(const rematPlanSpec_t *spec) {
    if (spec->policy != REMAT_PLAN_SEQUENCE &&
        (spec->sequence != NULL || spec->sequenceLen != 0u)) {
        PRINT_ERROR("rematPlanBuild: sequence or sequenceLen is set, but the policy %d is not "
                    "REMAT_PLAN_SEQUENCE",
                    (int)spec->policy);
        exit(1);
    }
    if (spec->policy != REMAT_PLAN_RETAIN_LIST && spec->retain != NULL) {
        PRINT_ERROR("rematPlanBuild: retain is set, but the policy %d is not "
                    "REMAT_PLAN_RETAIN_LIST",
                    (int)spec->policy);
        exit(1);
    }
}

/* RETAIN_LIST's checks before the counting pass reads the mask. A model whose
 * loss backward runs but no BACKWARD step does (n = 1 under cross entropy, or
 * only the last layer trains) has no REFORWARD window: a dropped ACT n could
 * never be recomputed before LOSS_BACKWARD reads it. */
static void requireARetainMaskWithAWindow(const rematPlanSpec_t *spec, const rematWireTable_t *t) {
    if (spec->retain == NULL) {
        PRINT_ERROR("rematPlanBuild: REMAT_PLAN_RETAIN_LIST needs a retain mask, got NULL");
        exit(1);
    }
    size_t n = t->modelSize;
    if (t->hasBackward && t->backwardTop < (ptrdiff_t)t->deepest && !spec->retain[n]) {
        PRINT_ERROR("rematPlanBuild: RETAIN_LIST drops ACT %zu but no REFORWARD window exists "
                    "(LOSS_BACKWARD reads it and no BACKWARD step runs); keep ACT %zu",
                    n, n);
        exit(1);
    }
}

/* Rule 5 for a generated program, before anything is reserved or cast to
 * uint16_t. A RETAIN_LIST program holds at most 3n + 2 steps, never more than
 * 2n + 2 + n(n+1)/2, so only UINT16_MAX can stop it. */
static void requireAProgramWithinTheBound(size_t numSteps, size_t n) {
    size_t bound = stepBound(n);
    if (numSteps > bound) {
        PRINT_ERROR("rematPlanBuild: the RETAIN_LIST program has %zu steps, above the step bound "
                    "%zu (min(UINT16_MAX, 2n + 2 + n(n+1)/2), n = %zu)",
                    numSteps, bound, n);
        exit(1);
    }
}

/* SEQUENCE's checks before anything reads the caller's records: the counting
 * pass and the copy read exactly sequenceLen of them. The checked byte count
 * of the copy is layoutProgram's, before reserveMemory; it cannot overflow
 * once the length is at most UINT16_MAX, so it has no death test. */
static void requireASequenceWithinTheBound(const rematPlanSpec_t *spec, size_t n) {
    if (spec->sequence == NULL) {
        PRINT_ERROR("rematPlanBuild: REMAT_PLAN_SEQUENCE needs a sequence, got NULL");
        exit(1);
    }
    if (spec->sequenceLen == 0u) {
        PRINT_ERROR("rematPlanBuild: REMAT_PLAN_SEQUENCE needs at least one step, got "
                    "sequenceLen 0");
        exit(1);
    }
    size_t bound = stepBound(n);
    if (spec->sequenceLen > bound) {
        PRINT_ERROR("rematPlanBuild: sequenceLen %zu is above the step bound %zu "
                    "(min(UINT16_MAX, 2n + 2 + n(n+1)/2), n = %zu)",
                    spec->sequenceLen, bound, n);
        exit(1);
    }
}

/* A SEQUENCE or RETAIN_LIST plan opens a range per write, up to one per step:
 * the cap bounds its quadratic plan work before anything is reserved.
 * STORE_ALL and LIVENESS keep at most numWires - 1 ranges and are not capped
 * here. */
static void requireRangesWithinTheCap(size_t numRanges) {
    if (numRanges > ODT_REMAT_MAX_RANGES) {
        PRINT_ERROR("rematPlanBuild: plan has %zu ranges, above ODT_REMAT_MAX_RANGES (%u)",
                    numRanges, (unsigned)ODT_REMAT_MAX_RANGES);
        exit(1);
    }
}

static size_t activationElements(const shape_t *shape) {
    size_t count = 1;
    for (size_t d = 0; d < shape->numberOfDimensions; d++) {
        count = planMul(count, shape->dimensions[d], "the element count of an activation");
    }
    return count;
}

/* The TRAIN program's recompute numbers. The element counts come from the
 * table's input key through the calcOutputShape chain the table derived its
 * wires from, on stack scratch of maxRank dims (the table keeps bytes, which
 * do not give BFP element counts). Grammar rule 1 runs every layer's FORWARD
 * once, so forwardCost is the sum over the layers. A layer's REFORWARDs are
 * counted by a scan of the steps, only in a program that recomputes: a
 * SEQUENCE or RETAIN_LIST plan, whose range cap keeps n at most
 * ODT_REMAT_MAX_RANGES (ACT 1..n each have a range; only GRAD wires share
 * one), so the scans read at most n * numSteps steps. */
static void recordRecomputeNumbers(rematPlan_t *p, const rematWireTable_t *t, layer_t **model) {
    const rematProgram_t *train = &p->train;
    size_t refwdSteps = 0;
    for (size_t s = 0; s < train->numSteps; s++) {
        refwdSteps += (train->steps[s].kind == REMAT_STEP_REFORWARD) ? 1u : 0u;
    }
    size_t maxRank = t->maxRank;
    size_t dimsA[maxRank], orderA[maxRank], dimsB[maxRank], orderB[maxRank];
    shape_t scratch[2] = {{.dimensions = dimsA, .orderOfDimensions = orderA},
                          {.dimensions = dimsB, .orderOfDimensions = orderB}};
    shape_t input = {.numberOfDimensions = t->inputRank,
                     .dimensions = t->inputDims,
                     .orderOfDimensions = t->inputOrder};
    shape_t *prev = &input;
    size_t forwardCost = 0;
    size_t recomputeCost = 0;
    for (size_t l = 0; l < t->modelSize; l++) {
        layer_t *layer = model[l];
        shape_t *out = &scratch[l % 2u];
        out->numberOfDimensions = (layer->type == FLATTEN) ? 2u : prev->numberOfDimensions;
        layerFunctions[layer->type].calcOutputShape(layer, prev, out);
        size_t cost = layerForwardCost(layer, activationElements(prev), activationElements(out));
        forwardCost = planAdd(forwardCost, cost, "forwardCost");
        if (refwdSteps > 0u) {
            size_t recomputes = 0;
            for (size_t s = 0; s < train->numSteps; s++) {
                if (train->steps[s].kind == REMAT_STEP_REFORWARD && train->steps[s].layer == l) {
                    recomputes++;
                }
            }
            recomputeCost =
                planAdd(recomputeCost, planMul(cost, recomputes, "recomputeCost"), "recomputeCost");
        }
        prev = out;
    }
    p->refwdSteps = refwdSteps;
    p->forwardCost = forwardCost;
    p->recomputeCost = recomputeCost;
}

bool rematPlanBuild(rematPlan_t **out, const rematWireTable_t *t, layer_t **model,
                    const rematPlanSpec_t *spec) {
    *out = NULL;
    const rematPlanSpec_t storeAll = {.policy = REMAT_PLAN_STORE_ALL};
    if (spec == NULL) {
        spec = &storeAll;
    }
    rematPlanPolicy_t policy = spec->policy;
    if (policy != REMAT_PLAN_STORE_ALL && policy != REMAT_PLAN_LIVENESS &&
        policy != REMAT_PLAN_SEQUENCE && policy != REMAT_PLAN_RETAIN_LIST) {
        PRINT_ERROR("rematPlanBuild: unknown policy %d", (int)policy);
        exit(1);
    }
    requireTheFieldsOfThePolicy(spec);
    requireTheTablesModel(t, model);
    size_t numSteps = rematTrainStepCount(t);
    size_t numMembers = spec->inPlaceDx ? rematFillInPlaceMarks(t, NULL) : 0u;
    size_t numRanges = t->numWires - 1u - numMembers;
    if (policy == REMAT_PLAN_SEQUENCE) {
        requireASequenceWithinTheBound(spec, t->modelSize);
        numSteps = spec->sequenceLen;
        numRanges = rematScanRangeCount(t, spec->sequence, numSteps, spec->inPlaceDx);
        requireRangesWithinTheCap(numRanges);
    } else if (policy == REMAT_PLAN_RETAIN_LIST) {
        requireARetainMaskWithAWindow(spec, t);
        numSteps = rematFillRetainSteps(t, model, spec->retain, spec->inPlaceDx, NULL, &numRanges);
        requireAProgramWithinTheBound(numSteps, t->modelSize);
        requireRangesWithinTheCap(numRanges);
    }
    size_t evalSteps = t->modelSize + 1u;
    size_t evalRanges = t->modelSize;
    programLayout_t trainAt = layoutProgram(sizeof(rematPlan_t), numSteps, numRanges);
    programLayout_t evalAt = layoutProgram(trainAt.end, evalSteps, evalRanges);
    /* The marks close the block: one byte per wire, no alignment. */
    size_t blockBytes =
        spec->inPlaceDx ? planAdd(evalAt.end, t->numWires, "blockBytes") : evalAt.end;
    uint8_t *block = reserveMemory(blockBytes);
    if (block == NULL) {
        return false;
    }
    rematPlan_t *p = (rematPlan_t *)block;
    p->policy = policy;
    p->blockBytes = blockBytes;
    p->inPlace = spec->inPlaceDx ? block + evalAt.end : NULL;
    if (p->inPlace != NULL) {
        (void)rematFillInPlaceMarks(t, p->inPlace);
    }
    rematProgram_t *train = &p->train;
    placeProgram(train, block, &trainAt, numSteps, numRanges);
    if (policy == REMAT_PLAN_SEQUENCE) {
        memcpy(train->steps, spec->sequence, numSteps * sizeof(rematStep_t));
        rematScanRanges(t, model, train->steps, numSteps, spec->inPlaceDx, train->ranges);
    } else if (policy == REMAT_PLAN_RETAIN_LIST) {
        (void)rematFillRetainSteps(t, model, spec->retain, spec->inPlaceDx, train->steps, NULL);
        rematScanRanges(t, model, train->steps, numSteps, spec->inPlaceDx, train->ranges);
    } else {
        rematFillTrainSteps(t, train->steps);
        rematFillTrainRanges(policy, t, model, numSteps, spec->inPlaceDx, train->ranges);
    }
    sortEndOrder(train);
    rematPlanValidateGrammar(train, t, model, REMAT_MODE_TRAIN, p->inPlace);
    train->peakLiveBytes = peakLiveBytesOf(train, t);
    recordRecomputeNumbers(p, t, model);
    rematProgram_t *eval = &p->eval;
    placeProgram(eval, block, &evalAt, evalSteps, evalRanges);
    rematFillEvalProgram(t, eval->steps, eval->ranges);
    sortEndOrder(eval);
    rematPlanValidateGrammar(eval, t, model, REMAT_MODE_EVAL, NULL);
    eval->peakLiveBytes = peakLiveBytesOf(eval, t);
    *out = p;
    return true;
}

void rematPlanFree(rematPlan_t *p) {
    freeReservedMemory(p);
}

const rematProgram_t *rematPlanProgram(const rematPlan_t *p, rematMode_t mode) {
    return (mode == REMAT_MODE_EVAL) ? &p->eval : &p->train;
}
