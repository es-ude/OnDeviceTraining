#define SOURCE_FILE "REMAT_PLAN"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

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
 * ARENA placement that follows it at init. */
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
 * blockBytes, via planAdd/planMul) and peakLiveBytes (accumulated with
 * planAdd in peakLiveBytesOf). rematTrainStepCount, rematBackwardStep and
 * numRanges = numWires - 1 are plain size_t arithmetic on modelSize,
 * backwardTop, deepest and numWires -- all bounded well under SIZE_MAX by the
 * table's numWires < REMAT_NONE guard, so they cannot overflow. In PR1 the
 * checked exits above are themselves unreachable -- each wire has one range,
 * so the peak never exceeds the table's checked total of wire bytes -- and
 * have no dedicated death test (PR6's multi-range and SEQUENCE plans make
 * them reachable). */
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

static size_t peakLiveBytesOf(const rematProgram_t *p, const rematWireTable_t *t) {
    rematWalk_t walk = {0};
    size_t live = 0;
    size_t peak = 0;
    for (walk.step = 0; walk.step < p->numSteps; walk.step++) {
        for (size_t r; (r = rematWalkOpening(p, &walk)) != REMAT_NONE;) {
            live = planAdd(live, t->wires[p->ranges[r].wire].bytes, "peakLiveBytes");
        }
        if (live > peak) {
            peak = live;
        }
        for (size_t r; (r = rematWalkClosing(p, &walk)) != REMAT_NONE;) {
            live -= t->wires[p->ranges[r].wire].bytes;
        }
    }
    return peak;
}

static size_t roundUpTo(size_t x, size_t align) {
    return planAdd(x, align - 1u, "blockBytes") & ~(align - 1u);
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

/* Linear in the ranges on purpose: independent of the walk and of endOrder. */
static void requireCovered(const rematProgram_t *p, size_t s, uint16_t wire, const char *role) {
    if (wire == 0) {
        return; /* ACT 0 is borrowed: always present */
    }
    for (size_t r = 0; r < p->numRanges; r++) {
        const rematRange_t *range = &p->ranges[r];
        if (range->wire == wire && range->begin <= s && s <= range->end) {
            return;
        }
    }
    const rematStep_t *st = &p->steps[s];
    PRINT_ERROR("rematPlanBuild: step #%zu (kind %u, layer %u) violates grammar rule 4: it %s wire "
                "%u outside any open range",
                s, (unsigned)st->kind, (unsigned)st->layer, role, (unsigned)wire);
    exit(1);
}

/* Rule 4 for one step that rules 1-3 already admitted: rule 1 bounds a FORWARD's
 * layer below n, LOSS_* steps carry layer n, and rule 3 keeps a BACKWARD's layer
 * in [deepest, top], so every operand id below is a real wire. */
static void requireOperandsCovered(const rematProgram_t *p, const rematWireTable_t *t,
                                   layer_t **model, size_t s) {
    size_t n = t->modelSize;
    size_t l = p->steps[s].layer;
    switch (p->steps[s].kind) {
    case REMAT_STEP_FORWARD:
        requireCovered(p, s, rematActId(t, l), "reads");
        requireCovered(p, s, rematActId(t, l + 1u), "writes");
        break;
    case REMAT_STEP_LOSS_FORWARD:
        requireCovered(p, s, rematActId(t, n), "reads");
        break;
    case REMAT_STEP_LOSS_BACKWARD:
        requireCovered(p, s, rematActId(t, n), "reads");
        requireCovered(p, s, rematGradId(t, n), "writes");
        break;
    default: { /* REMAT_STEP_BACKWARD */
        uint16_t gradIn =
            ((ptrdiff_t)l == t->backwardTop) ? rematGradId(t, n) : rematGradId(t, l + 1u);
        requireCovered(p, s, gradIn, "reads");
        if (layerBackwardReadsInput(model[l])) {
            requireCovered(p, s, rematActId(t, l), "reads");
        }
        if (l > t->deepest) {
            requireCovered(p, s, rematGradId(t, l), "writes");
        }
        break;
    }
    }
}

void rematPlanValidateGrammar(const rematProgram_t *p, const rematWireTable_t *t, layer_t **model) {
    size_t n = t->modelSize;
    size_t nextForward = 0;
    ptrdiff_t nextBackward = t->backwardTop;
    bool lossForward = false;
    bool lossBackward = false;
    for (size_t s = 0; s < p->numSteps; s++) {
        const rematStep_t *st = &p->steps[s];
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
        default:
            PRINT_ERROR("rematPlanBuild: step #%zu (kind %u, layer %u) violates grammar: unknown "
                        "step kind",
                        s, (unsigned)st->kind, (unsigned)st->layer);
            exit(1);
        }
        requireOperandsCovered(p, t, model, s);
    }
    if (!lossForward) {
        grammarEndExit(p->numSteps, "rule 1: no LOSS_FORWARD");
    }
    if (lossBackward != t->hasBackward) {
        grammarEndExit(p->numSteps, "rule 2: LOSS_BACKWARD missing");
    }
    if (t->hasBackward && nextBackward >= (ptrdiff_t)t->deepest) {
        grammarEndExit(p->numSteps, "rule 3: BACKWARD set incomplete");
    }
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

bool rematPlanBuild(rematPlan_t **out, const rematWireTable_t *t, layer_t **model,
                    const rematPlanSpec_t *spec) {
    *out = NULL;
    rematPlanPolicy_t policy = (spec == NULL) ? REMAT_PLAN_STORE_ALL : spec->policy;
    if (policy != REMAT_PLAN_STORE_ALL && policy != REMAT_PLAN_LIVENESS) {
        PRINT_ERROR("rematPlanBuild: unknown policy %d", (int)policy);
        exit(1);
    }
    requireTheTablesModel(t, model);
    size_t numSteps = rematTrainStepCount(t);
    size_t numRanges = t->numWires - 1u;
    size_t stepsAt = roundUpTo(sizeof(rematPlan_t), _Alignof(rematStep_t));
    size_t rangesAt = roundUpTo(
        planAdd(stepsAt, planMul(numSteps, sizeof(rematStep_t), "blockBytes"), "blockBytes"),
        _Alignof(rematRange_t));
    size_t endOrderAt = roundUpTo(
        planAdd(rangesAt, planMul(numRanges, sizeof(rematRange_t), "blockBytes"), "blockBytes"),
        _Alignof(uint16_t));
    size_t blockBytes =
        planAdd(endOrderAt, planMul(numRanges, sizeof(uint16_t), "blockBytes"), "blockBytes");
    uint8_t *block = reserveMemory(blockBytes);
    if (block == NULL) {
        return false;
    }
    rematPlan_t *p = (rematPlan_t *)block;
    p->policy = policy;
    p->blockBytes = blockBytes;
    rematProgram_t *train = &p->train;
    train->numSteps = numSteps;
    train->steps = (rematStep_t *)(block + stepsAt);
    train->numRanges = numRanges;
    train->ranges = (rematRange_t *)(block + rangesAt);
    train->endOrder = (uint16_t *)(block + endOrderAt);
    rematFillTrainSteps(t, train->steps);
    rematFillTrainRanges(policy, t, model, numSteps, train->ranges);
    sortEndOrder(train);
    rematPlanValidateGrammar(train, t, model);
    train->peakLiveBytes = peakLiveBytesOf(train, t);
    *out = p;
    return true;
}

void rematPlanFree(rematPlan_t *p) {
    freeReservedMemory(p);
}
