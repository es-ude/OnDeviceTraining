#define SOURCE_FILE "REMAT_PLAN"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "Common.h"
#include "Layer.h"
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
 * ARENA placement that follows it at init (plan Assumption 25). */
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

/* D60: checked here are the plan-block size (stepsAt/rangesAt/endOrderAt/
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

bool rematPlanBuild(rematPlan_t **out, const rematWireTable_t *t, layer_t **model,
                    const rematPlanSpec_t *spec) {
    *out = NULL;
    rematPlanPolicy_t policy = (spec == NULL) ? REMAT_PLAN_STORE_ALL : spec->policy;
    if (policy != REMAT_PLAN_STORE_ALL && policy != REMAT_PLAN_LIVENESS) {
        PRINT_ERROR("rematPlanBuild: unknown policy %d", (int)policy);
        exit(1);
    }
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
    train->peakLiveBytes = peakLiveBytesOf(train, t);
    *out = p;
    return true;
}

void rematPlanFree(rematPlan_t *p) {
    freeReservedMemory(p);
}
