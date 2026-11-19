#define SOURCE_FILE "REMAT_PLAN"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "CheckedSize.h"
#include "Common.h"
#include "Layer.h"
#include "LayerConfigAccess.h"
#include "LossFunction.h"
#include "Model.h"
#include "OpTable.h"
#include "RematPlan.h"
#include "RematPlanPolicy.h"
#include "RematScheduler.h"
#include "StorageApi.h"

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

/* Overflow-checked here are the plan-block size (the array offsets and
 * blockBytes, via planAdd/planMul), the step bound, the two
 * peaks, peakLiveBytes and the activation-only peak (accumulated with planAdd
 * in peakBytesUpTo, each under its own label), the activations' element
 * counts, and the two costs, forwardCost and recomputeCost.
 * The deriver's step, operand and range counts are plain size_t sums over
 * the table's nodes and the caller's steps -- all bounded well under SIZE_MAX
 * by the table's numWires < REMAT_NONE guard and the step bound, so they
 * cannot overflow. Only the two costs' exits are reachable (a layer's cost has no
 * bound the table checked) and have death tests. The others have none: a
 * program holds at most UINT16_MAX steps and no more ranges than steps, and
 * grammar rule 6 keeps the ranges of one wire apart, so a peak never exceeds
 * the table's checked total of wire bytes; an element count is a product the
 * table checked when it derived the same shape. */
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

/* One program's arrays inside the plan block, from `at` on: steps, operands,
 * nodeOps, ranges, endOrder, each at its own alignment; `end` is the first byte
 * after them. */
typedef struct programLayout {
    size_t stepsAt, operandsAt, nodeOpsAt, rangesAt, endOrderAt, end;
} programLayout_t;

static programLayout_t layoutProgram(size_t at, rematDeriveSize_t size, size_t numNodeOps) {
    programLayout_t l;
    l.stepsAt = roundUpTo(at, _Alignof(rematStep_t));
    l.operandsAt = roundUpTo(
        planAdd(l.stepsAt, planMul(size.numSteps, sizeof(rematStep_t), "blockBytes"), "blockBytes"),
        _Alignof(rematOperand_t));
    l.nodeOpsAt = roundUpTo(planAdd(l.operandsAt,
                                    planMul(size.numOperands, sizeof(rematOperand_t), "blockBytes"),
                                    "blockBytes"),
                            _Alignof(rematNodeOps_t));
    l.rangesAt =
        roundUpTo(planAdd(l.nodeOpsAt, planMul(numNodeOps, sizeof(rematNodeOps_t), "blockBytes"),
                          "blockBytes"),
                  _Alignof(rematRange_t));
    l.endOrderAt =
        roundUpTo(planAdd(l.rangesAt, planMul(size.numRanges, sizeof(rematRange_t), "blockBytes"),
                          "blockBytes"),
                  _Alignof(uint16_t));
    l.end = planAdd(l.endOrderAt, planMul(size.numRanges, sizeof(uint16_t), "blockBytes"),
                    "blockBytes");
    return l;
}

static void placeProgram(rematProgram_t *p, uint8_t *block, const programLayout_t *l,
                         rematDeriveSize_t size) {
    p->numSteps = size.numSteps;
    p->steps = (rematStep_t *)(block + l->stepsAt);
    p->numOperands = size.numOperands;
    p->operands = (rematOperand_t *)(block + l->operandsAt);
    p->nodeOps = (rematNodeOps_t *)(block + l->nodeOpsAt);
    p->numRanges = size.numRanges;
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

/* The grammar's own reading of the model, kept apart from the deriver's
 * inputs on purpose: the table's flags are what the deriver reads, so a
 * grammar that trusted them would share its errors. Chain form: buffer b
 * needs a gradient iff it lies above the first trainable node, and node j
 * runs its BACKWARD iff it lies at or above that node -- except, under
 * CrossEntropy, the sink Softmax, whose backward the combined p - y replaces.
 * The DAG form walks the wiring instead; grammarNeeds and grammarRuns are the
 * two places that widen. */
typedef struct grammarModel {
    const rematWireTable_t *t;
    const model_t *m;
    size_t n;
    size_t firstTrainable; /* n: nothing trains */
    bool crossEntropy;
} grammarModel_t;

static grammarModel_t grammarModelOf(const rematWireTable_t *t, const model_t *m) {
    grammarModel_t g = {.t = t,
                        .m = m,
                        .n = t->modelSize,
                        .firstTrainable = t->modelSize,
                        .crossEntropy = t->lossType == CROSS_ENTROPY};
    for (size_t j = 0; j < g.n; j++) {
        parameter_t *w = NULL;
        parameter_t *b = NULL;
        if (layerParameters(m->layers[j], &w, &b) && !layerIsFrozen(m->layers[j])) {
            g.firstTrainable = j;
            break;
        }
    }
    return g;
}

static bool grammarNeeds(const grammarModel_t *g, size_t b) {
    return b > g->firstTrainable;
}

static bool grammarRuns(const grammarModel_t *g, size_t j) {
    return j >= g->firstTrainable && !(g->crossEntropy && j == g->n - 1u);
}

/* The loss backward runs iff the buffer its gradient lands in needs one: ACT n
 * under MSE, the Softmax's input under CrossEntropy. */
static bool grammarHasBackward(const grammarModel_t *g) {
    return grammarNeeds(g, g->crossEntropy ? g->n - 1u : g->n);
}

/* Rule 3's cursor: the highest node below `from` that runs its BACKWARD, -1
 * for none. It only moves down, so a whole program costs O(n). */
static ptrdiff_t nextRunningBelow(const grammarModel_t *g, ptrdiff_t from) {
    for (ptrdiff_t j = from - 1; j >= 0; j--) {
        if (grammarRuns(g, (size_t)j)) {
            return j;
        }
    }
    return -1;
}

/* The GRAD wire holding buffer b's gradient: the table's gradIdOf[b], taken
 * only when its record is that gradient (kind GRAD, index b; under
 * CrossEntropy the Softmax input's gradient is the seed, index n), else the
 * record found by a scan. REMAT_NONE when no record holds it. */
static uint16_t gradWireOf(const grammarModel_t *g, size_t b) {
    const rematWireTable_t *t = g->t;
    size_t index = (g->crossEntropy && b == g->n - 1u) ? g->n : b;
    uint16_t id = t->gradIdOf[b];
    if (id < t->numWires && t->wires[id].kind == REMAT_WIRE_GRAD && t->wires[id].index == index) {
        return id;
    }
    for (size_t w = 0; w < t->numWires; w++) {
        if (t->wires[w].kind == REMAT_WIRE_GRAD && t->wires[w].index == index) {
            return (uint16_t)w;
        }
    }
    return REMAT_NONE;
}

#define GRAMMAR_MAX_OPERANDS (1u + 2u * MODEL_MAX_ARITY)

typedef struct expectedSlice {
    rematOperand_t ops[GRAMMAR_MAX_OPERANDS];
    uint8_t numReads;
    uint8_t numWrites;
} expectedSlice_t;

static void expectOperand(expectedSlice_t *e, uint16_t wire, opMode_t mode) {
    e->ops[e->numReads + e->numWrites] = (rematOperand_t){.wire = wire, .mode = (uint8_t)mode};
}

static void expectRead(expectedSlice_t *e, uint16_t wire, opMode_t mode) {
    expectOperand(e, wire, mode);
    e->numReads++;
}

static void expectWrite(expectedSlice_t *e, uint16_t wire) {
    expectOperand(e, wire, OP_MODE_WRITE);
    e->numWrites++;
}

/* The operands a step the kind rules admitted must carry, from the model
 * alone: those rules bound a FORWARD's and a REFORWARD's layer below n, a
 * LOSS_* step's at n and a BACKWARD's to a node that runs. ACT b is wire b. */
static expectedSlice_t expectedSlice(const grammarModel_t *g, const rematStep_t *st) {
    expectedSlice_t e = {.numReads = 0, .numWrites = 0};
    size_t n = g->n;
    size_t j = st->layer;
    uint16_t ids[MODEL_MAX_ARITY];
    switch (st->kind) {
    case REMAT_STEP_FORWARD:
    case REMAT_STEP_REFORWARD: {
        uint8_t k = modelNodeInputs(g->m, j, ids);
        for (uint8_t i = 0; i < k; i++) {
            expectRead(&e, ids[i], OP_MODE_READ);
        }
        expectWrite(&e, (uint16_t)(j + 1u));
        break;
    }
    case REMAT_STEP_LOSS_FORWARD:
        expectRead(&e, (uint16_t)n, OP_MODE_READ);
        break;
    case REMAT_STEP_LOSS_BACKWARD:
        expectRead(&e, (uint16_t)n, OP_MODE_READ);
        expectWrite(&e, gradWireOf(g, n));
        break;
    case REMAT_STEP_BACKWARD: {
        uint8_t k = modelNodeInputs(g->m, j, ids);
        opMode_t input = layerBackwardReadsInput(g->m->layers[j]) ? OP_MODE_READ : OP_MODE_META;
        expectRead(&e, gradWireOf(g, j + 1u), OP_MODE_READ);
        for (uint8_t i = 0; i < k; i++) {
            expectRead(&e, ids[i], input);
        }
        for (uint8_t i = 0; i < k; i++) {
            expectWrite(&e, grammarNeeds(g, ids[i]) ? gradWireOf(g, ids[i]) : REMAT_NONE);
        }
        break;
    }
    default:
        break; /* unreachable: the kind switch exits on an unknown kind first */
    }
    return e;
}

static const char *modeName(uint8_t mode) {
    switch (mode) {
    case OP_MODE_READ:
        return "READ";
    case OP_MODE_META:
        return "META";
    case OP_MODE_WRITE:
        return "WRITE";
    case OP_MODE_ACC:
        return "ACC";
    default:
        return "unknown mode";
    }
}

#define RULE8_PREFIX "rematPlanBuild: step #%zu (kind %u, layer %u) violates grammar rule 8: "

/* Rule 8: the step's slice lies inside the pool, starts where nodeOps puts
 * its node's slice of that kind, and holds exactly the operands the model
 * gives the step. Every later rule reads the slice, so it runs first. */
static void requireTheNodesSlice(const rematProgram_t *p, const grammarModel_t *g, size_t s) {
    const rematStep_t *st = &p->steps[s];
    unsigned kind = st->kind;
    unsigned layer = st->layer;
    if ((size_t)st->operands + st->numReads + st->numWrites > p->numOperands) {
        PRINT_ERROR(RULE8_PREFIX "its operand slice lies outside the pool", s, kind, layer);
        exit(1);
    }
    bool backward = st->kind == REMAT_STEP_LOSS_BACKWARD || st->kind == REMAT_STEP_BACKWARD;
    const rematNodeOps_t *node = &p->nodeOps[st->layer];
    if (st->operands != (backward ? node->backward : node->forward)) {
        PRINT_ERROR(RULE8_PREFIX "its operand slice is not its node's (nodeOps)", s, kind, layer);
        exit(1);
    }
    expectedSlice_t e = expectedSlice(g, st);
    if (st->numReads != e.numReads || st->numWrites != e.numWrites) {
        PRINT_ERROR(RULE8_PREFIX "%u reads and %u writes, expected %u and %u", s, kind, layer,
                    (unsigned)st->numReads, (unsigned)st->numWrites, (unsigned)e.numReads,
                    (unsigned)e.numWrites);
        exit(1);
    }
    const rematOperand_t *ops = &p->operands[st->operands];
    for (size_t i = 0; i < (size_t)e.numReads + e.numWrites; i++) {
        if (ops[i].wire != e.ops[i].wire || ops[i].mode != e.ops[i].mode) {
            PRINT_ERROR(RULE8_PREFIX "operand %zu is wire %u (%s), expected wire %u (%s)", s, kind,
                        layer, i, (unsigned)ops[i].wire, modeName(ops[i].mode),
                        (unsigned)e.ops[i].wire, modeName(e.ops[i].mode));
            exit(1);
        }
    }
}

/* Rule 4 (every READ operand inside a range of its wire, a member inside its
 * root's; META reads no bytes) and rule 6's write rule, over the slice rule 8
 * verified. A member's write lands in the range its root's write opened, so
 * rule 6 skips it. Its rule-4 check cannot fail first: the same step reads
 * wire - 1 of the same chain, which resolves to the same root and is checked
 * before it. */
static void requireOperandsCovered(const rematProgram_t *p, const uint8_t *inPlace, size_t s) {
    const rematStep_t *st = &p->steps[s];
    const rematOperand_t *ops = &p->operands[st->operands];
    for (size_t i = 0; i < st->numReads; i++) {
        if (ops[i].mode == OP_MODE_READ) {
            requireCovered(p, s, ops[i].wire, rematChainRoot(inPlace, ops[i].wire), "reads");
        }
    }
    for (size_t i = st->numReads; i < (size_t)st->numReads + st->numWrites; i++) {
        uint16_t wire = ops[i].wire;
        if (wire == REMAT_NONE) {
            continue;
        }
        uint16_t root = rematChainRoot(inPlace, wire);
        requireCovered(p, s, wire, root, "writes");
        if (root == wire && ops[i].mode == OP_MODE_WRITE) {
            requireNoEarlierRangeOpen(p, s, wire);
        }
    }
}

/* Whether a verified slice holds wire among its reads (READ) or its writes
 * (WRITE) in that mode. */
static bool sliceHolds(const rematProgram_t *p, const rematStep_t *st, uint16_t wire,
                       opMode_t mode) {
    const rematOperand_t *ops = &p->operands[st->operands];
    size_t from = (mode == OP_MODE_WRITE) ? st->numReads : 0u;
    size_t to = (mode == OP_MODE_WRITE) ? (size_t)st->numReads + st->numWrites : st->numReads;
    for (size_t i = from; i < to; i++) {
        if (ops[i].wire == wire && ops[i].mode == mode) {
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

/* Rule 6 over the ranges, once every step has passed: each range names a
 * wire of the table; no range names an in-place member (its root's range
 * holds it); each range lies inside the stream; the array is sorted by
 * begin, ties by wire id (the walk opens ranges in array order); each range
 * opens at a step whose write set holds its wire with WRITE; and a range a
 * REFORWARD opens holds a READ of that wire, so no recompute is dead. O(R)
 * plus the lengths of the REFORWARD ranges. */
static void requireRangesWellFormed(const rematProgram_t *p, const rematWireTable_t *t,
                                    const uint8_t *inPlace) {
    for (size_t r = 0; r < p->numRanges; r++) {
        const rematRange_t *range = &p->ranges[r];
        if (range->wire >= t->numWires) {
            rangeExit(r, range, "ranges name a wire of the table");
        }
        if (inPlace != NULL && inPlace[range->wire] != 0u) {
            rangeExit(r, range, "no range names an in-place member");
        }
        if (range->begin > range->end || range->end >= p->numSteps) {
            rangeExit(r, range, "ranges lie inside the stream");
        }
        const rematRange_t *prev = (r > 0) ? &p->ranges[r - 1u] : NULL;
        if (prev != NULL && (prev->begin > range->begin ||
                             (prev->begin == range->begin && prev->wire >= range->wire))) {
            rangeExit(r, range, "ranges sorted by begin, ties by wire id");
        }
        const rematStep_t *opener = &p->steps[range->begin];
        if (!sliceHolds(p, opener, range->wire, OP_MODE_WRITE)) {
            rangeExit(r, range, "a range opens where its wire is written");
        }
        if (opener->kind == REMAT_STEP_REFORWARD) {
            bool read = false;
            for (size_t s = range->begin + 1u; s <= range->end && !read; s++) {
                read = sliceHolds(p, &p->steps[s], range->wire, OP_MODE_READ);
            }
            if (!read) {
                rangeExit(r, range, "a REFORWARD's output is read inside its range");
            }
        }
    }
}

/* The stored marks, against the grammar's own reading of the model (the
 * generator decides them apart from it): a mark sits on the dx GRAD l of a
 * ReLU, Dropout or Flatten, so wire - 1 is a GRAD too, never ACT n; the
 * gradient BACKWARD(l) reads is wire - 1; and the two wires share one dtype,
 * FLOAT32 or SYM_INT32, and their bytes. BFP wires never alias. It runs
 * before the steps, so rematChainRoot only ever walks down a valid chain.
 * The gradient BACKWARD(l) reads is the wire whose record holds GRAD l + 1
 * (gradWireOf), not the table's raw gradIdOf lookup. */
static void requireMarksOnAliasPairs(const grammarModel_t *g, const uint8_t *inPlace) {
    const rematWireTable_t *t = g->t;
    const model_t *m = g->m;
    size_t n = t->modelSize;
    for (size_t w = 0; w < t->numWires; w++) {
        if (inPlace[w] == 0u) {
            continue;
        }
        const rematWire_t *out = &t->wires[w];
        if (out->kind != REMAT_WIRE_GRAD || out->index == n) {
            PRINT_ERROR("rematPlanBuild: wire %zu carries an in-place mark, but it is not a "
                        "layer's dx wire (GRAD l, l < n)",
                        w);
            exit(1);
        }
        size_t l = out->index;
        layerType_t type = m->layers[l]->type;
        if (type != RELU && type != DROPOUT && type != FLATTEN) {
            PRINT_ERROR("rematPlanBuild: wire %zu carries an in-place mark, but layer %zu is not a "
                        "ReLU, Dropout or Flatten",
                        w, l);
            exit(1);
        }
        uint16_t gradIn = gradWireOf(g, l + 1u);
        if (gradIn != w - 1u) {
            PRINT_ERROR("rematPlanBuild: wire %zu carries an in-place mark, but BACKWARD(%zu) "
                        "reads wire %u, not wire %zu",
                        w, l, (unsigned)gradIn, w - 1u);
            exit(1);
        }
        const rematWire_t *in = &t->wires[gradIn];
        if (out->dtype == BFP || in->dtype == BFP) {
            PRINT_ERROR("rematPlanBuild: wire %zu carries an in-place mark, but BFP wires never "
                        "alias",
                        w);
            exit(1);
        }
        if (out->dtype != in->dtype || out->bytes != in->bytes ||
            (out->dtype != FLOAT32 && out->dtype != SYM_INT32)) {
            PRINT_ERROR("rematPlanBuild: wire %zu carries an in-place mark, but wires %u and %zu "
                        "differ in dtype or bytes, or are neither FLOAT32 nor SYM_INT32",
                        w, (unsigned)gradIn, w);
            exit(1);
        }
    }
}

void rematPlanValidateGrammar(const rematProgram_t *p, const rematWireTable_t *t, const model_t *m,
                              rematMode_t mode, const uint8_t *inPlace) {
    grammarModel_t g = grammarModelOf(t, m);
    if (inPlace != NULL) {
        requireMarksOnAliasPairs(&g, inPlace);
    }
    bool hasBackward = grammarHasBackward(&g);
    size_t n = t->modelSize;
    size_t nextForward = 0;
    ptrdiff_t nextBackward = nextRunningBelow(&g, (ptrdiff_t)n); /* -1: no BACKWARD pending */
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
            if (!hasBackward || !lossForward || lossBackward || st->layer != n) {
                grammarExit(s, st,
                            "rule 2: LOSS_BACKWARD iff hasBackward, once, after LOSS_FORWARD");
            }
            lossBackward = true;
            break;
        case REMAT_STEP_BACKWARD:
            if (!lossBackward || nextBackward < 0 || (ptrdiff_t)st->layer != nextBackward) {
                grammarExit(s, st,
                            "rule 3: BACKWARD top..deepest, descending, after LOSS_BACKWARD");
            }
            nextBackward = nextRunningBelow(&g, nextBackward);
            break;
        case REMAT_STEP_REFORWARD:
            /* The checker's window, coded here on its own. hasBackward needs no
             * test: without it no node runs, so nothing is pending. */
            if (!lossForward || nextBackward < 0 ||
                (ptrdiff_t)st->layer >= (lossBackward ? nextBackward : (ptrdiff_t)n)) {
                grammarExit(s, st,
                            "rule 6: a REFORWARD comes after LOSS_FORWARD, below a pending "
                            "backward step");
            }
            if (layerForwardUpdatesState(m->layers[st->layer])) {
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
        requireTheNodesSlice(p, &g, s);
        requireOperandsCovered(p, inPlace, s);
    }
    if (!lossForward) {
        grammarEndExit(p->numSteps, "rule 1: no LOSS_FORWARD");
    }
    if (mode == REMAT_MODE_TRAIN && lossBackward != hasBackward) {
        grammarEndExit(p->numSteps, "rule 2: LOSS_BACKWARD missing");
    }
    if (mode == REMAT_MODE_TRAIN && hasBackward && nextBackward >= 0) {
        grammarEndExit(p->numSteps, "rule 3: BACKWARD set incomplete");
    }
    requireRangesWellFormed(p, t, inPlace);
}

/* The deriver reads the table and the grammar the model passed here: a model
 * other than the table's would surface as a grammar violation far from its
 * cause, so the key facts are compared first, by name, over the table's layer
 * count. */
static void requireTheTablesModel(const rematWireTable_t *t, const model_t *m) {
    for (size_t i = 0; i < t->modelSize; i++) {
        if ((uint8_t)m->layers[i]->type != t->layerType[i]) {
            PRINT_ERROR("rematPlanBuild: the model differs from the table's key at "
                        "'layerType[%zu]': table %u, model %u",
                        i, (unsigned)t->layerType[i], (unsigned)m->layers[i]->type);
            exit(1);
        }
        uint8_t frozen = layerIsFrozen(m->layers[i]) ? 1u : 0u;
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
 * loss backward runs but no BACKWARD step does has no REFORWARD window: a
 * dropped ACT n could never be recomputed before LOSS_BACKWARD reads it.
 * DEFENSIVE: unreachable today, since the loss gradient lands in a buffer
 * whose producer runs its BACKWARD whenever that buffer needs a gradient (ACT
 * n's under MSE, the Softmax input's under CrossEntropy). Kept so a future
 * loss that skips more than its sink cannot silently drop ACT n. */
static bool anyNodeRunsBackward(const rematWireTable_t *t) {
    for (size_t j = 0; j < t->modelSize; j++) {
        if (t->runsBackward[j]) {
            return true;
        }
    }
    return false;
}

static void requireARetainMaskWithAWindow(const rematPlanSpec_t *spec, const rematWireTable_t *t) {
    if (spec->retain == NULL) {
        PRINT_ERROR("rematPlanBuild: REMAT_PLAN_RETAIN_LIST needs a retain mask, got NULL");
        exit(1);
    }
    size_t n = t->modelSize;
    if (t->hasBackward && !anyNodeRunsBackward(t) && !spec->retain[n]) {
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

/* SEQUENCE's checks before the counting pass and the copy: the length checks
 * run before any record is read, so every later pass reads exactly sequenceLen
 * of them. The checked byte count of the copy is layoutProgram's, before
 * reserveMemory; it cannot overflow once the length is at most UINT16_MAX, so
 * it has no death test. */
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
    /* The plan derives every step's operand slice; a caller's record names
     * the step only. */
    for (size_t s = 0; s < spec->sequenceLen; s++) {
        const rematStep_t *st = &spec->sequence[s];
        if (st->numReads != 0u || st->numWrites != 0u || st->reserved != 0u || st->operands != 0u) {
            PRINT_ERROR("rematPlanBuild: sequence records carry no operands (step #%zu: "
                        "numReads %u, numWrites %u, reserved %u, operands %u)",
                        s, (unsigned)st->numReads, (unsigned)st->numWrites, (unsigned)st->reserved,
                        (unsigned)st->operands);
            exit(1);
        }
    }
}

/* A step's slice starts at a uint16_t offset, and UINT16_MAX itself is
 * REMAT_NONE, the empty-slice mark in nodeOps: a pool of more than UINT16_MAX
 * operands could start a real slice there. The pool does not depend on the
 * policy: five operands per node that runs its BACKWARD on a chain, so this
 * stops long chains of about 13107 nodes. */
static void requireAnOperandPoolWithinTheOffsets(size_t numOperands) {
    if (numOperands > (size_t)UINT16_MAX) {
        PRINT_ERROR("rematPlanBuild: the TRAIN program needs %zu operands, above UINT16_MAX "
                    "(operand offsets are uint16_t, and UINT16_MAX is REMAT_NONE)",
                    numOperands);
        exit(1);
    }
}

/* A SEQUENCE or RETAIN_LIST plan opens a range per write, up to one per step:
 * the cap bounds its quadratic plan work before anything is reserved.
 * STORE_ALL and LIVENESS open each wire's range once, at most numWires - 1,
 * and are not capped here. */
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
 * table's ACT shapes (its wire bytes do not give BFP element counts). Grammar rule 1 runs every
 * layer's FORWARD once, so forwardCost is the sum over the layers. A layer's REFORWARDs are counted
 * by a scan of the steps, only in a program that recomputes: a SEQUENCE or RETAIN_LIST plan, whose
 * range cap keeps n at most ODT_REMAT_MAX_RANGES (ACT 1..n each have a range; only GRAD wires share
 * one), so the scans read at most n * numSteps steps. */
static void recordRecomputeNumbers(rematPlan_t *p, const rematWireTable_t *t, const model_t *m) {
    const rematProgram_t *train = &p->train;
    size_t refwdSteps = 0;
    for (size_t s = 0; s < train->numSteps; s++) {
        refwdSteps += (train->steps[s].kind == REMAT_STEP_REFORWARD) ? 1u : 0u;
    }
    size_t forwardCost = 0;
    size_t recomputeCost = 0;
    for (size_t l = 0; l < t->modelSize; l++) {
        size_t cost = layerForwardCost(m->layers[l], activationElements(&t->actShapes[l]),
                                       activationElements(&t->actShapes[l + 1u]));
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
    }
    p->refwdSteps = refwdSteps;
    p->forwardCost = forwardCost;
    p->recomputeCost = recomputeCost;
}

bool rematPlanBuild(rematPlan_t **out, const rematWireTable_t *t, const model_t *m,
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
    requireTheTablesModel(t, m);
    if (policy == REMAT_PLAN_SEQUENCE) {
        requireASequenceWithinTheBound(spec, t->modelSize);
    } else if (policy == REMAT_PLAN_RETAIN_LIST) {
        requireARetainMaskWithAWindow(spec, t);
    }
    rematDeriveSize_t trainSize = rematDeriveTrainSize(t, spec);
    if (policy == REMAT_PLAN_RETAIN_LIST) {
        requireAProgramWithinTheBound(trainSize.numSteps, t->modelSize);
    }
    if (policy == REMAT_PLAN_SEQUENCE || policy == REMAT_PLAN_RETAIN_LIST) {
        requireRangesWithinTheCap(trainSize.numRanges);
    }
    requireAnOperandPoolWithinTheOffsets(trainSize.numOperands);
    rematDeriveSize_t evalSize = rematDeriveEvalSize(t);
    size_t numNodeOps = t->modelSize + 1u;
    programLayout_t trainAt = layoutProgram(sizeof(rematPlan_t), trainSize, numNodeOps);
    programLayout_t evalAt = layoutProgram(trainAt.end, evalSize, numNodeOps);
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
    placeProgram(train, block, &trainAt, trainSize);
    rematDeriveTrain(t, spec, train);
    rematScanRanges(t, train, policy, p->inPlace, train->ranges);
    sortEndOrder(train);
    rematPlanValidateGrammar(train, t, m, REMAT_MODE_TRAIN, p->inPlace);
    train->peakLiveBytes = peakLiveBytesOf(train, t);
    recordRecomputeNumbers(p, t, m);
    rematProgram_t *eval = &p->eval;
    placeProgram(eval, block, &evalAt, evalSize);
    rematDeriveEval(t, eval);
    rematScanRanges(t, eval, REMAT_PLAN_LIVENESS, NULL, eval->ranges);
    sortEndOrder(eval);
    rematPlanValidateGrammar(eval, t, m, REMAT_MODE_EVAL, NULL);
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
