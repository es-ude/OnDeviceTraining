#define SOURCE_FILE "REMAT_DERIVE"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "Common.h"
#include "Layer.h"
#include "LayerConfigAccess.h"
#include "LossFunction.h"
#include "Model.h"
#include "OpTable.h"
#include "RematPlan.h"
#include "RematPlanPolicy.h"

bool rematHasBackward(size_t n, lossFuncType_t lt, const uint8_t *needsGrad) {
    return needsGrad[lt == CROSS_ENTROPY ? n - 1u : n] != 0;
}

/* The GRAD an earlier contribution to buffer b opened, or REMAT_NONE when slot (j, i) is b's
 * first contribution. Contributions arrive in walk order, so the first already written slot that
 * reads b from a node running backward is the opener. O(edges) per call, so O(edges^2) per walk:
 * fine for the walk's once-per-init-and-bind use. */
static uint16_t openGradOf(const model_t *m, const uint8_t *runsBackward, const uint16_t *dxWire,
                           size_t j, size_t i, uint16_t b) {
    for (size_t c = m->numLayers; c-- > j;) {
        if (!runsBackward[c]) {
            continue;
        }
        uint16_t ids[MODEL_MAX_ARITY];
        uint8_t k = modelNodeInputs(m, c, ids);
        size_t end = (c == j) ? i : k;
        for (size_t s = 0; s < end; s++) {
            if (ids[s] == b) {
                return dxWire[c * MODEL_MAX_ARITY + s];
            }
        }
    }
    return REMAT_NONE;
}

/* The seed GRAD n: under CrossEntropy with a BFP-declaring head
 * (crossEntropyHeadDeclaresBfp) it IS the last Softmax's dx wire (the combined gradient replaces
 * its backward), so its template is the Softmax's propLossQ, type-derived. Every other seed -- MSE,
 * and CrossEntropy on a FLOAT32 / SYM head, whose propLossQ stays unused -- inherits ACT n, the
 * model output it is the gradient of. A Flatten passes its dx through, so the GRAD or PART it
 * writes inherits the ACT it is the gradient of. Under CrossEntropy no contribution reaches the
 * seed: the Softmax input n - 1 has the sink as its only consumer. */
size_t rematDeriveGradWires(const model_t *m, lossFuncType_t lt, const uint8_t *needsGrad,
                            const uint8_t *runsBackward, rematWireFact_t *facts, uint16_t *dxWire,
                            uint8_t *dxMode) {
    size_t n = m->numLayers;
    size_t count = 0;
    if (rematHasBackward(n, lt, needsGrad)) {
        if (facts != NULL) {
            bool seedTakesPropLossQ =
                lt == CROSS_ENTROPY && crossEntropyHeadDeclaresBfp(m->layers[n - 1u]);
            facts[n + 1u] =
                (rematWireFact_t){.kind = REMAT_WIRE_GRAD,
                                  .index = (uint16_t)n,
                                  .inheritFrom = seedTakesPropLossQ ? REMAT_NONE : (uint16_t)n};
        }
        count++;
    }
    for (size_t j = n; j-- > 0;) {
        uint16_t ids[MODEL_MAX_ARITY];
        uint8_t k = modelNodeInputs(m, j, ids);
        layerType_t type = m->layers[j]->type;
        for (size_t i = 0; i < MODEL_MAX_ARITY; i++) {
            size_t slot = j * MODEL_MAX_ARITY + i;
            if (i >= k || !runsBackward[j]) {
                dxWire[slot] = REMAT_NONE;
                dxMode[slot] = OP_MODE_READ;
                continue;
            }
            uint16_t b = ids[i];
            if (!needsGrad[b]) {
                dxWire[slot] = REMAT_NONE;
                dxMode[slot] = OP_MODE_WRITE;
                continue;
            }
            uint16_t open = openGradOf(m, runsBackward, dxWire, j, i, b);
            if (open != REMAT_NONE && opFunctions[type].accumulatesDx) {
                dxWire[slot] = open;
                dxMode[slot] = OP_MODE_ACC;
                continue;
            }
            size_t id = n + 1u + count++;
            if (facts != NULL) {
                facts[id] = (rematWireFact_t){.kind = (open == REMAT_NONE) ? REMAT_WIRE_GRAD
                                                                           : REMAT_WIRE_PART,
                                              .index = b,
                                              .inheritFrom = (type == FLATTEN) ? b : REMAT_NONE};
            }
            dxWire[slot] = (uint16_t)id;
            dxMode[slot] = OP_MODE_WRITE;
        }
    }
    return count;
}

uint8_t rematNodeInputs(const rematWireTable_t *t, size_t j, uint16_t ids[MODEL_MAX_ARITY]) {
    if (t->isChain) {
        ids[0] = (uint16_t)j;
        return 1u;
    }
    uint8_t k = (uint8_t)(t->inputStart[j + 1u] - t->inputStart[j]);
    for (uint8_t i = 0; i < k; i++) {
        ids[i] = t->inputIds[t->inputStart[j] + i];
    }
    return k;
}

/* The pool writer's cursor; with ops == NULL it only counts. It also counts what the slices it
 * wrote hold: reads, writes, and the writes rematScanRanges opens a range for (members decided by
 * inPlaceDx), so the sizing pass and pointAtSlice read the slice shape from the one writer. */
typedef struct poolCursor {
    const rematWireTable_t *t;
    bool inPlaceDx;
    rematOperand_t *ops;
    rematNodeOps_t *nodeOps;
    size_t at;
    size_t numReads;
    size_t numWrites;
    size_t numOpens;
} poolCursor_t;

static poolCursor_t countingCursor(const rematWireTable_t *t, bool inPlaceDx) {
    return (poolCursor_t){.t = t, .inPlaceDx = inPlaceDx};
}

/* Field by field, so the padding byte of the zeroed plan block stays zero. */
static void put(poolCursor_t *c, uint16_t wire, opMode_t mode) {
    if (c->ops != NULL) {
        c->ops[c->at].wire = wire;
        c->ops[c->at].mode = (uint8_t)mode;
    }
    c->at++;
}

static void putRead(poolCursor_t *c, uint16_t wire, opMode_t mode) {
    put(c, wire, mode);
    c->numReads++;
}

static void putWrite(poolCursor_t *c, uint16_t wire, opMode_t mode) {
    put(c, wire, mode);
    c->numWrites++;
    bool member = c->inPlaceDx && wire != REMAT_NONE && rematInPlaceMember(c->t, wire);
    if (wire != REMAT_NONE && mode == OP_MODE_WRITE && !member) {
        c->numOpens++;
    }
}

/* Node j's FORWARD slice; j = n is the LOSS_FORWARD slice. */
static void writeForwardSlice(poolCursor_t *c, size_t j) {
    const rematWireTable_t *t = c->t;
    size_t n = t->modelSize;
    if (c->nodeOps != NULL) {
        c->nodeOps[j].forward = (uint16_t)c->at;
    }
    if (j == n) {
        putRead(c, (uint16_t)n, OP_MODE_READ);
        return;
    }
    uint16_t ids[MODEL_MAX_ARITY];
    uint8_t k = rematNodeInputs(t, j, ids);
    for (uint8_t i = 0; i < k; i++) {
        putRead(c, ids[i], OP_MODE_READ);
    }
    putWrite(c, (uint16_t)(j + 1u), OP_MODE_WRITE);
}

/* Node j's BACKWARD slice; j = n is the LOSS_BACKWARD slice. Only for a slice the program has:
 * a backward runs, and node j < n runs its BACKWARD. */
static void writeBackwardSlice(poolCursor_t *c, size_t j) {
    const rematWireTable_t *t = c->t;
    size_t n = t->modelSize;
    if (c->nodeOps != NULL) {
        c->nodeOps[j].backward = (uint16_t)c->at;
    }
    if (j == n) {
        putRead(c, (uint16_t)n, OP_MODE_READ);
        putWrite(c, t->gradIdOf[n], OP_MODE_WRITE);
        return;
    }
    uint16_t ids[MODEL_MAX_ARITY];
    uint8_t k = rematNodeInputs(t, j, ids);
    putRead(c, t->gradIdOf[j + 1u], OP_MODE_READ);
    for (uint8_t i = 0; i < k; i++) {
        putRead(c, ids[i], t->readsInput[j] ? OP_MODE_READ : OP_MODE_META);
    }
    for (uint8_t i = 0; i < k; i++) {
        size_t slot = j * MODEL_MAX_ARITY + i;
        putWrite(c, t->dxWire[slot], (opMode_t)t->dxMode[slot]);
    }
}

/* The canonical pool, in pool order: the FORWARD slice of every node, the LOSS_FORWARD slice;
 * then, for a TRAIN program with a backward, the LOSS_BACKWARD slice and the BACKWARD slice of
 * every runsBackward node, descending. Each slice appears once, so the cursor's counts are those
 * of the default TRAIN (or the EVAL) program. The offsets are cast to uint16_t only while writing,
 * after the build checked the count against UINT16_MAX. */
static void writePool(poolCursor_t *c, bool train) {
    const rematWireTable_t *t = c->t;
    size_t n = t->modelSize;
    if (c->nodeOps != NULL) {
        for (size_t j = 0; j <= n; j++) {
            c->nodeOps[j] = (rematNodeOps_t){REMAT_NONE, REMAT_NONE, REMAT_NONE};
        }
    }
    for (size_t j = 0; j <= n; j++) {
        writeForwardSlice(c, j);
    }
    if (!train || !t->hasBackward) {
        return;
    }
    writeBackwardSlice(c, n);
    for (size_t j = n; j-- > 0;) {
        if (t->runsBackward[j]) {
            writeBackwardSlice(c, j);
        }
    }
}

/* The slice a step points at: node *node's FORWARD slice (backward false) or BACKWARD slice. False
 * for every step the grammar rejects later (an unknown kind, a layer outside its kind's range, a
 * node without that slice), so nothing reads past the pool before the grammar exits. Entry n of
 * nodeOps holds the loss slices, so a FORWARD or BACKWARD must check its layer before it looks. */
static bool sliceOf(const rematWireTable_t *t, const rematStep_t *st, size_t *node,
                    bool *backward) {
    size_t n = t->modelSize;
    size_t l = st->layer;
    *node = l;
    switch (st->kind) {
    case REMAT_STEP_FORWARD:
    case REMAT_STEP_REFORWARD:
        *backward = false;
        return l < n;
    case REMAT_STEP_LOSS_FORWARD:
        *backward = false;
        return l == n;
    case REMAT_STEP_LOSS_BACKWARD:
        *backward = true;
        return l == n && t->hasBackward;
    case REMAT_STEP_BACKWARD:
        *backward = true;
        return l < n && t->hasBackward && t->runsBackward[l];
    default:
        return false;
    }
}

/* The counts of one step's slice, from a counting run of its writer; zero for a rejected step. */
static poolCursor_t sliceCounts(const rematWireTable_t *t, bool inPlaceDx, const rematStep_t *st) {
    poolCursor_t c = countingCursor(t, inPlaceDx);
    size_t node;
    bool backward;
    if (sliceOf(t, st, &node, &backward)) {
        if (backward) {
            writeBackwardSlice(&c, node);
        } else {
            writeForwardSlice(&c, node);
        }
    }
    return c;
}

/* Points a step at its node's slice; a rejected step gets the empty slice. */
static void pointAtSlice(const rematWireTable_t *t, const rematNodeOps_t *nodeOps,
                         rematStep_t *st) {
    size_t node;
    bool backward;
    if (!sliceOf(t, st, &node, &backward)) {
        st->numReads = 0;
        st->numWrites = 0;
        st->operands = 0;
        return;
    }
    poolCursor_t c = sliceCounts(t, false, st);
    st->numReads = (uint8_t)c.numReads;
    st->numWrites = (uint8_t)c.numWrites;
    st->operands = backward ? nodeOps[node].backward : nodeOps[node].forward;
}

static size_t trainStepCount(const rematWireTable_t *t) {
    size_t n = t->modelSize;
    if (!t->hasBackward) {
        return n + 1u;
    }
    size_t backward = 0;
    for (size_t j = 0; j < n; j++) {
        backward += t->runsBackward[j] ? 1u : 0u;
    }
    return n + 2u + backward;
}

rematDeriveSize_t rematDeriveTrainSize(const rematWireTable_t *t, const rematPlanSpec_t *spec) {
    poolCursor_t pool = countingCursor(t, spec->inPlaceDx);
    writePool(&pool, true);
    size_t numSteps;
    size_t numRanges;
    switch (spec->policy) {
    case REMAT_PLAN_SEQUENCE:
        numSteps = spec->sequenceLen;
        numRanges = 0;
        for (size_t s = 0; s < numSteps; s++) {
            numRanges += sliceCounts(t, spec->inPlaceDx, &spec->sequence[s]).numOpens;
        }
        break;
    case REMAT_PLAN_RETAIN_LIST:
        /* The default program plus REFORWARDs, each reopening the ACT its FORWARD slice writes. */
        numSteps = rematFillRetainSteps(t, spec->retain, NULL);
        numRanges = pool.numOpens + (numSteps - trainStepCount(t));
        break;
    default:
        numSteps = trainStepCount(t);
        numRanges = pool.numOpens;
        break;
    }
    return (rematDeriveSize_t){
        .numSteps = numSteps, .numOperands = pool.at, .numRanges = numRanges};
}

rematDeriveSize_t rematDeriveEvalSize(const rematWireTable_t *t) {
    poolCursor_t pool = countingCursor(t, false);
    writePool(&pool, false);
    return (rematDeriveSize_t){
        .numSteps = t->modelSize + 1u, .numOperands = pool.at, .numRanges = pool.numOpens};
}

void rematDeriveTrain(const rematWireTable_t *t, const rematPlanSpec_t *spec, rematProgram_t *p) {
    switch (spec->policy) {
    case REMAT_PLAN_SEQUENCE:
        memcpy(p->steps, spec->sequence, p->numSteps * sizeof(rematStep_t));
        break;
    case REMAT_PLAN_RETAIN_LIST:
        (void)rematFillRetainSteps(t, spec->retain, p->steps);
        break;
    default:
        rematFillTrainSteps(t, p->steps);
        break;
    }
    poolCursor_t pool = countingCursor(t, false);
    pool.ops = p->operands;
    pool.nodeOps = p->nodeOps;
    writePool(&pool, true);
    for (size_t s = 0; s < p->numSteps; s++) {
        pointAtSlice(t, p->nodeOps, &p->steps[s]);
    }
}

void rematDeriveEval(const rematWireTable_t *t, rematProgram_t *p) {
    size_t n = t->modelSize;
    poolCursor_t pool = countingCursor(t, false);
    pool.ops = p->operands;
    pool.nodeOps = p->nodeOps;
    writePool(&pool, false);
    for (size_t j = 0; j < n; j++) {
        p->steps[j] = (rematStep_t){.kind = REMAT_STEP_FORWARD, .layer = (uint16_t)j};
    }
    p->steps[n] = (rematStep_t){.kind = REMAT_STEP_LOSS_FORWARD, .layer = (uint16_t)n};
    for (size_t s = 0; s <= n; s++) {
        pointAtSlice(t, p->nodeOps, &p->steps[s]);
    }
}

/* Extends the newest range of wire, searching back from the last one opened. */
static void extendNewest(rematRange_t *ranges, size_t opened, uint16_t wire, size_t s) {
    for (size_t r = opened; r > 0; r--) {
        if (ranges[r - 1u].wire == wire) {
            ranges[r - 1u].end = (uint16_t)s;
            return;
        }
    }
}

void rematScanRanges(const rematWireTable_t *t, const rematProgram_t *p, rematPlanPolicy_t policy,
                     const uint8_t *inPlace, rematRange_t *ranges) {
    size_t opened = 0;
    for (size_t s = 0; s < p->numSteps; s++) {
        const rematStep_t *st = &p->steps[s];
        const rematOperand_t *ops = &p->operands[st->operands];
        for (size_t i = 0; i < st->numReads; i++) {
            if (ops[i].mode == OP_MODE_READ) {
                extendNewest(ranges, opened, rematChainRoot(inPlace, ops[i].wire), s);
            }
        }
        for (size_t i = st->numReads; i < (size_t)st->numReads + st->numWrites; i++) {
            uint16_t wire = ops[i].wire;
            if (wire == REMAT_NONE) {
                continue;
            }
            uint16_t root = rematChainRoot(inPlace, wire);
            if (root == wire && ops[i].mode == OP_MODE_WRITE) {
                ranges[opened++] =
                    (rematRange_t){.wire = wire, .begin = (uint16_t)s, .end = (uint16_t)s};
            } else {
                extendNewest(ranges, opened, root, s);
            }
        }
    }
    if (policy == REMAT_PLAN_STORE_ALL) {
        for (size_t r = 0; r < opened; r++) {
            if (t->wires[ranges[r].wire].kind == REMAT_WIRE_ACT) {
                ranges[r].end = (uint16_t)(p->numSteps - 1u);
            }
        }
    }
}
