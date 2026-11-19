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

static void setSlot(uint16_t *dxWire, uint8_t *dxMode, size_t slot, uint16_t wire, opMode_t mode) {
    if (dxWire != NULL) {
        dxWire[slot] = wire;
        dxMode[slot] = (uint8_t)mode;
    }
}

/* The seed GRAD n: under CrossEntropy with a BFP-declaring head
 * (crossEntropyHeadDeclaresBfp) it IS the last Softmax's dx wire (the combined gradient replaces
 * its backward), so its template is the Softmax's propLossQ, type-derived. Every other seed -- MSE,
 * and CrossEntropy on a FLOAT32 / SYM head, whose propLossQ stays unused -- inherits ACT n, the
 * model output it is the gradient of. A Flatten passes its dx through, so its GRAD inherits the
 * ACT it is the gradient of. */
size_t rematDeriveGradWires(const model_t *m, lossFuncType_t lt, const uint8_t *needsGrad,
                            const uint8_t *runsBackward, rematWireFact_t *facts, uint16_t *dxWire,
                            uint8_t *dxMode) {
    size_t n = m->numLayers;
    size_t count = 0;
    bool hasBackward = needsGrad[lt == CROSS_ENTROPY ? n - 1u : n] != 0;
    if (hasBackward) {
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
        for (size_t i = 0; i < MODEL_MAX_ARITY; i++) {
            size_t slot = j * MODEL_MAX_ARITY + i;
            if (i >= k || !runsBackward[j]) {
                setSlot(dxWire, dxMode, slot, REMAT_NONE, OP_MODE_READ);
                continue;
            }
            uint16_t b = ids[i];
            if (!needsGrad[b]) {
                setSlot(dxWire, dxMode, slot, REMAT_NONE, OP_MODE_WRITE);
                continue;
            }
            size_t id = n + 1u + count++;
            if (facts != NULL) {
                bool passThrough = m->layers[j]->type == FLATTEN;
                facts[id] = (rematWireFact_t){.kind = REMAT_WIRE_GRAD,
                                              .index = b,
                                              .inheritFrom = passThrough ? b : REMAT_NONE};
            }
            setSlot(dxWire, dxMode, slot, (uint16_t)id, OP_MODE_WRITE);
        }
    }
    return count;
}

/* Node j's input buffers, which are its input ACT wires. The table holds no wiring yet, so every
 * node of a model it accepts is a chain node that reads buffer j. */
static uint8_t nodeInputs(const rematWireTable_t *t, size_t j, uint16_t ids[MODEL_MAX_ARITY]) {
    (void)t;
    ids[0] = (uint16_t)j;
    return 1u;
}

/* The pool writer's cursor; with ops == NULL it only counts. */
typedef struct poolCursor {
    rematOperand_t *ops;
    rematNodeOps_t *nodeOps;
    size_t at;
} poolCursor_t;

/* Field by field, so the padding byte of the zeroed plan block stays zero. */
static void put(poolCursor_t *c, uint16_t wire, opMode_t mode) {
    if (c->ops != NULL) {
        c->ops[c->at].wire = wire;
        c->ops[c->at].mode = (uint8_t)mode;
    }
    c->at++;
}

static void openForward(poolCursor_t *c, size_t j) {
    if (c->nodeOps != NULL) {
        c->nodeOps[j].forward = (uint16_t)c->at;
    }
}

static void openBackward(poolCursor_t *c, size_t j) {
    if (c->nodeOps != NULL) {
        c->nodeOps[j].backward = (uint16_t)c->at;
    }
}

/* The canonical pool, in pool order: the FORWARD slice of every node, the LOSS_FORWARD slice;
 * then, for a TRAIN program with a backward, the LOSS_BACKWARD slice and the BACKWARD slice of
 * every runsBackward node, descending. Returns the operand count. The offsets are cast to
 * uint16_t only while writing, after the build checked the count against UINT16_MAX + 1. */
static size_t writePool(const rematWireTable_t *t, bool train, rematOperand_t *ops,
                        rematNodeOps_t *nodeOps) {
    size_t n = t->modelSize;
    poolCursor_t c = {.ops = ops, .nodeOps = nodeOps, .at = 0};
    if (nodeOps != NULL) {
        for (size_t j = 0; j <= n; j++) {
            nodeOps[j] = (rematNodeOps_t){REMAT_NONE, REMAT_NONE, REMAT_NONE};
        }
    }
    uint16_t ids[MODEL_MAX_ARITY];
    for (size_t j = 0; j < n; j++) {
        uint8_t k = nodeInputs(t, j, ids);
        openForward(&c, j);
        for (uint8_t i = 0; i < k; i++) {
            put(&c, ids[i], OP_MODE_READ);
        }
        put(&c, (uint16_t)(j + 1u), OP_MODE_WRITE);
    }
    openForward(&c, n);
    put(&c, (uint16_t)n, OP_MODE_READ);
    if (!train || !t->hasBackward) {
        return c.at;
    }
    openBackward(&c, n);
    put(&c, (uint16_t)n, OP_MODE_READ);
    put(&c, t->gradIdOf[n], OP_MODE_WRITE);
    for (size_t j = n; j-- > 0;) {
        if (!t->runsBackward[j]) {
            continue;
        }
        uint8_t k = nodeInputs(t, j, ids);
        openBackward(&c, j);
        put(&c, t->gradIdOf[j + 1u], OP_MODE_READ);
        for (uint8_t i = 0; i < k; i++) {
            put(&c, ids[i], t->readsInput[j] ? OP_MODE_READ : OP_MODE_META);
        }
        for (uint8_t i = 0; i < k; i++) {
            size_t slot = j * MODEL_MAX_ARITY + i;
            put(&c, t->dxWire[slot], (opMode_t)t->dxMode[slot]);
        }
    }
    return c.at;
}

/* Points a step at its node's slice by kind. Every guard here is for a step the grammar rejects
 * later (an unknown kind, a layer outside its kind's range, a node without that slice): it gets
 * the empty slice, so nothing reads past the pool before the grammar exits. Entry n of nodeOps
 * holds the loss slices, so a FORWARD or BACKWARD must check its layer before it looks. */
static void pointAtSlice(const rematWireTable_t *t, const rematNodeOps_t *nodeOps,
                         rematStep_t *st) {
    size_t n = t->modelSize;
    size_t l = st->layer;
    uint16_t ids[MODEL_MAX_ARITY];
    uint16_t at = REMAT_NONE;
    uint8_t numReads = 0;
    uint8_t numWrites = 0;
    switch (st->kind) {
    case REMAT_STEP_FORWARD:
    case REMAT_STEP_REFORWARD:
        if (l < n) {
            at = nodeOps[l].forward;
            numReads = nodeInputs(t, l, ids);
            numWrites = 1u;
        }
        break;
    case REMAT_STEP_LOSS_FORWARD:
        if (l == n) {
            at = nodeOps[n].forward;
            numReads = 1u;
        }
        break;
    case REMAT_STEP_LOSS_BACKWARD:
        if (l == n) {
            at = nodeOps[n].backward;
            numReads = 1u;
            numWrites = 1u;
        }
        break;
    case REMAT_STEP_BACKWARD:
        if (l < n) {
            uint8_t k = nodeInputs(t, l, ids);
            at = nodeOps[l].backward;
            numReads = (uint8_t)(1u + k);
            numWrites = k;
        }
        break;
    default:
        break;
    }
    if (at == REMAT_NONE) {
        at = 0;
        numReads = 0;
        numWrites = 0;
    }
    st->numReads = numReads;
    st->numWrites = numWrites;
    st->operands = at;
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
    size_t numSteps;
    switch (spec->policy) {
    case REMAT_PLAN_SEQUENCE:
        numSteps = spec->sequenceLen;
        break;
    case REMAT_PLAN_RETAIN_LIST:
        numSteps = rematFillRetainSteps(t, spec->retain, spec->inPlaceDx, NULL, NULL);
        break;
    default:
        numSteps = trainStepCount(t);
        break;
    }
    return (rematDeriveSize_t){.numSteps = numSteps, .numOperands = writePool(t, true, NULL, NULL)};
}

rematDeriveSize_t rematDeriveEvalSize(const rematWireTable_t *t) {
    return (rematDeriveSize_t){.numSteps = t->modelSize + 1u,
                               .numOperands = writePool(t, false, NULL, NULL)};
}

void rematDeriveTrain(const rematWireTable_t *t, const rematPlanSpec_t *spec, rematProgram_t *p) {
    switch (spec->policy) {
    case REMAT_PLAN_SEQUENCE:
        memcpy(p->steps, spec->sequence, p->numSteps * sizeof(rematStep_t));
        break;
    case REMAT_PLAN_RETAIN_LIST:
        (void)rematFillRetainSteps(t, spec->retain, spec->inPlaceDx, p->steps, NULL);
        break;
    default:
        rematFillTrainSteps(t, p->steps);
        break;
    }
    (void)writePool(t, true, p->operands, p->nodeOps);
    for (size_t s = 0; s < p->numSteps; s++) {
        pointAtSlice(t, p->nodeOps, &p->steps[s]);
    }
}

void rematDeriveEval(const rematWireTable_t *t, rematProgram_t *p) {
    size_t n = t->modelSize;
    (void)writePool(t, false, p->operands, p->nodeOps);
    for (size_t j = 0; j < n; j++) {
        p->steps[j] = (rematStep_t){.kind = REMAT_STEP_FORWARD, .layer = (uint16_t)j};
    }
    p->steps[n] = (rematStep_t){.kind = REMAT_STEP_LOSS_FORWARD, .layer = (uint16_t)n};
    for (size_t s = 0; s <= n; s++) {
        pointAtSlice(t, p->nodeOps, &p->steps[s]);
    }
}
