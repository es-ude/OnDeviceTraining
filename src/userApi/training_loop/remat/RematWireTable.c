#define SOURCE_FILE "REMAT_WIRE_TABLE"

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
#include "Quantization.h"
#include "RematPlan.h"
#include "RematPlanPolicy.h"
#include "StorageApi.h"
#include "Tensor.h"

const char *rematWireKindName(uint8_t kind) {
    return kind == REMAT_WIRE_ACT ? "ACT" : kind == REMAT_WIRE_GRAD ? "GRAD" : "PART";
}

/* Every size product and sum of the table goes through these; f names the wire in an overflow exit
 * (NULL: the table's own arrays). */
static void exitSizeOverflow(const rematWireFact_t *f, const char *quantity) {
    if (f == NULL) {
        /* Reachable only at the pre-guard "numWires = n + 1" sum in
         * rematWireTableInit (and its hasBackward extension), before numWires
         * is checked against REMAT_NONE -- for an absurd n approaching
         * SIZE_MAX. Every other NULL call runs after that guard, on values it
         * already bounds. Kept so a NULL wire never reaches the wire-naming
         * format below. */
        PRINT_ERROR("remat: size overflow computing %s of the table arrays", quantity);
    } else {
        PRINT_ERROR("remat: size overflow computing %s of wire %s %u", quantity,
                    rematWireKindName(f->kind), (unsigned)f->index);
    }
    exit(1);
}

static size_t mulSize(size_t a, size_t b, const rematWireFact_t *f, const char *quantity) {
    size_t out;
    if (!checkedMulSize(a, b, &out)) {
        exitSizeOverflow(f, quantity);
    }
    return out;
}

static size_t addSize(size_t a, size_t b, const rematWireFact_t *f, const char *quantity) {
    size_t out;
    if (!checkedAddSize(a, b, &out)) {
        exitSizeOverflow(f, quantity);
    }
    return out;
}

static size_t roundUpSize(size_t x, size_t align, const rematWireFact_t *f, const char *quantity) {
    return addSize(x, align - 1u, f, quantity) & ~(align - 1u);
}

static size_t elementsOf(const shape_t *shape, const rematWireFact_t *f) {
    size_t count = 1;
    for (size_t d = 0; d < shape->numberOfDimensions; d++) {
        count = mulSize(count, shape->dimensions[d], f, "elements");
    }
    return count;
}

/* calcNumberOfBytesForData (Tensor.c:94-116), checked: every dtype a borrowed
 * ACT 0 may have; slab wires are FLOAT32, SYM_INT32 or BFP (init enforces it). */
static size_t wireBytes(const quantization_t *q, size_t elements, const rematWireFact_t *f) {
    size_t bits;
    switch (q->type) {
    case FLOAT32:
    case INT32:
    case SYM_INT32:
        return mulSize(elements, sizeof(int32_t), f, "bytes");
    case SYM:
        bits = ((const symQConfig_t *)q->qConfig)->qBits;
        break;
    case ASYM:
        bits = ((const asymQConfig_t *)q->qConfig)->qBits;
        break;
    case BOOL:
        bits = 1u;
        break;
    case BFP:
        bits = ((const bfpQConfig_t *)q->qConfig)->mantissaBits;
        break;
    default:
        PRINT_ERROR("remat: wire %s %u has unknown qtype %d", rematWireKindName(f->kind),
                    (unsigned)f->index, (int)q->type);
        exit(1);
    }
    return addSize(mulSize(bits, elements, f, "bytes"), 7u, f, "bytes") / 8u;
}

rematBfpGroups_t rematBfpWireGrouping(const bfpQConfig_t *tmpl, size_t elements, uint8_t kind,
                                      size_t index) {
    if (tmpl->groupSize == 0 || tmpl->groupSize == elements) {
        return (rematBfpGroups_t){.numGroups = 1, .groupSize = 0};
    }
    if (elements % tmpl->groupSize != 0) {
        PRINT_ERROR("remat: BFP groupSize %zu does not divide the %zu elements of wire %s %zu -- "
                    "pick a divisor or a per-tensor {1,0} template",
                    tmpl->groupSize, elements, rematWireKindName(kind), index);
        exit(1);
    }
    return (rematBfpGroups_t){.numGroups = elements / tmpl->groupSize,
                              .groupSize = tmpl->groupSize};
}

/* Production order: with ACT j at id j and GRAD/PART wires in the order BACKWARD writes them
 * (rematDeriveGradWires), every range begins in wire-id order. */
static void numberWires(rematWireFact_t *facts, const model_t *m, lossFuncType_t lt,
                        const uint8_t *needsGrad, const uint8_t *runsBackward, uint16_t *dxWire,
                        uint8_t *dxMode) {
    size_t n = m->numLayers;
    for (size_t j = 0; j <= n; j++) {
        facts[j] = (rematWireFact_t){
            .kind = REMAT_WIRE_ACT, .index = (uint16_t)j, .inheritFrom = REMAT_NONE};
    }
    (void)rematDeriveGradWires(m, lt, needsGrad, runsBackward, facts, dxWire, dxMode);
}

static void setWireFacts(rematWireFact_t *f, const quantization_t *tmpl, uint8_t rank,
                         size_t elements) {
    f->tmpl = tmpl;
    f->rank = rank;
    f->elements = elements;
    f->dtype = (uint8_t)tmpl->type;
    f->bytes = wireBytes(tmpl, elements, f);
    f->numGroups = 0;
    bool slabWire = !(f->kind == REMAT_WIRE_ACT && f->index == 0);
    if (slabWire && tmpl->type == BFP) {
        f->numGroups = rematBfpWireGrouping(tmpl->qConfig, elements, f->kind, f->index).numGroups;
    }
}

/* A gradient wire (GRAD or PART) takes the shape of the ACT it is the gradient of, and either
 * inherits that ACT's template or takes its writer's backwardWireQ. */
static void setGradWireFacts(rematWireFact_t *facts, size_t id, layer_t *writer) {
    rematWireFact_t *f = &facts[id];
    const rematWireFact_t *act = &facts[f->index];
    const quantization_t *tmpl =
        (f->inheritFrom != REMAT_NONE) ? facts[f->inheritFrom].tmpl : backwardWireQ(writer);
    setWireFacts(f, tmpl, act->rank, act->elements);
}

/* ACT j takes layerOutputQ(model[j-1]), or for a Flatten the template of its input ACT (for a
 * Flatten on buffer 0 the caller's live input config): config fields only, never a scale or
 * exponents. The seed's writer is the last node (its template matters only for a BFP-declaring
 * CE head: the last Softmax's dx wire); every other gradient wire's writer is the node whose dx
 * slot WRITEs it, so dxWire/dxMode must be numbered first. shapes[0..n] is modelShapePass's output
 * for this input; ACT j's rank and elements come from shapes[j]. */
static void deriveFacts(rematWireFact_t *facts, size_t numWires, const model_t *m,
                        const shape_t *shapes, const tensor_t *input, const uint16_t *dxWire,
                        const uint8_t *dxMode) {
    size_t n = m->numLayers;
    setWireFacts(&facts[0], input->quantization, (uint8_t)shapes[0].numberOfDimensions,
                 elementsOf(&shapes[0], &facts[0]));
    for (size_t j = 1; j <= n; j++) {
        layer_t *layer = m->layers[j - 1];
        const quantization_t *tmpl;
        if (layer->type == FLATTEN) {
            uint16_t ids[MODEL_MAX_ARITY];
            (void)modelNodeInputs(m, j - 1, ids);
            tmpl = facts[ids[0]].tmpl;
        } else {
            tmpl = layerOutputQ(layer);
        }
        setWireFacts(&facts[j], tmpl, (uint8_t)shapes[j].numberOfDimensions,
                     elementsOf(&shapes[j], &facts[j]));
    }
    if (numWires == n + 1u) {
        return;
    }
    setGradWireFacts(facts, n + 1u, m->layers[n - 1u]);
    for (size_t j = n; j-- > 0;) {
        for (size_t i = 0; i < MODEL_MAX_ARITY; i++) {
            size_t slot = j * MODEL_MAX_ARITY + i;
            if (dxWire[slot] != REMAT_NONE && dxMode[slot] == OP_MODE_WRITE) {
                setGradWireFacts(facts, dxWire[slot], m->layers[j]);
            }
        }
    }
}

static bool isMerge(const model_t *m, size_t j) {
    return opFunctions[m->layers[j]->type].arity > 1u;
}

static void requireFloat32MergeWire(const rematWireFact_t *f) {
    if (f->dtype != FLOAT32) {
        PRINT_ERROR("rematWireTableInit: non-FLOAT32 merge or accumulate: wire %s %u has dtype %u",
                    rematWireKindName(f->kind), (unsigned)f->index, (unsigned)f->dtype);
        exit(1);
    }
}

/* Merge numerics are FLOAT32 only (addAccumulate exits on any other dtype at run time), so a
 * non-chain model is refused at init, before any block holds its wires, when a merge-touched wire
 * is not FLOAT32: every ACT an ADD reads (addLayerInit already makes its output FLOAT32), every
 * GRAD an ADD's backward reads as dy, every GRAD something accumulates into (an ADD's ACC slot or
 * a SUM step) and every PART. Needs the facts and the dx slots numbered. Never called on a chain,
 * which has no merge, so a chain's SYM_INT32 / BFP wires stay admitted. */
static void requireFloat32MergeWires(const rematWireFact_t *facts, size_t numWires,
                                     const model_t *m, lossFuncType_t lt, const uint16_t *dxWire,
                                     const uint8_t *dxMode) {
    size_t n = m->numLayers;
    for (size_t j = 0; j < n; j++) {
        if (!isMerge(m, j)) {
            continue;
        }
        uint16_t ids[MODEL_MAX_ARITY];
        uint8_t k = modelNodeInputs(m, j, ids);
        for (uint8_t i = 0; i < k; i++) {
            requireFloat32MergeWire(&facts[ids[i]]);
        }
    }
    for (size_t id = n + 1u; id < numWires; id++) {
        const rematWireFact_t *f = &facts[id];
        if (f->kind == REMAT_WIRE_PART) {
            requireFloat32MergeWire(f);
            /* Its SUM target: the GRAD an earlier contribution opened, so numbered before it. */
            size_t open = id - 1u;
            while (facts[open].kind != REMAT_WIRE_GRAD || facts[open].index != f->index) {
                open--;
            }
            requireFloat32MergeWire(&facts[open]);
            continue;
        }
        /* GRAD b is node b - 1's dy; under CROSS_ENTROPY the seed GRAD n is the Softmax input's
         * gradient, so node n - 2's dy. */
        size_t b = (lt == CROSS_ENTROPY && f->index == n) ? n - 1u : f->index;
        if (b > 0u && isMerge(m, b - 1u)) {
            requireFloat32MergeWire(f);
        }
    }
    for (size_t slot = 0; slot < n * MODEL_MAX_ARITY; slot++) {
        if (dxMode[slot] == OP_MODE_ACC) {
            requireFloat32MergeWire(&facts[dxWire[slot]]);
        }
    }
}

/* Once per init, on the templates, after gradIdOf is filled: a scale-transparent layer copies the
 * codes of the wire it reads into the wire it writes. Forward, node j reads ACT in_0(j) and writes
 * ACT j+1. Backward, a running node reads its output's gradient GRAD gradIdOf[j+1] (the seed under
 * CROSS_ENTROPY too) and writes its slot-0 dx wire. Reads the wiring from the table's copy, so
 * that copy must be filled first. */
static void requireTransparentCopyWidths(const rematWireFact_t *facts, const model_t *m,
                                         const rematWireTable_t *t) {
    size_t n = m->numLayers;
    for (size_t j = 0; j < n; j++) {
        uint16_t ids[MODEL_MAX_ARITY];
        (void)rematNodeInputs(t, j, ids);
        requireTransparentCopyWidth(m->layers[j], j, "forward", facts[ids[0]].tmpl,
                                    facts[j + 1u].tmpl);
    }
    for (size_t j = n; j-- > 0;) { /* BACKWARD order: the first violation it would meet exits */
        uint16_t dx = t->dxWire[j * MODEL_MAX_ARITY];
        if (!t->runsBackward[j] || dx == REMAT_NONE) {
            continue;
        }
        requireTransparentCopyWidth(m->layers[j], j, "backward", facts[t->gradIdOf[j + 1u]].tmpl,
                                    facts[dx].tmpl);
    }
}

static rematWire_t recordOf(const rematWireFact_t *f, tensor_t *hdr) {
    return (rematWire_t){.kind = f->kind,
                         .dtype = f->dtype,
                         .rank = f->rank,
                         .borrowed = (f->kind == REMAT_WIRE_ACT && f->index == 0) ? 1u : 0u,
                         .index = f->index,
                         .inheritFrom = f->inheritFrom,
                         .bytes = f->bytes,
                         .expCapacity = f->numGroups,
                         .hdr = hdr};
}

/* One cursor rule for the sizing and the placing pass: with base
 * == NULL only the cursor advances; with the reserved block the offsets become
 * pointers. Both passes run the same code, so they cannot disagree. */
typedef struct slabLayout {
    uint8_t *base;
    size_t cursor;
} slabLayout_t;

static void *slabPlace(slabLayout_t *layout, size_t align, size_t count, size_t size,
                       const rematWireFact_t *f) {
    size_t start = roundUpSize(layout->cursor, align, f, "slabBytes");
    layout->cursor = addSize(start, mulSize(count, size, f, "slabBytes"), f, "slabBytes");
    return layout->base == NULL ? NULL : layout->base + start;
}

#define SLAB_PLACE(layout, T, count, f)                                                            \
    ((T *)slabPlace((layout), _Alignof(T), (count), sizeof(T), (f)))

/* count shapes, then their dims and their orders, maxRank entries each: the
 * storage modelShapePass writes into. Init sizes and places its transient
 * derivation shapes and the table's actShapes with this one rule. */
static shape_t *placeShapes(slabLayout_t *layout, size_t count, size_t maxRank) {
    shape_t *shapes = SLAB_PLACE(layout, shape_t, count, NULL);
    size_t entries = mulSize(count, maxRank, NULL, "shape storage");
    size_t *dims = SLAB_PLACE(layout, size_t, entries, NULL);
    size_t *order = SLAB_PLACE(layout, size_t, entries, NULL);
    if (shapes != NULL) {
        for (size_t j = 0; j < count; j++) {
            shapes[j] = (shape_t){.dimensions = dims + j * maxRank,
                                  .orderOfDimensions = order + j * maxRank};
        }
    }
    return shapes;
}

/* isChain: no wiring copy, zero bytes. */
static rematWireTable_t *layoutTable(slabLayout_t *layout, const rematWireFact_t *facts,
                                     size_t numWires, size_t n, bool isChain, size_t numEdges,
                                     size_t inputRank, size_t maxRank) {
    rematWireTable_t *t = SLAB_PLACE(layout, rematWireTable_t, 1u, NULL);
    rematWire_t *wires = SLAB_PLACE(layout, rematWire_t, numWires, NULL);
    uint16_t *gradIdOf = SLAB_PLACE(layout, uint16_t, n + 1u, NULL);
    uint8_t *layerType = SLAB_PLACE(layout, uint8_t, n, NULL);
    uint8_t *frozen = SLAB_PLACE(layout, uint8_t, n, NULL);
    size_t slots = mulSize(n, MODEL_MAX_ARITY, NULL, "dx slots");
    uint8_t *runsBackward = SLAB_PLACE(layout, uint8_t, n, NULL);
    uint8_t *needsGrad = SLAB_PLACE(layout, uint8_t, n + 1u, NULL);
    uint8_t *readsInput = SLAB_PLACE(layout, uint8_t, n, NULL);
    uint16_t *dxWire = SLAB_PLACE(layout, uint16_t, slots, NULL);
    uint8_t *dxMode = SLAB_PLACE(layout, uint8_t, slots, NULL);
    uint16_t *inputStart = NULL;
    uint16_t *inputIds = NULL;
    if (!isChain) {
        inputStart = SLAB_PLACE(layout, uint16_t, n + 1u, NULL);
        inputIds = SLAB_PLACE(layout, uint16_t, numEdges, NULL);
    }
    size_t *inputDims = SLAB_PLACE(layout, size_t, inputRank, NULL);
    size_t *inputOrder = SLAB_PLACE(layout, size_t, inputRank, NULL);
    /* The ACT shapes and the bind's derivation scratch: after the fixed-size key,
     * before the headers, so the exponent arrays stay the block's tail. The
     * cursor rule aligns them like every other object. */
    shape_t *actShapes = placeShapes(layout, n + 1u, maxRank);
    rematWireFact_t *bindScratch = SLAB_PLACE(layout, rematWireFact_t, numWires, NULL);
    if (t != NULL) {
        t->wires = wires;
        t->gradIdOf = gradIdOf;
        t->layerType = layerType;
        t->frozen = frozen;
        t->runsBackward = runsBackward;
        t->needsGrad = needsGrad;
        t->readsInput = readsInput;
        t->dxWire = dxWire;
        t->dxMode = dxMode;
        t->inputStart = inputStart;
        t->inputIds = inputIds;
        t->inputDims = inputDims;
        t->inputOrder = inputOrder;
        t->actShapes = actShapes;
        t->bindScratch = bindScratch;
        wires[0] = recordOf(&facts[0], NULL);
    }
    for (size_t id = 1; id < numWires; id++) {
        const rematWireFact_t *f = &facts[id];
        tensor_t *hdr = SLAB_PLACE(layout, tensor_t, 1u, f);
        shape_t *shape = SLAB_PLACE(layout, shape_t, 1u, f);
        size_t *dims = SLAB_PLACE(layout, size_t, f->rank, f);
        size_t *order = SLAB_PLACE(layout, size_t, f->rank, f);
        quantization_t *q = SLAB_PLACE(layout, quantization_t, 1u, f);
        void *qConfig = NULL;
        if (f->dtype == SYM_INT32) {
            qConfig = SLAB_PLACE(layout, symInt32QConfig_t, 1u, f);
        } else if (f->dtype == BFP) {
            qConfig = SLAB_PLACE(layout, bfpQConfig_t, 1u, f);
        }
        if (t != NULL) {
            *hdr = (tensor_t){.data = NULL, .shape = shape, .quantization = q, .sparsity = NULL};
            *shape = (shape_t){
                .numberOfDimensions = f->rank, .dimensions = dims, .orderOfDimensions = order};
            *q = (quantization_t){.type = (qtype_t)f->dtype, .qConfig = qConfig};
            wires[id] = recordOf(f, hdr);
        }
    }
    /* Exponent arrays go at the tail: a 1-byte per-tensor array
     * never sits between two pointer-bearing headers, and the last one abuts
     * the block end, where ASan sees an overrun (Task 5). */
    for (size_t id = 1; id < numWires; id++) {
        const rematWireFact_t *f = &facts[id];
        if (f->dtype != BFP) {
            continue;
        }
        uint8_t *exponents = SLAB_PLACE(layout, uint8_t, f->numGroups, f);
        if (t != NULL) {
            ((bfpQConfig_t *)wires[id].hdr->quantization->qConfig)->exponents = exponents;
        }
    }
    return t;
}

bool rematWireTableInit(rematWireTable_t **out, const model_t *m, lossConfig_t loss,
                        const tensor_t *inputLike) {
    *out = NULL;
    modelRequireValid(m, "rematWireTableInit");
    size_t n = m->numLayers;

    /* The gradient flags and the dx slots, derived once here and copied into the table: dxWire
     * first, so it is 2-aligned without padding. Freed with the facts. */
    size_t slots = mulSize(n, MODEL_MAX_ARITY, NULL, "dx slots");
    size_t flagBytes =
        addSize(mulSize(slots, sizeof(uint16_t) + 1u, NULL, "gradient flags"),
                addSize(mulSize(n, 2u, NULL, "gradient flags"), 1u, NULL, "gradient flags"), NULL,
                "gradient flags");
    uint8_t *flagBlock = reserveMemory(flagBytes);
    if (flagBlock == NULL) {
        return false;
    }
    uint16_t *dxWire = (uint16_t *)flagBlock;
    uint8_t *needsGrad = flagBlock + slots * sizeof(uint16_t);
    uint8_t *runsBackward = needsGrad + n + 1u;
    uint8_t *dxMode = runsBackward + n;
    modelGradFlags(m, loss.funcType, needsGrad, runsBackward);
    /* CrossEntropy's backward IS the combined Softmax + CE gradient p - y, so the sink's own
     * backward never runs: it must be the Softmax, or a layer would silently go untrained, and
     * its config must declare one of the two heads. With nothing to train nothing is skipped. */
    if (loss.funcType == CROSS_ENTROPY && needsGrad[n]) {
        requireCrossEntropyHead(m->layers[n - 1u], "rematWireTableInit");
    }
    bool hasBackward = rematHasBackward(n, loss.funcType, needsGrad);
    bool isChain = modelIsChain(m);
    size_t numEdges = isChain ? 0u : m->inputStart[n];
    size_t numWires = addSize(
        addSize(n, 1u, NULL, "numWires"),
        rematDeriveGradWires(m, loss.funcType, needsGrad, runsBackward, NULL, dxWire, dxMode), NULL,
        "numWires");
    if (numWires >= REMAT_NONE) {
        PRINT_ERROR("rematWireTableInit: %zu wires reach REMAT_NONE (0xFFFF): wire ids are "
                    "uint16_t",
                    numWires);
        exit(1);
    }
    size_t inputRank = inputLike->shape->numberOfDimensions;
    if (inputRank > UINT8_MAX) {
        PRINT_ERROR("rematWireTableInit: wire ACT 0 has rank %zu, above the uint8_t rank field "
                    "(255)",
                    inputRank);
        exit(1);
    }
    size_t maxRank = inputRank > 2u ? inputRank : 2u;

    /* The table's size depends on the derived facts (ranks, dtypes, exponent
     * counts), so init derives into a transient scratch of numWires facts and
     * n + 1 shapes -- sized from numWires and maxRank alone -- and frees it
     * before returning. Every bind derives into the table's own copies. */
    rematWireFact_t *facts =
        reserveMemory(mulSize(numWires, sizeof(rematWireFact_t), NULL, "derivation scratch"));
    if (facts == NULL) {
        freeReservedMemory(flagBlock);
        return false;
    }
    slabLayout_t shapeSizing = {.base = NULL, .cursor = 0};
    (void)placeShapes(&shapeSizing, n + 1u, maxRank);
    uint8_t *shapeBlock = reserveMemory(shapeSizing.cursor);
    if (shapeBlock == NULL) {
        freeReservedMemory(facts);
        freeReservedMemory(flagBlock);
        return false;
    }
    slabLayout_t shapePlacing = {.base = shapeBlock, .cursor = 0};
    shape_t *shapes = placeShapes(&shapePlacing, n + 1u, maxRank);
    modelShapePass(m, inputLike->shape, shapes, maxRank);
    numberWires(facts, m, loss.funcType, needsGrad, runsBackward, dxWire, dxMode);
    deriveFacts(facts, numWires, m, shapes, inputLike, dxWire, dxMode);

    size_t totalBytes = 0;
    for (size_t id = 1; id < numWires; id++) {
        const rematWireFact_t *f = &facts[id];
        if (f->dtype != FLOAT32 && f->dtype != SYM_INT32 && f->dtype != BFP) {
            PRINT_ERROR("rematWireTableInit: wire %s %u has dtype %u; remat wires are FLOAT32, "
                        "SYM_INT32 or BFP",
                        rematWireKindName(f->kind), (unsigned)f->index, (unsigned)f->dtype);
            exit(1);
        }
        if (f->bytes == 0) {
            PRINT_ERROR("rematWireTableInit: wire %s %u has zero bytes (#160)",
                        rematWireKindName(f->kind), (unsigned)f->index);
            exit(1);
        }
        /* Makes every later sum over wires (liveBytes, peakLiveBytes) provably in range. */
        totalBytes = addSize(totalBytes, f->bytes, f, "total wire bytes");
    }
    if (!isChain) {
        requireFloat32MergeWires(facts, numWires, m, loss.funcType, dxWire, dxMode);
    }

    slabLayout_t sizing = {.base = NULL, .cursor = 0};
    (void)layoutTable(&sizing, facts, numWires, n, isChain, numEdges, inputRank, maxRank);
    uint8_t *block = reserveMemory(sizing.cursor);
    if (block == NULL) {
        freeReservedMemory(shapeBlock);
        freeReservedMemory(facts);
        freeReservedMemory(flagBlock);
        return false;
    }
    slabLayout_t placing = {.base = block, .cursor = 0};
    rematWireTable_t *t =
        layoutTable(&placing, facts, numWires, n, isChain, numEdges, inputRank, maxRank);
    modelShapePass(m, inputLike->shape, t->actShapes, maxRank);

    t->modelSize = n;
    t->lossType = loss.funcType;
    t->hasBackward = hasBackward;
    t->numWires = numWires;
    for (size_t j = 0; j <= n; j++) {
        t->gradIdOf[j] = REMAT_NONE;
    }
    for (size_t id = n + 1u; id < numWires; id++) {
        if (t->wires[id].kind == REMAT_WIRE_GRAD) {
            t->gradIdOf[t->wires[id].index] = (uint16_t)id;
        }
    }
    if (loss.funcType == CROSS_ENTROPY && hasBackward) {
        t->gradIdOf[n - 1u] = t->gradIdOf[n]; /* the seed is the Softmax input's gradient */
    }
    for (size_t i = 0; i < n; i++) {
        t->layerType[i] = (uint8_t)m->layers[i]->type;
        t->frozen[i] = layerIsFrozen(m->layers[i]) ? 1u : 0u;
        t->readsInput[i] = layerBackwardReadsInput(m->layers[i]) ? 1u : 0u;
    }
    memcpy(t->needsGrad, needsGrad, n + 1u);
    memcpy(t->runsBackward, runsBackward, n);
    memcpy(t->dxWire, dxWire, slots * sizeof(uint16_t));
    memcpy(t->dxMode, dxMode, slots);
    t->isChain = isChain ? 1u : 0u;
    t->numEdges = numEdges;
    if (!isChain) {
        memcpy(t->inputStart, m->inputStart, (n + 1u) * sizeof(uint16_t));
        memcpy(t->inputIds, m->inputIds, numEdges * sizeof(uint16_t));
    }
    requireTransparentCopyWidths(facts, m, t);
    t->inputRank = inputRank;
    size_t inputKeyBytes = mulSize(inputRank, sizeof(size_t), NULL, "input key bytes");
    memcpy(t->inputDims, inputLike->shape->dimensions, inputKeyBytes);
    memcpy(t->inputOrder, inputLike->shape->orderOfDimensions, inputKeyBytes);
    t->inputType = (uint8_t)inputLike->quantization->type;
    t->maxRank = (uint8_t)maxRank;
    t->slabBytes = sizing.cursor;
    freeReservedMemory(shapeBlock);
    freeReservedMemory(facts);
    freeReservedMemory(flagBlock);
    *out = t;
    return true;
}

void rematWireTableFree(rematWireTable_t *t) {
    freeReservedMemory(t);
}

static void copyGradShape(shape_t *dst, const shape_t *src) {
    memcpy(dst->dimensions, src->dimensions, src->numberOfDimensions * sizeof(size_t));
    dst->numberOfDimensions = src->numberOfDimensions;
    setOrderOfDimsForNewTensor(dst->numberOfDimensions, dst->orderOfDimensions);
}

/* The only road into a slab BFP config. Phase 3 of the table bind and the
 * inherited-GRAD path of rematWireBind both come here, so neither can write
 * exponents[0..numGroups) past the slab's reserved expCapacity. */
static void bindBfpInto(rematWireTable_t *t, uint16_t id, const bfpQConfig_t *tmpl,
                        size_t elements) {
    const rematWire_t *w = &t->wires[id];
    rematBfpGroups_t g = rematBfpWireGrouping(tmpl, elements, w->kind, w->index);
    if (g.numGroups > w->expCapacity) {
        PRINT_ERROR("remat: wire %s %u needs %zu BFP exponent groups, above its expCapacity %zu",
                    rematWireKindName(w->kind), (unsigned)w->index, g.numGroups, w->expCapacity);
        exit(1);
    }
    quantization_t *q = w->hdr->quantization;
    bfpQConfig_t *qc = q->qConfig;
    initBfpQConfigGroupedInto(tmpl->mantissaBits, tmpl->exponentBits, tmpl->roundingMode,
                              g.numGroups, g.groupSize, qc->exponents, qc);
    initBfpQuantization(qc, q);
}

/* Config fields from the template; dynamic state fresh (SYM scale 1, BFP
 * exponents at the bias). The slab's structural pointers were set at init. */
static void writeWireConfig(rematWireTable_t *t, uint16_t id, const quantization_t *tmpl,
                            size_t elements) {
    quantization_t *q = t->wires[id].hdr->quantization;
    if (tmpl->type == FLOAT32) {
        initFloat32Quantization(q);
    } else if (tmpl->type == SYM_INT32) {
        const symInt32QConfig_t *src = tmpl->qConfig;
        symInt32QConfig_t *dst = q->qConfig;
        initSymInt32QConfigWithQMaxBits(src->roundingMode, dst, src->qMaxBits);
        initSymInt32Quantization(dst, q);
    } else { /* BFP: the only other wire dtype */
        bindBfpInto(t, id, tmpl->qConfig, elements);
    }
}

/* The type-derived GRAD headers of phase 3. */
static void writeGradHeaders(rematWireTable_t *t, const rematWireFact_t *facts) {
    for (size_t id = t->modelSize + 1u; id < t->numWires; id++) {
        rematWire_t *w = &t->wires[id];
        if (w->inheritFrom != REMAT_NONE) {
            continue; /* derived from the LIVE ACT header when its range opens (rematWireBind) */
        }
        copyGradShape(w->hdr->shape, t->wires[w->index].hdr->shape);
        writeWireConfig(t, (uint16_t)id, facts[id].tmpl, facts[id].elements);
    }
}

/* Phase 3 of the table bind: write the headers. ACT j's shape is phase 1's actShapes[j]. */
static void writeHeaders(rematWireTable_t *t, rematMode_t mode, tensor_t *input,
                         const rematWireFact_t *facts) {
    t->wires[0].hdr = input;
    t->wires[0].bytes = facts[0].bytes; /* a width edit on a packed input is adopted */
    for (size_t j = 1; j <= t->modelSize; j++) {
        shape_t *dst = t->wires[j].hdr->shape;
        const shape_t *src = &t->actShapes[j];
        size_t rankBytes = src->numberOfDimensions * sizeof(size_t);
        dst->numberOfDimensions = src->numberOfDimensions;
        memcpy(dst->dimensions, src->dimensions, rankBytes);
        memcpy(dst->orderOfDimensions, src->orderOfDimensions, rankBytes);
        writeWireConfig(t, (uint16_t)j, facts[j].tmpl, facts[j].elements);
    }
    if (mode == REMAT_MODE_TRAIN) { /* eval binds no GRAD wire */
        writeGradHeaders(t, facts);
    }
    for (size_t id = 0; id < t->numWires; id++) {
        t->wires[id].bindGen = 0;
    }
    t->liveBytes = 0;
    t->observedPeakLiveBytes = 0;
}

static void exitModelKey(const char *fact, size_t built, size_t live) {
    PRINT_ERROR("rematWireTableBind: key mismatch on '%s': built %zu, live %zu (a key change "
                "needs a fresh init)",
                fact, built, live);
    exit(1);
}

static void exitModelKeyAt(const char *fact, size_t i, size_t built, size_t live) {
    PRINT_ERROR("rematWireTableBind: key mismatch on '%s[%zu]': built %zu, live %zu (a key "
                "change needs a fresh init)",
                fact, i, built, live);
    exit(1);
}

static void exitWireKey(const rematWire_t *w, const char *field, size_t built, size_t live) {
    PRINT_ERROR("rematWireTableBind: key mismatch on wire %s %u, field '%s': built %zu, live %zu "
                "(a key change needs a fresh init)",
                rematWireKindName(w->kind), (unsigned)w->index, field, built, live);
    exit(1);
}

static void exitInputKeyAt(const char *field, size_t k, size_t built, size_t live) {
    PRINT_ERROR("rematWireTableBind: key mismatch on wire ACT 0, field '%s[%zu]': built %zu, "
                "live %zu (a key change needs a fresh init)",
                field, k, built, live);
    exit(1);
}

/* Phase 1 step 1 of the table bind: the model facts and ACT 0, before any shape is
 * derived. With ACT 0's rank and every layer type unchanged, each derived rank
 * is a pure function of them, so the built maxRank bounds the scratch.
 * needsGrad, hasBackward and the dx slots follow from the compared facts, and so do
 * the per-wire kind and rank. The wiring compares come first because on a caller-scheduler
 * path nothing validates the live model: only its match with the built wiring, which the init
 * validated, makes the live CSR safe for the readers after it (modelGradFlags here,
 * modelShapePass and rematDeriveGradWires in phase 1). The live side goes through modelIsChain,
 * so a builder-made chain keys as the MODEL_CHAIN it encodes. */
static void requireModelKey(const rematWireTable_t *t, const model_t *m, lossFuncType_t lt,
                            const tensor_t *input) {
    size_t n = m->numLayers;
    if (n != t->modelSize) {
        exitModelKey("modelSize", t->modelSize, n);
    }
    if (lt != t->lossType) {
        exitModelKey("lossType", (size_t)t->lossType, (size_t)lt);
    }
    for (size_t i = 0; i < n; i++) {
        if ((uint8_t)m->layers[i]->type != t->layerType[i]) {
            exitModelKeyAt("layerType", i, t->layerType[i], (size_t)m->layers[i]->type);
        }
    }
    size_t liveIsChain = modelIsChain(m) ? 1u : 0u;
    if (liveIsChain != t->isChain) {
        exitModelKey("isChain", t->isChain, liveIsChain);
    }
    if (!t->isChain) {
        for (size_t i = 0; i <= n; i++) {
            if (m->inputStart[i] != t->inputStart[i]) {
                exitModelKeyAt("inputStart", i, t->inputStart[i], m->inputStart[i]);
            }
        }
        for (size_t e = 0; e < t->numEdges; e++) {
            if (m->inputIds[e] != t->inputIds[e]) {
                exitModelKeyAt("inputIds", e, t->inputIds[e], m->inputIds[e]);
            }
        }
    }
    /* The live flags go into the bind scratch, unused until phase 1 derives into it; it holds
     * numWires facts, far more than the 2n + 1 bytes. */
    uint8_t *liveNeedsGrad = (uint8_t *)t->bindScratch;
    uint8_t *liveRunsBackward = liveNeedsGrad + n + 1u;
    modelGradFlags(m, lt, liveNeedsGrad, liveRunsBackward);
    /* The head config is no key field, so the init's head check runs again on the live model. */
    if (lt == CROSS_ENTROPY && liveNeedsGrad[n]) {
        requireCrossEntropyHead(m->layers[n - 1u], "rematWireTableBind");
    }
    for (size_t i = 0; i < n; i++) {
        if (liveRunsBackward[i] != t->runsBackward[i]) {
            exitModelKeyAt("runsBackward", i, t->runsBackward[i], liveRunsBackward[i]);
        }
    }
    for (size_t i = 0; i < n; i++) {
        size_t frozen = layerIsFrozen(m->layers[i]) ? 1u : 0u;
        if (frozen != t->frozen[i]) {
            exitModelKeyAt("frozen", i, t->frozen[i], frozen);
        }
    }
    const shape_t *shape = input->shape;
    if (shape->numberOfDimensions != t->inputRank) {
        exitWireKey(&t->wires[0], "rank", t->inputRank, shape->numberOfDimensions);
    }
    for (size_t k = 0; k < t->inputRank; k++) {
        if (shape->dimensions[k] != t->inputDims[k]) {
            exitInputKeyAt("dims", k, t->inputDims[k], shape->dimensions[k]);
        }
    }
    for (size_t k = 0; k < t->inputRank; k++) {
        if (shape->orderOfDimensions[k] != t->inputOrder[k]) {
            exitInputKeyAt("order", k, t->inputOrder[k], shape->orderOfDimensions[k]);
        }
    }
    if ((uint8_t)input->quantization->type != t->inputType) {
        exitWireKey(&t->wires[0], "dtype", t->inputType, (size_t)input->quantization->type);
    }
}

/* Phase 2 of the table bind: check the full per-wire key, before phase 3
 * writes anything (check before write). numGroups <= expCapacity is the only capacity compare. */
static void requireWireKey(const rematWireTable_t *t, const rematWireFact_t *facts) {
    for (size_t id = 1; id < t->numWires; id++) {
        const rematWire_t *w = &t->wires[id];
        const rematWireFact_t *f = &facts[id];
        if (f->dtype != w->dtype) {
            exitWireKey(w, "dtype", w->dtype, f->dtype);
        }
        if (f->bytes != w->bytes) {
            exitWireKey(w, "bytes", w->bytes, f->bytes);
        }
        if (f->numGroups > w->expCapacity) {
            PRINT_ERROR("rematWireTableBind: key mismatch on wire %s %u, field 'numGroups': %zu "
                        "groups exceed expCapacity %zu (a key change needs a fresh init)",
                        rematWireKindName(w->kind), (unsigned)w->index, f->numGroups,
                        w->expCapacity);
            exit(1);
        }
    }
}

void rematWireTableBind(rematWireTable_t *t, const model_t *m, lossFuncType_t lt, rematMode_t mode,
                        tensor_t *input) {
    requireModelKey(t, m, lt, input);
    /* Phase 1 writes only into the table block: the shapes, then the per-wire facts. */
    modelShapePass(m, input->shape, t->actShapes, t->maxRank);
    rematWireFact_t *facts = t->bindScratch;
    numberWires(facts, m, t->lossType, t->needsGrad, t->runsBackward, t->dxWire, t->dxMode);
    deriveFacts(facts, t->numWires, m, t->actShapes, input, t->dxWire, t->dxMode);
    requireWireKey(t, facts);
    writeHeaders(t, mode, input, facts);
}

void rematWireTableUnbind(rematWireTable_t *t) {
    t->wires[0].hdr = NULL;
    t->inPlace = NULL;
}

/* Names a record in a size-overflow exit (the checked helpers take a rematWireFact_t). */
static rematWireFact_t nameOf(const rematWire_t *w) {
    return (rematWireFact_t){.kind = w->kind, .index = w->index};
}

/* Overflow-checked like every size product, via the same elementsOf phase 1
 * uses. Unreachable in practice: the source ACT header was derived at this
 * call's table bind from checked facts. */
static size_t liveElements(const shape_t *shape, const rematWire_t *w) {
    rematWireFact_t name = nameOf(w);
    return elementsOf(shape, &name);
}

/* Inherited GRAD headers are derived when the range opens, after the forward
 * that produced their source, so a producer that wrote a config field of its
 * output is seen. Check before write: every check (dtype, rank, the live
 * payload bytes, and inside bindBfpInto the grouping and the expCapacity
 * bound) runs before the first slab write, so the config is written before
 * the shape. */
static void deriveInheritedHeader(rematWireTable_t *t, uint16_t id) {
    rematWire_t *w = &t->wires[id];
    const tensor_t *src = t->wires[w->inheritFrom].hdr;
    if ((uint8_t)src->quantization->type != w->dtype) {
        PRINT_ERROR("rematWireBind: wire %s %u field 'dtype': its source ACT %u is live as dtype "
                    "%d, the slab reserved dtype %u",
                    rematWireKindName(w->kind), (unsigned)w->index, (unsigned)w->inheritFrom,
                    (int)src->quantization->type, (unsigned)w->dtype);
        exit(1);
    }
    if (src->shape->numberOfDimensions != w->rank) {
        PRINT_ERROR("rematWireBind: wire %s %u field 'rank': its source ACT %u is live at rank "
                    "%zu, the slab reserved rank %u",
                    rematWireKindName(w->kind), (unsigned)w->index, (unsigned)w->inheritFrom,
                    src->shape->numberOfDimensions, (unsigned)w->rank);
        exit(1);
    }
    size_t elements = liveElements(src->shape, w);
    rematWireFact_t name = nameOf(w);
    size_t liveBytes = wireBytes(src->quantization, elements, &name);
    if (liveBytes != w->bytes) {
        PRINT_ERROR("rematWireBind: wire %s %u field 'bytes': its source ACT %u is live at %zu "
                    "bytes, the slab reserved %zu",
                    rematWireKindName(w->kind), (unsigned)w->index, (unsigned)w->inheritFrom,
                    liveBytes, w->bytes);
        exit(1);
    }
    writeWireConfig(t, id, src->quantization, elements);
    copyGradShape(w->hdr->shape, src->shape);
}

#ifdef ODT_REMAT_VERIFY
/* Test builds (ODT_REMAT_VERIFY). Poison at Bind stops calloc zeros from
 * masking a read of never-written bytes on the first call (the arena reuses
 * bytes without zeroing); poison at Release, while the bytes are still owned,
 * makes a read of released bytes loud. FLOAT32 gets a signalling NaN. */
#define REMAT_POISON_FLOAT32_BITS 0x7FA00000u
#define REMAT_POISON_BFP_BYTE 0xA5u

static void poisonWireBytes(const rematWire_t *w, uint8_t *bytes) {
    if (w->dtype == BFP) {
        memset(bytes, REMAT_POISON_BFP_BYTE, w->bytes);
        return;
    }
    uint32_t pattern = (w->dtype == FLOAT32) ? REMAT_POISON_FLOAT32_BITS : (uint32_t)INT32_MIN;
    for (size_t offset = 0; offset < w->bytes; offset += sizeof pattern) {
        memcpy(bytes + offset, &pattern, sizeof pattern);
    }
}
#endif

/* An in-place member of the borrowed marks: a wire whose layer's BACKWARD
 * writes into the bytes of the wire before it. */
static bool isMember(const rematWireTable_t *t, size_t w) {
    return t->inPlace != NULL && w < t->numWires && t->inPlace[w] != 0u;
}

/* Bind and Release name only a chain's root, as rows name only ranges. */
static void requireNotAMember(const rematWireTable_t *t, uint16_t w, const char *call,
                              const char *verb) {
    if (!isMember(t, w)) {
        return;
    }
    uint16_t root = w;
    while (isMember(t, root)) {
        root--;
    }
    PRINT_ERROR("%s: wire %s %u is an in-place member of %s %u; %s the chain's root", call,
                rematWireKindName(t->wires[w].kind), (unsigned)t->wires[w].index,
                rematWireKindName(t->wires[root].kind), (unsigned)t->wires[root].index, verb);
    exit(1);
}

/* A root's bind binds its chain: every member takes the root's bytes and a
 * bind generation of its own, and an inherited member (Flatten's dx) derives
 * its header here, through the path of every inherited wire, since it is
 * never bound on its own. Members follow their root in id order. The root's
 * bytes are poisoned and counted once. */
static void bindMembers(rematWireTable_t *t, uint16_t root, uint8_t *bytes) {
    for (size_t m = root + 1u; isMember(t, m); m++) {
        rematWire_t *rec = &t->wires[m];
        if (rec->hdr->data != NULL) {
            PRINT_ERROR("rematWireBind: wire %s %u, an in-place member of %s %u, is already bound",
                        rematWireKindName(rec->kind), (unsigned)rec->index,
                        rematWireKindName(t->wires[root].kind), (unsigned)t->wires[root].index);
            exit(1);
        }
        if (rec->inheritFrom != REMAT_NONE) {
            deriveInheritedHeader(t, (uint16_t)m);
        }
        rec->hdr->data = bytes;
        rec->bindGen++;
    }
}

static void releaseMembers(rematWireTable_t *t, uint16_t root) {
    const uint8_t *bytes = t->wires[root].hdr->data;
    for (size_t m = root + 1u; isMember(t, m); m++) {
        rematWire_t *rec = &t->wires[m];
        if (rec->hdr->data != bytes) {
            PRINT_ERROR("rematWireRelease: wire %s %u, an in-place member of %s %u, does not hold "
                        "its root's bytes",
                        rematWireKindName(rec->kind), (unsigned)rec->index,
                        rematWireKindName(t->wires[root].kind), (unsigned)t->wires[root].index);
            exit(1);
        }
        rec->hdr->data = NULL;
    }
}

void rematWireBind(rematWireTable_t *t, uint16_t w, uint8_t *bytes) {
    if (w >= t->numWires) {
        PRINT_ERROR("rematWireBind: wire id %u out of range (numWires %zu)", (unsigned)w,
                    t->numWires);
        exit(1);
    }
    if (bytes == NULL) {
        PRINT_ERROR("rematWireBind: bytes is NULL");
        exit(1);
    }
    rematWire_t *rec = &t->wires[w];
    if (rec->borrowed) {
        PRINT_ERROR("rematWireBind: wire ACT 0 is the caller's borrowed input; it is never bound");
        exit(1);
    }
    requireNotAMember(t, w, "rematWireBind", "bind");
    if (rec->hdr->data != NULL) {
        PRINT_ERROR("rematWireBind: wire %s %u is already bound", rematWireKindName(rec->kind),
                    (unsigned)rec->index);
        exit(1);
    }
    if (rec->inheritFrom != REMAT_NONE) {
        deriveInheritedHeader(t, w);
    }
    rec->hdr->data = bytes;
#ifdef ODT_REMAT_VERIFY
    poisonWireBytes(rec, bytes);
#endif
    rec->bindGen++;
    bindMembers(t, w, bytes);
    /* Overflow-checked, though each wire counts at most once (a bound wire cannot
     * be bound again), so init's checked total of wire bytes already bounds the
     * sum: this exit is unreachable and has no dedicated test. */
    rematWireFact_t name = nameOf(rec);
    t->liveBytes = addSize(t->liveBytes, rec->bytes, &name, "liveBytes");
    if (t->liveBytes > t->observedPeakLiveBytes) {
        t->observedPeakLiveBytes = t->liveBytes;
    }
}

void rematWireRelease(rematWireTable_t *t, uint16_t w) {
    if (w >= t->numWires) {
        PRINT_ERROR("rematWireRelease: wire id %u out of range (numWires %zu)", (unsigned)w,
                    t->numWires);
        exit(1);
    }
    rematWire_t *rec = &t->wires[w];
    if (rec->borrowed) {
        PRINT_ERROR("rematWireRelease: wire ACT 0 is the caller's borrowed input; it is never "
                    "released");
        exit(1);
    }
    requireNotAMember(t, w, "rematWireRelease", "release");
    if (rec->hdr->data == NULL) {
        PRINT_ERROR("rematWireRelease: wire %s %u is not bound", rematWireKindName(rec->kind),
                    (unsigned)rec->index);
        exit(1);
    }
    if (rec->bytes > t->liveBytes) {
        PRINT_ERROR("rematWireRelease: wire %s %u live bytes would underflow (released more than "
                    "bound since the last table bind)",
                    rematWireKindName(rec->kind), (unsigned)rec->index);
        exit(1);
    }
    releaseMembers(t, w);
#ifdef ODT_REMAT_VERIFY
    poisonWireBytes(rec, rec->hdr->data);
#endif
    rec->hdr->data = NULL;
    t->liveBytes -= rec->bytes;
}

tensor_t *rematWireHdr(const rematWireTable_t *t, uint16_t w) {
    return t->wires[w].hdr;
}

tensor_t *rematActHdr(const rematWireTable_t *t, size_t j) {
    return t->wires[j].hdr;
}

tensor_t *rematGradHdr(const rematWireTable_t *t, size_t j) {
    uint16_t id = t->gradIdOf[j];
    return id == REMAT_NONE ? NULL : t->wires[id].hdr;
}

uint16_t rematActId(const rematWireTable_t *t, size_t j) {
    (void)t; /* ACT j is wire j by construction; the table stays in the signature for callers */
    return (uint16_t)j;
}

uint16_t rematGradId(const rematWireTable_t *t, size_t j) {
    return t->gradIdOf[j];
}

size_t rematWireBytes(const rematWireTable_t *t, uint16_t w) {
    return t->wires[w].bytes;
}
