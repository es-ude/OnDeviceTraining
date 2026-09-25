#define SOURCE_FILE "REMAT_WIRE_TABLE"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "Common.h"
#include "Layer.h"
#include "LayerConfigAccess.h"
#include "LossFunction.h"
#include "Quantization.h"
#include "RematCheckedSize.h"
#include "RematPlan.h"
#include "StorageApi.h"
#include "Tensor.h"

static const char *wireKindName(uint8_t kind) {
    return kind == REMAT_WIRE_ACT ? "ACT" : "GRAD";
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
                    wireKindName(f->kind), (unsigned)f->index);
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
        PRINT_ERROR("remat: wire %s %u has unknown qtype %d", wireKindName(f->kind),
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
                    tmpl->groupSize, elements, wireKindName(kind), index);
        exit(1);
    }
    return (rematBfpGroups_t){.numGroups = elements / tmpl->groupSize,
                              .groupSize = tmpl->groupSize};
}

static size_t gradsBelowSeed(size_t deepest, ptrdiff_t top) {
    return top > (ptrdiff_t)deepest ? (size_t)(top - (ptrdiff_t)deepest) : 0u;
}

/* Production order (plan Assumption 4): with ACT j at id j and GRADs in the
 * order BACKWARD writes them, every range begins in wire-id order. */
static void numberWires(rematWireFact_t *facts, layer_t **model, size_t n, size_t deepest,
                        ptrdiff_t top, bool hasBackward) {
    for (size_t j = 0; j <= n; j++) {
        facts[j] = (rematWireFact_t){
            .kind = REMAT_WIRE_ACT, .index = (uint16_t)j, .inheritFrom = REMAT_NONE};
    }
    if (!hasBackward) {
        return;
    }
    size_t id = n + 1u;
    facts[id++] = (rematWireFact_t){
        .kind = REMAT_WIRE_GRAD, .index = (uint16_t)n, .inheritFrom = (uint16_t)n};
    for (ptrdiff_t l = top; l > (ptrdiff_t)deepest; l--) {
        bool passThrough = model[l]->type == FLATTEN; /* type-derived: plan Assumption 5 */
        facts[id++] = (rematWireFact_t){.kind = REMAT_WIRE_GRAD,
                                        .index = (uint16_t)l,
                                        .inheritFrom = passThrough ? (uint16_t)l : REMAT_NONE};
    }
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

/* ACT j takes layerOutputQ(model[j-1]), or for Flatten the template of ACT j-1
 * (for Flatten-at-0 the caller's live input config): config fields only, never
 * a scale or exponents. A GRAD takes backwardWireQ(model[l]) and ACT l's
 * shape, or inherits its ACT's template. Shapes ping-pong between two scratch
 * shapes of maxRank entries (plan Assumption 6). */
static void deriveFacts(rematWireFact_t *facts, size_t numWires, layer_t **model, size_t n,
                        const tensor_t *input, size_t maxRank) {
    size_t dimsA[maxRank], orderA[maxRank], dimsB[maxRank], orderB[maxRank];
    shape_t scratch[2] = {{.dimensions = dimsA, .orderOfDimensions = orderA},
                          {.dimensions = dimsB, .orderOfDimensions = orderB}};
    shape_t *prev = input->shape;
    setWireFacts(&facts[0], input->quantization, (uint8_t)prev->numberOfDimensions,
                 elementsOf(prev, &facts[0]));
    for (size_t j = 1; j <= n; j++) {
        layer_t *layer = model[j - 1];
        shape_t *out = &scratch[j % 2u];
        out->numberOfDimensions = (layer->type == FLATTEN) ? 2u : prev->numberOfDimensions;
        layerFunctions[layer->type].calcOutputShape(layer, prev, out);
        const quantization_t *tmpl =
            (layer->type == FLATTEN) ? facts[j - 1].tmpl : layerOutputQ(layer);
        setWireFacts(&facts[j], tmpl, (uint8_t)out->numberOfDimensions, elementsOf(out, &facts[j]));
        prev = out;
    }
    for (size_t id = n + 1u; id < numWires; id++) {
        rematWireFact_t *f = &facts[id];
        const rematWireFact_t *act = &facts[f->index];
        const quantization_t *tmpl = (f->inheritFrom != REMAT_NONE)
                                         ? facts[f->inheritFrom].tmpl
                                         : backwardWireQ(model[f->index]);
        setWireFacts(f, tmpl, act->rank, act->elements);
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

/* One cursor rule for the sizing and the placing pass (spec §3.3): with base
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

static rematWireTable_t *layoutTable(slabLayout_t *layout, const rematWireFact_t *facts,
                                     size_t numWires, size_t n, size_t inputRank) {
    rematWireTable_t *t = SLAB_PLACE(layout, rematWireTable_t, 1u, NULL);
    rematWire_t *wires = SLAB_PLACE(layout, rematWire_t, numWires, NULL);
    uint16_t *gradIdOf = SLAB_PLACE(layout, uint16_t, n + 1u, NULL);
    uint8_t *layerType = SLAB_PLACE(layout, uint8_t, n, NULL);
    uint8_t *frozen = SLAB_PLACE(layout, uint8_t, n, NULL);
    size_t *inputDims = SLAB_PLACE(layout, size_t, inputRank, NULL);
    size_t *inputOrder = SLAB_PLACE(layout, size_t, inputRank, NULL);
    /* The bind's derivation scratch (plan Assumption 29): after the fixed-size
     * key, before the headers, so the exponent arrays stay the block's tail. The
     * cursor rule aligns it for rematWireFact_t like every other object. */
    rematWireFact_t *bindScratch = SLAB_PLACE(layout, rematWireFact_t, numWires, NULL);
    if (t != NULL) {
        t->wires = wires;
        t->gradIdOf = gradIdOf;
        t->layerType = layerType;
        t->frozen = frozen;
        t->inputDims = inputDims;
        t->inputOrder = inputOrder;
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
    /* Exponent arrays go at the tail (spec §3.3): a 1-byte per-tensor array
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

bool rematWireTableInit(rematWireTable_t **out, layer_t **model, size_t n, lossConfig_t loss,
                        const tensor_t *inputLike) {
    *out = NULL;
    if (n == 0) {
        PRINT_ERROR("rematWireTableInit: modelSize == 0: nothing to schedule");
        exit(1);
    }
    size_t deepest;
    ptrdiff_t top;
    rematBackwardRange(model, n, loss.funcType, &deepest, &top);
    bool hasBackward = deepest < n;
    size_t numWires = addSize(n, 1u, NULL, "numWires");
    if (hasBackward) {
        numWires = addSize(numWires, addSize(1u, gradsBelowSeed(deepest, top), NULL, "numWires"),
                           NULL, "numWires");
    }
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
     * counts), so init derives into a transient scratch of numWires facts --
     * sized from numWires alone -- and frees it before returning (plan
     * Assumption 29). Every bind derives into the table's own copy. */
    rematWireFact_t *facts =
        reserveMemory(mulSize(numWires, sizeof(rematWireFact_t), NULL, "derivation scratch"));
    if (facts == NULL) {
        return false;
    }
    numberWires(facts, model, n, deepest, top, hasBackward);
    deriveFacts(facts, numWires, model, n, inputLike, maxRank);

    size_t totalBytes = 0;
    for (size_t id = 1; id < numWires; id++) {
        const rematWireFact_t *f = &facts[id];
        if (f->dtype != FLOAT32 && f->dtype != SYM_INT32 && f->dtype != BFP) {
            PRINT_ERROR("rematWireTableInit: wire %s %u has dtype %u; remat wires are FLOAT32, "
                        "SYM_INT32 or BFP",
                        wireKindName(f->kind), (unsigned)f->index, (unsigned)f->dtype);
            exit(1);
        }
        if (f->bytes == 0) {
            PRINT_ERROR("rematWireTableInit: wire %s %u has zero bytes (#160)",
                        wireKindName(f->kind), (unsigned)f->index);
            exit(1);
        }
        /* Makes every later sum over wires (liveBytes, peakLiveBytes) provably in range. */
        totalBytes = addSize(totalBytes, f->bytes, f, "total wire bytes");
    }

    slabLayout_t sizing = {.base = NULL, .cursor = 0};
    (void)layoutTable(&sizing, facts, numWires, n, inputRank);
    uint8_t *block = reserveMemory(sizing.cursor);
    if (block == NULL) {
        freeReservedMemory(facts);
        return false;
    }
    slabLayout_t placing = {.base = block, .cursor = 0};
    rematWireTable_t *t = layoutTable(&placing, facts, numWires, n, inputRank);

    t->modelSize = n;
    t->lossType = loss.funcType;
    t->deepest = deepest;
    t->backwardTop = top;
    t->hasBackward = hasBackward;
    t->numWires = numWires;
    for (size_t j = 0; j <= n; j++) {
        t->gradIdOf[j] = REMAT_NONE;
    }
    for (size_t id = n + 1u; id < numWires; id++) {
        t->gradIdOf[t->wires[id].index] = (uint16_t)id;
    }
    for (size_t i = 0; i < n; i++) {
        t->layerType[i] = (uint8_t)model[i]->type;
        t->frozen[i] = layerIsFrozen(model[i]) ? 1u : 0u;
    }
    t->inputRank = inputRank;
    size_t inputKeyBytes = mulSize(inputRank, sizeof(size_t), NULL, "input key bytes");
    memcpy(t->inputDims, inputLike->shape->dimensions, inputKeyBytes);
    memcpy(t->inputOrder, inputLike->shape->orderOfDimensions, inputKeyBytes);
    t->inputType = (uint8_t)inputLike->quantization->type;
    t->maxRank = (uint8_t)maxRank;
    t->slabBytes = sizing.cursor;
    freeReservedMemory(facts);
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

/* C2: the only road into a slab BFP config. Phase 3 of the table bind and the
 * inherited-GRAD path of rematWireBind both come here, so neither can write
 * exponents[0..numGroups) past the slab's reserved expCapacity. */
static void bindBfpInto(rematWireTable_t *t, uint16_t id, const bfpQConfig_t *tmpl,
                        size_t elements) {
    const rematWire_t *w = &t->wires[id];
    rematBfpGroups_t g = rematBfpWireGrouping(tmpl, elements, w->kind, w->index);
    if (g.numGroups > w->expCapacity) {
        PRINT_ERROR("remat: wire %s %u needs %zu BFP exponent groups, above its expCapacity %zu",
                    wireKindName(w->kind), (unsigned)w->index, g.numGroups, w->expCapacity);
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

/* Phase 3 (spec §3.4). calcOutputShape is a pure function of the config and
 * the input shape, so recomputing into the slab reproduces phase 1's shapes
 * without keeping them in scratch. */
static void writeHeaders(rematWireTable_t *t, layer_t **model, tensor_t *input,
                         const rematWireFact_t *facts) {
    t->wires[0].hdr = input;
    t->wires[0].bytes = facts[0].bytes; /* a width edit on a packed input is adopted (RF3) */
    for (size_t j = 1; j <= t->modelSize; j++) {
        layer_t *layer = model[j - 1];
        tensor_t *hdr = t->wires[j].hdr;
        hdr->shape->numberOfDimensions = t->wires[j].rank;
        layerFunctions[layer->type].calcOutputShape(layer, t->wires[j - 1].hdr->shape, hdr->shape);
        writeWireConfig(t, (uint16_t)j, facts[j].tmpl, facts[j].elements);
    }
    for (size_t id = t->modelSize + 1u; id < t->numWires; id++) {
        rematWire_t *w = &t->wires[id];
        if (w->inheritFrom != REMAT_NONE) {
            continue; /* derived from the LIVE ACT header when its range opens (rematWireBind) */
        }
        copyGradShape(w->hdr->shape, t->wires[w->index].hdr->shape);
        writeWireConfig(t, (uint16_t)id, facts[id].tmpl, facts[id].elements);
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
                wireKindName(w->kind), (unsigned)w->index, field, built, live);
    exit(1);
}

static void exitInputKeyAt(const char *field, size_t k, size_t built, size_t live) {
    PRINT_ERROR("rematWireTableBind: key mismatch on wire ACT 0, field '%s[%zu]': built %zu, "
                "live %zu (a key change needs a fresh init)",
                field, k, built, live);
    exit(1);
}

/* Phase 1 step 1 (spec §3.4): the model facts and ACT 0, before any shape is
 * derived. With ACT 0's rank and every layer type unchanged, each derived rank
 * is a pure function of them, so the built maxRank bounds the scratch.
 * backwardTop and hasBackward follow from the compared facts, and so do the
 * per-wire kind and rank (plan Assumption 9). */
static void requireModelKey(const rematWireTable_t *t, layer_t **model, size_t n, lossFuncType_t lt,
                            const tensor_t *input) {
    if (n != t->modelSize) {
        exitModelKey("modelSize", t->modelSize, n);
    }
    if (lt != t->lossType) {
        exitModelKey("lossType", (size_t)t->lossType, (size_t)lt);
    }
    for (size_t i = 0; i < n; i++) {
        if ((uint8_t)model[i]->type != t->layerType[i]) {
            exitModelKeyAt("layerType", i, t->layerType[i], (size_t)model[i]->type);
        }
    }
    size_t deepest;
    ptrdiff_t top;
    rematBackwardRange(model, n, lt, &deepest, &top);
    if (deepest != t->deepest) {
        exitModelKey("deepest", t->deepest, deepest);
    }
    for (size_t i = 0; i < n; i++) {
        size_t frozen = layerIsFrozen(model[i]) ? 1u : 0u;
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

/* Phase 2 (spec §3.4, D54): the full per-wire key, before phase 3 writes
 * anything. numGroups <= expCapacity is the only capacity compare. */
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
                        wireKindName(w->kind), (unsigned)w->index, f->numGroups, w->expCapacity);
            exit(1);
        }
    }
}

void rematWireTableBind(rematWireTable_t *t, layer_t **model, size_t n, lossFuncType_t lt,
                        tensor_t *input) {
    requireModelKey(t, model, n, lt, input);
    /* O(maxRank) stack (four size_t[maxRank] shape arrays); the per-wire
     * facts live in the table block. */
    rematWireFact_t *facts = t->bindScratch;
    numberWires(facts, model, t->modelSize, t->deepest, t->backwardTop, t->hasBackward);
    deriveFacts(facts, t->numWires, model, t->modelSize, input, t->maxRank);
    requireWireKey(t, facts);
    writeHeaders(t, model, input, facts);
}

void rematWireTableUnbind(rematWireTable_t *t) {
    t->wires[0].hdr = NULL;
}

/* Names a record in a size-overflow exit (the checked helpers take a rematWireFact_t). */
static rematWireFact_t nameOf(const rematWire_t *w) {
    return (rematWireFact_t){.kind = w->kind, .index = w->index};
}

/* Checked like every size product (D60), via the same elementsOf phase 1
 * uses. Unreachable in practice: the source ACT header was derived at this
 * call's table bind from checked facts. */
static size_t liveElements(const shape_t *shape, const rematWire_t *w) {
    rematWireFact_t name = nameOf(w);
    return elementsOf(shape, &name);
}

/* Spec §3.4 phase 3 item 3 / §3.10: today's post-forward initGradTensor
 * timing, so a producer that wrote a config field of its output is seen.
 * D54 order: every check (dtype, rank, the live payload bytes, and inside
 * bindBfpInto the grouping and the C2 capacity) runs before the first slab
 * write, so the config is written before the shape. */
static void deriveInheritedHeader(rematWireTable_t *t, uint16_t id) {
    rematWire_t *w = &t->wires[id];
    const tensor_t *src = t->wires[w->inheritFrom].hdr;
    if ((uint8_t)src->quantization->type != w->dtype) {
        PRINT_ERROR("rematWireBind: wire %s %u field 'dtype': its source ACT %u is live as dtype "
                    "%d, the slab reserved dtype %u",
                    wireKindName(w->kind), (unsigned)w->index, (unsigned)w->inheritFrom,
                    (int)src->quantization->type, (unsigned)w->dtype);
        exit(1);
    }
    if (src->shape->numberOfDimensions != w->rank) {
        PRINT_ERROR("rematWireBind: wire %s %u field 'rank': its source ACT %u is live at rank "
                    "%zu, the slab reserved rank %u",
                    wireKindName(w->kind), (unsigned)w->index, (unsigned)w->inheritFrom,
                    src->shape->numberOfDimensions, (unsigned)w->rank);
        exit(1);
    }
    size_t elements = liveElements(src->shape, w);
    rematWireFact_t name = nameOf(w);
    size_t liveBytes = wireBytes(src->quantization, elements, &name);
    if (liveBytes != w->bytes) {
        PRINT_ERROR("rematWireBind: wire %s %u field 'bytes': its source ACT %u is live at %zu "
                    "bytes, the slab reserved %zu",
                    wireKindName(w->kind), (unsigned)w->index, (unsigned)w->inheritFrom, liveBytes,
                    w->bytes);
        exit(1);
    }
    writeWireConfig(t, id, src->quantization, elements);
    copyGradShape(w->hdr->shape, src->shape);
}

#ifdef ODT_REMAT_VERIFY
/* Test builds (spec §3.10, §7.1). Poison at Bind stops calloc zeros from
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
    if (rec->hdr->data != NULL) {
        PRINT_ERROR("rematWireBind: wire %s %u is already bound", wireKindName(rec->kind),
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
    /* Checked (D60), though each wire counts at most once (a bound wire cannot
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
    if (rec->hdr->data == NULL) {
        PRINT_ERROR("rematWireRelease: wire %s %u is not bound", wireKindName(rec->kind),
                    (unsigned)rec->index);
        exit(1);
    }
    if (rec->bytes > t->liveBytes) {
        PRINT_ERROR("rematWireRelease: wire %s %u live bytes would underflow (released more than "
                    "bound since the last table bind)",
                    wireKindName(rec->kind), (unsigned)rec->index);
        exit(1);
    }
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
