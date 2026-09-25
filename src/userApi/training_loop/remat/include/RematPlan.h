#ifndef ODT_REMAT_PLAN_H
#define ODT_REMAT_PLAN_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "Layer.h"
#include "LossFunction.h"
#include "Quantization.h"
#include "Tensor.h"

/* Internal to the remat libraries (#4, spec §3-§4): the row inits, the
 * dispatch, RematCheck and tests include it. Placement-free: no byte offset
 * lives here (ARENA keeps its offsets privately, spec §5.5).
 *
 * BORROWED STORAGE (Codex F1): every slab header's shape arrays,
 * quantization_t, qConfig and BFP exponent array is interior storage of the
 * ONE table block. A slab header, or any quantization_t reachable from one,
 * must never reach freeTensor, freeShape, freeQuantization,
 * freeReservedMemory or a deserialize destination (deserializeQConfig
 * reallocates exponents): rematWireTableFree releases the whole table with a
 * single freeReservedMemory. bfpQConfig_t has no owning/borrowed flag, so this
 * rule is the only guard.
 *
 * SIZE ARITHMETIC (D60, Codex N1): every size product and sum RematPlan
 * computes itself is overflow-checked and exits naming the wire and the
 * quantity. An overflow inside a layer's calcOutputShape callback (e.g.
 * convTranspose1dOutputLength, SlidingWindow1d.c:120-124) is the layer's
 * responsibility, tracked with spec §16.1 item 7. */

#define REMAT_NONE ((uint16_t)0xFFFFu)

typedef enum rematWireKind { REMAT_WIRE_ACT = 0, REMAT_WIRE_GRAD } rematWireKind_t;

typedef struct rematWire {
    uint8_t kind;         /* rematWireKind_t */
    uint8_t dtype;        /* qtype_t; key */
    uint8_t rank;         /* key */
    uint8_t borrowed;     /* ACT 0 only: the caller's tensor; never bound, released or poisoned */
    uint16_t index;       /* j of ACT j / GRAD j */
    uint16_t inheritFrom; /* GRAD whose config comes from a LIVE ACT header when it is bound (the
                             loss seed, Flatten's dx): that ACT's index; REMAT_NONE otherwise */
    size_t bytes;         /* exact dtype-aware payload bytes, dims[0] = B (key); never padded */
    size_t expCapacity; /* BFP exponent slots reserved in the slab; a bind needs numGroups <= it */
    tensor_t *hdr; /* slab header; ACT 0: the caller's tensor between bind and unbind, else NULL */
} rematWire_t;

/* What a wire takes from a model and an input (spec §3.4 phase 1): the
 * derivation record of rematWireTableInit and of every rematWireTableBind.
 * Internal to RematWireTable.c; declared here so tests can size the
 * derivation scratch, and so rematWireTable_t can hold the bind scratch. */
typedef struct rematWireFact {
    const quantization_t *tmpl; /* whose config this wire takes */
    size_t elements;
    size_t bytes;
    size_t numGroups; /* BFP slab wires only, else 0 */
    uint16_t index;
    uint16_t inheritFrom;
    uint8_t kind;
    uint8_t dtype;
    uint8_t rank;
} rematWireFact_t;

typedef struct rematWireTable {
    size_t modelSize;
    lossFuncType_t lossType;
    size_t deepest; /* built facts: a bind compares them with the live model */
    ptrdiff_t backwardTop;
    bool hasBackward;
    size_t numWires;
    rematWire_t *wires; /* ACT j at id j (0..n), then GRAD n, top, top-1, ..., deepest+1 */
    uint16_t *gradIdOf; /* [n+1]: id of GRAD j, REMAT_NONE if it does not exist */
    uint8_t *layerType; /* [n], key */
    uint8_t *frozen;    /* [n], key */
    size_t inputRank;   /* ACT 0 key: rank, dims, order, dtype */
    size_t *inputDims;
    size_t *inputOrder;
    uint8_t inputType;
    uint8_t maxRank; /* max(inputRank, 2): bounds every wire rank, sizes the bind scratch */
    rematWireFact_t *bindScratch; /* [numWires], inside this block: every bind's phase-1 facts */
    size_t slabBytes;             /* the whole table block, padding included, no trailing pad */
} rematWireTable_t;

typedef struct rematBfpGroups {
    size_t numGroups;
    size_t groupSize;
} rematBfpGroups_t;

/* THE BFP wire-grouping rule (the driver inlines it at CalculateGradsSequential.c:231-246 and
 * :320-333, InferenceApi.c:78-91): a template's widths are shape-agnostic, so a wire derives
 * its own grouping from its element count. groupSize 0, or == elements, is per-tensor {1, 0}
 * (the derived {1, N} would break the config grammar); otherwise groupSize must divide the
 * elements, else it exits naming the wire. */
rematBfpGroups_t rematBfpWireGrouping(const bfpQConfig_t *tmpl, size_t elements, uint8_t kind,
                                      size_t index);

/* deepest = deepestTrainableIndex (n = nothing trains); top = n-1, or n-2
 * under CROSS_ENTROPY: the positional rule of CalculateGradsSequential.c:77-80
 * in SIGNED arithmetic (n == 1 under CE gives -1, D20). One shared function:
 * the plan uses it on the built model, the checker on the live one. */
void rematBackwardRange(layer_t **model, size_t n, lossFuncType_t lt, size_t *deepest,
                        ptrdiff_t *top);

/* Sizes and places the whole table in ONE reserveMemory block: records, key,
 * and one header per slab wire. The headers are linked, but their content
 * (dims, config) is written by every rematWireTableBind. Returns false only
 * when reserveMemory fails. */
bool rematWireTableInit(rematWireTable_t **out, layer_t **model, size_t n, lossConfig_t loss,
                        const tensor_t *inputLike);
/* NULL-safe. One freeReservedMemory; never freeTensor on a slab header. */
void rematWireTableFree(rematWireTable_t *t);

/* Re-derives every header's CONTENT from the live model and input (spec §3.4):
 * shapes, config fields from the current templates, fresh dynamic state (SYM
 * scale 1, BFP exponents at the stored bias). ACT 0 = input, verbatim.
 * Inherited GRAD headers are derived later, by rematWireBind. Reserves
 * nothing. */
void rematWireTableBind(rematWireTable_t *t, layer_t **model, size_t n, lossFuncType_t lt,
                        tensor_t *input);
void rematWireTableUnbind(rematWireTable_t *t); /* ACT 0 hdr = NULL */

/* Read-only accessors (rows, the checker, tests). */
tensor_t *rematWireHdr(const rematWireTable_t *t, uint16_t w);
tensor_t *rematActHdr(const rematWireTable_t *t, size_t j);  /* j == 0: the bound input or NULL */
tensor_t *rematGradHdr(const rematWireTable_t *t, size_t j); /* j == n: the seed; NULL if absent */
uint16_t rematActId(const rematWireTable_t *t, size_t j);
uint16_t rematGradId(const rematWireTable_t *t, size_t j); /* REMAT_NONE if absent */
size_t rematWireBytes(const rematWireTable_t *t, uint16_t w);

#endif // ODT_REMAT_PLAN_H
