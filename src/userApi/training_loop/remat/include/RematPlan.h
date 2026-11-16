#ifndef ODT_REMAT_PLAN_H
#define ODT_REMAT_PLAN_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "Layer.h"
#include "LossFunction.h"
#include "Model.h"
#include "Quantization.h"
#include "RematScheduler.h"
#include "Tensor.h"

/* Internal to the remat libraries: the row inits, the dispatch, RematCheck
 * and tests include it. Placement-free: no byte offset lives here (ARENA
 * keeps its offsets privately).
 *
 * BORROWED STORAGE: every slab header's shape arrays, quantization_t, qConfig
 * and BFP exponent array is interior storage of the ONE table block. Never
 * pass a slab header, or a quantization_t reachable from one, to freeTensor,
 * freeShape, freeQuantization, freeReservedMemory or a deserialize
 * destination (deserializeQConfig reallocates exponents). bfpQConfig_t has no
 * owning/borrowed flag, so this rule is the only guard.
 *
 * SIZE ARITHMETIC: every size product and sum RematPlan
 * computes itself is overflow-checked and exits naming the wire and the
 * quantity. An overflow inside a layer's calcOutputShape callback (e.g.
 * convTranspose1dOutputLength, SlidingWindow1d.c:120-124) is the layer's
 * responsibility. */

#define REMAT_NONE ((uint16_t)0xFFFFu)

/* The range budget, one definition for the plan and the rows: a
 * REMAT_PLAN_SEQUENCE or REMAT_PLAN_RETAIN_LIST plan above it exits in
 * rematPlanBuild before its block is reserved, under every row, and ARENA
 * checks every plan against it at init. HEAP admits STORE_ALL and LIVENESS
 * (at most numWires - 1 ranges) of any size. It bounds the quadratic plan work of the
 * two capped kinds: the end-order sort O(R^2), grammar rule 4's coverage scan
 * O(S * R) and ARENA's placement check O(R^2). A plan fact, not an OOM. Host
 * test builds may raise it; its value belongs in the plan dump, like
 * ODT_WIRE_ALIGN. */
#ifndef ODT_REMAT_MAX_RANGES
#define ODT_REMAT_MAX_RANGES 1024u
#endif
_Static_assert(ODT_REMAT_MAX_RANGES < REMAT_NONE,
               "range ids are uint16_t, and REMAT_NONE means no range");

typedef enum rematWireKind { REMAT_WIRE_ACT = 0, REMAT_WIRE_GRAD, REMAT_WIRE_PART } rematWireKind_t;

typedef struct rematWire {
    uint8_t kind;         /* rematWireKind_t */
    uint8_t dtype;        /* qtype_t; key */
    uint8_t rank;         /* key */
    uint8_t borrowed;     /* ACT 0 only: the caller's tensor; never bound, released or poisoned */
    uint16_t index;       /* the buffer b of ACT, GRAD or PART b */
    uint16_t inheritFrom; /* GRAD or PART whose config comes from a LIVE ACT header when it
                             is bound (the loss seed, a Flatten's dx): that ACT's index;
                             REMAT_NONE otherwise (the CrossEntropy seed of a BFP-declaring head
                             is the last Softmax's dx: its propLossQ) */
    size_t bytes;         /* exact dtype-aware payload bytes, dims[0] = B (key); never padded */
    size_t expCapacity; /* BFP exponent slots reserved in the slab; a bind needs numGroups <= it */
    uint32_t bindGen;   /* ++ on every rematWireBind; 0 after every rematWireTableBind */
    tensor_t *hdr; /* slab header; ACT 0: the caller's tensor between bind and unbind, else NULL */
} rematWire_t;

/* What a wire takes from a model and an input (phase 1 of the bind): the
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
    bool hasBackward;
    uint8_t isChain; /* modelIsChain of the built model: then the wiring below is NULL and 0 */
    /* rematWireTableInitEval: ACT wires only, every gradient flag 0 and every dx slot NONE; the
     * bind keys neither the loss type nor runsBackward */
    bool evalOnly;
    size_t numWires;
    rematWire_t *wires;    /* ACT j at id j (0..n), then GRAD and PART wires in production order */
    uint16_t *gradIdOf;    /* [n+1]: the GRAD holding buffer b's gradient, else REMAT_NONE; under
                              CROSS_ENTROPY the seed is the Softmax input's gradient too */
    uint8_t *layerType;    /* [n], key */
    uint8_t *frozen;       /* [n], key */
    uint8_t *runsBackward; /* [n], key (modelGradFlags) */
    uint8_t *needsGrad;    /* [n + 1], follows from the key */
    uint8_t *readsInput;   /* [n], layerBackwardReadsInput: from type and frozen, both keyed */
    uint16_t *dxWire; /* [n * MODEL_MAX_ARITY]: the wire node j's BACKWARD writes for input slot i,
                         REMAT_NONE = no dx (rematDeriveGradWires) */
    uint8_t *dxMode;  /* [n * MODEL_MAX_ARITY]: opMode_t of that slot; READ when nothing runs */
    /* The table's own copy of the model's CSR (the model's arrays may be a temporary): node j
     * reads inputIds[inputStart[j] .. inputStart[j+1]) */
    uint16_t *inputStart; /* [n + 1] */
    uint16_t *inputIds;   /* [numEdges] */
    size_t numEdges;
    size_t inputRank; /* ACT 0 key: rank, dims, order, dtype */
    size_t *inputDims;
    size_t *inputOrder;
    uint8_t inputType;
    uint8_t maxRank;    /* max(inputRank, 2): bounds every wire rank, sizes actShapes */
    shape_t *actShapes; /* [n + 1]: the shape pass on the bound input; dims, order in-block */
    rematWireFact_t *bindScratch; /* [numWires], inside this block: every bind's phase-1 facts */
    size_t liveBytes;             /* rises in rematWireBind, falls in rematWireRelease */
    size_t observedPeakLiveBytes; /* its high-water mark since the last rematWireTableBind */
    size_t slabBytes;             /* the whole table block, padding included, no trailing pad */
    /* The plan's in-place marks (rematPlan_t.inPlace), borrowed for a training
     * call, else NULL. Rows bind only ranges and never see a chain, so the SDK
     * binds and releases a chain's members with its root. */
    const uint8_t *inPlace;
} rematWireTable_t;

typedef struct rematBfpGroups {
    size_t numGroups;
    size_t groupSize;
} rematBfpGroups_t;

/* THE BFP wire-grouping rule of every training wire (InferenceApi.c's initBufferOutput and
 * initBufferInput keep a copy for the inference buffers): a template's widths are
 * shape-agnostic, so a wire derives its own grouping from its element count. groupSize 0,
 * or groupSize == elements, is per-tensor {1, 0} (the derived {1, N} would break the config
 * grammar); otherwise groupSize must divide the elements, else it exits naming the wire. */
rematBfpGroups_t rematBfpWireGrouping(const bfpQConfig_t *tmpl, size_t elements, uint8_t kind,
                                      size_t index);

/* Sizes and places the whole table in ONE reserveMemory block: records, key,
 * and one header per slab wire. The headers are linked, but their content
 * (dims, config) is written by every rematWireTableBind. Under CROSS_ENTROPY, when a backward
 * runs, the skipped last layer must be the Softmax and declare one of the two model heads
 * (requireCrossEntropyHead), else it exits; every bind checks the live model the same way. */
void rematWireTableInit(rematWireTable_t **out, const model_t *m, lossConfig_t loss,
                        const tensor_t *inputLike);
/* The loss-free table of an inference scheduler: the ACT wires alone, no loss head check.
 * Validates the model like rematWireTableInit and refuses a non-FLOAT32 ACT that an ADD reads.
 * Its bind keys no loss type, and its plan comes from rematPlanBuildEval only. */
void rematWireTableInitEval(rematWireTable_t **out, const model_t *m, const tensor_t *inputLike);
/* NULL-safe. One freeReservedMemory; never freeTensor on a slab header. */
void rematWireTableFree(rematWireTable_t *t);

/* Re-derives every header's CONTENT from the live model and input:
 * shapes, config fields from the current templates, fresh dynamic state (SYM
 * scale 1, BFP exponents at the stored bias). ACT 0 = input, verbatim.
 * Inherited GRAD headers are derived later, by rematWireBind. Reserves
 * nothing. An EVAL bind runs the same key check and ACT derivation but writes
 * no GRAD header: eval binds no GRAD wire. On an eval-only table lt is ignored: the
 * key holds no loss type and no runsBackward. */
void rematWireTableBind(rematWireTable_t *t, const model_t *m, lossFuncType_t lt, rematMode_t mode,
                        tensor_t *input);
void rematWireTableUnbind(rematWireTable_t *t); /* ACT 0 hdr = NULL, inPlace = NULL */

/* The row SDK: the ONLY writers of a header's ->data. Call after this
 * call's rematWireTableBind; bind and release chain roots only; bytes must be
 * non-NULL and satisfy the wire's size and alignment. Bind derives an
 * inherited GRAD header (the seed, Flatten's dx) from the LIVE ACT header,
 * whose data need not be resident (under LIVENESS a Flatten dx GRAD binds
 * after its source ACT died); the source's dtype, rank and payload size are
 * the ones the table bind checked. Both exit on the borrowed ACT 0, on an
 * unbalanced call, on a NULL bytes and on a wire id at or above numWires. While the table borrows a
 * plan's in-place marks (a training call), Bind and Release of a chain's root
 * also bind and release its members: each member takes the root's bytes and
 * keeps its own header, liveBytes counts the chain once, and both calls exit
 * when they name a member. */
void rematWireBind(rematWireTable_t *t, uint16_t w, uint8_t *bytes);
void rematWireRelease(rematWireTable_t *t, uint16_t w);

/* Read-only accessors (rows, the checker, tests). rematWireHdr and
 * rematWireBytes take ids below numWires too, but -- unlike Bind/Release --
 * do not check it (read every step, off the exit path). */
tensor_t *rematWireHdr(const rematWireTable_t *t, uint16_t w);
tensor_t *rematActHdr(const rematWireTable_t *t, size_t j);  /* j == 0: the bound input or NULL */
tensor_t *rematGradHdr(const rematWireTable_t *t, size_t j); /* j == n: the seed; NULL if absent */
uint16_t rematActId(const rematWireTable_t *t, size_t j);
uint16_t rematGradId(const rematWireTable_t *t, size_t j); /* REMAT_NONE if absent */
size_t rematWireBytes(const rematWireTable_t *t, uint16_t w);
/* "ACT", "GRAD" or "PART": the one wire-kind name every remat message prints. */
const char *rematWireKindName(uint8_t kind);

/* One inclusive lifetime of a wire: begin = the step that writes it, end = its
 * last reader (policy-dependent). STORE_ALL and LIVENESS give each slab wire
 * one range; SEQUENCE and RETAIN_LIST plans open one range per write, so a
 * recomputed wire has several, which never overlap. An in-place member
 * (rematPlan_t.inPlace) has none: its chain root's range runs on to the last
 * read of the chain. No placement (ARENA offsets are private to the row). */
typedef struct rematRange {
    uint16_t wire, begin, end;
} rematRange_t;

/* One operand of a step's slice. Its wire is REMAT_NONE only for a write: that
 * input gets no dx. */
typedef struct rematOperand {
    uint16_t wire;
    uint8_t mode; /* opMode_t */
} rematOperand_t;

/* Where each node's canonical slices start in its program's pool, REMAT_NONE
 * for none. FORWARD and every REFORWARD of a node share one slice. Entry n
 * holds the loss: forward = LOSS_FORWARD, backward = LOSS_BACKWARD. */
typedef struct rematNodeOps {
    uint16_t forward;
    uint16_t backward;
    uint16_t sum; /* always REMAT_NONE on a chain */
} rematNodeOps_t;

typedef struct rematProgram {
    size_t numSteps;
    rematStep_t *steps;
    size_t numOperands;       /* <= UINT16_MAX: offsets; UINT16_MAX is REMAT_NONE */
    rematOperand_t *operands; /* each step's slice: its reads, then its writes */
    rematNodeOps_t *nodeOps;  /* [n + 1] */
    size_t numRanges;
    rematRange_t *ranges; /* begin-sorted, ties by wire id */
    uint16_t *endOrder;   /* range ids sorted by end, ties by wire id */
    size_t peakLiveBytes; /* exact bytes, ACT 0 excluded: the POET x-axis, wires_peak_b */
} rematProgram_t;

typedef struct rematPlan {
    rematPlanPolicy_t policy;
    rematProgram_t train;
    /* FORWARD 0..n-1, LOSS_FORWARD; ACT j lives [FORWARD(j-1), FORWARD(j)],
     * ACT n [FORWARD(n-1), LOSS_FORWARD]; no GRAD range. The same for every
     * policy: eval never recomputes. */
    rematProgram_t eval;
    /* The TRAIN program's REFORWARD steps and layerForwardCost summed over its
     * FORWARD and over its REFORWARD steps: computed at build, because
     * rematSchedulerReport sees no model. */
    size_t refwdSteps;
    size_t forwardCost;
    size_t recomputeCost;
    size_t blockBytes; /* the one plan block, struct included */
    /* [numWires] inside the block when the spec set inPlaceDx, else NULL.
     * inPlace[w] == 1: wire w is an in-place member, which its layer's
     * BACKWARD writes into the bytes of wire w - 1, the gradient that step
     * reads. The first unmarked wire below a member is its chain's root, and
     * the root's range is the chain's only one. TRAIN only: an EVAL program
     * has no backward step. */
    uint8_t *inPlace;
} rematPlan_t;

/* Generates the TRAIN and the EVAL program for the table's built model in ONE
 * reserveMemory block. The caller guarantees `m` is the model the table was
 * just built on: the deriver reads only the table, `m` feeds the grammar and
 * the costs, so a different model is not detected. A DAG model plans under
 * STORE_ALL and LIVENESS only: SEQUENCE, RETAIN_LIST and inPlaceDx exit on it,
 * since recompute and in-place read chain positions. */
void rematPlanBuild(rematPlan_t **out, const rematWireTable_t *t, const model_t *m,
                    const rematPlanSpec_t *spec);
/* The plan of an eval-only table: the EVAL program alone, the same as rematPlanBuild's, and a
 * zeroed TRAIN program (no steps, so every TRAIN number of the report is 0). Same `m` rule and
 * return value as rematPlanBuild. */
void rematPlanBuildEval(rematPlan_t **out, const rematWireTable_t *t, const model_t *m);
void rematPlanFree(rematPlan_t *p); /* NULL-safe */

/* The program a call in `mode` walks. */
const rematProgram_t *rematPlanProgram(const rematPlan_t *p, rematMode_t mode);

/* The activation-only peak of a program: the maximum over its steps of the
 * live bytes of ACT wires (ACT 0, the caller's input, has no range); the
 * same scan as peakLiveBytes, not the ACT share at that peak's step. */
size_t rematProgramActPeakBytes(const rematProgram_t *p, const rematWireTable_t *t);

/* O(1) amortised: the next range with begin == w->step (resp. end == w->step,
 * in endOrder), or REMAT_NONE. Call both every step, openings first. A row
 * zeroes its rematWalk_t at the start of each call. */
size_t rematWalkOpening(const rematProgram_t *p, rematWalk_t *w);
size_t rematWalkClosing(const rematProgram_t *p, rematWalk_t *w);

/* Validates a completed program once, before it can run: step ordering,
 * operand coverage, recomputation legality, the fan-out contributions of a
 * GRAD (one WRITE opens it; later ones ACC from an accumulating row or write a
 * PART the next step SUMs in) and alias-chain lifetimes (rules 1-4 and 6-8).
 * Independent of RematCheck and of the deriver -- it reads wiring, read-set
 * and needs-grad off `m` with its own code -- so a mutation in one is caught
 * by the other; tests also run it on tampered programs. Exits naming the step
 * or range index and the rule. inPlace is the plan's marks (NULL: none); marks
 * on a DAG exit. Its counts live in one transient reserveMemory block. */
void rematPlanValidateGrammar(const rematProgram_t *p, const rematWireTable_t *t, const model_t *m,
                              rematMode_t mode, const uint8_t *inPlace);

#endif // ODT_REMAT_PLAN_H
