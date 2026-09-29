#ifndef ODT_REMAT_SCHEDULER_H
#define ODT_REMAT_SCHEDULER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "Layer.h"
#include "LossFunction.h"
#include "Tensor.h"

/* The remat scheduler's public header (#4, spec §5.1): the step, policy, spec
 * and walk types the static plan shares, and the scheduler with its rows
 * (ARENA, HEAP), its const vtable and the rematBegin/Next/Done/End dispatch. */

/* APPEND-ONLY: stored in plan tables and (PR6) caller/ir2c sequences. */
typedef enum rematStepKind {
    REMAT_STEP_FORWARD = 0,   /* layer l: ACT l -> ACT l+1 */
    REMAT_STEP_LOSS_FORWARD,  /* layer == n: output snapshot + loss of ACT n; forward->backward
                                 boundary */
    REMAT_STEP_LOSS_BACKWARD, /* layer == n: ACT n -> GRAD n (the seed) */
    REMAT_STEP_BACKWARD,      /* layer l: (ACT l, GRAD in(l)) -> GRAD l, or grads-only at deepest */
} rematStepKind_t;

/* One record, three roles: plan-table entry, the value a row's next() fills
 * in, and (PR6) the offline sequence record. No tensor pointers: the driver
 * resolves operands positionally. No explicit pad byte until a byte-image
 * consumer (#62 ir2c .rodata) needs one (spec §16.1 item 4d); uint16_t
 * alignment keeps the implicit pad at one byte on every target. */
typedef struct rematStep {
    uint8_t kind;   /* rematStepKind_t */
    uint16_t layer; /* model index; == modelSize for LOSS_*; the table's wire-count guard
                       implies modelSize < UINT16_MAX */
} rematStep_t;
_Static_assert(sizeof(rematStep_t) == 4, "fixed-width step record");

typedef enum rematPlanPolicy { REMAT_PLAN_STORE_ALL = 0, REMAT_PLAN_LIVENESS } rematPlanPolicy_t;

/* NULL, or a zero-initialised struct, means STORE_ALL (the trainingRunOptions_t
 * idiom, TrainingLoopApi.h:113-115). */
typedef struct rematPlanSpec {
    rematPlanPolicy_t policy;
} rematPlanSpec_t;

/* The static-plan cursor: step index, next range to open (begin order), next
 * range to close (endOrder). Defined here because the scheduler struct embeds
 * it. */
typedef struct rematWalk {
    size_t step, open, close;
} rematWalk_t;

typedef enum rematSchedulerType {
    REMAT_ARENA = 0,
    REMAT_HEAP
} rematSchedulerType_t; /* APPEND-ONLY */

typedef struct rematWireTable rematWireTable_t; /* RematPlan.h (internal) */
typedef struct rematPlan rematPlan_t;           /* RematPlan.h (internal) */
typedef struct rematScheduler rematScheduler_t;

typedef void (*rematBeginFn_t)(rematScheduler_t *s); /* runs AFTER the shared bind */
typedef bool (*rematNextFn_t)(rematScheduler_t *s, rematStep_t *step); /* false = stream complete */
typedef void (*rematDoneFn_t)(rematScheduler_t *s, const rematStep_t *step);
typedef void (*rematEndFn_t)(rematScheduler_t *s);
typedef void (*rematDeinitFn_t)(rematScheduler_t *s); /* row-private blocks only */

/* Every slot is mandatory for every row (the Optimizer.h wording). */
typedef struct rematSchedulerFunctions {
    const char *name; /* "arena" | "heap": every violation message names the row */
    rematBeginFn_t begin;
    rematNextFn_t next;
    rematDoneFn_t done;
    rematEndFn_t end;
    rematDeinitFn_t deinit;
} rematSchedulerFunctions_t;

/* Indexed by rematSchedulerType_t. const on purpose: flash-resident on the
 * MCU and no mutable module global -- a deliberate deviation from the
 * non-const optimizerFunctions[] / layerFunctions[] / lossFunctions[]. */
extern const rematSchedulerFunctions_t rematSchedulerFunctions[];

/* Caller-owned (stack, static or a struct field); ONE per concurrently
 * running training stream. */
struct rematScheduler {
    rematSchedulerType_t type;
    /* &rematSchedulerFunctions[type], set by the row's init. A test may point
     * its own instance at a decorator table (D24); the driver validates a
     * wrong table like a correct one. */
    const rematSchedulerFunctions_t *fns;
    rematWireTable_t *wires; /* shared buffer table: one reserveMemory block */
    rematPlan_t *plan;       /* shared static plan: one reserveMemory block, placement-free */
    bool inCall;             /* rematBegin..rematEnd; a re-entry guard, NOT a lock */
    rematWalk_t walk;        /* the static-plan cursor of the current call */
    /* The call protocol, owned by the dispatch for every row: next() handed
     * out `handed`, and done() has not answered it yet. */
    bool handedOut;
    rematStep_t handed;
    union {
        /* ARENA-private (R5). Two blocks (D55 as amended by Codex N3): offsets
         * first, placed into and verified, then the arena data block. */
        struct {
            uint8_t *base;          /* the resident arena data block */
            size_t bytes;           /* its size; non-zero from the verified placement on */
            size_t *offsets;        /* [numRanges] */
            size_t peakPlacedBytes; /* the peak of the placed (8-rounded) sums */
        } arena;
    } row;
};

/* One init per row (the LrScheduler idiom). Returns false iff a reserveMemory
 * failed; s is then safe for rematSchedulerDeinit and rematSchedulerReport,
 * whose flags say which fields are valid. A model the table cannot describe,
 * or whose size arithmetic overflows, exits naming it. */
bool rematArenaInit(rematScheduler_t *s, layer_t **model, size_t n, lossConfig_t loss,
                    const tensor_t *inputLike, const rematPlanSpec_t *spec);
/* The peer row (spec §5.6): table and plan only; one exactly-sized block per
 * range, reserved when the range opens and freed when it closes. */
bool rematHeapInit(rematScheduler_t *s, layer_t **model, size_t n, lossConfig_t loss,
                   const tensor_t *inputLike, const rematPlanSpec_t *spec);
/* NULL-safe and idempotent: the row's blocks (fns->deinit), then the plan and
 * the table. Exits inside a call. */
void rematSchedulerDeinit(rematScheduler_t *s);

/* Feeds the harness keys (spec §14); a field is valid only under the flag
 * that declares it. The three flags imply one another in order (D55 as
 * amended by Codex N3); a did-not-run point records what its flags allow. */
typedef struct rematReport {
    rematSchedulerType_t type;
    rematPlanPolicy_t policy;
    bool planned; /* table + plan built: numSteps, peakLiveBytes, metadataBytes */
    /* ARENA: placement computed + verified (the three arena fields). HEAP:
     * == planned, arena fields 0; it places each range in its own block. */
    bool placed;
    /* ARENA: the resident arena block exists. HEAP: == planned; its blocks are
     * reserved per range at each step, so init leaves nothing pending. */
    bool dataReserved;
    size_t numSteps;
    size_t peakLiveBytes;         /* plan, exact bytes, ACT 0 excluded: POET x-axis, wires_peak_b */
    size_t observedPeakLiveBytes; /* SDK accounting over the last call, every row (P8) */
    size_t arenaBytes;            /* = peakLiveBytes + arenaPadBytes + arenaGapBytes */
    size_t arenaPadBytes;         /* peakPlacedBytes - peakLiveBytes (alignment) -> arena_pad_b */
    size_t arenaGapBytes;         /* arenaBytes - peakPlacedBytes (FFD heuristic) -> arena_gap_b */
    size_t metadataBytes; /* table block + plan block + the offsets block -> wire_metadata_b */
} rematReport_t;
void rematSchedulerReport(const rematScheduler_t *s, rematReport_t *out);

/* THE entry points (the optimizerStep precedent): only drivers call them; raw
 * fns-> calls only in unit tests. rematBegin binds the table (key check,
 * per-bind derivation, ACT 0 = input) before the row's begin; rematEnd
 * unbinds it after the row's end. */
void rematBegin(rematScheduler_t *s, layer_t **model, size_t n, lossConfig_t loss, tensor_t *input);
bool rematNext(rematScheduler_t *s, rematStep_t *step);
void rematDone(rematScheduler_t *s, const rematStep_t *step);
void rematEnd(rematScheduler_t *s);

#endif // ODT_REMAT_SCHEDULER_H
