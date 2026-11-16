#define SOURCE_FILE "REMAT_SCHEDULER"

#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>

#include "Common.h"
#include "RematPlan.h"
#include "RematRows.h"
#include "RematScheduler.h"

const rematSchedulerFunctions_t rematSchedulerFunctions[] = {
    [REMAT_ARENA] = {"arena", rematArenaBegin, rematArenaNext, rematArenaDone, rematArenaEnd,
                     rematArenaDeinit},
    [REMAT_HEAP] = {"heap", rematHeapBegin, rematHeapNext, rematHeapDone, rematHeapEnd,
                    rematHeapDeinit},
};
/* The enum has no count member, so appending a row means updating
 * REMAT_HEAP below to the new last member, or a missing entry goes unseen. */
_Static_assert(sizeof rematSchedulerFunctions / sizeof rematSchedulerFunctions[0] ==
                   REMAT_HEAP + 1u,
               "one vtable entry per rematSchedulerType_t member");

static void requireInCall(const rematScheduler_t *s, const char *call) {
    if (!s->inCall) {
        PRINT_ERROR("remat[%s]: %s outside a call (no rematBegin since the last rematEnd)",
                    s->fns == NULL ? "uninitialised" : s->fns->name, call);
        exit(1);
    }
}

/* Both begins: the guards name the verb the caller used. The mode is set
 * before the row's begin, which walks that mode's program. */
static void beginCall(rematScheduler_t *s, const char *verb, const model_t *m, lossFuncType_t lt,
                      rematMode_t mode, tensor_t *input) {
    if (s->fns == NULL || s->wires == NULL || s->plan == NULL) {
        PRINT_ERROR("%s: scheduler not initialised (never initialised, or deinitialised)", verb);
        exit(1);
    }
    if (s->inCall) {
        PRINT_ERROR("%s: scheduler '%s' re-entered", verb, s->fns->name);
        exit(1);
    }
    if (mode == REMAT_MODE_TRAIN && s->wires->evalOnly) {
        PRINT_ERROR("%s: eval-only scheduler '%s' (rematHeapInitEval) has no TRAIN program; "
                    "init a training scheduler to train",
                    verb, s->fns->name);
        exit(1);
    }
    rematWireTableBind(s->wires, m, lt, mode, input);
    /* After the table bind, for a training call only: an evaluation program
     * has no backward step, so no chain. */
    s->wires->inPlace = (mode == REMAT_MODE_TRAIN) ? s->plan->inPlace : NULL;
    s->mode = mode;
    s->inCall = true;
    s->fns->begin(s);
}

void rematBegin(rematScheduler_t *s, const model_t *m, lossConfig_t loss, tensor_t *input) {
    beginCall(s, "rematBegin", m, loss.funcType, REMAT_MODE_TRAIN, input);
}

void rematBeginEval(rematScheduler_t *s, const model_t *m, lossFuncType_t lossType,
                    tensor_t *input) {
    beginCall(s, "rematBeginEval", m, lossType, REMAT_MODE_EVAL, input);
}

/* next() and done() alternate, and done() answers exactly the step next()
 * handed out. Checked here rather than in the rows: the protocol is the
 * same for every row. */
bool rematNext(rematScheduler_t *s, rematStep_t *st) {
    requireInCall(s, "rematNext");
    if (s->handedOut) {
        PRINT_ERROR("remat[%s]: rematNext while (kind %u, layer %u) is still handed out (rematDone "
                    "not called)",
                    s->fns->name, (unsigned)s->handed.kind, (unsigned)s->handed.layer);
        exit(1);
    }
    if (!s->fns->next(s, st)) {
        return false;
    }
    s->handed = *st;
    s->handedOut = true;
    return true;
}

void rematDone(rematScheduler_t *s, const rematStep_t *st) {
    requireInCall(s, "rematDone");
    if (!s->handedOut) {
        PRINT_ERROR("remat[%s]: rematDone for (kind %u, layer %u) with no step handed out "
                    "(rematNext not called since the last rematDone, or it returned false)",
                    s->fns->name, (unsigned)st->kind, (unsigned)st->layer);
        exit(1);
    }
    if (st->kind != s->handed.kind || st->layer != s->handed.layer) {
        PRINT_ERROR("remat[%s]: rematDone for a step next() did not hand out: next() handed out "
                    "(kind %u, layer %u), done() got (kind %u, layer %u)",
                    s->fns->name, (unsigned)s->handed.kind, (unsigned)s->handed.layer,
                    (unsigned)st->kind, (unsigned)st->layer);
        exit(1);
    }
    s->fns->done(s, st);
    s->handedOut = false;
}

void rematEnd(rematScheduler_t *s) {
    requireInCall(s, "rematEnd");
    s->fns->end(s);
    rematWireTableUnbind(s->wires);
    s->inCall = false;
}

void rematSchedulerDeinit(rematScheduler_t *s) {
    if (s == NULL || s->fns == NULL) {
        return; /* never initialised, or already deinitialised */
    }
    if (s->inCall) {
        PRINT_ERROR("remat[%s]: rematSchedulerDeinit inside a call", s->fns->name);
        exit(1);
    }
    s->fns->deinit(s);
    rematPlanFree(s->plan);
    rematWireTableFree(s->wires);
    *s = (rematScheduler_t){0};
}

/* walk.open needs no check: grammar rule 6 keeps every range inside the
 * stream and the ranges begin-sorted, so the walk opens all of them by the
 * last step, and a range that closed without opening would already have
 * exited in rematWireRelease (its wire is not bound). */
void rematRequireWalkComplete(const rematScheduler_t *s, const char *row) {
    const rematProgram_t *p = rematPlanProgram(s->plan, s->mode);
    if (s->walk.step != p->numSteps || s->walk.close != p->numRanges) {
        PRINT_ERROR("remat[%s]: rematEnd before the walk completed: %zu of %zu steps done, "
                    "%zu of %zu ranges closed",
                    row, s->walk.step, p->numSteps, s->walk.close, p->numRanges);
        exit(1);
    }
}

static void reportArena(const rematScheduler_t *s, const rematProgram_t *p, rematReport_t *out) {
    /* Cannot wrap: numRanges < REMAT_NONE, and init reserved this product. */
    out->metadataBytes += p->numRanges * sizeof(size_t);
    out->placed = true;
    out->dataReserved = true;
    out->arenaBytes = s->row.arena.bytes;
    /* Neither difference can wrap: placed >= exact bytes per wire, and the
     * verified co-live ranges fit disjointly inside the arena. */
    out->arenaPadBytes = s->row.arena.peakPlacedBytes - p->peakLiveBytes;
    out->arenaGapBytes = s->row.arena.bytes - s->row.arena.peakPlacedBytes;
}

void rematSchedulerReport(const rematScheduler_t *s, rematReport_t *out) {
    *out = (rematReport_t){.type = s->type};
    if (s->wires == NULL || s->plan == NULL) {
        return; /* never initialised, or deinitialised: no field is valid */
    }
    const rematProgram_t *p = &s->plan->train;
    out->policy = s->plan->policy;
    out->planned = true;
    out->numSteps = p->numSteps;
    out->refwdSteps = s->plan->refwdSteps;
    out->forwardCost = s->plan->forwardCost;
    out->recomputeCost = s->plan->recomputeCost;
    out->peakLiveBytes = p->peakLiveBytes;
    out->activationsPeakBytes = rematProgramActPeakBytes(p, s->wires);
    out->observedPeakLiveBytes = s->wires->observedPeakLiveBytes;
    /* Sums of resident block sizes: they cannot wrap. */
    out->metadataBytes = s->wires->slabBytes + s->plan->blockBytes;
    switch (s->type) {
    case REMAT_ARENA:
        reportArena(s, p, out);
        break;
    case REMAT_HEAP:
        /* No placement and no resident block: its arena fields are 0. */
        out->placed = true;
        out->dataReserved = true;
        break;
    }
}
