#define SOURCE_FILE "REMAT_SCHEDULER"

#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>

#include "Common.h"
#include "RematCheckedSize.h"
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

void rematBegin(rematScheduler_t *s, layer_t **model, size_t n, lossConfig_t loss,
                tensor_t *input) {
    if (s->fns == NULL || s->wires == NULL || s->plan == NULL) {
        PRINT_ERROR("rematBegin: scheduler not initialised (never initialised, or its init "
                    "returned false and was ignored)");
        exit(1);
    }
    if (s->inCall) {
        PRINT_ERROR("rematBegin: scheduler '%s' re-entered", s->fns->name);
        exit(1);
    }
    rematWireTableBind(s->wires, model, n, loss.funcType, input);
    s->inCall = true;
    s->fns->begin(s);
}

/* next() and done() alternate, and done() answers exactly the step next()
 * handed out. Checked here rather than in the rows: the protocol is the
 * same for every row, including those whose steps are not the plan's
 * (EVICT's synthesized REFORWARDs). */
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

void rematRequireWalkComplete(const rematScheduler_t *s, const char *row) {
    const rematProgram_t *p = &s->plan->train; /* PR3: the program of the call's mode */
    if (s->walk.step != p->numSteps || s->walk.close != p->numRanges) {
        PRINT_ERROR("remat[%s]: rematEnd before the walk completed: %zu of %zu steps done, "
                    "%zu of %zu ranges closed",
                    row, s->walk.step, p->numSteps, s->walk.close, p->numRanges);
        exit(1);
    }
}

/* Every size sum is overflow-checked. Resident block sizes cannot reach SIZE_MAX
 * together, so this exit is unreachable and has no dedicated test. */
static size_t reportAdd(size_t a, size_t b) {
    size_t out;
    if (!checkedAddSize(a, b, &out)) {
        PRINT_ERROR("rematSchedulerReport: size overflow computing metadataBytes");
        exit(1);
    }
    return out;
}

/* The ARENA half of the report: each flag derived from the block it names
 * (the offsets block, then the arena data block). */
static void reportArena(const rematScheduler_t *s, const rematProgram_t *p, rematReport_t *out) {
    if (s->row.arena.offsets != NULL) {
        /* Cannot wrap: numRanges < REMAT_NONE, and init reserved this product. */
        out->metadataBytes = reportAdd(out->metadataBytes, p->numRanges * sizeof(size_t));
    }
    if (s->row.arena.bytes != 0u) {
        out->placed = true;
        out->arenaBytes = s->row.arena.bytes;
        /* Neither difference can wrap: placed >= exact bytes per wire, and the
         * verified co-live ranges fit disjointly inside the arena. */
        out->arenaPadBytes = s->row.arena.peakPlacedBytes - p->peakLiveBytes;
        out->arenaGapBytes = s->row.arena.bytes - s->row.arena.peakPlacedBytes;
    }
    out->dataReserved = s->row.arena.base != NULL;
}

void rematSchedulerReport(const rematScheduler_t *s, rematReport_t *out) {
    *out = (rematReport_t){.type = s->type};
    if (s->wires == NULL || s->plan == NULL) {
        return; /* init stopped before the plan existed: no field is valid */
    }
    const rematProgram_t *p = &s->plan->train;
    out->policy = s->plan->policy;
    out->planned = true;
    out->numSteps = p->numSteps;
    out->peakLiveBytes = p->peakLiveBytes;
    out->observedPeakLiveBytes = s->wires->observedPeakLiveBytes;
    out->metadataBytes = reportAdd(s->wires->slabBytes, s->plan->blockBytes);
    switch (s->type) {
    case REMAT_ARENA:
        reportArena(s, p, out);
        break;
    case REMAT_HEAP:
        /* No placement and no resident block: a planned HEAP is ready to run,
         * its arena fields 0 by definition. */
        out->placed = true;
        out->dataReserved = true;
        break;
    }
}
