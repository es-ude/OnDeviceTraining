#define SOURCE_FILE "REMAT_SCHEDULER"

#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>

#include "Common.h"
#include "RematCheckedSize.h"
#include "RematPlan.h"
#include "RematRows.h"
#include "RematScheduler.h"

void rematSchedulerDeinit(rematScheduler_t *s) {
    if (s == NULL) {
        return;
    }
    rematArenaDeinit(s); /* PR1c: s->fns->deinit(s), behind the fns and inCall guards */
    rematPlanFree(s->plan);
    rematWireTableFree(s->wires);
    *s = (rematScheduler_t){0};
}

/* D60 covers every size sum. Resident block sizes cannot reach SIZE_MAX
 * together, so this exit is unreachable and has no dedicated test. */
static size_t reportAdd(size_t a, size_t b) {
    size_t out;
    if (!checkedAddSize(a, b, &out)) {
        PRINT_ERROR("rematSchedulerReport: size overflow computing metadataBytes");
        exit(1);
    }
    return out;
}

/* The ARENA half of the report (spec §5.1, D55 as amended by Codex N3): each
 * flag derived from the block it names. */
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
         * its arena fields 0 by definition (plan Assumption 4). */
        out->placed = true;
        out->dataReserved = true;
        break;
    }
}
