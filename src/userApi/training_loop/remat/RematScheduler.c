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
    /* ARENA is the only row until PR1c, whose HEAP reports placed == planned
     * with the arena fields 0. */
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
