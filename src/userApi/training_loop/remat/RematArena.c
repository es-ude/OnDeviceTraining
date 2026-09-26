#define SOURCE_FILE "REMAT_ARENA"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "Common.h"
#include "Layer.h"
#include "LossFunction.h"
#include "RematCheckedSize.h"
#include "RematPlace.h"
#include "RematPlan.h"
#include "RematScheduler.h"
#include "StorageApi.h"
#include "Tensor.h"

#define ARENA_UNPLACED SIZE_MAX

static const char *arenaWireKind(uint8_t kind) {
    return kind == REMAT_WIRE_ACT ? "ACT" : "GRAD";
}

/* Every size sum of the placement names the wire it was computing (D60). */
static size_t arenaAdd(const rematWireTable_t *t, uint16_t w, size_t a, size_t b,
                       const char *quantity) {
    size_t out;
    if (!checkedAddSize(a, b, &out)) {
        PRINT_ERROR("remat[arena]: size overflow computing %s at wire %s %u", quantity,
                    arenaWireKind(t->wires[w].kind), (unsigned)t->wires[w].index);
        exit(1);
    }
    return out;
}

size_t arenaPlaced(const rematWireTable_t *t, uint16_t w) {
    return arenaAdd(t, w, rematWireBytes(t, w), ODT_WIRE_ALIGN - 1u, "placed bytes") &
           ~(size_t)(ODT_WIRE_ALIGN - 1u);
}

/* Inclusive intervals (spec §4.2): ranges that meet at one step are co-live. */
static bool arenaCoLive(const rematRange_t *a, const rematRange_t *b) {
    return a->begin <= b->end && b->begin <= a->end;
}

static bool arenaPlacesBefore(const rematWireTable_t *t, const rematProgram_t *p, size_t a,
                              size_t b) {
    size_t placedA = arenaPlaced(t, p->ranges[a].wire);
    size_t placedB = arenaPlaced(t, p->ranges[b].wire);
    if (placedA != placedB) {
        return placedA > placedB;
    }
    if (p->ranges[a].begin != p->ranges[b].begin) {
        return p->ranges[a].begin < p->ranges[b].begin;
    }
    return p->ranges[a].wire < p->ranges[b].wire;
}

/* O(R) selection per range keeps the placement order without an order array. */
static size_t arenaNextToPlace(const rematWireTable_t *t, const rematProgram_t *p,
                               const size_t *offsets) {
    size_t next = p->numRanges;
    for (size_t r = 0; r < p->numRanges; r++) {
        if (offsets[r] == ARENA_UNPLACED &&
            (next == p->numRanges || arenaPlacesBefore(t, p, r, next))) {
            next = r;
        }
    }
    return next;
}

static size_t arenaPeakPlacedBytes(const rematWireTable_t *t, const rematProgram_t *p) {
    rematWalk_t walk = {0};
    size_t live = 0;
    size_t peak = 0;
    for (walk.step = 0; walk.step < p->numSteps; walk.step++) {
        for (size_t r; (r = rematWalkOpening(p, &walk)) != REMAT_NONE;) {
            uint16_t w = p->ranges[r].wire;
            live = arenaAdd(t, w, live, arenaPlaced(t, w), "peakPlacedBytes");
        }
        if (live > peak) {
            peak = live;
        }
        for (size_t r; (r = rematWalkClosing(p, &walk)) != REMAT_NONE;) {
            live -= arenaPlaced(t, p->ranges[r].wire);
        }
    }
    return peak;
}

bool arenaPlaceFirstFitDecreasing(const rematWireTable_t *t, const rematProgram_t *p,
                                  size_t *offsets, size_t *bytes, size_t *peakPlacedBytes) {
    /* numRanges < REMAT_NONE (the table's wire-id guard): the uint16_t ids and
     * this product fit. */
    uint16_t *byOffset = reserveMemory(p->numRanges * sizeof(uint16_t));
    if (byOffset == NULL) {
        return false;
    }
    for (size_t r = 0; r < p->numRanges; r++) {
        offsets[r] = ARENA_UNPLACED;
    }
    size_t end = 0;
    for (size_t numPlaced = 0; numPlaced < p->numRanges; numPlaced++) {
        size_t r = arenaNextToPlace(t, p, offsets);
        uint16_t w = p->ranges[r].wire;
        size_t size = arenaPlaced(t, w);
        /* byOffset holds the placed ranges by ascending offset. Sweeping it, off
         * is always 0 or the end of a co-live placed range, and it moves past a
         * co-live range only when that range overlaps [off, off + size); the
         * first co-live range starting at or above off + size therefore closes
         * the lowest feasible candidate of the spec's rule (plan Assumption 7).
         * O(R) per range, O(R^2) overall. */
        size_t off = 0;
        for (size_t i = 0; i < numPlaced; i++) {
            size_t q = byOffset[i];
            if (!arenaCoLive(&p->ranges[q], &p->ranges[r])) {
                continue;
            }
            if (offsets[q] >= arenaAdd(t, w, off, size, "offset + placed bytes")) {
                break;
            }
            uint16_t qw = p->ranges[q].wire;
            size_t qEnd = arenaAdd(t, qw, offsets[q], arenaPlaced(t, qw), "offset + placed bytes");
            if (qEnd > off) {
                off = qEnd;
            }
        }
        offsets[r] = off;
        size_t slot = numPlaced;
        while (slot > 0 && offsets[byOffset[slot - 1]] > off) {
            byOffset[slot] = byOffset[slot - 1];
            slot--;
        }
        byOffset[slot] = (uint16_t)r;
        size_t rEnd = arenaAdd(t, w, off, size, "arena bytes");
        if (rEnd > end) {
            end = rEnd;
        }
    }
    freeReservedMemory(byOffset);
    *bytes = end;
    *peakPlacedBytes = arenaPeakPlacedBytes(t, p);
    return true;
}

bool rematArenaInit(rematScheduler_t *s, layer_t **model, size_t n, lossConfig_t loss,
                    const tensor_t *inputLike, const rematPlanSpec_t *spec) {
    *s = (rematScheduler_t){.type = REMAT_ARENA};
    if (!rematWireTableInit(&s->wires, model, n, loss, inputLike)) {
        return false;
    }
    /* The identical model reaches both: rematPlanBuild checks it against the
     * table's key. */
    return rematPlanBuild(&s->plan, s->wires, model, spec);
}
