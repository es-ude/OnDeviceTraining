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
#include "RematRows.h"
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

/* D60 before any row reservation: every placed(w) and their sum over all
 * ranges. An offset is 0 or the end of a chain of distinct co-live ranges, so
 * this total bounds every offset + placed the placement computes: after this
 * pass its own checked sums cannot fire (plan Assumption 8). */
static void arenaRequirePlaceableSizes(const rematWireTable_t *t, const rematProgram_t *p) {
    size_t total = 0;
    for (size_t r = 0; r < p->numRanges; r++) {
        uint16_t w = p->ranges[r].wire;
        total = arenaAdd(t, w, total, arenaPlaced(t, w), "placed bytes total");
    }
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

void arenaVerifyPlacement(const rematWireTable_t *t, const rematProgram_t *p, const size_t *offsets,
                          size_t bytes) {
    for (size_t r = 0; r < p->numRanges; r++) {
        const rematWire_t *w = &t->wires[p->ranges[r].wire];
        size_t size = arenaPlaced(t, p->ranges[r].wire);
        if (offsets[r] % ODT_WIRE_ALIGN != 0u) {
            PRINT_ERROR("remat[arena]: placement verifier: wire %s %u at offset %zu is not a "
                        "multiple of ODT_WIRE_ALIGN (%u)",
                        arenaWireKind(w->kind), (unsigned)w->index, offsets[r],
                        (unsigned)ODT_WIRE_ALIGN);
            exit(1);
        }
        /* offsets[r] + size would wrap on an imported offset near SIZE_MAX. */
        if (offsets[r] > bytes || size > bytes - offsets[r]) {
            PRINT_ERROR("remat[arena]: placement verifier: wire %s %u at offset %zu (%zu placed "
                        "bytes) ends past the arena's %zu bytes",
                        arenaWireKind(w->kind), (unsigned)w->index, offsets[r], size, bytes);
            exit(1);
        }
    }
    /* Every range now lies inside [0, bytes), so the sums below cannot wrap. */
    for (size_t a = 0; a < p->numRanges; a++) {
        for (size_t b = a + 1u; b < p->numRanges; b++) {
            if (!arenaCoLive(&p->ranges[a], &p->ranges[b])) {
                continue;
            }
            size_t endA = offsets[a] + arenaPlaced(t, p->ranges[a].wire);
            size_t endB = offsets[b] + arenaPlaced(t, p->ranges[b].wire);
            if (offsets[a] < endB && offsets[b] < endA) {
                const rematWire_t *wa = &t->wires[p->ranges[a].wire];
                const rematWire_t *wb = &t->wires[p->ranges[b].wire];
                PRINT_ERROR("remat[arena]: placement verifier: wires %s %u and %s %u are co-live "
                            "but share bytes (offsets %zu and %zu)",
                            arenaWireKind(wa->kind), (unsigned)wa->index, arenaWireKind(wb->kind),
                            (unsigned)wb->index, offsets[a], offsets[b]);
                exit(1);
            }
        }
    }
}

bool rematArenaInit(rematScheduler_t *s, layer_t **model, size_t n, lossConfig_t loss,
                    const tensor_t *inputLike, const rematPlanSpec_t *spec) {
    *s = (rematScheduler_t){.type = REMAT_ARENA};
    if (!rematWireTableInit(&s->wires, model, n, loss, inputLike)) {
        return false;
    }
    /* Early bound, before the plan block exists (Leo, Codex PR1b-plan triage):
     * in PR1 every slab wire gets exactly one range, so the plan would hold
     * numWires - 1 ranges. The post-build check below stays authoritative. */
    if (s->wires->numWires - 1u > ODT_REMAT_MAX_RANGES) {
        PRINT_ERROR("remat[arena]: model needs %zu ranges, above ODT_REMAT_MAX_RANGES (%u)",
                    s->wires->numWires - 1u, (unsigned)ODT_REMAT_MAX_RANGES);
        exit(1);
    }
    /* The identical model reaches both: rematPlanBuild checks it against the
     * table's key. */
    if (!rematPlanBuild(&s->plan, s->wires, model, spec)) {
        return false;
    }
    const rematProgram_t *p = &s->plan->train;
    /* Authoritative: PR6 plans (RETAIN_LIST, SEQUENCE) hold more ranges than
     * wires. Unreachable in PR1 behind the pre-check above (no dedicated
     * test). */
    if (p->numRanges > ODT_REMAT_MAX_RANGES) {
        PRINT_ERROR("remat[arena]: plan has %zu ranges, above ODT_REMAT_MAX_RANGES (%u)",
                    p->numRanges, (unsigned)ODT_REMAT_MAX_RANGES);
        exit(1); /* a model/plan fact, not an OOM: the table-init idiom */
    }
    arenaRequirePlaceableSizes(s->wires, p);
    /* D55 as amended by Codex N3: the offsets table is its own small block,
     * placed into and verified before the arena data block exists, so a
     * failed data reservation still reports every analytic field. The limit
     * above bounds the product. numRanges >= 1: ACT n always has a range, so
     * neither reservation below is reserveMemory(0). */
    s->row.arena.offsets = reserveMemory(p->numRanges * sizeof(size_t));
    if (s->row.arena.offsets == NULL) {
        return false; /* planned, !placed */
    }
    size_t bytes;
    if (!arenaPlaceFirstFitDecreasing(s->wires, p, s->row.arena.offsets, &bytes,
                                      &s->row.arena.peakPlacedBytes)) {
        return false; /* the placement's scratch: planned, !placed */
    }
    arenaVerifyPlacement(s->wires, p, s->row.arena.offsets, bytes);
    s->row.arena.bytes = bytes; /* placed: every arena field of the report is valid */
    s->row.arena.base = reserveMemory(bytes);
    return s->row.arena.base != NULL; /* false: planned, placed, !dataReserved */
}

void rematArenaDeinit(rematScheduler_t *s) {
    freeReservedMemory(s->row.arena.base); /* NULL after a failed data reservation */
    freeReservedMemory(s->row.arena.offsets);
    s->row.arena.base = NULL;
    s->row.arena.offsets = NULL;
}
