#define SOURCE_FILE "REMAT_ARENA"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

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

/* Every size sum of the placement is overflow-checked and names the wire it
 * was computing. */
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

/* Overflow-checked before any row reservation: every placed(w) and their sum over all
 * ranges. An offset is 0 or the end of a chain of distinct co-live ranges, so
 * this total bounds every offset + placed the placement computes: after this
 * pass its own checked sums cannot fire. */
static void arenaRequirePlaceableSizes(const rematWireTable_t *t, const rematProgram_t *p) {
    size_t total = 0;
    for (size_t r = 0; r < p->numRanges; r++) {
        uint16_t w = p->ranges[r].wire;
        total = arenaAdd(t, w, total, arenaPlaced(t, w), "placed bytes total");
    }
}

/* Inclusive intervals: ranges that meet at one step are co-live. */
static bool arenaCoLive(const rematRange_t *a, const rematRange_t *b) {
    return a->begin <= b->end && b->begin <= a->end;
}

/* What a placement order compares. The last three fields are read only by the
 * members that set them. */
typedef struct arenaOrder {
    const rematWireTable_t *t;
    const rematProgram_t *p;
    const size_t *stepLive;      /* breadth/area: placed bytes live at each step */
    const uint16_t *busiestStep; /* breadth/area: per range, its step of most live bytes */
    uint16_t boundary;           /* two-ended: the first backward step */
} arenaOrder_t;

typedef bool (*arenaBefore_t)(const arenaOrder_t *o, size_t a, size_t b);

static size_t arenaRangePlaced(const rematWireTable_t *t, const rematProgram_t *p, size_t r) {
    return arenaPlaced(t, p->ranges[r].wire);
}

/* First-fit-decreasing: placed size descending, then begin, then wire id. */
static bool arenaPlacesBefore(const arenaOrder_t *o, size_t a, size_t b) {
    size_t placedA = arenaRangePlaced(o->t, o->p, a);
    size_t placedB = arenaRangePlaced(o->t, o->p, b);
    if (placedA != placedB) {
        return placedA > placedB;
    }
    if (o->p->ranges[a].begin != o->p->ranges[b].begin) {
        return o->p->ranges[a].begin < o->p->ranges[b].begin;
    }
    return o->p->ranges[a].wire < o->p->ranges[b].wire;
}

static bool arenaSizeThenEndBefore(const arenaOrder_t *o, size_t a, size_t b) {
    size_t placedA = arenaRangePlaced(o->t, o->p, a);
    size_t placedB = arenaRangePlaced(o->t, o->p, b);
    if (placedA != placedB) {
        return placedA > placedB;
    }
    const rematRange_t *ra = &o->p->ranges[a];
    const rematRange_t *rb = &o->p->ranges[b];
    if (ra->end != rb->end) {
        return ra->end > rb->end;
    }
    if (ra->begin != rb->begin) {
        return ra->begin < rb->begin;
    }
    return a < b;
}

/* Begin, then end descending, then placed size descending, then range id. */
static bool arenaStreamBefore(const arenaOrder_t *o, size_t a, size_t b) {
    const rematRange_t *ra = &o->p->ranges[a];
    const rematRange_t *rb = &o->p->ranges[b];
    if (ra->begin != rb->begin) {
        return ra->begin < rb->begin;
    }
    if (ra->end != rb->end) {
        return ra->end > rb->end;
    }
    size_t placedA = arenaRangePlaced(o->t, o->p, a);
    size_t placedB = arenaRangePlaced(o->t, o->p, b);
    if (placedA != placedB) {
        return placedA > placedB;
    }
    return a < b;
}

/* O(R) selection per range keeps the placement order without an order array. */
static size_t arenaNextToPlace(const arenaOrder_t *o, arenaBefore_t before, const size_t *offsets) {
    size_t next = o->p->numRanges;
    for (size_t r = 0; r < o->p->numRanges; r++) {
        if (offsets[r] == ARENA_UNPLACED && (next == o->p->numRanges || before(o, r, next))) {
            next = r;
        }
    }
    return next;
}

/* The ranges placed so far, by ascending offset. */
typedef struct arenaFit {
    const rematWireTable_t *t;
    const rematProgram_t *p;
    size_t *offsets;
    uint16_t *byOffset;
    size_t numPlaced;
} arenaFit_t;

/* numRanges <= ODT_REMAT_MAX_RANGES < REMAT_NONE (rematArenaInit checks the
 * cap first): the uint16_t ids and this product fit. */
static bool arenaFitInit(arenaFit_t *f, const rematWireTable_t *t, const rematProgram_t *p,
                         size_t *offsets) {
    *f = (arenaFit_t){.t = t, .p = p, .offsets = offsets};
    f->byOffset = reserveMemory(p->numRanges * sizeof(uint16_t));
    if (f->byOffset == NULL) {
        return false;
    }
    for (size_t r = 0; r < p->numRanges; r++) {
        offsets[r] = ARENA_UNPLACED;
    }
    return true;
}

/* The first-fit rule: the lowest offset that is 0 or the end of a co-live
 * placed range and overlaps no co-live placed range. Sweeping byOffset, off is
 * always such a candidate, and it moves past a co-live range only when that
 * range overlaps [off, off + size); the first co-live range starting at or
 * above off + size therefore closes the lowest feasible candidate.
 * O(R) per range, O(R^2) overall. */
static size_t arenaLowestFit(const arenaFit_t *f, size_t r) {
    uint16_t w = f->p->ranges[r].wire;
    size_t size = arenaPlaced(f->t, w);
    size_t off = 0;
    for (size_t i = 0; i < f->numPlaced; i++) {
        size_t q = f->byOffset[i];
        if (!arenaCoLive(&f->p->ranges[q], &f->p->ranges[r])) {
            continue;
        }
        if (f->offsets[q] >= arenaAdd(f->t, w, off, size, "offset + placed bytes")) {
            break;
        }
        uint16_t qw = f->p->ranges[q].wire;
        size_t qEnd =
            arenaAdd(f->t, qw, f->offsets[q], arenaPlaced(f->t, qw), "offset + placed bytes");
        if (qEnd > off) {
            off = qEnd;
        }
    }
    return off;
}

static void arenaFitRecord(arenaFit_t *f, size_t r, size_t off) {
    f->offsets[r] = off;
    size_t slot = f->numPlaced;
    while (slot > 0 && f->offsets[f->byOffset[slot - 1]] > off) {
        f->byOffset[slot] = f->byOffset[slot - 1];
        slot--;
    }
    f->byOffset[slot] = (uint16_t)r;
    f->numPlaced++;
}

/* max(offset + placed): the bytes a layout needs. */
static size_t arenaExtent(const rematWireTable_t *t, const rematProgram_t *p,
                          const size_t *offsets) {
    size_t end = 0;
    for (size_t r = 0; r < p->numRanges; r++) {
        uint16_t w = p->ranges[r].wire;
        size_t rEnd = arenaAdd(t, w, offsets[r], arenaPlaced(t, w), "arena bytes");
        if (rEnd > end) {
            end = rEnd;
        }
    }
    return end;
}

/* The peak of the placed bytes live at a step; with stepLive != NULL also
 * each step's sum. */
static size_t arenaPeakPlacedBytes(const rematWireTable_t *t, const rematProgram_t *p,
                                   size_t *stepLive) {
    rematWalk_t walk = {0};
    size_t live = 0;
    size_t peak = 0;
    for (walk.step = 0; walk.step < p->numSteps; walk.step++) {
        for (size_t r; (r = rematWalkOpening(p, &walk)) != REMAT_NONE;) {
            uint16_t w = p->ranges[r].wire;
            live = arenaAdd(t, w, live, arenaPlaced(t, w), "peakPlacedBytes");
        }
        if (stepLive != NULL) {
            stepLive[walk.step] = live;
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

static bool arenaFirstFitInOrder(const arenaOrder_t *o, arenaBefore_t before, size_t *offsets,
                                 size_t *bytes) {
    arenaFit_t f;
    if (!arenaFitInit(&f, o->t, o->p, offsets)) {
        return false;
    }
    while (f.numPlaced < o->p->numRanges) {
        size_t r = arenaNextToPlace(o, before, offsets);
        arenaFitRecord(&f, r, arenaLowestFit(&f, r));
    }
    freeReservedMemory(f.byOffset);
    *bytes = arenaExtent(o->t, o->p, offsets);
    return true;
}

size_t arenaPlacedPeakBytes(const rematWireTable_t *t, const rematProgram_t *p) {
    return arenaPeakPlacedBytes(t, p, NULL);
}

bool arenaPlaceFirstFitDecreasing(const rematWireTable_t *t, const rematProgram_t *p,
                                  size_t placedPeak, size_t *offsets, size_t *bytes) {
    (void)placedPeak;
    const arenaOrder_t o = {.t = t, .p = p};
    return arenaFirstFitInOrder(&o, arenaPlacesBefore, offsets, bytes);
}

bool arenaPlaceSizeThenEnd(const rematWireTable_t *t, const rematProgram_t *p, size_t placedPeak,
                           size_t *offsets, size_t *bytes) {
    (void)placedPeak;
    const arenaOrder_t o = {.t = t, .p = p};
    return arenaFirstFitInOrder(&o, arenaSizeThenEndBefore, offsets, bytes);
}

static bool arenaBeganLater(const rematProgram_t *p, size_t a, size_t b) {
    if (p->ranges[a].begin != p->ranges[b].begin) {
        return p->ranges[a].begin > p->ranges[b].begin;
    }
    return a > b;
}

/* Sweeping the co-live placed ranges by offset gives the gaps between them and
 * the open space above the last one. low = the start of the first gap that
 * holds the range; high = the highest offset at which a gap holds it below
 * placedPeak. All operands are offsets and placed sizes inside the bound that
 * arenaRequirePlaceableSizes checked, and each difference is taken only when
 * it cannot wrap. */
static size_t arenaPingPongOffset(const arenaFit_t *f, size_t r, size_t placedPeak) {
    const rematProgram_t *p = f->p;
    size_t size = arenaRangePlaced(f->t, p, r);
    size_t prev = 0;
    size_t low = 0;
    bool haveLow = false;
    size_t high = 0;
    bool haveHigh = false;
    size_t latest = p->numRanges;
    for (size_t i = 0; i < f->numPlaced; i++) {
        size_t q = f->byOffset[i];
        if (!arenaCoLive(&p->ranges[q], &p->ranges[r])) {
            continue;
        }
        if (latest == p->numRanges || arenaBeganLater(p, q, latest)) {
            latest = q;
        }
        size_t gapEnd = f->offsets[q];
        if (gapEnd > prev) {
            if (!haveLow && gapEnd - prev >= size) {
                low = prev;
                haveLow = true;
            }
            size_t top = gapEnd < placedPeak ? gapEnd : placedPeak;
            if (top >= prev && top - prev >= size) {
                high = top - size;
                haveHigh = true;
            }
        }
        size_t qEnd = arenaAdd(f->t, p->ranges[q].wire, gapEnd, arenaRangePlaced(f->t, p, q),
                               "offset + placed bytes");
        if (qEnd > prev) {
            prev = qEnd;
        }
    }
    if (!haveLow) {
        low = prev;
    }
    if (placedPeak >= prev && placedPeak - prev >= size) {
        high = placedPeak - size;
        haveHigh = true;
    }
    if (latest == p->numRanges || !haveHigh) {
        return low;
    }
    size_t latestOff = f->offsets[latest];
    size_t latestEnd = arenaAdd(f->t, p->ranges[latest].wire, latestOff,
                                arenaRangePlaced(f->t, p, latest), "offset + placed bytes");
    /* 2 x latestOff + placed < placedPeak, without the doubled sum. */
    bool lowerHalf = latestEnd <= placedPeak && latestOff < placedPeak - latestEnd;
    return lowerHalf ? high : low;
}

bool arenaPlacePingPong(const rematWireTable_t *t, const rematProgram_t *p, size_t placedPeak,
                        size_t *offsets, size_t *bytes) {
    const arenaOrder_t o = {.t = t, .p = p};
    arenaFit_t f;
    if (!arenaFitInit(&f, t, p, offsets)) {
        return false;
    }
    while (f.numPlaced < p->numRanges) {
        size_t r = arenaNextToPlace(&o, arenaStreamBefore, offsets);
        arenaFitRecord(&f, r, arenaPingPongOffset(&f, r, placedPeak));
    }
    freeReservedMemory(f.byOffset);
    *bytes = arenaExtent(t, p, offsets);
    return true;
}

/* placed x lifetime as hi * 2^16 + lo: exact for every placed size, because a
 * lifetime is at most 2^16 steps. */
static void arenaArea(const arenaOrder_t *o, size_t r, size_t *hi, size_t *lo) {
    size_t placed = arenaRangePlaced(o->t, o->p, r);
    size_t steps = (size_t)o->p->ranges[r].end - o->p->ranges[r].begin + 1u;
    size_t low = (placed & 0xFFFFu) * steps;
    *hi = (placed >> 16) * steps + (low >> 16);
    *lo = low & 0xFFFFu;
}

/* The order: the range whose busiest step holds more live placed bytes first
 * (the earlier step on equal bytes); within one step, larger placed size x
 * lifetime first; then the lower wire id. Wire id suffices as the last key: two
 * ranges of one wire are never co-live, so their order cannot change a
 * layout. */
static bool arenaBreadthAreaBefore(const arenaOrder_t *o, size_t a, size_t b) {
    uint16_t stepA = o->busiestStep[a];
    uint16_t stepB = o->busiestStep[b];
    if (stepA != stepB) {
        if (o->stepLive[stepA] != o->stepLive[stepB]) {
            return o->stepLive[stepA] > o->stepLive[stepB];
        }
        return stepA < stepB;
    }
    size_t hiA, loA, hiB, loB;
    arenaArea(o, a, &hiA, &loA);
    arenaArea(o, b, &hiB, &loB);
    if (hiA != hiB) {
        return hiA > hiB;
    }
    if (loA != loB) {
        return loA > loB;
    }
    return o->p->ranges[a].wire < o->p->ranges[b].wire;
}

/* A range is ordered at the first visited step that holds it: its step of
 * most live bytes, the earliest on ties. One scratch block holds the live
 * bytes per step and that step per range. */
bool arenaPlaceBreadthArea(const rematWireTable_t *t, const rematProgram_t *p, size_t placedPeak,
                           size_t *offsets, size_t *bytes) {
    (void)placedPeak;
    size_t *stepLive =
        reserveMemory(p->numSteps * sizeof(size_t) + p->numRanges * sizeof(uint16_t));
    if (stepLive == NULL) {
        return false;
    }
    uint16_t *busiestStep = (uint16_t *)(stepLive + p->numSteps);
    (void)arenaPeakPlacedBytes(t, p, stepLive);
    for (size_t r = 0; r < p->numRanges; r++) {
        uint16_t best = p->ranges[r].begin;
        for (uint16_t s = best; s < p->ranges[r].end; s++) {
            if (stepLive[s + 1u] > stepLive[best]) {
                best = (uint16_t)(s + 1u);
            }
        }
        busiestStep[r] = best;
    }
    const arenaOrder_t o = {.t = t, .p = p, .stepLive = stepLive, .busiestStep = busiestStep};
    bool placed = arenaFirstFitInOrder(&o, arenaBreadthAreaBefore, offsets, bytes);
    freeReservedMemory(stepLive);
    return placed;
}

bool arenaPlanHasGradRange(const rematWireTable_t *t, const rematProgram_t *p) {
    for (size_t r = 0; r < p->numRanges; r++) {
        if (t->wires[p->ranges[r].wire].kind == REMAT_WIRE_GRAD) {
            return true;
        }
    }
    return false;
}

static bool arenaStored(const arenaOrder_t *o, size_t r) {
    return o->p->ranges[r].begin < o->boundary && o->boundary <= o->p->ranges[r].end;
}

/* Every stored range before every other; the stored ones by (begin, end
 * descending, range id), the others in the stream order. */
static bool arenaTwoEndedBefore(const arenaOrder_t *o, size_t a, size_t b) {
    bool storedA = arenaStored(o, a);
    if (storedA != arenaStored(o, b)) {
        return storedA;
    }
    if (!storedA) {
        return arenaStreamBefore(o, a, b);
    }
    const rematRange_t *ra = &o->p->ranges[a];
    const rematRange_t *rb = &o->p->ranges[b];
    if (ra->begin != rb->begin) {
        return ra->begin < rb->begin;
    }
    if (ra->end != rb->end) {
        return ra->end > rb->end;
    }
    return a < b;
}

static size_t arenaRangeEnd(const rematWireTable_t *t, const rematProgram_t *p,
                            const size_t *offsets, size_t r) {
    return arenaAdd(t, p->ranges[r].wire, offsets[r], arenaRangePlaced(t, p, r),
                    "offset + placed bytes");
}

bool arenaPlaceTwoEnded(const rematWireTable_t *t, const rematProgram_t *p, size_t placedPeak,
                        size_t *offsets, size_t *bytes) {
    (void)placedPeak;
    if (!arenaPlanHasGradRange(t, p)) {
        PRINT_ERROR("remat[arena]: two-ended placement needs a GRAD range");
        exit(1);
    }
    arenaOrder_t o = {.t = t, .p = p, .boundary = UINT16_MAX};
    for (size_t r = 0; r < p->numRanges; r++) {
        if (t->wires[p->ranges[r].wire].kind == REMAT_WIRE_GRAD &&
            p->ranges[r].begin < o.boundary) {
            o.boundary = p->ranges[r].begin;
        }
    }
    arenaFit_t f;
    if (!arenaFitInit(&f, t, p, offsets)) {
        return false;
    }
    bool others = false;
    for (size_t k = 0; k < p->numRanges; k++) {
        size_t r = arenaNextToPlace(&o, arenaTwoEndedBefore, offsets);
        if (!others && !arenaStored(&o, r)) {
            others = true;
            f.numPlaced = 0; /* the others first fit on their own */
        }
        arenaFitRecord(&f, r, arenaLowestFit(&f, r));
    }
    freeReservedMemory(f.byOffset);
    /* Each group is first fit on its own, so an end within a group is at most
     * that group's placed total, and a stored end plus another end stays
     * inside the bound arenaRequirePlaceableSizes checked. */
    size_t height = arenaExtent(t, p, offsets);
    for (size_t a = 0; a < p->numRanges; a++) {
        for (size_t b = 0; b < p->numRanges; b++) {
            if (arenaStored(&o, a) && !arenaStored(&o, b) &&
                arenaCoLive(&p->ranges[a], &p->ranges[b])) {
                size_t pair = arenaAdd(t, p->ranges[b].wire, arenaRangeEnd(t, p, offsets, a),
                                       arenaRangeEnd(t, p, offsets, b), "two-ended height");
                if (pair > height) {
                    height = pair;
                }
            }
        }
    }
    /* height >= every end, so the mirrored offset cannot wrap. */
    for (size_t r = 0; r < p->numRanges; r++) {
        if (!arenaStored(&o, r)) {
            offsets[r] = height - arenaRangeEnd(t, p, offsets, r);
        }
    }
    *bytes = arenaExtent(t, p, offsets);
    return true;
}

typedef bool (*arenaMember_t)(const rematWireTable_t *t, const rematProgram_t *p, size_t placedPeak,
                              size_t *offsets, size_t *bytes);

/* Fixed at compile time; the order decides ties. A new member is one entry
 * here and its twin in the Python planner's copy of this placer. */
static const struct arenaMemberEntry {
    arenaMember_t place;
    bool needsGradRange;
} arenaMembers[] = {
    {arenaPlaceFirstFitDecreasing, false}, {arenaPlaceBreadthArea, false},
    {arenaPlacePingPong, false},           {arenaPlaceTwoEnded, true},
    {arenaPlaceSizeThenEnd, false},
};

bool arenaPlacePortfolio(const rematWireTable_t *t, const rematProgram_t *p, size_t *offsets,
                         size_t *bytes, size_t *peakPlacedBytes) {
    size_t placedPeak = arenaPeakPlacedBytes(t, p, NULL);
    if (!arenaMembers[0].place(t, p, placedPeak, offsets, bytes)) {
        return false;
    }
    size_t *trial = reserveMemory(p->numRanges * sizeof(size_t));
    if (trial == NULL) {
        return false;
    }
    bool hasGradRange = arenaPlanHasGradRange(t, p);
    bool placed = true;
    for (size_t m = 1; placed && m < sizeof arenaMembers / sizeof arenaMembers[0]; m++) {
        if (arenaMembers[m].needsGradRange && !hasGradRange) {
            continue;
        }
        size_t trialBytes = 0;
        placed = arenaMembers[m].place(t, p, placedPeak, trial, &trialBytes);
        if (placed && trialBytes < *bytes) {
            memcpy(offsets, trial, p->numRanges * sizeof(size_t));
            *bytes = trialBytes;
        }
    }
    freeReservedMemory(trial);
    *peakPlacedBytes = placedPeak;
    return placed;
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
#ifdef ODT_MEM_PROFILE
    size_t mark = memProfileCurrentBytes();
#endif
    *s = (rematScheduler_t){.type = REMAT_ARENA, .fns = &rematSchedulerFunctions[REMAT_ARENA]};
    if (!rematWireTableInit(&s->wires, model, n, loss, inputLike)) {
        return false;
    }
    /* Early bound, before the plan block exists: every TRAIN plan writes each
     * slab wire at least once, so without inPlaceDx it holds at least
     * numWires - 1 ranges (exactly that many under STORE_ALL and LIVENESS).
     * An in-place plan holds fewer, one per chain, so near the cap this bound
     * may reject a model the post-build check would admit; such a model has
     * over 500 layers, far beyond MCU scale. The post-build check below stays
     * authoritative. */
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
    /* Authoritative: the placement below relies on it. No input reaches it
     * today, so it has no dedicated test: STORE_ALL and LIVENESS stop at the
     * pre-check above, and rematPlanBuild already caps SEQUENCE and
     * RETAIN_LIST plans. */
    if (p->numRanges > ODT_REMAT_MAX_RANGES) {
        PRINT_ERROR("remat[arena]: plan has %zu ranges, above ODT_REMAT_MAX_RANGES (%u)",
                    p->numRanges, (unsigned)ODT_REMAT_MAX_RANGES);
        exit(1); /* a model/plan fact, not an OOM: the table-init idiom */
    }
    arenaRequirePlaceableSizes(s->wires, p);
    /* The offsets table is its own small block,
     * placed into and verified before the arena data block exists, so a
     * failed data reservation still reports every analytic field. The limit
     * above bounds the product. numRanges >= 1: ACT n always has a range, so
     * neither reservation below is reserveMemory(0). */
    s->row.arena.offsets = reserveMemory(p->numRanges * sizeof(size_t));
    if (s->row.arena.offsets == NULL) {
        return false; /* planned, !placed */
    }
    size_t bytes;
    if (!arenaPlacePortfolio(s->wires, p, s->row.arena.offsets, &bytes,
                             &s->row.arena.peakPlacedBytes)) {
        return false; /* the placement's scratch: planned, !placed */
    }
    arenaVerifyPlacement(s->wires, p, s->row.arena.offsets, bytes);
    s->row.arena.bytes = bytes; /* placed: every arena field of the report is valid */
    s->row.arena.base = reserveMemory(bytes);
    if (s->row.arena.base == NULL) {
        return false; /* planned, placed, !dataReserved */
    }
    ODT_ASAN_POISON(s->row.arena.base, bytes); /* unaddressable until a range opens */
#ifdef ODT_MEM_PROFILE
    rematRequireReservedMatchesReport(s, memProfileCurrentBytes() - mark);
#endif
    return true;
}

void rematArenaBegin(rematScheduler_t *s) {
    if (s->row.arena.offsets == NULL || s->row.arena.base == NULL) {
        PRINT_ERROR("remat[arena]: rematBegin on a scheduler whose arena was never reserved "
                    "(rematArenaInit returned false)");
        exit(1);
    }
    s->walk = (rematWalk_t){0};
}

/* TRAIN: the verified placement. EVAL, two-ended inside the same arena: ACT j
 * even at offset 0, odd top-aligned. ACT j and ACT j+1 are co-live at TRAIN's
 * FORWARD(j), so the verified layout proves placed(j) + placed(j+1) <= bytes;
 * bytes is a multiple of ODT_WIRE_ALIGN, so a top offset is aligned; ACT 0 is
 * borrowed. Eval therefore adds no byte to the arena. */
static size_t arenaOffset(const rematScheduler_t *s, const rematProgram_t *p, size_t r) {
    if (s->mode == REMAT_MODE_TRAIN) {
        return s->row.arena.offsets[r];
    }
    uint16_t w = p->ranges[r].wire;
    return (w % 2u == 0u) ? 0u : s->row.arena.bytes - arenaPlaced(s->wires, w);
}

bool rematArenaNext(rematScheduler_t *s, rematStep_t *st) {
    const rematProgram_t *p = rematPlanProgram(s->plan, s->mode);
    if (s->walk.step == p->numSteps) {
        return false;
    }
    for (size_t r; (r = rematWalkOpening(p, &s->walk)) != REMAT_NONE;) {
        /* The id comes from the range that was placed, never from rematGradId. */
        uint16_t w = p->ranges[r].wire;
        uint8_t *b = s->row.arena.base + arenaOffset(s, p, r);
        ODT_ASAN_UNPOISON(b, rematWireBytes(s->wires, w)); /* exact bytes: the pad stays poisoned */
        rematWireBind(s->wires, w, b); /* bindGen++, accounting, VERIFY poison-at-bind */
    }
    *st = p->steps[s->walk.step];
    return true;
}

void rematArenaDone(rematScheduler_t *s, const rematStep_t *st) {
    (void)st; /* the dispatch checks that done() answers the step next() handed out */
    const rematProgram_t *p = rematPlanProgram(s->plan, s->mode);
    for (size_t r; (r = rematWalkClosing(p, &s->walk)) != REMAT_NONE;) {
        uint16_t w = p->ranges[r].wire;
        uint8_t *b = rematWireHdr(s->wires, w)->data;
        rematWireRelease(s->wires, w);                /* VERIFY poison first, while still owned */
        ODT_ASAN_POISON(b, arenaPlaced(s->wires, w)); /* then unaddressable */
    }
    s->walk.step++;
}

void rematArenaEnd(rematScheduler_t *s) {
    rematRequireWalkComplete(s, "arena");
}

void rematArenaDeinit(rematScheduler_t *s) {
    if (s->row.arena.base != NULL) { /* NULL after a failed data reservation */
        ODT_ASAN_UNPOISON(s->row.arena.base, s->row.arena.bytes);
        freeReservedMemory(s->row.arena.base);
    }
    freeReservedMemory(s->row.arena.offsets);
    s->row.arena.base = NULL;
    s->row.arena.offsets = NULL;
}
