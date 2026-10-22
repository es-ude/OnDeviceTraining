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

/* What a placement order compares. */
typedef struct arenaOrder {
    const rematWireTable_t *t;
    const rematProgram_t *p;
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

bool arenaPlaceFirstFitDecreasing(const rematWireTable_t *t, const rematProgram_t *p,
                                  size_t *offsets, size_t *bytes, size_t *peakPlacedBytes) {
    const arenaOrder_t o = {.t = t, .p = p};
    if (!arenaFirstFitInOrder(&o, arenaPlacesBefore, offsets, bytes)) {
        return false;
    }
    *peakPlacedBytes = arenaPeakPlacedBytes(t, p);
    return true;
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
    if (!arenaPlaceFirstFitDecreasing(s->wires, p, s->row.arena.offsets, &bytes,
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
