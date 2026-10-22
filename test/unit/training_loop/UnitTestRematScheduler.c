#define SOURCE_FILE "UNIT_TEST_REMAT_SCHEDULER"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "AsanDeath.h"
#include "Common.h"
#include "Conv1dApi.h"
#include "DeathTest.h"
#include "FlattenApi.h"
#include "Layer.h"
#include "LayerConfigAccess.h"
#include "LayerQuant.h"
#include "Linear.h"
#include "LinearApi.h"
#include "LossFunction.h"
#include "Pool1dApi.h"
#include "QuantLayerApi.h"
#include "Quantization.h"
#include "Relu.h"
#include "ReluApi.h"
#include "RematPlace.h"
#include "RematPlan.h"
#include "RematRows.h"
#include "RematScheduler.h"
#include "RematTestFixtures.h"
#include "SoftmaxApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

/* ---- init builds the shared table and plan; the report ---- */

void testArenaInitBuildsTheTableAndThePlan(void) {
    fixture_t f;
    buildHarModel(&f);
    rematScheduler_t s = initArena(&f, NULL);
    TEST_ASSERT_EQUAL_INT(REMAT_ARENA, s.type);
    TEST_ASSERT_NOT_NULL(s.wires);
    TEST_ASSERT_NOT_NULL(s.plan);
    TEST_ASSERT_EQUAL_size_t(HAR_N, s.wires->modelSize);
    TEST_ASSERT_EQUAL_INT(REMAT_PLAN_STORE_ALL, s.plan->policy);
    freeFixture(&f, &s);
}

void testReportEchoesTypePolicyAndPlanFacts(void) {
    fixture_t f;
    buildHarModel(&f);
    rematScheduler_t s = initArena(&f, &g_liveness);
    rematReport_t r;
    rematSchedulerReport(&s, &r);
    TEST_ASSERT_EQUAL_INT(REMAT_ARENA, r.type);
    TEST_ASSERT_EQUAL_INT(REMAT_PLAN_LIVENESS, r.policy);
    TEST_ASSERT_TRUE(r.planned);
    TEST_ASSERT_EQUAL_size_t(25, r.numSteps); /* 12 + 1 + 1 + 11 */
    freeFixture(&f, &s);
}

static size_t reportedPeakOnHar(const rematPlanSpec_t *spec) {
    fixture_t f;
    buildHarModel(&f);
    rematScheduler_t s = initArena(&f, spec);
    rematReport_t r;
    rematSchedulerReport(&s, &r);
    freeFixture(&f, &s);
    return r.peakLiveBytes;
}

/* Scan-model pins, read from the report. */
void testReportPeakLiveBytesHarStoreAllIs74288(void) {
    TEST_ASSERT_EQUAL_size_t(74288, reportedPeakOnHar(NULL));
}

void testReportPeakLiveBytesHarLivenessIs49152(void) {
    TEST_ASSERT_EQUAL_size_t(49152, reportedPeakOnHar(&g_liveness));
}

/* Never initialised, failed before the plan existed, or deinitialised --
 * the report reads nothing through the NULL table or plan. */
void testReportOnAZeroedSchedulerIsEmpty(void) {
    rematScheduler_t s = {0};
    rematReport_t r;
    memset(&r, 0xA5, sizeof r);
    rematSchedulerReport(&s, &r);
    TEST_ASSERT_FALSE(r.planned);
    TEST_ASSERT_EQUAL_size_t(0, r.numSteps);
    TEST_ASSERT_EQUAL_size_t(0, r.peakLiveBytes);
    TEST_ASSERT_EQUAL_size_t(0, r.metadataBytes);
}

void testReportMetadataCountsTheResidentBlocks(void) {
    fixture_t f;
    buildHarModel(&f);
    rematScheduler_t s = initArena(&f, NULL);
    rematReport_t r;
    rematSchedulerReport(&s, &r);
    /* slab + plan block + the offsets block: 4,848 + 568 + 23 * 8 = 5,600 on LP64 */
    TEST_ASSERT_EQUAL_size_t(s.wires->slabBytes + s.plan->blockBytes +
                                 s.plan->train.numRanges * sizeof(size_t),
                             r.metadataBytes);
    freeFixture(&f, &s);
}

void testDeinitIsNullSafeAndIdempotent(void) {
    rematSchedulerDeinit(NULL);
    fixture_t f;
    buildHarModel(&f);
    rematScheduler_t s = initArena(&f, NULL);
    rematSchedulerDeinit(&s);
    TEST_ASSERT_NULL(s.wires);
    TEST_ASSERT_NULL(s.plan);
    rematSchedulerDeinit(&s);
    freeModel(f.model, f.n);
}

/* The row's deinit slot must be repeatable on its own, not only through
 * rematSchedulerDeinit's zeroing. */
void testArenaDeinitIsIdempotentOnItsOwn(void) {
    fixture_t f;
    buildHarModel(&f);
    rematScheduler_t s = initArena(&f, NULL);
    rematSchedulerFunctions[REMAT_ARENA].deinit(&s);
    TEST_ASSERT_NULL(s.row.arena.base);
    TEST_ASSERT_NULL(s.row.arena.offsets);
    rematSchedulerFunctions[REMAT_ARENA].deinit(&s);
    freeFixture(&f, &s);
}

#ifdef ODT_MEM_PROFILE
/* The live-byte counter is real only under ODT_MEM_PROFILE (unit_test_debug,
 * asan, ubsan); the plain unit_test preset compiles this out. */
void testDeinitReturnsEveryInitBlock(void) {
    fixture_t f;
    buildHarModel(&f);
    size_t before = memProfileCurrentBytes();
    rematScheduler_t s = initArena(&f, &g_liveness);
    TEST_ASSERT_TRUE(memProfileCurrentBytes() > before);
    rematSchedulerDeinit(&s);
    TEST_ASSERT_EQUAL_size_t(before, memProfileCurrentBytes());
    freeModel(f.model, f.n);
}
#endif

/* ---- aligned first-fit-decreasing placement ---- */

typedef struct builtPlan {
    rematWireTable_t *t;
    rematPlan_t *p;
} builtPlan_t;

/* Table and plan without the row, so the placement is tested on its own. */
static builtPlan_t buildTableAndPlan(layer_t **model, size_t n, lossFuncType_t lt,
                                     const tensor_t *x, const rematPlanSpec_t *spec) {
    builtPlan_t b = {NULL, NULL};
    TEST_ASSERT_TRUE(rematWireTableInit(&b.t, model, n, defaultLossConfig(lt), x));
    TEST_ASSERT_TRUE(rematPlanBuild(&b.p, b.t, model, spec));
    return b;
}

static void freeTableAndPlan(builtPlan_t *b) {
    rematPlanFree(b->p);
    rematWireTableFree(b->t);
}

void testArenaPlacedRoundsEveryWireUpToTheWireAlignment(void) {
    fixture_t f;
    buildF1Model(&f);
    builtPlan_t b = buildTableAndPlan(f.model, f.n, f.lt, f.x, NULL);
    TEST_ASSERT_EQUAL_size_t(5, rematWireBytes(b.t, 1));
    TEST_ASSERT_EQUAL_size_t(4, rematWireBytes(b.t, 2));
    TEST_ASSERT_EQUAL_size_t(4, rematWireBytes(b.t, 3));
    TEST_ASSERT_EQUAL_size_t(8, arenaPlaced(b.t, 1));
    TEST_ASSERT_EQUAL_size_t(8, arenaPlaced(b.t, 2));
    TEST_ASSERT_EQUAL_size_t(8, arenaPlaced(b.t, 3));
    freeTableAndPlan(&b);
    freeModel(f.model, f.n);

    /* HAR: every wire is a multiple of 8 (8192, 4096, 256, 24 B), so the
     * placement pads nothing -- the basis of the arenaPadBytes == 0 pin. */
    fixture_t h;
    buildHarModel(&h);
    builtPlan_t hb = buildTableAndPlan(h.model, h.n, h.lt, h.x, NULL);
    for (uint16_t w = 1; w < hb.t->numWires; w++) {
        TEST_ASSERT_EQUAL_size_t(rematWireBytes(hb.t, w), arenaPlaced(hb.t, w));
    }
    freeTableAndPlan(&hb);
    freeModel(h.model, h.n);
}

static void assertF1Placement(const rematPlanSpec_t *spec) {
    fixture_t f;
    buildF1Model(&f);
    builtPlan_t b = buildTableAndPlan(f.model, f.n, f.lt, f.x, spec);
    const rematProgram_t *p = &b.p->train;
    TEST_ASSERT_EQUAL_size_t(3, p->numRanges);
    for (size_t r = 0; r < 3u; r++) {
        TEST_ASSERT_EQUAL_UINT16(r + 1u, p->ranges[r].wire);
    }
    size_t offsets[3];
    size_t bytes = 0;
    size_t peak = 0;
    TEST_ASSERT_TRUE(arenaPlaceFirstFitDecreasing(b.t, p, offsets, &bytes, &peak));
    TEST_ASSERT_EQUAL_size_t(0, offsets[0]);  /* ACT 1 */
    TEST_ASSERT_EQUAL_size_t(8, offsets[1]);  /* ACT 2; unaligned: 5 */
    TEST_ASSERT_EQUAL_size_t(16, offsets[2]); /* the seed; unaligned: 9 */
    TEST_ASSERT_EQUAL_size_t(24, bytes);
    freeTableAndPlan(&b);
    freeModel(f.model, f.n);
}

/* The alignment pin. */
void testArenaOffsetsAligned(void) {
    assertF1Placement(NULL);
    assertF1Placement(&g_liveness);
}

void testFfdPeakPlacedBytesIsThePeakOfPlacedSums(void) {
    fixture_t f;
    buildF1Model(&f);
    builtPlan_t b = buildTableAndPlan(f.model, f.n, f.lt, f.x, NULL);
    size_t offsets[3];
    size_t bytes = 0;
    size_t peak = 0;
    TEST_ASSERT_TRUE(arenaPlaceFirstFitDecreasing(b.t, &b.p->train, offsets, &bytes, &peak));
    TEST_ASSERT_EQUAL_size_t(24, peak); /* all three co-live at step 3: 8 + 8 + 8, exact 13 */
    freeTableAndPlan(&b);
    freeModel(f.model, f.n);

    fixture_t h;
    buildHarModel(&h);
    builtPlan_t hb = buildTableAndPlan(h.model, h.n, h.lt, h.x, NULL);
    size_t harOffsets[23];
    TEST_ASSERT_EQUAL_size_t(23, hb.p->train.numRanges);
    TEST_ASSERT_TRUE(arenaPlaceFirstFitDecreasing(hb.t, &hb.p->train, harOffsets, &bytes, &peak));
    TEST_ASSERT_EQUAL_size_t(74288, peak); /* HAR pads nothing: the scan-model STORE_ALL peak */
    freeTableAndPlan(&hb);
    freeModel(h.model, h.n);
}

#define ORACLE_MAX_RANGES 40u
#define ORACLE_MAX_STEPS 30u

/* A hand-built table and program. The placement reads only wires[].bytes
 * (kind and index name a wire in an exit), the ranges, endOrder and
 * numSteps. */
typedef struct randomPlan {
    rematWire_t wires[ORACLE_MAX_RANGES + 1u];
    rematRange_t ranges[ORACLE_MAX_RANGES];
    uint16_t endOrder[ORACLE_MAX_RANGES];
    rematWireTable_t table;
    rematProgram_t program;
} randomPlan_t;

/* endOrder, the table and the program of a plan whose ranges and wires are set. */
static void finishPlan(randomPlan_t *rp, size_t numRanges, size_t numSteps) {
    for (size_t i = 0; i < numRanges; i++) {
        size_t k = i;
        while (k > 0 && rp->ranges[rp->endOrder[k - 1]].end > rp->ranges[i].end) {
            rp->endOrder[k] = rp->endOrder[k - 1];
            k--;
        }
        rp->endOrder[k] = (uint16_t)i;
    }
    rp->table = (rematWireTable_t){.numWires = numRanges + 1u, .wires = rp->wires};
    rp->program = (rematProgram_t){.numSteps = numSteps,
                                   .steps = NULL,
                                   .numRanges = numRanges,
                                   .ranges = rp->ranges,
                                   .endOrder = rp->endOrder};
}

static void buildRandomPlan(randomPlan_t *rp, uint32_t *state) {
    size_t numSteps = 1u + nextRandom(state) % ORACLE_MAX_STEPS;
    size_t numRanges = 1u + nextRandom(state) % ORACLE_MAX_RANGES;
    for (size_t r = 0; r < numRanges; r++) {
        uint16_t begin = (uint16_t)(nextRandom(state) % numSteps);
        uint16_t end = (uint16_t)(begin + nextRandom(state) % (numSteps - begin));
        size_t k = r;
        while (k > 0 && rp->ranges[k - 1].begin > begin) {
            rp->ranges[k] = rp->ranges[k - 1];
            k--;
        }
        rp->ranges[k] = (rematRange_t){.wire = 0, .begin = begin, .end = end};
    }
    /* Wire ids are a Fisher-Yates permutation of 1..numRanges, drawn from the
     * same seeded stream, decoupled from begin order: if wire id tracked
     * array (begin) position, as it used to, the placement's "begin, then wire
     * id" tie-break would collapse to "wire id" and never be exercised. */
    uint16_t perm[ORACLE_MAX_RANGES];
    for (size_t r = 0; r < numRanges; r++) {
        perm[r] = (uint16_t)(r + 1u);
    }
    for (size_t r = numRanges; r > 1u; r--) {
        size_t j = nextRandom(state) % r;
        uint16_t tmp = perm[r - 1u];
        perm[r - 1u] = perm[j];
        perm[j] = tmp;
    }
    rp->wires[0] = (rematWire_t){.kind = REMAT_WIRE_ACT, .index = 0, .borrowed = 1u};
    for (size_t r = 0; r < numRanges; r++) {
        size_t bytes = 1u + nextRandom(state) % 100u; /* mostly unaligned */
        if (nextRandom(state) % 4u == 0u) {
            bytes *= 64u;
        }
        rp->ranges[r].wire = perm[r];
        rp->wires[perm[r]] =
            (rematWire_t){.kind = REMAT_WIRE_ACT, .index = perm[r], .bytes = bytes};
    }
    finishPlan(rp, numRanges, numSteps);
}

typedef struct handRange {
    size_t bytes;
    uint16_t begin, end;
    uint8_t kind;
} handRange_t;

typedef struct handPlan {
    handRange_t ranges[5];
    size_t numRanges;
    size_t numSteps;
} handPlan_t;

/* Range r is wire r + 1; the ranges come in begin order. */
static void buildHandPlan(randomPlan_t *rp, const handPlan_t *hp) {
    rp->wires[0] = (rematWire_t){.kind = REMAT_WIRE_ACT, .index = 0, .borrowed = 1u};
    for (size_t r = 0; r < hp->numRanges; r++) {
        const handRange_t *hr = &hp->ranges[r];
        uint16_t w = (uint16_t)(r + 1u);
        rp->wires[w] = (rematWire_t){.kind = hr->kind, .index = w, .bytes = hr->bytes};
        rp->ranges[r] = (rematRange_t){.wire = w, .begin = hr->begin, .end = hr->end};
    }
    finishPlan(rp, hp->numRanges, hp->numSteps);
}

/* Hand plans, each placed at its peak by exactly one portfolio member; the
 * member tests below derive each layout. */
static const handPlan_t g_sizeThenEndPlan = {{{16, 0, 3, REMAT_WIRE_ACT},
                                              {16, 1, 4, REMAT_WIRE_ACT},
                                              {8, 3, 5, REMAT_WIRE_ACT},
                                              {8, 4, 5, REMAT_WIRE_ACT},
                                              {24, 5, 5, REMAT_WIRE_GRAD}},
                                             5,
                                             6};
static const handPlan_t g_pingPongPlan = {{{24, 0, 2, REMAT_WIRE_ACT},
                                           {16, 1, 3, REMAT_WIRE_ACT},
                                           {16, 2, 2, REMAT_WIRE_GRAD},
                                           {32, 3, 3, REMAT_WIRE_ACT}},
                                          4,
                                          4};
static const handPlan_t g_breadthAreaPlan = {{{16, 0, 3, REMAT_WIRE_ACT},
                                              {24, 1, 2, REMAT_WIRE_GRAD},
                                              {24, 2, 4, REMAT_WIRE_GRAD},
                                              {8, 3, 4, REMAT_WIRE_GRAD},
                                              {32, 4, 4, REMAT_WIRE_GRAD}},
                                             5,
                                             5};
static const handPlan_t g_twoEndedPlan = {{{32, 0, 1, REMAT_WIRE_ACT},
                                           {16, 1, 3, REMAT_WIRE_GRAD},
                                           {16, 2, 3, REMAT_WIRE_GRAD},
                                           {24, 3, 5, REMAT_WIRE_GRAD},
                                           {32, 4, 4, REMAT_WIRE_GRAD}},
                                          5,
                                          6};

static size_t oraclePlaced(size_t bytes) {
    return (bytes + 7u) / 8u * 8u;
}

static bool oracleCoLive(const rematRange_t *a, const rematRange_t *b) {
    return a->begin <= b->end && b->begin <= a->end;
}

static size_t oracleRangePlaced(const randomPlan_t *rp, size_t r) {
    return oraclePlaced(rp->wires[rp->ranges[r].wire].bytes);
}

static bool oraclePlacesFirst(const randomPlan_t *rp, size_t a, size_t b) {
    if (oracleRangePlaced(rp, a) != oracleRangePlaced(rp, b)) {
        return oracleRangePlaced(rp, a) > oracleRangePlaced(rp, b);
    }
    if (rp->ranges[a].begin != rp->ranges[b].begin) {
        return rp->ranges[a].begin < rp->ranges[b].begin;
    }
    return rp->ranges[a].wire < rp->ranges[b].wire;
}

static size_t oraclePeakPlaced(const randomPlan_t *rp) {
    size_t peak = 0;
    for (size_t s = 0; s < rp->program.numSteps; s++) {
        size_t sum = 0;
        for (size_t r = 0; r < rp->program.numRanges; r++) {
            if (rp->ranges[r].begin <= s && s <= rp->ranges[r].end) {
                sum += oracleRangePlaced(rp, r);
            }
        }
        if (sum > peak) {
            peak = sum;
        }
    }
    return peak;
}

/* The first-fit rule taken literally, O(R^3): every candidate (0, or the end of a
 * co-live placed range) against every co-live placed range; the lowest
 * feasible one wins. */
static size_t oracleFirstFitDecreasing(const randomPlan_t *rp, size_t *offsets,
                                       size_t *peakPlaced) {
    size_t numRanges = rp->program.numRanges;
    bool placed[ORACLE_MAX_RANGES] = {false};
    size_t bytes = 0;
    for (size_t k = 0; k < numRanges; k++) {
        size_t w = numRanges;
        for (size_t r = 0; r < numRanges; r++) {
            if (!placed[r] && (w == numRanges || oraclePlacesFirst(rp, r, w))) {
                w = r;
            }
        }
        size_t size = oracleRangePlaced(rp, w);
        size_t best = SIZE_MAX;
        for (size_t c = 0; c <= numRanges; c++) {
            size_t candidate;
            if (c == numRanges) {
                candidate = 0;
            } else if (placed[c] && oracleCoLive(&rp->ranges[c], &rp->ranges[w])) {
                candidate = offsets[c] + oracleRangePlaced(rp, c);
            } else {
                continue;
            }
            bool fits = true;
            for (size_t q = 0; q < numRanges && fits; q++) {
                if (placed[q] && oracleCoLive(&rp->ranges[q], &rp->ranges[w])) {
                    fits = candidate >= offsets[q] + oracleRangePlaced(rp, q) ||
                           offsets[q] >= candidate + size;
                }
            }
            if (fits && candidate < best) {
                best = candidate;
            }
        }
        offsets[w] = best;
        placed[w] = true;
        if (best + size > bytes) {
            bytes = best + size;
        }
    }
    *peakPlaced = oraclePeakPlaced(rp);
    return bytes;
}

/* The placement, within the O(R^2 log R) bound, must reproduce the
 * rule exactly, so no pin depends on how it is implemented. */
void testFfdMatchesTheNaiveOracleOnRandomPlans(void) {
    uint32_t state = 0x2545F491u;
    for (size_t trial = 0; trial < 500u; trial++) {
        randomPlan_t rp;
        buildRandomPlan(&rp, &state);
        size_t expected[ORACLE_MAX_RANGES];
        size_t actual[ORACLE_MAX_RANGES];
        size_t expectedPeak = 0;
        size_t actualBytes = 0;
        size_t actualPeak = 0;
        size_t expectedBytes = oracleFirstFitDecreasing(&rp, expected, &expectedPeak);
        TEST_ASSERT_TRUE(arenaPlaceFirstFitDecreasing(&rp.table, &rp.program, actual, &actualBytes,
                                                      &actualPeak));
        for (size_t r = 0; r < rp.program.numRanges; r++) {
            TEST_ASSERT_EQUAL_size_t_MESSAGE(expected[r], actual[r],
                                             "offset differs from the oracle");
        }
        TEST_ASSERT_EQUAL_size_t_MESSAGE(expectedBytes, actualBytes, "arena bytes differ");
        TEST_ASSERT_EQUAL_size_t_MESSAGE(expectedPeak, actualPeak, "peakPlacedBytes differs");
    }
}

/* A borrowed [1, SIZE_MAX/4] FLOAT32 input: ACT 1 holds 4 * (2^62 - 1) =
 * SIZE_MAX - 3 bytes, which table init accepts; rounding it up to 8 wraps. */
static void placeTheOnlyWire(builtPlan_t *b) {
    (void)arenaPlaced(b->t, 1);
}

void testArenaPlacedExitsNamingTheWireOnOverflow(void) {
    layer_t *model[1] = {makeRelu(&g_floatQ)};
    inputLike_t in;
    tensor_t *x = makeInput(&in, (size_t[]){1, SIZE_MAX / 4u}, 2, &g_floatQ);
    builtPlan_t b = buildTableAndPlan(model, 1, MSE, x, NULL);
    ASSERT_EXITS_WITH_OUTPUT(1, "size overflow computing placed bytes at wire ACT 1",
                             placeTheOnlyWire(&b));
    freeTableAndPlan(&b);
    freeModel(model, 1);
}

#ifdef ODT_MEM_PROFILE
/* The candidate list is a temporary block, released before the
 * placement returns ("freed before init returns"). */
void testFfdReleasesItsScratch(void) {
    fixture_t h;
    buildHarModel(&h);
    builtPlan_t b = buildTableAndPlan(h.model, h.n, h.lt, h.x, &g_liveness);
    size_t offsets[23];
    size_t bytes = 0;
    size_t peak = 0;
    size_t before = memProfileCurrentBytes();
    TEST_ASSERT_TRUE(arenaPlaceFirstFitDecreasing(b.t, &b.p->train, offsets, &bytes, &peak));
    TEST_ASSERT_EQUAL_size_t(before, memProfileCurrentBytes());
    freeTableAndPlan(&b);
    freeModel(h.model, h.n);
}
#endif

/* ---- the portfolio's other members ---- */

typedef bool (*placeMember_t)(const rematWireTable_t *t, const rematProgram_t *p, size_t placedPeak,
                              size_t *offsets, size_t *bytes);

static void assertMemberPlacesTheHandPlan(placeMember_t place, const handPlan_t *hp,
                                          const size_t *expected, size_t expectedBytes) {
    randomPlan_t rp;
    buildHandPlan(&rp, hp);
    size_t offsets[ORACLE_MAX_RANGES];
    size_t bytes = 0;
    TEST_ASSERT_TRUE(place(&rp.table, &rp.program, oraclePeakPlaced(&rp), offsets, &bytes));
    for (size_t r = 0; r < hp->numRanges; r++) {
        TEST_ASSERT_EQUAL_size_t_MESSAGE(expected[r], offsets[r], "offset differs");
    }
    TEST_ASSERT_EQUAL_size_t(expectedBytes, bytes);
}

/* Size, then the later end first: ranges 4, 1, 0, 2, 3. Range 2 [3,5] climbs
 * past 1 (0..16), 4 (0..24) and 0 (16..32) to 32; range 3 [4,5] fits under
 * range 2 at 24. 40 B = the peak (16 + 16 + 8 at step 3); first-fit-decreasing
 * needs 48. */
void testSizeThenEndPlacesTheHandPlan(void) {
    static const size_t expected[5] = {16, 0, 32, 24, 0};
    assertMemberPlacesTheHandPlan(arenaPlaceSizeThenEnd, &g_sizeThenEndPlan, expected, 40);
}

/* Height 56 = the peak at step 2 (24 + 16 + 16). Range 1 begins under range 0
 * (offset 0, lower half), so it takes the top: 56 - 16 = 40. Range 2's latest
 * co-live range is range 1 (upper half): the low gap 24..40. Range 3 [3,3]
 * sees only range 1 and takes 0. First-fit-decreasing needs 64. */
void testPingPongPlacesTheHandPlan(void) {
    static const size_t expected[4] = {0, 40, 24, 0};
    assertMemberPlacesTheHandPlan(arenaPlacePingPong, &g_pingPongPlan, expected, 56);
}

/* Steps 2 and 4 hold 64 B, the peak; step 2 comes first: ranges 2 (24 x 3
 * steps), 0 (16 x 4) and 1 (24 x 2) in area order, then step 4's ranges 4
 * and 3. First fit stacks 2, 0, 1 at 0, 24, 40, puts 4 above 2 at 24 and 3
 * above 4 and 0 at 56. First-fit-decreasing needs 80. */
void testBreadthAreaPlacesTheHandPlan(void) {
    static const size_t expected[5] = {24, 40, 0, 56, 24};
    assertMemberPlacesTheHandPlan(arenaPlaceBreadthArea, &g_breadthAreaPlan, expected, 64);
}

/* Placed size times lifetime is compared exactly: 2^60 B x 16 steps = 2^64
 * exceeds 2^61 B x 4 steps = 2^63, so the longer range is placed first,
 * although the first product wraps to 0 in a size_t. Both meet the busiest
 * step 0 (3 x 2^60 B). */
void testBreadthAreaComparesAreasWithoutWrapping(void) {
    static const handPlan_t hp = {
        {{(size_t)1 << 60, 0, 15, REMAT_WIRE_ACT}, {(size_t)1 << 61, 0, 3, REMAT_WIRE_ACT}}, 2, 16};
    static const size_t expected[2] = {0, (size_t)1 << 60};
    assertMemberPlacesTheHandPlan(arenaPlaceBreadthArea, &hp, expected, (size_t)3 << 60);
}

/* The first backward step is 1 (range 1, the first GRAD); range 0 [0,1] is
 * alive across it and stays at the bottom. The others first fit on their own
 * at 0, 16, 32, 0; their extent, 56 (range 3 at 32 + 24), already keeps range
 * 1 (end 16) clear of range 0 (end 32), the only co-live pair across the
 * groups, so they are mirrored below 56. Every other member needs 72. */
void testTwoEndedPlacesTheHandPlan(void) {
    static const size_t expected[5] = {0, 40, 24, 0, 24};
    assertMemberPlacesTheHandPlan(arenaPlaceTwoEnded, &g_twoEndedPlan, expected, 56);
}

/* A plan without a backward has no GRAD range: one ReLU under MSE (no layer
 * with parameters, so nothing is trained). */
void testPlanHasGradRangeOnlyWithABackward(void) {
    layer_t *model[1] = {makeRelu(&g_floatQ)};
    inputLike_t in;
    tensor_t *x = makeInput(&in, (size_t[]){1, 4}, 2, &g_floatQ);
    builtPlan_t b = buildTableAndPlan(model, 1, MSE, x, NULL);
    TEST_ASSERT_FALSE(arenaPlanHasGradRange(b.t, &b.p->train));
    freeTableAndPlan(&b);
    freeModel(model, 1);
    fixture_t h;
    buildHarModel(&h);
    builtPlan_t hb = buildTableAndPlan(h.model, h.n, h.lt, h.x, NULL);
    TEST_ASSERT_TRUE(arenaPlanHasGradRange(hb.t, &hb.p->train));
    freeTableAndPlan(&hb);
    freeModel(h.model, h.n);
}

static void placeTwoEndedWithoutGrad(void) {
    static const handPlan_t hp = {{{8, 0, 0, REMAT_WIRE_ACT}}, 1, 1};
    randomPlan_t rp;
    buildHandPlan(&rp, &hp);
    size_t offsets[1];
    size_t bytes = 0;
    (void)arenaPlaceTwoEnded(&rp.table, &rp.program, 8, offsets, &bytes);
}

void testTwoEndedExitsOnAPlanWithoutGradRanges(void) {
    ASSERT_EXITS_WITH_OUTPUT(1, "remat[arena]: two-ended placement needs a GRAD range",
                             placeTwoEndedWithoutGrad());
}

/* The random plans hold only ACT wires; two-ended needs GRAD ranges, so its
 * run marks the later half of each plan's ranges GRAD, as a backward follows
 * the forward. */
static void markLaterHalfGrad(randomPlan_t *rp) {
    for (size_t r = rp->program.numRanges / 2u; r < rp->program.numRanges; r++) {
        rp->wires[rp->ranges[r].wire].kind = REMAT_WIRE_GRAD;
    }
}

/* Every layout a member returns passes the verifier, its bytes is the extent
 * of its offsets, and no layout is below the placed peak. */
static void assertMemberVerifiesOnRandomPlans(placeMember_t place, bool withGrad) {
    uint32_t state = 0x9E3779B9u;
    for (size_t trial = 0; trial < 500u; trial++) {
        randomPlan_t rp;
        buildRandomPlan(&rp, &state);
        if (withGrad) {
            markLaterHalfGrad(&rp);
        }
        size_t offsets[ORACLE_MAX_RANGES];
        size_t bytes = 0;
        size_t peak = oraclePeakPlaced(&rp);
        TEST_ASSERT_TRUE(place(&rp.table, &rp.program, peak, offsets, &bytes));
        arenaVerifyPlacement(&rp.table, &rp.program, offsets, bytes);
        size_t extent = 0;
        for (size_t r = 0; r < rp.program.numRanges; r++) {
            if (offsets[r] + oracleRangePlaced(&rp, r) > extent) {
                extent = offsets[r] + oracleRangePlaced(&rp, r);
            }
        }
        TEST_ASSERT_EQUAL_size_t_MESSAGE(extent, bytes, "bytes is not the layout's extent");
        TEST_ASSERT_TRUE_MESSAGE(bytes >= peak, "arena below the placed peak");
    }
}

void testNewMembersVerifyOnRandomPlans(void) {
    assertMemberVerifiesOnRandomPlans(arenaPlaceSizeThenEnd, false);
    assertMemberVerifiesOnRandomPlans(arenaPlacePingPong, false);
    assertMemberVerifiesOnRandomPlans(arenaPlaceBreadthArea, false);
    assertMemberVerifiesOnRandomPlans(arenaPlaceTwoEnded, true);
}

#ifdef ODT_MEM_PROFILE
static void assertMemberReleasesItsScratch(placeMember_t place) {
    fixture_t h;
    buildHarModel(&h);
    builtPlan_t b = buildTableAndPlan(h.model, h.n, h.lt, h.x, NULL);
    size_t offsets[23];
    size_t bytes = 0;
    size_t before = memProfileCurrentBytes();
    TEST_ASSERT_TRUE(place(b.t, &b.p->train, 74288, offsets, &bytes));
    TEST_ASSERT_EQUAL_size_t(before, memProfileCurrentBytes());
    freeTableAndPlan(&b);
    freeModel(h.model, h.n);
}

void testNewMembersReleaseTheirScratch(void) {
    assertMemberReleasesItsScratch(arenaPlaceSizeThenEnd);
    assertMemberReleasesItsScratch(arenaPlacePingPong);
    assertMemberReleasesItsScratch(arenaPlaceBreadthArea);
    assertMemberReleasesItsScratch(arenaPlaceTwoEnded);
}
#endif

/* ---- the portfolio ---- */

/* First-fit-decreasing is a member, so no layout is larger than its; the
 * chosen layout passes the verifier, and the placed peak does not depend on
 * the layout. Every second plan gets GRAD ranges, so two-ended runs. */
void testPortfolioIsNeverWorseThanFfdOnRandomPlans(void) {
    uint32_t state = 0x85EBCA6Bu;
    for (size_t trial = 0; trial < 500u; trial++) {
        randomPlan_t rp;
        buildRandomPlan(&rp, &state);
        if (trial % 2u == 1u) {
            markLaterHalfGrad(&rp);
        }
        size_t ffd[ORACLE_MAX_RANGES];
        size_t best[ORACLE_MAX_RANGES];
        size_t ffdBytes = 0;
        size_t ffdPeak = 0;
        size_t bytes = 0;
        size_t peak = 0;
        TEST_ASSERT_TRUE(
            arenaPlaceFirstFitDecreasing(&rp.table, &rp.program, ffd, &ffdBytes, &ffdPeak));
        TEST_ASSERT_TRUE(arenaPlacePortfolio(&rp.table, &rp.program, best, &bytes, &peak));
        arenaVerifyPlacement(&rp.table, &rp.program, best, bytes);
        TEST_ASSERT_TRUE_MESSAGE(bytes <= ffdBytes, "portfolio larger than first-fit-decreasing");
        TEST_ASSERT_EQUAL_size_t(ffdPeak, peak);
    }
}

/* HAR LIVENESS: first-fit-decreasing, breadth/area, ping-pong and
 * size-then-end each place 49152 B with a layout of their own (two-ended
 * 49688 B); the first member keeps its layout. */
void testPortfolioKeepsTheEarlierMemberOnATie(void) {
    fixture_t h;
    buildHarModel(&h);
    builtPlan_t b = buildTableAndPlan(h.model, h.n, h.lt, h.x, &g_liveness);
    const rematProgram_t *p = &b.p->train;
    size_t ffd[23];
    size_t later[23];
    size_t best[23];
    size_t ffdBytes = 0;
    size_t laterBytes = 0;
    size_t bytes = 0;
    size_t peak = 0;
    TEST_ASSERT_TRUE(arenaPlaceFirstFitDecreasing(b.t, p, ffd, &ffdBytes, &peak));
    TEST_ASSERT_TRUE(arenaPlaceSizeThenEnd(b.t, p, peak, later, &laterBytes));
    TEST_ASSERT_EQUAL_size_t(49152, ffdBytes);
    TEST_ASSERT_EQUAL_size_t(49152, laterBytes);
    TEST_ASSERT_TRUE(memcmp(ffd, later, p->numRanges * sizeof(size_t)) != 0);
    TEST_ASSERT_TRUE(arenaPlacePortfolio(b.t, p, best, &bytes, &peak));
    TEST_ASSERT_EQUAL_size_t(49152, bytes);
    for (size_t r = 0; r < p->numRanges; r++) {
        TEST_ASSERT_EQUAL_size_t_MESSAGE(ffd[r], best[r], "not the first member's layout");
    }
    freeTableAndPlan(&b);
    freeModel(h.model, h.n);
}

static size_t portfolioBytesOf(const handPlan_t *hp) {
    randomPlan_t rp;
    buildHandPlan(&rp, hp);
    size_t offsets[ORACLE_MAX_RANGES];
    size_t bytes = 0;
    size_t peak = 0;
    TEST_ASSERT_TRUE(arenaPlacePortfolio(&rp.table, &rp.program, offsets, &bytes, &peak));
    TEST_ASSERT_EQUAL_size_t(peak, bytes);
    return bytes;
}

/* Each hand plan reaches its peak through one member only, so dropping any
 * member from the portfolio raises one of these four numbers. On HAR,
 * mnist_cnn and HAR finetune two-ended is never the only order at the peak; it
 * stays for models outside them. */
void testEveryPortfolioMemberIsTheOnlyOneAtThePeakOfSomePlan(void) {
    TEST_ASSERT_EQUAL_size_t(64, portfolioBytesOf(&g_breadthAreaPlan));
    TEST_ASSERT_EQUAL_size_t(56, portfolioBytesOf(&g_pingPongPlan));
    TEST_ASSERT_EQUAL_size_t(56, portfolioBytesOf(&g_twoEndedPlan));
    TEST_ASSERT_EQUAL_size_t(40, portfolioBytesOf(&g_sizeThenEndPlan));
}

#ifdef ODT_MEM_PROFILE
/* The trial layouts' block is released before the portfolio returns, so an
 * init still reserves exactly its report's arena + metadata bytes. */
void testPortfolioReleasesItsScratch(void) {
    fixture_t h;
    buildHarModel(&h);
    builtPlan_t b = buildTableAndPlan(h.model, h.n, h.lt, h.x, NULL);
    size_t offsets[23];
    size_t bytes = 0;
    size_t peak = 0;
    size_t before = memProfileCurrentBytes();
    TEST_ASSERT_TRUE(arenaPlacePortfolio(b.t, &b.p->train, offsets, &bytes, &peak));
    TEST_ASSERT_EQUAL_size_t(before, memProfileCurrentBytes());
    freeTableAndPlan(&b);
    freeModel(h.model, h.n);
}
#endif

/* ---- the placement verifier ---- */

typedef struct placedPlan {
    fixture_t f;
    builtPlan_t b;
    size_t offsets[23];
    size_t bytes;
} placedPlan_t;

static void placeFixture(placedPlan_t *pp, void (*build)(fixture_t *),
                         const rematPlanSpec_t *spec) {
    build(&pp->f);
    pp->b = buildTableAndPlan(pp->f.model, pp->f.n, pp->f.lt, pp->f.x, spec);
    size_t peak = 0;
    TEST_ASSERT_TRUE(pp->b.p->train.numRanges <= 23u);
    TEST_ASSERT_TRUE(
        arenaPlaceFirstFitDecreasing(pp->b.t, &pp->b.p->train, pp->offsets, &pp->bytes, &peak));
}

static void freePlacedFixture(placedPlan_t *pp) {
    freeTableAndPlan(&pp->b);
    freeModel(pp->f.model, pp->f.n);
}

static void verifyPlaced(placedPlan_t *pp) {
    arenaVerifyPlacement(pp->b.t, &pp->b.p->train, pp->offsets, pp->bytes);
}

static void assertTheFfdPlacementVerifies(void (*build)(fixture_t *), const rematPlanSpec_t *spec) {
    placedPlan_t pp;
    placeFixture(&pp, build, spec);
    ASSERT_EXITS_WITH(0, verifyPlaced(&pp));
    freePlacedFixture(&pp);
}

/* HAR's FFD layouts reuse bytes between ranges that are never co-live (e.g.
 * LIVENESS puts ACT 2 and ACT 4 both at 8,192): legal, and accepted. */
void testVerifierAcceptsTheFfdPlacementOnTheZoo(void) {
    assertTheFfdPlacementVerifies(buildHarModel, NULL);
    assertTheFfdPlacementVerifies(buildHarModel, &g_liveness);
    assertTheFfdPlacementVerifies(buildF1Model, NULL);
    assertTheFfdPlacementVerifies(buildF1Model, &g_liveness);
}

/* F1 ranges: 0 = ACT 1 [0,4], 1 = ACT 2 ([1,4] or [1,3]), 2 = the seed GRAD 2 [3,4]. */
void testVerifierExitsNamingBothWiresOnCoLiveOverlap(void) {
    placedPlan_t pp;
    placeFixture(&pp, buildF1Model, NULL);
    pp.offsets[1] = 0; /* ACT 2 onto ACT 1 */
    ASSERT_EXITS_WITH_OUTPUT(1, "wires ACT 1 and ACT 2 are co-live but share bytes",
                             verifyPlaced(&pp));
    freePlacedFixture(&pp);
}

/* Inclusive intervals: under LIVENESS ACT 2 ends at step 3, where the seed
 * begins; they must not share bytes. */
void testVerifierTreatsRangesMeetingAtOneStepAsCoLive(void) {
    placedPlan_t pp;
    placeFixture(&pp, buildF1Model, &g_liveness);
    TEST_ASSERT_EQUAL_UINT16(3, pp.b.p->train.ranges[1].end);
    TEST_ASSERT_EQUAL_UINT16(3, pp.b.p->train.ranges[2].begin);
    pp.offsets[2] = pp.offsets[1];
    ASSERT_EXITS_WITH_OUTPUT(1, "wires ACT 2 and GRAD 2 are co-live but share bytes",
                             verifyPlaced(&pp));
    freePlacedFixture(&pp);
}

/* [0,8), [12,20), [24,32) inside 32 bytes: disjoint, only the alignment is wrong. */
void testVerifierExitsOnAMisalignedOffset(void) {
    placedPlan_t pp;
    placeFixture(&pp, buildF1Model, NULL);
    pp.offsets[1] = 12;
    pp.offsets[2] = 24;
    pp.bytes = 32;
    ASSERT_EXITS_WITH_OUTPUT(1, "wire ACT 2 at offset 12 is not a multiple of ODT_WIRE_ALIGN (8)",
                             verifyPlaced(&pp));
    freePlacedFixture(&pp);
}

void testVerifierExitsOnARangePastTheArenaEnd(void) {
    placedPlan_t pp;
    placeFixture(&pp, buildF1Model, NULL);
    pp.bytes = 16; /* the seed sits at [16, 24) */
    ASSERT_EXITS_WITH_OUTPUT(1, "(8 placed bytes) ends past the arena's 16 bytes",
                             verifyPlaced(&pp));
    freePlacedFixture(&pp);
}

/* The seed's exact 4 bytes fit [16, 20); its placed 8 do not: the bound is
 * off + placed, not off + exact. */
void testVerifierBoundsPlacedNotExactBytes(void) {
    placedPlan_t pp;
    placeFixture(&pp, buildF1Model, NULL);
    pp.bytes = 20;
    ASSERT_EXITS_WITH_OUTPUT(1, "(8 placed bytes) ends past the arena's 20 bytes",
                             verifyPlaced(&pp));
    freePlacedFixture(&pp);
}

/* An imported offset of SIZE_MAX - 7 (a multiple of 8) makes
 * off + placed wrap to 0; the bound must not be computed that way. */
void testVerifierRejectsAnOffsetNearSizeMax(void) {
    placedPlan_t pp;
    placeFixture(&pp, buildF1Model, NULL);
    pp.offsets[2] = SIZE_MAX - 7u;
    ASSERT_EXITS_WITH_OUTPUT(1, "(8 placed bytes) ends past the arena's 24 bytes",
                             verifyPlaced(&pp));
    freePlacedFixture(&pp);
}

/* ---- init completes: offsets block, placement, verifier, arena block ---- */

static rematReport_t reportAfterInit(void (*build)(fixture_t *), const rematPlanSpec_t *spec) {
    fixture_t f;
    build(&f);
    rematScheduler_t s = initArena(&f, spec);
    rematReport_t r;
    rematSchedulerReport(&s, &r);
    freeFixture(&f, &s);
    return r;
}

void testArenaInitPlacesVerifiesAndReservesTheArena(void) {
    fixture_t f;
    buildHarModel(&f);
    rematScheduler_t s = initArena(&f, &g_liveness);
    rematReport_t r;
    rematSchedulerReport(&s, &r);
    TEST_ASSERT_TRUE(r.planned);
    TEST_ASSERT_TRUE(r.placed);
    TEST_ASSERT_TRUE(r.dataReserved);
    TEST_ASSERT_NOT_NULL(s.row.arena.offsets);
    TEST_ASSERT_NOT_NULL(s.row.arena.base);
    TEST_ASSERT_EQUAL_size_t(s.row.arena.bytes, r.arenaBytes);
    freeFixture(&f, &s);
}

static void assertArenaDecomposition(void (*build)(fixture_t *), const rematPlanSpec_t *spec) {
    rematReport_t r = reportAfterInit(build, spec);
    TEST_ASSERT_TRUE(r.placed);
    TEST_ASSERT_EQUAL_size_t(r.arenaBytes, r.peakLiveBytes + r.arenaPadBytes + r.arenaGapBytes);
}

void testReportArenaBytesIsPeakPlusPadPlusGap(void) {
    assertArenaDecomposition(buildHarModel, NULL);
    assertArenaDecomposition(buildHarModel, &g_liveness);
    assertArenaDecomposition(buildF1Model, NULL);
    assertArenaDecomposition(buildF1Model, &g_liveness);
}

/* Every HAR wire is a multiple of 8 (8192, 4096, 256, 24 B): nothing to pad. */
void testReportPadIsZeroOnHar(void) {
    TEST_ASSERT_EQUAL_size_t(0, reportAfterInit(buildHarModel, NULL).arenaPadBytes);
    TEST_ASSERT_EQUAL_size_t(0, reportAfterInit(buildHarModel, &g_liveness).arenaPadBytes);
}

/* F1: 5 + 4 + 4 = 13 live bytes at step 3 take 8 + 8 + 8 = 24 placed bytes. */
void testReportPadOnTheF1ModelIsEleven(void) {
    for (size_t policy = 0; policy < 2u; policy++) {
        rematReport_t r = reportAfterInit(buildF1Model, policy == 0u ? NULL : &g_liveness);
        TEST_ASSERT_EQUAL_size_t(13, r.peakLiveBytes);
        TEST_ASSERT_EQUAL_size_t(24, r.arenaBytes);
        TEST_ASSERT_EQUAL_size_t(11, r.arenaPadBytes);
        TEST_ASSERT_EQUAL_size_t(0, r.arenaGapBytes);
    }
}

/* The portfolio places HAR at its placed peak, a change needs a stated
 * reason: STORE_ALL 74,288 B (only ping-pong reaches it; first-fit-decreasing
 * alone leaves a 4,096 B gap) and LIVENESS 49,152 B. */
void testReportArenaBytesHarRegressionGuard(void) {
    TEST_ASSERT_EQUAL_size_t(74288, reportAfterInit(buildHarModel, NULL).arenaBytes);
    TEST_ASSERT_EQUAL_size_t(49152, reportAfterInit(buildHarModel, &g_liveness).arenaBytes);
}

/* HAR with ACT 3, ACT 7 and ACT 12 kept (retain[3] = retain[7] = retain[12] =
 * true, the rest false), and with ACT 12 dropped too: both place exactly their
 * 32,768 B peak (first-fit-decreasing alone needs 36,864 B). */
void testReportArenaBytesHarRetainRegressionGuard(void) {
    static const bool keep3And7And12[HAR_N + 1] = {[3] = true, [7] = true, [12] = true};
    static const bool keep3And7[HAR_N + 1] = {[3] = true, [7] = true};
    const bool *masks[2] = {keep3And7And12, keep3And7};
    for (size_t m = 0; m < 2u; m++) {
        const rematPlanSpec_t spec = {.policy = REMAT_PLAN_RETAIN_LIST, .retain = masks[m]};
        rematReport_t r = reportAfterInit(buildHarModel, &spec);
        TEST_ASSERT_EQUAL_size_t(32768, r.arenaBytes);
        TEST_ASSERT_EQUAL_size_t(32768, r.peakLiveBytes);
        TEST_ASSERT_EQUAL_size_t(24576, r.activationsPeakBytes);
    }
}

/* ACT 3, 7 and 12 kept (every other entry false): 7 REFORWARDs that cost
 * 301056 of the forwards' 502150, derived in UnitTestRematPlan above its
 * RETAIN_LIST tests; the same on both rows. */
void testReportCarriesTheRecomputeNumbersOfARetainListPlan(void) {
    static const bool keep3And7And12[HAR_N + 1] = {[3] = true, [7] = true, [12] = true};
    const rematPlanSpec_t spec = {.policy = REMAT_PLAN_RETAIN_LIST, .retain = keep3And7And12};
    for (int row = 0; row < 2; row++) {
        fixture_t f;
        buildHarModel(&f);
        rematScheduler_t s = (row == 0) ? initArena(&f, &spec) : initHeap(&f, &spec);
        rematReport_t r;
        rematSchedulerReport(&s, &r);
        TEST_ASSERT_EQUAL_size_t(7, r.refwdSteps);
        TEST_ASSERT_EQUAL_size_t(502150, r.forwardCost);
        TEST_ASSERT_EQUAL_size_t(301056, r.recomputeCost);
        freeFixture(&f, &s);
    }
}

static void buildFinetuneStage2Model(fixture_t *f) {
    buildHarFixture(f, true);
}

/* Like the HAR guard above, every value is the placed peak; a change needs a
 * stated reason. mnist_cnn: STORE_ALL 175,824 B (only ping-pong reaches it;
 * first-fit-decreasing alone 188,368) and LIVENESS 112,896 B. Finetune stage
 * 2: STORE_ALL 57,928 B; LIVENESS 16,384 B, a chain of 8,192 / 8,192 / 4,096 B
 * buffers that ping-pong alternates between bottom and top (first-fit-
 * decreasing gives the 4,096 B buffer a third slot: 20,480 B). */
void testReportArenaBytesMnistCnnAndFinetuneRegressionGuards(void) {
    TEST_ASSERT_EQUAL_size_t(175824, reportAfterInit(buildMnistCnnModel, NULL).arenaBytes);
    TEST_ASSERT_EQUAL_size_t(112896, reportAfterInit(buildMnistCnnModel, &g_liveness).arenaBytes);
    TEST_ASSERT_EQUAL_size_t(57928, reportAfterInit(buildFinetuneStage2Model, NULL).arenaBytes);
    TEST_ASSERT_EQUAL_size_t(16384,
                             reportAfterInit(buildFinetuneStage2Model, &g_liveness).arenaBytes);
}

/* The arena equals the placed peak on two HAR retain plans where only one
 * member reaches it. Keeping ACT 1, 3, 7 and 12: 40960 B, only through
 * breadth/area (first-fit-decreasing 45056). Keeping ACT 4, 6, 7, 10 and 12
 * with inPlaceDx: 28928 B, only through size-then-end (two-ended 29184,
 * first-fit-decreasing 32768); it is the cheapest in-place plan whose placed
 * peak is at most 28928 B, so the planner's frontier needs this arena. */
void testReportArenaBytesHarRetainPlansAtTheirPeak(void) {
    static const bool keep1_3_7_12[HAR_N + 1] = {[1] = true, [3] = true, [7] = true, [12] = true};
    static const bool keep4_6_7_10_12[HAR_N + 1] = {
        [4] = true, [6] = true, [7] = true, [10] = true, [12] = true};
    const rematPlanSpec_t a = {.policy = REMAT_PLAN_RETAIN_LIST, .retain = keep1_3_7_12};
    rematReport_t r = reportAfterInit(buildHarModel, &a);
    TEST_ASSERT_EQUAL_size_t(40960, r.peakLiveBytes);
    TEST_ASSERT_EQUAL_size_t(40960, r.arenaBytes);
    const rematPlanSpec_t b = {
        .policy = REMAT_PLAN_RETAIN_LIST, .retain = keep4_6_7_10_12, .inPlaceDx = true};
    r = reportAfterInit(buildHarModel, &b);
    TEST_ASSERT_EQUAL_size_t(28928, r.peakLiveBytes);
    TEST_ASSERT_EQUAL_size_t(28928, r.arenaBytes);
}

/* The report's activation-only peak is the TRAIN program's, on either row
 * (the hand-derived HAR values of UnitTestRematPlan). */
void testReportActivationsPeakOnHarOnBothRows(void) {
    TEST_ASSERT_EQUAL_size_t(57904, reportAfterInit(buildHarModel, NULL).activationsPeakBytes);
    TEST_ASSERT_EQUAL_size_t(41216,
                             reportAfterInit(buildHarModel, &g_liveness).activationsPeakBytes);
    fixture_t f;
    buildHarModel(&f);
    rematScheduler_t s = initHeap(&f, &g_liveness);
    rematReport_t r;
    rematSchedulerReport(&s, &r);
    freeFixture(&f, &s);
    TEST_ASSERT_EQUAL_size_t(41216, r.activationsPeakBytes);
}

/* The alignment property: random chains of FLOAT32 and packed BFP wires
 * (odd byte counts) under both policies, every offset and the arena size a
 * multiple of ODT_WIRE_ALIGN. It runs in a child so a verifier exit in init
 * reads as this test's verdict. */
#define MIXED_MAX_RELUS 6u
typedef struct mixedChain {
    uint8_t exponents[3][1];
    bfpQConfig_t bfpQc[3];
    quantization_t bfpQ[3];
    layer_t *model[MIXED_MAX_RELUS + 1u];
    size_t n;
    inputLike_t in;
    tensor_t *x;
} mixedChain_t;

/* A trainable Linear first, so every ReLU above it also gets a GRAD wire of
 * its own dtype, then 1..6 ReLUs whose template is FLOAT32 or BFP m = 3/5/7
 * (per tensor). The feature count 1..9 makes FLOAT32 wires 4 mod 8 when odd. */
static void buildMixedChain(mixedChain_t *c, uint32_t *state) {
    static const uint8_t mantissaBits[3] = {3, 5, 7};
    for (size_t k = 0; k < 3u; k++) {
        initBfpQConfigInto(mantissaBits[k], 8, HALF_AWAY, c->exponents[k], &c->bfpQc[k]);
        c->bfpQ[k] = (quantization_t){.type = BFP, .qConfig = &c->bfpQc[k]};
    }
    size_t features = 1u + nextRandom(state) % 9u;
    c->model[0] = makeLinear(features, features, false, 0);
    c->n = 2u + nextRandom(state) % MIXED_MAX_RELUS;
    for (size_t i = 1; i < c->n; i++) {
        uint32_t pick = nextRandom(state) % 4u;
        c->model[i] = makeRelu(pick == 3u ? &g_floatQ : &c->bfpQ[pick]);
    }
    c->x = makeInput(&c->in, (size_t[]){1, features}, 2, &g_floatQ);
}

static void initMixedChainsAndExitWithTheAlignmentVerdict(void) {
    uint32_t state = 0x5EED1234u;
    bool aligned = true;
    for (size_t trial = 0; trial < 64u; trial++) {
        mixedChain_t c;
        buildMixedChain(&c, &state);
        for (size_t policy = 0; policy < 2u; policy++) {
            rematScheduler_t s;
            if (!rematArenaInit(&s, c.model, c.n, defaultLossConfig(MSE), c.x,
                                policy == 0u ? NULL : &g_liveness)) {
                _exit(3);
            }
            for (size_t r = 0; r < s.plan->train.numRanges; r++) {
                aligned = aligned && s.row.arena.offsets[r] % ODT_WIRE_ALIGN == 0u;
            }
            aligned = aligned && s.row.arena.bytes % ODT_WIRE_ALIGN == 0u;
            rematSchedulerDeinit(&s);
        }
        freeModel(c.model, c.n);
    }
    _exit(aligned ? 0 : 2);
}

void testArenaOffsetsAlignedOnRandomMixedChains(void) {
    ASSERT_EXITS_WITH(0, initMixedChainsAndExitWithTheAlignmentVerdict());
}

#ifdef ODT_MEM_PROFILE
/* Resident means exactly these blocks: table, plan, offsets, arena. */
void testArenaInitReservesExactlyMetadataPlusArena(void) {
    fixture_t f;
    buildHarModel(&f);
    size_t before = memProfileCurrentBytes();
    rematScheduler_t s = initArena(&f, &g_liveness);
    rematReport_t r;
    rematSchedulerReport(&s, &r);
    TEST_ASSERT_EQUAL_size_t(r.metadataBytes + r.arenaBytes, memProfileCurrentBytes() - before);
    freeFixture(&f, &s);
}
#endif

#ifndef ODT_TEST_ASAN
/* Did-not-run pin. ReLU over a borrowed [1, 2^60] FLOAT32 input
 * (never read at init) under MSE passes table init unchanged: one 2^62-byte wire, one range, no
 * backward. Reserving 2^62 B fails on every 64-bit host, and the report must still carry every
 * analytic field. Host-only (LP64); skipped under ASan, which aborts on oversized requests unless
 * allocator_may_return_null=1. macOS malloc prints a "can't allocate region" warning to stderr
 * here; that is expected. */
void testArenaInitFailureKeepsAnalyticReport(void) {
    layer_t *model[1] = {makeRelu(&g_floatQ)};
    inputLike_t in;
    tensor_t *x = makeInput(&in, (size_t[]){1, (size_t)1 << 60}, 2, &g_floatQ);
    size_t before = memProfileCurrentBytes();
    rematScheduler_t s;
    TEST_ASSERT_FALSE(rematArenaInit(&s, model, 1, defaultLossConfig(MSE), x, NULL));
    rematReport_t r;
    rematSchedulerReport(&s, &r);
    TEST_ASSERT_TRUE(r.planned);
    TEST_ASSERT_TRUE(r.placed);
    TEST_ASSERT_FALSE(r.dataReserved);
    TEST_ASSERT_EQUAL_size_t(2, r.numSteps);
    TEST_ASSERT_EQUAL_size_t((size_t)1 << 62, r.peakLiveBytes);
    TEST_ASSERT_EQUAL_size_t(0, r.refwdSteps);
    TEST_ASSERT_EQUAL_size_t((size_t)1 << 60, r.forwardCost);
    TEST_ASSERT_EQUAL_size_t(0, r.recomputeCost);
    TEST_ASSERT_EQUAL_size_t((size_t)1 << 62, r.arenaBytes);
    TEST_ASSERT_EQUAL_size_t(0, r.arenaPadBytes);
    TEST_ASSERT_EQUAL_size_t(0, r.arenaGapBytes);
    TEST_ASSERT_EQUAL_size_t(s.wires->slabBytes + s.plan->blockBytes + sizeof(size_t),
                             r.metadataBytes);
    TEST_ASSERT_TRUE(r.metadataBytes > 0u);
    rematSchedulerDeinit(&s);
    TEST_ASSERT_EQUAL_size_t(before, memProfileCurrentBytes());
    freeModel(model, 1);
}
#endif

/* ---- init's named exits before any row reservation ---- */

/* The child prints how many bytes are live when it exits, so the parent can
 * check "before any row reservation": only the shared table and plan may
 * exist. Real only under ODT_MEM_PROFILE; elsewhere both sides read 0. */
static size_t g_memBeforeExit;

static void printReservedBeforeExit(void) {
    printf("reservedBeforeExit=%zu\n", memProfileCurrentBytes() - g_memBeforeExit);
}

static void arenaInitExpectingAnExit(layer_t **model, size_t n, lossFuncType_t lt,
                                     const tensor_t *x) {
    rematScheduler_t s;
    g_memBeforeExit = memProfileCurrentBytes();
    (void)atexit(printReservedBeforeExit);
    (void)rematArenaInit(&s, model, n, defaultLossConfig(lt), x, NULL);
}

static size_t sharedBlockBytes(layer_t **model, size_t n, lossFuncType_t lt, const tensor_t *x) {
#ifdef ODT_MEM_PROFILE
    builtPlan_t b = buildTableAndPlan(model, n, lt, x, NULL);
    size_t bytes = b.t->slabBytes + b.p->blockBytes;
    freeTableAndPlan(&b);
    return bytes;
#else
    (void)model;
    (void)n;
    (void)lt;
    (void)x;
    return 0u;
#endif
}

static void assertExitsBeforeAnyRowReservation(layer_t **model, size_t n, lossFuncType_t lt,
                                               const tensor_t *x, const char *message) {
    ASSERT_EXITS_WITH_OUTPUT(1, message, arenaInitExpectingAnExit(model, n, lt, x));
    char reserved[64];
    (void)snprintf(reserved, sizeof reserved, "reservedBeforeExit=%zu\n",
                   sharedBlockBytes(model, n, lt, x));
    ASSERT_EXITS_WITH_OUTPUT(1, reserved, arenaInitExpectingAnExit(model, n, lt, x));
}

/* A chain of one ReLU object over [1,1] FLOAT32 under MSE: nothing trains, so
 * there is no GRAD wire and exactly n ranges (ACT 1..n). */
static void fillReluChain(layer_t **model, size_t n, layer_t *relu) {
    for (size_t i = 0; i < n; i++) {
        model[i] = relu;
    }
}

/* The chain arrays below live on the test stack (8 KiB at the default); a host
 * build that raises the limit much further must reserve them instead. */
_Static_assert(ODT_REMAT_MAX_RANGES <= 4096u, "the ReLU-chain arrays live on the test stack");

static size_t tableBlockBytes(layer_t **model, size_t n, lossFuncType_t lt, const tensor_t *x) {
#ifdef ODT_MEM_PROFILE
    rematWireTable_t *t = NULL;
    TEST_ASSERT_TRUE(rematWireTableInit(&t, model, n, defaultLossConfig(lt), x));
    size_t bytes = t->slabBytes;
    rematWireTableFree(t);
    return bytes;
#else
    (void)model;
    (void)n;
    (void)lt;
    (void)x;
    return 0u;
#endif
}

/* An oversized chain is refused from the table alone, before the plan block
 * is reserved: only the table is live at the exit, and the message is the
 * pre-check's, not the post-build check's ("plan has ... ranges"). */
void testArenaInitExitsAboveMaxRangesBeforeThePlanIsBuilt(void) {
    layer_t *relu = makeRelu(&g_floatQ);
    layer_t *model[ODT_REMAT_MAX_RANGES + 1u];
    fillReluChain(model, ODT_REMAT_MAX_RANGES + 1u, relu);
    inputLike_t in;
    tensor_t *x = makeInput(&in, (size_t[]){1, 1}, 2, &g_floatQ);
    char message[96];
    (void)snprintf(message, sizeof message,
                   "model needs %u ranges, above ODT_REMAT_MAX_RANGES (%u)",
                   (unsigned)(ODT_REMAT_MAX_RANGES + 1u), (unsigned)ODT_REMAT_MAX_RANGES);
    ASSERT_EXITS_WITH_OUTPUT(1, message,
                             arenaInitExpectingAnExit(model, ODT_REMAT_MAX_RANGES + 1u, MSE, x));
    char reserved[64];
    (void)snprintf(reserved, sizeof reserved, "reservedBeforeExit=%zu\n",
                   tableBlockBytes(model, ODT_REMAT_MAX_RANGES + 1u, MSE, x));
    ASSERT_EXITS_WITH_OUTPUT(1, reserved,
                             arenaInitExpectingAnExit(model, ODT_REMAT_MAX_RANGES + 1u, MSE, x));
    freeReluLayer(relu);
}

static void initAndExitWithTheVerdict(layer_t **model, size_t n, const tensor_t *x) {
    rematScheduler_t s;
    bool ok = rematArenaInit(&s, model, n, defaultLossConfig(MSE), x, NULL) &&
              s.plan->train.numRanges == n;
    rematSchedulerDeinit(&s);
    _exit(ok ? 0 : 2);
}

/* The limit is inclusive -- a plan of exactly ODT_REMAT_MAX_RANGES
 * ranges initialises. */
void testArenaInitAcceptsExactlyMaxRanges(void) {
    layer_t *relu = makeRelu(&g_floatQ);
    layer_t *model[ODT_REMAT_MAX_RANGES];
    fillReluChain(model, ODT_REMAT_MAX_RANGES, relu);
    inputLike_t in;
    tensor_t *x = makeInput(&in, (size_t[]){1, 1}, 2, &g_floatQ);
    ASSERT_EXITS_WITH(0, initAndExitWithTheVerdict(model, ODT_REMAT_MAX_RANGES, x));
    freeReluLayer(relu);
}

/* ---- the range cap of the new plan kinds (HEAP and ARENA) ---- */

/* The ReLU chain's STORE_ALL steps, written out: FORWARD 0..n-1, LOSS_FORWARD.
 * As a SEQUENCE they open one range per FORWARD, n in all. */
static size_t fillChainSequence(rematStep_t *steps, size_t n) {
    for (size_t l = 0; l < n; l++) {
        steps[l] = (rematStep_t){.kind = REMAT_STEP_FORWARD, .layer = (uint16_t)l};
    }
    steps[n] = (rematStep_t){.kind = REMAT_STEP_LOSS_FORWARD, .layer = (uint16_t)n};
    return n + 1u;
}

static void heapInitAndExitWithTheVerdict(layer_t **model, size_t n, const tensor_t *x,
                                          const rematPlanSpec_t *spec, size_t numRanges) {
    rematScheduler_t s;
    bool ok = rematHeapInit(&s, model, n, defaultLossConfig(MSE), x, spec) &&
              s.plan->train.numRanges == numRanges;
    rematSchedulerDeinit(&s);
    _exit(ok ? 0 : 2);
}

static void heapInitWithSpecExpectingAnExit(layer_t **model, size_t n, const tensor_t *x,
                                            const rematPlanSpec_t *spec) {
    rematScheduler_t s;
    g_memBeforeExit = memProfileCurrentBytes();
    (void)atexit(printReservedBeforeExit);
    (void)rematHeapInit(&s, model, n, defaultLossConfig(MSE), x, spec);
}

static void arenaInitWithSpecExpectingAnExit(layer_t **model, size_t n, const tensor_t *x,
                                             const rematPlanSpec_t *spec) {
    rematScheduler_t s;
    (void)rematArenaInit(&s, model, n, defaultLossConfig(MSE), x, spec);
}

/* HEAP admits a STORE_ALL or LIVENESS plan of any size: the cap covers only
 * the plan kinds that can hold more ranges than wires. */
void testHeapInitAcceptsAStoreAllPlanAboveMaxRanges(void) {
    layer_t *relu = makeRelu(&g_floatQ);
    layer_t *model[ODT_REMAT_MAX_RANGES + 1u];
    fillReluChain(model, ODT_REMAT_MAX_RANGES + 1u, relu);
    inputLike_t in;
    tensor_t *x = makeInput(&in, (size_t[]){1, 1}, 2, &g_floatQ);
    ASSERT_EXITS_WITH(0, heapInitAndExitWithTheVerdict(model, ODT_REMAT_MAX_RANGES + 1u, x, NULL,
                                                       ODT_REMAT_MAX_RANGES + 1u));
    freeReluLayer(relu);
}

/* The same chain's steps as a SEQUENCE: 1025 ranges exit in rematPlanBuild,
 * before the plan block exists (only the table is live at the exit). */
void testHeapInitExitsOnASequenceAboveMaxRangesBeforeThePlanBlock(void) {
    layer_t *relu = makeRelu(&g_floatQ);
    layer_t *model[ODT_REMAT_MAX_RANGES + 1u];
    fillReluChain(model, ODT_REMAT_MAX_RANGES + 1u, relu);
    rematStep_t steps[ODT_REMAT_MAX_RANGES + 2u];
    rematPlanSpec_t spec = {.policy = REMAT_PLAN_SEQUENCE,
                            .sequence = steps,
                            .sequenceLen = fillChainSequence(steps, ODT_REMAT_MAX_RANGES + 1u)};
    inputLike_t in;
    tensor_t *x = makeInput(&in, (size_t[]){1, 1}, 2, &g_floatQ);
    char message[96];
    (void)snprintf(message, sizeof message,
                   "rematPlanBuild: plan has %u ranges, above ODT_REMAT_MAX_RANGES (%u)",
                   (unsigned)(ODT_REMAT_MAX_RANGES + 1u), (unsigned)ODT_REMAT_MAX_RANGES);
    ASSERT_EXITS_WITH_OUTPUT(
        1, message, heapInitWithSpecExpectingAnExit(model, ODT_REMAT_MAX_RANGES + 1u, x, &spec));
    char reserved[64];
    (void)snprintf(reserved, sizeof reserved, "reservedBeforeExit=%zu\n",
                   tableBlockBytes(model, ODT_REMAT_MAX_RANGES + 1u, MSE, x));
    ASSERT_EXITS_WITH_OUTPUT(
        1, reserved, heapInitWithSpecExpectingAnExit(model, ODT_REMAT_MAX_RANGES + 1u, x, &spec));
    freeReluLayer(relu);
}

void testHeapInitAcceptsASequenceOfExactlyMaxRanges(void) {
    layer_t *relu = makeRelu(&g_floatQ);
    layer_t *model[ODT_REMAT_MAX_RANGES];
    fillReluChain(model, ODT_REMAT_MAX_RANGES, relu);
    rematStep_t steps[ODT_REMAT_MAX_RANGES + 1u];
    rematPlanSpec_t spec = {.policy = REMAT_PLAN_SEQUENCE,
                            .sequence = steps,
                            .sequenceLen = fillChainSequence(steps, ODT_REMAT_MAX_RANGES)};
    inputLike_t in;
    tensor_t *x = makeInput(&in, (size_t[]){1, 1}, 2, &g_floatQ);
    ASSERT_EXITS_WITH(0, heapInitAndExitWithTheVerdict(model, ODT_REMAT_MAX_RANGES, x, &spec,
                                                       ODT_REMAT_MAX_RANGES));
    freeReluLayer(relu);
}

/* The minimum-memory schedule of a chain of 44 trainable Linear(1->1) under
 * MSE: after LOSS_BACKWARD, BACKWARD(l) for l = 43 down to 0, each preceded
 * by the REFORWARD run 0..l-1 that regenerates its input from ACT 0. Writes:
 * 44 FORWARDs, LOSS_BACKWARD, the 43 BACKWARDs above deepest, and
 * 43 * 44 / 2 = 946 REFORWARDs: 1034 ranges over 88 slab wires, so ARENA's
 * pre-check (88 wires) passes and the plan's own cap decides. */
#define MIN_MEMORY_CHAIN_N 44u
static size_t fillMinMemorySequence(rematStep_t *steps, size_t n) {
    size_t s = fillChainSequence(steps, n);
    steps[s++] = (rematStep_t){.kind = REMAT_STEP_LOSS_BACKWARD, .layer = (uint16_t)n};
    for (size_t l = n; l-- > 0;) {
        for (size_t k = 0; k < l; k++) {
            steps[s++] = (rematStep_t){.kind = REMAT_STEP_REFORWARD, .layer = (uint16_t)k};
        }
        steps[s++] = (rematStep_t){.kind = REMAT_STEP_BACKWARD, .layer = (uint16_t)l};
    }
    return s;
}

void testArenaInitExitsOnASequenceAboveMaxRangesInThePlanBuild(void) {
    layer_t *model[MIN_MEMORY_CHAIN_N];
    for (size_t i = 0; i < MIN_MEMORY_CHAIN_N; i++) {
        model[i] = makeLinear(1, 1, false, i);
    }
    rematStep_t steps[1036];
    rematPlanSpec_t spec = {.policy = REMAT_PLAN_SEQUENCE,
                            .sequence = steps,
                            .sequenceLen = fillMinMemorySequence(steps, MIN_MEMORY_CHAIN_N)};
    TEST_ASSERT_EQUAL_size_t(1036, spec.sequenceLen);
    inputLike_t in;
    tensor_t *x = makeInput(&in, (size_t[]){1, 1}, 2, &g_floatQ);
    ASSERT_EXITS_WITH_OUTPUT(1,
                             "rematPlanBuild: plan has 1034 ranges, above ODT_REMAT_MAX_RANGES "
                             "(1024)",
                             arenaInitWithSpecExpectingAnExit(model, MIN_MEMORY_CHAIN_N, x, &spec));
    freeModel(model, MIN_MEMORY_CHAIN_N);
}

/* ACT 1 holds 4 * (2^62 - 1) = SIZE_MAX - 3 bytes; rounding it up to 8 wraps. */
void testArenaInitExitsOnAPlacedSizeOverflowBeforeAnyRowReservation(void) {
    layer_t *model[1] = {makeRelu(&g_floatQ)};
    inputLike_t in;
    tensor_t *x = makeInput(&in, (size_t[]){1, SIZE_MAX / 4u}, 2, &g_floatQ);
    assertExitsBeforeAnyRowReservation(model, 1, MSE, x,
                                       "size overflow computing placed bytes at wire ACT 1");
    freeModel(model, 1);
}

/* ACT 1 = 2^63 B; the AvgPool (k 2, stride 1, VALID) output has length
 * 2^61 - 1, so ACT 2 = 2^63 - 4 B; the table's total 2^64 - 4 fits, but the
 * placed sizes 2^63 + 2^63 do not. */
void testArenaInitExitsOnAnArenaSumOverflowBeforeAnyRowReservation(void) {
    layerQuant_t lq;
    layerQuantInitUniform(&lq, &g_floatQ);
    layer_t *model[2] = {makeRelu(&g_floatQ),
                         avgPool1dLayerInit(&(avgPool1dInit_t){.kernelSize = 2, .stride = 1}, &lq)};
    inputLike_t in;
    tensor_t *x = makeInput(&in, (size_t[]){1, 1, (size_t)1 << 61}, 3, &g_floatQ);
    assertExitsBeforeAnyRowReservation(model, 2, MSE, x,
                                       "size overflow computing placed bytes total at wire ACT 2");
    freeModel(model, 2);
}

/* ---- the row's per-step entry points ---- */

/* One call through the dispatch, on either row. */
static void bindAndBegin(fixture_t *f, rematScheduler_t *s) {
    rematBegin(s, f->model, f->n, defaultLossConfig(f->lt), f->x);
}

static void endAndUnbind(rematScheduler_t *s) {
    rematEnd(s);
}

/* One call in `mode`: rematBegin for TRAIN, rematBeginEval for EVAL. */
static void bindAndBeginIn(fixture_t *f, rematScheduler_t *s, rematMode_t mode) {
    if (mode == REMAT_MODE_EVAL) {
        rematBeginEval(s, f->model, f->n, f->lt, f->x);
    } else {
        bindAndBegin(f, s);
    }
}

static size_t walkAll(fixture_t *f, rematScheduler_t *s) {
    bindAndBegin(f, s);
    rematStep_t st;
    size_t steps = 0;
    while (rematNext(s, &st)) {
        rematDone(s, &st);
        steps++;
    }
    endAndUnbind(s);
    return steps;
}

void testArenaNextHandsOutThePlanStepsInOrder(void) {
    fixture_t f;
    buildHarModel(&f);
    rematScheduler_t s = initArena(&f, NULL);
    const rematProgram_t *p = &s.plan->train;
    bindAndBegin(&f, &s);
    rematStep_t st;
    size_t i = 0;
    while (rematNext(&s, &st)) {
        TEST_ASSERT_TRUE(i < p->numSteps);
        TEST_ASSERT_EQUAL_UINT8(p->steps[i].kind, st.kind);
        TEST_ASSERT_EQUAL_UINT16(p->steps[i].layer, st.layer);
        rematDone(&s, &st);
        i++;
    }
    TEST_ASSERT_EQUAL_size_t(25, i);
    endAndUnbind(&s);
    freeFixture(&f, &s);
}

void testArenaNextAfterTheStreamCompletedReturnsFalse(void) {
    fixture_t f;
    buildF1Model(&f);
    rematScheduler_t s = initArena(&f, &g_liveness);
    bindAndBegin(&f, &s);
    rematStep_t st;
    while (rematNext(&s, &st)) {
        rematDone(&s, &st);
    }
    size_t live = s.wires->liveBytes;
    TEST_ASSERT_FALSE(rematNext(&s, &st));
    TEST_ASSERT_EQUAL_size_t(live, s.wires->liveBytes);
    endAndUnbind(&s);
    freeFixture(&f, &s);
}

/* PR1a carry: the walk restarts at every begin, on the same resident arena. */
void testArenaSecondCallRestartsTheWalk(void) {
    fixture_t f;
    buildF1Model(&f);
    rematScheduler_t s = initArena(&f, &g_liveness);
    TEST_ASSERT_EQUAL_size_t(5, walkAll(&f, &s));
    TEST_ASSERT_EQUAL_size_t(5, walkAll(&f, &s));
    TEST_ASSERT_EQUAL_UINT32(1, s.wires->wires[1].bindGen); /* the table bind reset it */
    freeFixture(&f, &s);
}

static void doneForAStepNextDidNotHandOut(fixture_t *f, rematScheduler_t *s) {
    bindAndBegin(f, s);
    rematStep_t st;
    (void)rematNext(s, &st);
    rematDone(s, &(rematStep_t){.kind = REMAT_STEP_BACKWARD, .layer = 0});
}

static void doneBeforeNext(fixture_t *f, rematScheduler_t *s) {
    bindAndBegin(f, s);
    rematDone(s, &s->plan->train.steps[0]);
}

static void doneWithoutNextAtLossForward(fixture_t *f, rematScheduler_t *s) {
    bindAndBegin(f, s);
    rematStep_t st;
    for (size_t k = 0; k < 2u; k++) {
        (void)rematNext(s, &st);
        rematDone(s, &st);
    }
    rematDone(s, &s->plan->train.steps[2]);
}

static void nextTwice(fixture_t *f, rematScheduler_t *s) {
    bindAndBegin(f, s);
    rematStep_t st;
    (void)rematNext(s, &st);
    (void)rematNext(s, &st);
}

static void doneAfterTheStreamCompleted(fixture_t *f, rematScheduler_t *s) {
    bindAndBegin(f, s);
    rematStep_t st;
    rematStep_t last = {0};
    while (rematNext(s, &st)) {
        rematDone(s, &st);
        last = st;
    }
    rematDone(s, &last);
}

static void endAfterOneStep(fixture_t *f, rematScheduler_t *s) {
    bindAndBegin(f, s);
    rematStep_t st;
    (void)rematNext(s, &st);
    rematDone(s, &st);
    rematEnd(s);
}

void testArenaEndExitsOnAnIncompleteWalk(void) {
    fixture_t f;
    buildF1Model(&f);
    rematScheduler_t s = initArena(&f, &g_liveness);
    ASSERT_EXITS_WITH_OUTPUT(
        1, "rematEnd before the walk completed: 1 of 5 steps done, 0 of 3 ranges closed",
        endAfterOneStep(&f, &s));
    freeFixture(&f, &s);
}

/* A range tampered after the build and ending past the last step would stay
 * bound across the call boundary: grammar rule 6 keeps a built plan's ranges
 * inside the stream, so only the walk's end check sees it. F1 STORE_ALL:
 * every range ends at step 4, the seed is last in endOrder. */
static void walkWithTheLastRangeLeftOpen(fixture_t *f, rematScheduler_t *s) {
    rematProgram_t *p = &s->plan->train;
    p->ranges[p->endOrder[p->numRanges - 1u]].end = (uint16_t)p->numSteps;
    (void)walkAll(f, s);
}

void testArenaEndExitsOnARangeLeftOpen(void) {
    fixture_t f;
    buildF1Model(&f);
    rematScheduler_t s = initArena(&f, NULL);
    ASSERT_EXITS_WITH_OUTPUT(
        1, "rematEnd before the walk completed: 5 of 5 steps done, 2 of 3 ranges closed",
        walkWithTheLastRangeLeftOpen(&f, &s));
    freeFixture(&f, &s);
}

static void beginOnAnUnreservedArena(fixture_t *f, rematScheduler_t *s) {
    s->row.arena.base = NULL; /* the state rematArenaInit leaves after a failed data block */
    bindAndBegin(f, s);
}

void testArenaBeginExitsWhenTheArenaWasNeverReserved(void) {
    fixture_t f;
    buildF1Model(&f);
    rematScheduler_t s = initArena(&f, NULL);
    ASSERT_EXITS_WITH_OUTPUT(1, "rematBegin on a scheduler whose arena was never reserved",
                             beginOnAnUnreservedArena(&f, &s));
    freeFixture(&f, &s);
}

/* ---- the row contract on every row ---- */

static void assertResident(const rematWireTable_t *t, uint16_t w, const char *what) {
    TEST_ASSERT_NOT_EQUAL_MESSAGE(REMAT_NONE, w, what);
    if (w == 0) {
        return; /* ACT 0 is the caller's borrowed input, checked once per call */
    }
    TEST_ASSERT_NOT_NULL_MESSAGE(rematWireHdr(t, w)->data, what);
}

/* Everything a step reads is resident and everything it writes is bound. A
 * BACKWARD reads ACT l only where layerBackwardReadsInput says so; elsewhere
 * ACT l may be NULL (W_dead, the LIVENESS state of a non-reading BACKWARD). */
static void assertOperandsResident(const rematWireTable_t *t, layer_t **model,
                                   const rematStep_t *st) {
    size_t n = t->modelSize;
    size_t l = st->layer;
    switch (st->kind) {
    case REMAT_STEP_FORWARD:
    case REMAT_STEP_REFORWARD:
        assertResident(t, rematActId(t, l), "FORWARD or REFORWARD reads its input");
        assertResident(t, rematActId(t, l + 1u), "FORWARD or REFORWARD writes its output");
        break;
    case REMAT_STEP_LOSS_FORWARD:
        assertResident(t, rematActId(t, n), "LOSS_FORWARD reads ACT n");
        break;
    case REMAT_STEP_LOSS_BACKWARD:
        assertResident(t, rematActId(t, n), "LOSS_BACKWARD reads ACT n");
        assertResident(t, rematGradId(t, n), "LOSS_BACKWARD writes the seed");
        break;
    case REMAT_STEP_BACKWARD: {
        uint16_t gradIn =
            ((ptrdiff_t)l == t->backwardTop) ? rematGradId(t, n) : rematGradId(t, l + 1u);
        assertResident(t, gradIn, "BACKWARD reads gradIn");
        if (layerBackwardReadsInput(model[l])) {
            assertResident(t, rematActId(t, l), "BACKWARD reads its input");
        }
        if (l > t->deepest) {
            assertResident(t, rematGradId(t, l), "BACKWARD writes its dx");
        }
        break;
    }
    default:
        TEST_FAIL_MESSAGE("a row handed out a step of unknown kind");
    }
}

/* The wire whose range holds this one's bytes: an in-place member walks down
 * to its chain's root; with inPlace NULL every wire is its own. */
static uint16_t rootOf(const uint8_t *inPlace, uint16_t w) {
    while (inPlace != NULL && inPlace[w] != 0u) {
        w--;
    }
    return w;
}

/* Two wires of one chain share their root's bytes and bind generation, and
 * nothing else: each keeps its own header, quantization, qConfig (FLOAT32 has
 * none) and shape. */
static void assertOneChain(const rematWireTable_t *t, uint16_t a, uint16_t b) {
    const tensor_t *ha = rematWireHdr(t, a);
    const tensor_t *hb = rematWireHdr(t, b);
    TEST_ASSERT_EQUAL_PTR_MESSAGE(ha->data, hb->data, "a chain's wires do not share its bytes");
    TEST_ASSERT_EQUAL_size_t(rematWireBytes(t, a), rematWireBytes(t, b));
    TEST_ASSERT_EQUAL_UINT32_MESSAGE(t->wires[a].bindGen, t->wires[b].bindGen,
                                     "a chain's wires differ in bind generation");
    TEST_ASSERT_TRUE_MESSAGE(ha != hb && ha->quantization != hb->quantization &&
                                 ha->shape != hb->shape,
                             "a chain's wires share a header, a quantization or a shape");
    TEST_ASSERT_TRUE_MESSAGE(ha->quantization->qConfig == NULL ||
                                 ha->quantization->qConfig != hb->quantization->qConfig,
                             "a chain's wires share a qConfig");
}

/* O(W^2), tests only. inPlace: the marks of the program walked (NULL: none);
 * the wires of one chain must share their bytes exactly, all others none. */
static void assertResidentHeadersDisjoint(const rematWireTable_t *t, const uint8_t *inPlace) {
    for (uint16_t a = 1; a < t->numWires; a++) {
        const uint8_t *da = rematWireHdr(t, a)->data;
        if (da == NULL) {
            continue;
        }
        for (uint16_t b = (uint16_t)(a + 1u); b < t->numWires; b++) {
            const uint8_t *db = rematWireHdr(t, b)->data;
            if (db == NULL) {
                continue;
            }
            if (rootOf(inPlace, a) == rootOf(inPlace, b)) {
                assertOneChain(t, a, b);
                continue;
            }
            /* Integer intervals: HEAP wires live in separate blocks, whose
             * pointers C does not order. */
            uintptr_t ua = (uintptr_t)da;
            uintptr_t ub = (uintptr_t)db;
            bool disjoint = ua + rematWireBytes(t, a) <= ub || ub + rematWireBytes(t, b) <= ua;
            TEST_ASSERT_TRUE_MESSAGE(disjoint, "two resident headers share bytes");
        }
    }
}

/* A range that ended releases its wire and every member of its chain. */
static void assertEndedRangesReleased(const rematProgram_t *p, const uint8_t *inPlace,
                                      const rematWireTable_t *t, size_t step) {
    for (size_t r = 0; r < p->numRanges; r++) {
        if (p->ranges[r].end != step) {
            continue;
        }
        uint16_t w = p->ranges[r].wire;
        do {
            TEST_ASSERT_NULL_MESSAGE(rematWireHdr(t, w)->data,
                                     "a range that ended at this step is still bound");
            w++;
        } while (w < t->numWires && rootOf(inPlace, w) == p->ranges[r].wire);
    }
}

/* A row binds a wire at its range's first step and releases it after its last
 * (P8), no earlier and no later: the memory check compares against the SDK's
 * liveBytes, which an early bind raises too, so only this pins the bound set.
 * After next a range ending at the step is still live; after done it is not.
 * An in-place member is live exactly while its chain root's range is. */
static void assertBoundIsExactlyTheLiveSet(const rematProgram_t *p, const uint8_t *inPlace,
                                           const rematWireTable_t *t, size_t step, bool afterDone) {
    for (uint16_t w = 1; w < t->numWires; w++) {
        uint16_t root = rootOf(inPlace, w);
        bool live = false;
        for (size_t r = 0; r < p->numRanges; r++) {
            const rematRange_t *rg = &p->ranges[r];
            if (rg->wire == root && rg->begin <= step &&
                (afterDone ? rg->end > step : rg->end >= step)) {
                live = true;
            }
        }
        TEST_ASSERT_EQUAL_MESSAGE(live, rematWireHdr(t, w)->data != NULL,
                                  "a wire is bound outside its range, or unbound inside it");
    }
}

/* Every bound wire's bytes suit every wire dtype: the ARENA
 * offsets are multiples of ODT_WIRE_ALIGN and HEAP blocks are max-aligned. */
static void assertBoundDataAligned(const rematWireTable_t *t) {
    for (uint16_t w = 1; w < t->numWires; w++) {
        const uint8_t *data = rematWireHdr(t, w)->data;
        if (data != NULL) {
            TEST_ASSERT_EQUAL_size_t_MESSAGE(0, (uintptr_t)data % ODT_WIRE_ALIGN,
                                             "a bound wire's data is not ODT_WIRE_ALIGN-aligned");
        }
    }
}

/* What a row holds beyond its init blocks at any point of a call. ARENA holds
 * every live byte in its resident block, so nothing. HEAP holds one block of
 * exactly bytes(w) per live range, and the counter adds requested bytes, not
 * the allocator header (StorageApi.c:41), so exactly the SDK's liveBytes.
 * Without ODT_MEM_PROFILE the counter reads 0 and so does this. */
static size_t heldInsideTheCall(const rematScheduler_t *s) {
#ifdef ODT_MEM_PROFILE
    return s->type == REMAT_HEAP ? s->wires->liveBytes : 0u;
#else
    (void)s;
    return 0u;
#endif
}

/* One call with every row-contract assert, plus: every bound wire aligned,
 * exactly the plan's live wires bound, and the row's reserved bytes exactly what it must hold after
 * every next and every done, peaking at the plan's peak on HEAP (0 on ARENA). */
static void walkWithTheContractChecks(fixture_t *f, rematScheduler_t *s, rematMode_t mode) {
    rematWireTable_t *t = s->wires;
    const rematProgram_t *p = rematPlanProgram(s->plan, mode);
    const uint8_t *inPlace = (mode == REMAT_MODE_TRAIN) ? s->plan->inPlace : NULL;
    size_t memAfterInit = memProfileCurrentBytes();
    size_t peakHeld = 0;
    bindAndBeginIn(f, s, mode);
    TEST_ASSERT_EQUAL_PTR(f->x, rematActHdr(t, 0));
    TEST_ASSERT_EQUAL_PTR_MESSAGE(inPlace, t->inPlace,
                                  "the table borrows the plan's marks for a training call only");
    rematStep_t st;
    size_t step = 0;
    while (rematNext(s, &st)) {
        assertOperandsResident(t, f->model, &st);
        assertResidentHeadersDisjoint(t, inPlace);
        assertBoundDataAligned(t);
        assertBoundIsExactlyTheLiveSet(p, inPlace, t, step, false);
        size_t held = memProfileCurrentBytes() - memAfterInit;
        TEST_ASSERT_EQUAL_size_t_MESSAGE(
            heldInsideTheCall(s), held, "the row holds other bytes than its live wires after next");
        if (held > peakHeld) {
            peakHeld = held;
        }
        rematDone(s, &st);
        assertEndedRangesReleased(p, inPlace, t, step);
        assertBoundIsExactlyTheLiveSet(p, inPlace, t, step, true);
        TEST_ASSERT_EQUAL_size_t_MESSAGE(
            heldInsideTheCall(s), memProfileCurrentBytes() - memAfterInit,
            "the row holds other bytes than its live wires after done");
        step++;
    }
    TEST_ASSERT_EQUAL_size_t(p->numSteps, step);
#ifdef ODT_MEM_PROFILE
    TEST_ASSERT_EQUAL_size_t(s->type == REMAT_HEAP ? p->peakLiveBytes : 0u, peakHeld);
#endif
    endAndUnbind(s);
    for (uint16_t w = 1; w < t->numWires; w++) {
        TEST_ASSERT_NULL_MESSAGE(rematWireHdr(t, w)->data, "a wire is still bound after end");
    }
    TEST_ASSERT_NULL(rematActHdr(t, 0));
    TEST_ASSERT_NULL_MESSAGE(t->inPlace, "the table still borrows the marks after end");
    TEST_ASSERT_EQUAL_size_t(memAfterInit, memProfileCurrentBytes());
}

/* Two calls: ARENA reuses its resident bytes without zeroing, HEAP reserves
 * fresh blocks (VERIFY poisons both at every bind on the test presets). */
static void assertRowContract(rowInit_t init, void (*build)(fixture_t *),
                              const rematPlanSpec_t *spec) {
    fixture_t f;
    build(&f);
    rematScheduler_t s = init(&f, spec);
    walkWithTheContractChecks(&f, &s, REMAT_MODE_TRAIN);
    walkWithTheContractChecks(&f, &s, REMAT_MODE_TRAIN);
    freeFixture(&f, &s);
}

/* Evaluation on a persistent scheduler between training calls: every call
 * walks its own mode's program under every contract check (on HEAP one
 * exactly-sized block per EVAL range, peaking at the EVAL program's peak), and
 * the training call after two eval calls is unaffected. */
static void assertRowContractAcrossModes(rowInit_t init, void (*build)(fixture_t *),
                                         const rematPlanSpec_t *spec) {
    fixture_t f;
    build(&f);
    rematScheduler_t s = init(&f, spec);
    walkWithTheContractChecks(&f, &s, REMAT_MODE_TRAIN);
    walkWithTheContractChecks(&f, &s, REMAT_MODE_EVAL);
    walkWithTheContractChecks(&f, &s, REMAT_MODE_EVAL);
    walkWithTheContractChecks(&f, &s, REMAT_MODE_TRAIN);
    freeFixture(&f, &s);
}

void testRowContractHeapHarAcrossModes(void) {
    assertRowContractAcrossModes(initHeap, buildHarModel, NULL);
}

void testRowContractHeapF1AcrossModesLiveness(void) {
    assertRowContractAcrossModes(initHeap, buildF1Model, &g_liveness);
}

/* On ARENA the walk also pins "eval adds 0 B": under ODT_MEM_PROFILE the
 * row holds nothing beyond its init blocks at any step of an eval call. */
void testRowContractArenaHarAcrossModes(void) {
    assertRowContractAcrossModes(initArena, buildHarModel, NULL);
}

void testRowContractArenaF1AcrossModesLiveness(void) {
    assertRowContractAcrossModes(initArena, buildF1Model, &g_liveness);
}

/* EVAL places two-ended inside the training arena: an even ACT at offset 0,
 * an odd ACT top-aligned at bytes - placed. ACT j and ACT j+1 are co-live at
 * TRAIN's FORWARD(j), so the verified TRAIN layout proves they fit together. */
static void assertEvalOffsetsAreTwoEnded(void (*build)(fixture_t *)) {
    fixture_t f;
    build(&f);
    rematScheduler_t s = initArena(&f, NULL);
    rematBeginEval(&s, f.model, f.n, f.lt, f.x);
    rematStep_t st;
    while (rematNext(&s, &st)) {
        for (uint16_t j = 1; j <= f.n; j++) {
            const uint8_t *data = rematActHdr(s.wires, j)->data;
            if (data == NULL) {
                continue;
            }
            size_t offset = (size_t)(data - s.row.arena.base);
            size_t expected =
                (j % 2u == 0u) ? 0u
                               : s.row.arena.bytes - arenaPlaced(s.wires, rematActId(s.wires, j));
            TEST_ASSERT_EQUAL_size_t_MESSAGE(expected, offset, "an EVAL ACT is not two-ended");
        }
        rematDone(&s, &st);
    }
    rematEnd(&s);
    freeFixture(&f, &s);
}

void testArenaEvalPlacesEvenActsAtTheBottomAndOddActsAtTheTop(void) {
    assertEvalOffsetsAreTwoEnded(buildHarModel);
    assertEvalOffsetsAreTwoEnded(buildF1Model);
}

void testRowContractArenaHarStoreAll(void) {
    assertRowContract(initArena, buildHarModel, NULL);
}

void testRowContractArenaHarLiveness(void) {
    assertRowContract(initArena, buildHarModel, &g_liveness);
}

void testRowContractArenaF1StoreAll(void) {
    assertRowContract(initArena, buildF1Model, NULL);
}

void testRowContractArenaF1Liveness(void) {
    assertRowContract(initArena, buildF1Model, &g_liveness);
}

void testRowContractHeapHarStoreAll(void) {
    assertRowContract(initHeap, buildHarModel, NULL);
}

void testRowContractHeapHarLiveness(void) {
    assertRowContract(initHeap, buildHarModel, &g_liveness);
}

void testRowContractHeapF1StoreAll(void) {
    assertRowContract(initHeap, buildF1Model, NULL);
}

void testRowContractHeapF1Liveness(void) {
    assertRowContract(initHeap, buildF1Model, &g_liveness);
}

/* A recompute plan rebinds ACT 1 and ACT 2 for REFORWARD(0) and REFORWARD(1)
 * after their first ranges closed. */
void testRowContractArenaOnARecomputePlan(void) {
    assertRowContract(initArena, buildMlp3Model, &g_mlp3Recompute);
}

void testRowContractHeapOnARecomputePlan(void) {
    assertRowContract(initHeap, buildMlp3Model, &g_mlp3Recompute);
}

static void assertRowContractOverRandomRetainMasks(rowInit_t init, bool inPlaceDx) {
    uint32_t state = 0x9E3779B9u;
    for (int trial = 0; trial < 8; trial++) {
        bool retain[HAR_N + 1];
        for (size_t j = 0; j <= HAR_N; j++) {
            retain[j] = nextRandom(&state) % 2u == 0u;
        }
        const rematPlanSpec_t spec = {
            .policy = REMAT_PLAN_RETAIN_LIST, .retain = retain, .inPlaceDx = inPlaceDx};
        assertRowContract(init, buildHarModel, &spec);
    }
}

void testRowContractArenaOnRandomRetainMasks(void) {
    assertRowContractOverRandomRetainMasks(initArena, false);
}

void testRowContractHeapOnRandomRetainMasks(void) {
    assertRowContractOverRandomRetainMasks(initHeap, false);
}

/* ---- in-place dx: the SDK binds a chain through its root ---- */

/* HAR under LIVENESS + inPlaceDx: the members 15 (GRAD 9, Flatten's dx), 17,
 * 20 and 23 bind and release with their roots 14, 16, 19 and 22. Every check
 * of the walk reads the chains from the plan's marks, across training and
 * evaluation calls; on HEAP the row holds exactly the SDK's liveBytes at every
 * step and peaks at the plan's peak, so a chain counts its bytes once. */
void testRowContractArenaHarLivenessInPlaceAcrossModes(void) {
    assertRowContractAcrossModes(initArena, buildHarModel, &g_livenessInPlace);
}

void testRowContractHeapHarLivenessInPlaceAcrossModes(void) {
    assertRowContractAcrossModes(initHeap, buildHarModel, &g_livenessInPlace);
}

/* GRAD ranges do not depend on the policy: STORE_ALL, RETAIN_LIST (ACT 3, 7
 * and 12 kept, and random masks) and MLP3's SEQUENCE recompute plan bind
 * their chains through the roots too. */
static void assertInPlaceRowContractUnderEveryPolicy(rowInit_t init) {
    static const bool keep3And7And12[HAR_N + 1] = {[3] = true, [7] = true, [12] = true};
    const rematPlanSpec_t storeAll = {.policy = REMAT_PLAN_STORE_ALL, .inPlaceDx = true};
    const rematPlanSpec_t retain = {
        .policy = REMAT_PLAN_RETAIN_LIST, .retain = keep3And7And12, .inPlaceDx = true};
    rematPlanSpec_t recompute = g_mlp3Recompute;
    recompute.inPlaceDx = true;
    assertRowContract(init, buildHarModel, &storeAll);
    assertRowContract(init, buildHarModel, &retain);
    assertRowContract(init, buildMlp3Model, &recompute);
    assertRowContractOverRandomRetainMasks(init, true);
}

void testRowContractArenaInPlaceUnderEveryPolicy(void) {
    assertInPlaceRowContractUnderEveryPolicy(initArena);
}

void testRowContractHeapInPlaceUnderEveryPolicy(void) {
    assertInPlaceRowContractUnderEveryPolicy(initHeap);
}

/* MLP3 with its ReLU's wires stored SYM_INT32 12 bits (ACT 2, GRAD 2 by the
 * last Linear's dx, GRAD 1): the in-place pair wire 6 over wire 5 carries a
 * qConfig on each header. The template outlives every fixture. */
static symInt32QConfig_t g_sym12Qc;
static quantization_t g_sym12Q = {.type = SYM_INT32, .qConfig = &g_sym12Qc};
static void buildSymChainModel(fixture_t *f) {
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &g_sym12Qc, 12);
    f->model[0] = makeLinear(2, 4, false, 0);
    f->model[1] = makeRelu(&g_sym12Q);
    f->model[2] = makeLinear(4, 2, false, 2);
    f->model[2]->config->linear->propLossQ = &g_sym12Q;
    f->n = MLP3_N;
    f->lt = MSE;
    f->x = makeResidentInput(&f->in, (size_t[]){1, 2}, 2);
}

/* Only ->data is shared: the walk's chain check finds two headers, two
 * quantizations, two qConfigs and two shapes for the pair, on both rows. */
void testAnInPlaceSymPairKeepsTwoHeadersOnBothRows(void) {
    const rowInit_t inits[] = {initArena, initHeap};
    for (size_t k = 0; k < 2u; k++) {
        fixture_t f;
        buildSymChainModel(&f);
        rematScheduler_t s = inits[k](&f, &g_livenessInPlace);
        TEST_ASSERT_EQUAL_UINT8(1, s.plan->inPlace[6]);
        TEST_ASSERT_NOT_NULL(rematWireHdr(s.wires, 5)->quantization->qConfig);
        TEST_ASSERT_NOT_NULL(rematWireHdr(s.wires, 6)->quantization->qConfig);
        walkWithTheContractChecks(&f, &s, REMAT_MODE_TRAIN);
        freeFixture(&f, &s);
    }
}

/* A member's header is derived when its root binds, through the same path as
 * every inherited wire: GRAD 9, Flatten's dx and a member of GRAD 10's chain,
 * takes ACT 9's dtype and shape at rootStep, although ACT 9's bytes died long
 * before (LIVENESS: BACKWARD(10), step 14; RETAIN_LIST keeping ACT 3, 7 and
 * 12: step 17). */
static void assertFlattenMemberDerivedAtItsRootsBind(rowInit_t init, const rematPlanSpec_t *spec,
                                                     size_t rootStep) {
    fixture_t f;
    buildHarModel(&f);
    rematScheduler_t s = init(&f, spec);
    TEST_ASSERT_EQUAL_UINT8(1, s.plan->inPlace[15]);
    bindAndBegin(&f, &s);
    rematStep_t st;
    size_t step = 0;
    while (rematNext(&s, &st)) {
        const tensor_t *member = rematGradHdr(s.wires, 9);
        if (step < rootStep) {
            TEST_ASSERT_NULL(member->data);
        } else if (step == rootStep) {
            const tensor_t *src = rematActHdr(s.wires, 9);
            TEST_ASSERT_NULL(src->data);
            TEST_ASSERT_EQUAL_PTR(rematGradHdr(s.wires, 10)->data, member->data);
            TEST_ASSERT_EQUAL_INT(FLOAT32, member->quantization->type);
            TEST_ASSERT_EQUAL_size_t(3, member->shape->numberOfDimensions);
            TEST_ASSERT_EQUAL_MEMORY(src->shape->dimensions, member->shape->dimensions,
                                     3 * sizeof(size_t));
        }
        rematDone(&s, &st);
        step++;
    }
    endAndUnbind(&s);
    freeFixture(&f, &s);
}

void testAnInPlaceFlattenMemberTakesItsHeaderAtItsRootsBind(void) {
    static const bool keep3And7And12[HAR_N + 1] = {[3] = true, [7] = true, [12] = true};
    const rematPlanSpec_t retain = {
        .policy = REMAT_PLAN_RETAIN_LIST, .retain = keep3And7And12, .inPlaceDx = true};
    assertFlattenMemberDerivedAtItsRootsBind(initArena, &g_livenessInPlace, 14);
    assertFlattenMemberDerivedAtItsRootsBind(initHeap, &g_livenessInPlace, 14);
    assertFlattenMemberDerivedAtItsRootsBind(initArena, &retain, 17);
    assertFlattenMemberDerivedAtItsRootsBind(initHeap, &retain, 17);
}

/* PR1a carry: under LIVENESS, Flatten's dx (GRAD 9) binds at BACKWARD(9)
 * after its source ACT 9 died. Pinned here so a planner change that keeps
 * ACT 9 alive cannot silently drop the harness's coverage of it. */
void testArenaHarLivenessBindsFlattenDxAfterItsSourceDied(void) {
    fixture_t f;
    buildHarModel(&f);
    TEST_ASSERT_EQUAL_INT(FLATTEN, f.model[9]->type);
    rematScheduler_t s = initArena(&f, &g_liveness);
    bindAndBegin(&f, &s);
    rematStep_t st;
    bool seen = false;
    while (rematNext(&s, &st)) {
        if (st.kind == REMAT_STEP_BACKWARD && st.layer == 9u) {
            TEST_ASSERT_NULL(rematActHdr(s.wires, 9)->data);
            TEST_ASSERT_NOT_NULL(rematGradHdr(s.wires, 9)->data);
            seen = true;
        }
        rematDone(&s, &st);
    }
    endAndUnbind(&s);
    TEST_ASSERT_TRUE(seen);
    freeFixture(&f, &s);
}

/* P8's table-level twin: binding at a range's first step and releasing after
 * its last makes the SDK's observed peak the plan's. Before any call the
 * observed peak is 0, so a report that copied the plan's peak would show.
 * Read before end, where a row that never releases would die. */
static void assertObservedPeakIsThePlannedPeak(rowInit_t init, void (*build)(fixture_t *),
                                               const rematPlanSpec_t *spec) {
    fixture_t f;
    build(&f);
    rematScheduler_t s = init(&f, spec);
    rematReport_t before;
    rematSchedulerReport(&s, &before);
    TEST_ASSERT_TRUE(before.peakLiveBytes > 0u);
    TEST_ASSERT_EQUAL_size_t(0, before.observedPeakLiveBytes);
    bindAndBegin(&f, &s);
    rematStep_t st;
    while (rematNext(&s, &st)) {
        rematDone(&s, &st);
    }
    rematReport_t r;
    rematSchedulerReport(&s, &r);
    TEST_ASSERT_EQUAL_size_t(s.wires->observedPeakLiveBytes, r.observedPeakLiveBytes);
    TEST_ASSERT_EQUAL_size_t(r.peakLiveBytes, r.observedPeakLiveBytes);
    endAndUnbind(&s);
    freeFixture(&f, &s);
}

void testReportObservedPeakEqualsThePlannedPeak(void) {
    const rowInit_t inits[] = {initArena, initHeap};
    for (size_t k = 0; k < 2u; k++) {
        assertObservedPeakIsThePlannedPeak(inits[k], buildHarModel, NULL);
        assertObservedPeakIsThePlannedPeak(inits[k], buildHarModel, &g_liveness);
        assertObservedPeakIsThePlannedPeak(inits[k], buildF1Model, NULL);
        assertObservedPeakIsThePlannedPeak(inits[k], buildF1Model, &g_liveness);
    }
}

/* P8 on an in-place plan: HAR under LIVENESS + inPlaceDx plans 41216 B
 * (49152 without), and a call observes exactly that on both rows, so each
 * chain counts its bytes once. The portfolio places it in exactly 41216 B
 * (first-fit-decreasing alone: 45056 B, a 3840 B gap); STORE_ALL + inPlaceDx
 * places its 70192 B peak. */
void testHarLivenessInPlaceObservesItsPlannedPeakOnBothRows(void) {
    assertObservedPeakIsThePlannedPeak(initArena, buildHarModel, &g_livenessInPlace);
    assertObservedPeakIsThePlannedPeak(initHeap, buildHarModel, &g_livenessInPlace);
    rematReport_t r = reportAfterInit(buildHarModel, &g_livenessInPlace);
    TEST_ASSERT_EQUAL_size_t(41216, r.peakLiveBytes);
    TEST_ASSERT_EQUAL_size_t(41216, r.arenaBytes);
    const rematPlanSpec_t storeAll = {.policy = REMAT_PLAN_STORE_ALL, .inPlaceDx = true};
    r = reportAfterInit(buildHarModel, &storeAll);
    TEST_ASSERT_EQUAL_size_t(70192, r.peakLiveBytes);
    TEST_ASSERT_EQUAL_size_t(70192, r.arenaBytes);
}

/* HAR under LIVENESS + inPlaceDx on HEAP, inside a training call: steps 0 to
 * k - 1 walked. BACKWARD(10), step 14, binds GRAD 10 (wire 14) and with it its
 * member GRAD 9 (wire 15). The bytes are a local: every child exits before it
 * returns. */
static void walkHarInPlaceTo(fixture_t *f, rematScheduler_t *s, size_t k) {
    bindAndBegin(f, s);
    rematStep_t st;
    for (size_t i = 0; i < k; i++) {
        TEST_ASSERT_TRUE(rematNext(s, &st));
        rematDone(s, &st);
    }
}

static void bindAMember(fixture_t *f, rematScheduler_t *s) {
    float bytes[64];
    walkHarInPlaceTo(f, s, 14);
    rematWireBind(s->wires, 15, (uint8_t *)bytes);
}

static void releaseAMember(fixture_t *f, rematScheduler_t *s) {
    rematStep_t st;
    walkHarInPlaceTo(f, s, 14);
    TEST_ASSERT_TRUE(rematNext(s, &st));
    rematWireRelease(s->wires, 15);
}

static void bindARootWhoseMemberIsBound(fixture_t *f, rematScheduler_t *s) {
    float bytes[64];
    rematStep_t st;
    walkHarInPlaceTo(f, s, 14);
    rematGradHdr(s->wires, 9)->data = (uint8_t *)bytes;
    (void)rematNext(s, &st);
}

static void releaseARootWhoseMemberMoved(fixture_t *f, rematScheduler_t *s) {
    float bytes[64];
    rematStep_t st;
    walkHarInPlaceTo(f, s, 14);
    TEST_ASSERT_TRUE(rematNext(s, &st));
    rematGradHdr(s->wires, 9)->data = (uint8_t *)bytes;
    rematWireRelease(s->wires, 14);
}

static void assertHarInPlaceExits(void (*misuse)(fixture_t *, rematScheduler_t *),
                                  const char *message) {
    fixture_t f;
    buildHarModel(&f);
    rematScheduler_t s = initHeap(&f, &g_livenessInPlace);
    ASSERT_EXITS_WITH_OUTPUT(1, message, misuse(&f, &s));
    freeFixture(&f, &s);
}

/* Rows bind only ranges, and a member has none: a Bind or Release that names
 * a member is a row bug. */
void testBindAndReleaseExitOnAnInPlaceMember(void) {
    assertHarInPlaceExits(bindAMember, "rematWireBind: wire GRAD 9 is an in-place member of GRAD "
                                       "10; bind the chain's root");
    assertHarInPlaceExits(releaseAMember, "rematWireRelease: wire GRAD 9 is an in-place member of "
                                          "GRAD 10; release the chain's root");
}

/* A member that left its root's bytes (bound on its own, or moved while
 * bound) stops the root's Bind or Release by name. */
void testARootsBindAndReleaseExitWhenAMemberLeftItsBytes(void) {
    assertHarInPlaceExits(bindARootWhoseMemberIsBound,
                          "rematWireBind: wire GRAD 9, an in-place member of GRAD 10, is already "
                          "bound");
    assertHarInPlaceExits(releaseARootWhoseMemberMoved,
                          "rematWireRelease: wire GRAD 9, an in-place member of GRAD 10, does not "
                          "hold its root's bytes");
}

static void beginAfterTheMembersDtypeChanged(fixture_t *f, rematScheduler_t *s) {
    f->model[1]->config->relu->propLossQ = &g_sym12Q;
    bindAndBegin(f, s);
}

/* The plan decided its pairs from the keyed wire facts: a call on a model
 * whose paired dx changed dtype since init exits at the key check, before any
 * chain is bound. */
void testAnInPlaceCallExitsAtTheKeyCheckWhenAMemberChangedDtype(void) {
    fixture_t f;
    buildMlp3Model(&f);
    rematScheduler_t s = initHeap(&f, &g_livenessInPlace);
    TEST_ASSERT_EQUAL_UINT8(1, s.plan->inPlace[6]);
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &g_sym12Qc, 12);
    ASSERT_EXITS_WITH_OUTPUT(
        1, "rematWireTableBind: key mismatch on wire GRAD 1, field 'dtype': built 1, live 2",
        beginAfterTheMembersDtypeChanged(&f, &s));
    freeFixture(&f, &s);
}

/* ---- ASan poisoning of the arena (asan preset only) ---- */

#ifdef ODT_TEST_ASAN
static void readTheArenaBeforeAnyRangeOpens(rematScheduler_t *s) {
    odtInstallAsanDeathExit();
    volatile uint8_t *arena = s->row.arena.base;
    (void)arena[0];
}

/* The whole arena is unaddressable until a range opens. */
void testArenaIsPoisonedUntilARangeOpens(void) {
    fixture_t f;
    buildF1Model(&f);
    rematScheduler_t s = initArena(&f, NULL);
    ASSERT_EXITS_WITH(ODT_ASAN_DEATH_EXIT, readTheArenaBeforeAnyRangeOpens(&s));
    freeFixture(&f, &s);
}

/* next() unpoisons exactly bytes(w). F1's ACT 1 is 5 BFP bytes at an 8-aligned
 * offset, so bytes 5..7 share a granule with the payload and must stay
 * poisoned: the granule-8 hypothesis, pinned. The marker is
 * flushed before the pad read because the death callback's _exit discards
 * buffered stdout. */
static void readThePadOfABoundWire(fixture_t *f, rematScheduler_t *s) {
    odtInstallAsanDeathExit();
    bindAndBegin(f, s);
    rematStep_t st;
    (void)rematNext(s, &st); /* FORWARD 0 binds ACT 1 */
    volatile uint8_t *act1 = rematWireHdr(s->wires, 1)->data;
    (void)act1[4];
    printf("payload-readable\n");
    (void)fflush(stdout);
    (void)act1[5];
}

void testArenaPadStaysPoisonedWhileBound(void) {
    fixture_t f;
    buildF1Model(&f);
    rematScheduler_t s = initArena(&f, NULL);
    ASSERT_EXITS_WITH_OUTPUT(ODT_ASAN_DEATH_EXIT, "payload-readable",
                             readThePadOfABoundWire(&f, &s));
    freeFixture(&f, &s);
}

/* The same in EVAL, where F1's ACT 1 (BFP, 5 B, odd) sits top-aligned: its
 * 3-byte pad is the arena's last bytes. */
static void readThePadOfATopPlacedEvalWire(fixture_t *f, rematScheduler_t *s) {
    odtInstallAsanDeathExit();
    rematBeginEval(s, f->model, f->n, f->lt, f->x);
    rematStep_t st;
    (void)rematNext(s, &st); /* FORWARD 0 binds ACT 1 */
    volatile uint8_t *act1 = rematWireHdr(s->wires, 1)->data;
    (void)act1[4];
    printf("payload-readable\n");
    (void)fflush(stdout);
    (void)act1[5];
}

void testArenaEvalPadStaysPoisonedWhileBound(void) {
    fixture_t f;
    buildF1Model(&f);
    rematScheduler_t s = initArena(&f, NULL);
    ASSERT_EXITS_WITH_OUTPUT(ODT_ASAN_DEATH_EXIT, "payload-readable",
                             readThePadOfATopPlacedEvalWire(&f, &s));
    freeFixture(&f, &s);
}

/* F1 LIVENESS: ACT 2 lives [1, 3]; a pointer saved while it was
 * bound must trip ASan once done() of step 3 released it. */
static void readAWireAfterItsRelease(fixture_t *f, rematScheduler_t *s) {
    odtInstallAsanDeathExit();
    bindAndBegin(f, s);
    rematStep_t st;
    volatile uint8_t *act2 = NULL;
    while (rematNext(s, &st)) {
        if (st.kind == REMAT_STEP_FORWARD && st.layer == 1u) {
            act2 = rematWireHdr(s->wires, 2)->data;
        }
        rematDone(s, &st);
        if (st.kind == REMAT_STEP_LOSS_BACKWARD) {
            break;
        }
    }
    if (act2 == NULL) {
        _exit(0); /* never captured: a NULL read's SEGV would also exit 86 */
    }
    (void)act2[0];
}

void testArenaReadAfterReleaseTripsAsan(void) {
    fixture_t f;
    buildF1Model(&f);
    rematScheduler_t s = initArena(&f, &g_liveness);
    ASSERT_EXITS_WITH(ODT_ASAN_DEATH_EXIT, readAWireAfterItsRelease(&f, &s));
    freeFixture(&f, &s);
}
#endif

/* ---- the HEAP row: init and report ---- */

void testHeapInitBuildsTheTableAndThePlan(void) {
    fixture_t f;
    buildHarModel(&f);
    rematScheduler_t s = initHeap(&f, &g_liveness);
    TEST_ASSERT_EQUAL_INT(REMAT_HEAP, s.type);
    TEST_ASSERT_EQUAL_size_t(HAR_N, s.wires->modelSize);
    TEST_ASSERT_EQUAL_INT(REMAT_PLAN_LIVENESS, s.plan->policy);
    freeFixture(&f, &s);
}

#ifdef ODT_MEM_PROFILE
/* HEAP holds no data between calls: init reserves the shared table and plan
 * blocks only. */
void testHeapInitReservesOnlyTheTableAndThePlan(void) {
    fixture_t f;
    buildHarModel(&f);
    size_t before = memProfileCurrentBytes();
    rematScheduler_t s = initHeap(&f, NULL);
    TEST_ASSERT_EQUAL_size_t(s.wires->slabBytes + s.plan->blockBytes,
                             memProfileCurrentBytes() - before);
    freeFixture(&f, &s);
}
#endif

/* On HEAP placed == planned and the arena fields are 0. HEAP keeps
 * no resident data block, so a successful init has nothing left to reserve:
 * dataReserved == planned. REMAT_HEAP != 0 makes the type
 * echo observable for the first time. */
static void assertHeapReport(const rematPlanSpec_t *spec, rematPlanPolicy_t policy,
                             size_t peakLiveBytes) {
    fixture_t f;
    buildHarModel(&f);
    rematScheduler_t s = initHeap(&f, spec);
    rematReport_t r;
    rematSchedulerReport(&s, &r);
    TEST_ASSERT_EQUAL_INT(REMAT_HEAP, r.type);
    TEST_ASSERT_EQUAL_INT(policy, r.policy);
    TEST_ASSERT_TRUE(r.planned);
    TEST_ASSERT_TRUE(r.placed);
    TEST_ASSERT_TRUE(r.dataReserved);
    TEST_ASSERT_EQUAL_size_t(25, r.numSteps);
    TEST_ASSERT_EQUAL_size_t(peakLiveBytes, r.peakLiveBytes);
    TEST_ASSERT_EQUAL_size_t(0, r.arenaBytes);
    TEST_ASSERT_EQUAL_size_t(0, r.arenaPadBytes);
    TEST_ASSERT_EQUAL_size_t(0, r.arenaGapBytes);
    /* Every forward once, nothing recomputed: HAR's forward cost, derived in
     * UnitTestRematPlan above its RETAIN_LIST tests. */
    TEST_ASSERT_EQUAL_size_t(0, r.refwdSteps);
    TEST_ASSERT_EQUAL_size_t(502150, r.forwardCost);
    TEST_ASSERT_EQUAL_size_t(0, r.recomputeCost);
    /* slab + plan block, no row table: 4,848 + 568 = 5,416 on LP64 */
    TEST_ASSERT_EQUAL_size_t(s.wires->slabBytes + s.plan->blockBytes, r.metadataBytes);
    freeFixture(&f, &s);
}

void testHeapReportIsPlacedAndReservedWithoutAnArena(void) {
    assertHeapReport(NULL, REMAT_PLAN_STORE_ALL, 74288);
    assertHeapReport(&g_liveness, REMAT_PLAN_LIVENESS, 49152);
}

/* A SEQUENCE plan's steps live in the one plan block, so HEAP init's
 * reserved == reported check (memory-counter builds) and metadataBytes count
 * them. MLP3 STORE_ALL holds 8 steps and 6 ranges, the recompute-twice
 * SEQUENCE 11 steps and 9 ranges. A TRAIN program costs 4 B per step and 8 B
 * per range (a 6 B range and its 2 B endOrder entry; every array is 2-aligned,
 * so no padding), and the EVAL program is the same in both: the block grows
 * by 4 * 3 + 8 * 3 = 36 B. */
void testHeapPlanBlockHoldsTheSequenceSteps(void) {
    const rematStep_t steps[] = MLP3_RECOMPUTE_TWICE_STEPS;
    layer_t *model[MLP3_N];
    buildMlp3(model);
    inputLike_t in;
    tensor_t *x = makeInput(&in, (size_t[]){1, 2}, 2, &g_floatQ);
    rematScheduler_t all;
    TEST_ASSERT_TRUE(rematHeapInit(&all, model, MLP3_N, defaultLossConfig(MSE), x, NULL));
    const rematPlanSpec_t spec = {
        .policy = REMAT_PLAN_SEQUENCE, .sequence = steps, .sequenceLen = COUNT_OF(steps)};
    rematScheduler_t seq;
    TEST_ASSERT_TRUE(rematHeapInit(&seq, model, MLP3_N, defaultLossConfig(MSE), x, &spec));
    TEST_ASSERT_EQUAL_size_t(all.plan->blockBytes + 36u, seq.plan->blockBytes);
    rematReport_t r;
    rematSchedulerReport(&seq, &r);
    TEST_ASSERT_EQUAL_size_t(11, r.numSteps);
    TEST_ASSERT_EQUAL_size_t(seq.wires->slabBytes + seq.plan->blockBytes, r.metadataBytes);
    rematSchedulerDeinit(&seq);
    rematSchedulerDeinit(&all);
    freeModel(model, MLP3_N);
}

static void heapInitExpectingAnExit(layer_t **model, const tensor_t *x) {
    rematScheduler_t s;
    (void)rematHeapInit(&s, model, 1, defaultLossConfig(MSE), x, NULL);
}

/* A borrowed [1, SIZE_MAX/4 + 2] FLOAT32 input (4 * N wraps to 4)
 * exits at rematHeapInit too, naming the wire. */
void testHeapInitExitsOnAByteCountOverflow(void) {
    layer_t *model[1] = {makeRelu(&g_floatQ)};
    inputLike_t in;
    tensor_t *x = makeInput(&in, (size_t[]){1, SIZE_MAX / 4u + 2u}, 2, &g_floatQ);
    ASSERT_EXITS_WITH_OUTPUT(1, "size overflow computing bytes of wire ACT 0",
                             heapInitExpectingAnExit(model, x));
    freeModel(model, 1);
}

/* ---- the HEAP row's entry points ---- */

void testHeapWalkHandsOutThePlanStepsAndEndsWithNothingBound(void) {
    fixture_t f;
    buildHarModel(&f);
    rematScheduler_t s = initHeap(&f, NULL);
    const rematProgram_t *p = &s.plan->train;
    bindAndBegin(&f, &s);
    rematStep_t st;
    size_t i = 0;
    while (rematNext(&s, &st)) {
        TEST_ASSERT_TRUE(i < p->numSteps);
        TEST_ASSERT_EQUAL_UINT8(p->steps[i].kind, st.kind);
        TEST_ASSERT_EQUAL_UINT16(p->steps[i].layer, st.layer);
        rematDone(&s, &st);
        i++;
    }
    TEST_ASSERT_EQUAL_size_t(25, i);
    endAndUnbind(&s);
    for (uint16_t w = 1; w < s.wires->numWires; w++) {
        TEST_ASSERT_NULL(rematWireHdr(s.wires, w)->data);
    }
    TEST_ASSERT_NULL(rematActHdr(s.wires, 0));
    freeFixture(&f, &s);
}

#ifdef ODT_MEM_PROFILE
/* The counter adds each block's requested bytes, not the allocator header
 * (StorageApi.c:41), so a row that reserves exactly bytes(w) per live range
 * holds exactly the SDK's liveBytes at every point of the call, and its
 * per-call peak of reserved bytes is the plan's peakLiveBytes. */
static void assertHeapHoldsTheLiveBytes(const rematPlanSpec_t *spec, size_t peakBytes) {
    fixture_t f;
    buildHarModel(&f);
    rematScheduler_t s = initHeap(&f, spec);
    size_t memAfterInit = memProfileCurrentBytes();
    bindAndBegin(&f, &s);
    rematStep_t st;
    size_t peak = 0;
    while (rematNext(&s, &st)) {
        size_t held = memProfileCurrentBytes() - memAfterInit;
        TEST_ASSERT_EQUAL_size_t(s.wires->liveBytes, held);
        if (held > peak) {
            peak = held;
        }
        rematDone(&s, &st);
        TEST_ASSERT_EQUAL_size_t(s.wires->liveBytes, memProfileCurrentBytes() - memAfterInit);
    }
    endAndUnbind(&s);
    TEST_ASSERT_EQUAL_size_t(peakBytes, peak);
    TEST_ASSERT_EQUAL_size_t(memAfterInit, memProfileCurrentBytes());
    freeFixture(&f, &s);
}

void testHeapHoldsExactlyTheLiveBytesAtEveryStep(void) {
    assertHeapHoldsTheLiveBytes(&g_liveness, 49152);
}

/* One block per range, the chain root's: 41216 B at the peak, not 49152. */
void testHeapHoldsExactlyTheLiveBytesAtEveryStepInPlace(void) {
    assertHeapHoldsTheLiveBytes(&g_livenessInPlace, 41216);
}
#endif

/* PR1a carry: the walk restarts at every begin. */
void testHeapSecondCallRestartsTheWalk(void) {
    fixture_t f;
    buildF1Model(&f);
    rematScheduler_t s = initHeap(&f, &g_liveness);
    TEST_ASSERT_EQUAL_size_t(5, walkAll(&f, &s));
    TEST_ASSERT_EQUAL_size_t(5, walkAll(&f, &s));
    TEST_ASSERT_EQUAL_UINT32(1, s.wires->wires[1].bindGen); /* the table bind reset it */
    freeFixture(&f, &s);
}

static void heapEndAfterOneStep(fixture_t *f, rematScheduler_t *s) {
    bindAndBegin(f, s);
    rematStep_t st;
    (void)rematNext(s, &st);
    rematDone(s, &st);
    rematEnd(s);
}

void testHeapEndExitsOnAnIncompleteWalk(void) {
    fixture_t f;
    buildF1Model(&f);
    rematScheduler_t s = initHeap(&f, &g_liveness);
    ASSERT_EXITS_WITH_OUTPUT(1,
                             "remat[heap]: rematEnd before the walk completed: 1 of 5 steps done, "
                             "0 of 3 ranges closed",
                             heapEndAfterOneStep(&f, &s));
    freeFixture(&f, &s);
}

static void heapEvalEndAfterOneStep(fixture_t *f, rematScheduler_t *s) {
    bindAndBeginIn(f, s, REMAT_MODE_EVAL);
    rematStep_t st;
    (void)rematNext(s, &st);
    rematDone(s, &st);
    rematEnd(s);
}

/* The end of an eval call checks the EVAL walk: F1's EVAL program has 3 steps
 * and 2 ranges. */
void testHeapEndInEvalChecksTheEvalWalk(void) {
    fixture_t f;
    buildF1Model(&f);
    rematScheduler_t s = initHeap(&f, NULL);
    ASSERT_EXITS_WITH_OUTPUT(1,
                             "remat[heap]: rematEnd before the walk completed: 1 of 3 steps done, "
                             "0 of 2 ranges closed",
                             heapEvalEndAfterOneStep(&f, &s));
    freeFixture(&f, &s);
}

static void beginEvalOnAZeroedScheduler(fixture_t *f) {
    rematScheduler_t s = {0};
    rematBeginEval(&s, f->model, f->n, f->lt, f->x);
}

/* rematBeginEval shares rematBegin's guards, under its own name. */
void testBeginEvalExitsOnASchedulerThatIsNotInitialised(void) {
    fixture_t f;
    buildF1Model(&f);
    ASSERT_EXITS_WITH_OUTPUT(1,
                             "rematBeginEval: scheduler not initialised (never initialised, or "
                             "its init returned false and was ignored)",
                             beginEvalOnAZeroedScheduler(&f));
    freeModel(f.model, f.n);
}

static void beginEvalInsideATrainingCall(fixture_t *f, rematScheduler_t *s) {
    bindAndBegin(f, s);
    rematBeginEval(s, f->model, f->n, f->lt, f->x);
}

void testBeginEvalExitsWhenReEntered(void) {
    fixture_t f;
    buildF1Model(&f);
    rematScheduler_t s = initHeap(&f, NULL);
    ASSERT_EXITS_WITH_OUTPUT(1, "rematBeginEval: scheduler 'heap' re-entered",
                             beginEvalInsideATrainingCall(&f, &s));
    freeFixture(&f, &s);
}

#ifndef ODT_TEST_ASAN
static void heapFirstNextOnAHugeWire(fixture_t *f, rematScheduler_t *s) {
    bindAndBegin(f, s);
    rematStep_t st;
    (void)rematNext(s, &st);
}

/* The row owns resource exits. ReLU over a borrowed [1, 2^60] FLOAT32
 * input under MSE (the did-not-run fixture): HEAP init reserves only the
 * table and plan, and FORWARD 0's 2^62-byte ACT 1 cannot be reserved on any
 * 64-bit host. Skipped under ASan, which aborts on oversized requests; macOS
 * malloc prints a "can't allocate region" warning to stderr here. */
void testHeapNextExitsNamingTheStepAndTheWireWhenAReservationFails(void) {
    fixture_t f;
    f.model[0] = makeRelu(&g_floatQ);
    f.n = 1;
    f.lt = MSE;
    f.x = makeInput(&f.in, (size_t[]){1, (size_t)1 << 60}, 2, &g_floatQ);
    rematScheduler_t s = initHeap(&f, NULL);
    ASSERT_EXITS_WITH_OUTPUT(1,
                             "remat[heap]: reserveMemory(4611686018427387904) failed at step #0 "
                             "for wire ACT 1",
                             heapFirstNextOnAHugeWire(&f, &s));
    freeFixture(&f, &s);
}
#endif

/* ---- the const vtable and the dispatch ---- */

void testEachInitInstallsItsRowsFunctionTable(void) {
    fixture_t f;
    buildF1Model(&f);
    rematScheduler_t a = initArena(&f, NULL);
    rematScheduler_t h = initHeap(&f, NULL);
    TEST_ASSERT_EQUAL_PTR(&rematSchedulerFunctions[REMAT_ARENA], a.fns);
    TEST_ASSERT_EQUAL_PTR(&rematSchedulerFunctions[REMAT_HEAP], h.fns);
    TEST_ASSERT_EQUAL_STRING("arena", a.fns->name);
    TEST_ASSERT_EQUAL_STRING("heap", h.fns->name);
    rematSchedulerDeinit(&a);
    rematSchedulerDeinit(&h);
    freeModel(f.model, f.n);
}

/* Every slot is mandatory for every row. */
void testEveryRowFillsEverySlot(void) {
    for (size_t row = REMAT_ARENA; row <= REMAT_HEAP; row++) {
        const rematSchedulerFunctions_t *fns = &rematSchedulerFunctions[row];
        TEST_ASSERT_NOT_NULL(fns->name);
        TEST_ASSERT_NOT_NULL(fns->begin);
        TEST_ASSERT_NOT_NULL(fns->next);
        TEST_ASSERT_NOT_NULL(fns->done);
        TEST_ASSERT_NOT_NULL(fns->end);
        TEST_ASSERT_NOT_NULL(fns->deinit);
    }
}

static void assertTheDispatchWalks(rowInit_t init) {
    fixture_t f;
    buildHarModel(&f);
    rematScheduler_t s = init(&f, NULL);
    const rematProgram_t *p = &s.plan->train;
    rematBegin(&s, f.model, f.n, defaultLossConfig(f.lt), f.x);
    TEST_ASSERT_TRUE(s.inCall);
    TEST_ASSERT_EQUAL_PTR(f.x, rematActHdr(s.wires, 0));
    rematStep_t st;
    size_t i = 0;
    while (rematNext(&s, &st)) {
        TEST_ASSERT_TRUE(i < p->numSteps);
        TEST_ASSERT_EQUAL_UINT8(p->steps[i].kind, st.kind);
        TEST_ASSERT_EQUAL_UINT16(p->steps[i].layer, st.layer);
        rematDone(&s, &st);
        i++;
    }
    TEST_ASSERT_EQUAL_size_t(25, i);
    rematEnd(&s);
    TEST_ASSERT_FALSE(s.inCall);
    TEST_ASSERT_NULL(rematActHdr(s.wires, 0));
    freeFixture(&f, &s);
}

void testTheDispatchWalksEitherRow(void) {
    assertTheDispatchWalks(initArena);
    assertTheDispatchWalks(initHeap);
}

/* A decorator row on the caller's own instance. The counters are
 * test-local state; the table itself is const. */
static size_t g_decoratedCalls[5];

static void countingBegin(rematScheduler_t *s) {
    g_decoratedCalls[0]++;
    rematSchedulerFunctions[REMAT_ARENA].begin(s);
}
static bool countingNext(rematScheduler_t *s, rematStep_t *st) {
    g_decoratedCalls[1]++;
    return rematSchedulerFunctions[REMAT_ARENA].next(s, st);
}
static void countingDone(rematScheduler_t *s, const rematStep_t *st) {
    g_decoratedCalls[2]++;
    rematSchedulerFunctions[REMAT_ARENA].done(s, st);
}
static void countingEnd(rematScheduler_t *s) {
    g_decoratedCalls[3]++;
    rematSchedulerFunctions[REMAT_ARENA].end(s);
}
static void countingDeinit(rematScheduler_t *s) {
    g_decoratedCalls[4]++;
    rematSchedulerFunctions[REMAT_ARENA].deinit(s);
}

static const rematSchedulerFunctions_t g_countingArena = {
    "counting-arena", countingBegin, countingNext, countingDone, countingEnd, countingDeinit};

void testTheDispatchCallsThroughTheInstancesFunctionTable(void) {
    memset(g_decoratedCalls, 0, sizeof g_decoratedCalls);
    fixture_t f;
    buildF1Model(&f);
    rematScheduler_t s = initArena(&f, NULL);
    s.fns = &g_countingArena;
    rematBegin(&s, f.model, f.n, defaultLossConfig(f.lt), f.x);
    rematStep_t st;
    while (rematNext(&s, &st)) {
        rematDone(&s, &st);
    }
    rematEnd(&s);
    TEST_ASSERT_EQUAL_size_t(1, g_decoratedCalls[0]);
    TEST_ASSERT_EQUAL_size_t(6, g_decoratedCalls[1]); /* 5 steps, then false */
    TEST_ASSERT_EQUAL_size_t(5, g_decoratedCalls[2]);
    TEST_ASSERT_EQUAL_size_t(1, g_decoratedCalls[3]);
    rematSchedulerDeinit(&s);
    TEST_ASSERT_EQUAL_size_t(1, g_decoratedCalls[4]); /* the row's deinit via fns */
    freeModel(f.model, f.n);
}

static void beginOnAZeroedScheduler(fixture_t *f) {
    rematScheduler_t s = {0};
    rematBegin(&s, f->model, f->n, defaultLossConfig(f->lt), f->x);
}

void testBeginExitsOnASchedulerThatWasNeverInitialised(void) {
    fixture_t f;
    buildF1Model(&f);
    ASSERT_EXITS_WITH_OUTPUT(1, "rematBegin: scheduler not initialised",
                             beginOnAZeroedScheduler(&f));
    freeModel(f.model, f.n);
}

/* The state rematHeapInit leaves when its plan block fails (fns and
 * table set, plan NULL); a caller that ignored the false must not enter a call. */
static void beginOnAHeapWhosePlanFailed(fixture_t *f, rematScheduler_t *s) {
    rematBegin(s, f->model, f->n, defaultLossConfig(f->lt), f->x);
}

void testBeginExitsOnAHeapWhoseInitReturnedFalse(void) {
    fixture_t f;
    buildF1Model(&f);
    size_t before = memProfileCurrentBytes();
    rematScheduler_t s = initHeap(&f, NULL);
    rematPlanFree(s.plan);
    s.plan = NULL;
    ASSERT_EXITS_WITH_OUTPUT(1, "rematBegin: scheduler not initialised",
                             beginOnAHeapWhosePlanFailed(&f, &s));
    rematSchedulerDeinit(&s); /* M-b: deinit after a failed HEAP init */
    TEST_ASSERT_NULL(s.fns);
    TEST_ASSERT_EQUAL_size_t(before, memProfileCurrentBytes());
    freeModel(f.model, f.n);
}

static void beginTwice(fixture_t *f, rematScheduler_t *s) {
    rematBegin(s, f->model, f->n, defaultLossConfig(f->lt), f->x);
    rematBegin(s, f->model, f->n, defaultLossConfig(f->lt), f->x);
}

/* The table bind resets liveBytes and bindGen but leaves ->data alone, so
 * a second begin on an unfinished call must not reach it. */
void testBeginExitsWhenTheCallIsReEntered(void) {
    fixture_t f;
    buildF1Model(&f);
    rematScheduler_t s = initHeap(&f, NULL);
    ASSERT_EXITS_WITH_OUTPUT(1, "rematBegin: scheduler 'heap' re-entered", beginTwice(&f, &s));
    freeFixture(&f, &s);
}

static void nextBeforeBegin(rematScheduler_t *s) {
    rematStep_t st;
    (void)rematNext(s, &st);
}

void testNextExitsOutsideACall(void) {
    fixture_t f;
    buildF1Model(&f);
    rematScheduler_t s = initHeap(&f, NULL);
    ASSERT_EXITS_WITH_OUTPUT(1, "remat[heap]: rematNext outside a call", nextBeforeBegin(&s));
    freeFixture(&f, &s);
}

/* m1: requireInCall's "uninitialised" branch (s->fns == NULL), untested until now. */
static void nextOnAZeroedScheduler(void) {
    rematScheduler_t s = {0};
    rematStep_t st;
    (void)rematNext(&s, &st);
}

void testNextExitsOnAnUninitialisedScheduler(void) {
    ASSERT_EXITS_WITH_OUTPUT(1, "remat[uninitialised]: rematNext outside a call",
                             nextOnAZeroedScheduler());
}

static void doneBeforeBegin(rematScheduler_t *s) {
    rematDone(s, &s->plan->train.steps[0]);
}

void testDoneExitsOutsideACall(void) {
    fixture_t f;
    buildF1Model(&f);
    rematScheduler_t s = initArena(&f, NULL);
    ASSERT_EXITS_WITH_OUTPUT(1, "remat[arena]: rematDone outside a call", doneBeforeBegin(&s));
    freeFixture(&f, &s);
}

static void endTwice(fixture_t *f, rematScheduler_t *s) {
    rematBegin(s, f->model, f->n, defaultLossConfig(f->lt), f->x);
    rematStep_t st;
    while (rematNext(s, &st)) {
        rematDone(s, &st);
    }
    rematEnd(s);
    rematEnd(s);
}

/* A second end would unbind and clear inCall twice; it must be named. */
void testEndExitsOutsideACall(void) {
    fixture_t f;
    buildF1Model(&f);
    rematScheduler_t s = initArena(&f, NULL);
    ASSERT_EXITS_WITH_OUTPUT(1, "remat[arena]: rematEnd outside a call", endTwice(&f, &s));
    freeFixture(&f, &s);
}

static void deinitInsideACall(fixture_t *f, rematScheduler_t *s) {
    rematBegin(s, f->model, f->n, defaultLossConfig(f->lt), f->x);
    rematStep_t st;
    (void)rematNext(s, &st);
    rematSchedulerDeinit(s);
}

/* HEAP holds a block for every open range mid-call; a deinit there would free
 * the table under bound wires and leak those blocks. */
void testDeinitExitsInsideACall(void) {
    fixture_t f;
    buildF1Model(&f);
    rematScheduler_t s = initHeap(&f, NULL);
    ASSERT_EXITS_WITH_OUTPUT(1, "remat[heap]: rematSchedulerDeinit inside a call",
                             deinitInsideACall(&f, &s));
    freeFixture(&f, &s);
}

/* ---- the call protocol, owned by the dispatch on every row ---- */

/* Each misuse on F1 (STORE_ALL) under both rows; the message names the row. */
static void assertMisuseExitsOnBothRows(void (*misuse)(fixture_t *, rematScheduler_t *),
                                        const char *rule) {
    const rowInit_t inits[] = {initArena, initHeap};
    for (size_t k = 0; k < 2u; k++) {
        fixture_t f;
        buildF1Model(&f);
        rematScheduler_t s = inits[k](&f, NULL);
        char message[160];
        (void)snprintf(message, sizeof message, "remat[%s]: %s", s.fns->name, rule);
        ASSERT_EXITS_WITH_OUTPUT(1, message, misuse(&f, &s));
        freeFixture(&f, &s);
    }
}

void testDoneExitsOnAStepNextDidNotHandOut(void) {
    assertMisuseExitsOnBothRows(doneForAStepNextDidNotHandOut,
                                "rematDone for a step next() did not hand out: next() handed out "
                                "(kind 0, layer 0), done() got (kind 3, layer 0)");
}

/* Step 0 opens ACT 1; a done() without its next() would leave it
 * unbound and stall the open cursor for the rest of the call. */
void testDoneExitsBeforeNextHandedTheStepOut(void) {
    assertMisuseExitsOnBothRows(doneBeforeNext, "rematDone for (kind 0, layer 0) with no step "
                                                "handed out");
}

/* F1 step 2 is LOSS_FORWARD, which opens no range (nor does a grads-only
 * BACKWARD at deepest): no open cursor can see the skipped next(), only the
 * handed-out flag. */
void testDoneExitsWhenNextWasSkippedAtAStepThatOpensNothing(void) {
    assertMisuseExitsOnBothRows(doneWithoutNextAtLossForward,
                                "rematDone for (kind 1, layer 2) with no step handed out");
}

void testNextExitsWhileTheHandedOutStepIsNotDone(void) {
    assertMisuseExitsOnBothRows(nextTwice, "rematNext while (kind 0, layer 0) is still handed "
                                           "out");
}

/* One done() too many must not reach the row, whose walk would index
 * steps[numSteps]. */
void testDoneExitsAfterTheStreamCompleted(void) {
    assertMisuseExitsOnBothRows(doneAfterTheStreamCompleted,
                                "rematDone for (kind 3, layer 1) with no step handed out");
}

/* ---- HEAP's right-boundary ASan redzones (asan preset only) ---- */

#ifdef ODT_TEST_ASAN
/* One block of exactly bytes(w) per range: F1's ACT 1 is 5 BFP bytes, so its
 * byte 5 lies in the block's redzone. The marker is flushed before the read
 * because the death callback's _exit discards buffered stdout. */
static void readOnePastAHeapWire(fixture_t *f, rematScheduler_t *s) {
    odtInstallAsanDeathExit();
    bindAndBegin(f, s);
    rematStep_t st;
    (void)rematNext(s, &st); /* FORWARD 0 binds ACT 1 */
    volatile uint8_t *act1 = rematWireHdr(s->wires, 1)->data;
    (void)act1[4];
    printf("payload-readable\n");
    (void)fflush(stdout);
    (void)act1[5];
}

void testHeapWireEndsAtItsExactBytes(void) {
    fixture_t f;
    buildF1Model(&f);
    rematScheduler_t s = initHeap(&f, NULL);
    ASSERT_EXITS_WITH_OUTPUT(ODT_ASAN_DEATH_EXIT, "payload-readable", readOnePastAHeapWire(&f, &s));
    freeFixture(&f, &s);
}

/* F1 LIVENESS: ACT 2 lives [1, 3]; its block is freed by done() of step 3, so
 * a pointer saved while it was bound trips ASan as a use after free. */
static void readAHeapWireAfterItsRelease(fixture_t *f, rematScheduler_t *s) {
    odtInstallAsanDeathExit();
    bindAndBegin(f, s);
    rematStep_t st;
    volatile uint8_t *act2 = NULL;
    while (rematNext(s, &st)) {
        if (st.kind == REMAT_STEP_FORWARD && st.layer == 1u) {
            act2 = rematWireHdr(s->wires, 2)->data;
        }
        rematDone(s, &st);
        if (st.kind == REMAT_STEP_LOSS_BACKWARD) {
            break;
        }
    }
    if (act2 == NULL) {
        _exit(0); /* never captured: a NULL read's SEGV would also exit 86 */
    }
    (void)act2[0];
}

void testHeapReadAfterReleaseTripsAsan(void) {
    fixture_t f;
    buildF1Model(&f);
    rematScheduler_t s = initHeap(&f, &g_liveness);
    ASSERT_EXITS_WITH(ODT_ASAN_DEATH_EXIT, readAHeapWireAfterItsRelease(&f, &s));
    freeFixture(&f, &s);
}

/* HAR LIVENESS + inPlaceDx: GRAD 9 (wire 15) holds GRAD 10's block, which
 * done() of BACKWARD(8), step 16, frees with the chain's one range, so a
 * pointer saved from the member while it was bound trips ASan as a use after
 * free. */
static void readAnInPlaceMemberAfterItsChainEnded(fixture_t *f, rematScheduler_t *s) {
    odtInstallAsanDeathExit();
    bindAndBegin(f, s);
    rematStep_t st;
    volatile uint8_t *member = NULL;
    while (rematNext(s, &st)) {
        if (st.kind == REMAT_STEP_BACKWARD && st.layer == 9u) {
            member = rematGradHdr(s->wires, 9)->data;
        }
        rematDone(s, &st);
        if (st.kind == REMAT_STEP_BACKWARD && st.layer == 8u) {
            break;
        }
    }
    if (member == NULL) {
        _exit(0); /* never captured: a NULL read's SEGV would also exit 86 */
    }
    (void)member[0];
}

void testHeapReadOfAnInPlaceMemberAfterItsChainEndedTripsAsan(void) {
    fixture_t f;
    buildHarModel(&f);
    rematScheduler_t s = initHeap(&f, &g_livenessInPlace);
    ASSERT_EXITS_WITH(ODT_ASAN_DEATH_EXIT, readAnInPlaceMemberAfterItsChainEnded(&f, &s));
    freeFixture(&f, &s);
}
#endif

/* The init self-check: the bytes the memory counter saw a successful init
 * reserve must equal what its report accounts for (arenaBytes +
 * metadataBytes). One unaccounted byte exits naming the row and both
 * numbers. */
static void selfCheckWithOneExtraByte(rowInit_t init) {
    fixture_t f;
    buildHarModel(&f);
    rematScheduler_t s = init(&f, NULL);
    rematReport_t r;
    rematSchedulerReport(&s, &r);
    rematRequireReservedMatchesReport(&s, r.arenaBytes + r.metadataBytes + 1u);
}

void testInitSelfCheckExitsOnAnUnaccountedByteOnHeap(void) {
    ASSERT_EXITS_WITH_OUTPUT(1,
                             "remat[heap]: init reserved 5433 B, but its report accounts for "
                             "5432 B (arenaBytes 0 + metadataBytes 5432)",
                             selfCheckWithOneExtraByte(initHeap));
}

void testInitSelfCheckExitsOnAnUnaccountedByteOnArena(void) {
    ASSERT_EXITS_WITH_OUTPUT(1,
                             "remat[arena]: init reserved 79905 B, but its report accounts for "
                             "79904 B (arenaBytes 74288 + metadataBytes 5616)",
                             selfCheckWithOneExtraByte(initArena));
}

#ifdef ODT_MEM_PROFILE
/* The same check reached through a real init: the wire table asks every layer
 * for its output shape while it is built, so a shape query that also reserves
 * 8 bytes once (kept reachable here) is a reservation no report term covers.
 * The init must then exit with its report's bytes + 8. */
static calcOutputShapeFn_t g_realOutputShape;
static void *g_unaccounted;

static void outputShapeReservingEightBytesOnce(layer_t *layer, shape_t *in, shape_t *out) {
    if (g_unaccounted == NULL) {
        g_unaccounted = reserveMemory(8);
    }
    g_realOutputShape(layer, in, out);
}

static void initWithAnUnaccountedReservation(rematSchedulerType_t row) {
    fixture_t f;
    buildHarModel(&f);
    layerType_t first = f.model[0]->type;
    g_realOutputShape = layerFunctions[first].calcOutputShape;
    layerFunctions[first].calcOutputShape = outputShapeReservingEightBytesOnce;
    rematScheduler_t s;
    bool built = (row == REMAT_ARENA)
                     ? rematArenaInit(&s, f.model, f.n, defaultLossConfig(f.lt), f.x, NULL)
                     : rematHeapInit(&s, f.model, f.n, defaultLossConfig(f.lt), f.x, NULL);
    if (!built) {
        exit(2);
    }
}

void testHeapInitExitsOnAnUnaccountedReservation(void) {
    ASSERT_EXITS_WITH_OUTPUT(1,
                             "remat[heap]: init reserved 5440 B, but its report accounts for "
                             "5432 B (arenaBytes 0 + metadataBytes 5432)",
                             initWithAnUnaccountedReservation(REMAT_HEAP));
}

void testArenaInitExitsOnAnUnaccountedReservation(void) {
    ASSERT_EXITS_WITH_OUTPUT(1,
                             "remat[arena]: init reserved 79912 B, but its report accounts for "
                             "79904 B (arenaBytes 74288 + metadataBytes 5616)",
                             initWithAnUnaccountedReservation(REMAT_ARENA));
}
#endif

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testArenaInitBuildsTheTableAndThePlan);
    RUN_TEST(testReportEchoesTypePolicyAndPlanFacts);
    RUN_TEST(testReportPeakLiveBytesHarStoreAllIs74288);
    RUN_TEST(testReportPeakLiveBytesHarLivenessIs49152);
    RUN_TEST(testReportOnAZeroedSchedulerIsEmpty);
    RUN_TEST(testReportMetadataCountsTheResidentBlocks);
    RUN_TEST(testDeinitIsNullSafeAndIdempotent);
    RUN_TEST(testArenaDeinitIsIdempotentOnItsOwn);
#ifdef ODT_MEM_PROFILE
    RUN_TEST(testDeinitReturnsEveryInitBlock);
#endif
    RUN_TEST(testArenaPlacedRoundsEveryWireUpToTheWireAlignment);
    RUN_TEST(testArenaOffsetsAligned);
    RUN_TEST(testFfdPeakPlacedBytesIsThePeakOfPlacedSums);
    RUN_TEST(testFfdMatchesTheNaiveOracleOnRandomPlans);
    RUN_TEST(testArenaPlacedExitsNamingTheWireOnOverflow);
#ifdef ODT_MEM_PROFILE
    RUN_TEST(testFfdReleasesItsScratch);
#endif
    RUN_TEST(testSizeThenEndPlacesTheHandPlan);
    RUN_TEST(testPingPongPlacesTheHandPlan);
    RUN_TEST(testBreadthAreaPlacesTheHandPlan);
    RUN_TEST(testBreadthAreaComparesAreasWithoutWrapping);
    RUN_TEST(testTwoEndedPlacesTheHandPlan);
    RUN_TEST(testPlanHasGradRangeOnlyWithABackward);
    RUN_TEST(testTwoEndedExitsOnAPlanWithoutGradRanges);
    RUN_TEST(testNewMembersVerifyOnRandomPlans);
#ifdef ODT_MEM_PROFILE
    RUN_TEST(testNewMembersReleaseTheirScratch);
#endif
    RUN_TEST(testPortfolioIsNeverWorseThanFfdOnRandomPlans);
    RUN_TEST(testPortfolioKeepsTheEarlierMemberOnATie);
    RUN_TEST(testEveryPortfolioMemberIsTheOnlyOneAtThePeakOfSomePlan);
#ifdef ODT_MEM_PROFILE
    RUN_TEST(testPortfolioReleasesItsScratch);
#endif
    RUN_TEST(testVerifierAcceptsTheFfdPlacementOnTheZoo);
    RUN_TEST(testVerifierExitsNamingBothWiresOnCoLiveOverlap);
    RUN_TEST(testVerifierTreatsRangesMeetingAtOneStepAsCoLive);
    RUN_TEST(testVerifierExitsOnAMisalignedOffset);
    RUN_TEST(testVerifierExitsOnARangePastTheArenaEnd);
    RUN_TEST(testVerifierBoundsPlacedNotExactBytes);
    RUN_TEST(testVerifierRejectsAnOffsetNearSizeMax);
    RUN_TEST(testArenaInitPlacesVerifiesAndReservesTheArena);
    RUN_TEST(testReportArenaBytesIsPeakPlusPadPlusGap);
    RUN_TEST(testReportPadIsZeroOnHar);
    RUN_TEST(testReportPadOnTheF1ModelIsEleven);
    RUN_TEST(testReportArenaBytesHarRegressionGuard);
    RUN_TEST(testReportArenaBytesHarRetainRegressionGuard);
    RUN_TEST(testReportCarriesTheRecomputeNumbersOfARetainListPlan);
    RUN_TEST(testReportArenaBytesMnistCnnAndFinetuneRegressionGuards);
    RUN_TEST(testReportArenaBytesHarRetainPlansAtTheirPeak);
    RUN_TEST(testReportActivationsPeakOnHarOnBothRows);
    RUN_TEST(testArenaOffsetsAlignedOnRandomMixedChains);
#ifdef ODT_MEM_PROFILE
    RUN_TEST(testArenaInitReservesExactlyMetadataPlusArena);
#endif
#ifndef ODT_TEST_ASAN
    RUN_TEST(testArenaInitFailureKeepsAnalyticReport);
#endif
    RUN_TEST(testArenaInitExitsAboveMaxRangesBeforeThePlanIsBuilt);
    RUN_TEST(testArenaInitAcceptsExactlyMaxRanges);
    RUN_TEST(testHeapInitAcceptsAStoreAllPlanAboveMaxRanges);
    RUN_TEST(testHeapInitExitsOnASequenceAboveMaxRangesBeforeThePlanBlock);
    RUN_TEST(testHeapInitAcceptsASequenceOfExactlyMaxRanges);
    RUN_TEST(testArenaInitExitsOnASequenceAboveMaxRangesInThePlanBuild);
    RUN_TEST(testArenaInitExitsOnAPlacedSizeOverflowBeforeAnyRowReservation);
    RUN_TEST(testArenaInitExitsOnAnArenaSumOverflowBeforeAnyRowReservation);
    RUN_TEST(testArenaNextHandsOutThePlanStepsInOrder);
    RUN_TEST(testArenaNextAfterTheStreamCompletedReturnsFalse);
    RUN_TEST(testArenaSecondCallRestartsTheWalk);
    RUN_TEST(testDoneExitsOnAStepNextDidNotHandOut);
    RUN_TEST(testDoneExitsBeforeNextHandedTheStepOut);
    RUN_TEST(testDoneExitsWhenNextWasSkippedAtAStepThatOpensNothing);
    RUN_TEST(testNextExitsWhileTheHandedOutStepIsNotDone);
    RUN_TEST(testDoneExitsAfterTheStreamCompleted);
    RUN_TEST(testArenaEndExitsOnAnIncompleteWalk);
    RUN_TEST(testArenaEndExitsOnARangeLeftOpen);
    RUN_TEST(testArenaBeginExitsWhenTheArenaWasNeverReserved);
    RUN_TEST(testRowContractArenaHarStoreAll);
    RUN_TEST(testRowContractArenaHarLiveness);
    RUN_TEST(testRowContractArenaF1StoreAll);
    RUN_TEST(testRowContractArenaF1Liveness);
    RUN_TEST(testRowContractHeapHarStoreAll);
    RUN_TEST(testRowContractHeapHarLiveness);
    RUN_TEST(testRowContractHeapF1StoreAll);
    RUN_TEST(testRowContractHeapF1Liveness);
    RUN_TEST(testRowContractArenaOnARecomputePlan);
    RUN_TEST(testRowContractHeapOnARecomputePlan);
    RUN_TEST(testRowContractArenaOnRandomRetainMasks);
    RUN_TEST(testRowContractHeapOnRandomRetainMasks);
    RUN_TEST(testRowContractHeapHarAcrossModes);
    RUN_TEST(testRowContractHeapF1AcrossModesLiveness);
    RUN_TEST(testRowContractArenaHarAcrossModes);
    RUN_TEST(testRowContractArenaF1AcrossModesLiveness);
    RUN_TEST(testArenaEvalPlacesEvenActsAtTheBottomAndOddActsAtTheTop);
    RUN_TEST(testArenaHarLivenessBindsFlattenDxAfterItsSourceDied);
    RUN_TEST(testReportObservedPeakEqualsThePlannedPeak);
    RUN_TEST(testRowContractArenaHarLivenessInPlaceAcrossModes);
    RUN_TEST(testRowContractHeapHarLivenessInPlaceAcrossModes);
    RUN_TEST(testRowContractArenaInPlaceUnderEveryPolicy);
    RUN_TEST(testRowContractHeapInPlaceUnderEveryPolicy);
    RUN_TEST(testAnInPlaceSymPairKeepsTwoHeadersOnBothRows);
    RUN_TEST(testAnInPlaceFlattenMemberTakesItsHeaderAtItsRootsBind);
    RUN_TEST(testHarLivenessInPlaceObservesItsPlannedPeakOnBothRows);
    RUN_TEST(testBindAndReleaseExitOnAnInPlaceMember);
    RUN_TEST(testARootsBindAndReleaseExitWhenAMemberLeftItsBytes);
    RUN_TEST(testAnInPlaceCallExitsAtTheKeyCheckWhenAMemberChangedDtype);
#ifdef ODT_TEST_ASAN
    RUN_TEST(testArenaIsPoisonedUntilARangeOpens);
    RUN_TEST(testArenaPadStaysPoisonedWhileBound);
    RUN_TEST(testArenaEvalPadStaysPoisonedWhileBound);
    RUN_TEST(testArenaReadAfterReleaseTripsAsan);
#endif
    RUN_TEST(testHeapInitBuildsTheTableAndThePlan);
#ifdef ODT_MEM_PROFILE
    RUN_TEST(testHeapInitReservesOnlyTheTableAndThePlan);
#endif
    RUN_TEST(testHeapReportIsPlacedAndReservedWithoutAnArena);
    RUN_TEST(testHeapPlanBlockHoldsTheSequenceSteps);
    RUN_TEST(testHeapInitExitsOnAByteCountOverflow);
    RUN_TEST(testHeapWalkHandsOutThePlanStepsAndEndsWithNothingBound);
#ifdef ODT_MEM_PROFILE
    RUN_TEST(testHeapHoldsExactlyTheLiveBytesAtEveryStep);
    RUN_TEST(testHeapHoldsExactlyTheLiveBytesAtEveryStepInPlace);
#endif
    RUN_TEST(testHeapSecondCallRestartsTheWalk);
    RUN_TEST(testHeapEndExitsOnAnIncompleteWalk);
    RUN_TEST(testHeapEndInEvalChecksTheEvalWalk);
    RUN_TEST(testBeginEvalExitsOnASchedulerThatIsNotInitialised);
    RUN_TEST(testBeginEvalExitsWhenReEntered);
#ifndef ODT_TEST_ASAN
    RUN_TEST(testHeapNextExitsNamingTheStepAndTheWireWhenAReservationFails);
#endif
    RUN_TEST(testEachInitInstallsItsRowsFunctionTable);
    RUN_TEST(testEveryRowFillsEverySlot);
    RUN_TEST(testTheDispatchWalksEitherRow);
    RUN_TEST(testTheDispatchCallsThroughTheInstancesFunctionTable);
    RUN_TEST(testBeginExitsOnASchedulerThatWasNeverInitialised);
    RUN_TEST(testBeginExitsOnAHeapWhoseInitReturnedFalse);
    RUN_TEST(testBeginExitsWhenTheCallIsReEntered);
    RUN_TEST(testNextExitsOutsideACall);
    RUN_TEST(testNextExitsOnAnUninitialisedScheduler);
    RUN_TEST(testDoneExitsOutsideACall);
    RUN_TEST(testEndExitsOutsideACall);
    RUN_TEST(testDeinitExitsInsideACall);
    RUN_TEST(testInitSelfCheckExitsOnAnUnaccountedByteOnHeap);
    RUN_TEST(testInitSelfCheckExitsOnAnUnaccountedByteOnArena);
#ifdef ODT_MEM_PROFILE
    RUN_TEST(testHeapInitExitsOnAnUnaccountedReservation);
    RUN_TEST(testArenaInitExitsOnAnUnaccountedReservation);
#endif
#ifdef ODT_TEST_ASAN
    RUN_TEST(testHeapWireEndsAtItsExactBytes);
    RUN_TEST(testHeapReadAfterReleaseTripsAsan);
    RUN_TEST(testHeapReadOfAnInPlaceMemberAfterItsChainEndedTripsAsan);
#endif
    return UNITY_END();
}
