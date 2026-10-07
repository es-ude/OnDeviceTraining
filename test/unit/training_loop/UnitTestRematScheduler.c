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
#include "LinearApi.h"
#include "LossFunction.h"
#include "Pool1dApi.h"
#include "QuantLayerApi.h"
#include "Quantization.h"
#include "RNG.h"
#include "ReluApi.h"
#include "RematPlace.h"
#include "RematPlan.h"
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
    /* slab + plan block + the offsets block: 4,848 + 544 + 23 * 8 = 5,576 on LP64 */
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
    *peakPlaced = 0;
    for (size_t s = 0; s < rp->program.numSteps; s++) {
        size_t sum = 0;
        for (size_t r = 0; r < numRanges; r++) {
            if (rp->ranges[r].begin <= s && s <= rp->ranges[r].end) {
                sum += oracleRangePlaced(rp, r);
            }
        }
        if (sum > *peakPlaced) {
            *peakPlaced = sum;
        }
    }
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

/* FFD heuristic; a change needs a stated reason. The one
 * pin recorded from the implementation, cross-checked against an independent
 * FFD of the first-fit rule on the hand-derived HAR ranges:
 * STORE_ALL leaves a 4,096 B gap above its 74,288 B peak, LIVENESS none. */
void testReportArenaBytesHarFfdRegressionGuard(void) {
    TEST_ASSERT_EQUAL_size_t(78384, reportAfterInit(buildHarModel, NULL).arenaBytes);
    TEST_ASSERT_EQUAL_size_t(49152, reportAfterInit(buildHarModel, &g_liveness).arenaBytes);
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
    c->model[0] = makeLinear(features, features, false);
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

/* A tampered or imported range ending past the last step would stay bound
 * across the call boundary; the grammar does not check range ends before
 * PR6. F1 STORE_ALL: every range ends at step 4, the seed is last in endOrder. */
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
        assertResident(t, rematActId(t, l), "FORWARD reads its input");
        assertResident(t, rematActId(t, l + 1u), "FORWARD writes its output");
        break;
    case REMAT_STEP_LOSS_FORWARD:
        assertResident(t, rematActId(t, n), "LOSS_FORWARD reads ACT n");
        break;
    case REMAT_STEP_LOSS_BACKWARD:
        assertResident(t, rematActId(t, n), "LOSS_BACKWARD reads ACT n");
        assertResident(t, rematGradId(t, n), "LOSS_BACKWARD writes the seed");
        break;
    default: { /* REMAT_STEP_BACKWARD */
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
    }
}

/* O(W^2), tests only. */
static void assertResidentHeadersDisjoint(const rematWireTable_t *t) {
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
            /* Integer intervals: HEAP wires live in separate blocks, whose
             * pointers C does not order. */
            uintptr_t ua = (uintptr_t)da;
            uintptr_t ub = (uintptr_t)db;
            bool disjoint = ua + rematWireBytes(t, a) <= ub || ub + rematWireBytes(t, b) <= ua;
            TEST_ASSERT_TRUE_MESSAGE(disjoint, "two resident headers share bytes");
        }
    }
}

static void assertEndedRangesReleased(const rematProgram_t *p, const rematWireTable_t *t,
                                      size_t step) {
    for (size_t r = 0; r < p->numRanges; r++) {
        if (p->ranges[r].end == step) {
            TEST_ASSERT_NULL_MESSAGE(rematWireHdr(t, p->ranges[r].wire)->data,
                                     "a range that ended at this step is still bound");
        }
    }
}

/* A row binds a wire at its range's first step and releases it after its last
 * (P8), no earlier and no later: the memory check compares against the SDK's
 * liveBytes, which an early bind raises too, so only this pins the bound set.
 * After next a range ending at the step is still live; after done it is not. */
static void assertBoundIsExactlyTheLiveSet(const rematProgram_t *p, const rematWireTable_t *t,
                                           size_t step, bool afterDone) {
    for (uint16_t w = 1; w < t->numWires; w++) {
        bool live = false;
        for (size_t r = 0; r < p->numRanges; r++) {
            const rematRange_t *rg = &p->ranges[r];
            if (rg->wire == w && rg->begin <= step &&
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
    size_t memAfterInit = memProfileCurrentBytes();
    size_t peakHeld = 0;
    bindAndBeginIn(f, s, mode);
    TEST_ASSERT_EQUAL_PTR(f->x, rematActHdr(t, 0));
    rematStep_t st;
    size_t step = 0;
    while (rematNext(s, &st)) {
        assertOperandsResident(t, f->model, &st);
        assertResidentHeadersDisjoint(t);
        assertBoundDataAligned(t);
        assertBoundIsExactlyTheLiveSet(p, t, step, false);
        size_t held = memProfileCurrentBytes() - memAfterInit;
        TEST_ASSERT_EQUAL_size_t_MESSAGE(
            heldInsideTheCall(s), held, "the row holds other bytes than its live wires after next");
        if (held > peakHeld) {
            peakHeld = held;
        }
        rematDone(s, &st);
        assertEndedRangesReleased(p, t, step);
        assertBoundIsExactlyTheLiveSet(p, t, step, true);
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
    TEST_ASSERT_EQUAL_size_t(memAfterInit, memProfileCurrentBytes());
}

/* Two calls: ARENA reuses its resident bytes without zeroing, HEAP reserves
 * fresh blocks (VERIFY poisons both at every bind on the test presets). While
 * the global stream exists (PR1-PR5c), a row neither draws from nor reseeds
 * it; the conv factories draw their initial weights, so the pin starts
 * after the model is built. */
static void assertRowContract(rowInit_t init, void (*build)(fixture_t *),
                              const rematPlanSpec_t *spec) {
    fixture_t f;
    build(&f);
    uint32_t seed = rngGetSeed();
    rematScheduler_t s = init(&f, spec);
    walkWithTheContractChecks(&f, &s, REMAT_MODE_TRAIN);
    walkWithTheContractChecks(&f, &s, REMAT_MODE_TRAIN);
    TEST_ASSERT_EQUAL_UINT32_MESSAGE(seed, rngGetSeed(), "a row touched the global RNG stream");
    freeFixture(&f, &s);
    TEST_ASSERT_EQUAL_UINT32_MESSAGE(seed, rngGetSeed(), "a row touched the global RNG stream");
}

/* Evaluation on a persistent scheduler between training calls: every call
 * walks its own mode's program under every contract check (on HEAP one
 * exactly-sized block per EVAL range, peaking at the EVAL program's peak), and
 * the training call after two eval calls is unaffected. */
static void assertRowContractAcrossModes(rowInit_t init, void (*build)(fixture_t *),
                                         const rematPlanSpec_t *spec) {
    fixture_t f;
    build(&f);
    uint32_t seed = rngGetSeed();
    rematScheduler_t s = init(&f, spec);
    walkWithTheContractChecks(&f, &s, REMAT_MODE_TRAIN);
    walkWithTheContractChecks(&f, &s, REMAT_MODE_EVAL);
    walkWithTheContractChecks(&f, &s, REMAT_MODE_EVAL);
    walkWithTheContractChecks(&f, &s, REMAT_MODE_TRAIN);
    TEST_ASSERT_EQUAL_UINT32_MESSAGE(seed, rngGetSeed(), "a row touched the global RNG stream");
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
    /* slab + plan block, no row table: 4,848 + 544 = 5,392 on LP64 */
    TEST_ASSERT_EQUAL_size_t(s.wires->slabBytes + s.plan->blockBytes, r.metadataBytes);
    freeFixture(&f, &s);
}

void testHeapReportIsPlacedAndReservedWithoutAnArena(void) {
    assertHeapReport(NULL, REMAT_PLAN_STORE_ALL, 74288);
    assertHeapReport(&g_liveness, REMAT_PLAN_LIVENESS, 49152);
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
void testHeapHoldsExactlyTheLiveBytesAtEveryStep(void) {
    fixture_t f;
    buildHarModel(&f);
    rematScheduler_t s = initHeap(&f, &g_liveness);
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
    TEST_ASSERT_EQUAL_size_t(49152, peak);
    TEST_ASSERT_EQUAL_size_t(memAfterInit, memProfileCurrentBytes());
    freeFixture(&f, &s);
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
    RUN_TEST(testReportArenaBytesHarFfdRegressionGuard);
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
    RUN_TEST(testRowContractHeapHarAcrossModes);
    RUN_TEST(testRowContractHeapF1AcrossModesLiveness);
    RUN_TEST(testRowContractArenaHarAcrossModes);
    RUN_TEST(testRowContractArenaF1AcrossModesLiveness);
    RUN_TEST(testArenaEvalPlacesEvenActsAtTheBottomAndOddActsAtTheTop);
    RUN_TEST(testArenaHarLivenessBindsFlattenDxAfterItsSourceDied);
    RUN_TEST(testReportObservedPeakEqualsThePlannedPeak);
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
    RUN_TEST(testHeapInitExitsOnAByteCountOverflow);
    RUN_TEST(testHeapWalkHandsOutThePlanStepsAndEndsWithNothingBound);
#ifdef ODT_MEM_PROFILE
    RUN_TEST(testHeapHoldsExactlyTheLiveBytesAtEveryStep);
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
#ifdef ODT_TEST_ASAN
    RUN_TEST(testHeapWireEndsAtItsExactBytes);
    RUN_TEST(testHeapReadAfterReleaseTripsAsan);
#endif
    return UNITY_END();
}
