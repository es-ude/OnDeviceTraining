#define SOURCE_FILE "UNIT_TEST_REMAT_SCHEDULER"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "Common.h"
#include "Conv1dApi.h"
#include "DeathTest.h"
#include "FlattenApi.h"
#include "Layer.h"
#include "LayerQuant.h"
#include "LinearApi.h"
#include "LossFunction.h"
#include "Pool1dApi.h"
#include "QuantLayerApi.h"
#include "Quantization.h"
#include "ReluApi.h"
#include "RematPlace.h"
#include "RematPlan.h"
#include "RematScheduler.h"
#include "SoftmaxApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

/* Fixture helpers copied from UnitTestRematPlan.c (plan Assumption 14).
 * Fixture layers borrow their wire templates, so one FLOAT32 template
 * outlives every fixture model. */
static quantization_t g_floatQ = {.type = FLOAT32, .qConfig = NULL};

static layer_t *makeLinear(size_t in, size_t out, bool frozen) {
    layerQuant_t lq;
    layerQuantInitUniform(&lq, &g_floatQ);
    return linearLayerInit(
        &(linearInit_t){.inFeatures = in,
                        .outFeatures = out,
                        .trainable = frozen ? TRAINABLE_FALSE : TRAINABLE_DEFAULT},
        &lq);
}

static layer_t *makeRelu(quantization_t *q) {
    return reluLayerInit(&(layerQuant_t){.outputQ = q, .propLossQ = q});
}

static layer_t *makeSoftmax(void) {
    layerQuant_t lq;
    layerQuantInitUniform(&lq, &g_floatQ);
    return softmaxLayerInit(&lq);
}

static layer_t *makeQuant(quantization_t *outputQ, quantization_t *propLossQ) {
    return quantLayerInit(&(layerQuant_t){.outputQ = outputQ, .propLossQ = propLossQ});
}

static void freeModel(layer_t **model, size_t n) {
    for (size_t i = 0; i < n; i++) {
        switch (model[i]->type) {
        case LINEAR:
            freeLinearLayer(model[i]);
            break;
        case RELU:
            freeReluLayer(model[i]);
            break;
        case SOFTMAX:
            freeSoftmaxLayer(model[i]);
            break;
        case CONV1D:
            freeConv1dLayer(model[i]);
            break;
        case MAXPOOL1D:
            freeMaxPool1dLayer(model[i]);
            break;
        case AVGPOOL1D:
            freeAvgPool1dLayer(model[i]);
            break;
        case FLATTEN:
            freeFlattenLayer(model[i]);
            break;
        case QUANTIZATION:
            freeQuantLayer(model[i]);
            break;
        default:
            TEST_FAIL_MESSAGE("freeModel: extend the switch for this layer type");
        }
    }
}

/* A borrowed input header on the caller's stack. Init and bind never read its
 * data, so data stays NULL. */
#define TEST_MAX_RANK 4
typedef struct inputLike {
    size_t dims[TEST_MAX_RANK];
    size_t order[TEST_MAX_RANK];
    shape_t shape;
    tensor_t tensor;
} inputLike_t;

static tensor_t *makeInput(inputLike_t *in, const size_t *dims, size_t rank, quantization_t *q) {
    TEST_ASSERT_TRUE(rank <= TEST_MAX_RANK);
    for (size_t d = 0; d < rank; d++) {
        in->dims[d] = dims[d];
        in->order[d] = d;
    }
    in->shape = (shape_t){
        .numberOfDimensions = rank, .dimensions = in->dims, .orderOfDimensions = in->order};
    in->tensor = (tensor_t){.data = NULL, .shape = &in->shape, .quantization = q, .sparsity = NULL};
    return &in->tensor;
}

/* examples/har_classifier/train_c.c:178-216 (B = 1). */
#define HAR_N 12
static void buildHar(layer_t **model, bool freezeConvs) {
    layerQuant_t lq;
    layerQuantInitUniform(&lq, &g_floatQ);
    trainable_t conv = freezeConvs ? TRAINABLE_FALSE : TRAINABLE_DEFAULT;
    model[0] = conv1dLayerInit(&(conv1dInit_t){.inChannels = 9,
                                               .outChannels = 16,
                                               .kernelSize = 7,
                                               .padding = SAME,
                                               .trainable = conv},
                               &lq);
    model[1] = reluLayerInit(&lq);
    model[2] = maxPool1dLayerInit(
        &(maxPool1dInit_t){.kernelSize = 2, .stride = 2, .inputChannels = 16, .inputLength = 128},
        &lq);
    model[3] = conv1dLayerInit(&(conv1dInit_t){.inChannels = 16,
                                               .outChannels = 32,
                                               .kernelSize = 5,
                                               .padding = SAME,
                                               .trainable = conv},
                               &lq);
    model[4] = reluLayerInit(&lq);
    model[5] = maxPool1dLayerInit(
        &(maxPool1dInit_t){.kernelSize = 2, .stride = 2, .inputChannels = 32, .inputLength = 64},
        &lq);
    model[6] = conv1dLayerInit(&(conv1dInit_t){.inChannels = 32,
                                               .outChannels = 64,
                                               .kernelSize = 3,
                                               .padding = SAME,
                                               .trainable = conv},
                               &lq);
    model[7] = reluLayerInit(&lq);
    model[8] = avgPool1dLayerInit(&(avgPool1dInit_t){.kernelSize = 32, .stride = 32}, &lq);
    model[9] = flattenLayerInit();
    model[10] = linearLayerInit(&(linearInit_t){.inFeatures = 64, .outFeatures = 6}, &lq);
    model[11] = softmaxLayerInit(&lq);
}

static tensor_t *makeHarInput(inputLike_t *in) {
    return makeInput(in, (size_t[]){1, 9, 128}, 3, &g_floatQ);
}

static uint32_t nextRandom(uint32_t *state) { /* xorshift32, test-local */
    uint32_t v = *state;
    v ^= v << 13;
    v ^= v >> 17;
    v ^= v << 5;
    *state = v;
    return v;
}

static const rematPlanSpec_t g_liveness = {.policy = REMAT_PLAN_LIVENESS};

/* One fixture model and its borrowed input. The BFP template of the F1 model
 * (Task 3) lives here too, because the model's layers borrow it: the fixture
 * must not move while the model is alive. */
typedef struct arenaFixture {
    layer_t *model[HAR_N];
    size_t n;
    lossFuncType_t lt;
    inputLike_t in;
    tensor_t *x;
    uint8_t bfpExponent[1];
    bfpQConfig_t bfpQc;
    quantization_t bfpQ;
} arenaFixture_t;

static void buildHarModel(arenaFixture_t *f) {
    buildHar(f->model, false);
    f->n = HAR_N;
    f->lt = CROSS_ENTROPY;
    f->x = makeHarInput(&f->in);
}

static rematScheduler_t initArena(arenaFixture_t *f, const rematPlanSpec_t *spec) {
    rematScheduler_t s;
    TEST_ASSERT_TRUE(rematArenaInit(&s, f->model, f->n, defaultLossConfig(f->lt), f->x, spec));
    TEST_ASSERT_NOT_NULL(s.wires);
    TEST_ASSERT_NOT_NULL(s.plan);
    return s;
}

static void freeFixture(arenaFixture_t *f, rematScheduler_t *s) {
    rematSchedulerDeinit(s);
    freeModel(f->model, f->n);
}

/* ---- init builds the shared table and plan; the report (spec §5.1, §12.2 item 10) ---- */

void testArenaInitBuildsTheTableAndThePlan(void) {
    arenaFixture_t f;
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
    arenaFixture_t f;
    buildHarModel(&f);
    rematScheduler_t s = initArena(&f, &g_liveness);
    rematReport_t r;
    rematSchedulerReport(&s, &r);
    TEST_ASSERT_EQUAL_INT(REMAT_ARENA, r.type);
    TEST_ASSERT_EQUAL_INT(REMAT_PLAN_LIVENESS, r.policy);
    TEST_ASSERT_TRUE(r.planned);
    TEST_ASSERT_EQUAL_size_t(25, r.numSteps); /* §4.3: 12 + 1 + 1 + 11 */
    freeFixture(&f, &s);
}

static size_t reportedPeakOnHar(const rematPlanSpec_t *spec) {
    arenaFixture_t f;
    buildHarModel(&f);
    rematScheduler_t s = initArena(&f, spec);
    rematReport_t r;
    rematSchedulerReport(&s, &r);
    freeFixture(&f, &s);
    return r.peakLiveBytes;
}

/* Scan-model pins (spec §1, §12.2 item 10), read from the report. */
void testReportPeakLiveBytesHarStoreAllIs74288(void) {
    TEST_ASSERT_EQUAL_size_t(74288, reportedPeakOnHar(NULL));
}

void testReportPeakLiveBytesHarLivenessIs49152(void) {
    TEST_ASSERT_EQUAL_size_t(49152, reportedPeakOnHar(&g_liveness));
}

/* RF3: never initialised, failed before the plan existed, or deinitialised --
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
    arenaFixture_t f;
    buildHarModel(&f);
    rematScheduler_t s = initArena(&f, NULL);
    rematReport_t r;
    rematSchedulerReport(&s, &r);
    TEST_ASSERT_EQUAL_size_t(s.wires->slabBytes + s.plan->blockBytes, r.metadataBytes);
    freeFixture(&f, &s);
}

void testDeinitIsNullSafeAndIdempotent(void) {
    rematSchedulerDeinit(NULL);
    arenaFixture_t f;
    buildHarModel(&f);
    rematScheduler_t s = initArena(&f, NULL);
    rematSchedulerDeinit(&s);
    TEST_ASSERT_NULL(s.wires);
    TEST_ASSERT_NULL(s.plan);
    rematSchedulerDeinit(&s);
    freeModel(f.model, f.n);
}

#ifdef ODT_MEM_PROFILE
/* The live-byte counter is real only under ODT_MEM_PROFILE (unit_test_debug,
 * asan, ubsan); the plain unit_test preset compiles this out. */
void testDeinitReturnsEveryInitBlock(void) {
    arenaFixture_t f;
    buildHarModel(&f);
    size_t before = memProfileCurrentBytes();
    rematScheduler_t s = initArena(&f, &g_liveness);
    TEST_ASSERT_TRUE(memProfileCurrentBytes() > before);
    rematSchedulerDeinit(&s);
    TEST_ASSERT_EQUAL_size_t(before, memProfileCurrentBytes());
    freeModel(f.model, f.n);
}
#endif

/* ---- aligned first-fit-decreasing placement (spec §5.5, §12.2 items 7 and 10) ---- */

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

/* The Codex F1 alignment model (spec §5.5): FLOAT32 [1,5] -> Quantization to
 * BFP m = 8 -> Linear 5 -> 1 under MSE. Wires: ACT 1 (BFP, 5 B), ACT 2
 * (FLOAT32, 4 B), the seed GRAD 2 (id 3, 4 B). Steps F0 0, F1 1, LF 2, LB 3,
 * B1 4. Unaligned FFD would place them at {0, 5, 9}. */
static void buildF1Model(arenaFixture_t *f) {
    initBfpQConfigInto(8, 8, HALF_AWAY, f->bfpExponent, &f->bfpQc);
    f->bfpQ = (quantization_t){.type = BFP, .qConfig = &f->bfpQc};
    f->model[0] = makeQuant(&f->bfpQ, &g_floatQ);
    f->model[1] = makeLinear(5, 1, false);
    f->n = 2;
    f->lt = MSE;
    f->x = makeInput(&f->in, (size_t[]){1, 5}, 2, &g_floatQ);
}

void testArenaPlacedRoundsEveryWireUpToTheWireAlignment(void) {
    arenaFixture_t f;
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
    arenaFixture_t h;
    buildHarModel(&h);
    builtPlan_t hb = buildTableAndPlan(h.model, h.n, h.lt, h.x, NULL);
    for (uint16_t w = 1; w < hb.t->numWires; w++) {
        TEST_ASSERT_EQUAL_size_t(rematWireBytes(hb.t, w), arenaPlaced(hb.t, w));
    }
    freeTableAndPlan(&hb);
    freeModel(h.model, h.n);
}

static void assertF1Placement(const rematPlanSpec_t *spec) {
    arenaFixture_t f;
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

/* §12.2 item 7: the spec's name for the alignment pin. */
void testArenaOffsetsAligned(void) {
    assertF1Placement(NULL);
    assertF1Placement(&g_liveness);
}

void testFfdPeakPlacedBytesIsThePeakOfPlacedSums(void) {
    arenaFixture_t f;
    buildF1Model(&f);
    builtPlan_t b = buildTableAndPlan(f.model, f.n, f.lt, f.x, NULL);
    size_t offsets[3];
    size_t bytes = 0;
    size_t peak = 0;
    TEST_ASSERT_TRUE(arenaPlaceFirstFitDecreasing(b.t, &b.p->train, offsets, &bytes, &peak));
    TEST_ASSERT_EQUAL_size_t(24, peak); /* all three co-live at step 3: 8 + 8 + 8, exact 13 */
    freeTableAndPlan(&b);
    freeModel(f.model, f.n);

    arenaFixture_t h;
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
     * array (begin) position, as it used to, the spec's "begin, then wire
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

/* The spec's rule taken literally, O(R^3): every candidate (0, or the end of a
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

/* Codex N3: the placement, within the O(R^2 log R) bound, must reproduce the
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
/* Codex N3: the candidate list is a temporary block, released before the
 * placement returns (spec §11.1: "freed before init returns"). */
void testFfdReleasesItsScratch(void) {
    arenaFixture_t h;
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

/* ---- the placement verifier (spec §5.5, R6) ---- */

typedef struct placedPlan {
    arenaFixture_t f;
    builtPlan_t b;
    size_t offsets[23];
    size_t bytes;
} placedPlan_t;

static void placeFixture(placedPlan_t *pp, void (*build)(arenaFixture_t *),
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

static void assertTheFfdPlacementVerifies(void (*build)(arenaFixture_t *),
                                          const rematPlanSpec_t *spec) {
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
 * begins; they must not share bytes (spec §4.2). */
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

/* RF5: an imported offset of SIZE_MAX - 7 (a multiple of 8) makes
 * off + placed wrap to 0; the bound must not be computed that way. */
void testVerifierRejectsAnOffsetNearSizeMax(void) {
    placedPlan_t pp;
    placeFixture(&pp, buildF1Model, NULL);
    pp.offsets[2] = SIZE_MAX - 7u;
    ASSERT_EXITS_WITH_OUTPUT(1, "(8 placed bytes) ends past the arena's 24 bytes",
                             verifyPlaced(&pp));
    freePlacedFixture(&pp);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testArenaInitBuildsTheTableAndThePlan);
    RUN_TEST(testReportEchoesTypePolicyAndPlanFacts);
    RUN_TEST(testReportPeakLiveBytesHarStoreAllIs74288);
    RUN_TEST(testReportPeakLiveBytesHarLivenessIs49152);
    RUN_TEST(testReportOnAZeroedSchedulerIsEmpty);
    RUN_TEST(testReportMetadataCountsTheResidentBlocks);
    RUN_TEST(testDeinitIsNullSafeAndIdempotent);
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
    return UNITY_END();
}
