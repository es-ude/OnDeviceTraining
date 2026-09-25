#define SOURCE_FILE "UNIT_TEST_REMAT_PLAN"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "AsanDeath.h"
#include "Common.h"
#include "Conv1d.h"
#include "Conv1dApi.h"
#include "DeathTest.h"
#include "Deserialize.h"
#include "FlattenApi.h"
#include "Layer.h"
#include "LayerNormApi.h"
#include "LayerQuant.h"
#include "LinearApi.h"
#include "LossFunction.h"
#include "Pool1dApi.h"
#include "QuantLayerApi.h"
#include "Quantization.h"
#include "ReluApi.h"
#include "RematPlan.h"
#include "Serialize.h"
#include "SoftmaxApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

/* Fixture layers borrow their wire templates, so one FLOAT32 template outlives
 * every fixture model. Tests that edit a template use a local one. */
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
        case LAYERNORM:
            freeLayerNormLayer(model[i]);
            break;
        default:
            TEST_FAIL_MESSAGE("freeModel: extend the switch for this layer type");
        }
    }
}

static layer_t *makeQuant(quantization_t *outputQ, quantization_t *propLossQ) {
    return quantLayerInit(&(layerQuant_t){.outputQ = outputQ, .propLossQ = propLossQ});
}

/* A borrowed input header on the caller's stack. Table init and bind never read
 * its data, so data stays NULL unless a test sets it. */
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

static rematWireTable_t *initTable(layer_t **model, size_t n, lossFuncType_t lt,
                                   const tensor_t *input) {
    rematWireTable_t *t = NULL;
    TEST_ASSERT_TRUE(rematWireTableInit(&t, model, n, defaultLossConfig(lt), input));
    TEST_ASSERT_NOT_NULL(t);
    return t;
}

/* examples/har_classifier/train_c.c:178-216 (B = 1); freezeConvs gives the
 * stage-2 backbone of train_c_finetune.c:171-216. */
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

static layer_t *makeLayerNorm(size_t features, bool frozen) {
    layerQuant_t lq;
    layerQuantInitUniform(&lq, &g_floatQ);
    return layerNormLayerInit(
        &(layerNormInit_t){.normalizedShape = (size_t[]){features},
                           .numNormDims = 1,
                           .trainable = frozen ? TRAINABLE_FALSE : TRAINABLE_DEFAULT},
        &lq);
}

static void assertDims(const shape_t *shape, const size_t *dims, size_t rank) {
    TEST_ASSERT_EQUAL_size_t(rank, shape->numberOfDimensions);
    for (size_t d = 0; d < rank; d++) {
        TEST_ASSERT_EQUAL_size_t(dims[d], shape->dimensions[d]);
    }
}

/* [Linear 2->8, Quant FLOAT32->BFP per-tensor] under MSE: ACT 1 (32 B), ACT 2
 * (BFP, 8 B, 1 exponent slot), the seed GRAD 2 (id 3, inherits ACT 2, 8 B,
 * its exponent byte the last object of the block) and GRAD 1 (id 4, 32 B). */
typedef struct seedFixture {
    uint8_t tmplExponent[1];
    bfpQConfig_t tmplQc;
    quantization_t tmplQ;
    layer_t *model[2];
    inputLike_t in;
    tensor_t *x;
    rematWireTable_t *t;
} seedFixture_t;

static void buildBfpSeedFixture(seedFixture_t *f) {
    initBfpQConfigInto(8, 8, HALF_AWAY, f->tmplExponent, &f->tmplQc);
    f->tmplQ = (quantization_t){.type = BFP, .qConfig = &f->tmplQc};
    f->model[0] = makeLinear(2, 8, false);
    f->model[1] = makeQuant(&f->tmplQ, &g_floatQ);
    f->x = makeInput(&f->in, (size_t[]){1, 2}, 2, &g_floatQ);
    f->t = initTable(f->model, 2, MSE, f->x);
    rematWireTableBind(f->t, f->model, 2, MSE, f->x);
}

static void freeSeedFixture(seedFixture_t *f) {
    rematWireTableFree(f->t);
    freeModel(f->model, 2);
}

#define SEED_ACT1 1u
#define SEED_ACT2 2u
#define SEED_GRAD2 3u

/* ---- rematBackwardRange (spec §4.3, §12.1) ---- */

void testBackwardRangeMseRunsFromLastLayerToDeepest(void) {
    layer_t *model[3] = {makeLinear(2, 4, false), makeRelu(&g_floatQ), makeLinear(4, 2, false)};
    size_t deepest = 99;
    ptrdiff_t top = 99;
    rematBackwardRange(model, 3, MSE, &deepest, &top);
    TEST_ASSERT_EQUAL_size_t(0, deepest);
    TEST_ASSERT_EQUAL_INT(2, (int)top);
    freeModel(model, 3);
}

void testBackwardRangeCrossEntropySkipsTheLastLayerPositionally(void) {
    layer_t *model[2] = {makeLinear(2, 3, false), makeSoftmax()};
    size_t deepest = 99;
    ptrdiff_t top = 99;
    rematBackwardRange(model, 2, CROSS_ENTROPY, &deepest, &top);
    TEST_ASSERT_EQUAL_size_t(0, deepest);
    TEST_ASSERT_EQUAL_INT(0, (int)top);
    freeModel(model, 2);
}

void testBackwardRangeTruncatesAtDeepestTrainable(void) {
    layer_t *model[3] = {makeLinear(2, 4, true), makeRelu(&g_floatQ), makeLinear(4, 2, false)};
    size_t deepest = 99;
    ptrdiff_t top = 99;
    rematBackwardRange(model, 3, MSE, &deepest, &top);
    TEST_ASSERT_EQUAL_size_t(2, deepest);
    TEST_ASSERT_EQUAL_INT(2, (int)top);
    freeModel(model, 3);
}

void testBackwardRangeAllFrozenReturnsModelSize(void) {
    layer_t *model[2] = {makeLinear(2, 4, true), makeRelu(&g_floatQ)};
    size_t deepest = 99;
    ptrdiff_t top = 99;
    rematBackwardRange(model, 2, MSE, &deepest, &top);
    TEST_ASSERT_EQUAL_size_t(2, deepest);
    TEST_ASSERT_EQUAL_INT(1, (int)top);
    freeModel(model, 2);
}

/* D20: n = 1 under CE keeps today's signed top = -1. */
void testBackwardRangeSingleLayerUnderCrossEntropyIsMinusOne(void) {
    layer_t *model[1] = {makeLinear(2, 3, false)};
    size_t deepest = 99;
    ptrdiff_t top = 99;
    rematBackwardRange(model, 1, CROSS_ENTROPY, &deepest, &top);
    TEST_ASSERT_EQUAL_size_t(0, deepest);
    TEST_ASSERT_EQUAL_INT(-1, (int)top);
    freeModel(model, 1);
}

/* ---- C1: the one BFP wire-grouping rule ---- */

/* The helper must reproduce the rule the driver inlines today
 * (CalculateGradsSequential.c:231-246 for ACT wires, :320-333 for dx wires,
 * InferenceApi.c:78-91 for inference buffers) on its canonical cases. PR1
 * leaves those copies alone (no driver change); PR2 replaces them. */
static void assertGrouping(size_t tmplGroups, size_t tmplGroupSize, size_t elements,
                           size_t numGroups, size_t groupSize) {
    uint8_t exponents[8];
    bfpQConfig_t tmpl;
    initBfpQConfigGroupedInto(8, 8, HALF_AWAY, tmplGroups, tmplGroupSize, exponents, &tmpl);
    rematBfpGroups_t g = rematBfpWireGrouping(&tmpl, elements, REMAT_WIRE_ACT, 1);
    TEST_ASSERT_EQUAL_size_t(numGroups, g.numGroups);
    TEST_ASSERT_EQUAL_size_t(groupSize, g.groupSize);
}

void testBfpWireGroupingMatchesTheDriverRule(void) {
    assertGrouping(1, 0, 16, 1, 0);  /* per-tensor template stays per-tensor */
    assertGrouping(2, 16, 16, 1, 0); /* groupSize == N: the {1,N} spelling becomes {1,0} */
    assertGrouping(2, 4, 16, 4, 4);  /* the template's numGroups is ignored: derived from N */
    assertGrouping(8, 2, 16, 8, 2);
}

void testBfpWireGroupingExitsNamingTheWireOnAnIndivisibleGroupSize(void) {
    uint8_t exponents[2];
    bfpQConfig_t tmpl;
    initBfpQConfigGroupedInto(8, 8, HALF_AWAY, 2, 3, exponents, &tmpl);
    ASSERT_EXITS_WITH_OUTPUT(1, "BFP groupSize 3 does not divide the 16 elements of wire GRAD 5",
                             (void)rematBfpWireGrouping(&tmpl, 16, REMAT_WIRE_GRAD, 5));
}

/* ---- wire numbering and records (spec §3.1, §3.2) ---- */

void testHarTableNumbersWiresInProductionOrder(void) {
    layer_t *model[HAR_N];
    buildHar(model, false);
    inputLike_t in;
    rematWireTable_t *t = initTable(model, HAR_N, CROSS_ENTROPY, makeHarInput(&in));

    /* ACT 0..12, then GRAD {12, 10, 9, ..., 1}: 23 slab headers + the borrowed ACT 0. */
    TEST_ASSERT_EQUAL_size_t(24, t->numWires);
    for (uint16_t j = 0; j <= 12; j++) {
        TEST_ASSERT_EQUAL_UINT8(REMAT_WIRE_ACT, t->wires[j].kind);
        TEST_ASSERT_EQUAL_UINT16(j, t->wires[j].index);
        TEST_ASSERT_EQUAL_UINT16(REMAT_NONE, t->wires[j].inheritFrom);
    }
    const uint16_t gradIndex[11] = {12, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
    for (uint16_t k = 0; k < 11; k++) {
        uint16_t id = (uint16_t)(13 + k);
        TEST_ASSERT_EQUAL_UINT8(REMAT_WIRE_GRAD, t->wires[id].kind);
        TEST_ASSERT_EQUAL_UINT16(gradIndex[k], t->wires[id].index);
        TEST_ASSERT_EQUAL_UINT16(id, t->gradIdOf[gradIndex[k]]);
    }
    /* The seed inherits ACT 12; Flatten (layer 9) passes its dx through, so
     * GRAD 9 inherits ACT 9; every other GRAD has a type-derived template. */
    TEST_ASSERT_EQUAL_UINT16(12, t->wires[13].inheritFrom);
    TEST_ASSERT_EQUAL_UINT16(9, t->wires[15].inheritFrom);
    TEST_ASSERT_EQUAL_UINT16(REMAT_NONE, t->wires[14].inheritFrom);
    TEST_ASSERT_EQUAL_UINT16(REMAT_NONE, t->wires[16].inheritFrom);
    /* No GRAD 11 under CE (the positional skip), no GRAD 0 (deepest, grads-only). */
    TEST_ASSERT_EQUAL_UINT16(REMAT_NONE, t->gradIdOf[11]);
    TEST_ASSERT_EQUAL_UINT16(REMAT_NONE, t->gradIdOf[0]);

    rematWireTableFree(t);
    freeModel(model, HAR_N);
}

void testHarTableRecordsBytesRanksAndKey(void) {
    layer_t *model[HAR_N];
    buildHar(model, false);
    inputLike_t in;
    rematWireTable_t *t = initTable(model, HAR_N, CROSS_ENTROPY, makeHarInput(&in));

    /* Scan-model bytes (plan header): ACT 0..12, then the GRADs in id order. */
    const size_t bytes[24] = {4608, 8192, 8192, 4096, 8192, 8192, 4096, 8192,
                              8192, 256,  256,  24,   24,   24,   256,  256,
                              8192, 8192, 4096, 8192, 8192, 4096, 8192, 8192};
    const uint8_t rank[24] = {3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2,
                              2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3};
    for (uint16_t id = 0; id < 24; id++) {
        TEST_ASSERT_EQUAL_size_t_MESSAGE(bytes[id], t->wires[id].bytes, "bytes");
        TEST_ASSERT_EQUAL_UINT8_MESSAGE(rank[id], t->wires[id].rank, "rank");
        TEST_ASSERT_EQUAL_UINT8(FLOAT32, t->wires[id].dtype);
        TEST_ASSERT_EQUAL_UINT8(id == 0 ? 1 : 0, t->wires[id].borrowed);
        TEST_ASSERT_EQUAL_size_t(0, t->wires[id].expCapacity);
    }
    TEST_ASSERT_NULL(
        t->wires[0].hdr); /* ACT 0 is the caller's, bound only between bind and unbind */

    TEST_ASSERT_EQUAL_size_t(HAR_N, t->modelSize);
    TEST_ASSERT_EQUAL_INT(CROSS_ENTROPY, t->lossType);
    TEST_ASSERT_EQUAL_size_t(0, t->deepest);
    TEST_ASSERT_EQUAL_INT(10, (int)t->backwardTop);
    TEST_ASSERT_TRUE(t->hasBackward);
    for (size_t i = 0; i < HAR_N; i++) {
        TEST_ASSERT_EQUAL_UINT8(model[i]->type, t->layerType[i]);
        TEST_ASSERT_EQUAL_UINT8(0, t->frozen[i]);
    }
    TEST_ASSERT_EQUAL_size_t(3, t->inputRank);
    const size_t dims[3] = {1, 9, 128};
    for (size_t d = 0; d < 3; d++) {
        TEST_ASSERT_EQUAL_size_t(dims[d], t->inputDims[d]);
        TEST_ASSERT_EQUAL_size_t(d, t->inputOrder[d]);
    }
    TEST_ASSERT_EQUAL_UINT8(FLOAT32, t->inputType);
    TEST_ASSERT_EQUAL_UINT8(3, t->maxRank);

    rematWireTableFree(t);
    freeModel(model, HAR_N);
}

/* RF5: one grouped template shared by wires of different sizes groups each
 * wire by its own element count (CalculateGradsSequential.c:215-224). */
void testSharedGroupedBfpTemplateGroupsPerWire(void) {
    uint8_t tmplExponents[2];
    bfpQConfig_t tmplQc;
    initBfpQConfigGroupedInto(8, 8, HALF_AWAY, 2, 4, tmplExponents, &tmplQc);
    quantization_t bfpQ = {.type = BFP, .qConfig = &tmplQc};
    layerQuant_t lq;
    layerQuantInitUniform(&lq, &g_floatQ);
    lq.outputQ = &bfpQ;
    layer_t *model[2] = {makeQuant(&bfpQ, &g_floatQ),
                         linearLayerInit(&(linearInit_t){.inFeatures = 4, .outFeatures = 16}, &lq)};
    inputLike_t in;
    rematWireTable_t *t = initTable(model, 2, MSE, makeInput(&in, (size_t[]){1, 4}, 2, &g_floatQ));

    /* ACT 1: 4 elements, groupSize 4 == N -> per-tensor, 1 slot, 4 B.
     * ACT 2: 16 elements -> 4 groups of 4, 16 B. Seed GRAD 2 inherits ACT 2.
     * deepest == top == 1, so no other GRAD. */
    TEST_ASSERT_EQUAL_size_t(4, t->numWires);
    TEST_ASSERT_EQUAL_UINT8(BFP, t->wires[1].dtype);
    TEST_ASSERT_EQUAL_size_t(1, t->wires[1].expCapacity);
    TEST_ASSERT_EQUAL_size_t(4, t->wires[1].bytes);
    TEST_ASSERT_EQUAL_UINT8(BFP, t->wires[2].dtype);
    TEST_ASSERT_EQUAL_size_t(4, t->wires[2].expCapacity);
    TEST_ASSERT_EQUAL_size_t(16, t->wires[2].bytes);
    TEST_ASSERT_EQUAL_UINT16(2, t->wires[3].inheritFrom);
    TEST_ASSERT_EQUAL_size_t(4, t->wires[3].expCapacity);
    TEST_ASSERT_EQUAL_size_t(16, t->wires[3].bytes);

    /* Pins the exponent tail's per-wire spacing so a SLAB_PLACE(...,
     * f->numGroups, ...) regression to a fixed count cannot slip past this
     * test. In wire order, each BFP wire's exponent array sits exactly its
     * OWN expCapacity bytes before the next one's, and the last (wire 3, the
     * seed) abuts the block end. */
    bfpQConfig_t *qc1 = t->wires[1].hdr->quantization->qConfig;
    bfpQConfig_t *qc2 = t->wires[2].hdr->quantization->qConfig;
    bfpQConfig_t *qc3 = t->wires[3].hdr->quantization->qConfig;
    TEST_ASSERT_EQUAL_PTR(qc1->exponents + t->wires[1].expCapacity, qc2->exponents);
    TEST_ASSERT_EQUAL_PTR(qc2->exponents + t->wires[2].expCapacity, qc3->exponents);
    TEST_ASSERT_EQUAL_PTR((uint8_t *)t + t->slabBytes, qc3->exponents + t->wires[3].expCapacity);

    rematWireTableFree(t);
    freeModel(model, 2);
}

/* ---- one block, released by a single release call (spec §2.3, §3.3; C5) ---- */

#ifdef ODT_MEM_PROFILE
/* The live-byte counter is real only under ODT_MEM_PROFILE (unit_test_debug,
 * asan, ubsan); the plain unit_test preset compiles this out. */
void testTableInitReservesOneBlockOfSlabBytesAndFreeReturnsIt(void) {
    layer_t *model[HAR_N];
    buildHar(model, false);
    inputLike_t in;
    tensor_t *x = makeHarInput(&in);
    size_t before = memProfileCurrentBytes();
    rematWireTable_t *t = initTable(model, HAR_N, CROSS_ENTROPY, x);
    TEST_ASSERT_EQUAL_size_t(before + t->slabBytes, memProfileCurrentBytes());
    rematWireTableFree(t);
    TEST_ASSERT_EQUAL_size_t(before, memProfileCurrentBytes());
    freeModel(model, HAR_N);
}
#endif

/* C5: slab headers borrow their shape, quantization_t, qConfig and
 * BFP exponents from the one table block, so the block's single
 * freeReservedMemory is the only legal release. The free runs in a forked
 * child: releasing an interior slab pointer aborts there (or, under
 * ODT_MEM_PROFILE, corrupts the live-byte count), and the parent sees it. */
static size_t g_memBeforeTable;

static void freeTableAndExitWithBaselineVerdict(rematWireTable_t *t) {
    rematWireTableFree(t);
    _exit(memProfileCurrentBytes() == g_memBeforeTable ? 0 : 2);
}

void testTableFreeReleasesExactlyTheOneBlock(void) {
    uint8_t exponent[1];
    bfpQConfig_t bfpQc;
    initBfpQConfigInto(8, 8, HALF_AWAY, exponent, &bfpQc);
    quantization_t bfpQ = {.type = BFP, .qConfig = &bfpQc};
    layer_t *model[1] = {makeQuant(&bfpQ, &g_floatQ)};
    inputLike_t in;
    tensor_t *x = makeInput(&in, (size_t[]){1, 8}, 2, &g_floatQ);
    g_memBeforeTable = memProfileCurrentBytes();
    rematWireTable_t *t = initTable(model, 1, MSE, x);
    ASSERT_EXITS_WITH(0, freeTableAndExitWithBaselineVerdict(t));
    rematWireTableFree(t);
    freeModel(model, 1);
}

void testTableFreeIsNullSafe(void) {
    ASSERT_EXITS_WITH(0, rematWireTableFree(NULL));
}

/* ---- slab alignment (spec §3.3, §12.2 item 7) ---- */

#define ASSERT_ALIGNED(ptr, T)                                                                     \
    TEST_ASSERT_EQUAL_UINT_MESSAGE(0u, (unsigned)((uintptr_t)(ptr) % _Alignof(T)),                 \
                                   #ptr " is misaligned for " #T)

/* Wire order SYM_INT32, BFP, FLOAT32: the 12-byte SYM qConfig and the odd-sized
 * uint8_t key arrays leave the cursor unaligned for what follows them, and an
 * exponent array placed inline would sit between the BFP and the FLOAT32
 * header instead of at the tail. */
void testSlabObjectsAlignedAndExponentsAtTheTail(void) {
    symInt32QConfig_t symQc;
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &symQc, 12);
    quantization_t symQ = {.type = SYM_INT32, .qConfig = &symQc};
    uint8_t exponent[1];
    bfpQConfig_t bfpQc;
    initBfpQConfigInto(8, 8, HALF_AWAY, exponent, &bfpQc);
    quantization_t bfpQ = {.type = BFP, .qConfig = &bfpQc};
    layer_t *model[3] = {makeQuant(&symQ, &g_floatQ), makeQuant(&bfpQ, &g_floatQ),
                         makeRelu(&g_floatQ)};
    inputLike_t in;
    rematWireTable_t *t = initTable(model, 3, MSE, makeInput(&in, (size_t[]){1, 3}, 2, &g_floatQ));

    ASSERT_ALIGNED(t->wires, rematWire_t);
    ASSERT_ALIGNED(t->gradIdOf, uint16_t);
    ASSERT_ALIGNED(t->inputDims, size_t);
    ASSERT_ALIGNED(t->inputOrder, size_t);
    uintptr_t headersEnd = 0;
    for (uint16_t id = 1; id < t->numWires; id++) {
        tensor_t *hdr = t->wires[id].hdr;
        ASSERT_ALIGNED(hdr, tensor_t);
        ASSERT_ALIGNED(hdr->shape, shape_t);
        ASSERT_ALIGNED(hdr->shape->dimensions, size_t);
        ASSERT_ALIGNED(hdr->shape->orderOfDimensions, size_t);
        ASSERT_ALIGNED(hdr->quantization, quantization_t);
        uintptr_t end = (uintptr_t)hdr->quantization + sizeof(quantization_t);
        if (hdr->quantization->type == SYM_INT32) {
            ASSERT_ALIGNED(hdr->quantization->qConfig, symInt32QConfig_t);
            end = (uintptr_t)hdr->quantization->qConfig + sizeof(symInt32QConfig_t);
        } else if (hdr->quantization->type == BFP) {
            ASSERT_ALIGNED(hdr->quantization->qConfig, bfpQConfig_t);
            end = (uintptr_t)hdr->quantization->qConfig + sizeof(bfpQConfig_t);
        } else {
            TEST_ASSERT_NULL(hdr->quantization->qConfig); /* FLOAT32 reserves no qConfig */
        }
        headersEnd = end > headersEnd ? end : headersEnd;
    }
    bfpQConfig_t *slabBfp = t->wires[2].hdr->quantization->qConfig;
    TEST_ASSERT_TRUE_MESSAGE((uintptr_t)slabBfp->exponents >= headersEnd,
                             "BFP exponents are not at the slab tail");
    /* The one BFP wire's exponent array is the last object and abuts the block end. */
    TEST_ASSERT_EQUAL_PTR((uint8_t *)t + t->slabBytes, slabBfp->exponents + 1);

    rematWireTableFree(t);
    freeModel(model, 3);
}

/* ---- read-only accessors (spec §3.10) and the linked, content-free headers ---- */

void testAccessorsReadTheTableAndHeadersAreLinked(void) {
    layer_t *model[HAR_N];
    buildHar(model, false);
    inputLike_t in;
    rematWireTable_t *t = initTable(model, HAR_N, CROSS_ENTROPY, makeHarInput(&in));

    TEST_ASSERT_EQUAL_UINT16(5, rematActId(t, 5));
    TEST_ASSERT_EQUAL_UINT16(13, rematGradId(t, 12));
    TEST_ASSERT_EQUAL_UINT16(14, rematGradId(t, 10));
    TEST_ASSERT_EQUAL_UINT16(REMAT_NONE, rematGradId(t, 11));
    TEST_ASSERT_NULL(rematGradHdr(t, 11));
    TEST_ASSERT_NULL(rematGradHdr(t, 0));
    TEST_ASSERT_EQUAL_PTR(t->wires[13].hdr, rematGradHdr(t, 12));
    TEST_ASSERT_EQUAL_PTR(t->wires[3].hdr, rematActHdr(t, 3));
    TEST_ASSERT_EQUAL_PTR(t->wires[20].hdr, rematWireHdr(t, 20));
    TEST_ASSERT_NULL(rematActHdr(t, 0));
    TEST_ASSERT_EQUAL_size_t(8192, rematWireBytes(t, 1));
    TEST_ASSERT_EQUAL_size_t(24, rematWireBytes(t, 13));

    for (uint16_t id = 1; id < t->numWires; id++) {
        tensor_t *hdr = rematWireHdr(t, id);
        TEST_ASSERT_NULL(hdr->data);
        TEST_ASSERT_NULL(hdr->sparsity);
        TEST_ASSERT_EQUAL_size_t(t->wires[id].rank, hdr->shape->numberOfDimensions);
        TEST_ASSERT_EQUAL_INT(t->wires[id].dtype, hdr->quantization->type);
    }

    rematWireTableFree(t);
    freeModel(model, HAR_N);
}

/* ---- table init's named exits (spec §3.2) ---- */

void testTableInitExitsOnAnEmptyModel(void) {
    layer_t *model[1] = {makeRelu(&g_floatQ)};
    inputLike_t in;
    tensor_t *x = makeInput(&in, (size_t[]){1, 4}, 2, &g_floatQ);
    rematWireTable_t *t = NULL;
    ASSERT_EXITS_WITH_OUTPUT(1, "modelSize == 0: nothing to schedule",
                             (void)rematWireTableInit(&t, model, 0, defaultLossConfig(MSE), x));
    freeModel(model, 1);
}

/* 65534 parameter-free layers under MSE: no GRAD, so n + 1 = 65535 wires, the
 * first count whose ids would include the REMAT_NONE sentinel. One ReLU is
 * reused for every slot; the model array itself is test-owned. */
void testTableInitExitsWhenWireIdsWouldReachRematNone(void) {
    const size_t n = 65534u;
    layer_t *relu = makeRelu(&g_floatQ);
    layer_t **model = reserveMemory(n * sizeof(layer_t *));
    TEST_ASSERT_NOT_NULL(model);
    for (size_t i = 0; i < n; i++) {
        model[i] = relu;
    }
    inputLike_t in;
    tensor_t *x = makeInput(&in, (size_t[]){1, 4}, 2, &g_floatQ);
    rematWireTable_t *t = NULL;
    ASSERT_EXITS_WITH_OUTPUT(1, "65535 wires reach REMAT_NONE (0xFFFF)",
                             (void)rematWireTableInit(&t, model, n, defaultLossConfig(MSE), x));
    freeReservedMemory(model);
    freeReluLayer(relu);
}

void testTableInitExitsOnAnInputRankAboveTheRankField(void) {
    size_t dims[256];
    size_t order[256];
    for (size_t d = 0; d < 256; d++) {
        dims[d] = 1;
        order[d] = d;
    }
    shape_t shape = {.numberOfDimensions = 256, .dimensions = dims, .orderOfDimensions = order};
    tensor_t x = {.data = NULL, .shape = &shape, .quantization = &g_floatQ, .sparsity = NULL};
    layer_t *model[1] = {makeRelu(&g_floatQ)};
    rematWireTable_t *t = NULL;
    ASSERT_EXITS_WITH_OUTPUT(1, "wire ACT 0 has rank 256, above the uint8_t rank field",
                             (void)rematWireTableInit(&t, model, 1, defaultLossConfig(MSE), &x));
    freeModel(model, 1);
}

void testTableInitExitsOnAnUnsupportedWireDtype(void) {
    float scale = 1.f;
    symQConfig_t symQc = {
        .scales = &scale, .numGroups = 1, .groupSize = 0, .roundingMode = HALF_AWAY, .qBits = 8};
    quantization_t symQ = {.type = SYM, .qConfig = &symQc};
    layer_t *model[1] = {makeRelu(&symQ)};
    inputLike_t in;
    tensor_t *x = makeInput(&in, (size_t[]){1, 4}, 2, &g_floatQ);
    rematWireTable_t *t = NULL;
    ASSERT_EXITS_WITH_OUTPUT(1, "wire ACT 1 has dtype 3; remat wires are FLOAT32, SYM_INT32 or BFP",
                             (void)rematWireTableInit(&t, model, 1, defaultLossConfig(MSE), x));
    freeModel(model, 1);
}

/* #160: a zero-size block is implementation-defined (a zero-byte request may return NULL). */
void testTableInitExitsOnAZeroByteWire(void) {
    layer_t *model[1] = {makeRelu(&g_floatQ)};
    inputLike_t in;
    tensor_t *x = makeInput(&in, (size_t[]){1, 0}, 2, &g_floatQ);
    rematWireTable_t *t = NULL;
    ASSERT_EXITS_WITH_OUTPUT(1, "wire ACT 1 has zero bytes (#160)",
                             (void)rematWireTableInit(&t, model, 1, defaultLossConfig(MSE), x));
    freeModel(model, 1);
}

/* §16.1 item 4g: the borrowed ACT 0 may be any dtype, packed ones included;
 * its bytes are exact for the checker's disjointness test. */
void testTableInitAcceptsAPackedBorrowedInputAndSizesItExactly(void) {
    float scale = 1.f;
    uint16_t zeroPoint = 0;
    symQConfig_t symQc = {
        .scales = &scale, .numGroups = 1, .groupSize = 0, .roundingMode = HALF_AWAY, .qBits = 4};
    asymQConfig_t asymQc = {.scales = &scale,
                            .zeroPoints = &zeroPoint,
                            .numGroups = 1,
                            .groupSize = 0,
                            .qBits = 6,
                            .roundingMode = HALF_AWAY};
    quantization_t inputQ[4] = {{.type = SYM, .qConfig = &symQc},
                                {.type = ASYM, .qConfig = &asymQc},
                                {.type = INT32, .qConfig = NULL},
                                {.type = BOOL, .qConfig = NULL}};
    const size_t expectedBytes[4] = {4, 6, 32, 1}; /* 8 elements: 4, 6, 32 and 1 bits each */
    layer_t *model[1] = {makeRelu(&g_floatQ)};
    for (size_t k = 0; k < 4; k++) {
        inputLike_t in;
        rematWireTable_t *t =
            initTable(model, 1, MSE, makeInput(&in, (size_t[]){1, 8}, 2, &inputQ[k]));
        TEST_ASSERT_EQUAL_size_t(expectedBytes[k], rematWireBytes(t, 0));
        rematWireTableFree(t);
    }
    freeModel(model, 1);
}

void testTableInitExitsOnAnUnknownInputQtype(void) {
    quantization_t unknownQ = {.type = (qtype_t)99, .qConfig = NULL};
    layer_t *model[1] = {makeRelu(&g_floatQ)};
    inputLike_t in;
    tensor_t *x = makeInput(&in, (size_t[]){1, 4}, 2, &unknownQ);
    rematWireTable_t *t = NULL;
    ASSERT_EXITS_WITH_OUTPUT(1, "wire ACT 0 has unknown qtype 99",
                             (void)rematWireTableInit(&t, model, 1, defaultLossConfig(MSE), x));
    freeModel(model, 1);
}

/* ---- checked size arithmetic (spec §3.8, D60) ---- */

/* The child prints how many bytes are live when it exits, so the parent can
 * check "exits before the table block is reserved". Real only under
 * ODT_MEM_PROFILE; on the plain preset both counters read 0. */
static size_t g_memBeforeExit;

static void printReservedBeforeExit(void) {
    printf("reservedBeforeExit=%zu\n", memProfileCurrentBytes() - g_memBeforeExit);
}

static void initExpectingAnExit(layer_t **model, size_t n, const tensor_t *x) {
    rematWireTable_t *t = NULL;
    g_memBeforeExit = memProfileCurrentBytes();
    (void)atexit(printReservedBeforeExit);
    (void)rematWireTableInit(&t, model, n, defaultLossConfig(MSE), x);
}

/* A borrowed [1, SIZE_MAX/4 + 2] FLOAT32 input: 4 * N wraps to 4. */
void testTableInitExitsOnAByteCountOverflowBeforeReservingTheTable(void) {
    layer_t *model[1] = {makeRelu(&g_floatQ)};
    inputLike_t in;
    tensor_t *x = makeInput(&in, (size_t[]){1, SIZE_MAX / 4u + 2u}, 2, &g_floatQ);
    ASSERT_EXITS_WITH_OUTPUT(1, "size overflow computing bytes of wire ACT 0",
                             initExpectingAnExit(model, 1, x));
    /* Only the transient derivation scratch (2 wires) is live at the exit: the
     * table block is never reserved (plan Assumption 29). */
#ifdef ODT_MEM_PROFILE
    size_t expectedReserved = 2u * sizeof(rematWireFact_t);
#else
    size_t expectedReserved = 0u; /* the counters are no-ops without ODT_MEM_PROFILE */
#endif
    char expected[64];
    (void)snprintf(expected, sizeof expected, "reservedBeforeExit=%zu", expectedReserved);
    ASSERT_EXITS_WITH_OUTPUT(1, expected, initExpectingAnExit(model, 1, x));
    freeModel(model, 1);
}

/* Four BFP wires of 2^62 elements with groupSize 1: each wire's bytes fit
 * (2^60), but their exponent tails sum to 2^64. */
void testTableInitExitsOnASlabSizeOverflow(void) {
    uint8_t inputExponent[1];
    bfpQConfig_t inputQc;
    initBfpQConfigInto(2, 8, HALF_AWAY, inputExponent, &inputQc);
    quantization_t inputQ = {.type = BFP, .qConfig = &inputQc};
    uint8_t tmplExponents[2];
    bfpQConfig_t tmplQc;
    initBfpQConfigGroupedInto(2, 8, HALF_AWAY, 2, 1, tmplExponents, &tmplQc);
    quantization_t tmplQ = {.type = BFP, .qConfig = &tmplQc};
    layer_t *relu = makeRelu(&tmplQ);
    layer_t *model[4] = {relu, relu, relu, relu};
    inputLike_t in;
    tensor_t *x = makeInput(&in, (size_t[]){1, (size_t)1 << 62}, 2, &inputQ);
    ASSERT_EXITS_WITH_OUTPUT(1, "size overflow computing slabBytes of wire ACT 4",
                             initExpectingAnExit(model, 4, x));
    freeReluLayer(relu);
}

/* Three FLOAT32 wires of SIZE_MAX/8 elements: each fits (just under 2^63
 * bytes), two sum to just under 2^64, the third overflows. The checked total
 * bounds every later sum over wires (plan Assumption 12). */
void testTableInitExitsOnATotalWireBytesOverflow(void) {
    layer_t *relu = makeRelu(&g_floatQ);
    layer_t *model[3] = {relu, relu, relu};
    inputLike_t in;
    tensor_t *x = makeInput(&in, (size_t[]){1, SIZE_MAX / 8u}, 2, &g_floatQ);
    ASSERT_EXITS_WITH_OUTPUT(1, "size overflow computing total wire bytes of wire ACT 3",
                             initExpectingAnExit(model, 3, x));
    freeReluLayer(relu);
}

/* ---- per-bind re-derivation (spec §3.4, §3.5) ---- */

void testBindWritesHarHeadersAndPointsAct0AtTheInput(void) {
    layer_t *model[HAR_N];
    buildHar(model, false);
    inputLike_t in;
    tensor_t *x = makeHarInput(&in);
    rematWireTable_t *t = initTable(model, HAR_N, CROSS_ENTROPY, x);
    rematWireTableBind(t, model, HAR_N, CROSS_ENTROPY, x);

    TEST_ASSERT_EQUAL_PTR(x, rematActHdr(t, 0));
    assertDims(rematActHdr(t, 1)->shape, (size_t[]){1, 16, 128}, 3);
    assertDims(rematActHdr(t, 3)->shape, (size_t[]){1, 16, 64}, 3);
    assertDims(rematActHdr(t, 9)->shape, (size_t[]){1, 64, 1}, 3);
    assertDims(rematActHdr(t, 10)->shape, (size_t[]){1, 64}, 2);
    assertDims(rematActHdr(t, 12)->shape, (size_t[]){1, 6}, 2);
    assertDims(rematGradHdr(t, 3)->shape, (size_t[]){1, 16, 64}, 3);
    assertDims(rematGradHdr(t, 10)->shape, (size_t[]){1, 64}, 2);
    for (uint16_t id = 1; id < t->numWires; id++) {
        TEST_ASSERT_NULL(rematWireHdr(t, id)->data);
        TEST_ASSERT_EQUAL_INT(FLOAT32, rematWireHdr(t, id)->quantization->type);
    }
    /* Inherited headers (the seed, Flatten's dx) wait for rematWireBind. */
    TEST_ASSERT_EQUAL_size_t(0, rematGradHdr(t, 12)->shape->dimensions[0]);
    TEST_ASSERT_EQUAL_size_t(0, rematGradHdr(t, 9)->shape->dimensions[0]);

    rematWireTableFree(t);
    freeModel(model, HAR_N);
}

/* Forward wires copy the upstream order (ReLU, LayerNorm); a dx wire always
 * gets identity order (CalculateGradsSequential.c:288-293). */
void testBindCopiesForwardOrderAndGivesGradsIdentityOrder(void) {
    layer_t *model[2] = {makeLayerNorm(4, false), makeRelu(&g_floatQ)};
    inputLike_t in;
    tensor_t *x = makeInput(&in, (size_t[]){1, 4}, 2, &g_floatQ);
    in.order[0] = 1;
    in.order[1] = 0;
    rematWireTable_t *t = initTable(model, 2, MSE, x);
    rematWireTableBind(t, model, 2, MSE, x);

    TEST_ASSERT_EQUAL_size_t(1, rematActHdr(t, 1)->shape->orderOfDimensions[0]);
    TEST_ASSERT_EQUAL_size_t(0, rematActHdr(t, 1)->shape->orderOfDimensions[1]);
    assertDims(rematGradHdr(t, 1)->shape, (size_t[]){1, 4}, 2);
    TEST_ASSERT_EQUAL_size_t(0, rematGradHdr(t, 1)->shape->orderOfDimensions[0]);
    TEST_ASSERT_EQUAL_size_t(1, rematGradHdr(t, 1)->shape->orderOfDimensions[1]);

    rematWireTableFree(t);
    freeModel(model, 2);
}

#ifdef ODT_MEM_PROFILE
/* G4: a bind derives into the slab and reserves nothing. */
void testBindAllocatesNothing(void) {
    uint8_t exponent[1];
    bfpQConfig_t bfpQc;
    initBfpQConfigInto(8, 8, HALF_AWAY, exponent, &bfpQc);
    quantization_t bfpQ = {.type = BFP, .qConfig = &bfpQc};
    layer_t *model[1] = {makeQuant(&bfpQ, &g_floatQ)};
    inputLike_t in;
    tensor_t *x = makeInput(&in, (size_t[]){1, 8}, 2, &g_floatQ);
    rematWireTable_t *t = initTable(model, 1, MSE, x);
    size_t before = memProfileCurrentBytes();
    rematWireTableBind(t, model, 1, MSE, x);
    TEST_ASSERT_EQUAL_size_t(before, memProfileCurrentBytes());
    rematWireTableFree(t);
    freeModel(model, 1);
}
#endif

/* §12.2 item 8 (table level): Quant outputQ @8 at build, @16 at the next bind;
 * the dynamic scale restarts at its init value every bind. */
void testBindRederivesSymQMaxBitsAndResetsScale(void) {
    symInt32QConfig_t symQc;
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &symQc, 8);
    quantization_t symQ = {.type = SYM_INT32, .qConfig = &symQc};
    layer_t *model[1] = {makeQuant(&symQ, &g_floatQ)};
    inputLike_t in;
    tensor_t *x = makeInput(&in, (size_t[]){1, 4}, 2, &g_floatQ);
    rematWireTable_t *t = initTable(model, 1, MSE, x);
    rematWireTableBind(t, model, 1, MSE, x);
    symInt32QConfig_t *slabQc = rematActHdr(t, 1)->quantization->qConfig;
    TEST_ASSERT_EQUAL_UINT8(8, slabQc->qMaxBits);

    slabQc->scale = 0.25f; /* a producer's OUT_WRITE epilogue */
    symQc.qMaxBits = 16;   /* key-preserving template edit between calls */
    rematWireTableBind(t, model, 1, MSE, x);
    TEST_ASSERT_EQUAL_UINT8(16, slabQc->qMaxBits);
    TEST_ASSERT_EQUAL_FLOAT(1.0f, slabQc->scale);

    rematWireTableFree(t);
    freeModel(model, 1);
}

/* §12.2 item 8 / §3.5: a deserializeModel into a skeleton whose wire width
 * differs is a key-preserving config edit (ODTS writes the layer's outputQ in
 * place, Deserialize.c:773), adopted at the next bind. */
void testBindAfterDeserializeModel(void) {
    symInt32QConfig_t savedQc;
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &savedQc, 8);
    quantization_t savedQ = {.type = SYM_INT32, .qConfig = &savedQc};
    layer_t *saved[1] = {makeQuant(&savedQ, &g_floatQ)};
    FILE *file = tmpfile();
    TEST_ASSERT_NOT_NULL(file);
    serializeModel(saved, 1, file);
    rewind(file);

    symInt32QConfig_t skeletonQc;
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &skeletonQc, 16);
    quantization_t skeletonQ = {.type = SYM_INT32, .qConfig = &skeletonQc};
    layer_t *skeleton[1] = {makeQuant(&skeletonQ, &g_floatQ)};
    inputLike_t in;
    tensor_t *x = makeInput(&in, (size_t[]){1, 4}, 2, &g_floatQ);
    rematWireTable_t *t = initTable(skeleton, 1, MSE, x);
    rematWireTableBind(t, skeleton, 1, MSE, x);
    symInt32QConfig_t *slabQc = rematActHdr(t, 1)->quantization->qConfig;
    TEST_ASSERT_EQUAL_UINT8(16, slabQc->qMaxBits);

    deserializeModel(skeleton, 1, file);
    (void)fclose(file);
    rematWireTableBind(t, skeleton, 1, MSE, x);
    TEST_ASSERT_EQUAL_UINT8(8, slabQc->qMaxBits);

    rematWireTableFree(t);
    freeModel(skeleton, 1);
    freeModel(saved, 1);
}

/* The rounding-mode half of §12.2's testBindRederivesRoundingModeAndDrawCount;
 * the draw-count half needs the driver (PR2). */
void testBindRederivesRoundingMode(void) {
    symInt32QConfig_t symQc;
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &symQc, 12);
    quantization_t symQ = {.type = SYM_INT32, .qConfig = &symQc};
    layer_t *model[1] = {makeQuant(&symQ, &g_floatQ)};
    inputLike_t in;
    tensor_t *x = makeInput(&in, (size_t[]){1, 4}, 2, &g_floatQ);
    rematWireTable_t *t = initTable(model, 1, MSE, x);
    rematWireTableBind(t, model, 1, MSE, x);
    symQc.roundingMode = SR_HALF_AWAY;
    rematWireTableBind(t, model, 1, MSE, x);
    symInt32QConfig_t *slabQc = rematActHdr(t, 1)->quantization->qConfig;
    TEST_ASSERT_EQUAL_INT(SR_HALF_AWAY, slabQc->roundingMode);
    rematWireTableFree(t);
    freeModel(model, 1);
}

/* §12.2 item 8: Flatten-at-0 re-inherits the live input's BFP grouping at
 * every bind -- {4,4} -> {2,8} shrinks within the built capacity and succeeds,
 * with fresh zero-state exponents. */
void testBindRederivesFlattenBfpGroupingFromTheLiveInput(void) {
    uint8_t inputExponents[4];
    bfpQConfig_t inputQc;
    initBfpQConfigGroupedInto(8, 8, HALF_AWAY, 4, 4, inputExponents, &inputQc);
    quantization_t inputQ = {.type = BFP, .qConfig = &inputQc};
    layer_t *model[1] = {flattenLayerInit()};
    inputLike_t in;
    tensor_t *x = makeInput(&in, (size_t[]){1, 4, 4}, 3, &inputQ);
    rematWireTable_t *t = initTable(model, 1, MSE, x);
    TEST_ASSERT_EQUAL_size_t(4, t->wires[1].expCapacity);
    rematWireTableBind(t, model, 1, MSE, x);
    bfpQConfig_t *slabQc = rematActHdr(t, 1)->quantization->qConfig;
    TEST_ASSERT_EQUAL_size_t(4, slabQc->numGroups);
    TEST_ASSERT_EQUAL_size_t(4, slabQc->groupSize);

    slabQc->exponents[0] = 3; /* a producer's exponent write */
    initBfpQConfigGroupedInto(8, 8, HALF_AWAY, 2, 8, inputExponents, &inputQc);
    rematWireTableBind(t, model, 1, MSE, x);
    TEST_ASSERT_EQUAL_size_t(2, slabQc->numGroups);
    TEST_ASSERT_EQUAL_size_t(8, slabQc->groupSize);
    TEST_ASSERT_EQUAL_UINT8(127, slabQc->exponents[0]); /* zero state: bias 2^(8-1)-1 */
    TEST_ASSERT_EQUAL_UINT8(127, slabQc->exponents[1]);

    rematWireTableFree(t);
    freeModel(model, 1);
}

/* §12.2 item 8: SYM@12 -> @8 on the input carries qMaxBits 8 onto the Flatten
 * wire, so a stale width cannot slip past the #227 operand guard. */
void testBindCarriesSymQMaxBitsOntoTheFlattenWire(void) {
    symInt32QConfig_t inputQc;
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &inputQc, 12);
    quantization_t inputQ = {.type = SYM_INT32, .qConfig = &inputQc};
    layer_t *model[1] = {flattenLayerInit()};
    inputLike_t in;
    tensor_t *x = makeInput(&in, (size_t[]){1, 2, 3}, 3, &inputQ);
    rematWireTable_t *t = initTable(model, 1, MSE, x);
    rematWireTableBind(t, model, 1, MSE, x);
    symInt32QConfig_t *slabQc = rematActHdr(t, 1)->quantization->qConfig;
    TEST_ASSERT_EQUAL_UINT8(12, slabQc->qMaxBits);
    inputQc.qMaxBits = 8;
    rematWireTableBind(t, model, 1, MSE, x);
    TEST_ASSERT_EQUAL_UINT8(8, slabQc->qMaxBits);
    rematWireTableFree(t);
    freeModel(model, 1);
}

/* RF3: a packed borrowed input re-quantized between calls (8 -> 4 bits) is a
 * config edit, not a key change; the checker sizes ACT 0 from the live input. */
void testBindFollowsTheLivePackedInputBytes(void) {
    float scale = 1.f;
    symQConfig_t inputQc = {
        .scales = &scale, .numGroups = 1, .groupSize = 0, .roundingMode = HALF_AWAY, .qBits = 8};
    quantization_t inputQ = {.type = SYM, .qConfig = &inputQc};
    layer_t *model[1] = {makeRelu(&g_floatQ)};
    inputLike_t in;
    tensor_t *x = makeInput(&in, (size_t[]){1, 8}, 2, &inputQ);
    rematWireTable_t *t = initTable(model, 1, MSE, x);
    TEST_ASSERT_EQUAL_size_t(8, rematWireBytes(t, 0));
    inputQc.qBits = 4;
    rematWireTableBind(t, model, 1, MSE, x);
    TEST_ASSERT_EQUAL_size_t(4, rematWireBytes(t, 0));
    rematWireTableFree(t);
    freeModel(model, 1);
}

/* P9 (table level): ACT 0 is the caller's tensor verbatim -- a table built on
 * sample A binds sample B, and B's header is not written. */
void testBindRunsSampleBOnATableBuiltOnSampleA(void) {
    layer_t *model[1] = {makeRelu(&g_floatQ)};
    inputLike_t inA;
    inputLike_t inB;
    float bytesB[4] = {1.f, 2.f, 3.f, 4.f};
    tensor_t *a = makeInput(&inA, (size_t[]){1, 4}, 2, &g_floatQ);
    tensor_t *b = makeInput(&inB, (size_t[]){1, 4}, 2, &g_floatQ);
    b->data = (uint8_t *)bytesB;
    inputLike_t snapshot = inB;
    rematWireTable_t *t = initTable(model, 1, MSE, a);
    rematWireTableBind(t, model, 1, MSE, a);
    rematWireTableBind(t, model, 1, MSE, b);
    TEST_ASSERT_EQUAL_PTR(b, rematActHdr(t, 0));
    TEST_ASSERT_EQUAL_PTR((uint8_t *)bytesB, b->data);
    TEST_ASSERT_EQUAL_MEMORY(snapshot.dims, inB.dims, sizeof inB.dims);
    TEST_ASSERT_EQUAL_MEMORY(snapshot.order, inB.order, sizeof inB.order);
    TEST_ASSERT_EQUAL_PTR(&g_floatQ, b->quantization);
    rematWireTableFree(t);
    freeModel(model, 1);
}

/* Plan Assumption 29: the bind's per-wire derivation scratch lives inside the
 * table block. LP64 layout arithmetic (not a scan-model pin): table struct 144
 * + 24 records x 40 + gradIdOf 26 + layerType/frozen 24, rounded to 8, + the
 * input key 48 = 1208; the bind scratch 960; the headers 2680. Total 4848.
 * Every host preset is LP64. */
void testHarSlabHoldsTheBindScratch(void) {
    layer_t *model[HAR_N];
    buildHar(model, false);
    inputLike_t in;
    rematWireTable_t *t = initTable(model, HAR_N, CROSS_ENTROPY, makeHarInput(&in));
    TEST_ASSERT_EQUAL_size_t(4848, t->slabBytes);
    const uint8_t *scratch = (const uint8_t *)t->bindScratch;
    TEST_ASSERT_TRUE(scratch >= (const uint8_t *)(t->inputOrder + t->inputRank));
    TEST_ASSERT_TRUE(scratch + t->numWires * sizeof(rematWireFact_t) <=
                     (const uint8_t *)t->wires[1].hdr);
    ASSERT_ALIGNED(t->bindScratch, rematWireFact_t);
    rematWireTableFree(t);
    freeModel(model, HAR_N);
}

void testUnbindClearsTheBorrowedInputOnly(void) {
    layer_t *model[1] = {makeRelu(&g_floatQ)};
    inputLike_t in;
    tensor_t *x = makeInput(&in, (size_t[]){1, 4}, 2, &g_floatQ);
    rematWireTable_t *t = initTable(model, 1, MSE, x);
    rematWireTableBind(t, model, 1, MSE, x);
    tensor_t *act1 = rematActHdr(t, 1);
    rematWireTableUnbind(t);
    TEST_ASSERT_NULL(rematActHdr(t, 0));
    TEST_ASSERT_EQUAL_PTR(act1, rematActHdr(t, 1));
    rematWireTableFree(t);
    freeModel(model, 1);
}

/* ---- the ASan death callback (spec §12.2 item 6, §16.1 item 4e; C6) ---- */

#ifdef ODT_TEST_ASAN
static void overrunAHeapBlockUnderTheCallback(void) {
    odtInstallAsanDeathExit();
    uint8_t *block = reserveMemory(4);
    volatile size_t past = 4;
    block[past] = 1;
}

/* The live-RED run that decides 4e: an ASan report inside a death-test child
 * must surface as exit 86, not as the SIGABRT of abort_on_error=1. */
void testAsanDeathCallbackExitsWithADistinctCode(void) {
    ASSERT_EXITS_WITH(ODT_ASAN_DEATH_EXIT, overrunAHeapBlockUnderTheCallback());
}
#endif

/* ---- the schedule key at bind (spec §3.4-§3.6, §12.2 item 8) ---- */

void testBindExitsOnAChangedModelSize(void) {
    layer_t *model[2] = {makeRelu(&g_floatQ), makeRelu(&g_floatQ)};
    inputLike_t in;
    tensor_t *x = makeInput(&in, (size_t[]){1, 4}, 2, &g_floatQ);
    rematWireTable_t *t = initTable(model, 2, MSE, x);
    ASSERT_EXITS_WITH_OUTPUT(1, "key mismatch on 'modelSize': built 2, live 1",
                             rematWireTableBind(t, model, 1, MSE, x));
    rematWireTableFree(t);
    freeModel(model, 2);
}

void testBindExitsOnAChangedLossType(void) {
    layer_t *model[2] = {makeLinear(4, 3, false), makeSoftmax()};
    inputLike_t in;
    tensor_t *x = makeInput(&in, (size_t[]){1, 4}, 2, &g_floatQ);
    rematWireTable_t *t = initTable(model, 2, CROSS_ENTROPY, x);
    ASSERT_EXITS_WITH_OUTPUT(1, "key mismatch on 'lossType': built 1, live 0",
                             rematWireTableBind(t, model, 2, MSE, x));
    rematWireTableFree(t);
    freeModel(model, 2);
}

void testBindExitsOnALayerTypeSwap(void) {
    layer_t *relu = makeRelu(&g_floatQ);
    layer_t *softmax = makeSoftmax();
    layer_t *model[2] = {makeRelu(&g_floatQ), relu};
    inputLike_t in;
    tensor_t *x = makeInput(&in, (size_t[]){1, 4}, 2, &g_floatQ);
    rematWireTable_t *t = initTable(model, 2, MSE, x);
    model[1] = softmax; /* same shape, same dtype: only the type changes */
    ASSERT_EXITS_WITH_OUTPUT(1, "key mismatch on 'layerType[1]': built 1, live 6",
                             rematWireTableBind(t, model, 2, MSE, x));
    rematWireTableFree(t);
    model[1] = relu;
    freeModel(model, 2);
    freeSoftmaxLayer(softmax);
}

/* Freezing the deepest trainable layer moves deepest (0 -> 3 on HAR). */
void testBindExitsWhenFreezingMovesDeepest(void) {
    layer_t *model[HAR_N];
    buildHar(model, false);
    inputLike_t in;
    tensor_t *x = makeHarInput(&in);
    rematWireTable_t *t = initTable(model, HAR_N, CROSS_ENTROPY, x);
    model[0]->config->conv1d->frozen = true;
    ASSERT_EXITS_WITH_OUTPUT(1, "key mismatch on 'deepest': built 0, live 3",
                             rematWireTableBind(t, model, HAR_N, CROSS_ENTROPY, x));
    model[0]->config->conv1d->frozen = false;
    rematWireTableFree(t);
    freeModel(model, HAR_N);
}

/* Freezing a layer above deepest keeps deepest; the frozen[] key catches it. */
void testBindExitsWhenFreezingALayerAboveDeepest(void) {
    layer_t *model[HAR_N];
    buildHar(model, false);
    inputLike_t in;
    tensor_t *x = makeHarInput(&in);
    rematWireTable_t *t = initTable(model, HAR_N, CROSS_ENTROPY, x);
    model[3]->config->conv1d->frozen = true;
    ASSERT_EXITS_WITH_OUTPUT(1, "key mismatch on 'frozen[3]': built 0, live 1",
                             rematWireTableBind(t, model, HAR_N, CROSS_ENTROPY, x));
    model[3]->config->conv1d->frozen = false;
    rematWireTableFree(t);
    freeModel(model, HAR_N);
}

void testBindExitsOnAChangedInputRank(void) {
    layer_t *model[1] = {makeRelu(&g_floatQ)};
    inputLike_t in;
    inputLike_t in3;
    rematWireTable_t *t = initTable(model, 1, MSE, makeInput(&in, (size_t[]){1, 4}, 2, &g_floatQ));
    tensor_t *x3 = makeInput(&in3, (size_t[]){1, 1, 4}, 3, &g_floatQ);
    ASSERT_EXITS_WITH_OUTPUT(1, "key mismatch on wire ACT 0, field 'rank': built 2, live 3",
                             rematWireTableBind(t, model, 1, MSE, x3));
    rematWireTableFree(t);
    freeModel(model, 1);
}

/* A B change (#152, D17: exact-B key) dies on ACT 0's dims[0], which phase 1
 * step 1 compares before any wire (plan Assumption 22). */
void testBindExitsOnAChangedBatch(void) {
    layer_t *model[1] = {makeRelu(&g_floatQ)};
    inputLike_t in;
    inputLike_t in2;
    rematWireTable_t *t = initTable(model, 1, MSE, makeInput(&in, (size_t[]){1, 4}, 2, &g_floatQ));
    tensor_t *x2 = makeInput(&in2, (size_t[]){2, 4}, 2, &g_floatQ);
    ASSERT_EXITS_WITH_OUTPUT(1, "key mismatch on wire ACT 0, field 'dims[0]': built 1, live 2",
                             rematWireTableBind(t, model, 1, MSE, x2));
    rematWireTableFree(t);
    freeModel(model, 1);
}

void testBindExitsOnAChangedInputOrder(void) {
    layer_t *model[1] = {makeRelu(&g_floatQ)};
    inputLike_t in;
    inputLike_t inT;
    rematWireTable_t *t = initTable(model, 1, MSE, makeInput(&in, (size_t[]){1, 4}, 2, &g_floatQ));
    tensor_t *xT = makeInput(&inT, (size_t[]){1, 4}, 2, &g_floatQ);
    inT.order[0] = 1;
    inT.order[1] = 0;
    ASSERT_EXITS_WITH_OUTPUT(1, "key mismatch on wire ACT 0, field 'order[0]': built 0, live 1",
                             rematWireTableBind(t, model, 1, MSE, xT));
    rematWireTableFree(t);
    freeModel(model, 1);
}

void testBindExitsOnAChangedInputDtype(void) {
    symInt32QConfig_t symQc;
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &symQc, 12);
    quantization_t symQ = {.type = SYM_INT32, .qConfig = &symQc};
    layer_t *model[1] = {makeRelu(&g_floatQ)};
    inputLike_t in;
    inputLike_t inS;
    rematWireTable_t *t = initTable(model, 1, MSE, makeInput(&in, (size_t[]){1, 4}, 2, &g_floatQ));
    tensor_t *xS = makeInput(&inS, (size_t[]){1, 4}, 2, &symQ);
    ASSERT_EXITS_WITH_OUTPUT(1, "key mismatch on wire ACT 0, field 'dtype': built 1, live 2",
                             rematWireTableBind(t, model, 1, MSE, xS));
    rematWireTableFree(t);
    freeModel(model, 1);
}

/* A BFP width edit changes the wire's bytes (8 -> 4 for 8 elements). */
void testBindExitsOnAChangedWireByteCount(void) {
    uint8_t exponent[1];
    bfpQConfig_t tmplQc;
    initBfpQConfigInto(8, 8, HALF_AWAY, exponent, &tmplQc);
    quantization_t tmplQ = {.type = BFP, .qConfig = &tmplQc};
    layer_t *model[1] = {makeQuant(&tmplQ, &g_floatQ)};
    inputLike_t in;
    tensor_t *x = makeInput(&in, (size_t[]){1, 8}, 2, &g_floatQ);
    rematWireTable_t *t = initTable(model, 1, MSE, x);
    tmplQc.mantissaBits = 4;
    ASSERT_EXITS_WITH_OUTPUT(1, "key mismatch on wire ACT 1, field 'bytes': built 8, live 4",
                             rematWireTableBind(t, model, 1, MSE, x));
    rematWireTableFree(t);
    freeModel(model, 1);
}

/* C1 (D54), dtype twin: a byte-neutral FLOAT32 -> SYM_INT32 template edit.
 * FLOAT32 reserved no qConfig, so a write before the check goes through a NULL
 * qConfig: a crash (or, under ASan, exit 86), never the named exit. */
static void bindUnderTheAsanCallback(rematWireTable_t *t, layer_t **model, size_t n,
                                     lossFuncType_t lt, tensor_t *x) {
    odtInstallAsanDeathExit();
    rematWireTableBind(t, model, n, lt, x);
}

void testBindFloatToSymTemplateEditExitsBeforeSlabWrite(void) {
    quantization_t tmplQ = {.type = FLOAT32, .qConfig = NULL};
    symInt32QConfig_t symQc;
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &symQc, 12);
    layer_t *model[1] = {makeRelu(&tmplQ)};
    inputLike_t in;
    tensor_t *x = makeInput(&in, (size_t[]){1, 4}, 2, &g_floatQ);
    rematWireTable_t *t = initTable(model, 1, MSE, x);
    tmplQ.type = SYM_INT32;
    tmplQ.qConfig = &symQc;
    ASSERT_EXITS_WITH_OUTPUT(1, "key mismatch on wire ACT 1, field 'dtype': built 1, live 2",
                             bindUnderTheAsanCallback(t, model, 1, MSE, x));
    rematWireTableFree(t);
    freeModel(model, 1);
}

/* C1 (D54), BFP twin: exactly one BFP wire, per-tensor (expCapacity 1), so its
 * exponent byte is the last object of the table block (plan Assumption 7). A
 * grouped edit derives 4 groups; a write before the check would put 3 bytes
 * past the block end, which ASan reports. */
void testBindGroupedBfpEditExitsBeforeSlabWrite(void) {
    uint8_t tmplExponents[4];
    bfpQConfig_t tmplQc;
    initBfpQConfigInto(8, 8, HALF_AWAY, tmplExponents, &tmplQc);
    quantization_t tmplQ = {.type = BFP, .qConfig = &tmplQc};
    layer_t *model[1] = {makeQuant(&tmplQ, &g_floatQ)};
    inputLike_t in;
    tensor_t *x = makeInput(&in, (size_t[]){1, 8}, 2, &g_floatQ);
    rematWireTable_t *t = initTable(model, 1, MSE, x);
    TEST_ASSERT_EQUAL_size_t(1, t->wires[1].expCapacity);
    initBfpQConfigGroupedInto(8, 8, HALF_AWAY, 4, 2, tmplExponents, &tmplQc);
    ASSERT_EXITS_WITH_OUTPUT(
        1, "key mismatch on wire ACT 1, field 'numGroups': 4 groups exceed expCapacity 1",
        bindUnderTheAsanCallback(t, model, 1, MSE, x));
    rematWireTableFree(t);
    freeModel(model, 1);
}

/* ---- the row SDK (spec §3.10) ---- */

void testWireBindSetsDataCountsBytesAndBumpsBindGen(void) {
    seedFixture_t f;
    buildBfpSeedFixture(&f);
    uint32_t act1[8];
    uint32_t act2[2];
    rematWireBind(f.t, SEED_ACT1, (uint8_t *)act1);
    TEST_ASSERT_EQUAL_PTR((uint8_t *)act1, rematWireHdr(f.t, SEED_ACT1)->data);
    TEST_ASSERT_EQUAL_UINT32(1, f.t->wires[SEED_ACT1].bindGen);
    TEST_ASSERT_EQUAL_size_t(32, f.t->liveBytes);
    rematWireBind(f.t, SEED_ACT2, (uint8_t *)act2);
    TEST_ASSERT_EQUAL_size_t(40, f.t->liveBytes);
    TEST_ASSERT_EQUAL_size_t(40, f.t->observedPeakLiveBytes);
    freeSeedFixture(&f);
}

void testWireReleaseClearsDataAndKeepsThePeak(void) {
    seedFixture_t f;
    buildBfpSeedFixture(&f);
    uint32_t act1[8];
    uint32_t act2[2];
    rematWireBind(f.t, SEED_ACT1, (uint8_t *)act1);
    rematWireBind(f.t, SEED_ACT2, (uint8_t *)act2);
    rematWireRelease(f.t, SEED_ACT1);
    TEST_ASSERT_NULL(rematWireHdr(f.t, SEED_ACT1)->data);
    TEST_ASSERT_EQUAL_size_t(8, f.t->liveBytes);
    TEST_ASSERT_EQUAL_size_t(40, f.t->observedPeakLiveBytes);
    TEST_ASSERT_EQUAL_UINT32(1, f.t->wires[SEED_ACT1].bindGen); /* Release keeps the generation */
    rematWireBind(f.t, SEED_ACT1, (uint8_t *)act1);
    TEST_ASSERT_EQUAL_UINT32(2, f.t->wires[SEED_ACT1].bindGen);
    freeSeedFixture(&f);
}

/* Phase 3 item 4: every bind starts a call -- generations, live bytes and the
 * observed peak restart at 0 (the peak is "over the last call", §3.8). */
void testTableBindResetsBindGenLiveBytesAndThePeak(void) {
    seedFixture_t f;
    buildBfpSeedFixture(&f);
    uint32_t act1[8];
    uint32_t act2[2];
    rematWireBind(f.t, SEED_ACT1, (uint8_t *)act1);
    rematWireBind(f.t, SEED_ACT2, (uint8_t *)act2);
    rematWireRelease(f.t, SEED_ACT1);
    rematWireRelease(f.t, SEED_ACT2);
    rematWireTableBind(f.t, f.model, 2, MSE, f.x);
    for (uint16_t id = 0; id < f.t->numWires; id++) {
        TEST_ASSERT_EQUAL_UINT32(0, f.t->wires[id].bindGen);
    }
    TEST_ASSERT_EQUAL_size_t(0, f.t->liveBytes);
    TEST_ASSERT_EQUAL_size_t(0, f.t->observedPeakLiveBytes);
    freeSeedFixture(&f);
}

/* The seed takes ACT 2's LIVE config fields when its range opens (today's
 * post-forward initGradTensor timing, CalculateGradsSequential.c:88), never
 * its exponents: a fresh zero state. */
void testWireBindDerivesTheSeedFromTheLiveActHeader(void) {
    seedFixture_t f;
    buildBfpSeedFixture(&f);
    uint32_t act2[2];
    uint32_t seed[2];
    rematWireBind(f.t, SEED_ACT2, (uint8_t *)act2);
    bfpQConfig_t *act2Qc = rematWireHdr(f.t, SEED_ACT2)->quantization->qConfig;
    act2Qc->exponents[0] = 5; /* the producer's OUT_WRITE */
    rematWireBind(f.t, SEED_GRAD2, (uint8_t *)seed);
    tensor_t *seedHdr = rematGradHdr(f.t, 2);
    assertDims(seedHdr->shape, (size_t[]){1, 8}, 2);
    TEST_ASSERT_EQUAL_size_t(0, seedHdr->shape->orderOfDimensions[0]);
    TEST_ASSERT_EQUAL_size_t(1, seedHdr->shape->orderOfDimensions[1]);
    TEST_ASSERT_EQUAL_INT(BFP, seedHdr->quantization->type);
    bfpQConfig_t *seedQc = seedHdr->quantization->qConfig;
    TEST_ASSERT_EQUAL_size_t(1, seedQc->numGroups);
    TEST_ASSERT_EQUAL_size_t(0, seedQc->groupSize);
    TEST_ASSERT_EQUAL_UINT8(8, seedQc->mantissaBits);
    TEST_ASSERT_EQUAL_UINT8(127, seedQc->exponents[0]);
    TEST_ASSERT_EQUAL_PTR((uint8_t *)seed, seedHdr->data);
    freeSeedFixture(&f);
}

void testWireBindInheritedSymTakesConfigNotScale(void) {
    symInt32QConfig_t symQc;
    initSymInt32QConfigWithQMaxBits(SR_HALF_AWAY, &symQc, 10);
    quantization_t symQ = {.type = SYM_INT32, .qConfig = &symQc};
    layer_t *model[2] = {makeLinear(2, 4, false), makeQuant(&symQ, &g_floatQ)};
    inputLike_t in;
    tensor_t *x = makeInput(&in, (size_t[]){1, 2}, 2, &g_floatQ);
    rematWireTable_t *t = initTable(model, 2, MSE, x);
    rematWireTableBind(t, model, 2, MSE, x);
    uint32_t act2[4];
    uint32_t seed[4];
    rematWireBind(t, 2, (uint8_t *)act2);
    symInt32QConfig_t *act2Qc = rematActHdr(t, 2)->quantization->qConfig;
    act2Qc->scale = 0.25f;
    rematWireBind(t, rematGradId(t, 2), (uint8_t *)seed);
    symInt32QConfig_t *seedQc = rematGradHdr(t, 2)->quantization->qConfig;
    TEST_ASSERT_EQUAL_UINT8(10, seedQc->qMaxBits);
    TEST_ASSERT_EQUAL_INT(SR_HALF_AWAY, seedQc->roundingMode);
    TEST_ASSERT_EQUAL_FLOAT(1.0f, seedQc->scale);
    rematWireTableFree(t);
    freeModel(model, 2);
}

/* FLOAT32 and SYM_INT32 both charge 4 B/element (wireBytes), so this dtype
 * edit is byte-neutral -- the bytes check cannot catch it, only the dtype
 * check can. Without it, initFloat32Quantization would silently orphan the
 * slab's reserved SYM_INT32 qConfig. */
void testWireBindInheritedGradChecksTheLiveDtypeWhenByteNeutral(void) {
    symInt32QConfig_t symQc;
    initSymInt32QConfigWithQMaxBits(SR_HALF_AWAY, &symQc, 10);
    quantization_t symQ = {.type = SYM_INT32, .qConfig = &symQc};
    layer_t *model[2] = {makeLinear(2, 4, false), makeQuant(&symQ, &g_floatQ)};
    inputLike_t in;
    tensor_t *x = makeInput(&in, (size_t[]){1, 2}, 2, &g_floatQ);
    rematWireTable_t *t = initTable(model, 2, MSE, x);
    rematWireTableBind(t, model, 2, MSE, x);
    uint32_t act2[4];
    uint32_t seed[4];
    rematWireBind(t, 2, (uint8_t *)act2);
    rematActHdr(t, 2)->quantization->type = FLOAT32;
    ASSERT_EXITS_WITH_OUTPUT(1, "rematWireBind: wire GRAD 2 field 'dtype'",
                             rematWireBind(t, rematGradId(t, 2), (uint8_t *)seed));
    rematWireTableFree(t);
    freeModel(model, 2);
}

/* C2 + D54 (table-level twin of PR2's decorator test): a live ACT header whose
 * grouping grew past the seed's slab capacity must exit by name before
 * initBfpQConfigGroupedInto writes 4 exponents into a 1-byte tail at the block
 * end. */
static void bindSeedUnderTheAsanCallback(seedFixture_t *f, uint8_t *bytes) {
    odtInstallAsanDeathExit();
    rematWireBind(f->t, SEED_GRAD2, bytes);
}

void testWireBindInheritedGradChecksCapacityBeforeWrite(void) {
    seedFixture_t f;
    buildBfpSeedFixture(&f);
    uint32_t act2[2];
    uint32_t seed[2];
    rematWireBind(f.t, SEED_ACT2, (uint8_t *)act2);
    bfpQConfig_t *act2Qc = rematWireHdr(f.t, SEED_ACT2)->quantization->qConfig;
    act2Qc->numGroups = 4; /* fields only: no exponent byte is written */
    act2Qc->groupSize = 2;
    ASSERT_EXITS_WITH_OUTPUT(1, "wire GRAD 2 needs 4 BFP exponent groups, above its expCapacity 1",
                             bindSeedUnderTheAsanCallback(&f, (uint8_t *)seed));
    freeSeedFixture(&f);
}

void testWireBindInheritedGradChecksTheLiveDtype(void) {
    seedFixture_t f;
    buildBfpSeedFixture(&f);
    uint32_t act2[2];
    uint32_t seed[2];
    rematWireBind(f.t, SEED_ACT2, (uint8_t *)act2);
    rematWireHdr(f.t, SEED_ACT2)->quantization->type = FLOAT32;
    ASSERT_EXITS_WITH_OUTPUT(1, "rematWireBind: wire GRAD 2 field 'dtype'",
                             rematWireBind(f.t, SEED_GRAD2, (uint8_t *)seed));
    rematWireHdr(f.t, SEED_ACT2)->quantization->type = BFP;
    freeSeedFixture(&f);
}

/* A live source whose payload size changed (BFP m8 -> m16 doubles 8 -> 16 B)
 * would overrun the seed's bytes; the check runs before any slab write. */
void testWireBindInheritedGradChecksTheLiveBytes(void) {
    seedFixture_t f;
    buildBfpSeedFixture(&f);
    uint32_t act2[4];
    uint32_t seed[4];
    rematWireBind(f.t, SEED_ACT2, (uint8_t *)act2);
    bfpQConfig_t *act2Qc = rematWireHdr(f.t, SEED_ACT2)->quantization->qConfig;
    act2Qc->mantissaBits = 16;
    ASSERT_EXITS_WITH_OUTPUT(1, "rematWireBind: wire GRAD 2 field 'bytes'",
                             rematWireBind(f.t, SEED_GRAD2, (uint8_t *)seed));
    act2Qc->mantissaBits = 8;
    freeSeedFixture(&f);
}

void testWireBindInheritedGradChecksTheLiveRank(void) {
    seedFixture_t f;
    buildBfpSeedFixture(&f);
    uint32_t act2[2];
    uint32_t seed[2];
    rematWireBind(f.t, SEED_ACT2, (uint8_t *)act2);
    rematWireHdr(f.t, SEED_ACT2)->shape->numberOfDimensions = 3;
    ASSERT_EXITS_WITH_OUTPUT(1, "rematWireBind: wire GRAD 2 field 'rank'",
                             rematWireBind(f.t, SEED_GRAD2, (uint8_t *)seed));
    rematWireHdr(f.t, SEED_ACT2)->shape->numberOfDimensions = 2;
    freeSeedFixture(&f);
}

/* The test above sets numberOfDimensions past the header's allocated
 * dims[2]/order[2] arrays, so it (coincidentally) also reaches the 'bytes'
 * exit through an out-of-bounds read of slab memory. This variant swaps in
 * fully-backed rank-3 arrays ([1,8] -> [1,8,1], a byte-count-preserving
 * reshape), so only the rank check can catch it. */
void testWireBindInheritedGradChecksTheLiveRankWhenByteNeutral(void) {
    seedFixture_t f;
    buildBfpSeedFixture(&f);
    uint32_t act2[2];
    uint32_t seed[2];
    rematWireBind(f.t, SEED_ACT2, (uint8_t *)act2);
    shape_t *act2Shape = rematWireHdr(f.t, SEED_ACT2)->shape;
    size_t *origDims = act2Shape->dimensions;
    size_t *origOrder = act2Shape->orderOfDimensions;
    size_t dims3[3] = {1, 8, 1};
    size_t order3[3] = {0, 1, 2};
    act2Shape->dimensions = dims3;
    act2Shape->orderOfDimensions = order3;
    act2Shape->numberOfDimensions = 3;
    ASSERT_EXITS_WITH_OUTPUT(1, "rematWireBind: wire GRAD 2 field 'rank'",
                             rematWireBind(f.t, SEED_GRAD2, (uint8_t *)seed));
    act2Shape->dimensions = origDims;
    act2Shape->orderOfDimensions = origOrder;
    act2Shape->numberOfDimensions = 2;
    freeSeedFixture(&f);
}

/* RF1: ACT 0 is the caller's tensor; a row that binds it would overwrite the
 * caller's ->data. */
void testWireBindRefusesTheBorrowedInput(void) {
    seedFixture_t f;
    buildBfpSeedFixture(&f);
    uint32_t bytes[2];
    ASSERT_EXITS_WITH_OUTPUT(1, "wire ACT 0 is the caller's borrowed input; it is never bound",
                             rematWireBind(f.t, 0, (uint8_t *)bytes));
    freeSeedFixture(&f);
}

void testWireReleaseRefusesTheBorrowedInput(void) {
    seedFixture_t f;
    buildBfpSeedFixture(&f);
    ASSERT_EXITS_WITH_OUTPUT(1, "wire ACT 0 is the caller's borrowed input; it is never released",
                             rematWireRelease(f.t, 0));
    freeSeedFixture(&f);
}

/* RF2: unbalanced SDK calls would double-count or wrap liveBytes. */
void testWireBindRefusesAnAlreadyBoundWire(void) {
    seedFixture_t f;
    buildBfpSeedFixture(&f);
    uint32_t act1[8];
    rematWireBind(f.t, SEED_ACT1, (uint8_t *)act1);
    ASSERT_EXITS_WITH_OUTPUT(1, "rematWireBind: wire ACT 1 is already bound",
                             rematWireBind(f.t, SEED_ACT1, (uint8_t *)act1));
    freeSeedFixture(&f);
}

void testWireReleaseRefusesAnUnboundWire(void) {
    seedFixture_t f;
    buildBfpSeedFixture(&f);
    ASSERT_EXITS_WITH_OUTPUT(1, "rematWireRelease: wire ACT 1 is not bound",
                             rematWireRelease(f.t, SEED_ACT1));
    freeSeedFixture(&f);
}

/* I2: REMAT_NONE is handed out by rematGradId and by the walk functions, and
 * an unchecked bind/release would write or subtract through a garbage
 * header, corrupting memory silently on an MCU. */
void testWireBindRefusesAWireIdOutOfRange(void) {
    seedFixture_t f;
    buildBfpSeedFixture(&f);
    uint32_t bytes[2];
    ASSERT_EXITS_WITH_OUTPUT(1, "rematWireBind: wire id 65535 out of range",
                             rematWireBind(f.t, REMAT_NONE, (uint8_t *)bytes));
    freeSeedFixture(&f);
}

void testWireReleaseRefusesAWireIdOutOfRange(void) {
    seedFixture_t f;
    buildBfpSeedFixture(&f);
    ASSERT_EXITS_WITH_OUTPUT(1, "rematWireRelease: wire id 65535 out of range",
                             rematWireRelease(f.t, REMAT_NONE));
    freeSeedFixture(&f);
}

/* m2: bytes == NULL would still increment liveBytes and bindGen, and under
 * ODT_REMAT_VERIFY poison a NULL pointer. */
void testWireBindRefusesNullBytes(void) {
    seedFixture_t f;
    buildBfpSeedFixture(&f);
    ASSERT_EXITS_WITH_OUTPUT(1, "rematWireBind: bytes is NULL",
                             rematWireBind(f.t, SEED_ACT1, NULL));
    freeSeedFixture(&f);
}

/* rematWireTableBind resets liveBytes to 0 but never clears a still-bound
 * wire's ->data (spec: only Bind/Release write it), so a wire left bound
 * across a rebind reads as still-bound. Releasing it then would subtract from
 * a liveBytes that no longer reflects it: checked, so it exits by name
 * instead of wrapping size_t. */
void testWireReleaseExitsWhenLiveBytesWouldUnderflow(void) {
    seedFixture_t f;
    buildBfpSeedFixture(&f);
    uint32_t act1[8];
    rematWireBind(f.t, SEED_ACT1, (uint8_t *)act1);
    rematWireTableBind(f.t, f.model, 2, MSE, f.x);
    ASSERT_EXITS_WITH_OUTPUT(1, "rematWireRelease: wire ACT 1 live bytes would underflow",
                             rematWireRelease(f.t, SEED_ACT1));
    freeSeedFixture(&f);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testBackwardRangeMseRunsFromLastLayerToDeepest);
    RUN_TEST(testBackwardRangeCrossEntropySkipsTheLastLayerPositionally);
    RUN_TEST(testBackwardRangeTruncatesAtDeepestTrainable);
    RUN_TEST(testBackwardRangeAllFrozenReturnsModelSize);
    RUN_TEST(testBackwardRangeSingleLayerUnderCrossEntropyIsMinusOne);
    RUN_TEST(testBfpWireGroupingMatchesTheDriverRule);
    RUN_TEST(testBfpWireGroupingExitsNamingTheWireOnAnIndivisibleGroupSize);
    RUN_TEST(testHarTableNumbersWiresInProductionOrder);
    RUN_TEST(testHarTableRecordsBytesRanksAndKey);
    RUN_TEST(testSharedGroupedBfpTemplateGroupsPerWire);
#ifdef ODT_MEM_PROFILE
    RUN_TEST(testTableInitReservesOneBlockOfSlabBytesAndFreeReturnsIt);
#endif
    RUN_TEST(testTableFreeReleasesExactlyTheOneBlock);
    RUN_TEST(testTableFreeIsNullSafe);
    RUN_TEST(testSlabObjectsAlignedAndExponentsAtTheTail);
    RUN_TEST(testAccessorsReadTheTableAndHeadersAreLinked);
    RUN_TEST(testTableInitExitsOnAnEmptyModel);
    RUN_TEST(testTableInitExitsWhenWireIdsWouldReachRematNone);
    RUN_TEST(testTableInitExitsOnAnInputRankAboveTheRankField);
    RUN_TEST(testTableInitExitsOnAnUnsupportedWireDtype);
    RUN_TEST(testTableInitExitsOnAZeroByteWire);
    RUN_TEST(testTableInitAcceptsAPackedBorrowedInputAndSizesItExactly);
    RUN_TEST(testTableInitExitsOnAnUnknownInputQtype);
    RUN_TEST(testTableInitExitsOnAByteCountOverflowBeforeReservingTheTable);
    RUN_TEST(testTableInitExitsOnASlabSizeOverflow);
    RUN_TEST(testTableInitExitsOnATotalWireBytesOverflow);
    RUN_TEST(testBindWritesHarHeadersAndPointsAct0AtTheInput);
    RUN_TEST(testBindCopiesForwardOrderAndGivesGradsIdentityOrder);
#ifdef ODT_MEM_PROFILE
    RUN_TEST(testBindAllocatesNothing);
#endif
    RUN_TEST(testBindRederivesSymQMaxBitsAndResetsScale);
    RUN_TEST(testBindAfterDeserializeModel);
    RUN_TEST(testBindRederivesRoundingMode);
    RUN_TEST(testBindRederivesFlattenBfpGroupingFromTheLiveInput);
    RUN_TEST(testBindCarriesSymQMaxBitsOntoTheFlattenWire);
    RUN_TEST(testBindFollowsTheLivePackedInputBytes);
    RUN_TEST(testBindRunsSampleBOnATableBuiltOnSampleA);
    RUN_TEST(testHarSlabHoldsTheBindScratch);
    RUN_TEST(testUnbindClearsTheBorrowedInputOnly);
#ifdef ODT_TEST_ASAN
    RUN_TEST(testAsanDeathCallbackExitsWithADistinctCode);
#endif
    RUN_TEST(testBindExitsOnAChangedModelSize);
    RUN_TEST(testBindExitsOnAChangedLossType);
    RUN_TEST(testBindExitsOnALayerTypeSwap);
    RUN_TEST(testBindExitsWhenFreezingMovesDeepest);
    RUN_TEST(testBindExitsWhenFreezingALayerAboveDeepest);
    RUN_TEST(testBindExitsOnAChangedInputRank);
    RUN_TEST(testBindExitsOnAChangedBatch);
    RUN_TEST(testBindExitsOnAChangedInputOrder);
    RUN_TEST(testBindExitsOnAChangedInputDtype);
    RUN_TEST(testBindExitsOnAChangedWireByteCount);
    RUN_TEST(testBindFloatToSymTemplateEditExitsBeforeSlabWrite);
    RUN_TEST(testBindGroupedBfpEditExitsBeforeSlabWrite);
    RUN_TEST(testWireBindSetsDataCountsBytesAndBumpsBindGen);
    RUN_TEST(testWireReleaseClearsDataAndKeepsThePeak);
    RUN_TEST(testTableBindResetsBindGenLiveBytesAndThePeak);
    RUN_TEST(testWireBindDerivesTheSeedFromTheLiveActHeader);
    RUN_TEST(testWireBindInheritedSymTakesConfigNotScale);
    RUN_TEST(testWireBindInheritedGradChecksTheLiveDtypeWhenByteNeutral);
    RUN_TEST(testWireBindInheritedGradChecksCapacityBeforeWrite);
    RUN_TEST(testWireBindInheritedGradChecksTheLiveDtype);
    RUN_TEST(testWireBindInheritedGradChecksTheLiveBytes);
    RUN_TEST(testWireBindInheritedGradChecksTheLiveRank);
    RUN_TEST(testWireBindInheritedGradChecksTheLiveRankWhenByteNeutral);
    RUN_TEST(testWireBindRefusesTheBorrowedInput);
    RUN_TEST(testWireReleaseRefusesTheBorrowedInput);
    RUN_TEST(testWireBindRefusesAnAlreadyBoundWire);
    RUN_TEST(testWireReleaseRefusesAnUnboundWire);
    RUN_TEST(testWireBindRefusesAWireIdOutOfRange);
    RUN_TEST(testWireReleaseRefusesAWireIdOutOfRange);
    RUN_TEST(testWireBindRefusesNullBytes);
    RUN_TEST(testWireReleaseExitsWhenLiveBytesWouldUnderflow);
    return UNITY_END();
}
