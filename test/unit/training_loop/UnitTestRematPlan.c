#define SOURCE_FILE "UNIT_TEST_REMAT_PLAN"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
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
#include "RematPlan.h"
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
    return UNITY_END();
}
