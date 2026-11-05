#define SOURCE_FILE "UNIT_TEST_QUANTIZATION_LAYER"

#include <math.h>
#include <stdint.h>
#include <string.h>

#include "DeathTest.h"
#include "KeyedDrawCheck.h"
#include "Layer.h"
#include "LayerExec.h"
#include "LayerQuant.h"
#include "QuantLayerApi.h"
#include "Quantization.h"
#include "QuantizationApi.h"
#include "QuantizationLayer.h"
#include "Rounding.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "TensorConversion.h"
#include "unity.h"

#include "BfpScatterRef.h"

void setUp(void) {}
void tearDown(void) {}

/* 2D tensor builders (UnitTestLayerNormIntegration pattern). shape_t backing
 * arrays come from reserveMemory because shape_t.dimensions is size_t*. */
static tensor_t *buildSym2D(size_t rows, size_t cols) {
    size_t *dims = reserveMemory(2 * sizeof(size_t));
    dims[0] = rows;
    dims[1] = cols;
    size_t *order = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 2, order);
    return initTensor(shape, quantizationInitSymInt32(HALF_AWAY), NULL);
}

static tensor_t *buildFloat2D(size_t rows, size_t cols) {
    size_t *dims = reserveMemory(2 * sizeof(size_t));
    dims[0] = rows;
    dims[1] = cols;
    size_t *order = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 2, order);
    return initTensor(shape, quantizationInitFloat(), NULL);
}

static tensor_t *buildAsym2D(size_t rows, size_t cols, uint8_t qBits) {
    size_t *dims = reserveMemory(2 * sizeof(size_t));
    dims[0] = rows;
    dims[1] = cols;
    size_t *order = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 2, order);
    return initTensor(shape, quantizationInitAsym(qBits, HALF_AWAY), NULL);
}

static tensor_t *buildBool2D(size_t rows, size_t cols) {
    size_t *dims = reserveMemory(2 * sizeof(size_t));
    dims[0] = rows;
    dims[1] = cols;
    size_t *order = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 2, order);
    return initTensor(shape, quantizationInitBool(), NULL);
}

void testQuantizationForwardSymToSymMatchesDirectRequant(void) {
    /* Linear-SYM-style accumulator-range input: |mantissa| >> 32767. */
    const int32_t mantissas[6] = {1500000, -730000, 42, 0, 999999, -1500000};
    tensor_t *input = buildSym2D(2, 3);
    memcpy(input->data, mantissas, sizeof(mantissas));
    ((symInt32QConfig_t *)input->quantization->qConfig)->scale = 1e-4f;

    tensor_t *expected = buildSym2D(2, 3);
    requantSymInt32Tensor(input, expected, NULL);

    /* Fixture sanity: the requant must actually change mantissas + scale,
     * otherwise the differential assert below is vacuous (scale-freeze class). */
    TEST_ASSERT_TRUE(((int32_t *)expected->data)[0] != 1500000);
    TEST_ASSERT_TRUE(((symInt32QConfig_t *)expected->quantization->qConfig)->scale != 1e-4f);

    tensor_t *actual = buildSym2D(2, 3);
    layer_t qLayer = {.type = QUANTIZATION, .config = NULL}; /* kernel reads only tensors */
    quantizationForward(&qLayer, input, actual, NULL);

    int32_t expectedMantissas[6];
    int32_t actualMantissas[6];
    memcpy(expectedMantissas, expected->data, sizeof(expectedMantissas));
    memcpy(actualMantissas, actual->data, sizeof(actualMantissas));
    float expectedScale = ((symInt32QConfig_t *)expected->quantization->qConfig)->scale;
    float actualScale = ((symInt32QConfig_t *)actual->quantization->qConfig)->scale;

    freeTensor(actual);
    freeTensor(expected);
    freeTensor(input);

    TEST_ASSERT_EQUAL_INT32_ARRAY(expectedMantissas, actualMantissas, 6);
    TEST_ASSERT_EQUAL_FLOAT(expectedScale, actualScale);
}

void testQuantizationForwardSymToFloatMatchesConvertTensor(void) {
    const int32_t mantissas[4] = {12000, -32767, 5, 32767};
    tensor_t *input = buildSym2D(2, 2);
    memcpy(input->data, mantissas, sizeof(mantissas));
    ((symInt32QConfig_t *)input->quantization->qConfig)->scale = 0.5f;

    tensor_t *expected = buildFloat2D(2, 2);
    convertTensor(input, expected, NULL);

    tensor_t *actual = buildFloat2D(2, 2);
    layer_t qLayer = {.type = QUANTIZATION, .config = NULL};
    quantizationForward(&qLayer, input, actual, NULL);

    float expectedVals[4];
    float actualVals[4];
    memcpy(expectedVals, expected->data, sizeof(expectedVals));
    memcpy(actualVals, actual->data, sizeof(actualVals));

    freeTensor(actual);
    freeTensor(expected);
    freeTensor(input);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expectedVals, actualVals, 4);
    TEST_ASSERT_EQUAL_FLOAT(6000.0f, actualVals[0]); /* anti-vacuity: 12000 * 0.5 */
}

void testQuantLayerConvertsAsymToSymInt32MatchesConvertTensor(void) {
    /* #266 regression, strengthened per the task-3 brief's contingency: a
     * plain float<->ASYM round trip already passes under the Task-2 shim
     * (arithmeticFromQuantization's storage-only fallback to ARITH_FLOAT32
     * happens to reconstruct the same float detour that convertFloatTensorTo*
     * /convertAsymTensorToFloatTensor already do). ASYM->SYM_INT32 is
     * genuinely broken by the shim: convertAsymTensorToSymInt32Tensor
     * preserves the ASYM tensor's OWN scale exactly (mantissa = code -
     * zeroPoint under the PR4 code domain, scale unchanged - a lossless
     * code shift). The shim instead
     * routes ASYM through a FLOAT32 detour (arithmeticFromQuantization(ASYM)
     * falls back to ARITH_FLOAT32), landing on convertFloatTensorToSymInt32Tensor,
     * which derives a FRESH absmax-based scale - a different mantissa AND a
     * different scale than the direct conversionMatrix path. */
    tensor_t *input = buildAsym2D(2, 2, 8);
    asymQConfig_t *inputAsymQC = input->quantization->qConfig;
    /* code-domain re-pin (group-quant PR4, D6): old value-domain zp -10 becomes +10;
     * mantissa image code - 10 reproduces the same integers. */
    inputAsymQC->scales[0] = 0.25f;
    inputAsymQC->zeroPoints[0] = 10;
    const uint8_t codes[4] = {0, 40, 80, 255};
    memcpy(input->data, codes, sizeof(codes));

    tensor_t *expected = buildSym2D(2, 2);
    convertTensor(input, expected, NULL);

    /* Fixture sanity: the direct conversion must preserve the ASYM scale
     * verbatim (not re-derive one), otherwise the differential assert below
     * is vacuous. */
    TEST_ASSERT_EQUAL_FLOAT(0.25f, ((symInt32QConfig_t *)expected->quantization->qConfig)->scale);

    tensor_t *actual = buildSym2D(2, 2);
    layer_t qLayer = {.type = QUANTIZATION, .config = NULL};
    quantizationForward(&qLayer, input, actual, NULL);

    int32_t expectedMantissas[4];
    int32_t actualMantissas[4];
    memcpy(expectedMantissas, expected->data, sizeof(expectedMantissas));
    memcpy(actualMantissas, actual->data, sizeof(actualMantissas));
    float expectedScale = ((symInt32QConfig_t *)expected->quantization->qConfig)->scale;
    float actualScale = ((symInt32QConfig_t *)actual->quantization->qConfig)->scale;

    freeTensor(actual);
    freeTensor(expected);
    freeTensor(input);

    TEST_ASSERT_EQUAL_INT32_ARRAY(expectedMantissas, actualMantissas, 4);
    TEST_ASSERT_EQUAL_FLOAT(expectedScale, actualScale);
}

void testQuantLayerAbortsOnUnpopulatedConversionCell(void) {
    /* BOOL -> ASYM has no real conversionMatrix cell: every entry in the BOOL
     * row (other than BOOL->BOOL, which never reaches the matrix - convertTensor's
     * same-type branch short-circuits first) is unsupportedConversionTypes,
     * which PRINT_ERRORs and exit(1)s (verified: the BOOL row's designated
     * initializer in src/tensor/TensorConversion.c:511-515 assigns
     * unsupportedConversionTypes to every non-BOOL column). */
    tensor_t *input = buildBool2D(1, 4);
    tensor_t *output = buildAsym2D(1, 4, 8);
    layer_t qLayer = {.type = QUANTIZATION, .config = NULL};

    ASSERT_EXITS_WITH_FAILURE(quantizationForward(&qLayer, input, output, NULL));

    freeTensor(output);
    freeTensor(input);
}

void testQuantizationBackwardRequantsDyMantissasLikeDirectRequant(void) {
    const int32_t dyMantissas[6] = {800000, -123456, 7, -800000, 65536, 0};
    tensor_t *loss = buildSym2D(2, 3);
    memcpy(loss->data, dyMantissas, sizeof(dyMantissas));
    ((symInt32QConfig_t *)loss->quantization->qConfig)->scale = 2e-5f;

    tensor_t *expected = buildSym2D(2, 3);
    requantSymInt32Tensor(loss, expected, NULL);

    tensor_t *forwardInput = buildSym2D(2, 3); /* unused: straight-through backward */
    tensor_t *propLoss = buildSym2D(2, 3);
    layer_t qLayer = {.type = QUANTIZATION, .config = NULL};
    quantizationBackward(&qLayer, forwardInput, loss, propLoss, NULL);

    int32_t expectedMantissas[6];
    int32_t actualMantissas[6];
    memcpy(expectedMantissas, expected->data, sizeof(expectedMantissas));
    memcpy(actualMantissas, propLoss->data, sizeof(actualMantissas));
    float expectedScale = ((symInt32QConfig_t *)expected->quantization->qConfig)->scale;
    float actualScale = ((symInt32QConfig_t *)propLoss->quantization->qConfig)->scale;

    freeTensor(propLoss);
    freeTensor(forwardInput);
    freeTensor(expected);
    freeTensor(loss);

    TEST_ASSERT_EQUAL_INT32_ARRAY(expectedMantissas, actualMantissas, 6);
    TEST_ASSERT_EQUAL_FLOAT(expectedScale, actualScale);
}

/* ---- BFP -> BFP backward: the integer re-block (arithmetic-bfp.md section 14) ----
 * The input layer's backward is straight-through. Between two BFP grids it is
 * integer: the loss codes become a BFP_RAW on the TARGET geometry (each code
 * under its own group's exponent, the raw exponent per target group from the
 * data, bfpRefRawFromRows), re-blocked by the funnel's epilogue in integers
 * (bfpRefReblockRaw) by the target template's rounding mode; under SR one
 * draw per element in element order from the op's RNG_TAG_OUT stream (op key
 * QUANTIZATION_OP_PROP_LOSS = 0 under RNG_TAG_BACKWARD). */

#define QL_BFP_N 24

static tensor_t *buildBfp2DWithCodes(size_t n, uint8_t m, roundingMode_t rm, size_t numGroups,
                                     size_t groupSize, const int32_t *codes, const uint8_t *exps) {
    size_t *dims = reserveMemory(2 * sizeof(size_t));
    dims[0] = 1;
    dims[1] = n;
    size_t *order = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 2, order);
    quantization_t *q = numGroups > 1 ? quantizationInitBfpGrouped(m, 8, rm, numGroups, groupSize)
                                      : quantizationInitBfp(m, 8, rm);
    tensor_t *t = initTensor(shape, q, NULL);
    if (codes != NULL) {
        int32_t tmp[QL_BFP_N];
        memcpy(tmp, codes, n * sizeof(int32_t));
        byteConversion((uint8_t *)tmp, 32, t->data, m, n);
        memcpy(((bfpQConfig_t *)q->qConfig)->exponents, exps, numGroups);
    }
    return t;
}

static void qlBfpBackwardCase(size_t srcGroups, size_t srcGroupSize, size_t dstGroups,
                              size_t dstGroupSize, uint8_t dstM, roundingMode_t rm, uint64_t seed) {
    int32_t codes[QL_BFP_N];
    uint8_t exps[QL_BFP_N];
    uint32_t s = 99u + (uint32_t)seed;
    for (size_t i = 0; i < QL_BFP_N; i++) {
        s = s * 1664525u + 1013904223u;
        codes[i] = (int32_t)((s >> 8) % 256u) - 128;
    }
    codes[5] = 0;
    for (size_t g = 0; g < srcGroups; g++) {
        s = s * 1664525u + 1013904223u;
        exps[g] = (uint8_t)(127 - 10 + (int)((s >> 8) % 21u));
    }
    tensor_t *loss =
        buildBfp2DWithCodes(QL_BFP_N, 8, HALF_AWAY, srcGroups, srcGroupSize, codes, exps);
    tensor_t *propLoss =
        buildBfp2DWithCodes(QL_BFP_N, dstM, rm, dstGroups, dstGroupSize, NULL, NULL);
    layer_t qLayer = {.type = QUANTIZATION, .config = NULL};
    layerExec_t exec = {.key = rngKeyFromSeed(seed)};
    quantizationBackward(&qLayer, NULL, loss, propLoss, rm == SR_HALF_AWAY ? &exec : NULL);

    int64_t sum[QL_BFP_N];
    int eAcc[QL_BFP_N];
    bfpRefOperand_t src = {codes, exps, srcGroupSize, 8, 8};
    for (size_t i = 0; i < QL_BFP_N; i++) {
        sum[i] = codes[i];
        eAcc[i] = bfpRefExp(&src, i);
    }
    int32_t mant[QL_BFP_N];
    int16_t rawE[QL_BFP_N];
    bfpRefStats_t st = {.fitsInt32 = true};
    bfpRefRawFromRows(sum, eAcc, QL_BFP_N, dstGroupSize, mant, rawE, &st);
    TEST_ASSERT_EQUAL_size_t(0, st.realigned);
    rngStream_t out = rngOpen(
        rngFold(rngFold(rngFold(exec.key, RNG_TAG_BACKWARD, 0u), RNG_TAG_OP, 0u), RNG_TAG_OUT, 0u));
    int32_t refCodes[QL_BFP_N];
    uint8_t refStored[QL_BFP_N];
    bfpRefReblockRaw(mant, rawE, QL_BFP_N, dstGroupSize, dstM, 8, rm == SR_HALF_AWAY ? &out : NULL,
                     refCodes, refStored);
    int32_t got[QL_BFP_N];
    unpackSignExtend(propLoss->data, dstM, 0, got, QL_BFP_N);
    TEST_ASSERT_EQUAL_INT32_ARRAY(refCodes, got, QL_BFP_N);
    TEST_ASSERT_EQUAL_UINT8_ARRAY(
        refStored, ((bfpQConfig_t *)propLoss->quantization->qConfig)->exponents, dstGroups);
    if (rm == HALF_AWAY) {
        /* exact regime: the same codes as the float requant cell */
        tensor_t *viaFloat =
            buildBfp2DWithCodes(QL_BFP_N, dstM, rm, dstGroups, dstGroupSize, NULL, NULL);
        requantBfpTensor(loss, viaFloat, NULL);
        int32_t fl[QL_BFP_N];
        unpackSignExtend(viaFloat->data, dstM, 0, fl, QL_BFP_N);
        TEST_ASSERT_EQUAL_INT32_ARRAY(fl, got, QL_BFP_N);
        freeTensor(viaFloat);
    }
    freeTensor(propLoss);
    freeTensor(loss);
}

void testQuantizationBackwardBfpToBfpReblocksInIntegers(void) {
    qlBfpBackwardCase(6, 4, 4, 6, 6, HALF_AWAY, 0u);    /* groups straddle */
    qlBfpBackwardCase(6, 4, 1, 0, 8, HALF_AWAY, 0u);    /* into per tensor */
    qlBfpBackwardCase(1, 0, 3, 8, 5, HALF_AWAY, 0u);    /* from per tensor */
    qlBfpBackwardCase(6, 4, 4, 6, 6, SR_HALF_AWAY, 7u); /* the op's OUT stream */
    qlBfpBackwardCase(6, 4, 2, 12, 8, SR_HALF_AWAY, 8u);
}

void testQuantizationCalcOutputShapeIdentityIncludingPermutedOrder(void) {
    size_t inDims[2] = {2, 3};
    size_t inOrder[2] = {1, 0}; /* transposed view */
    shape_t inputShape = {
        .dimensions = inDims, .numberOfDimensions = 2, .orderOfDimensions = inOrder};

    size_t outDims[2] = {0, 0};
    size_t outOrder[2] = {0, 0};
    shape_t outputShape = {
        .dimensions = outDims, .numberOfDimensions = 0, .orderOfDimensions = outOrder};

    layer_t qLayer = {.type = QUANTIZATION, .config = NULL};
    quantizationCalcOutputShape(&qLayer, &inputShape, &outputShape);

    TEST_ASSERT_EQUAL_UINT(2, outputShape.numberOfDimensions);
    TEST_ASSERT_EQUAL_UINT(2, outputShape.dimensions[0]);
    TEST_ASSERT_EQUAL_UINT(3, outputShape.dimensions[1]);
    TEST_ASSERT_EQUAL_UINT(1, outputShape.orderOfDimensions[0]); /* permutation preserved */
    TEST_ASSERT_EQUAL_UINT(0, outputShape.orderOfDimensions[1]);
}

void testLayerFunctionsVtableHasQuantizationRow(void) {
    TEST_ASSERT_EQUAL_PTR((void *)quantizationForward,
                          (void *)layerFunctions[QUANTIZATION].forward);
    TEST_ASSERT_EQUAL_PTR((void *)quantizationBackward,
                          (void *)layerFunctions[QUANTIZATION].backward);
    TEST_ASSERT_EQUAL_PTR((void *)quantizationCalcOutputShape,
                          (void *)layerFunctions[QUANTIZATION].calcOutputShape);
}

void testQuantizationConfigUnionMemberRoundTrip(void) {
    quantization_t *fq = quantizationInitSymInt32(HALF_AWAY);
    quantization_t *bq = quantizationInitSymInt32(HALF_AWAY);
    quantizationConfig_t cfg = {.outputQ = fq, .propLossQ = bq, .ownsQuantizations = false};
    layerConfig_t lc = {.quantization = &cfg};
    layer_t layer;
    initLayer(&layer, QUANTIZATION, &lc);

    TEST_ASSERT_EQUAL_INT(QUANTIZATION, layer.type);
    TEST_ASSERT_EQUAL_PTR(fq, layer.config->quantization->outputQ);
    TEST_ASSERT_EQUAL_PTR(bq, layer.config->quantization->propLossQ);

    freeQuantization(bq);
    freeQuantization(fq);
}

void testQuantLayerInitBorrowingStoresQuantPointersVerbatim(void) {
    quantization_t *fq = quantizationInitSymInt32(HALF_AWAY);
    quantization_t *bq = quantizationInitSymInt32(HALF_AWAY);
    layerQuant_t lq = {.outputQ = fq, .propLossQ = bq};

    layer_t *layer = quantLayerInit(&lq);

    TEST_ASSERT_NOT_NULL(layer);
    TEST_ASSERT_EQUAL_INT(QUANTIZATION, layer->type);
    quantizationConfig_t *cfg = layer->config->quantization;
    TEST_ASSERT_NOT_NULL(cfg);
    TEST_ASSERT_FALSE(cfg->ownsQuantizations);
    TEST_ASSERT_EQUAL_PTR(fq, cfg->outputQ);
    TEST_ASSERT_EQUAL_PTR(bq, cfg->propLossQ);

    freeQuantLayer(layer);
    /* Borrowing: caller-owned quantizations survive the layer teardown. */
    TEST_ASSERT_EQUAL_INT(SYM_INT32, fq->type);
    freeQuantization(bq);
    freeQuantization(fq);
}

void testQuantLayerInitOwningDeepCopiesQuantizations(void) {
    quantization_t *q = quantizationInitSymInt32(HALF_AWAY);
    layerQuant_t lq = {.outputQ = q, .propLossQ = q};

    layer_t *layer = quantLayerInitOwning(&lq);

    quantizationConfig_t *cfg = layer->config->quantization;
    TEST_ASSERT_TRUE(cfg->ownsQuantizations);
    TEST_ASSERT_NOT_EQUAL(q, cfg->outputQ);
    TEST_ASSERT_NOT_EQUAL(q, cfg->propLossQ);
    TEST_ASSERT_NOT_EQUAL(cfg->outputQ, cfg->propLossQ); /* two separate copies */
    TEST_ASSERT_EQUAL_INT(SYM_INT32, cfg->outputQ->type);
    TEST_ASSERT_EQUAL_INT(SYM_INT32, cfg->propLossQ->type);

    freeQuantLayer(layer);
    freeQuantization(q);
}

void testQuantLayerInitOwningFreesAllAllocationsWithoutLeak(void) {
    /* Build + free 5 layers — leak-checked by the CI valgrind/LSan sweep
     * (UnitTestLinear testLinearLayerInitOwningFreesAllAllocationsWithoutLeak
     * precedent). */
    for (int i = 0; i < 5; i++) {
        quantization_t *q = quantizationInitSymInt32(HALF_AWAY);
        layerQuant_t lq = {.outputQ = q, .propLossQ = q};
        layer_t *layer = quantLayerInitOwning(&lq);
        freeQuantLayer(layer);
        freeQuantization(q);
    }
    TEST_PASS();
}

/* ---- the layer converts with its CONVERT stream ---- */

/* Python, tags as numbers (FORWARD 4, BACKWARD 5, CONVERT 12):
 *   uv run python -c "from examples._shared.keyed_rng import (fold, key_from_seed as kfs,
 *   open_stream as o); k = kfs(0x1a7e); print([hex(o(fold(fold(k, p, 0), 12, 0)))
 *   for p in (4, 5)])" */
#define QL_EXEC_TEST_SEED 0x1a7eu

static conversionFunction_t qlSavedCell;
static size_t qlCellCalls;
static bool qlCellHasStream;
static uint32_t qlCellState;

static void qlRecordingCell(tensor_t *in, tensor_t *out, rngStream_t *rs) {
    qlCellCalls++;
    qlCellHasStream = rs != NULL;
    qlCellState = (rs != NULL) ? rs->state : 0u;
    qlSavedCell(in, out, rs);
}

static void qlAssertConvertStream(uint32_t phaseTag, uint32_t pythonState) {
    layerExec_t exec = {.key = rngKeyFromSeed(QL_EXEC_TEST_SEED)};
    rngKey_t phase;
    TEST_ASSERT_EQUAL_UINT32(1u, (uint32_t)qlCellCalls);
    TEST_ASSERT_TRUE(qlCellHasStream);
    rngKey_t convertKey = rngFold(*layerExecPhaseKey(&exec, phaseTag, &phase), RNG_TAG_CONVERT, 0u);
    TEST_ASSERT_EQUAL_HEX32(rngOpen(convertKey).state, qlCellState);
    TEST_ASSERT_EQUAL_HEX32(pythonState, qlCellState);
}

static void qlRunForward(const layerExec_t *exec) {
    tensor_t *input = buildFloat2D(2, 2);
    tensor_t *output = buildSym2D(2, 2);
    layer_t qLayer = {.type = QUANTIZATION, .config = NULL};
    qlSavedCell = conversionMatrix[FLOAT32][SYM_INT32];
    conversionMatrix[FLOAT32][SYM_INT32] = qlRecordingCell;
    qlCellCalls = 0;

    quantizationForward(&qLayer, input, output, exec);

    conversionMatrix[FLOAT32][SYM_INT32] = qlSavedCell;
    freeTensor(output);
    freeTensor(input);
}

/* FLOAT32 -> SYM_INT32 goes through [FLOAT32][SYM_INT32] with (FORWARD, CONVERT). */
void testQuantizationForwardConvertsWithItsForwardConvertStream(void) {
    layerExec_t exec = {.key = rngKeyFromSeed(QL_EXEC_TEST_SEED)};

    qlRunForward(&exec);

    qlAssertConvertStream(RNG_TAG_FORWARD, 0x1ccf7b1cu);
}

/* The backward requantizes the dy through the SYM_INT32 diagonal with
 * (BACKWARD, CONVERT): forward and backward never share a stream. */
void testQuantizationBackwardConvertsWithItsBackwardConvertStream(void) {
    layerExec_t exec = {.key = rngKeyFromSeed(QL_EXEC_TEST_SEED)};
    tensor_t *loss = buildSym2D(2, 2);
    tensor_t *forwardInput = buildSym2D(2, 2);
    tensor_t *propLoss = buildSym2D(2, 2);
    layer_t qLayer = {.type = QUANTIZATION, .config = NULL};
    qlSavedCell = conversionMatrix[SYM_INT32][SYM_INT32];
    conversionMatrix[SYM_INT32][SYM_INT32] = qlRecordingCell;
    qlCellCalls = 0;

    quantizationBackward(&qLayer, forwardInput, loss, propLoss, &exec);

    conversionMatrix[SYM_INT32][SYM_INT32] = qlSavedCell;
    freeTensor(propLoss);
    freeTensor(forwardInput);
    freeTensor(loss);
    qlAssertConvertStream(RNG_TAG_BACKWARD, 0x143ac223u);
}

void testQuantizationForwardWithoutAContextConvertsWithoutAStream(void) {
    qlRunForward(NULL);

    TEST_ASSERT_EQUAL_UINT32(1u, (uint32_t)qlCellCalls);
    TEST_ASSERT_FALSE(qlCellHasStream);
}

/* The state from the Python mirror, tags as numbers (FORWARD 4, CONVERT 12):
 *   uv run python -c "from examples._shared.keyed_rng import fold, key_from_seed as kfs,
 *   open_stream as o; print(hex(o(fold(fold(kfs(0x9a01), 4, 0), 12, 0))))" */
#define QL_KEYED_SEED 0x9a01u
#define QL_KEYED_CONVERT_STATE 0xbba72004u

/* A FLOAT32 [2, 8] requantized SR into an 8-bit SYM_INT32 output: 16
 * elements, each rounded by its own draw of the (FORWARD, CONVERT) stream. */
void testQuantizationSrForwardIsAPureFunctionOfItsConvertStream(void) {
    tensor_t *input = buildFloat2D(2, 8);
    for (size_t i = 0; i < 16; i++) {
        ((float *)input->data)[i] = 0.37f * (float)((i * 7) % 16) - 3.1f;
    }
    layer_t qLayer = {.type = QUANTIZATION, .config = NULL};
    keyedSymForward_t f = {
        .layer = &qLayer, .input = input, .outDims = {2, 8}, .outRank = 2, .forwardRounding = NULL};

    assertKeyedSymForwardDrawsFrom(
        &f, QL_KEYED_SEED,
        keyedPurposeStream(QL_KEYED_SEED, RNG_TAG_FORWARD, RNG_TAG_CONVERT).state,
        QL_KEYED_CONVERT_STATE);

    freeTensor(input);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testQuantizationForwardSymToSymMatchesDirectRequant);
    RUN_TEST(testQuantizationForwardSymToFloatMatchesConvertTensor);
    RUN_TEST(testQuantLayerConvertsAsymToSymInt32MatchesConvertTensor);
    RUN_TEST(testQuantLayerAbortsOnUnpopulatedConversionCell);
    RUN_TEST(testQuantizationBackwardRequantsDyMantissasLikeDirectRequant);
    RUN_TEST(testQuantizationBackwardBfpToBfpReblocksInIntegers);
    RUN_TEST(testQuantizationCalcOutputShapeIdentityIncludingPermutedOrder);
    RUN_TEST(testLayerFunctionsVtableHasQuantizationRow);
    RUN_TEST(testQuantizationConfigUnionMemberRoundTrip);
    RUN_TEST(testQuantLayerInitBorrowingStoresQuantPointersVerbatim);
    RUN_TEST(testQuantLayerInitOwningDeepCopiesQuantizations);
    RUN_TEST(testQuantLayerInitOwningFreesAllAllocationsWithoutLeak);
    RUN_TEST(testQuantizationForwardConvertsWithItsForwardConvertStream);
    RUN_TEST(testQuantizationBackwardConvertsWithItsBackwardConvertStream);
    RUN_TEST(testQuantizationForwardWithoutAContextConvertsWithoutAStream);
    RUN_TEST(testQuantizationSrForwardIsAPureFunctionOfItsConvertStream);
    return UNITY_END();
}
