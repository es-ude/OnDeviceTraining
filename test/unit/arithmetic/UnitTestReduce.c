#define SOURCE_FILE "UNIT_TEST_REDUCE"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#include "DeathTest.h"
#include "MinMax.h"
#include "Quantization.h"
#include "QuantizationApi.h"
#include "Reduce.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

/* Build a FLOAT32 tensor of the given rank/dims from optional row-major data
 * (NULL -> calloc-zero). Mirrors UnitTestLayerNorm.c's buildFloatTensorND.
 * Tests are the sanctioned allocation site (reserveMemory/StorageApi); the
 * alloc-locality rule only forbids raw malloc/free. Caller frees via freeTensor. */
static tensor_t *buildFloatTensorND(size_t numDims, const size_t *dimsIn, const float *vals) {
    size_t *dims = reserveMemory(numDims * sizeof(size_t));
    for (size_t i = 0; i < numDims; i++) {
        dims[i] = dimsIn[i];
    }
    size_t *order = reserveMemory(numDims * sizeof(size_t));
    setOrderOfDimsForNewTensor(numDims, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, numDims, order);
    tensor_t *t = initTensor(shape, quantizationInitFloat(), NULL);
    if (vals != NULL) {
        tensorFillFromFloatBuffer(t, (float *)vals, calcNumberOfElementsByShape(shape));
    }
    return t;
}

/* Build a SYM_INT32 (HALF_AWAY, qMaxBits=ODT_SYM_OPERAND_QMAXBITS=12) tensor; float
 * vals are quantized via tensorFillFromFloatBuffer -> convertFloatTensorToSymInt32Tensor
 * (absmax -> scale, round-clamp). NULL vals -> zero mantissas, scale 1.0. Caller
 * frees via freeTensor. */
static tensor_t *buildSymInt32TensorND(size_t numDims, const size_t *dimsIn, const float *vals) {
    size_t *dims = reserveMemory(numDims * sizeof(size_t));
    for (size_t i = 0; i < numDims; i++) {
        dims[i] = dimsIn[i];
    }
    size_t *order = reserveMemory(numDims * sizeof(size_t));
    setOrderOfDimsForNewTensor(numDims, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, numDims, order);
    tensor_t *t = initTensor(shape, quantizationInitSymInt32(HALF_AWAY), NULL);
    if (vals != NULL) {
        tensorFillFromFloatBuffer(t, vals, calcNumberOfElementsByShape(shape));
    }
    return t;
}

/* Like buildSymInt32TensorND but with an explicit qMaxBits (for the wide-operand
 * fail-fast fixture). */
static tensor_t *buildSymInt32TensorNDBits(size_t numDims, const size_t *dimsIn, const float *vals,
                                           uint8_t qMaxBits) {
    size_t *dims = reserveMemory(numDims * sizeof(size_t));
    for (size_t i = 0; i < numDims; i++) {
        dims[i] = dimsIn[i];
    }
    size_t *order = reserveMemory(numDims * sizeof(size_t));
    setOrderOfDimsForNewTensor(numDims, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, numDims, order);
    tensor_t *t = initTensor(shape, quantizationInitSymInt32WithBits(HALF_AWAY, qMaxBits), NULL);
    if (vals != NULL) {
        tensorFillFromFloatBuffer(t, vals, calcNumberOfElementsByShape(shape));
    }
    return t;
}

static float symScaleOf(tensor_t *t) {
    return ((symInt32QConfig_t *)t->quantization->qConfig)->scale;
}

/* ---- MinMax scalar helpers (added for Task 3's SYM kernels; used here by
 *      rsqrtSymInt32's absmax loop). Trivial hand tests. ---- */

void testAbsFloat32(void) {
    float a = absFloat32(-3.5f);
    float b = absFloat32(2.0f);
    float c = absFloat32(0.0f);
    TEST_ASSERT_FLOAT_WITHIN(1e-9f, 3.5f, a);
    TEST_ASSERT_FLOAT_WITHIN(1e-9f, 2.0f, b);
    TEST_ASSERT_FLOAT_WITHIN(1e-9f, 0.0f, c);
}

void testMaxFloat32s(void) {
    float a = maxFloat32s(2.0f, 5.0f);
    float b = maxFloat32s(-1.0f, -4.0f);
    float c = maxFloat32s(3.0f, 3.0f);
    TEST_ASSERT_FLOAT_WITHIN(1e-9f, 5.0f, a);
    TEST_ASSERT_FLOAT_WITHIN(1e-9f, -1.0f, b);
    TEST_ASSERT_FLOAT_WITHIN(1e-9f, 3.0f, c);
}

/* ---- FLOAT32 reductions ---- */

void testMeanOverLastAxisFloat32(void) {
    // in = [[1,2,3],[4,5,6]] shape [2,3]; k=1 -> mean over axis 1 -> leading dims [2]
    size_t inDims[] = {2, 3};
    tensor_t *in = buildFloatTensorND(2, inDims, (float[]){1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
    size_t outDims[] = {2};
    tensor_t *meanOut = buildFloatTensorND(1, outDims, NULL);

    meanOverTrailingAxesFloat32(in, 1, meanOut);

    float m0 = ((float *)meanOut->data)[0];
    float m1 = ((float *)meanOut->data)[1];

    freeTensor(meanOut);
    freeTensor(in);

    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 2.0f, m0);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 5.0f, m1);
}

void testMeanOverTrailing2AxesFloat32(void) {
    // in shape [2,2,3]; leading [2], trailing [2,3] reduced together (N=6); k=2 -> meanOut [2]
    size_t inDims[] = {2, 2, 3};
    tensor_t *in = buildFloatTensorND(
        3, inDims, (float[]){1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f});
    size_t outDims[] = {2};
    tensor_t *meanOut = buildFloatTensorND(1, outDims, NULL);

    meanOverTrailingAxesFloat32(in, 2, meanOut);

    float m0 = ((float *)meanOut->data)[0];
    float m1 = ((float *)meanOut->data)[1];

    freeTensor(meanOut);
    freeTensor(in);

    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 3.5f, m0); // mean{1..6}
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 9.5f, m1); // mean{7..12}
}

void testVarianceBiasedFloat32(void) {
    // in [[1,2,3],[4,5,6]], k=1, meanIn={2,5}; biased var per row = (1+0+1)/3 = 0.6667
    size_t inDims[] = {2, 3};
    tensor_t *in = buildFloatTensorND(2, inDims, (float[]){1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
    size_t outDims[] = {2};
    tensor_t *meanIn = buildFloatTensorND(1, outDims, (float[]){2.f, 5.f});
    tensor_t *varOut = buildFloatTensorND(1, outDims, NULL);

    varianceBiasedOverTrailingAxesFloat32(in, 1, meanIn, varOut);

    float v0 = ((float *)varOut->data)[0];
    float v1 = ((float *)varOut->data)[1];

    freeTensor(varOut);
    freeTensor(meanIn);
    freeTensor(in);

    TEST_ASSERT_FLOAT_WITHIN(1e-5f, 0.6666667f, v0);
    TEST_ASSERT_FLOAT_WITHIN(1e-5f, 0.6666667f, v1);
}

void testRsqrtFloat32EpsInside(void) {
    TEST_ASSERT_FLOAT_WITHIN(1e-5f, 0.8944228f, rsqrtFloat32(1.25f, 1e-5f)); // 1/sqrt(1.25001)
}

void testRsqrtFloat32ZeroVar(void) {
    TEST_ASSERT_FLOAT_WITHIN(1e-2f, 316.2278f, rsqrtFloat32(0.0f, 1e-5f)); // eps guards /0
}

void testMeanHonorsTransposedView(void) {
    // Permutation-awareness (spec §4). Physical [3,2] buffer, transposeTensor(0,1)
    // -> logical [2,3] with logical (r,c) = phys[c*2 + r]. A contiguous-only
    // implementation would read phys[0..2]/phys[3..5] and get {2.33, 4.67}.
    size_t pdims[] = {3, 2};
    tensor_t *in = buildFloatTensorND(2, pdims, (float[]){1.f, 4.f, 2.f, 5.f, 3.f, 6.f});
    transposeTensor(in, 0, 1); /* logical [2,3], physical unchanged */
    size_t outDims[] = {2};
    tensor_t *meanOut = buildFloatTensorND(1, outDims, NULL);

    meanOverTrailingAxesFloat32(in, 1, meanOut);

    float m0 = ((float *)meanOut->data)[0]; // mean logical row0 = {1,2,3}
    float m1 = ((float *)meanOut->data)[1]; // mean logical row1 = {4,5,6}

    freeTensor(meanOut);
    freeTensor(in);

    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 2.0f, m0);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 5.0f, m1);
}

/* ---- SYM_INT32 reductions (twin-sanity vs the FLOAT32 dequant) ---- */

void testMeanSymInt32MatchesFloatTwin(void) {
    // symIn (mantissas q, scale s) shape [2,3]; floatIn = q*s (exact dequant twin).
    size_t inDims[] = {2, 3};
    tensor_t *symIn = buildSymInt32TensorND(2, inDims, (float[]){1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
    int32_t *q = (int32_t *)symIn->data;
    float s = symScaleOf(symIn);
    float twin[6];
    for (size_t i = 0; i < 6; i++) {
        twin[i] = (float)q[i] * s;
    }
    tensor_t *floatIn = buildFloatTensorND(2, inDims, twin);

    size_t outDims[] = {2};
    tensor_t *meanSym = buildFloatTensorND(1, outDims, NULL);
    tensor_t *meanFloat = buildFloatTensorND(1, outDims, NULL);

    meanOverTrailingAxesSymInt32(symIn, 1, meanSym);
    meanOverTrailingAxesFloat32(floatIn, 1, meanFloat);

    float ms0 = ((float *)meanSym->data)[0];
    float ms1 = ((float *)meanSym->data)[1];
    float mf0 = ((float *)meanFloat->data)[0];
    float mf1 = ((float *)meanFloat->data)[1];

    freeTensor(meanFloat);
    freeTensor(meanSym);
    freeTensor(floatIn);
    freeTensor(symIn);

    TEST_ASSERT_FLOAT_WITHIN(5e-4f, mf0, ms0);
    TEST_ASSERT_FLOAT_WITHIN(5e-4f, mf1, ms1);
}

void testVarianceSymInt32MatchesFloatTwin(void) {
    size_t inDims[] = {2, 3};
    tensor_t *symIn = buildSymInt32TensorND(2, inDims, (float[]){1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
    int32_t *q = (int32_t *)symIn->data;
    float s = symScaleOf(symIn);
    float twin[6];
    for (size_t i = 0; i < 6; i++) {
        twin[i] = (float)q[i] * s;
    }
    tensor_t *floatIn = buildFloatTensorND(2, inDims, twin);

    size_t outDims[] = {2};
    tensor_t *meanIn = buildFloatTensorND(1, outDims, NULL);
    meanOverTrailingAxesFloat32(floatIn, 1, meanIn); // shared mean for both variance calls

    tensor_t *varSym = buildFloatTensorND(1, outDims, NULL);
    tensor_t *varFloat = buildFloatTensorND(1, outDims, NULL);

    varianceBiasedOverTrailingAxesSymInt32(symIn, 1, meanIn, varSym);
    varianceBiasedOverTrailingAxesFloat32(floatIn, 1, meanIn, varFloat);

    float vs0 = ((float *)varSym->data)[0];
    float vs1 = ((float *)varSym->data)[1];
    float vf0 = ((float *)varFloat->data)[0];
    float vf1 = ((float *)varFloat->data)[1];

    freeTensor(varFloat);
    freeTensor(varSym);
    freeTensor(meanIn);
    freeTensor(floatIn);
    freeTensor(symIn);

    TEST_ASSERT_FLOAT_WITHIN(5e-4f, vf0, vs0);
    TEST_ASSERT_FLOAT_WITHIN(5e-4f, vf1, vs1);
}

void testRsqrtSymInt32MatchesFloatTwin(void) {
    // Elementwise rsqrt; dynamic absmax (inputs not all equal). Compare the
    // dequantized output against rsqrt of the DEQUANTIZED inputs (SYM rsqrt is
    // imperfect by design -> loose tol).
    size_t dims[] = {3};
    tensor_t *symIn = buildSymInt32TensorND(1, dims, (float[]){1.25f, 4.0f, 0.25f});
    tensor_t *symOut = buildSymInt32TensorND(1, dims, NULL);

    rsqrtSymInt32(symIn, 1e-5f, symOut);

    int32_t *qin = (int32_t *)symIn->data;
    float sin = symScaleOf(symIn);
    int32_t *qout = (int32_t *)symOut->data;
    float sout = symScaleOf(symOut);

    float got[3];
    float expected[3];
    for (size_t i = 0; i < 3; i++) {
        got[i] = (float)qout[i] * sout;
        expected[i] = rsqrtFloat32((float)qin[i] * sin, 1e-5f);
    }

    freeTensor(symOut);
    freeTensor(symIn);

    for (size_t i = 0; i < 3; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-2f, expected[i], got[i]);
    }
}

void testRsqrtSymInt32RejectsWideOutput(void) {
    // Valid <=16-bit input; out configured with qMaxBits=17. This must be
    // rejected by the OUTPUT's own requant-target guard (dynamic-scale write),
    // not the input's value-sum operand guard -> fail fast either way, but the
    // two guards police different contracts (see reduceValidateSymRequantTarget).
    size_t dims[] = {3};
    tensor_t *symIn = buildSymInt32TensorND(1, dims, (float[]){1.25f, 4.0f, 0.25f});
    tensor_t *wideOut = buildSymInt32TensorNDBits(1, dims, NULL, 17);

    ASSERT_EXITS_WITH_FAILURE(rsqrtSymInt32(symIn, 1e-5f, wideOut));

    freeTensor(wideOut);
    freeTensor(symIn);
}

void testMeanWholeTensorKEqualsRank(void) {
    // k == rank -> K = 1 (whole-tensor reduction). mean over all six elements
    // {1..6} of a [2,3] tensor = 21/6 = 3.5.
    size_t inDims[] = {2, 3};
    tensor_t *in = buildFloatTensorND(2, inDims, (float[]){1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
    size_t outDims[] = {1};
    tensor_t *meanOut = buildFloatTensorND(1, outDims, NULL);

    meanOverTrailingAxesFloat32(in, 2, meanOut);

    float m0 = ((float *)meanOut->data)[0];

    freeTensor(meanOut);
    freeTensor(in);

    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 3.5f, m0);
}

void testReduceRejectsKGreaterThanRank(void) {
    // k=3 on a rank-2 tensor: k > rank underflows `rank - k` (size_t) in
    // reducePhysOffset -> an OOB stack write. blockGeom must fail fast first.
    size_t inDims[] = {2, 3};
    tensor_t *in = buildFloatTensorND(2, inDims, (float[]){1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
    size_t outDims[] = {1};
    tensor_t *meanOut = buildFloatTensorND(1, outDims, NULL);

    ASSERT_EXITS_WITH_FAILURE(meanOverTrailingAxesFloat32(in, 3, meanOut));

    freeTensor(meanOut);
    freeTensor(in);
}

void testMeanSymInt32RejectsWideOperand(void) {
    // qMaxBits=17 breaks the value-sum bound (<=16) -> fail fast.
    size_t inDims[] = {2, 3};
    tensor_t *wideIn =
        buildSymInt32TensorNDBits(2, inDims, (float[]){1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, 17);
    size_t outDims[] = {2};
    tensor_t *meanOut = buildFloatTensorND(1, outDims, NULL);

    ASSERT_EXITS_WITH_FAILURE(meanOverTrailingAxesSymInt32(wideIn, 1, meanOut));

    freeTensor(meanOut);
    freeTensor(wideIn);
}

void testMeanSymInt32RejectsOverlongAxis(void) {
    // N == 2^(32-qMaxBits) with qMaxBits=16 -> N == 65536, exactly AT the
    // value-sum boundary. Conservative bound -- the worst-case sum reaches
    // exactly INT32_MIN at N = 2^(32-qMaxBits); the guard rejects from the
    // boundary on.
    size_t inDims[] = {1, 65536};
    tensor_t *in = buildSymInt32TensorNDBits(2, inDims, NULL, 16);
    size_t outDims[] = {1};
    tensor_t *meanOut = buildFloatTensorND(1, outDims, NULL);

    ASSERT_EXITS_WITH_FAILURE(meanOverTrailingAxesSymInt32(in, 1, meanOut));

    freeTensor(meanOut);
    freeTensor(in);
}

void testMeanSymInt32RejectsZeroWidthOperand(void) {
    // qMaxBits=0 is a degenerate/invalid SYM operand -- it is also the one value
    // that would make the N-guard's shift amount (32-qMaxBits) hit 32, which is
    // UB for a size_t shift on 32-bit MCU targets. reduceValidateSymOperand must
    // reject it before the shift is ever computed.
    size_t inDims[] = {1, 3};
    tensor_t *in = buildSymInt32TensorNDBits(2, inDims, NULL, 0);
    size_t outDims[] = {1};
    tensor_t *meanOut = buildFloatTensorND(1, outDims, NULL);

    ASSERT_EXITS_WITH_FAILURE(meanOverTrailingAxesSymInt32(in, 1, meanOut));

    freeTensor(meanOut);
    freeTensor(in);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testAbsFloat32);
    RUN_TEST(testMaxFloat32s);
    RUN_TEST(testMeanOverLastAxisFloat32);
    RUN_TEST(testMeanOverTrailing2AxesFloat32);
    RUN_TEST(testMeanWholeTensorKEqualsRank);
    RUN_TEST(testReduceRejectsKGreaterThanRank);
    RUN_TEST(testVarianceBiasedFloat32);
    RUN_TEST(testRsqrtFloat32EpsInside);
    RUN_TEST(testRsqrtFloat32ZeroVar);
    RUN_TEST(testMeanHonorsTransposedView);
    RUN_TEST(testMeanSymInt32MatchesFloatTwin);
    RUN_TEST(testVarianceSymInt32MatchesFloatTwin);
    RUN_TEST(testRsqrtSymInt32MatchesFloatTwin);
    RUN_TEST(testRsqrtSymInt32RejectsWideOutput);
    RUN_TEST(testMeanSymInt32RejectsWideOperand);
    RUN_TEST(testMeanSymInt32RejectsOverlongAxis);
    RUN_TEST(testMeanSymInt32RejectsZeroWidthOperand);
    return UNITY_END();
}
