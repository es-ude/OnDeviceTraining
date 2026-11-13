#define SOURCE_FILE "UNIT_TEST_REDUCE"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#include "BfpKernelSupport.h"
#include "DeathTest.h"
#include "Quantization.h"
#include "QuantizationApi.h"
#include "Reduce.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

/* FLOAT32 tensor from row-major vals (NULL -> zeros). Release with freeTensor. */
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

/* SYM_INT32 tensor at qMaxBits = ODT_SYM_OPERAND_QMAXBITS, quantized from vals;
 * NULL vals -> zero mantissas, scale 1.0. Release with freeTensor. */
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

/* Like buildSymInt32TensorND but with an explicit qMaxBits. */
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
    return odtScaleToFloat(((symInt32QConfig_t *)t->quantization->qConfig)->scale);
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
    // Permutation-awareness. Physical [3,2] buffer, transposeTensor(0,1)
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

void testVarianceHonorsTransposedRank3View(void) {
    // BatchNorm1d's statistics view: physical [m=2, C=3, T=2] identity
    // buffer, transposeTensor(0,1) -> logical [C, m, T]; k = 2 reduces each
    // channel over its m*T = 4 values. x[b][c][t] = 10*c + 3*b + t.
    //   channel c values {10c, 10c+1, 10c+3, 10c+4}: mean 10c+2, biased var 2.5
    // A contiguous-block reader would take phys[4c..4c+3] and mix channels.
    size_t pdims[] = {2, 3, 2};
    float x[12];
    for (size_t b = 0; b < 2; b++) {
        for (size_t c = 0; c < 3; c++) {
            for (size_t t = 0; t < 2; t++) {
                x[(b * 3 + c) * 2 + t] = (float)(10 * c + 3 * b + t);
            }
        }
    }
    tensor_t *in = buildFloatTensorND(3, pdims, x);
    transposeTensor(in, 0, 1); /* logical [3, 2, 2], physical unchanged */
    size_t statDims[] = {3};
    tensor_t *meanOut = buildFloatTensorND(1, statDims, NULL);
    tensor_t *varOut = buildFloatTensorND(1, statDims, NULL);

    meanOverTrailingAxesFloat32(in, 2, meanOut);
    varianceBiasedOverTrailingAxesFloat32(in, 2, meanOut, varOut);

    float m[3];
    float v[3];
    for (size_t c = 0; c < 3; c++) {
        m[c] = ((float *)meanOut->data)[c];
        v[c] = ((float *)varOut->data)[c];
    }
    freeTensor(varOut);
    freeTensor(meanOut);
    freeTensor(in);

    for (size_t c = 0; c < 3; c++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, 10.0f * (float)c + 2.0f, m[c]);
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, 2.5f, v[c]);
    }
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

/* ---- BFP reductions ---- */

/* Unpacked-BFP scratch (the executeOp kernel-side form: int32 codes, not packed
 * bytes), so built by hand. Shape [2,6], groups of 4, exponents {126,127,129} =
 * scales {0.5,1,4}. Unequal scales expose ignored exponents; group 1 spans the row
 * boundary, so a fold at the wrong boundary moves both row means. The caller owns
 * the metadata storage, which must outlive the tensor. */
static int32_t kReduceBfpCodes[12] = {10, -20, 30, 40, 50, -60, 7, 9, -5, 11, 13, -17};
static uint8_t kReduceBfpExponents[3] = {126, 127, 129};

static void buildReduceBfpScratch(tensor_t *t, shape_t *shape, size_t *dims, size_t *order,
                                  quantization_t *q, bfpQConfig_t *qc) {
    dims[0] = 2;
    dims[1] = 6;
    order[0] = 0;
    order[1] = 1;
    setShape(shape, dims, 2, order);
    qc->exponents = kReduceBfpExponents;
    qc->numGroups = 3;
    qc->groupSize = 4;
    qc->roundingMode = HALF_AWAY;
    qc->mantissaBits = 8;
    qc->exponentBits = 8;
    initBfpQuantization(qc, q);
    setTensorValues(t, (uint8_t *)kReduceBfpCodes, shape, q, NULL);
}

void testMeanBfpSegmentFoldAcrossGroupBoundary(void) {
    tensor_t t;
    shape_t shape;
    size_t dims[2];
    size_t order[2];
    quantization_t q;
    bfpQConfig_t qc;
    buildReduceBfpScratch(&t, &shape, dims, order, &q, &qc);

    size_t outDims[] = {2};
    tensor_t *meanOut = buildFloatTensorND(1, outDims, NULL);

    meanOverTrailingAxesBfp(&t, 1, meanOut);

    float m0 = ((float *)meanOut->data)[0];
    float m1 = ((float *)meanOut->data)[1];

    freeTensor(meanOut);

    // row0: g0 partial 10-20+30+40 = 60 -> *0.5 = 30; g1 partial 50-60 = -10
    // -> *1 = -10; mean = 20/6. row1: g1 partial 7+9 = 16 -> *1 = 16; g2 partial
    // -5+11+13-17 = 2 -> *4 = 8; mean = 24/6 = 4.
    TEST_ASSERT_EQUAL_FLOAT(20.0f / 6.0f, m0);
    TEST_ASSERT_EQUAL_FLOAT(4.0f, m1);
}

void testVarianceBfpDequantCenterSquare(void) {
    // Same fixture. row0 dequants to {5,-10,15,20,50,-60}, mean 10/3
    // -> var = 61050/54 = 1130.5555...; row1 dequants to {7,9,-20,44,52,-68},
    // mean 4 -> var = 9698/6 = 1616.3333... (float32 accumulation stays well
    // inside the 0.01 tolerance at these magnitudes).
    tensor_t t;
    shape_t shape;
    size_t dims[2];
    size_t order[2];
    quantization_t q;
    bfpQConfig_t qc;
    buildReduceBfpScratch(&t, &shape, dims, order, &q, &qc);

    size_t outDims[] = {2};
    tensor_t *meanOut = buildFloatTensorND(1, outDims, NULL);
    tensor_t *varOut = buildFloatTensorND(1, outDims, NULL);

    meanOverTrailingAxesBfp(&t, 1, meanOut);
    varianceBiasedOverTrailingAxesBfp(&t, 1, meanOut, varOut);

    float v0 = ((float *)varOut->data)[0];
    float v1 = ((float *)varOut->data)[1];

    freeTensor(varOut);
    freeTensor(meanOut);

    TEST_ASSERT_FLOAT_WITHIN(0.01f, 1130.5555f, v0);
    TEST_ASSERT_FLOAT_WITHIN(0.01f, 1616.3333f, v1);
}

void testMeanBfpPerTensorSentinel(void) {
    // {1,0} per-tensor: one segment per row (group 0 everywhere), exponent 128
    // -> scale 2.0. Codes {1,2,3, 4,5,6} shape [2,3] -> means {2*6/3, 2*15/3}.
    static int32_t codes[6] = {1, 2, 3, 4, 5, 6};
    static uint8_t exps[1] = {128};
    tensor_t t;
    shape_t shape;
    size_t dims[2] = {2, 3};
    size_t order[2] = {0, 1};
    setShape(&shape, dims, 2, order);
    bfpQConfig_t qc = {.exponents = exps,
                       .numGroups = 1,
                       .groupSize = 0,
                       .roundingMode = HALF_AWAY,
                       .mantissaBits = 8,
                       .exponentBits = 8};
    quantization_t q;
    initBfpQuantization(&qc, &q);
    setTensorValues(&t, (uint8_t *)codes, &shape, &q, NULL);

    size_t outDims[] = {2};
    tensor_t *meanOut = buildFloatTensorND(1, outDims, NULL);

    meanOverTrailingAxesBfp(&t, 1, meanOut);

    float m0 = ((float *)meanOut->data)[0];
    float m1 = ((float *)meanOut->data)[1];

    freeTensor(meanOut);

    TEST_ASSERT_EQUAL_FLOAT(4.0f, m0);
    TEST_ASSERT_EQUAL_FLOAT(10.0f, m1);
}

void testMeanBfpSumHeadroomDeath(void) {
    // bfpSumSegmentLimit(16) = INT32_MAX >> 15 = 65535; a per-tensor {1,0}
    // operand's segment run IS the reduction length, so N = 65536 must die.
    size_t n = 65536;
    int32_t *codes = reserveMemory(n * sizeof(int32_t));
    static uint8_t exps[1] = {127};
    tensor_t t;
    shape_t shape;
    size_t dims[1] = {65536};
    size_t order[1] = {0};
    setShape(&shape, dims, 1, order);
    bfpQConfig_t qc = {.exponents = exps,
                       .numGroups = 1,
                       .groupSize = 0,
                       .roundingMode = HALF_AWAY,
                       .mantissaBits = 16,
                       .exponentBits = 8};
    quantization_t q;
    initBfpQuantization(&qc, &q);
    setTensorValues(&t, (uint8_t *)codes, &shape, &q, NULL);

    size_t outDims[] = {1};
    tensor_t *meanOut = buildFloatTensorND(1, outDims, NULL);

    ASSERT_EXITS_WITH_FAILURE(meanOverTrailingAxesBfp(&t, 1, meanOut));

    freeTensor(meanOut);
    freeReservedMemory(codes);
}

void testBfpSumHeadroomRejectsZeroMantissaWidth(void) {
    bfpQConfig_t qC = {.exponents = NULL,
                       .numGroups = 1,
                       .groupSize = 0,
                       .roundingMode = HALF_AWAY,
                       .mantissaBits = 0,
                       .exponentBits = 8};
    /* reductionLen must be 0 so the segment-length guard cannot mask a missing
     * zero-width rejection. */
    ASSERT_EXITS_WITH_FAILURE(bfpValidateSumHeadroom(&qC, 0, "test"));
}

/* ---- sumSquaresOverTrailingAxesFloat32 ---- */

void testSumSquaresTrailingAxes(void) {
    // in [2,3]: rows {1,2,3},{-2,0,2}; k=1 -> ssqOut [2] = {14, 8}.
    size_t inDims[] = {2, 3};
    tensor_t *in = buildFloatTensorND(2, inDims, (float[]){1.0f, 2.0f, 3.0f, -2.0f, 0.0f, 2.0f});
    size_t outDims[] = {2};
    tensor_t *ssq = buildFloatTensorND(1, outDims, NULL);

    sumSquaresOverTrailingAxesFloat32(in, 1, ssq);

    float s0 = ((float *)ssq->data)[0];
    float s1 = ((float *)ssq->data)[1];

    freeTensor(ssq);
    freeTensor(in);

    TEST_ASSERT_EQUAL_FLOAT(14.0f, s0);
    TEST_ASSERT_EQUAL_FLOAT(8.0f, s1);
}

void testSumSquaresWholeTensor(void) {
    // k == rank -> single scalar.
    size_t inDims[] = {4};
    tensor_t *in = buildFloatTensorND(1, inDims, (float[]){1.0f, 1.0f, 2.0f, 3.0f});
    size_t outDims[] = {1};
    tensor_t *ssq = buildFloatTensorND(1, outDims, NULL);

    sumSquaresOverTrailingAxesFloat32(in, 1, ssq);

    float s0 = ((float *)ssq->data)[0];

    freeTensor(ssq);
    freeTensor(in);

    TEST_ASSERT_EQUAL_FLOAT(15.0f, s0);
}

void testSumSquaresPermutationAware(void) {
    // Transposed view: [2,3] storage viewed as [3,2]; k=1 over the VIEW's
    // trailing dim -> 3 row-SSQs of the transposed logical rows.
    size_t inDims[] = {2, 3};
    tensor_t *in = buildFloatTensorND(2, inDims, (float[]){1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    transposeTensor(in, 0, 1); // logical [3,2]: rows {1,4},{2,5},{3,6}
    size_t outDims[] = {3};
    tensor_t *ssq = buildFloatTensorND(1, outDims, NULL);

    sumSquaresOverTrailingAxesFloat32(in, 1, ssq);

    float s0 = ((float *)ssq->data)[0];
    float s1 = ((float *)ssq->data)[1];
    float s2 = ((float *)ssq->data)[2];

    freeTensor(ssq);
    transposeTensor(in, 0, 1);
    freeTensor(in);

    TEST_ASSERT_EQUAL_FLOAT(17.0f, s0);
    TEST_ASSERT_EQUAL_FLOAT(29.0f, s1);
    TEST_ASSERT_EQUAL_FLOAT(45.0f, s2);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testMeanOverLastAxisFloat32);
    RUN_TEST(testMeanOverTrailing2AxesFloat32);
    RUN_TEST(testMeanWholeTensorKEqualsRank);
    RUN_TEST(testVarianceBiasedFloat32);
    RUN_TEST(testRsqrtFloat32EpsInside);
    RUN_TEST(testRsqrtFloat32ZeroVar);
    RUN_TEST(testMeanHonorsTransposedView);
    RUN_TEST(testVarianceHonorsTransposedRank3View);
    RUN_TEST(testMeanSymInt32MatchesFloatTwin);
    RUN_TEST(testVarianceSymInt32MatchesFloatTwin);
    RUN_TEST(testMeanSymInt32RejectsWideOperand);
    RUN_TEST(testMeanSymInt32RejectsOverlongAxis);
    RUN_TEST(testMeanSymInt32RejectsZeroWidthOperand);
    RUN_TEST(testMeanBfpSegmentFoldAcrossGroupBoundary);
    RUN_TEST(testVarianceBfpDequantCenterSquare);
    RUN_TEST(testMeanBfpPerTensorSentinel);
    RUN_TEST(testMeanBfpSumHeadroomDeath);
    RUN_TEST(testBfpSumHeadroomRejectsZeroMantissaWidth);
    RUN_TEST(testSumSquaresTrailingAxes);
    RUN_TEST(testSumSquaresWholeTensor);
    RUN_TEST(testSumSquaresPermutationAware);
    return UNITY_END();
}
