#include <math.h>
#include <stdbool.h>
#include <string.h>

#include "BfpRawFixture.h"
#include "BfpScatterRef.h"
#include "ConvTranspose1dKernel.h"
#include "DeathTest.h"
#include "QuantizationApi.h"
#include "RNG.h"
#include "SlidingWindow1d.h"
#include "StorageApi.h"
#include "SymAccumGuard.h"
#include "SymGroupedWeight.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "TestScale.h"
#include "expected_bfp_convT1d.h"
#include "expected_conv_transpose_1d_kernel.h"
#include "unity.h"

/* Golden comparisons: exact int32/int16 equality, bitwise float equality. */
#define GOLD_I32(e, a, n) TEST_ASSERT_EQUAL_INT32_ARRAY((e), (a), (n))
#define GOLD_I16(e, a, n) TEST_ASSERT_EQUAL_INT16_ARRAY((e), (a), (n))
#define GOLD_F32(e, a, n) TEST_ASSERT_EQUAL_MEMORY((e), (a), (n) * sizeof(float))
#define GOLD_F32_SCALAR(e, a) TEST_ASSERT_EQUAL_MEMORY(&(float){(e)}, &(float){(a)}, sizeof(float))
#define GOLD_U32_SCALAR(e, a) TEST_ASSERT_EQUAL_UINT32((e), (a))
#include "expected_convT1d_grouped_fold.h"
#include "golden_convT1d_scatter.h"

/* Grouped-weight ctx fixture: the stored dtype's code domain fixes the
 * weight's operand range (SymGroupedWeight.h); the guard runner below picks
 * the domain through gCtWeightDomain. */
static symGroupedWeight_t symGroupedWeightSym(const symQConfig_t *groups) {
    return (symGroupedWeight_t){.groups = groups, .domain = SYM_CODES_TWOS_COMPLEMENT};
}

static tensor_t *makeFloatTensor(size_t const *dims, size_t numDims, float const *data) {
    size_t *ownedDims = reserveMemory(numDims * sizeof(size_t));
    memcpy(ownedDims, dims, numDims * sizeof(size_t));
    size_t *order = reserveMemory(numDims * sizeof(size_t));
    setOrderOfDimsForNewTensor(numDims, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, ownedDims, numDims, order);
    tensor_t *t = initTensor(shape, quantizationInitFloat(), NULL);
    if (data != NULL) {
        tensorFillFromFloatBuffer(t, data, calcNumberOfElementsByTensor(t));
    }
    return t;
}

void testConvTranspose1dKernelBasic() {
    float xData[] = {1.0f, 2.0f, 3.0f};
    size_t xDims[] = {1, 1, 3};
    tensor_t *x = makeFloatTensor(xDims, 3, xData);

    float wData[] = {2.0f, 4.0f};
    size_t wDims[] = {1, 1, 2}; // [Cin=1, Cout/groups=1, K=2]
    tensor_t *w = makeFloatTensor(wDims, 3, wData);

    size_t yDims[] = {1, 1, 4};
    tensor_t *y = makeFloatTensor(yDims, 3, NULL);

    kernel_t kernel;
    initKernel(&kernel, 2, VALID, 1, 1); // size, paddingType, dilation, stride

    convTranspose1dKernelFloat32(x, w, NULL, &kernel, 1, 0, y);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expectedConvT1dForward_basic, y->data,
                                  expectedConvT1dForward_basic_len);
}

void testConvTranspose1dKernelWithStride() {
    float xData[] = {1.0f, 2.0f, 3.0f};
    size_t xDims[] = {1, 1, 3};
    tensor_t *x = makeFloatTensor(xDims, 3, xData);

    float wData[] = {2.0f, 4.0f};
    size_t wDims[] = {1, 1, 2};
    tensor_t *w = makeFloatTensor(wDims, 3, wData);

    // Lout = (3-1)*2 + (2-1)*1 + 0 + 1 = 6
    size_t yDims[] = {1, 1, 6};
    tensor_t *y = makeFloatTensor(yDims, 3, NULL);

    kernel_t kernel;
    initKernel(&kernel, 2, VALID, 1, 2); // size, padding, dilation, stride=2

    convTranspose1dKernelFloat32(x, w, NULL, &kernel, 1, 0, y);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expectedConvT1dForward_withStride, y->data,
                                  expectedConvT1dForward_withStride_len);
}

void testConvTranspose1dKernelWithOutputPadding() {
    float xData[] = {1.0f, 2.0f, 3.0f};
    size_t xDims[] = {1, 1, 3};
    tensor_t *x = makeFloatTensor(xDims, 3, xData);

    float wData[] = {2.0f, 4.0f};
    size_t wDims[] = {1, 1, 2};
    tensor_t *w = makeFloatTensor(wDims, 3, wData);

    // Lout = (3-1)*2 + (2-1)*1 + 1 + 1 = 7 with outputPadding=1
    size_t yDims[] = {1, 1, 7};
    tensor_t *y = makeFloatTensor(yDims, 3, NULL);

    kernel_t kernel;
    initKernel(&kernel, 2, VALID, 1, 2);

    convTranspose1dKernelFloat32(x, w, NULL, &kernel, 1, 1, y);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expectedConvT1dForward_withOutputPadding, y->data,
                                  expectedConvT1dForward_withOutputPadding_len);
}

void testConvTranspose1dKernelWithGroups() {
    // Lout = (3-1)*1 + (2-1)*1 + 0 + 1 = 4
    size_t yDims[] = {1, 4, 4};
    tensor_t *y = makeFloatTensor(yDims, 3, NULL);

    size_t xDims[] = {1, 4, 3};
    tensor_t *x = makeFloatTensor(xDims, 3, inputConvT1dForward_groups);

    size_t wDims[] = {4, 2, 2}; // [Cin=4, Cout/groups=2, K=2]
    tensor_t *w = makeFloatTensor(wDims, 3, weightConvT1dForward_groups);

    kernel_t kernel;
    initKernel(&kernel, 2, VALID, 1, 1);

    convTranspose1dKernelFloat32(x, w, NULL, &kernel, 2, 0, y);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expectedConvT1dForward_groups, y->data,
                                  expectedConvT1dForward_groups_len);
}

void testConvTranspose1dKernelIsConvBackwardAdjoint() {
    // Setup: from generator, we have:
    //   inputConvT1dForward_adjointCheck   = grad_y from a Conv1d backward,
    //   weightConvT1dForward_adjointCheck  = the forward Conv1d's W,
    //   expectedConvT1dForward_adjointCheck = autograd-derived dL/dx
    //
    // Conv1d setup that produced these: x:[1,2,5], w:[3,2,3], no bias,
    //   stride=1, padding=0, dilation=1, groups=1
    //   -> y:[1,3,3]
    //
    // For ConvTranspose1d (the adjoint): grad_y is "input" with shape [1,3,3];
    //   weight is reused as [Cin_t, Cout_t/g, K] = [3,2,3].
    //   Output shape = [1, 2, 5] (= original x shape).
    //
    // Lout = (L-1)*stride + dilation*(K-1) + outputPadding + 1
    //      = 2*1 + 1*2 + 0 + 1 = 5  ✓

    size_t xDims[] = {1, 3, 3};
    tensor_t *gy = makeFloatTensor(xDims, 3, inputConvT1dForward_adjointCheck);

    size_t wDims[] = {3, 2, 3};
    tensor_t *w = makeFloatTensor(wDims, 3, weightConvT1dForward_adjointCheck);

    size_t yDims[] = {1, 2, 5};
    tensor_t *gx = makeFloatTensor(yDims, 3, NULL);

    kernel_t kernel;
    initKernel(&kernel, 3, VALID, 1, 1);

    convTranspose1dKernelFloat32(gy, w, NULL, &kernel, 1, 0, gx);

    // expected = autograd-derived dL/dx; should match within float tolerance
    for (size_t i = 0; i < expectedConvT1dForward_adjointCheck_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expectedConvT1dForward_adjointCheck[i],
                                 ((float *)gx->data)[i]);
    }
}

void testConvTranspose1dKernelSamePaddingSymmetric() {
    // Adjoint of a Conv1d(K=3, stride=1, dilation=1, SAME) on inputLen=5.
    // Forward Conv1d geometry: padLeft=1, padRight=1, outputLen=5.
    // Adjoint takes lossGrad of shape [1,1,5] and scatters into propLoss [1,1,5].
    //
    // Hand-derived expected propLoss for an all-ones lossGrad against a known
    // small weight: this produces exactly the column sums of the unrolled
    // correlation matrix W, padded.
    //
    //   Forward: y[i] = sum_k x[i + k - 1] * w[k]   (with OOB skipped)
    //     y[0] = w[1]*x[0] + w[2]*x[1]              (padLeft=1 cuts w[0])
    //     y[1] = w[0]*x[0] + w[1]*x[1] + w[2]*x[2]
    //     y[2] = w[0]*x[1] + w[1]*x[2] + w[2]*x[3]
    //     y[3] = w[0]*x[2] + w[1]*x[3] + w[2]*x[4]
    //     y[4] = w[0]*x[3] + w[1]*x[4]              (padRight=1 cuts w[2])
    //
    //   Adjoint with lossGrad=ones:
    //     propLoss[0] = w[1] + w[0]                 = (sum of w-positions hitting x[0])
    //     propLoss[1] = w[2] + w[1] + w[0]
    //     propLoss[2] = w[2] + w[1] + w[0]
    //     propLoss[3] = w[2] + w[1] + w[0]
    //     propLoss[4] = w[2] + w[1]
    //
    // For w = [2, 4, 8]:
    //     propLoss = [4+2, 8+4+2, 8+4+2, 8+4+2, 8+4] = [6, 14, 14, 14, 12]

    float lossGradData[] = {1, 1, 1, 1, 1};
    size_t lossGradDims[] = {1, 1, 5};
    tensor_t *lossGrad = makeFloatTensor(lossGradDims, 3, lossGradData);

    float weightData[] = {2, 4, 8};
    size_t weightDims[] = {1, 1, 3};
    tensor_t *weight = makeFloatTensor(weightDims, 3, weightData);

    size_t propLossDims[] = {1, 1, 5};
    tensor_t *propLoss = makeFloatTensor(propLossDims, 3, NULL);

    kernel_t kernel;
    initKernel(&kernel, 3, SAME, 1, 1); // size, paddingType, dilation, stride

    convTranspose1dKernelFloat32(lossGrad, weight, NULL, &kernel, 1, 0, propLoss);

    float expected[] = {6, 14, 14, 14, 12};
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expected, propLoss->data, 5);
}

void testConvTranspose1dKernelSamePaddingAsymmetric() {
    // Adjoint of a Conv1d(K=4, stride=1, dilation=1, SAME) on inputLen=5.
    // Total pad = 4-1 = 3; padLeft=1, padRight=2 (PyTorch right-biased).
    //
    //   Forward: y[i] = sum_k x[i + k - 1] * w[k]
    //     y[0] = w[1]*x[0] + w[2]*x[1] + w[3]*x[2]              (k=0 cut)
    //     y[1] = w[0]*x[0] + w[1]*x[1] + w[2]*x[2] + w[3]*x[3]
    //     y[2] = w[0]*x[1] + w[1]*x[2] + w[2]*x[3] + w[3]*x[4]
    //     y[3] = w[0]*x[2] + w[1]*x[3] + w[2]*x[4]              (k=3 cut: idx=6 >= 5)
    //     y[4] = w[0]*x[3] + w[1]*x[4]                          (k=2,3 cut)
    //
    //   Adjoint with lossGrad=ones:
    //     propLoss[0] = w[1] + w[0]                              (hits y[0], y[1])
    //     propLoss[1] = w[2] + w[1] + w[0]
    //     propLoss[2] = w[3] + w[2] + w[1] + w[0]
    //     propLoss[3] = w[3] + w[2] + w[1] + w[0]
    //     propLoss[4] = w[3] + w[2] + w[1]
    //
    // For w = [1, 2, 4, 8]:
    //     propLoss = [3, 7, 15, 15, 14]

    float lossGradData[] = {1, 1, 1, 1, 1};
    size_t lossGradDims[] = {1, 1, 5};
    tensor_t *lossGrad = makeFloatTensor(lossGradDims, 3, lossGradData);

    float weightData[] = {1, 2, 4, 8};
    size_t weightDims[] = {1, 1, 4};
    tensor_t *weight = makeFloatTensor(weightDims, 3, weightData);

    size_t propLossDims[] = {1, 1, 5};
    tensor_t *propLoss = makeFloatTensor(propLossDims, 3, NULL);

    kernel_t kernel;
    initKernel(&kernel, 4, SAME, 1, 1);

    convTranspose1dKernelFloat32(lossGrad, weight, NULL, &kernel, 1, 0, propLoss);

    float expected[] = {3, 7, 15, 15, 14};
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expected, propLoss->data, 5);
}

void testConvTranspose1dKernelAdjointSameGrouped() {
    size_t gyDims[] = {2, 4, 6};
    tensor_t *gy = makeFloatTensor(gyDims, 3, inputConvT1d_adjointSameGrouped);

    size_t wDims[] = {4, 2, 3};
    tensor_t *w = makeFloatTensor(wDims, 3, weightConvT1d_adjointSameGrouped);

    size_t propLossDims[] = {2, 4, 6};
    tensor_t *propLoss = makeFloatTensor(propLossDims, 3, NULL);

    kernel_t kernel;
    initKernel(&kernel, 3, SAME, 1, 1);

    convTranspose1dKernelFloat32(gy, w, NULL, &kernel, 2, 0, propLoss);

    for (size_t i = 0; i < expectedConvT1d_adjointSameGrouped_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expectedConvT1d_adjointSameGrouped[i],
                                 ((float *)propLoss->data)[i]);
    }
}

/* ---- BFP epic PR2 (Task 5): convTranspose1dKernelBfpGather (D9) ----------
 *
 * Operands arrive in the funnel's UNPACKED-BFP scratch form: ->data holds
 * int32 sign-extended mantissa codes, ->quantization is BFP with a live
 * bfpQConfig_t (stack-fixture idiom, Quantization.h). Output is RAW FLOAT32
 * -- the kernel never rounds and never width-restores. The gold fixture
 * lives in the exact float regime (generate_expected_bfp_convT1d.py asserts
 * it, including its float-scatter cross-check), so expectations are BIT-
 * pinned via TEST_ASSERT_EQUAL_MEMORY, not a tolerance. */

/* Gold-fixture BFP input tensor (shared across the gold and twin tests);
 * exponents overridable so the twin can force power-of-two per-tensor scales
 * onto the same mantissas. */
static void setupBfpConvTInput(tensor_t *tensor, shape_t *shape, size_t *dims, size_t *order,
                               bfpQConfig_t *qC, quantization_t *q, uint8_t *exponents) {
    dims[0] = (size_t)kBfpConvTBatch;
    dims[1] = (size_t)kBfpConvTInChannels;
    dims[2] = (size_t)kBfpConvTInputLength;
    order[0] = 0;
    order[1] = 1;
    order[2] = 2;
    setShape(shape, dims, 3, order);
    /* sizeof(fixture) sizing happens at the caller (a regenerated gold with a
     * different group count fails loudly at the numGroups check instead of
     * silently short-copying) */
    qC->exponents = exponents;
    qC->numGroups = (size_t)kBfpConvTInNumGroups;
    qC->groupSize = (size_t)kBfpConvTInGroupSize;
    qC->roundingMode = HALF_AWAY;
    qC->mantissaBits = (uint8_t)kBfpConvTInMantissaBits;
    qC->exponentBits = (uint8_t)kBfpConvTInExponentBits;
    initBfpQuantization(qC, q);
    setTensorValues(tensor, (uint8_t *)kBfpConvTInCodes, shape, q, NULL);
}

static void setupBfpConvTWeight(tensor_t *tensor, shape_t *shape, size_t *dims, size_t *order,
                                bfpQConfig_t *qC, quantization_t *q, uint8_t *exponents) {
    dims[0] = (size_t)kBfpConvTInChannels;
    dims[1] = (size_t)kBfpConvTOutChannels; /* [Cin, Cout/groups, K], conv groups == 1 */
    dims[2] = (size_t)kBfpConvTKernelSize;
    order[0] = 0;
    order[1] = 1;
    order[2] = 2;
    setShape(shape, dims, 3, order);
    qC->exponents = exponents;
    qC->numGroups = (size_t)kBfpConvTWNumGroups;
    qC->groupSize = (size_t)kBfpConvTWGroupSize;
    qC->roundingMode = HALF_AWAY;
    qC->mantissaBits = (uint8_t)kBfpConvTWMantissaBits;
    qC->exponentBits = (uint8_t)kBfpConvTWExponentBits;
    initBfpQuantization(qC, q);
    setTensorValues(tensor, (uint8_t *)kBfpConvTWCodes, shape, q, NULL);
}

static void setupBfpConvTOutput(tensor_t *tensor, shape_t *shape, size_t *dims, size_t *order,
                                quantization_t *q, float *data) {
    dims[0] = (size_t)kBfpConvTBatch;
    dims[1] = (size_t)kBfpConvTOutChannels;
    dims[2] = (size_t)kBfpConvTOutLen;
    order[0] = 0;
    order[1] = 1;
    order[2] = 2;
    setShape(shape, dims, 3, order);
    initFloat32Quantization(q);
    setTensorValues(tensor, (uint8_t *)data, shape, q, NULL);
}

void testConvTranspose1dKernelBfpGatherMatchesGold(void) {
    tensor_t inTensor;
    size_t inDims[3], inOrder[3];
    shape_t inShape;
    uint8_t inExponents[sizeof(kBfpConvTInExponents)];
    memcpy(inExponents, kBfpConvTInExponents, sizeof(inExponents));
    bfpQConfig_t inQC;
    quantization_t inQ;
    setupBfpConvTInput(&inTensor, &inShape, inDims, inOrder, &inQC, &inQ, inExponents);

    tensor_t wTensor;
    size_t wDims[3], wOrder[3];
    shape_t wShape;
    uint8_t wExponents[sizeof(kBfpConvTWExponents)];
    memcpy(wExponents, kBfpConvTWExponents, sizeof(wExponents));
    bfpQConfig_t wQC;
    quantization_t wQ;
    setupBfpConvTWeight(&wTensor, &wShape, wDims, wOrder, &wQC, &wQ, wExponents);

    tensor_t biasTensor;
    size_t biasDims[] = {(size_t)kBfpConvTOutChannels};
    size_t biasOrder[] = {0};
    shape_t biasShape;
    setShape(&biasShape, biasDims, 1, biasOrder);
    uint8_t biasExponents[sizeof(kBfpConvTBiasExponents)];
    memcpy(biasExponents, kBfpConvTBiasExponents, sizeof(biasExponents));
    bfpQConfig_t biasQC = {.exponents = biasExponents,
                           .numGroups = 1,
                           .groupSize = 0,
                           .roundingMode = HALF_AWAY,
                           .mantissaBits = (uint8_t)kBfpConvTBiasMantissaBits,
                           .exponentBits = (uint8_t)kBfpConvTBiasExponentBits};
    quantization_t biasQ;
    initBfpQuantization(&biasQC, &biasQ);
    setTensorValues(&biasTensor, (uint8_t *)kBfpConvTBiasCodes, &biasShape, &biasQ, NULL);

    tensor_t outTensor;
    float outData[24];
    size_t outDims[3], outOrder[3];
    shape_t outShape;
    quantization_t outQ;
    setupBfpConvTOutput(&outTensor, &outShape, outDims, outOrder, &outQ, outData);

    kernel_t kernel;
    initKernel(&kernel, (size_t)kBfpConvTKernelSize, VALID, 1, (size_t)kBfpConvTStride);

    convTranspose1dKernelBfpGather(&inTensor, &wTensor, &biasTensor, &kernel, 1,
                                   (size_t)kBfpConvTOutputPadding, &outTensor);

    TEST_ASSERT_EQUAL_MEMORY(kBfpConvTExpected, outTensor.data,
                             kBfpConvTExpected_len * sizeof(float));
}

void testConvTranspose1dKernelBfpGatherNoBiasZeroSeeds(void) {
    tensor_t inTensor;
    size_t inDims[3], inOrder[3];
    shape_t inShape;
    uint8_t inExponents[sizeof(kBfpConvTInExponents)];
    memcpy(inExponents, kBfpConvTInExponents, sizeof(inExponents));
    bfpQConfig_t inQC;
    quantization_t inQ;
    setupBfpConvTInput(&inTensor, &inShape, inDims, inOrder, &inQC, &inQ, inExponents);

    tensor_t wTensor;
    size_t wDims[3], wOrder[3];
    shape_t wShape;
    uint8_t wExponents[sizeof(kBfpConvTWExponents)];
    memcpy(wExponents, kBfpConvTWExponents, sizeof(wExponents));
    bfpQConfig_t wQC;
    quantization_t wQ;
    setupBfpConvTWeight(&wTensor, &wShape, wDims, wOrder, &wQC, &wQ, wExponents);

    tensor_t outTensor;
    float outData[24];
    size_t outDims[3], outOrder[3];
    shape_t outShape;
    quantization_t outQ;
    setupBfpConvTOutput(&outTensor, &outShape, outDims, outOrder, &outQ, outData);

    kernel_t kernel;
    initKernel(&kernel, (size_t)kBfpConvTKernelSize, VALID, 1, (size_t)kBfpConvTStride);

    convTranspose1dKernelBfpGather(&inTensor, &wTensor, NULL, &kernel, 1,
                                   (size_t)kBfpConvTOutputPadding, &outTensor);

    TEST_ASSERT_EQUAL_MEMORY(kBfpConvTNoBiasExpected, outTensor.data,
                             kBfpConvTNoBiasExpected_len * sizeof(float));
}

/* BFP epic PR2: the gather forwards kernel->dilation into
 * convTranspose1dTapsAt only inside the BFP arm; every other BFP fixture runs
 * dilation=1, where a hardcoded 1 is an arithmetic identity. This fixture's
 * contributor enumeration (out_len 14 = (5-1)*2 + 2*2 + 1 + 1) genuinely
 * depends on the dilation. */
void testConvTranspose1dKernelBfpGatherDilation2MatchesGold(void) {
    tensor_t inTensor;
    size_t inDims[3], inOrder[3];
    shape_t inShape;
    uint8_t inExponents[sizeof(kBfpConvTInExponents)];
    memcpy(inExponents, kBfpConvTInExponents, sizeof(inExponents));
    bfpQConfig_t inQC;
    quantization_t inQ;
    setupBfpConvTInput(&inTensor, &inShape, inDims, inOrder, &inQC, &inQ, inExponents);

    tensor_t wTensor;
    size_t wDims[3], wOrder[3];
    shape_t wShape;
    uint8_t wExponents[sizeof(kBfpConvTWExponents)];
    memcpy(wExponents, kBfpConvTWExponents, sizeof(wExponents));
    bfpQConfig_t wQC;
    quantization_t wQ;
    setupBfpConvTWeight(&wTensor, &wShape, wDims, wOrder, &wQC, &wQ, wExponents);

    tensor_t biasTensor;
    size_t biasDims[] = {(size_t)kBfpConvTOutChannels};
    size_t biasOrder[] = {0};
    shape_t biasShape;
    setShape(&biasShape, biasDims, 1, biasOrder);
    uint8_t biasExponents[sizeof(kBfpConvTBiasExponents)];
    memcpy(biasExponents, kBfpConvTBiasExponents, sizeof(biasExponents));
    bfpQConfig_t biasQC = {.exponents = biasExponents,
                           .numGroups = 1,
                           .groupSize = 0,
                           .roundingMode = HALF_AWAY,
                           .mantissaBits = (uint8_t)kBfpConvTBiasMantissaBits,
                           .exponentBits = (uint8_t)kBfpConvTBiasExponentBits};
    quantization_t biasQ;
    initBfpQuantization(&biasQC, &biasQ);
    setTensorValues(&biasTensor, (uint8_t *)kBfpConvTBiasCodes, &biasShape, &biasQ, NULL);

    tensor_t outTensor;
    float outData[28]; /* batch * outChannels * kBfpConvTDilOutLen */
    size_t outDims[] = {(size_t)kBfpConvTBatch, (size_t)kBfpConvTOutChannels,
                        (size_t)kBfpConvTDilOutLen};
    size_t outOrder[] = {0, 1, 2};
    shape_t outShape;
    setShape(&outShape, outDims, 3, outOrder);
    quantization_t outQ;
    initFloat32Quantization(&outQ);
    setTensorValues(&outTensor, (uint8_t *)outData, &outShape, &outQ, NULL);

    kernel_t kernel;
    initKernel(&kernel, (size_t)kBfpConvTKernelSize, VALID, (size_t)kBfpConvTDilDilation,
               (size_t)kBfpConvTStride);

    convTranspose1dKernelBfpGather(&inTensor, &wTensor, &biasTensor, &kernel, 1,
                                   (size_t)kBfpConvTOutputPadding, &outTensor);

    TEST_ASSERT_EQUAL_MEMORY(kBfpConvTDilExpected, outTensor.data,
                             kBfpConvTDilExpected_len * sizeof(float));
}

/* BFP epic PR2, the ConvT sibling of UnitTestMatmul.c's
 * testMatmulBfpGroupedBiasBindsPerGroupExponent: same bias VALUES stored
 * grouped {numGroups=2, groupSize=1} with non-uniform exponents (goldgen
 * asserts a group-0 collapse differs), expected output bit-identical to the
 * per-tensor gold -- including the tap-free outputPadding tail, which is
 * PURE bias seed and therefore the most direct group-binding probe. */
void testConvTranspose1dKernelBfpGroupedBiasBindsPerGroupExponent(void) {
    tensor_t inTensor;
    size_t inDims[3], inOrder[3];
    shape_t inShape;
    uint8_t inExponents[sizeof(kBfpConvTInExponents)];
    memcpy(inExponents, kBfpConvTInExponents, sizeof(inExponents));
    bfpQConfig_t inQC;
    quantization_t inQ;
    setupBfpConvTInput(&inTensor, &inShape, inDims, inOrder, &inQC, &inQ, inExponents);

    tensor_t wTensor;
    size_t wDims[3], wOrder[3];
    shape_t wShape;
    uint8_t wExponents[sizeof(kBfpConvTWExponents)];
    memcpy(wExponents, kBfpConvTWExponents, sizeof(wExponents));
    bfpQConfig_t wQC;
    quantization_t wQ;
    setupBfpConvTWeight(&wTensor, &wShape, wDims, wOrder, &wQC, &wQ, wExponents);

    tensor_t biasTensor;
    size_t biasDims[] = {(size_t)kBfpConvTOutChannels};
    size_t biasOrder[] = {0};
    shape_t biasShape;
    setShape(&biasShape, biasDims, 1, biasOrder);
    uint8_t biasExponents[sizeof(kBfpConvTBiasGroupedExponents)];
    memcpy(biasExponents, kBfpConvTBiasGroupedExponents, sizeof(biasExponents));
    bfpQConfig_t biasQC = {.exponents = biasExponents,
                           .numGroups = (size_t)kBfpConvTBiasGroupedNumGroups,
                           .groupSize = 1,
                           .roundingMode = HALF_AWAY,
                           .mantissaBits = (uint8_t)kBfpConvTBiasMantissaBits,
                           .exponentBits = (uint8_t)kBfpConvTBiasExponentBits};
    quantization_t biasQ;
    initBfpQuantization(&biasQC, &biasQ);
    setTensorValues(&biasTensor, (uint8_t *)kBfpConvTBiasGroupedCodes, &biasShape, &biasQ, NULL);

    tensor_t outTensor;
    float outData[24];
    size_t outDims[3], outOrder[3];
    shape_t outShape;
    quantization_t outQ;
    setupBfpConvTOutput(&outTensor, &outShape, outDims, outOrder, &outQ, outData);

    kernel_t kernel;
    initKernel(&kernel, (size_t)kBfpConvTKernelSize, VALID, 1, (size_t)kBfpConvTStride);

    convTranspose1dKernelBfpGather(&inTensor, &wTensor, &biasTensor, &kernel, 1,
                                   (size_t)kBfpConvTOutputPadding, &outTensor);

    TEST_ASSERT_EQUAL_MEMORY(kBfpConvTExpected, outTensor.data,
                             kBfpConvTExpected_len * sizeof(float));
}

/* BFP power-of-two twin, mirroring UnitTestConv1dKernel.c's
 * testConv1dKernelBfpPowerOfTwoBitIdenticalToGroupedSym at the ConvT gather:
 * identical mantissas (the gold fixture's codes); BFP input per-tensor stored
 * 126 (2^-1 == 0.5f) <-> SYM inScale 0.5f; BFP weight grouped {numGroups=3,
 * groupSize=8} with every stored exponent 125 (2^-2 == 0.25f) <-> SYM
 * weightGroups scales all 0.25f (sAcc = 0.125f, per-product rescale factor
 * exactly 1.0). WHY bit-identity holds although the SYM kernel SCATTERS
 * (adds in (ic, inPos, k) order) and the BFP kernel GATHERS (adds in
 * (tap, ic) order): every product is an integer times a power-of-two scale,
 * |products| <= 24*6 and per-element sums <= ~1300 << 2^24, so EVERY
 * intermediate (partials, folds, accumulator states, the final dequant
 * multiply by 0.125f) is exactly representable in float32 -- no step ever
 * rounds, and exact float addition is associative/commutative, making the
 * add ORDER irrelevant. NULL bias on both sides (the SYM bias path rounds
 * through rescaleIntoAccumulatorScale and is not part of the twin claim).
 * The outputPadding tail (outPos 11) compares +0.0f against +0.0f. */
void testConvTranspose1dKernelBfpGatherPowerOfTwoBitIdenticalToGroupedSym(void) {
    size_t inDims[3], inOrder[3], wDims[3], wOrder[3], outDims[3], outOrder[3];
    shape_t inShape, wShape, outShape;

    kernel_t kernel;
    initKernel(&kernel, (size_t)kBfpConvTKernelSize, VALID, 1, (size_t)kBfpConvTStride);

    tensor_t inBfpTensor;
    uint8_t inExponents[] = {126}; /* 2^(126-127) == 0.5f */
    bfpQConfig_t inBfpQC = {.exponents = inExponents,
                            .numGroups = 1,
                            .groupSize = 0,
                            .roundingMode = HALF_AWAY,
                            .mantissaBits = 6,
                            .exponentBits = 8};
    quantization_t inBfpQ;
    inDims[0] = (size_t)kBfpConvTBatch;
    inDims[1] = (size_t)kBfpConvTInChannels;
    inDims[2] = (size_t)kBfpConvTInputLength;
    inOrder[0] = 0;
    inOrder[1] = 1;
    inOrder[2] = 2;
    setShape(&inShape, inDims, 3, inOrder);
    initBfpQuantization(&inBfpQC, &inBfpQ);
    setTensorValues(&inBfpTensor, (uint8_t *)kBfpConvTInCodes, &inShape, &inBfpQ, NULL);

    tensor_t wBfpTensor;
    uint8_t wExponents[] = {125, 125, 125}; /* 2^(125-127) == 0.25f */
    bfpQConfig_t wBfpQC = {.exponents = wExponents,
                           .numGroups = 3,
                           .groupSize = 8,
                           .roundingMode = HALF_AWAY,
                           .mantissaBits = 4,
                           .exponentBits = 8};
    quantization_t wBfpQ;
    wDims[0] = (size_t)kBfpConvTInChannels;
    wDims[1] = (size_t)kBfpConvTOutChannels;
    wDims[2] = (size_t)kBfpConvTKernelSize;
    wOrder[0] = 0;
    wOrder[1] = 1;
    wOrder[2] = 2;
    setShape(&wShape, wDims, 3, wOrder);
    initBfpQuantization(&wBfpQC, &wBfpQ);
    setTensorValues(&wBfpTensor, (uint8_t *)kBfpConvTWCodes, &wShape, &wBfpQ, NULL);

    tensor_t outBfpTensor;
    float outBfpData[24];
    quantization_t outBfpQ;
    outDims[0] = (size_t)kBfpConvTBatch;
    outDims[1] = (size_t)kBfpConvTOutChannels;
    outDims[2] = (size_t)kBfpConvTOutLen;
    outOrder[0] = 0;
    outOrder[1] = 1;
    outOrder[2] = 2;
    setShape(&outShape, outDims, 3, outOrder);
    initFloat32Quantization(&outBfpQ);
    setTensorValues(&outBfpTensor, (uint8_t *)outBfpData, &outShape, &outBfpQ, NULL);

    convTranspose1dKernelBfpGather(&inBfpTensor, &wBfpTensor, NULL, &kernel, 1,
                                   (size_t)kBfpConvTOutputPadding, &outBfpTensor);

    tensor_t inSymTensor;
    symInt32QConfig_t inSymQC;
    initSymInt32QConfig(HALF_AWAY, &inSymQC);
    inSymQC.scale = odtTestScale(0.5f);
    quantization_t inSymQ;
    initSymInt32Quantization(&inSymQC, &inSymQ);
    setTensorValues(&inSymTensor, (uint8_t *)kBfpConvTInCodes, &inShape, &inSymQ, NULL);

    tensor_t wSymTensor;
    symInt32QConfig_t wSymQC;
    initSymInt32QConfig(HALF_AWAY, &wSymQC);
    wSymQC.scale =
        odtTestScale(1.0f); /* poisoned scratch scale -- never read, scales live in weightGroups */
    quantization_t wSymQ;
    initSymInt32Quantization(&wSymQC, &wSymQ);
    setTensorValues(&wSymTensor, (uint8_t *)kBfpConvTWCodes, &wShape, &wSymQ, NULL);

    odtScale_t scales[3] = {odtTestScale(0.25f), odtTestScale(0.25f), odtTestScale(0.25f)};
    symQConfig_t weightGroups = {
        .scales = scales, .numGroups = 3, .groupSize = 8, .qBits = 8, .roundingMode = HALF_AWAY};

    tensor_t outSymTensor;
    int32_t outSymData[24];
    symInt32QConfig_t outSymQC;
    initSymInt32QConfig(HALF_AWAY, &outSymQC);
    quantization_t outSymQ;
    initSymInt32Quantization(&outSymQC, &outSymQ);
    setTensorValues(&outSymTensor, (uint8_t *)outSymData, &outShape, &outSymQ, NULL);

    symGroupedWeight_t groupedWeight = symGroupedWeightSym(&weightGroups);
    convTranspose1dKernelSymInt32Grouped(&inSymTensor, &wSymTensor, NULL, &kernel, 1,
                                         (size_t)kBfpConvTOutputPadding, &outSymTensor,
                                         &groupedWeight, NULL);

    for (size_t i = 0; i < 24; i++) {
        float symDequant = (float)outSymData[i] * odtScaleToFloat(outSymQC.scale);
        TEST_ASSERT_EQUAL_MEMORY(&symDequant, &outBfpData[i], sizeof(float));
    }
}

/* Geometry parity (BFP D9): the gather must resolve the adjoint-SAME geometry
 * exactly like the scatter kernels (shared convT1dResolveGeometry) -- same
 * fixture as testConvTranspose1dKernelSamePaddingSymmetric (K=3, SAME,
 * stride 1, Lin=5, padLeft recovered as 1), run BOTH the FLOAT32 scatter
 * (on the float values) and the BFP gather (on the same values as unit-scale
 * mantissas, stored exponent 127) and require BIT-identical output: integer
 * products, sums <= 14 -- exact regime, so any divergence is a geometry/tap
 * error, never rounding. */
void testConvTranspose1dKernelBfpGatherAdjointSameParityWithScatter(void) {
    float lossGradData[] = {1, 1, 1, 1, 1};
    size_t lossGradDims[] = {1, 1, 5};
    tensor_t *lossGrad = makeFloatTensor(lossGradDims, 3, lossGradData);

    float weightData[] = {2, 4, 8};
    size_t weightDims[] = {1, 1, 3};
    tensor_t *weight = makeFloatTensor(weightDims, 3, weightData);

    size_t propLossDims[] = {1, 1, 5};
    tensor_t *propLoss = makeFloatTensor(propLossDims, 3, NULL);

    kernel_t kernel;
    initKernel(&kernel, 3, SAME, 1, 1);

    convTranspose1dKernelFloat32(lossGrad, weight, NULL, &kernel, 1, 0, propLoss);

    tensor_t inTensor;
    int32_t inCodes[] = {1, 1, 1, 1, 1};
    size_t inDims[] = {1, 1, 5};
    size_t inOrder[] = {0, 1, 2};
    shape_t inShape;
    setShape(&inShape, inDims, 3, inOrder);
    uint8_t inExponents[] = {127}; /* scale 1.0f: mantissas ARE the float values */
    bfpQConfig_t inQC = {.exponents = inExponents,
                         .numGroups = 1,
                         .groupSize = 0,
                         .roundingMode = HALF_AWAY,
                         .mantissaBits = 8,
                         .exponentBits = 8};
    quantization_t inQ;
    initBfpQuantization(&inQC, &inQ);
    setTensorValues(&inTensor, (uint8_t *)inCodes, &inShape, &inQ, NULL);

    tensor_t wTensor;
    int32_t wCodes[] = {2, 4, 8};
    size_t wDims[] = {1, 1, 3};
    size_t wOrder[] = {0, 1, 2};
    shape_t wShape;
    setShape(&wShape, wDims, 3, wOrder);
    uint8_t wExponents[] = {127};
    bfpQConfig_t wQC = {.exponents = wExponents,
                        .numGroups = 1,
                        .groupSize = 0,
                        .roundingMode = HALF_AWAY,
                        .mantissaBits = 8,
                        .exponentBits = 8};
    quantization_t wQ;
    initBfpQuantization(&wQC, &wQ);
    setTensorValues(&wTensor, (uint8_t *)wCodes, &wShape, &wQ, NULL);

    tensor_t outTensor;
    float outData[5];
    size_t outDims[] = {1, 1, 5};
    size_t outOrder[] = {0, 1, 2};
    shape_t outShape;
    setShape(&outShape, outDims, 3, outOrder);
    quantization_t outQ;
    initFloat32Quantization(&outQ);
    setTensorValues(&outTensor, (uint8_t *)outData, &outShape, &outQ, NULL);

    convTranspose1dKernelBfpGather(&inTensor, &wTensor, NULL, &kernel, 1, 0, &outTensor);

    TEST_ASSERT_EQUAL_MEMORY(propLoss->data, outTensor.data, 5 * sizeof(float));
}

/* Geometry parity, validation side: a VALID output-length mismatch must die
 * in the gather exactly as it does in the scatter kernels (the shared
 * convT1dResolveGeometry): Lout for Lin=3, K=2, stride 1, outputPadding 0 is
 * 4, the tensor claims 5. */
void testConvTranspose1dKernelBfpGatherValidLengthMismatchDies(void) {
    tensor_t inTensor;
    int32_t inCodes[] = {1, 1, 1};
    size_t inDims[] = {1, 1, 3};
    size_t inOrder[] = {0, 1, 2};
    shape_t inShape;
    setShape(&inShape, inDims, 3, inOrder);
    uint8_t inExponents[] = {127};
    bfpQConfig_t inQC = {.exponents = inExponents,
                         .numGroups = 1,
                         .groupSize = 0,
                         .roundingMode = HALF_AWAY,
                         .mantissaBits = 8,
                         .exponentBits = 8};
    quantization_t inQ;
    initBfpQuantization(&inQC, &inQ);
    setTensorValues(&inTensor, (uint8_t *)inCodes, &inShape, &inQ, NULL);

    tensor_t wTensor;
    int32_t wCodes[] = {1, 1};
    size_t wDims[] = {1, 1, 2};
    size_t wOrder[] = {0, 1, 2};
    shape_t wShape;
    setShape(&wShape, wDims, 3, wOrder);
    uint8_t wExponents[] = {127};
    bfpQConfig_t wQC = {.exponents = wExponents,
                        .numGroups = 1,
                        .groupSize = 0,
                        .roundingMode = HALF_AWAY,
                        .mantissaBits = 8,
                        .exponentBits = 8};
    quantization_t wQ;
    initBfpQuantization(&wQC, &wQ);
    setTensorValues(&wTensor, (uint8_t *)wCodes, &wShape, &wQ, NULL);

    tensor_t outTensor;
    float outData[5];
    size_t outDims[] = {1, 1, 5}; /* expected VALID Lout is 4 */
    size_t outOrder[] = {0, 1, 2};
    shape_t outShape;
    setShape(&outShape, outDims, 3, outOrder);
    quantization_t outQ;
    initFloat32Quantization(&outQ);
    setTensorValues(&outTensor, (uint8_t *)outData, &outShape, &outQ, NULL);

    kernel_t kernel;
    initKernel(&kernel, 2, VALID, 1, 1);

    ASSERT_EXITS_WITH_FAILURE(
        convTranspose1dKernelBfpGather(&inTensor, &wTensor, NULL, &kernel, 1, 0, &outTensor));
}

void testConvTranspose1dKernelBfpGatherHeadroomGuardDies(void) {
    /* per-tensor m=16 operands with reduction Cin/groups * K = 2 >
     * bfpSegmentLimit(16, 16) == 1 -- boundary-tight on purpose: a limit+1
     * off-by-one in the guard would let exactly this segment length through
     * (3 would still die under that mutation). */
    tensor_t inTensor;
    int32_t inData[] = {1, 1, 1, 1};
    size_t inDims[] = {1, 1, 4};
    size_t inOrder[] = {0, 1, 2};
    shape_t inShape;
    setShape(&inShape, inDims, 3, inOrder);
    uint8_t inExponents[] = {127};
    bfpQConfig_t inQC = {.exponents = inExponents,
                         .numGroups = 1,
                         .groupSize = 0,
                         .roundingMode = HALF_AWAY,
                         .mantissaBits = 16,
                         .exponentBits = 8};
    quantization_t inQ;
    initBfpQuantization(&inQC, &inQ);
    setTensorValues(&inTensor, (uint8_t *)inData, &inShape, &inQ, NULL);

    tensor_t wTensor;
    int32_t wData[] = {1, 1};
    size_t wDims[] = {1, 1, 2};
    size_t wOrder[] = {0, 1, 2};
    shape_t wShape;
    setShape(&wShape, wDims, 3, wOrder);
    uint8_t wExponents[] = {127};
    bfpQConfig_t wQC = {.exponents = wExponents,
                        .numGroups = 1,
                        .groupSize = 0,
                        .roundingMode = HALF_AWAY,
                        .mantissaBits = 16,
                        .exponentBits = 8};
    quantization_t wQ;
    initBfpQuantization(&wQC, &wQ);
    setTensorValues(&wTensor, (uint8_t *)wData, &wShape, &wQ, NULL);

    tensor_t outTensor;
    float outData[5];
    size_t outDims[] = {1, 1, 5}; /* (4-1)*1 + 1 + 0 + 1 */
    size_t outOrder[] = {0, 1, 2};
    shape_t outShape;
    setShape(&outShape, outDims, 3, outOrder);
    quantization_t outQ;
    initFloat32Quantization(&outQ);
    setTensorValues(&outTensor, (uint8_t *)outData, &outShape, &outQ, NULL);

    kernel_t kernel;
    initKernel(&kernel, 2, VALID, 1, 1);

    ASSERT_EXITS_WITH_FAILURE(
        convTranspose1dKernelBfpGather(&inTensor, &wTensor, NULL, &kernel, 1, 0, &outTensor));
}

/* Group-shape fail-fast (Task 3/4 precedent): bfpGroupOf divides by groupSize
 * with no relation to numGroups, so a mismatched config ({numGroups=2,
 * groupSize=4} on 20 elements: 2*4 == 8 != 20) would silently read
 * exponents[] out of bounds. The kernel must route every operand through
 * validateBfpQConfigShape before touching data. */
void testConvTranspose1dKernelBfpGatherRejectsMismatchedGroupShape(void) {
    tensor_t inTensor;
    int32_t inData[20] = {0};
    size_t inDims[] = {1, 2, 10};
    size_t inOrder[] = {0, 1, 2};
    shape_t inShape;
    setShape(&inShape, inDims, 3, inOrder);
    uint8_t inExponents[] = {127, 127};
    bfpQConfig_t inQC = {.exponents = inExponents,
                         .numGroups = 2,
                         .groupSize = 4, /* 2*4 == 8 != 20 elements */
                         .roundingMode = HALF_AWAY,
                         .mantissaBits = 6,
                         .exponentBits = 8};
    quantization_t inQ;
    initBfpQuantization(&inQC, &inQ);
    setTensorValues(&inTensor, (uint8_t *)inData, &inShape, &inQ, NULL);

    tensor_t wTensor;
    int32_t wData[6] = {0};
    size_t wDims[] = {2, 1, 3}; /* [Cin=2, Cout/groups=1, K=3] */
    size_t wOrder[] = {0, 1, 2};
    shape_t wShape;
    setShape(&wShape, wDims, 3, wOrder);
    uint8_t wExponents[] = {127};
    bfpQConfig_t wQC = {.exponents = wExponents,
                        .numGroups = 1,
                        .groupSize = 0,
                        .roundingMode = HALF_AWAY,
                        .mantissaBits = 4,
                        .exponentBits = 8};
    quantization_t wQ;
    initBfpQuantization(&wQC, &wQ);
    setTensorValues(&wTensor, (uint8_t *)wData, &wShape, &wQ, NULL);

    tensor_t outTensor;
    float outData[12];
    size_t outDims[] = {1, 1, 12}; /* (10-1)*1 + 2 + 0 + 1 */
    size_t outOrder[] = {0, 1, 2};
    shape_t outShape;
    setShape(&outShape, outDims, 3, outOrder);
    quantization_t outQ;
    initFloat32Quantization(&outQ);
    setTensorValues(&outTensor, (uint8_t *)outData, &outShape, &outQ, NULL);

    kernel_t kernel;
    initKernel(&kernel, 3, VALID, 1, 1);

    ASSERT_EXITS_WITH_FAILURE(
        convTranspose1dKernelBfpGather(&inTensor, &wTensor, NULL, &kernel, 1, 0, &outTensor));
}

/* conv-groups=2 parity gold (#416, BFP PR3 Task 4): SPLIT channels
 * (inChPerGroup = 4, outChPerGroup = 2, not depthwise), so the gather's
 * inLo/oc_offset channel arithmetic is load-bearing -- the PR2 fixtures all
 * ran groups=1. Same stack-fixture idiom and VALID stride-2 outputPadding-1
 * geometry as testConvTranspose1dKernelBfpGatherMatchesGold, gold from
 * convT1d_bfp_gather_ref(conv_groups=2) with the full vacuity self-checks
 * under split channels (the generator's asymmetric-scaling pin keeps a
 * group-blind gather observable). */
void testConvTranspose1dKernelBfpGatherConvGroups2MatchesGold(void) {
    tensor_t inTensor;
    size_t inDims[] = {(size_t)kBfpConvTBatch, (size_t)kBfpConvTG2InChannels,
                       (size_t)kBfpConvTInputLength};
    size_t inOrder[] = {0, 1, 2};
    shape_t inShape;
    setShape(&inShape, inDims, 3, inOrder);
    uint8_t inExponents[sizeof(kBfpConvTG2InExponents)];
    memcpy(inExponents, kBfpConvTG2InExponents, sizeof(inExponents));
    bfpQConfig_t inQC = {.exponents = inExponents,
                         .numGroups = (size_t)kBfpConvTG2InNumGroups,
                         .groupSize = (size_t)kBfpConvTG2InGroupSize,
                         .roundingMode = HALF_AWAY,
                         .mantissaBits = (uint8_t)kBfpConvTG2InMantissaBits,
                         .exponentBits = (uint8_t)kBfpConvTG2InExponentBits};
    quantization_t inQ;
    initBfpQuantization(&inQC, &inQ);
    setTensorValues(&inTensor, (uint8_t *)kBfpConvTG2InCodes, &inShape, &inQ, NULL);

    tensor_t wTensor;
    size_t wDims[] = {(size_t)kBfpConvTG2InChannels,
                      (size_t)kBfpConvTG2OutChannels / (size_t)kBfpConvTG2ConvGroups,
                      (size_t)kBfpConvTKernelSize};
    size_t wOrder[] = {0, 1, 2};
    shape_t wShape;
    setShape(&wShape, wDims, 3, wOrder);
    uint8_t wExponents[sizeof(kBfpConvTG2WExponents)];
    memcpy(wExponents, kBfpConvTG2WExponents, sizeof(wExponents));
    bfpQConfig_t wQC = {.exponents = wExponents,
                        .numGroups = (size_t)kBfpConvTG2WNumGroups,
                        .groupSize = (size_t)kBfpConvTG2WGroupSize,
                        .roundingMode = HALF_AWAY,
                        .mantissaBits = (uint8_t)kBfpConvTG2WMantissaBits,
                        .exponentBits = (uint8_t)kBfpConvTG2WExponentBits};
    quantization_t wQ;
    initBfpQuantization(&wQC, &wQ);
    setTensorValues(&wTensor, (uint8_t *)kBfpConvTG2WCodes, &wShape, &wQ, NULL);

    tensor_t biasTensor;
    size_t biasDims[] = {(size_t)kBfpConvTG2OutChannels};
    size_t biasOrder[] = {0};
    shape_t biasShape;
    setShape(&biasShape, biasDims, 1, biasOrder);
    uint8_t biasExponents[sizeof(kBfpConvTG2BiasExponents)];
    memcpy(biasExponents, kBfpConvTG2BiasExponents, sizeof(biasExponents));
    bfpQConfig_t biasQC = {.exponents = biasExponents,
                           .numGroups = 1,
                           .groupSize = 0,
                           .roundingMode = HALF_AWAY,
                           .mantissaBits = (uint8_t)kBfpConvTG2BiasMantissaBits,
                           .exponentBits = (uint8_t)kBfpConvTG2BiasExponentBits};
    quantization_t biasQ;
    initBfpQuantization(&biasQC, &biasQ);
    setTensorValues(&biasTensor, (uint8_t *)kBfpConvTG2BiasCodes, &biasShape, &biasQ, NULL);

    tensor_t outTensor;
    float outData[48]; /* batch * kBfpConvTG2OutChannels * kBfpConvTOutLen */
    size_t outDims[] = {(size_t)kBfpConvTBatch, (size_t)kBfpConvTG2OutChannels,
                        (size_t)kBfpConvTOutLen};
    size_t outOrder[] = {0, 1, 2};
    shape_t outShape;
    setShape(&outShape, outDims, 3, outOrder);
    quantization_t outQ;
    initFloat32Quantization(&outQ);
    setTensorValues(&outTensor, (uint8_t *)outData, &outShape, &outQ, NULL);

    kernel_t kernel;
    initKernel(&kernel, (size_t)kBfpConvTKernelSize, VALID, 1, (size_t)kBfpConvTStride);

    convTranspose1dKernelBfpGather(&inTensor, &wTensor, &biasTensor, &kernel,
                                   (size_t)kBfpConvTG2ConvGroups, (size_t)kBfpConvTOutputPadding,
                                   &outTensor);

    TEST_ASSERT_EQUAL_MEMORY(kBfpConvTG2Expected, outTensor.data,
                             kBfpConvTG2Expected_len * sizeof(float));
}

void setUp() {}
void tearDown() {}

/* ---- SYM int32 accumulator guard (SymAccumGuard.h) ---------------------- */

#define CT_MAXN 1100
static int32_t gCtX[CT_MAXN];
static int32_t gCtW[CT_MAXN];
static int32_t gCtY[CT_MAXN * 2];
static uint8_t gCtOutBits; /* output qMaxBits after the kernel returned */

/* Operand whose storage dtype ctRunOutCh swaps to FLOAT32 (dtype-validation tests). */
typedef enum { CT_POISON_NONE, CT_POISON_X, CT_POISON_W, CT_POISON_BIAS, CT_POISON_Y } ctPoison_t;
static ctPoison_t gCtPoison = CT_POISON_NONE;

/* Code domain of the grouped weight ctRunOutCh hands the grouped kernel (a
 * stored SYM weight unless a test says ASYM image). */
static symCodeDomain_t gCtWeightDomain = SYM_CODES_TWOS_COMPLEMENT;

/* Output wires start at a width no stamp test expects, so a missing stamp can
 * never pass as the right one (the kernels never read the output width). */
#define UNSTAMPED_WIDTH 5

static void ctFill(size_t n, int32_t xCode, int32_t wCode, bool wAlternates) {
    for (size_t i = 0; i < n; i++) {
        gCtX[i] = xCode;
        gCtW[i] = (wAlternates && (i % 2 == 1)) ? -wCode : wCode;
    }
}

#define CT_MAX_OUT_CH 2

/* x [1, inCh, len], w [inCh, outCh, K], VALID stride 1 -> y [1, outCh, len + K - 1];
 * result in gCtY. biasCodes: NULL or one code per output channel. numGroups
 * == 0 -> per-tensor weight kernel, else the grouped kernel with equal scales. */
static void ctRunOutCh(size_t inCh, size_t len, size_t k, uint8_t xBits, uint8_t wBits,
                       size_t outCh, const int32_t *biasCodes, size_t numGroups) {
    TEST_ASSERT_TRUE(outCh <= CT_MAX_OUT_CH);
    size_t outLen = len + k - 1;
    size_t xDims[] = {1, inCh, len}, wDims[] = {inCh, outCh, k}, yDims[] = {1, outCh, outLen};
    size_t order[] = {0, 1, 2}, biasDims[] = {outCh};
    shape_t xS, wS, yS, biasS;
    setShape(&xS, xDims, 3, order);
    setShape(&wS, wDims, 3, order);
    setShape(&yS, yDims, 3, order);
    setShape(&biasS, biasDims, 1, order);
    symInt32QConfig_t xQC, wQC, yQC, biasQC;
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &xQC, xBits);
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &wQC, wBits);
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &yQC, UNSTAMPED_WIDTH);
    initSymInt32QConfig(HALF_AWAY, &biasQC);
    quantization_t xQ, wQ, yQ, biasQ;
    initSymInt32Quantization(&xQC, &xQ);
    initSymInt32Quantization(&wQC, &wQ);
    initSymInt32Quantization(&yQC, &yQ);
    initSymInt32Quantization(&biasQC, &biasQ);
    if (gCtPoison == CT_POISON_X) {
        initFloat32Quantization(&xQ);
    }
    if (gCtPoison == CT_POISON_W) {
        initFloat32Quantization(&wQ);
    }
    if (gCtPoison == CT_POISON_BIAS) {
        initFloat32Quantization(&biasQ);
    }
    if (gCtPoison == CT_POISON_Y) {
        initFloat32Quantization(&yQ);
    }
    int32_t biasData[CT_MAX_OUT_CH] = {0};
    for (size_t oc = 0; biasCodes != NULL && oc < outCh; oc++) {
        biasData[oc] = biasCodes[oc];
    }
    tensor_t x, w, yT, bias;
    setTensorValues(&x, (uint8_t *)gCtX, &xS, &xQ, NULL);
    setTensorValues(&w, (uint8_t *)gCtW, &wS, &wQ, NULL);
    setTensorValues(&yT, (uint8_t *)gCtY, &yS, &yQ, NULL);
    setTensorValues(&bias, (uint8_t *)biasData, &biasS, &biasQ, NULL);
    kernel_t kernel;
    initKernel(&kernel, k, VALID, 1, 1);
    if (numGroups == 0) {
        convTranspose1dKernelSymInt32(&x, &w, biasCodes ? &bias : NULL, &kernel, 1, 0, &yT, NULL);
    } else {
        odtScale_t scales[8] = {odtTestScale(1), odtTestScale(1), odtTestScale(1), odtTestScale(1),
                                odtTestScale(1), odtTestScale(1), odtTestScale(1), odtTestScale(1)};
        symQConfig_t groups = {.scales = scales,
                               .numGroups = numGroups,
                               .groupSize = inCh * outCh * k / numGroups,
                               .qBits = wBits,
                               .roundingMode = HALF_AWAY};
        symGroupedWeight_t groupedWeight = {.groups = &groups, .domain = gCtWeightDomain};
        convTranspose1dKernelSymInt32Grouped(&x, &w, biasCodes ? &bias : NULL, &kernel, 1, 0, &yT,
                                             &groupedWeight, NULL);
    }
    gCtOutBits = yQC.qMaxBits;
}

/* One output channel and one bias code: the shape every guard test uses. */
static void ctRun(size_t inCh, size_t len, size_t k, uint8_t xBits, uint8_t wBits,
                  const int32_t *biasCode, size_t numGroups) {
    ctRunOutCh(inCh, len, k, xBits, wBits, 1, biasCode, numGroups);
}

/* ctRun with the grouped weight read as a stored ASYM weight's image. */
static void ctRunAsymImage(size_t inCh, size_t len, size_t k, uint8_t xBits, uint8_t wBits,
                           const int32_t *biasCode, size_t numGroups) {
    gCtWeightDomain = SYM_CODES_ASYM_IMAGE;
    ctRun(inCh, len, k, xBits, wBits, biasCode, numGroups);
    gCtWeightDomain = SYM_CODES_TWOS_COMPLEMENT;
}

/* y[o] = bias + sum over (ic, inPos, tap) with inPos + tap == o of x * w, in int64 */
static void ctExpected(size_t inCh, size_t len, size_t k, int32_t bias, int32_t *out) {
    for (size_t o = 0; o < len + k - 1; o++) {
        int64_t sum = bias;
        for (size_t ic = 0; ic < inCh; ic++) {
            for (size_t tap = 0; tap < k; tap++) {
                if (o >= tap && o - tap < len) {
                    sum += (int64_t)gCtX[ic * len + (o - tap)] * gCtW[ic * k + tap];
                }
            }
        }
        out[o] = (int32_t)sum;
    }
}

static void ctAssertExact(size_t inCh, size_t len, size_t k, int32_t bias) {
    int32_t expected[CT_MAXN * 2];
    ctExpected(inCh, len, k, bias, expected);
    TEST_ASSERT_EQUAL_INT32_ARRAY(expected, gCtY, len + k - 1);
}

void testConvTranspose1dKernelSymGuardProvenSafeN511RunsExact(void) {
    ctFill(511, -2048, -2048, false);
    ctRun(511, 1, 1, 12, 12, NULL, 0);
    TEST_ASSERT_EQUAL_INT32(511 * 4194304, gCtY[0]);
}

void testConvTranspose1dKernelSymGuardDiesAtFirstWrappingNChannels(void) {
    ctFill(512, -2048, -2048, false);
    ASSERT_EXITS_WITH_OUTPUT(1, "convTranspose1dKernelSymInt32", ctRun(512, 1, 1, 12, 12, NULL, 0));
}

void testConvTranspose1dKernelSymGuardDiesAtN513PositiveCodes(void) {
    ctFill(513, 2047, 2047, false);
    ASSERT_EXITS_WITH_OUTPUT(1, "N=513", ctRun(513, 1, 1, 12, 12, NULL, 0));
}

void testConvTranspose1dKernelSymGuardCountsKernelTapsInN(void) {
    ctFill(512, -2048, -2048, false); /* inCh = 1: y[511] scatters 512 taps */
    ASSERT_EXITS_WITH_OUTPUT(1, "N=512", ctRun(1, 512, 512, 12, 12, NULL, 0));
}

void testConvTranspose1dKernelSymGuardCheckedPathStaysExactWhenValuesFit(void) {
    ctFill(601, 2047, 2047, true);
    ctRun(601, 1, 1, 12, 12, NULL, 0);
    ctAssertExact(601, 1, 1, 0);
}

void testConvTranspose1dKernelSymGuardPostScatterBiasIsCounted(void) {
    int32_t seed = 4194304; /* scatter alone = 511 * 2^22 fits; the bias pass tips it */
    ctFill(511, -2048, -2048, false);
    ASSERT_EXITS_WITH_OUTPUT(1, "max|seed|=4194304", ctRun(511, 1, 1, 12, 12, &seed, 0));
    int32_t small = 4194303;
    ctRun(511, 1, 1, 12, 12, &small, 0);
    TEST_ASSERT_EQUAL_INT32(511 * 4194304 + 4194303, gCtY[0]);
}

void testConvTranspose1dKernelSymGuardWidthsFollowTheOperandsNotTheDefault(void) {
    ctFill(31, -2048, -32768, false);
    ctRun(31, 1, 1, 12, 16, NULL, 0);
    TEST_ASSERT_EQUAL_INT32(31 * 67108864, gCtY[0]);
    ctFill(32, -2048, -32768, false);
    ASSERT_EXITS_WITH_OUTPUT(1, "widths (12, 16)", ctRun(32, 1, 1, 12, 16, NULL, 0));
}

void testConvTranspose1dKernelSymGuardWideProductOverflowAborts(void) {
    ctFill(1, 524288, 524288, false);
    ASSERT_EXITS_WITH_OUTPUT(1, "convTranspose1dKernelSymInt32", ctRun(1, 1, 1, 20, 20, NULL, 0));
    ctFill(2, 100, 200, false);
    ctRun(2, 1, 1, 20, 20, NULL, 0);
    TEST_ASSERT_EQUAL_INT32(2 * 20000, gCtY[0]);
}

void testConvTranspose1dKernelSymGroupedGuardDiesAtFirstWrappingN(void) {
    ctFill(1024, -2048, -2048, false);
    ASSERT_EXITS_WITH_OUTPUT(1, "convTranspose1dKernelSymInt32Grouped",
                             ctRun(1024, 1, 1, 12, 12, NULL, 4));
}

void testConvTranspose1dKernelSymGroupedGuardWidthsFollowTheInputWidth(void) {
    /* the input width follows its own declaration: int16 x int12 products are <= 2^26, so 31 terms
     * fit and 32 do not; the banner names a stored SYM int12 weight's exact range as 12 bits */
    ctFill(30, -32768, -2048, false);
    ctRun(30, 1, 1, 16, 12, NULL, 2);
    TEST_ASSERT_EQUAL_INT32(30 * 67108864, gCtY[0]);
    ctFill(32, -32768, -2048, false);
    ASSERT_EXITS_WITH_OUTPUT(1, "widths (16, 12)", ctRun(32, 1, 1, 16, 12, NULL, 2));
}

void testConvTranspose1dKernelSymGroupedGuardWideInputProductOverflowAborts(void) {
    /* 25-bit input x int12 weight: (2^24) * (2^11) = 2^35 leaves int32 on one product */
    ctFill(2, -16777216, -2048, false);
    ASSERT_EXITS_WITH_OUTPUT(1, "convTranspose1dKernelSymInt32Grouped",
                             ctRun(2, 1, 1, 25, 12, NULL, 2));
}

void testConvTranspose1dKernelSymGroupedGuardCountsKernelTapsInN(void) {
    ctFill(512, -2048, -2048, false); /* inCh = 1, K = 512 taps in 2 groups */
    ASSERT_EXITS_WITH_OUTPUT(1, "N=512", ctRun(1, 512, 512, 12, 12, NULL, 2));
}

void testConvTranspose1dKernelSymGroupedGuardCheckedPathStaysExactWhenValuesFit(void) {
    ctFill(600, 2047, 2047, true);
    ctRun(600, 1, 1, 12, 12, NULL, 3);
    ctAssertExact(600, 1, 1, 0);
}

/* 511 full-scale int12 terms (7 groups): a stored SYM weight's exact range
 * proves them (511 * 2^22 fits) and the safe scatter runs; a stored ASYM
 * weight's image (+-4095 reachable) is unprovable and the checked scatter runs
 * -- both exact. */
void testConvTranspose1dKernelSymGroupedGuardFullScaleN511RunsExact(void) {
    ctFill(511, -2048, -2048, false);
    ctRun(511, 1, 1, 12, 12, NULL, 7);
    TEST_ASSERT_EQUAL_INT32(511 * 4194304, gCtY[0]);
    ctRunAsymImage(511, 1, 1, 12, 12, NULL, 7);
    TEST_ASSERT_EQUAL_INT32(511 * 4194304, gCtY[0]);
}

void testConvTranspose1dKernelSymGroupedGuardPostScatterBiasIsCounted(void) {
    int32_t seed = 4194304;
    ctFill(511, -2048, -2048, false);
    ASSERT_EXITS_WITH_OUTPUT(1, "max|seed|=4194304", ctRun(511, 1, 1, 12, 12, &seed, 7));
}

/* A stored grouped ASYM weight unpacks to code - zp[g], reaching +-(2^qBits - 1)
 * = +-4095 at qBits 12; the kernel takes that range from the ctx's code domain:
 * 511 terms of 2047 x 4095 abort (the SYM range would have run them unchecked
 * and wrapped), 256 terms (2145911040) are proven at the exact image range and
 * run exact (the image's holding width 13 proves only 255). */
void testConvTranspose1dKernelSymGroupedGuardAsymImageWeightKeepsItsFullRange(void) {
    ctFill(511, 2047, 4095, false);
    ASSERT_EXITS_WITH_OUTPUT(1, "convTranspose1dKernelSymInt32Grouped",
                             ctRunAsymImage(511, 1, 1, 12, 12, NULL, 7));
    ctFill(256, 2047, 4095, false);
    ctRunAsymImage(256, 1, 1, 12, 12, NULL, 2);
    TEST_ASSERT_EQUAL_INT32(256 * 2047 * 4095, gCtY[0]);
}

/* A grouped weight wider than int12 used to exit at the validator; now it takes
 * the checked path (int12 x int16 products over 2 x 16 terms are unprovable).
 * 30 terms of -2048 x -32768 fit int32 through two groups; 32 terms (2^31)
 * wrap. */
void testConvTranspose1dKernelSymGroupedGuardWideWeightRunsExactWhenItFitsAndAbortsWhenItWraps(
    void) {
    ctFill(30, -2048, -32768, false);
    ctRun(30, 1, 1, 12, 16, NULL, 2);
    TEST_ASSERT_EQUAL_INT32(30 * 67108864, gCtY[0]);
    ctFill(32, -2048, -32768, false);
    ASSERT_EXITS_WITH_OUTPUT(1, "convTranspose1dKernelSymInt32Grouped",
                             ctRun(32, 1, 1, 12, 16, NULL, 2));
}

/* See the Matmul twin: each rescaled product is 1024 * 32767 here, so the
 * #189 reserve is not the issue; the scatter accumulates 80 of them (group 2
 * negated, one zero). The sum fits; all-positive wraps. */
static void wideGroupedFill(bool secondGroupNegative) {
    ctFill(80, 1024, 32767, false);
    if (secondGroupNegative) {
        for (size_t i = 40; i < 80; i++) {
            gCtW[i] = -32767;
        }
        gCtW[79] = 0;
    }
}

void testConvTranspose1dKernelSymGroupedWideScatterRunsExactWhenSumFits(void) {
    wideGroupedFill(true);
    ctRun(80, 1, 1, 12, 16, NULL, 2);
    TEST_ASSERT_EQUAL_INT32(1024 * 32767, gCtY[0]);
}

void testConvTranspose1dKernelSymGroupedWideScatterThatWrapsHitsTheF9Guard(void) {
    wideGroupedFill(false);
    ASSERT_EXITS_WITH_OUTPUT(1, "convTranspose1dKernelSymInt32Grouped",
                             ctRun(80, 1, 1, 12, 16, NULL, 2));
}

/* ---- #516: the output declares the accumulator's static width ------------ */

/* int4 x int4, 3 input channels scattering into one output element: the stamp
 * is the static range [-168, 192] -> 9, never the data (3). The bias lands in
 * a separate pass AFTER the scatter, so the scatter site alone would still say
 * 9; the stamp must count the seed the bias pass added, on the side its sign
 * picks: +1000 -> [832, 1192] -> 12, -1870 -> [-2038, -1678] -> 12 (max|seed|
 * on both sides, 2062, would say 13). */
void testConvTranspose1dKernelSymStampsResultWidthIncludesBiasSeed(void) {
    ctFill(3, 1, 1, false);
    ctRun(3, 1, 1, 4, 4, NULL, 0);
    TEST_ASSERT_EQUAL_INT32(3, gCtY[0]);
    symAccumSite_t site = symAccumSiteInit("t", 3, 4, 4, 0);
    TEST_ASSERT_EQUAL_UINT8(symAccumResultBits(&site), gCtOutBits);
    TEST_ASSERT_EQUAL_UINT8(9, gCtOutBits);
    int32_t bias = 1000;
    ctRun(3, 1, 1, 4, 4, &bias, 0);
    TEST_ASSERT_EQUAL_INT32(1003, gCtY[0]);
    symAccumSite_t seeded = symAccumSiteInit("t", 3, 4, 4, 1000);
    TEST_ASSERT_EQUAL_UINT8(symAccumResultBits(&seeded), gCtOutBits);
    TEST_ASSERT_EQUAL_UINT8(12, gCtOutBits);
    bias = -1870;
    ctRun(3, 1, 1, 4, 4, &bias, 0);
    TEST_ASSERT_EQUAL_INT32(3 - 1870, gCtY[0]);
    TEST_ASSERT_EQUAL_UINT8(12, gCtOutBits);
}

/* int4 x grouped int4 (2 groups x 4) over 8 scattered products, all scales
 * 1.0: the fold path with F = 8 (ConvTranspose1dKernel.c, SymGroupFold.h), so
 * each x folds to x * 2^(F-1) = 128x at A = 2^-7 and the stamp is the fold
 * site -- x widened to [-8 * 256 - 1, 7 * 256 + 1] = [-2049, 1793], no rescale
 * slack -- over the weight's EXACT code range. A stored SYM int4 weight is
 * [-8, 7]: one term spans [-14344, 16392], 8 terms [-114752, 131136] -> 19.
 * The post-scatter bias lands at A too (1024 -> seed 131072), lifting the
 * positive side to 262208 -> 20, so the stamp has to be the fold site WITH the
 * bias pass's seed (the weight counted one bit wide, [-16, 15], would say 20
 * seedless). A bias of -1200 (seed -153600) widens the negative side instead:
 * 114752 + 153600 = 268352 -> 20; -1023 (seed -130944) stays at 19. */
void testConvTranspose1dKernelSymGroupedStampsFoldWidthIncludesBiasSeed(void) {
    ctFill(8, 1, 1, false);
    ctRun(8, 1, 1, 4, 4, NULL, 2);
    TEST_ASSERT_EQUAL_INT32(8 * 128, gCtY[0]);
    symAccumSite_t fold = symAccumSiteInitRanges("t", 8, -2049, 1793, -8, 7, 0);
    TEST_ASSERT_EQUAL_UINT8(symAccumResultBits(&fold), gCtOutBits);
    TEST_ASSERT_EQUAL_UINT8(19, gCtOutBits);
    int32_t bias = 1024;
    ctRun(8, 1, 1, 4, 4, &bias, 2);
    TEST_ASSERT_EQUAL_INT32(1024 * 128 + 8 * 128, gCtY[0]);
    symAccumSite_t seeded = fold;
    symAccumSiteSetSeedRange(&seeded, 131072, 131072);
    TEST_ASSERT_EQUAL_UINT8(symAccumResultBits(&seeded), gCtOutBits);
    TEST_ASSERT_EQUAL_UINT8(20, gCtOutBits);
    bias = -1200;
    ctRun(8, 1, 1, 4, 4, &bias, 2);
    TEST_ASSERT_EQUAL_INT32(8 * 128 - 1200 * 128, gCtY[0]);
    TEST_ASSERT_EQUAL_UINT8(20, gCtOutBits);
    bias = -1023;
    ctRun(8, 1, 1, 4, 4, &bias, 2);
    TEST_ASSERT_EQUAL_UINT8(19, gCtOutBits);
}

/* The same scatter over a stored ASYM int4 weight's image [-15, 15]: each fold
 * term is +-30735, 8 terms 245880 per side -> 19 seedless (the image counted
 * one bit wide, [-16, 15], says 20); bias 500 (seed 64000) -> 309880 -> 20
 * (the SYM range, 131136 + 64000 = 195136, says 19). */
void testConvTranspose1dKernelSymGroupedAsymImageStampsItsFullRange(void) {
    ctFill(8, 1, 1, false);
    ctRunAsymImage(8, 1, 1, 4, 4, NULL, 2);
    TEST_ASSERT_EQUAL_INT32(8 * 128, gCtY[0]);
    TEST_ASSERT_EQUAL_UINT8(19, gCtOutBits);
    int32_t bias = 500;
    ctRunAsymImage(8, 1, 1, 4, 4, &bias, 2);
    TEST_ASSERT_EQUAL_INT32(500 * 128 + 8 * 128, gCtY[0]);
    TEST_ASSERT_EQUAL_UINT8(20, gCtOutBits);
}

/* The bias pass seeds one channel at a time, so the stamp must carry the
 * largest seed whichever channel holds it: 2 output channels x 3 int4 x int4
 * products (192 -> 9 alone) seeded 1000 / 1 stamp 1192 -> 12 in both channel
 * orders, where a stamp kept from the last (or the first) channel would read
 * 193 -> 9 in one of them. */
void testConvTranspose1dKernelSymStampCountsTheLargestBiasSeedOverChannels(void) {
    int32_t largeOnFirst[] = {1000, 1};
    ctFill(6, 1, 1, false);
    ctRunOutCh(3, 1, 1, 4, 4, 2, largeOnFirst, 0);
    TEST_ASSERT_EQUAL_INT32(1003, gCtY[0]);
    TEST_ASSERT_EQUAL_INT32(4, gCtY[1]);
    symAccumSite_t seeded = symAccumSiteInit("t", 3, 4, 4, 1000);
    TEST_ASSERT_EQUAL_UINT8(symAccumResultBits(&seeded), gCtOutBits);
    TEST_ASSERT_EQUAL_UINT8(12, gCtOutBits);
    int32_t largeOnLast[] = {1, 1000};
    ctRunOutCh(3, 1, 1, 4, 4, 2, largeOnLast, 0);
    TEST_ASSERT_EQUAL_INT32(4, gCtY[0]);
    TEST_ASSERT_EQUAL_INT32(1003, gCtY[1]);
    TEST_ASSERT_EQUAL_UINT8(12, gCtOutBits);
    /* each side takes its own extreme over the channels: seeds 1000 / -1870
     * span [-168 - 1870, 192 + 1000] = [-2038, 1192] -> 12, where the largest
     * magnitude on both sides (2062) would say 13 */
    int32_t mixedSigns[] = {1000, -1870};
    ctRunOutCh(3, 1, 1, 4, 4, 2, mixedSigns, 0);
    TEST_ASSERT_EQUAL_INT32(1003, gCtY[0]);
    TEST_ASSERT_EQUAL_INT32(3 - 1870, gCtY[1]);
    TEST_ASSERT_EQUAL_UINT8(12, gCtOutBits);
}

/* Operand dtype validation: a non-SYM_INT32 operand would be read through the
 * wrong qConfig struct, so each kernel must abort naming itself and the operand. */
static const int32_t kCtBias = 0;
#define CT_DTYPE_DEATH(name, poison, numGroups, bias, banner)                                      \
    void name(void) {                                                                              \
        ctFill(2, 3, 3, false);                                                                    \
        gCtPoison = poison;                                                                        \
        ASSERT_EXITS_WITH_OUTPUT(1, banner, ctRun(2, 1, 1, 12, 12, bias, numGroups));              \
        gCtPoison = CT_POISON_NONE;                                                                \
    }

CT_DTYPE_DEATH(testConvT1dKernelSymRejectsFloatInput, CT_POISON_X, 0, NULL,
               "convTranspose1dKernelSymInt32: input must be SYM_INT32")
CT_DTYPE_DEATH(testConvT1dKernelSymRejectsFloatWeight, CT_POISON_W, 0, NULL,
               "convTranspose1dKernelSymInt32: weight must be SYM_INT32")
CT_DTYPE_DEATH(testConvT1dKernelSymRejectsFloatBias, CT_POISON_BIAS, 0, &kCtBias,
               "convTranspose1dKernelSymInt32: bias must be SYM_INT32")
CT_DTYPE_DEATH(testConvT1dKernelSymRejectsFloatOutput, CT_POISON_Y, 0, NULL,
               "convTranspose1dKernelSymInt32: output must be SYM_INT32")
CT_DTYPE_DEATH(testConvT1dKernelSymGroupedRejectsFloatInput, CT_POISON_X, 2, NULL,
               "convTranspose1dKernelSymInt32Grouped: input must be SYM_INT32")
CT_DTYPE_DEATH(testConvT1dKernelSymGroupedRejectsFloatWeight, CT_POISON_W, 2, NULL,
               "convTranspose1dKernelSymInt32Grouped: weight must be SYM_INT32")
CT_DTYPE_DEATH(testConvT1dKernelSymGroupedRejectsFloatBias, CT_POISON_BIAS, 2, &kCtBias,
               "convTranspose1dKernelSymInt32Grouped: bias must be SYM_INT32")
CT_DTYPE_DEATH(testConvT1dKernelSymGroupedRejectsFloatOutput, CT_POISON_Y, 2, NULL,
               "convTranspose1dKernelSymInt32Grouped: output must be SYM_INT32")

/* ---- Conv1d dx under ARITH_BFP: the integer scatter -----------------------
 * convTranspose1dKernelBfpScatter walks the FLOAT32 scatter loop, keeps an
 * int32 accumulator per dx element on its (batch, conv group) E_acc and emits
 * the BFP_RAW: per raw target group the max E_acc of its rows, every sum
 * re-gridded to it. References: BfpScatterRef.h (int64, output-centric, from
 * the definition) and, for the worst case, the exact value count * 2^(2m-2) *
 * 2^(Egy+Ew). */

static uint32_t ctLcg(uint32_t *s) {
    *s = *s * 1664525u + 1013904223u;
    return *s >> 8;
}

typedef struct ctDxCase {
    size_t batch, cin, length, cout, groups, kernelSize, stride, dilation, padding;
    uint8_t m;
    size_t gyGroupSize, wGroupSize;
    size_t rawGroupSize; /* the dx raw's target geometry; 0 = per tensor {1,0} */
} ctDxCase_t;

typedef struct ctDxResult {
    bfpRawFixture_t raw;
    int32_t refMant[BFP_RAW_FIXTURE_MAX_OUT];
    int16_t refE[BFP_RAW_FIXTURE_MAX_GROUPS];
    size_t n, numGroups;
    bfpRefStats_t st;
} ctDxResult_t;

/* Runs the kernel into a sentinel-prefilled BFP_RAW and the reference on the
 * same operands. */
static void ctRunBfpScatterDx(const ctDxCase_t *c, const int32_t *gyCodes, const uint8_t *gyExps,
                              const int32_t *wCodes, const uint8_t *wExps, ctDxResult_t *r) {
    kernel_t kernel;
    initKernelExplicit(&kernel, c->kernelSize, c->padding, c->dilation, c->stride);
    windowGeometry1d_t geom = windowGeometry1dCalc(c->length, &kernel);
    size_t cinG = c->cin / c->groups;
    size_t gyElems = c->batch * c->cout * geom.outputLength;
    size_t wElems = c->cout * cinG * c->kernelSize;
    size_t gyGroups = c->gyGroupSize == 0 ? 1 : gyElems / c->gyGroupSize;
    size_t wGroups = c->wGroupSize == 0 ? 1 : wElems / c->wGroupSize;

    size_t gyDims[3] = {c->batch, c->cout, geom.outputLength};
    size_t wDims[3] = {c->cout, cinG, c->kernelSize};
    bfpScratchFixture_t gy;
    bfpScratchFixture_t w;
    bfpScratchInit(&gy, 3, gyDims, gyCodes, gyExps, gyGroups, c->gyGroupSize, c->m, 8);
    bfpScratchInit(&w, 3, wDims, wCodes, wExps, wGroups, c->wGroupSize, c->m, 8);

    r->n = c->batch * c->cin * c->length;
    r->numGroups = c->rawGroupSize == 0 ? 1 : r->n / c->rawGroupSize;
    size_t outDims[3] = {c->batch, c->cin, c->length};
    bfpRawInit(&r->raw, 3, outDims, r->numGroups, c->rawGroupSize);

    convTranspose1dKernelBfpScatter(&gy.tensor, &w.tensor, &kernel, c->groups, 0, &r->raw.tensor);

    bfpRefGeom_t G = {.batch = c->batch,
                      .cin = c->cin,
                      .length = c->length,
                      .cout = c->cout,
                      .groups = c->groups,
                      .kernelSize = c->kernelSize,
                      .stride = c->stride,
                      .dilation = c->dilation,
                      .padLeft = geom.padLeft,
                      .outLength = geom.outputLength};
    bfpRefOperand_t gyRef = {gyCodes, gyExps, c->gyGroupSize, c->m, 8};
    bfpRefOperand_t wRef = {wCodes, wExps, c->wGroupSize, c->m, 8};
    r->st = bfpRefConv1dDxRaw(&G, &gyRef, &wRef, c->rawGroupSize, r->refMant, r->refE);
}

static void ctAssertDxMatchesReference(const ctDxResult_t *r) {
    TEST_ASSERT_TRUE(r->st.fitsInt32);
    assertRawEquals(&r->raw, r->refMant, r->n, r->refE, r->numGroups);
}

/* Worst case at m = 16: every code -2^15, so every product is +2^30 and an
 * interior dx element sums Cout * K = 9 of them (overlapping taps). The guard
 * must right-shift (g > 0); without it, or with a headroom count that misses
 * the overlap, the int32 accumulator overflows. The raw value mant * 2^E is
 * exactly count * 2^30 * 2^(3 - 5), on E = -2 + g. */
void testConvTranspose1dKernelBfpScatterWorstCaseM16IsExact(void) {
    ctDxCase_t c = {.batch = 2,
                    .cin = 2,
                    .length = 6,
                    .cout = 3,
                    .groups = 1,
                    .kernelSize = 3,
                    .stride = 1,
                    .dilation = 1,
                    .padding = 1,
                    .m = 16,
                    .gyGroupSize = 0,
                    .wGroupSize = 0,
                    .rawGroupSize = 0};
    int32_t gyCodes[36];
    int32_t wCodes[18];
    for (size_t i = 0; i < 36; i++) {
        gyCodes[i] = -32768;
    }
    for (size_t i = 0; i < 18; i++) {
        wCodes[i] = -32768;
    }
    uint8_t gyExps[] = {127 + 3};
    uint8_t wExps[] = {127 - 5};
    ctDxResult_t r;
    ctRunBfpScatterDx(&c, gyCodes, gyExps, wCodes, wExps, &r);

    ctAssertDxMatchesReference(&r);
    int g = bfpRefGuard(9, 16, 16);
    TEST_ASSERT_TRUE(g > 0);
    TEST_ASSERT_EQUAL_INT16(-2 + g, r.raw.exps[0]);
    for (size_t b = 0; b < 2; b++) {
        for (size_t ic = 0; ic < 2; ic++) {
            for (size_t i = 0; i < 6; i++) {
                int64_t count = 0;
                for (size_t k = 0; k < 3; k++) {
                    long long t = (long long)i + 1 - (long long)k;
                    count += 3 * (t >= 0 && t < 6);
                }
                size_t idx = (b * 2 + ic) * 6 + i;
                TEST_ASSERT_EQUAL_INT64(count << (30 - g), (int64_t)r.raw.mant[idx]);
            }
        }
    }
}

/* m = 8, conv groups 2, stride 2 (two of the three taps can meet on one dx
 * element), grouped operands with a 24-binade exponent spread: products far
 * below the row top are rounded, products near it are shifted left. Raw groups
 * of 6 over rows of 9 straddle rows, conv groups and the batch, so some sums
 * are re-gridded to a higher group exponent. Bit-exact against the int64
 * reference. */
void testConvTranspose1dKernelBfpScatterMixedExponentsMatchesReference(void) {
    ctDxCase_t c = {.batch = 2,
                    .cin = 4,
                    .length = 9,
                    .cout = 4,
                    .groups = 2,
                    .kernelSize = 3,
                    .stride = 2,
                    .dilation = 1,
                    .padding = 1,
                    .m = 8,
                    .gyGroupSize = 4,
                    .wGroupSize = 6,
                    .rawGroupSize = 6};
    int32_t gyCodes[40]; /* [2, 4, 5] */
    int32_t wCodes[24];  /* [4, 2, 3] */
    uint8_t gyExps[10];
    uint8_t wExps[4];
    uint32_t s = 4242u;
    for (size_t i = 0; i < 40; i++) {
        gyCodes[i] = (int32_t)(ctLcg(&s) % 256u) - 128;
    }
    for (size_t i = 0; i < 24; i++) {
        wCodes[i] = (int32_t)(ctLcg(&s) % 256u) - 128;
    }
    for (size_t i = 0; i < 10; i++) {
        gyExps[i] = (uint8_t)(127 - 12 + (int)(ctLcg(&s) % 25u));
    }
    for (size_t i = 0; i < 4; i++) {
        wExps[i] = (uint8_t)(127 - 6 + (int)(ctLcg(&s) % 13u));
    }
    ctDxResult_t r;
    ctRunBfpScatterDx(&c, gyCodes, gyExps, wCodes, wExps, &r);

    ctAssertDxMatchesReference(&r);
    TEST_ASSERT_TRUE(r.st.rounded > 0);
    TEST_ASSERT_TRUE(r.st.leftShift > 0);
}

/* Same walk at m = 16 with spread exponents: the guard is positive, so even
 * the row-top products are rounded. Per-tensor raw over two conv groups. */
void testConvTranspose1dKernelBfpScatterMixedExponentsM16MatchesReference(void) {
    ctDxCase_t c = {.batch = 1,
                    .cin = 4,
                    .length = 9,
                    .cout = 4,
                    .groups = 2,
                    .kernelSize = 3,
                    .stride = 2,
                    .dilation = 1,
                    .padding = 1,
                    .m = 16,
                    .gyGroupSize = 4,
                    .wGroupSize = 6,
                    .rawGroupSize = 0};
    int32_t gyCodes[20]; /* [1, 4, 5] */
    int32_t wCodes[24];
    uint8_t gyExps[5];
    uint8_t wExps[4];
    uint32_t s = 99u;
    for (size_t i = 0; i < 20; i++) {
        gyCodes[i] = (int32_t)(ctLcg(&s) % 65536u) - 32768;
    }
    for (size_t i = 0; i < 24; i++) {
        wCodes[i] = (int32_t)(ctLcg(&s) % 65536u) - 32768;
    }
    for (size_t i = 0; i < 5; i++) {
        gyExps[i] = (uint8_t)(127 - 4 + (int)(ctLcg(&s) % 9u));
    }
    for (size_t i = 0; i < 4; i++) {
        wExps[i] = (uint8_t)(127 - 3 + (int)(ctLcg(&s) % 7u));
    }
    ctDxResult_t r;
    ctRunBfpScatterDx(&c, gyCodes, gyExps, wCodes, wExps, &r);

    ctAssertDxMatchesReference(&r);
    TEST_ASSERT_TRUE(r.st.rounded > 0);
}

/* Taps whose dx position falls outside the output never land: with stride 2
 * and dilation 2 the odd dx positions get no product and must read exactly 0
 * (the sentinel prefill shows they were written). */
void testConvTranspose1dKernelBfpScatterUnreachedPositionsAreZero(void) {
    ctDxCase_t c = {.batch = 1,
                    .cin = 1,
                    .length = 9,
                    .cout = 2,
                    .groups = 1,
                    .kernelSize = 2,
                    .stride = 2,
                    .dilation = 2,
                    .padding = 0,
                    .m = 8,
                    .gyGroupSize = 0,
                    .wGroupSize = 0,
                    .rawGroupSize = 3};
    int32_t gyCodes[8]; /* [1, 2, 4] */
    int32_t wCodes[4];
    for (size_t i = 0; i < 8; i++) {
        gyCodes[i] = (int32_t)i - 3;
    }
    for (size_t i = 0; i < 4; i++) {
        wCodes[i] = 5 - (int32_t)i;
    }
    uint8_t gyExps[] = {127};
    uint8_t wExps[] = {127};
    ctDxResult_t r;
    ctRunBfpScatterDx(&c, gyCodes, gyExps, wCodes, wExps, &r);

    ctAssertDxMatchesReference(&r);
    for (size_t i = 1; i < 9; i += 2) {
        TEST_ASSERT_EQUAL_INT32(0, r.raw.mant[i]);
    }
}

/* Stride 3 = kernel size: no two taps meet, so the headroom count is
 * in_channels/groups products, not times K. A guard over the nominal K would
 * shift one bit more and round the products differently. */
void testConvTranspose1dKernelBfpScatterHeadroomCountsOnlyMeetingTaps(void) {
    ctDxCase_t c = {.batch = 1,
                    .cin = 2,
                    .length = 9,
                    .cout = 2,
                    .groups = 1,
                    .kernelSize = 3,
                    .stride = 3,
                    .dilation = 1,
                    .padding = 0,
                    .m = 16,
                    .gyGroupSize = 0,
                    .wGroupSize = 0,
                    .rawGroupSize = 0};
    int32_t gyCodes[6]; /* [1, 2, 3] */
    int32_t wCodes[12]; /* [2, 2, 3] */
    uint32_t s = 5u;
    for (size_t i = 0; i < 6; i++) {
        gyCodes[i] = (int32_t)(ctLcg(&s) % 65536u) - 32768;
    }
    for (size_t i = 0; i < 12; i++) {
        wCodes[i] = (int32_t)(ctLcg(&s) % 65536u) - 32768;
    }
    uint8_t gyExps[] = {127};
    uint8_t wExps[] = {127};
    ctDxResult_t r;
    ctRunBfpScatterDx(&c, gyCodes, gyExps, wCodes, wExps, &r);

    ctAssertDxMatchesReference(&r);
    TEST_ASSERT_TRUE(r.st.rounded > 0);
}

/* A raw group straddling two batches whose E_acc differ: the group takes the
 * larger one, the lower batch's sums are re-gridded to it with HALF_AWAY.
 * Batch 1's loss sits 20 binades above batch 0's. At m = 8 the guard is
 * g = -15 (every sum carries 15 zero low bits on its own grid), so only a gap
 * beyond 15 binades rounds: here the low batch loses 5 bits. Raw groups of 4
 * over 6 elements per batch: group 1 = batch 0 elements 4-5 + batch 1
 * elements 0-1, so its LATER row is the higher one. A kernel that kept each
 * row's own E_acc, or published the group's first row's, fails the mantissa
 * or the exponent check. */
void testConvTranspose1dKernelBfpScatterRawGroupStraddlingBatchesTakesTheMaxExponent(void) {
    ctDxCase_t c = {.batch = 2,
                    .cin = 2,
                    .length = 3,
                    .cout = 2,
                    .groups = 1,
                    .kernelSize = 2,
                    .stride = 1,
                    .dilation = 1,
                    .padding = 0,
                    .m = 8,
                    .gyGroupSize = 4, /* one loss group per batch: [2, 2, 2] */
                    .wGroupSize = 0,
                    .rawGroupSize = 4};
    int32_t gyCodes[8] = {37, -91, 5, 127, -3, 77, -101, 19};
    int32_t wCodes[8] = {-128, 55, 13, -7, 99, -33, 61, 1};
    uint8_t gyExps[] = {127 - 10, 127 + 10};
    uint8_t wExps[] = {127};
    ctDxResult_t r;
    ctRunBfpScatterDx(&c, gyCodes, gyExps, wCodes, wExps, &r);

    ctAssertDxMatchesReference(&r);
    TEST_ASSERT_TRUE(r.st.realigned > 0);
    TEST_ASSERT_EQUAL_INT16(r.raw.exps[2], r.raw.exps[1]); /* group 1 takes batch 1's */
    TEST_ASSERT_EQUAL_INT16(r.raw.exps[2] - 20, r.raw.exps[0]);
}

/* Zero groups never set E_acc: loss [2, 2, 4] in one group per row, every
 * row in the zero state (E = 0) except (batch 1, channel 1), live at E = -28;
 * weight per tensor at E = 0. Batch 1's E_acc must follow its one live row,
 * not its zero row (the pre-pass), and the per-tensor raw, which straddles
 * both batches, must follow batch 1, not batch 0 that holds no live row (the
 * re-grid). Every dx value equals its exact int64 sum. */
void testConvTranspose1dKernelBfpScatterZeroGroupsDoNotSetTheAccumulator(void) {
    ctDxCase_t c = {.batch = 2,
                    .cin = 1,
                    .length = 6,
                    .cout = 2,
                    .groups = 1,
                    .kernelSize = 3,
                    .stride = 1,
                    .dilation = 1,
                    .padding = 0,
                    .m = 8,
                    .gyGroupSize = 4,
                    .wGroupSize = 0,
                    .rawGroupSize = 0};
    int32_t gyCodes[16] = {0};
    static const int32_t live[4] = {113, -77, 91, -59};
    memcpy(&gyCodes[12], live, sizeof(live));
    int32_t wCodes[6] = {101, -67, 45, 13, -99, 87}; /* [cout 2][cinG 1][K 3] */
    uint8_t gyExps[] = {127, 127, 127, 127 - 28};
    uint8_t wExps[] = {127};
    ctDxResult_t r;
    ctRunBfpScatterDx(&c, gyCodes, gyExps, wCodes, wExps, &r);

    for (size_t b = 0; b < 2; b++) {
        for (size_t p = 0; p < 6; p++) {
            int64_t exact = 0;
            for (size_t oc = 0; oc < 2; oc++) {
                for (size_t k = 0; k < 3; k++) {
                    if (p >= k && p - k < 4) {
                        exact += (int64_t)gyCodes[(b * 2 + oc) * 4 + (p - k)] * wCodes[oc * 3 + k];
                    }
                }
            }
            assertBfpValueExact(r.raw.mant[b * 6 + p], r.raw.exps[0], exact, -28, b * 6 + p);
        }
    }
}

/* A straddle gap beyond the int32 range (100 + 100 binades): the low batch's
 * sums re-grid to exactly 0, the high batch's stay exact. */
void testConvTranspose1dKernelBfpScatterRawGroupHugeGapFlushesTheLowRows(void) {
    ctDxCase_t c = {.batch = 2,
                    .cin = 1,
                    .length = 2,
                    .cout = 1,
                    .groups = 1,
                    .kernelSize = 1,
                    .stride = 1,
                    .dilation = 1,
                    .padding = 0,
                    .m = 8,
                    .gyGroupSize = 2,
                    .wGroupSize = 0,
                    .rawGroupSize = 0};
    int32_t gyCodes[4] = {64, -3, 127, -128};
    int32_t wCodes[1] = {-77};
    uint8_t gyExps[] = {127 + 100, 127 - 100};
    uint8_t wExps[] = {127};
    ctDxResult_t r;
    ctRunBfpScatterDx(&c, gyCodes, gyExps, wCodes, wExps, &r);

    ctAssertDxMatchesReference(&r);
    TEST_ASSERT_TRUE(r.st.realigned > 0);
    TEST_ASSERT_EQUAL_INT32(0, r.raw.mant[2]);
    TEST_ASSERT_EQUAL_INT32(0, r.raw.mant[3]);
    TEST_ASSERT_TRUE(r.raw.mant[0] != 0 && r.raw.mant[1] != 0);
}

/* The raw contract: the kernel writes a BFP_RAW only. */
static void ctScatterIntoFloatOutputBody(void) {
    ctDxCase_t c = {.batch = 1,
                    .cin = 1,
                    .length = 2,
                    .cout = 1,
                    .groups = 1,
                    .kernelSize = 1,
                    .stride = 1,
                    .dilation = 1,
                    .padding = 0,
                    .m = 8};
    kernel_t kernel;
    initKernelExplicit(&kernel, c.kernelSize, c.padding, c.dilation, c.stride);
    int32_t gyCodes[2] = {1, 2};
    int32_t wCodes[1] = {3};
    uint8_t exps[] = {127};
    size_t gyDims[3] = {1, 1, 2};
    size_t wDims[3] = {1, 1, 1};
    bfpScratchFixture_t gy;
    bfpScratchFixture_t w;
    bfpScratchInit(&gy, 3, gyDims, gyCodes, exps, 1, 0, 8, 8);
    bfpScratchInit(&w, 3, wDims, wCodes, exps, 1, 0, 8, 8);
    float out[2];
    size_t outDims[3] = {1, 1, 2};
    size_t outOrder[3] = {0, 1, 2};
    shape_t outShape;
    setShape(&outShape, outDims, 3, outOrder);
    quantization_t outQ;
    initFloat32Quantization(&outQ);
    tensor_t outTensor;
    setTensorValues(&outTensor, (uint8_t *)out, &outShape, &outQ, NULL);
    convTranspose1dKernelBfpScatter(&gy.tensor, &w.tensor, &kernel, 1, 0, &outTensor);
}

void testConvTranspose1dKernelBfpScatterRejectsFloatOutput(void) {
    ASSERT_EXITS_WITH_OUTPUT(1, "convTranspose1dKernelBfpScatter: output must be BFP_RAW",
                             ctScatterIntoFloatOutputBody());
}

/* ---- Goldens captured before the valid-tap-range refactor ----------------
 * Outputs recorded from the scatter kernels as they were before the ConvT
 * valid-tap range refactor, which must be bit-identical (SR stream included).
 * K 5 on the EXPLICIT adjoint path, the only one that clips taps. Geometry A
 * (stride 2, dilation 2, padLeft 3; Lin 5, Lout 12) clips low taps at inPos 0
 * and high taps at inPos 4; B (same kernel, Lin 2, Lout 5) clips both ends at
 * once (inPos 0 keeps k 2..3 only). With padLeft 3 and stride/dilation 2 the
 * clip offset padLeft - inPos*stride is always odd, so C (stride 3, dilation
 * 2, padLeft 4; Lin 4, Lout 11) adds offsets that divide by the dilation
 * (inPos 0) and ones that do not (inPos 1), and a high clip at inPos 3.
 * Operands are distinct, nonzero LCG codes; group scales are distinct with one
 * maximum, so every rescale but one has a factor != 1 and an extra or missing
 * rounding shows. */

typedef struct ctGoldGeom {
    size_t lin, lout, stride, dilation, padding;
} ctGoldGeom_t;

#define CT_GOLD_NUM_GEOMS 3
static const ctGoldGeom_t kCtGoldGeoms[CT_GOLD_NUM_GEOMS] = {
    {5, 12, 2, 2, 3}, {2, 5, 2, 2, 3}, {4, 11, 3, 2, 4}};

/* Nonzero code in [-r, -1] u [1, r]. */
static int32_t ctGoldCode(uint32_t *s, int32_t r) {
    int32_t v = (int32_t)(ctLcg(s) % (uint32_t)(2 * r)) - r;
    return v >= 0 ? v + 1 : v;
}

static void ctGoldKernel(kernel_t *kernel, const ctGoldGeom_t *geom) {
    initKernelExplicit(kernel, 5, geom->padding, geom->dilation, geom->stride);
}

/* Distinct dyadic scales, unique maximum, non-monotonic in g. */
static float ctGoldGroupScale(size_t g, size_t numGroups) {
    return (float)(5 + 3 * ((g * 7) % numGroups)) / 64.0f;
}

/* Stack tensor (3-D or 1-D) over caller data; not copyable after init. */
typedef struct ctGoldTensor {
    tensor_t t;
    shape_t s;
    size_t dims[3];
    size_t order[3];
    symInt32QConfig_t qc;
    quantization_t q;
} ctGoldTensor_t;

static void ctGoldShape(ctGoldTensor_t *g, size_t ndim, const size_t *dims) {
    for (size_t i = 0; i < ndim; i++) {
        g->dims[i] = dims[i];
        g->order[i] = i;
    }
    setShape(&g->s, g->dims, ndim, g->order);
}

static void ctGoldFloat(ctGoldTensor_t *g, size_t ndim, const size_t *dims, float *data) {
    ctGoldShape(g, ndim, dims);
    initFloat32Quantization(&g->q);
    setTensorValues(&g->t, (uint8_t *)data, &g->s, &g->q, NULL);
}

static void ctGoldSym(ctGoldTensor_t *g, size_t ndim, const size_t *dims, int32_t *data,
                      float scale, uint8_t bits, roundingMode_t rm) {
    ctGoldShape(g, ndim, dims);
    initSymInt32QConfigWithQMaxBits(rm, &g->qc, bits);
    g->qc.scale = odtScaleFromFloat(scale, ODT_SCALE_NEAREST);
    initSymInt32Quantization(&g->qc, &g->q);
    setTensorValues(&g->t, (uint8_t *)data, &g->s, &g->q, NULL);
}

/* batch 2, Cin 4, Cout 6, conv groups 2: weight [4, 3, 5] */
#define CT_GOLD_B 2
#define CT_GOLD_CIN 4
#define CT_GOLD_COUT 6
#define CT_GOLD_GROUPS 2
#define CT_GOLD_MAX_OUT (CT_GOLD_B * CT_GOLD_COUT * 12)

static void ctGoldRunFloat(const ctGoldGeom_t *geom, float *y) {
    uint32_t s = 911u + (uint32_t)geom->lin;
    float x[CT_GOLD_B * CT_GOLD_CIN * 5], w[CT_GOLD_CIN * 3 * 5], bias[CT_GOLD_COUT];
    for (size_t i = 0; i < CT_GOLD_B * CT_GOLD_CIN * geom->lin; i++) {
        x[i] = (float)ctGoldCode(&s, 4);
    }
    for (size_t i = 0; i < CT_GOLD_CIN * 3 * 5; i++) {
        w[i] = (float)ctGoldCode(&s, 3);
    }
    for (size_t i = 0; i < CT_GOLD_COUT; i++) {
        bias[i] = (float)ctGoldCode(&s, 5);
    }
    size_t xDims[] = {CT_GOLD_B, CT_GOLD_CIN, geom->lin}, wDims[] = {CT_GOLD_CIN, 3, 5};
    size_t yDims[] = {CT_GOLD_B, CT_GOLD_COUT, geom->lout}, bDims[] = {CT_GOLD_COUT};
    ctGoldTensor_t xT, wT, yT, bT;
    ctGoldFloat(&xT, 3, xDims, x);
    ctGoldFloat(&wT, 3, wDims, w);
    ctGoldFloat(&yT, 3, yDims, y);
    ctGoldFloat(&bT, 1, bDims, bias);
    kernel_t kernel;
    ctGoldKernel(&kernel, geom);
    convTranspose1dKernelFloat32(&xT.t, &wT.t, &bT.t, &kernel, CT_GOLD_GROUPS, 0, &yT.t);
}

void testConvTScatterFloatGolden(void) {
    const float *expected[CT_GOLD_NUM_GEOMS] = {kCtGoldFloatA, kCtGoldFloatB, kCtGoldFloatC};
    for (size_t gi = 0; gi < CT_GOLD_NUM_GEOMS; gi++) {
        float y[CT_GOLD_MAX_OUT];
        ctGoldRunFloat(&kCtGoldGeoms[gi], y);
        size_t n = CT_GOLD_B * CT_GOLD_COUT * kCtGoldGeoms[gi].lout;
        GOLD_F32(expected[gi], y, n);
    }
}

typedef struct ctGoldSymOut {
    int32_t y[CT_GOLD_MAX_OUT];
    float scale;
    uint8_t bits;
} ctGoldSymOut_t;

static void ctGoldRunSym(const ctGoldGeom_t *geom, ctGoldSymOut_t *out) {
    uint32_t s = 1213u + (uint32_t)geom->lin;
    int32_t x[CT_GOLD_B * CT_GOLD_CIN * 5], w[CT_GOLD_CIN * 3 * 5], bias[CT_GOLD_COUT];
    for (size_t i = 0; i < CT_GOLD_B * CT_GOLD_CIN * geom->lin; i++) {
        x[i] = ctGoldCode(&s, 100);
    }
    for (size_t i = 0; i < CT_GOLD_CIN * 3 * 5; i++) {
        w[i] = ctGoldCode(&s, 60);
    }
    for (size_t i = 0; i < CT_GOLD_COUT; i++) {
        bias[i] = ctGoldCode(&s, 200);
    }
    size_t xDims[] = {CT_GOLD_B, CT_GOLD_CIN, geom->lin}, wDims[] = {CT_GOLD_CIN, 3, 5};
    size_t yDims[] = {CT_GOLD_B, CT_GOLD_COUT, geom->lout}, bDims[] = {CT_GOLD_COUT};
    ctGoldTensor_t xT, wT, yT, bT;
    ctGoldSym(&xT, 3, xDims, x, 0.25f, 8, HALF_AWAY);
    ctGoldSym(&wT, 3, wDims, w, 0.125f, 8, HALF_AWAY);
    ctGoldSym(&yT, 3, yDims, out->y, 1.0f, UNSTAMPED_WIDTH, HALF_AWAY);
    ctGoldSym(&bT, 1, bDims, bias, 0.3f, 12, HALF_AWAY);
    kernel_t kernel;
    ctGoldKernel(&kernel, geom);
    convTranspose1dKernelSymInt32(&xT.t, &wT.t, &bT.t, &kernel, CT_GOLD_GROUPS, 0, &yT.t, NULL);
    out->scale = odtScaleToFloat(yT.qc.scale);
    out->bits = yT.qc.qMaxBits;
}

void testConvTScatterSymGolden(void) {
    const int32_t *expected[CT_GOLD_NUM_GEOMS] = {kCtGoldSymA, kCtGoldSymB, kCtGoldSymC};
    for (size_t gi = 0; gi < CT_GOLD_NUM_GEOMS; gi++) {
        ctGoldSymOut_t out;
        ctGoldRunSym(&kCtGoldGeoms[gi], &out);
        size_t n = CT_GOLD_B * CT_GOLD_COUT * kCtGoldGeoms[gi].lout;
        GOLD_I32(expected[gi], out.y, n);
        GOLD_F32_SCALAR(kCtGoldSymScale, out.scale);
        GOLD_U32_SCALAR(kCtGoldSymBits[gi], out.bits);
    }
}

/* The BFP Conv1d-dx scatter walk (the forward conv's view: dx [2, 4, L] from
 * gy [2, 6, Lgy] and w [6, 2, 5], conv groups 2) on geometries A and C.
 * Grouped operands with a wide exponent spread, so products are both rounded
 * and left-shifted; raw target groups straddle the conv groups, so some sums
 * are re-gridded. */
static void ctGoldRunBfp(const ctDxCase_t *c, uint32_t seed, ctDxResult_t *r) {
    kernel_t kernel;
    initKernelExplicit(&kernel, c->kernelSize, c->padding, c->dilation, c->stride);
    size_t gyN = c->batch * c->cout * windowGeometry1dCalc(c->length, &kernel).outputLength;
    size_t wN = c->cout * (c->cin / c->groups) * c->kernelSize;
    int32_t gyCodes[60];
    int32_t wCodes[60];
    uint8_t gyExps[15];
    uint8_t wExps[10];
    TEST_ASSERT_TRUE(gyN <= 60 && wN <= 60);
    TEST_ASSERT_TRUE(gyN / c->gyGroupSize <= 15 && wN / c->wGroupSize <= 10);
    uint32_t s = seed;
    for (size_t i = 0; i < gyN; i++) {
        gyCodes[i] = ctGoldCode(&s, 127);
    }
    for (size_t i = 0; i < wN; i++) {
        wCodes[i] = ctGoldCode(&s, 127);
    }
    for (size_t i = 0; i < gyN / c->gyGroupSize; i++) {
        gyExps[i] = (uint8_t)(127 - 12 + (int)(ctLcg(&s) % 25u));
    }
    for (size_t i = 0; i < wN / c->wGroupSize; i++) {
        wExps[i] = (uint8_t)(127 - 6 + (int)(ctLcg(&s) % 13u));
    }
    ctRunBfpScatterDx(c, gyCodes, gyExps, wCodes, wExps, r);
    ctAssertDxMatchesReference(r);
    TEST_ASSERT_TRUE(r->st.rounded > 0);
    TEST_ASSERT_TRUE(r->st.leftShift > 0);
    TEST_ASSERT_TRUE(r->st.realigned > 0);
}

void testConvTScatterBfpGolden(void) {
    /* A: L 12 (Lgy 5), raw groups of 16 over conv groups of 24 dx elements */
    ctDxCase_t a = {.batch = 2,
                    .cin = 4,
                    .length = 12,
                    .cout = 6,
                    .groups = 2,
                    .kernelSize = 5,
                    .stride = 2,
                    .dilation = 2,
                    .padding = 3,
                    .m = 8,
                    .gyGroupSize = 4,
                    .wGroupSize = 6,
                    .rawGroupSize = 16};
    ctDxResult_t r;
    ctGoldRunBfp(&a, 5150u, &r);
    TEST_ASSERT_EQUAL_size_t(96, r.n);
    TEST_ASSERT_EQUAL_size_t(6, r.numGroups);
    GOLD_I32(kCtGoldBfpMantA, r.raw.mant, 96);
    GOLD_I16(kCtGoldBfpExpA, r.raw.exps, 6);

    /* C: L 11 (Lgy 4), raw groups of 8 over conv groups of 22 dx elements */
    ctDxCase_t c = a;
    c.length = 11;
    c.stride = 3;
    c.padding = 4;
    c.rawGroupSize = 8;
    ctGoldRunBfp(&c, 6160u, &r);
    TEST_ASSERT_EQUAL_size_t(88, r.n);
    TEST_ASSERT_EQUAL_size_t(11, r.numGroups);
    GOLD_I32(kCtGoldBfpMantC, r.raw.mant, 88);
    GOLD_I16(kCtGoldBfpExpC, r.raw.exps, 11);
}

/* Grouped scatter: batch 2, Cin 2, Cout 7, conv groups 1, weight [2, 7, 5] =
 * 70 codes in 10 quantization groups of 7 (a group of 7 straddles the K = 5
 * kernel rows). The SR case also pins the stream state after the call: the
 * grouped scatter draws once per product and once per bias seed. */
#define CT_GOLD_G_COUT 7
#define CT_GOLD_G_NUM_GROUPS 10
#define CT_GOLD_G_MAX_OUT (CT_GOLD_B * CT_GOLD_G_COUT * 12)

typedef struct ctGoldGroupedOut {
    int32_t y[CT_GOLD_G_MAX_OUT];
    float scale;
    uint8_t bits;
} ctGoldGroupedOut_t;

static void ctGoldRunGrouped(const ctGoldGeom_t *geom, uint8_t xBits, roundingMode_t rm,
                             rngStream_t *rs, ctGoldGroupedOut_t *out) {
    uint32_t s = 7717u + (uint32_t)geom->lin;
    int32_t x[CT_GOLD_B * 2 * 5], w[2 * CT_GOLD_G_COUT * 5], bias[CT_GOLD_G_COUT];
    for (size_t i = 0; i < CT_GOLD_B * 2 * geom->lin; i++) {
        x[i] = ctGoldCode(&s, 100);
    }
    for (size_t i = 0; i < 2 * CT_GOLD_G_COUT * 5; i++) {
        w[i] = ctGoldCode(&s, 100);
    }
    for (size_t i = 0; i < CT_GOLD_G_COUT; i++) {
        bias[i] = ctGoldCode(&s, 200);
    }
    odtScale_t scales[CT_GOLD_G_NUM_GROUPS];
    for (size_t g = 0; g < CT_GOLD_G_NUM_GROUPS; g++) {
        scales[g] = odtTestScale(ctGoldGroupScale(g, CT_GOLD_G_NUM_GROUPS));
    }
    symQConfig_t groups = {.scales = scales,
                           .numGroups = CT_GOLD_G_NUM_GROUPS,
                           .groupSize = 7,
                           .qBits = 8,
                           .roundingMode = rm};
    symGroupedWeight_t groupedWeight = symGroupedWeightSym(&groups);
    size_t xDims[] = {CT_GOLD_B, 2, geom->lin}, wDims[] = {2, CT_GOLD_G_COUT, 5};
    size_t yDims[] = {CT_GOLD_B, CT_GOLD_G_COUT, geom->lout}, bDims[] = {CT_GOLD_G_COUT};
    ctGoldTensor_t xT, wT, yT, bT;
    ctGoldSym(&xT, 3, xDims, x, 0.25f, xBits, rm);
    ctGoldSym(&wT, 3, wDims, w, 1.0f, 8, rm); /* grouped scratch: scale poisoned, mode = op's */
    ctGoldSym(&yT, 3, yDims, out->y, 1.0f, UNSTAMPED_WIDTH, rm);
    ctGoldSym(&bT, 1, bDims, bias, 0.3f, 12, rm);
    kernel_t kernel;
    ctGoldKernel(&kernel, geom);
    convTranspose1dKernelSymInt32Grouped(&xT.t, &wT.t, &bT.t, &kernel, 1, 0, &yT.t, &groupedWeight,
                                         rs);
    out->scale = odtScaleToFloat(yT.qc.scale);
    out->bits = yT.qc.qMaxBits;
}

/* int8 input: the fold path (F = 8). The gold is the Python-int fold reference
 * (generate_expected_convT1d_grouped_fold.py), which reproduces this fixture's
 * LCG operands; A = {inScale.m, inScale.e + E + 15 - F}. */
void testConvTGroupedGoldenHalfAway(void) {
    const int32_t *expected[CT_GOLD_NUM_GEOMS] = {kCtFoldHaA, kCtFoldHaB, kCtFoldHaC};
    const float foldScale =
        odtScaleToFloat((odtScale_t){.m = (uint16_t)kCtFoldAccM, .e = (int16_t)kCtFoldAccE});
    for (size_t gi = 0; gi < CT_GOLD_NUM_GEOMS; gi++) {
        ctGoldGroupedOut_t out;
        ctGoldRunGrouped(&kCtGoldGeoms[gi], 8, HALF_AWAY, NULL, &out);
        size_t n = CT_GOLD_B * CT_GOLD_G_COUT * kCtGoldGeoms[gi].lout;
        GOLD_I32(expected[gi], out.y, n);
        GOLD_F32_SCALAR(foldScale, out.scale);
        GOLD_U32_SCALAR((uint32_t)kCtFoldBits[gi], out.bits);
    }
}

/* SR on the fold path, recorded from the kernel (kCtFoldSr*, the Python
 * reference has no C RNG). This fixture's k/64 group scales make every fold
 * exact, so the codes equal the HALF_AWAY gold except each bias seed, which SR
 * may lift by 1 (checked when recorded); the stream state pins one draw per
 * fold and per bias seed. SR rounding of inexact folds is pinned by
 * testConvTGroupedFoldSrIsUnbiased and SymGroupFold's own tests. */
void testConvTGroupedGoldenSr(void) {
    const int32_t *expected[CT_GOLD_NUM_GEOMS] = {kCtFoldSrA, kCtFoldSrB, kCtFoldSrC};
    const float foldScale =
        odtScaleToFloat((odtScale_t){.m = (uint16_t)kCtFoldAccM, .e = (int16_t)kCtFoldAccE});
    for (size_t gi = 0; gi < CT_GOLD_NUM_GEOMS; gi++) {
        rngStream_t rs = rngOpen(rngKeyFromSeed(0xC0DE0000u + gi));
        uint32_t before = rs.state;
        ctGoldGroupedOut_t out;
        ctGoldRunGrouped(&kCtGoldGeoms[gi], 8, SR_HALF_AWAY, &rs, &out);
        size_t n = CT_GOLD_B * CT_GOLD_G_COUT * kCtGoldGeoms[gi].lout;
        TEST_ASSERT_NOT_EQUAL_UINT32(before, rs.state);
        GOLD_I32(expected[gi], out.y, n);
        GOLD_U32_SCALAR(kCtFoldSrState[gi], rs.state);
        GOLD_F32_SCALAR(foldScale, out.scale);
        GOLD_U32_SCALAR((uint32_t)kCtFoldBits[gi], out.bits);
    }
}

/* ---- linq fold T3: the grouped scatter's per-group integer fold ----------
 * Each input code meeting group g is folded once into the accumulator scale
 * A = {inScale.m, inScale.e + E + 15 - F} (SymGroupFold.h); the products are
 * then plain int32 MACs. Golds: generate_expected_convT1d_grouped_fold.py
 * (Python ints). Fixture: batch 2, Cin 2, Cout 3, K 5, Lin 4, stride 2,
 * dilation 2, VALID (Lout 15); int8 operands. */

#define CT_FOLD_B 2
#define CT_FOLD_CIN 2
#define CT_FOLD_COUT 3
#define CT_FOLD_K 5
#define CT_FOLD_LIN 4
#define CT_FOLD_W_N (CT_FOLD_CIN * CT_FOLD_COUT * CT_FOLD_K)
#define CT_FOLD_X_N (CT_FOLD_B * CT_FOLD_CIN * CT_FOLD_LIN)
#define CT_FOLD_OUT_N (CT_FOLD_B * CT_FOLD_COUT * 15)
#define CT_FOLD_MAX_GROUPS 10

typedef struct ctFoldOut {
    int32_t y[CT_FOLD_OUT_N];
    odtScale_t scale;
    uint8_t bits;
} ctFoldOut_t;

static void ctFoldScales(const float *values, size_t n, odtScale_t *scales) {
    for (size_t g = 0; g < n; g++) {
        scales[g] = odtTestScale(values[g]);
    }
}

/* x / w default to the gold operands when NULL. */
static void ctFoldRun(const int32_t *x, const int32_t *w, const odtScale_t *scales,
                      size_t numGroups, size_t groupSize, bool withBias, roundingMode_t rm,
                      rngStream_t *rs, ctFoldOut_t *out) {
    TEST_ASSERT_EQUAL_size_t(CT_FOLD_OUT_N, kConvTFoldChannelOut_len);
    int32_t xData[CT_FOLD_X_N], wData[CT_FOLD_W_N], biasData[CT_FOLD_COUT];
    memcpy(xData, x != NULL ? x : kConvTFoldX, sizeof(xData));
    memcpy(wData, w != NULL ? w : kConvTFoldW, sizeof(wData));
    memcpy(biasData, kConvTFoldBias, sizeof(biasData));
    odtScale_t groupScales[CT_FOLD_MAX_GROUPS];
    memcpy(groupScales, scales, numGroups * sizeof(odtScale_t));
    symQConfig_t groups = {.scales = groupScales,
                           .numGroups = numGroups,
                           .groupSize = groupSize,
                           .qBits = 8,
                           .roundingMode = rm};
    symGroupedWeight_t groupedWeight = symGroupedWeightSym(&groups);
    size_t xDims[] = {CT_FOLD_B, CT_FOLD_CIN, CT_FOLD_LIN};
    size_t wDims[] = {CT_FOLD_CIN, CT_FOLD_COUT, CT_FOLD_K};
    size_t yDims[] = {CT_FOLD_B, CT_FOLD_COUT, (size_t)kConvTFoldOutLen};
    size_t bDims[] = {CT_FOLD_COUT};
    ctGoldTensor_t xT, wT, yT, bT;
    ctGoldSym(&xT, 3, xDims, xData, kConvTFoldXScale, 8, rm);
    ctGoldSym(&wT, 3, wDims, wData, 1.0f, 8, rm);
    ctGoldSym(&yT, 3, yDims, out->y, 1.0f, UNSTAMPED_WIDTH, rm);
    ctGoldSym(&bT, 1, bDims, biasData, kConvTFoldBiasScale, 12, rm);
    kernel_t kernel;
    initKernel(&kernel, CT_FOLD_K, VALID, 2, 2);
    convTranspose1dKernelSymInt32Grouped(&xT.t, &wT.t, withBias ? &bT.t : NULL, &kernel, 1, 0,
                                         &yT.t, &groupedWeight, rs);
    out->scale = odtScaleLoad(&yT.qc.scale);
    out->bits = yT.qc.qMaxBits;
}

static void ctFoldAssertGold(const float *wScales, size_t numGroups, size_t groupSize,
                             const int32_t *expected, int32_t accM, int32_t accE, int32_t bits) {
    odtScale_t scales[CT_FOLD_MAX_GROUPS];
    ctFoldScales(wScales, numGroups, scales);
    ctFoldOut_t out;
    ctFoldRun(NULL, NULL, scales, numGroups, groupSize, true, HALF_AWAY, NULL, &out);
    GOLD_I32(expected, out.y, CT_FOLD_OUT_N);
    TEST_ASSERT_EQUAL_UINT16((uint16_t)accM, out.scale.m);
    TEST_ASSERT_EQUAL_INT16((int16_t)accE, out.scale.e);
    TEST_ASSERT_EQUAL_UINT8((uint8_t)bits, out.bits);
}

/* groupSize 15 = Cout * K: each input channel's weight block is one group. */
void testConvTGroupedFoldGoldenHalfAwayChannel(void) {
    ctFoldAssertGold(kConvTFoldChannelWScales, (size_t)kConvTFoldChannelNumGroups,
                     (size_t)kConvTFoldChannelGroupSize, kConvTFoldChannelOut,
                     kConvTFoldChannelAccM, kConvTFoldChannelAccE, kConvTFoldChannelBits);
}

/* groupSize 3 against K = 5: groups straddle the tap rows (cursor path). */
void testConvTGroupedFoldGoldenHalfAwayGroupSize3(void) {
    ctFoldAssertGold(kConvTFoldGroup3WScales, (size_t)kConvTFoldGroup3NumGroups,
                     (size_t)kConvTFoldGroup3GroupSize, kConvTFoldGroup3Out, kConvTFoldGroup3AccM,
                     kConvTFoldGroup3AccE, kConvTFoldGroup3Bits);
}

/* The wire describes the codes: scale {inScale.m, inScale.e + E + 15 - F},
 * with F = 8 here (int8 x int8 over N = Cin * K = 10 terms is far inside
 * int32 at x * 2^8), and qMaxBits the width of the fold site -- x widened to
 * [xMin * 2^F - 1, xMax * 2^F + 1], no rescale slack -- with no bias. */
void testConvTGroupedFoldStampsFoldScaleAndWidth(void) {
    odtScale_t scales[CT_FOLD_MAX_GROUPS];
    ctFoldScales(kConvTFoldGroup3WScales, (size_t)kConvTFoldGroup3NumGroups, scales);
    int32_t eMax = INT32_MIN;
    for (size_t g = 0; g < (size_t)kConvTFoldGroup3NumGroups; g++) {
        eMax = scales[g].e > eMax ? scales[g].e : eMax;
    }
    ctFoldOut_t out;
    ctFoldRun(NULL, NULL, scales, (size_t)kConvTFoldGroup3NumGroups,
              (size_t)kConvTFoldGroup3GroupSize, false, HALF_AWAY, NULL, &out);
    const int32_t frac = 8;
    odtScale_t inScale = odtTestScale(kConvTFoldXScale);
    TEST_ASSERT_EQUAL_UINT16(inScale.m, out.scale.m);
    TEST_ASSERT_EQUAL_INT16((int16_t)(inScale.e + eMax + 15 - frac), out.scale.e);
    symAccumSite_t site = symAccumSiteInitRanges("t", CT_FOLD_CIN * CT_FOLD_K, -128 * 256 - 1,
                                                 127 * 256 + 1, -128, 127, 0);
    TEST_ASSERT_TRUE(symAccumProvenSafe(&site));
    TEST_ASSERT_EQUAL_UINT8(symAccumResultBits(&site), out.bits);
    TEST_ASSERT_EQUAL_UINT8(27, out.bits);
}

/* Review Focus 1: a group whose exponent sits 40 (shift 47) or 100 (shift
 * 107, clamped to 63) below E folds every code to 0 under HALF_AWAY, so the
 * output equals the run with that group's weights zeroed -- no garbage from
 * an oversized shift (run under UBSan). Under SR a fold may only land on 0 or
 * +-1, so each output moves by at most the sum of |w| over that group. */
void testConvTGroupedFoldTinyGroupScaleContributesNothingNotGarbage(void) {
    odtScale_t scales[CT_FOLD_MAX_GROUPS];
    ctFoldScales(kConvTFoldGroup3WScales, (size_t)kConvTFoldGroup3NumGroups, scales);
    int32_t eMax = INT32_MIN;
    for (size_t g = 0; g < (size_t)kConvTFoldGroup3NumGroups; g++) {
        eMax = scales[g].e > eMax ? scales[g].e : eMax;
    }
    const size_t tiny[2] = {2, 6};
    scales[tiny[0]].e = (int16_t)(eMax - 40);
    scales[tiny[1]].e = (int16_t)(eMax - 100);
    int32_t wZeroed[CT_FOLD_W_N];
    memcpy(wZeroed, kConvTFoldW, sizeof(wZeroed));
    int32_t tinyAbsSum = 0;
    for (size_t t = 0; t < 2; t++) {
        for (size_t i = tiny[t] * 3; i < tiny[t] * 3 + 3; i++) {
            tinyAbsSum += wZeroed[i] < 0 ? -wZeroed[i] : wZeroed[i];
            wZeroed[i] = 0;
        }
    }
    TEST_ASSERT_TRUE(tinyAbsSum > 0);
    const size_t numGroups = (size_t)kConvTFoldGroup3NumGroups;
    const size_t groupSize = (size_t)kConvTFoldGroup3GroupSize;

    ctFoldOut_t tinyOut, zeroOut;
    ctFoldRun(NULL, NULL, scales, numGroups, groupSize, true, HALF_AWAY, NULL, &tinyOut);
    ctFoldRun(NULL, wZeroed, scales, numGroups, groupSize, true, HALF_AWAY, NULL, &zeroOut);
    GOLD_I32(zeroOut.y, tinyOut.y, CT_FOLD_OUT_N);
    TEST_ASSERT_EQUAL_INT16(zeroOut.scale.e, tinyOut.scale.e);

    rngStream_t rsTiny = rngOpen(rngKeyFromSeed(0x7171u));
    rngStream_t rsZero = rngOpen(rngKeyFromSeed(0x7171u));
    ctFoldRun(NULL, NULL, scales, numGroups, groupSize, false, SR_HALF_AWAY, &rsTiny, &tinyOut);
    ctFoldRun(NULL, wZeroed, scales, numGroups, groupSize, false, SR_HALF_AWAY, &rsZero, &zeroOut);
    for (size_t i = 0; i < CT_FOLD_OUT_N; i++) {
        int32_t d = tinyOut.y[i] - zeroOut.y[i];
        TEST_ASSERT_TRUE(d <= tinyAbsSum && d >= -tinyAbsSum);
    }
}

/* Review Focus 3: a 17-bit input takes the legacy float path, bit-identical
 * to the pre-fold kernel -- kCtGoldGrouped{Ha,Sr}* and kCtGoldGroupedScale are
 * the goldens recorded before the fold existed (codes, scale and the SR
 * stream). The stamped width (#516) is the legacy rescaled site over the
 * 17-bit x range with the bias pass's seed range; this fixture's seeds
 * (|bias| <= 200 at 0.3 / sAcc 0.125, i.e. <= 480) sit far below its 2^26
 * boundary, so their part is pinned by ctLegacyStampCountsTheSeed. */
static uint8_t ctLegacyBits(size_t nTerms, int32_t wMin, int32_t wMax, int32_t seedMin,
                            int32_t seedMax) {
    int32_t xMin, xMax;
    symAccumWidthRange(17, &xMin, &xMax);
    symAccumSite_t site = symAccumSiteInitRangesRescaled("t", nTerms, xMin, xMax, wMin, wMax, 0);
    symAccumSiteSetSeedRange(&site, seedMin, seedMax);
    return symAccumResultBits(&site);
}

/* 17-bit x times grouped int4 over 8 scattered products, scales 1 (sAcc = 1,
 * so a bias b seeds as b): the rescaled site's positive side is 8 * 2^16 * 8
 * plus slack (4194313) -> 24 bits seedless; a bias of 2^22 lifts it past
 * 2^23 -> 25. */
static void ctLegacyStampCountsTheSeed(void) {
    ctFill(8, 1, 1, false);
    int32_t bias = 1 << 22;
    ctRun(8, 1, 1, 17, 4, &bias, 2);
    TEST_ASSERT_EQUAL_INT32(8 + (1 << 22), gCtY[0]);
    TEST_ASSERT_EQUAL_UINT8(ctLegacyBits(8, -8, 7, bias, bias), gCtOutBits);
    TEST_ASSERT_EQUAL_UINT8(25, gCtOutBits);
    TEST_ASSERT_EQUAL_UINT8(24, ctLegacyBits(8, -8, 7, 0, 0));
}

void testConvTGroupedFoldFallsBackToLegacyForWideInput(void) {
    const int32_t *expectedHa[CT_GOLD_NUM_GEOMS] = {kCtGoldGroupedHaA, kCtGoldGroupedHaB,
                                                    kCtGoldGroupedHaC};
    const int32_t *expectedSr[CT_GOLD_NUM_GEOMS] = {kCtGoldGroupedSrA, kCtGoldGroupedSrB,
                                                    kCtGoldGroupedSrC};
    for (size_t gi = 0; gi < CT_GOLD_NUM_GEOMS; gi++) {
        size_t n = CT_GOLD_B * CT_GOLD_G_COUT * kCtGoldGeoms[gi].lout;
        ctGoldGroupedOut_t out;
        ctGoldRunGrouped(&kCtGoldGeoms[gi], 17, HALF_AWAY, NULL, &out);
        GOLD_I32(expectedHa[gi], out.y, n);
        GOLD_F32_SCALAR(kCtGoldGroupedScale, out.scale);
        /* any seed range inside [-480, 480] gives the seedless width here */
        TEST_ASSERT_EQUAL_UINT8(ctLegacyBits(2 * 5, -128, 127, 0, 0),
                                ctLegacyBits(2 * 5, -128, 127, -480, 480));
        GOLD_U32_SCALAR(ctLegacyBits(2 * 5, -128, 127, 0, 0), out.bits);
        rngStream_t rs = rngOpen(rngKeyFromSeed(0xC0DE0000u + gi));
        ctGoldRunGrouped(&kCtGoldGeoms[gi], 17, SR_HALF_AWAY, &rs, &out);
        GOLD_I32(expectedSr[gi], out.y, n);
        GOLD_U32_SCALAR(kCtGoldGroupedSrState[gi], rs.state);
        GOLD_F32_SCALAR(kCtGoldGroupedScale, out.scale);
    }
    ctLegacyStampCountsTheSeed();
}

/* SR folds are unbiased: over 4096 streams the mean dequantized output sits
 * within 3 sigma (empirical, of that mean) of the exact dx, both for the sum
 * over all outputs and -- at 5 sigma, 90 elements -- per element; an element
 * with no spread must be exact. No bias (its seed is the float primitive's). */
#define CT_FOLD_SR_RUNS 4096
void testConvTGroupedFoldSrIsUnbiased(void) {
    odtScale_t scales[CT_FOLD_MAX_GROUPS];
    ctFoldScales(kConvTFoldGroup3WScales, (size_t)kConvTFoldGroup3NumGroups, scales);
    const double inScale = (double)kConvTFoldXScale;
    double exact[CT_FOLD_OUT_N] = {0};
    for (size_t b = 0; b < CT_FOLD_B; b++) {
        for (size_t ic = 0; ic < CT_FOLD_CIN; ic++) {
            for (size_t p = 0; p < CT_FOLD_LIN; p++) {
                int32_t xv = kConvTFoldX[(b * CT_FOLD_CIN + ic) * CT_FOLD_LIN + p];
                for (size_t oc = 0; oc < CT_FOLD_COUT; oc++) {
                    for (size_t k = 0; k < CT_FOLD_K; k++) {
                        size_t wIdx = (ic * CT_FOLD_COUT + oc) * CT_FOLD_K + k;
                        size_t o = p * 2 + k * 2;
                        exact[(b * CT_FOLD_COUT + oc) * 15 + o] +=
                            (double)xv * kConvTFoldW[wIdx] * inScale *
                            (double)odtScaleToFloat(scales[wIdx / 3]);
                    }
                }
            }
        }
    }
    static double sum[CT_FOLD_OUT_N], sumSq[CT_FOLD_OUT_N];
    memset(sum, 0, sizeof(sum));
    memset(sumSq, 0, sizeof(sumSq));
    double totSum = 0.0, totSumSq = 0.0;
    double acc = 0.0;
    for (uint32_t r = 0; r < CT_FOLD_SR_RUNS; r++) {
        rngStream_t rs = rngOpen(rngKeyFromSeed(0x5EED0000u + r));
        ctFoldOut_t out;
        ctFoldRun(NULL, NULL, scales, (size_t)kConvTFoldGroup3NumGroups,
                  (size_t)kConvTFoldGroup3GroupSize, false, SR_HALF_AWAY, &rs, &out);
        acc = (double)odtScaleToFloat(out.scale);
        double tot = 0.0;
        for (size_t i = 0; i < CT_FOLD_OUT_N; i++) {
            double v = (double)out.y[i] * acc;
            sum[i] += v;
            sumSq[i] += v * v;
            tot += v;
        }
        totSum += tot;
        totSumSq += tot * tot;
    }
    const double n = (double)CT_FOLD_SR_RUNS;
    double exactTot = 0.0;
    for (size_t i = 0; i < CT_FOLD_OUT_N; i++) {
        double mean = sum[i] / n;
        double var = sumSq[i] / n - mean * mean;
        double sigmaMean = sqrt(var > 0.0 ? var : 0.0) / sqrt(n);
        if (sigmaMean < 1e-3 * acc) {
            TEST_ASSERT_TRUE(fabs(mean - exact[i]) <= 1e-9 * (fabs(exact[i]) + acc));
        } else {
            TEST_ASSERT_TRUE(fabs(mean - exact[i]) <= 5.0 * sigmaMean);
        }
        exactTot += exact[i];
    }
    double totMean = totSum / n;
    double totVar = totSumSq / n - totMean * totMean;
    TEST_ASSERT_TRUE(totVar > 0.0);
    TEST_ASSERT_TRUE(fabs(totMean - exactTot) <= 3.0 * sqrt(totVar) / sqrt(n));
}

/* The zero rule: an input code of 0 is never folded, so an all-zero gradient
 * draws nothing from the SR stream and leaves an all-zero output -- on the
 * cursor-free channel path and on the cursor path. */
void testConvTGroupedFoldZeroGradientDrawsNothing(void) {
    const int32_t zeros[CT_FOLD_X_N] = {0};
    struct {
        const float *wScales;
        size_t numGroups, groupSize;
    } shapes[2] = {
        {kConvTFoldChannelWScales, (size_t)kConvTFoldChannelNumGroups,
         (size_t)kConvTFoldChannelGroupSize},
        {kConvTFoldGroup3WScales, (size_t)kConvTFoldGroup3NumGroups,
         (size_t)kConvTFoldGroup3GroupSize},
    };
    for (size_t s = 0; s < 2; s++) {
        odtScale_t scales[CT_FOLD_MAX_GROUPS];
        ctFoldScales(shapes[s].wScales, shapes[s].numGroups, scales);
        rngStream_t rs = rngOpen(rngKeyFromSeed(0x2E40u));
        const uint32_t before = rs.state;
        ctFoldOut_t out;
        ctFoldRun(zeros, NULL, scales, shapes[s].numGroups, shapes[s].groupSize, false,
                  SR_HALF_AWAY, &rs, &out);
        TEST_ASSERT_EQUAL_UINT32(before, rs.state);
        for (size_t i = 0; i < CT_FOLD_OUT_N; i++) {
            TEST_ASSERT_EQUAL_INT32(0, out.y[i]);
        }
    }
}

/* Review I1: a ConvT1d-forward bias seeded at the fold's A is 2^F times finer
 * than at the legacy sAcc, so the bias must take part in the F proof. Shape:
 * int12 input with absmax 0.05, per-channel int8 weights with absmax 0.1,
 * Cin 3, K 5, and a bias of about 1.0 -- seeded at A_{F=8} it is ~1e10, far
 * past the seed-rescale guard, while legacy's sAcc held it easily. The kernel
 * must pick a lower F (or legacy) and not abort, and its output must sit
 * within the documented bound of the exact value: 0.5 * scale per fold or
 * product rounding, i.e. 0.5 * outScale * (sum|w| + C) over the contributing
 * products, plus 0.5 * outScale for the bias seed. */
#define CT_BIAS_CIN 3
#define CT_BIAS_COUT 2
#define CT_BIAS_K 5
#define CT_BIAS_LIN 6
#define CT_BIAS_LOUT (CT_BIAS_LIN + CT_BIAS_K - 1)
void testConvTGroupedFoldLargeBiasLowersFInsteadOfAborting(void) {
    int32_t x[CT_BIAS_CIN * CT_BIAS_LIN], w[CT_BIAS_CIN * CT_BIAS_COUT * CT_BIAS_K];
    int32_t bias[CT_BIAS_COUT] = {2000, -1700};
    int32_t y[CT_BIAS_COUT * CT_BIAS_LOUT];
    uint32_t s = 4242u;
    for (size_t i = 0; i < sizeof(x) / sizeof(x[0]); i++) {
        x[i] = ctGoldCode(&s, 2047);
    }
    for (size_t i = 0; i < sizeof(w) / sizeof(w[0]); i++) {
        w[i] = ctGoldCode(&s, 127);
    }
    const float inScale = 0.05f / 2047.0f;
    const float biasScale = 1.0f / 2000.0f;
    const float wScaleValues[CT_BIAS_CIN] = {0.1f / 127.0f, 0.07f / 127.0f, 0.045f / 127.0f};
    odtScale_t scales[CT_BIAS_CIN];
    for (size_t g = 0; g < CT_BIAS_CIN; g++) {
        scales[g] = odtScaleFromFloat(wScaleValues[g], ODT_SCALE_NEAREST);
    }
    symQConfig_t groups = {.scales = scales,
                           .numGroups = CT_BIAS_CIN,
                           .groupSize = CT_BIAS_COUT * CT_BIAS_K,
                           .qBits = 8,
                           .roundingMode = HALF_AWAY};
    symGroupedWeight_t groupedWeight = symGroupedWeightSym(&groups);
    size_t xDims[] = {1, CT_BIAS_CIN, CT_BIAS_LIN},
           wDims[] = {CT_BIAS_CIN, CT_BIAS_COUT, CT_BIAS_K};
    size_t yDims[] = {1, CT_BIAS_COUT, CT_BIAS_LOUT}, bDims[] = {CT_BIAS_COUT};
    ctGoldTensor_t xT, wT, yT, bT;
    ctGoldSym(&xT, 3, xDims, x, inScale, 12, HALF_AWAY);
    ctGoldSym(&wT, 3, wDims, w, 1.0f, 8, HALF_AWAY);
    ctGoldSym(&yT, 3, yDims, y, 1.0f, UNSTAMPED_WIDTH, HALF_AWAY);
    ctGoldSym(&bT, 1, bDims, bias, biasScale, 12, HALF_AWAY);
    kernel_t kernel;
    initKernel(&kernel, CT_BIAS_K, VALID, 1, 1);
    convTranspose1dKernelSymInt32Grouped(&xT.t, &wT.t, &bT.t, &kernel, 1, 0, &yT.t, &groupedWeight,
                                         NULL);

    /* it stays on the fold path (A carries inScale's mantissa) at an F below 8 */
    odtScale_t out = odtScaleLoad(&yT.qc.scale), in = odtScaleLoad(&xT.qc.scale);
    int32_t eMax = INT32_MIN;
    for (size_t g = 0; g < CT_BIAS_CIN; g++) {
        eMax = scales[g].e > eMax ? scales[g].e : eMax;
    }
    const int32_t frac = in.e + eMax + 15 - out.e;
    TEST_ASSERT_EQUAL_UINT16(in.m, out.m);
    /* the bias seed is ~1.07e10 at F = 8 and ~1.34e9 at F = 5, past the seed
     * guard (~1.07e9); F = 4 (~6.7e8) is the largest that admits it */
    TEST_ASSERT_EQUAL_INT32(4, frac);

    const double outScale = (double)odtScaleToFloat(yT.qc.scale);
    const double sx = (double)odtScaleToFloat(xT.qc.scale);
    const double sb = (double)odtScaleToFloat(bT.qc.scale);
    for (size_t oc = 0; oc < CT_BIAS_COUT; oc++) {
        for (size_t o = 0; o < CT_BIAS_LOUT; o++) {
            double exact = (double)bias[oc] * sb;
            double absW = 0.0, count = 0.0;
            for (size_t ic = 0; ic < CT_BIAS_CIN; ic++) {
                for (size_t k = 0; k < CT_BIAS_K; k++) {
                    if (o < k || o - k >= CT_BIAS_LIN) {
                        continue;
                    }
                    int32_t wv = w[(ic * CT_BIAS_COUT + oc) * CT_BIAS_K + k];
                    exact += (double)x[ic * CT_BIAS_LIN + (o - k)] * wv * sx *
                             (double)odtScaleToFloat(scales[ic]);
                    absW += wv < 0 ? -wv : wv;
                    count += 1.0;
                }
            }
            double got = (double)y[oc * CT_BIAS_LOUT + o] * outScale;
            double bound = 0.5 * outScale * (absW + count + 1.0) + 1e-6 * fabs(exact);
            TEST_ASSERT_TRUE(fabs(got - exact) <= bound);
        }
    }
}

/* Review I2: conv groups 2 (Cin 4, Cout 6, outChPerGroup 3) on both fold
 * loops -- the per-ic group index over the global ic, the aligned predicate
 * per conv group, and the cursor across conv-group slabs. Python-int golds
 * (convt_grouped_fold_ref with conv_groups). */
#define CT_CG_CIN 4
#define CT_CG_COUT 6
#define CT_CG_GROUPS 2
#define CT_CG_W_N (CT_CG_CIN * (CT_CG_COUT / CT_CG_GROUPS) * CT_FOLD_K)
#define CT_CG_OUT_N (CT_FOLD_B * CT_CG_COUT * 15)
#define CT_CG_MAX_GROUPS 15
static void ctFoldCgAssertGold(const float *wScales, size_t numGroups, size_t groupSize,
                               const int32_t *expected, int32_t accM, int32_t accE, int32_t bits) {
    TEST_ASSERT_EQUAL_size_t(CT_CG_OUT_N, kConvTFoldCgAlignedOut_len);
    int32_t x[CT_FOLD_B * CT_CG_CIN * CT_FOLD_LIN], w[CT_CG_W_N], bias[CT_CG_COUT];
    int32_t y[CT_CG_OUT_N];
    memcpy(x, kConvTFoldCgX, sizeof(x));
    memcpy(w, kConvTFoldCgW, sizeof(w));
    memcpy(bias, kConvTFoldCgBias, sizeof(bias));
    odtScale_t scales[CT_CG_MAX_GROUPS];
    ctFoldScales(wScales, numGroups, scales);
    symQConfig_t groups = {.scales = scales,
                           .numGroups = numGroups,
                           .groupSize = groupSize,
                           .qBits = 8,
                           .roundingMode = HALF_AWAY};
    symGroupedWeight_t groupedWeight = symGroupedWeightSym(&groups);
    size_t xDims[] = {CT_FOLD_B, CT_CG_CIN, CT_FOLD_LIN};
    size_t wDims[] = {CT_CG_CIN, CT_CG_COUT / CT_CG_GROUPS, CT_FOLD_K};
    size_t yDims[] = {CT_FOLD_B, CT_CG_COUT, (size_t)kConvTFoldOutLen};
    size_t bDims[] = {CT_CG_COUT};
    ctGoldTensor_t xT, wT, yT, bT;
    ctGoldSym(&xT, 3, xDims, x, kConvTFoldXScale, 8, HALF_AWAY);
    ctGoldSym(&wT, 3, wDims, w, 1.0f, 8, HALF_AWAY);
    ctGoldSym(&yT, 3, yDims, y, 1.0f, UNSTAMPED_WIDTH, HALF_AWAY);
    ctGoldSym(&bT, 1, bDims, bias, kConvTFoldBiasScale, 12, HALF_AWAY);
    kernel_t kernel;
    initKernel(&kernel, CT_FOLD_K, VALID, 2, 2);
    convTranspose1dKernelSymInt32Grouped(&xT.t, &wT.t, &bT.t, &kernel, CT_CG_GROUPS, 0, &yT.t,
                                         &groupedWeight, NULL);
    GOLD_I32(expected, y, CT_CG_OUT_N);
    odtScale_t out = odtScaleLoad(&yT.qc.scale);
    TEST_ASSERT_EQUAL_UINT16((uint16_t)accM, out.m);
    TEST_ASSERT_EQUAL_INT16((int16_t)accE, out.e);
    TEST_ASSERT_EQUAL_UINT8((uint8_t)bits, yT.qc.qMaxBits);
}

void testConvTGroupedFoldGoldenConvGroups2Aligned(void) {
    ctFoldCgAssertGold(kConvTFoldCgAlignedWScales, (size_t)kConvTFoldCgAlignedNumGroups,
                       (size_t)kConvTFoldCgAlignedGroupSize, kConvTFoldCgAlignedOut,
                       kConvTFoldCgAlignedAccM, kConvTFoldCgAlignedAccE, kConvTFoldCgAlignedBits);
}

void testConvTGroupedFoldGoldenConvGroups2Cursor(void) {
    ctFoldCgAssertGold(kConvTFoldCgCursorWScales, (size_t)kConvTFoldCgCursorNumGroups,
                       (size_t)kConvTFoldCgCursorGroupSize, kConvTFoldCgCursorOut,
                       kConvTFoldCgCursorAccM, kConvTFoldCgCursorAccE, kConvTFoldCgCursorBits);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testConvT1dKernelSymRejectsFloatInput);
    RUN_TEST(testConvT1dKernelSymRejectsFloatWeight);
    RUN_TEST(testConvT1dKernelSymRejectsFloatBias);
    RUN_TEST(testConvT1dKernelSymRejectsFloatOutput);
    RUN_TEST(testConvT1dKernelSymGroupedRejectsFloatInput);
    RUN_TEST(testConvT1dKernelSymGroupedRejectsFloatWeight);
    RUN_TEST(testConvT1dKernelSymGroupedRejectsFloatBias);
    RUN_TEST(testConvT1dKernelSymGroupedRejectsFloatOutput);
    RUN_TEST(testConvTranspose1dKernelSymStampsResultWidthIncludesBiasSeed);
    RUN_TEST(testConvTranspose1dKernelSymGroupedStampsFoldWidthIncludesBiasSeed);
    RUN_TEST(testConvTranspose1dKernelSymGroupedAsymImageStampsItsFullRange);
    RUN_TEST(testConvTranspose1dKernelSymStampCountsTheLargestBiasSeedOverChannels);
    RUN_TEST(testConvTranspose1dKernelSymGroupedWideScatterRunsExactWhenSumFits);
    RUN_TEST(testConvTranspose1dKernelSymGroupedWideScatterThatWrapsHitsTheF9Guard);
    RUN_TEST(
        testConvTranspose1dKernelSymGroupedGuardWideWeightRunsExactWhenItFitsAndAbortsWhenItWraps);
    RUN_TEST(testConvTranspose1dKernelSymGroupedGuardAsymImageWeightKeepsItsFullRange);
    RUN_TEST(testConvTranspose1dKernelSymGuardProvenSafeN511RunsExact);
    RUN_TEST(testConvTranspose1dKernelSymGuardDiesAtFirstWrappingNChannels);
    RUN_TEST(testConvTranspose1dKernelSymGuardDiesAtN513PositiveCodes);
    RUN_TEST(testConvTranspose1dKernelSymGuardCountsKernelTapsInN);
    RUN_TEST(testConvTranspose1dKernelSymGuardCheckedPathStaysExactWhenValuesFit);
    RUN_TEST(testConvTranspose1dKernelSymGuardPostScatterBiasIsCounted);
    RUN_TEST(testConvTranspose1dKernelSymGuardWidthsFollowTheOperandsNotTheDefault);
    RUN_TEST(testConvTranspose1dKernelSymGuardWideProductOverflowAborts);
    RUN_TEST(testConvTranspose1dKernelSymGroupedGuardDiesAtFirstWrappingN);
    RUN_TEST(testConvTranspose1dKernelSymGroupedGuardWidthsFollowTheInputWidth);
    RUN_TEST(testConvTranspose1dKernelSymGroupedGuardCountsKernelTapsInN);
    RUN_TEST(testConvTranspose1dKernelSymGroupedGuardWideInputProductOverflowAborts);
    RUN_TEST(testConvTranspose1dKernelSymGroupedGuardCheckedPathStaysExactWhenValuesFit);
    RUN_TEST(testConvTranspose1dKernelSymGroupedGuardFullScaleN511RunsExact);
    RUN_TEST(testConvTranspose1dKernelSymGroupedGuardPostScatterBiasIsCounted);
    RUN_TEST(testConvTranspose1dKernelBasic);
    RUN_TEST(testConvTranspose1dKernelWithStride);
    RUN_TEST(testConvTranspose1dKernelWithOutputPadding);
    RUN_TEST(testConvTranspose1dKernelWithGroups);
    RUN_TEST(testConvTranspose1dKernelIsConvBackwardAdjoint);
    RUN_TEST(testConvTranspose1dKernelSamePaddingSymmetric);
    RUN_TEST(testConvTranspose1dKernelSamePaddingAsymmetric);
    RUN_TEST(testConvTranspose1dKernelAdjointSameGrouped);
    RUN_TEST(testConvTranspose1dKernelBfpGatherMatchesGold);
    RUN_TEST(testConvTranspose1dKernelBfpGatherNoBiasZeroSeeds);
    RUN_TEST(testConvTranspose1dKernelBfpGatherDilation2MatchesGold);
    RUN_TEST(testConvTranspose1dKernelBfpGroupedBiasBindsPerGroupExponent);
    RUN_TEST(testConvTranspose1dKernelBfpGatherPowerOfTwoBitIdenticalToGroupedSym);
    RUN_TEST(testConvTranspose1dKernelBfpGatherAdjointSameParityWithScatter);
    RUN_TEST(testConvTranspose1dKernelBfpGatherValidLengthMismatchDies);
    RUN_TEST(testConvTranspose1dKernelBfpGatherHeadroomGuardDies);
    RUN_TEST(testConvTranspose1dKernelBfpGatherRejectsMismatchedGroupShape);
    RUN_TEST(testConvTranspose1dKernelBfpGatherConvGroups2MatchesGold);
    RUN_TEST(testConvTranspose1dKernelBfpScatterWorstCaseM16IsExact);
    RUN_TEST(testConvTranspose1dKernelBfpScatterMixedExponentsMatchesReference);
    RUN_TEST(testConvTranspose1dKernelBfpScatterMixedExponentsM16MatchesReference);
    RUN_TEST(testConvTranspose1dKernelBfpScatterUnreachedPositionsAreZero);
    RUN_TEST(testConvTranspose1dKernelBfpScatterHeadroomCountsOnlyMeetingTaps);
    RUN_TEST(testConvTranspose1dKernelBfpScatterRawGroupStraddlingBatchesTakesTheMaxExponent);
    RUN_TEST(testConvTranspose1dKernelBfpScatterRawGroupHugeGapFlushesTheLowRows);
    RUN_TEST(testConvTranspose1dKernelBfpScatterZeroGroupsDoNotSetTheAccumulator);
    RUN_TEST(testConvTranspose1dKernelBfpScatterRejectsFloatOutput);
    RUN_TEST(testConvTScatterFloatGolden);
    RUN_TEST(testConvTScatterSymGolden);
    RUN_TEST(testConvTScatterBfpGolden);
    RUN_TEST(testConvTGroupedGoldenHalfAway);
    RUN_TEST(testConvTGroupedGoldenSr);
    RUN_TEST(testConvTGroupedFoldGoldenHalfAwayChannel);
    RUN_TEST(testConvTGroupedFoldGoldenHalfAwayGroupSize3);
    RUN_TEST(testConvTGroupedFoldStampsFoldScaleAndWidth);
    RUN_TEST(testConvTGroupedFoldTinyGroupScaleContributesNothingNotGarbage);
    RUN_TEST(testConvTGroupedFoldFallsBackToLegacyForWideInput);
    RUN_TEST(testConvTGroupedFoldSrIsUnbiased);
    RUN_TEST(testConvTGroupedFoldZeroGradientDrawsNothing);
    RUN_TEST(testConvTGroupedFoldLargeBiasLowersFInsteadOfAborting);
    RUN_TEST(testConvTGroupedFoldGoldenConvGroups2Aligned);
    RUN_TEST(testConvTGroupedFoldGoldenConvGroups2Cursor);
    return UNITY_END();
}
