#include <stdbool.h>
#include <string.h>

#include "ConvTranspose1dKernel.h"
#include "DeathTest.h"
#include "QuantizationApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "expected_bfp_convT1d.h"
#include "expected_conv_transpose_1d_kernel.h"
#include "unity.h"

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
    inSymQC.scale = 0.5f;
    quantization_t inSymQ;
    initSymInt32Quantization(&inSymQC, &inSymQ);
    setTensorValues(&inSymTensor, (uint8_t *)kBfpConvTInCodes, &inShape, &inSymQ, NULL);

    tensor_t wSymTensor;
    symInt32QConfig_t wSymQC;
    initSymInt32QConfig(HALF_AWAY, &wSymQC);
    wSymQC.scale = 1.0f; /* poisoned scratch scale -- never read, scales live in weightGroups */
    quantization_t wSymQ;
    initSymInt32Quantization(&wSymQC, &wSymQ);
    setTensorValues(&wSymTensor, (uint8_t *)kBfpConvTWCodes, &wShape, &wSymQ, NULL);

    float scales[3] = {0.25f, 0.25f, 0.25f};
    symQConfig_t weightGroups = {
        .scales = scales, .numGroups = 3, .groupSize = 8, .qBits = 8, .roundingMode = HALF_AWAY};

    tensor_t outSymTensor;
    int32_t outSymData[24];
    symInt32QConfig_t outSymQC;
    initSymInt32QConfig(HALF_AWAY, &outSymQC);
    quantization_t outSymQ;
    initSymInt32Quantization(&outSymQC, &outSymQ);
    setTensorValues(&outSymTensor, (uint8_t *)outSymData, &outShape, &outSymQ, NULL);

    convTranspose1dKernelSymInt32Grouped(&inSymTensor, &wSymTensor, NULL, &kernel, 1,
                                         (size_t)kBfpConvTOutputPadding, &outSymTensor,
                                         &weightGroups, NULL);

    for (size_t i = 0; i < 24; i++) {
        float symDequant = (float)outSymData[i] * outSymQC.scale;
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

static void ctFill(size_t n, int32_t xCode, int32_t wCode, bool wAlternates) {
    for (size_t i = 0; i < n; i++) {
        gCtX[i] = xCode;
        gCtW[i] = (wAlternates && (i % 2 == 1)) ? -wCode : wCode;
    }
}

/* x [1, inCh, len], w [inCh, 1, K], VALID stride 1 -> y [1, 1, len + K - 1];
 * result in gCtY. numGroups == 0 -> per-tensor weight kernel, else the grouped
 * kernel with equal scales. */
static void ctRun(size_t inCh, size_t len, size_t k, uint8_t xBits, uint8_t wBits,
                  const int32_t *biasCode, size_t numGroups) {
    size_t outLen = len + k - 1;
    size_t xDims[] = {1, inCh, len}, wDims[] = {inCh, 1, k}, yDims[] = {1, 1, outLen};
    size_t order[] = {0, 1, 2}, biasDims[] = {1};
    shape_t xS, wS, yS, biasS;
    setShape(&xS, xDims, 3, order);
    setShape(&wS, wDims, 3, order);
    setShape(&yS, yDims, 3, order);
    setShape(&biasS, biasDims, 1, order);
    symInt32QConfig_t xQC, wQC, yQC, biasQC;
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &xQC, xBits);
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &wQC, wBits);
    initSymInt32QConfig(HALF_AWAY, &yQC);
    initSymInt32QConfig(HALF_AWAY, &biasQC);
    quantization_t xQ, wQ, yQ, biasQ;
    initSymInt32Quantization(&xQC, &xQ);
    initSymInt32Quantization(&wQC, &wQ);
    initSymInt32Quantization(&yQC, &yQ);
    initSymInt32Quantization(&biasQC, &biasQ);
    int32_t biasData = biasCode ? *biasCode : 0;
    tensor_t x, w, yT, bias;
    setTensorValues(&x, (uint8_t *)gCtX, &xS, &xQ, NULL);
    setTensorValues(&w, (uint8_t *)gCtW, &wS, &wQ, NULL);
    setTensorValues(&yT, (uint8_t *)gCtY, &yS, &yQ, NULL);
    setTensorValues(&bias, (uint8_t *)&biasData, &biasS, &biasQ, NULL);
    kernel_t kernel;
    initKernel(&kernel, k, VALID, 1, 1);
    if (numGroups == 0) {
        convTranspose1dKernelSymInt32(&x, &w, biasCode ? &bias : NULL, &kernel, 1, 0, &yT, NULL);
    } else {
        float scales[8] = {1, 1, 1, 1, 1, 1, 1, 1};
        symQConfig_t groups = {.scales = scales,
                               .numGroups = numGroups,
                               .groupSize = inCh * k / numGroups,
                               .qBits = wBits,
                               .roundingMode = HALF_AWAY};
        convTranspose1dKernelSymInt32Grouped(&x, &w, biasCode ? &bias : NULL, &kernel, 1, 0, &yT,
                                             &groups, NULL);
    }
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
     * fit and 32 do not */
    ctFill(30, -32768, -2048, false);
    ctRun(30, 1, 1, 16, 12, NULL, 2);
    TEST_ASSERT_EQUAL_INT32(30 * 67108864, gCtY[0]);
    ctFill(32, -32768, -2048, false);
    ASSERT_EXITS_WITH_OUTPUT(1, "widths (16, 13)", ctRun(32, 1, 1, 16, 12, NULL, 2));
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

/* 511 full-scale terms: the weight counts one bit wider (ASYM mantissa image), so the
 * bound no longer proves this and the checked loop runs; it must stay exact. */
void testConvTranspose1dKernelSymGroupedGuardFullScaleN511RunsExact(void) {
    ctFill(511, -2048, -2048, false);
    ctRun(511, 1, 1, 12, 12, NULL, 7);
    TEST_ASSERT_EQUAL_INT32(511 * 4194304, gCtY[0]);
}

void testConvTranspose1dKernelSymGroupedGuardPostScatterBiasIsCounted(void) {
    int32_t seed = 4194304;
    ctFill(511, -2048, -2048, false);
    ASSERT_EXITS_WITH_OUTPUT(1, "max|seed|=4194304", ctRun(511, 1, 1, 12, 12, &seed, 7));
}

void testConvTranspose1dKernelSymGroupedGuardCountsTheAsymMantissaBitWider(void) {
    /* grouped ASYM weights unpack to code - zp (up to 2^qBits - 1 = 4095 at qBits 12) */
    ctFill(511, 2047, 4095, false);
    ASSERT_EXITS_WITH_OUTPUT(1, "convTranspose1dKernelSymInt32Grouped",
                             ctRun(511, 1, 1, 12, 12, NULL, 7));
}

/* A grouped weight wider than int12 used to exit at the validator; now it takes
 * the checked path (17 bits counted). 30 terms of -2048 x -32768 fit int32
 * through two groups; 32 terms (2^31) wrap. */
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

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testConvTranspose1dKernelSymGroupedWideScatterRunsExactWhenSumFits);
    RUN_TEST(testConvTranspose1dKernelSymGroupedWideScatterThatWrapsHitsTheF9Guard);
    RUN_TEST(
        testConvTranspose1dKernelSymGroupedGuardWideWeightRunsExactWhenItFitsAndAbortsWhenItWraps);
    RUN_TEST(testConvTranspose1dKernelSymGroupedGuardCountsTheAsymMantissaBitWider);
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
    return UNITY_END();
}
