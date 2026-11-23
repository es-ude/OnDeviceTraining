#include <math.h>
#include <stdbool.h>
#include <string.h>

#include "BfpRawFixture.h"
#include "Conv1dKernel.h"
#include "DeathTest.h"
#include "Matmul.h"
#include "QuantizationApi.h"
#include "RNG.h"
#include "StorageApi.h"
#include "SymAccumGuard.h"
#include "SymGroupCursor.h"
#include "SymGroupedWeight.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "TestScale.h"
#include "expected_bfp_conv1d.h"
#include "expected_conv1d_kernel.h"
#include "unity.h"

/* Golden comparisons: exact int32 equality, bitwise float equality. */
#define GOLD_I32(e, a, n) TEST_ASSERT_EQUAL_INT32_ARRAY((e), (a), (n))
#define GOLD_F32_SCALAR(e, a) TEST_ASSERT_EQUAL_MEMORY(&(float){(e)}, &(float){(a)}, sizeof(float))
#define GOLD_U32_SCALAR(e, a) TEST_ASSERT_EQUAL_UINT32((e), (a))
#include "expected_conv1d_grouped_fold.h"
#include "golden_conv1d_grouped.h"

/* Grouped-weight ctx fixture: the stored dtype's code domain fixes the
 * weight's operand range (SymGroupedWeight.h); the guard runner below picks
 * the domain through gCgWeightDomain. */
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

void testConv1dKernelSingleChannelSingleBatch() {
    float xData[] = {1.0f, 2.0f, 3.0f, 4.0f};
    size_t xDims[] = {1, 1, 4};
    tensor_t *x = makeFloatTensor(xDims, 3, xData);

    float wData[] = {2.0f, 4.0f};
    size_t wDims[] = {1, 1, 2};
    tensor_t *w = makeFloatTensor(wDims, 3, wData);

    size_t yDims[] = {1, 1, 3};
    tensor_t *y = makeFloatTensor(yDims, 3, NULL);

    kernel_t kernel;
    initKernel(&kernel, 2, VALID, 1, 1); // size, padding, dilation, stride

    conv1dKernelFloat32(x, w, NULL, &kernel, 1, y);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expectedConv1dForward_singleChannelSingleBatch, y->data,
                                  expectedConv1dForward_singleChannelSingleBatch_len);
}

void testConv1dKernelMultiChannelWithBias() {
    // x: [1, 3, 5] = arange(15)
    float xData[15];
    for (size_t i = 0; i < 15; i++) {
        xData[i] = (float)i;
    }
    size_t xDims[] = {1, 3, 5};
    tensor_t *x = makeFloatTensor(xDims, 3, xData);

    // w: [2, 3, 3] = arange(18) * 0.1
    float wData[18];
    for (size_t i = 0; i < 18; i++) {
        wData[i] = (float)i * 0.1f;
    }
    size_t wDims[] = {2, 3, 3};
    tensor_t *w = makeFloatTensor(wDims, 3, wData);

    float bData[] = {0.5f, -0.5f};
    size_t bDims[] = {2};
    tensor_t *b = makeFloatTensor(bDims, 1, bData);

    size_t yDims[] = {1, 2, 3};
    tensor_t *y = makeFloatTensor(yDims, 3, NULL);

    kernel_t kernel;
    initKernel(&kernel, 3, VALID, 1, 1);

    conv1dKernelFloat32(x, w, b, &kernel, 1, y);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expectedConv1dForward_multiChannelWithBias, y->data,
                                  expectedConv1dForward_multiChannelWithBias_len);
}

void testConv1dKernelMultiBatch() {
    // x and w come from the generator; their seeds are fixed.
    // x: [4, 2, 4]; w: [2, 2, 2]; output: [4, 2, 3]
    size_t yDims[] = {4, 2, 3};
    tensor_t *y = makeFloatTensor(yDims, 3, NULL);

    size_t xDims[] = {4, 2, 4};
    tensor_t *x = makeFloatTensor(xDims, 3, inputConv1dForward_multiBatch);

    size_t wDims[] = {2, 2, 2};
    tensor_t *w = makeFloatTensor(wDims, 3, weightConv1dForward_multiBatch);

    kernel_t kernel;
    initKernel(&kernel, 2, VALID, 1, 1);

    conv1dKernelFloat32(x, w, NULL, &kernel, 1, y);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expectedConv1dForward_multiBatch, y->data,
                                  expectedConv1dForward_multiBatch_len);
}

void testConv1dKernelGroupsDepthwise() {
    size_t yDims[] = {1, 4, 4}; // [B=1, Cout=4, Lout=5-2+1=4]
    tensor_t *y = makeFloatTensor(yDims, 3, NULL);

    size_t xDims[] = {1, 4, 5};
    tensor_t *x = makeFloatTensor(xDims, 3, inputConv1dForward_groupsDepthwise);

    size_t wDims[] = {4, 1, 2}; // [Cout=4, Cin/groups=1, K=2]
    tensor_t *w = makeFloatTensor(wDims, 3, weightConv1dForward_groupsDepthwise);

    kernel_t kernel;
    initKernel(&kernel, 2, VALID, 1, 1);

    conv1dKernelFloat32(x, w, NULL, &kernel, 4, y);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expectedConv1dForward_groupsDepthwise, y->data,
                                  expectedConv1dForward_groupsDepthwise_len);
}

void testConv1dKernelGroupsGrouped() {
    size_t yDims[] = {1, 8, 4}; // [B=1, Cout=8, Lout=4]
    tensor_t *y = makeFloatTensor(yDims, 3, NULL);

    size_t xDims[] = {1, 4, 5};
    tensor_t *x = makeFloatTensor(xDims, 3, inputConv1dForward_groupsGrouped);

    size_t wDims[] = {8, 2, 2}; // [Cout=8, Cin/groups=2, K=2]
    tensor_t *w = makeFloatTensor(wDims, 3, weightConv1dForward_groupsGrouped);

    kernel_t kernel;
    initKernel(&kernel, 2, VALID, 1, 1);

    conv1dKernelFloat32(x, w, NULL, &kernel, 2, y);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expectedConv1dForward_groupsGrouped, y->data,
                                  expectedConv1dForward_groupsGrouped_len);
}

void testConv1dKernelStrideDilation() {
    float xData[] = {1.0f, 0.0f, 2.0f, 0.0f, 0.0f, 0.0f, 3.0f, 0.0f, 4.0f};
    size_t xDims[] = {1, 1, 9};
    tensor_t *x = makeFloatTensor(xDims, 3, xData);

    float wData[] = {2.0f, 4.0f};
    size_t wDims[] = {1, 1, 2};
    tensor_t *w = makeFloatTensor(wDims, 3, wData);

    size_t yDims[] = {1, 1, 3};
    tensor_t *y = makeFloatTensor(yDims, 3, NULL);

    kernel_t kernel;
    initKernel(&kernel, 2, VALID, 2, 3); // size, padding, dilation, stride

    conv1dKernelFloat32(x, w, NULL, &kernel, 1, y);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expectedConv1dForward_strideDilation, y->data,
                                  expectedConv1dForward_strideDilation_len);
}

void testConv1dKernelSamePadding() {
    float xData[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    size_t xDims[] = {1, 1, 5};
    tensor_t *x = makeFloatTensor(xDims, 3, xData);

    float wData[] = {1.0f, 2.0f, 3.0f};
    size_t wDims[] = {1, 1, 3};
    tensor_t *w = makeFloatTensor(wDims, 3, wData);

    size_t yDims[] = {1, 1, 5};
    tensor_t *y = makeFloatTensor(yDims, 3, NULL);

    kernel_t kernel;
    initKernel(&kernel, 3, SAME, 1, 1);

    conv1dKernelFloat32(x, w, NULL, &kernel, 1, y);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expectedConv1dForward_samePadding, y->data,
                                  expectedConv1dForward_samePadding_len);
}

void testConv1dKernelExplicitPaddingStride2() {
    // ECG enc1 geometry: K=7, stride=2, EXPLICIT symmetric padding=3.
    // Gold from PyTorch F.conv1d(..., stride=2, padding=3) — see the generator's
    // fixture_explicit_padding. This is the layer-level parity guard for explicit
    // padding: a stride>1 conv that must reproduce PyTorch's padding=N exactly.
    float xData[10];
    for (size_t i = 0; i < 10; i++) {
        xData[i] = (float)(i + 1);
    }
    size_t xDims[] = {1, 1, 10};
    tensor_t *x = makeFloatTensor(xDims, 3, xData);

    float wData[] = {0.1f, -0.2f, 0.3f, -0.4f, 0.5f, -0.6f, 0.7f};
    size_t wDims[] = {1, 1, 7};
    tensor_t *w = makeFloatTensor(wDims, 3, wData);

    float bData[] = {0.25f};
    size_t bDims[] = {1};
    tensor_t *b = makeFloatTensor(bDims, 1, bData);

    size_t yDims[] = {1, 1, 5};
    tensor_t *y = makeFloatTensor(yDims, 3, NULL);

    kernel_t kernel;
    initKernelExplicit(&kernel, 7, 3, 1, 2); // size, padding, dilation, stride

    conv1dKernelFloat32(x, w, b, &kernel, 1, y);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expectedConv1dForward_explicitPadding, y->data,
                                  expectedConv1dForward_explicitPadding_len);
}

/* ---- conv1dKernelBfp ----------------------------------------------------
 *
 * Operands arrive in the funnel's UNPACKED-BFP scratch form: ->data holds
 * int32 sign-extended mantissa codes, ->quantization is BFP with a live
 * bfpQConfig_t (stack-fixture idiom, Quantization.h). Output is RAW FLOAT32
 * -- the kernel never rounds and never width-restores (both are the funnel's
 * job, not the kernel's). The gold fixture lives in the exact float regime
 * (generate_expected_bfp_conv1d.py asserts it), so expectations are BIT-
 * pinned via TEST_ASSERT_EQUAL_MEMORY, not a tolerance. */

void testConv1dKernelBfpMatchesGold(void) {
    tensor_t inTensor;
    size_t inDims[] = {(size_t)kBfpConvBatch, (size_t)kBfpConvInChannels,
                       (size_t)kBfpConvInputLength};
    size_t inOrder[] = {0, 1, 2};
    shape_t inShape;
    setShape(&inShape, inDims, 3, inOrder);
    /* sizeof(fixture) sizing: a regenerated gold with a different group count
     * fails loudly at the numGroups check instead of silently short-copying */
    uint8_t inExponents[sizeof(kBfpConvInExponents)];
    memcpy(inExponents, kBfpConvInExponents, sizeof(inExponents));
    bfpQConfig_t inQC = {.exponents = inExponents,
                         .numGroups = (size_t)kBfpConvInNumGroups,
                         .groupSize = (size_t)kBfpConvInGroupSize,
                         .roundingMode = HALF_AWAY,
                         .mantissaBits = (uint8_t)kBfpConvInMantissaBits,
                         .exponentBits = (uint8_t)kBfpConvInExponentBits};
    quantization_t inQ;
    initBfpQuantization(&inQC, &inQ);
    setTensorValues(&inTensor, (uint8_t *)kBfpConvInCodes, &inShape, &inQ, NULL);

    tensor_t wTensor;
    size_t wDims[] = {(size_t)kBfpConvOutChannels, (size_t)kBfpConvInChannels,
                      (size_t)kBfpConvKernelSize};
    size_t wOrder[] = {0, 1, 2};
    shape_t wShape;
    setShape(&wShape, wDims, 3, wOrder);
    uint8_t wExponents[sizeof(kBfpConvWExponents)];
    memcpy(wExponents, kBfpConvWExponents, sizeof(wExponents));
    bfpQConfig_t wQC = {.exponents = wExponents,
                        .numGroups = (size_t)kBfpConvWNumGroups,
                        .groupSize = (size_t)kBfpConvWGroupSize,
                        .roundingMode = HALF_AWAY,
                        .mantissaBits = (uint8_t)kBfpConvWMantissaBits,
                        .exponentBits = (uint8_t)kBfpConvWExponentBits};
    quantization_t wQ;
    initBfpQuantization(&wQC, &wQ);
    setTensorValues(&wTensor, (uint8_t *)kBfpConvWCodes, &wShape, &wQ, NULL);

    tensor_t biasTensor;
    size_t biasDims[] = {(size_t)kBfpConvOutChannels};
    size_t biasOrder[] = {0};
    shape_t biasShape;
    setShape(&biasShape, biasDims, 1, biasOrder);
    uint8_t biasExponents[sizeof(kBfpConvBiasExponents)];
    memcpy(biasExponents, kBfpConvBiasExponents, sizeof(biasExponents));
    bfpQConfig_t biasQC = {.exponents = biasExponents,
                           .numGroups = 1,
                           .groupSize = 0,
                           .roundingMode = HALF_AWAY,
                           .mantissaBits = (uint8_t)kBfpConvBiasMantissaBits,
                           .exponentBits = (uint8_t)kBfpConvBiasExponentBits};
    quantization_t biasQ;
    initBfpQuantization(&biasQC, &biasQ);
    setTensorValues(&biasTensor, (uint8_t *)kBfpConvBiasCodes, &biasShape, &biasQ, NULL);

    tensor_t outTensor;
    float outData[10];
    size_t outDims[] = {(size_t)kBfpConvBatch, (size_t)kBfpConvOutChannels, (size_t)kBfpConvOutLen};
    size_t outOrder[] = {0, 1, 2};
    shape_t outShape;
    setShape(&outShape, outDims, 3, outOrder);
    quantization_t outQ;
    initFloat32Quantization(&outQ);
    setTensorValues(&outTensor, (uint8_t *)outData, &outShape, &outQ, NULL);

    kernel_t kernel;
    initKernelExplicit(&kernel, (size_t)kBfpConvKernelSize, (size_t)kBfpConvPadding, 1,
                       (size_t)kBfpConvStride); // size, padding, dilation, stride

    conv1dKernelBfp(&inTensor, &wTensor, &biasTensor, &kernel, 1, &outTensor);

    TEST_ASSERT_EQUAL_MEMORY(kBfpConvExpected, outTensor.data,
                             kBfpConvExpected_len * sizeof(float));
}

void testConv1dKernelBfpNoBiasZeroSeeds(void) {
    tensor_t inTensor;
    size_t inDims[] = {(size_t)kBfpConvBatch, (size_t)kBfpConvInChannels,
                       (size_t)kBfpConvInputLength};
    size_t inOrder[] = {0, 1, 2};
    shape_t inShape;
    setShape(&inShape, inDims, 3, inOrder);
    uint8_t inExponents[sizeof(kBfpConvInExponents)];
    memcpy(inExponents, kBfpConvInExponents, sizeof(inExponents));
    bfpQConfig_t inQC = {.exponents = inExponents,
                         .numGroups = (size_t)kBfpConvInNumGroups,
                         .groupSize = (size_t)kBfpConvInGroupSize,
                         .roundingMode = HALF_AWAY,
                         .mantissaBits = (uint8_t)kBfpConvInMantissaBits,
                         .exponentBits = (uint8_t)kBfpConvInExponentBits};
    quantization_t inQ;
    initBfpQuantization(&inQC, &inQ);
    setTensorValues(&inTensor, (uint8_t *)kBfpConvInCodes, &inShape, &inQ, NULL);

    tensor_t wTensor;
    size_t wDims[] = {(size_t)kBfpConvOutChannels, (size_t)kBfpConvInChannels,
                      (size_t)kBfpConvKernelSize};
    size_t wOrder[] = {0, 1, 2};
    shape_t wShape;
    setShape(&wShape, wDims, 3, wOrder);
    uint8_t wExponents[sizeof(kBfpConvWExponents)];
    memcpy(wExponents, kBfpConvWExponents, sizeof(wExponents));
    bfpQConfig_t wQC = {.exponents = wExponents,
                        .numGroups = (size_t)kBfpConvWNumGroups,
                        .groupSize = (size_t)kBfpConvWGroupSize,
                        .roundingMode = HALF_AWAY,
                        .mantissaBits = (uint8_t)kBfpConvWMantissaBits,
                        .exponentBits = (uint8_t)kBfpConvWExponentBits};
    quantization_t wQ;
    initBfpQuantization(&wQC, &wQ);
    setTensorValues(&wTensor, (uint8_t *)kBfpConvWCodes, &wShape, &wQ, NULL);

    tensor_t outTensor;
    float outData[10];
    size_t outDims[] = {(size_t)kBfpConvBatch, (size_t)kBfpConvOutChannels, (size_t)kBfpConvOutLen};
    size_t outOrder[] = {0, 1, 2};
    shape_t outShape;
    setShape(&outShape, outDims, 3, outOrder);
    quantization_t outQ;
    initFloat32Quantization(&outQ);
    setTensorValues(&outTensor, (uint8_t *)outData, &outShape, &outQ, NULL);

    kernel_t kernel;
    initKernelExplicit(&kernel, (size_t)kBfpConvKernelSize, (size_t)kBfpConvPadding, 1,
                       (size_t)kBfpConvStride);

    conv1dKernelBfp(&inTensor, &wTensor, NULL, &kernel, 1, &outTensor);

    TEST_ASSERT_EQUAL_MEMORY(kBfpConvNoBiasExpected, outTensor.data,
                             kBfpConvNoBiasExpected_len * sizeof(float));
}

/* The dilation factor lives in the BFP arm's own
 * tap walk (inputIdx = firstValidInputIdx + i * geom.dilation) -- every other
 * BFP fixture runs dilation=1, where a dropped factor is an arithmetic
 * identity. This fixture's taps step by 2; a hardcode-1 mutant reads
 * p, p+1, p+2 instead of p, p+2, p+4 and misses the gold. */
void testConv1dKernelBfpDilation2MatchesGold(void) {
    tensor_t inTensor;
    size_t inDims[] = {(size_t)kBfpConvBatch, (size_t)kBfpConvInChannels,
                       (size_t)kBfpConvInputLength};
    size_t inOrder[] = {0, 1, 2};
    shape_t inShape;
    setShape(&inShape, inDims, 3, inOrder);
    uint8_t inExponents[sizeof(kBfpConvInExponents)];
    memcpy(inExponents, kBfpConvInExponents, sizeof(inExponents));
    bfpQConfig_t inQC = {.exponents = inExponents,
                         .numGroups = (size_t)kBfpConvInNumGroups,
                         .groupSize = (size_t)kBfpConvInGroupSize,
                         .roundingMode = HALF_AWAY,
                         .mantissaBits = (uint8_t)kBfpConvInMantissaBits,
                         .exponentBits = (uint8_t)kBfpConvInExponentBits};
    quantization_t inQ;
    initBfpQuantization(&inQC, &inQ);
    setTensorValues(&inTensor, (uint8_t *)kBfpConvInCodes, &inShape, &inQ, NULL);

    tensor_t wTensor;
    size_t wDims[] = {(size_t)kBfpConvOutChannels, (size_t)kBfpConvInChannels,
                      (size_t)kBfpConvKernelSize};
    size_t wOrder[] = {0, 1, 2};
    shape_t wShape;
    setShape(&wShape, wDims, 3, wOrder);
    uint8_t wExponents[sizeof(kBfpConvWExponents)];
    memcpy(wExponents, kBfpConvWExponents, sizeof(wExponents));
    bfpQConfig_t wQC = {.exponents = wExponents,
                        .numGroups = (size_t)kBfpConvWNumGroups,
                        .groupSize = (size_t)kBfpConvWGroupSize,
                        .roundingMode = HALF_AWAY,
                        .mantissaBits = (uint8_t)kBfpConvWMantissaBits,
                        .exponentBits = (uint8_t)kBfpConvWExponentBits};
    quantization_t wQ;
    initBfpQuantization(&wQC, &wQ);
    setTensorValues(&wTensor, (uint8_t *)kBfpConvWCodes, &wShape, &wQ, NULL);

    tensor_t biasTensor;
    size_t biasDims[] = {(size_t)kBfpConvOutChannels};
    size_t biasOrder[] = {0};
    shape_t biasShape;
    setShape(&biasShape, biasDims, 1, biasOrder);
    uint8_t biasExponents[sizeof(kBfpConvBiasExponents)];
    memcpy(biasExponents, kBfpConvBiasExponents, sizeof(biasExponents));
    bfpQConfig_t biasQC = {.exponents = biasExponents,
                           .numGroups = 1,
                           .groupSize = 0,
                           .roundingMode = HALF_AWAY,
                           .mantissaBits = (uint8_t)kBfpConvBiasMantissaBits,
                           .exponentBits = (uint8_t)kBfpConvBiasExponentBits};
    quantization_t biasQ;
    initBfpQuantization(&biasQC, &biasQ);
    setTensorValues(&biasTensor, (uint8_t *)kBfpConvBiasCodes, &biasShape, &biasQ, NULL);

    tensor_t outTensor;
    float outData[8]; /* batch * outChannels * kBfpConvDilOutLen */
    size_t outDims[] = {(size_t)kBfpConvBatch, (size_t)kBfpConvOutChannels,
                        (size_t)kBfpConvDilOutLen};
    size_t outOrder[] = {0, 1, 2};
    shape_t outShape;
    setShape(&outShape, outDims, 3, outOrder);
    quantization_t outQ;
    initFloat32Quantization(&outQ);
    setTensorValues(&outTensor, (uint8_t *)outData, &outShape, &outQ, NULL);

    kernel_t kernel;
    initKernelExplicit(&kernel, (size_t)kBfpConvKernelSize, (size_t)kBfpConvPadding,
                       (size_t)kBfpConvDilDilation, (size_t)kBfpConvStride);

    conv1dKernelBfp(&inTensor, &wTensor, &biasTensor, &kernel, 1, &outTensor);

    TEST_ASSERT_EQUAL_MEMORY(kBfpConvDilExpected, outTensor.data,
                             kBfpConvDilExpected_len * sizeof(float));
}

/* The conv sibling of UnitTestMatmul.c's
 * testMatmulBfpGroupedBiasBindsPerGroupExponent: same bias VALUES stored
 * grouped {numGroups=2, groupSize=1} with non-uniform exponents (goldgen
 * asserts a group-0 collapse differs), expected output bit-identical to the
 * per-tensor gold. */
void testConv1dKernelBfpGroupedBiasBindsPerGroupExponent(void) {
    tensor_t inTensor;
    size_t inDims[] = {(size_t)kBfpConvBatch, (size_t)kBfpConvInChannels,
                       (size_t)kBfpConvInputLength};
    size_t inOrder[] = {0, 1, 2};
    shape_t inShape;
    setShape(&inShape, inDims, 3, inOrder);
    uint8_t inExponents[sizeof(kBfpConvInExponents)];
    memcpy(inExponents, kBfpConvInExponents, sizeof(inExponents));
    bfpQConfig_t inQC = {.exponents = inExponents,
                         .numGroups = (size_t)kBfpConvInNumGroups,
                         .groupSize = (size_t)kBfpConvInGroupSize,
                         .roundingMode = HALF_AWAY,
                         .mantissaBits = (uint8_t)kBfpConvInMantissaBits,
                         .exponentBits = (uint8_t)kBfpConvInExponentBits};
    quantization_t inQ;
    initBfpQuantization(&inQC, &inQ);
    setTensorValues(&inTensor, (uint8_t *)kBfpConvInCodes, &inShape, &inQ, NULL);

    tensor_t wTensor;
    size_t wDims[] = {(size_t)kBfpConvOutChannels, (size_t)kBfpConvInChannels,
                      (size_t)kBfpConvKernelSize};
    size_t wOrder[] = {0, 1, 2};
    shape_t wShape;
    setShape(&wShape, wDims, 3, wOrder);
    uint8_t wExponents[sizeof(kBfpConvWExponents)];
    memcpy(wExponents, kBfpConvWExponents, sizeof(wExponents));
    bfpQConfig_t wQC = {.exponents = wExponents,
                        .numGroups = (size_t)kBfpConvWNumGroups,
                        .groupSize = (size_t)kBfpConvWGroupSize,
                        .roundingMode = HALF_AWAY,
                        .mantissaBits = (uint8_t)kBfpConvWMantissaBits,
                        .exponentBits = (uint8_t)kBfpConvWExponentBits};
    quantization_t wQ;
    initBfpQuantization(&wQC, &wQ);
    setTensorValues(&wTensor, (uint8_t *)kBfpConvWCodes, &wShape, &wQ, NULL);

    tensor_t biasTensor;
    size_t biasDims[] = {(size_t)kBfpConvOutChannels};
    size_t biasOrder[] = {0};
    shape_t biasShape;
    setShape(&biasShape, biasDims, 1, biasOrder);
    uint8_t biasExponents[sizeof(kBfpConvBiasGroupedExponents)];
    memcpy(biasExponents, kBfpConvBiasGroupedExponents, sizeof(biasExponents));
    bfpQConfig_t biasQC = {.exponents = biasExponents,
                           .numGroups = (size_t)kBfpConvBiasGroupedNumGroups,
                           .groupSize = 1,
                           .roundingMode = HALF_AWAY,
                           .mantissaBits = (uint8_t)kBfpConvBiasMantissaBits,
                           .exponentBits = (uint8_t)kBfpConvBiasExponentBits};
    quantization_t biasQ;
    initBfpQuantization(&biasQC, &biasQ);
    setTensorValues(&biasTensor, (uint8_t *)kBfpConvBiasGroupedCodes, &biasShape, &biasQ, NULL);

    tensor_t outTensor;
    float outData[10];
    size_t outDims[] = {(size_t)kBfpConvBatch, (size_t)kBfpConvOutChannels, (size_t)kBfpConvOutLen};
    size_t outOrder[] = {0, 1, 2};
    shape_t outShape;
    setShape(&outShape, outDims, 3, outOrder);
    quantization_t outQ;
    initFloat32Quantization(&outQ);
    setTensorValues(&outTensor, (uint8_t *)outData, &outShape, &outQ, NULL);

    kernel_t kernel;
    initKernelExplicit(&kernel, (size_t)kBfpConvKernelSize, (size_t)kBfpConvPadding, 1,
                       (size_t)kBfpConvStride);

    conv1dKernelBfp(&inTensor, &wTensor, &biasTensor, &kernel, 1, &outTensor);

    TEST_ASSERT_EQUAL_MEMORY(kBfpConvExpected, outTensor.data,
                             kBfpConvExpected_len * sizeof(float));
}

/* BFP power-of-two twin, the BFP sibling of the layer-level
 * testConv1dForwardGroupedEqualScalesBitIdenticalToScalar (UnitTestConv1d.c)
 * asserted at KERNEL level, mirroring UnitTestMatmul.c's
 * testMatmulBfpPowerOfTwoBitIdenticalToGroupedSym: identical mantissas (the
 * gold fixture's codes); BFP weight grouped {numGroups=6, groupSize=2} with
 * every stored exponent 125 (e=8, bias 127 -> 2^-2 == 0.25f) <-> SYM twin
 * scales all 0.25f; BFP input per-tensor stored 126 (2^-1 == 0.5f) <-> SYM
 * inScale 0.5f. All products and partials are far below 2^24 (exact floats),
 * every BFP fold is a pure exponent shift, and the SYM side's equal-power-
 * of-two rescales are exact round trips -- so the BFP float output must be
 * BIT-IDENTICAL to the SYM grouped path's dequantized output, clipped
 * stride-2 windows included. */
void testConv1dKernelBfpPowerOfTwoBitIdenticalToGroupedSym(void) {
    size_t inDims[] = {(size_t)kBfpConvBatch, (size_t)kBfpConvInChannels,
                       (size_t)kBfpConvInputLength};
    size_t inOrder[] = {0, 1, 2};
    shape_t inShape;
    setShape(&inShape, inDims, 3, inOrder);
    size_t wDims[] = {(size_t)kBfpConvOutChannels, (size_t)kBfpConvInChannels,
                      (size_t)kBfpConvKernelSize};
    size_t wOrder[] = {0, 1, 2};
    shape_t wShape;
    setShape(&wShape, wDims, 3, wOrder);
    size_t outDims[] = {(size_t)kBfpConvBatch, (size_t)kBfpConvOutChannels, (size_t)kBfpConvOutLen};
    size_t outOrder[] = {0, 1, 2};
    shape_t outShape;
    setShape(&outShape, outDims, 3, outOrder);

    kernel_t kernel;
    initKernelExplicit(&kernel, (size_t)kBfpConvKernelSize, (size_t)kBfpConvPadding, 1,
                       (size_t)kBfpConvStride);

    tensor_t inBfpTensor;
    uint8_t inExponents[] = {126}; /* 2^(126-127) == 0.5f */
    bfpQConfig_t inBfpQC = {.exponents = inExponents,
                            .numGroups = 1,
                            .groupSize = 0,
                            .roundingMode = HALF_AWAY,
                            .mantissaBits = 8,
                            .exponentBits = 8};
    quantization_t inBfpQ;
    initBfpQuantization(&inBfpQC, &inBfpQ);
    setTensorValues(&inBfpTensor, (uint8_t *)kBfpConvInCodes, &inShape, &inBfpQ, NULL);

    tensor_t wBfpTensor;
    uint8_t wExponents[] = {125, 125, 125, 125, 125, 125}; /* 2^(125-127) == 0.25f */
    bfpQConfig_t wBfpQC = {.exponents = wExponents,
                           .numGroups = 6,
                           .groupSize = 2,
                           .roundingMode = HALF_AWAY,
                           .mantissaBits = 8,
                           .exponentBits = 8};
    quantization_t wBfpQ;
    initBfpQuantization(&wBfpQC, &wBfpQ);
    setTensorValues(&wBfpTensor, (uint8_t *)kBfpConvWCodes, &wShape, &wBfpQ, NULL);

    tensor_t outBfpTensor;
    float outBfpData[10];
    quantization_t outBfpQ;
    initFloat32Quantization(&outBfpQ);
    setTensorValues(&outBfpTensor, (uint8_t *)outBfpData, &outShape, &outBfpQ, NULL);

    conv1dKernelBfp(&inBfpTensor, &wBfpTensor, NULL, &kernel, 1, &outBfpTensor);

    tensor_t inSymTensor;
    symInt32QConfig_t inSymQC;
    initSymInt32QConfig(HALF_AWAY, &inSymQC);
    inSymQC.scale = odtTestScale(0.5f);
    quantization_t inSymQ;
    initSymInt32Quantization(&inSymQC, &inSymQ);
    setTensorValues(&inSymTensor, (uint8_t *)kBfpConvInCodes, &inShape, &inSymQ, NULL);

    tensor_t wSymTensor;
    symInt32QConfig_t wSymQC;
    initSymInt32QConfig(HALF_AWAY, &wSymQC);
    wSymQC.scale =
        odtTestScale(1.0f); /* poisoned scratch scale -- never read, scales live in weightGroups */
    quantization_t wSymQ;
    initSymInt32Quantization(&wSymQC, &wSymQ);
    setTensorValues(&wSymTensor, (uint8_t *)kBfpConvWCodes, &wShape, &wSymQ, NULL);

    odtScale_t scales[6] = {odtTestScale(0.25f), odtTestScale(0.25f), odtTestScale(0.25f),
                            odtTestScale(0.25f), odtTestScale(0.25f), odtTestScale(0.25f)};
    symQConfig_t weightGroups = {
        .scales = scales, .numGroups = 6, .groupSize = 2, .qBits = 8, .roundingMode = HALF_AWAY};

    tensor_t outSymTensor;
    int32_t outSymData[10];
    symInt32QConfig_t outSymQC;
    initSymInt32QConfig(HALF_AWAY, &outSymQC);
    quantization_t outSymQ;
    initSymInt32Quantization(&outSymQC, &outSymQ);
    setTensorValues(&outSymTensor, (uint8_t *)outSymData, &outShape, &outSymQ, NULL);

    symGroupedWeight_t groupedWeight = symGroupedWeightSym(&weightGroups);
    conv1dKernelSymInt32Grouped(&inSymTensor, &wSymTensor, NULL, &kernel, 1, &outSymTensor,
                                &groupedWeight, NULL);

    for (size_t i = 0; i < 10; i++) {
        float symDequant = (float)outSymData[i] * odtScaleToFloat(outSymQC.scale);
        TEST_ASSERT_EQUAL_MEMORY(&symDequant, &outBfpData[i], sizeof(float));
    }
}

void testConv1dKernelBfpHeadroomGuardDies(void) {
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
    float outData[3];
    size_t outDims[] = {1, 1, 3};
    size_t outOrder[] = {0, 1, 2};
    shape_t outShape;
    setShape(&outShape, outDims, 3, outOrder);
    quantization_t outQ;
    initFloat32Quantization(&outQ);
    setTensorValues(&outTensor, (uint8_t *)outData, &outShape, &outQ, NULL);

    kernel_t kernel;
    initKernel(&kernel, 2, VALID, 1, 1);

    ASSERT_EXITS_WITH_FAILURE(conv1dKernelBfp(&inTensor, &wTensor, NULL, &kernel, 1, &outTensor));
}

/* conv-groups=2 parity gold: SPLIT channels
 * (inChPerGroup = outChPerGroup = 2, not depthwise), so the group loop's
 * in_lo/w_base channel arithmetic is load-bearing -- the other fixtures
 * all run groups=1. Same stack-fixture idiom and EXPLICIT-padded stride-2
 * geometry family as testConv1dKernelBfpMatchesGold, gold from
 * conv1d_bfp_ref(conv_groups=2) with the full vacuity self-checks under
 * split channels. */
void testConv1dKernelBfpConvGroups2MatchesGold(void) {
    tensor_t inTensor;
    size_t inDims[] = {(size_t)kBfpConvBatch, (size_t)kBfpConvG2InChannels,
                       (size_t)kBfpConvG2InputLength};
    size_t inOrder[] = {0, 1, 2};
    shape_t inShape;
    setShape(&inShape, inDims, 3, inOrder);
    uint8_t inExponents[sizeof(kBfpConvG2InExponents)];
    memcpy(inExponents, kBfpConvG2InExponents, sizeof(inExponents));
    bfpQConfig_t inQC = {.exponents = inExponents,
                         .numGroups = (size_t)kBfpConvG2InNumGroups,
                         .groupSize = (size_t)kBfpConvG2InGroupSize,
                         .roundingMode = HALF_AWAY,
                         .mantissaBits = (uint8_t)kBfpConvG2InMantissaBits,
                         .exponentBits = (uint8_t)kBfpConvG2InExponentBits};
    quantization_t inQ;
    initBfpQuantization(&inQC, &inQ);
    setTensorValues(&inTensor, (uint8_t *)kBfpConvG2InCodes, &inShape, &inQ, NULL);

    tensor_t wTensor;
    size_t wDims[] = {(size_t)kBfpConvG2OutChannels,
                      (size_t)kBfpConvG2InChannels / (size_t)kBfpConvG2ConvGroups,
                      (size_t)kBfpConvKernelSize};
    size_t wOrder[] = {0, 1, 2};
    shape_t wShape;
    setShape(&wShape, wDims, 3, wOrder);
    uint8_t wExponents[sizeof(kBfpConvG2WExponents)];
    memcpy(wExponents, kBfpConvG2WExponents, sizeof(wExponents));
    bfpQConfig_t wQC = {.exponents = wExponents,
                        .numGroups = (size_t)kBfpConvG2WNumGroups,
                        .groupSize = (size_t)kBfpConvG2WGroupSize,
                        .roundingMode = HALF_AWAY,
                        .mantissaBits = (uint8_t)kBfpConvG2WMantissaBits,
                        .exponentBits = (uint8_t)kBfpConvG2WExponentBits};
    quantization_t wQ;
    initBfpQuantization(&wQC, &wQ);
    setTensorValues(&wTensor, (uint8_t *)kBfpConvG2WCodes, &wShape, &wQ, NULL);

    tensor_t biasTensor;
    size_t biasDims[] = {(size_t)kBfpConvG2OutChannels};
    size_t biasOrder[] = {0};
    shape_t biasShape;
    setShape(&biasShape, biasDims, 1, biasOrder);
    uint8_t biasExponents[sizeof(kBfpConvG2BiasExponents)];
    memcpy(biasExponents, kBfpConvG2BiasExponents, sizeof(biasExponents));
    bfpQConfig_t biasQC = {.exponents = biasExponents,
                           .numGroups = 1,
                           .groupSize = 0,
                           .roundingMode = HALF_AWAY,
                           .mantissaBits = (uint8_t)kBfpConvG2BiasMantissaBits,
                           .exponentBits = (uint8_t)kBfpConvG2BiasExponentBits};
    quantization_t biasQ;
    initBfpQuantization(&biasQC, &biasQ);
    setTensorValues(&biasTensor, (uint8_t *)kBfpConvG2BiasCodes, &biasShape, &biasQ, NULL);

    tensor_t outTensor;
    float outData[20];
    size_t outDims[] = {(size_t)kBfpConvBatch, (size_t)kBfpConvG2OutChannels,
                        (size_t)kBfpConvG2OutLen};
    size_t outOrder[] = {0, 1, 2};
    shape_t outShape;
    setShape(&outShape, outDims, 3, outOrder);
    quantization_t outQ;
    initFloat32Quantization(&outQ);
    setTensorValues(&outTensor, (uint8_t *)outData, &outShape, &outQ, NULL);

    kernel_t kernel;
    initKernelExplicit(&kernel, (size_t)kBfpConvKernelSize, (size_t)kBfpConvPadding, 1,
                       (size_t)kBfpConvStride); // size, padding, dilation, stride

    conv1dKernelBfp(&inTensor, &wTensor, &biasTensor, &kernel, (size_t)kBfpConvG2ConvGroups,
                    &outTensor);

    TEST_ASSERT_EQUAL_MEMORY(kBfpConvG2Expected, outTensor.data,
                             kBfpConvG2Expected_len * sizeof(float));
}

/* ---- Integer-only forward: conv1dKernelBfpRaw --------------------------
 *
 * Same operand form as conv1dKernelBfp; the output is a BFP_RAW tensor
 * (int32 mantissas + one UNBIASED int16 exponent per raw target group) that
 * the funnel re-blocks or dequantizes. Gold: generate_expected_bfp_conv1d.py
 * (conv1d_bfp_raw_ref), pinned bit for bit on mantissas AND exponents, plus
 * the float64 exact values within the documented fold bound
 * 1/2 (S+1) 2^E_acc. The scratch-operand and BFP_RAW stack fixtures are the
 * shared BfpRawFixture.h (the pool tests use them too). */

/* The raw's value per element, exactly: mant * 2^E in double (|mant| < 2^31,
 * |E| small: no rounding), against the float64 exact sum within the bound. */
static void assertRawWithinBound(const bfpRawFixture_t *f, size_t n, const double *exact,
                                 const double *bound) {
    size_t gsz = f->qc.groupSize == 0 ? n : f->qc.groupSize;
    for (size_t i = 0; i < n; i++) {
        double v = ldexp((double)f->mant[i], f->exps[i / gsz]);
        TEST_ASSERT_TRUE_MESSAGE(fabs(v - exact[i]) <= bound[i],
                                 "raw value outside the fold bound");
    }
}

/* The main fixture's operands (grouped input m6, grouped weight m4, clipped
 * stride-2 windows); `wExps` lets a test swap the weight's exponent table. */
static void bfpRawMainOperands(bfpScratchFixture_t *in, bfpScratchFixture_t *w,
                               const uint8_t *wExps) {
    size_t inDims[] = {(size_t)kBfpConvBatch, (size_t)kBfpConvInChannels,
                       (size_t)kBfpConvInputLength};
    bfpScratchInit(in, 3, inDims, kBfpConvInCodes, kBfpConvInExponents, (size_t)kBfpConvInNumGroups,
                   (size_t)kBfpConvInGroupSize, (uint8_t)kBfpConvInMantissaBits,
                   (uint8_t)kBfpConvInExponentBits);
    size_t wDims[] = {(size_t)kBfpConvOutChannels, (size_t)kBfpConvInChannels,
                      (size_t)kBfpConvKernelSize};
    bfpScratchInit(w, 3, wDims, kBfpConvWCodes, wExps, (size_t)kBfpConvWNumGroups,
                   (size_t)kBfpConvWGroupSize, (uint8_t)kBfpConvWMantissaBits,
                   (uint8_t)kBfpConvWExponentBits);
}

static void bfpRawMainBias(bfpScratchFixture_t *bias) {
    size_t biasDims[] = {(size_t)kBfpConvOutChannels};
    bfpScratchInit(bias, 1, biasDims, kBfpConvBiasCodes, kBfpConvBiasExponents, 1, 0,
                   (uint8_t)kBfpConvBiasMantissaBits, (uint8_t)kBfpConvBiasExponentBits);
}

static void bfpRawMainKernel(kernel_t *kernel, size_t dilation) {
    initKernelExplicit(kernel, (size_t)kBfpConvKernelSize, (size_t)kBfpConvPadding, dilation,
                       (size_t)kBfpConvStride);
}

void testConv1dKernelBfpRawMatchesGold(void) {
    bfpScratchFixture_t in, w, bias;
    bfpRawMainOperands(&in, &w, kBfpConvWExponents);
    bfpRawMainBias(&bias);
    size_t outDims[] = {(size_t)kBfpConvBatch, (size_t)kBfpConvOutChannels, (size_t)kBfpConvOutLen};
    bfpRawFixture_t out;
    bfpRawInit(&out, 3, outDims, 1, 0);
    kernel_t kernel;
    bfpRawMainKernel(&kernel, 1);

    conv1dKernelBfpRaw(&in.tensor, &w.tensor, &bias.tensor, &kernel, 1, &out.tensor);

    assertRawEquals(&out, kBfpConvRawMant, kBfpConvRawMant_len, kBfpConvRawExps, 1);
    assertRawWithinBound(&out, kBfpConvRawMant_len, kBfpConvRawExact, kBfpConvRawBound);
}

void testConv1dKernelBfpRawNoBiasMatchesGold(void) {
    bfpScratchFixture_t in, w;
    bfpRawMainOperands(&in, &w, kBfpConvWExponents);
    size_t outDims[] = {(size_t)kBfpConvBatch, (size_t)kBfpConvOutChannels, (size_t)kBfpConvOutLen};
    bfpRawFixture_t out;
    bfpRawInit(&out, 3, outDims, 1, 0);
    kernel_t kernel;
    bfpRawMainKernel(&kernel, 1);

    conv1dKernelBfpRaw(&in.tensor, &w.tensor, NULL, &kernel, 1, &out.tensor);

    assertRawEquals(&out, kBfpConvRawNoBiasMant, kBfpConvRawNoBiasMant_len, kBfpConvRawNoBiasExps,
                    1);
}

/* Grouped bias {2, 1}: each channel's bias segment is aligned by ITS group's
 * exponent (a group-0 collapse is off by a power of two on channel 1). */
void testConv1dKernelBfpRawGroupedBiasBindsPerGroupExponent(void) {
    bfpScratchFixture_t in, w, bias;
    bfpRawMainOperands(&in, &w, kBfpConvWExponents);
    size_t biasDims[] = {(size_t)kBfpConvOutChannels};
    bfpScratchInit(&bias, 1, biasDims, kBfpConvBiasGroupedCodes, kBfpConvBiasGroupedExponents,
                   (size_t)kBfpConvBiasGroupedNumGroups, 1, (uint8_t)kBfpConvBiasMantissaBits,
                   (uint8_t)kBfpConvBiasExponentBits);
    size_t outDims[] = {(size_t)kBfpConvBatch, (size_t)kBfpConvOutChannels, (size_t)kBfpConvOutLen};
    bfpRawFixture_t out;
    bfpRawInit(&out, 3, outDims, 1, 0);
    kernel_t kernel;
    bfpRawMainKernel(&kernel, 1);

    conv1dKernelBfpRaw(&in.tensor, &w.tensor, &bias.tensor, &kernel, 1, &out.tensor);

    assertRawEquals(&out, kBfpConvRawBiasGroupedMant, kBfpConvRawBiasGroupedMant_len,
                    kBfpConvRawBiasGroupedExps, 1);
}

/* Raw target groups of 2 over the [1, 2, 5] output: group 2 holds the last
 * element of channel 0 and the first of channel 1, whose row maxima differ
 * (channel 1's weight exponents are bumped by 3), so its exponent must be
 * the LARGER row's and the straddled elements align to it. */
void testConv1dKernelBfpRawStraddlingTargetGroupTakesTheLargerRow(void) {
    bfpScratchFixture_t in, w, bias;
    bfpRawMainOperands(&in, &w, kBfpConvRawGs2WExponents);
    bfpRawMainBias(&bias);
    size_t outDims[] = {(size_t)kBfpConvBatch, (size_t)kBfpConvOutChannels, (size_t)kBfpConvOutLen};
    bfpRawFixture_t out;
    bfpRawInit(&out, 3, outDims, kBfpConvRawGs2Mant_len / (size_t)kBfpConvRawGs2GroupSize,
               (size_t)kBfpConvRawGs2GroupSize);
    kernel_t kernel;
    bfpRawMainKernel(&kernel, 1);

    conv1dKernelBfpRaw(&in.tensor, &w.tensor, &bias.tensor, &kernel, 1, &out.tensor);

    assertRawEquals(&out, kBfpConvRawGs2Mant, kBfpConvRawGs2Mant_len, kBfpConvRawGs2Exps,
                    kBfpConvRawGs2Exps_len);
}

void testConv1dKernelBfpRawDilation2MatchesGold(void) {
    bfpScratchFixture_t in, w, bias;
    bfpRawMainOperands(&in, &w, kBfpConvWExponents);
    bfpRawMainBias(&bias);
    size_t outDims[] = {(size_t)kBfpConvBatch, (size_t)kBfpConvOutChannels,
                        (size_t)kBfpConvDilOutLen};
    bfpRawFixture_t out;
    bfpRawInit(&out, 3, outDims, 1, 0);
    kernel_t kernel;
    bfpRawMainKernel(&kernel, (size_t)kBfpConvDilDilation);

    conv1dKernelBfpRaw(&in.tensor, &w.tensor, &bias.tensor, &kernel, 1, &out.tensor);

    assertRawEquals(&out, kBfpConvRawDilMant, kBfpConvRawDilMant_len, kBfpConvRawDilExps, 1);
}

void testConv1dKernelBfpRawConvGroups2MatchesGold(void) {
    bfpScratchFixture_t in, w, bias;
    size_t inDims[] = {(size_t)kBfpConvBatch, (size_t)kBfpConvG2InChannels,
                       (size_t)kBfpConvG2InputLength};
    bfpScratchInit(&in, 3, inDims, kBfpConvG2InCodes, kBfpConvG2InExponents,
                   (size_t)kBfpConvG2InNumGroups, (size_t)kBfpConvG2InGroupSize,
                   (uint8_t)kBfpConvG2InMantissaBits, (uint8_t)kBfpConvG2InExponentBits);
    size_t wDims[] = {(size_t)kBfpConvG2OutChannels,
                      (size_t)kBfpConvG2InChannels / (size_t)kBfpConvG2ConvGroups,
                      (size_t)kBfpConvKernelSize};
    bfpScratchInit(&w, 3, wDims, kBfpConvG2WCodes, kBfpConvG2WExponents,
                   (size_t)kBfpConvG2WNumGroups, (size_t)kBfpConvG2WGroupSize,
                   (uint8_t)kBfpConvG2WMantissaBits, (uint8_t)kBfpConvG2WExponentBits);
    size_t biasDims[] = {(size_t)kBfpConvG2OutChannels};
    bfpScratchInit(&bias, 1, biasDims, kBfpConvG2BiasCodes, kBfpConvG2BiasExponents, 1, 0,
                   (uint8_t)kBfpConvG2BiasMantissaBits, (uint8_t)kBfpConvG2BiasExponentBits);
    size_t outDims[] = {(size_t)kBfpConvBatch, (size_t)kBfpConvG2OutChannels,
                        (size_t)kBfpConvG2OutLen};
    bfpRawFixture_t out;
    bfpRawInit(&out, 3, outDims, 1, 0);
    kernel_t kernel;
    bfpRawMainKernel(&kernel, 1);

    conv1dKernelBfpRaw(&in.tensor, &w.tensor, &bias.tensor, &kernel, (size_t)kBfpConvG2ConvGroups,
                       &out.tensor);

    assertRawEquals(&out, kBfpConvG2RawMant, kBfpConvG2RawMant_len, kBfpConvG2RawExps, 1);
}

/* Both alignment arms inside ONE output: the input groups sit 24 binades
 * apart and the fold guard is negative, so output 3's max-exponent segment
 * is left-shifted exactly while its low-exponent segment is right-shifted
 * onto a HALF_AWAY tie (3.5 -> 4). */
void testConv1dKernelBfpRawAlignsTwoSided(void) {
    bfpScratchFixture_t in, w;
    size_t inDims[] = {1, 1, (size_t)kBfpConvSpreadInputLength};
    bfpScratchInit(&in, 3, inDims, kBfpConvSpreadInCodes, kBfpConvSpreadInExponents,
                   kBfpConvSpreadInExponents_len, (size_t)kBfpConvSpreadInGroupSize,
                   (uint8_t)kBfpConvSpreadInMantissaBits, 8);
    size_t wDims[] = {1, 1, 2};
    bfpScratchInit(&w, 3, wDims, kBfpConvSpreadWCodes, kBfpConvSpreadWExponents, 1, 0,
                   (uint8_t)kBfpConvSpreadWMantissaBits, 8);
    size_t outDims[] = {1, 1, (size_t)kBfpConvSpreadOutLen};
    bfpRawFixture_t out;
    bfpRawInit(&out, 3, outDims, 1, 0);
    kernel_t kernel;
    initKernel(&kernel, 2, VALID, 1, 1);

    conv1dKernelBfpRaw(&in.tensor, &w.tensor, NULL, &kernel, 1, &out.tensor);

    TEST_ASSERT_TRUE(kBfpConvSpreadG < 0);
    assertRawEquals(&out, kBfpConvSpreadRawMant, kBfpConvSpreadRawMant_len, kBfpConvSpreadRawExps,
                    1);
    assertRawWithinBound(&out, kBfpConvSpreadRawMant_len, kBfpConvSpreadRawExact,
                         kBfpConvSpreadRawBound);
    /* The rounding arm is observable: output 3 is not exact. */
    TEST_ASSERT_TRUE(ldexp((double)out.mant[3], out.exps[0]) != kBfpConvSpreadRawExact[3]);
}

/* m16 x m16: bfpSegmentLimit == 1, so every product is its own segment --
 * the same operands conv1dKernelBfp fails fast on (headroom guard) now RUN
 * through run splitting, within the fold bound. */
void testConv1dKernelBfpRawSplitsEveryProductAtM16(void) {
    bfpScratchFixture_t in, w;
    uint8_t unit[] = {127};
    size_t inDims[] = {1, 1, 4};
    bfpScratchInit(&in, 3, inDims, kBfpConvM16InCodes, unit, 1, 0, 16, 8);
    size_t wDims[] = {1, 1, 2};
    bfpScratchInit(&w, 3, wDims, kBfpConvM16WCodes, unit, 1, 0, 16, 8);
    size_t outDims[] = {1, 1, 3};
    bfpRawFixture_t out;
    bfpRawInit(&out, 3, outDims, 1, 0);
    kernel_t kernel;
    initKernel(&kernel, 2, VALID, 1, 1);

    conv1dKernelBfpRaw(&in.tensor, &w.tensor, NULL, &kernel, 1, &out.tensor);

    assertRawEquals(&out, kBfpConvM16RawMant, kBfpConvM16RawMant_len, kBfpConvM16RawExps, 1);
    assertRawWithinBound(&out, kBfpConvM16RawMant_len, kBfpConvM16RawExact, kBfpConvM16RawBound);
}

/* Dilation 2 over input groups of 3: output 1 taps inputs 1, 3, 5, so a
 * dilated step crosses an input-group boundary that a unit-step run count
 * would miss (tap 3 folded at group 0's exponent). */
void testConv1dKernelBfpRawDilatedRunStopsAtTheInputGroupEnd(void) {
    bfpScratchFixture_t in, w;
    size_t inDims[] = {1, 1, 9};
    bfpScratchInit(&in, 3, inDims, kBfpConvDilGrpInCodes, kBfpConvDilGrpInExponents, 3, 3, 4, 8);
    uint8_t unit[] = {127};
    size_t wDims[] = {1, 1, 3};
    bfpScratchInit(&w, 3, wDims, kBfpConvDilGrpWCodes, unit, 1, 0, 4, 8);
    size_t outDims[] = {1, 1, 5};
    bfpRawFixture_t out;
    bfpRawInit(&out, 3, outDims, 1, 0);
    kernel_t kernel;
    initKernel(&kernel, 3, VALID, 2, 1);

    conv1dKernelBfpRaw(&in.tensor, &w.tensor, NULL, &kernel, 1, &out.tensor);

    assertRawEquals(&out, kBfpConvDilGrpRawMant, kBfpConvDilGrpRawMant_len, kBfpConvDilGrpRawExps,
                    1);
    assertRawWithinBound(&out, kBfpConvDilGrpRawMant_len, kBfpConvDilGrpRawExact,
                         kBfpConvDilGrpRawBound);
}

/* m16 x m8 at the code floors: every product is exactly 2^22 and a 600-tap
 * reduction overflows int32 unless the run splits at bfpSegmentLimit (511);
 * the two partials align exactly under g = 1. */
void testConv1dKernelBfpRawSplitsLongRunAtTheSegmentLimit(void) {
    static int32_t xCodes[600];
    static int32_t wCodes[600];
    for (size_t i = 0; i < (size_t)kBfpConvLongTaps; i++) {
        xCodes[i] = -32768;
        wCodes[i] = -128;
    }
    bfpScratchFixture_t in, w;
    uint8_t unit[] = {127};
    size_t inDims[] = {1, 1, (size_t)kBfpConvLongTaps};
    bfpScratchInit(&in, 3, inDims, xCodes, unit, 1, 0, 16, 8);
    size_t wDims[] = {1, 1, (size_t)kBfpConvLongTaps};
    bfpScratchInit(&w, 3, wDims, wCodes, unit, 1, 0, 8, 8);
    size_t outDims[] = {1, 1, 1};
    bfpRawFixture_t out;
    bfpRawInit(&out, 3, outDims, 1, 0);
    kernel_t kernel;
    initKernel(&kernel, (size_t)kBfpConvLongTaps, VALID, 1, 1);

    conv1dKernelBfpRaw(&in.tensor, &w.tensor, NULL, &kernel, 1, &out.tensor);

    int16_t expExp = (int16_t)kBfpConvLongRawExp;
    assertRawEquals(&out, &kBfpConvLongRawMant, 1, &expExp, 1);
}

/* ---- Worst-case headroom: every code at a grid edge, at the proof's limits.
 * The exact sum is formed in int64 HERE (the test's reference; the kernel
 * never uses int64) and the raw must reproduce it exactly: mant * 2^E_acc ==
 * sum, with the expected g. Under a guard search that returned 1 for every
 * g >= 2 these accumulators overflowed. */

#define WORST_MAX_TAPS 2400
static int32_t gWorstX[WORST_MAX_TAPS];
static int32_t gWorstW[WORST_MAX_TAPS];

/* One output of a [1, 1, taps] x [1, 1, taps] reduction (K = taps, VALID),
 * every x = xCode, every w = wCode, both per-tensor at stored exponent 127
 * (unbiased 0); optional bias (code, stored exponent). */
static void runWorstCase(size_t taps, int32_t xCode, uint8_t mx, int32_t wCode, uint8_t mw,
                         const int32_t *biasCode, uint8_t mBias, uint8_t biasStored,
                         bfpRawFixture_t *out) {
    TEST_ASSERT_TRUE(taps <= WORST_MAX_TAPS);
    for (size_t i = 0; i < taps; i++) {
        gWorstX[i] = xCode;
        gWorstW[i] = wCode;
    }
    bfpScratchFixture_t in, w, bias;
    uint8_t unit[] = {127};
    size_t inDims[] = {1, 1, taps};
    bfpScratchInit(&in, 3, inDims, gWorstX, unit, 1, 0, mx, 8);
    size_t wDims[] = {1, 1, taps};
    bfpScratchInit(&w, 3, wDims, gWorstW, unit, 1, 0, mw, 8);
    if (biasCode != NULL) {
        uint8_t biasExp[] = {biasStored};
        size_t biasDims[] = {1};
        bfpScratchInit(&bias, 1, biasDims, biasCode, biasExp, 1, 0, mBias, 8);
    }
    size_t outDims[] = {1, 1, 1};
    bfpRawInit(out, 3, outDims, 1, 0);
    kernel_t kernel;
    initKernel(&kernel, taps, VALID, 1, 1);
    conv1dKernelBfpRaw(&in.tensor, &w.tensor, biasCode != NULL ? &bias.tensor : NULL, &kernel, 1,
                       &out->tensor);
}

static void assertRawIsExactly(const bfpRawFixture_t *out, int64_t exactSum, int expectedE) {
    TEST_ASSERT_EQUAL_INT16((int16_t)expectedE, out->exps[0]);
    TEST_ASSERT_TRUE(expectedE >= 0);
    int64_t value = (int64_t)out->mant[0] * ((int64_t)1 << (unsigned)expectedE);
    TEST_ASSERT_EQUAL_INT64(exactSum, value);
}

/* m16 x m16, N = 80 (g = 6, every product its own segment): all codes at the
 * floor -2^15 -> 80 products of exactly 2^30; then the floor against the
 * ceiling +2^15-1. Both exact in the raw. */
void testConv1dKernelBfpRawWorstCaseM16xM16AllCodesAtTheEdges(void) {
    bfpRawFixture_t out;
    runWorstCase(80, -32768, 16, -32768, 16, NULL, 0, 0, &out);
    assertRawIsExactly(&out, (int64_t)80 * ((int64_t)1 << 30), 6);
    TEST_ASSERT_EQUAL_INT32(80 << 24, out.mant[0]);

    runWorstCase(80, -32768, 16, 32767, 16, NULL, 0, 0, &out);
    assertRawIsExactly(&out, (int64_t)80 * (-32768LL * 32767LL), 6);
}

/* m16 x m8, N = 2400 (g = 3): the run splits at 511 products (four full
 * segments + 356), every product exactly 2^22; exact in the raw. */
void testConv1dKernelBfpRawWorstCaseM16xM8SplitRunsAtTheEdges(void) {
    bfpRawFixture_t out;
    runWorstCase(2400, -32768, 16, -128, 8, NULL, 0, 0, &out);
    assertRawIsExactly(&out, (int64_t)2400 * ((int64_t)1 << 22), 3);

    runWorstCase(2400, 32767, 16, -128, 8, NULL, 0, 0, &out);
    assertRawIsExactly(&out, (int64_t)2400 * (32767LL * -128LL), 3);
}

/* m16 x m16, N = 4 with a m16 bias at its floor (g = 2): the bias segment
 * is the extra term of the proof; exact. */
void testConv1dKernelBfpRawWorstCaseWithBiasAtTheFloor(void) {
    int32_t biasCode = -32768;
    bfpRawFixture_t out;
    runWorstCase(4, -32768, 16, -32768, 16, &biasCode, 16, 127, &out);
    assertRawIsExactly(&out, (int64_t)4 * ((int64_t)1 << 30) - 32768, 2);
}

/* Bias exponent ABOVE every product's (stored 147 = +20 unbiased): E_top is
 * the bias exponent, the products (m8 x m8, N = 80, one segment of 80 * 2^14)
 * are right-shifted by 10 exactly, the bias is left-shifted by 10 (g = -10);
 * the raw reproduces the int64 sum exactly. */
void testConv1dKernelBfpRawBiasExponentDominatesTheProducts(void) {
    int32_t biasCode = -128;
    bfpRawFixture_t out;
    runWorstCase(80, -128, 8, -128, 8, &biasCode, 8, 147, &out);
    assertRawIsExactly(&out, (int64_t)80 * ((int64_t)1 << 14) + (int64_t)-128 * ((int64_t)1 << 20),
                       10);
    TEST_ASSERT_EQUAL_INT32(-129792, out.mant[0]);
}

/* ---- Zero groups never set E_acc -------------------------------------------
 * An all-zero group stores the zero state (stored = bias, E = 0). Live data
 * 28 binades lower must keep every bit: E_acc follows the groups holding a
 * nonzero code, never a zero group's exponent (a zero code adds nothing to
 * any sum, so the headroom proof does not need it). Checked against the
 * exact int64 value of every output, not a mirror of the rule. */

static const int32_t kZgLive[] = {113, -77, 91, -59};
static const int32_t kZgW[] = {101, -67, 45};

/* conv1dKernelBfpRaw with one output channel, K = 3 VALID, per-tensor raw;
 * returns the exact sum of output i in units of 2^eLive. */
static int64_t zgConvExact(const int32_t *x, size_t i) {
    int64_t s = 0;
    for (size_t k = 0; k < 3; k++) {
        s += (int64_t)x[i + k] * kZgW[k];
    }
    return s;
}

/* Input [1, 1, 8] in groups of 4: group 0 all zero in the zero state, group 1
 * live at E = -28; weight per tensor at E = 0. */
void testConv1dKernelBfpRawZeroInputGroupDoesNotSetTheAccumulator(void) {
    int32_t x[8] = {0, 0, 0, 0};
    memcpy(&x[4], kZgLive, sizeof(kZgLive));
    uint8_t xE[] = {127, 127 - 28};
    uint8_t wE[] = {127};
    bfpScratchFixture_t in, w;
    size_t inDims[] = {1, 1, 8};
    size_t wDims[] = {1, 1, 3};
    bfpScratchInit(&in, 3, inDims, x, xE, 2, 4, 8, 8);
    bfpScratchInit(&w, 3, wDims, kZgW, wE, 1, 0, 8, 8);
    bfpRawFixture_t out;
    size_t outDims[] = {1, 1, 6};
    bfpRawInit(&out, 3, outDims, 1, 0);
    kernel_t kernel;
    initKernel(&kernel, 3, VALID, 1, 1);

    conv1dKernelBfpRaw(&in.tensor, &w.tensor, NULL, &kernel, 1, &out.tensor);

    for (size_t i = 0; i < 6; i++) {
        assertBfpValueExact(out.mant[i], out.exps[0], zgConvExact(x, i), -28, i);
    }
}

/* A zero bias code (zero state, E = 0) next to products at E = -28. */
void testConv1dKernelBfpRawZeroBiasDoesNotSetTheAccumulator(void) {
    uint8_t xE[] = {127 - 28};
    uint8_t wE[] = {127};
    int32_t biasCode[] = {0};
    uint8_t biasE[] = {127};
    bfpScratchFixture_t in, w, bias;
    size_t inDims[] = {1, 1, 4};
    size_t wDims[] = {1, 1, 3};
    size_t biasDims[] = {1};
    bfpScratchInit(&in, 3, inDims, kZgLive, xE, 1, 0, 8, 8);
    bfpScratchInit(&w, 3, wDims, kZgW, wE, 1, 0, 8, 8);
    bfpScratchInit(&bias, 1, biasDims, biasCode, biasE, 1, 0, 8, 8);
    bfpRawFixture_t out;
    size_t outDims[] = {1, 1, 2};
    bfpRawInit(&out, 3, outDims, 1, 0);
    kernel_t kernel;
    initKernel(&kernel, 3, VALID, 1, 1);

    conv1dKernelBfpRaw(&in.tensor, &w.tensor, &bias.tensor, &kernel, 1, &out.tensor);

    for (size_t i = 0; i < 2; i++) {
        assertBfpValueExact(out.mant[i], out.exps[0], zgConvExact(kZgLive, i), -28, i);
    }
}

/* Exact power-of-two grids: the raw's dequant (float)mant * 2^E must be
 * bit-identical to the float kernel's rounding-free fold on the same codes. */
void testConv1dKernelBfpRawPowerOfTwoDequantMatchesFloatKernel(void) {
    uint8_t inExps[] = {126};
    uint8_t wExps[] = {125, 125, 125, 125, 125, 125};
    bfpScratchFixture_t in, w;
    size_t inDims[] = {(size_t)kBfpConvBatch, (size_t)kBfpConvInChannels,
                       (size_t)kBfpConvInputLength};
    bfpScratchInit(&in, 3, inDims, kBfpConvInCodes, inExps, 1, 0, 8, 8);
    size_t wDims[] = {(size_t)kBfpConvOutChannels, (size_t)kBfpConvInChannels,
                      (size_t)kBfpConvKernelSize};
    bfpScratchInit(&w, 3, wDims, kBfpConvWCodes, wExps, 6, 2, 8, 8);
    size_t outDims[] = {(size_t)kBfpConvBatch, (size_t)kBfpConvOutChannels, (size_t)kBfpConvOutLen};
    bfpRawFixture_t raw;
    bfpRawInit(&raw, 3, outDims, 1, 0);
    kernel_t kernel;
    bfpRawMainKernel(&kernel, 1);

    conv1dKernelBfpRaw(&in.tensor, &w.tensor, NULL, &kernel, 1, &raw.tensor);

    tensor_t floatOut;
    float floatData[10];
    shape_t floatShape;
    size_t floatOrder[] = {0, 1, 2};
    setShape(&floatShape, outDims, 3, floatOrder);
    quantization_t floatQ;
    initFloat32Quantization(&floatQ);
    setTensorValues(&floatOut, (uint8_t *)floatData, &floatShape, &floatQ, NULL);
    conv1dKernelBfp(&in.tensor, &w.tensor, NULL, &kernel, 1, &floatOut);

    for (size_t i = 0; i < 10; i++) {
        float deq = ldexpf((float)raw.mant[i], raw.exps[0]);
        TEST_ASSERT_EQUAL_MEMORY(&floatData[i], &deq, sizeof(float));
    }
}

static void conv1dKernelBfpRawBadGeometryBody(void) {
    bfpScratchFixture_t in, w;
    bfpRawMainOperands(&in, &w, kBfpConvWExponents);
    size_t outDims[] = {(size_t)kBfpConvBatch, (size_t)kBfpConvOutChannels, (size_t)kBfpConvOutLen};
    bfpRawFixture_t out;
    bfpRawInit(&out, 3, outDims, 3, 4); /* 3 * 4 != 10 */
    kernel_t kernel;
    bfpRawMainKernel(&kernel, 1);
    conv1dKernelBfpRaw(&in.tensor, &w.tensor, NULL, &kernel, 1, &out.tensor);
}

void testConv1dKernelBfpRawRejectsMismatchedRawGeometry(void) {
    ASSERT_EXITS_WITH_FAILURE(conv1dKernelBfpRawBadGeometryBody());
}

void setUp() {}
void tearDown() {}

/* ---- SYM int32 accumulator guard (SymAccumGuard.h) ---------------------- */

#define CG_MAXN 1100
static int32_t gCgX[CG_MAXN];
static int32_t gCgW[CG_MAXN];
/* The output wire outlives cgRun so a chain test can feed it on as an operand
 * with the quantization the kernel stamped. */
static int32_t gCgY[CG_MAXN];
static symInt32QConfig_t gCgYQC;
static quantization_t gCgYQ;
static uint8_t gCgOutBits; /* output qMaxBits after the kernel returned */

/* Code domain of the grouped weight cgRun hands the grouped kernel (a stored
 * SYM weight unless a test says ASYM image). */
static symCodeDomain_t gCgWeightDomain = SYM_CODES_TWOS_COMPLEMENT;

/* Output wires start at a width no stamp test expects, so a missing stamp can
 * never pass as the right one (the kernels never read the output width). */
#define UNSTAMPED_WIDTH 5

static void cgFill(size_t n, int32_t xCode, int32_t wCode, bool wAlternates) {
    for (size_t i = 0; i < n; i++) {
        gCgX[i] = xCode;
        gCgW[i] = (wAlternates && (i % 2 == 1)) ? -wCode : wCode;
    }
}

static int32_t cgExpected(size_t n) {
    int64_t sum = 0;
    for (size_t i = 0; i < n; i++) {
        sum += (int64_t)gCgX[i] * gCgW[i];
    }
    return (int32_t)sum;
}

/* One output element: x [1, inCh, len], w [1, inCh, K] (flat reduction order
 * ic-major, tap-minor), VALID stride 1, so N = inCh * K terms when len == K.
 * numGroups == 0 -> per-tensor weight (conv1dKernelSymInt32); else the grouped
 * kernel with equal scales. biasCode != NULL adds a scale-1 bias. */
static int32_t cgRun(size_t inCh, size_t len, size_t k, uint8_t xBits, uint8_t wBits,
                     const int32_t *biasCode, size_t numGroups) {
    size_t outLen = len - k + 1;
    size_t xDims[] = {1, inCh, len}, wDims[] = {1, inCh, k}, yDims[] = {1, 1, outLen};
    size_t order[] = {0, 1, 2}, biasDims[] = {1};
    shape_t xS, wS, yS, biasS;
    setShape(&xS, xDims, 3, order);
    setShape(&wS, wDims, 3, order);
    setShape(&yS, yDims, 3, order);
    setShape(&biasS, biasDims, 1, order);
    symInt32QConfig_t xQC, wQC, biasQC;
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &xQC, xBits);
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &wQC, wBits);
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &gCgYQC, UNSTAMPED_WIDTH);
    initSymInt32QConfig(HALF_AWAY, &biasQC);
    quantization_t xQ, wQ, biasQ;
    initSymInt32Quantization(&xQC, &xQ);
    initSymInt32Quantization(&wQC, &wQ);
    initSymInt32Quantization(&gCgYQC, &gCgYQ);
    initSymInt32Quantization(&biasQC, &biasQ);
    int32_t *y = gCgY;
    int32_t biasData = biasCode ? *biasCode : 0;
    tensor_t x, w, yT, bias;
    setTensorValues(&x, (uint8_t *)gCgX, &xS, &xQ, NULL);
    setTensorValues(&w, (uint8_t *)gCgW, &wS, &wQ, NULL);
    setTensorValues(&yT, (uint8_t *)y, &yS, &gCgYQ, NULL);
    setTensorValues(&bias, (uint8_t *)&biasData, &biasS, &biasQ, NULL);
    kernel_t kernel;
    initKernel(&kernel, k, VALID, 1, 1);
    if (numGroups == 0) {
        conv1dKernelSymInt32(&x, &w, biasCode ? &bias : NULL, &kernel, 1, &yT, NULL);
    } else {
        odtScale_t scales[8] = {odtTestScale(1), odtTestScale(1), odtTestScale(1), odtTestScale(1),
                                odtTestScale(1), odtTestScale(1), odtTestScale(1), odtTestScale(1)};
        symQConfig_t groups = {.scales = scales,
                               .numGroups = numGroups,
                               .groupSize = inCh * k / numGroups,
                               .qBits = wBits,
                               .roundingMode = HALF_AWAY};
        symGroupedWeight_t groupedWeight = {.groups = &groups, .domain = gCgWeightDomain};
        conv1dKernelSymInt32Grouped(&x, &w, biasCode ? &bias : NULL, &kernel, 1, &yT,
                                    &groupedWeight, NULL);
    }
    gCgOutBits = gCgYQC.qMaxBits;
    return y[0];
}

/* cgRun with the grouped weight read as a stored ASYM weight's image. */
static int32_t cgRunAsymImage(size_t inCh, size_t len, size_t k, uint8_t xBits, uint8_t wBits,
                              const int32_t *biasCode, size_t numGroups) {
    gCgWeightDomain = SYM_CODES_ASYM_IMAGE;
    int32_t y = cgRun(inCh, len, k, xBits, wBits, biasCode, numGroups);
    gCgWeightDomain = SYM_CODES_TWOS_COMPLEMENT;
    return y;
}

/* Channel-aligned twin of cgRun's grouped branch: outC output channels, each
 * row (inCh * K codes, a copy of gCgW's first inCh * K) its own group of scale
 * 1, so groupSize == the row length and the kernel takes its cursor-free
 * aligned loop. One output element per channel, written to gCgY[oc]; every
 * channel gets the same bias code when biasCode != NULL. */
#define CG_ALIGNED_MAX_OUTC 2
static int32_t gCgWAligned[CG_ALIGNED_MAX_OUTC * CG_MAXN];

static void cgRunAligned(size_t outC, size_t inCh, size_t k, uint8_t bits,
                         const int32_t *biasCode) {
    TEST_ASSERT_TRUE(outC >= 2 && outC <= CG_ALIGNED_MAX_OUTC);
    size_t row = inCh * k;
    for (size_t oc = 0; oc < outC; oc++) {
        for (size_t i = 0; i < row; i++) {
            gCgWAligned[oc * row + i] = gCgW[i];
        }
    }
    size_t xDims[] = {1, inCh, k}, wDims[] = {outC, inCh, k}, yDims[] = {1, outC, 1};
    size_t order[] = {0, 1, 2}, biasDims[] = {outC};
    shape_t xS, wS, yS, biasS;
    setShape(&xS, xDims, 3, order);
    setShape(&wS, wDims, 3, order);
    setShape(&yS, yDims, 3, order);
    setShape(&biasS, biasDims, 1, order);
    symInt32QConfig_t xQC, wQC, biasQC;
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &xQC, bits);
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &wQC, bits);
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &gCgYQC, UNSTAMPED_WIDTH);
    initSymInt32QConfig(HALF_AWAY, &biasQC);
    quantization_t xQ, wQ, biasQ;
    initSymInt32Quantization(&xQC, &xQ);
    initSymInt32Quantization(&wQC, &wQ);
    initSymInt32Quantization(&gCgYQC, &gCgYQ);
    initSymInt32Quantization(&biasQC, &biasQ);
    int32_t biasData[CG_ALIGNED_MAX_OUTC];
    for (size_t oc = 0; oc < outC; oc++) {
        biasData[oc] = biasCode ? *biasCode : 0;
    }
    tensor_t x, w, yT, bias;
    setTensorValues(&x, (uint8_t *)gCgX, &xS, &xQ, NULL);
    setTensorValues(&w, (uint8_t *)gCgWAligned, &wS, &wQ, NULL);
    setTensorValues(&yT, (uint8_t *)gCgY, &yS, &gCgYQ, NULL);
    setTensorValues(&bias, (uint8_t *)biasData, &biasS, &biasQ, NULL);
    kernel_t kernel;
    initKernel(&kernel, k, VALID, 1, 1);
    odtScale_t scales[CG_ALIGNED_MAX_OUTC] = {odtTestScale(1), odtTestScale(1)};
    symQConfig_t groups = {.scales = scales,
                           .numGroups = outC,
                           .groupSize = row,
                           .qBits = bits,
                           .roundingMode = HALF_AWAY};
    symGroupedWeight_t groupedWeight = {.groups = &groups, .domain = SYM_CODES_TWOS_COMPLEMENT};
    conv1dKernelSymInt32Grouped(&x, &w, biasCode ? &bias : NULL, &kernel, 1, &yT, &groupedWeight,
                                NULL);
}

/* 600 terms of +-2047^2 per channel: unprovable, so the checked aligned loop
 * runs, and both channels come out exact. */
void testConv1dKernelSymGroupedAlignedGuardCheckedPathStaysExactWhenValuesFit(void) {
    cgFill(600, 2047, 2047, true);
    cgRunAligned(2, 600, 1, 12, NULL);
    TEST_ASSERT_EQUAL_INT32(cgExpected(600), gCgY[0]);
    TEST_ASSERT_EQUAL_INT32(cgExpected(600), gCgY[1]);
}

/* 512 terms of 2^22 per channel: the raw row partial reaches 2^31. */
void testConv1dKernelSymGroupedAlignedGuardDiesOnRawGroupPartialOverflow(void) {
    cgFill(512, -2048, -2048, false);
    ASSERT_EXITS_WITH_OUTPUT(1, "conv1dKernelSymInt32Grouped (raw group partial)",
                             cgRunAligned(2, 512, 1, 12, NULL));
}

/* 511 terms of 2^22 fit the partial; the 2^22 bias seed tips the rescaled
 * accumulator to 2^31. */
void testConv1dKernelSymGroupedAlignedGuardDiesOnRescaledAccumulatorOverflow(void) {
    int32_t seed = 4194304;
    cgFill(511, -2048, -2048, false);
    ASSERT_EXITS_WITH_OUTPUT(1, "conv1dKernelSymInt32Grouped (rescaled accumulator)",
                             cgRunAligned(2, 511, 1, 12, &seed));
}

void testConv1dKernelSymGuardProvenSafeN511RunsExact(void) {
    cgFill(511, -2048, -2048, false);
    TEST_ASSERT_EQUAL_INT32(511 * 4194304, cgRun(511, 1, 1, 12, 12, NULL, 0));
}

void testConv1dKernelSymGuardDiesAtFirstWrappingNChannels(void) {
    cgFill(512, -2048, -2048, false);
    ASSERT_EXITS_WITH_OUTPUT(1, "conv1dKernelSymInt32", (void)cgRun(512, 1, 1, 12, 12, NULL, 0));
}

void testConv1dKernelSymGuardDiesAtN513PositiveCodes(void) {
    cgFill(513, 2047, 2047, false);
    ASSERT_EXITS_WITH_OUTPUT(1, "N=513", (void)cgRun(513, 1, 1, 12, 12, NULL, 0));
}

void testConv1dKernelSymGuardCountsKernelTapsInN(void) {
    cgFill(512, -2048, -2048, false); /* inCh = 1, K = 512 taps drive the reduction */
    ASSERT_EXITS_WITH_OUTPUT(1, "N=512", (void)cgRun(1, 512, 512, 12, 12, NULL, 0));
}

void testConv1dKernelSymGuardCheckedPathStaysExactWhenValuesFit(void) {
    cgFill(601, 2047, 2047, true);
    TEST_ASSERT_EQUAL_INT32(cgExpected(601), cgRun(601, 1, 1, 12, 12, NULL, 0));
}

void testConv1dKernelSymGuardBiasSeedThatTipsTheBoundIsCounted(void) {
    int32_t seed = 4194304;
    cgFill(511, -2048, -2048, false);
    ASSERT_EXITS_WITH_OUTPUT(1, "max|seed|=4194304", (void)cgRun(511, 1, 1, 12, 12, &seed, 0));
    int32_t small = 4194303;
    TEST_ASSERT_EQUAL_INT32(511 * 4194304 + 4194303, cgRun(511, 1, 1, 12, 12, &small, 0));
}

void testConv1dKernelSymGuardWidthsFollowTheOperandsNotTheDefault(void) {
    /* int12 x int16: product <= 2^26, so 32 terms of -2048 x -32768 hit 2^31 */
    cgFill(31, -2048, -32768, false);
    TEST_ASSERT_EQUAL_INT32(31 * 67108864, cgRun(31, 1, 1, 12, 16, NULL, 0));
    cgFill(32, -2048, -32768, false);
    ASSERT_EXITS_WITH_OUTPUT(1, "widths (12, 16)", (void)cgRun(32, 1, 1, 12, 16, NULL, 0));
}

void testConv1dKernelSymGuardWideProductOverflowAborts(void) {
    /* 20-bit codes: one product (2^19)^2 = 2^38 already leaves int32 */
    cgFill(1, 524288, 524288, false);
    ASSERT_EXITS_WITH_OUTPUT(1, "conv1dKernelSymInt32", (void)cgRun(1, 1, 1, 20, 20, NULL, 0));
    cgFill(2, 100, 200, false); /* wide but fitting: checked path returns the exact sum */
    TEST_ASSERT_EQUAL_INT32(2 * 20000, cgRun(2, 1, 1, 20, 20, NULL, 0));
}

void testConv1dKernelSymGroupedGuardDiesOnRawGroupPartialOverflow(void) {
    cgFill(1024, -2048, -2048, false); /* 2 groups x 512 */
    ASSERT_EXITS_WITH_OUTPUT(1, "conv1dKernelSymInt32Grouped",
                             (void)cgRun(1024, 1, 1, 12, 12, NULL, 2));
}

void testConv1dKernelSymGroupedGuardDiesOnRescaledAccumulatorOverflow(void) {
    cgFill(1024, -2048, -2048, false); /* 4 x 256: partials 2^30 fit, their sum does not */
    ASSERT_EXITS_WITH_OUTPUT(1, "conv1dKernelSymInt32Grouped",
                             (void)cgRun(1024, 1, 1, 12, 12, NULL, 4));
}

void testConv1dKernelSymGroupedGuardCheckedPathStaysExactWhenValuesFit(void) {
    cgFill(600, 2047, 2047, true);
    TEST_ASSERT_EQUAL_INT32(cgExpected(600), cgRun(600, 1, 1, 12, 12, NULL, 3));
}

/* 511 full-scale int12 terms (7 groups): a stored SYM weight's exact range
 * proves them (511 * 2^22 fits) and the safe loop runs; a stored ASYM weight's
 * image (+-4095 reachable) is unprovable and the checked loop runs -- both exact. */
void testConv1dKernelSymGroupedGuardFullScaleN511RunsExact(void) {
    cgFill(511, -2048, -2048, false);
    TEST_ASSERT_EQUAL_INT32(511 * 4194304, cgRun(511, 1, 1, 12, 12, NULL, 7));
    TEST_ASSERT_EQUAL_INT32(511 * 4194304, cgRunAsymImage(511, 1, 1, 12, 12, NULL, 7));
}

/* The banner names the weight's exact range by its holding width: a stored SYM
 * int12 weight is 12 bits, not 13. */
void testConv1dKernelSymGroupedGuardWidthsFollowTheInputWidth(void) {
    cgFill(30, -32768, -2048, false);
    TEST_ASSERT_EQUAL_INT32(30 * 67108864, cgRun(30, 1, 1, 16, 12, NULL, 2));
    cgFill(32, -32768, -2048, false);
    ASSERT_EXITS_WITH_OUTPUT(1, "widths (16, 12)", (void)cgRun(32, 1, 1, 16, 12, NULL, 2));
}

void testConv1dKernelSymGroupedGuardWideInputProductOverflowAborts(void) {
    cgFill(2, -16777216, -2048, false); /* 25-bit input x int12 weight: one product is 2^35 */
    ASSERT_EXITS_WITH_OUTPUT(1, "conv1dKernelSymInt32Grouped",
                             (void)cgRun(2, 1, 1, 25, 12, NULL, 2));
}

void testConv1dKernelSymGroupedGuardCountsKernelTapsInN(void) {
    cgFill(512, -2048, -2048, false); /* inCh = 1, K = 512 taps in 2 groups */
    ASSERT_EXITS_WITH_OUTPUT(1, "N=512", (void)cgRun(1, 512, 512, 12, 12, NULL, 2));
}

/* A stored grouped ASYM weight unpacks to code - zp[g], reaching +-(2^qBits - 1)
 * = +-4095 at qBits 12; the kernel takes that range from the ctx's code domain:
 * 511 terms of 2047 x 4095 abort (the SYM range would have run them unchecked
 * and wrapped), 256 terms (2145911040) are proven at the exact image range and
 * run exact (the image's holding width 13 proves only 255). */
void testConv1dKernelSymGroupedGuardAsymImageWeightKeepsItsFullRange(void) {
    cgFill(511, 2047, 4095, false);
    ASSERT_EXITS_WITH_OUTPUT(1, "conv1dKernelSymInt32Grouped",
                             (void)cgRunAsymImage(511, 1, 1, 12, 12, NULL, 7));
    cgFill(256, 2047, 4095, false);
    TEST_ASSERT_EQUAL_INT32(256 * 2047 * 4095, cgRunAsymImage(256, 1, 1, 12, 12, NULL, 2));
}

/* A grouped weight wider than int12 used to exit at the validator; now it takes
 * the checked path (int12 x int16 products over 2 x 16 terms are unprovable).
 * 30 terms of -2048 x -32768 = 2013265920 fit int32 through two groups; 32
 * terms (2^31) wrap. */
void testConv1dKernelSymGroupedGuardWideWeightRunsExactWhenItFitsAndAbortsWhenItWraps(void) {
    cgFill(30, -2048, -32768, false);
    TEST_ASSERT_EQUAL_INT32(30 * 67108864, cgRun(30, 1, 1, 12, 16, NULL, 2));
    cgFill(32, -2048, -32768, false);
    ASSERT_EXITS_WITH_OUTPUT(1, "conv1dKernelSymInt32Grouped",
                             (void)cgRun(32, 1, 1, 12, 16, NULL, 2));
}

/* See the Matmul twin: two groups of 40 terms, raw partials 1342136320 (above
 * the seed reserve, inside int32). */
static void wideGroupedFill(bool secondGroupNegative) {
    cgFill(80, 1024, 32767, false);
    if (secondGroupNegative) {
        for (size_t i = 40; i < 80; i++) {
            gCgW[i] = -32767;
        }
        gCgW[79] = 0;
    }
}

void testConv1dKernelSymGroupedWidePartialAboveSeedGuardRunsExactWhenSumFits(void) {
    wideGroupedFill(true);
    TEST_ASSERT_EQUAL_INT32(1024 * 32767, cgRun(80, 1, 1, 12, 16, NULL, 2));
}

void testConv1dKernelSymGroupedWidePartialSumThatWrapsHitsTheAccumulatorGuard(void) {
    wideGroupedFill(false);
    ASSERT_EXITS_WITH_OUTPUT(1, "conv1dKernelSymInt32Grouped (rescaled accumulator)",
                             (void)cgRun(80, 1, 1, 12, 16, NULL, 2));
}

/* ---- The output declares the accumulator's static width ------------ */

/* int4 x int4 over inCh * K = 3 terms of code 1: the stamp is the static range
 * [-168, 192] -> 9, never the data (3). With a bias of 1000 at the accumulator
 * scale [-168 + 1000, 1192] -> 12; with a bias of -1870 [-2038, -1678] -> 12,
 * where max|seed| on both sides (2062) would say 13. */
void testConv1dKernelSymStampsResultWidthIncludesSeed(void) {
    cgFill(3, 1, 1, false);
    TEST_ASSERT_EQUAL_INT32(3, cgRun(3, 1, 1, 4, 4, NULL, 0));
    symAccumSite_t site = symAccumSiteInit("t", 3, 4, 4, 0);
    TEST_ASSERT_EQUAL_UINT8(symAccumResultBits(&site), gCgOutBits);
    TEST_ASSERT_EQUAL_UINT8(9, gCgOutBits);
    int32_t bias = 1000;
    TEST_ASSERT_EQUAL_INT32(1003, cgRun(3, 1, 1, 4, 4, &bias, 0));
    symAccumSite_t seeded = symAccumSiteInit("t", 3, 4, 4, 1000);
    TEST_ASSERT_EQUAL_UINT8(symAccumResultBits(&seeded), gCgOutBits);
    TEST_ASSERT_EQUAL_UINT8(12, gCgOutBits);
    bias = -1870;
    TEST_ASSERT_EQUAL_INT32(3 - 1870, cgRun(3, 1, 1, 4, 4, &bias, 0));
    TEST_ASSERT_EQUAL_UINT8(12, gCgOutBits);
}

/* int4 x grouped int4 (2 groups x 4) over 8 terms of code 1, scales 1: the
 * fold path with F = 8 (Conv1dKernel.c, SymGroupFold.h), so the codes are the
 * sum times 2^(F-1) = 128 at A = 2^-7 and a bias b seeds as 128b. The width is
 * the fold site's over the weight's EXACT code range (SymGroupedWeight.h): a
 * stored SYM int4 weight is [-8, 7] against x widened to [-2049, 1793], one
 * term spans [-14344, 16392], 8 terms [-114752, 131136] -> 19 bits seedless.
 * Bias 1024 (seed 131072) -> 262208 -> 20, bias 1023 -> 262080 -> 19, bias
 * -1200 -> -268352 on the negative side -> 20, bias -1023 -> 19. */
void testConv1dKernelSymGroupedStampsFoldWidthIncludesBiasSeed(void) {
    cgFill(8, 1, 1, false);
    TEST_ASSERT_EQUAL_INT32(8 * 128, cgRun(8, 1, 1, 4, 4, NULL, 2));
    symAccumSite_t fold = symAccumSiteInitRanges("t", 8, -2049, 1793, -8, 7, 0);
    TEST_ASSERT_EQUAL_UINT8(symAccumResultBits(&fold), gCgOutBits);
    TEST_ASSERT_EQUAL_UINT8(19, gCgOutBits);
    int32_t bias = 1024;
    TEST_ASSERT_EQUAL_INT32((8 + 1024) * 128, cgRun(8, 1, 1, 4, 4, &bias, 2));
    TEST_ASSERT_EQUAL_UINT8(20, gCgOutBits);
    bias = 1023;
    TEST_ASSERT_EQUAL_INT32((8 + 1023) * 128, cgRun(8, 1, 1, 4, 4, &bias, 2));
    TEST_ASSERT_EQUAL_UINT8(19, gCgOutBits);
    bias = -1200;
    TEST_ASSERT_EQUAL_INT32((8 - 1200) * 128, cgRun(8, 1, 1, 4, 4, &bias, 2));
    TEST_ASSERT_EQUAL_UINT8(20, gCgOutBits);
    bias = -1023;
    TEST_ASSERT_EQUAL_INT32((8 - 1023) * 128, cgRun(8, 1, 1, 4, 4, &bias, 2));
    TEST_ASSERT_EQUAL_UINT8(19, gCgOutBits);
}

/* The same fold site over a stored ASYM int4 weight's image [-15, 15]: +-30735
 * per term, 245880 per side -> 19 seedless (the image counted one bit wide,
 * [-16, 15], says 20); bias 500 (seed 64000) -> 309880 -> 20 (the SYM range
 * says 19). */
void testConv1dKernelSymGroupedAsymImageStampsItsFullRange(void) {
    cgFill(8, 1, 1, false);
    TEST_ASSERT_EQUAL_INT32(8 * 128, cgRunAsymImage(8, 1, 1, 4, 4, NULL, 2));
    TEST_ASSERT_EQUAL_UINT8(19, gCgOutBits);
    int32_t bias = 500;
    TEST_ASSERT_EQUAL_INT32((8 + 500) * 128, cgRunAsymImage(8, 1, 1, 4, 4, &bias, 2));
    TEST_ASSERT_EQUAL_UINT8(20, gCgOutBits);
}

/* A conv output chained into matmul: the raw 1x1x1 product -2048 * -2048 = 2^22
 * re-enters a SYM matmul through a 2-D view that shares the quantization the
 * kernel stamped. Declared 24 bits the (24, 12) site is unprovable and the
 * checked product aborts naming the stamped width; at the pre-stamp width the
 * guard proved one term safe and the product -2^33 wrapped silently. */
static void cgChainIntoMatmul(void) {
    cgFill(1, -2048, -2048, false);
    (void)cgRun(1, 1, 1, 12, 12, NULL, 0);
    size_t dims[] = {1, 1}, order[] = {0, 1};
    shape_t vS, cS, oS;
    setShape(&vS, dims, 2, order);
    setShape(&cS, dims, 2, order);
    setShape(&oS, dims, 2, order);
    symInt32QConfig_t cQC, oQC;
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &cQC, 12);
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &oQC, UNSTAMPED_WIDTH);
    quantization_t cQ, oQ;
    initSymInt32Quantization(&cQC, &cQ);
    initSymInt32Quantization(&oQC, &oQ);
    int32_t cCode = -2048, outCode = 0;
    tensor_t v, c, out;
    setTensorValues(&v, (uint8_t *)gCgY, &vS, &gCgYQ, NULL);
    setTensorValues(&c, (uint8_t *)&cCode, &cS, &cQ, NULL);
    setTensorValues(&out, (uint8_t *)&outCode, &oS, &oQ, NULL);
    matmulSymInt32Tensors(&v, &c, &out);
}

void testConv1dKernelSymChainedRawOutputOverflowAborts(void) {
    cgFill(1, -2048, -2048, false);
    TEST_ASSERT_EQUAL_INT32(4194304, cgRun(1, 1, 1, 12, 12, NULL, 0));
    TEST_ASSERT_EQUAL_UINT8(24, gCgOutBits); /* B = 2^11 * 2^11 = 2^22 -> 24 */
    ASSERT_EXITS_WITH_OUTPUT(1, "widths (24, 12)", cgChainIntoMatmul());
}

/* ---- Grouped-kernel goldens (LCG fixture) -------------------------------
 * kCgGold* were captured before the per-element group division became a
 * range cursor (that refactor was bit-identical, SR stream included); since
 * linq fold T4 they pin the LEGACY float path (17-bit input), and the int8
 * fixture's fold-path golds are kCgFold* (Python-int HALF_AWAY, kernel-recorded
 * SR). batch 2, Cin 2 (conv groups 1, so
 * inChPerGroup 2), L 8, K 5, stride 2, dilation 2: one weight row is 10 codes.
 * Every quantization group size {3, 4, 7, 10 (per channel)} runs under SAME
 * (padLeft 3), EXPLICIT 3, and EXPLICIT 9 (>= the dilated reach 9: output 0
 * is fully padded; output 1 sees only tap 4 per channel, weight indices 4 and
 * 9 of a row, so with group size 3 the walk jumps over a whole group). A row
 * is entered mid-group wherever left padding clips taps. Group scales are
 * distinct, dyadic, with one maximum, so all but one group rescale with a
 * factor != 1. */

typedef struct cgGoldCase {
    paddingType_t padType;
    size_t padding, groupSize, outC, outLen;
} cgGoldCase_t;

#define CG_GOLD_NUM_CASES 12
#define CG_GOLD_B 2
#define CG_GOLD_CIN 2
#define CG_GOLD_L 8
#define CG_GOLD_MAX_OUT (CG_GOLD_B * 7 * 9)

static const cgGoldCase_t kCgGoldCases[CG_GOLD_NUM_CASES] = {
    {SAME, 0, 3, 3, 4},     {SAME, 0, 4, 2, 4},     {SAME, 0, 7, 7, 4},     {SAME, 0, 10, 3, 4},
    {EXPLICIT, 3, 3, 3, 3}, {EXPLICIT, 3, 4, 2, 3}, {EXPLICIT, 3, 7, 7, 3}, {EXPLICIT, 3, 10, 3, 3},
    {EXPLICIT, 9, 3, 3, 9}, {EXPLICIT, 9, 4, 2, 9}, {EXPLICIT, 9, 7, 7, 9}, {EXPLICIT, 9, 10, 3, 9},
};

static const int32_t *const kCgGoldHa[CG_GOLD_NUM_CASES] = {
    kCgGoldHa0, kCgGoldHa1, kCgGoldHa2, kCgGoldHa3, kCgGoldHa4,  kCgGoldHa5,
    kCgGoldHa6, kCgGoldHa7, kCgGoldHa8, kCgGoldHa9, kCgGoldHa10, kCgGoldHa11};
static const int32_t *const kCgGoldSr[CG_GOLD_NUM_CASES] = {
    kCgGoldSr0, kCgGoldSr1, kCgGoldSr2, kCgGoldSr3, kCgGoldSr4,  kCgGoldSr5,
    kCgGoldSr6, kCgGoldSr7, kCgGoldSr8, kCgGoldSr9, kCgGoldSr10, kCgGoldSr11};

static uint32_t cgGoldLcg(uint32_t *s) {
    *s = *s * 1664525u + 1013904223u;
    return *s >> 8;
}

/* Nonzero code in [-r, -1] u [1, r]. */
static int32_t cgGoldCode(uint32_t *s, int32_t r) {
    int32_t v = (int32_t)(cgGoldLcg(s) % (uint32_t)(2 * r)) - r;
    return v >= 0 ? v + 1 : v;
}

/* Distinct dyadic scales, unique maximum, non-monotonic in g. */
static float cgGoldGroupScale(size_t g, size_t numGroups) {
    return (float)(5 + 3 * ((g * 7) % numGroups)) / 64.0f;
}

typedef struct cgGoldOut {
    int32_t y[CG_GOLD_MAX_OUT];
    float scale;
    uint8_t bits;
    int32_t bias[7]; /* codes */
    float biasScale;
} cgGoldOut_t;

/* xBits: the input's declared width (8; 17 takes the legacy float path). */
static void cgGoldRun(const cgGoldCase_t *c, uint8_t xBits, roundingMode_t rm, rngStream_t *rs,
                      cgGoldOut_t *out) {
    uint32_t s = 4049u + (uint32_t)(c->groupSize * 16 + c->padding);
    size_t wElems = c->outC * CG_GOLD_CIN * 5;
    size_t numGroups = wElems / c->groupSize;
    TEST_ASSERT_EQUAL_size_t(wElems, numGroups * c->groupSize);
    int32_t x[CG_GOLD_B * CG_GOLD_CIN * CG_GOLD_L], w[7 * CG_GOLD_CIN * 5];
    for (size_t i = 0; i < CG_GOLD_B * CG_GOLD_CIN * CG_GOLD_L; i++) {
        x[i] = cgGoldCode(&s, 100);
    }
    for (size_t i = 0; i < wElems; i++) {
        w[i] = cgGoldCode(&s, 100);
    }
    for (size_t i = 0; i < c->outC; i++) {
        out->bias[i] = cgGoldCode(&s, 200);
    }
    odtScale_t scales[10];
    for (size_t g = 0; g < numGroups; g++) {
        scales[g] = odtTestScale(cgGoldGroupScale(g, numGroups));
    }
    symQConfig_t groups = {.scales = scales,
                           .numGroups = numGroups,
                           .groupSize = c->groupSize,
                           .qBits = 8,
                           .roundingMode = rm};
    symGroupedWeight_t groupedWeight = symGroupedWeightSym(&groups);

    size_t xDims[] = {CG_GOLD_B, CG_GOLD_CIN, CG_GOLD_L}, wDims[] = {c->outC, CG_GOLD_CIN, 5};
    size_t yDims[] = {CG_GOLD_B, c->outC, c->outLen}, bDims[] = {c->outC};
    size_t order[] = {0, 1, 2};
    shape_t xS, wS, yS, bS;
    setShape(&xS, xDims, 3, order);
    setShape(&wS, wDims, 3, order);
    setShape(&yS, yDims, 3, order);
    setShape(&bS, bDims, 1, order);
    symInt32QConfig_t xQC, wQC, yQC, bQC;
    initSymInt32QConfigWithQMaxBits(rm, &xQC, xBits);
    initSymInt32QConfigWithQMaxBits(rm, &wQC, 8); /* grouped scratch: mode = the op's */
    initSymInt32QConfigWithQMaxBits(rm, &yQC, UNSTAMPED_WIDTH);
    initSymInt32QConfigWithQMaxBits(rm, &bQC, 12);
    xQC.scale = odtTestScale(0.25f);
    bQC.scale = odtScaleFromFloat(0.3f, ODT_SCALE_NEAREST); /* NEAREST15(0.3), canonical */
    quantization_t xQ, wQ, yQ, bQ;
    initSymInt32Quantization(&xQC, &xQ);
    initSymInt32Quantization(&wQC, &wQ);
    initSymInt32Quantization(&yQC, &yQ);
    initSymInt32Quantization(&bQC, &bQ);
    tensor_t xT, wT, yT, bT;
    setTensorValues(&xT, (uint8_t *)x, &xS, &xQ, NULL);
    setTensorValues(&wT, (uint8_t *)w, &wS, &wQ, NULL);
    setTensorValues(&yT, (uint8_t *)out->y, &yS, &yQ, NULL);
    setTensorValues(&bT, (uint8_t *)out->bias, &bS, &bQ, NULL);
    kernel_t kernel;
    if (c->padType == SAME) {
        initKernel(&kernel, 5, SAME, 2, 2);
    } else {
        initKernelExplicit(&kernel, 5, c->padding, 2, 2);
    }
    conv1dKernelSymInt32Grouped(&xT, &wT, &bT, &kernel, 1, &yT, &groupedWeight, rs);
    out->scale = odtScaleToFloat(yQC.scale);
    out->bits = yQC.qMaxBits;
    out->biasScale = odtScaleToFloat(bQC.scale);
}

static float cgFoldScale(int32_t accM, int32_t accE) {
    return odtScaleToFloat((odtScale_t){.m = (uint16_t)accM, .e = (int16_t)accE});
}

/* SR against the HALF_AWAY gold on a fixture whose combines are all exact
 * (k/64 group scales, generator-asserted): every code equals the gold except
 * through the bias seed, which SR may lift by 1 -- the same lift for every
 * (b, outPos) of one output channel. */
static void cgAssertSrIsHaUpToSeedLift(const int32_t *ha, const int32_t *sr, size_t batch,
                                       size_t outC, size_t outLen) {
    for (size_t oc = 0; oc < outC; oc++) {
        const int32_t lift = sr[oc * outLen] - ha[oc * outLen];
        TEST_ASSERT_TRUE(lift >= -1 && lift <= 1);
        for (size_t b = 0; b < batch; b++) {
            for (size_t p = 0; p < outLen; p++) {
                size_t i = (b * outC + oc) * outLen + p;
                TEST_ASSERT_EQUAL_INT32(ha[i] + lift, sr[i]);
            }
        }
    }
}

static const int32_t *const kCgFoldHa[CG_GOLD_NUM_CASES] = {
    kCgFoldHa0, kCgFoldHa1, kCgFoldHa2, kCgFoldHa3, kCgFoldHa4,  kCgFoldHa5,
    kCgFoldHa6, kCgFoldHa7, kCgFoldHa8, kCgFoldHa9, kCgFoldHa10, kCgFoldHa11};
static const int32_t *const kCgFoldSr[CG_GOLD_NUM_CASES] = {
    kCgFoldSr0, kCgFoldSr1, kCgFoldSr2, kCgFoldSr3, kCgFoldSr4,  kCgFoldSr5,
    kCgFoldSr6, kCgFoldSr7, kCgFoldSr8, kCgFoldSr9, kCgFoldSr10, kCgFoldSr11};

/* int8 input: the fold path (F = 8). The gold is the Python-int fold reference
 * (generate_expected_conv1d_grouped_fold.py), which reproduces this fixture's
 * LCG operands; scale A = {inScale.m, inScale.e + E + 15 - F}, qMaxBits the
 * fold site's width with the bias seeds. */
void testConv1dGroupedGoldenHalfAway(void) {
    for (size_t ci = 0; ci < CG_GOLD_NUM_CASES; ci++) {
        const cgGoldCase_t *c = &kCgGoldCases[ci];
        cgGoldOut_t out;
        cgGoldRun(c, 8, HALF_AWAY, NULL, &out);
        GOLD_I32(kCgFoldHa[ci], out.y, CG_GOLD_B * c->outC * c->outLen);
        GOLD_F32_SCALAR(cgFoldScale(kCgFoldAccM[ci], kCgFoldAccE[ci]), out.scale);
        GOLD_U32_SCALAR((uint32_t)kCgFoldBits[ci], out.bits);
        if (c->padding == 9) {
            /* output 0 has no valid tap: it is the bias seed alone */
            for (size_t b = 0; b < CG_GOLD_B; b++) {
                for (size_t oc = 0; oc < c->outC; oc++) {
                    int32_t seed = (int32_t)roundf((float)out.bias[oc] * out.biasScale / out.scale);
                    TEST_ASSERT_EQUAL_INT32(seed, out.y[(b * c->outC + oc) * c->outLen]);
                }
            }
        }
    }
}

/* SR on the fold path, recorded from the kernel (kCgFoldSr*, the Python
 * reference has no C RNG); checked here against the HALF_AWAY gold up to the
 * bias seed's lift. The stream state pins one draw per combine and per bias
 * seed. SR rounding of inexact combines: testConv1dGroupedFoldSrIsUnbiased. */
void testConv1dGroupedGoldenSr(void) {
    for (size_t ci = 0; ci < CG_GOLD_NUM_CASES; ci++) {
        const cgGoldCase_t *c = &kCgGoldCases[ci];
        rngStream_t rs = rngOpen(rngKeyFromSeed(0xC1D00000u + ci));
        uint32_t before = rs.state;
        cgGoldOut_t out;
        cgGoldRun(c, 8, SR_HALF_AWAY, &rs, &out);
        TEST_ASSERT_NOT_EQUAL_UINT32(before, rs.state);
        cgAssertSrIsHaUpToSeedLift(kCgFoldHa[ci], out.y, CG_GOLD_B, c->outC, c->outLen);
        GOLD_I32(kCgFoldSr[ci], out.y, CG_GOLD_B * c->outC * c->outLen);
        GOLD_U32_SCALAR(kCgFoldSrState[ci], rs.state);
        GOLD_F32_SCALAR(cgFoldScale(kCgFoldAccM[ci], kCgFoldAccE[ci]), out.scale);
        GOLD_U32_SCALAR((uint32_t)kCgFoldBits[ci], out.bits);
    }
}

/* Conv groups 2 (the cases above all run conv groups 1): batch 2, Cin 4,
 * conv groups 2 (inChPerGroup 2), outC 4 (outChPerGroup 2), L 8, K 5, stride
 * 2, dilation 2, EXPLICIT 3, so outLen 3 and left padding enters every row
 * mid-row. A weight row (one oc) is 10 codes; group sizes 4 and 8 straddle
 * both the ic rows (5 codes) and the oc rows (legacy cursor loop, fold
 * segment walk); 10 and 20 are channel-aligned (one group per row / per two
 * rows: the cursor-free loops). */
#define CG2_GOLD_NUM_CASES 2
#define CG2_GOLD_B 2
#define CG2_GOLD_CIN 4
#define CG2_GOLD_OUTC 4
#define CG2_GOLD_L 8
#define CG2_GOLD_OUTLEN 3
#define CG2_GOLD_W (CG2_GOLD_OUTC * 2 * 5)
#define CG2_GOLD_Y (CG2_GOLD_B * CG2_GOLD_OUTC * CG2_GOLD_OUTLEN)

typedef struct cg2GoldOut {
    int32_t y[CG2_GOLD_Y];
    float scale;
    uint8_t bits;
} cg2GoldOut_t;

static void cg2GoldRun(size_t groupSize, uint8_t xBits, roundingMode_t rm, rngStream_t *rs,
                       cg2GoldOut_t *out) {
    uint32_t s = 7121u + (uint32_t)groupSize;
    size_t numGroups = CG2_GOLD_W / groupSize;
    int32_t x[CG2_GOLD_B * CG2_GOLD_CIN * CG2_GOLD_L], w[CG2_GOLD_W], bias[CG2_GOLD_OUTC];
    for (size_t i = 0; i < CG2_GOLD_B * CG2_GOLD_CIN * CG2_GOLD_L; i++) {
        x[i] = cgGoldCode(&s, 100);
    }
    for (size_t i = 0; i < CG2_GOLD_W; i++) {
        w[i] = cgGoldCode(&s, 100);
    }
    for (size_t i = 0; i < CG2_GOLD_OUTC; i++) {
        bias[i] = cgGoldCode(&s, 200);
    }
    odtScale_t scales[10];
    for (size_t g = 0; g < numGroups; g++) {
        scales[g] = odtTestScale(cgGoldGroupScale(g, numGroups));
    }
    symQConfig_t groups = {.scales = scales,
                           .numGroups = numGroups,
                           .groupSize = groupSize,
                           .qBits = 8,
                           .roundingMode = rm};
    symGroupedWeight_t groupedWeight = symGroupedWeightSym(&groups);

    size_t xDims[] = {CG2_GOLD_B, CG2_GOLD_CIN, CG2_GOLD_L}, wDims[] = {CG2_GOLD_OUTC, 2, 5};
    size_t yDims[] = {CG2_GOLD_B, CG2_GOLD_OUTC, CG2_GOLD_OUTLEN}, bDims[] = {CG2_GOLD_OUTC};
    size_t order[] = {0, 1, 2};
    shape_t xS, wS, yS, bS;
    setShape(&xS, xDims, 3, order);
    setShape(&wS, wDims, 3, order);
    setShape(&yS, yDims, 3, order);
    setShape(&bS, bDims, 1, order);
    symInt32QConfig_t xQC, wQC, yQC, bQC;
    initSymInt32QConfigWithQMaxBits(rm, &xQC, xBits);
    initSymInt32QConfigWithQMaxBits(rm, &wQC, 8);
    initSymInt32QConfigWithQMaxBits(rm, &yQC, UNSTAMPED_WIDTH);
    initSymInt32QConfigWithQMaxBits(rm, &bQC, 12);
    xQC.scale = odtTestScale(0.25f);
    bQC.scale = odtScaleFromFloat(0.3f, ODT_SCALE_NEAREST); /* NEAREST15(0.3), canonical */
    quantization_t xQ, wQ, yQ, bQ;
    initSymInt32Quantization(&xQC, &xQ);
    initSymInt32Quantization(&wQC, &wQ);
    initSymInt32Quantization(&yQC, &yQ);
    initSymInt32Quantization(&bQC, &bQ);
    tensor_t xT, wT, yT, bT;
    setTensorValues(&xT, (uint8_t *)x, &xS, &xQ, NULL);
    setTensorValues(&wT, (uint8_t *)w, &wS, &wQ, NULL);
    setTensorValues(&yT, (uint8_t *)out->y, &yS, &yQ, NULL);
    setTensorValues(&bT, (uint8_t *)bias, &bS, &bQ, NULL);
    kernel_t kernel;
    initKernelExplicit(&kernel, 5, 3, 2, 2);
    conv1dKernelSymInt32Grouped(&xT, &wT, &bT, &kernel, 2, &yT, &groupedWeight, rs);
    out->scale = odtScaleToFloat(yQC.scale);
    out->bits = yQC.qMaxBits;
}

typedef struct cg2FoldGold {
    size_t groupSize;
    const int32_t *ha, *sr;
    uint32_t srState;
    int32_t accM, accE, bits;
    uint32_t srSeed;
} cg2FoldGold_t;

static void cg2AssertFoldGold(const cg2FoldGold_t *g) {
    cg2GoldOut_t ha, sr;
    cg2GoldRun(g->groupSize, 8, HALF_AWAY, NULL, &ha);
    rngStream_t rs = rngOpen(rngKeyFromSeed(g->srSeed));
    uint32_t before = rs.state;
    cg2GoldRun(g->groupSize, 8, SR_HALF_AWAY, &rs, &sr);
    TEST_ASSERT_NOT_EQUAL_UINT32(before, rs.state);
    GOLD_I32(g->ha, ha.y, CG2_GOLD_Y);
    cgAssertSrIsHaUpToSeedLift(g->ha, sr.y, CG2_GOLD_B, CG2_GOLD_OUTC, CG2_GOLD_OUTLEN);
    GOLD_I32(g->sr, sr.y, CG2_GOLD_Y);
    GOLD_U32_SCALAR(g->srState, rs.state);
    GOLD_F32_SCALAR(cgFoldScale(g->accM, g->accE), ha.scale);
    GOLD_F32_SCALAR(cgFoldScale(g->accM, g->accE), sr.scale);
    GOLD_U32_SCALAR((uint32_t)g->bits, ha.bits);
    GOLD_U32_SCALAR((uint32_t)g->bits, sr.bits);
}

/* Segment walk (group sizes 4, 8), fold path: Python-int HALF_AWAY gold, SR
 * recorded from the kernel. */
void testConv1dGroupedGoldenConvGroups2(void) {
    const cg2FoldGold_t golds[CG2_GOLD_NUM_CASES] = {
        {4, kCg2FoldHa0, kCg2FoldSr0, kCg2FoldSrState[0], kCg2FoldAccM[0], kCg2FoldAccE[0],
         kCg2FoldBits[0], 0xC2D00000u},
        {8, kCg2FoldHa1, kCg2FoldSr1, kCg2FoldSrState[1], kCg2FoldAccM[1], kCg2FoldAccE[1],
         kCg2FoldBits[1], 0xC2D00001u},
    };
    for (size_t ci = 0; ci < CG2_GOLD_NUM_CASES; ci++) {
        cg2AssertFoldGold(&golds[ci]);
    }
}

/* Channel-aligned groups on the cg2 fixture: group size 10 is one group per
 * weight row (group == oc), 20 spans two rows (group == oc / 2). Pins the
 * group lookup of the cursor-free channel-aligned fold loop. */
void testConv1dGroupedGoldenConvGroups2ChannelAligned(void) {
    const cg2FoldGold_t golds[2] = {
        {10, kCg2AlignedFoldHa0, kCg2AlignedFoldSr0, kCg2AlignedFoldSrState[0],
         kCg2AlignedFoldAccM[0], kCg2AlignedFoldAccE[0], kCg2AlignedFoldBits[0], 0xC2A00000u},
        {20, kCg2AlignedFoldHa1, kCg2AlignedFoldSr1, kCg2AlignedFoldSrState[1],
         kCg2AlignedFoldAccM[1], kCg2AlignedFoldAccE[1], kCg2AlignedFoldBits[1], 0xC2A00001u},
    };
    for (size_t ci = 0; ci < 2; ci++) {
        cg2AssertFoldGold(&golds[ci]);
    }
}

/* Review Focus 3: a 17-bit input takes the legacy float path, bit-identical
 * to the pre-fold kernel -- kCgGold*, kCg2Gold* and kCg2AlignedGold* are the
 * goldens recorded before the fold existed (codes, scale and the SR stream).
 * Covers both legacy loops: the cursor loop (cgGold, cg2 4/8) and the
 * channel-aligned loop (cgGold group size 10, cg2 10/20). The stamped width
 * (#516) is the legacy rescaled site over the 17-bit x range with the bias
 * seeds at sAcc; cgLegacyStampCountsTheSeed pins the seed's part of it. */
static uint8_t cgLegacyBits(size_t nTerms, int32_t wMin, int32_t wMax, int32_t seedMin,
                            int32_t seedMax) {
    int32_t xMin, xMax;
    symAccumWidthRange(17, &xMin, &xMax);
    symAccumSite_t site = symAccumSiteInitRangesRescaled("t", nTerms, xMin, xMax, wMin, wMax, 0);
    symAccumSiteSetSeedRange(&site, seedMin, seedMax);
    return symAccumResultBits(&site);
}

/* 17-bit x times grouped int4 over 8 terms, scales 1 (sAcc = 1, so a bias b
 * seeds as b): the rescaled site's positive side is 8 * 2^16 * 8 plus slack
 * (4194313) -> 24 bits seedless; a bias of 2^22 lifts it past 2^23 -> 25. */
static void cgLegacyStampCountsTheSeed(void) {
    cgFill(8, 1, 1, false);
    int32_t bias = 1 << 22;
    TEST_ASSERT_EQUAL_INT32(8 + (1 << 22), cgRun(8, 1, 1, 17, 4, &bias, 2));
    TEST_ASSERT_EQUAL_UINT8(cgLegacyBits(8, -8, 7, bias, bias), gCgOutBits);
    TEST_ASSERT_EQUAL_UINT8(25, gCgOutBits);
    TEST_ASSERT_EQUAL_UINT8(24, cgLegacyBits(8, -8, 7, 0, 0));
}

void testConv1dGroupedFoldFallsBackToLegacyForWideInput(void) {
    for (size_t ci = 0; ci < CG_GOLD_NUM_CASES; ci++) {
        const cgGoldCase_t *c = &kCgGoldCases[ci];
        size_t n = CG_GOLD_B * c->outC * c->outLen;
        cgGoldOut_t out;
        cgGoldRun(c, 17, HALF_AWAY, NULL, &out);
        GOLD_I32(kCgGoldHa[ci], out.y, n);
        GOLD_F32_SCALAR(kCgGoldScale[ci], out.scale);
        int32_t seedMin = 0, seedMax = 0;
        for (size_t oc = 0; oc < c->outC; oc++) {
            /* rescaleIntoAccumulatorScale(HALF_AWAY): round half away of the float quotient */
            int32_t seed = (int32_t)roundf((float)out.bias[oc] * out.biasScale / out.scale);
            seedMin = (oc == 0 || seed < seedMin) ? seed : seedMin;
            seedMax = (oc == 0 || seed > seedMax) ? seed : seedMax;
        }
        GOLD_U32_SCALAR(cgLegacyBits(CG_GOLD_CIN * 5, -128, 127, seedMin, seedMax), out.bits);
        rngStream_t rs = rngOpen(rngKeyFromSeed(0xC1D00000u + ci));
        cgGoldRun(c, 17, SR_HALF_AWAY, &rs, &out);
        GOLD_I32(kCgGoldSr[ci], out.y, n);
        GOLD_U32_SCALAR(kCgGoldSrState[ci], rs.state);
        GOLD_F32_SCALAR(kCgGoldScale[ci], out.scale);
    }
    const size_t groupSizes[4] = {4, 8, 10, 20};
    const int32_t *ha[4] = {kCg2GoldHa0, kCg2GoldHa1, kCg2AlignedGoldHa0, kCg2AlignedGoldHa1};
    const int32_t *sr[4] = {kCg2GoldSr0, kCg2GoldSr1, kCg2AlignedGoldSr0, kCg2AlignedGoldSr1};
    const uint32_t state[4] = {kCg2GoldSrState[0], kCg2GoldSrState[1], kCg2AlignedGoldSrState[0],
                               kCg2AlignedGoldSrState[1]};
    const float scale[4] = {kCg2GoldScale[0], kCg2GoldScale[1], kCg2AlignedGoldScale[0],
                            kCg2AlignedGoldScale[1]};
    const uint32_t seeds[4] = {0xC2D00000u, 0xC2D00001u, 0xC2A00000u, 0xC2A00001u};
    for (size_t ci = 0; ci < 4; ci++) {
        cg2GoldOut_t out;
        cg2GoldRun(groupSizes[ci], 17, HALF_AWAY, NULL, &out);
        GOLD_I32(ha[ci], out.y, CG2_GOLD_Y);
        GOLD_F32_SCALAR(scale[ci], out.scale);
        rngStream_t rs = rngOpen(rngKeyFromSeed(seeds[ci]));
        cg2GoldRun(groupSizes[ci], 17, SR_HALF_AWAY, &rs, &out);
        GOLD_I32(sr[ci], out.y, CG2_GOLD_Y);
        GOLD_U32_SCALAR(state[ci], rs.state);
        GOLD_F32_SCALAR(scale[ci], out.scale);
    }
    cgLegacyStampCountsTheSeed();
}

/* ---- linq fold T4: the grouped gather's integer combine -------------------
 * Each group run's raw int32 partial is combined once into the accumulator
 * scale A = {inScale.m, inScale.e + E + 15 - F} (SymGroupFold.h):
 * acc += round(partial * m_g * 2^-(E - e_g + 15 - F)). Golds:
 * generate_expected_conv1d_grouped_fold.py (Python ints). Fixtures: batch 2,
 * K 5, L 8, stride 2, dilation 2; channel/group2: Cin 2, Cout 3, EXPLICIT 9
 * (Lout 9, output 0 has no valid tap); cg: Cin 4, Cout 4, conv groups 2,
 * EXPLICIT 3 (Lout 3). int8 operands, non-dyadic group scales. */

#define C1F_B 2
#define C1F_K 5
#define C1F_L 8
#define C1F_MAX_CIN 4
#define C1F_MAX_COUT 4
#define C1F_MAX_W 40
#define C1F_MAX_OUT (C1F_B * C1F_MAX_COUT * 9)
#define C1F_MAX_GROUPS 20

typedef struct c1FoldFixture {
    size_t cin, cout, convGroups, padding, outLen;
    const int32_t *x, *w, *bias;
} c1FoldFixture_t;

typedef struct c1FoldOut {
    int32_t y[C1F_MAX_OUT];
    odtScale_t scale;
    uint8_t bits;
} c1FoldOut_t;

static const c1FoldFixture_t kC1FoldFx = {
    2, 3, 1, 9, 9, kConv1dFoldX, kConv1dFoldW, kConv1dFoldBias};
static const c1FoldFixture_t kC1FoldCg = {
    4, 4, 2, 3, 3, kConv1dFoldCgX, kConv1dFoldCgW, kConv1dFoldCgBias};

static size_t c1FoldOutN(const c1FoldFixture_t *f) {
    return C1F_B * f->cout * f->outLen;
}

static void c1FoldScales(const float *values, size_t n, odtScale_t *scales) {
    for (size_t g = 0; g < n; g++) {
        scales[g] = odtTestScale(values[g]);
    }
}

/* x / w default to the fixture's operands when NULL. */
static void c1FoldRun(const c1FoldFixture_t *f, const int32_t *x, const int32_t *w,
                      const odtScale_t *scales, size_t numGroups, size_t groupSize, bool withBias,
                      roundingMode_t rm, rngStream_t *rs, c1FoldOut_t *out) {
    const size_t xN = C1F_B * f->cin * C1F_L;
    const size_t wN = f->cout * (f->cin / f->convGroups) * C1F_K;
    TEST_ASSERT_EQUAL_size_t(wN, numGroups * groupSize);
    int32_t xData[C1F_B * C1F_MAX_CIN * C1F_L], wData[C1F_MAX_W], biasData[C1F_MAX_COUT];
    memcpy(xData, x != NULL ? x : f->x, xN * sizeof(int32_t));
    memcpy(wData, w != NULL ? w : f->w, wN * sizeof(int32_t));
    memcpy(biasData, f->bias, f->cout * sizeof(int32_t));
    odtScale_t groupScales[C1F_MAX_GROUPS];
    memcpy(groupScales, scales, numGroups * sizeof(odtScale_t));
    symQConfig_t groups = {.scales = groupScales,
                           .numGroups = numGroups,
                           .groupSize = groupSize,
                           .qBits = 8,
                           .roundingMode = rm};
    symGroupedWeight_t groupedWeight = symGroupedWeightSym(&groups);
    size_t xDims[] = {C1F_B, f->cin, C1F_L}, wDims[] = {f->cout, f->cin / f->convGroups, C1F_K};
    size_t yDims[] = {C1F_B, f->cout, f->outLen}, bDims[] = {f->cout};
    size_t order[] = {0, 1, 2};
    shape_t xS, wS, yS, bS;
    setShape(&xS, xDims, 3, order);
    setShape(&wS, wDims, 3, order);
    setShape(&yS, yDims, 3, order);
    setShape(&bS, bDims, 1, order);
    symInt32QConfig_t xQC, wQC, yQC, bQC;
    initSymInt32QConfigWithQMaxBits(rm, &xQC, 8);
    initSymInt32QConfigWithQMaxBits(rm, &wQC, 8);
    initSymInt32QConfigWithQMaxBits(rm, &yQC, UNSTAMPED_WIDTH);
    initSymInt32QConfigWithQMaxBits(rm, &bQC, 12);
    xQC.scale = odtTestScale(kConv1dFoldXScale);
    bQC.scale = odtTestScale(kConv1dFoldBiasScale);
    quantization_t xQ, wQ, yQ, bQ;
    initSymInt32Quantization(&xQC, &xQ);
    initSymInt32Quantization(&wQC, &wQ);
    initSymInt32Quantization(&yQC, &yQ);
    initSymInt32Quantization(&bQC, &bQ);
    tensor_t xT, wT, yT, bT;
    setTensorValues(&xT, (uint8_t *)xData, &xS, &xQ, NULL);
    setTensorValues(&wT, (uint8_t *)wData, &wS, &wQ, NULL);
    setTensorValues(&yT, (uint8_t *)out->y, &yS, &yQ, NULL);
    setTensorValues(&bT, (uint8_t *)biasData, &bS, &bQ, NULL);
    kernel_t kernel;
    initKernelExplicit(&kernel, C1F_K, f->padding, 2, 2);
    conv1dKernelSymInt32Grouped(&xT, &wT, withBias ? &bT : NULL, &kernel, f->convGroups, &yT,
                                &groupedWeight, rs);
    out->scale = odtScaleLoad(&yQC.scale);
    out->bits = yQC.qMaxBits;
}

static void c1FoldAssertGold(const c1FoldFixture_t *f, const float *wScales, size_t numGroups,
                             size_t groupSize, const int32_t *expected, int32_t accM, int32_t accE,
                             int32_t bits) {
    odtScale_t scales[C1F_MAX_GROUPS];
    c1FoldScales(wScales, numGroups, scales);
    c1FoldOut_t out;
    c1FoldRun(f, NULL, NULL, scales, numGroups, groupSize, true, HALF_AWAY, NULL, &out);
    GOLD_I32(expected, out.y, c1FoldOutN(f));
    TEST_ASSERT_EQUAL_UINT16((uint16_t)accM, out.scale.m);
    TEST_ASSERT_EQUAL_INT16((int16_t)accE, out.scale.e);
    TEST_ASSERT_EQUAL_UINT8((uint8_t)bits, out.bits);
}

/* groupSize 10 = Cin * K: one group per output channel (cursor-free loop). */
void testConv1dGroupedFoldGoldenHalfAwayChannel(void) {
    TEST_ASSERT_EQUAL_size_t(c1FoldOutN(&kC1FoldFx), kConv1dFoldChannelOut_len);
    c1FoldAssertGold(&kC1FoldFx, kConv1dFoldChannelWScales, (size_t)kConv1dFoldChannelNumGroups,
                     (size_t)kConv1dFoldChannelGroupSize, kConv1dFoldChannelOut,
                     kConv1dFoldChannelAccM, kConv1dFoldChannelAccE, kConv1dFoldChannelBits);
}

/* groupSize 2 against K = 5: groups straddle the tap rows, several combines
 * per window (segment walk). */
void testConv1dGroupedFoldGoldenHalfAwayGroupSize2(void) {
    c1FoldAssertGold(&kC1FoldFx, kConv1dFoldGroup2WScales, (size_t)kConv1dFoldGroup2NumGroups,
                     (size_t)kConv1dFoldGroup2GroupSize, kConv1dFoldGroup2Out,
                     kConv1dFoldGroup2AccM, kConv1dFoldGroup2AccE, kConv1dFoldGroup2Bits);
}

/* conv groups 2 on both fold loops: aligned (groupSize 10, one group per row)
 * and segment walk (groupSize 4, groups straddle the ic and oc rows). */
void testConv1dGroupedFoldGoldenConvGroups2(void) {
    TEST_ASSERT_EQUAL_size_t(c1FoldOutN(&kC1FoldCg), kConv1dFoldCgAlignedOut_len);
    c1FoldAssertGold(&kC1FoldCg, kConv1dFoldCgAlignedWScales, (size_t)kConv1dFoldCgAlignedNumGroups,
                     (size_t)kConv1dFoldCgAlignedGroupSize, kConv1dFoldCgAlignedOut,
                     kConv1dFoldCgAlignedAccM, kConv1dFoldCgAlignedAccE, kConv1dFoldCgAlignedBits);
    c1FoldAssertGold(&kC1FoldCg, kConv1dFoldCgCursorWScales, (size_t)kConv1dFoldCgCursorNumGroups,
                     (size_t)kConv1dFoldCgCursorGroupSize, kConv1dFoldCgCursorOut,
                     kConv1dFoldCgCursorAccM, kConv1dFoldCgCursorAccE, kConv1dFoldCgCursorBits);
}

/* The wire describes the codes: scale {inScale.m, inScale.e + E + 15 - F},
 * F = 8 here (int8 x int8 over N = Cin * K = 10 terms), and qMaxBits the width
 * of the fold site -- x widened to [xMin * 2^F - 1, xMax * 2^F + 1], no
 * rescale slack -- with the bias seeds' range (none without bias). */
void testConv1dGroupedFoldStampsFoldScaleAndWidth(void) {
    odtScale_t scales[C1F_MAX_GROUPS];
    const size_t numGroups = (size_t)kConv1dFoldGroup2NumGroups;
    c1FoldScales(kConv1dFoldGroup2WScales, numGroups, scales);
    int32_t eMax = INT32_MIN;
    for (size_t g = 0; g < numGroups; g++) {
        eMax = scales[g].e > eMax ? scales[g].e : eMax;
    }
    const int32_t frac = 8;
    odtScale_t inScale = odtTestScale(kConv1dFoldXScale);
    symAccumSite_t site =
        symAccumSiteInitRanges("t", 2 * C1F_K, -128 * 256 - 1, 127 * 256 + 1, -128, 127, 0);
    TEST_ASSERT_TRUE(symAccumProvenSafe(&site));

    c1FoldOut_t out;
    c1FoldRun(&kC1FoldFx, NULL, NULL, scales, numGroups, (size_t)kConv1dFoldGroup2GroupSize, false,
              HALF_AWAY, NULL, &out);
    TEST_ASSERT_EQUAL_UINT16(inScale.m, out.scale.m);
    TEST_ASSERT_EQUAL_INT16((int16_t)(inScale.e + eMax + 15 - frac), out.scale.e);
    TEST_ASSERT_EQUAL_UINT8(symAccumResultBits(&site), out.bits);
    TEST_ASSERT_EQUAL_UINT8((uint8_t)kConv1dFoldChannelBitsNoBias, out.bits);

    c1FoldRun(&kC1FoldFx, NULL, NULL, scales, numGroups, (size_t)kConv1dFoldGroup2GroupSize, true,
              HALF_AWAY, NULL, &out);
    symAccumSiteSetSeeds(&site, kConv1dFoldGroup2Seeds, kConv1dFoldGroup2Seeds_len);
    TEST_ASSERT_EQUAL_UINT8(symAccumResultBits(&site), out.bits);
    TEST_ASSERT_EQUAL_UINT8((uint8_t)kConv1dFoldGroup2Bits, out.bits);
}

/* A window with no valid tap (output 0 under EXPLICIT 9) combines nothing:
 * its output is the bias seed alone, and under SR the stream advances by
 * exactly one draw per combine the reference counts -- none for the empty
 * windows -- on the aligned loop (channel) and the segment walk (group2). */
void testConv1dGroupedFoldEmptyWindowCombinesNothing(void) {
    struct {
        const float *wScales;
        size_t numGroups, groupSize;
        const int32_t *seeds;
        int32_t combines;
    } shapes[2] = {
        {kConv1dFoldChannelWScales, (size_t)kConv1dFoldChannelNumGroups,
         (size_t)kConv1dFoldChannelGroupSize, kConv1dFoldChannelSeeds, kConv1dFoldChannelCombines},
        {kConv1dFoldGroup2WScales, (size_t)kConv1dFoldGroup2NumGroups,
         (size_t)kConv1dFoldGroup2GroupSize, kConv1dFoldGroup2Seeds, kConv1dFoldGroup2Combines},
    };
    for (size_t s = 0; s < 2; s++) {
        odtScale_t scales[C1F_MAX_GROUPS];
        c1FoldScales(shapes[s].wScales, shapes[s].numGroups, scales);
        c1FoldOut_t out;
        c1FoldRun(&kC1FoldFx, NULL, NULL, scales, shapes[s].numGroups, shapes[s].groupSize, true,
                  HALF_AWAY, NULL, &out);
        for (size_t b = 0; b < C1F_B; b++) {
            for (size_t oc = 0; oc < kC1FoldFx.cout; oc++) {
                TEST_ASSERT_EQUAL_INT32(shapes[s].seeds[oc],
                                        out.y[(b * kC1FoldFx.cout + oc) * kC1FoldFx.outLen]);
            }
        }

        rngStream_t rs = rngOpen(rngKeyFromSeed(0xE3D7u));
        rngStream_t expected = rngOpen(rngKeyFromSeed(0xE3D7u));
        for (int32_t d = 0; d < shapes[s].combines; d++) {
            (void)rngNextU32(&expected);
        }
        c1FoldRun(&kC1FoldFx, NULL, NULL, scales, shapes[s].numGroups, shapes[s].groupSize, false,
                  SR_HALF_AWAY, &rs, &out);
        TEST_ASSERT_EQUAL_UINT32(expected.state, rs.state);
    }
}

/* SR combines are unbiased: over 2048 streams the mean output sits within 3
 * sigma (empirical, of that mean) of the exact conv for the sum over all
 * outputs, and within 5 sigma per element; an element with no spread must be
 * exact. No bias (its seed is the float primitive's). Both fold loops. */
#define C1F_SR_RUNS 2048
static void c1FoldAssertSrUnbiased(const float *wScales, size_t numGroups, size_t groupSize) {
    const c1FoldFixture_t *f = &kC1FoldFx;
    odtScale_t scales[C1F_MAX_GROUPS];
    c1FoldScales(wScales, numGroups, scales);
    const double inScale = (double)kConv1dFoldXScale;
    const size_t n = c1FoldOutN(f);
    double exact[C1F_MAX_OUT] = {0};
    for (size_t b = 0; b < C1F_B; b++) {
        for (size_t oc = 0; oc < f->cout; oc++) {
            for (size_t p = 0; p < f->outLen; p++) {
                for (size_t ic = 0; ic < f->cin; ic++) {
                    for (size_t k = 0; k < C1F_K; k++) {
                        long pos = (long)(p * 2 + k * 2) - (long)f->padding;
                        if (pos < 0 || pos >= C1F_L) {
                            continue;
                        }
                        size_t wIdx = (oc * f->cin + ic) * C1F_K + k;
                        exact[(b * f->cout + oc) * f->outLen + p] +=
                            (double)f->x[(b * f->cin + ic) * C1F_L + (size_t)pos] * f->w[wIdx] *
                            inScale * (double)odtScaleToFloat(scales[wIdx / groupSize]);
                    }
                }
            }
        }
    }
    /* statistics on the deviation from the exact value in codes of A: the
     * values themselves (~1e3) would swamp the per-code variance in double */
    c1FoldOut_t out;
    c1FoldRun(f, NULL, NULL, scales, numGroups, groupSize, false, HALF_AWAY, NULL, &out);
    const double acc = (double)odtScaleToFloat(out.scale);
    static double sum[C1F_MAX_OUT], sumSq[C1F_MAX_OUT];
    memset(sum, 0, sizeof(sum));
    memset(sumSq, 0, sizeof(sumSq));
    double totSum = 0.0, totSumSq = 0.0;
    for (uint32_t r = 0; r < C1F_SR_RUNS; r++) {
        rngStream_t rs = rngOpen(rngKeyFromSeed(0x5EED1000u + r));
        c1FoldRun(f, NULL, NULL, scales, numGroups, groupSize, false, SR_HALF_AWAY, &rs, &out);
        TEST_ASSERT_TRUE(acc == (double)odtScaleToFloat(out.scale));
        double tot = 0.0;
        for (size_t i = 0; i < n; i++) {
            double d = (double)out.y[i] - exact[i] / acc;
            sum[i] += d;
            sumSq[i] += d * d;
            tot += d;
        }
        totSum += tot;
        totSumSq += tot * tot;
    }
    const double runs = (double)C1F_SR_RUNS;
    for (size_t i = 0; i < n; i++) {
        double mean = sum[i] / runs;
        double var = sumSq[i] / runs - mean * mean;
        double sigmaMean = sqrt(var > 0.0 ? var : 0.0) / sqrt(runs);
        if (sigmaMean < 1e-3) {
            TEST_ASSERT_TRUE(fabs(mean) <= 1e-6);
        } else {
            TEST_ASSERT_TRUE(fabs(mean) <= 5.0 * sigmaMean);
        }
    }
    double totMean = totSum / runs;
    double totVar = totSumSq / runs - totMean * totMean;
    TEST_ASSERT_TRUE(totVar > 0.0);
    TEST_ASSERT_TRUE(fabs(totMean) <= 3.0 * sqrt(totVar) / sqrt(runs));
}

void testConv1dGroupedFoldSrIsUnbiased(void) {
    c1FoldAssertSrUnbiased(kConv1dFoldChannelWScales, (size_t)kConv1dFoldChannelNumGroups,
                           (size_t)kConv1dFoldChannelGroupSize);
    c1FoldAssertSrUnbiased(kConv1dFoldGroup2WScales, (size_t)kConv1dFoldGroup2NumGroups,
                           (size_t)kConv1dFoldGroup2GroupSize);
}

/* The bias seeds the accumulator at A, 2^F times finer than the legacy sAcc,
 * so it takes part in the F proof (the convT twin is
 * testConvTGroupedFoldLargeBiasLowersFInsteadOfAborting). Shape: int12 input
 * with absmax 0.05, per-channel int8 weights with absmax 0.1 / 0.07, Cin 3,
 * K 5, and a bias of about +-1.0 -- seeded at A_{F=8} it is ~1e10, far past the
 * seed-rescale guard. The kernel must pick a lower F and not abort, and its
 * output must sit within 0.5 * A per combine plus 0.5 * A for the seed of the
 * exact value. */
#define C1F_BIAS_CIN 3
#define C1F_BIAS_COUT 2
#define C1F_BIAS_K 5
#define C1F_BIAS_L 9
#define C1F_BIAS_LOUT (C1F_BIAS_L - C1F_BIAS_K + 1)
/* the seed is ~1.07e10 at F = 8 and ~1.34e9 at F = 5, past the seed guard
 * (~1.07e9); F = 4 (~6.7e8) is the largest that admits it */
#define C1F_BIAS_EXPECTED_F 4
void testConv1dGroupedFoldLargeBiasLowersFInsteadOfAborting(void) {
    int32_t x[C1F_BIAS_CIN * C1F_BIAS_L], w[C1F_BIAS_COUT * C1F_BIAS_CIN * C1F_BIAS_K];
    int32_t bias[C1F_BIAS_COUT] = {2000, -1700};
    int32_t y[C1F_BIAS_COUT * C1F_BIAS_LOUT];
    uint32_t s = 4243u;
    for (size_t i = 0; i < sizeof(x) / sizeof(x[0]); i++) {
        x[i] = cgGoldCode(&s, 2047);
    }
    for (size_t i = 0; i < sizeof(w) / sizeof(w[0]); i++) {
        w[i] = cgGoldCode(&s, 127);
    }
    const float wScaleValues[C1F_BIAS_COUT] = {0.1f / 127.0f, 0.07f / 127.0f};
    odtScale_t scales[C1F_BIAS_COUT];
    for (size_t g = 0; g < C1F_BIAS_COUT; g++) {
        scales[g] = odtScaleFromFloat(wScaleValues[g], ODT_SCALE_NEAREST);
    }
    symQConfig_t groups = {.scales = scales,
                           .numGroups = C1F_BIAS_COUT,
                           .groupSize = C1F_BIAS_CIN * C1F_BIAS_K,
                           .qBits = 8,
                           .roundingMode = HALF_AWAY};
    symGroupedWeight_t groupedWeight = symGroupedWeightSym(&groups);
    size_t xDims[] = {1, C1F_BIAS_CIN, C1F_BIAS_L},
           wDims[] = {C1F_BIAS_COUT, C1F_BIAS_CIN, C1F_BIAS_K};
    size_t yDims[] = {1, C1F_BIAS_COUT, C1F_BIAS_LOUT}, bDims[] = {C1F_BIAS_COUT};
    size_t order[] = {0, 1, 2};
    shape_t xS, wS, yS, bS;
    setShape(&xS, xDims, 3, order);
    setShape(&wS, wDims, 3, order);
    setShape(&yS, yDims, 3, order);
    setShape(&bS, bDims, 1, order);
    symInt32QConfig_t xQC, wQC, yQC, bQC;
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &xQC, 12);
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &wQC, 8);
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &yQC, UNSTAMPED_WIDTH);
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &bQC, 12);
    xQC.scale = odtScaleFromFloat(0.05f / 2047.0f, ODT_SCALE_NEAREST);
    bQC.scale = odtScaleFromFloat(1.0f / 2000.0f, ODT_SCALE_NEAREST);
    quantization_t xQ, wQ, yQ, bQ;
    initSymInt32Quantization(&xQC, &xQ);
    initSymInt32Quantization(&wQC, &wQ);
    initSymInt32Quantization(&yQC, &yQ);
    initSymInt32Quantization(&bQC, &bQ);
    tensor_t xT, wT, yT, bT;
    setTensorValues(&xT, (uint8_t *)x, &xS, &xQ, NULL);
    setTensorValues(&wT, (uint8_t *)w, &wS, &wQ, NULL);
    setTensorValues(&yT, (uint8_t *)y, &yS, &yQ, NULL);
    setTensorValues(&bT, (uint8_t *)bias, &bS, &bQ, NULL);
    kernel_t kernel;
    initKernel(&kernel, C1F_BIAS_K, VALID, 1, 1);
    conv1dKernelSymInt32Grouped(&xT, &wT, &bT, &kernel, 1, &yT, &groupedWeight, NULL);

    /* it stays on the fold path (A carries inScale's mantissa) at an F below 8 */
    odtScale_t out = odtScaleLoad(&yQC.scale), in = odtScaleLoad(&xQC.scale);
    int32_t eMax = scales[0].e > scales[1].e ? scales[0].e : scales[1].e;
    TEST_ASSERT_EQUAL_UINT16(in.m, out.m);
    TEST_ASSERT_EQUAL_INT32(C1F_BIAS_EXPECTED_F, in.e + eMax + 15 - out.e);

    const double outScale = (double)odtScaleToFloat(yQC.scale);
    const double sx = (double)odtScaleToFloat(xQC.scale);
    const double sb = (double)odtScaleToFloat(bQC.scale);
    for (size_t oc = 0; oc < C1F_BIAS_COUT; oc++) {
        for (size_t p = 0; p < C1F_BIAS_LOUT; p++) {
            double exact = (double)bias[oc] * sb;
            for (size_t ic = 0; ic < C1F_BIAS_CIN; ic++) {
                for (size_t k = 0; k < C1F_BIAS_K; k++) {
                    exact += (double)x[ic * C1F_BIAS_L + p + k] *
                             w[(oc * C1F_BIAS_CIN + ic) * C1F_BIAS_K + k] * sx *
                             (double)odtScaleToFloat(scales[oc]);
                }
            }
            double got = (double)y[oc * C1F_BIAS_LOUT + p] * outScale;
            /* one combine (per-channel group) + the seed, plus float headroom */
            double bound = 0.5 * outScale * 2.0 + 1e-6 * fabs(exact);
            TEST_ASSERT_TRUE(fabs(got - exact) <= bound);
        }
    }
}

/* ---- linq fold T6: groups that do not line up with a weight row ----------
 * kCuFold* (golden_conv1d_grouped.h) were recorded from the cursor loop
 * before the segment walk replaced it; the walk must reproduce them, the SR
 * stream included. Non-dyadic group scales make the combines inexact, so a
 * misplaced or extra SR draw moves output codes, not just the stream. */
#define CU_K 5
#define CU_B 2
#define CU_MAX_X (CU_B * 6 * 10)
#define CU_MAX_W 60
#define CU_MAX_OUTC 4
#define CU_MAX_Y (CU_B * CU_MAX_OUTC * 10)
#define CU_MAX_GROUPS 20

typedef struct cuCase {
    size_t cin, convGroups, cout, len, stride, dilation;
    paddingType_t padType;
    size_t padding, groupSize, outLen;
} cuCase_t;

typedef struct cuOut {
    int32_t y[CU_MAX_Y];
    odtScale_t scale;
    uint8_t bits;
    int32_t eMax; /* the largest group scale exponent, for the fold stamp */
} cuOut_t;

static size_t cuWeights(const cuCase_t *c) {
    return c->cout * (c->cin / c->convGroups) * CU_K;
}

static size_t cuOutN(const cuCase_t *c) {
    return CU_B * c->cout * c->outLen;
}

static odtScale_t cuInScale(void) {
    return odtScaleFromFloat(0.037f, ODT_SCALE_NEAREST);
}

static void cuRun(const cuCase_t *c, uint32_t seed, roundingMode_t rm, rngStream_t *rs,
                  cuOut_t *out) {
    uint32_t s = seed;
    const size_t xN = CU_B * c->cin * c->len, wN = cuWeights(c);
    const size_t numGroups = wN / c->groupSize;
    TEST_ASSERT_EQUAL_size_t(wN, numGroups * c->groupSize);
    TEST_ASSERT_TRUE(xN <= CU_MAX_X && wN <= CU_MAX_W && c->cout <= CU_MAX_OUTC);
    TEST_ASSERT_TRUE(numGroups <= CU_MAX_GROUPS && cuOutN(c) <= CU_MAX_Y);
    int32_t x[CU_MAX_X], w[CU_MAX_W], bias[CU_MAX_OUTC];
    for (size_t i = 0; i < xN; i++) {
        x[i] = cgGoldCode(&s, 127);
    }
    for (size_t i = 0; i < wN; i++) {
        w[i] = cgGoldCode(&s, 127);
    }
    for (size_t i = 0; i < c->cout; i++) {
        bias[i] = cgGoldCode(&s, 900);
    }
    odtScale_t scales[CU_MAX_GROUPS];
    out->eMax = INT32_MIN;
    for (size_t g = 0; g < numGroups; g++) {
        float v = 0.0113f + 0.0037f * (float)((g * 7) % numGroups);
        scales[g] = odtScaleFromFloat(v, ODT_SCALE_NEAREST);
        out->eMax = scales[g].e > out->eMax ? scales[g].e : out->eMax;
    }
    symQConfig_t groups = {.scales = scales,
                           .numGroups = numGroups,
                           .groupSize = c->groupSize,
                           .qBits = 8,
                           .roundingMode = rm};
    symGroupedWeight_t groupedWeight = symGroupedWeightSym(&groups);
    size_t xDims[] = {CU_B, c->cin, c->len}, wDims[] = {c->cout, c->cin / c->convGroups, CU_K};
    size_t yDims[] = {CU_B, c->cout, c->outLen}, bDims[] = {c->cout};
    size_t order[] = {0, 1, 2};
    shape_t xS, wS, yS, bS;
    setShape(&xS, xDims, 3, order);
    setShape(&wS, wDims, 3, order);
    setShape(&yS, yDims, 3, order);
    setShape(&bS, bDims, 1, order);
    symInt32QConfig_t xQC, wQC, yQC, bQC;
    initSymInt32QConfigWithQMaxBits(rm, &xQC, 8);
    initSymInt32QConfigWithQMaxBits(rm, &wQC, 8);
    initSymInt32QConfigWithQMaxBits(rm, &yQC, UNSTAMPED_WIDTH);
    initSymInt32QConfigWithQMaxBits(rm, &bQC, 12);
    xQC.scale = cuInScale();
    bQC.scale = odtScaleFromFloat(0.3f, ODT_SCALE_NEAREST);
    quantization_t xQ, wQ, yQ, bQ;
    initSymInt32Quantization(&xQC, &xQ);
    initSymInt32Quantization(&wQC, &wQ);
    initSymInt32Quantization(&yQC, &yQ);
    initSymInt32Quantization(&bQC, &bQ);
    tensor_t xT, wT, yT, bT;
    setTensorValues(&xT, (uint8_t *)x, &xS, &xQ, NULL);
    setTensorValues(&wT, (uint8_t *)w, &wS, &wQ, NULL);
    setTensorValues(&yT, (uint8_t *)out->y, &yS, &yQ, NULL);
    setTensorValues(&bT, (uint8_t *)bias, &bS, &bQ, NULL);
    kernel_t kernel;
    if (c->padType == SAME) {
        initKernel(&kernel, CU_K, SAME, c->dilation, c->stride);
    } else {
        initKernelExplicit(&kernel, CU_K, c->padding, c->dilation, c->stride);
    }
    conv1dKernelSymInt32Grouped(&xT, &wT, &bT, &kernel, c->convGroups, &yT, &groupedWeight, rs);
    out->scale = odtScaleLoad(&yQC.scale);
    out->bits = yQC.qMaxBits;
}

/* The fold path ran, not the legacy float fallback: the stamp is
 * {inScale.m, inScale.e + eMax + 15 - F} with F >= 1. */
static void cuAssertFoldPath(const cuOut_t *out) {
    const odtScale_t in = cuInScale();
    const int32_t frac = in.e + out->eMax + 15 - out->scale.e;
    TEST_ASSERT_EQUAL_UINT16(in.m, out->scale.m);
    TEST_ASSERT_TRUE(frac >= 1 && frac <= 15);
}

static const cuCase_t kCuCases[4] = {
    /* straddle, SAME: row 10, G 4 crosses the ic and oc rows; stride 2 */
    {2, 1, 4, 9, 2, 1, SAME, 0, 4, 5},
    /* short-segment edge: row 10, G 3, EXPLICIT 3 */
    {2, 1, 3, 8, 1, 1, EXPLICIT, 3, 3, 10},
    /* group size 1: one combine per valid tap */
    {2, 1, 2, 10, 2, 2, EXPLICIT, 4, 1, 5},
    /* conv groups 2: row 15, G 12 (G / K 2, G % K 2); dilation 2 */
    {6, 2, 4, 8, 1, 2, EXPLICIT, 2, 12, 4},
};

static void cuAssertGold(size_t ci) {
    const cuCase_t *c = &kCuCases[ci];
    static const int32_t *const ha[4] = {kCuFoldHa0, kCuFoldHa1, kCuFoldHa2, kCuFoldHa3};
    static const int32_t *const sr[4] = {kCuFoldSr0, kCuFoldSr1, kCuFoldSr2, kCuFoldSr3};
    const uint32_t seed = 0xC6D00000u + (uint32_t)ci;
    const size_t n = cuOutN(c);
    cuOut_t out;
    cuRun(c, seed, HALF_AWAY, NULL, &out);
    cuAssertFoldPath(&out);
    GOLD_I32(ha[ci], out.y, n);
    TEST_ASSERT_EQUAL_UINT16((uint16_t)kCuFoldAccM[ci], out.scale.m);
    TEST_ASSERT_EQUAL_INT16((int16_t)kCuFoldAccE[ci], out.scale.e);
    TEST_ASSERT_EQUAL_UINT8((uint8_t)kCuFoldBits[ci], out.bits);

    rngStream_t rs = rngOpen(rngKeyFromSeed(seed));
    cuRun(c, seed, SR_HALF_AWAY, &rs, &out);
    cuAssertFoldPath(&out);
    GOLD_I32(sr[ci], out.y, n);
    GOLD_U32_SCALAR(kCuFoldSrState[ci], rs.state);
    TEST_ASSERT_EQUAL_UINT16((uint16_t)kCuFoldAccM[ci], out.scale.m);
    TEST_ASSERT_EQUAL_INT16((int16_t)kCuFoldAccE[ci], out.scale.e);
    TEST_ASSERT_EQUAL_UINT8((uint8_t)kCuFoldBits[ci], out.bits);
}

/* Brute force over (row, group): window outPos has a valid tap, yet some
 * group piece of a row, shorter than K, has none -- a piece the walk must
 * skip without a combine (and without an SR draw). EXPLICIT padding only. */
static bool cuWindowSkipsAShortPiece(const cuCase_t *c, size_t outPos) {
    const size_t rowLen = (c->cin / c->convGroups) * CU_K, g = c->groupSize;
    bool anyValid = false, skips = false;
    for (size_t oc = 0; oc < c->cout; oc++) {
        const size_t rowStart = oc * rowLen, rowEnd = rowStart + rowLen;
        for (size_t grp = rowStart / g; grp * g < rowEnd; grp++) {
            const size_t lo = grp * g > rowStart ? grp * g : rowStart;
            const size_t hi = (grp + 1) * g < rowEnd ? (grp + 1) * g : rowEnd;
            size_t valid = 0;
            for (size_t i = lo; i < hi; i++) {
                const long pos =
                    (long)(outPos * c->stride + ((i - rowStart) % CU_K) * c->dilation) -
                    (long)c->padding;
                valid += pos >= 0 && pos < (long)c->len;
            }
            anyValid = anyValid || valid > 0;
            skips = skips || (valid == 0 && hi - lo < CU_K);
        }
    }
    return anyValid && skips;
}

void testConv1dGroupedFoldUnalignedStraddleSame(void) {
    cuAssertGold(0);
}

/* Review Focus 1: in the first window only taps 3..4 are valid, so row 0's
 * 3-tap piece (ch 0, taps 0..2) gets none; in the last window only taps 0..1
 * are, so row 0's 1-tap piece (ch 1, tap 4) gets none. */
void testConv1dGroupedFoldUnalignedShortSegmentEdge(void) {
    const cuCase_t *c = &kCuCases[1];
    TEST_ASSERT_EQUAL(EXPLICIT, c->padType);
    TEST_ASSERT_TRUE(cuWindowSkipsAShortPiece(c, 0));
    TEST_ASSERT_TRUE(cuWindowSkipsAShortPiece(c, c->outLen - 1));
    cuAssertGold(1);
}

/* Review Focus 3: G 1 < K, so G / K == 0 and every step is a carry-only one. */
void testConv1dGroupedFoldUnalignedGroupSize1(void) {
    cuAssertGold(2);
}

/* Review Focus 4: the row base is per conv group. */
void testConv1dGroupedFoldUnalignedConvGroups2(void) {
    cuAssertGold(3);
}

/* ---- SymGroupCursor.h row walk: the group pieces of one weight row ------
 * Each expected row is {group, chLo, kLo, chHi, kHi}; the row end is
 * (rows, 0). */
static void sgAssertRowPieces(size_t groupSize, size_t kernelSize, size_t rows, size_t rowStart,
                              const size_t (*expected)[5], size_t n) {
    const symGroupRowWalk_t w = symGroupRowWalkInit(groupSize, kernelSize, rows);
    symGroupRowPiece_t p = symGroupRowFirstPiece(&w, rowStart);
    for (size_t i = 0; i < n; i++) {
        if (i > 0) {
            TEST_ASSERT_TRUE(symGroupRowNextPiece(&w, &p));
        }
        TEST_ASSERT_EQUAL_size_t(expected[i][0], p.group);
        TEST_ASSERT_EQUAL_size_t(expected[i][1], p.chLo);
        TEST_ASSERT_EQUAL_size_t(expected[i][2], p.kLo);
        TEST_ASSERT_EQUAL_size_t(expected[i][3], p.chHi);
        TEST_ASSERT_EQUAL_size_t(expected[i][4], p.kHi);
    }
    TEST_ASSERT_FALSE(symGroupRowNextPiece(&w, &p));
}

/* G 4 < row 10: the row's groups straddle the channel and the row ends. */
void testSymGroupRowWalkGroupSize4Kernel5(void) {
    const size_t at0[3][5] = {{0, 0, 0, 0, 4}, {1, 0, 4, 1, 3}, {2, 1, 3, 2, 0}};
    sgAssertRowPieces(4, 5, 2, 0, at0, 3);
    const size_t at10[3][5] = {{2, 0, 0, 0, 2}, {3, 0, 2, 1, 1}, {4, 1, 1, 2, 0}};
    sgAssertRowPieces(4, 5, 2, 10, at10, 3);
    /* rowStart off the tap grid; the last group runs past the row end */
    const size_t at13[3][5] = {{3, 0, 0, 0, 3}, {4, 0, 3, 1, 2}, {5, 1, 2, 2, 0}};
    sgAssertRowPieces(4, 5, 2, 13, at13, 3);
}

/* G 3 < K 5 (G / K 0): boundaries inside one channel's taps; at rowStart 10
 * a piece ends exactly on a channel start (kHi 0). */
void testSymGroupRowWalkGroupSize3Kernel5(void) {
    const size_t at0[4][5] = {{0, 0, 0, 0, 3}, {1, 0, 3, 1, 1}, {2, 1, 1, 1, 4}, {3, 1, 4, 2, 0}};
    sgAssertRowPieces(3, 5, 2, 0, at0, 4);
    const size_t at10[4][5] = {{3, 0, 0, 0, 2}, {4, 0, 2, 1, 0}, {5, 1, 0, 1, 3}, {6, 1, 3, 2, 0}};
    sgAssertRowPieces(3, 5, 2, 10, at10, 4);
}

void testSymGroupRowWalkGroupSize1Kernel3(void) {
    const size_t at0[6][5] = {{0, 0, 0, 0, 1}, {1, 0, 1, 0, 2}, {2, 0, 2, 1, 0},
                              {3, 1, 0, 1, 1}, {4, 1, 1, 1, 2}, {5, 1, 2, 2, 0}};
    sgAssertRowPieces(1, 3, 2, 0, at0, 6);
    const size_t at6[6][5] = {{6, 0, 0, 0, 1}, {7, 0, 1, 0, 2},  {8, 0, 2, 1, 0},
                              {9, 1, 0, 1, 1}, {10, 1, 1, 1, 2}, {11, 1, 2, 2, 0}};
    sgAssertRowPieces(1, 3, 2, 6, at6, 6);
}

/* G 12 > K 5 (G / K 2, G % K 2): a piece end past the row clamps to (rows, 0). */
void testSymGroupRowWalkGroupSize12Kernel5(void) {
    const size_t rows2At8[2][5] = {{0, 0, 0, 0, 4}, {1, 0, 4, 2, 0}};
    sgAssertRowPieces(12, 5, 2, 8, rows2At8, 2);
    /* one group covers the whole row */
    const size_t rows2At0[1][5] = {{0, 0, 0, 2, 0}};
    sgAssertRowPieces(12, 5, 2, 0, rows2At0, 1);
    /* rowStart on a group boundary: the first piece is a full group */
    const size_t rows3At0[2][5] = {{0, 0, 0, 2, 2}, {1, 2, 2, 3, 0}};
    sgAssertRowPieces(12, 5, 3, 0, rows3At0, 2);
    const size_t rows3At15[2][5] = {{1, 0, 0, 1, 4}, {2, 1, 4, 3, 0}};
    sgAssertRowPieces(12, 5, 3, 15, rows3At15, 2);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testConv1dKernelSymStampsResultWidthIncludesSeed);
    RUN_TEST(testConv1dKernelSymGroupedStampsFoldWidthIncludesBiasSeed);
    RUN_TEST(testConv1dKernelSymGroupedAsymImageStampsItsFullRange);
    RUN_TEST(testConv1dKernelSymChainedRawOutputOverflowAborts);
    RUN_TEST(testConv1dKernelSymGroupedWidePartialAboveSeedGuardRunsExactWhenSumFits);
    RUN_TEST(testConv1dKernelSymGroupedWidePartialSumThatWrapsHitsTheAccumulatorGuard);
    RUN_TEST(testConv1dKernelSymGroupedGuardWideWeightRunsExactWhenItFitsAndAbortsWhenItWraps);
    RUN_TEST(testConv1dKernelSymGroupedGuardAsymImageWeightKeepsItsFullRange);
    RUN_TEST(testConv1dKernelSymGuardProvenSafeN511RunsExact);
    RUN_TEST(testConv1dKernelSymGuardDiesAtFirstWrappingNChannels);
    RUN_TEST(testConv1dKernelSymGuardDiesAtN513PositiveCodes);
    RUN_TEST(testConv1dKernelSymGuardCountsKernelTapsInN);
    RUN_TEST(testConv1dKernelSymGuardCheckedPathStaysExactWhenValuesFit);
    RUN_TEST(testConv1dKernelSymGuardBiasSeedThatTipsTheBoundIsCounted);
    RUN_TEST(testConv1dKernelSymGuardWidthsFollowTheOperandsNotTheDefault);
    RUN_TEST(testConv1dKernelSymGuardWideProductOverflowAborts);
    RUN_TEST(testConv1dKernelSymGroupedGuardDiesOnRawGroupPartialOverflow);
    RUN_TEST(testConv1dKernelSymGroupedGuardDiesOnRescaledAccumulatorOverflow);
    RUN_TEST(testConv1dKernelSymGroupedGuardCheckedPathStaysExactWhenValuesFit);
    RUN_TEST(testConv1dKernelSymGroupedGuardFullScaleN511RunsExact);
    RUN_TEST(testConv1dKernelSymGroupedGuardWidthsFollowTheInputWidth);
    RUN_TEST(testConv1dKernelSymGroupedGuardWideInputProductOverflowAborts);
    RUN_TEST(testConv1dKernelSymGroupedGuardCountsKernelTapsInN);
    RUN_TEST(testConv1dKernelSymGroupedAlignedGuardCheckedPathStaysExactWhenValuesFit);
    RUN_TEST(testConv1dKernelSymGroupedAlignedGuardDiesOnRawGroupPartialOverflow);
    RUN_TEST(testConv1dKernelSymGroupedAlignedGuardDiesOnRescaledAccumulatorOverflow);
    RUN_TEST(testConv1dKernelSingleChannelSingleBatch);
    RUN_TEST(testConv1dKernelMultiChannelWithBias);
    RUN_TEST(testConv1dKernelMultiBatch);
    RUN_TEST(testConv1dKernelGroupsDepthwise);
    RUN_TEST(testConv1dKernelGroupsGrouped);
    RUN_TEST(testConv1dKernelStrideDilation);
    RUN_TEST(testConv1dKernelSamePadding);
    RUN_TEST(testConv1dKernelExplicitPaddingStride2);
    RUN_TEST(testConv1dKernelBfpMatchesGold);
    RUN_TEST(testConv1dKernelBfpNoBiasZeroSeeds);
    RUN_TEST(testConv1dKernelBfpDilation2MatchesGold);
    RUN_TEST(testConv1dKernelBfpGroupedBiasBindsPerGroupExponent);
    RUN_TEST(testConv1dKernelBfpPowerOfTwoBitIdenticalToGroupedSym);
    RUN_TEST(testConv1dKernelBfpHeadroomGuardDies);
    RUN_TEST(testConv1dKernelBfpConvGroups2MatchesGold);
    RUN_TEST(testConv1dKernelBfpRawMatchesGold);
    RUN_TEST(testConv1dKernelBfpRawNoBiasMatchesGold);
    RUN_TEST(testConv1dKernelBfpRawGroupedBiasBindsPerGroupExponent);
    RUN_TEST(testConv1dKernelBfpRawStraddlingTargetGroupTakesTheLargerRow);
    RUN_TEST(testConv1dKernelBfpRawDilation2MatchesGold);
    RUN_TEST(testConv1dKernelBfpRawConvGroups2MatchesGold);
    RUN_TEST(testConv1dKernelBfpRawAlignsTwoSided);
    RUN_TEST(testConv1dKernelBfpRawSplitsEveryProductAtM16);
    RUN_TEST(testConv1dKernelBfpRawSplitsLongRunAtTheSegmentLimit);
    RUN_TEST(testConv1dKernelBfpRawDilatedRunStopsAtTheInputGroupEnd);
    RUN_TEST(testConv1dKernelBfpRawWorstCaseM16xM16AllCodesAtTheEdges);
    RUN_TEST(testConv1dKernelBfpRawWorstCaseM16xM8SplitRunsAtTheEdges);
    RUN_TEST(testConv1dKernelBfpRawWorstCaseWithBiasAtTheFloor);
    RUN_TEST(testConv1dKernelBfpRawBiasExponentDominatesTheProducts);
    RUN_TEST(testConv1dKernelBfpRawPowerOfTwoDequantMatchesFloatKernel);
    RUN_TEST(testConv1dKernelBfpRawRejectsMismatchedRawGeometry);
    RUN_TEST(testConv1dKernelBfpRawZeroInputGroupDoesNotSetTheAccumulator);
    RUN_TEST(testConv1dKernelBfpRawZeroBiasDoesNotSetTheAccumulator);
    RUN_TEST(testConv1dGroupedGoldenHalfAway);
    RUN_TEST(testConv1dGroupedGoldenSr);
    RUN_TEST(testConv1dGroupedGoldenConvGroups2);
    RUN_TEST(testConv1dGroupedGoldenConvGroups2ChannelAligned);
    RUN_TEST(testConv1dGroupedFoldFallsBackToLegacyForWideInput);
    RUN_TEST(testConv1dGroupedFoldGoldenHalfAwayChannel);
    RUN_TEST(testConv1dGroupedFoldGoldenHalfAwayGroupSize2);
    RUN_TEST(testConv1dGroupedFoldGoldenConvGroups2);
    RUN_TEST(testConv1dGroupedFoldStampsFoldScaleAndWidth);
    RUN_TEST(testConv1dGroupedFoldEmptyWindowCombinesNothing);
    RUN_TEST(testConv1dGroupedFoldSrIsUnbiased);
    RUN_TEST(testConv1dGroupedFoldLargeBiasLowersFInsteadOfAborting);
    RUN_TEST(testConv1dGroupedFoldUnalignedStraddleSame);
    RUN_TEST(testConv1dGroupedFoldUnalignedShortSegmentEdge);
    RUN_TEST(testConv1dGroupedFoldUnalignedGroupSize1);
    RUN_TEST(testConv1dGroupedFoldUnalignedConvGroups2);
    RUN_TEST(testSymGroupRowWalkGroupSize4Kernel5);
    RUN_TEST(testSymGroupRowWalkGroupSize3Kernel5);
    RUN_TEST(testSymGroupRowWalkGroupSize1Kernel3);
    RUN_TEST(testSymGroupRowWalkGroupSize12Kernel5);
    return UNITY_END();
}
