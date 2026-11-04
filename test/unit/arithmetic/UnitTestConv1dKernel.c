#include <math.h>
#include <stdbool.h>
#include <string.h>

#include "BfpRawFixture.h"
#include "Conv1dKernel.h"
#include "DeathTest.h"
#include "Matmul.h"
#include "QuantizationApi.h"
#include "StorageApi.h"
#include "SymAccumGuard.h"
#include "SymGroupedWeight.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "expected_bfp_conv1d.h"
#include "expected_conv1d_kernel.h"
#include "unity.h"

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
    // ECG enc1 geometry (issue #177): K=7, stride=2, EXPLICIT symmetric padding=3.
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

/* ---- BFP epic PR2 (Task 4): conv1dKernelBfp -----------------------------
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

/* BFP epic PR2: the dilation factor lives in the BFP arm's own
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

/* BFP epic PR2, the conv sibling of UnitTestMatmul.c's
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
    inSymQC.scale = 0.5f;
    quantization_t inSymQ;
    initSymInt32Quantization(&inSymQC, &inSymQ);
    setTensorValues(&inSymTensor, (uint8_t *)kBfpConvInCodes, &inShape, &inSymQ, NULL);

    tensor_t wSymTensor;
    symInt32QConfig_t wSymQC;
    initSymInt32QConfig(HALF_AWAY, &wSymQC);
    wSymQC.scale = 1.0f; /* poisoned scratch scale -- never read, scales live in weightGroups */
    quantization_t wSymQ;
    initSymInt32Quantization(&wSymQC, &wSymQ);
    setTensorValues(&wSymTensor, (uint8_t *)kBfpConvWCodes, &wShape, &wSymQ, NULL);

    float scales[6] = {0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f};
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
        float symDequant = (float)outSymData[i] * outSymQC.scale;
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

/* Group-shape fail-fast (Task 3 precedent): bfpGroupOf divides by groupSize
 * with no relation to numGroups, so a mismatched config ({numGroups=2,
 * groupSize=4} on 20 elements: 2*4 == 8 != 20) would silently read
 * exponents[] out of bounds. The kernel must route every operand through
 * validateBfpQConfigShape before touching data. */
void testConv1dKernelBfpRejectsMismatchedGroupShape(void) {
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
    int32_t wData[12] = {0};
    size_t wDims[] = {2, 2, 3};
    size_t wOrder[] = {0, 1, 2};
    shape_t wShape;
    setShape(&wShape, wDims, 3, wOrder);
    uint8_t wExponents[] = {127, 127, 127, 127, 127, 127};
    bfpQConfig_t wQC = {.exponents = wExponents,
                        .numGroups = 6,
                        .groupSize = 2,
                        .roundingMode = HALF_AWAY,
                        .mantissaBits = 4,
                        .exponentBits = 8};
    quantization_t wQ;
    initBfpQuantization(&wQC, &wQ);
    setTensorValues(&wTensor, (uint8_t *)wData, &wShape, &wQ, NULL);

    tensor_t outTensor;
    float outData[10];
    size_t outDims[] = {1, 2, 5};
    size_t outOrder[] = {0, 1, 2};
    shape_t outShape;
    setShape(&outShape, outDims, 3, outOrder);
    quantization_t outQ;
    initFloat32Quantization(&outQ);
    setTensorValues(&outTensor, (uint8_t *)outData, &outShape, &outQ, NULL);

    kernel_t kernel;
    initKernelExplicit(&kernel, 3, 1, 1, 2);

    ASSERT_EXITS_WITH_FAILURE(conv1dKernelBfp(&inTensor, &wTensor, NULL, &kernel, 1, &outTensor));
}

/* conv-groups=2 parity gold (#416, BFP PR3 Task 3): SPLIT channels
 * (inChPerGroup = outChPerGroup = 2, not depthwise), so the group loop's
 * in_lo/w_base channel arithmetic is load-bearing -- the PR2 fixtures all
 * ran groups=1. Same stack-fixture idiom and EXPLICIT-padded stride-2
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

static void conv1dKernelBfpRawFloatOutputBody(void) {
    bfpScratchFixture_t in, w;
    bfpRawMainOperands(&in, &w, kBfpConvWExponents);
    tensor_t out;
    float outData[10];
    size_t outDims[] = {(size_t)kBfpConvBatch, (size_t)kBfpConvOutChannels, (size_t)kBfpConvOutLen};
    size_t outOrder[] = {0, 1, 2};
    shape_t outShape;
    setShape(&outShape, outDims, 3, outOrder);
    quantization_t outQ;
    initFloat32Quantization(&outQ);
    setTensorValues(&out, (uint8_t *)outData, &outShape, &outQ, NULL);
    kernel_t kernel;
    bfpRawMainKernel(&kernel, 1);
    conv1dKernelBfpRaw(&in.tensor, &w.tensor, NULL, &kernel, 1, &out);
}

void testConv1dKernelBfpRawRejectsNonRawOutput(void) {
    ASSERT_EXITS_WITH_FAILURE(conv1dKernelBfpRawFloatOutputBody());
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

/* Operand whose storage dtype cgRun swaps to FLOAT32 (dtype-validation tests). */
typedef enum { CG_POISON_NONE, CG_POISON_X, CG_POISON_W, CG_POISON_BIAS, CG_POISON_Y } cgPoison_t;
static cgPoison_t gCgPoison = CG_POISON_NONE;

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
    if (gCgPoison == CG_POISON_X) {
        initFloat32Quantization(&xQ);
    }
    if (gCgPoison == CG_POISON_W) {
        initFloat32Quantization(&wQ);
    }
    if (gCgPoison == CG_POISON_BIAS) {
        initFloat32Quantization(&biasQ);
    }
    if (gCgPoison == CG_POISON_Y) {
        initFloat32Quantization(&gCgYQ);
    }
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
        float scales[8] = {1, 1, 1, 1, 1, 1, 1, 1};
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
 * the #189 reserve, inside int32). */
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

void testConv1dKernelSymGroupedWidePartialSumThatWrapsHitsTheF9Guard(void) {
    wideGroupedFill(false);
    ASSERT_EXITS_WITH_OUTPUT(1, "conv1dKernelSymInt32Grouped (rescaled accumulator)",
                             (void)cgRun(80, 1, 1, 12, 16, NULL, 2));
}

/* ---- #516: the output declares the accumulator's static width ------------ */

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

/* int4 x grouped int4 (2 groups x 4) over 8 terms plus a bias at the
 * accumulator scale. The output is the RESCALED accumulator, so its width is
 * the rescaled site's over the weight's EXACT code range (Conv1dKernel.c,
 * SymGroupedWeight.h). A stored SYM int4 weight is [-8, 7]: one term spans
 * [-56, 64], 8 slacked terms [-457, 521] (slack 9 per side); bias 1023 -> 1544
 * -> 12, bias 500 -> 1021 -> 11, bias -1023 -> 1480 on the negative side ->
 * 12. Counting the weight one bit wide ([-16, 15], 1033 per side) would say
 * 13 for the bias 1023. The raw group partial alone is 11 (4 * 2^7 = 512). */
void testConv1dKernelSymGroupedStampsRescaledWidth(void) {
    cgFill(8, 1, 1, false);
    int32_t bias = 1023;
    TEST_ASSERT_EQUAL_INT32(1023 + 8, cgRun(8, 1, 1, 4, 4, &bias, 2));
    symAccumSite_t rescaled = symAccumSiteInitRescaled("t", 8, 4, 4, 1023);
    TEST_ASSERT_EQUAL_UINT8(symAccumResultBits(&rescaled), gCgOutBits);
    TEST_ASSERT_EQUAL_UINT8(12, gCgOutBits);
    bias = 500;
    TEST_ASSERT_EQUAL_INT32(500 + 8, cgRun(8, 1, 1, 4, 4, &bias, 2));
    TEST_ASSERT_EQUAL_UINT8(11, gCgOutBits);
    bias = -1023;
    TEST_ASSERT_EQUAL_INT32(8 - 1023, cgRun(8, 1, 1, 4, 4, &bias, 2));
    TEST_ASSERT_EQUAL_UINT8(12, gCgOutBits);
}

/* The same site over a stored ASYM int4 weight's image [-15, 15]: +-120 per
 * term, 969 per side; bias 54 -> 1023 -> 11 (the image counted one bit wide
 * says 12), bias 500 -> 1469 -> 12 (the SYM range says 11). */
void testConv1dKernelSymGroupedAsymImageStampsItsFullRange(void) {
    cgFill(8, 1, 1, false);
    int32_t bias = 54;
    TEST_ASSERT_EQUAL_INT32(54 + 8, cgRunAsymImage(8, 1, 1, 4, 4, &bias, 2));
    TEST_ASSERT_EQUAL_UINT8(11, gCgOutBits);
    bias = 500;
    TEST_ASSERT_EQUAL_INT32(500 + 8, cgRunAsymImage(8, 1, 1, 4, 4, &bias, 2));
    TEST_ASSERT_EQUAL_UINT8(12, gCgOutBits);
}

/* The #516 repro on a conv output: the raw 1x1x1 product -2048 * -2048 = 2^22
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

/* Operand dtype validation: a non-SYM_INT32 operand would be read through the
 * wrong qConfig struct, so each kernel must abort naming itself and the operand. */
static const int32_t kBias = 0;
#define CG_DTYPE_DEATH(name, poison, numGroups, bias, banner)                                      \
    void name(void) {                                                                              \
        cgFill(2, 3, 3, false);                                                                    \
        gCgPoison = poison;                                                                        \
        ASSERT_EXITS_WITH_OUTPUT(1, banner, (void)cgRun(2, 1, 1, 12, 12, bias, numGroups));        \
        gCgPoison = CG_POISON_NONE;                                                                \
    }

CG_DTYPE_DEATH(testConv1dKernelSymRejectsFloatInput, CG_POISON_X, 0, NULL,
               "conv1dKernelSymInt32: input must be SYM_INT32")
CG_DTYPE_DEATH(testConv1dKernelSymRejectsFloatWeight, CG_POISON_W, 0, NULL,
               "conv1dKernelSymInt32: weight must be SYM_INT32")
CG_DTYPE_DEATH(testConv1dKernelSymRejectsFloatBias, CG_POISON_BIAS, 0, &kBias,
               "conv1dKernelSymInt32: bias must be SYM_INT32")
CG_DTYPE_DEATH(testConv1dKernelSymRejectsFloatOutput, CG_POISON_Y, 0, NULL,
               "conv1dKernelSymInt32: output must be SYM_INT32")
CG_DTYPE_DEATH(testConv1dKernelSymGroupedRejectsFloatInput, CG_POISON_X, 2, NULL,
               "conv1dKernelSymInt32Grouped: input must be SYM_INT32")
CG_DTYPE_DEATH(testConv1dKernelSymGroupedRejectsFloatWeight, CG_POISON_W, 2, NULL,
               "conv1dKernelSymInt32Grouped: weight must be SYM_INT32")
CG_DTYPE_DEATH(testConv1dKernelSymGroupedRejectsFloatBias, CG_POISON_BIAS, 2, &kBias,
               "conv1dKernelSymInt32Grouped: bias must be SYM_INT32")
CG_DTYPE_DEATH(testConv1dKernelSymGroupedRejectsFloatOutput, CG_POISON_Y, 2, NULL,
               "conv1dKernelSymInt32Grouped: output must be SYM_INT32")

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testConv1dKernelSymRejectsFloatInput);
    RUN_TEST(testConv1dKernelSymRejectsFloatWeight);
    RUN_TEST(testConv1dKernelSymRejectsFloatBias);
    RUN_TEST(testConv1dKernelSymRejectsFloatOutput);
    RUN_TEST(testConv1dKernelSymGroupedRejectsFloatInput);
    RUN_TEST(testConv1dKernelSymGroupedRejectsFloatWeight);
    RUN_TEST(testConv1dKernelSymGroupedRejectsFloatBias);
    RUN_TEST(testConv1dKernelSymGroupedRejectsFloatOutput);
    RUN_TEST(testConv1dKernelSymStampsResultWidthIncludesSeed);
    RUN_TEST(testConv1dKernelSymGroupedStampsRescaledWidth);
    RUN_TEST(testConv1dKernelSymGroupedAsymImageStampsItsFullRange);
    RUN_TEST(testConv1dKernelSymChainedRawOutputOverflowAborts);
    RUN_TEST(testConv1dKernelSymGroupedWidePartialAboveSeedGuardRunsExactWhenSumFits);
    RUN_TEST(testConv1dKernelSymGroupedWidePartialSumThatWrapsHitsTheF9Guard);
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
    RUN_TEST(testConv1dKernelBfpRejectsMismatchedGroupShape);
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
    RUN_TEST(testConv1dKernelBfpRawWorstCaseM16xM16AllCodesAtTheEdges);
    RUN_TEST(testConv1dKernelBfpRawWorstCaseM16xM8SplitRunsAtTheEdges);
    RUN_TEST(testConv1dKernelBfpRawWorstCaseWithBiasAtTheFloor);
    RUN_TEST(testConv1dKernelBfpRawBiasExponentDominatesTheProducts);
    RUN_TEST(testConv1dKernelBfpRawPowerOfTwoDequantMatchesFloatKernel);
    RUN_TEST(testConv1dKernelBfpRawRejectsNonRawOutput);
    RUN_TEST(testConv1dKernelBfpRawRejectsMismatchedRawGeometry);
    return UNITY_END();
}
