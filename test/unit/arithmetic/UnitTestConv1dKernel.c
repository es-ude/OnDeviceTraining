#include <string.h>

#include "Conv1dKernel.h"
#include "DeathTest.h"
#include "QuantizationApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "expected_bfp_conv1d.h"
#include "expected_conv1d_kernel.h"
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

/* PR2 self-review finding 1: the dilation factor lives in the BFP arm's own
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

/* PR2 self-review finding 3, the conv sibling of UnitTestMatmul.c's
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

/* BFP power-of-two twin (spec §8c), the BFP sibling of the layer-level
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

    conv1dKernelSymInt32Grouped(&inSymTensor, &wSymTensor, NULL, &kernel, 1, &outSymTensor,
                                &weightGroups);

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

void setUp() {}
void tearDown() {}

int main(void) {
    UNITY_BEGIN();
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
    return UNITY_END();
}
