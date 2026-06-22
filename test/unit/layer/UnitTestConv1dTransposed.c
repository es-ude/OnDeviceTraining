#include <stdlib.h>

#include "Conv1dTransposed.h"
#include "ConvTranspose1dKernel.h"
#include "DeathTest.h"
#include "Layer.h"
#include "QuantizationApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "expected_conv1d_transposed.h"
#include "unity.h"

// Helper: build a Conv1dTransposed layer manually (no UserAPI in Phase 1)
typedef struct convT1dRunResult {
    parameter_t *weights;
    parameter_t *bias;
    layer_t layer;
    layerConfig_t lc;
    conv1dTransposedConfig_t cfg;
    tensor_t *input;
    tensor_t *output;
    kernel_t kernel;
    quantization_t *q;
} convT1dRunResult_t;

static void convT1dBuild(convT1dRunResult_t *r, float const *weightData, size_t const *weightDims,
                         float const *biasData, size_t const *biasDims, int hasBias,
                         float const *inputData, size_t const *inputDims, size_t kSize,
                         size_t dilation, size_t stride, size_t groups, size_t outputPadding,
                         float *outputBuf, size_t const *outputDims) {
    tensor_t *weightParam = tensorInitFloat((float *)weightData, (size_t *)weightDims, 3, NULL);
    tensor_t *weightGrad = gradInitFloat(weightParam, NULL);
    r->weights = parameterInit(weightParam, weightGrad);

    if (hasBias) {
        tensor_t *biasParam = tensorInitFloat((float *)biasData, (size_t *)biasDims, 1, NULL);
        tensor_t *biasGrad = gradInitFloat(biasParam, NULL);
        r->bias = parameterInit(biasParam, biasGrad);
    } else {
        r->bias = NULL;
    }

    initKernel(&r->kernel, kSize, VALID, dilation, stride);
    r->q = quantizationInitFloat();

    initConv1dTransposedConfigWithWeightsAndBias(&r->cfg, &r->kernel, r->weights, r->bias, groups,
                                                 outputPadding, r->q, r->q, r->q, r->q);
    r->layer.type = CONV1D_TRANSPOSED;
    r->lc.conv1dTransposed = &r->cfg;
    r->layer.config = &r->lc;

    r->input = tensorInitFloat((float *)inputData, (size_t *)inputDims, 3, NULL);
    r->output = tensorInitFloat(outputBuf, (size_t *)outputDims, 3, NULL);
}

/* Build a SYM_INT32 (HALF_AWAY, qMaxBits=12 operands) tensor from a float fixture:
 * values are quantized via tensorFillFromFloatBuffer (absmax->scale, round-clamp).
 * The fixtures are dequant-round-trip-stable (sym_gold.stable_dequant_i12) so the C
 * side lands on exactly the gold mantissas+scale. NULL vals -> zero mantissas, scale 1.0. */
static tensor_t *buildSymTensor(size_t numDims, const size_t *dimsIn, const float *vals) {
    size_t *dims = reserveMemory(numDims * sizeof(size_t));
    for (size_t i = 0; i < numDims; i++) {
        dims[i] = dimsIn[i];
    }
    size_t *order = reserveMemory(numDims * sizeof(size_t));
    setOrderOfDimsForNewTensor(numDims, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, numDims, order);
    tensor_t *t = initTensor(shape, quantizationInitSymInt32WithBits(HALF_AWAY, 12), NULL);
    if (vals != NULL) {
        tensorFillFromFloatBuffer(t, vals, calcNumberOfElementsByShape(shape));
    }
    return t;
}

static parameter_t *buildSymParam(size_t numDims, const size_t *dimsIn, const float *vals) {
    tensor_t *p = buildSymTensor(numDims, dimsIn, vals);
    tensor_t *g = gradInitSymInt32(p, HALF_AWAY, NULL);
    return parameterInit(p, g);
}

static float symScaleOf(tensor_t *t) {
    return ((symInt32QConfig_t *)t->quantization->qConfig)->scale;
}

void testConv1dTransposedForwardSymSingleChannelSingleBatch() {
    size_t weightDims[] = {1, 1, 2};
    size_t inputDims[] = {1, 1, 3};
    size_t outputDims[] = {1, 1, 4}; /* Lout=(3-1)*1+1*(2-1)+0+1=4 */

    parameter_t *weights = buildSymParam(3, weightDims, weight_convT1dSym_singleChannelSingleBatch);
    tensor_t *input = buildSymTensor(3, inputDims, input_convT1dSym_singleChannelSingleBatch);
    tensor_t *output = buildSymTensor(3, outputDims, NULL);

    kernel_t kernel;
    initKernel(&kernel, 2, VALID, 1, 1);
    quantization_t *sq = quantizationInitSymInt32(HALF_AWAY);
    conv1dTransposedConfig_t cfg;
    static layerConfig_t lc;
    static layer_t layer;
    initConv1dTransposedConfigWithWeightsAndBias(&cfg, &kernel, weights, NULL, 1, 0, sq, sq, sq,
                                                 sq);
    layer.type = CONV1D_TRANSPOSED;
    lc.conv1dTransposed = &cfg;
    layer.config = &lc;

    conv1dTransposedForward(&layer, input, output);

    int32_t *m = (int32_t *)output->data;
    for (size_t i = 0; i < expectedForward_convT1dSym_singleChannelSingleBatch_len; i++) {
        TEST_ASSERT_INT_WITHIN(forwardMantissaTol_convT1dSym_singleChannelSingleBatch,
                               expectedForward_convT1dSym_singleChannelSingleBatch[i], m[i]);
    }
    float scale = symScaleOf(output);
    TEST_ASSERT_FLOAT_WITHIN(expectedForwardScale_convT1dSym_singleChannelSingleBatch * 1e-4f,
                             expectedForwardScale_convT1dSym_singleChannelSingleBatch, scale);
    for (size_t i = 0; i < expectedForwardDequant_convT1dSym_singleChannelSingleBatch_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(forwardDequantTol_convT1dSym_singleChannelSingleBatch,
                                 expectedForwardDequant_convT1dSym_singleChannelSingleBatch[i],
                                 (float)m[i] * scale);
    }
}

void testConv1dTransposedForwardSingleChannelSingleBatch() {
    convT1dRunResult_t r;
    size_t weightDims[] = {1, 1, 2};
    size_t inputDims[] = {1, 1, 3};
    size_t outputDims[] = {1, 1, 4};
    float outputData[1 * 1 * 4] = {0};

    convT1dBuild(&r, weight_convT1d_singleChannelSingleBatch, weightDims, NULL, NULL, 0,
                 input_convT1d_singleChannelSingleBatch, inputDims, 2, 1, 1, 1, 0, outputData,
                 outputDims);

    conv1dTransposedForward(&r.layer, r.input, r.output);

    for (size_t i = 0; i < expectedForward_convT1d_singleChannelSingleBatch_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expectedForward_convT1d_singleChannelSingleBatch[i],
                                 ((float *)r.output->data)[i]);
    }
}

void testConv1dTransposedForwardMultiChannelWithBias() {
    convT1dRunResult_t r;
    size_t weightDims[] = {3, 2, 2};
    size_t biasDims[] = {2};
    size_t inputDims[] = {1, 3, 4};
    size_t outputDims[] = {1, 2, 5}; // (4-1)*1 + 1*(2-1) + 0 + 1 = 5
    float outputData[1 * 2 * 5] = {0};

    convT1dBuild(&r, weight_convT1d_multiChannelWithBias, weightDims,
                 bias_convT1d_multiChannelWithBias, biasDims, 1, input_convT1d_multiChannelWithBias,
                 inputDims, 2, 1, 1, 1, 0, outputData, outputDims);

    conv1dTransposedForward(&r.layer, r.input, r.output);

    for (size_t i = 0; i < expectedForward_convT1d_multiChannelWithBias_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expectedForward_convT1d_multiChannelWithBias[i],
                                 ((float *)r.output->data)[i]);
    }
}

void testConv1dTransposedCalcOutputShape() {
    // Cin=3, Cout=2 (= 2*1 groups=1), K=2, stride=2, dilation=1, outputPadding=1.
    // outLen = (4-1)*2 + 1*(2-1) + 1 + 1 = 9
    convT1dRunResult_t r;
    size_t weightDims[] = {3, 2, 2};
    size_t inputDims[] = {1, 3, 4};
    size_t outputDims[] = {1, 1, 1}; // nonzero so gradInitFloat on output works
    float dummyOut[1] = {0};

    convT1dBuild(&r, weight_convT1d_multiChannelWithBias, weightDims, NULL, NULL, 0,
                 input_convT1d_multiChannelWithBias, inputDims, 2, 1, 2, 1, 1, dummyOut,
                 outputDims);

    // shape_t uses pointer fields — must point to valid stack arrays.
    size_t inDims[3] = {1, 3, 4};
    size_t inOrder[3] = {0, 1, 2};
    shape_t inShape = {.dimensions = inDims, .orderOfDimensions = inOrder, .numberOfDimensions = 3};

    size_t outDims[3] = {0, 0, 0};
    size_t outOrder[3] = {0, 0, 0};
    shape_t outShape = {
        .dimensions = outDims, .orderOfDimensions = outOrder, .numberOfDimensions = 0};
    conv1dTransposedCalcOutputShape(&r.layer, &inShape, &outShape);

    TEST_ASSERT_EQUAL_size_t(3u, outShape.numberOfDimensions);
    TEST_ASSERT_EQUAL_size_t(1u, outShape.dimensions[0]);
    TEST_ASSERT_EQUAL_size_t(2u, outShape.dimensions[1]);
    TEST_ASSERT_EQUAL_size_t(9u, outShape.dimensions[2]);
}

void testConv1dTransposedBackwardSingleChannelWithBias() {
    convT1dRunResult_t r;
    size_t weightDims[] = {1, 1, 2};
    size_t biasDims[] = {1};
    size_t inputDims[] = {1, 1, 3};
    size_t outputDims[] = {1, 1, 4};
    float outputData[1 * 1 * 4] = {0};

    convT1dBuild(&r, weight_convT1d_singleChannelWithBias, weightDims,
                 bias_convT1d_singleChannelWithBias, biasDims, 1,
                 input_convT1d_singleChannelWithBias, inputDims, 2, 1, 1, 1, 0, outputData,
                 outputDims);

    conv1dTransposedForward(&r.layer, r.input, r.output); // sanity

    float lossGradData[4];
    for (size_t i = 0; i < 4; i++) {
        lossGradData[i] = 1.0f;
    }
    tensor_t *lossGrad = tensorInitFloat(lossGradData, outputDims, 3, NULL);

    float propLossData[3] = {0};
    tensor_t *propLoss = tensorInitFloat(propLossData, inputDims, 3, NULL);

    conv1dTransposedBackward(&r.layer, r.input, lossGrad, propLoss);

    for (size_t i = 0; i < expectedPropLoss_convT1d_singleChannelWithBias_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expectedPropLoss_convT1d_singleChannelWithBias[i],
                                 ((float *)propLoss->data)[i]);
    }
    for (size_t i = 0; i < expectedWeightGrad_convT1d_singleChannelWithBias_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expectedWeightGrad_convT1d_singleChannelWithBias[i],
                                 ((float *)r.weights->grad->data)[i]);
    }
    for (size_t i = 0; i < expectedBiasGrad_convT1d_singleChannelWithBias_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expectedBiasGrad_convT1d_singleChannelWithBias[i],
                                 ((float *)r.bias->grad->data)[i]);
    }
}

void testConv1dTransposedBackwardStride2() {
    convT1dRunResult_t r;
    size_t weightDims[] = {1, 1, 2};
    size_t inputDims[] = {1, 1, 3};
    // outLen = (3-1)*2 + 1*(2-1) + 0 + 1 = 6
    size_t outputDims[] = {1, 1, 6};
    float outputData[6] = {0};

    convT1dBuild(&r, weight_convT1d_stride2, weightDims, NULL, NULL, 0, input_convT1d_stride2,
                 inputDims, 2, 1, 2, 1, 0, outputData, outputDims);

    conv1dTransposedForward(&r.layer, r.input, r.output);

    float lossGradData[6];
    for (size_t i = 0; i < 6; i++) {
        lossGradData[i] = 1.0f;
    }
    tensor_t *lossGrad = tensorInitFloat(lossGradData, outputDims, 3, NULL);

    float propLossData[3] = {0};
    tensor_t *propLoss = tensorInitFloat(propLossData, inputDims, 3, NULL);

    conv1dTransposedBackward(&r.layer, r.input, lossGrad, propLoss);

    for (size_t i = 0; i < expectedPropLoss_convT1d_stride2_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expectedPropLoss_convT1d_stride2[i],
                                 ((float *)propLoss->data)[i]);
    }
    for (size_t i = 0; i < expectedWeightGrad_convT1d_stride2_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expectedWeightGrad_convT1d_stride2[i],
                                 ((float *)r.weights->grad->data)[i]);
    }
}

void testConv1dTransposedForwardMultiBatch() {
    convT1dRunResult_t r;
    size_t weightDims[] = {2, 2, 2};
    size_t inputDims[] = {3, 2, 4};
    // outLen = (4-1)*1 + 1*(2-1) + 0 + 1 = 5
    size_t outputDims[] = {3, 2, 5};
    float outputData[3 * 2 * 5] = {0};

    convT1dBuild(&r, weight_convT1d_multiBatch, weightDims, NULL, NULL, 0, input_convT1d_multiBatch,
                 inputDims, 2, 1, 1, 1, 0, outputData, outputDims);

    conv1dTransposedForward(&r.layer, r.input, r.output);

    for (size_t i = 0; i < expectedForward_convT1d_multiBatch_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expectedForward_convT1d_multiBatch[i],
                                 ((float *)r.output->data)[i]);
    }
}

void testConv1dTransposedForwardGroupsDepthwise() {
    convT1dRunResult_t r;
    size_t weightDims[] = {4, 1, 2};
    size_t inputDims[] = {1, 4, 4};
    // outLen = (4-1)*1 + 1*(2-1) + 0 + 1 = 5; Cout = 1*4 = 4
    size_t outputDims[] = {1, 4, 5};
    float outputData[1 * 4 * 5] = {0};

    convT1dBuild(&r, weight_convT1d_groupsDepthwise, weightDims, NULL, NULL, 0,
                 input_convT1d_groupsDepthwise, inputDims, 2, 1, 1, 4, 0, outputData, outputDims);

    conv1dTransposedForward(&r.layer, r.input, r.output);

    for (size_t i = 0; i < expectedForward_convT1d_groupsDepthwise_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expectedForward_convT1d_groupsDepthwise[i],
                                 ((float *)r.output->data)[i]);
    }
}

void testConv1dTransposedBackwardGroupsDepthwise() {
    convT1dRunResult_t r;
    size_t weightDims[] = {4, 1, 2};
    size_t inputDims[] = {1, 4, 4};
    size_t outputDims[] = {1, 4, 5};
    float outputData[1 * 4 * 5] = {0};

    convT1dBuild(&r, weight_convT1d_groupsDepthwise, weightDims, NULL, NULL, 0,
                 input_convT1d_groupsDepthwise, inputDims, 2, 1, 1, 4, 0, outputData, outputDims);

    conv1dTransposedForward(&r.layer, r.input, r.output);

    float lossGradData[1 * 4 * 5];
    for (size_t i = 0; i < 20; i++) {
        lossGradData[i] = 1.0f;
    }
    tensor_t *lossGrad = tensorInitFloat(lossGradData, outputDims, 3, NULL);

    float propLossData[1 * 4 * 4] = {0};
    tensor_t *propLoss = tensorInitFloat(propLossData, inputDims, 3, NULL);

    conv1dTransposedBackward(&r.layer, r.input, lossGrad, propLoss);

    for (size_t i = 0; i < expectedPropLoss_convT1d_groupsDepthwise_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expectedPropLoss_convT1d_groupsDepthwise[i],
                                 ((float *)propLoss->data)[i]);
    }
    for (size_t i = 0; i < expectedWeightGrad_convT1d_groupsDepthwise_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expectedWeightGrad_convT1d_groupsDepthwise[i],
                                 ((float *)r.weights->grad->data)[i]);
    }
}

void testConv1dTransposedForwardGroupsGrouped() {
    convT1dRunResult_t r;
    size_t weightDims[] = {4, 4, 2};
    size_t biasDims[] = {8};
    size_t inputDims[] = {1, 4, 4};
    // outLen = (4-1)*1 + 1*(2-1) + 0 + 1 = 5; Cout = 4*2 = 8
    size_t outputDims[] = {1, 8, 5};
    float outputData[1 * 8 * 5] = {0};

    convT1dBuild(&r, weight_convT1d_groupsGrouped, weightDims, bias_convT1d_groupsGrouped, biasDims,
                 1, input_convT1d_groupsGrouped, inputDims, 2, 1, 1, 2, 0, outputData, outputDims);

    conv1dTransposedForward(&r.layer, r.input, r.output);

    for (size_t i = 0; i < expectedForward_convT1d_groupsGrouped_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expectedForward_convT1d_groupsGrouped[i],
                                 ((float *)r.output->data)[i]);
    }
}

void testConv1dTransposedBackwardGroupsGrouped() {
    convT1dRunResult_t r;
    size_t weightDims[] = {4, 4, 2};
    size_t biasDims[] = {8};
    size_t inputDims[] = {1, 4, 4};
    size_t outputDims[] = {1, 8, 5};
    float outputData[1 * 8 * 5] = {0};

    convT1dBuild(&r, weight_convT1d_groupsGrouped, weightDims, bias_convT1d_groupsGrouped, biasDims,
                 1, input_convT1d_groupsGrouped, inputDims, 2, 1, 1, 2, 0, outputData, outputDims);

    conv1dTransposedForward(&r.layer, r.input, r.output);

    float lossGradData[1 * 8 * 5];
    for (size_t i = 0; i < 40; i++) {
        lossGradData[i] = 1.0f;
    }
    tensor_t *lossGrad = tensorInitFloat(lossGradData, outputDims, 3, NULL);

    float propLossData[1 * 4 * 4] = {0};
    tensor_t *propLoss = tensorInitFloat(propLossData, inputDims, 3, NULL);

    conv1dTransposedBackward(&r.layer, r.input, lossGrad, propLoss);

    for (size_t i = 0; i < expectedPropLoss_convT1d_groupsGrouped_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expectedPropLoss_convT1d_groupsGrouped[i],
                                 ((float *)propLoss->data)[i]);
    }
    for (size_t i = 0; i < expectedWeightGrad_convT1d_groupsGrouped_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expectedWeightGrad_convT1d_groupsGrouped[i],
                                 ((float *)r.weights->grad->data)[i]);
    }
    for (size_t i = 0; i < expectedBiasGrad_convT1d_groupsGrouped_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expectedBiasGrad_convT1d_groupsGrouped[i],
                                 ((float *)r.bias->grad->data)[i]);
    }
}

void testConv1dTransposedForwardStride2WithOutputPadding() {
    convT1dRunResult_t r;
    size_t weightDims[] = {1, 1, 2};
    size_t biasDims[] = {1};
    size_t inputDims[] = {1, 1, 3};
    // outLen = (3-1)*2 + 1*(2-1) + 1 + 1 = 7
    size_t outputDims[] = {1, 1, 7};
    float outputData[7] = {0};

    convT1dBuild(&r, weight_convT1d_stride2WithOutputPadding, weightDims,
                 bias_convT1d_stride2WithOutputPadding, biasDims, 1,
                 input_convT1d_stride2WithOutputPadding, inputDims, 2, 1, 2, 1, 1, outputData,
                 outputDims);

    conv1dTransposedForward(&r.layer, r.input, r.output);

    for (size_t i = 0; i < expectedForward_convT1d_stride2WithOutputPadding_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expectedForward_convT1d_stride2WithOutputPadding[i],
                                 ((float *)r.output->data)[i]);
    }
}

void testConv1dTransposedForwardDilation2() {
    convT1dRunResult_t r;
    size_t weightDims[] = {1, 1, 2};
    size_t inputDims[] = {1, 1, 4};
    // outLen = (4-1)*1 + 2*(2-1) + 0 + 1 = 6
    size_t outputDims[] = {1, 1, 6};
    float outputData[6] = {0};

    convT1dBuild(&r, weight_convT1d_dilation2, weightDims, NULL, NULL, 0, input_convT1d_dilation2,
                 inputDims, 2, 2, 1, 1, 0, outputData, outputDims);

    conv1dTransposedForward(&r.layer, r.input, r.output);

    for (size_t i = 0; i < expectedForward_convT1d_dilation2_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expectedForward_convT1d_dilation2[i],
                                 ((float *)r.output->data)[i]);
    }
}

void testConv1dTransposedCalcOutputShapeWithGroups() {
    convT1dRunResult_t r;
    size_t weightDims[] = {4, 4, 2};
    size_t inputDims[] = {1, 4, 4};
    size_t outputDims[] = {0, 0, 0};
    float dummyOut[1] = {0};

    convT1dBuild(&r, weight_convT1d_groupsGrouped, weightDims, NULL, NULL, 0,
                 input_convT1d_groupsGrouped, inputDims, 2, 1, 1, 2, 0, dummyOut, outputDims);

    // Caller-allocated dimensions/orderOfDimensions (shape_t.dimensions is size_t* not array)
    size_t inDimsForShape[3] = {1, 4, 4};
    size_t inOrder[3] = {0, 1, 2};
    shape_t inShape = {
        .numberOfDimensions = 3, .dimensions = inDimsForShape, .orderOfDimensions = inOrder};

    size_t outDimsForShape[3] = {0, 0, 0};
    size_t outOrder[3] = {0, 0, 0};
    shape_t outShape = {
        .numberOfDimensions = 0, .dimensions = outDimsForShape, .orderOfDimensions = outOrder};

    conv1dTransposedCalcOutputShape(&r.layer, &inShape, &outShape);

    TEST_ASSERT_EQUAL_size_t(3u, outShape.numberOfDimensions);
    TEST_ASSERT_EQUAL_size_t(1u, outShape.dimensions[0]); // batch
    TEST_ASSERT_EQUAL_size_t(8u,
                             outShape.dimensions[1]); // Cout = outChPerGroup * groups = 4 * 2 = 8
    TEST_ASSERT_EQUAL_size_t(5u, outShape.dimensions[2]); // outLen = (4-1)*1 + 1*(2-1) + 0 + 1 = 5
}

void testConv1dTransposedRegistryDispatch() {
    // Verify layerFunctions[CONV1D_TRANSPOSED] entries point at the right fns.
    TEST_ASSERT_NOT_NULL(layerFunctions[CONV1D_TRANSPOSED].forward);
    TEST_ASSERT_NOT_NULL(layerFunctions[CONV1D_TRANSPOSED].backward);
    TEST_ASSERT_NOT_NULL(layerFunctions[CONV1D_TRANSPOSED].calcOutputShape);
    // Identity check: dispatch matches direct call.
    TEST_ASSERT_TRUE(layerFunctions[CONV1D_TRANSPOSED].forward == conv1dTransposedForward);
    TEST_ASSERT_TRUE(layerFunctions[CONV1D_TRANSPOSED].backward == conv1dTransposedBackward);
    TEST_ASSERT_TRUE(layerFunctions[CONV1D_TRANSPOSED].calcOutputShape ==
                     conv1dTransposedCalcOutputShape);
}

void testConv1dTransposedForwardSymSingleChannelWithBias() {
    size_t weightDims[] = {1, 1, 2};
    size_t biasDims[] = {1};
    size_t inputDims[] = {1, 1, 3};
    size_t outputDims[] = {1, 1, 4};

    parameter_t *weights = buildSymParam(3, weightDims, weight_convT1dSym_singleChannelWithBias);
    parameter_t *bias = buildSymParam(1, biasDims, bias_convT1dSym_singleChannelWithBias);
    tensor_t *input = buildSymTensor(3, inputDims, input_convT1dSym_singleChannelWithBias);
    tensor_t *output = buildSymTensor(3, outputDims, NULL);

    kernel_t kernel;
    initKernel(&kernel, 2, VALID, 1, 1);
    quantization_t *sq = quantizationInitSymInt32(HALF_AWAY);
    conv1dTransposedConfig_t cfg;
    static layerConfig_t lc;
    static layer_t layer;
    initConv1dTransposedConfigWithWeightsAndBias(&cfg, &kernel, weights, bias, 1, 0, sq, sq, sq,
                                                 sq);
    layer.type = CONV1D_TRANSPOSED;
    lc.conv1dTransposed = &cfg;
    layer.config = &lc;

    conv1dTransposedForward(&layer, input, output);

    int32_t *m = (int32_t *)output->data;
    for (size_t i = 0; i < expectedForward_convT1dSym_singleChannelWithBias_len; i++) {
        TEST_ASSERT_INT_WITHIN(forwardMantissaTol_convT1dSym_singleChannelWithBias,
                               expectedForward_convT1dSym_singleChannelWithBias[i], m[i]);
    }
    float scale = symScaleOf(output);
    TEST_ASSERT_FLOAT_WITHIN(expectedForwardScale_convT1dSym_singleChannelWithBias * 1e-4f,
                             expectedForwardScale_convT1dSym_singleChannelWithBias, scale);
    for (size_t i = 0; i < expectedForwardDequant_convT1dSym_singleChannelWithBias_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(forwardDequantTol_convT1dSym_singleChannelWithBias,
                                 expectedForwardDequant_convT1dSym_singleChannelWithBias[i],
                                 (float)m[i] * scale);
    }
}

void testConv1dTransposedForwardSymStride2() {
    size_t weightDims[] = {1, 1, 2};
    size_t inputDims[] = {1, 1, 3};
    size_t outputDims[] = {1, 1, 6}; /* Lout=(3-1)*2+1*(2-1)+0+1=6 */

    parameter_t *weights = buildSymParam(3, weightDims, weight_convT1dSym_stride2Sym);
    tensor_t *input = buildSymTensor(3, inputDims, input_convT1dSym_stride2Sym);
    tensor_t *output = buildSymTensor(3, outputDims, NULL);

    kernel_t kernel;
    initKernel(&kernel, 2, VALID, 1, 2);
    quantization_t *sq = quantizationInitSymInt32(HALF_AWAY);
    conv1dTransposedConfig_t cfg;
    static layerConfig_t lc;
    static layer_t layer;
    initConv1dTransposedConfigWithWeightsAndBias(&cfg, &kernel, weights, NULL, 1, 0, sq, sq, sq,
                                                 sq);
    layer.type = CONV1D_TRANSPOSED;
    lc.conv1dTransposed = &cfg;
    layer.config = &lc;

    conv1dTransposedForward(&layer, input, output);

    int32_t *m = (int32_t *)output->data;
    for (size_t i = 0; i < expectedForward_convT1dSym_stride2Sym_len; i++) {
        TEST_ASSERT_INT_WITHIN(forwardMantissaTol_convT1dSym_stride2Sym,
                               expectedForward_convT1dSym_stride2Sym[i], m[i]);
    }
    float scale = symScaleOf(output);
    TEST_ASSERT_FLOAT_WITHIN(expectedForwardScale_convT1dSym_stride2Sym * 1e-4f,
                             expectedForwardScale_convT1dSym_stride2Sym, scale);
    for (size_t i = 0; i < expectedForwardDequant_convT1dSym_stride2Sym_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(forwardDequantTol_convT1dSym_stride2Sym,
                                 expectedForwardDequant_convT1dSym_stride2Sym[i],
                                 (float)m[i] * scale);
    }
}

void testConv1dTransposedForwardSymStride2OutputPadding() {
    size_t weightDims[] = {1, 1, 2};
    size_t biasDims[] = {1};
    size_t inputDims[] = {1, 1, 3};
    size_t outputDims[] = {1, 1, 7}; /* Lout=(3-1)*2+1*(2-1)+1+1=7 */

    parameter_t *weights = buildSymParam(3, weightDims, weight_convT1dSym_stride2OutputPaddingSym);
    parameter_t *bias = buildSymParam(1, biasDims, bias_convT1dSym_stride2OutputPaddingSym);
    tensor_t *input = buildSymTensor(3, inputDims, input_convT1dSym_stride2OutputPaddingSym);
    tensor_t *output = buildSymTensor(3, outputDims, NULL);

    kernel_t kernel;
    initKernel(&kernel, 2, VALID, 1, 2);
    quantization_t *sq = quantizationInitSymInt32(HALF_AWAY);
    conv1dTransposedConfig_t cfg;
    static layerConfig_t lc;
    static layer_t layer;
    initConv1dTransposedConfigWithWeightsAndBias(&cfg, &kernel, weights, bias, 1, 1, sq, sq, sq,
                                                 sq);
    layer.type = CONV1D_TRANSPOSED;
    lc.conv1dTransposed = &cfg;
    layer.config = &lc;

    conv1dTransposedForward(&layer, input, output);

    int32_t *m = (int32_t *)output->data;
    for (size_t i = 0; i < expectedForward_convT1dSym_stride2OutputPaddingSym_len; i++) {
        TEST_ASSERT_INT_WITHIN(forwardMantissaTol_convT1dSym_stride2OutputPaddingSym,
                               expectedForward_convT1dSym_stride2OutputPaddingSym[i], m[i]);
    }
    float scale = symScaleOf(output);
    TEST_ASSERT_FLOAT_WITHIN(expectedForwardScale_convT1dSym_stride2OutputPaddingSym * 1e-4f,
                             expectedForwardScale_convT1dSym_stride2OutputPaddingSym, scale);
    for (size_t i = 0; i < expectedForwardDequant_convT1dSym_stride2OutputPaddingSym_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(forwardDequantTol_convT1dSym_stride2OutputPaddingSym,
                                 expectedForwardDequant_convT1dSym_stride2OutputPaddingSym[i],
                                 (float)m[i] * scale);
    }
}

void testConv1dTransposedCalcWeightGradsSymGroupsGrouped() {
    size_t weightDims[] = {4, 4, 2}; /* [Cin=4, Cout/groups=4, K=2], groups=2 -> Cout=8 */
    size_t biasDims[] = {8};
    size_t inputDims[] = {1, 4, 4};
    size_t lossDims[] = {1, 8, 5};

    parameter_t *weights = buildSymParam(3, weightDims, weight_convT1dSym_groupsGroupedSym);
    parameter_t *bias = buildSymParam(1, biasDims, bias_convT1dSym_groupsGroupedSym);
    tensor_t *input = buildSymTensor(3, inputDims, input_convT1dSym_groupsGroupedSym);
    tensor_t *lossGrad = buildSymTensor(3, lossDims, lossGrad_convT1dSym_groupsGroupedSym);

    kernel_t kernel;
    initKernel(&kernel, 2, VALID, 1, 1);
    conv1dTransposedConfig_t cfg;
    quantization_t *sq = quantizationInitSymInt32(HALF_AWAY);
    initConv1dTransposedConfigWithWeightsAndBias(&cfg, &kernel, weights, bias, 2, 0, sq, sq, sq,
                                                 sq);

    conv1dTransposedCalcWeightGradsSymInt32(&cfg, input, lossGrad);

    int32_t *m = (int32_t *)weights->grad->data;
    for (size_t i = 0; i < expectedWeightGrad_convT1dSym_groupsGroupedSym_len; i++) {
        TEST_ASSERT_INT_WITHIN(weightGradMantissaTol_convT1dSym_groupsGroupedSym,
                               expectedWeightGrad_convT1dSym_groupsGroupedSym[i], m[i]);
    }
    float scale = symScaleOf(weights->grad);
    TEST_ASSERT_FLOAT_WITHIN(expectedWeightGradScale_convT1dSym_groupsGroupedSym * 1e-4f,
                             expectedWeightGradScale_convT1dSym_groupsGroupedSym, scale);
    for (size_t i = 0; i < expectedWeightGradDequant_convT1dSym_groupsGroupedSym_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(weightGradDequantTol_convT1dSym_groupsGroupedSym,
                                 expectedWeightGradDequant_convT1dSym_groupsGroupedSym[i],
                                 (float)m[i] * scale);
    }
}

void testConv1dTransposedCalcBiasGradsSymMultiChannel() {
    size_t weightDims[] = {3, 2, 2}; /* [Cin=3, Cout/groups=2, K=2] */
    size_t biasDims[] = {2};
    size_t lossDims[] = {1, 2, 5};

    parameter_t *weights = buildSymParam(3, weightDims, weight_convT1dSym_multiChannelWithBiasSym);
    parameter_t *bias = buildSymParam(1, biasDims, bias_convT1dSym_multiChannelWithBiasSym);
    tensor_t *lossGrad = buildSymTensor(3, lossDims, lossGrad_convT1dSym_multiChannelWithBiasSym);

    kernel_t kernel;
    initKernel(&kernel, 2, VALID, 1, 1);
    conv1dTransposedConfig_t cfg;
    quantization_t *sq = quantizationInitSymInt32(HALF_AWAY);
    initConv1dTransposedConfigWithWeightsAndBias(&cfg, &kernel, weights, bias, 1, 0, sq, sq, sq,
                                                 sq);

    conv1dTransposedCalcBiasGradsSymInt32(&cfg, lossGrad);

    int32_t *m = (int32_t *)bias->grad->data;
    for (size_t i = 0; i < expectedBiasGrad_convT1dSym_multiChannelWithBiasSym_len; i++) {
        TEST_ASSERT_INT_WITHIN(biasGradMantissaTol_convT1dSym_multiChannelWithBiasSym,
                               expectedBiasGrad_convT1dSym_multiChannelWithBiasSym[i], m[i]);
    }
    float scale = symScaleOf(bias->grad);
    TEST_ASSERT_FLOAT_WITHIN(expectedBiasGradScale_convT1dSym_multiChannelWithBiasSym * 1e-4f,
                             expectedBiasGradScale_convT1dSym_multiChannelWithBiasSym, scale);
    for (size_t i = 0; i < expectedBiasGradDequant_convT1dSym_multiChannelWithBiasSym_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(biasGradDequantTol_convT1dSym_multiChannelWithBiasSym,
                                 expectedBiasGradDequant_convT1dSym_multiChannelWithBiasSym[i],
                                 (float)m[i] * scale);
    }
}

void testConv1dTransposedBackwardSymStride2OutputPadding() {
    size_t weightDims[] = {1, 1, 2};
    size_t biasDims[] = {1};
    size_t inputDims[] = {1, 1, 3};
    size_t outputDims[] = {1, 1, 7}; /* Lout=(3-1)*2+1*(2-1)+1+1=7 */

    parameter_t *weights = buildSymParam(3, weightDims, weight_convT1dSym_stride2OutputPaddingSym);
    parameter_t *bias = buildSymParam(1, biasDims, bias_convT1dSym_stride2OutputPaddingSym);
    tensor_t *input = buildSymTensor(3, inputDims, input_convT1dSym_stride2OutputPaddingSym);
    tensor_t *lossGrad = buildSymTensor(3, outputDims, lossGrad_convT1dSym_stride2OutputPaddingSym);
    tensor_t *propLoss = buildSymTensor(3, inputDims, NULL);

    kernel_t kernel;
    initKernel(&kernel, 2, VALID, 1, 2); /* K=2, VALID, dilation=1, stride=2 */
    conv1dTransposedConfig_t cfg;
    quantization_t *sq = quantizationInitSymInt32(HALF_AWAY);
    static layerConfig_t lc;
    static layer_t layer;
    initConv1dTransposedConfigWithWeightsAndBias(&cfg, &kernel, weights, bias, 1, 1, sq, sq, sq,
                                                 sq);
    layer.type = CONV1D_TRANSPOSED;
    lc.conv1dTransposed = &cfg;
    layer.config = &lc;

    conv1dTransposedBackward(&layer, input, lossGrad, propLoss);

    /* propLoss (dx) */
    int32_t *dx = (int32_t *)propLoss->data;
    for (size_t i = 0; i < expectedPropLoss_convT1dSym_stride2OutputPaddingSym_len; i++) {
        TEST_ASSERT_INT_WITHIN(propLossMantissaTol_convT1dSym_stride2OutputPaddingSym,
                               expectedPropLoss_convT1dSym_stride2OutputPaddingSym[i], dx[i]);
    }
    float dxScale = symScaleOf(propLoss);
    TEST_ASSERT_FLOAT_WITHIN(expectedPropLossScale_convT1dSym_stride2OutputPaddingSym * 1e-4f,
                             expectedPropLossScale_convT1dSym_stride2OutputPaddingSym, dxScale);
    for (size_t i = 0; i < expectedPropLossDequant_convT1dSym_stride2OutputPaddingSym_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(propLossDequantTol_convT1dSym_stride2OutputPaddingSym,
                                 expectedPropLossDequant_convT1dSym_stride2OutputPaddingSym[i],
                                 (float)dx[i] * dxScale);
    }
    /* weightGrad */
    int32_t *dw = (int32_t *)weights->grad->data;
    float dwScale = symScaleOf(weights->grad);
    for (size_t i = 0; i < expectedWeightGrad_convT1dSym_stride2OutputPaddingSym_len; i++) {
        TEST_ASSERT_INT_WITHIN(weightGradMantissaTol_convT1dSym_stride2OutputPaddingSym,
                               expectedWeightGrad_convT1dSym_stride2OutputPaddingSym[i], dw[i]);
        TEST_ASSERT_FLOAT_WITHIN(weightGradDequantTol_convT1dSym_stride2OutputPaddingSym,
                                 expectedWeightGradDequant_convT1dSym_stride2OutputPaddingSym[i],
                                 (float)dw[i] * dwScale);
    }
    /* biasGrad */
    int32_t *db = (int32_t *)bias->grad->data;
    float dbScale = symScaleOf(bias->grad);
    for (size_t i = 0; i < expectedBiasGrad_convT1dSym_stride2OutputPaddingSym_len; i++) {
        TEST_ASSERT_INT_WITHIN(biasGradMantissaTol_convT1dSym_stride2OutputPaddingSym,
                               expectedBiasGrad_convT1dSym_stride2OutputPaddingSym[i], db[i]);
        TEST_ASSERT_FLOAT_WITHIN(biasGradDequantTol_convT1dSym_stride2OutputPaddingSym,
                                 expectedBiasGradDequant_convT1dSym_stride2OutputPaddingSym[i],
                                 (float)db[i] * dbScale);
    }
}

void testConv1dTransposedBackwardSymGroupsGrouped() {
    size_t weightDims[] = {4, 4, 2};
    size_t biasDims[] = {8};
    size_t inputDims[] = {1, 4, 4};
    size_t outputDims[] = {1, 8, 5};
    size_t propLossDims[] = {1, 4, 4};

    parameter_t *weights = buildSymParam(3, weightDims, weight_convT1dSym_groupsGroupedSym);
    parameter_t *bias = buildSymParam(1, biasDims, bias_convT1dSym_groupsGroupedSym);
    tensor_t *input = buildSymTensor(3, inputDims, input_convT1dSym_groupsGroupedSym);
    tensor_t *lossGrad = buildSymTensor(3, outputDims, lossGrad_convT1dSym_groupsGroupedSym);
    tensor_t *propLoss = buildSymTensor(3, propLossDims, NULL);

    kernel_t kernel;
    initKernel(&kernel, 2, VALID, 1, 1);
    conv1dTransposedConfig_t cfg;
    quantization_t *sq = quantizationInitSymInt32(HALF_AWAY);
    static layerConfig_t lc;
    static layer_t layer;
    initConv1dTransposedConfigWithWeightsAndBias(&cfg, &kernel, weights, bias, 2, 0, sq, sq, sq,
                                                 sq);
    layer.type = CONV1D_TRANSPOSED;
    lc.conv1dTransposed = &cfg;
    layer.config = &lc;

    conv1dTransposedBackward(&layer, input, lossGrad, propLoss);

    /* propLoss (dx) */
    int32_t *dx = (int32_t *)propLoss->data;
    for (size_t i = 0; i < expectedPropLoss_convT1dSym_groupsGroupedSym_len; i++) {
        TEST_ASSERT_INT_WITHIN(propLossMantissaTol_convT1dSym_groupsGroupedSym,
                               expectedPropLoss_convT1dSym_groupsGroupedSym[i], dx[i]);
    }
    float dxScale = symScaleOf(propLoss);
    TEST_ASSERT_FLOAT_WITHIN(expectedPropLossScale_convT1dSym_groupsGroupedSym * 1e-4f,
                             expectedPropLossScale_convT1dSym_groupsGroupedSym, dxScale);
    for (size_t i = 0; i < expectedPropLossDequant_convT1dSym_groupsGroupedSym_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(propLossDequantTol_convT1dSym_groupsGroupedSym,
                                 expectedPropLossDequant_convT1dSym_groupsGroupedSym[i],
                                 (float)dx[i] * dxScale);
    }
    /* weightGrad */
    int32_t *dw = (int32_t *)weights->grad->data;
    float dwScale = symScaleOf(weights->grad);
    for (size_t i = 0; i < expectedWeightGrad_convT1dSym_groupsGroupedSym_len; i++) {
        TEST_ASSERT_INT_WITHIN(weightGradMantissaTol_convT1dSym_groupsGroupedSym,
                               expectedWeightGrad_convT1dSym_groupsGroupedSym[i], dw[i]);
        TEST_ASSERT_FLOAT_WITHIN(weightGradDequantTol_convT1dSym_groupsGroupedSym,
                                 expectedWeightGradDequant_convT1dSym_groupsGroupedSym[i],
                                 (float)dw[i] * dwScale);
    }
    /* biasGrad */
    int32_t *db = (int32_t *)bias->grad->data;
    float dbScale = symScaleOf(bias->grad);
    for (size_t i = 0; i < expectedBiasGrad_convT1dSym_groupsGroupedSym_len; i++) {
        TEST_ASSERT_INT_WITHIN(biasGradMantissaTol_convT1dSym_groupsGroupedSym,
                               expectedBiasGrad_convT1dSym_groupsGroupedSym[i], db[i]);
        TEST_ASSERT_FLOAT_WITHIN(biasGradDequantTol_convT1dSym_groupsGroupedSym,
                                 expectedBiasGradDequant_convT1dSym_groupsGroupedSym[i],
                                 (float)db[i] * dbScale);
    }
}

void testConv1dTransposedBackwardSymDilation2() {
    size_t weightDims[] = {1, 1, 2};
    size_t inputDims[] = {1, 1, 4};
    size_t outputDims[] = {1, 1, 6};
    size_t propLossDims[] = {1, 1, 4};

    parameter_t *weights = buildSymParam(3, weightDims, weight_convT1dSym_dilation2Sym);
    tensor_t *input = buildSymTensor(3, inputDims, input_convT1dSym_dilation2Sym);
    tensor_t *lossGrad = buildSymTensor(3, outputDims, lossGrad_convT1dSym_dilation2Sym);
    tensor_t *propLoss = buildSymTensor(3, propLossDims, NULL);

    kernel_t kernel;
    initKernel(&kernel, 2, VALID, 2, 1); /* dilation=2, stride=1 */
    conv1dTransposedConfig_t cfg;
    quantization_t *sq = quantizationInitSymInt32(HALF_AWAY);
    static layerConfig_t lc;
    static layer_t layer;
    initConv1dTransposedConfigWithWeightsAndBias(&cfg, &kernel, weights, NULL, 1, 0, sq, sq, sq,
                                                 sq);
    layer.type = CONV1D_TRANSPOSED;
    lc.conv1dTransposed = &cfg;
    layer.config = &lc;

    conv1dTransposedBackward(&layer, input, lossGrad, propLoss);

    /* propLoss (dx) */
    int32_t *dx = (int32_t *)propLoss->data;
    for (size_t i = 0; i < expectedPropLoss_convT1dSym_dilation2Sym_len; i++) {
        TEST_ASSERT_INT_WITHIN(propLossMantissaTol_convT1dSym_dilation2Sym,
                               expectedPropLoss_convT1dSym_dilation2Sym[i], dx[i]);
    }
    float dxScale = symScaleOf(propLoss);
    TEST_ASSERT_FLOAT_WITHIN(expectedPropLossScale_convT1dSym_dilation2Sym * 1e-4f,
                             expectedPropLossScale_convT1dSym_dilation2Sym, dxScale);
    for (size_t i = 0; i < expectedPropLossDequant_convT1dSym_dilation2Sym_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(propLossDequantTol_convT1dSym_dilation2Sym,
                                 expectedPropLossDequant_convT1dSym_dilation2Sym[i],
                                 (float)dx[i] * dxScale);
    }
    /* weightGrad */
    int32_t *dw = (int32_t *)weights->grad->data;
    float dwScale = symScaleOf(weights->grad);
    for (size_t i = 0; i < expectedWeightGrad_convT1dSym_dilation2Sym_len; i++) {
        TEST_ASSERT_INT_WITHIN(weightGradMantissaTol_convT1dSym_dilation2Sym,
                               expectedWeightGrad_convT1dSym_dilation2Sym[i], dw[i]);
        TEST_ASSERT_FLOAT_WITHIN(weightGradDequantTol_convT1dSym_dilation2Sym,
                                 expectedWeightGradDequant_convT1dSym_dilation2Sym[i],
                                 (float)dw[i] * dwScale);
    }
    /* no biasGrad: bias == NULL */
}

/* ---------------------------------------------------------------------------
 * Shape-guard death tests (#232).
 *
 * Conv1dTransposed weight layout is [Cin, Cout/groups, K], so weight Cout is
 * dimensions[1] * groups (mirror image of Conv1d). The weightGrad helpers stride
 * lossGrad by `batch` (from forwardInput) and write the weight grad by
 * `outChannels` (from lossGrad); the biasGrad helpers write the bias grad by
 * `outChannels`. Each guard must fail-fast via exit(1). FLOAT helpers are static
 * and exercised through conv1dTransposedBackward; SYM helpers are called
 * directly. Data is all-zero — guards read shapes only. convT1dBuild does not
 * run forward, so intentionally inconsistent layers are safe to build here.
 * ------------------------------------------------------------------------- */

void testConv1dTransposedWeightGradFloatRejectsBatchMismatch() {
    size_t weightDims[] = {1, 1, 2}; // [Cin=1, Cout/groups=1, K=2], groups=1 -> Cout=1
    size_t inputDims[] = {2, 1, 3};  // forward batch 2
    size_t outputDims[] = {2, 1, 4};
    float weightData[2] = {0};
    float inputData[6] = {0};
    float outBuf[8] = {0};
    convT1dRunResult_t r;
    convT1dBuild(&r, weightData, weightDims, NULL, NULL, 0, inputData, inputDims, 2, 1, 1, 1, 0,
                 outBuf, outputDims);

    size_t lossDims[] = {1, 1, 4}; // lossGrad batch 1 != forward batch 2
    float lossData[4] = {0};
    tensor_t *lossGrad = tensorInitFloat(lossData, lossDims, 3, NULL);
    size_t propDims[] = {2, 1, 3};
    float propBuf[6] = {0};
    tensor_t *propLoss = tensorInitFloat(propBuf, propDims, 3, NULL);

    ASSERT_EXITS_WITH_FAILURE(conv1dTransposedBackward(&r.layer, r.input, lossGrad, propLoss));
}

void testConv1dTransposedWeightGradFloatRejectsOutChannelMismatch() {
    size_t weightDims[] = {1, 1, 2}; // weight Cout = 1
    size_t inputDims[] = {1, 1, 3};
    size_t outputDims[] = {1, 1, 4};
    float weightData[2] = {0};
    float inputData[3] = {0};
    float outBuf[4] = {0};
    convT1dRunResult_t r;
    convT1dBuild(&r, weightData, weightDims, NULL, NULL, 0, inputData, inputDims, 2, 1, 1, 1, 0,
                 outBuf, outputDims);

    size_t lossDims[] = {1, 3, 4}; // outChannels 3 != weight Cout 1
    float lossData[12] = {0};
    tensor_t *lossGrad = tensorInitFloat(lossData, lossDims, 3, NULL);
    size_t propDims[] = {1, 1, 3};
    float propBuf[3] = {0};
    tensor_t *propLoss = tensorInitFloat(propBuf, propDims, 3, NULL);

    ASSERT_EXITS_WITH_FAILURE(conv1dTransposedBackward(&r.layer, r.input, lossGrad, propLoss));
}

void testConv1dTransposedWeightGradSymRejectsBatchMismatch() {
    size_t weightDims[] = {4, 4, 2}; // [Cin=4, Cout/groups=4, K=2], groups=2 -> Cout=8
    size_t inputDims[] = {2, 4, 4};  // forward batch 2
    size_t lossDims[] = {1, 8, 5};   // lossGrad batch 1 != forward batch 2

    parameter_t *weights = buildSymParam(3, weightDims, NULL);
    tensor_t *input = buildSymTensor(3, inputDims, NULL);
    tensor_t *lossGrad = buildSymTensor(3, lossDims, NULL);

    kernel_t kernel;
    initKernel(&kernel, 2, VALID, 1, 1);
    conv1dTransposedConfig_t cfg;
    quantization_t *sq = quantizationInitSymInt32(HALF_AWAY);
    initConv1dTransposedConfigWithWeightsAndBias(&cfg, &kernel, weights, NULL, 2, 0, sq, sq, sq,
                                                 sq);

    ASSERT_EXITS_WITH_FAILURE(conv1dTransposedCalcWeightGradsSymInt32(&cfg, input, lossGrad));
}

void testConv1dTransposedWeightGradSymRejectsOutChannelMismatch() {
    size_t weightDims[] = {4, 4, 2}; // groups=2 -> weight Cout = 8
    size_t inputDims[] = {1, 4, 4};
    size_t lossDims[] = {1, 10, 5}; // outChannels 10 != weight Cout 8

    parameter_t *weights = buildSymParam(3, weightDims, NULL);
    tensor_t *input = buildSymTensor(3, inputDims, NULL);
    tensor_t *lossGrad = buildSymTensor(3, lossDims, NULL);

    kernel_t kernel;
    initKernel(&kernel, 2, VALID, 1, 1);
    conv1dTransposedConfig_t cfg;
    quantization_t *sq = quantizationInitSymInt32(HALF_AWAY);
    initConv1dTransposedConfigWithWeightsAndBias(&cfg, &kernel, weights, NULL, 2, 0, sq, sq, sq,
                                                 sq);

    ASSERT_EXITS_WITH_FAILURE(conv1dTransposedCalcWeightGradsSymInt32(&cfg, input, lossGrad));
}

void testConv1dTransposedBiasGradFloatRejectsOutChannelMismatch() {
    /* weight Cout == lossGrad outChannels so weightGrad passes; bias Cout differs
     * so the biasGrad guard must fire. convT1dBuild runs no forward, so the
     * inconsistent layer is safe; only backward runs, in the child. */
    size_t weightDims[] = {1, 2, 2}; // [Cin=1, Cout/groups=2, K=2], groups=1 -> Cout=2
    size_t biasDims[] = {1};         // bias Cout = 1 (intentionally inconsistent)
    size_t inputDims[] = {1, 1, 3};
    float weightData[4] = {0};
    float biasData[1] = {0};
    float inputData[3] = {0};
    float outBuf[8] = {0};
    size_t outputDims[] = {1, 2, 4};
    convT1dRunResult_t r;
    convT1dBuild(&r, weightData, weightDims, biasData, biasDims, 1, inputData, inputDims, 2, 1, 1,
                 1, 0, outBuf, outputDims);

    size_t lossDims[] = {1, 2, 4}; // outChannels 2 == weight Cout, != bias Cout 1
    float lossData[8] = {0};
    tensor_t *lossGrad = tensorInitFloat(lossData, lossDims, 3, NULL);
    size_t propDims[] = {1, 1, 3};
    float propBuf[3] = {0};
    tensor_t *propLoss = tensorInitFloat(propBuf, propDims, 3, NULL);

    ASSERT_EXITS_WITH_FAILURE(conv1dTransposedBackward(&r.layer, r.input, lossGrad, propLoss));
}

void testConv1dTransposedBiasGradSymRejectsOutChannelMismatch() {
    size_t weightDims[] = {3, 2, 2}; // K=2 satisfies the config kernel-size check
    size_t biasDims[] = {1};         // bias Cout = 1
    size_t lossDims[] = {1, 3, 5};   // outChannels 3 != bias Cout 1

    parameter_t *weights = buildSymParam(3, weightDims, NULL);
    parameter_t *bias = buildSymParam(1, biasDims, NULL);
    tensor_t *lossGrad = buildSymTensor(3, lossDims, NULL);

    kernel_t kernel;
    initKernel(&kernel, 2, VALID, 1, 1);
    conv1dTransposedConfig_t cfg;
    quantization_t *sq = quantizationInitSymInt32(HALF_AWAY);
    initConv1dTransposedConfigWithWeightsAndBias(&cfg, &kernel, weights, bias, 1, 0, sq, sq, sq,
                                                 sq);

    ASSERT_EXITS_WITH_FAILURE(conv1dTransposedCalcBiasGradsSymInt32(&cfg, lossGrad));
}

void setUp() {}
void tearDown() {}

int main() {
    UNITY_BEGIN();
    RUN_TEST(testConv1dTransposedForwardSingleChannelSingleBatch);
    RUN_TEST(testConv1dTransposedForwardMultiChannelWithBias);
    RUN_TEST(testConv1dTransposedCalcOutputShape);
    RUN_TEST(testConv1dTransposedBackwardSingleChannelWithBias);
    RUN_TEST(testConv1dTransposedBackwardStride2);
    RUN_TEST(testConv1dTransposedForwardMultiBatch);
    RUN_TEST(testConv1dTransposedForwardGroupsDepthwise);
    RUN_TEST(testConv1dTransposedBackwardGroupsDepthwise);
    RUN_TEST(testConv1dTransposedForwardGroupsGrouped);
    RUN_TEST(testConv1dTransposedBackwardGroupsGrouped);
    RUN_TEST(testConv1dTransposedForwardStride2WithOutputPadding);
    RUN_TEST(testConv1dTransposedForwardDilation2);
    RUN_TEST(testConv1dTransposedCalcOutputShapeWithGroups);
    RUN_TEST(testConv1dTransposedRegistryDispatch);
    RUN_TEST(testConv1dTransposedForwardSymSingleChannelSingleBatch);
    RUN_TEST(testConv1dTransposedForwardSymSingleChannelWithBias);
    RUN_TEST(testConv1dTransposedForwardSymStride2);
    RUN_TEST(testConv1dTransposedForwardSymStride2OutputPadding);
    RUN_TEST(testConv1dTransposedCalcWeightGradsSymGroupsGrouped);
    RUN_TEST(testConv1dTransposedCalcBiasGradsSymMultiChannel);
    RUN_TEST(testConv1dTransposedBackwardSymStride2OutputPadding);
    RUN_TEST(testConv1dTransposedBackwardSymGroupsGrouped);
    RUN_TEST(testConv1dTransposedBackwardSymDilation2);
    RUN_TEST(testConv1dTransposedWeightGradFloatRejectsBatchMismatch);
    RUN_TEST(testConv1dTransposedWeightGradFloatRejectsOutChannelMismatch);
    RUN_TEST(testConv1dTransposedWeightGradSymRejectsBatchMismatch);
    RUN_TEST(testConv1dTransposedWeightGradSymRejectsOutChannelMismatch);
    RUN_TEST(testConv1dTransposedBiasGradFloatRejectsOutChannelMismatch);
    RUN_TEST(testConv1dTransposedBiasGradSymRejectsOutChannelMismatch);
    return UNITY_END();
}
