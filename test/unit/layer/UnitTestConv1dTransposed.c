#include <stdlib.h>
#include <string.h>

#include "Conv1dTransposed.h"
#include "Conv1dTransposedApi.h"
#include "ConvTranspose1dKernel.h"
#include "DeathTest.h"
#include "Layer.h"
#include "LayerCommon.h"
#include "LayerQuant.h"
#include "QuantizationApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "expected_conv1d_transposed.h"
#include "expected_convT1d_grouped.h"
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

static void convT1dBuild(convT1dRunResult_t *r, float const *weightData, size_t const *weightDims,
                         float const *biasData, size_t const *biasDims, int hasBias,
                         float const *inputData, size_t const *inputDims, size_t kSize,
                         size_t dilation, size_t stride, size_t groups, size_t outputPadding,
                         float *outputBuf, size_t const *outputDims) {
    tensor_t *weightParam = makeFloatTensor(weightDims, 3, weightData);
    tensor_t *weightGrad = gradInitFloat(weightParam, NULL);
    r->weights = parameterInit(weightParam, weightGrad);

    if (hasBias) {
        tensor_t *biasParam = makeFloatTensor(biasDims, 1, biasData);
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

    r->input = makeFloatTensor(inputDims, 3, inputData);
    r->output = makeFloatTensor(outputDims, 3, NULL);
    (void)outputBuf;
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

/* Re-gold (spec D5): conv1dTransposedForward now routes SYM through
 * executeOp's OUT_WRITE epilogue, which requants the raw s_in*s_w
 * accumulator wire through the conversionMatrix diagonal
 * (requantSymInt32Tensor) instead of writing it unrestored (pre-PR1b.2
 * behavior — the fixture this test asserted against was the raw wire).
 * Dequant-equivalence: restored mantissa*restoredScale == raw
 * mantissa*rawScale within representation tolerance (same real value
 * re-expressed at a different int12 scale) — verified by
 * generate_expected_conv1d_transposed.py's `emulate_sym_convT` self-check
 * (fwd_err <= fwd_tol against the float64 PyTorch-autograd reference,
 * computed on the RESTORED fwd_deq/fwd_scale). Same re-gold class as Task 2's
 * propLoss/Task 3's LayerNorm/Conv1d's own forward pins (ratified spec D5
 * principle). Applies identically to the 3 other
 * testConv1dTransposedForwardSym* tests below. */
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
    TEST_ASSERT_FLOAT_WITHIN(expectedForwardScale_convT1dSym_singleChannelSingleBatch *
                                 forwardScaleTol_convT1dSym_singleChannelSingleBatch,
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
    tensor_t *lossGrad = makeFloatTensor(outputDims, 3, lossGradData);

    tensor_t *propLoss = makeFloatTensor(inputDims, 3, NULL);

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
    tensor_t *lossGrad = makeFloatTensor(outputDims, 3, lossGradData);

    tensor_t *propLoss = makeFloatTensor(inputDims, 3, NULL);

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
    tensor_t *lossGrad = makeFloatTensor(outputDims, 3, lossGradData);

    tensor_t *propLoss = makeFloatTensor(inputDims, 3, NULL);

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
    tensor_t *lossGrad = makeFloatTensor(outputDims, 3, lossGradData);

    tensor_t *propLoss = makeFloatTensor(inputDims, 3, NULL);

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
    TEST_ASSERT_FLOAT_WITHIN(expectedForwardScale_convT1dSym_singleChannelWithBias *
                                 forwardScaleTol_convT1dSym_singleChannelWithBias,
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
    TEST_ASSERT_FLOAT_WITHIN(expectedForwardScale_convT1dSym_stride2Sym *
                                 forwardScaleTol_convT1dSym_stride2Sym,
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
    TEST_ASSERT_FLOAT_WITHIN(expectedForwardScale_convT1dSym_stride2OutputPaddingSym *
                                 forwardScaleTol_convT1dSym_stride2OutputPaddingSym,
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

/* Re-gold (spec D5): conv1dTransposedBackward's dx wire (propLoss) is a
 * *produced* wire, not a passthrough — conv1dKernelSymInt32 (the VALID
 * gather adjoint) emits the raw s_loss*s_w mantissa, and executeOp's
 * OUT_WRITE epilogue then requants it through the conversionMatrix diagonal
 * to propLossQ's declared width (int12) before conv1dTransposedBackward
 * returns it — the same restoration every other producer wire gets (forward
 * output, weightGrad, biasGrad). The propLoss expectations below are
 * therefore POST-restoration values, not the kernel's raw output; the old
 * #187 fail-fast (produced propLoss tensor must be SYM) is retired along
 * with the raw-wire validator (docs/conventions/arithmetic-sym.md).
 * Dequant-equivalence: restored mantissa*restoredScale == raw
 * mantissa*rawScale within representation tolerance — verified by
 * generate_expected_conv1d_transposed.py's `emulate_sym_convT` dx section
 * (`_requant_absmax_i12_f32`, dx_err <= dx_tol against the float64
 * PyTorch-autograd reference, computed on the RESTORED dx_deq/dx_scale).
 * Same re-gold class as this file's forward pins above. Applies identically
 * to the 2 other testConv1dTransposedBackwardSym* tests below. */
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
    tensor_t *lossGrad = makeFloatTensor(lossDims, 3, lossData);
    size_t propDims[] = {2, 1, 3};
    tensor_t *propLoss = makeFloatTensor(propDims, 3, NULL);

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
    tensor_t *lossGrad = makeFloatTensor(lossDims, 3, lossData);
    size_t propDims[] = {1, 1, 3};
    tensor_t *propLoss = makeFloatTensor(propDims, 3, NULL);

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
    tensor_t *lossGrad = makeFloatTensor(lossDims, 3, lossData);
    size_t propDims[] = {1, 1, 3};
    tensor_t *propLoss = makeFloatTensor(propDims, 3, NULL);

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

/* #380 PR1 Task 2: create-time trainable knob (trainable_t). */
static layer_t *buildFloatConv1dTransposedWithTrainable(trainable_t trainable) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);
    layer_t *layer = conv1dTransposedLayerInitOwning(
        &(conv1dTransposedInit_t){
            .inChannels = 2, .outChannels = 3, .kernelSize = 3, .trainable = trainable},
        &lq);
    freeQuantization(q);
    return layer;
}

void testConv1dTransposedFactoryFrozenElidesGrads(void) {
    layer_t *layer = buildFloatConv1dTransposedWithTrainable(TRAINABLE_FALSE);
    conv1dTransposedConfig_t *cfg = layer->config->conv1dTransposed;
    bool weightsGradNull = cfg->weights->grad == NULL;
    bool biasGradNull = cfg->bias->grad == NULL;
    bool frozen = layerIsFrozen(layer);
    freeConv1dTransposedLayer(layer);
    TEST_ASSERT_TRUE(weightsGradNull);
    TEST_ASSERT_TRUE(biasGradNull);
    TEST_ASSERT_TRUE(frozen);
}

void testConv1dTransposedFactoryDefaultAllocatesGrads(void) {
    layer_t *layer = buildFloatConv1dTransposedWithTrainable(TRAINABLE_DEFAULT);
    conv1dTransposedConfig_t *cfg = layer->config->conv1dTransposed;
    bool weightsGradPresent = cfg->weights->grad != NULL;
    bool biasGradPresent = cfg->bias->grad != NULL;
    bool frozen = layerIsFrozen(layer);
    freeConv1dTransposedLayer(layer);
    TEST_ASSERT_TRUE(weightsGradPresent);
    TEST_ASSERT_TRUE(biasGradPresent);
    TEST_ASSERT_FALSE(frozen);
}

/* #380 PR1 Task 5: backward guard -- a frozen twin must skip the weight/bias
 * grad writes entirely (buffers stay all-zero) while still producing a
 * propLoss byte-identical to its trainable twin. Hand-seeded FLOAT32
 * fixtures via the file's own convT1dBuild helper (no borrowed builder
 * exists for ConvT1d) -- deterministic, no RNG, so the two twins start out
 * bit-identical; only `frozen` differs. */
void testConv1dTransposedBackwardFrozenTwinPropLossIdenticalGradsZero(void) {
    convT1dRunResult_t rA, rB;
    size_t weightDims[] = {1, 1, 2};
    size_t biasDims[] = {1};
    size_t inputDims[] = {1, 1, 3};
    size_t outputDims[] = {1, 1, 4};
    float outputDataA[1 * 1 * 4] = {0};
    float outputDataB[1 * 1 * 4] = {0};

    convT1dBuild(&rA, (float[]){1.f, -1.f}, weightDims, (float[]){0.5f}, biasDims, 1,
                 (float[]){1.f, 2.f, 3.f}, inputDims, 2, 1, 1, 1, 0, outputDataA, outputDims);
    convT1dBuild(&rB, (float[]){1.f, -1.f}, weightDims, (float[]){0.5f}, biasDims, 1,
                 (float[]){1.f, 2.f, 3.f}, inputDims, 2, 1, 1, 1, 0, outputDataB, outputDims);
    rB.cfg.frozen = true;

    tensor_t *lossGrad = makeFloatTensor(outputDims, 3, (float[]){1.f, 1.f, 1.f, 1.f});
    tensor_t *propLossTrainable = makeFloatTensor(inputDims, 3, NULL);
    tensor_t *propLossFrozen = makeFloatTensor(inputDims, 3, NULL);

    conv1dTransposedBackward(&rA.layer, rA.input, lossGrad, propLossTrainable);
    conv1dTransposedBackward(&rB.layer, rB.input, lossGrad, propLossFrozen);

    size_t numWeights = calcNumberOfElementsByTensor(rA.weights->param);
    size_t numBias = calcNumberOfElementsByTensor(rA.bias->param);
    size_t numPropLoss = calcNumberOfElementsByTensor(propLossTrainable);

    bool trainableWeightGradNonzero = false;
    bool frozenWeightGradAllZero = true;
    for (size_t i = 0; i < numWeights; i++) {
        if (((float *)rA.weights->grad->data)[i] != 0.0f) {
            trainableWeightGradNonzero = true;
        }
        if (((float *)rB.weights->grad->data)[i] != 0.0f) {
            frozenWeightGradAllZero = false;
        }
    }
    bool trainableBiasGradNonzero = false;
    bool frozenBiasGradAllZero = true;
    for (size_t i = 0; i < numBias; i++) {
        if (((float *)rA.bias->grad->data)[i] != 0.0f) {
            trainableBiasGradNonzero = true;
        }
        if (((float *)rB.bias->grad->data)[i] != 0.0f) {
            frozenBiasGradAllZero = false;
        }
    }
    bool propLossIdentical =
        memcmp(propLossTrainable->data, propLossFrozen->data,
               calcNumberOfBytesForData(propLossTrainable->quantization, numPropLoss)) == 0;

    TEST_ASSERT_TRUE_MESSAGE(trainableWeightGradNonzero,
                             "trainable twin weight grad must be written (nonzero)");
    TEST_ASSERT_TRUE_MESSAGE(frozenWeightGradAllZero,
                             "frozen twin weight grad must stay untouched (all-zero)");
    TEST_ASSERT_TRUE_MESSAGE(trainableBiasGradNonzero,
                             "trainable twin bias grad must be written (nonzero)");
    TEST_ASSERT_TRUE_MESSAGE(frozenBiasGradAllZero,
                             "frozen twin bias grad must stay untouched (all-zero)");
    TEST_ASSERT_TRUE_MESSAGE(propLossIdentical, "propLoss must be byte-identical between twins");
}

/* Factory-frozen layer (grads == NULL, Task 2): conv1dTransposedBackward must
 * complete without dereferencing the (absent) grad buffers -- the ASan gate
 * catches any NULL/OOB deref if the guard is missing or misplaced. */
void testConv1dTransposedBackwardFrozenFactoryLayerRunsWithoutGradBuffers(void) {
    layer_t *layer = buildFloatConv1dTransposedWithTrainable(TRAINABLE_FALSE);

    size_t inputDims[] = {1, 2, 4};
    size_t outputDims[] = {1, 3, 6}; /* Lout=(4-1)*1+1*(3-1)+0+1=6 */
    tensor_t *input =
        makeFloatTensor(inputDims, 3, (float[]){1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f});
    tensor_t *lossGrad = makeFloatTensor(outputDims, 3,
                                         (float[]){1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f,
                                                   1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f});
    tensor_t *propLoss = makeFloatTensor(inputDims, 3, NULL);

    conv1dTransposedBackward(layer, input, lossGrad, propLoss);

    bool gradStillNull = layer->config->conv1dTransposed->weights->grad == NULL;
    freeConv1dTransposedLayer(layer);

    TEST_ASSERT_TRUE(gradStillNull);
}

/* #380 PR2 Task 1: propLoss == NULL is a grads-only call -- weight/bias grads
 * must be computed exactly as with a real propLoss, and no dx memory may be
 * touched. Twin fixture (both TRAINABLE, hand-seeded, bit-identical) via the
 * file's own convT1dBuild helper (no borrowed builder exists for ConvT1d),
 * mirroring the PR1 frozen-twin fixture above: twin A gets a real propLoss
 * buffer, twin B gets a literal NULL. Pre-guard, twin B's call dereferences
 * the NULL propLoss and crashes (RED); post-guard, weight/bias grads match
 * twin A's byte-for-byte and twin A's propLoss is non-degenerate. */
void testConv1dTransposedBackwardNullPropLossComputesGradsOnly(void) {
    convT1dRunResult_t rA, rB;
    size_t weightDims[] = {1, 1, 2};
    size_t biasDims[] = {1};
    size_t inputDims[] = {1, 1, 3};
    size_t outputDims[] = {1, 1, 4};
    float outputDataA[1 * 1 * 4] = {0};
    float outputDataB[1 * 1 * 4] = {0};

    convT1dBuild(&rA, (float[]){1.f, -1.f}, weightDims, (float[]){0.5f}, biasDims, 1,
                 (float[]){1.f, 2.f, 3.f}, inputDims, 2, 1, 1, 1, 0, outputDataA, outputDims);
    convT1dBuild(&rB, (float[]){1.f, -1.f}, weightDims, (float[]){0.5f}, biasDims, 1,
                 (float[]){1.f, 2.f, 3.f}, inputDims, 2, 1, 1, 1, 0, outputDataB, outputDims);

    /* Non-uniform lossGrad (unlike the frozen-twin fixture's all-ones): with
     * weight [1,-1], a uniform lossGrad makes the adjoint-conv propLoss
     * identically zero (lossGrad[i]-lossGrad[i+1] == 0), which would make the
     * propLossA non-degeneracy assertion below vacuous. */
    tensor_t *lossGrad = makeFloatTensor(outputDims, 3, (float[]){1.f, 2.f, 3.f, 4.f});
    tensor_t *propLossA = makeFloatTensor(inputDims, 3, NULL);

    conv1dTransposedBackward(&rA.layer, rA.input, lossGrad, propLossA);
    conv1dTransposedBackward(&rB.layer, rB.input, lossGrad, NULL);

    size_t numWeights = calcNumberOfElementsByTensor(rA.weights->param);
    size_t numBias = calcNumberOfElementsByTensor(rA.bias->param);
    size_t numPropLoss = calcNumberOfElementsByTensor(propLossA);

    bool weightGradsIdentical =
        memcmp(rA.weights->grad->data, rB.weights->grad->data,
               calcNumberOfBytesForData(rA.weights->grad->quantization, numWeights)) == 0;
    bool biasGradsIdentical =
        memcmp(rA.bias->grad->data, rB.bias->grad->data,
               calcNumberOfBytesForData(rA.bias->grad->quantization, numBias)) == 0;
    bool propLossANonDegenerate = false;
    for (size_t i = 0; i < numPropLoss; i++) {
        if (((float *)propLossA->data)[i] != 0.0f) {
            propLossANonDegenerate = true;
        }
    }

    TEST_ASSERT_TRUE_MESSAGE(
        weightGradsIdentical,
        "weight grads must be byte-identical between the real-propLoss and NULL-propLoss twins");
    TEST_ASSERT_TRUE_MESSAGE(
        biasGradsIdentical,
        "bias grads must be byte-identical between the real-propLoss and NULL-propLoss twins");
    TEST_ASSERT_TRUE_MESSAGE(propLossANonDegenerate,
                             "twin A's propLoss must be non-degenerate (nonzero), proving the "
                             "NULL round only skipped dx");
}

/* ---- Group-quant PR3 (Task 2): ConvT1d forward with a grouped SYM weight.
 *
 * PR2's deny-pin (testConvT1dForwardGroupedWeightsFailFast, the death test
 * these gold tests REPLACE) is deliberately INVERTED here: grouped ConvT1d
 * forward is now the shipped behavior, routed through the grouped SCATTER
 * core convTranspose1dKernelSymInt32Grouped (per-PRODUCT rescale into
 * s_acc = inScale*max_g(scales[g]) -- a scatter has no per-(target, group)
 * run across which a raw partial could be carried, unlike the gather cores'
 * running-partial idiom).
 *
 * Fixture: Cin=2, Cout=3, K=3, Lin=4, B=1, VALID, stride=1, dilation=1,
 * outputPadding=0 (Lout=6), int12 codes
 * (generate_expected_convT1d_grouped.py). Both grouped-shape fixtures share
 * the SAME weight/input/bias mantissas -- only the group SHAPE differs.
 * NOTE on "per-channel" in ConvT1d's [Cin, Cout, K] storage: contiguous
 * groups of Cout*K elements span one INPUT-channel slab, so the perChannel
 * fixture means per-INPUT-channel groups (a per-OUTPUT-channel grouping is
 * not expressible as contiguous storage groups in this layout, unlike
 * Conv1d's [Cout, Cin, K]). */

/* Builds a SYM_INT32 (HALF_AWAY, int12) tensor with EXPLICIT int32 mantissas
 * + scale -- no absmax requantization, unlike buildSymTensor (which derives
 * both from a float source via tensorFillFromFloatBuffer). Needed here so
 * the fixture lands on EXACTLY the gold's mantissas (mirrors
 * UnitTestConv1d.c's helper of the same name). */
static tensor_t *buildSymInt32TensorExact(size_t numDims, const size_t *dimsIn,
                                          const int32_t *mantissas, float scale) {
    size_t *dims = reserveMemory(numDims * sizeof(size_t));
    for (size_t i = 0; i < numDims; i++) {
        dims[i] = dimsIn[i];
    }
    size_t *order = reserveMemory(numDims * sizeof(size_t));
    setOrderOfDimsForNewTensor(numDims, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, numDims, order);
    tensor_t *t = initTensor(shape, quantizationInitSymInt32(HALF_AWAY), NULL);
    size_t n = calcNumberOfElementsByShape(shape);
    for (size_t i = 0; i < n; i++) {
        ((int32_t *)t->data)[i] = mantissas[i];
    }
    ((symInt32QConfig_t *)t->quantization->qConfig)->scale = scale;
    return t;
}

/*! Builds the shared grouped-SYM weight/bias/input ConvT1d layer (borrowed,
 *  no grad buffers -- grouped grads are a future axis, #300). No borrowed
 *  builder exists for ConvT1d, so the layer is hand-wired into the caller's
 *  fixture struct (cfg/lc/layer must outlive the forward call). `numGroups`/
 *  `groupSize`/`wScales` vary per fixture (perChannel vs general); the
 *  weight/bias/input mantissas are the SAME shared fixture data. */
typedef struct convTGroupedFixture {
    layer_t layer;
    layerConfig_t lc;
    conv1dTransposedConfig_t cfg;
    kernel_t kernel;
    tensor_t *input;
} convTGroupedFixture_t;

static void buildGroupedConvTFixture(convTGroupedFixture_t *f, quantization_t *q, size_t numGroups,
                                     size_t groupSize, const float *wScales) {
    size_t weightDims[] = {(size_t)kConvTGroupedInChannels, (size_t)kConvTGroupedOutChannels,
                           (size_t)kConvTGroupedKernelSize};
    size_t *ownedWeightDims = reserveMemory(3 * sizeof(size_t));
    memcpy(ownedWeightDims, weightDims, sizeof(weightDims));
    size_t *weightOrder = reserveMemory(3 * sizeof(size_t));
    setOrderOfDimsForNewTensor(3, weightOrder);
    shape_t *weightShape = reserveMemory(sizeof(shape_t));
    setShape(weightShape, ownedWeightDims, 3, weightOrder);
    tensor_t *weightsParam = initTensor(
        weightShape, quantizationInitSymGrouped(12, HALF_AWAY, numGroups, groupSize), NULL);
    size_t numWeightElems = (size_t)kConvTGroupedInChannels * (size_t)kConvTGroupedOutChannels *
                            (size_t)kConvTGroupedKernelSize;
    byteConversion((uint8_t *)kConvTGroupedWMantissas, 32, weightsParam->data, 12, numWeightElems);
    symQConfig_t *weightQC = weightsParam->quantization->qConfig;
    for (size_t g = 0; g < numGroups; g++) {
        weightQC->scales[g] = wScales[g];
    }
    parameter_t *weights = parameterInit(weightsParam, NULL);

    size_t biasDims[] = {(size_t)kConvTGroupedOutChannels};
    tensor_t *biasParam =
        buildSymInt32TensorExact(1, biasDims, kConvTGroupedBiasMantissas, kConvTGroupedBiasScale);
    parameter_t *bias = parameterInit(biasParam, NULL);

    size_t inputDims[] = {(size_t)kConvTGroupedBatch, (size_t)kConvTGroupedInChannels,
                          (size_t)kConvTGroupedInputLength};
    f->input = buildSymInt32TensorExact(3, inputDims, kConvTGroupedXMantissas, kConvTGroupedXScale);

    initKernel(&f->kernel, (size_t)kConvTGroupedKernelSize, VALID, 1, 1);
    initConv1dTransposedConfigWithWeightsAndBias(&f->cfg, &f->kernel, weights, bias, 1, 0, q, q, q,
                                                 q);
    f->layer.type = CONV1D_TRANSPOSED;
    f->lc.conv1dTransposed = &f->cfg;
    f->layer.config = &f->lc;
}

/* Compares against the RAW scatter-core gold EXACTLY, not with a loose
 * tolerance: the output tensor here is FLOAT32 while forwardMath stays
 * SYM_INT32 (grouped weight), so conv1dTransposedForward's executeOp
 * epilogue takes the SYM_INT32->FLOAT32 conversionMatrix cell -- a single
 * EXACT `(float)mantissa * scale` per element, NOT the SYM_INT32 diagonal's
 * absmax-derived fresh-scale requant. That requant is deliberately dodged:
 * dequantizing approximately preserves the represented real value REGARDLESS
 * of which internal scale the kernel's rescales used, so a tolerance-based
 * comparison cannot reliably distinguish a correct s_acc from a
 * WRONG-but-still-sound one (e.g. scales[0] instead of max) -- the same
 * blind spot UnitTestConv1d.c's grouped tests documented empirically. The
 * exact FLOAT32 wire is bit-for-bit the deterministic float32 formula
 * generate_expected_convT1d_grouped.py's convT1d_grouped_ref computes
 * (python-int products, rescale_f32 == rescaleIntoAccumulatorScale
 * (HALF_AWAY) bit-for-bit), so ANY divergence in the kernel's internal
 * arithmetic (wrong s_acc, wrong per-product group, sum-then-rescale-once,
 * wrong rounding) changes the compared value measurably. Still routes
 * through conv1dTransposedForward -> executeOp's grouped-operand gate, so
 * the layer's groupedSymOperandPos wiring is under test too. */
void testConvT1dForwardGroupedPerChannelMatchesGold(void) {
    quantization_t *testQ = quantizationInitSymInt32(HALF_AWAY);
    convTGroupedFixture_t f;
    buildGroupedConvTFixture(&f, testQ, (size_t)kConvTPerChannelNumGroups,
                             (size_t)kConvTPerChannelGroupSize, kConvTPerChannelWScales);

    size_t outputDims[] = {(size_t)kConvTGroupedBatch, (size_t)kConvTGroupedOutChannels,
                           (size_t)kConvTGroupedOutLen};
    tensor_t *output = makeFloatTensor(outputDims, 3, NULL);

    conv1dTransposedForward(&f.layer, f.input, output);

    float *captured = (float *)output->data;
    for (size_t i = 0; i < kConvTPerChannelOutMantissas_len; i++) {
        float expected = (float)kConvTPerChannelOutMantissas[i] * kConvTPerChannelOutScale;
        TEST_ASSERT_EQUAL_FLOAT(expected, captured[i]);
    }
}

void testConvT1dForwardGroupedGeneralMatchesGold(void) {
    quantization_t *testQ = quantizationInitSymInt32(HALF_AWAY);
    convTGroupedFixture_t f;
    buildGroupedConvTFixture(&f, testQ, (size_t)kConvTGeneralNumGroups,
                             (size_t)kConvTGeneralGroupSize, kConvTGeneralWScales);

    size_t outputDims[] = {(size_t)kConvTGroupedBatch, (size_t)kConvTGroupedOutChannels,
                           (size_t)kConvTGroupedOutLen};
    tensor_t *output = makeFloatTensor(outputDims, 3, NULL);

    conv1dTransposedForward(&f.layer, f.input, output);

    float *captured = (float *)output->data;
    for (size_t i = 0; i < kConvTGeneralOutMantissas_len; i++) {
        float expected = (float)kConvTGeneralOutMantissas[i] * kConvTGeneralOutScale;
        TEST_ASSERT_EQUAL_FLOAT(expected, captured[i]);
    }
}

/* Equal-scales grouped twin: every group's scale is the SAME power-of-two
 * value (0.25f) and inScale (0.5f, kConvTGroupedXScale) is also a power of
 * two, so sAcc = inScale*maxScale and every product's paramScale
 * (inScale*scales[g]) are BIT-IDENTICAL float32 values -- and multiplying a
 * float by a power-of-two scale then dividing by the SAME value is an EXACT
 * round trip (pure exponent shifts; the int products, <= 60*40 = 2400, are
 * exactly representable), so round_half_away(product * paramScale / sAcc)
 * reproduces `product` exactly, per product. The grouped kernel's raw output
 * must therefore be BIT-IDENTICAL to the scalar (per-tensor SYM_INT32)
 * convTranspose1dKernelSymInt32 path on the SAME mantissas with weight scale
 * == the common group scale -- both wires here are FLOAT32 (same exact-
 * dequant reasoning as the gold tests above), so identical raw (mantissa,
 * scale) pairs dequantize to identical floats. Asserted by the generator's
 * self-check (v) on the emulation side too. */
void testConvT1dForwardGroupedEqualScalesBitIdenticalToScalar(void) {
    const float commonScale = 0.25f;
    quantization_t *testQ = quantizationInitSymInt32(HALF_AWAY);

    float groupScales[2] = {commonScale, commonScale};
    convTGroupedFixture_t grouped;
    buildGroupedConvTFixture(&grouped, testQ, 2, 9, groupScales);

    size_t outputDims[] = {(size_t)kConvTGroupedBatch, (size_t)kConvTGroupedOutChannels,
                           (size_t)kConvTGroupedOutLen};
    tensor_t *groupedOutput = makeFloatTensor(outputDims, 3, NULL);
    conv1dTransposedForward(&grouped.layer, grouped.input, groupedOutput);

    size_t weightDims[] = {(size_t)kConvTGroupedInChannels, (size_t)kConvTGroupedOutChannels,
                           (size_t)kConvTGroupedKernelSize};
    tensor_t *scalarWeightParam =
        buildSymInt32TensorExact(3, weightDims, kConvTGroupedWMantissas, commonScale);
    parameter_t *scalarWeights = parameterInit(scalarWeightParam, NULL);

    size_t biasDims[] = {(size_t)kConvTGroupedOutChannels};
    tensor_t *scalarBiasParam =
        buildSymInt32TensorExact(1, biasDims, kConvTGroupedBiasMantissas, kConvTGroupedBiasScale);
    parameter_t *scalarBias = parameterInit(scalarBiasParam, NULL);

    size_t inputDims[] = {(size_t)kConvTGroupedBatch, (size_t)kConvTGroupedInChannels,
                          (size_t)kConvTGroupedInputLength};
    tensor_t *scalarInput =
        buildSymInt32TensorExact(3, inputDims, kConvTGroupedXMantissas, kConvTGroupedXScale);

    kernel_t scalarKernel;
    initKernel(&scalarKernel, (size_t)kConvTGroupedKernelSize, VALID, 1, 1);
    conv1dTransposedConfig_t scalarCfg;
    static layerConfig_t scalarLc;
    static layer_t scalarLayer;
    initConv1dTransposedConfigWithWeightsAndBias(&scalarCfg, &scalarKernel, scalarWeights,
                                                 scalarBias, 1, 0, testQ, testQ, testQ, testQ);
    scalarLayer.type = CONV1D_TRANSPOSED;
    scalarLc.conv1dTransposed = &scalarCfg;
    scalarLayer.config = &scalarLc;

    tensor_t *scalarOutput = makeFloatTensor(outputDims, 3, NULL);
    conv1dTransposedForward(&scalarLayer, scalarInput, scalarOutput);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY((float *)scalarOutput->data, (float *)groupedOutput->data,
                                  kConvTPerChannelOutMantissas_len);
}

/* FLOAT32-math arm with the SAME grouped-SYM/SYM_INT32 storage (only
 * forwardMath differs): exercises the funnel's group-aware FLOAT32 dequant
 * (convertSymTensorToFloat32Tensor) gated by the FLOAT32-arm
 * groupedSymOperandPos declaration -- the arm-parity lesson from the PR2
 * final review's Conv1d asymmetry (Fix 3b): omitting the declaration on the
 * FLOAT32 arm makes this test die with the funnel's deny message.
 *
 * Tolerance derivation (per-product rescale error model, |err| <=
 * 0.5*C*s_acc): the float path is (near-)exact real arithmetic, while the
 * SYM gold rounds ONCE PER CONTRIBUTING PRODUCT (scatter core -- unlike the
 * gather cores' once-per-group-run). C for this fixture's geometry (Lin=4,
 * K=3, stride=1, dilation=1, Lout=6; window clipping counts): valid taps per
 * (oc, outIdx) are k in [outIdx-3, outIdx] ∩ [0, 2], i.e. 1,2,3,3,2,1 for
 * outIdx 0..5, times Cin=2 => C = 2,4,6,6,4,2; C_max = Cin*K = 6
 * (= kConvTGroupedMaxProductsPerOut, generator-asserted). Each product's
 * HALF_AWAY rescale rounds by at most 0.5 quanta of s_acc; the bias seed
 * adds ONE more rounding <= 0.5. Bound: 0.5*(C_max+1)*s_acc = 3.5*s_acc,
 * plus 1e-6f headroom for float32 arithmetic noise. */
void testConvT1dForwardGroupedFloatPathAgreesWithinTolerance(void) {
    quantization_t *floatQ = quantizationInitFloat();
    convTGroupedFixture_t f;
    buildGroupedConvTFixture(&f, floatQ, (size_t)kConvTPerChannelNumGroups,
                             (size_t)kConvTPerChannelGroupSize, kConvTPerChannelWScales);

    size_t outputDims[] = {(size_t)kConvTGroupedBatch, (size_t)kConvTGroupedOutChannels,
                           (size_t)kConvTGroupedOutLen};
    tensor_t *output = makeFloatTensor(outputDims, 3, NULL);

    conv1dTransposedForward(&f.layer, f.input, output);

    float *captured = (float *)output->data;
    const float tolerance =
        0.5f * (float)(kConvTGroupedMaxProductsPerOut + 1) * kConvTPerChannelOutScale + 1e-6f;
    for (size_t i = 0; i < kConvTPerChannelOutMantissas_len; i++) {
        float expected = (float)kConvTPerChannelOutMantissas[i] * kConvTPerChannelOutScale;
        TEST_ASSERT_FLOAT_WITHIN(tolerance, expected, captured[i]);
    }
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
    RUN_TEST(testConv1dTransposedFactoryFrozenElidesGrads);
    RUN_TEST(testConv1dTransposedFactoryDefaultAllocatesGrads);
    RUN_TEST(testConv1dTransposedBackwardFrozenTwinPropLossIdenticalGradsZero);
    RUN_TEST(testConv1dTransposedBackwardFrozenFactoryLayerRunsWithoutGradBuffers);
    RUN_TEST(testConv1dTransposedBackwardNullPropLossComputesGradsOnly);
    RUN_TEST(testConvT1dForwardGroupedPerChannelMatchesGold);
    RUN_TEST(testConvT1dForwardGroupedGeneralMatchesGold);
    RUN_TEST(testConvT1dForwardGroupedEqualScalesBitIdenticalToScalar);
    RUN_TEST(testConvT1dForwardGroupedFloatPathAgreesWithinTolerance);
    return UNITY_END();
}
