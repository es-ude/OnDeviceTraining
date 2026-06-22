#include <stdlib.h>

#include "Conv1d.h"
#include "Conv1dApi.h"
#include "ConvTranspose1dKernel.h"
#include "DeathTest.h"
#include "Layer.h"
#include "QuantizationApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "expected_conv1d.h"
#include "unity.h"

typedef struct conv1dFixtureSetup {
    // Pointer-to-caller-stack dims: tensorInitFloat stores a raw pointer, not a copy.
    // The pointed-to arrays must outlive the tensor — i.e. must live on the test
    // function's stack, not on conv1dRunForward's stack.
    size_t const *weightDims; // length 3
    size_t const *biasDims;   // length 1 (or NULL when hasBias==0)
    size_t const *inputDims;  // length 3
    size_t const *outputDims; // length 3
    int hasBias;
    size_t kSize;
    paddingType_t padding;
    size_t paddingAmount; // used only when padding == EXPLICIT
    size_t dilation;
    size_t stride;
    size_t groups;
    float const *weightData;
    float const *biasData;
    float const *inputData;
} conv1dFixtureSetup_t;

typedef struct conv1dRunResult {
    parameter_t *weights;
    parameter_t *bias;
    layer_t *layer;
    tensor_t *input;
    tensor_t *output;
    quantization_t *q;
} conv1dRunResult_t;

// Builds the layer (using direct initConv1dConfigWithWeightsAndBias when groups != 1
// — bypassing the UserAPI that hardcodes groups=1) and runs forward.
// Caller owns output buffer.
static conv1dRunResult_t conv1dRunForward(conv1dFixtureSetup_t s, float *outputBuf) {
    // Non-reentrant: function-local static storage for kernel/config/layer is
    // overwritten on each call. Safe for Unity tests, which execute serially
    // with no concurrent calls. If a future test invokes this helper twice
    // and tries to use both layers concurrently, the second call will silently
    // clobber the first.
    conv1dRunResult_t r = {0};

    tensor_t *weightParam = tensorInitFloat((float *)s.weightData, (size_t *)s.weightDims, 3, NULL);
    tensor_t *weightGrad = gradInitFloat(weightParam, NULL);
    r.weights = parameterInit(weightParam, weightGrad);

    if (s.hasBias) {
        tensor_t *biasParam = tensorInitFloat((float *)s.biasData, (size_t *)s.biasDims, 1, NULL);
        tensor_t *biasGrad = gradInitFloat(biasParam, NULL);
        r.bias = parameterInit(biasParam, biasGrad);
    } else {
        r.bias = NULL;
    }

    // kernelStore is static so its address remains valid after this function returns;
    // conv1dLayerInit / initConv1dConfigWithWeightsAndBias both store the kernel pointer.
    static kernel_t kernelStore;
    if (s.padding == EXPLICIT) {
        initKernelExplicit(&kernelStore, s.kSize, s.paddingAmount, s.dilation, s.stride);
    } else {
        initKernel(&kernelStore, s.kSize, s.padding, s.dilation, s.stride);
    }
    r.q = quantizationInitFloat();

    if (s.groups == 1) {
        r.layer = conv1dLayerInitLegacy(r.weights, r.bias, &kernelStore, r.q, r.q, r.q, r.q);
    } else {
        // Phase-2 will expose groups via UserAPI; here we go around the UserAPI.
        // All statics so their addresses remain valid after this function returns.
        static conv1dConfig_t cfg;
        static layerConfig_t lc;
        static layer_t l;
        initConv1dConfigWithWeightsAndBias(&cfg, &kernelStore, r.weights, r.bias, s.groups, r.q,
                                           r.q, r.q, r.q);
        l.type = CONV1D;
        lc.conv1d = &cfg;
        l.config = &lc;
        r.layer = &l;
    }

    r.input = tensorInitFloat((float *)s.inputData, (size_t *)s.inputDims, 3, NULL);
    r.output = tensorInitFloat(outputBuf, (size_t *)s.outputDims, 3, NULL);
    conv1dForward(r.layer, r.input, r.output);

    return r;
}

/* Build a SYM_INT32 (HALF_AWAY, qMaxBits=16) tensor from a float fixture: values
 * are quantized via tensorFillFromFloatBuffer (absmax->scale, round-clamp). The
 * fixtures are dequant-round-trip-stable (sym_gold.stable_dequant) so the C side
 * lands on exactly the gold mantissas+scale. NULL vals -> zero mantissas, scale 1.0. */
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

void testConv1dForwardMultiChannelWithBias() {
    size_t weightDims[] = {2, 3, 3};
    tensor_t *weightParam =
        tensorInitFloat((float *)weight_conv1d_multiChannelWithBias, weightDims, 3, NULL);
    tensor_t *weightGrad = gradInitFloat(weightParam, NULL);
    parameter_t *weights = parameterInit(weightParam, weightGrad);

    size_t biasDims[] = {2};
    tensor_t *biasParam =
        tensorInitFloat((float *)bias_conv1d_multiChannelWithBias, biasDims, 1, NULL);
    tensor_t *biasGrad = gradInitFloat(biasParam, NULL);
    parameter_t *bias = parameterInit(biasParam, biasGrad);

    kernel_t kernel;
    initKernel(&kernel, 3, VALID, 1, 1);
    quantization_t *q = quantizationInitFloat();
    layer_t *conv1d = conv1dLayerInitLegacy(weights, bias, &kernel, q, q, q, q);

    size_t inputDims[] = {1, 3, 5};
    tensor_t *input =
        tensorInitFloat((float *)input_conv1d_multiChannelWithBias, inputDims, 3, NULL);

    float outputData[1 * 2 * 3] = {0};
    size_t outputDims[] = {1, 2, 3};
    tensor_t *output = tensorInitFloat(outputData, outputDims, 3, NULL);

    conv1dForward(conv1d, input, output);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expectedForward_conv1d_multiChannelWithBias, output->data,
                                  expectedForward_conv1d_multiChannelWithBias_len);
}

void testConv1dForwardSingleChannelSingleBatch() {
    size_t weightDims[] = {1, 1, 2};
    tensor_t *weightParam =
        tensorInitFloat((float *)weight_conv1d_singleChannelSingleBatch, weightDims, 3, NULL);
    tensor_t *weightGrad = gradInitFloat(weightParam, NULL);
    parameter_t *weights = parameterInit(weightParam, weightGrad);

    kernel_t kernel;
    initKernel(&kernel, 2, VALID, 1, 1);

    quantization_t *q = quantizationInitFloat();
    layer_t *conv1d = conv1dLayerInitLegacy(weights, NULL, &kernel, q, q, q, q);

    size_t inputDims[] = {1, 1, 4};
    tensor_t *input =
        tensorInitFloat((float *)input_conv1d_singleChannelSingleBatch, inputDims, 3, NULL);

    float outputData[3] = {0};
    size_t outputDims[] = {1, 1, 3};
    tensor_t *output = tensorInitFloat(outputData, outputDims, 3, NULL);

    conv1dForward(conv1d, input, output);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expectedForward_conv1d_singleChannelSingleBatch, output->data,
                                  expectedForward_conv1d_singleChannelSingleBatch_len);
}

void testConv1dBackwardSingleChannelWithBias() {
    size_t weightDims[] = {1, 1, 2};
    tensor_t *weightParam =
        tensorInitFloat((float *)weight_conv1d_singleChannelWithBias, weightDims, 3, NULL);
    float weightGradData[2] = {0};
    tensor_t *weightGrad = tensorInitFloat(weightGradData, weightDims, 3, NULL);
    parameter_t *weights = parameterInit(weightParam, weightGrad);

    size_t biasDims[] = {1};
    tensor_t *biasParam =
        tensorInitFloat((float *)bias_conv1d_singleChannelWithBias, biasDims, 1, NULL);
    float biasGradData[1] = {0};
    tensor_t *biasGrad = tensorInitFloat(biasGradData, biasDims, 1, NULL);
    parameter_t *bias = parameterInit(biasParam, biasGrad);

    kernel_t kernel;
    initKernel(&kernel, 2, VALID, 1, 1);
    quantization_t *q = quantizationInitFloat();
    layer_t *conv1d = conv1dLayerInitLegacy(weights, bias, &kernel, q, q, q, q);

    size_t inputDims[] = {1, 1, 4};
    tensor_t *input =
        tensorInitFloat((float *)input_conv1d_singleChannelWithBias, inputDims, 3, NULL);

    // forward (sanity — also fills output)
    float outputData[3] = {0};
    size_t outputDims[] = {1, 1, 3};
    tensor_t *output = tensorInitFloat(outputData, outputDims, 3, NULL);
    conv1dForward(conv1d, input, output);

    // lossGrad = ones (matches what the generator used for autograd)
    float lossGradData[3];
    for (size_t i = 0; i < 3; i++) {
        lossGradData[i] = 1.0f;
    }
    tensor_t *lossGrad = tensorInitFloat(lossGradData, outputDims, 3, NULL);

    // propLoss buffer caller-owned, pre-zeroed
    float propLossData[4] = {0};
    tensor_t *propLoss = tensorInitFloat(propLossData, inputDims, 3, NULL);

    conv1dBackward(conv1d, input, lossGrad, propLoss);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expectedPropLoss_conv1d_singleChannelWithBias, propLoss->data,
                                  expectedPropLoss_conv1d_singleChannelWithBias_len);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expectedWeightGrad_conv1d_singleChannelWithBias,
                                  weights->grad->data,
                                  expectedWeightGrad_conv1d_singleChannelWithBias_len);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expectedBiasGrad_conv1d_singleChannelWithBias, bias->grad->data,
                                  expectedBiasGrad_conv1d_singleChannelWithBias_len);
}

void testConv1dBackwardSamePaddingSymmetric() {
    size_t weightDims[] = {1, 1, 3};
    tensor_t *weightParam =
        tensorInitFloat((float *)weight_conv1d_samePaddingSymmetric, weightDims, 3, NULL);
    float weightGradData[3] = {0};
    tensor_t *weightGrad = tensorInitFloat(weightGradData, weightDims, 3, NULL);
    parameter_t *weights = parameterInit(weightParam, weightGrad);

    kernel_t kernel;
    initKernel(&kernel, 3, SAME, 1, 1);
    quantization_t *q = quantizationInitFloat();
    layer_t *conv1d = conv1dLayerInitLegacy(weights, NULL, &kernel, q, q, q, q);

    size_t inputDims[] = {1, 1, 5};
    tensor_t *input =
        tensorInitFloat((float *)input_conv1d_samePaddingSymmetric, inputDims, 3, NULL);

    float outputData[5] = {0};
    size_t outputDims[] = {1, 1, 5};
    tensor_t *output = tensorInitFloat(outputData, outputDims, 3, NULL);
    conv1dForward(conv1d, input, output);

    float lossGradData[5];
    for (size_t i = 0; i < 5; i++) {
        lossGradData[i] = 1.0f;
    }
    tensor_t *lossGrad = tensorInitFloat(lossGradData, outputDims, 3, NULL);

    float propLossData[5] = {0};
    tensor_t *propLoss = tensorInitFloat(propLossData, inputDims, 3, NULL);

    conv1dBackward(conv1d, input, lossGrad, propLoss);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expectedPropLoss_conv1d_samePaddingSymmetric, propLoss->data,
                                  expectedPropLoss_conv1d_samePaddingSymmetric_len);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expectedWeightGrad_conv1d_samePaddingSymmetric,
                                  weights->grad->data,
                                  expectedWeightGrad_conv1d_samePaddingSymmetric_len);
}

void testConv1dForwardMultiBatch() {
    size_t weightDims[3] = {2, 2, 2};
    size_t inputDims[3] = {4, 2, 4};
    size_t outputDims[3] = {4, 2, 3};
    float outputData[4 * 2 * 3] = {0};
    conv1dFixtureSetup_t s = {
        .weightDims = weightDims,
        .biasDims = NULL,
        .inputDims = inputDims,
        .outputDims = outputDims,
        .hasBias = 0,
        .kSize = 2,
        .padding = VALID,
        .dilation = 1,
        .stride = 1,
        .groups = 1,
        .weightData = weight_conv1d_multiBatch,
        .biasData = NULL,
        .inputData = input_conv1d_multiBatch,
    };
    conv1dRunResult_t r = conv1dRunForward(s, outputData);

    TEST_ASSERT_EQUAL_size_t(expectedForward_conv1d_multiBatch_len, 4 * 2 * 3);
    for (size_t i = 0; i < expectedForward_conv1d_multiBatch_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expectedForward_conv1d_multiBatch[i],
                                 ((float *)r.output->data)[i]);
    }
}

void testConv1dForwardGroupsDepthwise() {
    size_t weightDims[3] = {4, 1, 2};
    size_t inputDims[3] = {1, 4, 5};
    size_t outputDims[3] = {1, 4, 4};
    float outputData[1 * 4 * 4] = {0};
    conv1dFixtureSetup_t s = {
        .weightDims = weightDims,
        .biasDims = NULL,
        .inputDims = inputDims,
        .outputDims = outputDims,
        .hasBias = 0,
        .kSize = 2,
        .padding = VALID,
        .dilation = 1,
        .stride = 1,
        .groups = 4,
        .weightData = weight_conv1d_groupsDepthwise,
        .biasData = NULL,
        .inputData = input_conv1d_groupsDepthwise,
    };
    conv1dRunResult_t r = conv1dRunForward(s, outputData);

    for (size_t i = 0; i < expectedForward_conv1d_groupsDepthwise_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expectedForward_conv1d_groupsDepthwise[i],
                                 ((float *)r.output->data)[i]);
    }
}

void testConv1dBackwardGroupsDepthwise() {
    size_t weightDims[3] = {4, 1, 2};
    size_t inputDims[3] = {1, 4, 5};
    size_t outputDims[3] = {1, 4, 4};
    float outputData[1 * 4 * 4] = {0};
    conv1dFixtureSetup_t s = {
        .weightDims = weightDims,
        .biasDims = NULL,
        .inputDims = inputDims,
        .outputDims = outputDims,
        .hasBias = 0,
        .kSize = 2,
        .padding = VALID,
        .dilation = 1,
        .stride = 1,
        .groups = 4,
        .weightData = weight_conv1d_groupsDepthwise,
        .biasData = NULL,
        .inputData = input_conv1d_groupsDepthwise,
    };
    conv1dRunResult_t r = conv1dRunForward(s, outputData);

    float lossGradData[1 * 4 * 4];
    for (size_t i = 0; i < 16; i++) {
        lossGradData[i] = 1.0f;
    }
    tensor_t *lossGrad = tensorInitFloat(lossGradData, outputDims, 3, NULL);

    float propLossData[1 * 4 * 5] = {0};
    tensor_t *propLoss = tensorInitFloat(propLossData, inputDims, 3, NULL);

    conv1dBackward(r.layer, r.input, lossGrad, propLoss);

    for (size_t i = 0; i < expectedPropLoss_conv1d_groupsDepthwise_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expectedPropLoss_conv1d_groupsDepthwise[i],
                                 ((float *)propLoss->data)[i]);
    }
    for (size_t i = 0; i < expectedWeightGrad_conv1d_groupsDepthwise_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expectedWeightGrad_conv1d_groupsDepthwise[i],
                                 ((float *)r.weights->grad->data)[i]);
    }
}

void testConv1dForwardGroupsGrouped() {
    size_t weightDims[3] = {8, 2, 2};
    size_t biasDims[1] = {8};
    size_t inputDims[3] = {1, 4, 5};
    size_t outputDims[3] = {1, 8, 4};
    float outputData[1 * 8 * 4] = {0};
    conv1dFixtureSetup_t s = {
        .weightDims = weightDims,
        .biasDims = biasDims,
        .inputDims = inputDims,
        .outputDims = outputDims,
        .hasBias = 1,
        .kSize = 2,
        .padding = VALID,
        .dilation = 1,
        .stride = 1,
        .groups = 2,
        .weightData = weight_conv1d_groupsGrouped,
        .biasData = bias_conv1d_groupsGrouped,
        .inputData = input_conv1d_groupsGrouped,
    };
    conv1dRunResult_t r = conv1dRunForward(s, outputData);

    for (size_t i = 0; i < expectedForward_conv1d_groupsGrouped_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expectedForward_conv1d_groupsGrouped[i],
                                 ((float *)r.output->data)[i]);
    }
}

void testConv1dBackwardGroupsGrouped() {
    size_t weightDims[3] = {8, 2, 2};
    size_t biasDims[1] = {8};
    size_t inputDims[3] = {1, 4, 5};
    size_t outputDims[3] = {1, 8, 4};
    float outputData[1 * 8 * 4] = {0};
    conv1dFixtureSetup_t s = {
        .weightDims = weightDims,
        .biasDims = biasDims,
        .inputDims = inputDims,
        .outputDims = outputDims,
        .hasBias = 1,
        .kSize = 2,
        .padding = VALID,
        .dilation = 1,
        .stride = 1,
        .groups = 2,
        .weightData = weight_conv1d_groupsGrouped,
        .biasData = bias_conv1d_groupsGrouped,
        .inputData = input_conv1d_groupsGrouped,
    };
    conv1dRunResult_t r = conv1dRunForward(s, outputData);

    float lossGradData[1 * 8 * 4];
    for (size_t i = 0; i < 32; i++) {
        lossGradData[i] = 1.0f;
    }
    tensor_t *lossGrad = tensorInitFloat(lossGradData, outputDims, 3, NULL);

    float propLossData[1 * 4 * 5] = {0};
    tensor_t *propLoss = tensorInitFloat(propLossData, inputDims, 3, NULL);

    conv1dBackward(r.layer, r.input, lossGrad, propLoss);

    for (size_t i = 0; i < expectedPropLoss_conv1d_groupsGrouped_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expectedPropLoss_conv1d_groupsGrouped[i],
                                 ((float *)propLoss->data)[i]);
    }
    for (size_t i = 0; i < expectedWeightGrad_conv1d_groupsGrouped_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expectedWeightGrad_conv1d_groupsGrouped[i],
                                 ((float *)r.weights->grad->data)[i]);
    }
    for (size_t i = 0; i < expectedBiasGrad_conv1d_groupsGrouped_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expectedBiasGrad_conv1d_groupsGrouped[i],
                                 ((float *)r.bias->grad->data)[i]);
    }
}

void testConv1dForwardStrideDilation() {
    size_t weightDims[3] = {1, 1, 2};
    size_t inputDims[3] = {1, 1, 9};
    size_t outputDims[3] = {1, 1, 3};
    float outputData[1 * 1 * 3] = {0};
    conv1dFixtureSetup_t s = {
        .weightDims = weightDims,
        .biasDims = NULL,
        .inputDims = inputDims,
        .outputDims = outputDims,
        .hasBias = 0,
        .kSize = 2,
        .padding = VALID,
        .dilation = 2,
        .stride = 3,
        .groups = 1,
        .weightData = weight_conv1d_strideDilation,
        .biasData = NULL,
        .inputData = input_conv1d_strideDilation,
    };
    conv1dRunResult_t r = conv1dRunForward(s, outputData);

    TEST_ASSERT_EQUAL_size_t(3u, expectedForward_conv1d_strideDilation_len);
    for (size_t i = 0; i < expectedForward_conv1d_strideDilation_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expectedForward_conv1d_strideDilation[i],
                                 ((float *)r.output->data)[i]);
    }
}

void testConv1dForwardSamePaddingAsymmetric() {
    size_t weightDims[3] = {1, 1, 4};
    size_t inputDims[3] = {1, 1, 5};
    size_t outputDims[3] = {1, 1, 5};
    float outputData[1 * 1 * 5] = {0};
    conv1dFixtureSetup_t s = {
        .weightDims = weightDims,
        .biasDims = NULL,
        .inputDims = inputDims,
        .outputDims = outputDims,
        .hasBias = 0,
        .kSize = 4,
        .padding = SAME,
        .dilation = 1,
        .stride = 1,
        .groups = 1,
        .weightData = weight_conv1d_samePaddingAsymmetric,
        .biasData = NULL,
        .inputData = input_conv1d_samePaddingAsymmetric,
    };
    conv1dRunResult_t r = conv1dRunForward(s, outputData);

    for (size_t i = 0; i < expectedForward_conv1d_samePaddingAsymmetric_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expectedForward_conv1d_samePaddingAsymmetric[i],
                                 ((float *)r.output->data)[i]);
    }
}

void testConv1dBackwardSamePaddingAsymmetric() {
    size_t weightDims[3] = {1, 1, 4};
    size_t inputDims[3] = {1, 1, 5};
    size_t outputDims[3] = {1, 1, 5};
    float outputData[1 * 1 * 5] = {0};
    conv1dFixtureSetup_t s = {
        .weightDims = weightDims,
        .biasDims = NULL,
        .inputDims = inputDims,
        .outputDims = outputDims,
        .hasBias = 0,
        .kSize = 4,
        .padding = SAME,
        .dilation = 1,
        .stride = 1,
        .groups = 1,
        .weightData = weight_conv1d_samePaddingAsymmetric,
        .biasData = NULL,
        .inputData = input_conv1d_samePaddingAsymmetric,
    };
    conv1dRunResult_t r = conv1dRunForward(s, outputData);

    float lossGradData[5];
    for (size_t i = 0; i < 5; i++) {
        lossGradData[i] = 1.0f;
    }
    tensor_t *lossGrad = tensorInitFloat(lossGradData, outputDims, 3, NULL);

    float propLossData[5] = {0};
    tensor_t *propLoss = tensorInitFloat(propLossData, inputDims, 3, NULL);

    conv1dBackward(r.layer, r.input, lossGrad, propLoss);

    for (size_t i = 0; i < expectedPropLoss_conv1d_samePaddingAsymmetric_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expectedPropLoss_conv1d_samePaddingAsymmetric[i],
                                 ((float *)propLoss->data)[i]);
    }
    for (size_t i = 0; i < expectedWeightGrad_conv1d_samePaddingAsymmetric_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expectedWeightGrad_conv1d_samePaddingAsymmetric[i],
                                 ((float *)r.weights->grad->data)[i]);
    }
}

void testConv1dForwardSamePaddingWithGroups() {
    size_t weightDims[3] = {4, 2, 3};
    size_t biasDims[1] = {4};
    size_t inputDims[3] = {2, 4, 6};
    size_t outputDims[3] = {2, 4, 6};
    float outputData[2 * 4 * 6] = {0};
    conv1dFixtureSetup_t s = {
        .weightDims = weightDims,
        .biasDims = biasDims,
        .inputDims = inputDims,
        .outputDims = outputDims,
        .hasBias = 1,
        .kSize = 3,
        .padding = SAME,
        .dilation = 1,
        .stride = 1,
        .groups = 2,
        .weightData = weight_conv1d_samePaddingWithGroups,
        .biasData = bias_conv1d_samePaddingWithGroups,
        .inputData = input_conv1d_samePaddingWithGroups,
    };
    conv1dRunResult_t r = conv1dRunForward(s, outputData);

    for (size_t i = 0; i < expectedForward_conv1d_samePaddingWithGroups_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expectedForward_conv1d_samePaddingWithGroups[i],
                                 ((float *)r.output->data)[i]);
    }
}

void testConv1dBackwardSamePaddingWithGroups() {
    size_t weightDims[3] = {4, 2, 3};
    size_t biasDims[1] = {4};
    size_t inputDims[3] = {2, 4, 6};
    size_t outputDims[3] = {2, 4, 6};
    float outputData[2 * 4 * 6] = {0};
    conv1dFixtureSetup_t s = {
        .weightDims = weightDims,
        .biasDims = biasDims,
        .inputDims = inputDims,
        .outputDims = outputDims,
        .hasBias = 1,
        .kSize = 3,
        .padding = SAME,
        .dilation = 1,
        .stride = 1,
        .groups = 2,
        .weightData = weight_conv1d_samePaddingWithGroups,
        .biasData = bias_conv1d_samePaddingWithGroups,
        .inputData = input_conv1d_samePaddingWithGroups,
    };
    conv1dRunResult_t r = conv1dRunForward(s, outputData);

    float lossGradData[2 * 4 * 6];
    for (size_t i = 0; i < 48; i++) {
        lossGradData[i] = 1.0f;
    }
    tensor_t *lossGrad = tensorInitFloat(lossGradData, outputDims, 3, NULL);

    float propLossData[2 * 4 * 6] = {0};
    tensor_t *propLoss = tensorInitFloat(propLossData, inputDims, 3, NULL);

    conv1dBackward(r.layer, r.input, lossGrad, propLoss);

    for (size_t i = 0; i < expectedPropLoss_conv1d_samePaddingWithGroups_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expectedPropLoss_conv1d_samePaddingWithGroups[i],
                                 ((float *)propLoss->data)[i]);
    }
    for (size_t i = 0; i < expectedWeightGrad_conv1d_samePaddingWithGroups_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expectedWeightGrad_conv1d_samePaddingWithGroups[i],
                                 ((float *)r.weights->grad->data)[i]);
    }
    for (size_t i = 0; i < expectedBiasGrad_conv1d_samePaddingWithGroups_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expectedBiasGrad_conv1d_samePaddingWithGroups[i],
                                 ((float *)r.bias->grad->data)[i]);
    }
}

void testConv1dForwardPointwise() {
    size_t weightDims[3] = {4, 3, 1};
    size_t biasDims[1] = {4};
    size_t inputDims[3] = {2, 3, 5};
    size_t outputDims[3] = {2, 4, 5};
    float outputData[2 * 4 * 5] = {0};
    conv1dFixtureSetup_t s = {
        .weightDims = weightDims,
        .biasDims = biasDims,
        .inputDims = inputDims,
        .outputDims = outputDims,
        .hasBias = 1,
        .kSize = 1,
        .padding = VALID,
        .dilation = 1,
        .stride = 1,
        .groups = 1,
        .weightData = weight_conv1d_pointwise,
        .biasData = bias_conv1d_pointwise,
        .inputData = input_conv1d_pointwise,
    };
    conv1dRunResult_t r = conv1dRunForward(s, outputData);

    for (size_t i = 0; i < expectedForward_conv1d_pointwise_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expectedForward_conv1d_pointwise[i],
                                 ((float *)r.output->data)[i]);
    }
}

void testConv1dBackwardPointwise() {
    size_t weightDims[3] = {4, 3, 1};
    size_t biasDims[1] = {4};
    size_t inputDims[3] = {2, 3, 5};
    size_t outputDims[3] = {2, 4, 5};
    float outputData[2 * 4 * 5] = {0};
    conv1dFixtureSetup_t s = {
        .weightDims = weightDims,
        .biasDims = biasDims,
        .inputDims = inputDims,
        .outputDims = outputDims,
        .hasBias = 1,
        .kSize = 1,
        .padding = VALID,
        .dilation = 1,
        .stride = 1,
        .groups = 1,
        .weightData = weight_conv1d_pointwise,
        .biasData = bias_conv1d_pointwise,
        .inputData = input_conv1d_pointwise,
    };
    conv1dRunResult_t r = conv1dRunForward(s, outputData);

    // Non-uniform lossGrad (from the generator), NOT all-ones: pins output-channel
    // dependence in the weight/bias/input gradients — the channel-mixing that defines
    // a pointwise (1x1) conv. See generate_expected_conv1d.py::fixture_pointwise.
    tensor_t *lossGrad = tensorInitFloat((float *)lossGrad_conv1d_pointwise, outputDims, 3, NULL);

    float propLossData[2 * 3 * 5] = {0};
    tensor_t *propLoss = tensorInitFloat(propLossData, inputDims, 3, NULL);

    conv1dBackward(r.layer, r.input, lossGrad, propLoss);

    for (size_t i = 0; i < expectedPropLoss_conv1d_pointwise_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expectedPropLoss_conv1d_pointwise[i],
                                 ((float *)propLoss->data)[i]);
    }
    for (size_t i = 0; i < expectedWeightGrad_conv1d_pointwise_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expectedWeightGrad_conv1d_pointwise[i],
                                 ((float *)r.weights->grad->data)[i]);
    }
    for (size_t i = 0; i < expectedBiasGrad_conv1d_pointwise_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expectedBiasGrad_conv1d_pointwise[i],
                                 ((float *)r.bias->grad->data)[i]);
    }
}

void testConv1dForwardExplicitPadding() {
    // ECG enc1 geometry (issue #177): K=7, stride=2, EXPLICIT symmetric padding=3.
    size_t weightDims[3] = {3, 2, 7};
    size_t biasDims[1] = {3};
    size_t inputDims[3] = {1, 2, 10};
    size_t outputDims[3] = {1, 3, 5}; // (10 + 2*3 - 7)/2 + 1 = 5
    float outputData[1 * 3 * 5] = {0};
    conv1dFixtureSetup_t s = {
        .weightDims = weightDims,
        .biasDims = biasDims,
        .inputDims = inputDims,
        .outputDims = outputDims,
        .hasBias = 1,
        .kSize = 7,
        .padding = EXPLICIT,
        .paddingAmount = 3,
        .dilation = 1,
        .stride = 2,
        .groups = 1,
        .weightData = weight_conv1d_explicitPadding,
        .biasData = bias_conv1d_explicitPadding,
        .inputData = input_conv1d_explicitPadding,
    };
    conv1dRunResult_t r = conv1dRunForward(s, outputData);

    for (size_t i = 0; i < expectedForward_conv1d_explicitPadding_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expectedForward_conv1d_explicitPadding[i],
                                 ((float *)r.output->data)[i]);
    }
}

void testConv1dBackwardExplicitPadding() {
    // Backward twin of the forward above. conv1dBackward delegates the input
    // gradient to the transposed-conv adjoint, which must also honour the
    // explicit pad — this is the regression guard for issue #177's training path.
    size_t weightDims[3] = {3, 2, 7};
    size_t biasDims[1] = {3};
    size_t inputDims[3] = {1, 2, 10};
    size_t outputDims[3] = {1, 3, 5};
    float outputData[1 * 3 * 5] = {0};
    conv1dFixtureSetup_t s = {
        .weightDims = weightDims,
        .biasDims = biasDims,
        .inputDims = inputDims,
        .outputDims = outputDims,
        .hasBias = 1,
        .kSize = 7,
        .padding = EXPLICIT,
        .paddingAmount = 3,
        .dilation = 1,
        .stride = 2,
        .groups = 1,
        .weightData = weight_conv1d_explicitPadding,
        .biasData = bias_conv1d_explicitPadding,
        .inputData = input_conv1d_explicitPadding,
    };
    conv1dRunResult_t r = conv1dRunForward(s, outputData);

    // Non-uniform lossGrad (from the generator), NOT all-ones — pins the output
    // channel in dL/dW (see generate_expected_conv1d.py::fixture_explicit_padding).
    tensor_t *lossGrad =
        tensorInitFloat((float *)lossGrad_conv1d_explicitPadding, outputDims, 3, NULL);

    float propLossData[1 * 2 * 10] = {0};
    tensor_t *propLoss = tensorInitFloat(propLossData, inputDims, 3, NULL);

    conv1dBackward(r.layer, r.input, lossGrad, propLoss);

    for (size_t i = 0; i < expectedPropLoss_conv1d_explicitPadding_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expectedPropLoss_conv1d_explicitPadding[i],
                                 ((float *)propLoss->data)[i]);
    }
    for (size_t i = 0; i < expectedWeightGrad_conv1d_explicitPadding_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expectedWeightGrad_conv1d_explicitPadding[i],
                                 ((float *)r.weights->grad->data)[i]);
    }
    for (size_t i = 0; i < expectedBiasGrad_conv1d_explicitPadding_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expectedBiasGrad_conv1d_explicitPadding[i],
                                 ((float *)r.bias->grad->data)[i]);
    }
}

void testConv1dForwardSymSingleChannelSingleBatch() {
    size_t weightDims[] = {1, 1, 2};
    size_t inputDims[] = {1, 1, 4};
    size_t outputDims[] = {1, 1, 3};

    parameter_t *weights = buildSymParam(3, weightDims, weight_conv1dSym_singleChannelSingleBatch);
    tensor_t *input = buildSymTensor(3, inputDims, input_conv1dSym_singleChannelSingleBatch);
    tensor_t *output = buildSymTensor(3, outputDims, NULL);

    kernel_t kernel;
    initKernel(&kernel, 2, VALID, 1, 1);
    quantization_t *sq = quantizationInitSymInt32(HALF_AWAY);
    layer_t *conv1d = conv1dLayerInitLegacy(weights, NULL, &kernel, sq, sq, sq, sq);

    conv1dForward(conv1d, input, output);

    int32_t *m = (int32_t *)output->data;
    for (size_t i = 0; i < expectedForward_conv1dSym_singleChannelSingleBatch_len; i++) {
        TEST_ASSERT_INT_WITHIN(forwardMantissaTol_conv1dSym_singleChannelSingleBatch,
                               expectedForward_conv1dSym_singleChannelSingleBatch[i], m[i]);
    }
    float scale = symScaleOf(output);
    TEST_ASSERT_FLOAT_WITHIN(expectedForwardScale_conv1dSym_singleChannelSingleBatch * 1e-4f,
                             expectedForwardScale_conv1dSym_singleChannelSingleBatch, scale);
    for (size_t i = 0; i < expectedForwardDequant_conv1dSym_singleChannelSingleBatch_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(forwardDequantTol_conv1dSym_singleChannelSingleBatch,
                                 expectedForwardDequant_conv1dSym_singleChannelSingleBatch[i],
                                 (float)m[i] * scale);
    }
}

void testConv1dForwardSymSingleChannelWithBias() {
    size_t weightDims[] = {1, 1, 2};
    size_t biasDims[] = {1};
    size_t inputDims[] = {1, 1, 4};
    size_t outputDims[] = {1, 1, 3};

    parameter_t *weights = buildSymParam(3, weightDims, weight_conv1dSym_singleChannelWithBias);
    parameter_t *bias = buildSymParam(1, biasDims, bias_conv1dSym_singleChannelWithBias);
    tensor_t *input = buildSymTensor(3, inputDims, input_conv1dSym_singleChannelWithBias);
    tensor_t *output = buildSymTensor(3, outputDims, NULL);

    kernel_t kernel;
    initKernel(&kernel, 2, VALID, 1, 1);
    quantization_t *sq = quantizationInitSymInt32(HALF_AWAY);
    layer_t *conv1d = conv1dLayerInitLegacy(weights, bias, &kernel, sq, sq, sq, sq);

    conv1dForward(conv1d, input, output);

    int32_t *m = (int32_t *)output->data;
    for (size_t i = 0; i < expectedForward_conv1dSym_singleChannelWithBias_len; i++) {
        TEST_ASSERT_INT_WITHIN(forwardMantissaTol_conv1dSym_singleChannelWithBias,
                               expectedForward_conv1dSym_singleChannelWithBias[i], m[i]);
    }
    float scale = symScaleOf(output);
    TEST_ASSERT_FLOAT_WITHIN(expectedForwardScale_conv1dSym_singleChannelWithBias * 1e-4f,
                             expectedForwardScale_conv1dSym_singleChannelWithBias, scale);
    for (size_t i = 0; i < expectedForwardDequant_conv1dSym_singleChannelWithBias_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(forwardDequantTol_conv1dSym_singleChannelWithBias,
                                 expectedForwardDequant_conv1dSym_singleChannelWithBias[i],
                                 (float)m[i] * scale);
    }
}

void testConv1dForwardSymPointwise() {
    size_t weightDims[] = {4, 3, 1};
    size_t biasDims[] = {4};
    size_t inputDims[] = {2, 3, 5};
    size_t outputDims[] = {2, 4, 5};

    parameter_t *weights = buildSymParam(3, weightDims, weight_conv1dSym_pointwise);
    parameter_t *bias = buildSymParam(1, biasDims, bias_conv1dSym_pointwise);
    tensor_t *input = buildSymTensor(3, inputDims, input_conv1dSym_pointwise);
    tensor_t *output = buildSymTensor(3, outputDims, NULL);

    kernel_t kernel;
    initKernel(&kernel, 1, VALID, 1, 1);
    quantization_t *sq = quantizationInitSymInt32(HALF_AWAY);
    layer_t *conv1d = conv1dLayerInitLegacy(weights, bias, &kernel, sq, sq, sq, sq);

    conv1dForward(conv1d, input, output);

    int32_t *m = (int32_t *)output->data;
    for (size_t i = 0; i < expectedForward_conv1dSym_pointwise_len; i++) {
        TEST_ASSERT_INT_WITHIN(forwardMantissaTol_conv1dSym_pointwise,
                               expectedForward_conv1dSym_pointwise[i], m[i]);
    }
    float scale = symScaleOf(output);
    TEST_ASSERT_FLOAT_WITHIN(expectedForwardScale_conv1dSym_pointwise * 1e-4f,
                             expectedForwardScale_conv1dSym_pointwise, scale);
    for (size_t i = 0; i < expectedForwardDequant_conv1dSym_pointwise_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(forwardDequantTol_conv1dSym_pointwise,
                                 expectedForwardDequant_conv1dSym_pointwise[i],
                                 (float)m[i] * scale);
    }
}

void testConv1dForwardSymExplicitPadding() {
    size_t weightDims[] = {3, 2, 7};
    size_t biasDims[] = {3};
    size_t inputDims[] = {1, 2, 10};
    size_t outputDims[] = {1, 3, 5};

    parameter_t *weights = buildSymParam(3, weightDims, weight_conv1dSym_explicitPadding);
    parameter_t *bias = buildSymParam(1, biasDims, bias_conv1dSym_explicitPadding);
    tensor_t *input = buildSymTensor(3, inputDims, input_conv1dSym_explicitPadding);
    tensor_t *output = buildSymTensor(3, outputDims, NULL);

    kernel_t kernel;
    initKernelExplicit(&kernel, 7, 3, 1, 2);
    quantization_t *sq = quantizationInitSymInt32(HALF_AWAY);
    layer_t *conv1d = conv1dLayerInitLegacy(weights, bias, &kernel, sq, sq, sq, sq);

    conv1dForward(conv1d, input, output);

    int32_t *m = (int32_t *)output->data;
    for (size_t i = 0; i < expectedForward_conv1dSym_explicitPadding_len; i++) {
        TEST_ASSERT_INT_WITHIN(forwardMantissaTol_conv1dSym_explicitPadding,
                               expectedForward_conv1dSym_explicitPadding[i], m[i]);
    }
    float scale = symScaleOf(output);
    TEST_ASSERT_FLOAT_WITHIN(expectedForwardScale_conv1dSym_explicitPadding * 1e-4f,
                             expectedForwardScale_conv1dSym_explicitPadding, scale);
    for (size_t i = 0; i < expectedForwardDequant_conv1dSym_explicitPadding_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(forwardDequantTol_conv1dSym_explicitPadding,
                                 expectedForwardDequant_conv1dSym_explicitPadding[i],
                                 (float)m[i] * scale);
    }
}

void testConv1dCalcWeightGradsSymGroupsGrouped() {
    size_t weightDims[] = {8, 2, 2};
    size_t biasDims[] = {8};
    size_t inputDims[] = {1, 4, 5};
    size_t lossDims[] = {1, 8, 4};

    parameter_t *weights = buildSymParam(3, weightDims, weight_conv1dSym_groupsGroupedSym);
    parameter_t *bias = buildSymParam(1, biasDims, bias_conv1dSym_groupsGroupedSym);
    tensor_t *input = buildSymTensor(3, inputDims, input_conv1dSym_groupsGroupedSym);
    tensor_t *lossGrad = buildSymTensor(3, lossDims, lossGrad_conv1dSym_groupsGroupedSym);

    kernel_t kernel;
    initKernel(&kernel, 2, VALID, 1, 1);
    conv1dConfig_t cfg;
    quantization_t *sq = quantizationInitSymInt32(HALF_AWAY);
    initConv1dConfigWithWeightsAndBias(&cfg, &kernel, weights, bias, 2, sq, sq, sq, sq);

    conv1dCalcWeightGradsSymInt32(&cfg, input, lossGrad);

    int32_t *m = (int32_t *)weights->grad->data;
    for (size_t i = 0; i < expectedWeightGrad_conv1dSym_groupsGroupedSym_len; i++) {
        TEST_ASSERT_INT_WITHIN(weightGradMantissaTol_conv1dSym_groupsGroupedSym,
                               expectedWeightGrad_conv1dSym_groupsGroupedSym[i], m[i]);
    }
    float scale = symScaleOf(weights->grad);
    TEST_ASSERT_FLOAT_WITHIN(expectedWeightGradScale_conv1dSym_groupsGroupedSym * 1e-4f,
                             expectedWeightGradScale_conv1dSym_groupsGroupedSym, scale);
    for (size_t i = 0; i < expectedWeightGradDequant_conv1dSym_groupsGroupedSym_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(weightGradDequantTol_conv1dSym_groupsGroupedSym,
                                 expectedWeightGradDequant_conv1dSym_groupsGroupedSym[i],
                                 (float)m[i] * scale);
    }
}

void testConv1dKernelSymScatterStrideDilation() {
    size_t weightDims[] = {1, 1, 2};
    size_t lossDims[] = {1, 1, 3};
    size_t propDims[] = {1, 1, 9};

    tensor_t *weight = buildSymTensor(3, weightDims, weight_conv1dSym_strideDilationSym);
    tensor_t *lossGrad = buildSymTensor(3, lossDims, lossGrad_conv1dSym_strideDilationSym);
    tensor_t *propLoss = buildSymTensor(3, propDims, NULL);

    kernel_t kernel;
    initKernel(&kernel, 2, VALID, 2, 3); /* size=2, VALID, dilation=2, stride=3 */

    convTranspose1dKernelSymInt32(lossGrad, weight, NULL, &kernel, 1, 0, propLoss);

    int32_t *m = (int32_t *)propLoss->data;
    for (size_t i = 0; i < expectedPropLoss_conv1dSym_strideDilationSym_len; i++) {
        TEST_ASSERT_INT_WITHIN(propLossMantissaTol_conv1dSym_strideDilationSym,
                               expectedPropLoss_conv1dSym_strideDilationSym[i], m[i]);
    }
    float scale = symScaleOf(propLoss);
    TEST_ASSERT_FLOAT_WITHIN(expectedPropLossScale_conv1dSym_strideDilationSym * 1e-4f,
                             expectedPropLossScale_conv1dSym_strideDilationSym, scale);
    for (size_t i = 0; i < expectedPropLossDequant_conv1dSym_strideDilationSym_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(propLossDequantTol_conv1dSym_strideDilationSym,
                                 expectedPropLossDequant_conv1dSym_strideDilationSym[i],
                                 (float)m[i] * scale);
    }
}

void testConv1dCalcBiasGradsSymPointwise() {
    size_t weightDims[] = {4, 3, 1};
    size_t biasDims[] = {4};
    size_t lossDims[] = {2, 4, 5};

    parameter_t *weights = buildSymParam(3, weightDims, weight_conv1dSym_pointwise);
    parameter_t *bias = buildSymParam(1, biasDims, bias_conv1dSym_pointwise);
    tensor_t *lossGrad = buildSymTensor(3, lossDims, lossGrad_conv1dSym_pointwise);

    kernel_t kernel;
    initKernel(&kernel, 1, VALID, 1, 1);
    conv1dConfig_t cfg;
    quantization_t *sq = quantizationInitSymInt32(HALF_AWAY);
    initConv1dConfigWithWeightsAndBias(&cfg, &kernel, weights, bias, 1, sq, sq, sq, sq);

    conv1dCalcBiasGradsSymInt32(&cfg, lossGrad);

    int32_t *m = (int32_t *)bias->grad->data;
    for (size_t i = 0; i < expectedBiasGrad_conv1dSym_pointwise_len; i++) {
        TEST_ASSERT_INT_WITHIN(biasGradMantissaTol_conv1dSym_pointwise,
                               expectedBiasGrad_conv1dSym_pointwise[i], m[i]);
    }
    float scale = symScaleOf(bias->grad);
    TEST_ASSERT_FLOAT_WITHIN(expectedBiasGradScale_conv1dSym_pointwise * 1e-4f,
                             expectedBiasGradScale_conv1dSym_pointwise, scale);
    for (size_t i = 0; i < expectedBiasGradDequant_conv1dSym_pointwise_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(biasGradDequantTol_conv1dSym_pointwise,
                                 expectedBiasGradDequant_conv1dSym_pointwise[i],
                                 (float)m[i] * scale);
    }
}

void testConv1dBackwardSymExplicitPadding() {
    size_t weightDims[] = {3, 2, 7};
    size_t biasDims[] = {3};
    size_t inputDims[] = {1, 2, 10};
    size_t outputDims[] = {1, 3, 5};

    parameter_t *weights = buildSymParam(3, weightDims, weight_conv1dSym_explicitPadding);
    parameter_t *bias = buildSymParam(1, biasDims, bias_conv1dSym_explicitPadding);
    tensor_t *input = buildSymTensor(3, inputDims, input_conv1dSym_explicitPadding);
    tensor_t *lossGrad = buildSymTensor(3, outputDims, lossGrad_conv1dSym_explicitPadding);
    tensor_t *propLoss = buildSymTensor(3, inputDims, NULL);

    kernel_t kernel;
    initKernelExplicit(&kernel, 7, 3, 1, 2); /* K=7, pad=3, dilation=1, stride=2 */
    conv1dConfig_t cfg;
    quantization_t *sq = quantizationInitSymInt32(HALF_AWAY);
    static layerConfig_t lc;
    static layer_t layer;
    initConv1dConfigWithWeightsAndBias(&cfg, &kernel, weights, bias, 1, sq, sq, sq, sq);
    layer.type = CONV1D;
    lc.conv1d = &cfg;
    layer.config = &lc;

    conv1dBackward(&layer, input, lossGrad, propLoss);

    /* propLoss (dx) */
    int32_t *dx = (int32_t *)propLoss->data;
    for (size_t i = 0; i < expectedPropLoss_conv1dSym_explicitPadding_len; i++) {
        TEST_ASSERT_INT_WITHIN(propLossMantissaTol_conv1dSym_explicitPadding,
                               expectedPropLoss_conv1dSym_explicitPadding[i], dx[i]);
    }
    float dxScale = symScaleOf(propLoss);
    TEST_ASSERT_FLOAT_WITHIN(expectedPropLossScale_conv1dSym_explicitPadding * 1e-4f,
                             expectedPropLossScale_conv1dSym_explicitPadding, dxScale);
    for (size_t i = 0; i < expectedPropLossDequant_conv1dSym_explicitPadding_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(propLossDequantTol_conv1dSym_explicitPadding,
                                 expectedPropLossDequant_conv1dSym_explicitPadding[i],
                                 (float)dx[i] * dxScale);
    }
    /* weightGrad */
    int32_t *dw = (int32_t *)weights->grad->data;
    float dwScale = symScaleOf(weights->grad);
    for (size_t i = 0; i < expectedWeightGrad_conv1dSym_explicitPadding_len; i++) {
        TEST_ASSERT_INT_WITHIN(weightGradMantissaTol_conv1dSym_explicitPadding,
                               expectedWeightGrad_conv1dSym_explicitPadding[i], dw[i]);
        TEST_ASSERT_FLOAT_WITHIN(weightGradDequantTol_conv1dSym_explicitPadding,
                                 expectedWeightGradDequant_conv1dSym_explicitPadding[i],
                                 (float)dw[i] * dwScale);
    }
    /* biasGrad */
    int32_t *db = (int32_t *)bias->grad->data;
    float dbScale = symScaleOf(bias->grad);
    for (size_t i = 0; i < expectedBiasGrad_conv1dSym_explicitPadding_len; i++) {
        TEST_ASSERT_INT_WITHIN(biasGradMantissaTol_conv1dSym_explicitPadding,
                               expectedBiasGrad_conv1dSym_explicitPadding[i], db[i]);
        TEST_ASSERT_FLOAT_WITHIN(biasGradDequantTol_conv1dSym_explicitPadding,
                                 expectedBiasGradDequant_conv1dSym_explicitPadding[i],
                                 (float)db[i] * dbScale);
    }
}

void testConv1dBackwardSymGroupsGrouped() {
    size_t weightDims[] = {8, 2, 2};
    size_t biasDims[] = {8};
    size_t inputDims[] = {1, 4, 5};
    size_t outputDims[] = {1, 8, 4};

    parameter_t *weights = buildSymParam(3, weightDims, weight_conv1dSym_groupsGroupedSym);
    parameter_t *bias = buildSymParam(1, biasDims, bias_conv1dSym_groupsGroupedSym);
    tensor_t *input = buildSymTensor(3, inputDims, input_conv1dSym_groupsGroupedSym);
    tensor_t *lossGrad = buildSymTensor(3, outputDims, lossGrad_conv1dSym_groupsGroupedSym);
    tensor_t *propLoss = buildSymTensor(3, inputDims, NULL);

    kernel_t kernel;
    initKernel(&kernel, 2, VALID, 1, 1);
    conv1dConfig_t cfg;
    quantization_t *sq = quantizationInitSymInt32(HALF_AWAY);
    static layerConfig_t lc;
    static layer_t layer;
    initConv1dConfigWithWeightsAndBias(&cfg, &kernel, weights, bias, 2, sq, sq, sq, sq);
    layer.type = CONV1D;
    lc.conv1d = &cfg;
    layer.config = &lc;

    conv1dBackward(&layer, input, lossGrad, propLoss);

    /* propLoss (dx) */
    int32_t *dx = (int32_t *)propLoss->data;
    for (size_t i = 0; i < expectedPropLoss_conv1dSym_groupsGroupedSym_len; i++) {
        TEST_ASSERT_INT_WITHIN(propLossMantissaTol_conv1dSym_groupsGroupedSym,
                               expectedPropLoss_conv1dSym_groupsGroupedSym[i], dx[i]);
    }
    float dxScale = symScaleOf(propLoss);
    TEST_ASSERT_FLOAT_WITHIN(expectedPropLossScale_conv1dSym_groupsGroupedSym * 1e-4f,
                             expectedPropLossScale_conv1dSym_groupsGroupedSym, dxScale);
    for (size_t i = 0; i < expectedPropLossDequant_conv1dSym_groupsGroupedSym_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(propLossDequantTol_conv1dSym_groupsGroupedSym,
                                 expectedPropLossDequant_conv1dSym_groupsGroupedSym[i],
                                 (float)dx[i] * dxScale);
    }
    /* weightGrad */
    int32_t *dw = (int32_t *)weights->grad->data;
    float dwScale = symScaleOf(weights->grad);
    for (size_t i = 0; i < expectedWeightGrad_conv1dSym_groupsGroupedSym_len; i++) {
        TEST_ASSERT_INT_WITHIN(weightGradMantissaTol_conv1dSym_groupsGroupedSym,
                               expectedWeightGrad_conv1dSym_groupsGroupedSym[i], dw[i]);
        TEST_ASSERT_FLOAT_WITHIN(weightGradDequantTol_conv1dSym_groupsGroupedSym,
                                 expectedWeightGradDequant_conv1dSym_groupsGroupedSym[i],
                                 (float)dw[i] * dwScale);
    }
    /* biasGrad */
    int32_t *db = (int32_t *)bias->grad->data;
    float dbScale = symScaleOf(bias->grad);
    for (size_t i = 0; i < expectedBiasGrad_conv1dSym_groupsGroupedSym_len; i++) {
        TEST_ASSERT_INT_WITHIN(biasGradMantissaTol_conv1dSym_groupsGroupedSym,
                               expectedBiasGrad_conv1dSym_groupsGroupedSym[i], db[i]);
        TEST_ASSERT_FLOAT_WITHIN(biasGradDequantTol_conv1dSym_groupsGroupedSym,
                                 expectedBiasGradDequant_conv1dSym_groupsGroupedSym[i],
                                 (float)db[i] * dbScale);
    }
}

void testConv1dBackwardSymStrideDilation() {
    size_t weightDims[] = {1, 1, 2};
    size_t inputDims[] = {1, 1, 9};
    size_t outputDims[] = {1, 1, 3};

    parameter_t *weights = buildSymParam(3, weightDims, weight_conv1dSym_strideDilationSym);
    tensor_t *input = buildSymTensor(3, inputDims, input_conv1dSym_strideDilationSym);
    tensor_t *lossGrad = buildSymTensor(3, outputDims, lossGrad_conv1dSym_strideDilationSym);
    tensor_t *propLoss = buildSymTensor(3, inputDims, NULL);

    kernel_t kernel;
    initKernel(&kernel, 2, VALID, 2, 3); /* size=2, VALID, dilation=2, stride=3 */
    conv1dConfig_t cfg;
    quantization_t *sq = quantizationInitSymInt32(HALF_AWAY);
    static layerConfig_t lc;
    static layer_t layer;
    initConv1dConfigWithWeightsAndBias(&cfg, &kernel, weights, NULL, 1, sq, sq, sq, sq);
    layer.type = CONV1D;
    lc.conv1d = &cfg;
    layer.config = &lc;

    conv1dBackward(&layer, input, lossGrad, propLoss);

    /* propLoss (dx) */
    int32_t *dx = (int32_t *)propLoss->data;
    for (size_t i = 0; i < expectedPropLoss_conv1dSym_strideDilationSym_len; i++) {
        TEST_ASSERT_INT_WITHIN(propLossMantissaTol_conv1dSym_strideDilationSym,
                               expectedPropLoss_conv1dSym_strideDilationSym[i], dx[i]);
    }
    float dxScale = symScaleOf(propLoss);
    TEST_ASSERT_FLOAT_WITHIN(expectedPropLossScale_conv1dSym_strideDilationSym * 1e-4f,
                             expectedPropLossScale_conv1dSym_strideDilationSym, dxScale);
    for (size_t i = 0; i < expectedPropLossDequant_conv1dSym_strideDilationSym_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(propLossDequantTol_conv1dSym_strideDilationSym,
                                 expectedPropLossDequant_conv1dSym_strideDilationSym[i],
                                 (float)dx[i] * dxScale);
    }
    /* weightGrad */
    int32_t *dw = (int32_t *)weights->grad->data;
    float dwScale = symScaleOf(weights->grad);
    for (size_t i = 0; i < expectedWeightGrad_conv1dSym_strideDilationSym_len; i++) {
        TEST_ASSERT_INT_WITHIN(weightGradMantissaTol_conv1dSym_strideDilationSym,
                               expectedWeightGrad_conv1dSym_strideDilationSym[i], dw[i]);
        TEST_ASSERT_FLOAT_WITHIN(weightGradDequantTol_conv1dSym_strideDilationSym,
                                 expectedWeightGradDequant_conv1dSym_strideDilationSym[i],
                                 (float)dw[i] * dwScale);
    }
    /* no biasGrad: bias == NULL */
}

/* ---------------------------------------------------------------------------
 * Shape-guard death tests (#232).
 *
 * The weightGrad helpers stride into lossGrad by `batch` (from forwardInput) and
 * write the weight grad by `outChannels` (from lossGrad). A mis-shaped lossGrad
 * is therefore a latent OOB read (too-few batches) / OOB write (outChannels !=
 * weight Cout). Each guard must fail-fast via exit(1). FLOAT helpers are static,
 * so they are exercised through the public conv1dBackward entry point; the SYM
 * helpers are exported and called directly. Data is all-zero — the guards read
 * shapes only and fire before any allocation or accumulation.
 * ------------------------------------------------------------------------- */

void testConv1dWeightGradFloatRejectsBatchMismatch() {
    size_t weightDims[] = {1, 1, 2};
    size_t inputDims[] = {2, 1, 5}; // forward batch 2
    size_t outputDims[] = {2, 1, 4};
    float weightData[2] = {0};
    float inputData[10] = {0};
    float outBuf[8] = {0};
    conv1dFixtureSetup_t s = {
        .weightDims = weightDims,
        .biasDims = NULL,
        .inputDims = inputDims,
        .outputDims = outputDims,
        .hasBias = 0,
        .kSize = 2,
        .padding = VALID,
        .paddingAmount = 0,
        .dilation = 1,
        .stride = 1,
        .groups = 1,
        .weightData = weightData,
        .biasData = NULL,
        .inputData = inputData,
    };
    conv1dRunResult_t r = conv1dRunForward(s, outBuf);

    size_t lossDims[] = {1, 1, 4}; // lossGrad batch 1 != forward batch 2
    float lossData[4] = {0};
    tensor_t *lossGrad = tensorInitFloat(lossData, lossDims, 3, NULL);
    size_t propDims[] = {2, 1, 5};
    float propBuf[10] = {0};
    tensor_t *propLoss = tensorInitFloat(propBuf, propDims, 3, NULL);

    ASSERT_EXITS_WITH_FAILURE(conv1dBackward(r.layer, r.input, lossGrad, propLoss));
}

void testConv1dWeightGradFloatRejectsOutChannelMismatch() {
    size_t weightDims[] = {1, 1, 2}; // weight Cout = 1
    size_t inputDims[] = {1, 1, 5};
    size_t outputDims[] = {1, 1, 4};
    float weightData[2] = {0};
    float inputData[5] = {0};
    float outBuf[4] = {0};
    conv1dFixtureSetup_t s = {
        .weightDims = weightDims,
        .biasDims = NULL,
        .inputDims = inputDims,
        .outputDims = outputDims,
        .hasBias = 0,
        .kSize = 2,
        .padding = VALID,
        .paddingAmount = 0,
        .dilation = 1,
        .stride = 1,
        .groups = 1,
        .weightData = weightData,
        .biasData = NULL,
        .inputData = inputData,
    };
    conv1dRunResult_t r = conv1dRunForward(s, outBuf);

    size_t lossDims[] = {1, 3, 4}; // outChannels 3 != weight Cout 1
    float lossData[12] = {0};
    tensor_t *lossGrad = tensorInitFloat(lossData, lossDims, 3, NULL);
    size_t propDims[] = {1, 1, 5};
    float propBuf[5] = {0};
    tensor_t *propLoss = tensorInitFloat(propBuf, propDims, 3, NULL);

    ASSERT_EXITS_WITH_FAILURE(conv1dBackward(r.layer, r.input, lossGrad, propLoss));
}

void testConv1dWeightGradSymRejectsBatchMismatch() {
    size_t weightDims[] = {8, 2, 2};
    size_t inputDims[] = {2, 4, 5}; // forward batch 2
    size_t lossDims[] = {1, 8, 4};  // lossGrad batch 1 != forward batch 2

    parameter_t *weights = buildSymParam(3, weightDims, NULL);
    tensor_t *input = buildSymTensor(3, inputDims, NULL);
    tensor_t *lossGrad = buildSymTensor(3, lossDims, NULL);

    kernel_t kernel;
    initKernel(&kernel, 2, VALID, 1, 1);
    conv1dConfig_t cfg;
    quantization_t *sq = quantizationInitSymInt32(HALF_AWAY);
    initConv1dConfigWithWeightsAndBias(&cfg, &kernel, weights, NULL, 2, sq, sq, sq, sq);

    ASSERT_EXITS_WITH_FAILURE(conv1dCalcWeightGradsSymInt32(&cfg, input, lossGrad));
}

void testConv1dWeightGradSymRejectsOutChannelMismatch() {
    size_t weightDims[] = {8, 2, 2}; // weight Cout = 8
    size_t inputDims[] = {1, 4, 5};
    size_t lossDims[] = {1, 10, 4}; // outChannels 10 != weight Cout 8

    parameter_t *weights = buildSymParam(3, weightDims, NULL);
    tensor_t *input = buildSymTensor(3, inputDims, NULL);
    tensor_t *lossGrad = buildSymTensor(3, lossDims, NULL);

    kernel_t kernel;
    initKernel(&kernel, 2, VALID, 1, 1);
    conv1dConfig_t cfg;
    quantization_t *sq = quantizationInitSymInt32(HALF_AWAY);
    initConv1dConfigWithWeightsAndBias(&cfg, &kernel, weights, NULL, 2, sq, sq, sq, sq);

    ASSERT_EXITS_WITH_FAILURE(conv1dCalcWeightGradsSymInt32(&cfg, input, lossGrad));
}

void testConv1dBiasGradFloatRejectsOutChannelMismatch() {
    /* weight Cout == lossGrad outChannels so the weightGrad guard passes; bias
     * Cout differs, so the biasGrad guard must fire. The layer is intentionally
     * inconsistent, so forward is skipped (it would OOB-read bias in the parent);
     * the layer is built directly and only backward is exercised, in the child. */
    size_t weightDims[] = {2, 1, 2}; // weight Cout = 2
    size_t biasDims[] = {1};         // bias Cout = 1 (intentionally inconsistent)
    size_t inputDims[] = {1, 1, 5};
    float weightData[4] = {0};
    float biasData[1] = {0};
    float inputData[5] = {0};

    tensor_t *weightParam = tensorInitFloat(weightData, weightDims, 3, NULL);
    parameter_t *weights = parameterInit(weightParam, gradInitFloat(weightParam, NULL));
    tensor_t *biasParam = tensorInitFloat(biasData, biasDims, 1, NULL);
    parameter_t *bias = parameterInit(biasParam, gradInitFloat(biasParam, NULL));

    kernel_t kernel;
    initKernel(&kernel, 2, VALID, 1, 1);
    quantization_t *q = quantizationInitFloat();
    layer_t *layer = conv1dLayerInitLegacy(weights, bias, &kernel, q, q, q, q);

    tensor_t *input = tensorInitFloat(inputData, inputDims, 3, NULL);
    size_t lossDims[] = {1, 2, 4}; // outChannels 2 == weight Cout, != bias Cout 1
    float lossData[8] = {0};
    tensor_t *lossGrad = tensorInitFloat(lossData, lossDims, 3, NULL);
    size_t propDims[] = {1, 1, 5};
    float propBuf[5] = {0};
    tensor_t *propLoss = tensorInitFloat(propBuf, propDims, 3, NULL);

    ASSERT_EXITS_WITH_FAILURE(conv1dBackward(layer, input, lossGrad, propLoss));
}

void testConv1dBiasGradSymRejectsOutChannelMismatch() {
    size_t weightDims[] = {3, 1, 1}; // K=1 to satisfy the config kernel-size check
    size_t biasDims[] = {1};         // bias Cout = 1
    size_t lossDims[] = {1, 3, 4};   // outChannels 3 != bias Cout 1

    parameter_t *weights = buildSymParam(3, weightDims, NULL);
    parameter_t *bias = buildSymParam(1, biasDims, NULL);
    tensor_t *lossGrad = buildSymTensor(3, lossDims, NULL);

    kernel_t kernel;
    initKernel(&kernel, 1, VALID, 1, 1);
    conv1dConfig_t cfg;
    quantization_t *sq = quantizationInitSymInt32(HALF_AWAY);
    initConv1dConfigWithWeightsAndBias(&cfg, &kernel, weights, bias, 1, sq, sq, sq, sq);

    ASSERT_EXITS_WITH_FAILURE(conv1dCalcBiasGradsSymInt32(&cfg, lossGrad));
}

void setUp() {}
void tearDown() {}

int main() {
    UNITY_BEGIN();
    RUN_TEST(testConv1dForwardMultiChannelWithBias);
    RUN_TEST(testConv1dForwardSingleChannelSingleBatch);
    RUN_TEST(testConv1dBackwardSingleChannelWithBias);
    RUN_TEST(testConv1dBackwardSamePaddingSymmetric);
    RUN_TEST(testConv1dForwardMultiBatch);
    RUN_TEST(testConv1dForwardGroupsDepthwise);
    RUN_TEST(testConv1dBackwardGroupsDepthwise);
    RUN_TEST(testConv1dForwardGroupsGrouped);
    RUN_TEST(testConv1dBackwardGroupsGrouped);
    RUN_TEST(testConv1dForwardStrideDilation);
    RUN_TEST(testConv1dForwardSamePaddingAsymmetric);
    RUN_TEST(testConv1dBackwardSamePaddingAsymmetric);
    RUN_TEST(testConv1dForwardSamePaddingWithGroups);
    RUN_TEST(testConv1dBackwardSamePaddingWithGroups);
    RUN_TEST(testConv1dForwardPointwise);
    RUN_TEST(testConv1dBackwardPointwise);
    RUN_TEST(testConv1dForwardExplicitPadding);
    RUN_TEST(testConv1dBackwardExplicitPadding);
    RUN_TEST(testConv1dForwardSymSingleChannelSingleBatch);
    RUN_TEST(testConv1dForwardSymSingleChannelWithBias);
    RUN_TEST(testConv1dForwardSymPointwise);
    RUN_TEST(testConv1dForwardSymExplicitPadding);
    RUN_TEST(testConv1dKernelSymScatterStrideDilation);
    RUN_TEST(testConv1dCalcWeightGradsSymGroupsGrouped);
    RUN_TEST(testConv1dCalcBiasGradsSymPointwise);
    RUN_TEST(testConv1dBackwardSymExplicitPadding);
    RUN_TEST(testConv1dBackwardSymGroupsGrouped);
    RUN_TEST(testConv1dBackwardSymStrideDilation);
    RUN_TEST(testConv1dWeightGradFloatRejectsBatchMismatch);
    RUN_TEST(testConv1dWeightGradFloatRejectsOutChannelMismatch);
    RUN_TEST(testConv1dWeightGradSymRejectsBatchMismatch);
    RUN_TEST(testConv1dWeightGradSymRejectsOutChannelMismatch);
    RUN_TEST(testConv1dBiasGradFloatRejectsOutChannelMismatch);
    RUN_TEST(testConv1dBiasGradSymRejectsOutChannelMismatch);
    return UNITY_END();
}
