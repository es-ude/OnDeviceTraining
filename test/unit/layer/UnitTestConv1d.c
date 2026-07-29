#include <stdlib.h>
#include <string.h>

#include "BorrowedLayer.h"
#include "Conv1d.h"
#include "Conv1dApi.h"
#include "ConvTranspose1dKernel.h"
#include "DeathTest.h"
#include "Layer.h"
#include "LayerCommon.h"
#include "LayerQuant.h"
#include "QuantizationApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "expected_conv1d.h"
#include "expected_conv1d_grouped.h"
#include "unity.h"

typedef struct conv1dFixtureSetup {
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

    tensor_t *weightParam = makeFloatTensor(s.weightDims, 3, s.weightData);
    tensor_t *weightGrad = gradInitFloat(weightParam, NULL);
    r.weights = parameterInit(weightParam, weightGrad);

    if (s.hasBias) {
        tensor_t *biasParam = makeFloatTensor(s.biasDims, 1, s.biasData);
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

    // The UserAPI factory (conv1dLayerInit) always allocates its own weights/bias
    // (KAIMING init requires FLOAT32 storage, LayerCommon.c requireFloat32) and
    // cannot borrow caller-built parameter_t/kernel_t, so this fixture goes
    // directly through initConv1dConfigWithWeightsAndBias for every groups value
    // (all statics so their addresses remain valid after this function returns).
    static conv1dConfig_t cfg;
    static layerConfig_t lc;
    static layer_t l;
    initConv1dConfigWithWeightsAndBias(&cfg, &kernelStore, r.weights, r.bias, s.groups, r.q, r.q,
                                       r.q, r.q);
    l.type = CONV1D;
    lc.conv1d = &cfg;
    l.config = &lc;
    r.layer = &l;

    r.input = makeFloatTensor(s.inputDims, 3, s.inputData);
    r.output = makeFloatTensor(s.outputDims, 3, NULL);
    (void)outputBuf;
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
    tensor_t *weightParam = makeFloatTensor(weightDims, 3, weight_conv1d_multiChannelWithBias);
    tensor_t *weightGrad = gradInitFloat(weightParam, NULL);
    parameter_t *weights = parameterInit(weightParam, weightGrad);

    size_t biasDims[] = {2};
    tensor_t *biasParam = makeFloatTensor(biasDims, 1, bias_conv1d_multiChannelWithBias);
    tensor_t *biasGrad = gradInitFloat(biasParam, NULL);
    parameter_t *bias = parameterInit(biasParam, biasGrad);

    kernel_t kernel;
    initKernel(&kernel, 3, VALID, 1, 1);
    quantization_t *q = quantizationInitFloat();
    layer_t *conv1d = buildBorrowedConv1dLayer(weights, bias, &kernel, q);

    size_t inputDims[] = {1, 3, 5};
    tensor_t *input = makeFloatTensor(inputDims, 3, input_conv1d_multiChannelWithBias);

    size_t outputDims[] = {1, 2, 3};
    tensor_t *output = makeFloatTensor(outputDims, 3, NULL);

    conv1dForward(conv1d, input, output);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expectedForward_conv1d_multiChannelWithBias, output->data,
                                  expectedForward_conv1d_multiChannelWithBias_len);
}

void testConv1dForwardSingleChannelSingleBatch() {
    size_t weightDims[] = {1, 1, 2};
    tensor_t *weightParam = makeFloatTensor(weightDims, 3, weight_conv1d_singleChannelSingleBatch);
    tensor_t *weightGrad = gradInitFloat(weightParam, NULL);
    parameter_t *weights = parameterInit(weightParam, weightGrad);

    kernel_t kernel;
    initKernel(&kernel, 2, VALID, 1, 1);

    quantization_t *q = quantizationInitFloat();
    layer_t *conv1d = buildBorrowedConv1dLayer(weights, NULL, &kernel, q);

    size_t inputDims[] = {1, 1, 4};
    tensor_t *input = makeFloatTensor(inputDims, 3, input_conv1d_singleChannelSingleBatch);

    size_t outputDims[] = {1, 1, 3};
    tensor_t *output = makeFloatTensor(outputDims, 3, NULL);

    conv1dForward(conv1d, input, output);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expectedForward_conv1d_singleChannelSingleBatch, output->data,
                                  expectedForward_conv1d_singleChannelSingleBatch_len);
}

void testConv1dBackwardSingleChannelWithBias() {
    size_t weightDims[] = {1, 1, 2};
    tensor_t *weightParam = makeFloatTensor(weightDims, 3, weight_conv1d_singleChannelWithBias);
    tensor_t *weightGrad = makeFloatTensor(weightDims, 3, NULL);
    parameter_t *weights = parameterInit(weightParam, weightGrad);

    size_t biasDims[] = {1};
    tensor_t *biasParam = makeFloatTensor(biasDims, 1, bias_conv1d_singleChannelWithBias);
    tensor_t *biasGrad = makeFloatTensor(biasDims, 1, NULL);
    parameter_t *bias = parameterInit(biasParam, biasGrad);

    kernel_t kernel;
    initKernel(&kernel, 2, VALID, 1, 1);
    quantization_t *q = quantizationInitFloat();
    layer_t *conv1d = buildBorrowedConv1dLayer(weights, bias, &kernel, q);

    size_t inputDims[] = {1, 1, 4};
    tensor_t *input = makeFloatTensor(inputDims, 3, input_conv1d_singleChannelWithBias);

    // forward (sanity — also fills output)
    size_t outputDims[] = {1, 1, 3};
    tensor_t *output = makeFloatTensor(outputDims, 3, NULL);
    conv1dForward(conv1d, input, output);

    // lossGrad = ones (matches what the generator used for autograd)
    float lossGradData[3];
    for (size_t i = 0; i < 3; i++) {
        lossGradData[i] = 1.0f;
    }
    tensor_t *lossGrad = makeFloatTensor(outputDims, 3, lossGradData);

    // propLoss buffer caller-owned, pre-zeroed
    tensor_t *propLoss = makeFloatTensor(inputDims, 3, NULL);

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
    tensor_t *weightParam = makeFloatTensor(weightDims, 3, weight_conv1d_samePaddingSymmetric);
    tensor_t *weightGrad = makeFloatTensor(weightDims, 3, NULL);
    parameter_t *weights = parameterInit(weightParam, weightGrad);

    kernel_t kernel;
    initKernel(&kernel, 3, SAME, 1, 1);
    quantization_t *q = quantizationInitFloat();
    layer_t *conv1d = buildBorrowedConv1dLayer(weights, NULL, &kernel, q);

    size_t inputDims[] = {1, 1, 5};
    tensor_t *input = makeFloatTensor(inputDims, 3, input_conv1d_samePaddingSymmetric);

    size_t outputDims[] = {1, 1, 5};
    tensor_t *output = makeFloatTensor(outputDims, 3, NULL);
    conv1dForward(conv1d, input, output);

    float lossGradData[5];
    for (size_t i = 0; i < 5; i++) {
        lossGradData[i] = 1.0f;
    }
    tensor_t *lossGrad = makeFloatTensor(outputDims, 3, lossGradData);

    tensor_t *propLoss = makeFloatTensor(inputDims, 3, NULL);

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
    tensor_t *lossGrad = makeFloatTensor(outputDims, 3, lossGradData);

    tensor_t *propLoss = makeFloatTensor(inputDims, 3, NULL);

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
    tensor_t *lossGrad = makeFloatTensor(outputDims, 3, lossGradData);

    tensor_t *propLoss = makeFloatTensor(inputDims, 3, NULL);

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
    tensor_t *lossGrad = makeFloatTensor(outputDims, 3, lossGradData);

    tensor_t *propLoss = makeFloatTensor(inputDims, 3, NULL);

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
    tensor_t *lossGrad = makeFloatTensor(outputDims, 3, lossGradData);

    tensor_t *propLoss = makeFloatTensor(inputDims, 3, NULL);

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
    tensor_t *lossGrad = makeFloatTensor(outputDims, 3, lossGrad_conv1d_pointwise);

    tensor_t *propLoss = makeFloatTensor(inputDims, 3, NULL);

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
    tensor_t *lossGrad = makeFloatTensor(outputDims, 3, lossGrad_conv1d_explicitPadding);

    tensor_t *propLoss = makeFloatTensor(inputDims, 3, NULL);

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

/* Re-gold (spec D5): conv1dForward now routes SYM through executeOp's
 * OUT_WRITE epilogue, which requants the raw s_in*s_w accumulator wire
 * through the conversionMatrix diagonal (requantSymInt32Tensor) instead of
 * writing it unrestored (pre-PR1b.2 behavior — the fixture this test asserted
 * against was the raw wire, characterizing exactly what a downstream
 * Quantization layer used to restore). Dequant-equivalence: restored
 * mantissa*restoredScale == raw mantissa*rawScale within representation
 * tolerance (both are exact re-expressions of the same real value at a
 * different int12 scale) — verified by generate_expected_conv1d.py's
 * `emulate_sym_conv` self-check (fwd_err <= fwd_tol against the float64
 * PyTorch-autograd reference, computed on the RESTORED fwd_deq/fwd_scale).
 * Same re-gold class as Task 2's propLoss/Task 3's LayerNorm forward pins
 * (ratified spec D5 principle, controller 2026-07-03). Applies identically
 * to the 3 other testConv1dForwardSym* tests below. */
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
    layer_t *conv1d = buildBorrowedConv1dLayer(weights, NULL, &kernel, sq);

    conv1dForward(conv1d, input, output);

    int32_t *m = (int32_t *)output->data;
    for (size_t i = 0; i < expectedForward_conv1dSym_singleChannelSingleBatch_len; i++) {
        TEST_ASSERT_INT_WITHIN(forwardMantissaTol_conv1dSym_singleChannelSingleBatch,
                               expectedForward_conv1dSym_singleChannelSingleBatch[i], m[i]);
    }
    float scale = symScaleOf(output);
    TEST_ASSERT_FLOAT_WITHIN(expectedForwardScale_conv1dSym_singleChannelSingleBatch *
                                 forwardScaleTol_conv1dSym_singleChannelSingleBatch,
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
    layer_t *conv1d = buildBorrowedConv1dLayer(weights, bias, &kernel, sq);

    conv1dForward(conv1d, input, output);

    int32_t *m = (int32_t *)output->data;
    for (size_t i = 0; i < expectedForward_conv1dSym_singleChannelWithBias_len; i++) {
        TEST_ASSERT_INT_WITHIN(forwardMantissaTol_conv1dSym_singleChannelWithBias,
                               expectedForward_conv1dSym_singleChannelWithBias[i], m[i]);
    }
    float scale = symScaleOf(output);
    TEST_ASSERT_FLOAT_WITHIN(expectedForwardScale_conv1dSym_singleChannelWithBias *
                                 forwardScaleTol_conv1dSym_singleChannelWithBias,
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
    layer_t *conv1d = buildBorrowedConv1dLayer(weights, bias, &kernel, sq);

    conv1dForward(conv1d, input, output);

    int32_t *m = (int32_t *)output->data;
    for (size_t i = 0; i < expectedForward_conv1dSym_pointwise_len; i++) {
        TEST_ASSERT_INT_WITHIN(forwardMantissaTol_conv1dSym_pointwise,
                               expectedForward_conv1dSym_pointwise[i], m[i]);
    }
    float scale = symScaleOf(output);
    TEST_ASSERT_FLOAT_WITHIN(expectedForwardScale_conv1dSym_pointwise *
                                 forwardScaleTol_conv1dSym_pointwise,
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
    layer_t *conv1d = buildBorrowedConv1dLayer(weights, bias, &kernel, sq);

    conv1dForward(conv1d, input, output);

    int32_t *m = (int32_t *)output->data;
    for (size_t i = 0; i < expectedForward_conv1dSym_explicitPadding_len; i++) {
        TEST_ASSERT_INT_WITHIN(forwardMantissaTol_conv1dSym_explicitPadding,
                               expectedForward_conv1dSym_explicitPadding[i], m[i]);
    }
    float scale = symScaleOf(output);
    TEST_ASSERT_FLOAT_WITHIN(expectedForwardScale_conv1dSym_explicitPadding *
                                 forwardScaleTol_conv1dSym_explicitPadding,
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

    /* Direct low-level kernel call — bypasses conv1dBackward's executeOp
     * funnel entirely, so this characterizes convTranspose1dKernelSymInt32's
     * own raw, unrestored output (the RawKernel fixtures), not the
     * funnel-restored propLoss wire conv1dBackward now produces (design D3;
     * see testConv1dBackwardSymStrideDilation for that). */
    convTranspose1dKernelSymInt32(lossGrad, weight, NULL, &kernel, 1, 0, propLoss);

    int32_t *m = (int32_t *)propLoss->data;
    for (size_t i = 0; i < expectedPropLossRawKernel_conv1dSym_strideDilationSym_len; i++) {
        TEST_ASSERT_INT_WITHIN(propLossRawKernelMantissaTol_conv1dSym_strideDilationSym,
                               expectedPropLossRawKernel_conv1dSym_strideDilationSym[i], m[i]);
    }
    float scale = symScaleOf(propLoss);
    TEST_ASSERT_FLOAT_WITHIN(expectedPropLossRawKernelScale_conv1dSym_strideDilationSym * 1e-4f,
                             expectedPropLossRawKernelScale_conv1dSym_strideDilationSym, scale);
    for (size_t i = 0; i < expectedPropLossRawKernelDequant_conv1dSym_strideDilationSym_len; i++) {
        TEST_ASSERT_FLOAT_WITHIN(propLossRawKernelDequantTol_conv1dSym_strideDilationSym,
                                 expectedPropLossRawKernelDequant_conv1dSym_strideDilationSym[i],
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

/* Re-gold (spec D5): conv1dBackward's dx wire (propLoss) is a *produced*
 * wire, not a passthrough — convTranspose1dKernelSymInt32 emits the raw
 * s_loss*s_w scatter-adjoint mantissa (characterized unrestored above by
 * testConv1dKernelSymScatterStrideDilation), and executeOp's OUT_WRITE
 * epilogue then requants it through the conversionMatrix diagonal to
 * propLossQ's declared width (int12) before conv1dBackward returns it — the
 * same restoration every other producer wire gets (forward output,
 * weightGrad, biasGrad). The propLoss expectations below are therefore
 * POST-restoration values, not the kernel's raw output; the old #187
 * fail-fast (produced propLoss tensor must be SYM) this file used to lean on
 * is retired along with the raw-wire validator
 * (docs/conventions/arithmetic-sym.md). Dequant-equivalence: restored
 * mantissa*restoredScale == raw mantissa*rawScale within representation
 * tolerance — verified by generate_expected_conv1d.py's `emulate_sym_conv`
 * dx section (`_requant_absmax_i12_f32`, dx_err <= dx_tol against the
 * float64 PyTorch-autograd reference, computed on the RESTORED
 * dx_deq/dx_scale). Same re-gold class as this file's forward pins above.
 * Applies identically to the 2 other testConv1dBackwardSym* tests below. */
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
    tensor_t *lossGrad = makeFloatTensor(lossDims, 3, lossData);
    size_t propDims[] = {2, 1, 5};
    tensor_t *propLoss = makeFloatTensor(propDims, 3, NULL);

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
    tensor_t *lossGrad = makeFloatTensor(lossDims, 3, lossData);
    size_t propDims[] = {1, 1, 5};
    tensor_t *propLoss = makeFloatTensor(propDims, 3, NULL);

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

    tensor_t *weightParam = makeFloatTensor(weightDims, 3, weightData);
    parameter_t *weights = parameterInit(weightParam, gradInitFloat(weightParam, NULL));
    tensor_t *biasParam = makeFloatTensor(biasDims, 1, biasData);
    parameter_t *bias = parameterInit(biasParam, gradInitFloat(biasParam, NULL));

    kernel_t kernel;
    initKernel(&kernel, 2, VALID, 1, 1);
    quantization_t *q = quantizationInitFloat();
    layer_t *layer = buildBorrowedConv1dLayer(weights, bias, &kernel, q);

    tensor_t *input = makeFloatTensor(inputDims, 3, inputData);
    size_t lossDims[] = {1, 2, 4}; // outChannels 2 == weight Cout, != bias Cout 1
    float lossData[8] = {0};
    tensor_t *lossGrad = makeFloatTensor(lossDims, 3, lossData);
    size_t propDims[] = {1, 1, 5};
    tensor_t *propLoss = makeFloatTensor(propDims, 3, NULL);

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

/* #380 PR1 Task 2: create-time trainable knob (trainable_t). */
static layer_t *buildFloatConv1dWithTrainable(trainable_t trainable) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);
    layer_t *layer = conv1dLayerInitOwning(
        &(conv1dInit_t){.inChannels = 2, .outChannels = 3, .kernelSize = 3, .trainable = trainable},
        &lq);
    freeQuantization(q);
    return layer;
}

void testConv1dFactoryFrozenElidesGrads(void) {
    layer_t *layer = buildFloatConv1dWithTrainable(TRAINABLE_FALSE);
    conv1dConfig_t *cfg = layer->config->conv1d;
    bool weightsGradNull = cfg->weights->grad == NULL;
    bool biasGradNull = cfg->bias->grad == NULL;
    bool frozen = layerIsFrozen(layer);
    freeConv1dLayer(layer);
    TEST_ASSERT_TRUE(weightsGradNull);
    TEST_ASSERT_TRUE(biasGradNull);
    TEST_ASSERT_TRUE(frozen);
}

void testConv1dFactoryDefaultAllocatesGrads(void) {
    layer_t *layer = buildFloatConv1dWithTrainable(TRAINABLE_DEFAULT);
    conv1dConfig_t *cfg = layer->config->conv1d;
    bool weightsGradPresent = cfg->weights->grad != NULL;
    bool biasGradPresent = cfg->bias->grad != NULL;
    bool frozen = layerIsFrozen(layer);
    freeConv1dLayer(layer);
    TEST_ASSERT_TRUE(weightsGradPresent);
    TEST_ASSERT_TRUE(biasGradPresent);
    TEST_ASSERT_FALSE(frozen);
}

/* #380 PR1 Task 5: backward guard -- a frozen twin must skip the weight/bias
 * grad writes entirely (buffers stay all-zero) while still producing a
 * propLoss byte-identical to its trainable twin. Hand-seeded FLOAT32
 * fixtures via buildBorrowedConv1dLayer (deterministic, no RNG) so the two
 * twins start out bit-identical; only `frozen` differs. */
void testConv1dBackwardFrozenTwinPropLossIdenticalGradsZero(void) {
    size_t weightDims[] = {1, 1, 2};
    size_t biasDims[] = {1};
    size_t inputDims[] = {1, 1, 4};
    size_t outputDims[] = {1, 1, 3};

    tensor_t *weightParamA = makeFloatTensor(weightDims, 3, (float[]){1.f, -1.f});
    tensor_t *weightGradA = gradInitFloat(weightParamA, NULL);
    parameter_t *weightsA = parameterInit(weightParamA, weightGradA);
    tensor_t *biasParamA = makeFloatTensor(biasDims, 1, (float[]){0.5f});
    tensor_t *biasGradA = gradInitFloat(biasParamA, NULL);
    parameter_t *biasA = parameterInit(biasParamA, biasGradA);
    kernel_t kernelA;
    initKernel(&kernelA, 2, VALID, 1, 1);
    quantization_t *q = quantizationInitFloat();
    layer_t *trainableTwin = buildBorrowedConv1dLayer(weightsA, biasA, &kernelA, q);

    tensor_t *weightParamB = makeFloatTensor(weightDims, 3, (float[]){1.f, -1.f});
    tensor_t *weightGradB = gradInitFloat(weightParamB, NULL);
    parameter_t *weightsB = parameterInit(weightParamB, weightGradB);
    tensor_t *biasParamB = makeFloatTensor(biasDims, 1, (float[]){0.5f});
    tensor_t *biasGradB = gradInitFloat(biasParamB, NULL);
    parameter_t *biasB = parameterInit(biasParamB, biasGradB);
    kernel_t kernelB;
    initKernel(&kernelB, 2, VALID, 1, 1);
    layer_t *frozenTwin = buildBorrowedConv1dLayer(weightsB, biasB, &kernelB, q);
    frozenTwin->config->conv1d->frozen = true;

    tensor_t *input = makeFloatTensor(inputDims, 3, (float[]){1.f, 2.f, 3.f, 4.f});
    tensor_t *lossGrad = makeFloatTensor(outputDims, 3, (float[]){1.f, 1.f, 1.f});
    tensor_t *propLossTrainable = makeFloatTensor(inputDims, 3, NULL);
    tensor_t *propLossFrozen = makeFloatTensor(inputDims, 3, NULL);

    conv1dBackward(trainableTwin, input, lossGrad, propLossTrainable);
    conv1dBackward(frozenTwin, input, lossGrad, propLossFrozen);

    size_t numWeights = calcNumberOfElementsByTensor(weightParamA);
    size_t numBias = calcNumberOfElementsByTensor(biasParamA);
    size_t numPropLoss = calcNumberOfElementsByTensor(propLossTrainable);

    bool trainableWeightGradNonzero = false;
    bool frozenWeightGradAllZero = true;
    for (size_t i = 0; i < numWeights; i++) {
        if (((float *)weightGradA->data)[i] != 0.0f) {
            trainableWeightGradNonzero = true;
        }
        if (((float *)weightGradB->data)[i] != 0.0f) {
            frozenWeightGradAllZero = false;
        }
    }
    bool trainableBiasGradNonzero = false;
    bool frozenBiasGradAllZero = true;
    for (size_t i = 0; i < numBias; i++) {
        if (((float *)biasGradA->data)[i] != 0.0f) {
            trainableBiasGradNonzero = true;
        }
        if (((float *)biasGradB->data)[i] != 0.0f) {
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

/* Factory-frozen layer (grads == NULL, Task 2): conv1dBackward must complete
 * without dereferencing the (absent) grad buffers -- the ASan gate catches
 * any NULL/OOB deref if the guard is missing or misplaced. */
void testConv1dBackwardFrozenFactoryLayerRunsWithoutGradBuffers(void) {
    layer_t *layer = buildFloatConv1dWithTrainable(TRAINABLE_FALSE);

    size_t inputDims[] = {1, 2, 5};
    size_t outputDims[] = {1, 3, 3};
    tensor_t *input =
        makeFloatTensor(inputDims, 3, (float[]){1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f});
    tensor_t *lossGrad =
        makeFloatTensor(outputDims, 3, (float[]){1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f});
    tensor_t *propLoss = makeFloatTensor(inputDims, 3, NULL);

    conv1dBackward(layer, input, lossGrad, propLoss);

    bool gradStillNull = layer->config->conv1d->weights->grad == NULL;
    freeConv1dLayer(layer);

    TEST_ASSERT_TRUE(gradStillNull);
}

/* #380 PR2 Task 1: propLoss == NULL is a grads-only call -- weight/bias grads
 * must be computed exactly as with a real propLoss, and no dx memory may be
 * touched. Twin fixture (both TRAINABLE, hand-seeded, bit-identical) mirrors
 * the PR1 frozen-twin idiom above: twin A gets a real propLoss buffer, twin B
 * gets a literal NULL. Pre-guard, twin B's call dereferences the NULL
 * propLoss and crashes (RED); post-guard, weight/bias grads match twin A's
 * byte-for-byte and twin A's propLoss is non-degenerate. */
void testConv1dBackwardNullPropLossComputesGradsOnly(void) {
    size_t weightDims[] = {1, 1, 2};
    size_t biasDims[] = {1};
    size_t inputDims[] = {1, 1, 4};

    tensor_t *weightParamA = makeFloatTensor(weightDims, 3, (float[]){1.f, -1.f});
    tensor_t *weightGradA = gradInitFloat(weightParamA, NULL);
    parameter_t *weightsA = parameterInit(weightParamA, weightGradA);
    tensor_t *biasParamA = makeFloatTensor(biasDims, 1, (float[]){0.5f});
    tensor_t *biasGradA = gradInitFloat(biasParamA, NULL);
    parameter_t *biasA = parameterInit(biasParamA, biasGradA);
    kernel_t kernelA;
    initKernel(&kernelA, 2, VALID, 1, 1);
    quantization_t *q = quantizationInitFloat();
    layer_t *twinA = buildBorrowedConv1dLayer(weightsA, biasA, &kernelA, q);

    tensor_t *weightParamB = makeFloatTensor(weightDims, 3, (float[]){1.f, -1.f});
    tensor_t *weightGradB = gradInitFloat(weightParamB, NULL);
    parameter_t *weightsB = parameterInit(weightParamB, weightGradB);
    tensor_t *biasParamB = makeFloatTensor(biasDims, 1, (float[]){0.5f});
    tensor_t *biasGradB = gradInitFloat(biasParamB, NULL);
    parameter_t *biasB = parameterInit(biasParamB, biasGradB);
    kernel_t kernelB;
    initKernel(&kernelB, 2, VALID, 1, 1);
    layer_t *twinB = buildBorrowedConv1dLayer(weightsB, biasB, &kernelB, q);

    tensor_t *input = makeFloatTensor(inputDims, 3, (float[]){1.f, 2.f, 3.f, 4.f});
    tensor_t *lossGrad = makeFloatTensor((size_t[]){1, 1, 3}, 3, (float[]){1.f, 1.f, 1.f});
    tensor_t *propLossA = makeFloatTensor(inputDims, 3, NULL);

    conv1dBackward(twinA, input, lossGrad, propLossA);
    conv1dBackward(twinB, input, lossGrad, NULL);

    size_t numWeights = calcNumberOfElementsByTensor(weightParamA);
    size_t numBias = calcNumberOfElementsByTensor(biasParamA);
    size_t numPropLoss = calcNumberOfElementsByTensor(propLossA);

    bool weightGradsIdentical =
        memcmp(weightGradA->data, weightGradB->data,
               calcNumberOfBytesForData(weightGradA->quantization, numWeights)) == 0;
    bool biasGradsIdentical =
        memcmp(biasGradA->data, biasGradB->data,
               calcNumberOfBytesForData(biasGradA->quantization, numBias)) == 0;
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

/* ---- Group-quant PR2 (Task 4): Conv1d forward with a grouped SYM weight --
 *
 * Fixture: IC=2, OC=3, K=3, L=6, B=1, VALID padding, stride=1 (outLen=4),
 * int12 codes (generate_expected_conv1d_grouped.py). Both grouped-shape
 * fixtures (perChannel/general) share the SAME weight/input/bias mantissas
 * -- only the group SHAPE (numGroups/groupSize/scales) differs, mirroring
 * generate_expected_group_matmul.py's Task-3 discipline. */

/* Builds a SYM_INT32 (HALF_AWAY, int12) tensor with EXPLICIT int32 mantissas
 * + scale -- no absmax requantization, unlike buildSymTensor (which derives
 * both from a float source via tensorFillFromFloatBuffer). Needed here so
 * the fixture lands on EXACTLY the gold's hand-picked mantissas. */
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

/*! Builds the shared grouped-SYM weight/bias/input Conv1d layer (borrowed,
 *  no grad buffers -- grouped grads are a future axis, #300 -- mirrors
 *  UnitTestLinear.c's buildGroupedFixtureLayer exactly). `numGroups`/
 *  `groupSize`/`wScales` vary per fixture (perChannel vs general vs the PR3
 *  samePad shape); `padding` is VALID for all PR2 fixtures and SAME for the
 *  PR3 padding-coverage fixture; the weight/bias/input mantissas are the
 *  SAME shared fixture data. */
static layer_t *buildGroupedConv1dFixtureLayer(quantization_t *q, size_t numGroups,
                                               size_t groupSize, const float *wScales,
                                               paddingType_t padding, tensor_t **inputOut) {
    size_t weightDims[] = {(size_t)kConv1dGroupedOutChannels, (size_t)kConv1dGroupedInChannels,
                           (size_t)kConv1dGroupedKernelSize};
    size_t *ownedWeightDims = reserveMemory(3 * sizeof(size_t));
    memcpy(ownedWeightDims, weightDims, sizeof(weightDims));
    size_t *weightOrder = reserveMemory(3 * sizeof(size_t));
    setOrderOfDimsForNewTensor(3, weightOrder);
    shape_t *weightShape = reserveMemory(sizeof(shape_t));
    setShape(weightShape, ownedWeightDims, 3, weightOrder);
    tensor_t *weightsParam = initTensor(
        weightShape, quantizationInitSymGrouped(12, HALF_AWAY, numGroups, groupSize), NULL);
    size_t numWeightElems = (size_t)kConv1dGroupedOutChannels * (size_t)kConv1dGroupedInChannels *
                            (size_t)kConv1dGroupedKernelSize;
    byteConversion((uint8_t *)kConv1dGroupedWMantissas, 32, weightsParam->data, 12, numWeightElems);
    symQConfig_t *weightQC = weightsParam->quantization->qConfig;
    for (size_t g = 0; g < numGroups; g++) {
        weightQC->scales[g] = wScales[g];
    }
    parameter_t *weights = parameterInit(weightsParam, NULL);

    size_t biasDims[] = {(size_t)kConv1dGroupedOutChannels};
    tensor_t *biasParam =
        buildSymInt32TensorExact(1, biasDims, kConv1dGroupedBiasMantissas, kConv1dGroupedBiasScale);
    parameter_t *bias = parameterInit(biasParam, NULL);

    size_t inputDims[] = {(size_t)kConv1dGroupedBatch, (size_t)kConv1dGroupedInChannels,
                          (size_t)kConv1dGroupedInputLength};
    tensor_t *input =
        buildSymInt32TensorExact(3, inputDims, kConv1dGroupedXMantissas, kConv1dGroupedXScale);
    *inputOut = input;

    kernel_t *kernel = reserveMemory(sizeof(kernel_t));
    initKernel(kernel, (size_t)kConv1dGroupedKernelSize, padding, 1, 1);

    return buildBorrowedConv1dLayer(weights, bias, kernel, q);
}

/* Compares against the RAW gather-core gold EXACTLY, not with a loose
 * tolerance: the output tensor here is FLOAT32 while forwardMath stays
 * SYM_INT32 (grouped weight), so conv1dForward's executeOp epilogue takes
 * the SYM_INT32->FLOAT32 conversionMatrix cell (convertSymInt32TensorToFloat32Tensor,
 * TensorConversion.c) -- a single EXACT `(float)mantissa * scale` per
 * element, NOT the SYM_INT32->SYM_INT32 diagonal's absmax-derived fresh-scale
 * requant every other SYM forward test in this file compares through. That
 * requant is deliberately dodged here: dequantizing (mantissa*scale)
 * approximately preserves the represented real value REGARDLESS of which
 * internal scale the kernel's rescale-combines used, so a requantized
 * (or otherwise dequantized-with-a-generous-tolerance) comparison cannot
 * reliably distinguish a correct s_acc from a WRONG-but-still-sound one
 * (e.g. scales[0] instead of max) -- confirmed empirically while building
 * this test (mutation (i) below silently passed against such a comparison).
 * The exact FLOAT32 wire has no such blind spot: it is bit-for-bit the same
 * deterministic float32 formula generate_expected_conv1d_grouped.py's
 * conv1d_grouped_ref computes (python-int MACs, rescale_f32 ==
 * rescaleIntoAccumulatorScale(HALF_AWAY) bit-for-bit), so ANY divergence in
 * the kernel's internal arithmetic (wrong s_acc, dropped combine, wrong
 * rounding mode) changes the compared value measurably. Still routes through
 * conv1dForward -> executeOp's grouped-operand gate, so mutation (iv) (the
 * layer's groupedSymOperandPos wiring) still applies. */
void testConv1dForwardGroupedPerChannelMatchesGold(void) {
    quantization_t *testQ = quantizationInitSymInt32(HALF_AWAY);
    tensor_t *input = NULL;
    layer_t *conv1d = buildGroupedConv1dFixtureLayer(testQ, (size_t)kPerChannelNumGroups,
                                                     (size_t)kPerChannelGroupSize,
                                                     kPerChannelWScales, VALID, &input);

    size_t outputDims[] = {(size_t)kConv1dGroupedBatch, (size_t)kConv1dGroupedOutChannels,
                           (size_t)kPerChannelOutLen};
    tensor_t *output = makeFloatTensor(outputDims, 3, NULL);

    conv1dForward(conv1d, input, output);

    float *captured = (float *)output->data;
    for (size_t i = 0; i < kPerChannelOutMantissas_len; i++) {
        float expected = (float)kPerChannelOutMantissas[i] * kPerChannelOutScale;
        TEST_ASSERT_EQUAL_FLOAT(expected, captured[i]);
    }
}

void testConv1dForwardGroupedGeneralMatchesGold(void) {
    quantization_t *testQ = quantizationInitSymInt32(HALF_AWAY);
    tensor_t *input = NULL;
    layer_t *conv1d =
        buildGroupedConv1dFixtureLayer(testQ, (size_t)kGeneralNumGroups, (size_t)kGeneralGroupSize,
                                       kGeneralWScales, VALID, &input);

    size_t outputDims[] = {(size_t)kConv1dGroupedBatch, (size_t)kConv1dGroupedOutChannels,
                           (size_t)kGeneralOutLen};
    tensor_t *output = makeFloatTensor(outputDims, 3, NULL);

    conv1dForward(conv1d, input, output);

    float *captured = (float *)output->data;
    for (size_t i = 0; i < kGeneralOutMantissas_len; i++) {
        float expected = (float)kGeneralOutMantissas[i] * kGeneralOutScale;
        TEST_ASSERT_EQUAL_FLOAT(expected, captured[i]);
    }
}

/* Equal-scales grouped twin: mirrors
 * testMatmulGroupedEqualScalesBitIdenticalToScalar (UnitTestMatmul.c) at the
 * LAYER level. Every group's scale is the SAME power-of-two value (0.25f)
 * and inScale (0.5f, kConv1dGroupedXScale) is also a power of two, so sAcc =
 * inScale*maxScale and every combine's paramScale (inScale*scales[g]) are
 * BIT-IDENTICAL float32 values -- and dividing a float by the SAME
 * power-of-two value it was just multiplied by is an EXACT round trip (pure
 * exponent shifts, no mantissa rounding), so
 * round_half_away(partial*paramScale/sAcc) reproduces `partial` exactly. The
 * grouped kernel's raw output must therefore be BIT-IDENTICAL to the scalar
 * (non-grouped, per-tensor SYM_INT32) conv1dKernelSymInt32 path run on the
 * SAME mantissas with weight scale == the common group scale -- both wires
 * here are FLOAT32 (same exact-dequant reasoning as the two tests above), so
 * identical raw (mantissa, scale) pairs dequantize to identical floats. */
void testConv1dForwardGroupedEqualScalesBitIdenticalToScalar(void) {
    const float commonScale = 0.25f;
    quantization_t *testQ = quantizationInitSymInt32(HALF_AWAY);

    float groupScales[3] = {commonScale, commonScale, commonScale};
    tensor_t *groupedInput = NULL;
    layer_t *groupedLayer =
        buildGroupedConv1dFixtureLayer(testQ, 3, 6, groupScales, VALID, &groupedInput);

    size_t outputDims[] = {(size_t)kConv1dGroupedBatch, (size_t)kConv1dGroupedOutChannels,
                           (size_t)kPerChannelOutLen};
    tensor_t *groupedOutput = makeFloatTensor(outputDims, 3, NULL);
    conv1dForward(groupedLayer, groupedInput, groupedOutput);

    size_t weightDims[] = {(size_t)kConv1dGroupedOutChannels, (size_t)kConv1dGroupedInChannels,
                           (size_t)kConv1dGroupedKernelSize};
    tensor_t *scalarWeightParam =
        buildSymInt32TensorExact(3, weightDims, kConv1dGroupedWMantissas, commonScale);
    parameter_t *scalarWeights = parameterInit(scalarWeightParam, NULL);

    size_t biasDims[] = {(size_t)kConv1dGroupedOutChannels};
    tensor_t *scalarBiasParam =
        buildSymInt32TensorExact(1, biasDims, kConv1dGroupedBiasMantissas, kConv1dGroupedBiasScale);
    parameter_t *scalarBias = parameterInit(scalarBiasParam, NULL);

    size_t inputDims[] = {(size_t)kConv1dGroupedBatch, (size_t)kConv1dGroupedInChannels,
                          (size_t)kConv1dGroupedInputLength};
    tensor_t *scalarInput =
        buildSymInt32TensorExact(3, inputDims, kConv1dGroupedXMantissas, kConv1dGroupedXScale);

    kernel_t *scalarKernel = reserveMemory(sizeof(kernel_t));
    initKernel(scalarKernel, (size_t)kConv1dGroupedKernelSize, VALID, 1, 1);
    layer_t *scalarLayer = buildBorrowedConv1dLayer(scalarWeights, scalarBias, scalarKernel, testQ);

    tensor_t *scalarOutput = makeFloatTensor(outputDims, 3, NULL);
    conv1dForward(scalarLayer, scalarInput, scalarOutput);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY((float *)scalarOutput->data, (float *)groupedOutput->data,
                                  kPerChannelOutMantissas_len);
}

/* Final-review Fix 3(b): Conv1d had no FLOAT32-math grouped-forward coverage
 * at all (unlike Linear, which pins both arms via
 * testLinearForwardGroupedFloatPathAgreesWithinTolerance) -- the review found
 * conv1dForward's FLOAT32 arm never declared groupedSymOperandPos, an
 * asymmetry that would have made Conv1d's FLOAT32-math grouped forward
 * regress the moment the funnel's FLOAT32-arm deny gate landed (Fix 3a).
 * Same perChannel fixture as testConv1dForwardGroupedPerChannelMatchesGold
 * (buildGroupedConv1dFixtureLayer with a FLOAT32 `q` instead of SYM_INT32 --
 * the weight/bias/input tensors are STILL grouped-SYM/SYM_INT32 storage,
 * exactly like the SYM_INT32-math sibling; only forwardMath differs),
 * exercising Task 2's grouped dequant (convertSymTensorToFloat32Tensor)
 * gated by Fix 3(a)'s FLOAT32-arm check instead of skipping it entirely.
 *
 * Tolerance (same 2-combines-per-output-element structure as
 * testLinearForwardGroupedFloatPathAgreesWithinTolerance's per-channel
 * fixture: groupSize==the full reduction length per output channel, so ONE
 * weight tail-combine + ONE bias-seed combine per output element, each
 * HALF_AWAY-rounding by at most 0.5 quanta): 1.0 * kPerChannelOutScale +
 * 1e-6f headroom for float32 arithmetic noise. */
void testConv1dForwardGroupedPerChannelFloatPathAgreesWithinTolerance(void) {
    quantization_t *floatQ = quantizationInitFloat();
    tensor_t *input = NULL;
    layer_t *conv1d = buildGroupedConv1dFixtureLayer(floatQ, (size_t)kPerChannelNumGroups,
                                                     (size_t)kPerChannelGroupSize,
                                                     kPerChannelWScales, VALID, &input);

    size_t outputDims[] = {(size_t)kConv1dGroupedBatch, (size_t)kConv1dGroupedOutChannels,
                           (size_t)kPerChannelOutLen};
    tensor_t *output = makeFloatTensor(outputDims, 3, NULL);

    conv1dForward(conv1d, input, output);

    float *captured = (float *)output->data;
    const float tolerance = 1.0f * kPerChannelOutScale + 1e-6f;
    for (size_t i = 0; i < kPerChannelOutMantissas_len; i++) {
        float expected = (float)kPerChannelOutMantissas[i] * kPerChannelOutScale;
        TEST_ASSERT_FLOAT_WITHIN(tolerance, expected, captured[i]);
    }
}

/* ---- Group-quant PR3 (Task 3): Conv1d backward dx with a grouped SYM
 * weight -- the adjoint SCATTER (convTranspose1dKernelSymInt32Grouped in the
 * adjoint role, per-PRODUCT rescale into s_acc = lossScale*max(scales)).
 * Fixture: the VALID forward fixture's weight/group shapes with a seeded
 * random SYM_INT32 lossGrad [1, 3, 4] (generate_expected_conv1d_grouped.py's
 * dx fixtures, torch-autograd cross-checked there). ---- */

/*! Shared dx-fixture plumbing (mirrors UnitTestLinear.c's
 *  runGroupedDxBackward): builds the grouped VALID fixture layer, FREEZES it
 *  (the borrowed fixture parameters carry no grad buffers, and dx is the
 *  ONLY backward op that consumes the (grouped) weight tensor -- the
 *  weight/bias grad ops take {fwdIn, loss}/{loss} and are pinned by the
 *  existing SYM backward tests -- so freezing scopes these tests to the dx
 *  wire; #380: frozen layers still propagate loss), then runs conv1dBackward
 *  with the gold lossGrad into a FLOAT32 propLoss wire and returns it. */
static tensor_t *runGroupedConv1dDxBackward(quantization_t *q, size_t numGroups, size_t groupSize,
                                            const float *wScales) {
    tensor_t *input = NULL;
    layer_t *conv1d =
        buildGroupedConv1dFixtureLayer(q, numGroups, groupSize, wScales, VALID, &input);
    conv1d->config->conv1d->frozen = true;

    size_t lossDims[] = {(size_t)kConv1dGroupedBatch, (size_t)kConv1dGroupedOutChannels,
                         (size_t)kConv1dDxFwdOutLen};
    tensor_t *lossGrad =
        buildSymInt32TensorExact(3, lossDims, kConv1dDxLossMantissas, kConv1dDxLossScale);

    size_t propLossDims[] = {(size_t)kConv1dGroupedBatch, (size_t)kConv1dGroupedInChannels,
                             (size_t)kConv1dGroupedInputLength};
    tensor_t *propLoss = makeFloatTensor(propLossDims, 3, NULL);

    conv1dBackward(conv1d, input, lossGrad, propLoss);
    return propLoss;
}

/* Exact FLOAT32-wire compare against the RAW adjoint-scatter gold (the
 * grouped forward tests' ratified design, see
 * testConv1dForwardGroupedPerChannelMatchesGold's comment on why the exact
 * dequant wire has no wrong-but-sound-scale blind spot): propLossMath stays
 * SYM_INT32, the propLoss WIRE is FLOAT32, so the OUT_WRITE epilogue takes
 * the exact SYM_INT32->FLOAT32 cell. Routes through conv1dBackward ->
 * executeOp's grouped-operand gate, so the dx opSpec's groupedSymOperandPos
 * declaration is under test too (removing it dies with the funnel deny). */
void testConv1dBackwardGroupedDxPerChannelMatchesGold(void) {
    quantization_t *testQ = quantizationInitSymInt32(HALF_AWAY);
    tensor_t *propLoss = runGroupedConv1dDxBackward(
        testQ, (size_t)kPerChannelNumGroups, (size_t)kPerChannelGroupSize, kPerChannelWScales);

    float *captured = (float *)propLoss->data;
    for (size_t i = 0; i < kDxPerChannelOutMantissas_len; i++) {
        float expected = (float)kDxPerChannelOutMantissas[i] * kDxPerChannelOutScale;
        TEST_ASSERT_EQUAL_FLOAT(expected, captured[i]);
    }
}

void testConv1dBackwardGroupedDxGeneralMatchesGold(void) {
    quantization_t *testQ = quantizationInitSymInt32(HALF_AWAY);
    tensor_t *propLoss = runGroupedConv1dDxBackward(testQ, (size_t)kGeneralNumGroups,
                                                    (size_t)kGeneralGroupSize, kGeneralWScales);

    float *captured = (float *)propLoss->data;
    for (size_t i = 0; i < kDxGeneralOutMantissas_len; i++) {
        float expected = (float)kDxGeneralOutMantissas[i] * kDxGeneralOutScale;
        TEST_ASSERT_EQUAL_FLOAT(expected, captured[i]);
    }
}

/* Equal-scales dx twin: same argument as
 * testConv1dForwardGroupedEqualScalesBitIdenticalToScalar, transplanted to
 * the adjoint scatter. All group scales are the SAME power-of-two value
 * (0.25f) and the loss scale (0.5f) is a power of two, so sAcc and every
 * product's paramScale are BIT-IDENTICAL float32 values and each per-product
 * rescale is an EXACT identity (multiply/divide by the same power of two are
 * pure exponent shifts; the products, <= 40*5 = 200, are exactly
 * representable) -- the grouped scatter's raw dx must be BIT-IDENTICAL to
 * the scalar convTranspose1dKernelSymInt32 adjoint on the same mantissas
 * with weight scale == the common group scale. Both wires are FLOAT32, so
 * identical raw (mantissa, scale) pairs dequantize to identical floats. */
void testConv1dBackwardGroupedDxEqualScalesBitIdenticalToScalar(void) {
    const float commonScale = 0.25f;
    quantization_t *testQ = quantizationInitSymInt32(HALF_AWAY);

    float groupScales[3] = {commonScale, commonScale, commonScale};
    tensor_t *groupedPropLoss = runGroupedConv1dDxBackward(testQ, 3, 6, groupScales);

    size_t weightDims[] = {(size_t)kConv1dGroupedOutChannels, (size_t)kConv1dGroupedInChannels,
                           (size_t)kConv1dGroupedKernelSize};
    tensor_t *scalarWeightParam =
        buildSymInt32TensorExact(3, weightDims, kConv1dGroupedWMantissas, commonScale);
    parameter_t *scalarWeights = parameterInit(scalarWeightParam, NULL);

    size_t biasDims[] = {(size_t)kConv1dGroupedOutChannels};
    tensor_t *scalarBiasParam =
        buildSymInt32TensorExact(1, biasDims, kConv1dGroupedBiasMantissas, kConv1dGroupedBiasScale);
    parameter_t *scalarBias = parameterInit(scalarBiasParam, NULL);

    size_t inputDims[] = {(size_t)kConv1dGroupedBatch, (size_t)kConv1dGroupedInChannels,
                          (size_t)kConv1dGroupedInputLength};
    tensor_t *scalarInput =
        buildSymInt32TensorExact(3, inputDims, kConv1dGroupedXMantissas, kConv1dGroupedXScale);

    kernel_t *scalarKernel = reserveMemory(sizeof(kernel_t));
    initKernel(scalarKernel, (size_t)kConv1dGroupedKernelSize, VALID, 1, 1);
    layer_t *scalarLayer = buildBorrowedConv1dLayer(scalarWeights, scalarBias, scalarKernel, testQ);
    scalarLayer->config->conv1d->frozen = true;

    size_t lossDims[] = {(size_t)kConv1dGroupedBatch, (size_t)kConv1dGroupedOutChannels,
                         (size_t)kConv1dDxFwdOutLen};
    tensor_t *lossGrad =
        buildSymInt32TensorExact(3, lossDims, kConv1dDxLossMantissas, kConv1dDxLossScale);
    tensor_t *scalarPropLoss = makeFloatTensor(inputDims, 3, NULL);
    conv1dBackward(scalarLayer, scalarInput, lossGrad, scalarPropLoss);

    bool nonDegenerate = false;
    for (size_t i = 0; i < kDxPerChannelOutMantissas_len; i++) {
        if (((float *)groupedPropLoss->data)[i] != 0.0f) {
            nonDegenerate = true;
        }
    }
    TEST_ASSERT_TRUE_MESSAGE(nonDegenerate, "grouped dx twin is vacuously all-zero");
    TEST_ASSERT_EQUAL_FLOAT_ARRAY((float *)scalarPropLoss->data, (float *)groupedPropLoss->data,
                                  kDxPerChannelOutMantissas_len);
}

/* FLOAT32 dx path on the SAME grouped-SYM weight and SYM_INT32 lossGrad
 * (only propLossMath differs): the executeOp prologue dequantizes both
 * operands (grouped weight through the group-aware SYM->FLOAT32 cell, gated
 * by the FLOAT32 arm's groupedSymOperandPos declaration -- the arm-parity
 * lesson from PR2's final review), then the float scatter computes the
 * reference value with NO per-product rounding.
 *
 * Tolerance derivation (scatter error model, |err| <= 0.5*C*s_acc): the SYM
 * gold rounds ONCE PER CONTRIBUTING PRODUCT. C for this geometry (VALID
 * forward K=3, stride=1, L=6, Lout=4): dx[l] receives one product per
 * (outPos, k) with outPos + k == l, i.e. 1,2,3,3,2,1 for l = 0..5, times
 * Cout = 3 => C = 3,6,9,9,6,3; C_max = Cout*K = 9
 * (= kConv1dDxMaxProductsPerOut, generator-asserted). dx has NO bias seed
 * (unlike the forward's +1 term). Bound: 0.5*C_max*s_acc = 4.5 *
 * kDxPerChannelOutScale, plus 1e-6f headroom for float32 noise. */
void testConv1dBackwardGroupedDxFloatPathAgreesWithinTolerance(void) {
    quantization_t *floatQ = quantizationInitFloat();
    tensor_t *propLoss = runGroupedConv1dDxBackward(
        floatQ, (size_t)kPerChannelNumGroups, (size_t)kPerChannelGroupSize, kPerChannelWScales);

    float *captured = (float *)propLoss->data;
    const float tolerance =
        0.5f * (float)kConv1dDxMaxProductsPerOut * kDxPerChannelOutScale + 1e-6f;
    for (size_t i = 0; i < kDxPerChannelOutMantissas_len; i++) {
        float expected = (float)kDxPerChannelOutMantissas[i] * kDxPerChannelOutScale;
        TEST_ASSERT_FLOAT_WITHIN(tolerance, expected, captured[i]);
    }
}

/* ---- Group-quant PR3 (Task 3): PR2's two disclosed forward coverage gaps.
 * (a) SAME padding x quant-groups: groupSize=2 places a group boundary
 * between two VISITED taps INSIDE a padding-clipped window (out_pos=0 clips
 * to taps k=1,2; their w_idx 1|2 straddle the g0|g1 boundary --
 * generator-PROVED geometry, assert_same_padding_geometry) -- the
 * per-element group lookup's raison d'être: a lookup held per (oc, ic) row
 * (run-based assumption) misattributes tap k=2 and FAILS this gold. ---- */
void testConv1dForwardGroupedSamePaddingMatchesGold(void) {
    quantization_t *testQ = quantizationInitSymInt32(HALF_AWAY);
    tensor_t *input = NULL;
    layer_t *conv1d = buildGroupedConv1dFixtureLayer(
        testQ, (size_t)kSamePadNumGroups, (size_t)kSamePadGroupSize, kSamePadWScales, SAME, &input);

    size_t outputDims[] = {(size_t)kConv1dGroupedBatch, (size_t)kConv1dGroupedOutChannels,
                           (size_t)kSamePadOutLen};
    tensor_t *output = makeFloatTensor(outputDims, 3, NULL);

    conv1dForward(conv1d, input, output);

    float *captured = (float *)output->data;
    for (size_t i = 0; i < kSamePadOutMantissas_len; i++) {
        float expected = (float)kSamePadOutMantissas[i] * kSamePadOutScale;
        TEST_ASSERT_EQUAL_FLOAT(expected, captured[i]);
    }
}

/* (b) conv-groups=2 x quant-groups: weight [4, 2, 3] (24 elements), quant
 * groupSize=4 -- quant boundaries {4,8,12,16,20} vs channel-row starts
 * {6,12,18}: the two group systems disagree (generator-asserted), so a
 * kernel that bound the quant group to channel structure instead of flat
 * storage would diverge. Layer hand-wired (the borrowed builder is
 * groups=1-only). */
static layer_t *buildConvGroupsGroupedFixtureLayer(quantization_t *q, tensor_t **inputOut) {
    size_t weightDims[] = {(size_t)kCgOutChannels, (size_t)kCgInChannels / (size_t)kCgConvGroups,
                           (size_t)kCgKernelSize};
    size_t *ownedWeightDims = reserveMemory(3 * sizeof(size_t));
    memcpy(ownedWeightDims, weightDims, sizeof(weightDims));
    size_t *weightOrder = reserveMemory(3 * sizeof(size_t));
    setOrderOfDimsForNewTensor(3, weightOrder);
    shape_t *weightShape = reserveMemory(sizeof(shape_t));
    setShape(weightShape, ownedWeightDims, 3, weightOrder);
    tensor_t *weightsParam = initTensor(
        weightShape,
        quantizationInitSymGrouped(12, HALF_AWAY, (size_t)kCgNumGroups, (size_t)kCgGroupSize),
        NULL);
    byteConversion((uint8_t *)kCgWMantissas, 32, weightsParam->data, 12, kCgWMantissas_len);
    symQConfig_t *weightQC = weightsParam->quantization->qConfig;
    for (size_t g = 0; g < (size_t)kCgNumGroups; g++) {
        weightQC->scales[g] = kCgWScales[g];
    }
    parameter_t *weights = parameterInit(weightsParam, NULL);

    size_t biasDims[] = {(size_t)kCgOutChannels};
    tensor_t *biasParam = buildSymInt32TensorExact(1, biasDims, kCgBiasMantissas, kCgBiasScale);
    parameter_t *bias = parameterInit(biasParam, NULL);

    size_t inputDims[] = {(size_t)kConv1dGroupedBatch, (size_t)kCgInChannels,
                          (size_t)kCgInputLength};
    *inputOut = buildSymInt32TensorExact(3, inputDims, kCgXMantissas, kCgXScale);

    kernel_t *kernel = reserveMemory(sizeof(kernel_t));
    initKernel(kernel, (size_t)kCgKernelSize, VALID, 1, 1);

    conv1dConfig_t *cfg = reserveMemory(sizeof(conv1dConfig_t));
    initConv1dConfigWithWeightsAndBias(cfg, kernel, weights, bias, (size_t)kCgConvGroups, q, q, q,
                                       q);
    layerConfig_t *layerCfg = reserveMemory(sizeof(layerConfig_t));
    layerCfg->conv1d = cfg;
    layer_t *layer = reserveMemory(sizeof(layer_t));
    initLayer(layer, CONV1D, layerCfg);
    return layer;
}

void testConv1dForwardGroupedConvGroupsMatchesGold(void) {
    quantization_t *testQ = quantizationInitSymInt32(HALF_AWAY);
    tensor_t *input = NULL;
    layer_t *conv1d = buildConvGroupsGroupedFixtureLayer(testQ, &input);

    size_t outputDims[] = {(size_t)kConv1dGroupedBatch, (size_t)kCgOutChannels, (size_t)kCgOutLen};
    tensor_t *output = makeFloatTensor(outputDims, 3, NULL);

    conv1dForward(conv1d, input, output);

    float *captured = (float *)output->data;
    for (size_t i = 0; i < kCgOutMantissas_len; i++) {
        float expected = (float)kCgOutMantissas[i] * kCgOutScale;
        TEST_ASSERT_EQUAL_FLOAT(expected, captured[i]);
    }
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
    RUN_TEST(testConv1dFactoryFrozenElidesGrads);
    RUN_TEST(testConv1dFactoryDefaultAllocatesGrads);
    RUN_TEST(testConv1dBackwardFrozenTwinPropLossIdenticalGradsZero);
    RUN_TEST(testConv1dBackwardFrozenFactoryLayerRunsWithoutGradBuffers);
    RUN_TEST(testConv1dBackwardNullPropLossComputesGradsOnly);
    RUN_TEST(testConv1dForwardGroupedPerChannelMatchesGold);
    RUN_TEST(testConv1dForwardGroupedGeneralMatchesGold);
    RUN_TEST(testConv1dForwardGroupedEqualScalesBitIdenticalToScalar);
    RUN_TEST(testConv1dForwardGroupedPerChannelFloatPathAgreesWithinTolerance);
    RUN_TEST(testConv1dBackwardGroupedDxPerChannelMatchesGold);
    RUN_TEST(testConv1dBackwardGroupedDxGeneralMatchesGold);
    RUN_TEST(testConv1dBackwardGroupedDxEqualScalesBitIdenticalToScalar);
    RUN_TEST(testConv1dBackwardGroupedDxFloatPathAgreesWithinTolerance);
    RUN_TEST(testConv1dForwardGroupedSamePaddingMatchesGold);
    RUN_TEST(testConv1dForwardGroupedConvGroupsMatchesGold);
    return UNITY_END();
}
