#define SOURCE_FILE "UNIT_TEST_CONV1D_TRANSPOSED_API"

#include <math.h>

#include "Conv1dTransposed.h"
#include "Conv1dTransposedApi.h"
#include "Kernel.h"
#include "Layer.h"
#include "LayerCommon.h"
#include "LayerQuant.h"
#include "QuantizationApi.h"
#include "RNG.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "unity.h"

void setUp() {}
void tearDown() {}

/*! Returns the max |value| over a FLOAT32 tensor's data buffer. */
static float maxAbsFloat(const tensor_t *t) {
    const float *vals = (const float *)t->data;
    size_t n = t->shape->dimensions[0];
    for (size_t d = 1; d < t->shape->numberOfDimensions; d++) {
        n *= t->shape->dimensions[d];
    }
    float m = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float a = fabsf(vals[i]);
        if (a > m) {
            m = a;
        }
    }
    return m;
}

void testConv1dTransposedLayerInitBorrowingBuildsLayerWithCorrectShape(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);

    layer_t *layer = conv1dTransposedLayerInit(
        &(conv1dTransposedInit_t){
            .inChannels = 16,
            .outChannels = 8,
            .kernelSize = 5,
            .stride = 5,
            .padding = VALID,
            .bias = BIAS_TRUE,
        },
        &lq);

    TEST_ASSERT_NOT_NULL(layer);
    TEST_ASSERT_EQUAL_INT(CONV1D_TRANSPOSED, layer->type);

    conv1dTransposedConfig_t *cfg = layer->config->conv1dTransposed;
    TEST_ASSERT_NOT_NULL(cfg);
    TEST_ASSERT_FALSE(cfg->ownsQuantizations);

    /* Borrowing variant stores pointers verbatim */
    TEST_ASSERT_EQUAL_PTR(q, cfg->forwardQ);
    TEST_ASSERT_EQUAL_PTR(q, cfg->weightGradQ);
    TEST_ASSERT_EQUAL_PTR(q, cfg->biasGradQ);
    TEST_ASSERT_EQUAL_PTR(q, cfg->propLossQ);

    /* Weight shape: [inChannels, outChannels/groups, kernelSize] per Conv1dTransposed.h:12.
     * Note SWAP from Conv1d. */
    TEST_ASSERT_NOT_NULL(cfg->weights);
    tensor_t *weightTensor = cfg->weights->param;
    TEST_ASSERT_NOT_NULL(weightTensor);
    TEST_ASSERT_EQUAL_UINT(3, weightTensor->shape->numberOfDimensions);
    TEST_ASSERT_EQUAL_UINT(16, weightTensor->shape->dimensions[0]); /* inChannels */
    TEST_ASSERT_EQUAL_UINT(8, weightTensor->shape->dimensions[1]);  /* outChannels / groups */
    TEST_ASSERT_EQUAL_UINT(5, weightTensor->shape->dimensions[2]);  /* kernelSize */

    /* Bias shape: [outChannels] */
    TEST_ASSERT_NOT_NULL(cfg->bias);
    tensor_t *biasTensor = cfg->bias->param;
    TEST_ASSERT_EQUAL_UINT(1, biasTensor->shape->numberOfDimensions);
    TEST_ASSERT_EQUAL_UINT(8, biasTensor->shape->dimensions[0]);

    /* Kernel populated from init struct */
    TEST_ASSERT_NOT_NULL(cfg->kernel);
    TEST_ASSERT_EQUAL_UINT(5, cfg->kernel->size);
    TEST_ASSERT_EQUAL_INT(VALID, cfg->kernel->paddingType);
    TEST_ASSERT_EQUAL_UINT(5, cfg->kernel->stride);

    /* groups + outputPadding defaulted to 1 / 0 */
    TEST_ASSERT_EQUAL_UINT(1, cfg->groups);
    TEST_ASSERT_EQUAL_UINT(0, cfg->outputPadding);

    freeConv1dTransposedLayer(layer);
    freeQuantization(q);
}

void testConv1dTransposedLayerInitBorrowingBiasFalseLeavesBiasNull(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);

    layer_t *layer = conv1dTransposedLayerInit(
        &(conv1dTransposedInit_t){
            .inChannels = 4,
            .outChannels = 2,
            .kernelSize = 3,
            .bias = BIAS_FALSE,
        },
        &lq);

    conv1dTransposedConfig_t *cfg = layer->config->conv1dTransposed;
    TEST_ASSERT_NULL(cfg->bias);

    freeConv1dTransposedLayer(layer);
    freeQuantization(q);
}

void testConv1dTransposedLayerInitBorrowingOutputPaddingPropagatesToConfig(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);

    layer_t *layer = conv1dTransposedLayerInit(
        &(conv1dTransposedInit_t){
            .inChannels = 4,
            .outChannels = 2,
            .kernelSize = 3,
            .stride = 2,
            .outputPadding = 1,
            .bias = BIAS_TRUE,
        },
        &lq);

    conv1dTransposedConfig_t *cfg = layer->config->conv1dTransposed;
    TEST_ASSERT_EQUAL_UINT(1, cfg->outputPadding);
    TEST_ASSERT_EQUAL_UINT(2, cfg->kernel->stride);

    freeConv1dTransposedLayer(layer);
    freeQuantization(q);
}

void testConv1dTransposedLayerInitOwningDeepCopiesQuantizations(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);

    layer_t *layer = conv1dTransposedLayerInitOwning(
        &(conv1dTransposedInit_t){
            .inChannels = 8,
            .outChannels = 4,
            .kernelSize = 3,
            .bias = BIAS_TRUE,
        },
        &lq);

    conv1dTransposedConfig_t *cfg = layer->config->conv1dTransposed;

    TEST_ASSERT_NOT_EQUAL(q, cfg->forwardQ);
    TEST_ASSERT_NOT_EQUAL(q, cfg->weightGradQ);
    TEST_ASSERT_NOT_EQUAL(q, cfg->biasGradQ);
    TEST_ASSERT_NOT_EQUAL(q, cfg->propLossQ);
    TEST_ASSERT_EQUAL_INT(q->type, cfg->forwardQ->type);
    TEST_ASSERT_TRUE(cfg->ownsQuantizations);

    freeConv1dTransposedLayer(layer);
    freeQuantization(q);
}

void testConv1dTransposedLayerInitOwningFreesAllAllocationsWithoutLeak(void) {
    for (int i = 0; i < 5; i++) {
        quantization_t *q = quantizationInitFloat();
        layerQuant_t lq;
        layerQuantInitUniform(&lq, q);

        layer_t *layer = conv1dTransposedLayerInitOwning(
            &(conv1dTransposedInit_t){
                .inChannels = 4,
                .outChannels = 2,
                .kernelSize = 3,
                .bias = BIAS_TRUE,
            },
            &lq);

        freeConv1dTransposedLayer(layer);
        freeQuantization(q);
    }
    TEST_PASS();
}

void testConv1dTransposedLayerInitKeepsFloat32Grad(void) {
    quantization_t *fwd = quantizationInitFloat();
    quantization_t *bwd = quantizationInitSymInt32(HALF_AWAY);
    layerQuant_t lq = {
        .forwardMath = fwd,
        .backwardMath = bwd,
        .weightStorage = fwd,
        .biasStorage = fwd,
    };

    layer_t *layer = conv1dTransposedLayerInit(
        &(conv1dTransposedInit_t){
            .inChannels = 2,
            .outChannels = 4,
            .kernelSize = 3,
            .bias = BIAS_TRUE,
        },
        &lq);

    conv1dTransposedConfig_t *cfg = layer->config->conv1dTransposed;
    int weightGradType = cfg->weights->grad->quantization->type;
    int biasGradType = cfg->bias->grad->quantization->type;

    freeConv1dTransposedLayer(layer);
    freeQuantization(bwd);
    freeQuantization(fwd);

    TEST_ASSERT_EQUAL_INT_MESSAGE(FLOAT32, weightGradType,
                                  "Conv1dTransposed weight grad must stay FLOAT32");
    TEST_ASSERT_EQUAL_INT_MESSAGE(FLOAT32, biasGradType,
                                  "Conv1dTransposed bias grad must stay FLOAT32");
}

void testConv1dTransposedLayerInitDefaultWeightsWithinPyTorchBound(void) {
    /* PyTorch default ConvTranspose1d init: weight ~ U(-1/sqrt(fan_in),
     * +1/sqrt(fan_in)), bias drawn from the same uniform; for the
     * [inChannels, outPerGroup, kernelSize] layout fan_in = outPerGroup*kernelSize. */
    const size_t inChannels = 64, outChannels = 32, kernelSize = 8;
    const size_t fanIn = outChannels * kernelSize; /* groups=1 -> outPerGroup = outChannels */
    const float bound = 1.0f / sqrtf((float)fanIn);

    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);

    rngSetSeed(99);
    layer_t *layer = conv1dTransposedLayerInit(
        &(conv1dTransposedInit_t){
            .inChannels = inChannels,
            .outChannels = outChannels,
            .kernelSize = kernelSize,
            .bias = BIAS_TRUE,
        },
        &lq);

    conv1dTransposedConfig_t *cfg = layer->config->conv1dTransposed;
    float weightMaxAbs = maxAbsFloat(cfg->weights->param);
    float biasMaxAbs = maxAbsFloat(cfg->bias->param);

    freeConv1dTransposedLayer(layer);
    freeQuantization(q);

    TEST_ASSERT_TRUE_MESSAGE(weightMaxAbs <= bound * 1.001f,
                             "ConvTranspose default weights exceed PyTorch bound 1/sqrt(fan_in)");
    TEST_ASSERT_TRUE_MESSAGE(
        weightMaxAbs >= bound * 0.85f,
        "ConvTranspose default weights far below PyTorch bound -> wrong scale");
    TEST_ASSERT_TRUE_MESSAGE(biasMaxAbs > 0.0f,
                             "ConvTranspose default bias is zero (PyTorch draws from a uniform)");
    TEST_ASSERT_TRUE_MESSAGE(biasMaxAbs <= bound * 1.001f,
                             "ConvTranspose default bias exceeds PyTorch bound 1/sqrt(fan_in)");
}

void testConv1dTransposedLayerInitKaimingUniformOverrideUsesHeBound(void) {
    /* Explicit weightInit = {INIT_KAIMING_UNIFORM} -> He, default gain sqrt(2):
     * uniform(+/- sqrt(6)/sqrt(fan_in)). Wider than the default bound; bias
     * stays PyTorch default uniform. */
    const size_t inChannels = 64, outChannels = 32, kernelSize = 8;
    const size_t fanIn = outChannels * kernelSize;
    const float defaultBound = 1.0f / sqrtf((float)fanIn);
    const float heBound = sqrtf(6.0f) / sqrtf((float)fanIn);

    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);

    rngSetSeed(99);
    layer_t *layer = conv1dTransposedLayerInit(
        &(conv1dTransposedInit_t){
            .inChannels = inChannels,
            .outChannels = outChannels,
            .kernelSize = kernelSize,
            .bias = BIAS_TRUE,
            .weightInit = {INIT_KAIMING_UNIFORM},
        },
        &lq);

    conv1dTransposedConfig_t *cfg = layer->config->conv1dTransposed;
    float weightMaxAbs = maxAbsFloat(cfg->weights->param);
    float biasMaxAbs = maxAbsFloat(cfg->bias->param);

    freeConv1dTransposedLayer(layer);
    freeQuantization(q);

    TEST_ASSERT_TRUE_MESSAGE(weightMaxAbs > defaultBound,
                             "He override did not widen weights beyond the PyTorch default bound");
    TEST_ASSERT_TRUE_MESSAGE(weightMaxAbs <= heBound * 1.001f,
                             "He weights exceed the sqrt(6)/sqrt(fan_in) bound");
    TEST_ASSERT_TRUE_MESSAGE(biasMaxAbs <= defaultBound * 1.001f,
                             "Bias must stay PyTorch default uniform regardless of weight scheme");
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testConv1dTransposedLayerInitBorrowingBuildsLayerWithCorrectShape);
    RUN_TEST(testConv1dTransposedLayerInitBorrowingBiasFalseLeavesBiasNull);
    RUN_TEST(testConv1dTransposedLayerInitBorrowingOutputPaddingPropagatesToConfig);
    RUN_TEST(testConv1dTransposedLayerInitOwningDeepCopiesQuantizations);
    RUN_TEST(testConv1dTransposedLayerInitOwningFreesAllAllocationsWithoutLeak);
    RUN_TEST(testConv1dTransposedLayerInitKeepsFloat32Grad);
    RUN_TEST(testConv1dTransposedLayerInitDefaultWeightsWithinPyTorchBound);
    RUN_TEST(testConv1dTransposedLayerInitKaimingUniformOverrideUsesHeBound);
    return UNITY_END();
}
