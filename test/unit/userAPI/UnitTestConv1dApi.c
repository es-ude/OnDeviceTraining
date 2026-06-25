#define SOURCE_FILE "UNIT_TEST_CONV1D_API"

#include <math.h>

#include "Conv1d.h"
#include "Conv1dApi.h"
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

void testConv1dLayerInitBorrowingBuildsLayerWithCorrectShape(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);

    layer_t *layer = conv1dLayerInit(
        &(conv1dInit_t){
            .inChannels = 3,
            .outChannels = 4,
            .kernelSize = 5,
            .padding = VALID,
            .stride = 1,
            .dilation = 1,
            .groups = 1,
            .bias = BIAS_TRUE,
        },
        &lq);

    TEST_ASSERT_NOT_NULL(layer);
    TEST_ASSERT_EQUAL_INT(CONV1D, layer->type);

    conv1dConfig_t *cfg = layer->config->conv1d;
    TEST_ASSERT_NOT_NULL(cfg);
    TEST_ASSERT_FALSE(cfg->ownsQuantizations);

    /* Borrowing variant stores pointers verbatim */
    TEST_ASSERT_EQUAL_PTR(q, cfg->forwardQ);
    TEST_ASSERT_EQUAL_PTR(q, cfg->weightGradQ);
    TEST_ASSERT_EQUAL_PTR(q, cfg->biasGradQ);
    TEST_ASSERT_EQUAL_PTR(q, cfg->propLossQ);

    /* Weights allocated with shape [outChannels, inChannels/groups, kernelSize] */
    TEST_ASSERT_NOT_NULL(cfg->weights);
    tensor_t *weightTensor = cfg->weights->param;
    TEST_ASSERT_NOT_NULL(weightTensor);
    TEST_ASSERT_EQUAL_UINT(3, weightTensor->shape->numberOfDimensions);
    TEST_ASSERT_EQUAL_UINT(4, weightTensor->shape->dimensions[0]); /* outChannels */
    TEST_ASSERT_EQUAL_UINT(3, weightTensor->shape->dimensions[1]); /* inChannels / groups */
    TEST_ASSERT_EQUAL_UINT(5, weightTensor->shape->dimensions[2]); /* kernelSize */

    /* Bias allocated with shape [outChannels] */
    TEST_ASSERT_NOT_NULL(cfg->bias);
    tensor_t *biasTensor = cfg->bias->param;
    TEST_ASSERT_NOT_NULL(biasTensor);
    TEST_ASSERT_EQUAL_UINT(1, biasTensor->shape->numberOfDimensions);
    TEST_ASSERT_EQUAL_UINT(4, biasTensor->shape->dimensions[0]);

    /* Kernel populated from init struct */
    TEST_ASSERT_NOT_NULL(cfg->kernel);
    TEST_ASSERT_EQUAL_UINT(5, cfg->kernel->size);
    TEST_ASSERT_EQUAL_INT(VALID, cfg->kernel->paddingType);
    TEST_ASSERT_EQUAL_UINT(1, cfg->kernel->stride);
    TEST_ASSERT_EQUAL_UINT(1, cfg->kernel->dilation);

    /* groups defaulted to 1 explicitly via init */
    TEST_ASSERT_EQUAL_UINT(1, cfg->groups);

    freeConv1dLayer(layer);
    freeQuantization(q);
}

void testConv1dLayerInitBorrowingBiasDefaultResolvesToTrue(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);

    layer_t *layer = conv1dLayerInit(
        &(conv1dInit_t){
            .inChannels = 1,
            .outChannels = 2,
            .kernelSize = 3,
            /* .bias omitted → BIAS_DEFAULT (0) → resolves to true */
        },
        &lq);

    conv1dConfig_t *cfg = layer->config->conv1d;
    TEST_ASSERT_NOT_NULL(cfg->bias);

    freeConv1dLayer(layer);
    freeQuantization(q);
}

void testConv1dLayerInitBorrowingBiasFalseLeavesBiasNull(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);

    layer_t *layer = conv1dLayerInit(
        &(conv1dInit_t){
            .inChannels = 1,
            .outChannels = 2,
            .kernelSize = 3,
            .bias = BIAS_FALSE,
        },
        &lq);

    conv1dConfig_t *cfg = layer->config->conv1d;
    TEST_ASSERT_NULL(cfg->bias);

    freeConv1dLayer(layer);
    freeQuantization(q);
}

void testConv1dLayerInitBorrowingPaddingDefaultIsValid(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);

    layer_t *layer = conv1dLayerInit(
        &(conv1dInit_t){
            .inChannels = 1,
            .outChannels = 1,
            .kernelSize = 3,
            /* .padding omitted → VALID (enum value 0) */
            /* .stride, .dilation, .groups omitted → 1 (resolved from 0) */
        },
        &lq);

    conv1dConfig_t *cfg = layer->config->conv1d;
    TEST_ASSERT_EQUAL_INT(VALID, cfg->kernel->paddingType);
    TEST_ASSERT_EQUAL_UINT(1, cfg->kernel->stride);
    TEST_ASSERT_EQUAL_UINT(1, cfg->kernel->dilation);
    TEST_ASSERT_EQUAL_UINT(1, cfg->groups);

    freeConv1dLayer(layer);
    freeQuantization(q);
}

void testConv1dLayerInitOwningDeepCopiesQuantizations(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);

    layer_t *layer = conv1dLayerInitOwning(
        &(conv1dInit_t){
            .inChannels = 3,
            .outChannels = 4,
            .kernelSize = 5,
            .bias = BIAS_TRUE,
        },
        &lq);

    conv1dConfig_t *cfg = layer->config->conv1d;

    /* Owning variant: cfg->forwardQ is a fresh allocation, NOT the original q */
    TEST_ASSERT_NOT_EQUAL(q, cfg->forwardQ);
    TEST_ASSERT_NOT_EQUAL(q, cfg->weightGradQ);
    TEST_ASSERT_NOT_EQUAL(q, cfg->biasGradQ);
    TEST_ASSERT_NOT_EQUAL(q, cfg->propLossQ);

    /* But the copy has equal type to the original */
    TEST_ASSERT_EQUAL_INT(q->type, cfg->forwardQ->type);

    /* ownsQuantizations flag is set */
    TEST_ASSERT_TRUE(cfg->ownsQuantizations);

    freeConv1dLayer(layer);
    freeQuantization(q);
}

void testConv1dLayerInitOwningFreesAllAllocationsWithoutLeak(void) {
    /* Build + free 5 layers — if anything leaks, LSan catches it in CI. */
    for (int i = 0; i < 5; i++) {
        quantization_t *q = quantizationInitFloat();
        layerQuant_t lq;
        layerQuantInitUniform(&lq, q);

        layer_t *layer = conv1dLayerInitOwning(
            &(conv1dInit_t){
                .inChannels = 8,
                .outChannels = 4,
                .kernelSize = 3,
                .bias = BIAS_TRUE,
            },
            &lq);

        freeConv1dLayer(layer);
        freeQuantization(q);
    }
    TEST_PASS();
}

void testConv1dLayerInitKeepsFloat32GradEvenWithSymInt32BackwardMath(void) {
    /* Conv1d backward is FLOAT32-only; its grad must stay FLOAT32 regardless of
     * backwardMath, so the gradInit plumbing defaults Conv1d to FLOAT32. */
    quantization_t *fwd = quantizationInitFloat();
    quantization_t *bwd = quantizationInitSymInt32(HALF_AWAY);
    layerQuant_t lq = {
        .forwardMath = fwd,
        .backwardMath = bwd,
        .weightStorage = fwd,
        .biasStorage = fwd,
    };

    layer_t *layer = conv1dLayerInit(
        &(conv1dInit_t){
            .inChannels = 2,
            .outChannels = 4,
            .kernelSize = 3,
            .bias = BIAS_TRUE,
        },
        &lq);

    conv1dConfig_t *cfg = layer->config->conv1d;
    int weightGradType = cfg->weights->grad->quantization->type;
    int biasGradType = cfg->bias->grad->quantization->type;

    freeConv1dLayer(layer);
    freeQuantization(bwd);
    freeQuantization(fwd);

    TEST_ASSERT_EQUAL_INT_MESSAGE(
        FLOAT32, weightGradType, "Conv1d weight grad must stay FLOAT32 (backward is FLOAT32-only)");
    TEST_ASSERT_EQUAL_INT_MESSAGE(FLOAT32, biasGradType,
                                  "Conv1d bias grad must stay FLOAT32 (backward is FLOAT32-only)");
}

void testConv1dLayerInitDefaultWeightsWithinPyTorchBound(void) {
    /* PyTorch default Conv1d init: weight ~ U(-1/sqrt(fan_in), +1/sqrt(fan_in)),
     * bias ~ U(-1/sqrt(fan_in), +1/sqrt(fan_in)); fan_in = inPerGroup*kernelSize.
     * Today the factory uses gain=sqrt(2) (He) -> bound sqrt(6)/sqrt(fan_in),
     * ~2.45x too wide, and bias is zero. Both must fail here pre-fix. */
    const size_t inChannels = 16, outChannels = 64, kernelSize = 8;
    const size_t fanIn = inChannels * kernelSize; /* groups=1 */
    const float bound = 1.0f / sqrtf((float)fanIn);

    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);

    rngSetSeed(123);
    layer_t *layer = conv1dLayerInit(
        &(conv1dInit_t){
            .inChannels = inChannels,
            .outChannels = outChannels,
            .kernelSize = kernelSize,
            .bias = BIAS_TRUE,
        },
        &lq);

    conv1dConfig_t *cfg = layer->config->conv1d;
    float weightMaxAbs = maxAbsFloat(cfg->weights->param);
    float biasMaxAbs = maxAbsFloat(cfg->bias->param);

    freeConv1dLayer(layer);
    freeQuantization(q);

    /* Weights must lie inside the PyTorch bound (with float slack). */
    TEST_ASSERT_TRUE_MESSAGE(weightMaxAbs <= bound * 1.001f,
                             "Conv1d default weights exceed PyTorch bound 1/sqrt(fan_in)");
    /* And nearly reach it (a uniform of 8192 samples gets very close). */
    TEST_ASSERT_TRUE_MESSAGE(weightMaxAbs >= bound * 0.85f,
                             "Conv1d default weights far below PyTorch bound -> wrong scale");
    /* Bias must be drawn from the same uniform: nonzero and within bound. */
    TEST_ASSERT_TRUE_MESSAGE(biasMaxAbs > 0.0f,
                             "Conv1d default bias is zero (PyTorch draws it from a uniform)");
    TEST_ASSERT_TRUE_MESSAGE(biasMaxAbs <= bound * 1.001f,
                             "Conv1d default bias exceeds PyTorch bound 1/sqrt(fan_in)");
}

void testConv1dLayerInitKaimingUniformOverrideUsesHeBound(void) {
    /* Explicit weightInit = {INIT_KAIMING_UNIFORM} -> He init, default gain
     * sqrt(2): kaimingUniform(sqrt(2), fan_in) = uniform(+/- sqrt(6)/sqrt(fan_in)).
     * Must be wider than the PyTorch default bound (proves the override took
     * effect) yet within the He bound. */
    const size_t inChannels = 16, outChannels = 64, kernelSize = 8;
    const size_t fanIn = inChannels * kernelSize; /* groups=1 */
    const float defaultBound = 1.0f / sqrtf((float)fanIn);
    const float heBound = sqrtf(6.0f) / sqrtf((float)fanIn);

    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);

    rngSetSeed(123);
    layer_t *layer = conv1dLayerInit(
        &(conv1dInit_t){
            .inChannels = inChannels,
            .outChannels = outChannels,
            .kernelSize = kernelSize,
            .bias = BIAS_TRUE,
            .weightInit = {INIT_KAIMING_UNIFORM},
        },
        &lq);

    conv1dConfig_t *cfg = layer->config->conv1d;
    float weightMaxAbs = maxAbsFloat(cfg->weights->param);
    /* Bias is ALWAYS the PyTorch default uniform(+/- 1/sqrt(fan_in)),
     * independent of the weight scheme. */
    float biasMaxAbs = maxAbsFloat(cfg->bias->param);

    freeConv1dLayer(layer);
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
    RUN_TEST(testConv1dLayerInitBorrowingBuildsLayerWithCorrectShape);
    RUN_TEST(testConv1dLayerInitBorrowingBiasDefaultResolvesToTrue);
    RUN_TEST(testConv1dLayerInitBorrowingBiasFalseLeavesBiasNull);
    RUN_TEST(testConv1dLayerInitBorrowingPaddingDefaultIsValid);
    RUN_TEST(testConv1dLayerInitOwningDeepCopiesQuantizations);
    RUN_TEST(testConv1dLayerInitOwningFreesAllAllocationsWithoutLeak);
    RUN_TEST(testConv1dLayerInitKeepsFloat32GradEvenWithSymInt32BackwardMath);
    RUN_TEST(testConv1dLayerInitDefaultWeightsWithinPyTorchBound);
    RUN_TEST(testConv1dLayerInitKaimingUniformOverrideUsesHeBound);
    return UNITY_END();
}
