#define SOURCE_FILE "UNIT_TEST_CONV1D_API"

#include "Conv1d.h"
#include "Conv1dApi.h"
#include "Kernel.h"
#include "Layer.h"
#include "LayerCommon.h"
#include "LayerQuant.h"
#include "QuantizationApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "unity.h"

void setUp() {}
void tearDown() {}

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

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testConv1dLayerInitBorrowingBuildsLayerWithCorrectShape);
    RUN_TEST(testConv1dLayerInitBorrowingBiasDefaultResolvesToTrue);
    RUN_TEST(testConv1dLayerInitBorrowingBiasFalseLeavesBiasNull);
    RUN_TEST(testConv1dLayerInitBorrowingPaddingDefaultIsValid);
    RUN_TEST(testConv1dLayerInitOwningDeepCopiesQuantizations);
    RUN_TEST(testConv1dLayerInitOwningFreesAllAllocationsWithoutLeak);
    RUN_TEST(testConv1dLayerInitKeepsFloat32GradEvenWithSymInt32BackwardMath);
    return UNITY_END();
}
