#define SOURCE_FILE "UNIT_TEST_POOL1D_API"

#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "ArithmeticType.h"
#include "AvgPool1d.h"
#include "DeathTest.h"
#include "InferenceApi.h"
#include "Kernel.h"
#include "Layer.h"
#include "LayerQuant.h"
#include "MaxPool1d.h"
#include "Pool1dApi.h"
#include "QuantizationApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "unity.h"

void setUp() {}
void tearDown() {}

/* ============================================================================
 * MaxPool1d
 * ========================================================================== */

void testMaxPool1dLayerInitBorrowingBuildsLayerWithKernelAndArgmax(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);

    /* For K=2 S=2 VALID on inputLength=64, outputLength = (64 - 2)/2 + 1 = 32 */
    layer_t *layer = maxPool1dLayerInit(
        &(maxPool1dInit_t){
            .kernelSize = 2,
            .stride = 2,
            .inputChannels = 16,
            .inputLength = 64,
        },
        &lq);

    TEST_ASSERT_NOT_NULL(layer);
    TEST_ASSERT_EQUAL_INT(MAXPOOL1D, layer->type);

    maxPool1dConfig_t *cfg = layer->config->maxPool1d;
    TEST_ASSERT_NOT_NULL(cfg);
    TEST_ASSERT_FALSE(cfg->ownsQuantizations);

    TEST_ASSERT_EQUAL_PTR(q, cfg->outputQ);
    TEST_ASSERT_EQUAL_PTR(q, cfg->propLossQ);
    TEST_ASSERT_EQUAL_INT(ARITH_FLOAT32, cfg->forwardMath.type);
    TEST_ASSERT_EQUAL_INT(ARITH_FLOAT32, cfg->propLossMath.type);

    /* Kernel correctness */
    TEST_ASSERT_NOT_NULL(cfg->kernel);
    TEST_ASSERT_EQUAL_UINT(2, cfg->kernel->size);
    TEST_ASSERT_EQUAL_INT(VALID, cfg->kernel->paddingType);
    TEST_ASSERT_EQUAL_UINT(2, cfg->kernel->stride);
    TEST_ASSERT_EQUAL_UINT(1, cfg->kernel->dilation);

    /* Argmax tensor shape: [1, inputChannels, outputLength] = [1, 16, 32] */
    TEST_ASSERT_NOT_NULL(cfg->argmaxIndices);
    TEST_ASSERT_EQUAL_UINT(3, cfg->argmaxIndices->shape->numberOfDimensions);
    TEST_ASSERT_EQUAL_UINT(1, cfg->argmaxIndices->shape->dimensions[0]);
    TEST_ASSERT_EQUAL_UINT(16, cfg->argmaxIndices->shape->dimensions[1]);
    TEST_ASSERT_EQUAL_UINT(32, cfg->argmaxIndices->shape->dimensions[2]);
    TEST_ASSERT_EQUAL_INT(INT32, cfg->argmaxIndices->quantization->type);

    freeMaxPool1dLayer(layer);
    freeQuantization(q);
}

void testMaxPool1dLayerInitBorrowingStrideDefaultsToKernelSize(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);

    /* stride omitted → defaults to kernelSize per PyTorch convention */
    layer_t *layer = maxPool1dLayerInit(
        &(maxPool1dInit_t){
            .kernelSize = 4,
            .inputChannels = 1,
            .inputLength = 16,
        },
        &lq);

    maxPool1dConfig_t *cfg = layer->config->maxPool1d;
    TEST_ASSERT_EQUAL_UINT(4, cfg->kernel->size);
    TEST_ASSERT_EQUAL_UINT(4, cfg->kernel->stride);
    /* outputLength = (16 - 4)/4 + 1 = 4 */
    TEST_ASSERT_EQUAL_UINT(4, cfg->argmaxIndices->shape->dimensions[2]);

    freeMaxPool1dLayer(layer);
    freeQuantization(q);
}

void testMaxPool1dLayerInitOwningDeepCopiesTwoQuantizations(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);

    layer_t *layer = maxPool1dLayerInitOwning(
        &(maxPool1dInit_t){
            .kernelSize = 2,
            .stride = 2,
            .inputChannels = 4,
            .inputLength = 8,
        },
        &lq);

    maxPool1dConfig_t *cfg = layer->config->maxPool1d;
    TEST_ASSERT_NOT_EQUAL(q, cfg->outputQ);
    TEST_ASSERT_NOT_EQUAL(q, cfg->propLossQ);
    TEST_ASSERT_EQUAL_INT(q->type, cfg->outputQ->type);
    TEST_ASSERT_EQUAL_INT(ARITH_FLOAT32, cfg->forwardMath.type);
    TEST_ASSERT_TRUE(cfg->ownsQuantizations);

    freeMaxPool1dLayer(layer);
    freeQuantization(q);
}

void testMaxPool1dLayerInitOwningRepeatedBuildFreeNoLeak(void) {
    for (int i = 0; i < 5; i++) {
        quantization_t *q = quantizationInitFloat();
        layerQuant_t lq;
        layerQuantInitUniform(&lq, q);

        layer_t *layer = maxPool1dLayerInitOwning(
            &(maxPool1dInit_t){
                .kernelSize = 2,
                .stride = 2,
                .inputChannels = 4,
                .inputLength = 8,
            },
            &lq);

        freeMaxPool1dLayer(layer);
        freeQuantization(q);
    }
    TEST_PASS();
}

/* ============================================================================
 * AvgPool1d
 * ========================================================================== */

void testAvgPool1dLayerInitBorrowingBuildsLayerWithKernel(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);

    layer_t *layer = avgPool1dLayerInit(
        &(avgPool1dInit_t){
            .kernelSize = 5,
            .stride = 5,
        },
        &lq);

    TEST_ASSERT_NOT_NULL(layer);
    TEST_ASSERT_EQUAL_INT(AVGPOOL1D, layer->type);

    avgPool1dConfig_t *cfg = layer->config->avgPool1d;
    TEST_ASSERT_NOT_NULL(cfg);
    TEST_ASSERT_FALSE(cfg->ownsQuantizations);

    TEST_ASSERT_EQUAL_PTR(q, cfg->outputQ);
    TEST_ASSERT_EQUAL_PTR(q, cfg->propLossQ);
    TEST_ASSERT_EQUAL_INT(ARITH_FLOAT32, cfg->forwardMath.type);
    TEST_ASSERT_EQUAL_INT(ARITH_FLOAT32, cfg->propLossMath.type);

    TEST_ASSERT_NOT_NULL(cfg->kernel);
    TEST_ASSERT_EQUAL_UINT(5, cfg->kernel->size);
    TEST_ASSERT_EQUAL_INT(VALID, cfg->kernel->paddingType);
    TEST_ASSERT_EQUAL_UINT(5, cfg->kernel->stride);

    freeAvgPool1dLayer(layer);
    freeQuantization(q);
}

void testAvgPool1dLayerInitBorrowingStrideDefaultsToKernelSize(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);

    layer_t *layer = avgPool1dLayerInit(
        &(avgPool1dInit_t){
            .kernelSize = 3,
            /* stride omitted → kernelSize=3 */
        },
        &lq);

    avgPool1dConfig_t *cfg = layer->config->avgPool1d;
    TEST_ASSERT_EQUAL_UINT(3, cfg->kernel->stride);

    freeAvgPool1dLayer(layer);
    freeQuantization(q);
}

void testAvgPool1dLayerInitOwningDeepCopiesTwoQuantizations(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);

    layer_t *layer = avgPool1dLayerInitOwning(
        &(avgPool1dInit_t){
            .kernelSize = 2,
            .stride = 2,
        },
        &lq);

    avgPool1dConfig_t *cfg = layer->config->avgPool1d;
    TEST_ASSERT_NOT_EQUAL(q, cfg->outputQ);
    TEST_ASSERT_NOT_EQUAL(q, cfg->propLossQ);
    TEST_ASSERT_TRUE(cfg->ownsQuantizations);

    freeAvgPool1dLayer(layer);
    freeQuantization(q);
}

/* ============================================================================
 * MaxPool1d argmax grow-on-demand (#152 PR3b, spec §6.7)
 * ========================================================================== */

static tensor_t *buildPoolTensor(size_t batch, size_t channels, size_t length, const float *src) {
    size_t *dims = reserveMemory(3 * sizeof(size_t));
    dims[0] = batch;
    dims[1] = channels;
    dims[2] = length;
    size_t *order = reserveMemory(3 * sizeof(size_t));
    setOrderOfDimsForNewTensor(3, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 3, order);
    tensor_t *t = initTensor(shape, quantizationInitFloat(), NULL);
    if (src != NULL) {
        tensorFillFromFloatBuffer(t, src, batch * channels * length);
    }
    return t;
}

/* K=2, S=2 over L=4 -> Lout=2; every row has its own argmax pattern so a
 * backward that read another row's indices would scatter to wrong cells. */
static const float POOL_ROWS[4 * 4] = {
    1.0f,  5.0f,  2.0f,  0.0f,  /* maxima 5, 2    argmax {1, 2} */
    7.0f,  3.0f,  0.0f,  9.0f,  /* maxima 7, 9    argmax {0, 3} */
    -1.0f, -4.0f, -8.0f, -6.0f, /* maxima -1, -6  argmax {0, 3} */
    0.5f,  2.5f,  -3.0f, -2.0f, /* maxima 2.5, -2 argmax {1, 3} */
};

static layer_t *buildGrowthPool(quantization_t *q) {
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);
    return maxPool1dLayerInit(
        &(maxPool1dInit_t){.kernelSize = 2, .stride = 2, .inputChannels = 1, .inputLength = 4},
        &lq);
}

void testMaxPool1dArgmaxGrowsOnceAndTracksBatch(void) {
    quantization_t *q = quantizationInitFloat();
    layer_t *layer = buildGrowthPool(q);
    maxPool1dConfig_t *cfg = layer->config->maxPool1d;
    tensor_t *argmax = cfg->argmaxIndices;

    tensor_t *in1 = buildPoolTensor(1, 1, 4, POOL_ROWS);
    tensor_t *out1 = buildPoolTensor(1, 1, 2, NULL);
    maxPool1dForward(layer, in1, out1);
    size_t batchAfterB1 = argmax->shape->dimensions[0];
    size_t capacityAfterB1 = cfg->argmaxCapacity;

    tensor_t *in4 = buildPoolTensor(4, 1, 4, POOL_ROWS);
    tensor_t *out4 = buildPoolTensor(4, 1, 2, NULL);
    maxPool1dForward(layer, in4, out4);
    size_t batchAfterB4 = argmax->shape->dimensions[0];
    size_t capacityAfterB4 = cfg->argmaxCapacity;
    const uint8_t *grownData = argmax->data;
    int32_t argmaxB4[8];
    memcpy(argmaxB4, argmax->data, sizeof argmaxB4);

    /* B=2 with rows 1 and 3: shrink in place, no new reservation. */
    tensor_t *in2 = buildPoolTensor(2, 1, 4, POOL_ROWS + 4);
    memcpy(in2->data + 4 * sizeof(float), POOL_ROWS + 12, 4 * sizeof(float));
    tensor_t *out2 = buildPoolTensor(2, 1, 2, NULL);
    maxPool1dForward(layer, in2, out2);
    size_t batchAfterB2 = argmax->shape->dimensions[0];
    bool b2KeptBuffer = argmax->data == grownData;
    int32_t argmaxB2[4];
    memcpy(argmaxB2, argmax->data, sizeof argmaxB2);

    /* Backward at B=2 must scatter through THIS call's rows (non-uniform
     * lossGrad: a row mix-up moves values, not just positions). */
    tensor_t *lossGrad = buildPoolTensor(2, 1, 2, (float[]){1.0f, 2.0f, 3.0f, 4.0f});
    tensor_t *propLoss = buildPoolTensor(2, 1, 4, NULL);
    maxPool1dBackward(layer, in2, lossGrad, propLoss);
    float gotPropLoss[8];
    memcpy(gotPropLoss, propLoss->data, sizeof gotPropLoss);

    /* Back to B=4: still within capacity -> same buffer again. */
    maxPool1dForward(layer, in4, out4);
    bool b4AgainKeptBuffer = argmax->data == grownData;
    size_t capacityAfterB4Again = cfg->argmaxCapacity;

    freeTensor(propLoss);
    freeTensor(lossGrad);
    freeTensor(out2);
    freeTensor(in2);
    freeTensor(out4);
    freeTensor(in4);
    freeTensor(out1);
    freeTensor(in1);
    freeMaxPool1dLayer(layer);
    freeQuantization(q);

    TEST_ASSERT_EQUAL_size_t(1, batchAfterB1);
    TEST_ASSERT_EQUAL_size_t(2, capacityAfterB1);
    TEST_ASSERT_EQUAL_size_t(4, batchAfterB4);
    TEST_ASSERT_EQUAL_size_t(8, capacityAfterB4);
    const int32_t expectedB4[8] = {1, 2, 0, 3, 0, 3, 1, 3};
    TEST_ASSERT_EQUAL_INT32_ARRAY(expectedB4, argmaxB4, 8);
    TEST_ASSERT_EQUAL_size_t(2, batchAfterB2);
    TEST_ASSERT_TRUE_MESSAGE(b2KeptBuffer, "shrinking B must not re-reserve");
    const int32_t expectedB2[4] = {0, 3, 1, 3};
    TEST_ASSERT_EQUAL_INT32_ARRAY(expectedB2, argmaxB2, 4);
    const float expectedPropLoss[8] = {1.0f, 0.0f, 0.0f, 2.0f, 0.0f, 3.0f, 0.0f, 4.0f};
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expectedPropLoss, gotPropLoss, 8);
    TEST_ASSERT_TRUE_MESSAGE(b4AgainKeptBuffer, "regrowing within capacity must not re-reserve");
    TEST_ASSERT_EQUAL_size_t(8, capacityAfterB4Again);
}

#ifdef ODT_MEM_PROFILE
/* Growth frees the old argmax block (CI runs detect_leaks=0, so only this
 * exact counter catches a dropped free): the one growing forward, factory
 * [1, 1, 2] -> [4, 1, 2], changes the live-byte count by exactly the new
 * block minus the old one. Every tensor is built before and freed after the
 * two readings; a FLOAT32-in, FLOAT32-out forward allocates nothing else. */
void testMaxPool1dArgmaxGrowthReturnsTheOldBlock(void) {
    quantization_t *q = quantizationInitFloat();
    layer_t *layer = buildGrowthPool(q);
    maxPool1dConfig_t *cfg = layer->config->maxPool1d;
    tensor_t *in4 = buildPoolTensor(4, 1, 4, POOL_ROWS);
    tensor_t *out4 = buildPoolTensor(4, 1, 2, NULL);
    size_t capacityBefore = cfg->argmaxCapacity;

    size_t before = memProfileMark();
    maxPool1dForward(layer, in4, out4);
    size_t after = memProfileMark();
    size_t capacityAfter = cfg->argmaxCapacity;

    freeTensor(out4);
    freeTensor(in4);
    freeMaxPool1dLayer(layer);
    freeQuantization(q);

    TEST_ASSERT_EQUAL_size_t(2, capacityBefore);
    TEST_ASSERT_EQUAL_size_t(8, capacityAfter);
    TEST_ASSERT_EQUAL_size_t_MESSAGE(before + (8 - 2) * sizeof(int32_t), after,
                                     "growth must free the old argmax block");
}
#endif /* ODT_MEM_PROFILE */

void testInferenceAtBatch4GrowsFactoryMaxPool(void) {
    quantization_t *q = quantizationInitFloat();
    layer_t *layer = buildGrowthPool(q);
    layer_t *model[1] = {layer};
    tensor_t *in4 = buildPoolTensor(4, 1, 4, POOL_ROWS);

    tensor_t *out = inference(model, 1, in4);
    size_t outRank = out->shape->numberOfDimensions;
    size_t outBatch = out->shape->dimensions[0];
    float gotOut[8];
    memcpy(gotOut, out->data, sizeof gotOut);
    size_t argmaxBatch = layer->config->maxPool1d->argmaxIndices->shape->dimensions[0];

    freeTensor(out);
    freeTensor(in4);
    freeMaxPool1dLayer(layer);
    freeQuantization(q);

    TEST_ASSERT_EQUAL_size_t(3, outRank);
    TEST_ASSERT_EQUAL_size_t(4, outBatch);
    const float expectedOut[8] = {5.0f, 2.0f, 7.0f, 9.0f, -1.0f, -6.0f, 2.5f, -2.0f};
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expectedOut, gotOut, 8);
    TEST_ASSERT_EQUAL_size_t(4, argmaxBatch);
}

void testMaxPool1dArgmaxGrowthRejectsSizeOverflow(void) {
    /* B = SIZE_MAX / 8 + 2 with C * Lout = 2 * 4 = 8: B * 8 wraps to exactly
     * 8, the factory's capacity -- an unchecked multiply would skip growth,
     * stamp dims[0] = B and let the kernel walk B rows of a 16-float buffer.
     * The shapes are stack-built: the guard fires before any data is read. */
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);
    layer_t *layer = maxPool1dLayerInit(
        &(maxPool1dInit_t){.kernelSize = 2, .stride = 2, .inputChannels = 2, .inputLength = 8},
        &lq);
    size_t hugeBatch = SIZE_MAX / 8 + 2;
    float inData[16] = {0};
    float outData[8] = {0};
    size_t inDims[3] = {hugeBatch, 2, 8};
    size_t outDims[3] = {hugeBatch, 2, 4};
    size_t order[3] = {0, 1, 2};
    shape_t inShape = {.numberOfDimensions = 3, .dimensions = inDims, .orderOfDimensions = order};
    shape_t outShape = {.numberOfDimensions = 3, .dimensions = outDims, .orderOfDimensions = order};
    quantization_t floatQ;
    initFloat32Quantization(&floatQ);
    tensor_t input = {.data = (uint8_t *)inData, .shape = &inShape, .quantization = &floatQ};
    tensor_t output = {.data = (uint8_t *)outData, .shape = &outShape, .quantization = &floatQ};

    ASSERT_EXITS_WITH_FAILURE(maxPool1dForward(layer, &input, &output));

    freeMaxPool1dLayer(layer);
    freeQuantization(q);
}

static tensor_t *buildInt32PoolTensor(size_t batch, size_t channels, size_t length) {
    size_t *dims = reserveMemory(3 * sizeof(size_t));
    dims[0] = batch;
    dims[1] = channels;
    dims[2] = length;
    size_t *order = reserveMemory(3 * sizeof(size_t));
    setOrderOfDimsForNewTensor(3, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 3, order);
    return initTensor(shape, quantizationInitInt32(), NULL);
}

void testMaxPool1dArgmaxAdoptsASwappedInBuffer(void) {
    /* Capacity is trusted only while argmaxCapacityData == argmaxIndices->data
     * (MaxPool1d.h). An argmax swapped in by hand after growth must be adopted
     * at its OWN element count: the stale capacity 8 would skip growth at B=4
     * and let the kernel write 8 indices into the 2-element buffer. */
    quantization_t *q = quantizationInitFloat();
    layer_t *layer = buildGrowthPool(q);
    maxPool1dConfig_t *cfg = layer->config->maxPool1d;
    tensor_t *factoryArgmax = cfg->argmaxIndices;
    tensor_t *in1 = buildPoolTensor(1, 1, 4, POOL_ROWS);
    tensor_t *out1 = buildPoolTensor(1, 1, 2, NULL);
    tensor_t *in4 = buildPoolTensor(4, 1, 4, POOL_ROWS);
    tensor_t *out4 = buildPoolTensor(4, 1, 2, NULL);
    maxPool1dForward(layer, in4, out4); /* the factory argmax grows to capacity 8 */

    tensor_t *swapped = buildInt32PoolTensor(1, 1, 2);
    uintptr_t swappedOriginalAddr = (uintptr_t)swapped->data;
    cfg->argmaxIndices = swapped;
    /* B = 1 fits the swapped buffer: adopted at its 2 elements, no growth. */
    maxPool1dForward(layer, in1, out1);
    size_t capacityAtB1 = cfg->argmaxCapacity;
    bool trackedAtB1 = cfg->argmaxCapacityData == swapped->data;
    bool keptAtB1 = (uintptr_t)swapped->data == swappedOriginalAddr;
    /* B = 4 exceeds the adopted 2 elements: the swapped argmax grows. */
    maxPool1dForward(layer, in4, out4);
    bool grewAtB4 = (uintptr_t)swapped->data != swappedOriginalAddr;
    size_t capacityAtB4 = cfg->argmaxCapacity;
    bool trackedAtB4 = cfg->argmaxCapacityData == swapped->data;
    int32_t argmaxB4[8];
    memcpy(argmaxB4, swapped->data, sizeof argmaxB4);

    cfg->argmaxIndices = factoryArgmax; /* the layer frees its own argmax */
    freeTensor(swapped);
    freeTensor(out4);
    freeTensor(in4);
    freeTensor(out1);
    freeTensor(in1);
    freeMaxPool1dLayer(layer);
    freeQuantization(q);

    TEST_ASSERT_EQUAL_size_t(2, capacityAtB1);
    TEST_ASSERT_TRUE_MESSAGE(trackedAtB1, "a swapped-in argmax must be adopted");
    TEST_ASSERT_TRUE_MESSAGE(keptAtB1, "B=1 fits the adopted buffer: no re-reservation");
    TEST_ASSERT_TRUE_MESSAGE(grewAtB4, "B=4 exceeds the adopted capacity: must grow");
    TEST_ASSERT_EQUAL_size_t(8, capacityAtB4);
    TEST_ASSERT_TRUE(trackedAtB4);
    const int32_t expectedB4[8] = {1, 2, 0, 3, 0, 3, 1, 3};
    TEST_ASSERT_EQUAL_INT32_ARRAY(expectedB4, argmaxB4, 8);
}

void testMaxPool1dArgmaxGrowthFailsFastWhenReservationFails(void) {
    /* B = SIZE_MAX / 4 with C * Lout = 1 * 1: B * C * Lout * 4 bytes =
     * SIZE_MAX - 3 does NOT wrap, but no allocator can serve it. reserveMemory
     * returns NULL (its own size-wrap guard under ODT_MEM_PROFILE, calloc
     * failure otherwise) and growth must fail fast instead of installing a
     * NULL argmax (spec §6.4, §6.7). Stack-built shapes: nothing is read. */
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);
    layer_t *layer = maxPool1dLayerInit(
        &(maxPool1dInit_t){.kernelSize = 2, .stride = 2, .inputChannels = 1, .inputLength = 2},
        &lq);
    size_t hugeBatch = SIZE_MAX / 4;
    float inData[2] = {0};
    float outData[1] = {0};
    size_t inDims[3] = {hugeBatch, 1, 2};
    size_t outDims[3] = {hugeBatch, 1, 1};
    size_t order[3] = {0, 1, 2};
    shape_t inShape = {.numberOfDimensions = 3, .dimensions = inDims, .orderOfDimensions = order};
    shape_t outShape = {.numberOfDimensions = 3, .dimensions = outDims, .orderOfDimensions = order};
    quantization_t floatQ;
    initFloat32Quantization(&floatQ);
    tensor_t input = {.data = (uint8_t *)inData, .shape = &inShape, .quantization = &floatQ};
    tensor_t output = {.data = (uint8_t *)outData, .shape = &outShape, .quantization = &floatQ};

    ASSERT_EXITS_WITH_FAILURE(maxPool1dForward(layer, &input, &output));

    freeMaxPool1dLayer(layer);
    freeQuantization(q);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testMaxPool1dLayerInitBorrowingBuildsLayerWithKernelAndArgmax);
    RUN_TEST(testMaxPool1dLayerInitBorrowingStrideDefaultsToKernelSize);
    RUN_TEST(testMaxPool1dLayerInitOwningDeepCopiesTwoQuantizations);
    RUN_TEST(testMaxPool1dLayerInitOwningRepeatedBuildFreeNoLeak);
    RUN_TEST(testAvgPool1dLayerInitBorrowingBuildsLayerWithKernel);
    RUN_TEST(testAvgPool1dLayerInitBorrowingStrideDefaultsToKernelSize);
    RUN_TEST(testAvgPool1dLayerInitOwningDeepCopiesTwoQuantizations);
    RUN_TEST(testMaxPool1dArgmaxGrowsOnceAndTracksBatch);
#ifdef ODT_MEM_PROFILE
    RUN_TEST(testMaxPool1dArgmaxGrowthReturnsTheOldBlock);
#endif
    RUN_TEST(testInferenceAtBatch4GrowsFactoryMaxPool);
    RUN_TEST(testMaxPool1dArgmaxGrowthRejectsSizeOverflow);
    RUN_TEST(testMaxPool1dArgmaxAdoptsASwappedInBuffer);
    RUN_TEST(testMaxPool1dArgmaxGrowthFailsFastWhenReservationFails);
    return UNITY_END();
}
