#include "CrossEntropy.h"
#include "LossFunction.h"
#include "MSE.h"
#include "QuantizationApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "unity.h"

void setUp() {}
void tearDown() {}

/* Build a model-output tensor with the requested shape so computeMeanScale*
 * can derive numFeaturesPerSample = numElements / dimensions[0] itself. The
 * tensor's data is uninitialized — only the shape is read. */
static tensor_t *buildOutputShapeTensor2D(size_t d0, size_t d1) {
    size_t *dims = reserveMemory(2 * sizeof(size_t));
    dims[0] = d0;
    dims[1] = d1;
    size_t *order = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 2, order);
    return initTensor(shape, quantizationInitFloat(), NULL);
}

/* Through the vtable: MSE scales by 1 / (N * F) with F = numElements / B, CE by 1 / N whatever
 * the feature count. The B = 4 row separates F from the element count. */
void testComputeMeanScaleDispatchesPerLossFamily(void) {
    const struct {
        size_t batch, features, totalSamples;
        float mse, ce;
    } cases[] = {
        {1, 10, 2, 1.0f / 20.0f, 1.0f / 2.0f},
        {1, 784, 32, 1.0f / (32.0f * 784.0f), 1.0f / 32.0f},
        {4, 10, 8, 1.0f / 80.0f, 1.0f / 8.0f},
    };
    for (size_t i = 0; i < sizeof cases / sizeof cases[0]; i++) {
        tensor_t *output = buildOutputShapeTensor2D(cases[i].batch, cases[i].features);
        float mseScale = lossFunctions[MSE].computeMeanScale(cases[i].totalSamples, output);
        float ceScale =
            lossFunctions[CROSS_ENTROPY].computeMeanScale(cases[i].totalSamples, output);
        freeTensor(output);
        TEST_ASSERT_FLOAT_WITHIN(1e-9f, cases[i].mse, mseScale);
        TEST_ASSERT_FLOAT_WITHIN(1e-7f, cases[i].ce, ceScale);
    }
}

void testLossFunctionsVtable_ForwardSlotBindsCrossEntropyDispatcher() {
    /* The vtable must route through the dtype dispatcher, not the raw float
     * impl — otherwise SYM_INT32 mantissas reaching CE forward are silently
     * reinterpreted as float bit patterns (no error, garbage loss). */
    TEST_ASSERT_TRUE(lossFunctions[CROSS_ENTROPY].forward == crossEntropyForward);
}

void testDefaultLossConfigIsMeanReducedAndUnweighted(void) {
    const lossFuncType_t types[] = {MSE, CROSS_ENTROPY};
    for (size_t i = 0; i < sizeof types / sizeof types[0]; i++) {
        lossConfig_t cfg = defaultLossConfig(types[i]);
        TEST_ASSERT_EQUAL_INT(types[i], cfg.funcType);
        TEST_ASSERT_EQUAL_INT(REDUCTION_MEAN, cfg.backwardReduction);
        TEST_ASSERT_NULL(cfg.classWeights);
    }
}

/* The integer mean divisor (the BFP mean scale's input): CE = N, MSE = N * F,
 * and each family's float scale is exactly 1.0f / (float)divisor. */
void testComputeMeanDivisor_CeIsNAndMseIsNTimesF(void) {
    tensor_t *output = buildOutputShapeTensor2D(1, 10);
    size_t mse = lossFunctions[MSE].computeMeanDivisor(6, output);
    size_t ce = lossFunctions[CROSS_ENTROPY].computeMeanDivisor(6, output);
    float mseScale = computeMeanScaleMSE(6, output);
    float ceScale = computeMeanScaleCE(6, output);
    freeTensor(output);
    TEST_ASSERT_EQUAL_size_t(60, mse);
    TEST_ASSERT_EQUAL_size_t(6, ce);
    TEST_ASSERT_TRUE(mseScale == 1.0f / (float)mse);
    TEST_ASSERT_TRUE(ceScale == 1.0f / (float)ce);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testComputeMeanScaleDispatchesPerLossFamily);
    RUN_TEST(testLossFunctionsVtable_ForwardSlotBindsCrossEntropyDispatcher);
    RUN_TEST(testDefaultLossConfigIsMeanReducedAndUnweighted);
    RUN_TEST(testComputeMeanDivisor_CeIsNAndMseIsNTimesF);
    return UNITY_END();
}
