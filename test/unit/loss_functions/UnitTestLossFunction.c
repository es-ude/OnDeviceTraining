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

void testComputeMeanScaleMSE_ScaleIsOneOverNTimesF() {
    /* B=1, F=10 → scale = 1 / (N * F) = 1 / (2 * 10). */
    tensor_t *output = buildOutputShapeTensor2D(1, 10);
    float scale = computeMeanScaleMSE(/* N */ 2, output);
    freeTensor(output);
    TEST_ASSERT_FLOAT_WITHIN(1e-7f, 1.0f / 20.0f, scale);
}

void testComputeMeanScaleMSE_LargerInputsScaleDenominatorAccordingly() {
    /* B=1, F=784 (MNIST flattened) → scale = 1 / (32 * 784). */
    tensor_t *output = buildOutputShapeTensor2D(1, 784);
    float scale = computeMeanScaleMSE(/* N */ 32, output);
    freeTensor(output);
    TEST_ASSERT_FLOAT_WITHIN(1e-7f, 1.0f / (32.0f * 784.0f), scale);
}

void testComputeMeanScaleCE_ScaleIsOneOverN() {
    /* CE convention: 1 / totalSamples (class-axis is internal — F is unused). */
    tensor_t *output = buildOutputShapeTensor2D(1, 10);
    float scale = computeMeanScaleCE(/* N */ 2, output);
    freeTensor(output);
    TEST_ASSERT_FLOAT_WITHIN(1e-7f, 1.0f / 2.0f, scale);
}

void testComputeMeanScaleCE_FeatureCountDoesNotAffectScale() {
    /* F variation must NOT change CE's mean scale. */
    tensor_t *outputA = buildOutputShapeTensor2D(1, 10);
    tensor_t *outputB = buildOutputShapeTensor2D(1, 1000);
    float scaleA = computeMeanScaleCE(8, outputA);
    float scaleB = computeMeanScaleCE(8, outputB);
    freeTensor(outputA);
    freeTensor(outputB);
    TEST_ASSERT_EQUAL_FLOAT(scaleA, scaleB);
}

void testLossFunctionsVtable_ComputeMeanScaleDispatchesToFamilyImpl() {
    /* B=1, F=10. MSE expects 1/(2*10)=0.05; CE expects 1/2=0.5. */
    tensor_t *output = buildOutputShapeTensor2D(1, 10);

    float mseScale = lossFunctions[MSE].computeMeanScale(2, output);
    float ceScale = lossFunctions[CROSS_ENTROPY].computeMeanScale(2, output);

    freeTensor(output);

    /* Distinct values prove both slots wired with the right family-specific impl. */
    TEST_ASSERT_FLOAT_WITHIN(1e-7f, 0.05f, mseScale);
    TEST_ASSERT_FLOAT_WITHIN(1e-7f, 0.5f, ceScale);
}

void testLossFunctionsVtable_ForwardSlotBindsCrossEntropyDispatcher() {
    /* The vtable must route through the dtype dispatcher, not the raw float
     * impl — otherwise SYM_INT32 mantissas reaching CE forward are silently
     * reinterpreted as float bit patterns (no error, garbage loss). */
    TEST_ASSERT_TRUE(lossFunctions[CROSS_ENTROPY].forward == crossEntropyForward);
}

/* === defaultLossConfig === */

void testDefaultLossConfig_MSE() {
    lossConfig_t cfg = defaultLossConfig(MSE);
    TEST_ASSERT_EQUAL_INT(MSE, cfg.funcType);
    TEST_ASSERT_EQUAL_INT(REDUCTION_MEAN, cfg.backwardReduction);
    TEST_ASSERT_NULL(cfg.classWeights);
}

void testDefaultLossConfig_CrossEntropy() {
    lossConfig_t cfg = defaultLossConfig(CROSS_ENTROPY);
    TEST_ASSERT_EQUAL_INT(CROSS_ENTROPY, cfg.funcType);
    TEST_ASSERT_EQUAL_INT(REDUCTION_MEAN, cfg.backwardReduction);
    TEST_ASSERT_NULL(cfg.classWeights);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testComputeMeanScaleMSE_ScaleIsOneOverNTimesF);
    RUN_TEST(testComputeMeanScaleMSE_LargerInputsScaleDenominatorAccordingly);
    RUN_TEST(testComputeMeanScaleCE_ScaleIsOneOverN);
    RUN_TEST(testComputeMeanScaleCE_FeatureCountDoesNotAffectScale);
    RUN_TEST(testLossFunctionsVtable_ComputeMeanScaleDispatchesToFamilyImpl);
    RUN_TEST(testLossFunctionsVtable_ForwardSlotBindsCrossEntropyDispatcher);
    RUN_TEST(testDefaultLossConfig_MSE);
    RUN_TEST(testDefaultLossConfig_CrossEntropy);
    return UNITY_END();
}
