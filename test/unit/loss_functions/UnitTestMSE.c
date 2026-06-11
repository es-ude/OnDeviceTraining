#include "MSE.h"
#include "QuantizationApi.h"
#include "Rounding.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "TensorConversion.h"
#include "unity.h"
#include <string.h>

void testMSEForward_MeanReturnsPerSampleMean() {
    /* Output (1D, 3 elements). Today B=1, so numFeaturesPerSample = 3. */
    size_t *outputDims = reserveMemory(1 * sizeof(size_t));
    outputDims[0] = 3;
    size_t *outputOrder = reserveMemory(1 * sizeof(size_t));
    setOrderOfDimsForNewTensor(1, outputOrder);
    shape_t *outputShape = reserveMemory(sizeof(shape_t));
    setShape(outputShape, outputDims, 1, outputOrder);
    tensor_t *output = initTensor(outputShape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(output, (float[]){1.f, 2.f, 3.f}, 3);

    size_t *labelDims = reserveMemory(1 * sizeof(size_t));
    labelDims[0] = 3;
    size_t *labelOrder = reserveMemory(1 * sizeof(size_t));
    setOrderOfDimsForNewTensor(1, labelOrder);
    shape_t *labelShape = reserveMemory(sizeof(shape_t));
    setShape(labelShape, labelDims, 1, labelOrder);
    tensor_t *label = initTensor(labelShape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(label, (float[]){2.f, 4.f, 6.f}, 3);

    float capturedLoss = mseLossForward(output, label, REDUCTION_MEAN);

    freeTensor(label);
    freeTensor(output);

    /* MEAN: ((1-2)² + (2-4)² + (3-6)²) / 3 = (1+4+9)/3 = 14/3 ≈ 4.667 */
    TEST_ASSERT_FLOAT_WITHIN(0.01f, 14.0f / 3.0f, capturedLoss);
}

void testMSEForward_SumReturnsRawSum() {
    size_t *outputDims = reserveMemory(1 * sizeof(size_t));
    outputDims[0] = 3;
    size_t *outputOrder = reserveMemory(1 * sizeof(size_t));
    setOrderOfDimsForNewTensor(1, outputOrder);
    shape_t *outputShape = reserveMemory(sizeof(shape_t));
    setShape(outputShape, outputDims, 1, outputOrder);
    tensor_t *output = initTensor(outputShape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(output, (float[]){1.f, 2.f, 3.f}, 3);

    size_t *labelDims = reserveMemory(1 * sizeof(size_t));
    labelDims[0] = 3;
    size_t *labelOrder = reserveMemory(1 * sizeof(size_t));
    setOrderOfDimsForNewTensor(1, labelOrder);
    shape_t *labelShape = reserveMemory(sizeof(shape_t));
    setShape(labelShape, labelDims, 1, labelOrder);
    tensor_t *label = initTensor(labelShape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(label, (float[]){2.f, 4.f, 6.f}, 3);

    float capturedLoss = mseLossForward(output, label, REDUCTION_SUM);

    freeTensor(label);
    freeTensor(output);

    /* SUM: 1 + 4 + 9 = 14 (no division). */
    TEST_ASSERT_FLOAT_WITHIN(0.01f, 14.0f, capturedLoss);
}

void testMSELossBackward_FloatWritesRawPerElementGrad() {
    size_t numberOfElements = 3;
    size_t dims[] = {numberOfElements};
    size_t orderOfDims[] = {0};
    shape_t shape = {.dimensions = dims, .orderOfDimensions = orderOfDims, .numberOfDimensions = 1};

    tensor_t modelOutput;
    quantization_t modelOutputQ;
    initFloat32Quantization(&modelOutputQ);
    float modelOutputData[] = {1.f, 2.f, -3.f};
    setTensorValues(&modelOutput, (uint8_t *)modelOutputData, &shape, &modelOutputQ, NULL);

    tensor_t label;
    quantization_t labelQ;
    initFloat32Quantization(&labelQ);
    float labelData[] = {-5.f, -4.f, 2.f};
    setTensorValues(&label, (uint8_t *)labelData, &shape, &labelQ, NULL);

    tensor_t result;
    quantization_t resultQ;
    initFloat32Quantization(&resultQ);
    float resultData[3];
    setTensorValues(&result, (uint8_t *)resultData, &shape, &resultQ, NULL);

    /* Raw per-element gradient: 2*(o-l). No /F division — that lives in
     * computeMeanScaleMSE applied at the optimizer step. */
    mseLossBackwardFloat(&modelOutput, &label, &result);

    /* delta = [6, 6, -5]; raw grad = 2*delta = [12, 12, -10] */
    float expected[] = {12.f, 12.f, -10.f};
    float *actual = (float *)result.data;
    for (size_t i = 0; i < 3; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expected[i], actual[i]);
    }
}

void testMSELossBackward_SymInt32WritesRawPerElementGrad() {
    size_t numberOfElements = 3;
    size_t dims[] = {numberOfElements};
    size_t orderOfDims[] = {0};
    shape_t shape = {.dimensions = dims, .orderOfDimensions = orderOfDims, .numberOfDimensions = 1};

    tensor_t modelOutput;
    quantization_t modelOutputQ;
    initFloat32Quantization(&modelOutputQ);
    float modelOutputData[] = {1.f, 2.f, -3.f};
    setTensorValues(&modelOutput, (uint8_t *)modelOutputData, &shape, &modelOutputQ, NULL);

    tensor_t modelOutputSymInt32;
    symInt32QConfig_t modelOutputSymInt32QC;
    initSymInt32QConfig(HALF_AWAY, &modelOutputSymInt32QC);
    quantization_t modelOutputSymInt32Q;
    initSymInt32Quantization(&modelOutputSymInt32QC, &modelOutputSymInt32Q);
    uint8_t modelOutputSymInt32Data[numberOfElements * sizeof(int32_t)];
    setTensorValuesForConversion(modelOutputSymInt32Data, &modelOutputSymInt32Q, &modelOutput,
                                 &modelOutputSymInt32);
    convertTensor(&modelOutput, &modelOutputSymInt32);

    tensor_t label;
    quantization_t labelQ;
    initFloat32Quantization(&labelQ);
    float labelData[] = {-5.f, -4.f, 2.f};
    setTensorValues(&label, (uint8_t *)labelData, &shape, &labelQ, NULL);

    tensor_t labelSymInt32;
    symInt32QConfig_t labelSymInt32QC;
    initSymInt32QConfig(HALF_AWAY, &labelSymInt32QC);
    quantization_t labelSymInt32Q;
    initSymInt32Quantization(&labelSymInt32QC, &labelSymInt32Q);
    uint8_t labelSymInt32Data[numberOfElements * sizeof(int32_t)];
    setTensorValuesForConversion(labelSymInt32Data, &labelSymInt32Q, &label, &labelSymInt32);
    convertTensor(&label, &labelSymInt32);

    tensor_t result;
    quantization_t resultQ;
    initFloat32Quantization(&resultQ);
    float resultData[numberOfElements];
    setTensorValues(&result, (uint8_t *)resultData, &shape, &resultQ, NULL);

    tensor_t resultSymInt32;
    symInt32QConfig_t resultSymInt32QC;
    initSymInt32QConfig(HALF_AWAY, &resultSymInt32QC);
    quantization_t resultSymInt32Q;
    initSymInt32Quantization(&resultSymInt32QC, &resultSymInt32Q);
    uint8_t resultSymInt32Data[numberOfElements * sizeof(int32_t)];
    memset(resultSymInt32Data, 0, numberOfElements * sizeof(int32_t));
    setTensorValuesForConversion(resultSymInt32Data, &resultSymInt32Q, &result, &resultSymInt32);

    mseLossBackward(&modelOutputSymInt32, &labelSymInt32, &resultSymInt32);
    convertTensor(&resultSymInt32, &result);

    /* Raw per-element gradient: same shape as float test, allow wider tolerance for fixed-point. */
    float expected[] = {12.f, 12.f, -10.f};
    float *actual = (float *)result.data;
    for (size_t i = 0; i < numberOfElements; i++) {
        TEST_ASSERT_FLOAT_WITHIN(0.5f, expected[i], actual[i]);
    }
}

void setUp() {}
void tearDown() {}

int main(void) {
    UNITY_BEGIN();

    RUN_TEST(testMSEForward_MeanReturnsPerSampleMean);
    RUN_TEST(testMSEForward_SumReturnsRawSum);

    RUN_TEST(testMSELossBackward_FloatWritesRawPerElementGrad);
    RUN_TEST(testMSELossBackward_SymInt32WritesRawPerElementGrad);

    return UNITY_END();
}
