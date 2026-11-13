#include "Div.h"
#include "Quantization.h"
#include "Tensor.h"
#include "TestScale.h"
#include "unity.h"

/* The stored quotient scale is a / b rounded ONCE onto the 15-bit dyadic grid
 * (NEAREST): 0.5 / 0.75 = 2/3 = 21845.33 * 2^-15 -> 21845 * 2^-15, not the
 * float32 quotient 0.6666667 stored raw. */
void testDivSymInt32ScaleIsCanonicalQuotient() {
    int32_t aData[] = {6, -9};
    int32_t bData[] = {2, 3};
    int32_t outData[] = {0, 0};
    size_t dims[] = {2};
    size_t orderOfDims[] = {0};
    shape_t shape = {.dimensions = dims, .orderOfDimensions = orderOfDims, .numberOfDimensions = 1};

    symInt32QConfig_t aQC;
    initSymInt32QConfig(HALF_AWAY, &aQC);
    aQC.scale = odtTestScale(0.5f);
    quantization_t aQ;
    initSymInt32Quantization(&aQC, &aQ);
    tensor_t aTensor;
    setTensorValues(&aTensor, (uint8_t *)aData, &shape, &aQ, NULL);

    symInt32QConfig_t bQC;
    initSymInt32QConfig(HALF_AWAY, &bQC);
    bQC.scale = odtTestScale(0.75f);
    quantization_t bQ;
    initSymInt32Quantization(&bQC, &bQ);
    tensor_t bTensor;
    setTensorValues(&bTensor, (uint8_t *)bData, &shape, &bQ, NULL);

    symInt32QConfig_t outQC;
    initSymInt32QConfig(HALF_AWAY, &outQC);
    quantization_t outQ;
    initSymInt32Quantization(&outQC, &outQ);
    tensor_t outTensor;
    setTensorValues(&outTensor, (uint8_t *)outData, &shape, &outQ, NULL);

    divSymInt32Tensors(&aTensor, &bTensor, &outTensor);

    int32_t expected[] = {3, -3};
    TEST_ASSERT_EQUAL_INT32_ARRAY(expected, outData, 2);
    TEST_ASSERT_TRUE(odtScaleToFloat(outQC.scale) == 21845.0f / 32768.0f);
}

void setUp() {}
void tearDown() {}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testDivSymInt32ScaleIsCanonicalQuotient);
    return UNITY_END();
}
