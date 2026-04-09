#define SOURCE_FILE "UNIT_TEST_TENSOR_API"

#include <stddef.h>
#include <string.h>

#include "TensorApi.h"
#include "Tensor.h"
#include "Quantization.h"
#include "unity.h"

void setUp() {}
void tearDown() {}

void testTensorInitWithDistribution_Zeros_InitializesProductOfDimsValues() {
    // dims = {2, 5} → product = 10, sum = 7
    // Bug: += gives 7, *= gives 10
    // Fill data with sentinel 42.0f, then ZEROS should overwrite exactly 10 values
    float data[10];
    for (size_t i = 0; i < 10; i++) {
        data[i] = 42.0f;
    }
    size_t dims[] = {2, 5};
    quantization_t q;
    initFloat32Quantization(&q);

    tensor_t *t = tensorInitWithDistribution(ZEROS, data, dims, 2, &q, NULL, 2, 5);

    // All 10 values should be zero
    float *values = (float *)t->data;
    for (size_t i = 0; i < 10; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-9f, 0.0f, values[i]);
    }
}

void testTensorInitWithDistribution_Ones_InitializesAllValues() {
    // dims = {3, 4} → product = 12, sum = 7
    // Fill data with 0.0f, then ONES should set exactly 12 values to 1.0f
    float data[12];
    memset(data, 0, sizeof(data));
    size_t dims[] = {3, 4};
    quantization_t q;
    initFloat32Quantization(&q);

    tensor_t *t = tensorInitWithDistribution(ONES, data, dims, 2, &q, NULL, 3, 4);

    float *values = (float *)t->data;
    for (size_t i = 0; i < 12; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-9f, 1.0f, values[i]);
    }
}

void testTensorInitWithDistribution_Normal_InitializesAllValues() {
    // dims = {4, 5} → product = 20, sum = 9
    // If only 9 values are initialized, remaining 11 stay at sentinel
    float data[20];
    float sentinel = -999.0f;
    for (size_t i = 0; i < 20; i++) {
        data[i] = sentinel;
    }
    size_t dims[] = {4, 5};
    quantization_t q;
    initFloat32Quantization(&q);

    tensor_t *t = tensorInitWithDistribution(NORMAL, data, dims, 2, &q, NULL, 4, 5);

    // With NORMAL distribution, values should NOT be the sentinel
    float *values = (float *)t->data;
    size_t sentinelCount = 0;
    for (size_t i = 0; i < 20; i++) {
        if (values[i] == sentinel) {
            sentinelCount++;
        }
    }
    // All 20 values should have been overwritten — none should remain as sentinel
    TEST_ASSERT_EQUAL_UINT(0, sentinelCount);
}

void testTensorInitWithDistribution_ShapeIsCorrect() {
    // Verify the resulting tensor has the correct shape dimensions
    float data[6] = {0};
    size_t dims[] = {2, 3};
    quantization_t q;
    initFloat32Quantization(&q);

    tensor_t *t = tensorInitWithDistribution(ZEROS, data, dims, 2, &q, NULL, 2, 3);

    TEST_ASSERT_EQUAL_UINT(2, t->shape->numberOfDimensions);
    size_t numElements = calcNumberOfElementsByTensor(t);
    TEST_ASSERT_EQUAL_UINT(6, numElements);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testTensorInitWithDistribution_Zeros_InitializesProductOfDimsValues);
    RUN_TEST(testTensorInitWithDistribution_Ones_InitializesAllValues);
    RUN_TEST(testTensorInitWithDistribution_Normal_InitializesAllValues);
    RUN_TEST(testTensorInitWithDistribution_ShapeIsCorrect);
    return UNITY_END();
}
