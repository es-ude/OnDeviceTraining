#include "Kernel.h"
#include "unity.h"

void testkernelGetWindowIndices1d() {
    kernel_t kernel = {.dilation = 1, .paddingType = VALID, .size = 2, .stride = 1};

    size_t indices[kernel.size];

    kernelGetWindowIndices1d(&kernel, 0 * kernel.stride, indices);
    size_t expected0[] = {0, 1};
    TEST_ASSERT_EQUAL_size_t_ARRAY(expected0, indices, kernel.size);

    kernelGetWindowIndices1d(&kernel, 1 * kernel.stride, indices);
    size_t expected1[] = {1, 2};
    TEST_ASSERT_EQUAL_size_t_ARRAY(expected1, indices, kernel.size);

    kernelGetWindowIndices1d(&kernel, 2 * kernel.stride, indices);
    size_t expected2[] = {2, 3};
    TEST_ASSERT_EQUAL_size_t_ARRAY(expected2, indices, kernel.size);
}

void testkernelGetWindowIndices1dWithStride() {
    kernel_t kernel = {.dilation = 1, .paddingType = VALID, .size = 2, .stride = 2};

    size_t indices[kernel.size];

    kernelGetWindowIndices1d(&kernel, 0 * kernel.stride, indices);
    size_t expected0[] = {0, 1};
    TEST_ASSERT_EQUAL_size_t_ARRAY(expected0, indices, kernel.size);

    kernelGetWindowIndices1d(&kernel, 1 * kernel.stride, indices);
    size_t expected1[] = {2, 3};
    TEST_ASSERT_EQUAL_size_t_ARRAY(expected1, indices, kernel.size);

    kernelGetWindowIndices1d(&kernel, 2 * kernel.stride, indices);
    size_t expected2[] = {4, 5};
    TEST_ASSERT_EQUAL_size_t_ARRAY(expected2, indices, kernel.size);
}

void testkernelGetWindowIndices1dWithStrideAndDilation() {
    kernel_t kernel = {.dilation = 2, .paddingType = VALID, .size = 2, .stride = 3};

    size_t indices[kernel.size];

    kernelGetWindowIndices1d(&kernel, 0 * kernel.stride, indices);
    size_t expected0[] = {0, 2};
    TEST_ASSERT_EQUAL_size_t_ARRAY(expected0, indices, kernel.size);

    kernelGetWindowIndices1d(&kernel, 1 * kernel.stride, indices);
    size_t expected1[] = {3, 5};
    TEST_ASSERT_EQUAL_size_t_ARRAY(expected1, indices, kernel.size);

    kernelGetWindowIndices1d(&kernel, 2 * kernel.stride, indices);
    size_t expected2[] = {6, 8};
    TEST_ASSERT_EQUAL_size_t_ARRAY(expected2, indices, kernel.size);
}

void testCalculatePaddingSize1d() {
    kernel_t kernel = {.dilation = 2, .paddingType = SAME, .size = 3, .stride = 2};

    size_t inputLengthPerChannel = 6;

    size_t actual = kernelCalculatePaddingSize1d(inputLengthPerChannel, &kernel);
    size_t expected = 9;
    TEST_ASSERT_EQUAL_size_t(expected, actual);
}

void setUp() {}
void tearDown() {}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testkernelGetWindowIndices1d);
    RUN_TEST(testkernelGetWindowIndices1dWithStride);
    RUN_TEST(testkernelGetWindowIndices1dWithStrideAndDilation);

    RUN_TEST(testCalculatePaddingSize1d);
    UNITY_END();
}
