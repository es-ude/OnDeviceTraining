#include "Quantization.h"
#include "Tensor.h"
#include "unity.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

void setUp() {}
void tearDown() {}

void test_calcBytesPerElement_Bool() {
    quantization_t q;
    initBoolQuantization(&q);
    TEST_ASSERT_EQUAL_size_t(1, calcBytesPerElement(&q));
}

void test_calcBitsPerElement_Bool() {
    quantization_t q;
    initBoolQuantization(&q);
    TEST_ASSERT_EQUAL_size_t(1, calcBitsPerElement(&q));
}

void test_tensorBoolGet_AfterZeroInit_AllFalse() {
    /* N=10 → 2 bytes; data is zero-initialized stack array */
    uint8_t data[2] = {0, 0};
    size_t dims[] = {10};
    size_t orderOfDims[] = {0};
    shape_t shape = {.numberOfDimensions = 1, .dimensions = dims, .orderOfDimensions = orderOfDims};
    quantization_t q;
    initBoolQuantization(&q);
    tensor_t t;
    setTensorValues(&t, data, &shape, &q, NULL);

    for (size_t i = 0; i < 10; i++) {
        TEST_ASSERT_FALSE(tensorBoolGet(&t, i));
    }
}

void test_tensorBoolSetGet_RoundTrip_KnownPattern() {
    /* Pattern with mix of byte boundaries (i=7→8) and last-byte padding. */
    uint8_t data[2] = {0, 0};
    size_t dims[] = {10};
    size_t orderOfDims[] = {0};
    shape_t shape = {.numberOfDimensions = 1, .dimensions = dims, .orderOfDimensions = orderOfDims};
    quantization_t q;
    initBoolQuantization(&q);
    tensor_t t;
    setTensorValues(&t, data, &shape, &q, NULL);

    const bool expected[10] = {true, false, true, true, false, false, true, false, false, true};
    for (size_t i = 0; i < 10; i++) {
        tensorBoolSet(&t, i, expected[i]);
    }
    for (size_t i = 0; i < 10; i++) {
        TEST_ASSERT_EQUAL(expected[i], tensorBoolGet(&t, i));
    }
}

void test_tensorBoolSet_PreservesNeighborBits() {
    /* Set bit 3, verify bits 0,1,2,4,5,6,7 are unchanged (zero). */
    uint8_t data[1] = {0};
    size_t dims[] = {8};
    size_t orderOfDims[] = {0};
    shape_t shape = {.numberOfDimensions = 1, .dimensions = dims, .orderOfDimensions = orderOfDims};
    quantization_t q;
    initBoolQuantization(&q);
    tensor_t t;
    setTensorValues(&t, data, &shape, &q, NULL);

    tensorBoolSet(&t, 3, true);

    /* Bit 3 only → byte value should be 0b00001000 = 8 (LSB-first). */
    TEST_ASSERT_EQUAL_UINT8(0b00001000, data[0]);
}

void test_tensorBoolSet_PaddingBitsRemainZero_N3() {
    /* N=3: bits 3..7 are padding, must stay zero after setting bits 0,1,2. */
    uint8_t data[1] = {0};
    size_t dims[] = {3};
    size_t orderOfDims[] = {0};
    shape_t shape = {.numberOfDimensions = 1, .dimensions = dims, .orderOfDimensions = orderOfDims};
    quantization_t q;
    initBoolQuantization(&q);
    tensor_t t;
    setTensorValues(&t, data, &shape, &q, NULL);

    tensorBoolSet(&t, 0, true);
    tensorBoolSet(&t, 1, true);
    tensorBoolSet(&t, 2, true);

    /* Bits 0,1,2 set; bits 3..7 zero → 0b00000111 = 7. */
    TEST_ASSERT_EQUAL_UINT8(0b00000111, data[0]);
}

void test_tensorBoolSet_ClearsBit() {
    /* Start with byte=0xFF, clear bit 2, expect 0xFB. */
    uint8_t data[1] = {0xFF};
    size_t dims[] = {8};
    size_t orderOfDims[] = {0};
    shape_t shape = {.numberOfDimensions = 1, .dimensions = dims, .orderOfDimensions = orderOfDims};
    quantization_t q;
    initBoolQuantization(&q);
    tensor_t t;
    setTensorValues(&t, data, &shape, &q, NULL);

    tensorBoolSet(&t, 2, false);

    TEST_ASSERT_EQUAL_UINT8(0b11111011, data[0]);
}

void test_copyTensorBool_AllBitsMatch() {
    /* Source: 10 elements with known pattern. */
    uint8_t srcData[2];
    uint8_t dstData[2] = {0, 0};

    /* Build src with pattern via accessor (round-tripped previously). */
    size_t srcDims[] = {10};
    size_t srcOrder[] = {0};
    shape_t srcShape = {
        .numberOfDimensions = 1, .dimensions = srcDims, .orderOfDimensions = srcOrder};
    quantization_t qSrc;
    initBoolQuantization(&qSrc);
    tensor_t src;
    /* Zero src first so set-only test is deterministic. */
    memset(srcData, 0, sizeof(srcData));
    setTensorValues(&src, srcData, &srcShape, &qSrc, NULL);

    const bool pattern[10] = {true, true, false, true, false, true, false, false, true, true};
    for (size_t i = 0; i < 10; i++) {
        tensorBoolSet(&src, i, pattern[i]);
    }

    /* Build dst with matching size; copyTensor will fill shape/quant/data. */
    size_t dstDims[1];
    size_t dstOrder[1];
    shape_t dstShape = {
        .numberOfDimensions = 0, .dimensions = dstDims, .orderOfDimensions = dstOrder};
    quantization_t qDst;
    initFloat32Quantization(&qDst); /* will be overwritten by copyQuantization */
    tensor_t dst;
    setTensorValues(&dst, dstData, &dstShape, &qDst, NULL);

    copyTensor(&dst, &src);

    /* Assert dst->quantization->type now BOOL. */
    TEST_ASSERT_EQUAL_INT(BOOL, dst.quantization->type);

    /* Assert bit-for-bit equality via accessor. */
    for (size_t i = 0; i < 10; i++) {
        TEST_ASSERT_EQUAL(pattern[i], tensorBoolGet(&dst, i));
    }
}

void test_printTensorBool_Smoke() {
    /* Just verify no crash on a small BOOL tensor; output goes to stdout. */
    uint8_t data[1] = {0b00000101}; /* bits 0 and 2 set */
    size_t dims[] = {3};
    size_t orderOfDims[] = {0};
    shape_t shape = {.numberOfDimensions = 1, .dimensions = dims, .orderOfDimensions = orderOfDims};
    quantization_t q;
    initBoolQuantization(&q);
    tensor_t t;
    setTensorValues(&t, data, &shape, &q, NULL);

    printTensor(&t); /* must return cleanly */
    TEST_PASS();
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_calcBytesPerElement_Bool);
    RUN_TEST(test_calcBitsPerElement_Bool);
    RUN_TEST(test_tensorBoolGet_AfterZeroInit_AllFalse);
    RUN_TEST(test_tensorBoolSetGet_RoundTrip_KnownPattern);
    RUN_TEST(test_tensorBoolSet_PreservesNeighborBits);
    RUN_TEST(test_tensorBoolSet_PaddingBitsRemainZero_N3);
    RUN_TEST(test_tensorBoolSet_ClearsBit);
    RUN_TEST(test_copyTensorBool_AllBitsMatch);
    RUN_TEST(test_printTensorBool_Smoke);
    return UNITY_END();
}
