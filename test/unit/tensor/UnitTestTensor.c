#include "DTypes.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "unity.h"

#include <stdlib.h>
#include <string.h>

void testGetBitmask() {
    uint8_t startbit = 1;
    uint8_t endbit = 4;
    uint8_t bitmask = getBitmask(startbit, endbit);
    uint8_t expected = 0b00001110;
    TEST_ASSERT_EQUAL(expected, bitmask);
}

void testGetBitmask2() {
    uint8_t startbit = 10;
    uint8_t endbit = 14;
    uint8_t bitmask = getBitmask(startbit, endbit);
    uint8_t expected = 0b00111100;
    TEST_ASSERT_EQUAL(expected, bitmask);
}

void testReadByte() {
    uint8_t startbit = 1;
    uint8_t endbit = 4;
    uint8_t data = 0b00101010;
    uint8_t actual = readByte(data, startbit, endbit);
    uint8_t expected = 0b00000101;
    TEST_ASSERT_EQUAL(expected, actual);
}

void testWriteByte() {
    uint8_t existing_data = 0b00000101;
    uint8_t data = 0b00000101;
    uint8_t newData = writeByte(existing_data, data, 3, 7);
    uint8_t expected = 0b00101101;
    TEST_ASSERT_EQUAL_UINT8(expected, newData);
}

void testWriteByte2() {
    uint8_t existing_data = 0b00000101;
    uint8_t data = 0b00010101;
    uint8_t newData = writeByte(existing_data, data, 3, 11);
    uint8_t expected = 0b10101101;
    TEST_ASSERT_EQUAL_UINT8(expected, newData);
}

void testByteFlattening() {
    // {1, 2, 78}
    uint8_t dataIn[] = {0b000000001, 0b00000110, 0b00111000, 0b00000001};
    size_t dataInBits = 9;

    size_t dataOutBits = 19;
    size_t numValues = 3;
    size_t numBytesDataOut = (dataOutBits * numValues - 1) / 8 + 1;
    uint8_t *dataOut = reserveMemory(numBytesDataOut);
    byteConversion(dataIn, dataInBits, dataOut, dataOutBits, numValues);

    /* CAPTURE before free. */
    uint8_t captured[numBytesDataOut];
    memcpy(captured, dataOut, numBytesDataOut);
    freeReservedMemory(dataOut);

    uint8_t expectedBytes[] = {0b000000001, 0b00000000, 0b00011000, 0b00000000,
                               0b10000000,  0b00010011, 0b00000000, 0b00000000};

    TEST_ASSERT_EQUAL_UINT8_ARRAY(expectedBytes, captured, numBytesDataOut);
}

void testByteFlattening2() {
    // {1, 2, 78}
    uint8_t dataIn[] = {0b000000001, 0b00000000, 0b00011000, 0b00000000,
                        0b10000000,  0b00010011, 0b00000000, 0b00000000};
    size_t dataInBits = 19;

    size_t dataOutBits = 9;
    size_t numValues = 3;
    size_t numBytesDataOut = (dataOutBits * numValues - 1) / 8 + 1;
    uint8_t *dataOut = reserveMemory(numBytesDataOut);
    byteConversion(dataIn, dataInBits, dataOut, dataOutBits, numValues);

    /* CAPTURE before free. */
    uint8_t captured[numBytesDataOut];
    memcpy(captured, dataOut, numBytesDataOut);
    freeReservedMemory(dataOut);

    uint8_t expectedBytes[] = {0b000000001, 0b00000110, 0b00111000, 0b00000001};

    TEST_ASSERT_EQUAL_UINT8_ARRAY(expectedBytes, captured, numBytesDataOut);
}

void testByteFlattening3() {
    // {1, 2, 78}
    uint8_t dataIn[] = {0b000000001, 0b00000000, 0b00001100, 0b00000000,
                        0b00000100,  0b00001011, 0b00000000, 0b00000000};
    size_t dataInBits = 8;

    size_t dataOutBits = 4;
    size_t numValues = 8;
    size_t numBytesDataOut = (dataOutBits * numValues - 1) / 8 + 1;
    uint8_t *dataOut = reserveMemory(numBytesDataOut);
    byteConversion(dataIn, dataInBits, dataOut, dataOutBits, numValues);

    /* CAPTURE before free. */
    uint8_t captured[numBytesDataOut];
    memcpy(captured, dataOut, numBytesDataOut);
    freeReservedMemory(dataOut);

    uint8_t expectedBytes[] = {0b000000001, 0b00001100, 0b10110100, 0b00000000};

    TEST_ASSERT_EQUAL_UINT8_ARRAY(expectedBytes, captured, numBytesDataOut);
}

void testByteFlattening4() {
    uint8_t dataIn[] = {0b11010000, 0b11101110, 0b01101111, 0b00000000};
    size_t dataInBits = 5;
    size_t dataOutBits = 8;
    size_t numBytesDataOut = 6;
    uint8_t *dataOut = reserveMemory(numBytesDataOut);
    byteConversion(dataIn, dataInBits, dataOut, dataOutBits, numBytesDataOut);

    /* CAPTURE before free. */
    uint8_t captured[numBytesDataOut];
    memcpy(captured, dataOut, numBytesDataOut);
    freeReservedMemory(dataOut);

    uint8_t expectedBytes[] = {16, 22, 27, 31, 6, 0};
    TEST_ASSERT_EQUAL_UINT8_ARRAY(expectedBytes, captured, numBytesDataOut);
}

void testByteFlattening5() {
    uint8_t dataIn[] = {0b11010000, 0b11101110, 0b01101111, 0b00000000};
    size_t dataInBits = 5;
    size_t dataOutBits = 32;
    size_t numValues = 6;
    size_t numBytesDataOut = 4 * numValues;
    uint8_t dataOut[numBytesDataOut];
    ;
    byteConversion(dataIn, dataInBits, dataOut, dataOutBits, numValues);
    uint8_t expectedBytes[] = {16, 0, 0, 0, 22, 0, 0, 0, 27, 0, 0, 0,
                               31, 0, 0, 0, 6,  0, 0, 0, 0,  0, 0, 0};
    TEST_ASSERT_EQUAL_UINT8_ARRAY(expectedBytes, dataOut, numBytesDataOut);
}

/* Regression for an ASan-detected heap-buffer-overflow surfaced by
 * testLinearBackwardFloatWithMismatchedQuantizations on the float32 -> asym8
 * conversion path. byteConversion's inner while-loop keeps iterating after
 * the output for a value is full to consume remaining input bits, accessing
 * dataOut[dataOutIndex] one byte past the end of the buffer. The OOB write
 * is a no-op on the value (writeByte with startbit==endbit), so the bug is
 * invisible to value assertions and only fires under ASan. */
void testByteConversion_NarrowingInt32ToByte_DoesNotOverreadOutputBuffer() {
    /* Two int32 values, little-endian: 0xAABBCCDD, 0xEEFF1122. */
    uint8_t dataIn[8] = {0xDD, 0xCC, 0xBB, 0xAA, 0x22, 0x11, 0xFF, 0xEE};
    size_t dataInBits = 32;
    size_t dataOutBits = 8;
    size_t numValues = 2;
    size_t numBytesDataOut = (numValues * dataOutBits - 1) / 8 + 1; /* = 2 */

    uint8_t *dataOut = reserveMemory(numBytesDataOut);
    byteConversion(dataIn, dataInBits, dataOut, dataOutBits, numValues);

    uint8_t captured[numBytesDataOut];
    memcpy(captured, dataOut, numBytesDataOut);
    freeReservedMemory(dataOut);

    uint8_t expectedBytes[2] = {0xDD, 0x22};
    TEST_ASSERT_EQUAL_UINT8_ARRAY(expectedBytes, captured, numBytesDataOut);
}

/* Companion to the narrowing test above, exercising the sub-byte path of #86.
 * 32-bit input -> 4-bit output, 4 values packed into 2 bytes. The same
 * loop-control bug shows up: after the last value's bits are written, the
 * inner while-loop continues consuming input bits and accesses
 * dataOut[dataOutIndex] one byte past the end of the buffer. */
void testByteConversion_NarrowingInt32ToSubByte_DoesNotOverreadOutputBuffer() {
    /* Four int32 values, low 4 bits matter: 0xA, 0xB, 0xC, 0xD. */
    uint8_t dataIn[16] = {
        0x0A, 0, 0, 0, 0x0B, 0, 0, 0, 0x0C, 0, 0, 0, 0x0D, 0, 0, 0,
    };
    size_t dataInBits = 32;
    size_t dataOutBits = 4;
    size_t numValues = 4;
    size_t numBytesDataOut = (numValues * dataOutBits - 1) / 8 + 1; /* = 2 */

    uint8_t *dataOut = reserveMemory(numBytesDataOut);
    byteConversion(dataIn, dataInBits, dataOut, dataOutBits, numValues);

    uint8_t captured[numBytesDataOut];
    memcpy(captured, dataOut, numBytesDataOut);
    freeReservedMemory(dataOut);

    /* byte 0 = (0xB << 4) | 0xA = 0xBA; byte 1 = (0xD << 4) | 0xC = 0xDC. */
    uint8_t expectedBytes[2] = {0xBA, 0xDC};
    TEST_ASSERT_EQUAL_UINT8_ARRAY(expectedBytes, captured, numBytesDataOut);
}

void testCopyTensor() {
    size_t numberOfValues = 3;
    tensor_t src;
    float data[] = {1.f, 2.f, 3.f};
    size_t dims[] = {1, numberOfValues};
    size_t numberOfDims = 2;
    size_t orderOfDims[] = {0, 1};
    shape_t shape = {
        .dimensions = dims, .numberOfDimensions = numberOfDims, .orderOfDimensions = orderOfDims};
    quantization_t q;
    initFloat32Quantization(&q);

    setTensorValues(&src, (uint8_t *)data, &shape, &q, NULL);

    tensor_t dest;
    float destData[numberOfValues];
    size_t destDims[2];
    size_t destNumberOfDims;
    size_t destOrderOfDims[2];
    shape_t destShape = {.dimensions = destDims,
                         .numberOfDimensions = destNumberOfDims,
                         .orderOfDimensions = destOrderOfDims};
    setTensorValues(&dest, (uint8_t *)destData, &destShape, &q, NULL);

    copyTensor(&dest, &src);

    float expectedData[] = {1.f, 2.f, 3.f};
    size_t expectedDims[] = {1, numberOfValues};
    size_t expectedNumberOfDims = 2;
    size_t expectedOrderOfDims[] = {0, 1};

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expectedData, dest.data, numberOfValues);
    TEST_ASSERT_EQUAL_size_t_ARRAY(expectedDims, dest.shape->dimensions, 2);
    TEST_ASSERT_EQUAL_size_t(expectedNumberOfDims, dest.shape->numberOfDimensions);
    TEST_ASSERT_EQUAL_size_t_ARRAY(expectedOrderOfDims, dest.shape->orderOfDimensions, 2);
}

void test_calcBitsPerElement_Sym_qBits3() {
    symQConfig_t cfg = {.scale = 1.0f, .qBits = 3, .roundingMode = HALF_AWAY};
    quantization_t q;
    initSymQuantization(&cfg, &q);
    TEST_ASSERT_EQUAL_size_t(3, calcBitsPerElement(&q));
}

void test_calcBytesPerElement_Sym_qBits3() {
    symQConfig_t cfg = {.scale = 1.0f, .qBits = 3, .roundingMode = HALF_AWAY};
    quantization_t q;
    initSymQuantization(&cfg, &q);
    /* ceil(3/8) = 1 */
    TEST_ASSERT_EQUAL_size_t(1, calcBytesPerElement(&q));
}

void test_calcNumberOfBytesForData_Sym_qBits3_N10() {
    symQConfig_t cfg = {.scale = 1.0f, .qBits = 3, .roundingMode = HALF_AWAY};
    quantization_t q;
    initSymQuantization(&cfg, &q);
    /* ceil(3*10 / 8) = ceil(30/8) = 4 */
    TEST_ASSERT_EQUAL_size_t(4, calcNumberOfBytesForData(&q, 10));
}

void test_calcNumberOfBytesForData_Sym_qBits5_N4() {
    symQConfig_t cfg = {.scale = 1.0f, .qBits = 5, .roundingMode = HALF_AWAY};
    quantization_t q;
    initSymQuantization(&cfg, &q);
    /* ceil(5*4 / 8) = ceil(20/8) = 3 */
    TEST_ASSERT_EQUAL_size_t(3, calcNumberOfBytesForData(&q, 4));
}

void setUp() {}
void tearDown() {}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testByteFlattening);
    RUN_TEST(testByteFlattening2);
    RUN_TEST(testByteFlattening3);
    RUN_TEST(testByteFlattening4);
    RUN_TEST(testByteFlattening5);
    RUN_TEST(testByteConversion_NarrowingInt32ToSubByte_DoesNotOverreadOutputBuffer);
    RUN_TEST(testByteConversion_NarrowingInt32ToByte_DoesNotOverreadOutputBuffer);

    RUN_TEST(testGetBitmask);
    RUN_TEST(testGetBitmask2);
    RUN_TEST(testWriteByte);
    RUN_TEST(testWriteByte2);
    RUN_TEST(testReadByte);

    RUN_TEST(testCopyTensor);

    RUN_TEST(test_calcBitsPerElement_Sym_qBits3);
    RUN_TEST(test_calcBytesPerElement_Sym_qBits3);
    RUN_TEST(test_calcNumberOfBytesForData_Sym_qBits3_N10);
    RUN_TEST(test_calcNumberOfBytesForData_Sym_qBits5_N4);
    return UNITY_END();
}
