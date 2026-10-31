#ifndef REFERENCE_BYTE_CONVERSION_H
#define REFERENCE_BYTE_CONVERSION_H

/* Test-only oracle: a verbatim copy of the bit-walking byteConversionAppend
 * engine (and its helpers) that Tensor.c shipped before #505, renamed with a
 * ref prefix. The differential test in UnitTestTensor.c pins the production
 * engine to it byte for byte. Header-only because add_elastic_ai_unit_test
 * does not compile MORE_SOURCES. */

#include <stddef.h>
#include <stdint.h>

static uint32_t refGetBitmask(uint32_t startbit, uint32_t endbit) {
    uint32_t endbitInternal = endbit - (startbit / 8) * 8;
    uint32_t startbitInternal = startbit - (startbit / 8) * 8;
    uint32_t counter = 0;
    uint32_t value = 1;
    for (size_t i = 0; i < 8; i++) {
        if ((i >= startbitInternal) & (endbitInternal > i)) {
            counter += value;
        }
        value *= 2;
    }
    return counter;
}

static uint8_t refReadByte(uint8_t data, uint8_t startbit, uint8_t endbit) {
    uint8_t bitmask = refGetBitmask(startbit, endbit);
    uint8_t intermediate = data & bitmask;
    intermediate >>= startbit - (startbit / 8) * 8;
    return intermediate;
}

static uint8_t refWriteByte(uint8_t existingData, uint8_t data, uint8_t startbit, uint8_t endbit) {
    uint8_t startbitInternal = startbit - (startbit / 8) * 8;
    uint8_t endbitInternal = endbit - (startbit / 8) * 8;
    uint8_t bitmask = refGetBitmask(startbitInternal, endbitInternal);
    data <<= startbitInternal;
    uint8_t intermediate = data & bitmask;
    existingData = (existingData & (uint8_t)~bitmask) | intermediate;
    return existingData;
}

static int refMax(int a, int b) {
    return (a > b) ? a : b;
}

static int refMin(int a, int b) {
    return (a < b) ? a : b;
}

static void refByteConversionAppend(uint8_t *dataIn, size_t dataInBits, uint8_t *dataOut,
                                    size_t dataOutBits, size_t numValues, size_t dstStartBit,
                                    size_t srcStartBit) {
    size_t dataOutIndex = dstStartBit / 8;
    size_t dataInIndex = srcStartBit / 8;
    int dataOutStartbit = (int)(dstStartBit % 8);
    int dataInStartbit = (int)(srcStartBit % 8);
    int dataInEndbit = dataInStartbit + (int)dataInBits;
    int dataOutEndbit = dataOutStartbit + (int)dataOutBits;
    for (size_t i = 0; i < numValues; i++) {
        while ((dataInStartbit < dataInEndbit) | (dataOutStartbit < dataOutEndbit)) {
            uint8_t data = 0;
            if (dataInStartbit < dataInEndbit) {
                data = refReadByte(dataIn[dataInIndex], dataInStartbit, dataInEndbit);
            }
            if (dataOutStartbit < dataOutEndbit) {
                dataOut[dataOutIndex] =
                    refWriteByte(dataOut[dataOutIndex], data, dataOutStartbit, dataOutEndbit);
            }

            int valuesRead = refMin(dataInEndbit - dataInStartbit, 8 - dataInStartbit % 8);
            int valuesWritten = refMin(dataOutEndbit - dataOutStartbit, 8 - dataOutStartbit % 8);
            int minValue = refMin(valuesRead, valuesWritten);

            uint8_t deltaIn = minValue;
            uint8_t deltaOut = minValue;
            if (dataInStartbit == dataInEndbit) {
                dataOutStartbit += valuesWritten;
                deltaOut = valuesWritten;
            } else {
                dataOutStartbit += minValue;
            }
            if (dataOutStartbit == dataOutEndbit) {
                dataInStartbit += valuesRead;
                deltaIn = valuesRead;
            } else {
                dataInStartbit += minValue;
            }

            if (dataInStartbit / 8 > (dataInStartbit - deltaIn) / 8) {
                dataInIndex += 1;
            }
            if (dataOutStartbit / 8 > (dataOutStartbit - deltaOut) / 8) {
                dataOutIndex += 1;
            }
        }
        dataInStartbit = dataInEndbit % 8;
        dataInEndbit = dataInStartbit + (int)dataInBits;
        dataOutStartbit = dataOutEndbit % 8;
        dataOutEndbit = dataOutStartbit + (int)dataOutBits;
    }
}

#endif
