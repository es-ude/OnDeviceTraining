#define SOURCE_FILE "SERIAL_WIRE"

#include <stdlib.h>
#include <string.h>

#include "Common.h"
#include "SerialWire.h"

void serialWriteBytes(const void *bytes, size_t numberOfBytes, FILE *f) {
    if (fwrite(bytes, 1, numberOfBytes, f) != numberOfBytes) {
        PRINT_ERROR("short write (%zu bytes): stream not writable or device full", numberOfBytes);
        exit(1);
    }
}

void serialWriteU8(uint8_t value, FILE *f) {
    serialWriteBytes(&value, 1, f);
}

void serialWriteU16LE(uint16_t value, FILE *f) {
    uint8_t bytes[2] = {(uint8_t)value, (uint8_t)(value >> 8)};
    serialWriteBytes(bytes, 2, f);
}

void serialWriteU32LE(uint32_t value, FILE *f) {
    uint8_t bytes[4] = {(uint8_t)value, (uint8_t)(value >> 8), (uint8_t)(value >> 16),
                        (uint8_t)(value >> 24)};
    serialWriteBytes(bytes, 4, f);
}

void serialWriteI32LE(int32_t value, FILE *f) {
    serialWriteU32LE((uint32_t)value, f);
}

void serialWriteF32LE(float value, FILE *f) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    serialWriteU32LE(bits, f);
}

void serialWriteSizeAsU32LE(size_t value, FILE *f) {
#if SIZE_MAX > UINT32_MAX
    if (value > (size_t)UINT32_MAX) {
        PRINT_ERROR("value %zu exceeds the u32 wire width", value);
        exit(1);
    }
#endif
    serialWriteU32LE((uint32_t)value, f);
}

void serialReadBytes(void *bytes, size_t numberOfBytes, FILE *f) {
    if (fread(bytes, 1, numberOfBytes, f) != numberOfBytes) {
        PRINT_ERROR("short read (%zu bytes): truncated or corrupt stream", numberOfBytes);
        exit(1);
    }
}

uint8_t serialReadU8(FILE *f) {
    uint8_t value;
    serialReadBytes(&value, 1, f);
    return value;
}

uint16_t serialReadU16LE(FILE *f) {
    uint8_t bytes[2];
    serialReadBytes(bytes, 2, f);
    return (uint16_t)((uint16_t)bytes[0] | ((uint16_t)bytes[1] << 8));
}

uint32_t serialReadU32LE(FILE *f) {
    uint8_t bytes[4];
    serialReadBytes(bytes, 4, f);
    return (uint32_t)bytes[0] | ((uint32_t)bytes[1] << 8) | ((uint32_t)bytes[2] << 16) |
           ((uint32_t)bytes[3] << 24);
}

int32_t serialReadI32LE(FILE *f) {
    return (int32_t)serialReadU32LE(f);
}

float serialReadF32LE(FILE *f) {
    uint32_t bits = serialReadU32LE(f);
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}
