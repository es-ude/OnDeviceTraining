#ifndef ODT_SERIAL_WIRE_H
#define ODT_SERIAL_WIRE_H

#include <stdint.h>
#include <stdio.h>

/* Checked wire primitives shared by the ODT on-disk formats (ODTS models,
 * ODTR PPCA checkpoints). Multi-byte scalars are encoded little-endian byte by
 * byte, so files are identical across 32/64-bit hosts. Every fwrite/fread is
 * length-checked and fails fast on short I/O.
 *
 * Bulk payloads (packed tensor storage) go through serialWriteBytes /
 * serialReadBytes verbatim in host byte order, so only little-endian hosts
 * are supported. */
#if defined(__BYTE_ORDER__) && (__BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__)
#error "ODT serial formats write bulk DATA payloads verbatim; little-endian hosts only"
#endif

void serialWriteBytes(const void *bytes, size_t numberOfBytes, FILE *f);
void serialWriteU8(uint8_t value, FILE *f);
void serialWriteU16LE(uint16_t value, FILE *f);
/*! linq S4 (ODTS v9): i16 carrier for a dyadic scale's exponent (odtScale_t.e). */
void serialWriteI16LE(int16_t value, FILE *f);
void serialWriteU32LE(uint32_t value, FILE *f);
void serialWriteI32LE(int32_t value, FILE *f);
void serialWriteF32LE(float value, FILE *f);
void serialWriteU64LE(uint64_t value, FILE *f);
/*! size_t carrier for counts/dims/kernel geometry: fails fast if the value
 *  cannot fit the fixed u32 wire width (only reachable on 64-bit hosts). */
void serialWriteSizeAsU32LE(size_t value, FILE *f);

void serialReadBytes(void *bytes, size_t numberOfBytes, FILE *f);
uint8_t serialReadU8(FILE *f);
uint16_t serialReadU16LE(FILE *f);
int16_t serialReadI16LE(FILE *f);
uint32_t serialReadU32LE(FILE *f);
int32_t serialReadI32LE(FILE *f);
float serialReadF32LE(FILE *f);
uint64_t serialReadU64LE(FILE *f);

#endif // ODT_SERIAL_WIRE_H
