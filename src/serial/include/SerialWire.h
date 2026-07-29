#ifndef ODT_SERIAL_WIRE_H
#define ODT_SERIAL_WIRE_H

#include <stdint.h>
#include <stdio.h>

/* Checked, endian-pinned wire primitives shared by every ODT on-disk format
 * (ODTS models, ODTR PPCA checkpoints). Multi-byte scalars are encoded
 * little-endian byte-by-byte — never as raw host integers — so files are
 * identical across 32/64-bit and BE/LE hosts (#370). Every fwrite/fread is
 * length-checked and fails fast (PRINT_ERROR + exit(1)) on short I/O, so a
 * truncated stream can never decode as silent garbage.
 *
 * Bulk DATA payloads (packed tensor storage) go through serialWriteBytes /
 * serialReadBytes verbatim; their element byte order is the host's. All
 * supported targets (x86-64/AArch64 hosts, Cortex-M) are little-endian —
 * pinned here so a big-endian port cannot silently produce byte-swapped
 * payloads. */
#if defined(__BYTE_ORDER__) && (__BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__)
#error "ODT serial formats write bulk DATA payloads verbatim; little-endian hosts only"
#endif

void serialWriteBytes(const void *bytes, size_t numberOfBytes, FILE *f);
void serialWriteU8(uint8_t value, FILE *f);
/*! group-quant PR4 (Task 4): u16 carrier for the ODTS v5 ASYM record's
 *  code-domain zeroPoints[] array (D6 caps qBits <= 16, so a code-domain zp
 *  always fits u16) -- same checked, endian-pinned pattern as the other
 *  fixed-width primitives below. */
void serialWriteU16LE(uint16_t value, FILE *f);
void serialWriteU32LE(uint32_t value, FILE *f);
void serialWriteI32LE(int32_t value, FILE *f);
void serialWriteF32LE(float value, FILE *f);
/*! size_t carrier for counts/dims/kernel geometry: fails fast if the value
 *  cannot fit the fixed u32 wire width (only reachable on 64-bit hosts). */
void serialWriteSizeAsU32LE(size_t value, FILE *f);

void serialReadBytes(void *bytes, size_t numberOfBytes, FILE *f);
uint8_t serialReadU8(FILE *f);
uint16_t serialReadU16LE(FILE *f);
uint32_t serialReadU32LE(FILE *f);
int32_t serialReadI32LE(FILE *f);
float serialReadF32LE(FILE *f);

#endif // ODT_SERIAL_WIRE_H
