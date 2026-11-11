#ifndef ODT_DYADIC_SCALE_H
#define ODT_DYADIC_SCALE_H

#include <stdbool.h>
#include <stdint.h>
#include <string.h>

/* Integer dyadic quantization scale: value = m * 2^e.
 *
 * Canonical form: m is normalised to 15 significant bits, 16384 <= m <= 32767,
 * so every positive value has exactly one representation and odtScaleEq is a
 * field comparison. Admissible range: the value is a normal float,
 * 2^-126 <= m * 2^e <= FLT_MAX, which for a canonical m is exactly
 * ODT_SCALE_E_MIN <= e <= ODT_SCALE_E_MAX. Every arithmetic and float-boundary
 * helper fails fast (PRINT_ERROR + exit(1)) on a non-canonical operand and on a
 * result outside that range; a zero-initialised odtScale_t is not canonical.
 * odtScaleStore / odtScaleLoad are unchecked 4-byte copies.
 *
 * Use it wherever a stored scale must be dyadic, so integer kernels on an MCU
 * without FPU can rescale with shifts and int32 multiplies. The arithmetic
 * helpers (Mul, MulRatio, Cmp, Eq) use int32/uint32 only: no float, no
 * division operator, no 64-bit type. Float appears only at the boundary
 * (FromFloat, ToFloat, CanonFloat), and only as IEEE-754 bit manipulation.
 *
 * Rounding: NEAREST rounds half away from zero (scales are positive, so half
 * up); CEIL and FLOOR are directed on the value. Each helper rounds once, from
 * the exact real result.
 *
 * Concurrency contract: a scale is written by a single producer and must not
 * be read concurrently with that write. Publishing a written scale to other
 * threads is the job of the tensor's synchronisation (the scheduler), not of
 * this module. odtScaleStore / odtScaleLoad copy the 4 bytes alias-safely
 * through one uint32_t; on arm-none-eabi gcc -O2/-Os for Cortex-M0+ and
 * Cortex-M33 each compiles to a single str / ldr (checked, not assumed). They
 * are not atomic read-modify-write operations. */

typedef struct {
    _Alignas(4) uint16_t m;
    int16_t e;
} odtScale_t;

_Static_assert(sizeof(odtScale_t) == 4, "odtScale_t must be one 32-bit word");
_Static_assert(_Alignof(odtScale_t) == 4, "odtScale_t must be word aligned");

#define ODT_SCALE_M_MIN 16384
#define ODT_SCALE_M_MAX 32767
#define ODT_SCALE_E_MIN (-140) /* 16384 * 2^-140 = 2^-126 = FLT_MIN */
#define ODT_SCALE_E_MAX 113    /* 32767 * 2^113 <= FLT_MAX < 16384 * 2^114 */

/* 2^k for k in [-126, 127]. A compound literal: usable in expressions and, as
 * a GNU C extension also accepted by clang, in static initializers. */
#define ODT_SCALE_POW2(k) ((odtScale_t){.m = (uint16_t)16384, .e = (int16_t)((k) - 14)})
#define ODT_SCALE_ONE ODT_SCALE_POW2(0)

typedef enum { ODT_SCALE_NEAREST, ODT_SCALE_CEIL, ODT_SCALE_FLOOR } odtScaleRound_t;

/* x must be a positive normal float whose rounded value stays <= FLT_MAX. */
odtScale_t odtScaleFromFloat(float x, odtScaleRound_t r);

/* Exact: 15 significant bits always fit a float. */
float odtScaleToFloat(odtScale_t s);

/* a * b, rounded once from the exact 30-bit mantissa product. */
odtScale_t odtScaleMul(odtScale_t a, odtScale_t b, odtScaleRound_t r);

/* s * num / den, rounded once from the exact quotient. num and den in
 * [1, 2^31]; 2^31 is admitted so an INT32_MIN magnitude can be passed. */
odtScale_t odtScaleMulRatio(odtScale_t s, uint32_t num, uint32_t den, odtScaleRound_t r);

/* -1, 0 or 1 as a < b, a == b, a > b by value. */
int odtScaleCmp(odtScale_t a, odtScale_t b);

bool odtScaleEq(odtScale_t a, odtScale_t b);

/* odtScaleToFloat(odtScaleFromFloat(x, r)): x snapped onto the dyadic grid. */
float odtScaleCanonFloat(float x, odtScaleRound_t r);

/* __builtin_assume_aligned: without it gcc cannot prove the pointer is word
 * aligned and, on Cortex-M0+ (no unaligned access), emits a memcpy call
 * instead of one str / ldr. The struct's _Alignas(4) makes the promise true.
 *
 * Precondition: dst / src point to a real odtScale_t object (4-byte aligned).
 * Never cast a byte-buffer offset to odtScale_t *; to read a scale from a byte
 * buffer (e.g. a serialized model), memcpy the 4 bytes into a local first. */
static inline void odtScaleStore(odtScale_t *dst, odtScale_t v) {
    uint32_t bits;
    memcpy(&bits, &v, sizeof bits);
    memcpy(__builtin_assume_aligned(dst, 4), &bits, sizeof bits);
}

static inline odtScale_t odtScaleLoad(const odtScale_t *src) {
    uint32_t bits;
    memcpy(&bits, __builtin_assume_aligned(src, 4), sizeof bits);
    odtScale_t v;
    memcpy(&v, &bits, sizeof v);
    return v;
}

#endif /* ODT_DYADIC_SCALE_H */
