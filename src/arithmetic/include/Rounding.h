#ifndef ROUNDING_H
#define ROUNDING_H
#include <math.h>
#include <stdint.h>
#include <string.h>

#include "RNG.h"

/*! @brief Describes rounding
 * HALF_AWAY = round half away from zero (C roundf(), C17 7.12.9.6)
 * SR_HALF_AWAY = stochastic rounding, one law: lift iff u < frac, with u = (top 24 bits of
 *                a U32) / 2^24 in [0, 1 - 2^-24], frac = x - trunc(x). Negative x: lower iff
 *                u < -frac. The up-probability is the fractional part; no sum x + u is
 *                formed, so an exact integer stays put for every u, and SR(-x) = -SR(x)
 *                for the same u
 */
typedef enum roundingMode { HALF_AWAY, SR_HALF_AWAY } roundingMode_t;

/* The per-element helpers below are static inline: the encode loops and requant kernels call them
 * once per element across translation units, and without LTO an out-of-line call (and the mode
 * switch) costs more than the rounding itself on Cortex-M. */

/* Stops the program: SR rounding was asked for without a stream. Out of line so the inline fast
 * path stays small. */
_Noreturn void roundByModeMissingStream(void);

/* C roundf(): rounds half away from zero (C17 7.12.9.6); float, not double, keeps soft-float MCUs
 * off the double path */
static inline int32_t roundHalfAway(const float input) {
    return roundf(input);
}

/* Stochastic rounding: lift iff u < frac (law: roundingMode_t above and
 * docs/conventions/arithmetic-sym.md), decided on the float's bits so a soft-float MCU makes no
 * float call. Bit-identical to the float form
 *   t = trunc(x), f = x - t, f < 0 ? t - (u < -f) : t + (u < f):
 * u = k * 2^-24 with integer k, so u < |f| <=> k < |f| * 2^24 <=> k < c = ceil(|f| * 2^24),
 * which the branches compute exactly from the mantissa (c < 2^24 fits uint32, no int64).
 * A negative x gives -(|t| + lift), the float form's t - lift; the negation runs in uint32 so
 * x = -2^31 cannot overflow. Exactly one draw per call, whatever x is; the caller guarantees a
 * finite x in [-2^31, 2^31) (+2^31 would wrap to INT32_MIN). Call it through roundByMode: it
 * does not check rs, which must be a non-NULL stream. */
static inline int32_t roundSRHalfAway(const float input, rngStream_t *rs) {
    const uint32_t k = rngNextU32(rs) >> 8;
    uint32_t bits;
    memcpy(&bits, &input, sizeof bits);
    const uint32_t biased = (bits >> 23) & 0xFFu;
    const uint32_t mantissa = bits & 0x7FFFFFu;
    uint32_t t = 0u;
    uint32_t c;
    if (biased == 0u) {
        c = mantissa != 0u; /* zero: c = 0; subnormal: 0 < |f| < 2^-24, so c = 1 */
    } else {
        const uint32_t m = mantissa | 0x800000u;
        const int32_t e = (int32_t)biased - 127;
        if (e >= 23) {
            t = m << (e - 23);
            c = 0u;
        } else if (e >= 0) {
            const uint32_t dropped = (uint32_t)(23 - e);
            t = m >> dropped;
            c = (m & ((1u << dropped) - 1u)) << (e + 1);
        } else {
            const uint32_t sh = (uint32_t)(-(e + 1)); /* |f| * 2^24 = m / 2^sh */
            if (sh == 0u) {
                c = m;
            } else if (sh >= 24u) {
                c = 1u;
            } else {
                c = (m >> sh) + ((m & ((1u << sh) - 1u)) != 0u);
            }
        }
    }
    const uint32_t r = t + (k < c);
    return (bits >> 31) ? (int32_t)(0u - r) : (int32_t)r;
}

static inline int32_t roundByMode(const float input, const roundingMode_t roundingMode,
                                  rngStream_t *rs) {
    switch (roundingMode) {
    case HALF_AWAY:
        return roundHalfAway(input);
    case SR_HALF_AWAY:
        if (rs == NULL) {
            roundByModeMissingStream();
        }
        return roundSRHalfAway(input, rs);
    }
    return 0;
}

/*! @brief Rescale a SYM_INT32 parameter mantissa from its own scale into an
 * accumulator's product-scale, returning the int32 seed to add into the
 * integer accumulator (forward bias seed for Matmul/Conv, LayerNorm affine
 * beta seed).
 *
 * The float->int32 cast is data-dependent and is UB on overflow
 * (C17 6.3.1.4, #189). When compiled with -DODT_SEED_GUARD (unit-test/CI
 * builds; off for release/MCU) the cast is guarded: it fails fast when the
 * rescaled value leaves no int32 headroom for one worst-case int16xint16
 * product (32768*32767). NaN-robust via !(x <= T): a 0*inf=NaN from an
 * underflowed accumulator scale is caught, not bypassed. Release builds run
 * the raw cast (UBSan #204 covers occurrences there). */
int32_t rescaleIntoAccumulatorScale(int32_t paramQ, float paramScale, float accumulatorScale,
                                    roundingMode_t roundingMode, rngStream_t *rs);

/*! @brief Rescale a raw int32 PARTIAL (a finished group's sum, or one product)
 * into an accumulator's scale, for the grouped kernels' combine step.
 *
 * Unlike rescaleIntoAccumulatorScale this reserves no headroom: the partial is
 * not a bias seed, so a value close to INT32_MAX is legitimate and the
 * following overflow-checked add (SymAccumGuard.h) decides whether the SUM
 * fits. It keeps only the float->int32 cast safe: in every build it fails fast
 * when the rescaled float is NaN or leaves [-2^31, 2^31). */
int32_t rescalePartialIntoAccumulatorScale(int32_t paramQ, float paramScale, float accumulatorScale,
                                           roundingMode_t roundingMode, rngStream_t *rs);

static inline int32_t clampInt32(int32_t input, int32_t min, int32_t max) {
    if (input < min) {
        return min;
    }
    if (input > max) {
        return max;
    }
    return input;
}

static inline float clamp(float input, float min, float max) {
    if (input < min) {
        return min;
    }
    if (input > max) {
        return max;
    }
    return input;
}

#endif // ROUNDING_H
