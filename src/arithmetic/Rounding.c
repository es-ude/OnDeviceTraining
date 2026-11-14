#define SOURCE_FILE "ROUNDING"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "Common.h"
#include "RNG.h"
#include "Rounding.h"

// C roundf(): rounds half away from zero (C17 7.12.9.6); float, not double, keeps soft-float MCUs
// off the double path
int32_t roundHalfAway(float input) {
    return roundf(input);
}

/* Stochastic rounding: lift iff u < frac (law: Rounding.h, docs/conventions/arithmetic-sym.md),
 * decided on the float's bits so a soft-float MCU makes no float call. Bit-identical to the float
 * form t = trunc(x), f = x - t, f < 0 ? t - (u < -f) : t + (u < f): u = k * 2^-24 with integer
 * k, so u < |f| <=> k < |f| * 2^24 <=> k < c = ceil(|f| * 2^24), which the branches compute
 * exactly from the mantissa (c < 2^24 fits uint32, no int64). A negative x gives
 * -(|t| + lift), the float form's t - lift; the negation runs in uint32 so x = -2^31 cannot
 * overflow. Exactly one draw per call, whatever x is; the caller guarantees a finite
 * x in [-2^31, 2^31) (+2^31 would wrap to INT32_MIN). */
static int32_t roundSRHalfAway(const float input, rngStream_t *rs) {
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

int32_t roundByMode(const float input, const roundingMode_t roundingMode, rngStream_t *rs) {
    switch (roundingMode) {
    case HALF_AWAY:
        return roundHalfAway(input);
    case SR_HALF_AWAY:
        if (rs == NULL) {
            PRINT_ERROR("roundByMode: SR rounding without a stream");
            exit(1);
        }
        return roundSRHalfAway(input, rs);
    }
    return 0;
}

int32_t rescaleIntoAccumulatorScale(int32_t paramQ, float paramScale, float accumulatorScale,
                                    roundingMode_t roundingMode, rngStream_t *rs) {
    float rescaled = (float)paramQ * paramScale / accumulatorScale;
#ifdef ODT_SEED_GUARD
    /* !(x <= T), NOT (x > T): also catches NaN (0*inf when accumulatorScale
     * underflows to 0). Reserve the real worst int16 product 32768*32767
     * (the -32768 case), not qMax*qMax. */
    if (!(fabsf(rescaled) <= 2147483647.0f - 32768.0f * 32767.0f)) {
        PRINT_ERROR("rescaleIntoAccumulatorScale: param scale incompatible with accumulator "
                    "scale — seed overflows int32 (#189)");
        exit(1);
    }
#endif
    return roundByMode(rescaled, roundingMode, rs);
}

int32_t rescalePartialIntoAccumulatorScale(int32_t paramQ, float paramScale, float accumulatorScale,
                                           roundingMode_t roundingMode, rngStream_t *rs) {
    float rescaled = (float)paramQ * paramScale / accumulatorScale;
    /* !(a && b) form so NaN fails; 2^31 is exactly representable and the
     * largest float below it (2147483520) survives the SR add and the round. */
    if (!(rescaled >= -2147483648.0f && rescaled < 2147483648.0f)) {
        PRINT_ERROR("rescalePartialIntoAccumulatorScale: rescaled partial %g is NaN or outside "
                    "int32 -- param scale incompatible with accumulator scale",
                    (double)rescaled);
        exit(1);
    }
    return roundByMode(rescaled, roundingMode, rs);
}

int32_t clampInt32(int32_t input, int32_t min, int32_t max) {
    if (input < min) {
        return min;
    }
    if (input > max) {
        return max;
    }
    return input;
}

float clamp(float input, float min, float max) {
    if (input < min) {
        return min;
    }
    if (input > max) {
        return max;
    }
    return input;
}
