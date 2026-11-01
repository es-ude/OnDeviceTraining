#define SOURCE_FILE "ROUNDING"

#include <math.h>
#include <stdlib.h>

#include "Common.h"
#include "RNG.h"
#include "Rounding.h"

// C roundf(): rounds half away from zero (C17 7.12.9.6); float, not double, keeps soft-float MCUs
// off the double path
int32_t roundHalfAway(float input) {
    return roundf(input);
}

/* Stochastic rounding: floor(x + u), u uniform in [0, 1), rounds up with probability equal to
 * the fractional part (Connolly, Higham & Mary 2021). The sum x + u is never formed in float:
 * its rounded value crosses the next integer for u > 1 - ulp(x)/2 (2^-14 at 2047, 1/2 on
 * [2^23, 2^24)), an upward bias that grows with |x|. Splitting x into trunc + exact fraction
 * keeps the same draw-to-outcome mapping as floor(x + u) in real arithmetic, exact for every
 * finite x in int32 range (1 - u is exact on the 2^-24 grid of u). An integer x stays put. */
static int32_t roundSRHalfAway(const float input, rngStream_t *rs) {
    const float u = rngNextUnitFloat(rs); /* exactly one draw, whatever x is */
    const int32_t t = (int32_t)input;     /* truncates toward zero */
    const float f = input - (float)t;     /* exact */
    return f < 0.0f ? t - (u < -f) : t + ((1.0f - u) <= f);
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
