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
 * the fractional part (Connolly, Higham & Mary 2021). An exact integer x + u (u == 0 on an
 * integer x) stays put on both sides of zero. floorf, not floor: soft-float MCUs stay off the
 * double path. */
static int32_t roundSRHalfAway(const float input, rngStream_t *rs) {
    return (int32_t)floorf(input + rngNextUnitFloat(rs));
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
