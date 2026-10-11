#define SOURCE_FILE "ROUNDING"

#include <math.h>
#include <stdlib.h>

#include "Common.h"
#include "RNG.h"
#include "Rounding.h"

// C round(): rounds half away from zero (C17 7.12.9.6)
int32_t roundHalfAway(float input) {
    return round(input);
}

float randfloat() {
    return rngNextFloat();
}

int32_t roundSRHalfAway(const float input, rngStream_t *rs) {
    return roundHalfAway(input + randfloat() - 0.5f);
}

int32_t roundByMode(const float input, const roundingMode_t roundingMode, rngStream_t *rs) {
    switch (roundingMode) {
    case HALF_AWAY:
        return roundHalfAway(input);
    case SR_HALF_AWAY:
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
