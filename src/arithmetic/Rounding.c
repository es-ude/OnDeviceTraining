#define SOURCE_FILE "ROUNDING"

#include <math.h>
#include <stdlib.h>

#include "Common.h"
#include "Rounding.h"

void roundByModeMissingStream(void) {
    PRINT_ERROR("roundByMode: SR rounding without a stream");
    exit(1);
}

int32_t rescaleIntoAccumulatorScale(int32_t paramQ, float paramScale, float accumulatorScale,
                                    roundingMode_t roundingMode, rngStream_t *rs) {
    float rescaled = (float)paramQ * paramScale / accumulatorScale;
    /* !(x <= T) also catches NaN (0*inf when accumulatorScale underflows to 0).
     * Reserve the worst int16 product 32768*32767, not qMax*qMax. */
    if (!(fabsf(rescaled) <= 2147483647.0f - 32768.0f * 32767.0f)) {
        PRINT_ERROR("rescaleIntoAccumulatorScale: param scale incompatible with accumulator "
                    "scale — seed overflows int32");
        exit(1);
    }
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
