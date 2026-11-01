#ifndef ROUNDING_H
#define ROUNDING_H
#include <stdint.h>

/*! @brief Describes rounding
 * HALF_AWAY = round half away from zero (C roundf(), C17 7.12.9.6)
 * SR_HALF_AWAY = stochastic rounding: exactly floor(x + u) with u uniform in [0, 1),
 *                i.e. round up with probability equal to the fractional part;
 *                the sum is not rounded in float (no lift across an integer for u
 *                near 1), and an exact integer stays put for every u
 */
typedef enum roundingMode { HALF_AWAY, SR_HALF_AWAY } roundingMode_t;

/* Pointer-only use: the identical typedef lives in RNG.h (C11 allows the
 * repetition), so this header needs no RNG include path. */
typedef struct rngStream rngStream_t;

int32_t roundByMode(float input, roundingMode_t roundingMode, rngStream_t *rs);

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

int32_t clampInt32(int32_t input, int32_t min, int32_t max);

float clamp(float input, float min, float max);

#endif // ROUNDING_H
