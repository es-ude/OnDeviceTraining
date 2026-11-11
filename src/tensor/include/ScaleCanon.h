#ifndef ODT_SCALE_CANON_H
#define ODT_SCALE_CANON_H

#include <stdint.h>

#include "DyadicScale.h"

/* The float-domain rules that keep every STORED SYM / SYM_INT32 / ASYM scale an
 * exact 15-bit dyadic value (DyadicScale.h) while the fields are still float.
 * Each rule rounds once; the value it returns is the one the codes are computed
 * on AND the one stored, so codes and metadata share one grid. Link
 * DyadicScale to use them.
 *
 * Which rule:
 * - a grid fitted to data (absMax / hi, (max - min) / qMax): canonFitScale,
 *   or canonFitScaleFromCodes when the exact absMax is an integer code times a
 *   stored scale. CEIL, so the absMax element still maps inside the top code.
 * - a stored product of two stored scales: canonProductScale (NEAREST, from the
 *   exact mantissa product -- never the float product snapped afterwards).
 * - a stored scale divided by an integer count (mean, pool window):
 *   canonRatioScale, NEAREST.
 * - a stored scale times an arbitrary coefficient: canonCoeffScale, NEAREST.
 * - a user-given or deserialized scale: canonFixedScale, NEAREST.
 * Operands are read with FromFloat(NEAREST), which is exact for a canonical
 * stored value.
 *
 * storeCanonScale is the store: it asserts canonicality under ODT_SCALE_STRICT.
 * Every helper fails fast (DyadicScale) on a non-positive, non-finite or
 * out-of-normal-range value. */

static inline void storeCanonScale(float *dst, float v) {
    odtScaleAssertCanonical(v);
    *dst = v;
}

static inline float canonFitScale(float absMax, float hi) {
    return (absMax == 0.f) ? 1.0f : odtScaleCanonFloat(absMax / hi, ODT_SCALE_CEIL);
}

/* absMax = maxAbsCode * inScale exactly; maxAbsCode in [0, 2^31], hi >= 1. */
static inline float canonFitScaleFromCodes(float inScale, uint32_t maxAbsCode, int32_t hi) {
    if (maxAbsCode == 0u) {
        return 1.0f;
    }
    return odtScaleToFloat(odtScaleMulRatio(odtScaleFromFloat(inScale, ODT_SCALE_NEAREST),
                                            maxAbsCode, (uint32_t)hi, ODT_SCALE_CEIL));
}

static inline float canonProductScale(float a, float b, odtScaleRound_t r) {
    return odtScaleToFloat(odtScaleMul(odtScaleFromFloat(a, ODT_SCALE_NEAREST),
                                       odtScaleFromFloat(b, ODT_SCALE_NEAREST), r));
}

/* s * num / den; num, den in [1, 2^31]. */
static inline float canonRatioScale(float s, uint32_t num, uint32_t den) {
    return odtScaleToFloat(
        odtScaleMulRatio(odtScaleFromFloat(s, ODT_SCALE_NEAREST), num, den, ODT_SCALE_NEAREST));
}

static inline float canonCoeffScale(float s, float factor) {
    return canonProductScale(s, odtScaleCanonFloat(factor, ODT_SCALE_NEAREST), ODT_SCALE_NEAREST);
}

static inline float canonFixedScale(float x) {
    return odtScaleCanonFloat(x, ODT_SCALE_NEAREST);
}

/* |v| as uint32, so INT32_MIN gives 2^31. */
static inline uint32_t canonAbsCode(int32_t v) {
    return v < 0 ? 0u - (uint32_t)v : (uint32_t)v;
}

#endif /* ODT_SCALE_CANON_H */
