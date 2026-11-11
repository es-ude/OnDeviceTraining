#define SOURCE_FILE "SYM_GROUP_FOLD"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "Common.h"
#include "DyadicScale.h"
#include "Quantization.h"
#include "RNG.h"
#include "Rounding.h"
#include "SymAccumGuard.h"
#include "SymGroupFold.h"

#define MUL_SHIFT_MAX_SHIFT 63u
#define HALF_FRAC32 0x80000000u

/* v * m as the two-word value t * 2^16 + r16 (r16 in [0, 2^16)), int32-only:
 * v = hi * 2^16 + lo with hi = v >> 16 (floor) and lo = v & 0xFFFF, so
 * |hi * m| <= 2^15 * 2^15 = 2^30 and lo * m < 2^16 * 2^15 = 2^31 (uint32);
 * the carry lo * m >> 16 < 2^15 keeps |t| <= 2^30 + 2^15. */
int32_t symMulShiftRound(int32_t v, uint16_t m, uint32_t s, roundingMode_t mode, rngStream_t *rs) {
    if (s > MUL_SHIFT_MAX_SHIFT) {
        s = MUL_SHIFT_MAX_SHIFT;
    }
    uint32_t u = 0;
    if (mode == SR_HALF_AWAY && s > 0u) {
        if (rs == NULL) {
            PRINT_ERROR("symMulShiftRound: SR shift without a stream");
            exit(1);
        }
        u = rngNextU32(rs);
    }

    const int32_t hi = v >> 16;
    const uint32_t lo = (uint32_t)v & 0xFFFFu;
    const uint32_t loProd = lo * (uint32_t)m;
    const int32_t t = hi * (int32_t)m + (int32_t)(loProd >> 16);
    const uint32_t r16 = loProd & 0xFFFFu;

    /* floor(v * m / 2^s), the top 32 bits of the dropped s-bit fraction, and
     * whether any dropped bit lies below those 32 (only for 32 < s < 48 can
     * a lower bit decide a HALF_AWAY tie). The floor sums wrap through uint32;
     * the precondition makes the final value fit. */
    int32_t floorQ;
    uint32_t frac32;
    bool sticky = false;
    if (s <= 16u) {
        floorQ = (int32_t)(((uint32_t)t << (16u - s)) + (r16 >> s));
        frac32 = s == 0u ? 0u : (r16 & ((1u << s) - 1u)) << (32u - s);
    } else if (s < 48u) {
        const uint32_t k = s - 16u; /* [1, 31] */
        const uint32_t tLow = (uint32_t)t & ((1u << k) - 1u);
        floorQ = t >> k;
        if (k <= 16u) {
            frac32 = (tLow << (32u - k)) | (r16 << (16u - k));
        } else {
            frac32 = (tLow << (32u - k)) | (r16 >> (k - 16u));
            sticky = (r16 & ((1u << (k - 16u)) - 1u)) != 0u;
        }
    } else {
        /* |v * m| <= 2^46 < 2^s: the floor is the sign, and the top 32 bits of
         * the s-bit fraction are the product shifted by s - 32 >= 16, i.e.
         * t >> (s - 48) in two's complement (s - 48 <= 15 after the clamp).
         * Every dropped lower bit sits below a frac32 that is < 2^31 (t >= 0)
         * or > 2^31 (t < 0), so no tie exists and sticky is never read. */
        floorQ = t < 0 ? -1 : 0;
        frac32 = (uint32_t)(t >> (s - 48u));
    }

    bool lift;
    if (mode == SR_HALF_AWAY) {
        lift = u < frac32;
    } else {
        /* half away from zero in the floor domain: above half always lifts; an
         * exact half lifts a non-negative product (away = up) and keeps a
         * negative one at its floor (away = down). The product is negative
         * iff t is (r16 >= 0). */
        lift = frac32 > HALF_FRAC32 || (frac32 == HALF_FRAC32 && (t >= 0 || sticky));
    }
    /* unsigned add: a floor of INT32_MIN - 1 (wrapped to INT32_MAX above) lifting to an in-range
     * INT32_MIN must not be a signed overflow */
    return (int32_t)((uint32_t)floorQ + (lift ? 1u : 0u));
}

static void requireCanonicalScale(odtScale_t s, const char *what) {
    if (s.m < ODT_SCALE_M_MIN || s.m > ODT_SCALE_M_MAX || s.e < ODT_SCALE_E_MIN ||
        s.e > ODT_SCALE_E_MAX) {
        PRINT_ERROR("symGroupFoldInit: %s {m = %u, e = %d} is not a canonical scale", what,
                    (unsigned)s.m, (int)s.e);
        exit(1);
    }
}

/* The seed bound at F (see SymGroupFold.h); > ODT_GROUP_FOLD_SEED_LIMIT when
 * it cannot be admitted. Overflow-free for every uint64 input. */
static uint64_t foldSeedAt(uint64_t seedAtFracMax, int32_t frac) {
    const uint32_t k = (uint32_t)(ODT_GROUP_FOLD_FRAC_BITS_MAX - frac);
    uint64_t seed = (seedAtFracMax >> k) + ((seedAtFracMax & (((uint64_t)1 << k) - 1u)) != 0u);
    if (seed == 0u || seed > ODT_GROUP_FOLD_SEED_LIMIT) {
        return seed;
    }
    const uint64_t mask22 = ((uint64_t)1 << 22) - 1u;
    return seed + (seed >> 22) + ((seed & mask22) != 0u) + 1u;
}

symGroupFold_t symGroupFoldInit(const symQConfig_t *weightGroups, odtScale_t inScale, uint8_t xBits,
                                size_t nMax, int32_t wMin, int32_t wMax,
                                uint64_t maxAbsSeedAtFracMax) {
    if (weightGroups == NULL || weightGroups->numGroups == 0) {
        PRINT_ERROR("symGroupFoldInit: NULL or empty weight group config");
        exit(1);
    }
    requireCanonicalScale(inScale, "inScale");

    symGroupFold_t f = {
        .legacy = true, .fracBits = 0, .eMax = 0, .accScale = inScale, .maxAbsSeed = 0};

    int32_t eMax = INT32_MIN;
    for (size_t g = 0; g < weightGroups->numGroups; g++) {
        odtScale_t s = odtScaleLoad(&weightGroups->scales[g]);
        requireCanonicalScale(s, "group scale");
        eMax = s.e > eMax ? s.e : eMax;
    }
    f.eMax = (int16_t)eMax;

    if (xBits > ODT_GROUP_FOLD_MAX_X_BITS) {
        return f;
    }
    int32_t xMin, xMax;
    symAccumWidthRange(xBits, &xMin, &xMax);

    /* xBits <= 16 and F <= 8: the widened range stays within +-(2^23 + 1). */
    for (int32_t frac = ODT_GROUP_FOLD_FRAC_BITS_MAX; frac >= ODT_GROUP_FOLD_FRAC_BITS_MIN;
         frac--) {
        const uint64_t seed = foldSeedAt(maxAbsSeedAtFracMax, frac);
        if (seed > ODT_GROUP_FOLD_SEED_LIMIT) {
            continue;
        }
        int32_t xLo = xMin * (1 << frac) - 1;
        int32_t xHi = xMax * (1 << frac) + 1;
        symAccumSite_t site =
            symAccumSiteInitRanges("symGroupFoldInit", nMax, xLo, xHi, wMin, wMax, seed);
        if (!symAccumProvenSafe(&site)) {
            continue;
        }
        int32_t accE = (int32_t)inScale.e + eMax + 15 - frac;
        if (accE < ODT_SCALE_E_MIN || accE > ODT_SCALE_E_MAX) {
            return f;
        }
        f.legacy = false;
        f.fracBits = (uint8_t)frac;
        f.accScale.m = inScale.m;
        f.accScale.e = (int16_t)accE;
        f.maxAbsSeed = seed;
        return f;
    }
    return f;
}
