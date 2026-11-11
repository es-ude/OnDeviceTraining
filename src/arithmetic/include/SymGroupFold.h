#ifndef SYM_GROUP_FOLD_H
#define SYM_GROUP_FOLD_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "DyadicScale.h"
#include "Quantization.h"
#include "Rounding.h"

/* Per-group integer fold for the grouped-scale SYM kernels.
 *
 * A grouped weight carries one dyadic scale s_g = m_g * 2^e_g per group. The
 * kernels fold an input code x (scale inScale) into group g once,
 *     x_g = round(x * m_g * 2^-(E - e_g + 15 - F)),   E = max_g e_g,
 * and then accumulate x_g * w in plain int32 at ONE accumulator scale
 *     A = inScale.m * 2^(inScale.e + E + 15 - F)
 * (exact dyadic: x * inScale * s_g = x_g * A up to the fold's rounding). With a
 * canonical m_g < 2^15 and e_g <= E, |x_g| <= |x| * 2^F + 1, so F extra fraction
 * bits keep the fold's rounding error at 2^-F of one input code while the
 * accumulator guard (SymAccumGuard.h) still proves the int32 sum safe.
 *
 * The multiplier is the fixed-point M = M0 * 2^-n of Jacob et al., CVPR 2018,
 * Sec. 2.2, Eq. (6), with M0 = m / 2^15 in [0.5, 1). */

#define ODT_GROUP_FOLD_FRAC_BITS_MIN 1
#define ODT_GROUP_FOLD_FRAC_BITS_MAX 8
#define ODT_GROUP_FOLD_MAX_X_BITS 16
/* The largest |seed| rescaleIntoAccumulatorScale's ODT_SEED_GUARD (Rounding.c)
 * admits: 2^31 - 1 minus the worst int16 product 32768 * 32767. */
#define ODT_GROUP_FOLD_SEED_LIMIT (2147483647u - 32768u * 32767u)

/* round(v * m * 2^-s); m in [0, 32768], any s >= 0 (s > 63 is treated as 63: |v*m| < 2^46,
 * so the true result already rounds to 0 under HALF_AWAY and the SR lift probability
 * changes by < 2^-17 -- documented clamp, as bfpShiftRightRounded clamps at 31); int32-only.
 * HALF_AWAY: exact. SR_HALF_AWAY: floor + lift iff u < frac, u = rngNextU32(rs),
 * frac = the top 32 bits of the dropped s-bit fraction (exact for s <= 32);
 * exactly one draw iff s > 0; zero fraction never lifts; SR with s > 0 and rs == NULL ->
 * named error, exit(1) (SR with s == 0 draws nothing and accepts a NULL stream).
 * Precondition (caller-proved, not checked): the returned value fits int32 -- for HALF_AWAY
 * the rounded result, for SR whichever of floor / floor + 1 the draw picks (both, for
 * every draw, when the real value v*m*2^-s lies in [INT32_MIN, INT32_MAX]). */
int32_t symMulShiftRound(int32_t v, uint16_t m, uint32_t s, roundingMode_t mode, rngStream_t *rs);

typedef struct symGroupFold {
    bool legacy;         /* true: caller must run today's float path */
    uint8_t fracBits;    /* F, valid iff !legacy */
    int16_t eMax;        /* E = max_g scales[g].e  (canonical m in [2^14, 2^15)) */
    odtScale_t accScale; /* A = {inScale.m, inScale.e + E + 15 - F} */
    uint64_t maxAbsSeed; /* the |seed| bound at A the proof used (slack included), 0 seedless */
} symGroupFold_t;

/* E, then the largest F in [MIN, MAX] whose guard site proves safe for
 * nMax terms of (x range widened to [xMin*2^F - 1, xMax*2^F + 1]) x [wMin, wMax]
 * plus |seed| <= seed_F, where the caller measures its seed (e.g. a bias
 * rescaled into the accumulator) at the FINEST accumulator scale, A at
 * F = MAX: maxAbsSeedAtFracMax >= |seed| there. A at F is 2^(MAX - F) times
 * coarser, so seed_F = ceil(maxAbsSeedAtFracMax / 2^(MAX - F)), plus
 * ceil(seed_F * 2^-22) + 1 of slack for the float32 rescale that forms the
 * real seed and its rounding lift (0 stays 0). F also needs
 * seed_F <= ODT_GROUP_FOLD_SEED_LIMIT (the seed-rescale guard). The bound used
 * is returned in .maxAbsSeed.
 * Legacy if xBits > MAX_X_BITS, or no F >= MIN is provable,
 * or A = {inScale.m, inScale.e + E + 15 - F} is outside the canonical exponent range
 * DyadicScale.h admits. x's range is the declared xBits two's-complement range.
 * Fails fast (named error, exit(1)) on a NULL / empty config or a non-canonical
 * inScale / group scale: the fold's |x_g| <= |x| * 2^F + 1 bound needs canonical m. */
symGroupFold_t symGroupFoldInit(const symQConfig_t *weightGroups, odtScale_t inScale, uint8_t xBits,
                                size_t nMax, int32_t wMin, int32_t wMax,
                                uint64_t maxAbsSeedAtFracMax);

/* E - s.e + 15 - F for one group's (local-copy) scale. */
static inline uint32_t symGroupFoldShift(const symGroupFold_t *f, odtScale_t s) {
    return (uint32_t)((int32_t)f->eMax - (int32_t)s.e + 15 - (int32_t)f->fracBits);
}

#endif // SYM_GROUP_FOLD_H
