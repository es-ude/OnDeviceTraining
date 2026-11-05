#ifndef BFP_KERNEL_SUPPORT_H
#define BFP_KERNEL_SUPPORT_H

#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "Common.h"
#include "Quantization.h"
#include "Tensor.h"

/* BFP epic PR2 (Task 3): shared support for the BFP GEMM kernels (Matmul.c
 * now, the conv kernels in later tasks). Header-only static inline -- no new
 * library. */

/*! Group id of a storage index; the {numGroups=1, groupSize=0} per-tensor
 * sentinel maps everything to group 0. */
static inline size_t bfpGroupOf(const bfpQConfig_t *qC, size_t storageIdx) {
    return qC->groupSize == 0 ? 0 : storageIdx / qC->groupSize;
}

/*! Current-group cache for the per-element lookups inside MAC loops: the
 * storage range [start, end) of `group`. Zero-initialise ({0}) for the empty
 * range, so the first lookup recomputes. One cache per operand config. */
typedef struct {
    size_t group;
    size_t start;
    size_t end;
} bfpGroupRange_t;

/*! Group id of `physIdx`, same result as bfpGroupOf, but the division runs
 * only when `physIdx` leaves the cached range (a group change in either
 * direction) -- an M0+ has no hardware divide, and the MAC loops walk long
 * same-group runs. The per-tensor sentinel installs [0, SIZE_MAX) once. */
static inline size_t bfpGroupAt(bfpGroupRange_t *r, const bfpQConfig_t *qC, size_t physIdx) {
    if (physIdx < r->start || physIdx >= r->end) {
        r->group = bfpGroupOf(qC, physIdx);
        if (qC->groupSize == 0) {
            r->start = 0;
            r->end = SIZE_MAX;
        } else {
            r->start = r->group * qC->groupSize;
            r->end = r->start + qC->groupSize;
        }
    }
    return r->group;
}

/*! Taps idx, idx + stride, ... (at most `cap`) that stay below `end`, the
 * exclusive end of the group range holding idx: the run a MAC loop may take
 * with one bfpGroupAt per operand instead of one per tap. No division (an M0+
 * has none in hardware): unit stride is one subtraction, a dilated stride
 * walks the run. Safe on the per-tensor sentinel end SIZE_MAX. */
static inline size_t bfpTapsBelow(size_t idx, size_t end, size_t stride, size_t cap) {
    size_t span = end - idx;
    if (stride == 1) {
        return span < cap ? span : cap;
    }
    size_t n = 0;
    while (n < cap) {
        n++;
        if (span <= stride) {
            break;
        }
        span -= stride;
    }
    return n;
}

/*! ldexpf((float)m, e), bit for bit, for every int32 m and every int e: the
 * fold of an int32 segment partial (or a single code) into a float
 * accumulator. Use it instead of ldexpf in per-fold / per-element BFP loops;
 * the libm call is the hot cost there.
 * Fast path e in [-126, 96]: 2^e is a normal float built from its exponent
 * field, and (float)m is 0 or 1 <= |m| <= 2^31, so m * 2^e lies in
 * [2^-126, 2^127] or is +0 -- representable, hence EXACT. An exact product
 * equals ldexpf's single rounding, and stays exact when the compiler fuses
 * it with the accumulator add (-ffp-contract=fast). Every other e (subnormal
 * results, overflow to inf, 2^e itself not normal) takes the ldexpf call.
 * Reach: an operand exponent with 2..8 exponent bits is stored - bias in
 * [-127, 128], so a product fold's e is in [-254, 256] -- trained tensors
 * stay well inside the fast path, the rest is correct, only slower. */
static inline float bfpLdexpInt32(int32_t m, int e) {
    if (e >= -126 && e <= 96) {
        uint32_t bits = (uint32_t)(e + 127) << 23;
        float pow2;
        memcpy(&pow2, &bits, sizeof(pow2));
        return (float)m * pow2;
    }
    return ldexpf((float)m, e);
}

/*! Max products one int32 block partial may accumulate: INT32_MAX >> (ma+mb-2).
 * Mantissas are in [-2^(m-1), 2^(m-1)-1], so |product| <= 2^(ma+mb-2) and
 * INT32_MAX >> (ma+mb-2) products fit int32 -- the kernels' no-int64 rule
 * holds by construction, never by wider accumulators. */
static inline size_t bfpSegmentLimit(uint8_t maMantissaBits, uint8_t mbMantissaBits) {
    unsigned shift = (unsigned)maMantissaBits + (unsigned)mbMantissaBits - 2u;
    if (shift >= 31u) {
        return 1u; /* two 2^30 products already overflow int32 */
    }
    return (size_t)(INT32_MAX >> shift);
}

/*! Fail-fast (#227 headroom) at kernel entry over the (widths, blocking, K)
 * triple: a same-exponent segment never accumulates more than
 * min(runA, runB, K) products (boundary events include every multiple of each
 * operand's groupSize, and distinct storage indices inside one group number
 * <= groupSize -- even on strided walks), so that bound must stay within
 * bfpSegmentLimit. Bias operands are exempt: they are value-seeds dequantized
 * to float before the reduction, never product operands. */
static inline void bfpValidateBlockHeadroom(const bfpQConfig_t *aQC, const bfpQConfig_t *bQC,
                                            size_t reductionLen, const char *what) {
    size_t runA = aQC->groupSize == 0 ? reductionLen : aQC->groupSize;
    size_t runB = bQC->groupSize == 0 ? reductionLen : bQC->groupSize;
    size_t maxSeg = runA < runB ? runA : runB;
    if (maxSeg > reductionLen) {
        maxSeg = reductionLen;
    }
    size_t limit = bfpSegmentLimit(aQC->mantissaBits, bQC->mantissaBits);
    if (maxSeg > limit) {
        PRINT_ERROR("%s: BFP block partial would overflow int32 -- max same-exponent segment "
                    "%zu exceeds %zu products for mantissa widths (%u, %u) (#227 headroom)",
                    what, maxSeg, limit, aQC->mantissaBits, bQC->mantissaBits);
        exit(1);
    }
}

/* ---- Integer fold (integer-only forward, arithmetic-bfp.md §13) ----------
 * A forward kernel keeps ONE int32 accumulator per output in units of
 * 2^E_acc (E_acc per raw target group) and enters every same-exponent
 * partial as partial * 2^-k, k = E_acc - E_segment. The helpers below are
 * the whole arithmetic of that step: no float, no division, no int64. */

/*! v * 2^-k rounded half away from zero, k >= 0. Magnitude form, overflow-
 * safe for every int32 (|INT32_MIN| = 2^31 is a legal uint32 magnitude):
 * t = mag >> (k-1) keeps the half bit as bit 0, so (t >> 1) + (t & 1) is
 * floor(mag / 2^k) plus one when the dropped fraction is >= 1/2. k >= 32
 * drops everything below 1/2 (a partial can never reach 2^31; the one
 * exact -1/2, INT32_MIN at k == 32, still rounds away). */
static inline int32_t bfpShrHalfAway(int32_t v, unsigned k) {
    if (k == 0u) {
        return v;
    }
    if (k >= 32u) {
        return (k == 32u && v == INT32_MIN) ? -1 : 0;
    }
    uint32_t mag = v < 0 ? 0u - (uint32_t)v : (uint32_t)v;
    uint32_t t = mag >> (k - 1u);
    uint32_t r = (t >> 1) + (t & 1u);
    return v < 0 ? -(int32_t)r : (int32_t)r;
}

/*! The fold alignment: k >= 0 is bfpShrHalfAway, k < 0 an EXACT left shift
 * (no rounding, no overflow -- the headroom proof of bfpFoldGuardBits
 * guarantees |v| * 2^-k fits int32 for every NONZERO segment and bias the
 * kernel can produce; -k <= 30 by the same proof). A zero partial may come
 * from a group that does not set E_acc (no nonzero code in the pre-pass run),
 * whose exponent can sit any distance above E_acc: -k >= 32 returns its 0
 * without the out-of-range shift. */
static inline int32_t bfpAlignPartial(int32_t v, int k) {
    if (k < 0) {
        return k > -32 ? (int32_t)((uint32_t)v << (unsigned)(-k)) : 0;
    }
    return bfpShrHalfAway(v, (unsigned)k);
}

/*! The static headroom proof of the integer fold, as the guard g that the
 * kernel adds to the largest segment exponent: E_acc = E_top + g, where
 * E_top bounds EVERY product-segment exponent AND the bias exponent (the
 * kernel max'es the bias exponent in). In units of 2^E_acc:
 *   - a product segment s of count_s products (|code| <= 2^(m-1), so
 *     |product| <= 2^(ma+mb-2)) contributes |partial_s| 2^(E_s - E_acc) <=
 *     count_s 2^(ma+mb-2) 2^-g; run splitting only partitions the N
 *     products, so sum count_s <= N over all segments and together they
 *     contribute <= N 2^(ma+mb-2) 2^-g. Each UNALIGNED partial separately
 *     obeys count_s <= bfpSegmentLimit, i.e. |partial_s| <= INT32_MAX;
 *   - the bias segment contributes |b| 2^(E_b - E_acc) <= 2^(mBias-1) 2^-g
 *     (E_b <= E_top by construction; when the bias exponent exceeds every
 *     product's, E_top = E_b and the products are shifted further right --
 *     they only shrink -- while the bias term is the one at 2^-g);
 *   - every HALF_AWAY right shift adds at most one half; a left shift is
 *     exact. At most N product segments exist (a split never makes an empty
 *     one) plus the bias, so the rounding slack is <= (N + 1) / 2; one whole
 *     unit per fold, N + 1, is reserved.
 * Summing these ABSOLUTE bounds bounds every intermediate accumulator value,
 * whatever the signs or the fold order, not only the final sum:
 *     (N 2^(ma+mb-2) + 2^(mBias-1)) 2^-g + (N + 1) <= INT32_MAX.
 * g is the smallest integer in [-30, 31] satisfying it. It is NEGATIVE
 * whenever the reduction is below the int32 limit (m = 8: g = -10 at N = 80),
 * so the max-exponent partials are left-shifted exactly (|v| 2^-g stays
 * within the bound, no overflow) and only segments more than -g binades
 * below E_top are rounded; it is >= 2 for m = 16 reductions beyond a few
 * products (m16 x m16: 2 at N = 4, 6 at N = 80; m16 x m8: 3 at N = 2400).
 * 64-bit arithmetic is fine here (setup, once per call), never in the fold;
 * every shift count below is in [0, 31]. mBias = 0 means no bias segment.
 * Fail-fast when no g <= 31 exists (every fold would flush to 0). */
static inline int bfpFoldGuardBits(size_t reductionLen, uint8_t ma, uint8_t mb, uint8_t mBias,
                                   const char *what) {
    unsigned productShift = (unsigned)ma + (unsigned)mb - 2u;
    if (reductionLen >= ((size_t)INT32_MAX >> 1) || reductionLen > (UINT64_MAX >> productShift)) {
        PRINT_ERROR("%s: reduction length %zu is beyond any int32 fold", what, reductionLen);
        exit(1);
    }
    uint64_t total =
        ((uint64_t)reductionLen << productShift) + (mBias > 0 ? (uint64_t)1 << (mBias - 1u) : 0u);
    uint64_t room = (uint64_t)INT32_MAX - ((uint64_t)reductionLen + 1u);
    int g = 0;
    if (total > room) {
        /* Right shifts needed: the smallest g in [1, 31] with total <= room 2^g
         * (room < 2^31, so room << 31 fits uint64). */
        while (total > (room << (unsigned)g)) {
            g++;
            if (g > 31) {
                PRINT_ERROR("%s: reduction length %zu at mantissa widths (%u, %u) needs more "
                            "than 31 guard bits -- no int32 fold can hold it",
                            what, reductionLen, (unsigned)ma, (unsigned)mb);
                exit(1);
            }
        }
    } else {
        /* Left shifts allowed: the most negative g in [-30, 0] with
         * total 2^-g <= room (total <= room < 2^31; shift counts 1..30). */
        while (g > -30 && (total << (unsigned)(1 - g)) <= room) {
            g--;
        }
    }
    return g;
}

/* ---- Integer training primitives (full-BFP training, arithmetic-bfp.md §15) ----
 * The integer backward, gradient accumulation and optimizer build on three
 * pieces: a normalized integer scalar for configuration coefficients, a
 * quotient/remainder form that carries the bits an alignment drops all the
 * way to the final rounding, and that final rounding (HALF_AWAY or SR on the
 * magnitude). No float, no division, no int64 below this line except in
 * bfpScalarFromFloat, which runs once per configuration value. */

/*! Bit length of v: bfpBitLength32(0) = 0, bfpBitLength32(2^31) = 32. A
 * binary search instead of __builtin_clz: a core without CLZ (Cortex-M0+)
 * would call libgcc for it. */
static inline unsigned bfpBitLength32(uint32_t v) {
    unsigned n = 0;
    if (v >> 16) {
        v >>= 16;
        n += 16;
    }
    if (v >> 8) {
        v >>= 8;
        n += 8;
    }
    if (v >> 4) {
        v >>= 4;
        n += 4;
    }
    if (v >> 2) {
        v >>= 2;
        n += 2;
    }
    if (v >> 1) {
        v >>= 1;
        n += 1;
    }
    return n + (v != 0);
}

/*! A configuration scalar (learning rate, momentum, 1/B, ...) as value =
 * q * 2^e: a normalized significand, 2^14 <= |q| <= 2^15 - 1 (15
 * significant bits), or q = 0, e = 0 for zero. A floating exponent, not one
 * global Q15 grid, so a small learning rate keeps all 15 bits. */
typedef struct bfpScalar {
    int32_t q;
    int16_t e;
} bfpScalar_t;

/*! Encode a float once, at configuration time (float is allowed there, never
 * in the training-step roots). Significand rounded half away from zero to 15
 * bits, so the relative error is <= 2^-15; exact for 0, 1 ({2^14, -14}),
 * every power of two (subnormals included) and every value whose significand
 * fits 15 bits. A significand that rounds up to 2^15 renormalizes to 2^14
 * with e + 1. Non-finite values fail fast. */
static inline bfpScalar_t bfpScalarFromFloat(float x, const char *what) {
    if (!isfinite(x)) {
        PRINT_ERROR("%s: non-finite scalar %f has no integer encoding", what, (double)x);
        exit(1);
    }
    if (x == 0.0f) {
        return (bfpScalar_t){.q = 0, .e = 0};
    }
    int ex = 0;
    const float f = frexpf(x, &ex);            /* |f| in [0.5, 1), x = f * 2^ex */
    int32_t q = (int32_t)roundf(f * 32768.0f); /* f * 2^15 is exact */
    if (q == 32768 || q == -32768) {
        q /= 2;
        ex += 1;
    }
    return (bfpScalar_t){.q = q, .e = (int16_t)(ex - 15)};
}

/*! mant * s.q as int32, the product's exponent being E_mant + s.e. Exact and
 * overflow-free for every BFP code (m <= 16, |mant| <= 2^15) against a
 * normalized scalar (|q| <= 2^15 - 1): |product| <= 2^30 - 2^15. */
static inline int32_t bfpScalarMulMant(int32_t mant, bfpScalar_t s) {
    return mant * s.q;
}

/*! A value on the grid 2^G in quotient/remainder form:
 *   value / 2^G = q + (frac + t) / 2^32,  t in [0, 1),  sticky = (t != 0).
 * q is the FLOOR quotient (signed), frac the top 32 bits of the remainder in
 * [0, 1), sticky whether any remainder bit below those 32 is set. Nothing an
 * alignment drops is lost before the final rounding: the top 32 bits are kept
 * exactly and the rest is summarized by sticky. */
typedef struct bfpQr {
    int32_t q;
    uint32_t frac;
    bool sticky;
} bfpQr_t;

/*! floor(v / 2^k) for k in [0, 31] without right-shifting a negative value
 * (implementation-defined in C): ~v >= 0 for v < 0, and
 * ~(~v >> k) = floor(v / 2^k). */
static inline int32_t bfpFloorShr(int32_t v, unsigned k) {
    return v >= 0 ? (int32_t)((uint32_t)v >> k) : (int32_t)~((uint32_t)~v >> k);
}

/*! The term m * 2^-k on the grid. k > 0 aligns right: the dropped bits go to
 * frac/sticky, for every k (k >= 64 leaves only the sign extension in frac).
 * k <= 0 is an exact left shift; the CALLER guarantees |m| * 2^-k < 2^31
 * (k <= -32 is only legal for m == 0). */
static inline bfpQr_t bfpQrOfTerm(int32_t m, int k) {
    bfpQr_t r = {.q = 0, .frac = 0, .sticky = false};
    if (k <= 0) {
        if (k > -32) {
            r.q = (int32_t)((uint32_t)m << (unsigned)(-k));
        }
        return r;
    }
    const int32_t sign = m < 0 ? -1 : 0;
    if (k < 32) {
        r.q = bfpFloorShr(m, (unsigned)k);
        r.frac = (uint32_t)m << (32u - (unsigned)k);
    } else if (k == 32) {
        r.q = sign;
        r.frac = (uint32_t)m;
    } else if (k < 64) {
        const unsigned j = (unsigned)k - 32u; /* frac = bits [j, j + 31] of the sign-extended m */
        r.q = sign;
        r.frac = (uint32_t)bfpFloorShr(m, j);
        r.sticky = ((uint32_t)m & ((1u << j) - 1u)) != 0;
    } else {
        r.q = sign;
        r.frac = (uint32_t)sign;
        r.sticky = m != 0;
    }
    return r;
}

/*! x + y on the same grid. The CALLER guarantees the sum's floor quotient fits
 * int32. Exact when at most one operand is sticky (both sticky: frac may be
 * one unit of 2^-32 low -- the two unknown tails can carry). */
static inline bfpQr_t bfpQrAdd(bfpQr_t x, bfpQr_t y) {
    bfpQr_t r;
    r.frac = x.frac + y.frac;
    const uint32_t carry = r.frac < x.frac;
    r.q = (int32_t)((uint32_t)x.q + (uint32_t)y.q + carry);
    r.sticky = x.sticky || y.sticky;
    return r;
}

/*! Re-grid onto the 2^k times coarser grid 2^(G + k), k >= 0, exactly: the
 * integer bits that fall below the new grid move into frac, frac's low bits
 * into sticky. Any k (the D6 flush regime reaches far beyond 32). */
static inline bfpQr_t bfpQrShr(bfpQr_t x, unsigned k) {
    if (k == 0) {
        return x;
    }
    bfpQr_t r;
    const int32_t sign = x.q < 0 ? -1 : 0;
    if (k < 32) {
        r.q = bfpFloorShr(x.q, k);
        r.frac = ((uint32_t)x.q << (32u - k)) | (x.frac >> k);
        r.sticky = x.sticky || (x.frac & ((1u << k) - 1u)) != 0;
    } else if (k == 32) {
        r.q = sign;
        r.frac = (uint32_t)x.q;
        r.sticky = x.sticky || x.frac != 0;
    } else if (k < 64) {
        const unsigned j = k - 32u;
        r.q = sign;
        r.frac = (uint32_t)bfpFloorShr(x.q, j);
        r.sticky = x.sticky || x.frac != 0 || ((uint32_t)x.q & ((1u << j) - 1u)) != 0;
    } else {
        r.q = sign;
        r.frac = (uint32_t)sign;
        r.sticky = x.sticky || x.frac != 0 || x.q != 0;
    }
    return r;
}

/*! |value| rounded UP to the grid (uint32, <= 2^31): the magnitude the
 * exponent rule may use. For value >= 0 it is q plus one if any remainder is
 * left; for value < 0 it is -q (|value| = -q - f, f in [0, 1)). The CALLER
 * guarantees q < INT32_MAX when value > 0 has a remainder. */
static inline uint32_t bfpQrCeilMagnitude(bfpQr_t x) {
    if (x.q < 0) {
        return 0u - (uint32_t)x.q;
    }
    return (uint32_t)x.q + (uint32_t)(x.frac != 0 || x.sticky);
}

/*! The final rounding onto the grid, on the MAGNITUDE (the sign is applied
 * after, so SR(-x) = -SR(x) for the same draw, the Rounding.h law), clamped
 * to [-negCap, posCap]. With |value| = M + (F + t) / 2^32, t in [0, 1):
 *   HALF_AWAY: lift iff F >= 2^31 (an exact half lifts; t cannot reach the
 *              next half);
 *   SR:        lift iff u < F, u = the draw -- the shipped "lift iff u <
 *              frac" law on a 32-bit grid. For a remainder of k <= 32 bits
 *              F is the whole remainder, so this is EXACT and makes the same
 *              decision as reblockBfpRawTensor's (u >> (32 - k)) < rem from
 *              the same draw. Beyond 32 bits the probability is truncated to
 *              F / 2^32: low by less than 2^-32, and a magnitude remainder
 *              below 2^-32 of a grid step never lifts -- the documented
 *              cutoff of the one-draw-per-element rule (arithmetic-bfp.md
 *              §15).
 * A negative value converts to magnitude form exactly, using sticky:
 * |q + f| = (-q - 1) + (1 - f), and floor((1 - f) 2^32) is ~frac when a tail
 * below frac exists, else 2^32 - frac. `u` is ignored under HALF_AWAY. */
static inline int32_t bfpQrRound(bfpQr_t x, roundingMode_t rm, uint32_t u, uint32_t posCap,
                                 uint32_t negCap) {
    const bool neg = x.q < 0;
    uint32_t mag;
    uint32_t f;
    if (!neg) {
        mag = (uint32_t)x.q;
        f = x.frac;
    } else if (x.frac == 0 && !x.sticky) {
        mag = 0u - (uint32_t)x.q;
        f = 0;
    } else {
        mag = ~(uint32_t)x.q; /* -q - 1 */
        f = x.sticky ? ~x.frac : 0u - x.frac;
    }
    const bool lift = rm == SR_HALF_AWAY ? u < f : f >= 0x80000000u;
    const uint32_t cap = neg ? negCap : posCap;
    uint32_t t = mag > cap ? cap : mag + (uint32_t)lift;
    if (t > cap) {
        t = cap;
    }
    return neg ? (int32_t)(0u - t) : (int32_t)t;
}

/*! The integer add-and-round: a * 2^-ka + b * 2^-kb summed EXACTLY on a
 * working grid (quotient/remainder: the dropped bits of both alignments are
 * kept), then re-gridded kd >= 0 steps coarser and rounded ONCE (bfpQrRound).
 * The CALLER picks the working grid so that both aligned terms stay below
 * 2^30 in magnitude (then the sum fits int32) and so that at most one term
 * drops more than 32 bits -- true whenever the larger term sits within two
 * bits of 2^30 on the grid, as the accumulate engine's E_c rule guarantees.
 * Draws nothing itself: under SR the caller passes u, one draw per element. */
static inline int32_t bfpAddAndRound(int32_t a, int ka, int32_t b, int kb, unsigned kd,
                                     roundingMode_t rm, uint32_t u, uint32_t posCap,
                                     uint32_t negCap) {
    const bfpQr_t s = bfpQrAdd(bfpQrOfTerm(a, ka), bfpQrOfTerm(b, kb));
    return bfpQrRound(bfpQrShr(s, kd), rm, u, posCap, negCap);
}

/*! Cursor for the integer-fold exponent pre-pass (Conv1d, AvgPool): the groups of `qC` that the
 * flat run [start, start + count) intersects, for runs visited in increasing `start` order (groups
 * are contiguous, so no division is needed). */
typedef struct bfpRowCursor {
    size_t group;
    size_t groupEnd; /* flat end (exclusive) of `group` */
} bfpRowCursor_t;

/*! Marker of bfpRowMaxExponent: the run holds no nonzero code. Never add to
 * it (bfpLiveExpSum); a max over markers and exponents is a plain max. */
#define BFP_NO_LIVE_EXP INT_MIN

/*! a + b of two pre-pass exponents, the marker if either is one: a product
 * with a factor from a run without nonzero codes is zero, so it bounds
 * nothing. */
static inline int bfpLiveExpSum(int a, int b) {
    return (a == BFP_NO_LIVE_EXP || b == BFP_NO_LIVE_EXP) ? BFP_NO_LIVE_EXP : a + b;
}

/*! The int16 form of a pre-pass exponent for the stack tables (a live
 * exponent is in [-254, 256], so INT16_MIN is free for the marker). */
static inline int16_t bfpLiveExpTo16(int e) {
    return e == BFP_NO_LIVE_EXP ? INT16_MIN : (int16_t)e;
}
static inline int bfpLiveExpFrom16(int16_t e) {
    return e == INT16_MIN ? BFP_NO_LIVE_EXP : (int)e;
}

/*! The top of a unit (an E_acc row or raw target group) whose every
 * contribution was the marker: the zero state's exponent 0, the value every
 * all-zero operand produced before (its sums are all 0, so any finite grid
 * holds them exactly). */
static inline int bfpLiveTopOrZero(int e) {
    return e == BFP_NO_LIVE_EXP ? 0 : e;
}

/*! Max unbiased exponent over the groups holding a NONZERO code of the run
 * [start, start + count) of `codes` (the operand's unpacked int32 scratch),
 * BFP_NO_LIVE_EXP when the run holds none. A zero code adds nothing to any
 * sum, so its group's exponent bounds nothing: an all-zero group in the zero
 * state (stored = bias, E = 0) would otherwise lift E_acc above live data
 * many binades lower and round it before the final rounding. One read per
 * code at most (a group whose exponent cannot raise the max is skipped
 * unread); no division. */
static inline int bfpRowMaxExponent(bfpRowCursor_t *c, const bfpQConfig_t *qC, int32_t expBias,
                                    const int32_t *codes, size_t start, size_t count) {
    size_t stop = start + count;
    if (qC->groupSize == 0) {
        for (size_t i = start; i < stop; i++) {
            if (codes[i] != 0) {
                return (int)qC->exponents[0] - expBias;
            }
        }
        return BFP_NO_LIVE_EXP;
    }
    while (start >= c->groupEnd) {
        c->group++;
        c->groupEnd += qC->groupSize;
    }
    size_t g = c->group;
    size_t gEnd = c->groupEnd;
    int e = BFP_NO_LIVE_EXP;
    size_t i = start;
    while (i < stop) {
        size_t segEnd = gEnd < stop ? gEnd : stop;
        int eg = (int)qC->exponents[g] - expBias;
        if (eg > e) {
            for (size_t j = i; j < segEnd; j++) {
                if (codes[j] != 0) {
                    e = eg;
                    break;
                }
            }
        }
        i = segEnd;
        if (i == gEnd) {
            g++;
            gEnd += qC->groupSize;
        }
    }
    return e;
}

/*! Entry check of an integer forward kernel's BFP_RAW output over
 * `outElems` elements (validateBfpRawQConfigShape: the only guard between
 * the kernel and an out-of-bounds exponents[]/mantissa write). An empty
 * output still gets every group exponent written (0), as the BFP_RAW
 * contract requires. Returns the element count per raw group (outElems for
 * the per-tensor {1,0}). */
static inline size_t bfpRawRequireOutput(bfpRawQConfig_t *rawQC, size_t outElems,
                                         const char *what) {
    validateBfpRawQConfigShape(rawQC, outElems, what);
    if (outElems == 0) {
        for (size_t t = 0; t < rawQC->numGroups; t++) {
            rawQC->exponents[t] = 0;
        }
    }
    return rawQC->groupSize == 0 ? outElems : rawQC->groupSize;
}

/*! The raw emit of the integer grad kernels (BFP_RAW_BACKWARD_ACC): `acc`
 * holds n int32 sums, row by row (rowLen elements per row, each row on its
 * own E_acc, eAccOfRow[row]); rewritten in place as the BFP_RAW mantissas on
 * rawQC's geometry (one unbiased exponent per raw target group, which may
 * straddle rows). The exponent comes from the DATA, not from the rows' E_acc
 * bounds: E_t = max over the group's nonzero sums of E_acc + bitlen(|sum|),
 * minus 31. The sum setting it has bitlen <= 31 (the fold guard keeps every
 * sum below INT32_MAX), so it moves left by 31 - bitlen <= 30, exactly; a
 * sum moving left elsewhere has a lower top, so it fits too; a sum moving
 * right ends below its top - E_t < 31 bits, so the half-away lift cannot
 * reach 2^31. Only a sum more than 31 binades below the group's top loses
 * bits (half away, < 2^-31 of the group's largest sum). E_acc bounds would
 * place the grid by the loosest row instead -- an inflated or empty row
 * (all-zero loss group, stored exponent = bias) re-grids every other row of
 * the group coarser than the 8-bit grad step before the accumulate's SR. An
 * all-zero group writes exponent 0 (the accumulate ignores the exponent of
 * an all-zero raw group). Every group exponent is written; zero sums are
 * never shifted. Runs of (row, group) are walked with cursors, no division;
 * one bit length per run. */
static inline void bfpRawEmitFromRows(int32_t *acc, size_t n, const int *eAccOfRow, size_t rowLen,
                                      bfpRawQConfig_t *rawQC, const char *what) {
    size_t gsz = bfpRawRequireOutput(rawQC, n, what);
    if (n == 0) {
        return;
    }
    int16_t *rawE = rawQC->exponents;
    for (int pass = 0; pass < 2; pass++) {
        size_t row = 0;
        size_t rowEnd = rowLen;
        size_t t = 0;
        size_t tEnd = gsz;
        int top = INT_MIN;
        size_t i = 0;
        while (i < n) {
            size_t runEnd = rowEnd < tEnd ? rowEnd : tEnd;
            int eRow = eAccOfRow[row];
            if (pass == 0) {
                uint32_t mx = 0;
                for (size_t j = i; j < runEnd; j++) {
                    uint32_t mag = acc[j] < 0 ? 0u - (uint32_t)acc[j] : (uint32_t)acc[j];
                    mx = mag > mx ? mag : mx;
                }
                if (mx != 0) {
                    int e = eRow + (int)bfpBitLength32(mx);
                    top = e > top ? e : top;
                }
                if (runEnd == tEnd) {
                    rawE[t] = (int16_t)(top == INT_MIN ? 0 : top - 31);
                    top = INT_MIN;
                }
            } else {
                int k = (int)rawE[t] - eRow;
                for (size_t j = i; j < runEnd; j++) {
                    if (acc[j] != 0) {
                        acc[j] = bfpAlignPartial(acc[j], k);
                    }
                }
            }
            if (runEnd == rowEnd) {
                row++;
                rowEnd += rowLen;
            }
            if (runEnd == tEnd) {
                t++;
                tEnd += gsz;
            }
            i = runEnd;
        }
    }
}

/*! The guard of a pure code SUM (AvgPool): the product proof with the second
 * factor bounded by 1 (`mb = 1`: 2^(m+1-2) = 2^(m-1) = max |code|), i.e. the
 * smallest g with K 2^(m-1) 2^-g + (K + 1) <= INT32_MAX. */
static inline int bfpSumFoldGuardBits(size_t terms, uint8_t mantissaBits, const char *what) {
    return bfpFoldGuardBits(terms, mantissaBits, 1, 0, what);
}

/*! Exact value compare ma * 2^ea > mb * 2^eb of two BFP codes (|code| <=
 * 2^15, m <= 16), no float: the code under the larger exponent is shifted
 * left by the gap d in uint32 (|code| 2^d <= 2^30 for d <= 15). For d >= 16
 * the shifted side has magnitude >= 2^16 > |other| unless it is zero, so its
 * sign decides -- and a zero there loses exactly to a negative other. Equal
 * values are not greater (the caller's first-strict-greater tie rule). */
static inline bool bfpValueGreater(int32_t ma, int ea, int32_t mb, int eb) {
    if (ea == eb) {
        return ma > mb;
    }
    if (ea > eb) {
        int d = ea - eb;
        if (d >= 16) {
            return ma > 0 || (ma == 0 && mb < 0);
        }
        return (int32_t)((uint32_t)ma << (unsigned)d) > mb;
    }
    int d = eb - ea;
    if (d >= 16) {
        return mb < 0 || (mb == 0 && ma > 0);
    }
    return ma > (int32_t)((uint32_t)mb << (unsigned)d);
}

/*! sum / count rounded half away from zero, count > 0 -- roundByMode
 * (HALF_AWAY) without leaving the integer domain, exact over the whole int32
 * range (uint32 magnitude; the tie test 2 rem >= count is written as
 * rem >= count - rem, so nothing overflows). One division per call. */
static inline int32_t roundedDivHalfAwayInt32(int32_t sum, int32_t count) {
    uint32_t mag = sum < 0 ? 0u - (uint32_t)sum : (uint32_t)sum;
    uint32_t d = (uint32_t)count;
    uint32_t q = mag / d;
    uint32_t rem = mag - q * d;
    q += rem >= d - rem;
    return sum < 0 ? (int32_t)(0u - q) : (int32_t)q;
}

/*! Max codes one int32 SUM partial may accumulate (PR3, biasGrad-family
 * reductions): a pure mantissa sum of g codes is bounded by g * 2^(m-1)
 * (codes in [-2^(m-1), 2^(m-1)-1]), so an int32 segment partial is sound for
 * g <= INT32_MAX >> (m-1). The PRODUCT helper's >> (ma+mb-2) bound does not
 * apply to single-operand sums. */
static inline size_t bfpSumSegmentLimit(uint8_t mantissaBits) {
    if (mantissaBits == 0) {
        PRINT_ERROR("bfpSumSegmentLimit: mantissaBits == 0 is not a BFP width (shift-UB; "
                    "the SYM twin rejects qMaxBits == 0 the same way)");
        exit(1);
    }
    return (size_t)(INT32_MAX >> (mantissaBits - 1));
}

/*! Fail-fast sum-headroom twin of bfpValidateBlockHeadroom, for kernels that
 * SUM one BFP operand's mantissas (no products): a same-exponent segment
 * never accumulates more than min(groupSize, reductionLen) codes (a group
 * holds groupSize storage elements total, so no walk -- strided or not --
 * visits more of one group; per-tensor {1,0} caps at the reduction length),
 * so that bound must stay within bfpSumSegmentLimit. */
static inline void bfpValidateSumHeadroom(const bfpQConfig_t *qC, size_t reductionLen,
                                          const char *what) {
    size_t maxSeg = qC->groupSize == 0 ? reductionLen : qC->groupSize;
    if (maxSeg > reductionLen) {
        maxSeg = reductionLen;
    }
    size_t limit = bfpSumSegmentLimit(qC->mantissaBits);
    if (maxSeg > limit) {
        PRINT_ERROR("%s: BFP sum partial would overflow int32 -- max same-exponent segment "
                    "%zu exceeds %zu codes for mantissa width %u",
                    what, maxSeg, limit, qC->mantissaBits);
        exit(1);
    }
}

/*! BFP epic PR4 (R-P2/R-P5): one wire's length gate. validateBfpQConfigShape
 * alone is NOT sufficient for the outside-funnel copiers: the per-tensor
 * sentinel {numGroups=1, groupSize=0} is legal for ANY element count, so two
 * per-tensor wires of DIFFERENT lengths pass every grid check while a verbatim
 * payload copy sized off one of them runs off the other's buffer. The element
 * count is therefore checked explicitly, next to the grid. */
static inline void bfpRequireElementCount(const bfpQConfig_t *qC, size_t elems, size_t expected,
                                          const char *what) {
    if (elems != expected) {
        PRINT_ERROR("%s: BFP wires must hold the same number of elements for a verbatim "
                    "code/exponent carry -- got %zu vs %zu (a per-tensor {1, 0} grid matches "
                    "any length, so the grid check alone cannot catch this)",
                    what, elems, expected);
        exit(1);
    }
    validateBfpQConfigShape(qC, elems);
}

/*! BFP epic PR4 (R-P2/R-P5): verbatim-transparency gate for the layers that
 * run OUTSIDE the executeOp funnel and COPY codes and exponents between two
 * BFP wires (Relu, Flatten, Dropout) instead of re-deriving them. Copying is
 * sound only when both wires hold the same number of elements AND share the
 * block grid AND the code/exponent widths; anything else is a re-block, which
 * belongs to the Quantization layer (the sanctioned re-block point,
 * docs/conventions/arithmetic-bfp.md §9 / BFP D8). */
static inline void bfpRequireSameGeometry(const bfpQConfig_t *aQC, size_t aElems,
                                          const bfpQConfig_t *bQC, size_t bElems,
                                          const char *what) {
    bfpRequireElementCount(bQC, bElems, aElems, what);
    validateBfpQConfigShape(aQC, aElems);
    if (aQC->numGroups != bQC->numGroups || aQC->groupSize != bQC->groupSize ||
        aQC->mantissaBits != bQC->mantissaBits || aQC->exponentBits != bQC->exponentBits) {
        PRINT_ERROR("%s: BFP wires must share {numGroups, groupSize, mantissaBits, "
                    "exponentBits} for a verbatim code/exponent carry -- got {%zu, %zu, %u, %u} "
                    "vs {%zu, %zu, %u, %u}; insert a Quantization layer to re-block",
                    what, aQC->numGroups, aQC->groupSize, (unsigned)aQC->mantissaBits,
                    (unsigned)aQC->exponentBits, bQC->numGroups, bQC->groupSize,
                    (unsigned)bQC->mantissaBits, (unsigned)bQC->exponentBits);
        exit(1);
    }
}

/*! PR6 (PR5 deferral 4): the R-P1/R-N1 produced-wire staging anchor, shared.
 * outputQ anchors forward ops, propLossQ anchors backward ops. Eager at op
 * entry; NULL-checked because userApi factories copy layerQuant_t slots by
 * value. */
static inline const bfpQConfig_t *bfpWireAnchor(const quantization_t *wireQ, const char *what) {
    if (wireQ == NULL || wireQ->type != BFP) {
        PRINT_ERROR("%s: ARITH_BFP requires a BFP-typed produced-wire config as the staging "
                    "width anchor (outputQ forward / propLossQ backward) -- see "
                    "docs/conventions/arithmetic-bfp.md",
                    what);
        exit(1);
    }
    return wireQ->qConfig;
}

/*! PR6 (PR5 deferral 4): fail-fast for paths without BFP semantics, shared. */
static inline void bfpRequireNoBfpWire(const tensor_t *t, const char *what) {
    if (t->quantization->type == BFP) {
        PRINT_ERROR("%s: no BFP semantics on this path -- keep BFP off this wire or use "
                    "FLOAT32 wires",
                    what);
        exit(1);
    }
}

#endif // BFP_KERNEL_SUPPORT_H
