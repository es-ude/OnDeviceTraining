#ifndef SYM_ACCUM_GUARD_H
#define SYM_ACCUM_GUARD_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "Common.h"
#include "Mul.h"

/* SYM int32 accumulator guard. Every SYM_INT32 product accumulator is
 * int32 (no int64 in kernels), so a long enough reduction of wide enough
 * codes wraps silently -- signed overflow, UB. The guard is a hybrid:
 *
 *  1. A STATIC bound per op proves the accumulator safe from the operands'
 *     signed code ranges and the op's actual seed: N_max * max|term| +
 *     max|seed| <= INT32_MAX, |term| over the product's four corners. A
 *     declared width q reads as the full two's-complement range
 *     [-2^(q-1), 2^(q-1) - 1] because -2^(q-1) IS a reachable code
 *     (fixed-scale requant saturation clamps to qMin), so an int12 x int12
 *     term is 2^22; a grouped weight's range comes from its code domain
 *     instead (SymGroupedWeight.h). When the bound holds the kernel runs its
 *     loop body unchanged -- same instructions, same TRACK_INSTRUCTIONS
 *     counters, same gold output.
 *  2. Otherwise the SAME loop body is instantiated with `checked == true`:
 *     every accumulate goes through __builtin_add_overflow (and the product
 *     through __builtin_mul_overflow when either width exceeds 16, where the
 *     product alone can leave int32), and the first real overflow aborts.
 *     A reduction the bound cannot prove but whose values happen to fit
 *     still returns the exact result.
 *
 * The bound arithmetic is 64-bit; the accumulator never is.
 *
 * ASSUMPTION: stored codes obey their declared qMaxBits/qBits. The width is
 * metadata and is not re-checked per code. Every funnel staging route
 * declares the true width of the codes it writes and the requant cells clamp
 * to it, so only a hand-built tensor whose codes exceed its own declared
 * width can defeat the proof. One carve-out: the funnel's grouped scratch
 * declares the STORED qBits (poisoned alongside the scale, ExecuteOp.h)
 * although a grouped ASYM weight unpacks to code - zp[g], one code wider on
 * each side; the grouped kernels take that operand's range from their
 * symGroupedWeight_t ctx (SymGroupedWeight.h), never from the scratch
 * declaration. Every public raw producer (matmul family, Conv1d / ConvT
 * kernels, mul, equal-scale add, linearCalcBiasGradsSymInt32; the Linear
 * weight grad and dx through the matmul) declares the static result width of
 * the accumulator it was written under (symAccumResultBits): the smallest
 * two's-complement width holding the accumulator's exact range, 32 meaning
 * the full raw int32 range (#516). The Conv1d / ConvT weight-grad scatter and
 * bias-grad sum are file-static funnel kernels: only the ACC epilogue sees
 * their raw wire, and it reads the scale and the target width, never the raw
 * width, so they keep the scratch width.
 *
 * Kernels use the pattern
 *     symAccumSite_t site = symAccumSiteInit(op, nMax, qa, qb, 0);
 *     symAccumSiteSetSeeds(&site, seed, nSeeds);   (a seeded kernel)
 *     if (symAccumProvenSafe(&site)) { loops(false, &site); }
 *     else                           { loops(true,  &site); }
 * with `loops` a static inline always_inline function whose `checked`
 * argument is a compile-time constant at both call sites, and the step
 * `acc = symMac(acc, a, b, checked, &site)`. An operand whose range is not a
 * declared width (the grouped kernels' weight) builds the site from explicit
 * signed ranges instead: symAccumSiteInitRanges(op, nMax, aMin, aMax, bMin,
 * bMax, 0); the width-based inits are that with the two's-complement ranges.
 *
 * VALUE SUMS (acc += code, no product): symAccumSiteInitSum is the same site
 * with b = the constant 1, so the proof reads N * 2^(q-1) <= INT32_MAX, and
 * the loop step is `acc = symAdd(acc, code, checked, &site)`. Used by the
 * Linear / Conv1d / Conv1dTransposed bias-grad sums. The checked loop aborts
 * on the first INTERMEDIATE overflow even when the final sum would have fit
 * (a mixed-sign run can leave int32 and come back); the proven loop never
 * overflows. Reduce and the pools still check N < 2^(32-q) themselves.
 * Header-only static inline, like BfpKernelSupport.h. */

#if defined(__GNUC__) || defined(__clang__)
#define SYM_ACCUM_ALWAYS_INLINE static inline __attribute__((always_inline))
#else
#define SYM_ACCUM_ALWAYS_INLINE static inline
#endif

/* The safe and the checked instantiation of a kernel's loop body each live in
 * their OWN non-inlined function. Inlined side by side into one function they
 * share a stack frame, which grows enough that the loops around the MAC spill
 * and every far frame access costs two instructions; separate frames keep the
 * safe path's register allocation identical to the unguarded kernel. */
#if defined(__GNUC__) || defined(__clang__)
#define SYM_ACCUM_NOINLINE static __attribute__((noinline))
#else
#define SYM_ACCUM_NOINLINE static
#endif

typedef struct symAccumSite {
    const char *op; /* names the op in the abort banner */
    size_t nMax;    /* max terms one accumulator receives */
    int32_t aMin;   /* signed code range of each operand: the proof's and the */
    int32_t aMax;   /* result width's input */
    int32_t bMin;
    int32_t bMax;
    uint8_t aBits; /* the holding width of each range (banner, wide-product check) */
    uint8_t bBits;
    uint64_t maxAbsSeed; /* max |seed|, up to 2^31 for INT32_MIN: the proof's seed room (banner) */
    int32_t seedMax;     /* signed extremes of the per-accumulator seeds: the result width's */
    int32_t seedMin;     /* two sides (no seed: [0, 0], the value a seedless kernel starts from) */
    bool wideProduct;    /* either width > 16: |a*b| may leave int32 */
    bool rescaled;       /* terms are float-rescaled partials: use the slacked bound */
    bool valueSum;       /* plain sum of stored codes (banner names the wire, not the knob) */
} symAccumSite_t;

/* |v| as an unsigned magnitude (INT32_MIN -> 2^31). */
static inline uint64_t symAccumMagnitude(int32_t v) {
    int64_t w = v;
    return (uint64_t)(w < 0 ? -w : w);
}

/* max |seed[i]| as an unsigned magnitude (INT32_MIN -> 2^31). NULL / n == 0 -> 0. */
static inline uint64_t symAccumMaxAbsSeed(const int32_t *seed, size_t n) {
    uint64_t m = 0;
    for (size_t i = 0; seed != NULL && i < n; i++) {
        uint64_t mag = symAccumMagnitude(seed[i]);
        if (mag > m) {
            m = mag;
        }
    }
    return m;
}

/* The seeds a site's accumulators start from, as the signed range
 * [seedMin, seedMax]: the proof reads their max magnitude, the result width
 * each side's extreme. */
static inline void symAccumSiteSetSeedRange(symAccumSite_t *s, int32_t seedMin, int32_t seedMax) {
    uint64_t minMag = symAccumMagnitude(seedMin);
    uint64_t maxMag = symAccumMagnitude(seedMax);
    s->seedMin = seedMin;
    s->seedMax = seedMax;
    s->maxAbsSeed = minMag > maxMag ? minMag : maxMag;
}

/* ... from the per-accumulator seed array; NULL / n == 0 -> [0, 0]. */
static inline void symAccumSiteSetSeeds(symAccumSite_t *s, const int32_t *seed, size_t n) {
    int32_t lo = 0, hi = 0;
    for (size_t i = 0; seed != NULL && i < n; i++) {
        lo = (i == 0 || seed[i] < lo) ? seed[i] : lo;
        hi = (i == 0 || seed[i] > hi) ? seed[i] : hi;
    }
    symAccumSiteSetSeedRange(s, lo, hi);
}

/* A declared width as the bound arithmetic reads it: a zero width is read as
 * 1 (degenerate) and widths clamp at 32 (an int32 code is at most 2^31). */
static inline unsigned symAccumClampedWidth(uint8_t bits) {
    return bits < 1 ? 1u : (bits > 32 ? 32u : bits);
}

/* Smallest two's-complement width w in [1, 32] holding the range
 * [-maxNegMag, maxPos]: maxPos <= 2^(w-1) - 1 and maxNegMag <= 2^(w-1) (the
 * negative side holds one more magnitude); 32 when the range leaves 31 bits
 * -- the raw int32 range, which the guard already reads as unprovable. The
 * guard's |code| <= 2^(w-1) reading of a declared width stays sound. */
static inline uint8_t symAccumWidthForRange(uint64_t maxPos, uint64_t maxNegMag) {
    uint8_t w = 1;
    while (w < 32 &&
           (maxPos > ((uint64_t)1 << (w - 1u)) - 1u || maxNegMag > ((uint64_t)1 << (w - 1u)))) {
        w++;
    }
    return w;
}

/* The two's-complement range a declared width reads as, [-2^(q-1), 2^(q-1) - 1]
 * (q clamped as above, so 32 is the full int32 range). */
static inline void symAccumWidthRange(uint8_t bits, int32_t *min, int32_t *max) {
    uint64_t half = (uint64_t)1 << (symAccumClampedWidth(bits) - 1u);
    *min = (int32_t)-(int64_t)half;
    *max = (int32_t)(half - 1u);
}

/* The holding width of a signed code range (its two's-complement width). */
static inline uint8_t symAccumRangeWidth(int32_t min, int32_t max) {
    return symAccumWidthForRange(max > 0 ? (uint64_t)max : 0,
                                 min < 0 ? (uint64_t)(-(int64_t)min) : 0);
}

/* Range of one product over two signed operand ranges: a product is monotone
 * in each factor, so its extremes are the largest and the smallest of the four
 * corners; the positive side and the negative magnitude each floor at 0. */
static inline void symAccumCornerRange(int32_t aMin, int32_t aMax, int32_t bMin, int32_t bMax,
                                       uint64_t *maxPos, uint64_t *maxNegMag) {
    int64_t corner[4] = {(int64_t)aMin * bMin, (int64_t)aMin * bMax, (int64_t)aMax * bMin,
                         (int64_t)aMax * bMax};
    int64_t hi = corner[0], lo = corner[0];
    for (size_t i = 1; i < 4; i++) {
        hi = corner[i] > hi ? corner[i] : hi;
        lo = corner[i] < lo ? corner[i] : lo;
    }
    *maxPos = hi > 0 ? (uint64_t)hi : 0;
    *maxNegMag = lo < 0 ? (uint64_t)(-lo) : 0;
}

/* Range of one product of declared widths (qa, qb): the positive corner
 * (-2^(qa-1)) * (-2^(qb-1)) = 2^(qa-1) * 2^(qb-1), the negative one
 * -2^(qa-1) * (2^(qb-1) - 1) or its mirror, whichever is larger -- one short
 * of the positive corner by the narrower operand's half range. */
static inline void symAccumProductRange(uint8_t aBits, uint8_t bBits, uint64_t *maxPos,
                                        uint64_t *maxNegMag) {
    int32_t aMin, aMax, bMin, bMax;
    symAccumWidthRange(aBits, &aMin, &aMax);
    symAccumWidthRange(bBits, &bMin, &bMax);
    symAccumCornerRange(aMin, aMax, bMin, bMax, maxPos, maxNegMag);
}

/* Site over explicit signed operand ranges (each a range the operand's codes
 * actually obey). Only the seed's magnitude is known here, so the seed range
 * is read as [-M, +M] within int32 (a +2^31 seed does not exist; INT32_MIN
 * does); a kernel that knows its seed array narrows it with
 * symAccumSiteSetSeeds. */
static inline symAccumSite_t symAccumSiteInitRanges(const char *op, size_t nMax, int32_t aMin,
                                                    int32_t aMax, int32_t bMin, int32_t bMax,
                                                    uint64_t maxAbsSeed) {
    uint8_t aBits = symAccumRangeWidth(aMin, aMax);
    uint8_t bBits = symAccumRangeWidth(bMin, bMax);
    symAccumSite_t s = {
        .op = op,
        .nMax = nMax,
        .aMin = aMin,
        .aMax = aMax,
        .bMin = bMin,
        .bMax = bMax,
        .aBits = aBits,
        .bBits = bBits,
        .maxAbsSeed = maxAbsSeed,
        .seedMax = maxAbsSeed > (uint64_t)INT32_MAX ? INT32_MAX : (int32_t)maxAbsSeed,
        .seedMin = maxAbsSeed >= ((uint64_t)1 << 31) ? INT32_MIN : -(int32_t)maxAbsSeed,
        .wideProduct = aBits > 16 || bBits > 16,
        .rescaled = false,
        .valueSum = false};
    return s;
}

/* Site over explicit ranges for an accumulator of float-rescaled terms (see
 * symAccumSiteTermLimit). */
static inline symAccumSite_t symAccumSiteInitRangesRescaled(const char *op, size_t nMax,
                                                            int32_t aMin, int32_t aMax,
                                                            int32_t bMin, int32_t bMax,
                                                            uint64_t maxAbsSeed) {
    symAccumSite_t s = symAccumSiteInitRanges(op, nMax, aMin, aMax, bMin, bMax, maxAbsSeed);
    s.rescaled = true;
    return s;
}

/* Site over declared widths: each operand's two's-complement range. */
static inline symAccumSite_t symAccumSiteInit(const char *op, size_t nMax, uint8_t aBits,
                                              uint8_t bBits, uint64_t maxAbsSeed) {
    int32_t aMin, aMax, bMin, bMax;
    symAccumWidthRange(aBits, &aMin, &aMax);
    symAccumWidthRange(bBits, &bMin, &bMax);
    return symAccumSiteInitRanges(op, nMax, aMin, aMax, bMin, bMax, maxAbsSeed);
}

/* Site for a plain sum of N stored codes of declared width `bits` (no product,
 * no seed): the code range against the constant 1 -- never a mirrored one-bit
 * code, which would cost a bit (the banner still names a one-bit second width). */
static inline symAccumSite_t symAccumSiteInitSum(const char *op, size_t nMax, uint8_t bits) {
    int32_t min, max;
    symAccumWidthRange(bits, &min, &max);
    symAccumSite_t s = symAccumSiteInitRanges(op, nMax, min, max, 1, 1, 0);
    s.bBits = 1;
    s.valueSum = true;
    return s;
}

/* Site over declared widths for an accumulator of float-rescaled terms (see
 * symAccumSiteTermLimit). */
static inline symAccumSite_t symAccumSiteInitRescaled(const char *op, size_t nMax, uint8_t aBits,
                                                      uint8_t bBits, uint64_t maxAbsSeed) {
    symAccumSite_t s = symAccumSiteInit(op, nMax, aBits, bBits, maxAbsSeed);
    s.rescaled = true;
    return s;
}

/* Range of one term the site's accumulator receives: the product's corners
 * over the operand ranges (a value sum's second range is [1, 1], so this is
 * the stored code's range). */
static inline void symAccumTermRange(const symAccumSite_t *s, uint64_t *maxPos,
                                     uint64_t *maxNegMag) {
    symAccumCornerRange(s->aMin, s->aMax, s->bMin, s->bMax, maxPos, maxNegMag);
}

/* Largest N for which N * max|term| + maxAbsSeed <= INT32_MAX, max|term| the
 * larger side of symAccumTermRange; 0 when even one term is not provable. The
 * comparison is a division so neither the bound nor a caller's N (up to
 * SIZE_MAX) can wrap.
 *
 * A RESCALED site's accumulator receives round((p * s_g) / sAcc) terms
 * (grouped kernels, scale ratio <= 1). In exact arithmetic |term| <= |p|, but
 * the float32 evaluation can land one ulp (2^-23 relative) above |p| and a
 * stochastic round adds up to 1 more, so each term is bounded by
 * mp * (1 + 2^-22) + 1 (twice the ulp slack, kept as the margin). With
 * room = INT32_MAX - maxAbsSeed this is
 *     N * (mp + 1) + N * mp / 2^22 <= room
 *  <=> N <= room * 2^22 / ((mp + 1) * 2^22 + mp),
 * evaluated in 64-bit (room * 2^22 and the denominator are < 2^54). */
static inline size_t symAccumSiteTermLimit(const symAccumSite_t *s) {
    uint64_t termPos, termNeg;
    symAccumTermRange(s, &termPos, &termNeg);
    uint64_t term = termPos > termNeg ? termPos : termNeg;
    if (s->maxAbsSeed > (uint64_t)INT32_MAX || term > (uint64_t)INT32_MAX) {
        return 0;
    }
    uint64_t room = (uint64_t)INT32_MAX - s->maxAbsSeed;
    if (s->rescaled) {
        return (size_t)((room << 22) / (((term + 1u) << 22) + term));
    }
    return term == 0 ? SIZE_MAX : (size_t)(room / term);
}

/* The term limit of declared widths (qa, qb): N * 2^(qa-1) * 2^(qb-1) +
 * maxAbsSeed <= INT32_MAX. */
static inline size_t symAccumTermLimit(uint8_t aBits, uint8_t bBits, uint64_t maxAbsSeed) {
    symAccumSite_t s = symAccumSiteInit("", 0, aBits, bBits, maxAbsSeed);
    return symAccumSiteTermLimit(&s);
}

/* ... and of a rescaled site over declared widths (the slacked bound above). */
static inline size_t symAccumTermLimitRescaled(uint8_t aBits, uint8_t bBits, uint64_t maxAbsSeed) {
    symAccumSite_t s = symAccumSiteInitRescaled("", 0, aBits, bBits, maxAbsSeed);
    return symAccumSiteTermLimit(&s);
}

/* Zero terms never accumulate, so N == 0 is safe whatever the ranges. */
static inline bool symAccumProvenSafe(const symAccumSite_t *s) {
    return s->nMax == 0 || s->nMax <= symAccumSiteTermLimit(s);
}

/* One side of a proven site's range: nMax terms of per-term bound `term`, the
 * rescaled site's slack on top (nMax * term * (1 + 2^-22) + nMax, the ceil
 * taken once -- the same per-term bound symAccumSiteTermLimit proves),
 * then shifted by the seed's extreme on this side (`seedShift`, the signed
 * seed that widens it: seedMax on the positive side, -seedMin on the negative)
 * and floored at 0: a side the seed pushes past zero costs no bits. */
static inline uint64_t symAccumSideBound(const symAccumSite_t *s, uint64_t term,
                                         int64_t seedShift) {
    uint64_t terms = (uint64_t)s->nMax * term;
    uint64_t bound = terms;
    if (s->rescaled) {
        bound += (uint64_t)s->nMax + ((terms + (((uint64_t)1 << 22) - 1u)) >> 22);
    }
    if (seedShift >= 0) {
        return bound + (uint64_t)seedShift;
    }
    uint64_t back = (uint64_t)(-seedShift);
    return bound > back ? bound - back : 0;
}

/* Declared width of an accumulator a kernel wrote under this site (#516):
 * 32 when !symAccumProvenSafe(s) (the bound exceeds int32, the checked loop
 * ran, the codes span the raw int32 range); otherwise the smallest
 * two's-complement width holding the accumulator's exact range
 *     [-(nMax * termNeg) + seedMin, nMax * termPos + seedMax]
 * (each side slacked on a rescaled site). Both sides stay within the
 * magnitude bound the term limit proves (max(termPos, termNeg), |seed| <=
 * maxAbsSeed), so provenSafe guarantees they fit INT32_MAX and nothing wraps
 * in the 64-bit evaluation. nMax == 0 -> the width of the seed range alone. */
static inline uint8_t symAccumResultBits(const symAccumSite_t *s) {
    if (!symAccumProvenSafe(s)) {
        return 32;
    }
    uint64_t termPos, termNeg;
    symAccumTermRange(s, &termPos, &termNeg);
    uint64_t maxPos = symAccumSideBound(s, termPos, s->seedMax);
    uint64_t maxNegMag = symAccumSideBound(s, termNeg, -(int64_t)s->seedMin);
    return symAccumWidthForRange(maxPos, maxNegMag);
}

/* Unconditional (all builds): the exit(1) stays even where PRINT_ERROR compiles out. */
static inline void symAccumAbort(const symAccumSite_t *s) {
    if (s->valueSum) {
        /* The summed wire is stored, so ODT_SYM_OPERAND_QMAXBITS (an operand clamp) is no lever. */
        PRINT_ERROR("%s: SYM int32 accumulator overflow -- N=%zu terms, value sum of %u-bit "
                    "codes; narrow the producing wire's qMaxBits or run this op under "
                    "ARITH_FLOAT32",
                    s->op, s->nMax, (unsigned)s->aBits);
        exit(1);
    }
    PRINT_ERROR("%s: SYM int32 accumulator overflow -- N=%zu terms, operand widths (%u, %u), "
                "max|seed|=%zu; narrow ODT_SYM_OPERAND_QMAXBITS (one lever) or run this op "
                "under ARITH_FLOAT32",
                s->op, s->nMax, (unsigned)s->aBits, (unsigned)s->bBits, (size_t)s->maxAbsSeed);
    exit(1);
}

static inline int32_t symAccumAddChecked(int32_t acc, int32_t term, const symAccumSite_t *s) {
    int32_t r;
    if (__builtin_add_overflow(acc, term, &r)) {
        symAccumAbort(s);
    }
    return r;
}

/* a*b with the wide-width product check: for widths <= 16 |a*b| <= 2^30
 * already, so only a wide site tests the multiply. Goes through mulInt32s so
 * TRACK_INSTRUCTIONS counts it exactly like the unchecked path. */
static inline int32_t symAccumProductChecked(int32_t a, int32_t b, const symAccumSite_t *s) {
    if (s->wideProduct) {
        int32_t unused;
        if (__builtin_mul_overflow(a, b, &unused)) {
            symAccumAbort(s);
        }
    }
    return mulInt32s(a, b);
}

/* acc + a*b with every step checked. */
static inline int32_t symAccumMacChecked(int32_t acc, int32_t a, int32_t b,
                                         const symAccumSite_t *s) {
    return symAccumAddChecked(acc, symAccumProductChecked(a, b, s), s);
}

/* A standalone product (the grouped ConvT scatter rescales each product before
 * accumulating it); `checked` must be a compile-time constant. */
SYM_ACCUM_ALWAYS_INLINE int32_t symProduct(int32_t a, int32_t b, const bool checked,
                                           const symAccumSite_t *s) {
    if (!checked) {
        return mulInt32s(a, b);
    }
    return symAccumProductChecked(a, b, s);
}

/* One accumulate step; `checked` must be a compile-time constant at the call
 * site so the unchecked instantiation is exactly `acc + mulInt32s(a, b)`. */
SYM_ACCUM_ALWAYS_INLINE int32_t symMac(int32_t acc, int32_t a, int32_t b, const bool checked,
                                       const symAccumSite_t *s) {
    if (!checked) {
        return acc + mulInt32s(a, b);
    }
    return symAccumMacChecked(acc, a, b, s);
}

/* Accumulate an already-formed term (rescaled partial, post-scatter bias seed). */
SYM_ACCUM_ALWAYS_INLINE int32_t symAdd(int32_t acc, int32_t term, const bool checked,
                                       const symAccumSite_t *s) {
    if (!checked) {
        return acc + term;
    }
    return symAccumAddChecked(acc, term, s);
}

#endif // SYM_ACCUM_GUARD_H
