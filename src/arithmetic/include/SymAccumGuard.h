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
 *     declared widths and the op's actual seed: N_max * 2^(qa-1) * 2^(qb-1) +
 *     max|seed| <= INT32_MAX. The full two's-complement magnitude 2^(q-1) is
 *     used because -2^(q-1) IS a reachable code (fixed-scale requant
 *     saturation clamps to qMin). When the bound holds the kernel runs its
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
 * width can defeat the proof.
 *
 * Kernels use the pattern
 *     symAccumSite_t site = symAccumSiteInit(op, nMax, qa, qb, maxAbsSeed);
 *     if (symAccumProvenSafe(&site)) { loops(false, &site); }
 *     else                           { loops(true,  &site); }
 * with `loops` a static inline always_inline function whose `checked`
 * argument is a compile-time constant at both call sites, and the step
 * `acc = symMac(acc, a, b, checked, &site)`. Out of scope: value-sum
 * accumulators; Reduce and the pools check N < 2^(32-q) themselves, the
 * Linear / Conv bias-grad sums have no run-time guard yet (follow-up).
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
    uint8_t aBits;  /* operand widths as declared (banner) */
    uint8_t bBits;
    uint64_t maxAbsSeed; /* max |seed|, up to 2^31 for INT32_MIN */
    bool wideProduct;    /* either width > 16: |a*b| may leave int32 */
    bool rescaled;       /* terms are float-rescaled partials: use the slacked bound */
} symAccumSite_t;

/* max |seed[i]| as an unsigned magnitude (INT32_MIN -> 2^31). NULL / n == 0 -> 0. */
static inline uint64_t symAccumMaxAbsSeed(const int32_t *seed, size_t n) {
    uint64_t m = 0;
    for (size_t i = 0; seed != NULL && i < n; i++) {
        int64_t v = seed[i];
        uint64_t mag = (uint64_t)(v < 0 ? -v : v);
        if (mag > m) {
            m = mag;
        }
    }
    return m;
}

/* Largest N for which N * 2^(qa-1) * 2^(qb-1) + maxAbsSeed <= INT32_MAX; 0 when
 * even one term is not provable. The comparison is a division so neither the
 * bound nor a caller's N (up to SIZE_MAX) can wrap. A zero width is read as 1
 * (degenerate) and widths clamp at 32 (an int32 code is at most 2^31). */
static inline size_t symAccumTermLimit(uint8_t aBits, uint8_t bBits, uint64_t maxAbsSeed) {
    unsigned qa = aBits < 1 ? 1u : (aBits > 32 ? 32u : aBits);
    unsigned qb = bBits < 1 ? 1u : (bBits > 32 ? 32u : bBits);
    uint64_t maxProduct = (uint64_t)1 << (qa + qb - 2u);
    if (maxAbsSeed > (uint64_t)INT32_MAX || maxProduct > (uint64_t)INT32_MAX) {
        return 0;
    }
    return (size_t)(((uint64_t)INT32_MAX - maxAbsSeed) / maxProduct);
}

/* Same as symAccumTermLimit for accumulators that receive RESCALED terms
 * (grouped kernels: round((p * s_g) / sAcc), scale ratio <= 1). In exact
 * arithmetic |term| <= |p|, but the float32 evaluation can land one ulp
 * (2^-23 relative) above |p| and a stochastic round adds up to 1 more, so each
 * term is bounded by maxProduct * (1 + 2^-22) + 1 (twice the ulp slack, kept
 * as the margin). With room = INT32_MAX - maxAbsSeed this is
 *     N * (mp + 1) + N * mp / 2^22 <= room
 *  <=> N <= room * 2^22 / ((mp + 1) * 2^22 + mp),
 * evaluated in 64-bit (room * 2^22 and the denominator are < 2^54). */
static inline size_t symAccumTermLimitRescaled(uint8_t aBits, uint8_t bBits, uint64_t maxAbsSeed) {
    unsigned qa = aBits < 1 ? 1u : (aBits > 32 ? 32u : aBits);
    unsigned qb = bBits < 1 ? 1u : (bBits > 32 ? 32u : bBits);
    uint64_t maxProduct = (uint64_t)1 << (qa + qb - 2u);
    if (maxAbsSeed > (uint64_t)INT32_MAX || maxProduct > (uint64_t)INT32_MAX) {
        return 0;
    }
    uint64_t room = (uint64_t)INT32_MAX - maxAbsSeed;
    return (size_t)((room << 22) / (((maxProduct + 1u) << 22) + maxProduct));
}

static inline symAccumSite_t symAccumSiteInit(const char *op, size_t nMax, uint8_t aBits,
                                              uint8_t bBits, uint64_t maxAbsSeed) {
    symAccumSite_t s = {.op = op,
                        .nMax = nMax,
                        .aBits = aBits,
                        .bBits = bBits,
                        .maxAbsSeed = maxAbsSeed,
                        .wideProduct = aBits > 16 || bBits > 16,
                        .rescaled = false};
    return s;
}

/* Site for an accumulator of float-rescaled terms (see symAccumTermLimitRescaled). */
static inline symAccumSite_t symAccumSiteInitRescaled(const char *op, size_t nMax, uint8_t aBits,
                                                      uint8_t bBits, uint64_t maxAbsSeed) {
    symAccumSite_t s = symAccumSiteInit(op, nMax, aBits, bBits, maxAbsSeed);
    s.rescaled = true;
    return s;
}

/* Zero terms never accumulate, so N == 0 is safe whatever the widths. */
static inline bool symAccumProvenSafe(const symAccumSite_t *s) {
    if (s->nMax == 0) {
        return true;
    }
    size_t limit = s->rescaled ? symAccumTermLimitRescaled(s->aBits, s->bBits, s->maxAbsSeed)
                               : symAccumTermLimit(s->aBits, s->bBits, s->maxAbsSeed);
    return s->nMax <= limit;
}

/* Unconditional (all builds): the exit(1) stays even where PRINT_ERROR compiles out. */
static inline void symAccumAbort(const symAccumSite_t *s) {
    PRINT_ERROR("%s: SYM int32 accumulator overflow -- N=%zu terms, operand widths (%u, %u), "
                "max|seed|=%lu; narrow ODT_SYM_OPERAND_QMAXBITS (one lever) or run this op "
                "under ARITH_FLOAT32",
                s->op, s->nMax, (unsigned)s->aBits, (unsigned)s->bBits,
                (unsigned long)s->maxAbsSeed);
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
