#ifndef BFP_KERNEL_SUPPORT_H
#define BFP_KERNEL_SUPPORT_H

#include <limits.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

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
 * guarantees |v| * 2^-k fits int32 for every segment and bias the kernel
 * can produce; -k <= 30 by the same proof). */
static inline int32_t bfpAlignPartial(int32_t v, int k) {
    if (k < 0) {
        return (int32_t)((uint32_t)v << (unsigned)(-k));
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

/*! Cursor for the integer-fold exponent pre-pass (Conv1d, AvgPool): the groups of `qC` that the
 * flat run [start, start + count) intersects, for runs visited in increasing `start` order (groups
 * are contiguous, so no division is needed). */
typedef struct bfpRowCursor {
    size_t group;
    size_t groupEnd; /* flat end (exclusive) of `group` */
} bfpRowCursor_t;

/*! Max unbiased exponent over the groups the run intersects. */
static inline int bfpRowMaxExponent(bfpRowCursor_t *c, const bfpQConfig_t *qC, int32_t expBias,
                                    size_t start, size_t count) {
    if (qC->groupSize == 0) {
        return (int)qC->exponents[0] - expBias;
    }
    while (start >= c->groupEnd) {
        c->group++;
        c->groupEnd += qC->groupSize;
    }
    size_t g = c->group;
    size_t gEnd = c->groupEnd;
    int e = (int)qC->exponents[g] - expBias;
    size_t stop = start + count;
    while (stop > gEnd) {
        g++;
        gEnd += qC->groupSize;
        int eg = (int)qC->exponents[g] - expBias;
        if (eg > e) {
            e = eg;
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
