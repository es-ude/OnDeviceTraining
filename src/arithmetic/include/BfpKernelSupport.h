#ifndef BFP_KERNEL_SUPPORT_H
#define BFP_KERNEL_SUPPORT_H

#include <limits.h>
#include <stdint.h>
#include <stdlib.h>

#include "Common.h"
#include "Quantization.h"

/* BFP epic PR2 (Task 3): shared support for the BFP GEMM kernels (Matmul.c
 * now, the conv kernels in later tasks). Header-only static inline -- no new
 * library. */

/*! Group id of a storage index; the {numGroups=1, groupSize=0} per-tensor
 * sentinel maps everything to group 0. */
static inline size_t bfpGroupOf(const bfpQConfig_t *qC, size_t storageIdx) {
    return qC->groupSize == 0 ? 0 : storageIdx / qC->groupSize;
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

/*! Max codes one int32 SUM partial may accumulate (PR3, biasGrad-family
 * reductions): a pure mantissa sum of g codes is bounded by g * 2^(m-1)
 * (codes in [-2^(m-1), 2^(m-1)-1]), so an int32 segment partial is sound for
 * g <= INT32_MAX >> (m-1). The PRODUCT helper's >> (ma+mb-2) bound does not
 * apply to single-operand sums. */
static inline size_t bfpSumSegmentLimit(uint8_t mantissaBits) {
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

#endif // BFP_KERNEL_SUPPORT_H
