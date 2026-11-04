#define SOURCE_FILE "BFP_SOFTMAX_EXP"

#include "BfpSoftmaxExp.h"

#include <stdlib.h>

#include "Common.h"
#include "RNG.h"

int32_t bfpShiftRightRounded(int32_t v, uint32_t k, bfpShiftRounding_t mode, rngStream_t *rs) {
    if (k == 0) {
        return v;
    }
    if (k > 31) {
        k = 31;
    }
    switch (mode) {
    case BFP_SHIFT_TRUNC:
        return v >> k; /* arithmetic shift on negatives = floor */
    case BFP_SHIFT_HALF_AWAY: {
        /* |v| <= 2^30 at every call site, so the half-add fits int32 */
        const int32_t magnitude = (v < 0) ? -v : v;
        const int32_t rounded = (magnitude + (1 << (k - 1))) >> k;
        return (v < 0) ? -rounded : rounded;
    }
    case BFP_SHIFT_SR: {
        /* floor-domain, sign-agnostic, unbiased: floor + Bernoulli(rem/2^k).
         * The unsigned re-shift reproduces f * 2^k in two's complement
         * without the signed-left-shift UB; rem ∈ [0, 2^k). The draw's top k
         * bits are u on a k-bit grid (integer, no float on the forward
         * path; for k <= 24 the same decision the 24-bit float draw took),
         * and u >= 0, so rem == 0 never increments. */
        if (rs == NULL) {
            PRINT_ERROR("bfpShiftRightRounded: SR shift without a stream");
            exit(1);
        }
        const int32_t floorQ = v >> k;
        const uint32_t rem = (uint32_t)(v - (int32_t)((uint32_t)floorQ << k));
        const uint32_t u = rngNextU32(rs) >> (32u - k);
        return floorQ + ((u < rem) ? 1 : 0);
    }
    }
    return 0;
}

int32_t bfpIExpQ(int32_t qW, bfpShiftRounding_t mode, rngStream_t *rs) {
    if (qW <= BFP_SOFTMAX_QFLOOR) {
        return 0; /* exp underflow: exact 0, no RNG draw */
    }
    const int32_t z = (-qW) / BFP_SOFTMAX_QLN2;   /* non-negative operands -> floor */
    const int32_t qp = qW + z * BFP_SOFTMAX_QLN2; /* in (-QLN2, 0] */
    const int32_t qL =
        (qp + BFP_SOFTMAX_QB) * (qp + BFP_SOFTMAX_QB) + BFP_SOFTMAX_QC; /* <= 748,954,122 */
    return bfpShiftRightRounded(qL, (uint32_t)z, mode, rs); /* z <= ZMAX by the QFLOOR gate */
}
