#ifndef BFP_SOFTMAX_EXP_H
#define BFP_SOFTMAX_EXP_H
#include <stdint.h>

/* Integer i-exp core for the BFP softmax (I-BERT Algorithm 3, Kim et al.
 * 2021, on a FIXED dyadic work grid 2^-F) plus the 3-mode integer
 * shift-rounding helper the kernels share. Pipeline contract:
 * docs/conventions/arithmetic-bfp.md §5.9 (R-S2).
 *
 * Headroom proof (why F = 14 and why no int64 is needed anywhere):
 * q_p ∈ (−QLN2, 0] so |q_p + q_b| ≤ q_b = 22167; the I-POLY square is
 * ≤ 22167² = 491,375,889; adding QC = 257,578,233 gives qL ≤ 748,954,122
 * < 2^30 < 2^31. At F = 15, q_b ≈ 44334, square ≈ 1.97e9, plus the F=15
 * scale QC ≈ 5.15e8 overflows int32 → F = 14 is maximal. */
#define BFP_SOFTMAX_EXP_FRAC_BITS 14 /* F: work grid 2^-F            */
#define BFP_SOFTMAX_QLN2 11356       /* floor(ln2 * 2^14)            */
#define BFP_SOFTMAX_QB 22167         /* floor(1.353 * 2^14)          */
#define BFP_SOFTMAX_QC 257578233     /* floor(0.344 * 2^28 / 0.3585) */
#define BFP_SOFTMAX_A 0.3585f        /* S_out = A * 2^-28            */
#define BFP_SOFTMAX_ZMAX 31
#define BFP_SOFTMAX_QFLOOR (-(BFP_SOFTMAX_ZMAX + 1) * BFP_SOFTMAX_QLN2) /* -363392 */

typedef enum bfpShiftRounding {
    BFP_SHIFT_TRUNC = 0, /* arithmetic >>: floor (two's complement). I-BERT-faithful default. */
    BFP_SHIFT_HALF_AWAY, /* round-half-away-from-zero of v / 2^k                              */
    BFP_SHIFT_SR,        /* floor + Bernoulli(remainder / 2^k), one rngNextFloat() draw       */
} bfpShiftRounding_t;

/*! @brief Rounded arithmetic right shift v / 2^k in one of three modes.
 *
 * k == 0 returns v unchanged (no RNG draw). k >= 31 is clamped to 31
 * (documented; only reachable via extreme exponent spreads whose true
 * results round to 0 or ±1). SR draws exactly ONE rngNextFloat() per call
 * with k > 0, and a zero remainder never increments (exact shifts stay
 * exact under SR). HALF_AWAY requires |v| ≤ 2^30 (holds at every call
 * site) so the magnitude add cannot overflow. */
int32_t bfpShiftRightRounded(int32_t v, uint32_t k, bfpShiftRounding_t mode);

/*! @brief Integer exponential on the 2^-F work grid.
 *
 * Input qW ≤ 0 at scale 2^-F; output q_e ≥ 0 at scale S_out = A * 2^-28,
 * i.e. exp(qW * 2^-F) ≈ q_e * 0.3585 * 2^-28. qW ≤ BFP_SOFTMAX_QFLOOR is
 * an exact 0 (exp underflow; also keeps z ≤ ZMAX and skips the RNG in SR
 * mode). `mode` picks the rounding of the final >> z renormalization. */
int32_t bfpIExpQ(int32_t qW, bfpShiftRounding_t mode);

#endif // BFP_SOFTMAX_EXP_H
