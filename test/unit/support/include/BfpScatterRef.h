#ifndef ODT_TEST_BFP_SCATTER_REF_H
#define ODT_TEST_BFP_SCATTER_REF_H

/*
 * Test-side int64 references for the integer BFP Conv1d dx / weightGrad
 * (test-only). Written from the definition, not from the kernels: each
 * destination enumerates its own contributors (an output-centric walk, while
 * the kernels scatter), sums in int64 and reports whether the sum fits int32.
 *
 * Definition: every destination row group R (dx: one (batch, conv group);
 * weightGrad: one output channel) has E_top(R) >= every product exponent that
 * reaches R (the row maxima of the two operands, paired as the kernel's
 * contributors pair), E_acc = E_top + g with g the int32 headroom guard over
 * N products per destination. Each product enters as product * 2^-(E_acc -
 * E_product), rounded half away from zero. weightGrad's value is
 * float(sum) * 2^E_acc; dx is the BFP_RAW of the kernel (one exponent per raw
 * target group, bfpRefConv1dDxRaw).
 */

#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "unity.h"

typedef struct bfpRefOperand {
    const int32_t *codes;
    const uint8_t *exps; /* stored (biased) exponent per group */
    size_t groupSize;    /* 0 = per tensor */
    uint8_t m;
    uint8_t e;
} bfpRefOperand_t;

typedef struct bfpRefGeom {
    size_t batch, cin, length, cout, groups, kernelSize, stride, dilation, padLeft, outLength;
} bfpRefGeom_t;

typedef struct bfpRefStats {
    size_t rounded;    /* products whose alignment dropped a nonzero fraction */
    size_t leftShift;  /* products shifted left (exact) */
    size_t realigned;  /* dx raw: sums whose re-grid to the raw group's E_t dropped a fraction */
    bool fitsInt32;    /* every destination sum fits int32 */
    int64_t absMaxSum; /* largest |destination sum| */
} bfpRefStats_t;

static inline int bfpRefExp(const bfpRefOperand_t *o, size_t idx) {
    size_t g = o->groupSize == 0 ? 0 : idx / o->groupSize;
    return (int)o->exps[g] - ((1 << (o->e - 1)) - 1);
}

static inline int bfpRefMaxExp(const bfpRefOperand_t *o, size_t start, size_t count) {
    int best = INT_MIN;
    for (size_t i = start; i < start + count; i++) {
        int e = bfpRefExp(o, i);
        best = e > best ? e : best;
    }
    return best;
}

/* Smallest g in [-30, 31] with n 2^(ma+mb-2) 2^-g + n + 1 <= INT32_MAX
 * (doubles are exact here: every term is an integer below 2^53). */
static inline int bfpRefGuard(size_t n, unsigned ma, unsigned mb) {
    double total = ldexp((double)n, (int)(ma + mb - 2u));
    double room = (double)INT32_MAX - (double)n - 1.0;
    for (int g = -30; g <= 31; g++) {
        if (ldexp(total, -g) <= room) {
            return g;
        }
    }
    return 99;
}

static inline int64_t bfpRefAlign(int64_t p, int k, bfpRefStats_t *st) {
    if (k <= 0) {
        st->leftShift += k < 0;
        return p * ((int64_t)1 << -k);
    }
    int64_t mag = p < 0 ? -p : p;
    int64_t r = k >= 62 ? 0 : (mag + ((int64_t)1 << (k - 1))) >> k;
    if (k >= 62 ? mag != 0 : (r << k) != mag) {
        st->rounded++;
    }
    return p < 0 ? -r : r;
}

static inline float bfpRefFinish(int64_t sum, int eAcc, bfpRefStats_t *st) {
    if (sum > INT32_MAX || sum < INT32_MIN) {
        st->fitsInt32 = false;
    }
    int64_t mag = sum < 0 ? -sum : sum;
    st->absMaxSum = mag > st->absMaxSum ? mag : st->absMaxSum;
    return ldexpf((float)sum, eAcc);
}

/* Taps that can meet on one dx position: k, k' with (k - k') * dilation a
 * multiple of stride, counted per stride residue of k * dilation. */
static inline size_t bfpRefMaxTapOverlap(const bfpRefGeom_t *G) {
    size_t best = 0;
    for (size_t r = 0; r < G->stride; r++) {
        size_t n = 0;
        for (size_t k = 0; k < G->kernelSize; k++) {
            n += (k * G->dilation) % G->stride == r;
        }
        best = n > best ? n : best;
    }
    return best;
}

/* dx[b, ic, i] = sum over oc in group(ic), k with t = (i + padLeft - k d) / s
 * integral in [0, outLength) of gy[b, oc, t] * W[oc, icl, k]: the int64 sum
 * of each destination on its row group's E_acc (sum[], eAcc[] per element). */
static inline void bfpRefConv1dDxSums(const bfpRefGeom_t *G, const bfpRefOperand_t *gy,
                                      const bfpRefOperand_t *w, int64_t *sum, int *eAccOut,
                                      bfpRefStats_t *st) {
    size_t cinG = G->cin / G->groups;
    size_t coutG = G->cout / G->groups;
    int g = bfpRefGuard(coutG * bfpRefMaxTapOverlap(G), gy->m, w->m);
    for (size_t b = 0; b < G->batch; b++) {
        for (size_t ic = 0; ic < G->cin; ic++) {
            size_t grp = ic / cinG;
            size_t icl = ic - grp * cinG;
            int eTop = INT_MIN;
            for (size_t ocl = 0; ocl < coutG; ocl++) {
                size_t oc = grp * coutG + ocl;
                int e = bfpRefMaxExp(gy, (b * G->cout + oc) * G->outLength, G->outLength) +
                        bfpRefMaxExp(w, oc * cinG * G->kernelSize, cinG * G->kernelSize);
                eTop = e > eTop ? e : eTop;
            }
            int eAcc = eTop + g;
            for (size_t i = 0; i < G->length; i++) {
                int64_t acc = 0;
                for (size_t ocl = 0; ocl < coutG; ocl++) {
                    size_t oc = grp * coutG + ocl;
                    for (size_t k = 0; k < G->kernelSize; k++) {
                        long long num =
                            (long long)i + (long long)G->padLeft - (long long)(k * G->dilation);
                        if (num < 0 || num % (long long)G->stride != 0) {
                            continue;
                        }
                        size_t t = (size_t)(num / (long long)G->stride);
                        if (t >= G->outLength) {
                            continue;
                        }
                        size_t gi = (b * G->cout + oc) * G->outLength + t;
                        size_t wi = (oc * cinG + icl) * G->kernelSize + k;
                        int64_t p = (int64_t)gy->codes[gi] * (int64_t)w->codes[wi];
                        acc += bfpRefAlign(p, eAcc - (bfpRefExp(gy, gi) + bfpRefExp(w, wi)), st);
                    }
                }
                size_t idx = (b * G->cin + ic) * G->length + i;
                if (acc > INT32_MAX || acc < INT32_MIN) {
                    st->fitsInt32 = false;
                }
                int64_t mag = acc < 0 ? -acc : acc;
                st->absMaxSum = mag > st->absMaxSum ? mag : st->absMaxSum;
                sum[idx] = acc;
                eAccOut[idx] = eAcc;
            }
        }
    }
}

/* dx as the BFP_RAW the kernel emits: per raw target group t (rawGroupSize
 * elements, 0 = the whole output) E_t = the max E_acc over the group's
 * elements; each sum is re-gridded from its own E_acc to 2^E_t, rounding half
 * away from zero (a gap of 62+ binades leaves 0). `mant` holds the output's
 * element count, `rawE` its group count. */
static inline bfpRefStats_t bfpRefConv1dDxRaw(const bfpRefGeom_t *G, const bfpRefOperand_t *gy,
                                              const bfpRefOperand_t *w, size_t rawGroupSize,
                                              int32_t *mant, int16_t *rawE) {
    bfpRefStats_t st = {.fitsInt32 = true};
    size_t n = G->batch * G->cin * G->length;
    int64_t sum[n > 0 ? n : 1];
    int eAcc[n > 0 ? n : 1];
    bfpRefConv1dDxSums(G, gy, w, sum, eAcc, &st);
    size_t gsz = rawGroupSize == 0 ? n : rawGroupSize;
    for (size_t t = 0; gsz > 0 && t < n / gsz; t++) {
        int eT = INT_MIN;
        for (size_t i = t * gsz; i < (t + 1) * gsz; i++) {
            eT = eAcc[i] > eT ? eAcc[i] : eT;
        }
        rawE[t] = (int16_t)eT;
        for (size_t i = t * gsz; i < (t + 1) * gsz; i++) {
            int k = eT - eAcc[i];
            int64_t mag = sum[i] < 0 ? -sum[i] : sum[i];
            int64_t r = k >= 62 ? 0 : (k == 0 ? mag : (mag + ((int64_t)1 << (k - 1))) >> k);
            if (k > 0 && (k >= 62 ? mag != 0 : (r << k) != mag)) {
                st.realigned++;
            }
            mant[i] = (int32_t)(sum[i] < 0 ? -r : r);
        }
    }
    return st;
}

/* gw[oc, icl, k] = sum over b, t with pos = t s + k d - padLeft in [0,
 * length) of x[b, ic, pos] * gy[b, oc, t]: the int64 sum of each element on
 * its output channel's E_acc (sum[], eAcc[] per element; the float view
 * bfpRefConv1dWeightGrad below, the BFP_RAW view bfpRefRawFromRows). */
static inline void bfpRefConv1dWeightGradSums(const bfpRefGeom_t *G, const bfpRefOperand_t *x,
                                              const bfpRefOperand_t *gy, int64_t *sumOut,
                                              int *eAccOut, bfpRefStats_t *st) {
    size_t cinG = G->cin / G->groups;
    size_t coutG = G->cout / G->groups;
    size_t maxTaps = 0;
    for (size_t k = 0; k < G->kernelSize; k++) {
        size_t n = 0;
        for (size_t t = 0; t < G->outLength; t++) {
            long long pos = (long long)(t * G->stride + k * G->dilation) - (long long)G->padLeft;
            n += pos >= 0 && pos < (long long)G->length;
        }
        maxTaps = n > maxTaps ? n : maxTaps;
    }
    int g = bfpRefGuard(G->batch * maxTaps, x->m, gy->m);
    for (size_t oc = 0; oc < G->cout; oc++) {
        size_t grp = oc / coutG;
        int eTop = INT_MIN;
        for (size_t b = 0; b < G->batch; b++) {
            int e = bfpRefMaxExp(x, (b * G->cin + grp * cinG) * G->length, cinG * G->length) +
                    bfpRefMaxExp(gy, (b * G->cout + oc) * G->outLength, G->outLength);
            eTop = e > eTop ? e : eTop;
        }
        int eAcc = eTop + g;
        for (size_t icl = 0; icl < cinG; icl++) {
            size_t ic = grp * cinG + icl;
            for (size_t k = 0; k < G->kernelSize; k++) {
                int64_t sum = 0;
                for (size_t b = 0; b < G->batch; b++) {
                    for (size_t t = 0; t < G->outLength; t++) {
                        long long pos =
                            (long long)(t * G->stride + k * G->dilation) - (long long)G->padLeft;
                        if (pos < 0 || pos >= (long long)G->length) {
                            continue;
                        }
                        size_t xi = (b * G->cin + ic) * G->length + (size_t)pos;
                        size_t gi = (b * G->cout + oc) * G->outLength + t;
                        int64_t p = (int64_t)x->codes[xi] * (int64_t)gy->codes[gi];
                        sum += bfpRefAlign(p, eAcc - (bfpRefExp(x, xi) + bfpRefExp(gy, gi)), st);
                    }
                }
                size_t idx = (oc * cinG + icl) * G->kernelSize + k;
                sumOut[idx] = sum;
                eAccOut[idx] = eAcc;
            }
        }
    }
}

static inline bfpRefStats_t bfpRefConv1dWeightGrad(const bfpRefGeom_t *G, const bfpRefOperand_t *x,
                                                   const bfpRefOperand_t *gy, float *gw) {
    bfpRefStats_t st = {.fitsInt32 = true};
    size_t n = G->cout * (G->cin / G->groups) * G->kernelSize;
    int64_t sum[n > 0 ? n : 1];
    int eAcc[n > 0 ? n : 1];
    bfpRefConv1dWeightGradSums(G, x, gy, sum, eAcc, &st);
    for (size_t i = 0; i < n; i++) {
        gw[i] = bfpRefFinish(sum[i], eAcc[i], &st);
    }
    return st;
}

/* bias grad gb[oc] = sum over b (outer), t (inner) of gy[b, oc, t]. Walked in
 * that order, consecutive codes of one loss group form a segment whose code
 * sum is exact; each segment sum enters the destination as sum * 2^-(E_acc -
 * E_group), rounded half away from zero, with E_acc = max over b of the loss
 * row (b, oc)'s max exponent + g, g the guard of a pure SUM of batch *
 * outLength codes (the product guard with mb = 1). */
static inline void bfpRefConv1dBiasGradSums(const bfpRefGeom_t *G, const bfpRefOperand_t *gy,
                                            int64_t *sumOut, int *eAccOut, bfpRefStats_t *st) {
    int g = bfpRefGuard(G->batch * G->outLength, gy->m, 1u);
    for (size_t oc = 0; oc < G->cout; oc++) {
        int eTop = INT_MIN;
        for (size_t b = 0; b < G->batch; b++) {
            int e = bfpRefMaxExp(gy, (b * G->cout + oc) * G->outLength, G->outLength);
            eTop = e > eTop ? e : eTop;
        }
        int eAcc = G->batch > 0 && G->outLength > 0 ? eTop + g : 0;
        int64_t sum = 0;
        int64_t seg = 0;
        size_t segGroup = SIZE_MAX;
        size_t segIdx = 0;
        for (size_t b = 0; b < G->batch; b++) {
            for (size_t t = 0; t < G->outLength; t++) {
                size_t gi = (b * G->cout + oc) * G->outLength + t;
                size_t grp = gy->groupSize == 0 ? 0 : gi / gy->groupSize;
                if (grp != segGroup && segGroup != SIZE_MAX) {
                    sum += bfpRefAlign(seg, eAcc - bfpRefExp(gy, segIdx), st);
                    seg = 0;
                }
                if (grp != segGroup) {
                    segGroup = grp;
                    segIdx = gi;
                }
                seg += gy->codes[gi];
            }
        }
        if (segGroup != SIZE_MAX) {
            sum += bfpRefAlign(seg, eAcc - bfpRefExp(gy, segIdx), st);
        }
        if (sum > INT32_MAX || sum < INT32_MIN) {
            st->fitsInt32 = false;
        }
        sumOut[oc] = sum;
        eAccOut[oc] = eAcc;
    }
}

static inline unsigned bfpRefBitLength64(uint64_t v) {
    unsigned n = 0;
    while (v != 0) {
        n++;
        v >>= 1;
    }
    return n;
}

/* A grad kernel's BFP_RAW (one exponent per raw target group of rawGroupSize
 * elements, 0 = the whole output), from the DATA: E_t = max over the group's
 * nonzero sums of E_acc + bitlen(|sum|), minus 31 (every re-gridded mantissa
 * then fits int32: the top one is shifted left exactly, a lower one is below
 * 2^30 after a half-away right shift); an all-zero group has nothing to
 * place (its exponent is left 0). Each sum is re-gridded from its own E_acc
 * to 2^E_t: exactly to the left, half away from zero to the right (62+
 * binades leave 0). st->realigned counts the right shifts that dropped a
 * fraction, st->leftShift is incremented per left-shifted nonzero sum. */
static inline void bfpRefRawFromRows(const int64_t *sum, const int *eAcc, size_t n,
                                     size_t rawGroupSize, int32_t *mant, int16_t *rawE,
                                     bfpRefStats_t *st) {
    size_t gsz = rawGroupSize == 0 ? n : rawGroupSize;
    for (size_t t = 0; gsz > 0 && t < n / gsz; t++) {
        int top = INT_MIN;
        for (size_t i = t * gsz; i < (t + 1) * gsz; i++) {
            if (sum[i] != 0) {
                uint64_t mag = sum[i] < 0 ? 0u - (uint64_t)sum[i] : (uint64_t)sum[i];
                int e = eAcc[i] + (int)bfpRefBitLength64(mag);
                top = e > top ? e : top;
            }
        }
        int eT = top == INT_MIN ? 0 : top - 31;
        rawE[t] = (int16_t)eT;
        for (size_t i = t * gsz; i < (t + 1) * gsz; i++) {
            int64_t mag = sum[i] < 0 ? -sum[i] : sum[i];
            int k = eT - eAcc[i];
            int64_t r;
            if (mag == 0) {
                r = 0;
            } else if (k <= 0) {
                TEST_ASSERT_TRUE(-k <= 62);
                r = mag << -k;
                st->leftShift += k < 0;
            } else {
                r = k >= 62 ? 0 : (mag + ((int64_t)1 << (k - 1))) >> k;
                if (k >= 62 ? mag != 0 : (r << k) != mag) {
                    st->realigned++;
                }
            }
            TEST_ASSERT_TRUE(r <= INT32_MAX);
            mant[i] = (int32_t)(sum[i] < 0 ? -r : r);
        }
    }
}

#endif // ODT_TEST_BFP_SCATTER_REF_H
