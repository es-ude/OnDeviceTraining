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
 * E_product), rounded half away from zero; the destination's value is
 * float(sum) * 2^E_acc.
 */

#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

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
 * integral in [0, outLength) of gy[b, oc, t] * W[oc, icl, k]. */
static inline bfpRefStats_t bfpRefConv1dDx(const bfpRefGeom_t *G, const bfpRefOperand_t *gy,
                                           const bfpRefOperand_t *w, float *dx) {
    bfpRefStats_t st = {.fitsInt32 = true};
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
                int64_t sum = 0;
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
                        sum += bfpRefAlign(p, eAcc - (bfpRefExp(gy, gi) + bfpRefExp(w, wi)), &st);
                    }
                }
                dx[(b * G->cin + ic) * G->length + i] = bfpRefFinish(sum, eAcc, &st);
            }
        }
    }
    return st;
}

/* gw[oc, icl, k] = sum over b, t with pos = t s + k d - padLeft in [0,
 * length) of x[b, ic, pos] * gy[b, oc, t]. */
static inline bfpRefStats_t bfpRefConv1dWeightGrad(const bfpRefGeom_t *G, const bfpRefOperand_t *x,
                                                   const bfpRefOperand_t *gy, float *gw) {
    bfpRefStats_t st = {.fitsInt32 = true};
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
                        sum += bfpRefAlign(p, eAcc - (bfpRefExp(x, xi) + bfpRefExp(gy, gi)), &st);
                    }
                }
                gw[(oc * cinG + icl) * G->kernelSize + k] = bfpRefFinish(sum, eAcc, &st);
            }
        }
    }
    return st;
}

#endif // ODT_TEST_BFP_SCATTER_REF_H
