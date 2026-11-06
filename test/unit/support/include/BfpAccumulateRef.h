#ifndef ODT_TEST_BFP_ACCUMULATE_REF_H
#define ODT_TEST_BFP_ACCUMULATE_REF_H

/*
 * Test-side exact reference of the integer BFP grad accumulate
 * (accumulateBfpRawIntoBfpRescale: BFP target += BFP_RAW increment), written
 * from its documented result (docs/conventions/arithmetic-bfp.md section 15),
 * not from the engine's quotient/remainder code. Per group:
 *   E_c = max(E_o + bitlen(A_o), E_r + bitlen(A_r)) - 30, capped at capE
 *   E_d = clamp(max(E_exact(sum), E_c), -bias, capE)     (all-zero sum: zero state)
 *   code = |sum| / 2^E_d rounded once: F = the top 32 bits of the remainder,
 *          SR lifts iff u < F, HALF_AWAY iff F >= 2^31; sign after; clamped
 *          to [-2^(m-1), 2^(m-1) - 1]; one draw per element in element order
 *          under SR on every path.
 * The exact sums live in 128-bit integers on the group's finer grid, so any
 * gap up to ~90 binades between the old and the raw exponent is exact (the
 * reference asserts that domain).
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "RNG.h"
#include "Rounding.h"
#include "unity.h"

typedef __int128 bfpAccRefI128_t;
typedef unsigned __int128 bfpAccRefU128_t;

static inline bfpAccRefU128_t bfpAccRefAbs(bfpAccRefI128_t v) {
    return v < 0 ? (bfpAccRefU128_t)0 - (bfpAccRefU128_t)v : (bfpAccRefU128_t)v;
}

static inline int bfpAccRefBitLen(bfpAccRefU128_t v) {
    int n = 0;
    while (v != 0) {
        n++;
        v >>= 1;
    }
    return n;
}

/* round(|v| / 2^d) by the documented magnitude law, sign after, clamped. */
static inline int32_t bfpAccRefRound(bfpAccRefI128_t v, int d, bool sr, uint32_t u, uint32_t posCap,
                                     uint32_t negCap) {
    const bool neg = v < 0;
    const bfpAccRefU128_t mag = bfpAccRefAbs(v);
    const uint32_t cap = neg ? negCap : posCap;
    bfpAccRefU128_t q;
    uint32_t f;
    if (d <= 0) {
        unsigned l = (unsigned)(-d);
        if (mag != 0 && (l >= 64 || bfpAccRefBitLen(mag) + (int)l > 100)) {
            return neg ? -(int32_t)negCap : (int32_t)posCap; /* saturates */
        }
        q = mag << l;
        f = 0;
    } else {
        unsigned k = (unsigned)d;
        q = k >= 128 ? 0 : mag >> k;
        bfpAccRefU128_t rem = k >= 128 ? mag : mag & (((bfpAccRefU128_t)1 << k) - 1u);
        if (k <= 32) {
            f = (uint32_t)(rem << (32u - k));
        } else if (k - 32u < 128u) {
            f = (uint32_t)(rem >> (k - 32u));
        } else {
            f = 0;
        }
    }
    bool lift = sr ? u < f : f >= 0x80000000u;
    bfpAccRefU128_t t = q + (bfpAccRefU128_t)lift;
    if (t > cap) {
        t = cap;
    }
    return neg ? (int32_t)(0u - (uint32_t)t) : (int32_t)t;
}

/* target (codes, stored exponents; updated in place) += raw (mant, unbiased
 * rawE per group), groups of groupSize elements (0 = per tensor). rs: the
 * stream for SR (NULL under HALF_AWAY). */
static inline void bfpAccRefAccumulate(int32_t *codes, uint8_t *stored, const int32_t *raw,
                                       const int16_t *rawE, size_t n, size_t groupSize, uint8_t m,
                                       uint8_t e, roundingMode_t rm, rngStream_t *rs) {
    const int bias = (1 << (e - 1)) - 1;
    int cap = (1 << e) - 1;
    if (cap > bias + 127) {
        cap = bias + 127;
    }
    const int capE = cap - bias;
    const uint32_t posCap = (1u << (m - 1)) - 1u;
    const uint32_t negCap = 1u << (m - 1);
    const size_t gsz = groupSize == 0 ? n : groupSize;
    for (size_t g = 0; gsz > 0 && g < n / gsz; g++) {
        const size_t start = g * gsz;
        const size_t end = start + gsz;
        const int eo = (int)stored[g] - bias;
        const int er = rawE[g];
        uint32_t ao = 0;
        uint32_t ar = 0;
        for (size_t i = start; i < end; i++) {
            uint32_t mo = codes[i] < 0 ? 0u - (uint32_t)codes[i] : (uint32_t)codes[i];
            uint32_t mr = raw[i] < 0 ? 0u - (uint32_t)raw[i] : (uint32_t)raw[i];
            ao = mo > ao ? mo : ao;
            ar = mr > ar ? mr : ar;
        }
        int ec = 0;
        if (ao != 0 || ar != 0) {
            int to = ao ? eo + bfpAccRefBitLen(ao) : INT32_MIN;
            int tr = ar ? er + bfpAccRefBitLen(ar) : INT32_MIN;
            ec = (to > tr ? to : tr) - 30;
            ec = ec > capE ? capE : ec;
        }
        const int emin = ao == 0 ? er : (ar == 0 ? eo : (eo < er ? eo : er));
        TEST_ASSERT_TRUE_MESSAGE((ao == 0 || eo - emin <= 90) && (ar == 0 || er - emin <= 90),
                                 "bfpAccRefAccumulate: exponent gap outside the reference");
        bfpAccRefI128_t vs[gsz];
        bfpAccRefU128_t vmax = 0;
        for (size_t i = start; i < end; i++) {
            bfpAccRefI128_t vo =
                ao == 0 ? 0 : (bfpAccRefI128_t)codes[i] * ((bfpAccRefI128_t)1 << (eo - emin));
            bfpAccRefI128_t vr =
                ar == 0 ? 0 : (bfpAccRefI128_t)raw[i] * ((bfpAccRefI128_t)1 << (er - emin));
            vs[i - start] = vo + vr;
            bfpAccRefU128_t a = bfpAccRefAbs(vo + vr);
            vmax = a > vmax ? a : vmax;
        }
        int ed = 0;
        if (vmax == 0) {
            stored[g] = (uint8_t)bias;
        } else {
            /* smallest E >= E_c with vmax <= qMax * 2^(E - emin) */
            int E = ec;
            for (;; E++) {
                int s = E - emin;
                bool ok;
                if (s >= 0) {
                    ok = s >= 100 || vmax <= ((bfpAccRefU128_t)posCap << s);
                } else {
                    ok = (vmax << (unsigned)(-s)) <= posCap;
                }
                if (ok || E >= capE) {
                    break;
                }
            }
            E = E < -bias ? -bias : E;
            E = E > capE ? capE : E;
            stored[g] = (uint8_t)(E + bias);
            ed = E;
        }
        for (size_t i = start; i < end; i++) {
            uint32_t u = rm == SR_HALF_AWAY ? rngNextU32(rs) : 0u;
            codes[i] = vmax == 0 ? 0
                                 : bfpAccRefRound(vs[i - start], ed - emin, rm == SR_HALF_AWAY, u,
                                                  posCap, negCap);
        }
    }
}

#endif // ODT_TEST_BFP_ACCUMULATE_REF_H
