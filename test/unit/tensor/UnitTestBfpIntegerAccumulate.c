#include "BfpKernelSupport.h"
#include "DTypes.h"
#include "DeathTest.h"
#include "Quantization.h"
#include "RNG.h"
#include "RngPinnedDraw.h"
#include "Tensor.h"
#include "TensorConversion.h"
#include "unity.h"

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

/* Integer training primitives (arithmetic-bfp.md §15): the configuration
 * scalar, the quotient/remainder add-and-round, and the integer
 * dynamic-rescale accumulate engine (BFP stored + BFP_RAW increment). Every
 * expected value is computed HERE, in int64 exact arithmetic, independent of
 * the quotient/remainder machinery under test. */

void setUp(void) {}
void tearDown(void) {}

/* ---- test-local exact helpers ---------------------------------------- */

static uint32_t testRngState = 0x2545F491u;

static uint32_t testRand(void) {
    uint32_t x = testRngState;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    testRngState = x;
    return x;
}

static int32_t testRandInt(int32_t lo, int32_t hi) { /* inclusive */
    return lo + (int32_t)(testRand() % (uint32_t)(hi - lo + 1));
}

/* A random signed value with magnitude < 2^bits (bits in [0, 31]); about one
 * in eight is zero, one in sixteen sits at the magnitude bound. */
static int32_t testRandSigned(unsigned bits) {
    uint32_t r = testRand();
    if (bits == 0 || (r & 7u) == 0) {
        return 0;
    }
    uint32_t mask = bits >= 32 ? UINT32_MAX : (1u << bits) - 1u;
    uint32_t mag = ((r & 15u) == 1) ? mask : (testRand() & mask);
    if (bits == 31 && (r & 63u) == 2) {
        return INT32_MIN; /* |INT32_MIN| = 2^31, the raw magnitude bound */
    }
    return (r & 16u) ? -(int32_t)mag : (int32_t)mag;
}

static uint64_t absI64(int64_t v) {
    return v < 0 ? 0u - (uint64_t)v : (uint64_t)v;
}

/* Exact rounding of |V| * 2^-D (V an exact int64, D any int) onto the grid,
 * the magnitude law the primitives document: F = top 32 bits of the
 * remainder, SR lifts iff u < F, HALF_AWAY iff F >= 2^31; sign after, clamp. */
static int32_t refRoundExact(int64_t v, int d, bool sr, uint32_t u, uint32_t posCap,
                             uint32_t negCap) {
    const bool neg = v < 0;
    const uint64_t mag = absI64(v);
    const uint32_t cap = neg ? negCap : posCap;
    uint64_t m;
    uint32_t f;
    if (d <= 0) {
        unsigned l = (unsigned)(-d);
        if (mag != 0 && (l >= 63 || mag > ((uint64_t)cap >> (l > 32 ? 32 : l)))) {
            return neg ? -(int32_t)negCap : (int32_t)posCap; /* saturates */
        }
        m = mag << l;
        f = 0;
    } else {
        unsigned k = (unsigned)d;
        m = k >= 64 ? 0 : mag >> k;
        uint64_t rem = k >= 64 ? mag : mag & ((UINT64_C(1) << k) - 1u);
        if (k <= 32) {
            f = (uint32_t)(rem << (32u - k));
        } else if (k - 32u < 64u) {
            f = (uint32_t)(rem >> (k - 32u));
        } else {
            f = 0;
        }
    }
    bool lift = sr ? u < f : f >= 0x80000000u;
    uint64_t t = m + (uint64_t)lift;
    if (t > cap) {
        t = cap;
    }
    return neg ? (int32_t)(0u - (uint32_t)t) : (int32_t)t;
}

/* ---- bfpScalar_t ----------------------------------------------------- */

static double scalarValue(bfpScalar_t s) {
    return ldexp((double)s.q, s.e);
}

static void assertNormalized(bfpScalar_t s) {
    int32_t a = s.q < 0 ? -s.q : s.q;
    TEST_ASSERT_TRUE_MESSAGE(a >= 16384 && a <= 32767, "significand not normalized to 15 bits");
}

void testScalarZeroOneAndPowersOfTwoAreExact(void) {
    bfpScalar_t z = bfpScalarFromFloat(0.0f, "t");
    TEST_ASSERT_EQUAL_INT32(0, z.q);
    TEST_ASSERT_EQUAL_INT16(0, z.e);
    bfpScalar_t nz = bfpScalarFromFloat(-0.0f, "t");
    TEST_ASSERT_EQUAL_INT32(0, nz.q);

    bfpScalar_t one = bfpScalarFromFloat(1.0f, "t");
    TEST_ASSERT_EQUAL_INT32(16384, one.q);
    TEST_ASSERT_EQUAL_INT16(-14, one.e);

    for (int p = -149; p <= 127; p++) {
        float x = ldexpf(1.0f, p);
        bfpScalar_t s = bfpScalarFromFloat(x, "t");
        TEST_ASSERT_EQUAL_INT32(16384, s.q);
        TEST_ASSERT_EQUAL_INT16(p - 14, s.e);
        bfpScalar_t sn = bfpScalarFromFloat(-x, "t");
        TEST_ASSERT_EQUAL_INT32(-16384, sn.q);
        TEST_ASSERT_EQUAL_INT16(p - 14, sn.e);
    }
}

/* Every finite float: normalized, and q is x * 2^-e rounded half away from
 * zero (x * 2^-e is exact in double). */
void testScalarRoundsSignificandHalfAwayTo15Bits(void) {
    const float cases[] = {0.1f,       0.9f,        0.99f,  1.0f / 3.0f, 1e-3f, 3.0e-5f, 0.999f,
                           123456.78f, -0.0123456f, 1e-40f, 3.4e38f,     7.0f,  -1.5f};
    for (size_t i = 0; i < sizeof cases / sizeof cases[0]; i++) {
        bfpScalar_t s = bfpScalarFromFloat(cases[i], "t");
        assertNormalized(s);
        double exactQ = ldexp((double)cases[i], -s.e);
        double expectQ = exactQ < 0 ? -floor(-exactQ + 0.5) : floor(exactQ + 0.5);
        TEST_ASSERT_EQUAL_INT32((int32_t)expectQ, s.q);
        double rel = fabs(scalarValue(s) - (double)cases[i]) / fabs((double)cases[i]);
        TEST_ASSERT_TRUE(rel <= ldexp(1.0, -15));
    }
    for (int i = 0; i < 20000; i++) {
        uint32_t bits = testRand() & 0x7F7FFFFFu; /* finite, any sign bit cleared */
        float x;
        memcpy(&x, &bits, sizeof x);
        if (x == 0.0f) {
            continue;
        }
        bfpScalar_t s = bfpScalarFromFloat(x, "t");
        assertNormalized(s);
        double exactQ = ldexp((double)x, -s.e);
        TEST_ASSERT_EQUAL_INT32((int32_t)floor(exactQ + 0.5), s.q);
    }
}

/* The carry case: a significand that rounds up to 2^15 renormalizes. */
void testScalarRoundUpToNextBinadeRenormalizes(void) {
    float x = 1.0f - ldexpf(1.0f, -17); /* f * 2^15 = 32767.75 -> 32768 */
    bfpScalar_t s = bfpScalarFromFloat(x, "t");
    TEST_ASSERT_EQUAL_INT32(16384, s.q);
    TEST_ASSERT_EQUAL_INT16(-14, s.e);
    bfpScalar_t sn = bfpScalarFromFloat(-x, "t");
    TEST_ASSERT_EQUAL_INT32(-16384, sn.q);
    TEST_ASSERT_EQUAL_INT16(-14, sn.e);
}

static void scalarInfBody(void) {
    (void)bfpScalarFromFloat(INFINITY, "t");
}
static void scalarNanBody(void) {
    (void)bfpScalarFromFloat(NAN, "t");
}

void testScalarRejectsNonFinite(void) {
    ASSERT_EXITS_WITH_FAILURE(scalarInfBody());
    ASSERT_EXITS_WITH_FAILURE(scalarNanBody());
}

/* 16-bit code x normalized 15-bit significand: exact in int32 at the
 * extremes (checked against int64). */
void testScalarMulMantIsExactAtExtremes(void) {
    const int32_t mants[] = {-32768, 32767, -1, 1, 0, 12345, -20000};
    const int32_t qs[] = {32767, -32767, 16384, -16384, 0, 23170};
    for (size_t i = 0; i < sizeof mants / sizeof mants[0]; i++) {
        for (size_t j = 0; j < sizeof qs / sizeof qs[0]; j++) {
            bfpScalar_t s = {.q = qs[j], .e = 3};
            TEST_ASSERT_EQUAL_INT64((int64_t)mants[i] * qs[j],
                                    (int64_t)bfpScalarMulMant(mants[i], s));
        }
    }
}

/* ---- quotient/remainder primitives ------------------------------------ */

/* bfpQrOfTerm keeps every dropped bit: q * 2^32 + frac is the exact
 * sign-extended value at the grid for k <= 32 (int64 check), and for k > 32
 * frac/sticky are the top 32 / remaining bits of the exact remainder. */
void testQrOfTermIsExactAtEveryShift(void) {
    const int32_t ms[] = {0, 1, -1, 5, -5, INT32_MAX, INT32_MIN, 0x12345678, -0x12345678, 3};
    const int ks[] = {0, 1, 2, 17, 31, 32, 33, 40, 63, 64, 65, 100, 32739};
    for (size_t i = 0; i < sizeof ms / sizeof ms[0]; i++) {
        for (size_t j = 0; j < sizeof ks / sizeof ks[0]; j++) {
            int32_t m = ms[i];
            int k = ks[j];
            bfpQr_t r = bfpQrOfTerm(m, k);
            /* reference: floor(m / 2^k) and the remainder, from int64 */
            int64_t fl;
            uint64_t rem; /* m - fl * 2^k, in [0, 2^k), only for k <= 62 */
            if (k <= 62) {
                int64_t p = INT64_C(1) << k;
                fl = (int64_t)m >= 0 ? (int64_t)m / p : -((-(int64_t)m + p - 1) / p);
                rem = (uint64_t)((int64_t)m - fl * p);
                uint32_t f = k <= 32 ? (uint32_t)(rem << (32 - k)) : (uint32_t)(rem >> (k - 32));
                bool st = k <= 32 ? false : (rem & ((UINT64_C(1) << (k - 32)) - 1u)) != 0;
                TEST_ASSERT_EQUAL_INT32((int32_t)fl, r.q);
                TEST_ASSERT_EQUAL_HEX32(f, r.frac);
                TEST_ASSERT_EQUAL(st, r.sticky);
            } else {
                /* |m| <= 2^31 <= 2^(k-32): frac is pure sign extension (bits
                 * [k-32, k-1] lie at or above bit 31); sticky is bits [0,
                 * k-33] of the sign-extended m -- bits [0, 30] at k = 63 */
                TEST_ASSERT_EQUAL_INT32(m < 0 ? -1 : 0, r.q);
                TEST_ASSERT_EQUAL_HEX32(m < 0 ? UINT32_MAX : 0u, r.frac);
                bool st = k == 63 ? ((uint32_t)m & 0x7FFFFFFFu) != 0 : m != 0;
                TEST_ASSERT_EQUAL(st, r.sticky);
            }
        }
    }
    bfpQr_t l = bfpQrOfTerm(-3, -5); /* exact left shift */
    TEST_ASSERT_EQUAL_INT32(-96, l.q);
    TEST_ASSERT_EQUAL_HEX32(0, l.frac);
    TEST_ASSERT_FALSE(l.sticky);
    bfpQr_t z = bfpQrOfTerm(0, -40); /* a zero term at any left shift */
    TEST_ASSERT_EQUAL_INT32(0, z.q);
}

/* bfpQrShr is exact at every re-grid distance: composing two shifts equals
 * one, and against int64 for the reachable range. */
void testQrShrIsExactAtEveryDistance(void) {
    for (int it = 0; it < 4000; it++) {
        int32_t m = testRandSigned(31);
        int k1 = testRandInt(0, 40);
        int k2 = testRandInt(0, 80);
        bfpQr_t once = bfpQrOfTerm(m, k1 + k2);
        bfpQr_t twice = bfpQrShr(bfpQrOfTerm(m, k1), (unsigned)k2);
        TEST_ASSERT_EQUAL_INT32(once.q, twice.q);
        TEST_ASSERT_EQUAL_HEX32(once.frac, twice.frac);
        TEST_ASSERT_EQUAL(once.sticky, twice.sticky);
    }
}

/* bfpAddAndRound against the exact int64 reference over random operands:
 * terms below 2^30 on the working grid (ka/kb <= 0 left-shifted, > 0 with a
 * remainder of up to 45 bits), at most one term past 32 dropped bits (the
 * documented exactness condition), destination 0..60 steps coarser, HALF_AWAY
 * and SR at the decision boundary and random draws. */
void testAddAndRoundMatchesExactReference(void) {
    for (int it = 0; it < 60000; it++) {
        int ka = testRandInt(-3, 45);
        int kb = testRandInt(-3, 45);
        if (ka > 32 && kb > 32) {
            kb = testRandInt(-3, 2);
        }
        int32_t a = testRandSigned(ka >= 1 ? 31 : (unsigned)(29 + ka));
        int32_t b = testRandSigned(kb >= 1 ? 31 : (unsigned)(29 + kb));
        unsigned kd = (unsigned)testRandInt(0, 60);
        int kmax = ka > kb ? ka : kb;
        if (kmax < 0) {
            kmax = 0;
        }
        if (kmax - ka > 31 || kmax - kb > 31) {
            continue; /* outside the int64 reference's reach */
        }
        int64_t v =
            (int64_t)a * (INT64_C(1) << (kmax - ka)) + (int64_t)b * (INT64_C(1) << (kmax - kb));
        const uint32_t caps[2][2] = {{0x7FFFFFFFu, 0x80000000u}, {127u, 128u}};
        const uint32_t *cap = caps[it & 1];
        int d = kmax + (int)kd;
        /* the boundary draws of the exact F */
        uint32_t fExact = 0;
        {
            uint64_t mag = absI64(v);
            if (d > 0 && d <= 32) {
                fExact = (uint32_t)((mag & ((UINT64_C(1) << d) - 1u)) << (32 - d));
            } else if (d > 32 && d < 96) {
                uint64_t rem = d >= 64 ? mag : mag & ((UINT64_C(1) << d) - 1u);
                fExact = d - 32 < 64 ? (uint32_t)(rem >> (d - 32)) : 0u;
            }
        }
        const uint32_t us[] = {0u,          1u,          fExact - 1u, fExact,
                               fExact + 1u, 0x80000000u, UINT32_MAX,  testRand()};
        TEST_ASSERT_EQUAL_INT32(refRoundExact(v, d, false, 0, cap[0], cap[1]),
                                bfpAddAndRound(a, ka, b, kb, kd, HALF_AWAY, 0, cap[0], cap[1]));
        for (size_t j = 0; j < sizeof us / sizeof us[0]; j++) {
            TEST_ASSERT_EQUAL_INT32(
                refRoundExact(v, d, true, us[j], cap[0], cap[1]),
                bfpAddAndRound(a, ka, b, kb, kd, SR_HALF_AWAY, us[j], cap[0], cap[1]));
        }
    }
}

/* SR is odd per draw: SR(-x) = -SR(x) for the same u (Rounding.h law), also
 * through the negative-to-magnitude conversion with a sticky tail. */
void testAddAndRoundSrIsOddPerDraw(void) {
    for (int it = 0; it < 20000; it++) {
        int ka = testRandInt(-3, 40);
        int kb = testRandInt(-3, 2);
        int32_t a = testRandSigned(ka >= 1 ? 30 : (unsigned)(28 + ka));
        int32_t b = testRandSigned(kb >= 1 ? 30 : (unsigned)(28 + kb));
        unsigned kd = (unsigned)testRandInt(0, 50);
        uint32_t u = testRand();
        int32_t pos = bfpAddAndRound(a, ka, b, kb, kd, SR_HALF_AWAY, u, 0x7FFFFFFFu, 0x7FFFFFFFu);
        int32_t neg = bfpAddAndRound(-a, ka, -b, kb, kd, SR_HALF_AWAY, u, 0x7FFFFFFFu, 0x7FFFFFFFu);
        TEST_ASSERT_EQUAL_INT32(pos, -neg);
    }
}

/* The documented cutoff of the one-draw rule: a magnitude remainder of
 * exactly 2^-32 of a grid step lifts iff u == 0; below that it never lifts;
 * the toward-zero twin moves for every u but UINT32_MAX. */
void testAddAndRoundTailCutoffIsTwoToMinus32(void) {
    /* 64 + 2^-32 (grid: 64 at k = 0, 1 at k = 32) */
    TEST_ASSERT_EQUAL_INT32(65, bfpAddAndRound(64, 0, 1, 32, 0, SR_HALF_AWAY, 0u, 127, 128));
    TEST_ASSERT_EQUAL_INT32(64, bfpAddAndRound(64, 0, 1, 32, 0, SR_HALF_AWAY, 1u, 127, 128));
    /* 64 + 1.5 * 2^-32: F = 1, the half below is truncated */
    TEST_ASSERT_EQUAL_INT32(65, bfpAddAndRound(64, 0, 3, 33, 0, SR_HALF_AWAY, 0u, 127, 128));
    TEST_ASSERT_EQUAL_INT32(64, bfpAddAndRound(64, 0, 3, 33, 0, SR_HALF_AWAY, 1u, 127, 128));
    /* 64 + 2^-33 and 64 + 2^-100: below the cutoff */
    TEST_ASSERT_EQUAL_INT32(64, bfpAddAndRound(64, 0, 1, 33, 0, SR_HALF_AWAY, 0u, 127, 128));
    TEST_ASSERT_EQUAL_INT32(64, bfpAddAndRound(64, 0, 1, 100, 0, SR_HALF_AWAY, 0u, 127, 128));
    /* 64 - 2^-40 = 63 + (1 - 2^-40): F = 2^32 - 1 */
    TEST_ASSERT_EQUAL_INT32(64,
                            bfpAddAndRound(64, 0, -1, 40, 0, SR_HALF_AWAY, 0xFFFFFFFEu, 127, 128));
    TEST_ASSERT_EQUAL_INT32(63,
                            bfpAddAndRound(64, 0, -1, 40, 0, SR_HALF_AWAY, 0xFFFFFFFFu, 127, 128));
    /* HALF_AWAY: exact half lifts, a hair below does not */
    TEST_ASSERT_EQUAL_INT32(65, bfpAddAndRound(64, 0, 1, 1, 0, HALF_AWAY, 0u, 127, 128));
    TEST_ASSERT_EQUAL_INT32(-65, bfpAddAndRound(-64, 0, -1, 1, 0, HALF_AWAY, 0u, 127, 128));
    TEST_ASSERT_EQUAL_INT32(64, bfpAddAndRound(129, 1, -1, 40, 0, HALF_AWAY, 0u, 127, 128));
}

/* ---- the integer rescale accumulate engine ---------------------------- */

#define ACC_MAX_N 4096
#define ACC_MAX_GROUPS 1024

typedef struct accFixture {
    size_t n;
    size_t dims[1];
    size_t order[1];
    shape_t shape;
    uint8_t exps[ACC_MAX_GROUPS];
    bfpQConfig_t qc;
    quantization_t q;
    uint8_t data[ACC_MAX_N * 2 + 8];
    tensor_t target;
    int32_t raw[ACC_MAX_N];
    int16_t rawE[ACC_MAX_GROUPS];
    bfpRawQConfig_t rawQC;
    quantization_t rawQ;
    tensor_t inc;
} accFixture_t;

static accFixture_t fixA;

static void accInit(accFixture_t *f, size_t n, size_t numGroups, size_t groupSize, uint8_t m,
                    uint8_t e, roundingMode_t rm, const int32_t *codes, const uint8_t *stored,
                    const int32_t *raw, const int16_t *rawE) {
    TEST_ASSERT_TRUE(n <= ACC_MAX_N && numGroups <= ACC_MAX_GROUPS);
    f->n = n;
    f->dims[0] = n;
    f->order[0] = 0;
    setShape(&f->shape, f->dims, 1, f->order);
    memcpy(f->exps, stored, numGroups);
    f->qc = (bfpQConfig_t){.exponents = f->exps,
                           .numGroups = numGroups,
                           .groupSize = groupSize,
                           .roundingMode = rm,
                           .mantissaBits = m,
                           .exponentBits = e};
    initBfpQuantization(&f->qc, &f->q);
    memset(f->data, 0xA5, sizeof f->data);
    if (n > 0) {
        int32_t tmp[ACC_MAX_N];
        memcpy(tmp, codes, n * sizeof(int32_t));
        byteConversion((uint8_t *)tmp, 32, f->data, m, n);
    }
    setTensorValues(&f->target, f->data, &f->shape, &f->q, NULL);
    if (n > 0) {
        memcpy(f->raw, raw, n * sizeof(int32_t));
    }
    memcpy(f->rawE, rawE, numGroups * sizeof(int16_t));
    f->rawQC =
        (bfpRawQConfig_t){.exponents = f->rawE, .numGroups = numGroups, .groupSize = groupSize};
    initBfpRawQuantization(&f->rawQC, &f->rawQ);
    setTensorValues(&f->inc, (uint8_t *)f->raw, &f->shape, &f->rawQ, NULL);
}

static void accCodes(const accFixture_t *f, int32_t *out) {
    unpackSignExtend(f->data, f->qc.mantissaBits, 0, out, f->n);
}

/* The engine's documented result, computed exactly (int64) per group:
 *   E_c = max(E_o + bitlen(A_o), E_r + bitlen(A_r)) - 30, capped at capE
 *   E_d = clamp(max(E_exact(sum), E_c), -bias, capE)   (all-zero sum: zero state)
 *   code = round(|sum| / 2^E_d) by refRoundExact, one draw per element.
 * Requires |E_o - E_r| <= 30 inside every group with both sides nonzero. */
static void refAccumulate(const int32_t *codes, const uint8_t *stored, const int32_t *raw,
                          const int16_t *rawE, size_t n, size_t numGroups, size_t groupSize,
                          uint8_t m, uint8_t e, roundingMode_t rm, rngStream_t *rs,
                          int32_t *outCodes, uint8_t *outStored) {
    const int bias = (1 << (e - 1)) - 1;
    int cap = (1 << e) - 1;
    if (cap > bias + 127) {
        cap = bias + 127;
    }
    const int capE = cap - bias;
    const uint32_t posCap = (1u << (m - 1)) - 1u;
    const uint32_t negCap = 1u << (m - 1);
    const size_t gsz = groupSize == 0 ? n : groupSize;
    for (size_t g = 0; g < numGroups; g++) {
        const size_t start = g * gsz;
        const size_t end = start + gsz;
        const int eo = (int)stored[g] - bias;
        const int er = rawE[g];
        uint32_t ao = 0;
        uint32_t ar = 0;
        for (size_t i = start; i < end; i++) {
            uint32_t mo = (uint32_t)absI64(codes[i]);
            uint32_t mr = (uint32_t)absI64(raw[i]);
            ao = mo > ao ? mo : ao;
            ar = mr > ar ? mr : ar;
        }
        int ec = 0;
        bool any = ao != 0 || ar != 0;
        if (any) {
            int to = INT32_MIN;
            int tr = INT32_MIN;
            if (ao) {
                to = eo + (int)(32 - __builtin_clz(ao));
            }
            if (ar) {
                tr = er + (int)(32 - __builtin_clz(ar));
            }
            ec = (to > tr ? to : tr) - 30;
            if (ec > capE) {
                ec = capE;
            }
        }
        /* exact sums on the grid 2^emin */
        const int emin = ao == 0 ? er : (ar == 0 ? eo : (eo < er ? eo : er));
        int64_t vs[ACC_MAX_N];
        uint64_t vmax = 0;
        bool huge = false;
        for (size_t i = start; i < end; i++) {
            int64_t vo = codes[i] == 0 ? 0 : (int64_t)codes[i] * ((int64_t)1 << (eo - emin));
            int64_t vr = raw[i] == 0 ? 0 : (int64_t)raw[i] * ((int64_t)1 << (er - emin));
            vs[i - start] = vo + vr;
            uint64_t a = absI64(vo + vr);
            vmax = a > vmax ? a : vmax;
        }
        (void)huge;
        int ed;
        if (vmax == 0) {
            outStored[g] = (uint8_t)bias;
            ed = 0;
        } else {
            /* smallest E >= ec with vmax <= qMax * 2^(E - emin) */
            int E = ec;
            for (;; E++) {
                int s = E - emin;
                bool ok;
                if (s >= 0) {
                    ok = s >= 48 || vmax <= ((uint64_t)posCap << s);
                } else {
                    ok = vmax <= posCap && (vmax << (unsigned)(-s)) <= posCap;
                }
                if (ok || E >= capE) {
                    break;
                }
            }
            if (E < -bias) {
                E = -bias;
            }
            if (E > capE) {
                E = capE;
            }
            outStored[g] = (uint8_t)(E + bias);
            ed = E;
        }
        for (size_t i = start; i < end; i++) {
            uint32_t u = rm == SR_HALF_AWAY ? rngNextU32(rs) : 0u;
            if (vmax == 0) {
                outCodes[i] = 0;
                continue;
            }
            outCodes[i] =
                refRoundExact(vs[i - start], ed - emin, rm == SR_HALF_AWAY, u, posCap, negCap);
        }
    }
}

static void assertEngineMatchesReference(size_t n, size_t numGroups, size_t groupSize, uint8_t m,
                                         uint8_t e, roundingMode_t rm, const int32_t *codes,
                                         const uint8_t *stored, const int32_t *raw,
                                         const int16_t *rawE, uint32_t seed) {
    static int32_t expCodes[ACC_MAX_N];
    static int32_t gotCodes[ACC_MAX_N];
    uint8_t expStored[ACC_MAX_GROUPS];
    rngStream_t refStream = rngOpen((rngKey_t){{seed, 7u}});
    rngStream_t engStream = rngOpen((rngKey_t){{seed, 7u}});
    refAccumulate(codes, stored, raw, rawE, n, numGroups, groupSize, m, e, rm, &refStream, expCodes,
                  expStored);
    accInit(&fixA, n, numGroups, groupSize, m, e, rm, codes, stored, raw, rawE);
    accumulateBfpRawIntoBfpRescale(&fixA.target, &fixA.inc, rm == SR_HALF_AWAY ? &engStream : NULL);
    accCodes(&fixA, gotCodes);
    TEST_ASSERT_EQUAL_UINT8_ARRAY(expStored, fixA.exps, numGroups);
    TEST_ASSERT_EQUAL_INT32_ARRAY(expCodes, gotCodes, n);
    if (rm == SR_HALF_AWAY) {
        /* one draw per element, in element order: both streams in step */
        TEST_ASSERT_EQUAL_UINT32(rngNextU32(&refStream), rngNextU32(&engStream));
    }
    /* the increment is read-only */
    TEST_ASSERT_EQUAL_INT32_ARRAY(raw, fixA.raw, n);
    TEST_ASSERT_EQUAL_INT16_ARRAY(rawE, fixA.rawE, numGroups);
}

/* Random grouped targets against the exact reference: per-tensor and grouped
 * geometries (group size 1 needs more fresh exponents than the window holds,
 * so the ring wraps), byte and sub-byte widths, multi-chunk n (the in-place
 * read-modify-write and the exponent publication after the last read), zero
 * and zero-old groups, cancellation, HALF_AWAY and SR. */
void testEngineMatchesExactReferenceOnRandomTargets(void) {
    static int32_t codes[ACC_MAX_N];
    static int32_t raw[ACC_MAX_N];
    uint8_t stored[ACC_MAX_GROUPS];
    int16_t rawE[ACC_MAX_GROUPS];
    const struct {
        size_t n, numGroups, groupSize;
        uint8_t m;
    } geo[] = {{1, 1, 0, 8},     {7, 1, 0, 4},      {600, 1, 0, 8},   {700, 700, 1, 5},
               {699, 233, 3, 8}, {640, 10, 64, 16}, {513, 1, 0, 2},   {770, 110, 7, 12},
               {300, 3, 100, 6}, {32, 2, 16, 8},    {1000, 8, 125, 3}};
    for (int round = 0; round < 6; round++) {
        for (size_t c = 0; c < sizeof geo / sizeof geo[0]; c++) {
            const size_t n = geo[c].n;
            const size_t ng = geo[c].numGroups;
            const size_t gsz = geo[c].groupSize == 0 ? n : geo[c].groupSize;
            const uint8_t m = geo[c].m;
            const uint8_t e = 8;
            const int bias = 127;
            for (size_t g = 0; g < ng; g++) {
                stored[g] = (uint8_t)(bias + testRandInt(-20, 20));
                unsigned rawBits = (unsigned)testRandInt(0, 31);
                rawE[g] = (int16_t)((int)stored[g] - bias + testRandInt(-30, 30) -
                                    (rawBits > 8 ? (int)rawBits - 8 : 0) / 2);
                int flavour = testRandInt(0, 9);
                for (size_t i = g * gsz; i < (g + 1) * gsz; i++) {
                    codes[i] = flavour == 0 ? 0 : testRandSigned(m - 1u);
                    if (codes[i] < -(1 << (m - 1))) {
                        codes[i] = -(1 << (m - 1));
                    }
                    raw[i] = flavour == 1 ? 0 : testRandSigned(rawBits);
                }
                if (flavour == 2) { /* exact cancellation of the first element */
                    int d = ((int)stored[g] - bias) - rawE[g];
                    if (d >= 0 && d <= 15) {
                        raw[g * gsz] = -codes[g * gsz] * (1 << d);
                    }
                }
                /* keep |E_o - E_r| <= 30 for the int64 reference */
                int gap = ((int)stored[g] - bias) - rawE[g];
                if (gap > 30 || gap < -30) {
                    rawE[g] = (int16_t)((int)stored[g] - bias);
                }
            }
            uint32_t seed = 0x1000u + (uint32_t)(round * 31 + (int)c);
            assertEngineMatchesReference(n, ng, geo[c].groupSize, m, e, HALF_AWAY, codes, stored,
                                         raw, rawE, seed);
            assertEngineMatchesReference(n, ng, geo[c].groupSize, m, e, SR_HALF_AWAY, codes, stored,
                                         raw, rawE, seed);
        }
    }
}

/* Dead zone (#279), pinned: w = 64 * Delta, increment -Delta/256 (a raw of
 * 2^20 codes at 2^-28). The sum 63.996 Delta keeps the exponent, and SR
 * must move it to 63 with probability exactly 1/256: lift iff
 * u < 255 * 2^24. The +Delta/256 twin moves to 65 iff u < 2^24. */
void testEngineDeadZoneUpdateMovesAtTheExactPinnedThreshold(void) {
    const int32_t codes[1] = {64};
    const uint8_t stored[1] = {127};
    const int16_t rawE[1] = {-28};
    const int32_t rawNeg[1] = {-(1 << 20)};
    const int32_t rawPos[1] = {1 << 20};
    const struct {
        const int32_t *raw;
        uint32_t u;
        int32_t expect;
    } cases[] = {{rawNeg, 255u << 24, 63}, {rawNeg, (255u << 24) - 1u, 64}, {rawNeg, 1u, 64},
                 {rawNeg, UINT32_MAX, 63}, {rawPos, (1u << 24) - 1u, 65},   {rawPos, 1u << 24, 64},
                 {rawPos, 1u, 65},         {rawPos, UINT32_MAX, 64}};
    for (size_t c = 0; c < sizeof cases / sizeof cases[0]; c++) {
        rngStream_t s = rngStreamWithNextDraw(cases[c].u);
        accInit(&fixA, 1, 1, 0, 8, 8, SR_HALF_AWAY, codes, stored, cases[c].raw, rawE);
        accumulateBfpRawIntoBfpRescale(&fixA.target, &fixA.inc, &s);
        int32_t got[1];
        accCodes(&fixA, got);
        TEST_ASSERT_EQUAL_UINT8(127, fixA.exps[0]);
        TEST_ASSERT_EQUAL_INT32(cases[c].expect, got[0]);
    }
}

/* Dead zone, statistically: 25 calls x 4096 elements, every element 64 Delta
 * - Delta/256, one seeded stream. The count of 63s must sit within 5 sigma
 * of 102400 / 256 = 400 (sigma ~ 20). A deterministic alignment of the
 * increment (round it to the destination grid before the SR) gives 0. */
void testEngineDeadZoneUpdateMovesWithProbabilityOneIn256(void) {
    static int32_t codes[4096];
    static int32_t raw[4096];
    static int32_t got[4096];
    const uint8_t stored[1] = {127};
    const int16_t rawE[1] = {-28};
    for (size_t i = 0; i < 4096; i++) {
        codes[i] = 64;
        raw[i] = -(1 << 20);
    }
    rngStream_t s = rngOpen((rngKey_t){{0xC0FFEEu, 3u}});
    long moved = 0;
    long other = 0;
    for (int call = 0; call < 25; call++) {
        accInit(&fixA, 4096, 1, 0, 8, 8, SR_HALF_AWAY, codes, stored, raw, rawE);
        accumulateBfpRawIntoBfpRescale(&fixA.target, &fixA.inc, &s);
        TEST_ASSERT_EQUAL_UINT8(127, fixA.exps[0]);
        accCodes(&fixA, got);
        for (size_t i = 0; i < 4096; i++) {
            moved += got[i] == 63;
            other += got[i] != 63 && got[i] != 64;
        }
    }
    TEST_ASSERT_EQUAL(0, other);
    TEST_ASSERT_TRUE_MESSAGE(moved >= 300 && moved <= 500, "dead-zone move rate is not 1/256");
}

/* SR is unbiased: the mean code over many seeded draws of 10 + 77/256 sits
 * within 5 sigma of the exact value (fraction carried by a raw at 2^-8 of
 * the code step), for both signs. */
void testEngineSrIsUnbiased(void) {
    static int32_t codes[4096];
    static int32_t raw[4096];
    static int32_t got[4096];
    const uint8_t stored[1] = {127};
    const int16_t rawE[1] = {-8};
    for (int sign = -1; sign <= 1; sign += 2) {
        for (size_t i = 0; i < 4096; i++) {
            codes[i] = 10 * sign;
            raw[i] = 77 * sign;
        }
        rngStream_t s = rngOpen((rngKey_t){{0xBEEFu + (uint32_t)(sign + 1), 1u}});
        double sum = 0;
        const int calls = 10;
        for (int call = 0; call < calls; call++) {
            accInit(&fixA, 4096, 1, 0, 8, 8, SR_HALF_AWAY, codes, stored, raw, rawE);
            accumulateBfpRawIntoBfpRescale(&fixA.target, &fixA.inc, &s);
            accCodes(&fixA, got);
            for (size_t i = 0; i < 4096; i++) {
                sum += got[i];
            }
        }
        /* 10 + 77/256 lands on E = -3: code 82 + 0.40625 (code steps) */
        TEST_ASSERT_EQUAL_UINT8(124, fixA.exps[0]);
        double nTot = 4096.0 * calls;
        double p = 0.40625;
        double mean = sum / nTot;
        double sigma = sqrt(p * (1 - p) / nTot);
        TEST_ASSERT_TRUE(fabs(mean - sign * (82.0 + p)) <= 5 * sigma);
    }
}

/* Large exponent gaps keep their tail (no k >= 32 flush): 64 + 2^-31 lifts
 * iff u < 2; the toward-zero 64 - 2^-40 moves unless u == UINT32_MAX;
 * 64 + 2^-32, 64 + 1.5 * 2^-32, 64 + 2^-40 and a gap of 32704 binades do not
 * lift at u = 1 (the documented 2^-32 cutoff; F = 1 would lift only at
 * u == 0, which the xorshift32 backend never draws -- the u == 0 boundary is
 * pinned on the primitive, testAddAndRoundTailCutoffIsTwoToMinus32). */
void testEngineLargeExponentGapKeepsTheTail(void) {
    const int32_t codes[1] = {64};
    const uint8_t stored[1] = {127};
    const struct {
        int32_t raw;
        int16_t rawE;
        uint32_t u;
        int32_t expect;
    } cases[] = {{3, -33, 1u, 64},
                 {1, -32, 1u, 64},
                 {1, -31, 1u, 65},
                 {1, -31, 2u, 64},
                 {-1, -40, 0xFFFFFFFEu, 64},
                 {-1, -40, 0xFFFFFFFFu, 63},
                 {1, -40, 1u, 64},
                 {INT32_MIN, -32704, 1u, 64},
                 {INT32_MAX, -32768, 1u, 64},
                 {1 << 30, -61, 1u, 65},
                 {1 << 30, -62, 1u, 64}};
    for (size_t c = 0; c < sizeof cases / sizeof cases[0]; c++) {
        int32_t raw[1] = {cases[c].raw};
        int16_t rawE[1] = {cases[c].rawE};
        rngStream_t s = rngStreamWithNextDraw(cases[c].u);
        accInit(&fixA, 1, 1, 0, 8, 8, SR_HALF_AWAY, codes, stored, raw, rawE);
        accumulateBfpRawIntoBfpRescale(&fixA.target, &fixA.inc, &s);
        int32_t got[1];
        accCodes(&fixA, got);
        TEST_ASSERT_EQUAL_UINT8_MESSAGE(127, fixA.exps[0], "exponent moved");
        TEST_ASSERT_EQUAL_INT32(cases[c].expect, got[0]);
    }
}

/* Cancellation: an exact zero sum is the zero state; a residue within the
 * working grid gets its exact fresh exponent; a residue cancelled below the
 * working grid E_c keeps E_c (the documented floor); a cancelled element
 * next to a live one takes the group's exponent. */
void testEngineCancellation(void) {
    int32_t got[2];
    {
        const int32_t codes[2] = {100, -37};
        const uint8_t stored[1] = {127};
        const int32_t raw[2] = {-(100 << 20), 37 << 20};
        const int16_t rawE[1] = {-20};
        accInit(&fixA, 2, 1, 0, 8, 8, HALF_AWAY, codes, stored, raw, rawE);
        accumulateBfpRawIntoBfpRescale(&fixA.target, &fixA.inc, NULL);
        accCodes(&fixA, got);
        TEST_ASSERT_EQUAL_UINT8(127, fixA.exps[0]); /* zero state = bias */
        TEST_ASSERT_EQUAL_INT32(0, got[0]);
        TEST_ASSERT_EQUAL_INT32(0, got[1]);
    }
    {
        /* 100 - (100 - 2^-8) = 2^-8: E = -14, code 64 (E_c = -23) */
        const int32_t codes[1] = {100};
        const uint8_t stored[1] = {127};
        const int32_t raw[1] = {-(100 << 20) + (1 << 12)};
        const int16_t rawE[1] = {-20};
        accInit(&fixA, 1, 1, 0, 8, 8, HALF_AWAY, codes, stored, raw, rawE);
        accumulateBfpRawIntoBfpRescale(&fixA.target, &fixA.inc, NULL);
        accCodes(&fixA, got);
        TEST_ASSERT_EQUAL_UINT8(127 - 14, fixA.exps[0]);
        TEST_ASSERT_EQUAL_INT32(64, got[0]);
    }
    {
        /* 100 - (100 - 2^-20) = 2^-20: below E_c = -23 + 7 bits -> E_d = E_c, code 8 */
        const int32_t codes[1] = {100};
        const uint8_t stored[1] = {127};
        const int32_t raw[1] = {-(100 << 20) + 1};
        const int16_t rawE[1] = {-20};
        accInit(&fixA, 1, 1, 0, 8, 8, HALF_AWAY, codes, stored, raw, rawE);
        accumulateBfpRawIntoBfpRescale(&fixA.target, &fixA.inc, NULL);
        accCodes(&fixA, got);
        TEST_ASSERT_EQUAL_UINT8(127 - 23, fixA.exps[0]);
        TEST_ASSERT_EQUAL_INT32(8, got[0]);
    }
    {
        /* element 0 cancels exactly, element 1 sets the exponent: 3 + 2^-1 */
        const int32_t codes[2] = {100, 3};
        const uint8_t stored[1] = {127};
        const int32_t raw[2] = {-(100 << 20), 1 << 19};
        const int16_t rawE[1] = {-20};
        accInit(&fixA, 2, 1, 0, 8, 8, HALF_AWAY, codes, stored, raw, rawE);
        accumulateBfpRawIntoBfpRescale(&fixA.target, &fixA.inc, NULL);
        accCodes(&fixA, got);
        TEST_ASSERT_EQUAL_UINT8(127 - 5, fixA.exps[0]); /* 3.5 <= 127 * 2^-5 */
        TEST_ASSERT_EQUAL_INT32(0, got[0]);
        TEST_ASSERT_EQUAL_INT32(112, got[1]);
    }
}

/* Exponent extremes at exponentBits = 4 (bias 7, stored [0, 15], E in
 * [-7, 8]): the flush regime SRs a value far below the smallest step with its
 * exact probability (stored 0, not the zero state); the cap regime saturates;
 * a raw beyond 2^30 of the cap step (the working grid capped at E = 8) still
 * computes its non-saturating neighbours exactly. */
void testEngineExponentExtremes(void) {
    int32_t got[3];
    {
        /* 2^-20 under E_min = -7: code 1 iff u < 2^32 * 2^-13 = 2^19 */
        const int32_t codes[1] = {0};
        const uint8_t stored[1] = {7};
        const int32_t raw[1] = {1};
        const int16_t rawE[1] = {-20};
        const uint32_t us[2] = {(1u << 19) - 1u, 1u << 19};
        const int32_t expect[2] = {1, 0};
        for (int c = 0; c < 2; c++) {
            rngStream_t s = rngStreamWithNextDraw(us[c]);
            accInit(&fixA, 1, 1, 0, 8, 4, SR_HALF_AWAY, codes, stored, raw, rawE);
            accumulateBfpRawIntoBfpRescale(&fixA.target, &fixA.inc, &s);
            accCodes(&fixA, got);
            TEST_ASSERT_EQUAL_UINT8(0, fixA.exps[0]);
            TEST_ASSERT_EQUAL_INT32(expect[c], got[0]);
        }
    }
    {
        /* 2^20 and -2^20 over the cap 127 * 2^8: stored 15, codes saturate */
        const int32_t codes[2] = {0, 0};
        const uint8_t stored[1] = {7};
        const int32_t raw[2] = {1, -1};
        const int16_t rawE[1] = {20};
        accInit(&fixA, 2, 1, 0, 8, 4, HALF_AWAY, codes, stored, raw, rawE);
        accumulateBfpRawIntoBfpRescale(&fixA.target, &fixA.inc, NULL);
        accCodes(&fixA, got);
        TEST_ASSERT_EQUAL_UINT8(15, fixA.exps[0]);
        TEST_ASSERT_EQUAL_INT32(127, got[0]);
        TEST_ASSERT_EQUAL_INT32(-128, got[1]);
    }
    {
        /* raw 2^30 * 2^20 saturates through the capped working grid; its
         * neighbours 5 * 2^8 + 0 and 127 * 2^8 - 2^20 are exact / clamp */
        const int32_t codes[3] = {0, 5, 127};
        const uint8_t stored[1] = {15};
        const int32_t raw[3] = {1 << 30, 0, -1};
        const int16_t rawE[1] = {20};
        accInit(&fixA, 3, 1, 0, 8, 4, HALF_AWAY, codes, stored, raw, rawE);
        accumulateBfpRawIntoBfpRescale(&fixA.target, &fixA.inc, NULL);
        accCodes(&fixA, got);
        TEST_ASSERT_EQUAL_UINT8(15, fixA.exps[0]);
        TEST_ASSERT_EQUAL_INT32(127, got[0]);
        TEST_ASSERT_EQUAL_INT32(5, got[1]);
        TEST_ASSERT_EQUAL_INT32(-128, got[2]);
    }
    {
        /* the same with negative saturation and SR (draws still consumed) */
        const int32_t codes[3] = {0, 5, 0};
        const uint8_t stored[1] = {15};
        const int32_t raw[3] = {INT32_MIN, 0, 0};
        const int16_t rawE[1] = {32767};
        rngStream_t s = rngOpen((rngKey_t){{99u, 1u}});
        rngStream_t ref = rngOpen((rngKey_t){{99u, 1u}});
        accInit(&fixA, 3, 1, 0, 8, 4, SR_HALF_AWAY, codes, stored, raw, rawE);
        accumulateBfpRawIntoBfpRescale(&fixA.target, &fixA.inc, &s);
        accCodes(&fixA, got);
        TEST_ASSERT_EQUAL_UINT8(15, fixA.exps[0]);
        TEST_ASSERT_EQUAL_INT32(-128, got[0]);
        TEST_ASSERT_EQUAL_INT32(5, got[1]);
        TEST_ASSERT_EQUAL_INT32(0, got[2]);
        for (int i = 0; i < 3; i++) {
            (void)rngNextU32(&ref);
        }
        TEST_ASSERT_EQUAL_UINT32(rngNextU32(&ref), rngNextU32(&s));
    }
    {
        /* raw exponent at the int16 floor: a nonzero raw that is all tail */
        const int32_t codes[1] = {1};
        const uint8_t stored[1] = {127};
        const int32_t raw[1] = {INT32_MIN};
        const int16_t rawE[1] = {INT16_MIN};
        accInit(&fixA, 1, 1, 0, 8, 8, HALF_AWAY, codes, stored, raw, rawE);
        accumulateBfpRawIntoBfpRescale(&fixA.target, &fixA.inc, NULL);
        accCodes(&fixA, got);
        TEST_ASSERT_EQUAL_UINT8(127 - 6, fixA.exps[0]); /* 1 - tiny: 127 * 2^-6 */
        TEST_ASSERT_EQUAL_INT32(64, got[0]);
    }
}

/* A fresh accumulator (all codes zero, e.g. after optimizerZeroGrad) takes
 * the increment's own exponent: a zero OLD side never dominates the working
 * grid, whatever its stored exponent. */
void testEngineFreshAccumulatorTakesTheIncrementGrid(void) {
    const int32_t codes[2] = {0, 0};
    const uint8_t stored[1] = {254};
    const int32_t raw[2] = {3, -1};
    const int16_t rawE[1] = {-40};
    accInit(&fixA, 2, 1, 0, 8, 8, HALF_AWAY, codes, stored, raw, rawE);
    accumulateBfpRawIntoBfpRescale(&fixA.target, &fixA.inc, NULL);
    int32_t got[2];
    accCodes(&fixA, got);
    TEST_ASSERT_EQUAL_UINT8(127 - 45, fixA.exps[0]); /* 3 * 2^-40 <= 127 * 2^-45 */
    TEST_ASSERT_EQUAL_INT32(96, got[0]);
    TEST_ASSERT_EQUAL_INT32(-32, got[1]);
}

/* HALF_AWAY consumes no draw and runs without a stream; n == 0 leaves the
 * canonical zero state (R7). */
void testEngineEmptyTargetIsZeroState(void) {
    const uint8_t stored[1] = {3};
    const int16_t rawE[1] = {5};
    accInit(&fixA, 0, 1, 0, 8, 8, SR_HALF_AWAY, NULL, stored, NULL, rawE);
    rngStream_t s = rngOpen((rngKey_t){{5u, 5u}});
    accumulateBfpRawIntoBfpRescale(&fixA.target, &fixA.inc, &s);
    TEST_ASSERT_EQUAL_UINT8(127, fixA.exps[0]);
}

static void engineAliasBody(void) {
    const int32_t codes[2] = {1, 2};
    const uint8_t stored[1] = {127};
    const int32_t raw[2] = {1, 2};
    const int16_t rawE[1] = {0};
    accInit(&fixA, 2, 1, 0, 8, 8, HALF_AWAY, codes, stored, raw, rawE);
    fixA.inc.data = fixA.target.data;
    accumulateBfpRawIntoBfpRescale(&fixA.target, &fixA.inc, NULL);
}

static void engineGeometryMismatchBody(void) {
    const int32_t codes[4] = {1, 2, 3, 4};
    const uint8_t stored[2] = {127, 127};
    const int32_t raw[4] = {1, 2, 3, 4};
    const int16_t rawE[2] = {0, 0};
    accInit(&fixA, 4, 2, 2, 8, 8, HALF_AWAY, codes, stored, raw, rawE);
    fixA.rawQC.numGroups = 1;
    fixA.rawQC.groupSize = 0;
    accumulateBfpRawIntoBfpRescale(&fixA.target, &fixA.inc, NULL);
}

static void engineSrWithoutStreamBody(void) {
    const int32_t codes[2] = {1, 2};
    const uint8_t stored[1] = {127};
    const int32_t raw[2] = {1, 2};
    const int16_t rawE[1] = {0};
    accInit(&fixA, 2, 1, 0, 8, 8, SR_HALF_AWAY, codes, stored, raw, rawE);
    accumulateBfpRawIntoBfpRescale(&fixA.target, &fixA.inc, NULL);
}

static void engineCountMismatchBody(void) {
    const int32_t codes[4] = {1, 2, 3, 4};
    const uint8_t stored[1] = {127};
    const int32_t raw[4] = {1, 2, 3, 4};
    const int16_t rawE[1] = {0};
    accInit(&fixA, 4, 1, 0, 8, 8, HALF_AWAY, codes, stored, raw, rawE);
    static size_t shortDims[1] = {3};
    static size_t shortOrder[1] = {0};
    static shape_t shortShape;
    setShape(&shortShape, shortDims, 1, shortOrder);
    fixA.inc.shape = &shortShape;
    accumulateBfpRawIntoBfpRescale(&fixA.target, &fixA.inc, NULL);
}

static void engineWrongDtypeBody(void) {
    const int32_t codes[2] = {1, 2};
    const uint8_t stored[1] = {127};
    const int32_t raw[2] = {1, 2};
    const int16_t rawE[1] = {0};
    accInit(&fixA, 2, 1, 0, 8, 8, HALF_AWAY, codes, stored, raw, rawE);
    accumulateBfpRawIntoBfpRescale(&fixA.target, &fixA.target, NULL);
}

/* The raw increment's storage must not overlap the target's: a payload that
 * starts inside the target's packed bytes, or raw exponents that sit on the
 * target's exponent bytes, would be rewritten while still being read. */
static uint8_t g_accOverlap[64] __attribute__((aligned(4)));
static void enginePayloadOverlapBody(void) {
    const int32_t codes[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    const uint8_t stored[1] = {127};
    const int32_t raw[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    const int16_t rawE[1] = {0};
    accInit(&fixA, 8, 1, 0, 8, 8, HALF_AWAY, codes, stored, raw, rawE);
    memcpy(g_accOverlap, fixA.data, 8);
    memcpy(g_accOverlap + 4, fixA.raw, 8 * sizeof(int32_t));
    fixA.target.data = g_accOverlap;
    fixA.inc.data = g_accOverlap + 4;
    accumulateBfpRawIntoBfpRescale(&fixA.target, &fixA.inc, NULL);
}
static void engineExponentOverlapBody(void) {
    const int32_t codes[2] = {1, 2};
    const uint8_t stored[2] = {127, 127};
    const int32_t raw[2] = {1, 2};
    const int16_t rawE[2] = {0, 0};
    accInit(&fixA, 2, 2, 1, 8, 8, HALF_AWAY, codes, stored, raw, rawE);
    fixA.rawQC.exponents = (int16_t *)(void *)fixA.exps;
    accumulateBfpRawIntoBfpRescale(&fixA.target, &fixA.inc, NULL);
}

void testEngineRejectsOverlappingStorage(void) {
    ASSERT_EXITS_WITH_OUTPUT(1, "payload overlaps", enginePayloadOverlapBody());
    ASSERT_EXITS_WITH_OUTPUT(1, "exponents overlap", engineExponentOverlapBody());
}

void testEngineRejectsBadInputs(void) {
    ASSERT_EXITS_WITH_FAILURE(engineAliasBody());
    ASSERT_EXITS_WITH_FAILURE(engineGeometryMismatchBody());
    ASSERT_EXITS_WITH_FAILURE(engineSrWithoutStreamBody());
    ASSERT_EXITS_WITH_FAILURE(engineCountMismatchBody());
    ASSERT_EXITS_WITH_FAILURE(engineWrongDtypeBody());
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testScalarZeroOneAndPowersOfTwoAreExact);
    RUN_TEST(testScalarRoundsSignificandHalfAwayTo15Bits);
    RUN_TEST(testScalarRoundUpToNextBinadeRenormalizes);
    RUN_TEST(testScalarRejectsNonFinite);
    RUN_TEST(testScalarMulMantIsExactAtExtremes);
    RUN_TEST(testQrOfTermIsExactAtEveryShift);
    RUN_TEST(testQrShrIsExactAtEveryDistance);
    RUN_TEST(testAddAndRoundMatchesExactReference);
    RUN_TEST(testAddAndRoundSrIsOddPerDraw);
    RUN_TEST(testAddAndRoundTailCutoffIsTwoToMinus32);
    RUN_TEST(testEngineMatchesExactReferenceOnRandomTargets);
    RUN_TEST(testEngineDeadZoneUpdateMovesAtTheExactPinnedThreshold);
    RUN_TEST(testEngineDeadZoneUpdateMovesWithProbabilityOneIn256);
    RUN_TEST(testEngineSrIsUnbiased);
    RUN_TEST(testEngineLargeExponentGapKeepsTheTail);
    RUN_TEST(testEngineCancellation);
    RUN_TEST(testEngineExponentExtremes);
    RUN_TEST(testEngineFreshAccumulatorTakesTheIncrementGrid);
    RUN_TEST(testEngineEmptyTargetIsZeroState);
    RUN_TEST(testEngineRejectsBadInputs);
    RUN_TEST(testEngineRejectsOverlappingStorage);
    return UNITY_END();
}
