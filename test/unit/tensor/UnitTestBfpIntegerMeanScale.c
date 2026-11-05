#include "BfpKernelSupport.h"
#include "DTypes.h"
#include "DeathTest.h"
#include "Quantization.h"
#include "RNG.h"
#include "Tensor.h"
#include "TensorConversion.h"
#include "unity.h"

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

/* The integer macro-batch mean scale (arithmetic-bfp.md §15.6): 1/count as a
 * scalar from the integer count, and the in-place integer scale of a BFP
 * gradient by such a scalar. Every expected value is computed HERE in int64
 * exact arithmetic, independent of the quotient/remainder machinery. */

void setUp(void) {}
void tearDown(void) {}

static uint32_t testRngState = 0x7A3C91E5u;

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

static unsigned bitLen64(uint64_t v) {
    unsigned n = 0;
    while (v) {
        n++;
        v >>= 1;
    }
    return n;
}

/* ---- bfpScalarFromCount ---------------------------------------------- */

void testScalarFromCountPowersOfTwoAreExact(void) {
    for (unsigned k = 0; k <= 30; k++) {
        bfpScalar_t s = bfpScalarFromCount((size_t)1 << k, "t");
        TEST_ASSERT_EQUAL_INT32(16384, s.q);
        TEST_ASSERT_EQUAL_INT16(-14 - (int)k, s.e);
    }
}

/* q = round(2^(14 + L) / count), L = bitlen(count), exact int64 division in
 * the test; no ties exist for a non-power-of-two count. Normalized, and the
 * relative error against the exact 1/count is <= 2^-15. */
static void assertCountScalar(uint32_t c) {
    bfpScalar_t s = bfpScalarFromCount(c, "t");
    TEST_ASSERT_TRUE(s.q >= 16384 && s.q <= 32767);
    const unsigned L = bitLen64(c);
    uint64_t num = (uint64_t)1 << (14 + L);
    uint64_t q = (2 * num + c) / (2 * (uint64_t)c); /* round half up, positive */
    int e = -(int)(14 + L);
    if ((c & (c - 1u)) == 0) {
        q = 16384;
        e = -(int)(14 + L - 1);
    } else if (q == 32768) {
        q = 16384;
        e += 1;
    }
    TEST_ASSERT_EQUAL_INT32((int32_t)q, s.q);
    TEST_ASSERT_EQUAL_INT16(e, s.e);
    double rel = fabs(ldexp((double)s.q, s.e) * (double)c - 1.0);
    TEST_ASSERT_TRUE(rel <= ldexp(1.0, -15));
}

void testScalarFromCountMatchesExactRounding(void) {
    for (uint32_t c = 1; c <= 70000; c++) {
        assertCountScalar(c);
    }
    for (int i = 0; i < 20000; i++) {
        uint32_t c = 1u + (testRand() & 0x7FFFFFFEu);
        assertCountScalar(c);
    }
    assertCountScalar(0x7FFFFFFFu);
    assertCountScalar(0x40000001u);
}

static void countZeroBody(void) {
    (void)bfpScalarFromCount(0, "t");
}
static void countHugeBody(void) {
    (void)bfpScalarFromCount((size_t)1u << 31, "t");
}

void testScalarFromCountRejectsZeroAndHuge(void) {
    ASSERT_EXITS_WITH_FAILURE(countZeroBody());
    ASSERT_EXITS_WITH_FAILURE(countHugeBody());
}

/* ---- scaleBfpTensorByScalarInPlace ----------------------------------- */

#define SC_MAX_N 4096
#define SC_MAX_GROUPS 4096

typedef struct {
    size_t n;
    size_t dims[1];
    size_t order[1];
    shape_t shape;
    uint8_t exps[SC_MAX_GROUPS];
    bfpQConfig_t qc;
    quantization_t q;
    uint8_t data[SC_MAX_N * 2 + 8];
    tensor_t t;
} scFixture_t;

static scFixture_t fix;

static void scInit(scFixture_t *f, size_t n, size_t numGroups, size_t groupSize, uint8_t m,
                   uint8_t e, roundingMode_t rm, const int32_t *codes, const uint8_t *stored) {
    TEST_ASSERT_TRUE(n <= SC_MAX_N && numGroups <= SC_MAX_GROUPS);
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
        int32_t tmp[SC_MAX_N];
        memcpy(tmp, codes, n * sizeof(int32_t));
        byteConversion((uint8_t *)tmp, 32, f->data, m, n);
    }
    setTensorValues(&f->t, f->data, &f->shape, &f->q, NULL);
}

static void scCodes(const scFixture_t *f, int32_t *out) {
    unpackSignExtend(f->data, f->qc.mantissaBits, 0, out, f->n);
}

static uint64_t absI64(int64_t v) {
    return v < 0 ? 0u - (uint64_t)v : (uint64_t)v;
}

/* Exact rounding of V * 2^-d (V exact int64, d >= 0) on the magnitude: F =
 * the top 32 bits of the remainder, SR lifts iff u < F, HALF_AWAY iff F >=
 * 2^31; sign after; clamp to the code range. */
static int32_t refRound(int64_t v, int d, bool sr, uint32_t u, uint32_t posCap, uint32_t negCap) {
    const bool neg = v < 0;
    const uint64_t mag = absI64(v);
    uint64_t m = d >= 64 ? 0 : mag >> d;
    uint64_t rem = d >= 64 ? mag : (d == 0 ? 0 : mag & ((UINT64_C(1) << d) - 1u));
    uint32_t f;
    if (d == 0) {
        f = 0;
    } else if (d <= 32) {
        f = (uint32_t)(rem << (32 - d));
    } else if (d - 32 < 64) {
        f = (uint32_t)(rem >> (d - 32));
    } else {
        f = 0;
    }
    bool lift = sr ? u < f : f >= 0x80000000u;
    const uint32_t cap = neg ? negCap : posCap;
    uint64_t t = m + (uint64_t)lift;
    if (t > cap) {
        t = cap;
    }
    return neg ? (int32_t)(0u - (uint32_t)t) : (int32_t)t;
}

/* The documented result, per group: A = absmax code; A == 0 -> zero state,
 * no draws. Otherwise P = q * A on the grid 2^(E + s.e); the fresh exponent
 * is the smallest E' with P <= qMax * 2^(E' - E - s.e), floored at stored 0;
 * d = E' - E - s.e. d == 14 with q == 2^14 (a power-of-two factor on a group
 * that neither renormalizes nor hits the floor): codes unchanged, no draws.
 * Otherwise every element is code * q rounded d steps down, one draw per
 * element in element order under SR. Returns the number of draws. */
static size_t refScale(const int32_t *codes, const uint8_t *stored, size_t n, size_t numGroups,
                       size_t groupSize, uint8_t m, uint8_t e, roundingMode_t rm, bfpScalar_t s,
                       rngStream_t *rs, int32_t *outCodes, uint8_t *outStored) {
    const int bias = (1 << (e - 1)) - 1;
    const uint64_t qMax = (1u << (m - 1)) - 1u;
    const uint32_t posCap = (uint32_t)qMax;
    const uint32_t negCap = 1u << (m - 1);
    const size_t gsz = groupSize == 0 ? n : groupSize;
    size_t draws = 0;
    for (size_t g = 0; g < numGroups; g++) {
        const size_t start = g * gsz;
        const size_t end = start + gsz;
        uint64_t a = 0;
        for (size_t i = start; i < end; i++) {
            uint64_t mag = absI64(codes[i]);
            a = mag > a ? mag : a;
        }
        if (a == 0) {
            outStored[g] = (uint8_t)bias;
            for (size_t i = start; i < end; i++) {
                outCodes[i] = 0;
            }
            continue;
        }
        const int eRaw = (int)stored[g] - bias + s.e;
        const uint64_t p = a * (uint64_t)s.q;
        int d = -40;
        while (!(d >= 0 ? p <= (qMax << d) : p <= (qMax >> -d))) {
            d++;
        }
        int eNew = eRaw + d;
        if (eNew < -bias) {
            eNew = -bias;
            d = eNew - eRaw;
        }
        /* the cap: reachable only by the negative code bound -2^(m-1) at s ~ 1 */
        const int capE = ((1 << e) - 1 > bias + 127 ? bias + 127 : (1 << e) - 1) - bias;
        if (eNew > capE) {
            eNew = capE;
            d = eNew - eRaw;
        }
        outStored[g] = (uint8_t)(eNew + bias);
        if (s.q == 16384 && d == 14) {
            for (size_t i = start; i < end; i++) {
                outCodes[i] = codes[i];
            }
            continue;
        }
        for (size_t i = start; i < end; i++) {
            uint32_t u = 0;
            if (rm == SR_HALF_AWAY) {
                u = rngNextU32(rs);
                draws++;
            }
            outCodes[i] =
                refRound((int64_t)codes[i] * s.q, d, rm == SR_HALF_AWAY, u, posCap, negCap);
        }
    }
    return draws;
}

static void assertScaleMatchesReference(size_t n, size_t numGroups, size_t groupSize, uint8_t m,
                                        uint8_t e, roundingMode_t rm, const int32_t *codes,
                                        const uint8_t *stored, bfpScalar_t s, uint32_t seed) {
    static int32_t expCodes[SC_MAX_N];
    static int32_t gotCodes[SC_MAX_N];
    static uint8_t expStored[SC_MAX_GROUPS];
    rngStream_t refStream = rngOpen((rngKey_t){{seed, 9u}});
    rngStream_t engStream = rngOpen((rngKey_t){{seed, 9u}});
    refScale(codes, stored, n, numGroups, groupSize, m, e, rm, s, &refStream, expCodes, expStored);
    scInit(&fix, n, numGroups, groupSize, m, e, rm, codes, stored);
    scaleBfpTensorByScalarInPlace(&fix.t, s, rm == SR_HALF_AWAY ? &engStream : NULL);
    scCodes(&fix, gotCodes);
    TEST_ASSERT_EQUAL_UINT8_ARRAY(expStored, fix.exps, numGroups);
    TEST_ASSERT_EQUAL_INT32_ARRAY(expCodes, gotCodes, n);
    if (rm == SR_HALF_AWAY) {
        /* the documented draw count, in element order: streams in step */
        TEST_ASSERT_EQUAL_UINT32(rngNextU32(&refStream), rngNextU32(&engStream));
    }
}

/* Random grouped targets against the exact reference: per-tensor and grouped
 * geometries (group size 1 wraps the window), byte and sub-byte widths,
 * multi-chunk n, zero groups, non-tight grids, groups near the exponent floor;
 * power-of-two and other counts; HALF_AWAY and SR. */
void testScaleMatchesExactReferenceOnRandomTargets(void) {
    static int32_t codes[SC_MAX_N];
    static uint8_t stored[SC_MAX_GROUPS];
    const struct {
        size_t n, numGroups, groupSize;
        uint8_t m, e;
    } geo[] = {{1, 1, 0, 8, 8},       {7, 1, 0, 4, 8},       {600, 1, 0, 8, 8},
               {700, 700, 1, 5, 8},   {699, 233, 3, 8, 8},   {640, 10, 64, 16, 8},
               {513, 1, 0, 2, 8},     {770, 110, 7, 12, 8},  {300, 3, 100, 6, 4},
               {32, 2, 16, 8, 3},     {1000, 8, 125, 3, 5},  {4096, 1, 0, 8, 8},
               {2048, 2048, 1, 8, 4}, {1024, 4, 256, 16, 8}, {999, 333, 3, 7, 6}};
    const size_t counts[] = {1, 2, 3, 5, 7, 48, 64, 100, 127, 129, 1000, 4096, 65535, 1u << 20};
    for (int round = 0; round < 4; round++) {
        for (size_t c = 0; c < sizeof geo / sizeof geo[0]; c++) {
            const size_t n = geo[c].n;
            const size_t ng = geo[c].numGroups;
            const size_t gsz = geo[c].groupSize == 0 ? n : geo[c].groupSize;
            const uint8_t m = geo[c].m;
            const uint8_t e = geo[c].e;
            const int bias = (1 << (e - 1)) - 1;
            const int maxStored = (1 << e) - 1 > bias + 127 ? bias + 127 : (1 << e) - 1;
            for (size_t g = 0; g < ng; g++) {
                int flavour = testRandInt(0, 9);
                /* stored near the floor in a third of the groups */
                stored[g] = (uint8_t)(flavour < 3 ? testRandInt(0, maxStored < 12 ? maxStored : 12)
                                                  : testRandInt(0, maxStored));
                int topBits = flavour == 4 ? testRandInt(1, m - 1) : m - 1; /* 4: not tight */
                for (size_t i = g * gsz; i < (g + 1) * gsz; i++) {
                    int32_t lim = (int32_t)((1u << (topBits - 0)) - 1u);
                    if (topBits >= m) {
                        lim = (1 << (m - 1)) - 1;
                    }
                    codes[i] = flavour == 0 ? 0 : testRandInt(-lim, lim);
                    if (flavour == 5 && i == g * gsz) {
                        codes[i] = -(1 << (m - 1)); /* the negative code bound */
                    }
                }
            }
            for (size_t k = 0; k < sizeof counts / sizeof counts[0]; k++) {
                bfpScalar_t s = bfpScalarFromCount(counts[k], "t");
                uint32_t seed = 0x2000u + (uint32_t)(round * 997 + (int)c * 31 + (int)k);
                assertScaleMatchesReference(n, ng, geo[c].groupSize, m, e, HALF_AWAY, codes, stored,
                                            s, seed);
                assertScaleMatchesReference(n, ng, geo[c].groupSize, m, e, SR_HALF_AWAY, codes,
                                            stored, s, seed);
            }
        }
    }
}

/* Power-of-two count on a tight grid away from the floor: the scale is the
 * exponent alone -- codes bit-unchanged, every stored exponent lowered by
 * log2(count), and not a single draw (the stream is untouched). */
void testScalePowerOfTwoIsAnExactExponentShiftWithoutDraws(void) {
    static int32_t codes[1000];
    static int32_t got[1000];
    uint8_t stored[8];
    for (size_t g = 0; g < 8; g++) {
        stored[g] = (uint8_t)(100 + g);
        for (size_t i = g * 125; i < (g + 1) * 125; i++) {
            codes[i] = testRandInt(-127, 127);
        }
        codes[g * 125] = (g & 1) ? 127 : -127; /* tight: absmax in the top binade */
    }
    for (unsigned k = 0; k <= 12; k++) {
        scInit(&fix, 1000, 8, 125, 8, 8, SR_HALF_AWAY, codes, stored);
        rngStream_t s = rngOpen((rngKey_t){{77u, k}});
        rngStream_t untouched = rngOpen((rngKey_t){{77u, k}});
        scaleBfpTensorByScalarInPlace(&fix.t, bfpScalarFromCount((size_t)1 << k, "t"), &s);
        scCodes(&fix, got);
        TEST_ASSERT_EQUAL_INT32_ARRAY(codes, got, 1000);
        for (size_t g = 0; g < 8; g++) {
            TEST_ASSERT_EQUAL_UINT8(stored[g] - k, fix.exps[g]);
        }
        TEST_ASSERT_EQUAL_UINT32(rngNextU32(&untouched), rngNextU32(&s));
    }
}

/* The exponent floor: stored 3, count 64 would need stored -3. The group
 * saturates at stored 0 and its codes shift right by the 3 missing steps,
 * rounded by SR from the stream: 100 / 8 = 12.5 lifts iff u < 2^31, -77 / 8 =
 * -9.625 lifts (to -10) iff u < 0.625 * 2^32; an exact 64 / 8 never moves. */
void testScaleAtTheExponentFloorRoundsTheShiftedCodes(void) {
    const int32_t codes[3] = {100, -77, 64};
    const uint8_t stored[1] = {3};
    for (size_t k = 0; k < 64; k++) { /* the stream's own draws decide; 64 seeds */
        rngStream_t s = rngOpen((rngKey_t){{0x5151u, (uint32_t)k}});
        rngStream_t peek = s;
        uint32_t u0 = rngNextU32(&peek);
        uint32_t u1 = rngNextU32(&peek);
        (void)rngNextU32(&peek); /* element 2: exact, its draw still consumed */
        scInit(&fix, 3, 1, 0, 8, 8, SR_HALF_AWAY, codes, stored);
        scaleBfpTensorByScalarInPlace(&fix.t, bfpScalarFromCount(64, "t"), &s);
        int32_t got[3];
        scCodes(&fix, got);
        TEST_ASSERT_EQUAL_UINT8(0, fix.exps[0]);
        TEST_ASSERT_EQUAL_INT32(u0 < 0x80000000u ? 13 : 12, got[0]);
        TEST_ASSERT_EQUAL_INT32(u1 < 0xA0000000u ? -10 : -9, got[1]);
        TEST_ASSERT_EQUAL_INT32(8, got[2]);
        /* three draws consumed, one per element */
        TEST_ASSERT_EQUAL_UINT32(rngNextU32(&peek), rngNextU32(&s));
    }
    /* HALF_AWAY at the floor: 12.5 -> 13, -9.625 -> -10, 8 -> 8 */
    scInit(&fix, 3, 1, 0, 8, 8, HALF_AWAY, codes, stored);
    scaleBfpTensorByScalarInPlace(&fix.t, bfpScalarFromCount(64, "t"), NULL);
    int32_t got[3];
    scCodes(&fix, got);
    TEST_ASSERT_EQUAL_UINT8(0, fix.exps[0]);
    TEST_ASSERT_EQUAL_INT32(13, got[0]);
    TEST_ASSERT_EQUAL_INT32(-10, got[1]);
    TEST_ASSERT_EQUAL_INT32(8, got[2]);
    /* far below the floor: everything below half a code step flushes under
     * HALF_AWAY, and under SR still lifts with its exact probability */
    const int32_t tiny[1] = {1};
    const uint8_t storedTiny[1] = {0};
    scInit(&fix, 1, 1, 0, 8, 8, HALF_AWAY, tiny, storedTiny);
    scaleBfpTensorByScalarInPlace(&fix.t, bfpScalarFromCount(1u << 20, "t"), NULL);
    scCodes(&fix, got);
    TEST_ASSERT_EQUAL_UINT8(0, fix.exps[0]);
    TEST_ASSERT_EQUAL_INT32(0, got[0]);
    size_t lifted = 0;
    rngStream_t s = rngOpen((rngKey_t){{0xF100Du, 3u}});
    for (int i = 0; i < 40000; i++) {
        scInit(&fix, 1, 1, 0, 8, 8, SR_HALF_AWAY, (const int32_t[]){1}, (const uint8_t[]){4});
        scaleBfpTensorByScalarInPlace(&fix.t, bfpScalarFromCount(1u << 10, "t"), &s);
        scCodes(&fix, got);
        TEST_ASSERT_TRUE(got[0] == 0 || got[0] == 1);
        lifted += (size_t)got[0];
    }
    /* 1 * 2^(4-10) at stored 0: 2^-6 of a code step, expected 625 of 40000 */
    double sigma = sqrt(40000.0 * (1.0 / 64) * (63.0 / 64));
    TEST_ASSERT_TRUE(fabs((double)lifted - 625.0) <= 5 * sigma);
}

/* A non-power-of-two count under SR is unbiased against the ENCODED scalar
 * code * q * 2^e (the 15-bit encoding of 1/count is a separate, deterministic
 * error <= 2^-15, asserted on bfpScalarFromCount): the mean code over many
 * seeded calls sits within 5 sigma of the exact value, for both signs. */
void testScaleNonPowerOfTwoIsUnbiasedUnderSr(void) {
    static int32_t codes[4096];
    static int32_t got[4096];
    const uint8_t stored[1] = {127};
    const size_t countsToTest[] = {3, 48, 100};
    for (size_t k = 0; k < 3; k++) {
        const bfpScalar_t s = bfpScalarFromCount(countsToTest[k], "t");
        for (int sign = -1; sign <= 1; sign += 2) {
            for (size_t i = 0; i < 4096; i++) {
                codes[i] = (i == 0 ? 127 : 93) * sign; /* element 0 pins the grid */
            }
            /* reference: 127 * q sets d; element i > 0 is 93 * q / 2^d */
            const uint64_t p = 127u * (uint64_t)s.q;
            int d = 0;
            while (p > (127u << d)) {
                d++;
            }
            const double exact = 93.0 * (double)s.q / ldexp(1.0, d);
            const double frac = exact - floor(exact);
            rngStream_t rs = rngOpen((rngKey_t){{0xACE5u + (uint32_t)k, (uint32_t)(sign + 1)}});
            double sum = 0;
            const int calls = 10;
            for (int call = 0; call < calls; call++) {
                scInit(&fix, 4096, 1, 0, 8, 8, SR_HALF_AWAY, codes, stored);
                scaleBfpTensorByScalarInPlace(&fix.t, s, &rs);
                scCodes(&fix, got);
                TEST_ASSERT_EQUAL_UINT8(127 + s.e + d, fix.exps[0]);
                for (size_t i = 1; i < 4096; i++) {
                    TEST_ASSERT_TRUE(got[i] == sign * (int32_t)floor(exact) ||
                                     got[i] == sign * ((int32_t)floor(exact) + 1));
                    sum += got[i];
                }
            }
            const double nTot = 4095.0 * calls;
            const double mean = sum / nTot;
            const double sigma = sqrt(frac * (1 - frac) / nTot);
            TEST_ASSERT_TRUE(fabs(mean - sign * exact) <= 5 * sigma + 1e-12);
        }
    }
}

/* Zero groups take the canonical zero state (stored = bias) whatever they
 * held, draw nothing, and leave their neighbours' draws in place; an empty
 * tensor is the zero state too. */
void testScaleZeroGroupsAndEmptyTensorAreZeroState(void) {
    const int32_t codes[6] = {0, 0, 5, -3, 0, 0};
    const uint8_t stored[3] = {200, 130, 9};
    scInit(&fix, 6, 3, 2, 8, 8, SR_HALF_AWAY, codes, stored);
    rngStream_t s = rngOpen((rngKey_t){{3u, 3u}});
    scaleBfpTensorByScalarInPlace(&fix.t, bfpScalarFromCount(4, "t"), &s);
    TEST_ASSERT_EQUAL_UINT8(127, fix.exps[0]);
    TEST_ASSERT_EQUAL_UINT8(127, fix.exps[2]);
    int32_t got[6];
    scCodes(&fix, got);
    TEST_ASSERT_EQUAL_INT32(0, got[0]);
    TEST_ASSERT_EQUAL_INT32(0, got[5]);

    scInit(&fix, 0, 1, 0, 8, 8, SR_HALF_AWAY, NULL, (const uint8_t[]){42});
    scaleBfpTensorByScalarInPlace(&fix.t, bfpScalarFromCount(3, "t"), &s);
    TEST_ASSERT_EQUAL_UINT8(127, fix.exps[0]);
}

static void scaleAboveOneBody(void) {
    scInit(&fix, 2, 1, 0, 8, 8, HALF_AWAY, (const int32_t[]){1, 2}, (const uint8_t[]){127});
    scaleBfpTensorByScalarInPlace(&fix.t, (bfpScalar_t){.q = 16385, .e = -14}, NULL);
}
static void scaleTwoBody(void) {
    scInit(&fix, 2, 1, 0, 8, 8, HALF_AWAY, (const int32_t[]){1, 2}, (const uint8_t[]){127});
    scaleBfpTensorByScalarInPlace(&fix.t, (bfpScalar_t){.q = 16384, .e = -13}, NULL);
}
static void scaleZeroBody(void) {
    scInit(&fix, 2, 1, 0, 8, 8, HALF_AWAY, (const int32_t[]){1, 2}, (const uint8_t[]){127});
    scaleBfpTensorByScalarInPlace(&fix.t, (bfpScalar_t){.q = 0, .e = 0}, NULL);
}
static void scaleNegativeBody(void) {
    scInit(&fix, 2, 1, 0, 8, 8, HALF_AWAY, (const int32_t[]){1, 2}, (const uint8_t[]){127});
    scaleBfpTensorByScalarInPlace(&fix.t, (bfpScalar_t){.q = -16384, .e = -20}, NULL);
}
static void scaleNotNormalizedBody(void) {
    scInit(&fix, 2, 1, 0, 8, 8, HALF_AWAY, (const int32_t[]){1, 2}, (const uint8_t[]){127});
    scaleBfpTensorByScalarInPlace(&fix.t, (bfpScalar_t){.q = 3, .e = -10}, NULL);
}
static void scaleSrWithoutStreamBody(void) {
    scInit(&fix, 2, 1, 0, 8, 8, SR_HALF_AWAY, (const int32_t[]){1, 2}, (const uint8_t[]){127});
    scaleBfpTensorByScalarInPlace(&fix.t, bfpScalarFromCount(2, "t"), NULL);
}
static void scaleWrongDtypeBody(void) {
    scInit(&fix, 2, 1, 0, 8, 8, HALF_AWAY, (const int32_t[]){1, 2}, (const uint8_t[]){127});
    quantization_t fq;
    initFloat32Quantization(&fq);
    float vals[2] = {1.f, 2.f};
    tensor_t ft;
    setTensorValues(&ft, (uint8_t *)vals, &fix.shape, &fq, NULL);
    scaleBfpTensorByScalarInPlace(&ft, bfpScalarFromCount(2, "t"), NULL);
}

void testScaleRejectsBadInputs(void) {
    ASSERT_EXITS_WITH_FAILURE(scaleAboveOneBody());
    ASSERT_EXITS_WITH_FAILURE(scaleTwoBody());
    ASSERT_EXITS_WITH_FAILURE(scaleZeroBody());
    ASSERT_EXITS_WITH_FAILURE(scaleNegativeBody());
    ASSERT_EXITS_WITH_FAILURE(scaleNotNormalizedBody());
    ASSERT_EXITS_WITH_FAILURE(scaleSrWithoutStreamBody());
    ASSERT_EXITS_WITH_FAILURE(scaleWrongDtypeBody());
}

/* The float walker this replaces is exact for a power-of-two factor too:
 * on random tight and non-tight per-tensor grids both land on the same codes
 * and exponent, so a HAR run with B = 64 cannot tell them apart. */
void testScalePowerOfTwoAgreesWithTheFloatWalker(void) {
    static int32_t codes[700];
    static int32_t gotInt[700];
    static int32_t gotFloat[700];
    for (int round = 0; round < 200; round++) {
        int top = testRandInt(1, 7);
        for (size_t i = 0; i < 700; i++) {
            codes[i] = testRandInt(-(1 << top) + 1, (1 << top) - 1);
        }
        if (round % 3 == 0) {
            codes[testRandInt(0, 699)] = -128;
        }
        /* stored <= 200: near the top of the float range the walker's
         * mant * 2^E overflows to inf and saturates the exponent -- the
         * integer path stays exact there, the two are compared below it */
        const uint8_t stored[1] = {(uint8_t)testRandInt(20, 200)};
        const unsigned k = (unsigned)testRandInt(0, 10);
        scInit(&fix, 700, 1, 0, 8, 8, SR_HALF_AWAY, codes, stored);
        rngStream_t s1 = rngOpen((rngKey_t){{9u, (uint32_t)round}});
        scaleBfpTensorByScalarInPlace(&fix.t, bfpScalarFromCount((size_t)1 << k, "t"), &s1);
        scCodes(&fix, gotInt);
        const uint8_t eInt = fix.exps[0];
        scInit(&fix, 700, 1, 0, 8, 8, SR_HALF_AWAY, codes, stored);
        rngStream_t s2 = rngOpen((rngKey_t){{9u, (uint32_t)round}});
        scaleBfpTensorInPlace(&fix.t, ldexpf(1.f, -(int)k), &s2);
        scCodes(&fix, gotFloat);
        TEST_ASSERT_EQUAL_UINT8(fix.exps[0], eInt);
        TEST_ASSERT_EQUAL_INT32_ARRAY(gotFloat, gotInt, 700);
    }
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testScalarFromCountPowersOfTwoAreExact);
    RUN_TEST(testScalarFromCountMatchesExactRounding);
    RUN_TEST(testScalarFromCountRejectsZeroAndHuge);
    RUN_TEST(testScaleMatchesExactReferenceOnRandomTargets);
    RUN_TEST(testScalePowerOfTwoIsAnExactExponentShiftWithoutDraws);
    RUN_TEST(testScaleAtTheExponentFloorRoundsTheShiftedCodes);
    RUN_TEST(testScaleNonPowerOfTwoIsUnbiasedUnderSr);
    RUN_TEST(testScaleZeroGroupsAndEmptyTensorAreZeroState);
    RUN_TEST(testScaleRejectsBadInputs);
    RUN_TEST(testScalePowerOfTwoAgreesWithTheFloatWalker);
    return UNITY_END();
}
