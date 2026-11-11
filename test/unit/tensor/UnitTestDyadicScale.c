#include "DeathTest.h"
#include "DyadicScale.h"
#include "unity.h"

#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

/* Every expected value is computed HERE, independently of the integer
 * machinery under test: FromFloat and Mul against a double reference (the
 * operands and the 30-bit product are exact in double), MulRatio against an
 * exact uint64 quotient/remainder reference (long double is only 64 bits wide
 * on arm64 macOS, so it is not exact for a 46-bit numerator over a 31-bit
 * divisor). */

void setUp(void) {}
void tearDown(void) {}

#define SCALE_E_MIN (-140)
#define SCALE_E_MAX 113
#define SWEEP_COUNT 1000000u

static const odtScaleRound_t allModes[] = {ODT_SCALE_NEAREST, ODT_SCALE_CEIL, ODT_SCALE_FLOOR};
static const char *const modeNames[] = {"NEAREST", "CEIL", "FLOOR"};

/* File-scope static initializers: the constants must be usable in static
 * tables, not only in expressions. */
static const odtScale_t staticTable[] = {ODT_SCALE_ONE, ODT_SCALE_POW2(-126), ODT_SCALE_POW2(127),
                                         ODT_SCALE_POW2(0)};

static uint32_t testRngState = 0x5EED1234u;

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

static float floatFromBits(uint32_t bits) {
    float x;
    memcpy(&x, &bits, sizeof x);
    return x;
}

static odtScale_t mk(uint16_t m, int16_t e) {
    odtScale_t s = {.m = m, .e = e};
    return s;
}

/* Unbounded result of the reference rounding: e may lie outside the
 * admissible range, which the caller then expects to fail fast. */
typedef struct {
    int32_t m;
    int32_t e;
} refScale_t;

/* q in [2^14, 2^15) is the truncated mantissa; aboveZero / atOrAboveHalf
 * describe the discarded fraction. */
static refScale_t refApplyRounding(uint32_t q, bool aboveZero, bool atOrAboveHalf, int32_t e,
                                   odtScaleRound_t r) {
    uint32_t m = q;
    if ((r == ODT_SCALE_NEAREST && atOrAboveHalf) || (r == ODT_SCALE_CEIL && aboveZero)) {
        m++;
    }
    if (m == 32768u) {
        m = 16384u;
        e++;
    }
    refScale_t out = {(int32_t)m, e};
    return out;
}

/* v > 0 exactly representable in double: round to 15 significant bits. */
static refScale_t refRoundDouble(double v, odtScaleRound_t r) {
    int k = 0;
    double f = frexp(v, &k); /* v = f * 2^k, f in [0.5, 1) */
    double t = f * 32768.0;  /* in [16384, 32768), exact */
    uint32_t q = (uint32_t)t;
    double frac = t - (double)q;
    return refApplyRounding(q, frac > 0.0, frac >= 0.5, k - 15, r);
}

/* value = p / den * 2^eBase, p >= 1, den >= 1, p < 2^47, den <= 2^31. */
static refScale_t refRoundRatio(uint64_t p, uint64_t den, int32_t eBase, odtScaleRound_t r) {
    uint64_t n = p;
    uint64_t d = den;
    int32_t t = 0; /* value = (n / d) * 2^(eBase - t) */
    while (n / d < 16384u) {
        n <<= 1;
        t++;
    }
    while (n / d >= 32768u) {
        d <<= 1;
        t--;
    }
    uint64_t q = n / d;
    uint64_t rem = n % d;
    return refApplyRounding((uint32_t)q, rem != 0, 2 * rem >= d, eBase - t, r);
}

static bool refInRange(refScale_t s) {
    return s.e >= SCALE_E_MIN && s.e <= SCALE_E_MAX;
}

static void assertScaleEqualsRef(refScale_t want, odtScale_t got, const char *what) {
    if ((int32_t)got.m != want.m || (int32_t)got.e != want.e) {
        char msg[256];
        snprintf(msg, sizeof msg, "%s: want {%d, %d}, got {%u, %d}", what, (int)want.m, (int)want.e,
                 (unsigned)got.m, (int)got.e);
        TEST_FAIL_MESSAGE(msg);
    }
}

void testScaleFromFloatRoundTripExactForDyadic(void) {
    for (int32_t e = SCALE_E_MIN; e <= SCALE_E_MAX; e++) {
        for (uint32_t m = 16384u; m <= 32767u; m++) {
            /* Every mantissa at the boundary and central exponents, a stride elsewhere. */
            bool fullRow = e == SCALE_E_MIN || e == SCALE_E_MAX || e == -14 || e == 0;
            if (!fullRow && (m % 97u) != 0 && m != 16384u && m != 32767u) {
                continue;
            }
            float x = ldexpf((float)m, (int)e);
            for (size_t i = 0; i < 3; i++) {
                odtScale_t s = odtScaleFromFloat(x, allModes[i]);
                TEST_ASSERT_EQUAL_UINT16(m, s.m);
                TEST_ASSERT_EQUAL_INT16(e, s.e);
                TEST_ASSERT_TRUE(odtScaleToFloat(s) == x);
                TEST_ASSERT_TRUE(odtScaleCanonFloat(x, allModes[i]) == x);
            }
        }
    }
}

void testScaleFromFloatRoundingModes(void) {
    /* Directed: half-way, just below and just above half in the dropped 9 bits. */
    {
        /* m0 = 16384, dropped bits 0x100 (exact half): NEAREST ties away. */
        float x = floatFromBits((127u << 23) | 0x100u);
        TEST_ASSERT_TRUE(odtScaleEq(mk(16385, -14), odtScaleFromFloat(x, ODT_SCALE_NEAREST)));
        TEST_ASSERT_TRUE(odtScaleEq(mk(16385, -14), odtScaleFromFloat(x, ODT_SCALE_CEIL)));
        TEST_ASSERT_TRUE(odtScaleEq(mk(16384, -14), odtScaleFromFloat(x, ODT_SCALE_FLOOR)));
        float below = floatFromBits((127u << 23) | 0x0FFu);
        TEST_ASSERT_TRUE(odtScaleEq(mk(16384, -14), odtScaleFromFloat(below, ODT_SCALE_NEAREST)));
        TEST_ASSERT_TRUE(odtScaleEq(mk(16385, -14), odtScaleFromFloat(below, ODT_SCALE_CEIL)));
        float lowest = floatFromBits((127u << 23) | 0x001u);
        TEST_ASSERT_TRUE(odtScaleEq(mk(16385, -14), odtScaleFromFloat(lowest, ODT_SCALE_CEIL)));
        TEST_ASSERT_TRUE(odtScaleEq(mk(16384, -14), odtScaleFromFloat(lowest, ODT_SCALE_NEAREST)));
        /* Carry out of m = 32767: 2 - 2^-23 rounds up to 2. */
        float carry = floatFromBits((127u << 23) | 0x7FFFFFu);
        TEST_ASSERT_TRUE(odtScaleEq(mk(16384, -13), odtScaleFromFloat(carry, ODT_SCALE_NEAREST)));
        TEST_ASSERT_TRUE(odtScaleEq(mk(16384, -13), odtScaleFromFloat(carry, ODT_SCALE_CEIL)));
        TEST_ASSERT_TRUE(odtScaleEq(mk(32767, -14), odtScaleFromFloat(carry, ODT_SCALE_FLOOR)));
        /* Range ends: FLT_MIN is admissible, FLT_MAX only when rounded down. */
        TEST_ASSERT_TRUE(
            odtScaleEq(mk(16384, SCALE_E_MIN), odtScaleFromFloat(FLT_MIN, ODT_SCALE_NEAREST)));
        TEST_ASSERT_TRUE(
            odtScaleEq(mk(32767, SCALE_E_MAX), odtScaleFromFloat(FLT_MAX, ODT_SCALE_FLOOR)));
        /* Just above FLT_MIN with CEIL stays in the lowest binade. */
        float aboveMin = floatFromBits((1u << 23) | 1u);
        TEST_ASSERT_TRUE(
            odtScaleEq(mk(16385, SCALE_E_MIN), odtScaleFromFloat(aboveMin, ODT_SCALE_CEIL)));
    }

    uint32_t checked = 0;
    for (uint32_t i = 0; i < SWEEP_COUNT; i++) {
        uint32_t biased = (uint32_t)testRandInt(1, 254);
        uint32_t bits = (biased << 23) | (testRand() & 0x7FFFFFu);
        float x = floatFromBits(bits);
        double xd = (double)x;
        for (size_t k = 0; k < 3; k++) {
            refScale_t want = refRoundDouble(xd, allModes[k]);
            if (!refInRange(want)) {
                continue; /* top-binade carry past FLT_MAX: covered by the death test */
            }
            odtScale_t got = odtScaleFromFloat(x, allModes[k]);
            char what[96];
            snprintf(what, sizeof what, "FromFloat(0x%08x, %s)", (unsigned)bits, modeNames[k]);
            assertScaleEqualsRef(want, got, what);
            double back = ldexp((double)got.m, got.e);
            double rel = fabs(back - xd) / xd;
            if (allModes[k] == ODT_SCALE_NEAREST) {
                TEST_ASSERT_TRUE_MESSAGE(rel <= ldexp(1.0, -15), what);
            } else if (allModes[k] == ODT_SCALE_CEIL) {
                TEST_ASSERT_TRUE_MESSAGE(back >= xd && rel < ldexp(1.0, -14), what);
            } else {
                TEST_ASSERT_TRUE_MESSAGE(back <= xd && rel < ldexp(1.0, -14), what);
            }
            TEST_ASSERT_TRUE_MESSAGE((double)odtScaleCanonFloat(x, allModes[k]) == back, what);
            checked++;
        }
    }
    TEST_ASSERT_TRUE(checked > 3u * SWEEP_COUNT - 100u);
}

void testScaleMulMatchesExactReference(void) {
    /* Double rounding counterexample: the integer product rounds once to
     * 24561; via a float product it would round twice to 24562. */
    odtScale_t p = odtScaleMul(mk(16385, -14), mk(24560, -14), ODT_SCALE_NEAREST);
    TEST_ASSERT_TRUE(odtScaleEq(mk(24561, -14), p));
    TEST_ASSERT_TRUE(
        odtScaleEq(mk(24562, -14), odtScaleMul(mk(16385, -14), mk(24560, -14), ODT_SCALE_CEIL)));
    TEST_ASSERT_TRUE(
        odtScaleEq(mk(24561, -14), odtScaleMul(mk(16385, -14), mk(24560, -14), ODT_SCALE_FLOOR)));

    /* ONE is the identity in every mode. */
    for (size_t k = 0; k < 3; k++) {
        TEST_ASSERT_TRUE(
            odtScaleEq(mk(23456, 7), odtScaleMul(mk(23456, 7), ODT_SCALE_ONE, allModes[k])));
    }

    /* Carry at m = 32767: 16385 * 32766 = 2^29 - 2. */
    TEST_ASSERT_TRUE(
        odtScaleEq(mk(16384, 1), odtScaleMul(mk(16385, -7), mk(32766, -7), ODT_SCALE_NEAREST)));
    TEST_ASSERT_TRUE(
        odtScaleEq(mk(32767, 0), odtScaleMul(mk(16385, -7), mk(32766, -7), ODT_SCALE_FLOOR)));

    /* Exponent boundaries: a carry lifts a product just below 2^-126 into
     * range; FLOOR of the same product is out of range (death test). */
    TEST_ASSERT_TRUE(
        odtScaleEq(mk(16384, SCALE_E_MIN),
                   odtScaleMul(mk(16385, SCALE_E_MIN), mk(32766, -15), ODT_SCALE_NEAREST)));
    TEST_ASSERT_TRUE(
        odtScaleEq(mk(32767, SCALE_E_MAX),
                   odtScaleMul(mk(16385, SCALE_E_MAX), mk(32766, -14), ODT_SCALE_FLOOR)));
    TEST_ASSERT_TRUE(odtScaleEq(
        ODT_SCALE_POW2(127), odtScaleMul(ODT_SCALE_POW2(127), ODT_SCALE_ONE, ODT_SCALE_NEAREST)));
    TEST_ASSERT_TRUE(
        odtScaleEq(ODT_SCALE_POW2(-126),
                   odtScaleMul(ODT_SCALE_POW2(-63), ODT_SCALE_POW2(-63), ODT_SCALE_NEAREST)));

    uint32_t checked = 0;
    for (uint32_t i = 0; i < SWEEP_COUNT; i++) {
        odtScale_t a = mk((uint16_t)testRandInt(16384, 32767), (int16_t)testRandInt(-77, 49));
        odtScale_t b = mk((uint16_t)testRandInt(16384, 32767), (int16_t)testRandInt(-77, 49));
        double exact = ldexp((double)((uint32_t)a.m * (uint32_t)b.m), a.e + b.e);
        for (size_t k = 0; k < 3; k++) {
            refScale_t want = refRoundDouble(exact, allModes[k]);
            if (!refInRange(want)) {
                continue;
            }
            char what[96];
            snprintf(what, sizeof what, "Mul({%u,%d},{%u,%d}, %s)", (unsigned)a.m, (int)a.e,
                     (unsigned)b.m, (int)b.e, modeNames[k]);
            assertScaleEqualsRef(want, odtScaleMul(a, b, allModes[k]), what);
            checked++;
        }
    }
    TEST_ASSERT_TRUE(checked > 2u * SWEEP_COUNT);
}

static void checkMulRatio(odtScale_t s, uint32_t num, uint32_t den) {
    for (size_t k = 0; k < 3; k++) {
        refScale_t want = refRoundRatio((uint64_t)s.m * num, den, s.e, allModes[k]);
        TEST_ASSERT_TRUE_MESSAGE(refInRange(want), "directed MulRatio case must be in range");
        char what[128];
        snprintf(what, sizeof what, "MulRatio({%u,%d}, %lu, %lu, %s)", (unsigned)s.m, (int)s.e,
                 (unsigned long)num, (unsigned long)den, modeNames[k]);
        assertScaleEqualsRef(want, odtScaleMulRatio(s, num, den, allModes[k]), what);
    }
}

static uint32_t randRatioOperand(void) {
    uint32_t bits = (uint32_t)testRandInt(1, 32);
    if (bits == 32) {
        return 0x80000000u; /* the 2^31 endpoint, about 1 in 32 draws */
    }
    uint32_t v = testRand() & ((1u << bits) - 1u);
    return v == 0 ? 1u : v;
}

void testScaleMulRatioMatchesExactReference(void) {
    const uint32_t two31 = 0x80000000u;

    /* Reference self-check on exact cases. */
    TEST_ASSERT_EQUAL_INT32(24577, refRoundRatio(49155, 1, 0, ODT_SCALE_FLOOR).m);

    /* Directed values with known answers. */
    TEST_ASSERT_TRUE(
        odtScaleEq(mk(23456, 7), odtScaleMulRatio(mk(23456, 7), 1, 1, ODT_SCALE_NEAREST)));
    TEST_ASSERT_TRUE(
        odtScaleEq(mk(23456, 8), odtScaleMulRatio(mk(23456, 7), 2, 1, ODT_SCALE_FLOOR)));
    TEST_ASSERT_TRUE(
        odtScaleEq(mk(23456, 38), odtScaleMulRatio(mk(23456, 7), two31, 1, ODT_SCALE_FLOOR)));
    TEST_ASSERT_TRUE(
        odtScaleEq(mk(23456, -24), odtScaleMulRatio(mk(23456, 7), 1, two31, ODT_SCALE_CEIL)));
    TEST_ASSERT_TRUE(
        odtScaleEq(mk(23456, 7), odtScaleMulRatio(mk(23456, 7), two31, two31, ODT_SCALE_CEIL)));
    /* Exact tie through a non-trivial divisor: 16385 * 15 / 5 = 49155 = 24577.5 * 2. */
    TEST_ASSERT_TRUE(
        odtScaleEq(mk(24578, 1), odtScaleMulRatio(mk(16385, 0), 15, 5, ODT_SCALE_NEAREST)));
    TEST_ASSERT_TRUE(
        odtScaleEq(mk(24578, 1), odtScaleMulRatio(mk(16385, 0), 15, 5, ODT_SCALE_CEIL)));
    TEST_ASSERT_TRUE(
        odtScaleEq(mk(24577, 1), odtScaleMulRatio(mk(16385, 0), 15, 5, ODT_SCALE_FLOOR)));
    /* Just below that tie (sticky set, round bit clear): NEAREST down, CEIL up. */
    {
        uint32_t num = 15u * (1u << 20) - 1u;
        uint32_t den = 5u * (1u << 20);
        TEST_ASSERT_TRUE(
            odtScaleEq(mk(24577, 1), odtScaleMulRatio(mk(16385, 0), num, den, ODT_SCALE_NEAREST)));
        TEST_ASSERT_TRUE(
            odtScaleEq(mk(24578, 1), odtScaleMulRatio(mk(16385, 0), num, den, ODT_SCALE_CEIL)));
        TEST_ASSERT_TRUE(
            odtScaleEq(mk(24577, 1), odtScaleMulRatio(mk(16385, 0), num, den, ODT_SCALE_FLOOR)));
    }
    /* Just above it: every mode but FLOOR goes up. */
    {
        uint32_t num = 15u * (1u << 20) + 1u;
        uint32_t den = 5u * (1u << 20);
        TEST_ASSERT_TRUE(
            odtScaleEq(mk(24578, 1), odtScaleMulRatio(mk(16385, 0), num, den, ODT_SCALE_NEAREST)));
        TEST_ASSERT_TRUE(
            odtScaleEq(mk(24577, 1), odtScaleMulRatio(mk(16385, 0), num, den, ODT_SCALE_FLOOR)));
    }
    /* Carry at m = 32767: 32767 * 131071 / 131068 = 32767.75. */
    TEST_ASSERT_TRUE(odtScaleEq(mk(16384, 1),
                                odtScaleMulRatio(mk(32767, 0), 131071, 131068, ODT_SCALE_NEAREST)));
    TEST_ASSERT_TRUE(
        odtScaleEq(mk(32767, 0), odtScaleMulRatio(mk(32767, 0), 131071, 131068, ODT_SCALE_FLOOR)));
    /* CEIL carry on the smallest excess: 32767 * 2^31 / (2^31 - 1). */
    TEST_ASSERT_TRUE(odtScaleEq(mk(16384, 1),
                                odtScaleMulRatio(mk(32767, 0), two31, two31 - 1u, ODT_SCALE_CEIL)));
    TEST_ASSERT_TRUE(odtScaleEq(
        mk(32767, 0), odtScaleMulRatio(mk(32767, 0), two31, two31 - 1u, ODT_SCALE_NEAREST)));
    /* Exponent boundaries, in range. */
    TEST_ASSERT_TRUE(
        odtScaleEq(mk(16384, SCALE_E_MAX),
                   odtScaleMulRatio(mk(16384, SCALE_E_MAX - 31), two31, 1, ODT_SCALE_NEAREST)));
    TEST_ASSERT_TRUE(
        odtScaleEq(mk(16384, SCALE_E_MIN),
                   odtScaleMulRatio(mk(16384, SCALE_E_MIN + 31), 1, two31, ODT_SCALE_NEAREST)));
    /* Just below 2^-126: NEAREST and CEIL round up into range. */
    TEST_ASSERT_TRUE(
        odtScaleEq(mk(16384, SCALE_E_MIN),
                   odtScaleMulRatio(mk(16384, SCALE_E_MIN), two31 - 1u, two31, ODT_SCALE_NEAREST)));
    TEST_ASSERT_TRUE(
        odtScaleEq(mk(16384, SCALE_E_MIN),
                   odtScaleMulRatio(mk(16384, SCALE_E_MIN), two31 - 1u, two31, ODT_SCALE_CEIL)));

    /* Directed operands through the reference. */
    const uint32_t ops[] = {1u, 2u, 3u, 5u, 7u, 0xFFFFu, 0x10000u, 0x10001u, 0x7FFFFFFFu, two31};
    const uint16_t ms[] = {16384u, 16385u, 24560u, 32766u, 32767u};
    for (size_t a = 0; a < sizeof ops / sizeof ops[0]; a++) {
        for (size_t b = 0; b < sizeof ops / sizeof ops[0]; b++) {
            for (size_t c = 0; c < sizeof ms / sizeof ms[0]; c++) {
                checkMulRatio(mk(ms[c], 0), ops[a], ops[b]);
            }
        }
    }

    uint32_t checked = 0;
    for (uint32_t i = 0; i < SWEEP_COUNT; i++) {
        odtScale_t s =
            mk((uint16_t)testRandInt(16384, 32767), (int16_t)testRandInt(SCALE_E_MIN, SCALE_E_MAX));
        uint32_t num = randRatioOperand();
        uint32_t den = randRatioOperand();
        for (size_t k = 0; k < 3; k++) {
            refScale_t want = refRoundRatio((uint64_t)s.m * num, den, s.e, allModes[k]);
            if (!refInRange(want)) {
                continue;
            }
            char what[128];
            snprintf(what, sizeof what, "MulRatio({%u,%d}, %lu, %lu, %s)", (unsigned)s.m, (int)s.e,
                     (unsigned long)num, (unsigned long)den, modeNames[k]);
            assertScaleEqualsRef(want, odtScaleMulRatio(s, num, den, allModes[k]), what);
            checked++;
        }
    }
    TEST_ASSERT_TRUE(checked > 2u * SWEEP_COUNT);
}

void testScaleCmpOrdersByValue(void) {
    TEST_ASSERT_EQUAL_INT(0, odtScaleCmp(mk(20000, 3), mk(20000, 3)));
    TEST_ASSERT_EQUAL_INT(-1, odtScaleCmp(mk(20000, 3), mk(20001, 3)));
    TEST_ASSERT_EQUAL_INT(1, odtScaleCmp(mk(20001, 3), mk(20000, 3)));
    /* A larger exponent wins even against the largest mantissa below it. */
    TEST_ASSERT_EQUAL_INT(-1, odtScaleCmp(mk(32767, 3), mk(16384, 4)));
    TEST_ASSERT_EQUAL_INT(1, odtScaleCmp(mk(16384, 4), mk(32767, 3)));
    TEST_ASSERT_EQUAL_INT(-1, odtScaleCmp(mk(32767, SCALE_E_MIN), mk(16384, SCALE_E_MAX)));
    TEST_ASSERT_EQUAL_INT(1, odtScaleCmp(mk(16384, SCALE_E_MAX), mk(32767, SCALE_E_MIN)));
    TEST_ASSERT_TRUE(odtScaleEq(mk(20000, 3), mk(20000, 3)));
    TEST_ASSERT_FALSE(odtScaleEq(mk(20000, 3), mk(20000, 4)));
    TEST_ASSERT_FALSE(odtScaleEq(mk(20000, 3), mk(20001, 3)));

    for (uint32_t i = 0; i < 200000u; i++) {
        odtScale_t a = mk((uint16_t)testRandInt(16384, 32767), (int16_t)testRandInt(-6, 6));
        odtScale_t b =
            (i & 1u) ? a : mk((uint16_t)testRandInt(16384, 32767), (int16_t)testRandInt(-6, 6));
        float fa = odtScaleToFloat(a);
        float fb = odtScaleToFloat(b);
        int want = (fa < fb) ? -1 : (fa > fb) ? 1 : 0;
        TEST_ASSERT_EQUAL_INT(want, odtScaleCmp(a, b));
        TEST_ASSERT_EQUAL_INT(-want, odtScaleCmp(b, a));
        TEST_ASSERT_EQUAL(want == 0, odtScaleEq(a, b));
    }
}

void testScaleRejectsInvalid(void) {
    /* FromFloat: only positive normal floats whose rounding stays <= FLT_MAX. */
    ASSERT_EXITS_WITH_OUTPUT(1, "not a positive normal float",
                             (void)odtScaleFromFloat(0.0f, ODT_SCALE_NEAREST));
    ASSERT_EXITS_WITH_OUTPUT(1, "not a positive normal float",
                             (void)odtScaleFromFloat(-0.0f, ODT_SCALE_CEIL));
    ASSERT_EXITS_WITH_OUTPUT(1, "not a positive normal float",
                             (void)odtScaleFromFloat(-1.0f, ODT_SCALE_NEAREST));
    ASSERT_EXITS_WITH_OUTPUT(
        1, "not a positive normal float",
        (void)odtScaleFromFloat(floatFromBits(0x7FC00000u), ODT_SCALE_NEAREST));
    ASSERT_EXITS_WITH_OUTPUT(1, "not a positive normal float",
                             (void)odtScaleFromFloat(floatFromBits(0x7F800000u), ODT_SCALE_FLOOR));
    ASSERT_EXITS_WITH_OUTPUT(
        1, "not a positive normal float",
        (void)odtScaleFromFloat(floatFromBits(0xFF800000u), ODT_SCALE_NEAREST));
    ASSERT_EXITS_WITH_OUTPUT(
        1, "not a positive normal float",
        (void)odtScaleFromFloat(floatFromBits(0x00400000u), ODT_SCALE_CEIL)); /* subnormal */
    ASSERT_EXITS_WITH_OUTPUT(1, "not a positive normal float",
                             (void)odtScaleFromFloat(floatFromBits(0x007FFFFFu),
                                                     ODT_SCALE_CEIL)); /* largest subnormal */
    ASSERT_EXITS_WITH_OUTPUT(
        1, "not a positive normal float",
        (void)odtScaleFromFloat(floatFromBits(0x00000001u), ODT_SCALE_CEIL)); /* 2^-149 */
    ASSERT_EXITS_WITH_OUTPUT(1, "outside the admissible range",
                             (void)odtScaleFromFloat(FLT_MAX, ODT_SCALE_NEAREST));
    ASSERT_EXITS_WITH_OUTPUT(1, "outside the admissible range",
                             (void)odtScaleFromFloat(FLT_MAX, ODT_SCALE_CEIL));
    ASSERT_EXITS_WITH_OUTPUT(1, "not a positive normal float",
                             (void)odtScaleCanonFloat(0.0f, ODT_SCALE_NEAREST));

    /* Non-canonical operands, e.g. a zero-initialised scale. */
    ASSERT_EXITS_WITH_OUTPUT(1, "not a canonical scale", (void)odtScaleToFloat(mk(0, 0)));
    ASSERT_EXITS_WITH_OUTPUT(1, "not a canonical scale", (void)odtScaleToFloat(mk(16383, 0)));
    ASSERT_EXITS_WITH_OUTPUT(1, "not a canonical scale", (void)odtScaleToFloat(mk(32768, 0)));
    ASSERT_EXITS_WITH_OUTPUT(1, "not a canonical scale",
                             (void)odtScaleToFloat(mk(16384, SCALE_E_MAX + 1)));
    ASSERT_EXITS_WITH_OUTPUT(1, "not a canonical scale",
                             (void)odtScaleToFloat(mk(32767, SCALE_E_MIN - 1)));
    ASSERT_EXITS_WITH_OUTPUT(1, "not a canonical scale",
                             (void)odtScaleMul(mk(0, 0), ODT_SCALE_ONE, ODT_SCALE_NEAREST));
    ASSERT_EXITS_WITH_OUTPUT(1, "not a canonical scale",
                             (void)odtScaleMul(ODT_SCALE_ONE, mk(0, 0), ODT_SCALE_NEAREST));
    ASSERT_EXITS_WITH_OUTPUT(1, "not a canonical scale",
                             (void)odtScaleMulRatio(mk(0, 0), 1, 1, ODT_SCALE_NEAREST));
    ASSERT_EXITS_WITH_OUTPUT(1, "not a canonical scale",
                             (void)odtScaleCmp(mk(0, 0), ODT_SCALE_ONE));
    ASSERT_EXITS_WITH_OUTPUT(1, "not a canonical scale",
                             (void)odtScaleCmp(ODT_SCALE_ONE, mk(16384, -200)));
    ASSERT_EXITS_WITH_OUTPUT(1, "not a canonical scale", (void)odtScaleEq(mk(0, 0), mk(0, 0)));

    /* Mul: results outside the range, including a FLOOR just below 2^-126 and
     * a NEAREST carry past FLT_MAX. */
    ASSERT_EXITS_WITH_OUTPUT(
        1, "outside the admissible range",
        (void)odtScaleMul(ODT_SCALE_POW2(127), ODT_SCALE_POW2(1), ODT_SCALE_NEAREST));
    ASSERT_EXITS_WITH_OUTPUT(
        1, "outside the admissible range",
        (void)odtScaleMul(ODT_SCALE_POW2(-126), ODT_SCALE_POW2(-1), ODT_SCALE_NEAREST));
    ASSERT_EXITS_WITH_OUTPUT(
        1, "outside the admissible range",
        (void)odtScaleMul(mk(16385, SCALE_E_MIN), mk(32766, -15), ODT_SCALE_FLOOR));
    ASSERT_EXITS_WITH_OUTPUT(
        1, "outside the admissible range",
        (void)odtScaleMul(mk(16385, SCALE_E_MAX), mk(32766, -14), ODT_SCALE_NEAREST));
    ASSERT_EXITS_WITH_OUTPUT(
        1, "outside the admissible range",
        (void)odtScaleMul(mk(32767, SCALE_E_MAX), mk(32767, SCALE_E_MAX), ODT_SCALE_FLOOR));
    ASSERT_EXITS_WITH_OUTPUT(
        1, "outside the admissible range",
        (void)odtScaleMul(mk(16384, SCALE_E_MIN), mk(16384, SCALE_E_MIN), ODT_SCALE_CEIL));

    /* MulRatio: num and den in [1, 2^31]; results outside the range. */
    ASSERT_EXITS_WITH_OUTPUT(1, "num",
                             (void)odtScaleMulRatio(ODT_SCALE_ONE, 0, 1, ODT_SCALE_NEAREST));
    ASSERT_EXITS_WITH_OUTPUT(
        1, "num", (void)odtScaleMulRatio(ODT_SCALE_ONE, 0x80000001u, 1, ODT_SCALE_NEAREST));
    ASSERT_EXITS_WITH_OUTPUT(
        1, "num", (void)odtScaleMulRatio(ODT_SCALE_ONE, UINT32_MAX, 1, ODT_SCALE_NEAREST));
    ASSERT_EXITS_WITH_OUTPUT(1, "den",
                             (void)odtScaleMulRatio(ODT_SCALE_ONE, 1, 0, ODT_SCALE_NEAREST));
    ASSERT_EXITS_WITH_OUTPUT(
        1, "den", (void)odtScaleMulRatio(ODT_SCALE_ONE, 1, 0x80000001u, ODT_SCALE_NEAREST));
    ASSERT_EXITS_WITH_OUTPUT(
        1, "outside the admissible range",
        (void)odtScaleMulRatio(mk(16384, SCALE_E_MAX), 2, 1, ODT_SCALE_NEAREST));
    ASSERT_EXITS_WITH_OUTPUT(
        1, "outside the admissible range",
        (void)odtScaleMulRatio(mk(32767, SCALE_E_MAX), 0x80000000u, 1, ODT_SCALE_FLOOR));
    ASSERT_EXITS_WITH_OUTPUT(1, "outside the admissible range",
                             (void)odtScaleMulRatio(mk(16384, SCALE_E_MIN), 1, 2, ODT_SCALE_CEIL));
    ASSERT_EXITS_WITH_OUTPUT(
        1, "outside the admissible range",
        (void)odtScaleMulRatio(mk(16384, SCALE_E_MIN), 1, 0x80000000u, ODT_SCALE_NEAREST));
    ASSERT_EXITS_WITH_OUTPUT(
        1, "outside the admissible range",
        (void)odtScaleMulRatio(mk(16384, SCALE_E_MIN), 0x7FFFFFFFu, 0x80000000u, ODT_SCALE_FLOOR));
    ASSERT_EXITS_WITH_OUTPUT(
        1, "outside the admissible range",
        (void)odtScaleMulRatio(mk(32767, SCALE_E_MAX), 131071, 131068, ODT_SCALE_NEAREST));
}

void testScaleConstants(void) {
    TEST_ASSERT_EQUAL_size_t(4, sizeof(odtScale_t));
    TEST_ASSERT_EQUAL_size_t(4, _Alignof(odtScale_t));

    TEST_ASSERT_EQUAL_UINT16(16384, ODT_SCALE_ONE.m);
    TEST_ASSERT_EQUAL_INT16(-14, ODT_SCALE_ONE.e);
    TEST_ASSERT_TRUE(odtScaleToFloat(ODT_SCALE_ONE) == 1.0f);
    for (size_t k = 0; k < 3; k++) {
        TEST_ASSERT_TRUE(odtScaleEq(ODT_SCALE_ONE, odtScaleFromFloat(1.0f, allModes[k])));
    }

    TEST_ASSERT_TRUE(odtScaleEq(ODT_SCALE_ONE, staticTable[0]));
    TEST_ASSERT_TRUE(odtScaleToFloat(staticTable[1]) == FLT_MIN);
    TEST_ASSERT_TRUE(odtScaleToFloat(staticTable[2]) == ldexpf(1.0f, 127));
    TEST_ASSERT_TRUE(odtScaleEq(ODT_SCALE_ONE, staticTable[3]));

    for (int k = -126; k <= 127; k++) {
        odtScale_t p = ODT_SCALE_POW2(k);
        TEST_ASSERT_EQUAL_UINT16(16384, p.m);
        TEST_ASSERT_EQUAL_INT16(k - 14, p.e);
        TEST_ASSERT_TRUE(odtScaleToFloat(p) == ldexpf(1.0f, k));
        TEST_ASSERT_TRUE(odtScaleEq(p, odtScaleFromFloat(ldexpf(1.0f, k), ODT_SCALE_NEAREST)));
    }

    /* Store/Load copy the whole value and touch only the target slot. */
    odtScale_t slots[3] = {ODT_SCALE_ONE, ODT_SCALE_ONE, ODT_SCALE_ONE};
    odtScaleStore(&slots[1], mk(30001, -77));
    TEST_ASSERT_TRUE(odtScaleEq(mk(30001, -77), odtScaleLoad(&slots[1])));
    TEST_ASSERT_TRUE(odtScaleEq(ODT_SCALE_ONE, odtScaleLoad(&slots[0])));
    TEST_ASSERT_TRUE(odtScaleEq(ODT_SCALE_ONE, odtScaleLoad(&slots[2])));
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testScaleFromFloatRoundTripExactForDyadic);
    RUN_TEST(testScaleFromFloatRoundingModes);
    RUN_TEST(testScaleMulMatchesExactReference);
    RUN_TEST(testScaleMulRatioMatchesExactReference);
    RUN_TEST(testScaleCmpOrdersByValue);
    RUN_TEST(testScaleRejectsInvalid);
    RUN_TEST(testScaleConstants);
    return UNITY_END();
}
