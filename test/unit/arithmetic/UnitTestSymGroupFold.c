#define SOURCE_FILE "UNIT_TEST_SYM_GROUP_FOLD"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "DeathTest.h"
#include "DyadicScale.h"
#include "Quantization.h"
#include "RNG.h"
#include "Rounding.h"
#include "SymGroupFold.h"
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

_Static_assert(_Generic(&symMulShiftRound,
                   int32_t (*)(int32_t, uint16_t, uint32_t, roundingMode_t, rngStream_t *): 1,
                   default: 0),
               "symMulShiftRound signature");
_Static_assert(_Generic(&symGroupFoldInit,
                   symGroupFold_t (*)(const symQConfig_t *, odtScale_t, uint8_t, size_t, int32_t,
                                      int32_t, uint64_t): 1,
                   default: 0),
               "symGroupFoldInit signature");

/* ---- int64 references (test-only; the kernel is int32-only) ---- */

static uint32_t clampShift(uint32_t s) {
    return s > 63u ? 63u : s;
}

/* round half away from zero of v * m * 2^-s */
static int64_t refHalfAway(int32_t v, uint16_t m, uint32_t s) {
    s = clampShift(s);
    int64_t p = (int64_t)v * m;
    if (s == 0) {
        return p;
    }
    int64_t mag = p < 0 ? -p : p;
    int64_t r = (mag + ((int64_t)1 << (s - 1u))) >> s;
    return p < 0 ? -r : r;
}

/* floor(v * m * 2^-s) and the top 32 bits of the dropped s-bit fraction */
static void refFloorFrac(int32_t v, uint16_t m, uint32_t s, int64_t *floorOut, uint32_t *frac32) {
    s = clampShift(s);
    int64_t p = (int64_t)v * m;
    *floorOut = p >> s; /* arithmetic shift = floor */
    uint64_t rem = s == 0 ? 0u : ((uint64_t)p & (((uint64_t)1 << s) - 1u));
    *frac32 = s <= 32u ? (uint32_t)(rem << (32u - s)) : (uint32_t)(rem >> (s - 32u));
}

static bool fitsInt32(int64_t x) {
    return x >= INT32_MIN && x <= INT32_MAX;
}

/* deterministic generator for the triples (independent of the SR stream) */
static uint32_t lcgState = 0x12345678u;
static uint32_t lcgNext(void) {
    lcgState = lcgState * 1664525u + 1013904223u;
    return lcgState;
}

static int32_t pickV(void) {
    static const int32_t edges[] = {INT32_MIN,
                                    INT32_MIN + 1,
                                    INT32_MAX,
                                    INT32_MAX - 1,
                                    0,
                                    1,
                                    -1,
                                    2,
                                    -2,
                                    65535,
                                    -65536,
                                    65536,
                                    32767,
                                    -32768,
                                    (1 << 30) + 7,
                                    -(1 << 30) - 3};
    uint32_t pick = lcgNext() % 4u;
    if (pick == 0u) {
        return edges[lcgNext() % (sizeof edges / sizeof edges[0])];
    }
    if (pick == 1u) {
        return (int32_t)(lcgNext() % 4096u) - 2048; /* small */
    }
    return (int32_t)lcgNext(); /* full range */
}

static uint16_t pickM(void) {
    static const uint16_t ms[] = {0, 1, 16384, 32767, 32768};
    uint32_t pick = lcgNext() % 6u;
    if (pick < 5u) {
        return ms[pick];
    }
    return (uint16_t)(lcgNext() % 32769u);
}

#define N_TRIPLES 250000

void testMulShiftRoundMatchesInt64ReferenceHalfAway(void) {
    lcgState = 0x12345678u;
    size_t checked = 0;
    for (size_t i = 0; i < N_TRIPLES; i++) {
        int32_t v = pickV();
        uint16_t m = pickM();
        uint32_t s = lcgNext() % 64u;
        int64_t ref = refHalfAway(v, m, s);
        if (!fitsInt32(ref)) {
            continue;
        }
        int32_t got = symMulShiftRound(v, m, s, HALF_AWAY, NULL);
        if ((int64_t)got != ref) {
            char msg[128];
            snprintf(msg, sizeof msg, "v=%ld m=%u s=%u", (long)v, (unsigned)m, (unsigned)s);
            TEST_ASSERT_EQUAL_INT64_MESSAGE(ref, (int64_t)got, msg);
        }
        checked++;
    }
    TEST_ASSERT_GREATER_THAN_size_t(200000, checked);

    /* review M1: the result is INT32_MIN while the floor is INT32_MIN - 1, so the
     * lift must not be a signed int32 add (UB). v*m / 2^14 = -2^31 - 0.000488. */
    TEST_ASSERT_EQUAL_INT64(INT32_MIN, refHalfAway(-2147352584, 16385, 14u));
    TEST_ASSERT_EQUAL_INT32(INT32_MIN, symMulShiftRound(-2147352584, 16385, 14u, HALF_AWAY, NULL));
}

/* review M1 under SR: floor = INT32_MIN - 1 is not an int32, so only the lifted
 * outcome (INT32_MIN) is in contract. The seed is pinned to a draw that lifts
 * (lift probability ~0.9995); the law is checked against the cloned draw. */
static void checkSrInt32MinLift(void) {
    const int32_t v = -2147352584;
    int64_t fl;
    uint32_t frac32;
    refFloorFrac(v, 16385, 14u, &fl, &frac32);
    TEST_ASSERT_EQUAL_INT64((int64_t)INT32_MIN - 1, fl);
    rngStream_t rs = rngOpen(rngKeyFromSeed(4u));
    rngStream_t clone = rs;
    uint32_t u = rngNextU32(&clone);
    TEST_ASSERT_TRUE_MESSAGE(u < frac32, "pinned seed must draw a lift");
    int32_t got = symMulShiftRound(v, 16385, 14u, SR_HALF_AWAY, &rs);
    TEST_ASSERT_EQUAL_INT64(fl + 1, (int64_t)got);
    TEST_ASSERT_EQUAL_UINT32(clone.state, rs.state);
}

void testMulShiftRoundSrFloorPlusLiftLaw(void) {
    lcgState = 0x12345678u;
    rngStream_t rs = rngOpen(rngKeyFromSeed(77u));
    size_t checked = 0, lifts = 0;
    for (size_t i = 0; i < N_TRIPLES; i++) {
        int32_t v = pickV();
        uint16_t m = pickM();
        uint32_t s = lcgNext() % 64u;
        int64_t fl;
        uint32_t frac32;
        refFloorFrac(v, m, s, &fl, &frac32);
        /* the SR result is floor or floor + 1: both must fit int32 */
        if (!fitsInt32(fl) || (frac32 != 0u && !fitsInt32(fl + 1))) {
            continue;
        }
        rngStream_t clone = rs;
        int32_t got = symMulShiftRound(v, m, s, SR_HALF_AWAY, &rs);
        bool lift = false;
        if (s > 0u) {
            uint32_t u = rngNextU32(&clone);
            lift = (u >> 0) < frac32;
        }
        TEST_ASSERT_EQUAL_UINT32(clone.state, rs.state);
        if ((int64_t)got != fl + (lift ? 1 : 0)) {
            char msg[128];
            snprintf(msg, sizeof msg, "v=%ld m=%u s=%u", (long)v, (unsigned)m, (unsigned)s);
            TEST_ASSERT_EQUAL_INT64_MESSAGE(fl + (lift ? 1 : 0), (int64_t)got, msg);
        }
        lifts += lift ? 1u : 0u;
        checked++;
    }
    TEST_ASSERT_GREATER_THAN_size_t(200000, checked);
    TEST_ASSERT_GREATER_THAN_size_t(1000, lifts); /* the law is exercised both ways */
    checkSrInt32MinLift();
}

void testMulShiftRoundSrDrawsOncePerCallWithShift(void) {
    rngStream_t rs = rngOpen(rngKeyFromSeed(5u));
    for (uint32_t s = 0; s < 70u; s++) {
        rngStream_t before = rs;
        (void)symMulShiftRound(12345, 30000, s, SR_HALF_AWAY, &rs);
        rngStream_t expected = before;
        if (s > 0u) {
            (void)rngNextU32(&expected);
        }
        TEST_ASSERT_EQUAL_UINT32(expected.state, rs.state);
    }
    /* HALF_AWAY never draws */
    rngStream_t before = rs;
    (void)symMulShiftRound(12345, 30000, 7u, HALF_AWAY, &rs);
    TEST_ASSERT_EQUAL_UINT32(before.state, rs.state);
}

void testMulShiftRoundExactValuesNeverLift(void) {
    rngStream_t rs = rngOpen(rngKeyFromSeed(11u));
    /* v * m divisible by 2^s: m = 2^14 contributes 14 bits, v the rest */
    for (int rep = 0; rep < 2000; rep++) {
        for (uint32_t s = 0; s <= 30u; s++) {
            int32_t q = (int32_t)(rngNextU32(&rs) % 2001u) - 1000;
            uint32_t vShift = s > 14u ? s - 14u : 0u;
            int32_t v = q * (1 << vShift);
            int64_t exact = ((int64_t)v * 16384) >> s;
            TEST_ASSERT_EQUAL_INT64(exact,
                                    (int64_t)symMulShiftRound(v, 16384, s, SR_HALF_AWAY, &rs));
        }
    }
    /* zero product, any shift */
    for (uint32_t s = 0; s < 70u; s++) {
        TEST_ASSERT_EQUAL_INT32(0, symMulShiftRound(INT32_MIN, 0, s, SR_HALF_AWAY, &rs));
        TEST_ASSERT_EQUAL_INT32(0, symMulShiftRound(0, 32768, s, SR_HALF_AWAY, &rs));
    }
}

void testMulShiftRoundHugeShift(void) {
    static const uint32_t shifts[] = {31, 32, 46, 47, 48, 63, 64, 200};
    static const int32_t vs[] = {INT32_MIN, INT32_MAX};
    for (size_t i = 0; i < sizeof shifts / sizeof shifts[0]; i++) {
        for (size_t j = 0; j < 2; j++) {
            uint32_t s = shifts[i];
            int32_t v = vs[j];
            int64_t ref = refHalfAway(v, 32767, s);
            TEST_ASSERT_EQUAL_INT64(ref, (int64_t)symMulShiftRound(v, 32767, s, HALF_AWAY, NULL));
            if (s >= 47u) {
                TEST_ASSERT_EQUAL_INT64(0, ref);
            }
            /* SR: floor or floor + 1 by the law */
            rngStream_t rs = rngOpen(rngKeyFromSeed(1000u + i * 2u + j));
            rngStream_t clone = rs;
            int64_t fl;
            uint32_t frac32;
            refFloorFrac(v, 32767, s, &fl, &frac32);
            uint32_t u = rngNextU32(&clone);
            int32_t got = symMulShiftRound(v, 32767, s, SR_HALF_AWAY, &rs);
            TEST_ASSERT_EQUAL_INT64(fl + (u < frac32 ? 1 : 0), (int64_t)got);
            /* s > 63 is s = 63, for both modes and the same draw */
            if (s > 63u) {
                rngStream_t a = rngOpen(rngKeyFromSeed(31u));
                rngStream_t b = rngOpen(rngKeyFromSeed(31u));
                TEST_ASSERT_EQUAL_INT32(symMulShiftRound(v, 32767, 63u, SR_HALF_AWAY, &a),
                                        symMulShiftRound(v, 32767, s, SR_HALF_AWAY, &b));
                TEST_ASSERT_EQUAL_INT32(symMulShiftRound(v, 32767, 63u, HALF_AWAY, NULL),
                                        symMulShiftRound(v, 32767, s, HALF_AWAY, NULL));
            }
        }
    }
    /* the documented magnitudes: ~ +-2^15 at s = 31, +-2^14 at s = 32 */
    TEST_ASSERT_EQUAL_INT32(-32767, symMulShiftRound(INT32_MIN, 32767, 31u, HALF_AWAY, NULL));
    TEST_ASSERT_EQUAL_INT32(32767, symMulShiftRound(INT32_MAX, 32767, 31u, HALF_AWAY, NULL));
    TEST_ASSERT_EQUAL_INT32(-16384, symMulShiftRound(INT32_MIN, 32767, 32u, HALF_AWAY, NULL));
    TEST_ASSERT_EQUAL_INT32(16383, symMulShiftRound(INT32_MAX, 32767, 32u, HALF_AWAY, NULL));
}

/* 32 < s < 48: the top 32 fraction bits can read exactly one half while a lower
 * bit is set. v * m = -(2^39 - 1) (8191 = 2^13 - 1 divides 2^39 - 1) at s = 40
 * is -0.5 + 2^-40: the floor is -1, the fraction 2^39 + 1, so frac32 = 2^31 with
 * a sticky low bit, and HALF_AWAY must give 0, not the negative tie's -1. */
void testMulShiftRoundHalfAwayTieNeedsLowBits(void) {
    const int32_t v = -67117057; /* (2^39 - 1) / 8191 */
    TEST_ASSERT_EQUAL_INT64(-(((int64_t)1 << 39) - 1), (int64_t)v * 8191);
    TEST_ASSERT_EQUAL_INT32(0, symMulShiftRound(v, 8191, 40u, HALF_AWAY, NULL));
    TEST_ASSERT_EQUAL_INT32(0, symMulShiftRound(-v, 8191, 40u, HALF_AWAY, NULL));
    /* the exact negative tie -2^39 / 2^40 = -0.5 rounds away to -1 */
    TEST_ASSERT_EQUAL_INT32(-1, symMulShiftRound(-(1 << 24), 32768, 40u, HALF_AWAY, NULL));
    TEST_ASSERT_EQUAL_INT32(1, symMulShiftRound(1 << 24, 32768, 40u, HALF_AWAY, NULL));
}

static void srWithoutAStream(void) {
    (void)symMulShiftRound(13, 16384, 3u, SR_HALF_AWAY, NULL);
}

void testMulShiftRoundSrWithoutAStreamExits(void) {
    ASSERT_EXITS_WITH_OUTPUT(1, "symMulShiftRound: SR shift without a stream", srWithoutAStream());
}

/* ---- symGroupFoldInit ---- */

static symQConfig_t groupedConfig(odtScale_t *scales, size_t numGroups) {
    return (symQConfig_t){.scales = scales,
                          .numGroups = numGroups,
                          .groupSize = 4,
                          .roundingMode = HALF_AWAY,
                          .qBits = 8};
}

static const odtScale_t IN_SCALE = {.m = 30000, .e = -10};

void testGroupFoldInitPicksLargestProvableF(void) {
    odtScale_t scales[2] = {{.m = 20000, .e = -20}, {.m = 16384, .e = -22}};
    symQConfig_t qc = groupedConfig(scales, 2);

    /* N = 4, int8 x int8: F = 8 fits easily */
    symGroupFold_t f8 = symGroupFoldInit(&qc, IN_SCALE, 8, 4, -128, 127, 0);
    TEST_ASSERT_FALSE(f8.legacy);
    TEST_ASSERT_EQUAL_UINT8(8, f8.fracBits);

    /* HAR dx: int12 gradient x int8 weight codes */
    symGroupFold_t har160 = symGroupFoldInit(&qc, IN_SCALE, 12, 160, -127, 127, 0);
    symGroupFold_t har192 = symGroupFoldInit(&qc, IN_SCALE, 12, 192, -127, 127, 0);
    TEST_ASSERT_FALSE(har160.legacy);
    TEST_ASSERT_FALSE(har192.legacy);
    TEST_ASSERT_EQUAL_UINT8(5, har160.fracBits);
    TEST_ASSERT_EQUAL_UINT8(5, har192.fracBits);

    /* only F = 0 fits: int12 x int8 at N = 5000 (F = 0: 5000 * 2048 * 128 < 2^31;
     * F = 1: 5000 * 4097 * 128 > 2^31) */
    symGroupFold_t only0 = symGroupFoldInit(&qc, IN_SCALE, 12, 5000, -128, 127, 0);
    TEST_ASSERT_TRUE(only0.legacy);

    /* inputs wider than 16 bits keep the float path */
    symGroupFold_t wide = symGroupFoldInit(&qc, IN_SCALE, 17, 4, -128, 127, 0);
    TEST_ASSERT_TRUE(wide.legacy);
}

void testGroupFoldInitLegacyWhenAccScaleLeavesCanonicalRange(void) {
    /* A.e = inScale.e + E + 15 - F; inScale at the bottom of the range and a
     * small E push it below ODT_SCALE_E_MIN */
    odtScale_t scales[2] = {{.m = 16384, .e = -60}, {.m = 16384, .e = -100}};
    symQConfig_t qc = groupedConfig(scales, 2);
    odtScale_t inScale = {.m = 16384, .e = ODT_SCALE_E_MIN + 10};
    symGroupFold_t f = symGroupFoldInit(&qc, inScale, 8, 4, -128, 127, 0);
    TEST_ASSERT_TRUE(f.legacy);

    /* the same groups with an inScale high enough: A stays canonical */
    odtScale_t inHigh = {.m = 16384, .e = 0};
    symGroupFold_t ok = symGroupFoldInit(&qc, inHigh, 8, 4, -128, 127, 0);
    TEST_ASSERT_FALSE(ok.legacy);
    TEST_ASSERT_EQUAL_INT16(0 + -60 + 15 - 8, ok.accScale.e);
}

void testGroupFoldInitAccScaleIsExactDyadic(void) {
    odtScale_t scales[2] = {{.m = 20000, .e = -20}, {.m = 16384, .e = -22}};
    symQConfig_t qc = groupedConfig(scales, 2);
    symGroupFold_t f = symGroupFoldInit(&qc, IN_SCALE, 8, 4, -128, 127, 0);
    TEST_ASSERT_FALSE(f.legacy);
    uint8_t F = f.fracBits;
    TEST_ASSERT_EQUAL_INT16(-20, f.eMax);
    TEST_ASSERT_EQUAL_UINT16(30000, f.accScale.m);
    TEST_ASSERT_EQUAL_INT16(-10 - 20 + 15 - F, f.accScale.e);
    odtScale_t s0 = odtScaleLoad(&qc.scales[0]);
    odtScale_t s1 = odtScaleLoad(&qc.scales[1]);
    TEST_ASSERT_EQUAL_UINT32(15u - F, symGroupFoldShift(&f, s0));
    TEST_ASSERT_EQUAL_UINT32(17u - F, symGroupFoldShift(&f, s1));
}

/* The last parameter is the seed magnitude measured at the finest A (F =
 * ODT_GROUP_FOLD_FRAC_BITS_MAX): the seed at F is 2^(MAX - F) times smaller,
 * plus slack ceil(seed_F * 2^-22) + 1 for the float rescale that forms it.
 * Each F needs that seed within the seed-rescale guard AND the site with it
 * proven safe. */
void testGroupFoldInitSeedAtFracMaxLowersF(void) {
    odtScale_t scales[2] = {{.m = 20000, .e = -20}, {.m = 16384, .e = -22}};
    symQConfig_t qc = groupedConfig(scales, 2);

    /* int8 x int8, N = 4: the accumulator has room, but the seed at F = 8
     * (1500000359 with slack) exceeds the seed guard 2^31 - 1 - 32768 * 32767;
     * at F = 7 it is 750000000 + 179 + 1 */
    symGroupFold_t guardLimited = symGroupFoldInit(&qc, IN_SCALE, 8, 4, -128, 127, 1500000000u);
    TEST_ASSERT_FALSE(guardLimited.legacy);
    TEST_ASSERT_EQUAL_UINT8(7, guardLimited.fracBits);
    TEST_ASSERT_EQUAL_UINT64(750000180u, guardLimited.maxAbsSeed);
    TEST_ASSERT_EQUAL_INT16(IN_SCALE.e - 20 + 15 - 7, guardLimited.accScale.e);

    /* HAR dx shape (int12 x int8, N = 160, seedless F = 5): the seed at F = 5
     * (900000216) leaves no room next to 160 * 4097 * 2^5 * 128 terms; at
     * F = 4 it is 450000000 + 108 + 1 */
    symGroupFold_t roomLimited = symGroupFoldInit(&qc, IN_SCALE, 12, 160, -127, 127, 7200000000u);
    TEST_ASSERT_FALSE(roomLimited.legacy);
    TEST_ASSERT_EQUAL_UINT8(4, roomLimited.fracBits);
    TEST_ASSERT_EQUAL_UINT64(450000109u, roomLimited.maxAbsSeed);

    /* no seed: no slack, F as before */
    symGroupFold_t seedless = symGroupFoldInit(&qc, IN_SCALE, 12, 160, -127, 127, 0);
    TEST_ASSERT_EQUAL_UINT8(5, seedless.fracBits);
    TEST_ASSERT_EQUAL_UINT64(0u, seedless.maxAbsSeed);
}

/* A seed too large at every F (even 2^7 times coarser than at F = 8) forces
 * the legacy path; the bound arithmetic must not wrap for any uint64 seed. */
void testGroupFoldInitSeedForcesLegacy(void) {
    odtScale_t scales[2] = {{.m = 20000, .e = -20}, {.m = 16384, .e = -22}};
    symQConfig_t qc = groupedConfig(scales, 2);
    symGroupFold_t big = symGroupFoldInit(&qc, IN_SCALE, 8, 4, -128, 127, (uint64_t)1 << 40);
    TEST_ASSERT_TRUE(big.legacy);
    symGroupFold_t huge = symGroupFoldInit(&qc, IN_SCALE, 8, 4, -128, 127, UINT64_MAX);
    TEST_ASSERT_TRUE(huge.legacy);
    /* 1e9 * 2^7 at F = 8 is 1e9 + 239 + 1 at F = 1 (fits the guard) and
     * 2e9 at F = 2 (does not): only F = 1 remains */
    symGroupFold_t edge =
        symGroupFoldInit(&qc, IN_SCALE, 8, 4, -128, 127, (uint64_t)1000000000u << 7);
    TEST_ASSERT_FALSE(edge.legacy);
    TEST_ASSERT_EQUAL_UINT8(1, edge.fracBits);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testMulShiftRoundMatchesInt64ReferenceHalfAway);
    RUN_TEST(testMulShiftRoundSrFloorPlusLiftLaw);
    RUN_TEST(testMulShiftRoundSrDrawsOncePerCallWithShift);
    RUN_TEST(testMulShiftRoundExactValuesNeverLift);
    RUN_TEST(testMulShiftRoundHugeShift);
    RUN_TEST(testMulShiftRoundHalfAwayTieNeedsLowBits);
    RUN_TEST(testMulShiftRoundSrWithoutAStreamExits);
    RUN_TEST(testGroupFoldInitPicksLargestProvableF);
    RUN_TEST(testGroupFoldInitLegacyWhenAccScaleLeavesCanonicalRange);
    RUN_TEST(testGroupFoldInitAccScaleIsExactDyadic);
    RUN_TEST(testGroupFoldInitSeedAtFracMaxLowersF);
    RUN_TEST(testGroupFoldInitSeedForcesLegacy);
    return UNITY_END();
}
