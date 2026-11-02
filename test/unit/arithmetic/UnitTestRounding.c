#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "DeathTest.h"
#include "RNG.h"
#include "RngPinnedDraw.h"
#include "Rounding.h"
#include "unity.h"

void testRescaleIntoAccumulatorScale() {
    /* 100 * 4.0 / 0.5 = 800 (exact) */
    TEST_ASSERT_EQUAL_INT32(800, rescaleIntoAccumulatorScale(100, 4.0f, 0.5f, HALF_AWAY, NULL));

    /* small accumulator scale -> large seed: 1 / 2^-10 = 1024 (exact) */
    TEST_ASSERT_EQUAL_INT32(1024,
                            rescaleIntoAccumulatorScale(1, 1.0f, 0.0009765625f, HALF_AWAY, NULL));

    /* half-away rounding: 3 * 1.0 / 2.0 = 1.5 -> 2 ; -3 -> -1.5 -> -2 */
    TEST_ASSERT_EQUAL_INT32(2, rescaleIntoAccumulatorScale(3, 1.0f, 2.0f, HALF_AWAY, NULL));
    TEST_ASSERT_EQUAL_INT32(-2, rescaleIntoAccumulatorScale(-3, 1.0f, 2.0f, HALF_AWAY, NULL));

    /* identical scales -> identity (ratio exactly 1.0) */
    TEST_ASSERT_EQUAL_INT32(12345,
                            rescaleIntoAccumulatorScale(12345, 0.25f, 0.25f, HALF_AWAY, NULL));
}

void testRoundHalfAway() {
    float input = 1.7f;
    int32_t actual = roundByMode(input, HALF_AWAY, NULL);
    int32_t expected = 2;
    TEST_ASSERT_EQUAL(expected, actual);

    input = 2.3f;
    actual = roundByMode(input, HALF_AWAY, NULL);
    expected = 2;
    TEST_ASSERT_EQUAL(expected, actual);

    input = 2.5f;
    actual = roundByMode(input, HALF_AWAY, NULL);
    expected = 3;
    TEST_ASSERT_EQUAL(expected, actual);

    input = -1.7f;
    actual = roundByMode(input, HALF_AWAY, NULL);
    expected = -2;
    TEST_ASSERT_EQUAL(expected, actual);

    input = -2.3f;
    actual = roundByMode(input, HALF_AWAY, NULL);
    expected = -2;
    TEST_ASSERT_EQUAL(expected, actual);

    input = -2.5f;
    actual = roundByMode(input, HALF_AWAY, NULL);
    expected = -3;
    TEST_ASSERT_EQUAL(expected, actual);
}

static void assertMatchesDoubleRound(const float x) {
    TEST_ASSERT_EQUAL_INT32((int32_t)round((double)x), roundByMode(x, HALF_AWAY, NULL));
}

/* HALF_AWAY must give the value C round() gives on the float promoted to
 * double, over ties, their float neighbours, a strided sweep of all finite
 * floats inside int32 range, and the int32 edges. */
void testRoundHalfAwayMatchesDoubleRoundOverFloatSweep() {
    for (int32_t k = -70000; k <= 70000; k++) {
        const float tie = (float)k + 0.5f;
        assertMatchesDoubleRound(tie);
        assertMatchesDoubleRound(nextafterf(tie, -INFINITY));
        assertMatchesDoubleRound(nextafterf(tie, INFINITY));
    }

    for (uint64_t bits = 0; bits <= UINT32_MAX; bits += 4099u) {
        const uint32_t pattern = (uint32_t)bits;
        float x;
        memcpy(&x, &pattern, sizeof x);
        if (isfinite(x) && fabsf(x) < 2147483520.0f) {
            assertMatchesDoubleRound(x);
        }
    }

    const float edges[] = {0.0f,        -0.0f,         0.49999997f,    -0.49999997f,  8388607.5f,
                           -8388607.5f, 2147483520.0f, -2147483520.0f, -2147483648.0f};
    for (size_t i = 0; i < sizeof edges / sizeof edges[0]; i++) {
        assertMatchesDoubleRound(edges[i]);
    }
}

/* SR rounding draws exactly one unit float from the stream it is handed. The
 * first draw of rngOpen(rngKeyFromSeed(31)):
 *   uv run python -c "from examples._shared.keyed_rng import (key_from_seed as kfs,
 *   open_stream as o, next_unit_float as u); print(u(o(kfs(31)))[1])"
 * prints 0.1786224... < 0.5 = frac(352.5), so SR lifts to 353 (law: Rounding.h). */
void testRoundByModeSrDrawsOneUnitFloatFromItsStream() {
    rngStream_t rs = rngOpen(rngKeyFromSeed(31u));
    rngStream_t oneStepOn = rngOpen(rngKeyFromSeed(31u));
    (void)rngNextU32(&oneStepOn);
    TEST_ASSERT_EQUAL_INT32(353, roundByMode(352.5f, SR_HALF_AWAY, &rs));
    TEST_ASSERT_EQUAL_HEX32(oneStepOn.state, rs.state);
}

/* #510: an exactly representable code (frac == 0) must come back unchanged on both
 * sides of zero (the old round(x + u - 0.5) formula sent every k <= 0 to k - 1). */
void testRoundByModeSrWithZeroDrawKeepsExactIntegers() {
    const float values[] = {0.f, -3.f, -2048.f, 3.f, -2047.f, 2047.f};
    for (size_t i = 0; i < sizeof values / sizeof values[0]; i++) {
        rngStream_t rs = rngStreamWithNextDraw(0x00000001u); /* top 24 bits 0: u == 0 */
        TEST_ASSERT_EQUAL_INT32((int32_t)values[i], roundByMode(values[i], SR_HALF_AWAY, &rs));
    }
}

/* x = -0.25, u = 0.25 == |frac|: not below it, so no lowering; the result is 0. */
void testRoundByModeSrSumOnIntegerTakesTheInteger() {
    rngStream_t rs = rngStreamWithNextDraw(0x40000000u); /* top 24 bits 0x400000: u == 0.25 */
    TEST_ASSERT_EQUAL_INT32(0, roundByMode(-0.25f, SR_HALF_AWAY, &rs));
}

/* SR must never form the float sum x + u, whose rounded value crosses the next integer
 * for u > 1 - ulp(x)/2 (a magnitude-growing upward bias). */
static int32_t srWithDraw(float x, uint32_t draw) {
    rngStream_t rs = rngStreamWithNextDraw(draw);
    return roundByMode(x, SR_HALF_AWAY, &rs);
}

/* 2047 + (1 - 2^-24) would round up to 2048 in float32; an exact integer never lifts. */
void testRoundByModeSrLargestDrawDoesNotLiftAcrossTheInteger() {
    TEST_ASSERT_EQUAL_INT32(2047, srWithDraw(2047.f, 0xFFFFFFFFu));
}

/* -2^24 + 0.75 would round to -2^24 + 1 in float32; an exact integer never moves. */
void testRoundByModeSrNegativeBinadeEdgeStaysOnTheInteger() {
    TEST_ASSERT_EQUAL_INT32(-16777216, srWithDraw(-16777216.f, 0xC0000000u));
    TEST_ASSERT_EQUAL_INT32(8388609, srWithDraw(8388609.f, 0xC0000000u));
}

/* Spacing is 0.5 on [2^22, 2^23): frac = 0.5. Lift iff u < 0.5: u = 0.25 lifts,
 * u = 0.5 does not. */
void testRoundByModeSrHalfIntegerInLargeBinadeFollowsTheFraction() {
    TEST_ASSERT_EQUAL_INT32(4194305, srWithDraw(4194304.5f, 0x40000000u));
    TEST_ASSERT_EQUAL_INT32(4194304, srWithDraw(4194304.5f, 0x80000000u));
}

/* frac = 3 * 2^-25 is far below a near-one draw: no lift. */
void testRoundByModeSrTinyPositiveWithNearOneDrawStaysZero() {
    TEST_ASSERT_EQUAL_INT32(0, srWithDraw(ldexpf(3.f, -25), 0xFFFFFE00u));
}

/* x = -2^-30: lower iff u < 2^-30, i.e. only u == 0; the smallest nonzero u (2^-24) is enough. */
void testRoundByModeSrTinyNegativeRoundsDownOnlyForZeroDraw() {
    TEST_ASSERT_EQUAL_INT32(-1, srWithDraw(ldexpf(-1.f, -30), 0x00000001u));
    TEST_ASSERT_EQUAL_INT32(0, srWithDraw(ldexpf(-1.f, -30), 0x00000100u));
}

/* Pins the law: lift iff u < frac(x) (negative x: lower iff u < -frac), u = (draw >> 8) / 2^24. */
void testRoundByModeSrLiftsIffDrawBelowFraction() {
    TEST_ASSERT_EQUAL_INT32(3, srWithDraw(2.25f, 0x1999999Au));   /* u ~ 0.1 < 0.25: lift */
    TEST_ASSERT_EQUAL_INT32(2, srWithDraw(2.25f, 0xE6666666u));   /* u ~ 0.9: stay */
    TEST_ASSERT_EQUAL_INT32(-3, srWithDraw(-2.25f, 0x1999999Au)); /* u ~ 0.1 < 0.25: lower */
    TEST_ASSERT_EQUAL_INT32(-2, srWithDraw(-2.25f, 0xE6666666u));
    TEST_ASSERT_EQUAL_INT32(2, srWithDraw(2.25f, 0xC0000000u));   /* u = 0.75: stay */
    TEST_ASSERT_EQUAL_INT32(2, srWithDraw(2.25f, 0x40000000u));   /* u = 0.25 == frac: stay */
    TEST_ASSERT_EQUAL_INT32(3, srWithDraw(2.25f, 0x3FFFFF00u));   /* u = 0.25 - 2^-24: lift */
    TEST_ASSERT_EQUAL_INT32(-2, srWithDraw(-2.25f, 0x40000000u)); /* u = 0.25 == |frac|: stay */
    TEST_ASSERT_EQUAL_INT32(-3, srWithDraw(-2.25f, 0x3FFFFF00u));
}

/* Odd per draw: SR(-x) == -SR(x) for the same u. */
void testRoundByModeSrIsOddPerDraw() {
    const float xs[] = {0.1f, 0.5f, 2.25f, 2.75f, 352.5f, 4194304.5f, 1000.999f};
    const uint32_t draws[] = {0x00000001u, 0x1999999Au, 0x40000000u,
                              0x80000000u, 0xE6666666u, 0xFFFFFFFFu};
    for (size_t i = 0; i < sizeof xs / sizeof xs[0]; i++) {
        for (size_t j = 0; j < sizeof draws / sizeof draws[0]; j++) {
            TEST_ASSERT_EQUAL_INT32(-srWithDraw(xs[i], draws[j]), srWithDraw(-xs[i], draws[j]));
        }
    }
}

static void roundSrWithoutAStream(void) {
    (void)roundByMode(352.5f, SR_HALF_AWAY, NULL);
}

/* No key, no randomness: SR rounding without a stream stops at the draw;
 * HALF_AWAY never draws and takes NULL (testRoundHalfAway). */
void testRoundByModeSrWithoutAStreamExits() {
    ASSERT_EXITS_WITH_OUTPUT(1, "roundByMode: SR rounding without a stream",
                             roundSrWithoutAStream());
}

void setUp() {}
void tearDown() {}

/* A partial near INT32_MAX is legitimate (no seed headroom reserved), but the
 * float->int32 cast must still be refused outside [-2^31, 2^31) and for NaN. */
void testRescalePartialKeepsOnlyCastSafety(void) {
    TEST_ASSERT_EQUAL_INT32(
        2147483520, rescalePartialIntoAccumulatorScale(2147483520, 1.0f, 1.0f, HALF_AWAY, NULL));
    TEST_ASSERT_EQUAL_INT32(
        INT32_MIN, rescalePartialIntoAccumulatorScale(INT32_MIN, 1.0f, 1.0f, HALF_AWAY, NULL));
    TEST_ASSERT_EQUAL_INT32(100,
                            rescalePartialIntoAccumulatorScale(100, 1.0f, 1.0f, HALF_AWAY, NULL));
    ASSERT_EXITS_WITH_OUTPUT(
        1, "outside int32",
        (void)rescalePartialIntoAccumulatorScale(INT32_MAX, 1.0f, 1.0f, HALF_AWAY, NULL));
    ASSERT_EXITS_WITH_OUTPUT(
        1, "outside int32",
        (void)rescalePartialIntoAccumulatorScale(1 << 20, 4096.0f, 1.0f, HALF_AWAY, NULL));
    ASSERT_EXITS_WITH_OUTPUT(
        1, "outside int32",
        (void)rescalePartialIntoAccumulatorScale(0, 1.0f, 0.0f, HALF_AWAY, NULL));
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testRescalePartialKeepsOnlyCastSafety);
    RUN_TEST(testRoundHalfAway);
    RUN_TEST(testRoundHalfAwayMatchesDoubleRoundOverFloatSweep);
    RUN_TEST(testRescaleIntoAccumulatorScale);
    RUN_TEST(testRoundByModeSrDrawsOneUnitFloatFromItsStream);
    RUN_TEST(testRoundByModeSrWithoutAStreamExits);
    RUN_TEST(testRoundByModeSrWithZeroDrawKeepsExactIntegers);
    RUN_TEST(testRoundByModeSrSumOnIntegerTakesTheInteger);
    RUN_TEST(testRoundByModeSrLargestDrawDoesNotLiftAcrossTheInteger);
    RUN_TEST(testRoundByModeSrNegativeBinadeEdgeStaysOnTheInteger);
    RUN_TEST(testRoundByModeSrHalfIntegerInLargeBinadeFollowsTheFraction);
    RUN_TEST(testRoundByModeSrTinyPositiveWithNearOneDrawStaysZero);
    RUN_TEST(testRoundByModeSrTinyNegativeRoundsDownOnlyForZeroDraw);
    RUN_TEST(testRoundByModeSrLiftsIffDrawBelowFraction);
    RUN_TEST(testRoundByModeSrIsOddPerDraw);

    return UNITY_END();
}
