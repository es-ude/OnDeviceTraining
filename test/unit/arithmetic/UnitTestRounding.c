#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "DeathTest.h"
#include "RNG.h"
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
 * prints 0.1786224..., so round(352.5 + j - 0.5) = 352 (HALF_AWAY: 353). */
void testRoundByModeSrDrawsOneUnitFloatFromItsStream() {
    rngStream_t rs = rngOpen(rngKeyFromSeed(31u));
    rngStream_t oneStepOn = rngOpen(rngKeyFromSeed(31u));
    (void)rngNextU32(&oneStepOn);
    TEST_ASSERT_EQUAL_INT32(352, roundByMode(352.5f, SR_HALF_AWAY, &rs));
    TEST_ASSERT_EQUAL_HEX32(oneStepOn.state, rs.state);
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

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testRoundHalfAway);
    RUN_TEST(testRoundHalfAwayMatchesDoubleRoundOverFloatSweep);
    RUN_TEST(testRescaleIntoAccumulatorScale);
    RUN_TEST(testRoundByModeSrDrawsOneUnitFloatFromItsStream);
    RUN_TEST(testRoundByModeSrWithoutAStreamExits);

    return UNITY_END();
}
