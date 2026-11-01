#include <stddef.h>
#include <stdint.h>

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
    RUN_TEST(testRescaleIntoAccumulatorScale);
    RUN_TEST(testRoundByModeSrDrawsOneUnitFloatFromItsStream);
    RUN_TEST(testRoundByModeSrWithoutAStreamExits);

    return UNITY_END();
}
