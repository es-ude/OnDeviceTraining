#include <stddef.h>

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

void setUp() {}
void tearDown() {}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testRoundHalfAway);
    RUN_TEST(testRescaleIntoAccumulatorScale);

    return UNITY_END();
}
