#include "Rounding.h"
#include "unity.h"

void testRoundHalfAway() {
    float input = 1.7f;
    int32_t actual = roundByMode(input, HALF_AWAY);
    int32_t expected = 2;
    TEST_ASSERT_EQUAL(expected, actual);

    input = 2.3f;
    actual = roundByMode(input, HALF_AWAY);
    expected = 2;
    TEST_ASSERT_EQUAL(expected, actual);

    input = 2.5f;
    actual = roundByMode(input, HALF_AWAY);
    expected = 3;
    TEST_ASSERT_EQUAL(expected, actual);

    input = -1.7f;
    actual = roundByMode(input, HALF_AWAY);
    expected = -2;
    TEST_ASSERT_EQUAL(expected, actual);

    input = -2.3f;
    actual = roundByMode(input, HALF_AWAY);
    expected = -2;
    TEST_ASSERT_EQUAL(expected, actual);

    input = -2.5f;
    actual = roundByMode(input, HALF_AWAY);
    expected = -3;
    TEST_ASSERT_EQUAL(expected, actual);
}

void setUp() {}
void tearDown() {}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testRoundHalfAway);

    return UNITY_END();
}
