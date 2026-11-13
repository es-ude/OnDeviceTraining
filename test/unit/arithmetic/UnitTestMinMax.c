#define SOURCE_FILE "UNIT_TEST_MIN_MAX"

#include "MinMax.h"
#include "unity.h"

void setUp() {}
void tearDown() {}

/* Element 0 holds a poison value, so an implementation that seeds the max from
 * values[0] before checking n returns 7.5f instead of 0.f. */
void testFindAbsMaxFloatEmptyReturnsZero(void) {
    float poisoned[2] = {-7.5f, 3.25f};
    TEST_ASSERT_EQUAL_FLOAT(0.0f, findAbsMaxFloat((uint8_t *)poisoned, 0));
}

/* The maximum sits neither at index 0 nor at the last index. */
void testFindAbsMaxFloatScansEveryElement(void) {
    float values[4] = {1.0f, -6.5f, 2.0f, 4.0f};
    TEST_ASSERT_EQUAL_FLOAT(6.5f, findAbsMaxFloat((uint8_t *)values, 4));
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testFindAbsMaxFloatEmptyReturnsZero);
    RUN_TEST(testFindAbsMaxFloatScansEveryElement);
    return UNITY_END();
}
