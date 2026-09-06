#define SOURCE_FILE "UNIT_TEST_MIN_MAX"

#include "MinMax.h"
#include "unity.h"

void setUp() {}
void tearDown() {}

/* #420 G4: findAbsMaxFloat seeded its running max from values[0] before the
 * loop, so numberOfElements == 0 was an out-of-bounds read (#160 family,
 * shared with the SYM engines). n == 0 is now DEFINED to return 0.f -- the
 * absmax of nothing -- which is what its downstream consumer already treats
 * as "no data": deriveBfpStoredExponent's absMax == 0 branch returns the
 * zero-state exponent (bias, scale 1.0).
 *
 * No death test: this test DEFINES behaviour instead of trapping it. The
 * buffer below is a real, fully in-bounds array whose element 0 holds a
 * POISON value, so the guard-absent mutant returns 7.5f instead of 0.f --
 * observable with no undefined behaviour anywhere in the mutant either. */
void testFindAbsMaxFloatEmptyReturnsZero(void) {
    float poisoned[2] = {-7.5f, 3.25f};
    TEST_ASSERT_EQUAL_FLOAT(0.0f, findAbsMaxFloat((uint8_t *)poisoned, 0));
}

void testFindAbsMaxFloatSingleElementReturnsItsMagnitude(void) {
    float values[1] = {-2.5f};
    TEST_ASSERT_EQUAL_FLOAT(2.5f, findAbsMaxFloat((uint8_t *)values, 1));
}

/* Pins that the n == 0 early return did not shorten the scan: the maximum
 * sits neither at index 0 nor at the last index. */
void testFindAbsMaxFloatScansEveryElement(void) {
    float values[4] = {1.0f, -6.5f, 2.0f, 4.0f};
    TEST_ASSERT_EQUAL_FLOAT(6.5f, findAbsMaxFloat((uint8_t *)values, 4));
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testFindAbsMaxFloatEmptyReturnsZero);
    RUN_TEST(testFindAbsMaxFloatSingleElementReturnsItsMagnitude);
    RUN_TEST(testFindAbsMaxFloatScansEveryElement);
    return UNITY_END();
}
