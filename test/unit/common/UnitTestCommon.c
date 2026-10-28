#define SOURCE_FILE "UnitTestCommon"

#include <stdint.h>

#include "CheckedSize.h"
#include "Common.h"
#include "unity.h"

_Static_assert(
    DLEVEL >= 1,
    "DEBUG_MODE_ERROR must be defined by default — PRINT_ERROR would be silently compiled out");

void setUp(void) {}
void tearDown(void) {}

void testDLevelDefaultIsErrorOrHigher(void) {
    TEST_ASSERT_GREATER_OR_EQUAL_INT(1, DLEVEL);
}

/* *out is unspecified after a false return, so only the true cases check it. */
void testCheckedMulSizeDetectsWrap(void) {
    size_t o = 0;
    TEST_ASSERT_FALSE(checkedMulSize(SIZE_MAX / 2u + 1u, 2u, &o));
    TEST_ASSERT_TRUE(checkedMulSize(SIZE_MAX / 3u, 3u, &o));
    TEST_ASSERT_EQUAL_size_t(SIZE_MAX / 3u * 3u, o);
    TEST_ASSERT_TRUE(checkedMulSize(0u, SIZE_MAX, &o));
    TEST_ASSERT_EQUAL_size_t(0u, o);
}

void testCheckedAddSizeDetectsWrap(void) {
    size_t o = 0;
    TEST_ASSERT_FALSE(checkedAddSize(SIZE_MAX, 1u, &o));
    TEST_ASSERT_TRUE(checkedAddSize(SIZE_MAX - 1u, 1u, &o));
    TEST_ASSERT_EQUAL_size_t(SIZE_MAX, o);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testDLevelDefaultIsErrorOrHigher);
    RUN_TEST(testCheckedMulSizeDetectsWrap);
    RUN_TEST(testCheckedAddSizeDetectsWrap);
    return UNITY_END();
}
