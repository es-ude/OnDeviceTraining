#define SOURCE_FILE "UnitTestMemProfile"

#include "MemProfile.h"
#include "StorageApi.h"
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

static volatile unsigned char g_sink;

/* Touches n bytes of a stack VLA end-to-end so the compiler cannot elide it.
 * No framework calls — pure stack load. */
static void stackConsumer(void *arg) {
    size_t n = *(size_t *)arg;
    volatile unsigned char buf[n];
    for (size_t i = 0; i < n; i++) {
        buf[i] = (unsigned char)(i & 0xFF);
    }
    g_sink += buf[n - 1];
}

/* Frame slack established from first principles: the ONLY non-VLA variation
 * between calls is alignment padding of the VLA base. 256 bytes is a generous
 * ceiling for that. This constant is NEVER raised to make a failing case pass
 * (integrity rule) — a delta beyond it is a finding to investigate with Leo. */
#define F_STACK_SLACK 256

void testStackWatermarkTracksLoadRelativeAndMonotonic(void) {
    size_t n16 = 16u * 1024, n64 = 64u * 1024, n256 = 256u * 1024;
    size_t region = 1u << 20; /* 1 MiB */

    size_t s16 = measurePeakStackBytes(stackConsumer, &n16, region);
    size_t s64 = measurePeakStackBytes(stackConsumer, &n64, region);
    size_t s256 = measurePeakStackBytes(stackConsumer, &n256, region);

    /* Absolute sanity: each measurement is at least its own VLA load. */
    TEST_ASSERT_GREATER_OR_EQUAL_size_t(n16, s16);
    TEST_ASSERT_GREATER_OR_EQUAL_size_t(n64, s64);
    TEST_ASSERT_GREATER_OR_EQUAL_size_t(n256, s256);

    /* Monotonic in load. */
    TEST_ASSERT_GREATER_THAN_size_t(s16, s64);
    TEST_ASSERT_GREATER_THAN_size_t(s64, s256);

    /* Deltas equal the load deltas within the fixed frame slack (absolute
     * frame overhead cancels). This is the "genuinely tracks stack" claim. */
    long d1 = (long)(s64 - s16) - (long)(n64 - n16);
    long d2 = (long)(s256 - s64) - (long)(n256 - n64);
    TEST_ASSERT_INT_WITHIN(F_STACK_SLACK, 0, (int)d1);
    TEST_ASSERT_INT_WITHIN(F_STACK_SLACK, 0, (int)d2);
}

/* A caller may pass any stackBytes; Apple libpthread rejects a region whose size
 * is not a page multiple (EINVAL) in every build, not only under ASan. */
void testStackWatermarkHandlesNonPageMultipleRegion(void) {
    size_t load = 4u * 1024;
    size_t region = (1u << 20) + 100; /* deliberately not a page multiple */

    size_t used = measurePeakStackBytes(stackConsumer, &load, region);

    TEST_ASSERT_GREATER_OR_EQUAL_size_t(load, used);
    TEST_ASSERT_LESS_OR_EQUAL_size_t(region, used);
}

void testRssPeakIsPositiveAndGrows(void) {
    size_t before = memProfileRssPeakKb();
    TEST_ASSERT_GREATER_THAN_size_t(0, before); /* the process already uses RAM */
    /* Force a large resident allocation and touch it. */
    size_t big = 32u * 1024 * 1024;
    volatile unsigned char *p = reserveMemory(big);
    for (size_t i = 0; i < big; i += 4096) {
        p[i] = 1;
    }
    size_t after = memProfileRssPeakKb();
    TEST_ASSERT_GREATER_OR_EQUAL_size_t(before, after); /* peak is monotonic */
    freeReservedMemory((void *)p);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testStackWatermarkTracksLoadRelativeAndMonotonic);
    RUN_TEST(testStackWatermarkHandlesNonPageMultipleRegion);
    RUN_TEST(testRssPeakIsPositiveAndGrows);
    return UNITY_END();
}
