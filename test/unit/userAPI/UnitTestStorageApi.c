#define SOURCE_FILE "UnitTestStorageApi"

#include <stddef.h>
#include <stdint.h>

#include "AsanDeath.h"
#include "DeathTest.h"
#include "StorageApi.h"
#include "unity.h"

/* Compile-time contract: reserveMemory returns the allocated payload
 * directly (void *), and freeReservedMemory accepts that same payload.
 * The old void ** handle-indirection is removed; see issue #99 comment
 * for the design rationale (a real handle-table allocator will reintroduce
 * a typed opaque handle type later, not void **). */
_Static_assert(_Generic((&reserveMemory), void *(*)(size_t): 1, default: 0),
               "reserveMemory must return void * (payload), not void ** (handle)");
_Static_assert(_Generic((&freeReservedMemory), void (*)(void *): 1, default: 0),
               "freeReservedMemory must take void * (payload)");

void testReserveMemoryReturnsZeroedWritablePayload(void) {
    uint8_t *payload = reserveMemory(64);
    TEST_ASSERT_NOT_NULL(payload);

    for (size_t i = 0; i < 64; ++i) {
        TEST_ASSERT_EQUAL_UINT8(0, payload[i]);
    }
    payload[0] = 0xAA;
    payload[63] = 0x55;
    TEST_ASSERT_EQUAL_UINT8(0xAA, payload[0]);
    TEST_ASSERT_EQUAL_UINT8(0x55, payload[63]);

    freeReservedMemory(payload);
}

void testReserveAndFreeRoundTrip(void) {
    int *payload = reserveMemory(sizeof(int));
    TEST_ASSERT_NOT_NULL(payload);

    *payload = 42;
    TEST_ASSERT_EQUAL_INT(42, *payload);

    freeReservedMemory(payload);
}

void testFreeReservedMemoryIsNullSafe(void) {
    freeReservedMemory(NULL);
}

static void reserveAndFreeZeroBytes(void) {
    freeReservedMemory(reserveMemory(0));
}

/* A 0-byte request is not a failure, even where calloc(1, 0) is NULL (#160). */
void testReserveMemoryOfZeroBytesReturns(void) {
    ASSERT_EXITS_WITH(0, reserveAndFreeZeroBytes());
}

#if defined(ODT_MEM_PROFILE) || !defined(ODT_TEST_ASAN)
/* SIZE_MAX wraps the profiling header and no calloc serves it. Skipped for a
 * plain calloc under ASan, which aborts on oversized requests. */
void testReserveMemoryExitsOnAWrappingSize(void) {
    ASSERT_EXITS_WITH_OUTPUT(1, "reserveMemory: cannot reserve", (void)reserveMemory(SIZE_MAX));
}
#endif

#if SIZE_MAX > UINT32_MAX && !defined(ODT_TEST_ASAN)
/* 2^62 bytes wrap nothing but no 64-bit host can reserve them. Skipped under
 * ASan, which aborts on oversized requests; macOS malloc prints a "can't
 * allocate region" warning to stderr here. */
void testReserveMemoryExitsWhenTheAllocationFails(void) {
    ASSERT_EXITS_WITH_OUTPUT(1, "reserveMemory: cannot reserve 4611686018427387904 bytes",
                             (void)reserveMemory((size_t)1 << 62));
}
#endif

#ifdef ODT_MEM_PROFILE
/* GT-1: exact-integer heap accounting. No tolerance exists to fudge.
 * Guarded by ODT_MEM_PROFILE: the counter is a no-op when the macro is
 * undefined, so these tests only compile/run in profiling builds — the
 * ASan/UBSan CI presets (ODT_MEM_PROFILE=ON); plain unit_test has it OFF. */
void testHeapCounterExactCurrentAndPeak(void) {
    memProfileReset();
    TEST_ASSERT_EQUAL_size_t(0, memProfileCurrentBytes());
    TEST_ASSERT_EQUAL_size_t(0, memProfilePeakBytes());

    void *a = reserveMemory(1000);
    TEST_ASSERT_EQUAL_size_t(1000, memProfileCurrentBytes());
    TEST_ASSERT_EQUAL_size_t(1000, memProfilePeakBytes());

    void *b = reserveMemory(500);
    TEST_ASSERT_EQUAL_size_t(1500, memProfileCurrentBytes());
    TEST_ASSERT_EQUAL_size_t(1500, memProfilePeakBytes());

    freeReservedMemory(a);
    TEST_ASSERT_EQUAL_size_t(500, memProfileCurrentBytes());
    TEST_ASSERT_EQUAL_size_t(1500, memProfilePeakBytes()); /* peak holds */

    void *c = reserveMemory(2000);
    TEST_ASSERT_EQUAL_size_t(2500, memProfileCurrentBytes());
    TEST_ASSERT_EQUAL_size_t(2500, memProfilePeakBytes()); /* new peak */

    freeReservedMemory(b);
    freeReservedMemory(c);
    TEST_ASSERT_EQUAL_size_t(0, memProfileCurrentBytes());
    TEST_ASSERT_EQUAL_size_t(2500, memProfilePeakBytes());
}

/* The size-header must preserve alignment + writability of the full request. */
void testReservedRegionUsableAndAligned(void) {
    memProfileReset();
    size_t n = 777;
    unsigned char *p = reserveMemory(n);
    TEST_ASSERT_EQUAL_UINT(0, (uintptr_t)p % _Alignof(max_align_t));
    for (size_t i = 0; i < n; i++) {
        p[i] = (unsigned char)(i & 0xFF); /* full-range write; ASan guards OOB */
    }
    TEST_ASSERT_EQUAL_size_t(n, memProfileCurrentBytes());
    freeReservedMemory(p);
    TEST_ASSERT_EQUAL_size_t(0, memProfileCurrentBytes());
}
#endif /* ODT_MEM_PROFILE */

void setUp() {}
void tearDown() {}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testReserveMemoryReturnsZeroedWritablePayload);
    RUN_TEST(testReserveAndFreeRoundTrip);
    RUN_TEST(testFreeReservedMemoryIsNullSafe);
    RUN_TEST(testReserveMemoryOfZeroBytesReturns);
#if defined(ODT_MEM_PROFILE) || !defined(ODT_TEST_ASAN)
    RUN_TEST(testReserveMemoryExitsOnAWrappingSize);
#endif
#if SIZE_MAX > UINT32_MAX && !defined(ODT_TEST_ASAN)
    RUN_TEST(testReserveMemoryExitsWhenTheAllocationFails);
#endif
#ifdef ODT_MEM_PROFILE
    RUN_TEST(testHeapCounterExactCurrentAndPeak);
    RUN_TEST(testReservedRegionUsableAndAligned);
#endif
    return UNITY_END();
}
