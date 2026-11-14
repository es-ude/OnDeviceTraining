#define SOURCE_FILE "STORAGE_API"

#include <stdlib.h>

#include "Common.h"
#include "StorageApi.h"

static void exitUnreservable(size_t numberOfBytes) {
    PRINT_ERROR("reserveMemory: cannot reserve %zu bytes", numberOfBytes);
    exit(1);
}

#ifdef ODT_MEM_PROFILE
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>

/* Prepended to every allocation; a union with max_align_t forces the user
 * region (immediately after) to keep max alignment. Stores the user size so
 * free() can decrement the exact byte count. */
typedef union memHeader {
    size_t size;
    max_align_t align;
} memHeader_t;

static atomic_size_t g_currentBytes = 0;
static atomic_size_t g_peakBytes = 0;

static void bumpPeak(size_t cur) {
    size_t prev = atomic_load_explicit(&g_peakBytes, memory_order_relaxed);
    while (cur > prev &&
           !atomic_compare_exchange_weak_explicit(&g_peakBytes, &prev, cur, memory_order_relaxed,
                                                  memory_order_relaxed)) {
        /* prev is reloaded by the CAS on failure */
    }
}

void *reserveMemory(size_t numberOfBytes) {
    if (numberOfBytes > SIZE_MAX - sizeof(memHeader_t)) {
        exitUnreservable(numberOfBytes); /* the header would wrap the size_t */
    }
    memHeader_t *base = calloc(1, sizeof(memHeader_t) + numberOfBytes);
    if (base == NULL) {
        exitUnreservable(numberOfBytes);
    }
    base->size = numberOfBytes;
    size_t cur = atomic_fetch_add_explicit(&g_currentBytes, numberOfBytes, memory_order_relaxed) +
                 numberOfBytes;
    bumpPeak(cur);
    return (void *)(base + 1);
}

void freeReservedMemory(void *ptr) {
    if (ptr == NULL) {
        return;
    }
    memHeader_t *base = (memHeader_t *)ptr - 1;
    atomic_fetch_sub_explicit(&g_currentBytes, base->size, memory_order_relaxed);
    free(base);
}

void memProfileReset(void) {
    atomic_store_explicit(&g_currentBytes, 0, memory_order_relaxed);
    atomic_store_explicit(&g_peakBytes, 0, memory_order_relaxed);
}
size_t memProfileCurrentBytes(void) {
    return atomic_load_explicit(&g_currentBytes, memory_order_relaxed);
}
size_t memProfilePeakBytes(void) {
    return atomic_load_explicit(&g_peakBytes, memory_order_relaxed);
}
size_t memProfileMark(void) {
    return memProfileCurrentBytes();
}

#else /* !ODT_MEM_PROFILE */

void *reserveMemory(size_t numberOfBytes) {
    void *p = calloc(1, numberOfBytes);
    /* Size 0 is not a failure: calloc(1, 0) may return NULL on newlib (#160). */
    if (p == NULL && numberOfBytes != 0) {
        exitUnreservable(numberOfBytes);
    }
    return p;
}
void freeReservedMemory(void *ptr) {
    free(ptr);
}
void memProfileReset(void) {}
size_t memProfileCurrentBytes(void) {
    return 0;
}
size_t memProfilePeakBytes(void) {
    return 0;
}
size_t memProfileMark(void) {
    return 0;
}

#endif
