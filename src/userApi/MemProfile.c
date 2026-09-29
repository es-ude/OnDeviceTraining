#define SOURCE_FILE "MEM_PROFILE"

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <unistd.h>

#include "MemProfile.h"

#define MEM_STACK_PAINT 0xA5u

typedef struct stackThunk {
    void (*fn)(void *);
    void *arg;
} stackThunk_t;

static void *stackThunkRunner(void *p) {
    stackThunk_t *t = (stackThunk_t *)p;
    t->fn(t->arg);
    return NULL;
}

size_t measurePeakStackBytes(void (*fn)(void *), void *arg, size_t stackBytes) {
    /* pthread_attr_setstack's region must be page-aligned in BOTH address and
     * size: POSIX permits (but does not require) that, and Apple libpthread
     * enforces it (glibc only checks size). A raw malloc() is not guaranteed
     * to return a page-aligned address, and stackBytes is caller-chosen, so
     * both need adjusting before the region reaches pthread_attr_setstack. */
    long pageSizeRaw = sysconf(_SC_PAGESIZE);
    if (pageSizeRaw <= 0) {
        fprintf(stderr, "MEM_PROFILE: sysconf(_SC_PAGESIZE) failed — failing loud\n");
        exit(1);
    }
    size_t pageSize = (size_t)pageSizeRaw;

    /* Round stackBytes up to a whole page. */
    size_t remainder = stackBytes % pageSize;
    size_t pad = remainder == 0 ? 0 : pageSize - remainder;
    if (stackBytes > SIZE_MAX - pad) {
        fprintf(stderr, "MEM_PROFILE: stackBytes overflowed while rounding up to a page "
                        "— failing loud\n");
        exit(1);
    }
    size_t roundedBytes = stackBytes + pad;

    /* raw allocation: measurement apparatus, deliberately not reserveMemory */
    void *regionRaw = NULL;
    int allocErr = posix_memalign(&regionRaw, pageSize, roundedBytes);
    if (allocErr != 0 || regionRaw == NULL) {
        fprintf(stderr,
                "MEM_PROFILE: posix_memalign for the stack region failed (rc=%d) — failing loud\n",
                allocErr);
        exit(1);
    }
    unsigned char *region = (unsigned char *)regionRaw;
    memset(region, (int)MEM_STACK_PAINT, roundedBytes);

    pthread_attr_t attr;
    pthread_attr_init(&attr);
    /* Fail loud on any pthread error: a rejected stack region (attr_setstack) would run the
     * workload on the DEFAULT stack, leaving `region` fully painted -> a silent used==0
     * measurement; a failed create would leave `th` uninitialized -> UB in join. Both are
     * worse than a crash for a measurement apparatus. */
    int setstackRc = pthread_attr_setstack(&attr, region, roundedBytes);
    if (setstackRc != 0) {
        fprintf(stderr,
                "MEM_PROFILE: pthread_attr_setstack rejected the stack region (rc=%d) "
                "— failing loud (measurement would be wrong)\n",
                setstackRc);
        exit(1);
    }

    stackThunk_t thunk = {.fn = fn, .arg = arg};
    pthread_t th;
    if (pthread_create(&th, &attr, stackThunkRunner, &thunk) != 0) {
        fprintf(stderr, "MEM_PROFILE: pthread_create failed — failing loud\n");
        exit(1);
    }
    if (pthread_join(th, NULL) != 0) {
        fprintf(stderr, "MEM_PROFILE: pthread_join failed — failing loud\n");
        exit(1);
    }
    pthread_attr_destroy(&attr);

    /* Stack grows down from the high end of [region, region+roundedBytes). The
     * untouched (still-painted) bytes are the low prefix; scan from the low end
     * for the first touched byte. used = roundedBytes - firstTouchedIndex. */
    size_t firstTouched = 0;
    while (firstTouched < roundedBytes && region[firstTouched] == MEM_STACK_PAINT) {
        firstTouched++;
    }
    size_t used = roundedBytes - firstTouched;
    free(region);
    return used;
}

size_t memProfileRssPeakKb(void) {
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) != 0) {
        /* Soft-fail sentinel (0 = "unavailable"): RSS is a COARSE SECONDARY anchor,
         * not a primary measurement like the heap counter / stack watermark. A 0
         * (never a real RSS) is more useful to a consumer than crashing the whole
         * run; contrast measurePeakStackBytes, which fails loud because a wrong
         * stack number silently corrupts the primary result. getrusage(RUSAGE_SELF)
         * essentially never fails in practice. */
        return 0;
    }
    /* ru_maxrss is KiB on Linux, bytes on macOS/Darwin. */
#if defined(__APPLE__)
    return (size_t)ru.ru_maxrss / 1024u;
#else
    return (size_t)ru.ru_maxrss;
#endif
}
