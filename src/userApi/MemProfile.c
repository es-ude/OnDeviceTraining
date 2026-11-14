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
    /* pthread_attr_setstack needs a page-aligned address and size on some
     * platforms (Apple libpthread enforces both). */
    long pageSizeRaw = sysconf(_SC_PAGESIZE);
    if (pageSizeRaw <= 0) {
        fprintf(stderr, "MEM_PROFILE: sysconf(_SC_PAGESIZE) failed — failing loud\n");
        exit(1);
    }
    size_t pageSize = (size_t)pageSizeRaw;

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
    /* A pthread failure invalidates the measurement: a rejected region runs fn
     * on the default stack (a silent 0), a failed create leaves th unset. */
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
        /* Return 0 when RSS is unavailable. */
        return 0;
    }
    /* ru_maxrss is KiB on Linux, bytes on macOS/Darwin. */
#if defined(__APPLE__)
    return (size_t)ru.ru_maxrss / 1024u;
#else
    return (size_t)ru.ru_maxrss;
#endif
}
