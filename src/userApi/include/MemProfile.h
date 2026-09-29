#ifndef MEM_PROFILE_H
#define MEM_PROFILE_H

#include <stddef.h>

/* Runs fn(arg) on a fresh pthread whose stack is a stackBytes region painted
 * with a sentinel, then returns the peak bytes touched (stack high-water).
 * stackBytes is rounded up to a whole page internally (pthread_attr_setstack
 * requires a page-aligned region on some platforms). Host-only measurement
 * apparatus; the region uses a raw page-aligned allocation, NOT
 * reserveMemory, so it never enters the heap counter. */
size_t measurePeakStackBytes(void (*fn)(void *), void *arg, size_t stackBytes);

/* Peak resident set size (KiB) from getrusage(RUSAGE_SELF). Includes
 * allocator/libc/dataset — a coarse process-level anchor, NOT the MCU number. */
size_t memProfileRssPeakKb(void);

#endif
