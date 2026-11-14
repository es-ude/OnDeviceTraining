#ifndef STORAGEAPI_H
#define STORAGEAPI_H

#include <stddef.h>

/* Zeroed memory. Exits (PRINT_ERROR + exit(1)) when the request cannot be met,
 * so callers never check for NULL; a 0-byte request may return NULL (#160). */
void *reserveMemory(size_t numberOfBytes);

/* NULL-safe (mirrors free(NULL)). */
void freeReservedMemory(void *ptr);

/* Memory profiling (active only when ODT_MEM_PROFILE is defined; otherwise
 * these are no-ops returning 0). Counts user bytes requested via
 * reserveMemory, net of freeReservedMemory. memProfileMark returns the current
 * live total for phase-boundary diffing. */
void memProfileReset(void);
size_t memProfileCurrentBytes(void);
size_t memProfilePeakBytes(void);
size_t memProfileMark(void);

#endif // STORAGEAPI_H
