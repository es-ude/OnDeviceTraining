#ifndef GROUP_CURSOR_H
#define GROUP_CURSOR_H

#include <stddef.h>
#include <stdint.h>

/*! Group index of a storage index (`groupSize == 0 ? 0 : idx / groupSize`),
 * dividing only when idx leaves the cached group's range. Use it in per-MAC
 * loops: any visit order (gaps, backward hops) returns the exact quotient.
 * groupSize 0 is the per-tensor sentinel (one group over everything). */
typedef struct groupCursor {
    size_t groupSize;
    size_t group;
    size_t lo;
    size_t hi;
} groupCursor_t;

static inline groupCursor_t groupCursorInit(size_t groupSize) {
    groupCursor_t c = {groupSize, 0, 0, groupSize == 0 ? SIZE_MAX : 0};
    return c;
}

static inline size_t groupCursorOf(groupCursor_t *c, size_t idx) {
    /* One unsigned compare covers both sides of [lo, hi). */
    if (idx - c->lo >= c->hi - c->lo) {
        /* Sentinel [0, SIZE_MAX) is left only at idx == SIZE_MAX. */
        if (c->groupSize == 0) {
            return 0;
        }
        c->group = idx / c->groupSize;
        c->lo = c->group * c->groupSize;
        /* Clamp, never wrap: a wrapped hi would alias small indices into the
         * last (partial) group. SIZE_MAX itself then just recomputes. */
        c->hi = c->lo <= SIZE_MAX - c->groupSize ? c->lo + c->groupSize : SIZE_MAX;
    }
    return c->group;
}

#endif
