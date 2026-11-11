#ifndef ODT_SYM_GROUP_CURSOR_H
#define ODT_SYM_GROUP_CURSOR_H

#include <stdbool.h>
#include <stddef.h>

/*! Storage range [start, end) of weight quantization group `group`, for the
 * grouped SYM conv kernels. Their visited weight indices come in contiguous
 * runs (one kernel row's valid taps) separated by gaps where padding clips
 * taps, so a kernel splits each run at `end` and the MAC loop between splits
 * needs no group lookup at all. Zero-initialise ({0}) for the empty range, so
 * the first lookup recomputes. The caller guarantees groupSize > 0 (the
 * grouped entries reject the per-tensor sentinel). Same idea as BFP's
 * bfpGroupAt (BfpKernelSupport.h), without its per-tensor branch. */
typedef struct {
    size_t group;
    size_t start;
    size_t end;
} symGroupCursor_t;

/*! True when idx lies in the cached group. One unsigned compare covers both
 * directions: an idx below start wraps to a huge offset. */
static inline bool symGroupCursorHolds(const symGroupCursor_t *c, size_t idx) {
    return idx - c->start < c->end - c->start;
}

/*! Moves the cursor to idx's group: the only division, run once per group
 * change instead of once per MAC (an M0+ has no hardware divide). */
static inline void symGroupCursorSeek(symGroupCursor_t *c, size_t idx, size_t groupSize) {
    c->group = idx / groupSize;
    c->start = c->group * groupSize;
    c->end = c->start + groupSize;
}

#endif // ODT_SYM_GROUP_CURSOR_H
