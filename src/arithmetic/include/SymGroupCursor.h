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

/*! One quantization-group piece of a weight row [rowStart, rowStart + rows * K),
 * the row viewed as `rows` channels of K taps. The piece spans (chLo, kLo) up to
 * (chHi, kHi) exclusive in (channel, tap) order; the row end is (rows, 0). */
typedef struct {
    size_t group;
    size_t chLo, kLo;
    size_t chHi, kHi;
} symGroupRowPiece_t;

/*! The per-call constants of the row walk: a group spans stepCh channels plus
 * stepK taps, so a boundary steps to the next one without a divide. */
typedef struct {
    size_t groupSize, kernelSize, rows;
    size_t stepCh, stepK;
} symGroupRowWalk_t;

/*! Once per kernel call. groupSize, kernelSize and rows are > 0. */
static inline symGroupRowWalk_t symGroupRowWalkInit(size_t groupSize, size_t kernelSize,
                                                    size_t rows) {
    return (symGroupRowWalk_t){.groupSize = groupSize,
                               .kernelSize = kernelSize,
                               .rows = rows,
                               .stepCh = groupSize / kernelSize,
                               .stepK = groupSize % kernelSize};
}

/*! The row's first piece: the one divide per row. The first boundary's tap
 * offset (< rows * K) becomes (channel, tap) by subtraction, at most rows - 1
 * steps, so that a second divide is not needed. */
static inline symGroupRowPiece_t symGroupRowFirstPiece(const symGroupRowWalk_t *w,
                                                       size_t rowStart) {
    symGroupRowPiece_t p = {.group = rowStart / w->groupSize, .chLo = 0, .kLo = 0};
    size_t toEnd = (p.group + 1) * w->groupSize - rowStart;
    if (toEnd >= w->rows * w->kernelSize) {
        p.chHi = w->rows;
        p.kHi = 0;
        return p;
    }
    size_t ch = 0;
    while (toEnd >= w->kernelSize) {
        toEnd -= w->kernelSize;
        ch++;
    }
    p.chHi = ch;
    p.kHi = toEnd;
    return p;
}

/*! Advances to the next piece of the same row; false at the row end. */
static inline bool symGroupRowNextPiece(const symGroupRowWalk_t *w, symGroupRowPiece_t *p) {
    if (p->chHi >= w->rows) {
        return false;
    }
    p->group++;
    p->chLo = p->chHi;
    p->kLo = p->kHi;
    size_t ch = p->chLo + w->stepCh;
    size_t k = p->kLo + w->stepK;
    if (k >= w->kernelSize) {
        k -= w->kernelSize;
        ch++;
    }
    if (ch >= w->rows) {
        ch = w->rows;
        k = 0;
    }
    p->chHi = ch;
    p->kHi = k;
    return true;
}

#endif // ODT_SYM_GROUP_CURSOR_H
