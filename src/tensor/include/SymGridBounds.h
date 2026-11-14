#ifndef ODT_SYM_GRID_BOUNDS_H
#define ODT_SYM_GRID_BOUNDS_H

#include <stdint.h>
#include <stdlib.h>

#include "Common.h"

/* The code grid of a q-bit symmetric width: lo = -2^(q-1), hi = 2^(q-1) - 1. */
typedef struct symGridBounds {
    int32_t lo;
    int32_t hi;
} symGridBounds_t;

/* Grid bounds of a SYM / SYM_INT32 width, formed in int32. Not powf(2, q-1) - 1:
 * from q = 26 float rounds that up to 2^(q-1), one code past the grid, and the
 * accumulator guard trusts the declared width. Sites may keep (float)hi for a
 * scale or a pre-clamp, but clamp rounded codes against these integer bounds.
 * Widths [1, 31] keep the shift and the float-to-int32 cast representable; the
 * check stays because SYM_INT32 widths are not range-checked at every entry
 * (a zero width, or a raw kernel output of width 32). */
static inline symGridBounds_t symGridBoundsOrExit(uint32_t q, const char *what) {
    if (q == 0 || q > 31) {
        PRINT_ERROR("%s: SYM width (%u) outside [1, 31]", what, (unsigned)q);
        exit(1);
    }
    const int32_t half = (int32_t)1 << (q - 1);
    symGridBounds_t g = {.lo = -half, .hi = half - 1};
    return g;
}

/* As symGridBoundsOrExit, for a site that DERIVES the scale from data
 * (scale = absMax / hi). A width-1 grid is {-1, 0}: hi == 0 leaves no positive
 * code to map absMax onto, so the quotient is infinite. Width 1 stays legal
 * where the scale is fixed and for a declared kernel output (a zero-term
 * result is honestly width 1); call this only where absMax != 0 is divided. */
static inline symGridBounds_t symGridBoundsForDerivedScaleOrExit(uint32_t q, const char *what) {
    const symGridBounds_t g = symGridBoundsOrExit(q, what);
    if (g.hi == 0) {
        PRINT_ERROR("%s: SYM width (%u) has no positive code; a scale derived from data over it "
                    "is infinite (use width >= 2, or a fixed scale)",
                    what, (unsigned)q);
        exit(1);
    }
    return g;
}

/* Bounds for a site that divides by hi only when absMax != 0: an all-zero
 * input takes the neutral scale 1 and never divides, so it keeps the plain
 * check; any other input needs the derived-scale check. */
static inline symGridBounds_t symGridBoundsForAbsMaxOrExit(uint32_t q, const char *what,
                                                           float absMax) {
    return (absMax == 0.f) ? symGridBoundsOrExit(q, what)
                           : symGridBoundsForDerivedScaleOrExit(q, what);
}

#endif /* ODT_SYM_GRID_BOUNDS_H */
