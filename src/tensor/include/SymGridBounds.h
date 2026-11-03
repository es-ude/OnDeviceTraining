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

/* Grid bounds of a SYM / SYM_INT32 width, formed in int32 (#515).
 *
 * Why not powf(2, q - 1) - 1: float32 carries 24 significant bits, so from
 * q = 26 on the grid maximum is inexact and rounds UP to 2^(q-1) -- one code
 * past the grid. A clamp against that bound emits a code the declared width
 * says cannot exist, and the SYM accumulator guard proves its int32 budget
 * from exactly that declared width. Sites keep (float)hi for deriving a
 * scale or pre-clamping a quotient (the same value as before for q <= 25);
 * the integer clamp after rounding is what pulls a wide-width code back onto
 * the grid.
 *
 * Why 31 is the ceiling (#202): 2^(q-1) has to exist in int32 -- for the
 * shift that forms it and for the cast back from the float pre-clamp bound
 * (float)hi, which rounds up to 2^(q-1) for q >= 26 and must stay inside
 * int32 after the quotient is rounded. 2^30 does, 2^31 does not. q = 0
 * would shift by -1. Every reachable width passes through this check before
 * the shift, so a config that escaped its init validation still fails fast
 * here, by `what`. */
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
 * check; any other input needs the derived-scale check. One validation. */
static inline symGridBounds_t symGridBoundsForAbsMaxOrExit(uint32_t q, const char *what,
                                                           float absMax) {
    return (absMax == 0.f) ? symGridBoundsOrExit(q, what)
                           : symGridBoundsForDerivedScaleOrExit(q, what);
}

#endif /* ODT_SYM_GRID_BOUNDS_H */
