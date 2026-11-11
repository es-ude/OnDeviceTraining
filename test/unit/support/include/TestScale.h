#ifndef ODT_TEST_SCALE_H
#define ODT_TEST_SCALE_H

#include "DyadicScale.h"
#include "unity.h"

/* A test fixture's stored scale from a float literal or expression. The
 * fixture value must already be canonical (DyadicScale.h): the conversion is
 * exact, so a test keeps the exact input it was written with. A
 * non-canonical value fails the test at the calling line instead of being
 * snapped silently. */
static inline odtScale_t odtTestScaleAt(float x, UNITY_LINE_TYPE line) {
    odtScale_t s = odtScaleFromFloat(x, ODT_SCALE_NEAREST);
    if (odtScaleToFloat(s) != x) {
        UNITY_TEST_FAIL(line, "odtTestScale: fixture scale is not a canonical dyadic value");
    }
    return s;
}

#define odtTestScale(x) odtTestScaleAt((x), __LINE__)

#endif /* ODT_TEST_SCALE_H */
