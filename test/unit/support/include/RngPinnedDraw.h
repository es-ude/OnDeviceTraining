#ifndef ODT_TEST_RNG_PINNED_DRAW_H
#define ODT_TEST_RNG_PINNED_DRAW_H

#include <stdint.h>

#include "RNG.h"
#include "unity.h"

/* Lets a test pin the SR uniform exactly: a stream whose next rngNextU32 draw
 * is a chosen value. Header-only; a consumer links odt_test_rng_pin and RNG. */

/* Inverse of one rngNextU32 step (xorshift32 13/17/5): the state whose NEXT draw is `draw`. */
static inline uint32_t rngStateWhoseNextDrawIs(uint32_t draw) {
    uint32_t b = draw ^ (draw << 5) ^ (draw << 10) ^ (draw << 15) ^ (draw << 20) ^ (draw << 25) ^
                 (draw << 30);
    uint32_t a = b ^ (b >> 17);
    return a ^ (a << 13) ^ (a << 26);
}

/* A fresh stream whose first draw is `draw`. The probe assert checks the
 * inversion on every use, so the helper cannot be silently wrong. */
static inline rngStream_t rngStreamWithNextDraw(uint32_t draw) {
    const uint32_t state = rngStateWhoseNextDrawIs(draw);
    TEST_ASSERT_NOT_EQUAL_UINT32(0u, state);
    rngStream_t probe = rngOpen((rngKey_t){{state, 0u}});
    TEST_ASSERT_EQUAL_UINT32(draw, rngNextU32(&probe));
    return rngOpen((rngKey_t){{state, 0u}});
}

#endif /* ODT_TEST_RNG_PINNED_DRAW_H */
