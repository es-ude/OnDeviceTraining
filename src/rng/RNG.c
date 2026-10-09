#define SOURCE_FILE "RNG"

#include <stdlib.h>

#include "Common.h"
#include "RNG.h"

static rng32_t rng = {.state = 1};

static uint32_t rngNext(rng32_t *rng) {
    uint32_t x = rng->state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    rng->state = x;
    return x;
}

static size_t rngBounded(rng32_t *rng, size_t bound) {
    uint32_t x;
    uint32_t limit = UINT32_MAX - (UINT32_MAX % bound);

    do {
        x = rngNext(rng);
    } while (x >= limit);

    return x % bound;
}

void rngShuffleIndices(size_t *indices, size_t n) {
    if (n < 2) {
        return;
    }

    for (size_t i = n - 1; i > 0; --i) {
        size_t j = rngBounded(&rng, i + 1);

        size_t tmp = indices[i];
        indices[i] = indices[j];
        indices[j] = tmp;
    }
}

void rngSetSeed(uint32_t seed) {
    rng.state = seed ? seed : UINT32_MAX;
}

uint32_t rngGetSeed(void) {
    return rng.state;
}

float rngNextFloatCtx(rng32_t *rngCtx) {
    /* 0x1p24f == 2^24 as a float literal: value-identical to the former
     * (float)(1 << 24), minus the (16-bit-int-only) UB of a wide untyped
     * shift -- the repo's one such site (PR #366 review). */
    return (float)(rngNext(rngCtx) >> 8) / 0x1p24f;
}

float rngNextFloat(void) {
    return rngNextFloatCtx(&rng);
}

/* Threefry-2x32 rotation constants; round i uses entry i % 8. */
static const unsigned threefryRotations[8] = {13, 15, 26, 6, 17, 29, 16, 24};

static uint32_t rotateLeft32(uint32_t x, unsigned r) {
    return (x << r) | (x >> (32u - r));
}

/* Add, rotate and xor only: no multiplier, which the M0+ lacks for the high
 * half. 20 rounds is Random123's default; 0x1BD11BDA is Threefry's key-schedule
 * parity constant. */
rngKey_t rngFold(rngKey_t parent, uint32_t tag, uint32_t index) {
    const uint32_t ks[3] = {parent.w[0], parent.w[1], parent.w[0] ^ parent.w[1] ^ 0x1BD11BDAu};
    uint32_t x0 = tag + ks[0];
    uint32_t x1 = index + ks[1];
    for (unsigned step = 0; step < 20u; ++step) {
        x0 += x1;
        x1 = rotateLeft32(x1, threefryRotations[step % 8u]) ^ x0;
        if (step % 4u == 3u) {
            const unsigned injection = (step + 1u) / 4u;
            x0 += ks[injection % 3u];
            x1 += ks[(injection + 1u) % 3u] + injection;
        }
    }
    return (rngKey_t){{x0, x1}};
}

rngKey_t rngKeyFromSeed(uint64_t seed) {
    return rngFold((rngKey_t){{0u, 0u}}, (uint32_t)seed, (uint32_t)(seed >> 32));
}

rngStream_t rngOpen(rngKey_t key) {
    const uint32_t state = key.w[0] ? key.w[0] : (key.w[1] ? key.w[1] : 1u);
    return (rngStream_t){state};
}

uint32_t rngNextU32(rngStream_t *stream) {
    uint32_t y = stream->state;
    y ^= y << 13;
    y ^= y >> 17;
    y ^= y << 5;
    stream->state = y;
    return y;
}

/* The top 24 bits fit a float's mantissa exactly, so the result never rounds
 * up to 1. */
float rngNextUnitFloat(rngStream_t *stream) {
    return (float)(rngNextU32(stream) >> 8) / 0x1p24f;
}

/* Today's rule, its >= and its residue bias included (a non-zero xorshift32
 * state never draws 0): identity with the global shuffle is the contract. */
size_t rngNextBounded(rngStream_t *stream, size_t bound) {
    /* Bound 0 has no valid draw. A draw is 32 bits wide, so a bound above
     * UINT32_MAX cannot be met: the 32-bit cast below would divide by zero
     * (2^32) or draw from a smaller range (2^32 + k gives [0, k)). Both are
     * caller bugs. */
    if (bound == 0) {
        PRINT_ERROR("rngNextBounded: bound is 0");
        exit(1);
    }
    if (bound > UINT32_MAX) {
        PRINT_ERROR("rngNextBounded: bound exceeds UINT32_MAX");
        exit(1);
    }
    const uint32_t bound32 = (uint32_t)bound;
    const uint32_t reject = UINT32_MAX - (UINT32_MAX % bound32);
    uint32_t draw;
    do {
        draw = rngNextU32(stream);
    } while (draw >= reject);
    return draw % bound32;
}
