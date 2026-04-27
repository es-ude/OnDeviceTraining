#ifndef RNG_H
#define RNG_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    uint32_t state;
} rng32_t;

// NOTE: not thread-safe — all functions below use module-global RNG state.
// When multi-threading support is added, migrate to context-passing variants.

void rngShuffleIndices(size_t *indices, size_t n);

void rngSetSeed(uint32_t seed);

uint32_t rngGetSeed(void);

float rngNextFloat(void);

#endif // RNG_H
