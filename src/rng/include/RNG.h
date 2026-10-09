#ifndef RNG_H
#define RNG_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    uint32_t state;
} rng32_t;

/* Keyed randomness. A key names one node of a key tree, and every random value
 * is a pure function of a key and a position, so values do not depend on call
 * order, thread or recomputation. The functions in this block keep no module
 * state: safe from any thread, with caller-owned streams. */
typedef struct rngKey {
    uint32_t w[2];
} rngKey_t;

/* One edge of the key tree: one Threefry-2x32-20 block (Salmon et al., SC'11)
 * with the parent as the Threefry key and (tag, index) as the counter; the
 * output block is the child. Its values are part of the value contract: a
 * change moves every keyed value, so they are fixed by known-answer tests. */
rngKey_t rngFold(rngKey_t parent, uint32_t tag, uint32_t index);

// NOTE: not thread-safe — all functions below use module-global RNG state.
// When multi-threading support is added, migrate to context-passing variants.

void rngShuffleIndices(size_t *indices, size_t n);

void rngSetSeed(uint32_t seed);

uint32_t rngGetSeed(void);

/* Context-passing variant: same draw math as rngNextFloat() on caller-owned
 * state. Use for streams that must not perturb the module-global stream
 * (e.g. PPCA replay sampling, #326). The global functions delegate here. */
float rngNextFloatCtx(rng32_t *rng);

float rngNextFloat(void);

#endif // RNG_H
