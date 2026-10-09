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

/* The root of a key tree: one Threefry block under the all-zero key over the
 * counter (low word, high word) of the seed. Every seed, 0 included, is an
 * ordinary seed. */
rngKey_t rngKeyFromSeed(uint64_t seed);

/* The draws under one key. Owned by its caller, usually on the stack of the op
 * that draws. Backend v1: keyed xorshift32, today's draw math. Start a stream
 * with rngOpen only: a hand-set state 0 would draw 0 forever. */
typedef struct rngStream {
    uint32_t state;
} rngStream_t;

/* v1 keeps one key word as the state: w[0], else w[1], else 1 (state 0 would
 * stick). Reopening a key replays its stream, which is how a recompute gets
 * the values of the first computation. */
rngStream_t rngOpen(rngKey_t key);

/* The next draw: one xorshift32 step (shifts 13, 17, 5), the math of
 * rngNextFloatCtx's step. */
uint32_t rngNextU32(rngStream_t *stream);

/* A float in [0, 1) from the next draw: the mapping of rngNextFloatCtx. */
float rngNextUnitFloat(rngStream_t *stream);

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
