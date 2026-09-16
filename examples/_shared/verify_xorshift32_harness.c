#define SOURCE_FILE "verify_xorshift32_harness"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "RNG.h"

/* Compile alongside src/rng/RNG.c. Usage: ./harness <n> <seed> [<k>]
 * Seeds, shuffles [0, n) once (dataLoaderInit's init shuffle), then applies
 * k (default 0) further rngShuffleIndices calls on the CURRENT permutation
 * WITHOUT reseeding (dataLoaderReshuffle, once per epoch > 0). Emits the
 * final permutation to stdout, space-separated, one trailing newline.
 */
int main(int argc, char **argv) {
    if (argc != 3 && argc != 4) {
        fprintf(stderr, "usage: %s <n> <seed> [<k>]\n", argv[0]);
        return 1;
    }
    size_t n = (size_t)strtoull(argv[1], NULL, 10);
    uint32_t seed = (uint32_t)strtoul(argv[2], NULL, 10);
    size_t k = (argc == 4) ? (size_t)strtoull(argv[3], NULL, 10) : 0;

    size_t *indices = malloc(n * sizeof(*indices));
    if (!indices) {
        return 2;
    }
    for (size_t i = 0; i < n; ++i) {
        indices[i] = i;
    }

    rngSetSeed(seed);
    rngShuffleIndices(indices, n);
    for (size_t r = 0; r < k; ++r) {
        rngShuffleIndices(indices, n);
    }

    for (size_t i = 0; i < n; ++i) {
        printf("%zu%c", indices[i], (i + 1 == n) ? '\n' : ' ');
    }
    free(indices);
    return 0;
}
