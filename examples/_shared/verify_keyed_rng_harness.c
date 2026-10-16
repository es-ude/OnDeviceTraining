#define SOURCE_FILE "verify_keyed_rng_harness"

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "RNG.h"

/* Compile alongside src/rng/RNG.c with -I src/rng/include -I src/common/include
 * -DDEBUG_MODE_ERROR=true (as CMake's Common target, so a bad bound names itself).
 * One command per run, numbers in decimal or 0x-hex, one line of output:
 *   key <seed>                      the root key: w0 w1 (hex)
 *   fold <w0> <w1> <tag> <index>    the child key: w0 w1 (hex)
 *   u32 <w0> <w1> <count>           count draws of the key's stream (hex)
 *   float <w0> <w1> <count>         count unit floats as bit patterns (hex)
 *   bounded <w0> <w1> <bound>...    one bounded draw per bound on one stream (decimal)
 *   shuffle <w0> <w1> <n>           0 ... n-1 shuffled over the key's stream by Fisher-Yates
 *                                   from the last slot down (decimal); exit 2 with a
 *                                   message when n indices do not fit in memory
 */
static unsigned long long number(const char *text) {
    return strtoull(text, NULL, 0);
}

static rngKey_t keyOf(char **words) {
    return (rngKey_t){{(uint32_t)number(words[0]), (uint32_t)number(words[1])}};
}

static void printKey(rngKey_t key) {
    printf("%08" PRIx32 " %08" PRIx32 "\n", key.w[0], key.w[1]);
}

int main(int argc, char **argv) {
    if (argc == 3 && strcmp(argv[1], "key") == 0) {
        printKey(rngKeyFromSeed((uint64_t)number(argv[2])));
        return 0;
    }
    if (argc == 6 && strcmp(argv[1], "fold") == 0) {
        printKey(rngFold(keyOf(argv + 2), (uint32_t)number(argv[4]), (uint32_t)number(argv[5])));
        return 0;
    }
    if (argc == 5 && (strcmp(argv[1], "u32") == 0 || strcmp(argv[1], "float") == 0)) {
        rngStream_t stream = rngOpen(keyOf(argv + 2));
        const unsigned long long count = number(argv[4]);
        for (unsigned long long i = 0; i < count; ++i) {
            uint32_t bits;
            if (argv[1][0] == 'u') {
                bits = rngNextU32(&stream);
            } else {
                const float value = rngNextUnitFloat(&stream);
                memcpy(&bits, &value, sizeof bits);
            }
            printf("%08" PRIx32 "%c", bits, (i + 1 == count) ? '\n' : ' ');
        }
        return 0;
    }
    if (argc >= 5 && strcmp(argv[1], "bounded") == 0) {
        rngStream_t stream = rngOpen(keyOf(argv + 2));
        for (int i = 4; i < argc; ++i) {
            printf("%zu%c", rngNextBounded(&stream, (size_t)number(argv[i])),
                   (i + 1 == argc) ? '\n' : ' ');
        }
        return 0;
    }
    if (argc == 5 && strcmp(argv[1], "shuffle") == 0) {
        rngStream_t stream = rngOpen(keyOf(argv + 2));
        const unsigned long long count = number(argv[4]);
        if (count > SIZE_MAX / sizeof(size_t)) {
            fprintf(stderr, "shuffle: n = %llu would overflow the buffer size\n", count);
            return 2;
        }
        const size_t n = (size_t)count;
        size_t *order = malloc((n > 0 ? n : 1) * sizeof *order);
        if (order == NULL) {
            fprintf(stderr, "shuffle: cannot allocate %zu indices\n", n);
            return 2;
        }
        for (size_t i = 0; i < n; ++i) {
            order[i] = i;
        }
        if (n >= 2) {
            for (size_t i = n - 1; i > 0; --i) {
                const size_t j = rngNextBounded(&stream, i + 1);
                const size_t swapped = order[i];
                order[i] = order[j];
                order[j] = swapped;
            }
        }
        for (size_t i = 0; i < n; ++i) {
            printf("%zu ", order[i]);
        }
        printf("\n");
        free(order);
        return 0;
    }
    fprintf(stderr, "usage: %s key|fold|u32|float|bounded|shuffle ...\n", argv[0]);
    return 1;
}
