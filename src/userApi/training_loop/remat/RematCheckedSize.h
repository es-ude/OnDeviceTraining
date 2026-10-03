#ifndef ODT_REMAT_CHECKED_SIZE_H
#define ODT_REMAT_CHECKED_SIZE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* Every size product and sum the remat libraries
 * compute goes through these, and the caller exits by name on false. Private
 * to src/userApi/training_loop/remat/ (PR1b's RematPlace.h reuses it). The
 * GNU builtins are the primary path (gcc >= 5, clang, arm-none-eabi-gcc);
 * the SIZE_MAX tests are the portable fallback for other compilers. */
#if defined(__has_builtin)
#if __has_builtin(__builtin_mul_overflow) && __has_builtin(__builtin_add_overflow)
#define REMAT_HAVE_OVERFLOW_BUILTINS 1
#endif
#endif
#if !defined(REMAT_HAVE_OVERFLOW_BUILTINS) && defined(__GNUC__) && __GNUC__ >= 5
#define REMAT_HAVE_OVERFLOW_BUILTINS 1
#endif

static inline bool checkedMulSize(size_t a, size_t b, size_t *out) {
#ifdef REMAT_HAVE_OVERFLOW_BUILTINS
    return !__builtin_mul_overflow(a, b, out);
#else
    if (a != 0u && SIZE_MAX / a < b) {
        return false;
    }
    *out = a * b;
    return true;
#endif
}

static inline bool checkedAddSize(size_t a, size_t b, size_t *out) {
#ifdef REMAT_HAVE_OVERFLOW_BUILTINS
    return !__builtin_add_overflow(a, b, out);
#else
    if (SIZE_MAX - a < b) {
        return false;
    }
    *out = a + b;
    return true;
#endif
}

#endif // ODT_REMAT_CHECKED_SIZE_H
