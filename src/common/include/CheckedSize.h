#ifndef ODT_CHECKED_SIZE_H
#define ODT_CHECKED_SIZE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* Size products and sums over untrusted or user-supplied counts (files,
 * configs, caller arguments) go through these; on false the caller fails
 * fast by name. True means *out holds the exact result. False means the
 * result wrapped and *out is unspecified (the builtins store the wrapped
 * value), so a caller never reads *out after false. */
#if defined(__has_builtin)
#if __has_builtin(__builtin_mul_overflow) && __has_builtin(__builtin_add_overflow)
#define ODT_HAVE_OVERFLOW_BUILTINS 1
#endif
#endif
#if !defined(ODT_HAVE_OVERFLOW_BUILTINS) && defined(__GNUC__) && __GNUC__ >= 5
#define ODT_HAVE_OVERFLOW_BUILTINS 1
#endif
#ifndef ODT_HAVE_OVERFLOW_BUILTINS
// clang-format off
#error "CheckedSize.h requires __builtin_mul_overflow/__builtin_add_overflow (GNU-compatible compiler)"
// clang-format on
#endif

static inline bool checkedMulSize(size_t a, size_t b, size_t *out) {
    return !__builtin_mul_overflow(a, b, out);
}

static inline bool checkedAddSize(size_t a, size_t b, size_t *out) {
    return !__builtin_add_overflow(a, b, out);
}

#endif // ODT_CHECKED_SIZE_H
