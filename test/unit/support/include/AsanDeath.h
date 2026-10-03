#ifndef ODT_TEST_ASAN_DEATH_H
#define ODT_TEST_ASAN_DEATH_H

/*
 * Death-test support for AddressSanitizer reports (#4 remat).
 *
 * The unit_test_asan test preset sets abort_on_error=1 (CMakePresets.json:251),
 * so an ASan report ends in SIGABRT, which a death test can only read as
 * "terminated by signal". Install the callback INSIDE the death-test
 * statement (the forked child): the sanitizer runtime calls it before it
 * honours abort_on_error, and it _exit()s with ODT_ASAN_DEATH_EXIT -- distinct
 * from 1 (every named guard's exit code) and 0 (the statement returned), so a
 * test expecting a named exit that hits ASan instead reads
 * "Expected 1 Was 86". _exit discards buffered stdout; the report goes to
 * stderr, which the death-test macros silence anyway.
 *
 * Outside ASan the install is a no-op, so tests that use it build and run on
 * every preset; only their ASan evidence is asan-specific.
 */

#define ODT_ASAN_DEATH_EXIT 86

#if defined(__SANITIZE_ADDRESS__)
#define ODT_TEST_ASAN 1
#elif defined(__has_feature)
#if __has_feature(address_sanitizer)
#define ODT_TEST_ASAN 1
#endif
#endif

#ifdef ODT_TEST_ASAN
#include <unistd.h>

/* Forward-declared, not included from <sanitizer/common_interface_defs.h>:
 * that header's install path is compiler-specific (AppleClang exposes it,
 * the project's pinned devenv clang 22 does not), while this function's
 * signature is a long-stable part of the compiler-rt ASan runtime ABI, so
 * declaring it ourselves is both simpler than an #if __has_include fallback
 * and portable across every ASan-capable compiler this project builds with. */
void __sanitizer_set_death_callback(void (*callback)(void));

static void odtAsanDeathExit(void) {
    _exit(ODT_ASAN_DEATH_EXIT);
}

static inline void odtInstallAsanDeathExit(void) {
    __sanitizer_set_death_callback(odtAsanDeathExit);
}
#else
static inline void odtInstallAsanDeathExit(void) {}
#endif

#endif /* ODT_TEST_ASAN_DEATH_H */
