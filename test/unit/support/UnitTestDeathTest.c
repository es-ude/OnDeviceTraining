#define SOURCE_FILE "UNIT_TEST_DEATH_TEST"

#include "Common.h"
#include "DeathTest.h"
#include "unity.h"

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

/* Fastest RED, identical on every compiler: without the macro a call is an
 * implicit declaration (clang: error; gcc: warning, then a void-argument
 * error). */
#ifndef ASSERT_EXITS_WITH_OUTPUT
#error "DeathTest.h must define ASSERT_EXITS_WITH_OUTPUT(code, substr, statement)"
#endif

void setUp(void) {}
void tearDown(void) {}

/* Exit status of a death-test child whose body failed a Unity assertion.
 * Distinct from 0 (the body passed) and 1 (the framework's fail-fast code). */
#define DEATH_TEST_UNITY_FAILED_EXIT 3

/* The framework's fail-fast idiom in the remat violation format:
 * a PRINT_ERROR banner on stdout, then exit(). */
static void violate(const char *rule, int code) {
    PRINT_ERROR("remat[ARENA]: step #3 BACKWARD(layer 1) violates '%s'", rule);
    exit(code);
}

/* Runs a body whose inner ASSERT_EXITS_WITH_OUTPUT is expected to FAIL. The
 * failure longjmps to this TEST_PROTECT instead of into the runner, Unity's
 * FAIL line is flushed to this child's stdout, and the exit status says
 * whether the body failed. The outer macro then checks both. */
static void runExpectingUnityFailure(void (*body)(void)) {
    if (TEST_PROTECT()) {
        body();
    }
    (void)fflush(stdout);
    _exit(Unity.CurrentTestFailed ? DEATH_TEST_UNITY_FAILED_EXIT : 0);
}

void testExitsWithOutputPassesOnMatchingCodeAndSubstring(void) {
    ASSERT_EXITS_WITH_OUTPUT(1, "violates 'residency'", violate("residency", 1));
}

static void substringAbsentBody(void) {
    ASSERT_EXITS_WITH_OUTPUT(1, "'residency'", violate("aliasing", 1));
}

void testExitsWithOutputFailsWhenSubstringAbsent(void) {
    ASSERT_EXITS_WITH_OUTPUT(DEATH_TEST_UNITY_FAILED_EXIT, "stdout lacks",
                             runExpectingUnityFailure(substringAbsentBody));
}

static void exitCodeDiffersBody(void) {
    ASSERT_EXITS_WITH_OUTPUT(1, "'residency'", violate("residency", 2));
}

void testExitsWithOutputFailsWhenExitCodeDiffers(void) {
    ASSERT_EXITS_WITH_OUTPUT(DEATH_TEST_UNITY_FAILED_EXIT, "exit code mismatch",
                             runExpectingUnityFailure(exitCodeDiffersBody));
}

static void printThenDieBySignal(void) {
    printf("violates 'residency'\n");
    (void)fflush(stdout);
    (void)raise(SIGTERM);
}

/* Expected code 0 on purpose: a signal death has WEXITSTATUS 0, so only the
 * exited-not-signalled check can reject it. SIGTERM: ASan does not intercept
 * it and it leaves no core file. */
static void killedBySignalBody(void) {
    ASSERT_EXITS_WITH_OUTPUT(0, "'residency'", printThenDieBySignal());
}

void testExitsWithOutputFailsWhenChildDiesBySignal(void) {
    ASSERT_EXITS_WITH_OUTPUT(DEATH_TEST_UNITY_FAILED_EXIT, "terminated by signal",
                             runExpectingUnityFailure(killedBySignalBody));
}

/* 256 KiB before the banner, more than any pipe buffer (Linux 64 KiB, macOS
 * 16-64 KiB): the parent must drain while the child runs, and the match must
 * survive the bounded capture window. */
static void floodThenViolate(void) {
    for (int line = 0; line < 4096; line++) {
        printf("%063d\n", line);
    }
    violate("residency", 1);
}

void testExitsWithOutputFindsBannerAfterFloodingThePipe(void) {
    ASSERT_EXITS_WITH_OUTPUT(1, "'residency'", floodThenViolate());
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testExitsWithOutputPassesOnMatchingCodeAndSubstring);
    RUN_TEST(testExitsWithOutputFailsWhenSubstringAbsent);
    RUN_TEST(testExitsWithOutputFailsWhenExitCodeDiffers);
    RUN_TEST(testExitsWithOutputFailsWhenChildDiesBySignal);
    RUN_TEST(testExitsWithOutputFindsBannerAfterFloodingThePipe);
    return UNITY_END();
}
