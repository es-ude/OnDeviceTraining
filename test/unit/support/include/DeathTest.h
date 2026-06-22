#ifndef ODT_TEST_DEATH_TEST_H
#define ODT_TEST_DEATH_TEST_H

/*
 * Fork-based death-test harness.
 *
 * Asserts that a statement terminates the process via exit(<code>) - the idiom
 * the framework uses for fail-fast guards (PRINT_ERROR(...); exit(1)). The
 * statement runs in a forked child so a *missing* guard, which may dereference
 * out of bounds and crash, cannot take down the parent test process: the parent
 * only inspects the child's exit status.
 *
 * Host-only: relies on POSIX fork()/waitpid(). ODT unit tests run on the host
 * (Linux CI / macOS dev), never on the MCU, so this is always available here.
 *
 * Outcomes inspected by the parent:
 *   - child called exit(code)        -> WIFEXITED && WEXITSTATUS == code  (pass)
 *   - statement returned (no exit)   -> child _exit(0): code mismatch     (fail)
 *   - child died by signal (SIGSEGV) -> !WIFEXITED: clear failure message (fail)
 */

#include <stdio.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include "unity.h"

#define ASSERT_EXITS_WITH(expectedCode, statement)                                                 \
    do {                                                                                           \
        fflush(stdout);                                                                            \
        fflush(stderr);                                                                            \
        pid_t _odtDeathPid = fork();                                                               \
        TEST_ASSERT_MESSAGE(_odtDeathPid >= 0, "fork() failed in death test");                     \
        if (_odtDeathPid == 0) {                                                                   \
            /* Child: silence the guard's PRINT_ERROR banner, run, and - if the                    \
             * statement does NOT exit - report "no death" via exit code 0. */                     \
            (void)freopen("/dev/null", "w", stdout);                                               \
            (void)freopen("/dev/null", "w", stderr);                                               \
            statement;                                                                             \
            _exit(0);                                                                              \
        }                                                                                          \
        int _odtDeathStatus = 0;                                                                   \
        (void)waitpid(_odtDeathPid, &_odtDeathStatus, 0);                                          \
        TEST_ASSERT_TRUE_MESSAGE(WIFEXITED(_odtDeathStatus),                                       \
                                 "death-test child terminated by signal, expected exit()");        \
        TEST_ASSERT_EQUAL_INT_MESSAGE((expectedCode), WEXITSTATUS(_odtDeathStatus),                \
                                      "death-test child exit code mismatch");                      \
    } while (0)

/* Convenience for the framework's fail-fast convention (exit(1)). */
#define ASSERT_EXITS_WITH_FAILURE(statement) ASSERT_EXITS_WITH(1, statement)

#endif /* ODT_TEST_DEATH_TEST_H */
