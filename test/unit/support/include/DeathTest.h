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

#include <errno.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
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

/* Bytes read from the child's stdout per read(), and the tail carried between
 * reads: an expectedSubstr of up to this length is found even across a read
 * boundary. */
#define ODT_DEATH_CAPTURE_CHUNK 4096u
/* Failure-message sizes: the stdout tail quoted in the message, and the message. */
#define ODT_DEATH_TAIL_BYTES 256u
#define ODT_DEATH_MSG_BYTES 512u

/* Drains fd to EOF and reports whether needle occurs anywhere in the stream.
 * Draining before waitpid() matters: a child printing more than the pipe buffer
 * blocks in write() until the parent reads. Memory stays bounded (no heap:
 * test/ is under the alloc-locality gate); the last bytes of the stream land in
 * tail for the failure message. */
static inline bool odtDeathDrainAndFind(int fd, const char *needle, char *tail, size_t tailSize) {
    char window[2 * ODT_DEATH_CAPTURE_CHUNK + 1];
    size_t kept = 0;
    bool found = (needle[0] == '\0');
    for (;;) {
        ssize_t got = read(fd, window + kept, ODT_DEATH_CAPTURE_CHUNK);
        if (got < 0 && errno == EINTR) {
            continue;
        }
        if (got <= 0) {
            break;
        }
        kept += (size_t)got;
        window[kept] = '\0';
        found = found || strstr(window, needle) != NULL;
        if (kept > ODT_DEATH_CAPTURE_CHUNK) {
            memmove(window, window + kept - ODT_DEATH_CAPTURE_CHUNK, ODT_DEATH_CAPTURE_CHUNK);
            kept = ODT_DEATH_CAPTURE_CHUNK;
        }
    }
    window[kept] = '\0';
    size_t from = (kept + 1 > tailSize) ? kept + 1 - tailSize : 0;
    memcpy(tail, window + from, kept - from + 1);
    return found;
}

/* Like ASSERT_EXITS_WITH, and additionally asserts that the child's stdout
 * contains expectedSubstr - e.g. the rule name of a PRINT_ERROR banner, so a
 * test cannot pass by dying on a different guard with the same exit code.
 * stdout goes through a pipe instead of /dev/null; stderr stays silenced.
 * Match text inside the message: PRINT_ERROR wraps it in ANSI colour codes, so
 * a substring spanning the start of the banner or its trailing newline fails.
 * The guard must leave through exit() (which flushes stdio), not _exit().
 * expectedSubstr is at most ODT_DEATH_CAPTURE_CHUNK bytes.
 * Searched as a C string: a NUL in the child's stdout before the text hides it. */
#define ASSERT_EXITS_WITH_OUTPUT(expectedCode, expectedSubstr, statement)                          \
    do {                                                                                           \
        const char *_odtNeedle = (expectedSubstr);                                                 \
        int _odtPipe[2];                                                                           \
        TEST_ASSERT_MESSAGE(pipe(_odtPipe) == 0, "pipe() failed in death test");                   \
        fflush(stdout);                                                                            \
        fflush(stderr);                                                                            \
        pid_t _odtDeathPid = fork();                                                               \
        if (_odtDeathPid < 0) {                                                                    \
            (void)close(_odtPipe[0]);                                                              \
            (void)close(_odtPipe[1]);                                                              \
        }                                                                                          \
        TEST_ASSERT_MESSAGE(_odtDeathPid >= 0, "fork() failed in death test");                     \
        if (_odtDeathPid == 0) {                                                                   \
            (void)close(_odtPipe[0]);                                                              \
            (void)dup2(_odtPipe[1], STDOUT_FILENO);                                                \
            if (_odtPipe[1] != STDOUT_FILENO) {                                                    \
                (void)close(_odtPipe[1]);                                                          \
            }                                                                                      \
            (void)freopen("/dev/null", "w", stderr);                                               \
            statement;                                                                             \
            (void)fflush(stdout);                                                                  \
            _exit(0);                                                                              \
        }                                                                                          \
        (void)close(_odtPipe[1]);                                                                  \
        char _odtTail[ODT_DEATH_TAIL_BYTES];                                                       \
        bool _odtFound = odtDeathDrainAndFind(_odtPipe[0], _odtNeedle, _odtTail, sizeof _odtTail); \
        (void)close(_odtPipe[0]);                                                                  \
        int _odtDeathStatus = 0;                                                                   \
        while (waitpid(_odtDeathPid, &_odtDeathStatus, 0) < 0 && errno == EINTR) {}                \
        char _odtMsg[ODT_DEATH_MSG_BYTES];                                                         \
        (void)snprintf(_odtMsg, sizeof _odtMsg,                                                    \
                       "death-test child terminated by signal %d, expected exit(); "               \
                       "stdout tail: \"%s\"",                                                      \
                       WIFSIGNALED(_odtDeathStatus) ? WTERMSIG(_odtDeathStatus) : 0, _odtTail);    \
        TEST_ASSERT_TRUE_MESSAGE(WIFEXITED(_odtDeathStatus), _odtMsg);                             \
        (void)snprintf(_odtMsg, sizeof _odtMsg,                                                    \
                       "death-test child exit code mismatch; stdout tail: \"%s\"", _odtTail);      \
        TEST_ASSERT_EQUAL_INT_MESSAGE((expectedCode), WEXITSTATUS(_odtDeathStatus), _odtMsg);      \
        (void)snprintf(_odtMsg, sizeof _odtMsg,                                                    \
                       "death-test child stdout lacks \"%s\"; stdout tail: \"%s\"", _odtNeedle,    \
                       _odtTail);                                                                  \
        TEST_ASSERT_TRUE_MESSAGE(_odtFound, _odtMsg);                                              \
    } while (0)

#endif /* ODT_TEST_DEATH_TEST_H */
