/* -finstrument-functions hook for trace_step.py (host only, not part of any
 * library): records every function entered while trainingEpochDefault runs and,
 * on the fourth trainingBatchDefault entry (three whole steps: the trainer's
 * first, wire-gated step and two plain ones, each with its optimizer step and
 * grad reset), writes the addresses relative to main to $STEP_TRACE_OUT and
 * exits. Compiled without instrumentation. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

/* Weak: CMake's compiler check links this object without the library. */
extern float trainingEpochDefault() __attribute__((weak));
extern float trainingBatchDefault() __attribute__((weak));
extern int main();

#define TRACE_SLOTS 65536u
#define TRACE_BATCHES 3

static uintptr_t slots[TRACE_SLOTS];
static int insideEpoch = 0;
static int batches = 0;

__attribute__((no_instrument_function)) static void dumpAndExit(void) {
    const char *path = getenv("STEP_TRACE_OUT");
    FILE *f = fopen(path != NULL ? path : "step_trace.addr", "w");
    if (f == NULL) {
        _exit(3);
    }
    for (size_t i = 0; i < TRACE_SLOTS; i++) {
        if (slots[i] != 0) {
            fprintf(f, "%llx\n", (unsigned long long)(slots[i] - (uintptr_t)main));
        }
    }
    fclose(f);
    _exit(0);
}

__attribute__((no_instrument_function)) void __cyg_profile_func_enter(void *fn, void *callSite) {
    (void)callSite;
    if (fn == (void *)trainingEpochDefault) {
        insideEpoch++;
    }
    if (insideEpoch == 0) {
        return;
    }
    if (fn == (void *)trainingBatchDefault && ++batches > TRACE_BATCHES) {
        dumpAndExit();
    }
    uintptr_t a = (uintptr_t)fn;
    size_t h = (size_t)((a >> 2) * 2654435761u) % TRACE_SLOTS;
    while (slots[h] != 0 && slots[h] != a) {
        h = (h + 1) % TRACE_SLOTS;
    }
    slots[h] = a;
}

__attribute__((no_instrument_function)) void __cyg_profile_func_exit(void *fn, void *callSite) {
    (void)callSite;
    if (fn == (void *)trainingEpochDefault) {
        insideEpoch--;
    }
}
