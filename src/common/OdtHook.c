#define SOURCE_FILE "ODT_HOOK"

#include <stddef.h>

#include "Common.h"
#include "OdtHook.h"

static odtHookFn_t hookFn = NULL;
static void *hookCtx = NULL;

void odtHookSet(odtHookFn_t fn, void *ctx) {
    hookFn = fn;
    hookCtx = ctx;
}

void odtHookFire(odtEvent_t event) {
    if (hookFn != NULL) {
        hookFn(hookCtx, event);
    }
}
