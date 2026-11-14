#ifndef ODT_HOOK_H
#define ODT_HOOK_H

/*! Process-wide hook for phase profiling. FORWARD and BACKWARD together cover each
 *  calculateGradsSequential / tracedGrads call, including recomputation, models whose
 *  backward runs for only some layers, and frozen models; recomputation adds no events,
 *  so every call fires exactly the four FORWARD/BACKWARD events. OPTIMIZER covers
 *  optimizerStep's parameter update, excluding gradient preparation (zeroing,
 *  mean-scaling, clipping); a raw optimizerFunctions[type].step() call emits no events.
 *  Install before training, keep ctx alive, and serialize instrumented training calls:
 *  the slot is unsynchronized and events carry no thread identity. NULL disables the
 *  hook (the default). Per-layer observation is traceSink_t (TraceApi.h). */

typedef enum {
    ODT_EVENT_FORWARD_BEGIN,
    ODT_EVENT_FORWARD_END,
    ODT_EVENT_BACKWARD_BEGIN,
    ODT_EVENT_BACKWARD_END,
    ODT_EVENT_OPTIMIZER_BEGIN,
    ODT_EVENT_OPTIMIZER_END,
} odtEvent_t;

typedef void (*odtHookFn_t)(void *ctx, odtEvent_t event);

/*! Install the process-wide hook; fn == NULL removes it. ctx is handed back
 *  verbatim on every event and is never dereferenced by the framework. */
void odtHookSet(odtHookFn_t fn, void *ctx);

/*! Internal phase-boundary notification. */
void odtHookFire(odtEvent_t event);

#endif /* ODT_HOOK_H */
