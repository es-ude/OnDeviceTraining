#ifndef ODT_HOOK_H
#define ODT_HOOK_H

/*! Generic phase hook for EXTERNAL profilers (energy rigs, cycle counters,
 *  latency tracers). The framework fires one event at each boundary of the
 *  three training-step phases; a caller-installed function turns them into
 *  whatever the profiler needs (GPIO markers, DWT_CYCCNT reads, log lines).
 *  Precondition for the #419 benchmark target. The dependency is one way:
 *  the profiler library (e.g. odt-energy-rig's rig_marker) never learns about
 *  ODT; only the example that wires the two together includes both headers.
 *
 *  What each phase spans (the FORWARD and BACKWARD pair TILE one whole
 *  calculateGradsSequential / tracedGrads call, no gap between them):
 *   - FORWARD:   activation-buffer allocation, every layer forward, the output
 *                snapshot copied into trainingStats_t, and the loss FORWARD.
 *   - BACKWARD:  the loss backward, every layer backward, activation teardown.
 *                Fires even when backward truncates at the deepest trainable
 *                layer or is skipped entirely (all-frozen model): every call
 *                yields exactly FORWARD_BEGIN, FORWARD_END, BACKWARD_BEGIN,
 *                BACKWARD_END in that order, so an external occurrence count
 *                per step is a constant, never a function of the model.
 *   - OPTIMIZER: the parameter update inside optimizerStep() (Optimizer.h),
 *                and only that -- grad zeroing, mean-scaling and clipping are
 *                outside the phase. A direct optimizerFunctions[type].step()
 *                call performs the same update but fires NO events; profiler
 *                and benchmark code must step via optimizerStep().
 *  tracedGrads fires the identical events (same body as
 *  calculateGradsSequential); its traceSink_t probes land INSIDE the phases.
 *
 *  Energy-marker rule: only phases of at least ~500 us are usable as energy
 *  markers -- below that a marker edge is not reliably resolvable against
 *  the profiler's sampling grid. Events bounding a shorter phase are for
 *  LATENCY profiling only, never for energy markers. There are deliberately
 *  no per-layer events here; per-layer observation is traceSink_t
 *  (TraceApi.h), a different tool for a different question.
 *
 *  One global slot: odtHookSet installs fn + ctx process-wide; NULL disables
 *  (the default). Install before training starts and do not swap it
 *  mid-step -- the slot is plain static state, not synchronized. One
 *  training step at a time: events carry no thread or invocation identity,
 *  so concurrent calculateGrads* / optimizerStep calls from several threads
 *  interleave their events indistinguishably -- serialize training while a
 *  hook is installed, or install none.
 *
 *  Cost when unset: one load and one branch per event. Six event kinds; a
 *  calculateGrads* call fires four, an optimizerStep two, so a macro-batch
 *  of B samples per optimizer update fires 4*B + 2 events. */

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

/*! Internal -- the framework's six call sites only: if (fn) fn(ctx, event). */
void odtHookFire(odtEvent_t event);

#endif /* ODT_HOOK_H */
