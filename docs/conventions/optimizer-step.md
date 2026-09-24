# Optimizer step entry

## Step through `optimizerStep()`, not the raw vtable

Every training loop, example, and benchmark steps the optimizer through
`optimizerStep()` (`src/optimizer/include/Optimizer.h`). The raw vtable call
`optimizerFunctions[type].step(optim)` performs the identical parameter update
but fires none of the `ODT_EVENT_OPTIMIZER_BEGIN`/`END` phase-hook events
(`src/common/include/OdtHook.h`, #429).

Why:
- An external profiler (energy rig, cycle counter, latency tracer, #419) counts
  on four events per `calculateGrads*` call and two per `optimizerStep`, so
  4·(b/m) + 2 per optimizer update for a macro-batch of b samples at
  `microBatchSize` m (4·b + 2 at the default m = 1). A raw-vtable stepper
  reports FORWARD/BACKWARD spans and silently zero OPTIMIZER spans.
- The stack-watermark probe (`examples/har_classifier/mem_instrument.c`) measures
  the production training path, and `trainingEpochDefault` steps through the
  wrapper, so the probe mirrors that call graph. The wrapper's own frame is
  tens of bytes and only shows up where the peak lies inside the step (the
  packed-SYM binary); provenance for the measured deltas lives with the budgets
  in `examples/_shared/check_stack_watermark.py` (#432).
- Every raw-vtable stepper in `examples/` is a copy-paste source for the next one.

Grad zeroing (`optimizerFunctions[type].zero` / `optimizerZeroGrad`),
mean-scaling and clipping stay separate calls; they are outside the measured
phase by design.

Enforcement:
- A CI job (`optimizer-step-entry` in `.github/workflows/ci.yml`, mirrored in the
  devenv `ci` script) runs `git grep -nE '(\.|->)step([^A-Za-z0-9_]|$)'` with
  the pathspecs `examples/*.c` `examples/*.h` (git wildcards cross `/`, so that
  is every C source and header under `examples/`) and fails the build on any
  match. The regex catches every member access to the vtable's `step` -- the
  call, `(*fns.step)(o)`, and an alias `stepFn_t f = fns.step;` -- not only the
  call spelling. Comment lines (leading `//`, `*` or `/*`) are excluded; a
  trailing comment on a code line is not, so do not mention `.step` there.
  Python files are not scanned: the PyTorch twins legitimately call
  `optimizer.step()`.
- Known limitation: a direct `sgdStepM()` / `adamWStep()` call is not policed.
  Those are implementation entry points declared only in the internal `Sgd.h` /
  `AdamW.h` headers, which no example includes (examples use `SgdApi.h`,
  `AdamWApi.h`, `Optimizer.h`); a per-function list would rot with every new
  optimizer, so the rule is stated here instead of grepped.
- Exceptions: unit tests under `test/` that step through the raw vtable
  (`UnitTestSgd.c`, `UnitTestAdamW.c`, but also `UnitTestLinear.c`,
  `UnitTestMultiLayerTraining.c`, `UnitTestOptimizerScaling.c`,
  `UnitTestTrainingLoopApi.c`, `UnitTestCalculateGradsSequential.c`)
  deliberately exercise the optimizer implementation without a hook in the
  loop; they are outside the gate's pathspec and stay as they are.
  `optimizerStep()` itself is covered by `test/unit/common/UnitTestOdtHook.c`.

## Every `optimizerFunctions[]` row implements `getLr` and `setLr`

Since the batch-size scheduler port (#445) `trainingRun` calls
`optimizerFunctions[type].getLr(optimizer)` once per epoch for **every**
caller -- to fill `epochInfo_t.learningRate` before the epoch trains -- not
only when an LR scheduler is wired. `LrScheduler` and a compensating
`BsScheduler` write through `setLr`. A new optimizer row that leaves either
accessor NULL therefore crashes on the first epoch of any `trainingRun`,
scheduler or not. Both current rows (`SGD_M`, `ADAM_W`) implement them
(`src/optimizer/Optimizer.c`); a new optimizer must too, and its unit test
should pin both slots (`TEST_ASSERT_NOT_NULL(optimizerFunctions[TYPE].getLr)`
and `setLr`, or a `trainingRun` smoke test).
