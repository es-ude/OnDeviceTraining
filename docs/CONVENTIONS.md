# Project Conventions

## Data Shape Convention

Datasets deliver samples in their natural geometric shape (e.g. `[C, H, W]`
for images, `[C, L]` for time series). Any `reshape`, `flatten`, or `view`
operation is the **first layer of the model**, not a preprocessing step in
the dataset. This:

- keeps dataset code independent of downstream model topology
- allows one dataset to feed models with different input ranks
- matches the PyTorch / Keras / elastic-ai.creator IR convention, so a future
  ir2c can compile each shape transform to a corresponding C layer

For flatten-to-2D, use `flattenLayerInit()` from `FlattenApi.h`.

## Sanitizer-driven memory bug detection

The C unit-test suite is run twice in CI: once normally (`c-build-and-test`),
and once under AddressSanitizer + UndefinedBehaviorSanitizer
(`c-asan-build-and-test`). The sanitizer job is a hard gate — any heap-OOB,
use-after-free, double-free, or UB diagnoses fails the PR. LeakSanitizer is
deliberately **off** (`detect_leaks=0`) in CI; see the opt-in recipe below.

### Local reproduction

The `unit_test_asan` preset is the source of truth. Same flags, same runtime
options as CI:

```bash
cmake --preset unit_test_asan
cmake --build --preset unit_test_asan
ctest --preset unit_test_asan
```

Or, in the devenv shell, the composite script:

```bash
run_asan_tests
```

Sanitizer flags (`-fsanitize=address,undefined -fno-sanitize=function
-fno-omit-frame-pointer -fno-sanitize-recover=all -g -O1`) propagate to every
target in the link graph via the configure preset — there is no opt-in per
target.

Runtime options the test preset sets:

- `ASAN_OPTIONS=detect_leaks=0:abort_on_error=1:halt_on_error=1:strict_string_checks=1:check_initialization_order=1`
- `UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1`

`halt_on_error=1` plus `-fno-sanitize-recover=all` means the **first** finding
aborts the test binary — earlier tests must run cleanly to surface later ones.
When triaging multiple unrelated failures, isolate by running individual test
binaries from `build/unit_test_asan/test/unit/...` directly.

### macOS toolchain requirement (LLVM ≥ 22)

macOS 26.4 changed the dyld shared-cache layout in a way that hangs
AddressSanitizer startup — `__asan_init` livelocks before `main()` (zero output,
~100% CPU) — for any compiler-rt **≤ 21.1.8**, which is the nixpkgs Darwin
default that `pkgs.clang` would otherwise provide. The upstream fix (LLVM
PR #182943, backported to `release/22.x`) ships in **LLVM ≥ 22**, so the devenv
`run_asan_tests` and `ci` scripts pin the ASan compiler to clang 22 (the
`nixpkgs-llvm22` input → `asanClang` in `devenv.nix`). The normal `gcc` build
and CI (Linux / apt-clang) are unaffected.

Running ASan outside devenv on macOS? Use clang ≥ 22, or Apple Command Line
Tools ≥ 26.5 (Apple backported the same fix into their clang 21). Apple CLT
≤ 26.3 will hang.

### Opt-in LeakSanitizer recipe

LSan is staged separately because it requires a cleanup convention every test
honours; see #82 for the umbrella. To run a single test or directory under LSan
during incremental cleanup work, override `detect_leaks` at the call site:

```bash
ASAN_OPTIONS="detect_leaks=1:abort_on_error=1:halt_on_error=1" \
  build/unit_test_asan/test/unit/<module>/UnitTest<Name>
```

For broader recon (e.g. surveying which tests currently leak), prefer the
valgrind-based recipe in `docs/superpowers/tools/lsan-recon/` — it produces
reproducible, fully-attributed per-test reports.

## Allocation Locality

Only `src/userApi/` may call `malloc`, `calloc`, `realloc`, or `free` directly. All other code (sub-layers under `src/`, tests under `test/`) must route allocations through `reserveMemory` and `freeReservedMemory` in `src/userApi/StorageApi.{c,h}`.

Why:
- MCU stack overflows are silent killers; routing through StorageApi keeps stack usage predictable and small.
- Reviewers know exactly where to look for memory issues: `src/userApi/`.
- A future handle-based allocator can subsume the entire allocation surface in one API change instead of touching every call site.

Enforcement:
- A CI job (`alloc-locality` in `.github/workflows/ci.yml`) runs `git grep` against `src/` and `test/` (excluding `src/userApi/`) and fails the build on any match. Comments are excluded from the match.
- Exceptions: none today. If a use-case arises that genuinely needs a direct alloc primitive outside `src/userApi/`, escalate via a PR comment so the rule itself can be revisited.

## Test memory discipline

Unit tests in `test/unit/**` follow a tiered idiom for memory cleanup. The
tier boundary is mechanical: tests that contain no `*Init*` calls (i.e.,
purely stack-allocated `tensor_t`/`shape_t`/`quantization_t` designated
initializers) stay in the **stack-only tier** and need no cleanup. Any test
that calls `*Init*` (= heap allocation through `reserveMemory`) is in the
**heap tier** and follows three rules.

### Rule 1 — Build via the post-#106 primitives

Heap tensors are built by:

```c
size_t *dims  = reserveMemory(N * sizeof(size_t));
/* ... populate dims[i] ... */
size_t *order = reserveMemory(N * sizeof(size_t));
setOrderOfDimsForNewTensor(N, order);
shape_t *s    = reserveMemory(sizeof(shape_t));
setShape(s, dims, N, order);
tensor_t *t   = initTensor(s, quantizationInitFloat(), NULL);
tensorFillFromFloatBuffer(t, src, count);   /* or initDistribution(t, &d); */
```

The deprecated `tensorInitFloat` / `tensorInitSymInt32` / `tensorInit*`
family must not be used in new tests. Their attributes emit
`-Wdeprecated-declarations` to surface accidental adoption.

A file-local factory like `makeFloatTensorForDistTest` in
`test/unit/tensor/UnitTestTensorApi.c` is fine when 3+ tests in the same
file repeat the construction. A *cross-file* helper is deferred until 3+
test files repeat the same construction.

### Rule 2 — Free in reverse-init order

`freeTensor` cascades to data + shape (with its dims and order blocks) +
quantization + sparsity + the tensor struct itself. Do not call
`freeShape` or `freeQuantization` on a shape/quantization that was already
consumed by `initTensor` — that is a double-free. The cascade table:

| Allocation                                | Cleanup call         | Cascades to                         |
|-------------------------------------------|----------------------|-------------------------------------|
| `initTensor(s, q, sp)`                    | `freeTensor(t)`      | data, shape (+dims, +order), q, sp  |
| `parameterInit(p, g)`                     | `freeParameter(par)` | param tensor + grad (if non-NULL)   |
| `linearLayerInitLegacy(...)`              | `freeLinearLayerLegacy(l)` | layer config wrapper only     |
| `reluLayerInitLegacy(...)`                | `freeReluLayerLegacy(l)` | layer config wrapper only       |
| `softmaxLayerInit(...)`                   | `freeSoftmaxLayer(l)`| layer config wrapper only           |
| `sgdMCreateOptim(...)`                    | `freeOptimSgdM(o)`   | all registered parameters + states  |
| `inference(...)` (returns `tensor_t *`)   | `freeTensor(out)`    | as above                            |
| `inferenceWithLoss(...)`                  | `freeInferenceStats` | stats struct + output tensor        |
| `calculateGradsSequential(...)`           | `freeTrainingStats`  | stats struct                        |

Layer free-functions release only the config wrapper, not the parameters
they reference. When an optimizer is in play, `freeOptimSgdM` takes
ownership of the parameter cleanup — do not also call `freeParameter` on
the same pointers.

### Rule 3 — Assert-last (capture, free, assert)

ODT's Unity build defines `UNITY_INCLUDE_SETJMP`, so a failing
`TEST_ASSERT_*` longjmps out of the test function and any code after it
does not run. To keep LSan output meaningful — failing tests should still
report zero leaks attributable to the test fixture — every heap-tier test
follows this three-block shape:

```c
void testFoo(void) {
    /* 1. Build heap fixtures (Rule 1). */
    quantization_t *q = quantizationInitFloat();
    /* ... etc ... */

    /* 2. Exercise the system, capture every assertion value into a
     *    stack local. Do not assert here. */
    float capturedLoss = inferenceWithLoss(model, ...)->loss;
    /* (capture more if needed) */

    /* 3. Free in reverse-init order (Rule 2). */
    freeTensor(t);
    /* ... etc ... */

    /* 4. Assert on the captured locals. */
    TEST_ASSERT_FLOAT_WITHIN(1e-4f, EXPECTED_LOSS, capturedLoss);
}
```

Reference exemplars in the tree: `test/unit/userAPI/UnitTestInferenceApi.c`,
`test/unit/userAPI/UnitTestMultiLayerTraining.c`,
`test/unit/tensor/UnitTestTensorApi.c::testInitDistribution_*`.

### Verification

A test file is considered idiom-compliant when, run under valgrind in the
`odt-lsan-recon:2026-04-22` Docker image with
`--leak-check=full --show-leak-kinds=all`, all four LEAK SUMMARY
categories report 0 bytes in 0 blocks (or valgrind emits "All heap blocks
were freed -- no leaks are possible"). The reproducible recipe and
container Dockerfile live in `docs/superpowers/tools/lsan-recon/`.

## Build-time gold-value generators (CMake + uv + PyTorch)

Some unit tests compare C-side numerics against PyTorch reference values. The
references are not committed: a Python script in the test directory emits a C
header (`expected_*.h`) at build time, which the test then `#include`s.

The wiring lives in `test/unit/<module>/CMakeLists.txt`:

```cmake
add_custom_command(
        OUTPUT ${GEN_HEADER}
        COMMAND uv run ${CMAKE_CURRENT_SOURCE_DIR}/generate_expected_<thing>.py
                --out ${GEN_HEADER}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/generate_expected_<thing>.py
        VERBATIM
)
add_custom_target(generate_expected_<thing> DEPENDS ${GEN_HEADER})
add_dependencies(UnitTest<Name> generate_expected_<thing>)
target_include_directories(UnitTest<Name> PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
```

Reference exemplars:
`test/unit/arithmetic/generate_expected_conv1d_kernel.py`,
`test/unit/arithmetic/generate_expected_conv_transpose_1d_kernel.py`.

### Generator-script conventions

- Use `repr(v) + "f"` to format C float literals, **not** `f"{v:.9g}"`.
  `repr` always preserves a decimal point or exponent, so `10.0f` stays valid.
  `:.9g` produces `10` and the trailing `f` then makes it an invalid integer
  suffix that gcc rejects.
- Self-check fixtures with `assert torch.allclose(...)` before emitting them,
  so generator-side numerical drift fails the build instead of silently
  shifting expected values.
- `torch` and `torchvision` are declared as direct dependencies in
  `pyproject.toml`. The decoupling is intentional: generator scripts
  import `torch` directly, so the dependency belongs at the project
  level rather than inherited from `elasticai-creator`.

### CI implication: every job that runs `cmake --build` MUST install uv

The custom command above is invoked by ninja during the build phase, not by
configure. Any CI job that produces or runs targets depending on a generated
header must therefore have `uv` on `PATH` at build time. In
`.github/workflows/ci.yml` this is `c-build-and-test` and
`c-asan-build-and-test`; both install uv via `astral-sh/setup-uv@v6` and
`uv sync` before `cmake --preset ...`.

Locally this is silent: `devenv.nix` puts `uv` on `PATH` for the whole shell,
so `cmake --build` finds it without any explicit setup. CI is stricter and
catches drift here before merge.

When introducing a new generator under a new test target, audit every CI job
that builds the affected preset and add the uv setup steps if missing.

## Loss API: microbatch contracts

Each loss function in `src/loss_functions/` exposes:

- `forward(modelOutput, label, reduction) → float`
- `backward(modelOutput, label, result) → void`
- `computeMeanScale(totalSamples, modelOutput) → float`

### Reduction split

`lossConfig_t.backwardReduction` is the user's training-strategy choice — it
drives whether `scaleOptimizerGradients` runs between `trainingBatchDefault`
and `optimFns.step`. It is a config field.

`forwardReduction` is a per-call parameter on every aggregator
(`trainingBatchDefault`, `evaluationBatch`, `evaluationEpoch`, `inferenceWithLoss`,
`calculateGradsFn_t`). It controls how the per-microbatch loss value is
reported. `trainingRun` is the only function that hardcodes it
(to `REDUCTION_MEAN`) so train and eval losses are comparable; lower-level
callers pick freely.

### Microbatch shape

`modelOutput->shape->dimensions[0]` is the microbatch dimension `B`. For
`B=1` today, output shape is `[F]` (the leading 1 is implicit). For `B>=1`
in the future, output shape is `[B, F]` and `numFeaturesPerSample = numElements / B`.

**Uniform-B assumption** (DataLoader contract): all microbatches in one
macro batch have equal `B`. The MEAN aggregator divides by total samples
(`Σ batch->size`) rather than by `(numberOfBatches × B)`, so non-uniform B
would skew the mean. ODT's DataLoader currently always produces uniform
batches via `dropLast=true`; non-uniform B is out of contract.

### Backward macro-scaling

Backward writes raw per-element gradients (`2(o-l)` for MSE, `(p-y)` for CE).
The macro-batch divisor lives at the optimizer:

- `lossFunctions[lossConfig.funcType].computeMeanScale(N, modelOutput)`
  returns the PyTorch-parity divisor (`1/(N*F)` for MSE, `1/N` for CE).
- `scaleOptimizerGradients(optimizer, factor)` multiplies every parameter's
  `grad` field by the factor in place.
- `trainingEpochDefault` calls these between accumulation and `step`,
  but only when `backwardReduction == REDUCTION_MEAN`.

For SUM (or future per-sample weighted variants — see #150), the backward
gradient flows through unscaled.

### Shape assertion (deferred)

Runtime assertion of the `dimensions[0] >= 1` contract is deferred to the
microbatch-B>1 umbrella (#152) — specifically #153. Today (B=1 only) the
assertion would be effectively a no-op; the protective value materialises
when B>1 becomes a real feature target.

## Quantized gradient accumulation — known precision Open Problem

As of the quantized-gradient prerequisite (`gradInit`, 2026-06-05) a trainable
layer's parameter gradient can be stored in the dtype its `backwardMath`
declares. For SYM_INT32 grads, the per-microbatch accumulation reuses the
existing `addSymInt32TensorsInplace` ("strategy A", dynamic-rescale): it
dequantizes both the running grad and the new microbatch grad to float, adds,
and re-quantizes the running sum to a new absmax-derived scale **on every
microbatch**.

This is functionally correct end-to-end today, but **not** numerically ideal:

- Quantization noise compounds with the number of microbatches M.
- The running-sum absmax is pinned by the heaviest microbatch, coarsening the
  LSB for the accumulated small-gradient mass.

Preliminary characterization (internal simulation, M=100, N=64, σ=1e-3 with a
10% ×50 heavy tail — *problem characterization only, not a basis for a chosen
solution*):

| Strategy | Final rel. error vs float64 | Float-free? |
|---|---|---|
| A — dynamic-rescale (current) | ~1.5e-4, **grows with M** (2.0e-5 @ step1 → 1.7e-4 @ step100) | No |
| B — fixed-scale integer accum | ~9.9e-5 | Yes |
| C — float accum, quantize-at-read | ~2.2e-5 | No |

We deliberately ship strategy A now and do **not** adopt B/C or any homegrown
numerical scheme. The resolution path is a literature review (stochastic-rounding
accumulators, error-feedback / residual accumulation, higher-precision master
grads, block/group scaling, …) → implement or improve a **published** technique.
Tracked as a separate research task (#218). This note is intentionally public
(not buried in a private spec) so contributors hitting accuracy issues in
quantized training know this is a known, expected limitation rather than a bug.

### Two accumulation schemes in-tree (both intentional)

- **Strategy A (dynamic-rescale)** — Linear SYM weight grads and LayerNorm
  gamma/beta grads: per-microbatch `addSymInt32TensorsInplace` (dequantize
  both operands with their own scales, float-add, requantize the running sum
  to a fresh absmax scale). Not float-free.
- **Fixed-scale integer accumulation** — Linear SYM bias grads
  (`linearCalcBiasGradsSymInt32`): increments are rescaled into the running
  grad's EXISTING scale and added in integer arithmetic; the scale is never
  re-derived during accumulation. The coarser resolution (LSB pinned by the
  running scale, which inits to 1.0) is inherent to the scheme.

  **Attribution note:** this fixed-scale integer bias-GRADIENT accumulation is
  ODT's own construction and is NOT prescribed by Deutel et al.
  (arXiv:2407.10734). The paper's quantization is *dynamic*: scales are
  re-derived from observed data — weights every SGD update (Eqs. 6-7) — and the
  method is framed throughout as "dynamic adaptation of the zero-point and
  scale parameters" (Sec. IV-E). The paper has a forward bias (int32 bias on
  the int32 MAC accumulator, Fig. 2) but describes no bias-*gradient*
  accumulation, and it nowhere states that any scale is held static *during
  training* (the only static/PTQ mention is post-training, at deployment) — so
  absent evidence to the contrary, assume its scales are dynamic. ODT's
  fixed-scale bias-grad scheme, which never re-derives the scale during
  accumulation, therefore DEVIATES from the paper's dynamic scaling; the ODT
  scheme that corresponds to Deutel is Strategy A (dynamic-rescale, above).
  What ODT also follows from Deutel: per-layer error requant (~Eq. 4) and the
  float-space SGD step (~Eqs. 5-7). Scheme choice + the init-scale resolution
  limit: #218.

This is a research framework: deliberate scheme differences like this one
MUST be documented here, so experimental design stays separable from
accidental inconsistency. LayerNorm uses strategy A for BOTH gamma and beta
per the 2026-06-05 LayerNorm spec.
