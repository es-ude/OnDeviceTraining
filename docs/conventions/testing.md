# Unit-test conventions

## Test-driven discipline (RED before implementation)

Every feature and bugfix is written test-first: the failing test (RED) is
authored and **its failure is observed and captured before** the
implementation exists. This is a hard house rule, not a preference.

- **Capture RED live.** The evidence that a test discriminates is its
  observed failure *against absent or unimplemented code* — a link error, an
  assertion failure, a death-test exit mismatch — recorded at the time,
  before the implementation is written.
- **Post-hoc reconstruction is not acceptable as primary evidence.**
  Reverting a finished implementation, running the already-written tests to
  manufacture a RED, then reapplying the code, is *not* test-first. A test
  authored after seeing the implementation can silently encode "what the code
  does" instead of "what the spec requires" — the exact failure mode TDD
  exists to prevent. If test-first was skipped, the remedy is to redo the
  tests from the spec, not to reconstruct a RED.
- **Mutation testing is required but is not a substitute.** Each test must be
  shown to catch a real defect: temporarily break the code, observe the
  specific test go RED, restore, confirm GREEN. This proves the test is not
  vacuous — but a non-vacuous test written post-hoc can still assert the wrong
  contract. Mutation evidence complements live-RED; it does not replace it.

Rationale of record: PR #276 shipped one optimizer task implemented before
its tests, with RED reconstructed post-hoc (source-revert + targeted
mutations, byte-identical reapply). It was accepted *that once* — a redo of
already-green, mutation-verified code was judged not worth the cost —
explicitly not as precedent. This section is the go-forward rule that keeps
it from recurring.

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

## Test memory discipline

Unit tests in `test/unit/**` follow a tiered idiom for memory cleanup. The
tier boundary is mechanical: tests that contain no `*Init*` calls (i.e.,
purely stack-allocated `tensor_t`/`shape_t`/`quantization_t` designated
initializers) stay in the **stack-only tier** and need no cleanup. Any test
that calls `*Init*` (= heap allocation through `reserveMemory`) is in the
**heap tier** and follows three rules.

### Stack-fixture idiom for `symQConfig_t` (group-quant PR1)

Since the always-array migration, `initSymQConfig` heap-allocates the
1-element `scales` array (`reserveMemory`), so every call needs a paired
`freeReservedMemory(qc.scales)` / `freeQuantization`. Tests that only need
a fixed per-tensor scale stay in the stack-only tier by building the struct
directly over a local backing array — never passed to any free call:

```c
float s[1] = {0.5f};
symQConfig_t qc = {
    .scales = s, .numGroups = 1, .groupSize = 0,
    .roundingMode = HALF_AWAY, .qBits = 8};
```

The array must outlive every use of the config (beware `tensorInitFloat`-style
dangling if the config escapes the scope). Reach for `initSymQConfig` + free
only when the test exercises the allocation topology itself.

**Never as a deserialize destination (group-quant PR2, Task 5 review fix).**
"Never passed to any free call" also rules out handing a stack fixture to
`deserializeTensor` / `deserializeParameter` / `ppcaReplaySetDeserialize` as
the skeleton being deserialized INTO. A file record whose `numGroups` differs
from the skeleton's own makes `deserializeQConfig`'s SYM arm
`freeReservedMemory()` the skeleton's *current* `scales` pointer before
`reserveMemory()`-ing a differently-sized one — there is no explicit
`free(qc.scales)` at the test's own call site to catch, but a stack-backed
array reaching that implicit free is exactly the same undefined behavior the
"never freed" rule exists to avoid. Build deserialize-destination fixtures
with `initSymQConfig`/`initSymQConfigGrouped` (or a real tensor via
`initTensor`) instead, even when the test only cares about a fixed
per-tensor scale.

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

The `tensorInitFloat` / `tensorInitSymInt32` / `tensorInit*` family was
removed in #182; it no longer exists, so new tests must use the Rule 1
pattern above.

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
| `softmaxLayerInit(...)`                   | `freeSoftmaxLayer(l)`| layer config wrapper only           |
| `sgdMCreateOptim(...)`                    | `freeOptim(o)`       | all registered parameters + states  |
| `inference(...)` (returns `tensor_t *`)   | `freeTensor(out)`    | as above                            |
| `inferenceWithLoss(...)`                  | `freeInferenceStats` | stats struct + output tensor        |
| `calculateGradsSequential(...)`           | `freeTrainingStats`  | stats struct                        |

Layer free-functions release only the config wrapper, not the parameters
they reference. When an optimizer is in play, `freeOptim` takes
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

### Torch-version coupling: pin, don't loosen

The bit-exact golds pin torch's INTERNAL kernel rounding (lerp FMA fusion,
addcdiv grouping, scheduler recursion paths), which torch does not guarantee
across versions. Gold breakage on a torch bump is expected maintenance:
RE-DERIVE the golds on the new pin — never loosen exact comparisons to
tolerances.

A separate, narrower caveat is the fp32-boundary/libm case: a fixture
deliberately placed on a float32 rounding boundary can see torch's own
computation disagree with the C-side closed form by a sub-ulp **double**
difference that depends on the host's libm (observed: Apple libm and glibc
`cos` land on opposite sides of the same boundary) — that drift is a
torch/libm implementation detail, not a regression, and must not gate gold
generation. The worked example is
`test/unit/optimizer/generate_expected_lr_scheduler.py`'s
`cos_discrim_b005_t26_e001` `CosineAnnealingLR` fixture (#327): it exists to
discriminate the two double associations of `pi*e/T_max` (`(pi*e)/T_max`,
torch's/correct, vs. `pi*(e/T_max)`, the bug #327 fixed), and incidentally
also sits on a float32 rounding boundary at epoch 13. Its `record_sequence`
call sets `boundary_tolerant=True`: the emitted gold is the closed-form value
itself (bit-identical to the C computation by construction) and torch's
output is merely sanity-checked within one float32 ULP, rather than compared
bit-for-bit like every other fixture.

A second worked example shows the same pattern triggered by a PLATFORM axis
rather than a version axis: torch kernel rounding can also differ per host
wheel, not just per torch release.
`test/unit/arithmetic/generate_expected_pointwise_fused.py`'s `addcdivDenom`
azero fixture (#353, found on the Linux CI runner while stabilizing #328) is
the case — torch's addcdiv vectorization fuses the final multiply-add on
Linux-AVX2 but not on macOS-ARM, and with an all-zero accumulator (0+x is
exact) that <=1-ulp difference survives to the emitted bits, so the SAME
torch version produces different golds on the two CI-relevant hosts. The
STANDARD (a~O(1)) addcdiv fixture still passes bit-for-bit on both
platforms — the term difference there sits below the accumulator's own ULP
and is absorbed — so the divergence is real but narrow: only the
zero-accumulator regime exposes it. Once a self-check exposes a platform (or
version) split like this, the fix is the same move as the #327
discriminator: demote torch from gold source to tolerance sanity check, and
promote the documented order — here the numpy float32 replication of the
six separate roundings (sqrt/div/mul/add, each correctly rounded) — to the
gold source itself, since it is bit-identical on every IEEE host and to the
C implementation by construction.

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

