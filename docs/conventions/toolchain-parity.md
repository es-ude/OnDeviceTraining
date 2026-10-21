# Toolchain parity

## What depends on the toolchain

Unit gold tests (1e-4-class `TEST_ASSERT_FLOAT_WITHIN`) and the example
float bit-parity/allclose gates (`c-bit-parity` job) compare C-side
numerics against fixed reference values. The comparison is only as stable
as the toolchain that produced the C side: FMA contraction
(`-ffp-contract`) changes which multiply-adds fuse into a single rounding
step, and libm implementations differ across compiler/libc versions. See
the kernel comments in `src/layer/LayerNorm.c` (forward absmax pre-check,
backward pass-A/pass-B expression matching) for concrete places where this
already mattered enough to work around in code.

## The pins

- `-ffp-contract=fast` is pinned explicitly in the `host` configure preset
  (`CMakePresets.json`) instead of being left to whatever the compiler
  defaults to. Sanitizer presets (`unit_test_asan`, `unit_test_ubsan`) set
  their own `CMAKE_C_FLAGS`, which under CMake-preset semantics fully
  replaces the inherited value rather than appending to it — the pin
  deliberately does not reach them.
- Every CI job runs on a pinned runner image (`ubuntu-24.04`, which ships
  gcc 13; one ARM job on `ubuntu-22.04`), never on `ubuntu-latest`. GitHub
  moved `ubuntu-latest` to 26.04 with gcc 15 in October 2026, and the guard
  below stopped CI. Moving to a newer image is a deliberate toolchain change,
  made together with the guard and checked against the bit-parity job.
- CI asserts `gcc -dumpversion` major == `13` in `c-build-and-test` and
  `c-bit-parity` — the two jobs whose output feeds gold comparisons. The
  `c-ubsan-build-and-test` job also runs the gold suite on gcc but is not
  asserted here (its preset's own `CMAKE_C_FLAGS` drops the pin, so it runs
  on gcc's implicit `=fast` default). The `c-asan-build-and-test` job runs a
  correctness gate, not a numerics gate; its local devenv reproduction is
  pinned to clang 22 for an unrelated reason (macOS `__asan_init` livelock
  on older LLVM — see the comment in `devenv.nix`).

## Parity rule (existing repo principle)

Host/dev toolchain, Docker, and CI must match versions exactly for
numerics work. **Current known gap (2026-07-07): devenv gcc 15.2.0 vs. CI
apt gcc 13.x.** Golds pass on both today, but treat any local gold delta
with suspicion until the gap is closed — it is not evidence by itself that
a change is wrong, nor that the toolchains agree in general. Closing it
(pin devenv down to gcc 13, or move CI onto the devenv toolchain) is an
open decision, not resolved by this doc.

## On a deliberate toolchain bump

1. Run the full gold suite under the old **and** new toolchain.
2. Run the `-ffp-contract=off` sensitivity probe to (re-)establish whether
   the suite is contraction-sensitive at the new version:
   ```bash
   cmake -S . -B build/fpoff -G Ninja -DDEBUG_MODE_DEBUG=ON \
     -DODT_MEM_PROFILE=ON -DCMAKE_C_FLAGS=-ffp-contract=off
   cmake --build build/fpoff --target all_unit_tests -j4
   ctest --test-dir build/fpoff -L unit --output-on-failure
   ```
3. Re-run the `c-bit-parity` job.
4. Update the `gcc major` value asserted in `ci.yml` in the same PR.
5. Document any gold movement in the PR body.

**Never loosen a tolerance to make a toolchain bump pass.** A tolerance
change is a numerics decision with its own review — it is not a CI fix.
