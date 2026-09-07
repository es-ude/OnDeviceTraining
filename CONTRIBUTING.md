# Contributing to OnDeviceTraining

Welcome! This repository is a pure-C framework for **inference and on-device
training (backpropagation) of DNNs on microcontrollers** — plus the beginnings
of a Python pipeline that will compile PyTorch models down to it. This page is
the map: where things live, how to build and test, and where the canonical
rules are written down. It links rather than duplicates — when in doubt, the
linked document wins.

## Start here

Read in this order:

1. [`README.md`](README.md) — motivation, design principles, roadmap
2. [`docs/FEATURES.md`](docs/FEATURES.md) — what the framework can do **today**
   (layer/optimizer/serialization matrix)
3. [`examples/README.md`](examples/README.md) — runnable end-to-end demos with
   PyTorch twins and bit-parity checks; the best way to see the user-facing API
4. [`docs/CONVENTIONS.md`](docs/CONVENTIONS.md) — contributor conventions index
   and the project vision (memory over float accuracy); per-subsystem rules
   live in [`docs/conventions/`](docs/conventions/)

## Repository map

| Path | What it is |
|---|---|
| `src/` | The C framework — one or more static libraries per subdirectory |
| `test/unit/` | Unity unit tests, one `UnitTest<LibName>.c` per library |
| `examples/` | End-to-end training demos (PyTorch reference + C twin + parity harness) |
| `python/odt/` | Python package skeleton: `providers/`, `ir2c/`, `resource_estimator/` (not started yet) |
| `docs/` | Feature matrix, conventions index, per-subsystem conventions |
| `cmake/` | Build helpers and external dependencies (Unity via FetchContent) |
| `.github/workflows/` | CI pipeline (see [CI](#branches-prs--ci)) |

The C framework is layered; upper layers only reach lower ones:

```text
src/userApi/          TrainingLoopApi, InferenceApi, TensorApi, SgdApi, StateDictApi, …
    ↓
src/layer/ + src/loss_functions/ + src/optimizer/
    ↓
src/arithmetic/       Matmul, Add, Comparison, Distributions, …
    ↓
src/tensor/           Tensor, Quantization, TensorConversion, DTypes
```

Support modules (`common/`, `csv/`, `data_loader/`, `rng/`, `serial/`) sit
alongside. Dispatch is via C-style vtables — global function-pointer arrays
(`layerFunctions[]`, `optimizerFunctions[]`, `lossFunctions[]`,
`conversionMatrix[][]`) indexed by enum. Within each module, public headers
live in its `include/` subdirectory and implementations at the directory root.

## Development environment

The supported setup is [devenv](https://devenv.sh) (requires Nix; devenv's
Getting Started covers installing both): `devenv shell`
provides cmake, ninja, gcc, arm-gcc, uv and Python 3.12 in known-good
versions, plus these scripts:

| Script | Does |
|---|---|
| `setup_cmake` | Configure the `unit_test` preset (with `CC=gcc`) |
| `build_unit_tests` | Build the unit-test suite |
| `run_ai_unit_tests` | Run all Unity unit tests via ctest |
| `run_asan_tests` | Configure + build + run the suite under ASan/UBSan |
| `clean_cmake` | Clean the `unit_test` build tree |
| `ci` | Most of the CI pipeline locally: allocation-locality gate, `clang-format` check, C tests, ASan/UBSan tests, Python tests |

Without devenv you need CMake ≥ 3.20, Ninja, a C compiler, and
[uv](https://docs.astral.sh/uv/) for anything Python. Python work always goes
through `uv run` / `uv sync` — never a bare `python` or `pip`.

## Build & test

The build is driven entirely by presets in
[`CMakePresets.json`](CMakePresets.json):

```sh
cmake --preset unit_test_debug          # configure
cmake --build --preset unit_test_debug  # build (target: all_unit_tests)
ctest --preset unit_test_debug          # run all unit tests
```

| Configure preset | Purpose |
|---|---|
| `unit_test` | Plain unit-test build (what CI runs) |
| `unit_test_error` / `unit_test_info` / `unit_test_debug` | Increasing log verbosity (`DEBUG_MODE_*`); `debug` also enables `ODT_MEM_PROFILE` |
| `unit_test_asan` | AddressSanitizer + UBSan. On macOS this needs compiler-rt ≥ LLVM 22 — see [`docs/conventions/testing.md`](docs/conventions/testing.md) (the devenv shell handles it) |
| `unit_test_ubsan` | Signed-overflow / float-cast UBSan build |
| `examples` | `BUILD_EXAMPLES=ON`; the default `all` target compiles **every** example binary |
| `examples_memprofile` | `examples` + `ODT_MEM_PROFILE=ON` |

Each preset builds into its own `build/<preset>/` directory. Test presets
exist for the six `unit_test*` presets; test binaries in
`build/<preset>/test/unit/` can also be executed directly. Two practical
notes:

- The DataLoader test needs MNIST fixture files — generate once with
  `uv run test/unit/data_loader/MNISTLoader.py`.
- The unit-test build itself shells out to `uv run` for build-time gold-value
  generators (PyTorch-computed expected values baked into headers), so `uv`
  must be on `PATH` even for "C-only" work — and the first build syncs the
  Python deps, including a sizable PyTorch download. The devenv shell brings
  `uv`; CI has a dedicated sync step for this.
- No `CMAKE_C_STANDARD` is set; the compiler default (typically gnu17) is
  what CI builds with.

Python tests: `uv run pytest` (from the repo root).

## Examples

Almost every example in [`examples/`](examples/README.md) pairs a PyTorch
reference with a C twin and a deterministic **bit-parity** gate that CI
enforces (the exception: `mixed_width_mlp` is a C-only acceptance demo) —
they double as living documentation of the user-facing API and as regression
tests. If your change touches training behavior, expect the bit-parity CI job
to notice. Determinism rules live in
[`examples/_shared/DETERMINISM.md`](examples/_shared/DETERMINISM.md).

## Adding code

Canonical rules: [`docs/CONVENTIONS.md`](docs/CONVENTIONS.md). The ones you
will hit immediately:

- `#define SOURCE_FILE "<filename>"` is the first line of every `.c` file
  (before the `Common.h` include).
- **Allocation locality**: `malloc`/`calloc`/`free` are allowed only in
  `src/userApi/`; everything else goes through
  `reserveMemory`/`freeReservedMemory` (StorageApi). CI greps for violations —
  see [`docs/conventions/allocation.md`](docs/conventions/allocation.md).
- Formatting is `clang-format` **21** (the major version CI pins; the devenv
  shell provides it) with the repo `.clang-format`, 4-space indent — enforced
  by CI. Other clang-format majors may format differently and fail the check.
- Datasets deliver samples in their natural geometric shape; any
  reshape/flatten is the first *layer* of the model — see
  [`docs/conventions/data-shape.md`](docs/conventions/data-shape.md).

### Adding a unit test

1. Create `test/unit/<module>/UnitTest<LibName>.c` — the name must match the
   CMake target of the library under test.
2. Register it in that directory's `CMakeLists.txt` with
   `add_elastic_ai_unit_test(LIB_UNDER_TEST <name> [MORE_LIBS ...])`
   (defined in [`test/unit/unit_test.cmake`](test/unit/unit_test.cmake)).
3. Tests use Unity (fetched automatically): `UNITY_BEGIN()`, `RUN_TEST(...)`,
   `UNITY_END()` — registration is manual, there is no test discovery.
   `setUp`/`tearDown` are empty stubs by convention.

More in [`docs/conventions/testing.md`](docs/conventions/testing.md)
(sanitizer gating, heap-tier memory discipline, build-time gold-value
generators).

## Branches, PRs & CI

- `main` is the default/release branch; **`develop` is the integration
  branch — branch off `develop` and target your PR back at `develop`**.
- Branch protection requires green checks before merge; direct pushes are
  blocked, all changes go through PRs.
- Commit subjects follow conventional-commit style
  (`feat(scope): …`, `fix(scope): …`, `docs: …`), as in the existing history.
- GitHub's `Closes #NN` auto-close only fires on merges to the default branch
  (`main`) — after a `develop` merge, close the issue manually.

CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs on pushes and
PRs to `main`/`develop`:

| Job | Checks |
|---|---|
| `alloc-locality` | No allocation primitives outside `src/userApi/` |
| `optimizer-step-entry` | `examples/` step the optimizer through `optimizerStep()`, never the raw vtable (#432) |
| `c-format-check` | `clang-format --dry-run -Werror` over `src`, `test`, `examples` |
| `c-build-and-test` | `unit_test` preset: configure, build, ctest |
| `c-asan-build-and-test` | The suite under ASan + UBSan |
| `c-ubsan-build-and-test` | The suite under overflow-focused UBSan |
| `c-bit-parity` | Builds **all** example binaries, then parity of the C twins against PyTorch (exact int32 predictions; ECG reconstructions via allclose) |
| `python-test` | `uv run pytest` |

The devenv `ci` script mirrors this pipeline locally (all but the UBSan and
bit-parity jobs) — run it before opening a PR. If your change touches
numerics or training behavior, also run the `unit_test_ubsan` preset and the
affected examples' parity checks yourself.

## The bigger picture

This repo is half of a two-repo pipeline with
[es-ude/elastic-ai.creator](https://github.com/es-ude/elastic-ai.creator):

```text
PyTorch model → torch2ir → IR + shapes → annotated IR → ir2c → C code → MCU
              ╰―― elastic-ai.creator ――╯               ╰――― this repo ―――╯
```

**creator** owns the IR definition, shape inference and protocol interfaces;
**this repo** owns the provider implementations, `ir2c` code generation, the
C training framework, resource estimation, and all hardware-specific logic.
The Python side here (`python/odt/`) is a skeleton awaiting that work.

## License

MIT — see [`LICENSE.md`](LICENSE.md).
