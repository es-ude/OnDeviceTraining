# On Device Training Framework

**OnDeviceTraining** is a lightweight C/CMake framework for **inference and on-device training (backpropagation)** of deep neural networks across two execution contexts:

- **MCU targets** (resource-constrained embedded systems)
- **PC/host builds** (fast iteration, debugging, reference behavior)

> *Framework for Inference and Training of Deep Neural Networks on MCU + PC*

---

## Motivation

Most TinyML stacks are built for inference-only. OnDeviceTraining targets the harder regime:

- local personalization / adaptation
- continual learning on streaming sensor data
- tiny fine-tuning loops under strict RAM/Flash/compute budgets

The project aims to provide a **research-friendly but engineering-minded** codebase for:

- training algorithms and memory/computation trade-offs
- portability across MCUs
- host-side debugging and reference behavior (“ground truth” runs)

---

## What this repository contains today

The repository is currently structured as a **CMake-based C project** and includes:

- `src/` — core sources
- `test/unit/` — unit tests
- `examples/` — end-to-end training demos (PyTorch reference + C twin + parity checks) — see [`examples/README.md`](examples/README.md)
- `docs/` — feature matrix, continual-learning guide, contributor conventions
- `python/` — the Python package `python/odt/` (the remat budget planner `python/odt/planner/`; the other subpackages are not started yet) and `python/tests/` (its tests and those of the example tooling)
- `cmake/` — build helpers
- `CMakePresets.json` — reproducible CMake configurations
- `pyproject.toml` / `uv.lock` — Python dependencies, managed with [uv](https://docs.astral.sh/uv/)
- `devenv.*` — a pinned developer environment (optional, depending on your setup)
- `.github/workflows/ci.yml` — the CI pipeline
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — repository map, build/test, and contribution workflow
- MIT license

What the framework supports today (layers, optimizers, quantization, serialization, …) is
tracked in [`docs/FEATURES.md`](docs/FEATURES.md) — that file is the source of truth for current
capabilities. Expect this project to evolve quickly.

---

## Design principles

- **Portability-first:** keep the training core independent of heavy runtimes and OS assumptions.
- **MCU realism:** optimize for peak RAM, temporary buffers, and predictable memory behavior.
- **Host equivalence:** run the same model code on PC for debugging/profiling and cross-checking.
- **Incremental complexity:** start minimal, then add optimizers, quantization, and memory knobs without breaking the baseline.

---

## Build options

Pass options at configure time (`cmake -B build -DODT_REMAT_CHECK=OFF`) or pick a preset from
[`CMakePresets.json`](CMakePresets.json). `cmake -LH <build-dir>` lists every option with its help
text. Projects that only `add_subdirectory(OnDeviceTraining/src)` get the same options and defaults.

| Option | Default | What it does / when to change it |
|---|---|---|
| `HOST` | `ON` | Host PC build. Set `OFF` for MCU firmware (defines `BUILDING_FOR_REMOTE`). |
| `BUILD_EXAMPLES` | `OFF` | Build the training programs under `examples/`. |
| `ODT_BUILD_TESTS` | `ON` | Build the unit tests (top-level builds only). Turn `OFF` when you only need the libraries. |
| `ODT_BUILD_DIFFERENTIAL` | `OFF` | Build the differential numerics driver in `test/differential`. |
| `DEBUG_MODE_INFO` | `OFF` | Print info-level log messages. Errors are always printed. |
| `DEBUG_MODE_DEBUG` | `OFF` | Print debug-level log messages (verbose; host debugging). |
| `ODT_MEM_PROFILE` | `OFF` | Track heap use of the framework's allocations to measure peak RAM. Adds overhead; leave `OFF` in firmware. |
| `ODT_TRACK_INSTRUCTIONS` | `OFF` | Count arithmetic operations in the kernels. Adds overhead; profiling only. |
| `ODT_REMAT_VERIFY` | `OFF` | Overwrite rematerialized buffers with poison bytes to catch stale reads. Test builds only. |
| `ODT_REMAT_CHECK` | `ON` | Validate every rematerialization step before running it. Set `OFF` on MCUs to save flash (about 5 KB) once your model is known to be valid. |
| `ODT_SCALE_STRICT` | `OFF` | Exit when a stored SYM/SYM_INT32/ASYM scale is not an exact 15-bit dyadic value. Test builds only. |
