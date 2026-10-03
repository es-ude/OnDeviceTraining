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
- `python/` — Python package skeleton (`python/odt/`, not started yet) and `python/tests/` (tests for the example tooling)
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
