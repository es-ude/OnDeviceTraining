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
- `cmake/` — build helpers
- `CMakePresets.json` — reproducible CMake configurations
- `devenv.*` — a pinned developer environment (optional, depending on your setup)
- MIT license

If you’re new: expect this project to evolve quickly. This README is written to clearly separate
**current scope** from **planned features**.

---

## Design principles

- **Portability-first:** keep the training core independent of heavy runtimes and OS assumptions.
- **MCU realism:** optimize for peak RAM, temporary buffers, and predictable memory behavior.
- **Host equivalence:** run the same model code on PC for debugging/profiling and cross-checking.
- **Incremental complexity:** start minimal, then add optimizers, quantization, and memory knobs without breaking the baseline.

---

## Roadmap (planned additions)

This section is a **direction**, not a promise.

### 1) Training core
- Forward + backward support for a growing set of “MCU-credible” building blocks
- Loss functions and training loops for supervised learning
- Optimizers (SGD, momentum, Adam variants) with configurable state footprint

### 2) Memory & compute optimization knobs
- Gradient checkpointing / recomputation strategies
- Buffer reuse and static memory planning
- Operator fusion where it reduces peak RAM or improves throughput

### 3) Quantization & low-precision training
- Quantized inference baseline with consistent numerics across host/MCU
- QAT-style flows and integer-friendly training variants
- Mixed precision strategies with explicit memory accounting

### 4) Tooling & usability
- Minimal “example zoo”, e.g.:
  - tiny MLP (XOR / toy classification)
  - time-series classifier
  - small conv net
- Profiling hooks and per-layer accounting (MACs, temporary buffers, parameter footprint)
- CI for host builds + sanity tests

### 5) Platform targets
- Reference ports for common MCU families (board + toolchain recipes)
- Clean hardware abstraction boundary to keep the training core platform-agnostic

--