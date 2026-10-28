# Allocation locality

## Allocation Locality

Only `src/userApi/` may call `malloc`, `calloc`, `realloc`, or `free` directly. All other code (sub-layers under `src/`, tests under `test/`) must route allocations through `reserveMemory` and `freeReservedMemory` in `src/userApi/StorageApi.{c,h}`.

Why:
- MCU stack overflows are silent killers; routing through StorageApi keeps stack usage predictable and small.
- Reviewers know exactly where to look for memory issues: `src/userApi/`.
- A future handle-based allocator can subsume the entire allocation surface in one API change instead of touching every call site.

Enforcement:
- A CI job (`alloc-locality` in `.github/workflows/ci.yml`) runs `git grep` against `src/` and `test/` (excluding `src/userApi/`) and fails the build on any match. Comments are excluded from the match.
- Exceptions: none today. If a use-case arises that genuinely needs a direct alloc primitive outside `src/userApi/`, escalate via a PR comment so the rule itself can be revisited.

## Checked size math

Size products and sums over counts that come from files, configs or callers go through `checkedMulSize`/`checkedAddSize` in `src/common/include/CheckedSize.h`; on `false` the caller fails fast by name and never reads the result. Carve-out: the memory-layer internals `reserveMemory` (`src/userApi/StorageApi.c`) and the stack-page rounding in `src/userApi/MemProfile.c` keep their hand-rolled `SIZE_MAX` guards on purpose (each guards one add of a header or page pad inside that layer itself); do not convert them.
