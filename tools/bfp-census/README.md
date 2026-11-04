# BFP forward float census

Counts the soft-float, libm, division, int64 and memcpy/memset calls that the BFP
forward kernels reach when built for Cortex-M0+ (no FPU, no hardware divide), where
each such call is real cost and a leftover float is a bug against the integer-only goal.

When to use: after touching any BFP forward kernel or conversion, and as the gate for
the integer-only forward work (`--strict` exits 1 on any hit outside the allow-list).

```
cmake --preset arm_cross_m0plus            # needs arm-none-eabi-gcc on PATH
cmake --build --preset arm_cross_m0plus
uv run tools/bfp-census/census.py          # report mode, always exit 0
uv run tools/bfp-census/census.py -v --strict
```

- `forward_roots.txt`: the functions that count as "BFP forward" (`<source.c> <function>`).
  The census follows direct calls from each root, within the object and across objects.
- `allowlist.txt`: permitted boundaries (input quantization, loss-facing dequant).
  `<function>` stops traversal there; `<function> <symbol>` allows one symbol in one function.
- Not followed: calls through function pointers or vtables; list such targets as roots.
- Reachability is an over-approximation when a reached function switches on dtype
  (hits in a non-BFP branch show up too); `-v` shows which root reaches each hit.
- Set `ARM_OBJDUMP` if `arm-none-eabi-objdump` is not on PATH.
