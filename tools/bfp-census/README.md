# BFP forward float census

Counts the soft-float, libm, division, int64 and memcpy/memset calls that the BFP
forward kernels reach when built for Cortex-M0+ (no FPU, no hardware divide), where
each such call is real cost and a leftover float is a bug against the integer-only goal.

When to use: after touching any BFP forward kernel or conversion, and as the gate for
the integer-only HAR forward (`--strict` exits 1 on any float/double/libm/division/int64
hit outside the allow-list, on a missing root, or on a stale allow-list entry;
memcpy/memset and clz are reported only).

```
cmake --preset arm_cross_m0plus            # needs arm-none-eabi-gcc on PATH
cmake --build --preset arm_cross_m0plus
uv run tools/bfp-census/census.py --strict -v                                   # the HAR gate
uv run tools/bfp-census/census.py --roots tools/bfp-census/forward_roots.txt    # every BFP forward kernel, report
uv run tools/bfp-census/census.py --strict --roots tools/bfp-census/har_backward_roots.txt  # the integer HAR backward so far
uv run tools/bfp-census/census.py --self-test
```

- `har_forward_roots.txt` (default): the HAR classifier's integer forward, the gate set.
  `forward_roots.txt`: every BFP forward kernel, including ops not yet integer (report).
  `har_backward_roots.txt`: the HAR backward paths that are integer so far (today the
  Conv1d and Linear dx and the dx-wire re-block, and the Conv1d and Linear weightGrad /
  biasGrad into BFP grads with their integer accumulate, and the macro-batch mean
  scale of BFP grads, and the SGD step on BFP storage: momentum store and parameter update; the
  ReLU, MaxPool1d, AvgPool1d, Flatten, standalone Softmax and input-layer dx; and the
  CrossEntropy loss boundary,
  the one allow-listed float island), gated with `--strict` like
  the forward; a path
  joins it when it turns integer.
  Format `<source.c> <function>`: the census follows direct calls from the root, within
  the object and across objects. `<source.c> <function> self`: only the function's own
  call sites count and nothing is followed -- for dispatchers whose callees span every
  dtype's arm, so their bodies are still checked.
- `allowlist.txt`: every entry names the ONE root it applies to (`<source.c>:<function>`),
  so an entry for one root never hides another root's hits. `<root> <function>` stops
  traversal there (the float boundaries: input quantization, loss-facing dequant);
  `<root> <function> <symbol> <count>` allows exactly that many call sites. More sites
  are hits; fewer make the entry stale, so a new division in an allowed function shows.
- Not followed: calls through function pointers or vtables; list such targets as roots.
  The dtype dispatchers (`executeOp`, `writeOutConversion`, the layers' `*Forward`
  switches) are `self` roots: a full walk from them reaches every dtype's arm. What a
  self root cannot see -- a float route through a callee -- is, on the BFP-wire forward,
  only the staging of a FLOAT32 operand, which is a run-time error in `executeOp`.
- A boundary root whose allow-list entry stops at the root itself walks nothing: its row
  documents the boundary, it does not check it.
- Reachability is an over-approximation when a reached function switches on dtype
  (hits in a non-BFP branch show up too); `-v` shows which root reaches each hit.
- Set `ARM_OBJDUMP` if `arm-none-eabi-objdump` is not on PATH.
