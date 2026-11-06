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
uv run tools/bfp-census/census.py --strict --roots tools/bfp-census/har_train_step_roots.txt  # the whole HAR training step
uv run tools/bfp-census/census.py --self-test      # + the step gate's negative controls when built
uv run tools/bfp-census/trace_step.py               # regenerate har_train_step_trace.txt (host build)
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
  `har_train_step_roots.txt`: the whole training step of the integer path (bfp8full,
  both heads): `include`s the forward and backward sets and adds the step's driver
  (epoch and batch loops, the calculate-grads interpreter, the loss value, the
  optimizer step, the grad reset, the RNG keys) and the targets the step reaches only
  through a pointer (the remat HEAP scheduler's rows, every HAR layer's
  `calcOutputShape`, the loader's `getBatch`); the --strict gate of the full path.
  Its only float is the loss island of the float head and the loss value (both
  heads): the samples and labels are pre-converted to BFP, and the input layer's
  converting arm is a declared boundary.
  `include <file>` (relative to the including file) pulls a roots file in; a root
  line seen before is dropped, so a union lists each root once.
  Format `<source.c> <function>`: the census follows direct calls from the root, within
  the object and across objects. `<source.c> <function> self`: only the function's own
  call sites count and nothing is followed -- for dispatchers whose callees span every
  dtype's arm, so their bodies are still checked.
- `allowlist.txt`: every entry names the ONE root it applies to (`<source.c>:<function>`),
  so an entry for one root never hides another root's hits. `<root> <function>` stops
  traversal there (the float boundaries: input quantization, loss-facing dequant);
  `<root> <function> <symbol> <count>` allows exactly that many call sites. More sites
  are hits; fewer make the entry stale, so a new division in an allowed function shows.
  An entry certifies call SITES, not how often they run: a division allowed in
  `calcElementIndexByIndices` still runs once per MAC (the shared index math the
  same-algorithm rule keeps on purpose), so a passing census does not prove there is
  no per-MAC software division.
- Not followed: calls through function pointers or vtables; list such targets as roots.
  `har_train_step_roots.txt` is closed over them by its `indirect
  har_train_step_indirect.txt` line: every function whose address the build takes
  (an ABS32 relocation: vtable and conversion-cell data, kernel and callback pointers
  loaded in code) must be covered by a root -- named, or reached by a walk -- or carry
  an `exclude <src.c> <function> <reason>` line saying why it never runs on the step;
  and every covered function that calls through a register (`blx`, or a `bx` that is
  not a return) must carry a `site <src.c> <function> <count> <targets>` line with
  exactly that count. A new layer type, kernel arm, conversion cell, scheduler row or
  callback therefore fails `--strict` until it is rooted or excluded; an entry that
  stops applying is stale. Targets outside the objects (the trainer's `getSample`,
  `getDatasetSize`, `calculateGradsFn`, a phase hook) show up as `site` lines.
  `--self-test` with a built `BUILD_DIR` injects forbidden helpers (float, division,
  int64) into the pointer-only targets (remat HEAP rows, `calcOutputShape`,
  `reluForward`, `getBatch`), drops a root and adds an undeclared pointer call; the
  step gate must fail on each, for that reason.
- The step set is closed over the direct calls of its `self` roots, statically: a
  self root checks only its own call sites, so every function it calls directly must
  be walked (or named) by some root, or be declared in the indirect file as
  `boundary <self.c> <self root> <callee.c> <callee> <reason>` -- an arm the bfp8full
  step does not take (another dtype, the m > 1 path, a cold reject), with the reason.
  Walked roots follow every direct call, and their only exits are the allow-list's
  stop entries, so the closure does not depend on what any run observed.
- As a cross-check, the `trace har_train_step_trace.txt` line names the
  `<translation unit> <function>` pairs an instrumented host run of both bfp8full
  heads executes (three steps; `uv run tools/bfp-census/trace_step.py` builds the
  trainer with `-finstrument-functions` and `step_trace_hook.c` under
  `build/step_trace` and reads the translation units from the macOS link map). Every
  pair that is a function of the arm objects must be walked by a root or carry a
  `trace-exclude <source.c> <function> <reason>` line (today: the float internals of
  the stopped boundaries); pairs outside the arm objects (the trainer, functions the
  arm build inlines into a walked caller) are skipped. The trace stores a fingerprint
  of `src/`, the trainer and the trace tool; `--strict` fails when it no longer
  matches ("re-run trace_step.py").
- Functions are identified by (translation unit, name) everywhere: roots, allow-list,
  exclusions, boundaries, trace. An allow-list function may be written `<name>` only
  when one object defines it; a name several objects define (static helpers,
  header `static inline` copies) must be written `<source.c>:<name>`, else the census
  stops with that message. A callee name the walk cannot resolve to one object fails
  `--strict`.
- The negative controls (`--self-test` with the arm build) inject forbidden helpers
  into the pointer-only and directly-called roots, graft a new helper with a float
  call onto `reluForward` (not in the trace), add a second static `clampInt32` with a
  float call in `Relu.c` (from the self root, from a walked root, and in the trace),
  drop roots, add an undeclared pointer call, fake a changed source fingerprint, and
  inject a float call into the input layer's straight-through and into the integer
  head's BFP-label read;
  each must fail `--strict` for its own reason.
  The dtype dispatchers (`executeOp`, `writeOutConversion`, the layers' `*Forward`
  switches) are `self` roots: a full walk from them reaches every dtype's arm. What a
  self root cannot see -- a float route through a callee -- is, on the BFP-wire forward,
  only the staging of a FLOAT32 operand, which is a run-time error in `executeOp`.
- A boundary root whose allow-list entry stops at the root itself walks nothing: its row
  documents the boundary, it does not check it.
- Reachability is an over-approximation when a reached function switches on dtype
  (hits in a non-BFP branch show up too); `-v` shows which root reaches each hit.
- Set `ARM_OBJDUMP` if `arm-none-eabi-objdump` is not on PATH.
