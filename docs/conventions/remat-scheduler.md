# Remat scheduler

The training driver (`CalculateGradsSequential.c`) runs every grads call as a
stream of steps that a **remat scheduler** hands out
(`src/userApi/training_loop/remat/`, #4). A scheduler is one **row** (ARENA or
HEAP today) over shared code: the buffer table and the static plan
(`RematPlan`), the dispatch (`RematScheduler.c`) and the validator
(`RematCheck`). A caller initialises one with `rematArenaInit` or
`rematHeapInit`, passes it as `trainingCall_t.remat` (`TrainingCall.h`), and
deinitialises it with `rematSchedulerDeinit`. A NULL scheduler means an
ephemeral HEAP scheduler with the STORE_ALL plan per grads call.

## The seam rule: rows own bytes, nothing else

A row writes a header's `data` pointer only through `rematWireBind` /
`rematWireRelease`, the row SDK in `RematWireTable.c`. Header content (shape,
quantization, qConfig, BFP exponents), layer and loss execution, hooks, trace
and randomness belong to the shared code and the driver. Rows cannot produce a
stale header, and the validator's bind-generation check stays sound.

Who owns which memory (every block comes from `reserveMemory` and goes back
through `freeReservedMemory`; bound wire bytes never come from raw `malloc`,
a static array or the stack; the scheduler struct and the validator state
are plain objects their owner places, e.g. on a stack frame):
- the `rematScheduler_t` struct and ACT 0 (the input sample, borrowed,
  never bound or poisoned): the caller;
- the buffer table with its header slab, and the static plan: `RematPlan`,
  one block each, from init to deinit;
- ARENA's offsets table and arena block: the ARENA row, resident from init to
  deinit; HEAP's per-range blocks: the HEAP row, from range begin to range end;
- the validator state: the driver's stack frame, one call.

## The row contract (R1–R8)

Every row satisfies these before it joins the conformance matrix.
- **R1 Bytes, never metadata.** Set `->data` only through `rematWireBind` /
  `rematWireRelease`. Never write header content.
- **R2 Values, not pointers.** `next` fills a caller-provided `rematStep_t`.
  The driver resolves operands through `RematCheck`; a row never hands them
  over.
- **R3 Versioned binding.** Every bind increments `bindGen`. Bind a wire only
  for a step that produces it (FORWARD, LOSS_BACKWARD, BACKWARD).
- **R4 No side effects; RNG-free.** Never run a layer or a loss, fire a hook,
  or call a trace sink. Call no `rng*` function: rows neither draw nor set
  anything, and never receive or derive a key.
- **R5 Placement is private.** Offsets, pools and tiers live in the row's
  member of `rematScheduler_t.row`. The shared table and plan stay
  placement-free.
- **R6 Offline inputs are verified.** An offline placement enters only as an
  ARENA init input and is always re-verified, never trusted.
- **R7 Readiness.** When `next` returns a step, every byte it reads is
  complete and every buffer it writes is bound.
- **R8 Failure ownership.** The validator exits on grammar, residency or
  bind-generation violations (a scheduler bug). The row exits on
  resource failure (`reserveMemory` NULL mid-call), naming the step and the
  wire. Only an allocation failure at init is recoverable (the init returns
  `false`); an unsupported model or a broken size or range limit exits at
  init too.
- **Lifecycle.** Every `rematSchedulerFunctions_t` slot is mandatory. `end`
  leaves no non-borrowed wire resident. `deinit` frees only row-private blocks
  and is safe after a failed init. One scheduler instance per concurrently
  running training stream.

## The plan contract: a persistent scheduler

A scheduler binds against the model as it is at each call.
- **An edit that keeps the key is adopted at the next bind**, exactly as
  without a scheduler: a template's qMaxBits or rounding mode, or a
  `deserializeModel` into a skeleton whose wire widths differ.
- **An edit that changes a key fact exits at the next bind**, naming the wire
  and the field (`rematWireTableBind: key mismatch on …`): the batch size, a
  rank, a dtype, a byte count, a layer type, `numGroups` above `expCapacity`,
  and **freezing or unfreezing a layer mid-run**. The exit fires before any
  header is written. To change a key fact, call `rematSchedulerDeinit` and
  initialise a fresh scheduler.
- A scheduler keyed to a different input shape than the one the grads level
  receives exits on its first call, naming ACT 0's field.
- **Evaluation binds the same key.** `inferenceWithLoss` on a scheduler
  exits at bind when its input differs from the training input the
  scheduler was keyed to (batch rows, rank, order, dtype), naming ACT 0's
  field. `trainingRun` avoids that for its own micro-batch artefacts: it
  hands evaluation the scheduler only when every eval chunk has the training
  row count (remat D19), judged once at entry on the eval loader's nominal
  count. A loader whose stream differs from its nominal count (a replay
  wrapper, for example) can still end on a ragged chunk, which exits at that
  chunk's bind. An eval sample whose own shape or dtype differs from the
  training sample's exits too: at its call's bind, or, when evaluation stacks
  samples (m > 1), already at the gather check if it is not FLOAT32 or
  differs from the first eval sample.

## Two tiers of checking

- **Always on**, release and firmware included: the validator's step rules
  and stream checks, and ARENA's init-time placement verifier. There is no
  opt-out macro.
- **`ODT_REMAT_VERIFY`** (a CMake option, default OFF): dtype-aware poison of
  wire bytes at bind and at release, and the driver's strict dead-wire check.
  ON in the `unit_test`, `unit_test_debug` and `unit_test_asan` presets, and in
  `unit_test_ubsan` through `unit_test_debug`; OFF in `unit_test_error`,
  `unit_test_info` and every non-test preset. It is a PUBLIC compile definition
  on `RematPlan`: a test whose `LIB_UNDER_TEST` is `RematScheduler`,
  `RematCheck` or `CalculateGradsSequential` lists `RematPlan` in `MORE_LIBS`
  before it `#ifdef`s on it.

## Adding a row

- **Additive edits, plus one token.** In `RematScheduler.h`: a member
  appended to `rematSchedulerType_t`, a member of the `row` union, the row's
  init declaration. Its entry points go into the private `RematRows.h`, its
  vtable entry into `RematScheduler.c`, whose `_Static_assert` swaps one
  token for the new last `rematSchedulerType_t` member (no count member).
  `CalculateGradsSequential.c` and `RematCheck.c` stay unchanged.
- **The proof.** The row joins the conformance matrix (new cells in `g_cells`,
  `test/unit/training_loop/UnitTestCalculateGradsConformance.c`) and every
  both-rows death in `UnitTestCalculateGradsDeaths.c`.
- **The gate.** A new row file under `remat/` is gated with no edit. A new
  **shared** file there goes on the gate's exclusion list (next section) with
  a one-line "shared, because …" reason.

## The `remat-row-contract` gate

A CI job (`remat-row-contract` in `.github/workflows/ci.yml`) enforces R1's
data-pointer rule and R4's hook, layer, loss and RNG bans as text, because no
CMake boundary can: `OdtHook.h` reaches every library through `Common__hdrs`,
and hook, layer, loss and RNG symbols resolve at the final link. The devenv
`ci` script runs the same run block; outside a git work tree (a jj secondary
workspace) it prints `remat-row-contract: SKIPPED … this is NOT a pass` and
continues, while the CI job fails closed.

**Files.** Every `.c` and `.h` under `src/userApi/training_loop/remat/`,
`include/` too, except an explicit list of shared files: `RematCheck.c`,
`RematPlan.c`, `RematPlanPolicy.c`, `RematPlanPolicy.h`, `RematScheduler.c`,
`RematWireTable.c` (the row SDK, the one legitimate `->data` writer),
`RematCheckedSize.h` and `RematRows.h`. A new file is therefore a row until
someone says otherwise. `RematPlace.h`, ARENA's private placement, is gated
as row code, and so is the public API in `include/`: its comments may name a
banned symbol, its code may not. Every exclusion carries a "shared, because
…" reason in the job, and a reviewer of any edit to the list checks that the
file is not a row: its functions are not among `RematRows.h`'s entry points,
and it is not a row-private header.

**Pattern** (`git grep -nE`; an explicit character class stands in for `\b`,
which macOS ERE ignores):
- `odtHook(Fire|Set)`, `layerFunctions`, `lossFunctions`: hooks, layer and
  loss execution;
- `rng[A-Z]…(` after a non-identifier character: every call of the RNG API,
  which names its functions `rng` plus an upper-case letter (`rngSetSeed`,
  `rngNextFloatCtx`, the keyed API that replaces the global stream); a
  lower-case `rngseed(` would not match, and the type `rng32_t` is not a call;
- `.data` or `->data` followed by `=`, a compound assignment (`+=`, `<<=`, …),
  `++` or `--`, or preceded by `++`/`--` (an operand without blanks, e.g.
  `++(s->wires[i].hdr->data)`): every write of a data pointer, in
  clang-format's spacing (`h->data`, `*hdr`), which `c-format-check`
  enforces. Element writes such as `h->data[0] = 1` (the row's own bytes) and
  comparisons such as `h->data == NULL` do not match.

A matching line is dropped when it starts with `//`, with a `/*` comment that
runs to the line end, or with a `*` followed by a blank, `/` or the line end
(a block-comment line). Code after a leading `/* … */`, a line starting
`*hdr = …` (a dereference) and a trailing comment on a code line are scanned.

**Self-checks.** The job fails if `RematArena.c` or `RematHeap.c` drops out
of the gated set (renaming a row means updating that list), and if `git grep`
itself fails. Before it passes, it plants at least one line per pattern arm, a
new row file and six control lines into a throwaway repository and checks
that each fires or stays silent as it should. It never plants into the
checkout.

**Stated honestly.** The gate is a text check, not a boundary. It does not
look at the rest of R1 and R4: a write of other header content
(`h->shape = …`, a whole-header store `*hdr = saved;`) and a trace-sink call
rest on review and the tests. Within its scope it misses a
write through a copied pointer (`uint8_t **p = &h->data; *p = x;`),
`memcpy(&h->data, …)`, a macro that expands to a banned call, a function
pointer taken without a call (`f = rngNextFloat;`), a call split across
lines, a parenthesized lvalue before the operator (`(h->data)++`), a
block comment between `data` and the operator (`h->data /* x */ = b;`), a
`++`/`--` operand that holds a blank (`++(f(a, b)->data)`), and row code in
a file that is not a `.c` or `.h`. Poison, ASan, the validator's bind
generations and the death tests catch those at test time. It also flags a
read whose index pre-increments (`p = w[++i].hdr->data;`): split that line;
a line of two block comments that names a banned symbol: merge them; and an
inner block-comment line that names one without a leading `*`: add the `*`.
`git grep` sees tracked files only: a local run misses a brand-new file until
git tracks it (after the next jj snapshot in a colocated checkout, after
`git add` elsewhere), while CI sees every committed file.

**Not gated yet.** A raw `malloc`/`calloc`/`realloc`/`free` in a row: the
`alloc-locality` job excludes `src/userApi/`, and adding the allocation family
here would trip over a string literal in `RematPlace.h`. The rule still holds:
rows allocate through `reserveMemory` only ([allocation.md](allocation.md)).

**EVICT.** A future EVICT row selects victims deterministically. If it is
ever given a key branch of its own, the `rng` arm needs an exception, added in
the job and stated here.

## Tests: the decorator seam and the death-string contract

- **Decorator rows** are the approved test seam: a test points its own
  instance's `fns` at a `const` table whose slots wrap the pass-throughs in
  `test/unit/training_loop/RematTestDecorators.h`. Never patch
  `rematSchedulerFunctions[]` (it is `const`).
- **Death strings live in the source**: the validator's rules in
  `RematCheck.c` (`remat[<row>]: step #<k> <KIND>(layer <l>) violates
  '<rule>'`), the dispatch's in `RematScheduler.c`, the key and bind exits in
  `RematWireTable.c`, the driver's in `CalculateGradsSequential.c`, and each
  row's resource exits in the row (R8). A death test
  (`ASSERT_EXITS_WITH_OUTPUT`, `UnitTestCalculateGradsDeaths.c`) copies a fixed
  substring from the source, never a paraphrase.
- **Mutations** disable one rule with a `false && ` prefix, which keeps the
  text unique for the inverse edit.
- **Behaviour worth knowing:**
  - on a tampered plan the order and range rules fire before any residency
    rule;
  - a re-bind without producing needs release-then-bind: a bare second bind
    exits in `rematWireBind` (`is already bound`);
  - a scheduler cannot be keyed to a raw 1-D sample for a Linear-first model
    (its init exits in `linearCalcOutputShape`), so wrong-shape tests key a
    wrong row count instead.

## Decision register

Code comments, tests and messages cite the remat design by ID. The design
document is not published, so every ID they cite or rely on is stated here,
each as its rule. Cite them with the family prefix: `remat D<n>` (design
decisions), `remat R<n>` (the row contract above), `remat P<n>`
(conformance properties); the rule name `W_dead` needs none. Superseded
decisions and those with nothing in force yet are left out.

**Design decisions.**
- **remat D3** API shape: the grads, batch, epoch and eval levels take a
  trailing NULLable `const trainingCall_t *call` (NULL = a zero-initialised
  call). It carries the scheduler (`remat`); later fields join the struct
  without a second signature sweep. No per-layer flags.
- **remat D9** Acceptance: a call on any row and plan is bit-identical to
  the store-all driver. Parameter grads, loss, output snapshot, SYM scales
  and BFP exponents compare by `memcmp`; while the global RNG stream exists,
  its state after the call matches too (remat P2).
- **remat D11** ARENA and HEAP are peer rows behind one swappable,
  step-by-step scheduler, which picks the steps. Placement offsets are
  ARENA-private. Every memory block comes from `reserveMemory`.
- **remat D17** The schedule key fixes the batch size exactly: a scheduler
  keyed to batch B exits at bind on any other B. No plan-per-B cache.
- **remat D19** Evaluation runs on the caller's scheduler: `inferenceWithLoss`
  with a non-NULL `call->remat` walks the plan's EVAL program (FORWARD and
  LOSS_FORWARD only) in that scheduler's memory, with ACT 0 the caller's
  input, borrowed. Output values, shape, dtype, dynamic quantization state
  (SYM scale, BFP exponents), the loss and the model state after the call
  are bit-identical to the NULL path's; the one difference is an input's
  sparsity marker, which this path drops (the output is unmarked) while the
  NULL path keeps it. `trainingRun` hands its eval calls `options->remat` only
  when every eval chunk has the training row count: `evalMicroBatchSize ==
  microBatchSize`, and the eval loader's nominal count
  (`datasetSize / batchSize * batchSize`) a multiple of it. Otherwise its
  eval calls get no scheduler, and the run completes as without one. The
  sample shape is not judged at entry: an eval sample whose shape or dtype
  differs from the key exits with a named error, at its call's bind or, when
  evaluation stacks samples, at the gather check. The public `evaluation*`
  functions pass none.
- **remat D20** n = 1 under CrossEntropy keeps the signed `top = -1`: the
  stream has a LOSS_BACKWARD step and no BACKWARD step.
- **remat D23** A step is a 4-byte `{kind, layer}` value (`rematStep_t`);
  the driver resolves its operands (R2).
- **remat D24** Rows dispatch through a `const` vtable and a per-instance
  `fns` pointer that the row's init sets. Tests inject faults with
  decorator rows on their own instance, never by patching the vtable.
- **remat D25** Evaluation inside the training arena adds no byte to it:
  ARENA places the EVAL program two-ended (an even ACT at offset 0, an odd
  ACT top-aligned), which the verified training layout proves fits, and HEAP
  reserves one block per EVAL range. An eval bind runs the full key check
  and writes no GRAD header. `inference()`, `inferenceBatched()` and the
  public `evaluation*` functions keep their signatures and their per-call
  buffers.
- **remat D26** ARENA's arena is resident: reserved at init, held until
  deinit.
- **remat D28** The validator (`RematCheck`) is always compiled in,
  firmware included. There is no opt-out macro.
- **remat D29** A violation mid-call exits (`exit(1)`), naming the row, the
  step and the rule. Only an allocation failure at init is recoverable: the
  init returns `false`, and nothing ran. An unsupported model or a broken
  size or range limit exits at init too (remat D55, D60).
- **remat D30** A NULL scheduler means an ephemeral HEAP scheduler with the
  STORE_ALL plan per grads call, bit-identical to the pre-remat driver.
- **remat D31** Three libraries: `RematPlan` (wire table and static plan),
  `RematScheduler` (dispatch and rows) and `RematCheck` (the validator).
  Rows write `->data` only through bind and release (R1).
- **remat D32** Offline planners enter as plan inputs. PAGED and EVICT are
  later rows that need no driver or interface change.
- **remat D33** ARENA conformance runs under `unit_test_asan` (its
  `undefined` sanitizer covers alignment) plus manual poisoning.
- **remat D42** Test strategy: an in-process Legacy oracle (the pre-remat
  driver, kept verbatim), a row-contract harness, deaths matched on message
  substrings, tampered schedulers, conformance P1-P10, poison on both rows.
- **remat D44** Keyed randomness (planned): every random value becomes a
  function of an explicit key and a counter, and the module-global stream
  goes away. In force now: rows and the validator call no RNG function.
- **remat D50** Rows never receive or derive a key and call no `rng`
  function (the `remat-row-contract` gate). A future EVICT row selects
  victims by a deterministic scan; sampling there is an open question.
- **remat D54** Two-phase bind: derive every key fact into stack scratch,
  compare the full key, including dtype, rank, byte count and `numGroups`
  against `expCapacity`, and only then write a header. `rematWireBind`'s
  inherited GRAD derivation follows the same order.
- **remat D55** ARENA init order: the offsets table in its own heap block,
  first-fit-decreasing placement into it, the placement verifier, then the
  arena data block. The report flags `planned`, `placed`, `dataReserved`
  imply one another in that order, so a failed init still reports what
  its flags allow. More than `ODT_REMAT_MAX_RANGES` (default 1024) ranges
  exit at init, naming it.
- **remat D57** The Legacy oracle retires when keyed randomness reaches the
  layer path; from then on fixtures compare against an explicit HEAP +
  STORE_ALL reference with the same call key. Until then nobody fixes it:
  a fix would move the oracle.
- **remat D60** Every size product and sum in the remat libraries is
  overflow-checked. An overflow exits at init, naming the wire and the
  quantity, before any reservation.

**Row contract** (in full above).
- **remat R1** A row writes a header's `data` pointer only through
  `rematWireBind` / `rematWireRelease`, and no other header content.
- **remat R2** `next` fills a caller-provided `rematStep_t`; the driver
  resolves the operands, a row never hands them over.
- **remat R3** Every bind increments `bindGen`; a wire is bound only for a
  step that produces it.
- **remat R4** A row runs no layer or loss, fires no hook, calls no trace
  sink and calls no `rng` function.
- **remat R5** Placement (offsets, pools, tiers) stays in the row's member
  of `rematScheduler_t.row`; the shared table and plan are placement-free.
- **remat R6** An offline placement enters only as an ARENA init input and
  is always re-verified.
- **remat R7** When `next` returns a step, every byte it reads is complete
  and every buffer it writes is bound.
- **remat R8** The validator exits on scheduler bugs, the row on resource
  failure mid-call; only an allocation failure at init is recoverable (the
  init returns `false`). R8 lifecycle: every vtable slot is mandatory, `end`
  leaves no non-borrowed wire resident, and `deinit` is safe after a failed
  init.

**Conformance properties** (each row and plan against the Legacy oracle).
- **remat P1** Parameter grads, loss, output snapshot, SYM scales and BFP
  exponents equal the oracle's byte for byte.
- **remat P2** The global RNG stream after the call equals the oracle's;
  once a call draws nothing from it, it is unchanged across the call; it
  retires with the stream.
- **remat P3** Exactly four hook events per call, in order.
- **remat P4** The trace's `(idx, phase)` sequence and tensors match.
- **remat P5** Once the caller frees the returned `trainingStats_t`,
  `ODT_MEM_PROFILE`'s current bytes are back at their pre-call value.
- **remat P6** Two consecutive calls on one persistent instance, with a
  key-preserving template edit (qMaxBits 8 -> 16) between them, equal two
  oracle calls.
- **remat P7** Key deaths are identical on every row: the key check is
  shared code.
- **remat P8** After a call, `observedPeakLiveBytes` equals the
  `peakLiveBytes` of the program the call walked (TRAIN, or EVAL for an
  eval call).
- **remat P9** The caller's input header and bytes stay unchanged, and a
  scheduler built on sample A runs sample B of the same shape.
- **remat P10** Clean under ASan and UBSan.

**Rule names.**
- **W_dead** A BACKWARD whose layer does not read its input
  (`layerBackwardReadsInput` false) may get that input as a header with
  `data == NULL` (the LIVENESS state). Under `ODT_REMAT_VERIFY` the driver
  passes such a dead header on every plan (strict W_dead), so a read the
  read-set table denies crashes under STORE_ALL too.
