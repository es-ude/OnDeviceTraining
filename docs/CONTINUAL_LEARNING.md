# Continual Learning: PPCA Generative Replay

Memory-bounded protection against catastrophic forgetting under sequential
domain shift (#326). Per class, a small probabilistic-PCA (PPCA) generator
models the input distribution; during fine-tuning on a new domain, synthetic
rehearsal samples drawn from that generator are mixed into every batch — no
raw exemplar from any earlier domain is ever stored. Persistent memory is
`O((k+1)·d)` per class, constant in the number of domains absorbed so far.

Module layout: `src/continual_learning/` (math + serialization, no
allocation), `src/userApi/continual_learning/` (create/free, the replay
data-loader wrapper — all `malloc`/`reserveMemory` lives here). Public
headers: `PpcaReplay.h`, `PpcaReplaySerialize.h`, `PpcaReplayApi.h`.

## When to use

This feature targets **domain-incremental** fine-tuning: the label set is
**fixed** (`numClasses` is set once, at generator-set creation, and never
grows) but the **input distribution shifts** between sessions — a new user, a
new sensor placement, a new deployment environment, a new recording site.
Each new domain arrives as its own fine-tuning session on the *same* model
and the *same* classes; PPCA replay keeps the earlier domains' accuracy from
being overwritten while fine-tuning on the new one.

It is explicitly **not**:

- **Task-incremental / multi-head learning.** There is no mechanism here for
  a growing output space or per-task heads — `ppcaReplaySetCreate(numClasses,
  cfg)` fixes the class count up front.
- **A substitute for full retraining.** When retraining from scratch on the
  union of all data is feasible, do that — it will out-perform any
  rehearsal-from-summary-statistics scheme. This feature is for the
  resource-constrained regime (on-device, memory-bounded, no room to keep
  raw exemplars around and no budget for repeated full retraining) where
  full retraining isn't an option.

## How it works

One PPCA generator per class, `ppcaReplay_t`:

```c
typedef struct {
    size_t dim, rank;       /* d, k */
    tensor_t *mean;         /* [d] */
    tensor_t *basis;        /* [k,d]; rows orthonormal; beyond-rank rows zeroed */
    tensor_t *eigvals;      /* [k]; descending; beyond-rank entries 0 */
    float sigma2, totalVar; /* totalVar = running total scatter */
    uint32_t count;         /* n, absorbed-sample count */
    arithmetic_t mergeMath, streamMath, sampleMath;
    float sigma2Floor, shrinkageGamma;
} ppcaReplay_t;
```

Each generator models its class as a `d`-dimensional Gaussian with a `k`-dim
principal subspace plus isotropic residual noise (Tipping & Bishop's PPCA).
`ppcaReplaySample(g, rng, out)` draws a synthetic sample:

```
x = mu + Σ_i sqrt(max(λ_i − σ², 0)) · z_i · u_i + σ·ε
```

summing over `k_eff = min(rank, count − 1)` kept components (`u_i` = basis
row `i`, `z_i, ε ~ N(0,1)` drawn via `randomNormalCtx` on a caller-owned
`rng32_t *` stream). The kernel **always** draws `rank + dim` variates —
components beyond `k_eff` are drawn and discarded — so the stream position
never depends on how many components a generator currently has (important
for determinism fixtures across updates). A freshly-updated generator with
`count == 1` has `k_eff == 0`: its first sample is just `mu + σ·ε`, isotropic
noise around the single point it has seen so far.

**Wiring into training — a data-loader wrapper, zero framework changes.**
`replayDataLoaderWrap(base, cfg)` (`PpcaReplayApi.h`) wraps an existing
`dataLoader_t`:

```c
typedef struct {
    ppcaReplaySet_t *set;
    size_t samplesPerClass; /* r */
    uint32_t minCount;      /* class eligible once generator count >= minCount */
    rng32_t *stream;        /* caller-owned sampling stream */
} replayLoaderConfig_t;
```

Its `getBatch` calls the base loader first, then appends `r` synthetic
samples **after** the real ones for every class whose generator has
`count >= minCount` — sample 0 stays real, so the `labelRef`/`meanScale`
contracts that assume a real first sample keep holding. Batches grow to
`baseBatch->size + eligible·r`; the optimizer's MEAN divisor already reads
`batch->size`, so it picks up the synthetic samples automatically — **the
replay wrapper adds no second divisor** (the double-scaling trap this
avoids). Item/label pools are lazily built from the *first* real batch,
mirroring its item/label shape and dtype via `getShapeLike`/`getQLike`
(labels must be FLOAT32 one-hot `[numClasses]`).

**Footgun, by design of the ownership split:** the wrapper *borrows* the
base loader's fields (indices, sizes, function pointers) rather than
copying them. Free it with **`freeReplayDataLoader`, never
`freeDataLoader`** — calling `freeDataLoader` on the wrapper would free the
base loader's `indices` array out from under the base loader, which the
caller still owns and must free separately.

## Memory formula

Per-class persistent footprint:

```
bytes_per_class ≈ (k+1)·d·b + 4k + O(1)
```

- `b` is bytes/element for the mean+basis storage dtype: `b = 4` for
  FLOAT32, `b ≈ 1` for packed SYM/ASYM at 8 bits.
- The `4k` term approximates eigvals stored FLOAT32 (4 bytes/entry) — the
  common choice, since eigvals feed directly into the sampling coefficient
  `sqrt(λ−σ²)` and are tiny relative to `d`, so precision usually outweighs
  the sub-byte saving. `eigvalsQ` is an independent storage config, though
  (`ppcaReplayConfig_t`), and can be packed too — eigvals are non-negative,
  so ASYM avoids wasting SYM's sign bit if you do.
- `O(1)` is `sizeof(ppcaReplay_t)` — a handful of scalars/pointers, negligible
  next to the `d`-scaled terms for any realistic input dimension.

This footprint does **not** grow with the number of domains absorbed: each
new domain updates the existing `(mean, basis, eigvals)` in place (see
*Update paths*), it never allocates a new slot per domain.

For the exact figure under any specific config, don't hand-compute the
approximation — call:

- **`ppcaReplayBytes(const ppcaReplay_t *g)`** — the real persistent
  footprint: `sizeof(ppcaReplay_t)` plus `calcNumberOfBytesForData` summed
  over the mean/basis/eigvals tensors at their *actual* configured dtypes.
- **`ppcaReplayIsoExemplarCount(const ppcaReplay_t *g, size_t bytesPerExemplar)`**
  — `ppcaReplayBytes(g) / bytesPerExemplar`, the natural iso-byte comparison
  against exemplar-buffer replay: "how many raw stored samples would this
  generator's footprint have bought instead?"

**Workspace** (shared scratch for path-A merges and path-B streaming,
`ppcaWorkspaceCreate(dim, rank, maxSessionSamples)`) is a *separate*,
one-time allocation — reused across every update call and every class, never
per-class or per-domain. `ppcaWorkspaceBytes(dim, rank, maxSessionSamples)`
returns the exact sum of this inventory (all float32, `p = rank +
maxSessionSamples + 1`):

| Buffer | Elements |
|---|---|
| `bT` (augmented matrix, transposed) | `p·d` |
| `gram` | `p²` |
| `eigvecs` | `p²` |
| `theta` (eigvals of the Gram matrix) | `p` |
| `lambdaOut` | `rank` |
| `rowScales` | `p` |
| `meanBatch` (path B reuses as `xFloat`) | `d` |
| `muOld` | `d` |
| `u` (path-B deflation buffer) | `d` |

(`sigma2Out`/`totalVarOut` are plain scalar fields of `ppcaWorkspace_t`, not
separately heap-allocated — they fold into the struct's own `O(1)`
overhead, not this table.) This workspace is "far beyond safe stack size"
for realistic `dim`/`maxSessionSamples` — it is `reserveMemory`-backed
(heap), created once, freed once with `freePpcaWorkspace`.

## Knob guidance

- **`rank` (k, the subspace dimension)** — start at **8**. Larger `k`
  captures more per-class input variance at `O(k·d)` storage/compute cost;
  too large relative to the number of samples a domain actually contributes
  risks fitting noisy tail eigenvalues (mitigate with `shrinkageGamma`).
- **`samplesPerClass` (r, synthetic samples per eligible class per batch)**
  — start **1–4**. Larger `r` biases each batch more toward rehearsal of
  earlier domains, at the cost of a proportionally larger batch
  (`batch->size` grows by up to `eligible · r`).
- **`minCount`** (gates immature generators out of replay: a class is
  eligible once its generator's `count >= minCount`) — start at **`>= 2k`**
  as a rule of thumb (enough absorbed samples for the merge's
  numerical-rank guard to trust `k` components). **Always keep `minCount >=
  1`.** `minCount == 0` makes a completely untouched generator
  (`count == 0`) eligible for replay — the loader will then try to sample
  it, and `ppcaReplaySample` fail-fasts (`PRINT_ERROR + exit(1)`, "generator
  has absorbed no data") at the very first batch that reaches it, since no
  domain has been absorbed into that generator yet.
- **`maxSessionSamples`** — sizes the shared merge workspace (passed to
  `ppcaWorkspaceCreate`/`ppcaReplaySetCreate` via `ppcaReplayConfig_t`). A
  single `ppcaReplayUpdate` call with more than `maxSessionSamples` rows
  fails fast. Size it for the largest single-session batch you intend to
  merge in one call (typically one domain's worth of a class's samples),
  **not** a global streaming budget — path B (CCIPCA) processes one sample
  at a time and is unaffected by this bound.
- **`sigma2Floor`** — numeric floor for the isotropic residual variance;
  also the generator's *initial* `sigma2` at `ppcaReplayCreate`, before any
  data has been absorbed. Prevents division-by-zero/negative variance under
  floating-point noise. Typical value: `1e-6f`.
- **`shrinkageGamma`** — small-sample eigenvalue shrinkage,
  `λᵢ ← (1−γ)·λᵢ + γ·λ̄` where `λ̄` is the mean of the top *kept*
  eigenvalues; `0` = off (the default). Turn it on when a domain contributes
  few samples relative to `k` (noisy tail eigenvalues), trading a bit of
  subspace fidelity for stability.
- **Storage configs** (`meanQ`/`basisQ`/`eigvalsQ` in `ppcaReplayConfig_t`,
  borrowed pointers cloned via `getQLike`) — the recommended pattern is
  **FLOAT32-active / packed-at-rest**: keep a generator's state FLOAT32
  while it is actively being updated across a session (this also matters
  for path-B streaming accuracy, see *Update paths*), then snapshot to
  SYM/ASYM at 8 bits via `executeConvert` at session or checkpoint
  boundaries for the memory win. `meanQ`/`basisQ`/`eigvalsQ` must each be
  FLOAT32, SYM, or ASYM — SYM_INT32 (a compute format, not storage, #261),
  INT32 (would silently value-cast through the conversion matrix), and BOOL
  (no conversion cell) are all rejected at `ppcaReplayCreate`.
- **`mergeMath`/`streamMath`/`sampleMath`** — must all be `{.type =
  ARITH_FLOAT32, ...}` in v1; see *Limitations*.

## Update paths

Two ways to fold a new domain's data into a generator; pick per class per
domain (they can be mixed across classes/domains, though in practice a
model will use one path consistently).

**Path A — session merge (`ppcaReplayUpdate`, recommended default).**

```c
void ppcaReplayUpdate(ppcaReplay_t *g, const tensor_t *samples /* [m,d] */, ppcaWorkspace_t *ws);
```

One call absorbs an entire session's samples at once (`1 <= m <=
ws->maxSessionSamples`, else fail-fast). It combines the old model `(n, μ,
U, λ, σ²)` with the new batch's `(m, μ_b)` via the Chan, Golub & LeVeque
(1983) two-set mean/scatter merge, **plus** the mean-correction column from
Ross et al. (2008): an extra augmented row `sqrt(n·m/(n+m))·(μ − μ_b)` folded
into the Gram matrix alongside the rescaled old basis and the centered new
samples. Without that correction row, the merged subspace would be biased
exactly when the session mean shifts relative to the old one — precisely
the domain-shift scenario this feature exists for. The augmented Gram
matrix is eigendecomposed (Jacobi), the top `min(rank, numerical_rank)`
components are rotated back and renormalized, and the spectrum/`sigma2` are
recomputed with the session folded in. Cost: `O(p²·d + p³)` per call with
`p = k+m+1` — dominated by the Gram/rotate-back matmul term when
`d ≥ k+m+1`, the common case for the memory-bounded regime this feature
targets — all in the shared workspace (zero steady-state allocation).

**Path B — CCIPCA streaming (`ppcaReplayUpdateStreaming`).**

```c
void ppcaReplayUpdateStreaming(ppcaReplay_t *g, const tensor_t *x /* [d] */, ppcaWorkspace_t *ws);
```

One call per sample, `O(k·d)`, per Weng, Zhang & Hwang (2003)'s CCIPCA:
incrementally rotates the basis toward each new observation without ever
forming a batch or a Gram matrix. This is implemented **deliberately
without** the amnesic discount CCIPCA offers (which would down-weight older
samples over time) — ODT's goal here is *retention*, not gradual
forgetting, so every absorbed sample counts equally, permanently. One
consequence worth knowing: both update paths compute their merge weights
(`n/(n+m)`, `1/(n+1)`, …) by casting the `uint32_t count` to `float`.
float32 represents integers exactly only up to `2^24` (~16.7M); a generator
streamed (path B) far beyond that horizon will see its weighting arithmetic
quietly lose exactness — long before `count` itself could overflow. It's a
soft, very large streaming horizon, not a hard cap enforced anywhere in
code — worth knowing before running an always-on generator across an
extremely long deployment. (At an exact `uint32` wrap, ~256× beyond that
horizon, `count` would pass through 0 and the `1/(n+1)`-family weights would
briefly degenerate rather than crash — academic, but stated for
completeness.)

Packed state (SYM/ASYM) is mechanically compatible with path B, but is
**grid-bounded**, not data-bounded: each streamed sample forces a
whole-tensor requantization of `basis`/`mean`/`eigvals` (no sub-tensor
funnel writes exist), so the running estimate's precision random-walks at
the quantization grid's scale rather than shrinking as more data arrives.
This is within the framework's memory-over-accuracy philosophy, not a bug,
but the FLOAT32-active/packed-at-rest pattern (*Knob guidance*, above) is
the recommended mitigation for accuracy-sensitive streaming use.

## Sequencing rule

The replay loader and the generators it draws from must be advanced in a
specific order relative to fine-tuning, or replay stops meaning what it's
supposed to mean:

1. **Pretrain** on domain 0.
2. Generators **absorb domain 0 immediately after pretraining** (one
   `ppcaReplayUpdate`/`ppcaReplayUpdateStreaming` call per class over
   domain 0's data) — domain 0 is the one exception to the "after
   fine-tuning" rule below, since there is no earlier domain for it to
   rehearse against.
3. For each subsequent domain `t = 1, 2, …`: **fine-tune** on domain `t`
   with the replay loader active — during this fine-tuning, replay draws
   synthetic samples from generators **as they stood after domain `t−1`**,
   i.e. strictly *before* domain `t`'s own data has been folded in. Only
   **after** domain `t`'s fine-tuning completes do the generators absorb
   domain `t`'s data, in preparation for domain `t+1`.

The invariant this protects: the model is never rehearsed on the very data
it is currently being fine-tuned toward — replay always looks *backward* at
already-completed domains. This is a calling-convention contract the
training loop must honor; nothing in the API enforces the ordering (no
generator "epoch" or "domain" counter exists to catch it if you call update
too early or too late) — get the sequence right in your own training-loop
composition.

## Checkpointing

```c
void ppcaReplaySetSerialize(const ppcaReplaySet_t *set, FILE *f);
void ppcaReplaySetDeserialize(ppcaReplaySet_t *skeleton, FILE *f);
```

(`PpcaReplaySerialize.h`) — the repo's first non-layer serialized state.
Wire format: `"ODTR"` magic, `u32 version(=2)`, `u32 numClasses`, then per
class: `u32 dim`, `u32 rank`, `u32 count`, `f32 sigma2`, `f32 totalVar`,
followed by the three tensors (`mean`, `basis`, `eigvals`) via the existing
tensor-tier `serializeTensor`/`deserializeTensor`. Since v2 (#370) every
scalar is fixed-width little-endian via the checked `SerialWire` primitives.
The embedded tensor records follow the CURRENT ODTS tensor-tier layout —
since group-quant PR1 that means the v4 SYM qconfig record (numGroups/
groupSize/scales[]); the ODTR container version does NOT track ODTS record
changes (an old checkpoint with SYM tensors fails only via downstream
guards — formal linkage tracked in #401).

**Deserialize fills a pre-built skeleton in place** — it does not allocate a
set for you. Build the skeleton with `ppcaReplaySetCreate(numClasses, cfg)`
using the *exact* `dim`/`rank`/storage config the checkpoint was written
with, then call `ppcaReplaySetDeserialize(skeleton, f)`. The skeleton's
`dim`, `rank`, dtype, and (for SYM/ASYM) `qBits` must match the checkpoint's
recorded values **per tensor** (`mean`, `basis`, `eigvals` independently) —
any mismatch is a fail-fast `PRINT_ERROR + exit(1)` (the "#316-class"
guard). A SYM tensor's `numGroups` is the one exception (group-quant PR2):
a file `numGroups` that differs from the skeleton's own is tolerated —
the shared ODTS tensor-tier deserialize reallocates the skeleton's
`scales[]` to the file's shape (same relax as any other ODTS SYM record);
the sentinel invariant (`numGroups==1 <=> groupSize==0`) and the
divisibility identity (`numGroups * groupSize == N`) are still enforced.
Mechanism: each tensor record's header (shape, dtype, qConfig) is read into
locals and validated against the skeleton *before* the public
`deserializeTensor` is allowed to touch it, then the stream is rewound and
the real read happens — this works around an open bug (#316) where
`deserializeTensor` itself overwrites the destination's shape/dtype from
file bytes before sizing its own payload read, so a wrapper can't check
"before overwrite" by calling it directly.

This requires a **seekable stream** — the peek-validate-rewind mechanism
does an `fseek` back to each record's start before consuming it, so a
plain `fopen`'d file works but a pipe or socket does not.

Checkpoints are architecture-portable since v2 (#370): all structural fields
are fixed-width little-endian, so `size_t` width no longer leaks into the
file. Bulk tensor DATA payloads are raw packed storage bytes — all supported
targets are little-endian (pinned at compile time in `SerialWire.h`).

## Limitations

- **Retention by design — no amnesic/forgetting factor anywhere.** Path A's
  merge weights every absorbed session equally forever; path B's CCIPCA
  update deliberately omits the amnesic discount Weng et al. offer. A
  generator never "forgets" an absorbed domain — that is the entire point
  of the feature (protect earlier domains from being overwritten). The
  flip side: there is no built-in mechanism to age out genuinely stale
  statistics if the true underlying distribution keeps drifting away from
  what earlier domains taught the generator.
- **Finite (if very large) streaming horizon.** Both update paths compute
  their merge/streaming weights via a `float` cast of the `uint32_t count`
  field; float32 loses exact integer representation beyond `2^24`
  (~16.7M). A generator streamed (path B) far past that point will see its
  weighting arithmetic silently lose precision, well before `count` itself
  could overflow — not a crash, just a slow accuracy drift with no fail-fast
  attached to it.
- **v1 is `ARITH_FLOAT32` compute only.** `mergeMath`/`streamMath`/
  `sampleMath` are declared per-op (`arithmetic_t`, the same by-value knob
  every layer uses), but only `ARITH_FLOAT32` is implemented —
  `ppcaValidateFloatArith` fail-fasts (at `ppcaReplayCreate` **and** at
  every update/sample call) if any is set to `ARITH_SYM_INT32`. An integer
  PPCA arithmetic path (which needs, among other things, an integer
  eigensolver) is literature-first future work, to be filed as a tracking
  issue rather than home-grown. State *storage* is independent of this and
  already accepts FLOAT32/SYM/ASYM.
- **float32 Gram-matrix conditioning.** Path A's merge forms a Gram matrix
  `G = Bᵀ·B` (eigendecomposed by `jacobiEigSymFloat32`, `JacobiEig.h`),
  which squares the condition number in float32 — a real numerical risk for
  ill-conditioned sessions. This risk is gated empirically by an
  incremental-vs-pooled-PCA regression test rather than by a runtime check;
  the design spec's contingency for that test failing is a one-sided Jacobi
  SVD directly on `Bᵀ` (avoiding the Gram matrix, and the condition-number
  squaring, entirely) — as of this writing that fallback is a **spec'd, not
  shipped** primitive: it isn't in `src/arithmetic/` and there is no config
  knob to select it. If you hit conditioning trouble in practice, that's the
  documented next step, not something you can flip on today.
- **Host-native checkpoint format.** Width/endianness are inherited from
  the model-serialization format — no cross-architecture normalization.

## Literature

- **Tipping & Bishop (1999), Probabilistic PCA** — the generative model and
  the sampling form `ppcaReplaySample` implements.
- **Ross, Lim, Lin, Yang (2008), Incremental Learning for Robust Visual
  Tracking** — the two-subspace merge with the mean-correction column that
  path A's Gram-matrix construction is built from.
- **Chan, Golub, LeVeque (1983)** — the two-set mean/variance merge
  underlying path A's `μ'`/`T'` update.
- **Weng, Zhang, Hwang (2003), CCIPCA** — path B's streaming update, used
  *without* the amnesic discount the paper offers (see *Limitations*:
  retention, not forgetting, is the design goal here).
- **Lopez-Paz & Ranzato (2017), GEM** — the ACC/BWT evaluation-metric
  definitions used by the sequential-domain demo.
- **FearNet (ICLR 2018), PASS (CVPR 2021)** — pseudo-rehearsal from
  per-class statistics; PASS's `k = 0` case (sampling reduces to
  `μ + σ·ε`) coincides with this feature's own `count == 1` /
  zero-subspace behavior.
- **Ravaglia et al. (IEEE JETCAS 2021)** — precedent for near-lossless
  8-bit replay-buffer storage, the empirical basis for recommending
  SYM/ASYM@8 as the packed-at-rest option above.
