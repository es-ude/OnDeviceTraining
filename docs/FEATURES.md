# Feature Matrix

Snapshot of what the C framework can do today, per subsystem. This is a
**living reference** — update it when a layer/optimizer capability lands or a
gate is lifted. Every row was audited against source; issue numbers point at the
tracking work. Legend: ✓ = supported · ~ = supported via a fallback (see note) ·
✗ = not supported / gated · n/a = not applicable (e.g. layer has no parameters).

Central axis: **arithmetic (compute) and storage are independent knobs.** A layer
declares compute per op as a by-value `arithmetic_t {ARITH_FLOAT32 | ARITH_SYM_INT32,
roundingMode}` (`forwardMath`, GEMM family also `weightGradMath`/`biasGradMath`, dx op
`propLossMath`); storage is the produced-wire `quantization_t*` (`outputQ`/`propLossQ`)
and the grad-storage knobs. `SYM_INT32` is a **compute** format, never durable grad
storage (#261); `SYM`/`ASYM` are packed **storage** formats. Active training paths are
FLOAT32 and SYM_INT32; SYM/ASYM/BOOL are partial.

## Layers (`layerType_t`, 13 total)

| Layer | Trainable | FLOAT32 arith | SYM_INT32 arith | Quant params | Quant grads | (De)serialize |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| `LINEAR` | ✓ | ✓ | ✓ native (all 4 ops) | ✗ (#270 gate) | ✓ SYM/ASYM | ✓ |
| `CONV1D` | ✓ | ✓ | ✓ native (all 4 ops) | ✗ (#270 gate) | ✓ SYM/ASYM | ✓ |
| `CONV1D_TRANSPOSED` | ✓ | ✓ | ✓ native (all 4 ops) | ✗ (#270 gate) | ✓ SYM/ASYM | ✓ |
| `LAYERNORM` | ✓ | ✓ | ✓ native (fwd+bwd) | ~ SYM_INT32 only | ~ SYM path only | ✓ |
| `GROUPNORM` | ✓ | ✓ | ✓ native (fwd+bwd) | ~ SYM_INT32 only | ~ SYM path only | ✓ |
| `RELU` | – | ✓ | ✓ native (fwd+bwd) | n/a | n/a | ✓ |
| `SOFTMAX` | – | ✓ | ~ dequant-to-float | n/a | n/a | ✓ |
| `FLATTEN` | – | ✓ | ✓ scale-transparent | n/a | n/a | ✓ |
| `DROPOUT` | – | ✓ | ✓ scale-transparent | n/a | n/a | ✓ |
| `MAXPOOL1D` | – | ✓ | ✗ | n/a | n/a | ✓ |
| `AVGPOOL1D` | – | ✓ | ✗ | n/a | n/a | ✓ |
| `ADAPTIVE_AVGPOOL1D` | – | ✓ | ✗ | n/a | n/a | ✓ |
| `QUANTIZATION` | – | converter node | converter node | n/a | n/a | ✓ |

Notes on the qualified cells:

- **`SYM_INT32 arith`** — *native* means an integer kernel selected by the op's
  `arithmetic_t.type` and routed through the `executeOp` funnel (raw int32 mantissas,
  width-restored at the producer). *scale-transparent* (Flatten/Dropout) copies int
  values verbatim and folds any factor into the wire scale, bypassing the funnel.
  *dequant-to-float* (Softmax backward) converts operands to FLOAT32, computes, and
  requantizes — no integer arithmetic. Pools (`MAXPOOL1D`/`AVGPOOL1D`/
  `ADAPTIVE_AVGPOOL1D`) have **only** a FLOAT32 kernel body; their forward still routes
  through the funnel, so a mismatched SYM_INT32 input is dequantized to float scratch
  first, but no SYM kernel exists (backward exits on non-FLOAT32).
- **`QUANTIZATION`** is a pure storage-to-storage conversion node (`executeConvert`,
  conversionMatrix), not an arithmetic layer — it deliberately changes dtype/scale.
- **Quant params** — trainable weight/bias storage. Linear/Conv are hard-gated to
  FLOAT32 by the `requireFloat32` init gate (#270); `getQLike` can clone SYM_INT32/
  SYM/ASYM but `initWeightTensor`/`initBiasTensor` fail fast. **LayerNorm and GroupNorm
  are the exceptions**: gamma/beta may be stored SYM_INT32 (no requireFloat32 gate), but
  SYM/ASYM are rejected by factory validation — hence "partial". No layer supports SYM/
  ASYM native param storage.
- **Quant grads** — the `weightGradStorage`/`biasGradStorage` knobs in `layerQuant_t`
  (plus `weightGradAccMode`/`biasGradAccMode`). Default is FLOAT32 everywhere; SYM and
  ASYM packed grad storage work for the four trainable layers via `gradInit`→`getQLike`
  (BOOL rejected, #269). Accumulate epilogue: SYM target honors both `OUT_ACC_FIXED_SCALE`
  and `OUT_ACC_DYNAMIC_RESCALE`; ASYM honors `DYNAMIC_RESCALE` only. **LayerNorm/GroupNorm
  caveat**: packed grads are only writable on the SYM_INT32 backward path — their FLOAT32
  backward raw-casts grads (and the dx wire) and rejects packed storage.

## Optimizer (`optimizerType_t`)

- **Two optimizer types**: `SGD_M` and `ADAM_W` (#328). The plain-`SGD` enum value
  and its unreachable code arm were deleted earlier (#308). `sgdMCreateOptim` is the
  only public SGD factory; passing `momentumFactor == 0.0f` is how a caller gets plain
  SGD, not a separate type: the factory allocates **no** momentum-state buffers in
  that case (`optim->states == NULL`) and `sgdStepM` runs a single stateless update
  op per parameter instead of the two-op momentum path (mathematically identical at
  momentum 0).
- **AdamW** (`adamWCreateOptim(lr, beta1, beta2, eps, weightDecay, model, sizeModel,
  momentQuant, updateMath)`, torch.optim.AdamW-faithful (documented-order replication;
  torch cross-checked to ≤1 ulp)) — weight decay is
  **decoupled** (`param *= 1 - lr*wd`), unlike SGD's coupled L2 (`grad += wd*param`).
  `beta1`/`beta2`/`eps`/`weightDecay` are stored **double**; every per-step scalar
  (`1-beta1`, `1-beta2`, the bias corrections, `-(lr/bc1)`, `1-lr*wd`) is composed in
  double and cast to float exactly once at the kernel-call boundary, mirroring
  PyTorch's own python-double composition — float-stored betas lose `1-beta` bits
  unrecoverably. The step is an op-for-op replication of torch's single-tensor
  AdamW sequence over the `PointwiseFused` primitives via three `executeOp` kernels
  (m: lerp / v: mul+addcmul / param: mul+addcdivDenom) — 5 elementwise passes per
  parameter per step, no denom tensor materialized. Each trainable parameter gets
  **2 moment buffers** (`m = stateBuffers[0]`, `v = stateBuffers[1]`), each a
  `getQLike` deep clone of the single caller-owned `momentQuant` template (FLOAT32
  default; quantized moment storage is a memory knob routed through the executeOp
  funnel like any other dtype, with no bit-parity claim). `updateMath` is
  FLOAT32-only, fail-fast at both create time and every `adamWStep` call (the #310
  pattern). `stepCount` (bias-correction exponent `t`) is **not** checkpointed —
  `StateDictApi` serializes only weights/biases, so a resumed run restarts bias
  correction at `t=1` (#350). The `m`-update's `lerp` bit-parity holds only for
  `beta1 > 0.5` (documented, not branched — validation still admits `[0, 1)`).
  Scheduler-compatible via the same `getLr`/`setLr` vtable row as SGD. Exercised by
  the `train_c_har_classifier_adamw` example.
- **Arithmetic** — `sgdMCreateOptim`'s `updateMath` parameter (by-value `arithmetic_t`,
  #310) selects what the update kernels compute in. Only `ARITH_FLOAT32` is
  implemented; any other type fails fast at optimizer-create time and again at
  `sgdStepM` (no integer/fixed-point update kernel exists yet). Rounding ownership:
  `updateMath.roundingMode` governs only the `executeOp` funnel's prologue (operand
  conversion into the compute format; inert for FLOAT32) — the OUT_WRITE epilogue
  rounds by each **target** tensor's own qConfig, e.g. a packed-SYM param's write-back
  rounding comes from its own `SYM_ROUNDING`-controlled config, independent of
  `updateMath`.
- **Quant grads** — ✓. The update kernels route through `executeOp` like every other
  op in the framework: the funnel dispatches per operand's ACTUAL dtype, so FLOAT32 /
  SYM_INT32 / SYM / ASYM grads are all consumed with no optimizer-level dtype switch.
  INT32 and BOOL grad storage are rejected at optimizer-create time.
- **Quant params** — dtype dispatch is a funnel property, not a whole-optimizer
  `qtype`: the OUT_WRITE epilogue requants into whatever dtype each parameter tensor
  actually carries, so FLOAT32/SYM_INT32/SYM/ASYM params all round-trip through the
  same code path. FLOAT32 and packed SYM are exercised today (har_classifier
  trainers); SYM_INT32 and ASYM are implied by the generic executeOp/conversionMatrix
  dispatch but not yet exercised by any example. Support is bounded only by what
  `conversionMatrix` covers for that dtype (BOOL has no conversion cell in either
  direction).
- **Features**: learning rate, coupled L2 weight decay (SGD) or decoupled weight
  decay (AdamW), classic heavy-ball momentum (SGD, no Nesterov/dampening),
  per-parameter momentum state (2 states for Linear/Conv/ConvT/LayerNorm, 0 for the
  rest, and 0 for every parameter when `momentumFactor == 0`), dtype-aware
  `scaleOptimizerGradients` (O(1) scale fold for quantized grads), and
  `optimizerZeroGrad` — shared across both rows. No bias-correction on SGD; AdamW
  has PyTorch-standard bias correction (`bc1`/`bc2`).
- **LR schedulers** (#327): `lrScheduler_t` (caller-owned, zero-alloc) with
  `STEP_LR`, `EXPONENTIAL_LR`, `COSINE_ANNEALING_LR` — PyTorch closed-form
  parity (double math, float cast at `setLr`), stepped by `trainingRun`
  (NULL-able param, once per epoch after the callback) or manually via
  `lrSchedulerStep` at any boundary. Optimizer-agnostic through the
  `getLr`/`setLr` vtable entries — both `SGD_M` and `ADAM_W` rows implement them,
  so a scheduler works unchanged across either optimizer.

## Serialization (`src/serial/`)

- **Layers** — all 12 `layerType_t` have matching `serialize` + `deserialize` arms,
  each round-trip tested under `test/unit/serial/`.
- **Dtypes** — all 6 `qtype_t` qconfigs serialize/deserialize symmetrically (INT32/
  FLOAT32/BOOL = type byte only; SYM_INT32/SYM/ASYM carry scale + rounding + bits [+
  zeroPoint]). Packed tensor data (SYM/ASYM sub-byte, BOOL 1-bit) is byte-tight via
  `calcNumberOfBytesForData` and round-trips exactly.
- **Model format** — `"ODTS"` magic + `version` (=1) + `layerCount` + per-layer type
  tag. Deserialize fail-fasts on magic / version / count / tag mismatch.
- **Contract** — deserialize **fills a pre-constructed model in place** (no allocation
  in the serial path); the caller must build a matching model first. It does not
  validate that a tensor's file dtype matches the destination's allocated qConfig.
- **Gaps** — `serializeSparsity` is a TODO stub (shape is fully serialized); wire format
  is host-native (`size_t` width + endianness, no normalization → cross-arch gap for a
  64-bit host writing an MCU-loaded model); `fwrite`/`fread` return values are ignored
  (no truncation detection). No save-to-disk of a full model exists above this layer —
  `StateDictApi` is weight-**load** only.

## Continual learning (`src/continual_learning/`, `src/userApi/continual_learning/`)

Memory-bounded generative replay against catastrophic forgetting under
sequential domain shift (#326). Full guide: **`docs/CONTINUAL_LEARNING.md`**
(when to use, memory formula, knob guidance, update paths, sequencing,
checkpointing, limitations, literature).

- **Per-class PPCA generator** — `ppcaReplay_t` holds `(mean[d], basis[k,d]`
  orthonormal, `eigvals[k]` descending, `sigma2, totalVar, count)`;
  `ppcaReplaySample` draws synthetic rehearsal samples reproducing the PPCA
  marginal (Tipping & Bishop 1999). Persistent footprint is `O((k+1)·d)` per
  class and **constant** in the number of domains absorbed so far
  (`ppcaReplayBytes`, `ppcaReplayIsoExemplarCount` for the iso-byte
  exemplar-buffer comparison).
- **Two update paths** — `ppcaReplayUpdate` (session merge, recommended
  default; Chan et al. 1983 two-set merge + the Ross et al. 2008
  mean-correction column) and `ppcaReplayUpdateStreaming` (CCIPCA, Weng et
  al. 2003, per-sample `O(k·d)`, deliberately **no** amnesic discount —
  retention by design, not tracked as a gap).
- **Training-loop integration is a `dataLoader_t` wrapper** —
  `replayDataLoaderWrap`/`freeReplayDataLoader` (`src/userApi/
  continual_learning/`), zero framework changes: appends `r` synthetic
  samples per eligible class after the real batch, so the optimizer's
  existing `batch->size` MEAN divisor scales them in automatically. Must be
  freed with `freeReplayDataLoader`, **never** `freeDataLoader` (the
  wrapper borrows the base loader's fields).
- **Storage vs. arithmetic** — same split as layers: `meanQ`/`basisQ`/
  `eigvalsQ` (FLOAT32/SYM/ASYM) are independent per-tensor storage configs;
  compute (`mergeMath`/`streamMath`/`sampleMath`) is v1 `ARITH_FLOAT32`
  only (`ppcaValidateFloatArith` fail-fasts on `ARITH_SYM_INT32`).
  FLOAT32-active/packed-at-rest (SYM/ASYM@8 via `executeConvert`) is the
  recommended pattern.
- **Serialization** — `ppcaReplaySetSerialize`/`ppcaReplaySetDeserialize`
  (`PpcaReplaySerialize.h`), the repo's first non-layer checkpoint format
  (`"ODTR"` magic), with a peek-validate-rewind guard against the
  #316-class deserialize-overwrite hazard; requires a seekable stream.

## Other subsystems

- **Loss functions** — `MSE` (FLOAT32 + SYM_INT32, fwd+bwd) and `CROSS_ENTROPY`
  (fwd FLOAT32-only; bwd FLOAT32 + ASYM). CE backward is the **fused softmax+CE
  gradient** (`softmaxOutput − target`), and the training loop skips the Softmax layer
  in backprop. Backward emits **raw per-element grads**; the mean divisor is deferred to
  the optimizer via `computeMeanScale` × `scaleOptimizerGradients`.
- **Training loop** — `trainingRun` → epoch → batch → pluggable `calculateGradsFn`.
  A "batch" is gradient accumulation over B=1 microbatches. Metrics: loss, accuracy,
  macro precision/recall/F1, and a caller-owned confusion matrix
  (`epochStats_t` / `classificationReport_t`). Three eval entry points.
- **Data loading** — `.npy` (`<f4`/`<i4`) and CSV; `DataLoader` with batch + shuffle
  (once at init, **no per-epoch reshuffle**) + `dropLast` (only `true` supported).
  No dedicated MNIST loader in `src/` (examples preprocess to `.npy`).
- **RNG** — global XorShift32, seedable and reproducible, **byte-mirrored in Python**
  and CI-verified. Drives weight init, the Dropout Bernoulli mask (swappable fill hook),
  and DataLoader shuffle.
- **Observer / trace** — `traceSink_t` callback facility (not a layer) that hands
  fwd/activation-grad/loss-grad/param tensors to a caller-supplied sink; used for
  layer-by-layer C-vs-PyTorch parity debugging (`npyDumpSink`, kws_raw harness).
- **Fused pointwise primitives** (`src/arithmetic/PointwiseFused.c`, #328) — three
  single-pass fused elementwise ops that are the compute core of AdamW's moment/param
  updates (`adamWStep`, see Optimizer section above — the only consumer today), each
  with a torch-parity **rounding contract**: `lerpFloat32TensorsInplace`
  (`Tensor.lerp_`, small-weight branch) uses an explicit `fmaf()`, bit-identical to
  ATen's FMA-fused CPU kernel for `|weight| < 0.5`; `addcmulFloat32TensorsInplace`
  (`Tensor.addcmul_`) and `addcdivDenomFloat32TensorsInplace` (`Tensor.addcdiv_` parity
  without materializing torch's denom tensor) keep their multiple roundings **separate
  and left-associated** — no fusion. The separate-rounding contract is enforced by
  `-ffp-contract=off` on the `PointwiseFused` compile target, not by the in-source
  `#pragma STDC FP_CONTRACT OFF` alone: GCC ignores that pragma, and its default
  `-ffp-contract=fast` silently fuses the non-lerp ops into `vfma` on ARM (invisible on
  x86-64 CI, which needs `-mfma` to contract). A shared **identity-order gate** fail-fasts
  unless every operand is FLOAT32 with matching dimensions and identity
  `orderOfDimensions`, which deliberately sidesteps the permuted-Inplace aliasing hazard
  (#339) and makes operand aliasing (e.g. `addcmul(a, b, b, s)`) safe by construction.
  `addcdivDenomFloat32TensorsInplace`'s rounding contract is bit-identical to torch's
  macOS-ARM kernel; torch's Linux-AVX2 vectorization fuses the final multiply-add and may
  differ by up to 1 ulp in zero-accumulator regimes (#353), so its gold fixtures pin the
  documented rounding order itself rather than a torch capture.
  A latent `ODT_TRACK_INSTRUCTIONS` CMake cache var (ON in the `unit_test_debug`/
  `unit_test_asan` presets, and, by inheritance, `unit_test_ubsan`; OFF by default
  elsewhere) compiles in per-element instruction
  counters (`getLerpInstructionCounter` et al.); no caller/estimator consumes them yet —
  this is the first library wired for the mechanism (#351 tracks enabling it on the
  legacy `Square`/`Matmul` libs, which predate it and have known blockers).
- **Quantization machinery** — the 4-phase `executeOp` funnel + `executeConvert`, a
  6×6 `conversionMatrix` (every FLOAT32/INT32/SYM_INT32/SYM/ASYM pair; BOOL unsupported
  in every direction), grad-accumulate modes, and `quantizeFloatToAsym` as the single
  `*→ASYM` helper.
- **Weight init** — PyTorch-compatible: `INIT_DEFAULT` reproduces
  `kaiming_uniform_(a=√5)`, plus kaiming/xavier schemes (`weightInit_t`). FLOAT32-only.
- **userApi** — `inference` (single/batched/with-loss), `StateDictApi` (weight **load**
  only), `LayerWeightsApi`, `ModelValidationApi` (near-stub after PR1b.2 retired the
  SYM-producer rule), `LayerQuant` uniform config, and `StorageApi` as the single
  allocation seam (alloc primitives only in `src/userApi/`).
- **Examples** — 7 end-to-end: `har_classifier`, `ecg_anomaly_ae`, `mnist_mlp`,
  `mnist_cnn`, `kws_mfcc`, `kws_raw` (+ trace harness), `mixed_width_mlp`.

## Known gaps / partial features

- Native SYM_INT32 params blocked for Linear/Conv by the `requireFloat32` gate (#270).
- No SYM/ASYM native param storage anywhere; no per-parameter optimizer dtype.
- Optimizer has no integer update kernel (SYM_INT32 = dequant→float→requant).
- CrossEntropy forward is FLOAT32-only; `classWeights` field is allocated but unused.
- Serialization: sparsity stub, non-portable wire format, no I/O error handling.
- `sparsityType_t` is scaffolding only (propagated but no kernel exploits it).
- Pooling layers are FLOAT32-only.
- Continual-learning (PPCA replay, #326) arithmetic is `ARITH_FLOAT32` only;
  no integer eigensolver exists yet. State storage is unaffected (FLOAT32/
  SYM/ASYM all accepted).
- `TRACK_INSTRUCTIONS` counters exist on the legacy `Square`/`Matmul` libs but are
  unsafe to enable: a name mismatch fails compilation for `Square`, both libs lack a
  reset helper, and `Matmul`'s SYM_INT32 path double-increments (#351).
