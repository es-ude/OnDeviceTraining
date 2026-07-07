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

- **Two optimizers**: `SGD` (plain) and `SGD_M` (momentum). No Adam/RMSProp/Nesterov.
  Only `sgdMCreateOptim` exists as a public factory and it **always** builds `SGD_M`;
  a plain-`SGD` optimizer must be hand-assembled. `optimizerInit` is declared but
  unimplemented (dangling).
- **Arithmetic** — the update is **always computed in FLOAT32**. There is a SYM_INT32
  dispatch arm, but it dequantizes params + grads to a stack float VLA, updates in
  float, and requantizes in place. No integer/fixed-point update kernel exists.
- **Quant grads** — ✓. Grads of dtype FLOAT32 / SYM_INT32 / SYM / ASYM are consumed;
  FLOAT32 takes a raw-cast fast path, the rest dequantize to a float VLA. INT32 and
  BOOL are rejected at optimizer-create time.
- **Quant params** — FLOAT32 and SYM_INT32 only (in-place dequant→update→requant).
  SYM/ASYM param storage hits the step's `switch` default and exits. `qtype` is a
  single whole-optimizer dtype, **not** per-parameter — every param must share it.
- **Features**: learning rate, coupled L2 weight decay, classic heavy-ball momentum
  (no Nesterov/dampening), per-parameter momentum state (2 states for Linear/Conv/
  ConvT/LayerNorm, 0 for the rest), dtype-aware `scaleOptimizerGradients` (O(1) scale
  fold for quantized grads), and `sgdZeroGrad`. No LR schedule / bias-correction.

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
