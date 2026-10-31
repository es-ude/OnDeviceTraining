# Data shape convention

## Data Shape Convention

Datasets deliver samples in their natural geometric shape (e.g. `[C, H, W]`
for images, `[C, L]` for time series). Any `reshape`, `flatten`, or `view`
operation is the **first layer of the model**, not a preprocessing step in
the dataset. This:

- keeps dataset code independent of downstream model topology
- allows one dataset to feed models with different input ranks
- matches the PyTorch / Keras / elastic-ai.creator IR convention, so a future
  ir2c can compile each shape transform to a corresponding C layer

For flatten-to-2D, use `flattenLayerInit()` from `FlattenApi.h`.

## Who adds the batch axis (#152 PR3a)

Datasets never carry a batch axis; the loop owns it.

- **Tensor-level entry points take batched tensors `[B, ...]`:** `inference`,
  `inferenceWithLoss`, `calculateGradsSequential`, `tracedGrads`. A caller that
  hands them ONE dataset sample wraps it first with `batchViewOf`
  (`src/userApi/tensor/include/BatchView.h`): a caller-owned, stack-allocated
  `[1, ...sample]` view that shares the sample's data, quantization and
  sparsity (no copy, no heap, nothing to free).
- **`batch_t` consumers take natural-shape samples and add axis 0 themselves:**
  `trainingBatchDefault`, `trainingEpochDefault` (its `labelRef` for
  `computeMeanScale` — this shifts the MSE MEAN gradient scale for labels
  that don't already start with a leading 1, see
  [loss.md](loss.md#microbatch-shape)), `evaluationBatch` (and its public
  entry point `evaluationEpoch`), `evaluateBatchInternal` (behind
  `evaluationEpochWithMetrics`, `evaluationEpochWithReport` and `trainingRun`)
  and `inferenceBatched`.
  At `microBatchSize` m > 1, `trainingBatchDefault` instead gathers each
  chunk of m samples into an `[m, ...]` copy (#152 PR3b, FLOAT32 only; see
  `trainingRunOptions_t.microBatchSize`). The evaluation consumers hand
  `[1, ...]` views at m = 1 and, at m > 1, `[rows, ...]` views over gather
  buffers filled across loader batches (#468). `numClasses` still comes from the
  raw sample label's element count (the per-sample class count) in `trainingRun`
  and `evaluationEpochWithMetrics`, and `evaluationEpochWithReport` checks it.
- Nothing auto-detects an existing batch axis: a sample that already carries a
  leading 1 is wrapped again (`[1, 1, ...]`). A missed or doubled wrap travels
  through rank-preserving layers (Relu, LayerNorm, Dropout) until the first
  layer that checks rank (Linear, Conv1d, Conv1dTransposed, the pools,
  GroupNorm, BatchNorm1d), which fails fast, or a Flatten, which absorbs it
  silently; if it reaches the output unabsorbed, the loss's shape check
  (#153) fails fast unless the label carries the same mistake — see
  "Migrating from `[1, ...]` samples" below.
- BatchNorm1d needs `n = m·T ≥ 2` values per channel in training: a
  `[m, C]` model needs `microBatchSize ≥ 2`; `[1, C, T ≥ 2]` trains on
  per-sample statistics (instance-norm-like) while eval uses running
  statistics — allowed for PyTorch parity (#460).
- A label's per-sample shape must equal the model output's per-sample shape
  (#153). Store a scalar regression target for a `[B, 1]` head as `(N, 1)`,
  so each sample is `[1]` and its view `[1, 1]`; an `(N,)` array loads as
  rank-0 samples whose `[1]` view the loss rejects.

```c
batchView_t itemView;
tensor_t *out = inference(model, modelSize, batchViewOf(&itemView, sample->item));
```

### Documented exception: mnist_cnn

The framework has no reshape layer, so `examples/mnist_cnn/train_c.c`
(`reshapeItemsToConv1d`) reshapes each `[1, 28, 28]` image into the
`[1, 784]` (channel, length) sample its first Conv1d consumes. That
dataset-side reshape is the one exception to "reshapes are the first model
layer"; it adds no batch axis — the loop does.

### Migrating from `[1, ...]` samples

A dataset or fixture still built on the pre-#152 convention (samples or
labels that already carry an explicit leading 1) needs auditing along three
axes, since nothing here auto-detects the old shape:

- **Items.** A first layer that checks input rank (Linear, Conv1d,
  Conv1dTransposed, the pools, GroupNorm, BatchNorm1d) fails loudly
  (`exit(1)`) on the now-doubled leading axis — the fastest signal that a
  dataset still needs updating. A Flatten-first model (e.g.
  `examples/mnist_cnn`) collapses the extra leading 1 and gives no error;
  a Relu-, LayerNorm- or Dropout-first model carries the extra axis onward,
  where it keeps traveling through any further rank-preserving layers until
  a later rank-checking layer fails fast, a later Flatten absorbs it
  silently, or — if neither follows — the loss's shape check (#153) fails
  fast at the output, unless the labels carry the same extra 1 — check the
  dataset directly instead of relying on a test failure.
- **Labels.** The loss dispatchers (`src/loss_functions/MSE.c`,
  `CrossEntropy.c`) require the label to have exactly the model output's rank
  and dimensions (#153), so an old-style label whose leading 1 was a batch
  axis fails fast once the loop wraps it to `[1, 1, ...]`: drop that leading
  1. A leading 1 that is a real axis (ECG's `[1, 140]` against a
  `[1, 1, 140]` output) stays.
- **User callbacks.** `calculateGradsFn_t` and `inferenceWithLossFn_t`
  implementations passed to `trainingBatchDefault` / `evaluationBatch` now
  receive stack `[1, ...]` views borrowed for the duration of the call,
  not the long-lived dataset tensors they used to see — see
  `TrainingLoopApi.h`.

## Weight layouts, orientation and padding

Layer shapes follow the PyTorch convention; the points a re-implementation
(or a reference model) has to get exactly right:

- **Linear** (`src/layer/Linear.c`): the weight is `[outFeatures, inFeatures]`,
  the bias `[outFeatures]`. Forward is `y = x Wᵀ + b` for `x` `[B, inFeatures]`
  (`linearForwardFloat` transposes the weight view, zero-copy, then runs
  `x @ Wᵀ`). Backward: `dx = dL W` (`dL` `[B, outFeatures]`, `W` as stored),
  `weightGrad = dLᵀ x` (`[outFeatures, inFeatures]`), `biasGrad` = per-feature
  sum of `dL` over the batch (`linearCalcBiasGradsFloat32`). Every matmul entry
  computes `a @ b` of its two arguments.
- **Conv1d** (`conv1dKernelFloat32`, `conv1dKernelSymInt32`): the weight is
  `[Cout, Cin/groups, K]`, flat index `(oc · Cin/groups + ic) · K + k`; input
  and output are `[B, C, L]`. It is a cross-correlation: kernel tap `k` reads
  input position `outPos · stride − padLeft + k · dilation`, the kernel is not
  flipped. Conv1dTransposed's layout is `[Cin, Cout/groups, K]` (see
  [arithmetic-sym.md](arithmetic-sym.md)).
- **Padding** (`windowGeometry1dCalc`, `src/arithmetic/SlidingWindow1d.c`),
  with `effK = dilation · (K − 1) + 1`: `VALID` pads nothing and has
  `outputLength = (L − effK) / stride + 1`, or **0 when `L < effK`**; `EXPLICIT`
  pads `padding` zeros on each side with the same formula on the padded
  length (also 0 when it is shorter than `effK`); `SAME` has
  `outputLength = ceil(L / stride)` and
  `totalPad = max(0, effK + (outputLength − 1) · stride − L)` split as
  `padLeft = totalPad / 2` (floor) and `padRight = totalPad − padLeft`, i.e.
  the extra zero of an odd total goes to the right (TF-minimal; PyTorch
  `padding=N` is `EXPLICIT`).
