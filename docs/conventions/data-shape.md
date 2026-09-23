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
  and `inferenceBatched`. The `numClasses` peeks in `trainingRun` and
  `evaluationEpochWithMetrics` read the raw sample label's element count (the
  per-sample class count).
- Nothing auto-detects an existing batch axis: a sample that already carries a
  leading 1 is wrapped again (`[1, 1, ...]`). A first layer that checks input
  rank (Linear, Conv1d, Conv1dTransposed, the pools, GroupNorm) fails fast on
  a missed or doubled wrap; a Flatten-, Relu-, LayerNorm- or Dropout-fronted
  model does not check rank and can absorb the mistake silently — see
  "Migrating from `[1, ...]` samples" below.

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
  Conv1dTransposed, the pools, GroupNorm) fails loudly (`exit(1)`) on the
  now-doubled leading axis — the fastest signal that a dataset still needs
  updating. A Flatten-first model (e.g. `examples/mnist_cnn`) or a model
  whose first layers are Relu, LayerNorm or Dropout does not check rank and
  produces the same output either way (the extra leading 1 multiplies out
  to the same element count), so these models give no error — check the
  dataset directly instead of relying on a test failure.
- **Labels.** `requireOperandMatchesOutput` (`src/loss_functions/MSE.c`,
  `CrossEntropy.c`) compares element count only, not rank, so an old-style
  `[1, ...]` label is accepted silently too. Under `REDUCTION_MEAN` this is
  exactly the MSE mean-scale change described in
  [loss.md](loss.md#microbatch-shape).
- **User callbacks.** `calculateGradsFn_t` and `inferenceWithLossFn_t`
  implementations passed to `trainingBatchDefault` / `evaluationBatch` now
  receive stack `[1, ...]` views borrowed for the duration of the call,
  not the long-lived dataset tensors they used to see — see
  `TrainingLoopApi.h`.
