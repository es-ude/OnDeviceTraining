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
  `computeMeanScale`), `evaluationBatch`, `evaluateBatchInternal` (behind
  `evaluationEpochWithMetrics`, `evaluationEpochWithReport` and `trainingRun`)
  and `inferenceBatched`. The `numClasses` peeks in `trainingRun` and
  `evaluationEpochWithMetrics` read the raw sample label's element count (the
  per-sample class count).
- Nothing auto-detects an existing batch axis: a sample that already carries a
  leading 1 is wrapped again (`[1, 1, ...]`), and a rank-checking first layer
  (Conv1d, the pools, Linear) fails fast.

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
