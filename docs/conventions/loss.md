# Loss & training-loop microbatch contracts

## Loss API: microbatch contracts

Each loss function in `src/loss_functions/` exposes:

- `forward(modelOutput, label, reduction) → float`
- `backward(modelOutput, label, result) → void`
- `computeMeanScale(totalSamples, modelOutput) → float`

### Reduction split

`lossConfig_t.backwardReduction` is the user's training-strategy choice — it
drives whether `scaleOptimizerGradients` runs between `trainingBatchDefault`
and `optimizerStep`. It is a config field.

`forwardReduction` is a per-call parameter on every aggregator
(`trainingBatchDefault`, `evaluationBatch`, `evaluationEpoch`, `inferenceWithLoss`,
`calculateGradsFn_t`). It controls how the per-microbatch loss value is
reported. `trainingRun` is the only function that hardcodes it
(to `REDUCTION_MEAN`) so train and eval losses are comparable; lower-level
callers pick freely.

### Microbatch shape

`modelOutput->shape->dimensions[0]` is the microbatch dimension `B`, and it
is always explicit: the output shape is `[B, ...]` and
`numFeaturesPerSample = numElements / B`. At `B=1` the training loop makes
the leading 1 explicit with `batchViewOf`, for the model input, the label
and the `labelRef` handed to `computeMeanScale` alike (see
[data-shape.md](data-shape.md), "Who adds the batch axis").

This changed the MSE MEAN gradient scale for labels that do not already
start with a leading 1 (#152 PR3a). Before this PR, `trainingEpochDefault`
passed `computeMeanScale` the raw label tensor: a rank-1 label `[F]` had
`dims[0] == F`, so `computeMeanScaleMSE` read `numFeaturesPerSample = F/F
== 1` and the MEAN gradient scale was `1/b`. Now `trainingEpochDefault`
passes the `batchViewOf` view of sample 0's label (`[1, F]`), so `dims[0]
== 1`, `numFeaturesPerSample == F`, and the scale is `1/(b·F)` — the
PyTorch value, and consistent with the MSE forward MEAN (which always
divides by every element, not just the batch). A `[C, L]` label moves the
same way, from `1/(b·L)` to `1/(b·C·L)`. In-tree users are unaffected
(ECG's label is `[1, 140]`; migrated fixtures are value-preserving), but an
external caller whose MSE labels do not start with a leading 1 now gets an
effective optimizer step F times smaller than before.

Softmax partitions its input by the same `B` rows (#152): each of the `B` rows
normalizes over its own `numElements / B` elements (a rank-1 input is one
row), so CE's fused `(p - y)` backward is per row as well.

**Uniform-B contract**: all microbatches in one macro batch have equal `B`.
At `trainingRunOptions_t.microBatchSize` 1 (the default) `trainingBatchDefault`
hands `calculateGradsFn` one `[1, ...]` batch view per sample; at m > 1 (#152,
FLOAT32 only) it gathers each chunk of m samples into one `[m, ...]` call, and
the loop enforces `b % m == 0` so there are no ragged tails:
`trainingRun` checks the train loader's batch and, before epoch 0, every
batch a batch-size scheduler will set (`bsSchedulerBatchSizeAt`);
`trainingEpochDefault` checks the loader batch; `trainingBatchDefault` checks
`batch->size` (the backstop for a replay loader, whose batch is
`base + eligible·r`). Each fails fast naming `b` and `m`. Known limitation:
Dropout fails fast at m > 1, because its caller-allocated mask holds one
sample's elements (follow-up issue).

The reported MEAN loss weights every chunk by its rows,
`Σ (chunkLoss × m) / batch->size`, which equals the per-sample mean for both
losses because each loss's MEAN already divides by its rows (CE by
`dimensions[0]`, MSE by all `m·F` elements). `batch->size` always counts
samples, so the gradient macro-scale below is unchanged by m.

### Backward macro-scaling

Backward writes raw per-element gradients (`2(o-l)` for MSE, `(p-y)` for CE).
The macro-batch divisor lives at the optimizer:

- `lossFunctions[lossConfig.funcType].computeMeanScale(N, modelOutput)`
  returns the PyTorch-parity divisor (`1/(N*F)` for MSE, `1/N` for CE).
- `scaleOptimizerGradients(optimizer, factor)` multiplies every parameter's
  `grad` field by the factor in place.
- `trainingEpochDefault` calls these between accumulation and `step`,
  but only when `backwardReduction == REDUCTION_MEAN`.

For SUM (or future per-sample weighted variants — see #150), the backward
gradient flows through unscaled.

### Shape assertion (deferred)

Runtime assertion of the `dimensions[0] >= 1` contract is deferred to #153.
B > 1 is reachable now, through `microBatchSize` (#152), so the assertion
would no longer be a no-op; until #153 lands, the loop builds each stacked
item and label with the same `dimensions[0] = m`.

