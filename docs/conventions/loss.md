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

**Uniform-B assumption** (DataLoader contract): all microbatches in one
macro batch have equal `B`. The MEAN aggregator divides by total samples
(`Σ batch->size`) rather than by `(numberOfBatches × B)`, so non-uniform B
would skew the mean. ODT's DataLoader currently always produces uniform
batches via `dropLast=true`; non-uniform B is out of contract.

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

Runtime assertion of the `dimensions[0] >= 1` contract is deferred to the
microbatch-B>1 umbrella (#152) — specifically #153. Today (B=1 only) the
assertion would be effectively a no-op; the protective value materialises
when B>1 becomes a real feature target.

