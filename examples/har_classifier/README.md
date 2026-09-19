# HAR Classifier — PyTorch + C Parity Demo

Trains a 6-class human-activity classifier on the UCI HAR dataset using the
1D-CNN layers exposed by both PyTorch (reference) and the ODT C framework. The
C model is built with the factory layer API (`conv1dLayerInit` + `layerQuant_t`)
and loads PyTorch weights through `StateDictApi`.

One binary, two verification modes:

- **Bit-parity** (what CI runs): `BIT_PARITY=1` loads PyTorch's trained weights
  into the C model and runs inference only — the C predictions must be
  **bit-identical** to PyTorch's. Deterministic and exact.
- **Train-from-scratch demo**: with no env var the C model trains from its own
  random init; `compare.py` checks final-state parity within tolerance and emits
  plots. Independent init, so it verifies *convergence*, not bits.

## Run it

```bash
# 1. Prepare data (downloads ~58 MB the first time; cached under data/raw/)
uv run python examples/har_classifier/prepare_data.py

# 2. Train the PyTorch reference + export weights (~30s on CPU)
uv run python examples/har_classifier/train_pytorch.py

# 3. Build the C trainer
cmake --preset examples
cmake --build --preset examples --target train_c_har_classifier

# 4a. Bit-parity check (exact — this is the CI gate)
BIT_PARITY=1 ./build/examples/examples/har_classifier/train_c_har_classifier
uv run python examples/_shared/compare_predictions.py \
  --pytorch examples/har_classifier/outputs/pytorch_predictions.npy \
  --c examples/har_classifier/outputs/c_predictions.npy --dtype int32

# 4b. …or the train-from-scratch demo + plots (several minutes)
./build/examples/examples/har_classifier/train_c_har_classifier
uv run python examples/har_classifier/compare.py
```

## Outputs

After the train-from-scratch demo, `examples/har_classifier/` contains:
- `data/{train,val,test}_{x,y}.npy`
- `logs/{pytorch,c}.json`
- `outputs/{pytorch,c}_predictions.npy`
- `plots/{loss_curves,accuracy_curves,confusion_matrix_pt,confusion_matrix_c}.png`

## Model

- Input: `[9, 128]` (9 IMU channels, 2.56 s window @ 50 Hz)
- 2 × `Conv1d → ReLU → MaxPool1d` blocks, then `Conv1d → ReLU`
- Global `AvgPool1d` → `Flatten → Linear → Softmax → CrossEntropy`
- ~10 K parameters

## Parity tolerance (train-from-scratch demo)

| Metric | Tolerance |
|---|---|
| test_acc  | ±2.5 pp absolute |
| test_loss | ±0.15 nats absolute |

The demo's two implementations use independent random init; the loss tolerance
is empirically calibrated. Bit-parity mode requires exact equality instead.
See `examples/_shared/DETERMINISM.md` for the full determinism contract.

## Float32 trainer knobs: LR / batch-size schedules and per-epoch reshuffle

`train_c_har_classifier` is fully env-configured; a plain run reproduces the
develop defaults **except** that the train loader now reshuffles every epoch
(`RESHUFFLE=1`, mirrored bit-exactly by `train_pytorch.py` — see
`examples/_shared/DETERMINISM.md`). `RESHUFFLE=0` on both sides restores the
old shuffle-once baseline.

| env | default | effect |
|---|---|---|
| `EPOCHS`, `LR`, `MOMENTUM`, `SEED`, `SHUFFLE_SEED`, `LOG_PATH` | 20, 0.01, 0.9, 42, 42, `logs/c.json` | as before |
| `BATCH_SIZE` | 64 | the train loader's initial batch (`config.batch` in the log) |
| `LR_SCHEDULE` | `none` | `none` \| `step` \| `exp` — torch StepLR / ExponentialLR (`LrScheduler`, #327): LR × `GAMMA` |
| `BS_SCHEDULE` | `none` | `none` \| `step` \| `exp` — the batch analogue (`BsScheduler`): batch ÷ `GAMMA`, `clamp(round(·), 1, MAX_BATCH_SIZE)` |
| `BS_LR_COMPENSATION` | 0 | 1 = the batch scheduler also writes `lr = LR · applied / exact` (the "BC" arm); combining it with `LR_SCHEDULE != none` is rejected up front, before the log file is opened (two LR writers; `trainingRun`'s guard is the backstop) |
| `GAMMA` | 1.0 | shared by both schedules so the arms are matched at the same γ; 1.0 = no-op |
| `STEP_SIZE` | 1 | step schedules only |
| `MAX_BATCH_SIZE` | train size / 10 = 661 | cap of the batch scheduler; must not exceed the train-set size (6617 samples) — a larger cap would let a growing schedule drive the batch past the dataset (zero batches per epoch), so it is rejected up front |
| `RESHUFFLE` | 1 | per-epoch reshuffle of the train loader (#381) |

`LR_SCHEDULE` and `BS_SCHEDULE` may both be set without compensation (LR decays
and the batch grows). Unknown schedule names, `BATCH_SIZE` outside `[1, 65535]`,
`STEP_SIZE < 1`, `GAMMA <= 0` and a `MAX_BATCH_SIZE` below `BATCH_SIZE` or above
the train-set size fail fast.

The run log (`examples/_shared/log_schema.py`) records every knob under
`config` (`lr_schedule`, `bs_schedule`, `bs_lr_compensation`, `gamma`,
`step_size`, `max_batch_size`, `reshuffle`) plus `toolchain` (the compiler's
`__VERSION__`, because FMA contraction and libm differ across compilers — see
`docs/conventions/toolchain-parity.md`), and per epoch the values the epoch
actually trained with: `lr`, `batch_size`, `parameter_updates`
(`= train size // batch_size`). Arms at a glance:

```bash
B=./build/examples/examples/har_classifier/train_c_har_classifier
LR_SCHEDULE=exp GAMMA=0.97 $B                                   # LS: LR x 0.97 per epoch
BS_SCHEDULE=exp GAMMA=0.97 BATCH_SIZE=1 $B                      # BS: batch / 0.97 per epoch, LR constant
BS_SCHEDULE=exp GAMMA=0.97 BATCH_SIZE=1 BS_LR_COMPENSATION=1 $B  # BC: as BS, LR corrected for rounding/cap
```

### Best-val-loss snapshot and divergence

`train_c_har_classifier` keeps a snapshot of the optimizer's parameters from
the epoch with the lowest validation loss so far (`stopOnNonFiniteLoss` is
always on for this harness; a NaN/Inf validation loss is never "best"). After
training, the snapshot is loaded and evaluated on the test set, then the
**final** (last-epoch) parameters are restored — `final.test_loss`/`test_acc`,
`outputs/c_predictions.npy` and the plots all keep describing the final model
exactly as before; the snapshot is only evaluated, never kept. The run log's
`final` block gains seven fields:

| field | meaning |
|---|---|
| `diverged` | 1 iff the run stopped at the first non-finite train or validation loss (in the last epoch this leaves `epochs_completed == EPOCHS`); 0 otherwise |
| `epochs_completed` | number of epochs whose callback ran (`== len(epochs)`; `<= config.epochs` when `diverged` — equal iff the divergence happened in the last epoch) |
| `best_val_epoch` | 0-based, the same index as `epochs[].epoch`; the epoch with the LOWEST validation loss, or `null` if none was finite |
| `best_val_loss` | that validation loss, or `null` |
| `best_val_acc` | validation accuracy AT that epoch (not the maximum val acc), or `null` |
| `test_loss_at_best_val` | test-set loss evaluated on the snapshotted (best-val) parameters, or `null` |
| `test_acc_at_best_val` | test-set accuracy evaluated on the snapshotted parameters, or `null` |

A diverging run (e.g. `BATCH_SIZE=1 LR=0.01` at the default develop LR, known
to diverge for a few seeds) no longer aborts (`#446`): it ends gracefully
after the epoch that produced the non-finite loss, exits 0, and writes
`diverged: 1` with `epochs_completed` (and the `epochs` array) shorter than
the requested `EPOCHS` unless the divergence happened in the last epoch.
Stdout gains a matching line after `FINAL test_loss=… test_acc=…`:

```
FINAL@best-val epoch=<n> val_loss=<f> val_acc=<f> test_loss=<f> test_acc=<f>
```

printed only when at least one epoch produced a finite validation loss (i.e.
a snapshot exists — divergence in epoch 0 leaves no snapshot, and the five
`best_val_*` / `*_at_best_val` fields are `null`).

## Packed-SYM weight-quantization memory study

A second trainer, `train_c_har_classifier_sym` (source `train_c_sym.c`), trains
the *same* model with weights + biases stored **packed sub-byte SYM@x** while the
whole backward pass, gradients, and optimizer momentum stay FLOAT32. It quantifies
the on-device memory cost/benefit of weight quantization across widths, against the
FLOAT32 trainer as the reference.

> **Reshuffle asymmetry (read before comparing arms).** Only the float trainer
> reshuffles per epoch by default (`RESHUFFLE=1`, see the knobs section above);
> `train_c_sym.c` and the AdamW trainer still shuffle once. A stored FLOAT32
> reference therefore no longer reproduces under plain defaults, and
> `compare_memory.py`'s "keeps up with FLOAT32" gate (`ACC_TOL = 0.01`) would
> compare a reshuffling float arm against non-reshuffling SYM arms. Regenerate
> the float reference with `RESHUFFLE=0`, or re-run all arms once the SYM
> trainers opt in. Making the per-epoch reshuffle the framework-wide default is
> a deferred follow-up (design Part C item 1), so this asymmetry is temporary.

Key design points (see `docs/CONVENTIONS.md` and the source comments):

- **Weights + biases** are packed `SYM@x` (`ceil(x·N/8)` bytes → @12 = 62.5%,
  @8 = 75%, @4 = 87.5% smaller than FLOAT32's `4N`). Everything else — gradients,
  momentum, activation wires — is FLOAT32.
- **Momentum is FLOAT32**, decoupled from the packed-SYM params via the optimizer's
  own-config knob (`sgdMCreateOptim(..., momentumQuant)`). A packed-SYM momentum
  would re-quantize the velocity to the same coarse levels as the weights.
- **Stochastic rounding** (`SR_HALF_AWAY`) on the training write-back lets a
  FLOAT32 gradient step smaller than one SYM level move the weight *in expectation*
  (the #279 dead-zone escape). Since #279 this is the **framework default**: the
  optimizer factories set seeded-SR write-back rounding (optimizer-owned, param
  storage qConfigs stay deterministic `HALF_AWAY`). `SYM_ROUNDING=det` opts out via
  `optimizerSetWriteBackRounding` to A/B the dead-zone claim — confirmed by the
  ≥10-seed sweep (SR recovers 96–97% of the dead-zone gap; see #279).

### Cosine LR schedule vs. the SYM dead zone (#327)

`LR_SCHEDULE=cosine` (with optional `LR_MIN`, default 0) runs the SYM binary
under `CosineAnnealingLR(T_max = EPOCHS)`: fast start near the int32-overflow
ceiling (#189), fine finish near the sub-grid stall. The cosine tail
deliberately pushes per-step updates BELOW one SYM quantization level — with
the default stochastic rounding (`SR_HALF_AWAY`) those updates survive in
expectation (#279); with `SYM_ROUNDING=det` they die in the dead zone. That
contrast is the point of the demo, not a defect. The PyTorch twin mirrors the
schedule via its `SCHEDULER`/`LR_MIN` module constants. Per-epoch LR lands in
the run log (`epochs[].lr`).

Sweep results (10 seeds, `run_matrix.py --configs sym8 sym8cos sym4 sym4cos`):

| config  | test_acc (mean±sd) | min    | max    | test_loss (mean±sd) | n  |
|---------|---------------------|--------|--------|----------------------|----|
| sym8    | 0.9026 ± 0.0074     | 0.8856 | 0.9104 | 0.3726 ± 0.0954      | 10 |
| sym8cos | 0.8996 ± 0.0076     | 0.8907 | 0.9148 | 0.3258 ± 0.0869      | 10 |
| sym4    | 0.8761 ± 0.0210     | 0.8388 | 0.9046 | 0.6111 ± 0.1671      | 10 |
| sym4cos | 0.8928 ± 0.0114     | 0.8758 | 0.9080 | 0.4212 ± 0.1001      | 10 |

At 8-bit, cosine vs. constant LR is a wash (Δmean=0.0030, well within 1 sd of
either); at 4-bit cosine's mean is higher (0.8928 vs 0.8761, Δ=0.0167) and
comparable in magnitude to sym4's own sd (0.0210) — not clearly outside noise
on 10 seeds — but cosine also cuts the run-to-run spread roughly in half (sd
0.0114 vs 0.0210), a variance-reduction pattern absent at 8-bit.

### SGD vs AdamW (#328)

`train_c_adamw.c` (target `train_c_har_classifier_adamw`) trains the same
model/data as `train_c.c` with `adamWCreateOptim` — decoupled weight decay
(`torch.optim.AdamW` single-tensor sequence, op-for-op) at the PyTorch
defaults: betas 0.9/0.999, eps 1e-8, weight decay 0.01, constant LR 0.001.
This is a defaults-vs-defaults comparison: the SGD demo runs lr 0.01 with
momentum 0.9 and **no** weight decay, the AdamW demo runs lr 0.001 with
decoupled wd 0.01 — the arms differ in both step size and regularization by
design. The PyTorch twin mirrors the arm via its `OPTIMIZER` module constant
(set `LR = 0.001` alongside `OPTIMIZER = "adamw"`).

Sweep results (10 seeds, 20 epochs, `run_matrix.py --configs float adamw`):

| config | test_acc (mean±sd) | min    | max    | test_loss (mean±sd) | n  |
|--------|---------------------|--------|--------|----------------------|----|
| float  | 0.8953 ± 0.0050     | 0.8870 | 0.9033 | 0.3280 ± 0.0571      | 10 |
| adamw  | 0.8988 ± 0.0038     | 0.8938 | 0.9053 | 0.4177 ± 0.0547      | 10 |

Accuracy is a wash (Δmean = 0.0035, within 1 sd of either arm; AdamW's spread
is slightly tighter). AdamW's final cross-entropy is consistently higher
(0.42 vs 0.33) at these defaults — the demo demonstrates the optimizer's
mechanics and cost, it does not claim a convergence win on this task.

Memory cost (darwin `examples_memprofile` RunLogs, this tree): optimizer
state is exactly 2× SGD-momentum's — 81 712 B analytic (m + v, two FLOAT32
buffers per parameter) vs 40 856 B (one momentum buffer); 83 848 B vs
42 076 B measured including the per-parameter `states_t` shells. Stack peak
is identical (27 784 B for both binaries — moment buffers live on the heap,
not the training-step stack; the absolute value is toolchain-dependent, the
SGD-vs-AdamW identity is not).
On the Linux CI runner (warn-only watermark REPORT job, PR #363) the same
identity holds: 31 920 B stack peak for both the SGD and AdamW binaries —
which is also why the AdamW binary shares the `float` watermark budget
bucket rather than getting its own key.

### Pretrain -> freeze -> finetune (#380)

`train_c_finetune.c` (target `train_c_har_classifier_finetune`) demonstrates
layer freezing end to end. **Stage 1** trains the full model exactly like
`train_c.c` and serializes it (`serializeModel`, ODTS v4) to
`outputs/har_pretrained.odts`. **Stage 2** rebuilds the identical topology but
with `.trainable = TRAINABLE_FALSE` on the three Conv1d factories, loads the
stage-1 checkpoint into it (deserialization is grad-presence *tolerant*: a
fully-trainable file loads cleanly into a frozen-backbone skeleton, the
conv layers' grad records are parsed and discarded), builds an optimizer over
it, and fine-tunes. Freezing makes the three conv layers optimizer-invisible
— the optimizer collects only the head `Linear`'s weight + bias
(`optim->sizeStates == 2`, asserted in the binary).

Env-overridable like the rest of the file's config: `STAGE1_EPOCHS`,
`STAGE2_EPOCHS` (default 20 each), plus the usual `LR`/`MOMENTUM`/`SEED`/
`SHUFFLE_SEED`/`LOG_PATH`.

Three memory metrics tell the freezing story — run the binary and read the
`FREEZE`-prefixed stdout lines:

- **`optstate_analytic_*_b` / `grads_*_b`** (full vs. frozen): the optimizer's
  momentum buffers and gradient buffers shrink from all four param layers to
  the head alone — 40 856 B -> 1 560 B (~96% smaller). Analytic: formula-derived
  from topology + dtype, so the figure is fixed regardless of how many epochs
  either stage runs. `params_*_b` does **not** shrink (40 856 B either way,
  same analytic basis) — freezing stops a layer from *training*, it doesn't
  evict its weights from memory; the conv backbone stays resident for
  inference.
- **`dx_peak_stage{1,2}_b`** (the headline number): PR2's backward truncation
  means stage 2 never computes or allocates a gradient wire below the head —
  there is no dx ping-pong at all, just the single CE+Softmax lossGrad seed.
  Analytic: 16 384 B (stage 1) -> 24 B (stage 2), a ~683x collapse —
  formula-derived from topology/dtype, not a per-run measurement.
- **`stack_peak_b`** (watermarked in CI): stage 2's training step skips the
  conv layers' backward entirely (no weight-grad conversion scratch), so its
  stack high-water sits an order of magnitude below the plain/AdamW `float`
  binaries' budget. Measured: 4 016 B vs. their 27 768 B (darwin;
  `check_stack_watermark.py`'s dedicated `finetune` bucket, not a re-key of
  `float`).

A 3-epoch/3-epoch smoke run (`STAGE1_EPOCHS=3 STAGE2_EPOCHS=3 SEED=1`):
stage-1 test_acc 0.6871, stage-2 test_acc 0.7170 — the head continues
improving on the same data after the backbone freezes, as expected.

### Full-SYM wires — the #206 acceptance run

`SYM_WIRES=1` switches `train_c_sym.c` from FLOAT32 activation/dx wires to the
full-SYM configuration: SYM_INT32 (int12) activation **and** dx wires, native
SYM dx compute on every layer (conv/linear/pools/ReLU), the softmax funnel
requantizing its output to SYM, the fake-quant CrossEntropy arms (#206)
consuming the SYM head, and the SYM-aware metrics argmax. Trainable-param
storage stays packed SYM@`SYM_BITS`, param **gradients stay FLOAT32** (#261);
write-back rounding is the ratified seeded-SR default (#279).

Sweep result (10 seeds, 40 epochs, `run_matrix.py --configs sym8w`,
`logs_206_fullsym/`; the float twin reuses the #279-sweep runs — the FLOAT32
path is numerically unchanged since, guarded by the bit-parity CI gate):

| config | test_acc (mean±sd) | min    | max    | n  |
|--------|---------------------|--------|--------|----|
| float  | 0.9065 ± 0.0111     | 0.8799 | 0.9203 | 10 |
| sym8w  | 0.9076 ± 0.0104     | 0.8901 | 0.9270 | 10 |

**Measured degradation: none** (Δmean = +0.0011, well within 1 sd of either
arm). The #206 acceptance criterion braced for a drop citing Deutel et al.
(arXiv:2407.10734) — but the paper's reported degradation is tied to 8-bit
**gradient** range, and this configuration deliberately keeps gradients
FLOAT32, so parity here is consistent with the paper rather than contradicting
it. Quantizing the gradient path is the open research axis (#218, Jan's
#137–#142 ladder), not this run.

### Group-granular quantization (#300)

`train_c_sym.c` also carries a `GROUP_MODE`/`GROUP_SIZE`/`WEIGHT_DTYPE` env
axis on top of everything above, wiring the group-quant epic's grouped
SYM/ASYM machinery (design spec:
`docs/superpowers/specs/2026-07-28-group-quantization-design.md`) into a real
model:

- `WEIGHT_DTYPE=sym|asym` (default `sym`) — packed SYM (scale-only) or ASYM
  (scale + uint16 code-domain zero-point) weight/bias storage, both at
  `SYM_BITS`. Biases are always per-tensor regardless of the weight axis below.
- `GROUP_MODE=tensor|channel|size` (default `tensor`, today's original
  behavior, byte-identical) — `channel` gives one scale/zero-point per output
  channel (`groupSize = N/outCh`); `size` takes an explicit `GROUP_SIZE=<n>`
  and, **whenever `n` does not evenly divide a given layer's own element
  count**, falls back to that layer's per-channel groupSize instead (never a
  short last group). The framework never guesses silently: every run's true
  per-layer resolved shape is in the log's `groups_resolved` field
  (`{"conv1": [numGroups, groupSize], ...}`), and `group_overhead_b` totals the
  scale/zero-point metadata bytes across all 8 param tensors.

The startup storage gate behind the `GATES PASS` line is the shared
`examples/_shared/param_gate.h` checker (SYM / ASYM / BFP arms, #417) — the
epic-#410 PR7 BFP trainer will reuse it.

Resolved shapes on HAR's actual topology (`N` = weight element count, `outCh`
= output channels, `pc = N/outCh` = the per-channel groupSize):

| layer  | shape       | N    | outCh | pc | channel  | G64 (GROUP_SIZE=64) | G32 (GROUP_SIZE=32) |
|--------|-------------|------|-------|----|----------|----------------------|----------------------|
| conv1  | [16, 9, 7]  | 1008 | 16    | 63 | [16, 63] | **[16, 63] (fallback: 1008 = 2⁴·3²·7, neither 64 nor 32 divides it)** | **[16, 63] (fallback)** |
| conv2  | [32, 16, 5] | 2560 | 32    | 80 | [32, 80] | [40, 64]             | [80, 32]             |
| conv3  | [64, 32, 3] | 6144 | 64    | 96 | [64, 96] | [96, 64]             | [192, 32]            |
| linear | [6, 64]     | 384  | 6     | 64 | [6, 64]  | [6, 64]              | [12, 32]             |

conv1 is the one layer whose element count (1008) is divisible by neither
64 nor 32, so it silently runs at per-channel
granularity under both `g64` and `g32` while the other three layers get their
requested group size — a property of the topology, not a bug; always read
`groups_resolved` rather than inferring group shapes from a config name.

**`run_matrix.py` arms** (`sym{4,6}pc`/`g64`/`g32`, `asym{4,6}`/`pc`/`g64`/`g32`
— 14 total, at the two coarse widths where the accuracy-per-byte frontier is
most interesting): see the `CONFIGS` dict's own header comment in
`run_matrix.py` for the exact per-arm env. Aggregate with the same
`compare_memory.py` used by the rest of this sweep.

**The #300 epic-acceptance sweep** (spec §8; run offline, NOT in CI — same
regime as the full study above):

```bash
uv run examples/har_classifier/run_matrix.py \
  --configs sym4 sym4pc sym4g64 sym4g32 asym4 asym4pc asym4g64 asym4g32 \
            sym6 sym6pc sym6g64 sym6g32 asym6 asym6pc asym6g64 asym6g32 \
  --seeds 1 2 3 4 5 6 7 8 9 10 --logs examples/har_classifier/logs_300_groups
```

16 arms × 10 seeds = 160 runs (~2× the full-study wall clock; `--jobs 8` and
`--resume` apply). Honest-aggregation rules are unchanged (multi-seed
aggregate only, ≥10 seeds for shippable claims, convergence failures are
recorded findings) with ONE addition for the frontier: the per-arm weight
footprint is `params_b + group_overhead_b` — the per-group scale/zero-point
metadata is real on-device memory and the whole point of the finer-granularity
arms is paying it for accuracy, so the frontier is reported against the SUM,
never `params_b` alone (spec §8's "including the §4 overhead, honestly
reported"). Note `compare_memory.py` reports `params_b` only — add each arm's
constant `config.group_overhead_b` from any of its logs by hand. `groups_resolved` in each log is the ground truth for what a
config name actually ran (see the fallback table above).

**ODTS v5 round-trip demo** (`ODTS_ROUNDTRIP=1`, default off): after the final
test eval, `train_c_sym.c` serializes the trained model to
`outputs/har_sym_group.odts`, builds a **fresh per-tensor skeleton** (same
topology, same `WEIGHT_DTYPE`, but always `GROUP_MODE=tensor` regardless of
the run's own group mode), deserializes the file into it, re-runs the same
test eval, and asserts the result is bit-identical to the original — loud
`stderr` + exit 1 on any mismatch. This is the concrete format-parity
evidence for the group-quant spec's ODTS §6: a file written by a **grouped**
run must load cleanly into a **per-tensor** reader via
`deserializeQConfig`'s realloc-on-numGroups-mismatch relax, not just in the
unit-test fixtures. On success the run's JSON log gains
`"config": {..., "odts_roundtrip": "ok"}`.

### Block floating point (#410)

`train_c_har_classifier_bfp` trains the same 12-layer model with classic block
floating point: packed `mantissaBits`-wide two's-complement mantissas sharing one
`exponentBits`-wide exponent per block (1 B of metadata per block, vs 4 B per float
scale for grouped SYM). Spec: epic design 2026-07-29 + PR7 design 2026-09-14.

**Knobs** (env): `BFP_MANTISSA_BITS` (2..16, default 8), `BFP_EXPONENT_BITS` (2..8, 8),
`BFP_WEIGHT_BLOCK=tensor|channel|N` (N falls back to per-channel where it does not divide
a weight's element count — conv1 at 32/64, like the SYM g32/g64 arms), `BFP_WIRE_BLOCK=
float|tensor|N` (`float` keeps FLOAT32 wires; N resolves PER WIRE and falls back to
per-tensor on the 6-element head wires), `BFP_MATH=native|fq`, `BFP_GRADS=0|1`,
`BFP_STATE=0|1` (needs GRADS=1), `BFP_ROUNDING=sr|det`; plus `LR/MOMENTUM/EPOCHS/SEED/
SHUFFLE_SEED/LR_SCHEDULE/LR_MIN/LOG_PATH`. `MOMENTUM` must be finite and > 0 — rejected at
the env gate (exit 1); behind it the STATE gate expects the optimizer's eight momentum
buffers to exist (`MOMENTUM=0` would allocate none).

**Config names encode every knob** (`--resume`-safe):
`bfp_wb{t|pc|N}_ab{f|t|N}_m{M}_e{E}_x{nat|fq}_g{0|1}_s{0|1}_r{sr|det}_l{const|cos}`.

**The arm ladder.** `abf` (FLOAT32 wires) is NOT a float arm: the GEMM math still runs
`ARITH_BFP` and stages each float operand per-tensor at the weight widths. `xfq` pins the
four GEMM math slots and Softmax's forward to FLOAT32 while Relu/pools keep their BFP arms
— it is the GEMM fake-quant reference curve, not a whole-model fake-quant twin (the pools'
FLOAT32 arms reject packed wires). `g1` / `g1s1` store grads / momentum as per-tensor BFP.

**Resolved shapes** (from the logs' `groups_resolved` / `wires_resolved`):

| weights | wb8 | wb32 | pc | t |
|---|---|---|---|---|
| conv1 (1008) | {126,8} | {16,63} (fallback) | {16,63} | {1,0} |
| conv2 (2560) | {320,8} | {80,32} | {32,80} | {1,0} |
| conv3 (6144) | {768,8} | {192,32} | {64,96} | {1,0} |
| linear (384) | {48,8} | {12,32} | {6,64} | {1,0} |

| wires | ab16 | ab32 | t |
|---|---|---|---|
| 2048-element (conv/relu out, most dx) | {128,16} | {64,32} | {1,0} |
| 1024-element (pool1/pool2 out, conv2/conv3 dx) | {64,16} | {32,32} | {1,0} |
| 64-element (pool3/flatten out, flatten/linear dx) | {4,16} | {2,32} | {1,0} |
| 6-element (linear/softmax out, loss grad) | {1,0} fallback | {1,0} fallback | {1,0} |

**Rounding.** `OUT_WRITE` rounds by the operation's mode, so the knobs are carried by the
math slots: forward packs and float-operand staging are always HALF_AWAY (deterministic
inference, evaluation never draws from the RNG); under `sr` the dx packs
(`propLossMath`), grad/state packs and the optimizer write-back use SR_HALF_AWAY; `det`
is HALF_AWAY everywhere. **Exception:** the CrossEntropy loss-grad pack (`p − y`, 6
elements) has no operation carrier (it clones softmax.out's config and packs through
`executeConvert`) and is HALF_AWAY under both — so `sr` vs `det` compares nine of the ten
dx seams. **At coarse widths this can matter a lot:** with m=4 (and more so e=4) the five
small components of `p − y` round deterministically to zero on that first pack — a dead
zone at the head that SR would escape elsewhere. A `_m4_`/`_e4_` arm that stalls at the
head is a candidate symptom of this exception, not necessarily of the width; see the PR7
design §3.5.1 for the follow-up (a loss-owned wire template).

**Finding (stage-1 smoke, 1 epoch, seed 1) — recorded, not fully diagnosed.** The
`bfp_wb32_ab16_m6_e8_xnat_g0_s0_rdet_lconst` arm (deterministic rounding, otherwise
identical to the anchor) did not learn: `train_loss=1.789` (≈ ln 6) and `test_acc=0.014`
— well below both the test set's 14% minimum class frequency and the same seed's
untrained `initial_val_acc≈0.128`, so this is neither a uniform/constant output (that
would score ≥14%) nor simple stasis near init. The `sr` arm at the same widths learns
normally (`test_acc≈0.386`). **Hypothesis (UNVERIFIED):** the #279 dead zone at full
strength — at m=6 every `lr·grad` step may be sub-ULP of the per-group weight grid, so
HALF_AWAY write-back rounds every update away while `sr`'s stochastic write-back escapes
it in expectation. A pure dead zone predicts stasis near init, not the observed active
divergence BELOW it, so the hypothesis as stated does not fully explain the number; the
mechanism has not been established, and this is an open question, not a conclusion.
Diagnostic for a follow-up: `LOG_CODE_MOVEMENT=1 BFP_ROUNDING=det … train_c_har_classifier_bfp`
(the #279 SYM instrumentation, inherited by the BFP trainer) — inspect `codes_changed_frac`
in the log. **This det-arm number is suspect until that diagnostic has run.** Concretely:
`LOG_CODE_MOVEMENT=1 EPOCHS=2 BFP_ROUNDING=det … train_c_har_classifier_bfp` —
`codes_changed_frac == 0.0` together with accuracy far from `initial_val_acc` points at an
eval/first-encode tooling bug rather than the kernel; `codes_changed_frac > 0` with
below-chance accuracy points at biased drift instead. A second, localizing run —
`BFP_ROUNDING=det BFP_WIRE_BLOCK=float EPOCHS=1 … train_c_har_classifier_bfp` — removes the
dx-pack seam and narrows the search further.

**Headroom.** Native BFP kernels accumulate int32 block partials; a config is rejected
before data load when `min(block, K) > INT32_MAX >> (2m−2)` for any HAR reduction
(63/80/96/64 forward, 128/64/32 weightGrad, 160/192/6 dx). The whole m ≤ 8 grid is safe;
the bound binds from m=14 with blocks ≥ 32, and, for the per-tensor arms (`wbt`, `abf`),
already from m=13 (conv1's weightGrad reduction L=128 exceeds the m=13 limit of 127).

**Memory.** The `memory` block now carries eleven categories: payload (`params_b`,
`grads_b`, `optstate_analytic_b`, `activations_b`, `io_b`, `pool_backward_b`,
`dx_peak_b`) and metadata (`group_overhead_b`, `grad_overhead_b`,
`optstate_overhead_b`, `wire_overhead_b`); `mcu_total_b` is their sum IN THE C
HARNESS since PR7 (SYM logs' totals grew by their group metadata that was previously
only reported under `config`). Accuracy-per-byte = accuracy / (payload + ALL metadata).
For the anchor config (`bfp_wb32_ab16_m6_e8_xnat_g0_s0_rsr_lconst`), the stage-1 smoke
reported `activations_b`=10858, `wire_overhead_b`=1162, `group_overhead_b`=304.

**Stage-1 matrix and run:**
```bash
cmake --preset examples_memprofile && cmake --build --preset examples_memprofile --target train_c_har_classifier train_c_har_classifier_sym train_c_har_classifier_bfp
uv run examples/har_classifier/run_matrix.py \
  --configs float sym6g32 \
            bfp_wb32_ab16_m6_e8_xnat_g0_s0_rsr_lconst bfp_wbt_ab16_m6_e8_xnat_g0_s0_rsr_lconst \
            bfp_wbpc_ab16_m6_e8_xnat_g0_s0_rsr_lconst bfp_wb8_ab16_m6_e8_xnat_g0_s0_rsr_lconst \
            bfp_wb32_abf_m6_e8_xnat_g0_s0_rsr_lconst bfp_wb32_abt_m6_e8_xnat_g0_s0_rsr_lconst \
            bfp_wb32_ab32_m6_e8_xnat_g0_s0_rsr_lconst bfp_wb32_ab16_m4_e8_xnat_g0_s0_rsr_lconst \
            bfp_wb32_ab16_m8_e8_xnat_g0_s0_rsr_lconst bfp_wb32_ab16_m6_e4_xnat_g0_s0_rsr_lconst \
            bfp_wb32_ab16_m6_e8_xnat_g0_s0_rdet_lconst bfp_wb32_ab16_m6_e8_xfq_g0_s0_rsr_lconst \
            bfp_wb32_ab16_m6_e8_xnat_g1_s0_rsr_lconst bfp_wb32_ab16_m6_e8_xnat_g1_s1_rsr_lconst \
  --seeds 1 2 3 4 5 6 7 8 9 10 --epochs 50 --jobs 8 --logs examples/har_classifier/logs_bfp
uv run examples/har_classifier/compare_memory.py --logs examples/har_classifier/logs_bfp
```
Runtime: measured, not estimated — but measured unoptimized. These numbers come from the
`examples_memprofile` preset, whose `CMakeCache.txt` has `CMAKE_BUILD_TYPE=` (empty: the
preset sets only `BUILD_EXAMPLES`/`ODT_MEM_PROFILE`, no optimization flag, i.e. -O0),
not the `-O2` the campaign discipline (spec §1/§13) calls for. The stage-1 smoke (1 epoch,
seed 1, 16 configs under `--jobs 4`) put the anchor
(`bfp_wb32_ab16_m6_e8_xnat_g0_s0_rsr_lconst`) at 273.6 s/epoch under that 4-way load
against `sym6g32`'s 86.4 s — ≈3.2× under the same 4-way load (273.6 s vs 86.4 s); the
anchor's solo time was 223.8 s/epoch. A 50-epoch run at these -O0 rates is therefore ≈3 h
per (config, seed); the full 16-config × 10-seed stage-1 matrix is ≈60 h wall at
`--jobs 8`. Treat all of the above as an **upper bound**: the spec §13 one-epoch `-O2`
probe (build with
`cmake --preset examples_memprofile -DCMAKE_C_FLAGS="-O2 -ffp-contract=fast"` then
`cmake --build --preset examples_memprofile --target train_c_har_classifier train_c_har_classifier_sym train_c_har_classifier_bfp`
— `-DCMAKE_C_FLAGS` REPLACES rather than extends the `host` preset's flags, so
`-ffp-contract=fast` must be repeated or the campaign build silently drops the CI/unit-test
numerics contract; do NOT use `CMAKE_BUILD_TYPE` instead, since `Release`/`RelWithDebInfo`
also add `-DNDEBUG`)
supersedes both these absolute times and the BFP:SYM ratio once run; do not plan the real
campaign's wall-clock budget off the numbers above. The stack-watermark `bfp` bucket is
uncalibrated (report-only) until its own PR.

### Build with memory profiling

Memory instrumentation is compiled in only under the `examples_memprofile` preset
(`-DODT_MEM_PROFILE`); the plain `examples` preset (the CI bit-parity build) is
byte-identical bare calloc/free.

```bash
cmake --preset examples_memprofile
cmake --build --preset examples_memprofile --target \
    train_c_har_classifier train_c_har_classifier_sym train_c_har_classifier_adamw
```

All three binaries are env-configured: `SEED`, `EPOCHS`, `LR`, `MOMENTUM`, `LOG_PATH`
(+ the float binary's `BATCH_SIZE`/`LR_SCHEDULE`/`BS_SCHEDULE`/`BS_LR_COMPENSATION`/`GAMMA`/
`STEP_SIZE`/`MAX_BATCH_SIZE`/`RESHUFFLE`, see above; + `SYM_BITS`, `SYM_ROUNDING`, `SYM_WIRES`, `LR_SCHEDULE`, `LR_MIN`, `GROUP_MODE`,
`GROUP_SIZE`, `WEIGHT_DTYPE`, `ODTS_ROUNDTRIP` for the SYM binary; the AdamW
binary ignores `MOMENTUM`). Each writes an extended RunLog JSON whose `memory` block
carries per-category analytic bytes, instrumented heap/stack/RSS peaks, and the **reconciliation
gap** (`heap_peak − mcu_total`, ≈ the host-resident dataset the MCU would
stream — recorded, never massaged).

### Offline sweep + honest aggregation

```bash
# Full study: {float, sym@12, sym@10, sym@8, sym@6, sym@4, sym@8cos, sym@4cos,
# adamw, sym@8det, sym@6det, sym@4det} × seed 1..10 = 120 runs.
# LONG (~60+ h offline; NOT wired into CI). Use --configs/--seeds/--epochs to smoke.
uv run examples/har_classifier/run_matrix.py                 # full
uv run examples/har_classifier/run_matrix.py --configs float sym8 --seeds 1 2 --epochs 2  # smoke

# Aggregate 10-seed mean±std + comparison table + 3 pictograms.
uv run examples/har_classifier/compare_memory.py --plots
```

**Read the numbers honestly.** The training loop streams the macro-batch one sample
at a time (**micro-batch B=1**; gradients accumulate at the optimizer), so the
concurrent activation peak is *one sample's* worth — not 64×. That makes the on-device
footprint roughly balanced: at FLOAT32 (~181 KB total) params, grads, and momentum are
each ~22% and activations ~31%. At SYM@8 the weight *category* shrinks **75%** (40 KB →
10 KB), which translates to a **material ~17% drop in total on-device training RAM**
(~181 KB → ~151 KB; ~19% at SYM@4). These totals are the **heap** categories only:
`mcu_total_b` excludes the training-step **stack** high-water (`stack_peak_b`, reported
separately, ≈27 KB float / ≈52 KB sym after the #296 packed-repack scratch). Including
the stack, the SYM totals still come out ahead (SYM@8 ≈202 KiB vs FLOAT32 ≈208 KiB of
provisioned RAM), so the stack **confirms** the SYM win rather than erasing it — this
reverses the pre-#296 picture; see `check_stack_watermark.py` for the current numbers.
As of #321 `mcu_total_b` also counts the backward-only on-device state the earlier
figures omitted — the persistent MaxPool argmax buffers (~8 KB, `pool_backward_b`) and
the transient dx ping-pong peak (~16 KB, `dx_peak_b`) — so the absolute totals above are
~25 KB higher than quoted; that delta is uniform across configs, so the relative SYM win
is unchanged. Regenerate the exact figures from a fresh sweep.
`compare_memory.py` reports the weight-category drop and the total-footprint drop
**separately** — they answer different questions. The
next wins are grads and momentum (each another ~22%), reachable via the optimizer's
per-config quant knob. Headline claims come **only** from the ≥10-seed aggregate; a
`--min-seeds` guard warns loudly on smoke-sized runs.

Sweep artifacts (`logs/`, `outputs/memory_summary.json`, `plots/har_mem_*.png`) are
gitignored like every other example's results — regenerate them locally.
