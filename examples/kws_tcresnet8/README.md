# kws_tcresnet8 — TC-ResNet8 keyword spotting, C and a PyTorch twin

TC-ResNet8 (Choi et al., "Temporal Convolution for Real-time Keyword Spotting on Mobile
Devices", 2019) trained on Speech Commands with ODT's FLOAT32 graph training, next to a PyTorch
twin. It is a residual 1D CNN: MFCC coefficients are the channels, time is the length.

```
stem:   Conv1d(40→16, k3, SAME, no bias) → BN → ReLU
block ×3 (c_out = 24, 32, 48):
  main:     Conv1d(k9, s2, SAME) → BN → ReLU → Conv1d(k9, s1, SAME) → BN
  shortcut: Conv1d(k1, s2) → BN → ReLU
  out:      Add → ReLU
head:   AdaptiveAvgPool1d(1) → Flatten → Dropout(0.5) → Linear(48→classes, no bias) → Softmax
```

`SAME` is TF-SAME (the odd pad goes on the right); the twin pads explicitly with the same rule,
because PyTorch's `padding='same'` rejects stride 2.

| file | what |
|---|---|
| `train_c.c` | the C program (target `train_c_kws_tcresnet8`); one binary for every run |
| `train_pytorch.py` | the PyTorch twin, written from the paper and the reference code |
| `same_start.py` | the same-start check (below) |
| `run_sweep.py` | the 10-seed sweep runner |
| `compare.py` | the C-vs-PyTorch comparison, loss curves and memory table |

## Data

Two setups, both written by `examples/kws_mfcc/prepare_data.py` (Speech Commands v0.02,
downloaded once into `examples/_shared/data/speech_commands/`):

| | setup | classes | input | command |
|---|---|---|---|---|
| A | the paper's | 12: yes, no, up, down, left, right, on, off, stop, go, silence, unknown | 40 MFCC, 30 ms window, 10 ms hop, no centre padding: [40, 98] | `KWS_CLASSES=12 uv run examples/kws_mfcc/prepare_data.py` → `examples/kws_mfcc/data/12class_paper/` |
| B | kws_mfcc's | 6: yes, no, up, down, silence, unknown | [40, 32] (hop 512) | `uv run examples/kws_mfcc/prepare_data.py` → `examples/kws_mfcc/data/6class/` |

The C program reads T and the class count from the `.npy` files, so one binary serves both.

## Knobs (environment variables, the same on both sides)

| variable | default | meaning |
|---|---|---|
| `KWS_DATA_DIR` | required | a directory with `{train,val,test}_{x,y}.npy` |
| `TCR_WIDTH` | `full` | `full`: stem 16, blocks 24/32/48 (65,168 parameters on A; paper: 66K). `half`: every channel count halved (17,080 on A) — an ODT variant for the Pico, not in the paper |
| `TCR_SEED` | 42 | initialisation, Dropout masks and sample order |
| `TCR_MICROBATCH` | 100 | rows per forward/backward call (must divide the batch of 100); BN statistics are taken over the micro-batch |
| `TCR_EPOCHS` | paper count | override the epoch count (the LR schedule keeps the paper's step size) |
| `TCR_OUT` | `examples/kws_tcresnet8/logs` | where `c.json` / `pytorch.json` go (schema: `examples/_shared/log_schema.py`) |
| `TCR_SAME_START` | unset | C only: load the twin's exported initial state; used by `same_start.py` |
| `TCR_SAME_START_STEPS` | 100 | C only, with `TCR_SAME_START`: steps to train (1 to 100); writes the losses, the state after the last step and the first 200 validation probabilities |

## Recipe

SGD, momentum 0.9, batch 100, LR 0.1, `STEP_LR` γ = 0.1. Weight decay 1e-3 on the conv and linear
weights only: two optimizer groups, the BN group without decay. Xavier-uniform init on both sides,
each from its own RNG. BN momentum 0.003 (TF decay 0.997), eps 1e-3. Dropout 0.5.

The paper drops the LR at 10k and 20k iterations and stops at 30k. Here the step size is the
whole number of epochs closest to 10k iterations (`round(10000 / iterations-per-epoch)`, round
half up), and training runs 3 step sizes:

| data | train samples | iterations/epoch | LR 0.1 | LR 0.01 | LR 0.001 | epochs |
|---|---|---|---|---|---|---|
| A | 36,923 | 369 | epochs 0–26 | 27–53 | 54–80 | 81 (29,889 iterations) |
| B | 18,660 | 186 | epochs 0–53 | 54–107 | 108–161 | 162 (30,132 iterations) |

## Deviations from the paper

Any accuracy from this example is quoted with these:

- No background-noise mixing or time-shift augmentation.
- No early stopping: final-epoch accuracy. The paper reports the best-validation checkpoint,
  averaged over 15 runs; here it is mean ± std over 10 seeds.
- Silence is synthetic low-amplitude noise; Google's version crops the background-noise files.
  The unknown sampling rate may differ.
- Speech Commands v0.02 (the paper does not name a version).
- The LR schedule is mapped to whole epochs (table above), so the iteration budget is ~30k, not
  exactly 30k.

Library note (not used here): with optimizer groups, ODT scales a cosine schedule's `etaMin`
by the group's LR multiplier too; torch does not. This example uses `STEP_LR` only.

## BatchNorm and the micro-batch

ODT's BatchNorm1d takes batch statistics over the micro-batch it is given; PyTorch takes them over
the whole batch. So the C-vs-PyTorch comparison runs at micro-batch = batch = 100. Micro-batch 4 is
the Pico-sized memory setting; its accuracy runs show what small-batch BN statistics cost. It also
shortens the running-statistics horizon: BN updates its running average once per micro-batch, so at
micro-batch 4 it updates 25 times per optimizer step instead of once.

## Same-start check

`same_start.py` exports the twin's initial weights and BN buffers, the C program loads them, and
both train the first steps of epoch 0 in the same sample order (`examples/_shared/keyed_sampler.py`
mirrors C's shuffle) with Dropout off. The results go to
`logs/same_start_<data>_<width>/same_start.json`.

```
uv run examples/kws_tcresnet8/same_start.py --data A --width full [--c-binary PATH]
```

It exits 0 only when all three checks pass:

1. **Step-1 loss:** C within a relative 1e-4 of PyTorch float32.
2. **State after one update:** parameters, SGD momentum, BN running mean and running variance,
   each compared with a float64 run of the twin (t64). Per kind, d(X) = max over tensors of
   max|X − t64| / max|t64|; C passes if d(C) ≤ 10 × max(d(t32), 1e-6). The 1e-6 floor (about
   8 float32 ulp) keeps the bound above one ulp when PyTorch float32 sits next to float64.
3. **Eval argmax:** after that update, C and PyTorch float32 agree on ≥ 199 of the first 200
   validation samples (running BN statistics, no Dropout).

The 100-step loss curve is printed and saved, with the first step where the two sides differ by
more than 1e-4, but it never fails the check. Any two float32 implementations part within a few
steps: a ReLU input near zero rounds to the other side, and LR 0.1 with momentum amplifies it.
PyTorch float32 parts from its own float64 run at step 3 on A full. C parts earlier than PyTorch
float32 mainly because its BatchNorm sums batch × time in one float32 accumulator. The
training-level C-vs-PyTorch test is the 10-seed comparison in `compare.py`.

Measured (seed 42, `examples` preset):

| check | A full | B half |
|---|---|---|
| 1. step-1 loss, relative | 8.9e-8 | 2.2e-7 |
| 2. param: d(C) / bound | 3.5e-3 / 1.7e-2 | 3.0e-6 / 1.0e-5 |
| 2. momentum: d(C) / bound | 3.5e-3 / 8.4e-2 | 4.9e-6 / 2.9e-5 |
| 2. running mean: d(C) / bound | 5.1e-6 / 1.0e-5 | 1.5e-6 / 1.0e-5 |
| 2. running var: d(C) / bound | 1.2e-6 / 1.0e-5 | 6.4e-7 / 1.0e-5 |
| 3. eval argmax agreement | 200/200 | 200/200 |
| curve: first step above 1e-4 (information) | 3 | 4 |

Both pass. On A full, ReLU ties already flip inside step 1 on both sides, so d(t32) is large
there too (param 1.7e-3, momentum 8.4e-3). The tightest margin is A full's running mean at half
its floored bound.

## Builds

The `examples` preset has no optimisation (-O0) and is far too slow for training runs (A, full
width: about 10 minutes per epoch). The sweep uses two optimized builds of the same toolchain:

```
cmake --preset examples_release            && cmake --build --preset examples_release --target train_c_kws_tcresnet8
cmake --preset examples_release_memprofile && cmake --build --preset examples_release_memprofile --target train_c_kws_tcresnet8
```

`examples_release` is `examples` with `CMAKE_BUILD_TYPE=Release` (training and timing);
`examples_release_memprofile` adds `ODT_MEM_PROFILE` (memory runs). `run_sweep.py` configures
and builds both itself. The optimized build rounds differently from an -O0 build (most likely
multiply-add contraction: every host preset sets `-ffp-contract=fast`), so the same seed gives
losses that agree early and drift apart over training; compare runs of one build only.

## Sweep and comparison

```
uv run examples/kws_tcresnet8/run_sweep.py --time-one      # 2 epochs per configuration and arm, solo; projects the sweep
uv run examples/kws_tcresnet8/run_sweep.py --dry-run       # list what would run
uv run examples/kws_tcresnet8/run_sweep.py --jobs 6 [--only 'A_full_*,*mem*']
uv run examples/kws_tcresnet8/compare.py
```

The sweep is data {A, B} × width {full, half} × seeds 0–9 for three arms — C at micro-batch 100,
PyTorch at 100, C at 4 (120 runs) — plus 8 one-epoch memory runs (each configuration at
micro-batch 100 and 4, `ODT_MEM_PROFILE` build; the heap peak is reached in the first step).
Runs are named `<data>_<width>_<side>_m<micro-batch>_s<seed>` and `<data>_<width>_mem_m<micro-batch>`
(the names `--only` matches). A run with a complete log is skipped, so a stopped sweep resumes.

Results land in `logs/sweep/<data>_<width>/<side>_m<mb>/seed<k>/{c,pytorch}.json` (console
output next to it), memory runs in `logs/sweep/<data>_<width>/mem_m<mb>/`, timing runs in
`logs/timing/`, and the plot in `plots/sweep/loss_curves.png`.

`compare.py` prints, per configuration, mean ± std of the test accuracy for the three arms and the
hypothesis test: the C and PyTorch means at micro-batch 100 must differ by less than one pooled
standard deviation (exit 1 otherwise). Missing seeds are listed. Diverged seeds are listed per arm
and kept out of the mean, the std and the hypothesis test; a configuration where only one side
diverged fails.

The memory table reports the training memory: the heap peak minus the loaded splits and minus the
train/val loader index arrays (one `size_t` per sample; shown in their own column), i.e.
parameters, gradients, optimizer state, activations and scratch. The table prints its margin to the
Pico 2 (RP2350) SRAM of 520 KiB. The number is the heap only, measured on a 64-bit host: it leaves
out the stack, static data and the firmware, and host pointers and `size_t` are twice the RP2350
width, so a device fit needs headroom beyond a positive margin. The test loader is created after the
peak is read.
