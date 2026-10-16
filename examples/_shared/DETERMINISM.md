# Determinism Contract

Every random value of a C training run is a pure function of a key and a
position: the value `i` of the stream that the key of path `p` under the root
`r` opens, F(r, p, i). The root comes from one seed, the path names who draws
(which update, call, layer, op and purpose), and nothing else enters: not the
call order, not the thread, not a recomputation, not any other draw. Running
the same seed twice gives the same values bit for bit, and so does running the
same work in another order. The keyed core replaced a module-global generator
in #4, so values drawn by builds before that change differ; within the keyed
core a value changes only if its key path or the backend below changes, and
known-answer tests pin both.

## Seeds

`examples/_shared/seeds.py` holds the example seeds:

- `SEED = 42` — the seed the C trainers hand to `trainingRngInit` by default
  (the root of their key tree), and the seed the PyTorch twins pass to
  `torch.manual_seed` and to `KeyedShuffleSampler`.
- `SPLIT_SEED = 42` — seeds the numpy train/validation split of every
  `prepare_data.py`; neither trainer draws with it.

## The key tree

The root is `rngKeyFromSeed(seed)`; every seed, 0 included, is an ordinary
seed. Each edge is `rngFold(parent, tag, index)`: one Threefry-2x32-20 block
with the parent as the Threefry key and `(tag, index)` as the counter. The tag
values are the `RNG_TAG_*` defines of `src/rng/include/RNG.h`, mirrored as
`TAGS` in `examples/_shared/keyed_rng.py` and checked against the header by a
test. Tags are append-only and never renumbered, 0 is never a tag and no two
tags share a value, so every path decodes one way. A tag that only a build flag
defines keeps its value, so a build without the flag has a gap.

The trainer's `trainingRng_t` (`src/userApi/training_loop/include/TrainingLoopApi.h`)
holds the root and two counters: `step` counts optimizer updates and `epoch`
completed epochs. Its helpers are the one place the upper part of the tree is
written out:

    root = rngKeyFromSeed(seed)                      trainingRngInit
    |-- STEP(step)                                   trainingRngStepKey
    |   |-- CALL(c)                                  trainingRngCallKey
    |   |   `-- LAYER(l) -- FORWARD(0) | BACKWARD(0)
    |   |       |-- OP(k) -- STAGE(s) | KERNEL(0) | OUT(0) | ACC(0) | ACC(1)
    |   |       `-- DROPOUT_MASK | DROPOUT_BRIDGE | SOFTMAX_SHIFT | NORM_DX
    |   |           | LOSS | CONVERT (index 0)
    |   `-- CL_REPLAY(0) -- CLASS(c) -- SLOT(i)       replay samples
    |-- UPDATE(step)                                 trainingRngUpdateKey
    |   `-- PARAM(p) -- OP(k) (write-back) | SCALE(0) (mean-scale)
    |-- EVAL(epoch) -- CALL(c) -- ... as under STEP  trainingRngEvalKey
    |-- SHUFFLE(epoch, or 0)                         trainingRngShuffleKey
    |-- SHUFFLE_EVAL(0)                              trainingRngShuffleEvalKey
    `-- INIT(l) -- PARAM(slot)                       trainingRngInitKey

`trainingRngAdvance` moves `step` after each update and
`trainingRngAdvanceEpoch` moves `epoch` after each epoch; a counter that would
pass `UINT32_MAX` stops the program with a named error rather than reuse keys.
A draw that needs a key and gets none stops the program with a named error; a
path that does not draw needs no key.

## Streams (backend v1: keyed xorshift32)

- `rngOpen(key)` starts a stream at `w[0]`, else `w[1]`, else 1 (state 0 would
  stick). Reopening a key replays its stream.
- `rngNextU32` is one xorshift32 step (shifts 13, 17, 5); the draw is the new
  state.
- `rngNextUnitFloat` takes the top 24 bits of the next draw over 2^24, a float
  in [0, 1).
- `rngNextBounded(stream, bound)` takes the next draw below
  `UINT32_MAX - UINT32_MAX % bound` (a draw at or above it is redrawn), modulo
  `bound`.
- A shuffle is Fisher-Yates from the identity order: from the last slot down,
  slot `i` swaps with `rngNextBounded(i + 1)`; fewer than two items draw
  nothing.
- `randomUniformCtx` returns `min + u * (max - min)`; `randomNormalCtx` is the
  Box-Muller transform (cosine branch) on `u1`, then `u2`, with `u1` redrawn
  while it is at or below 1e-7.

## Threads

No function in `RNG.h` keeps mutable module state: every stream is owned by
its caller and every key is a value, so the functions are safe from any
thread. A stream itself is not shared: give each thread its own (opening the
same key twice yields the same values). The Dropout mask sampler is a
separate, process-wide setting (`bernoulliSetFillMaskFn`): install it before
training starts.

## Init weights

Each side uses its own native init, and init weights are NOT exchanged between
the implementations: that keeps the parity story robust to known C-side init
drift. PyTorch uses `torch.nn.init` defaults (Kaiming uniform for
`Conv1d`/`Linear` weights, uniform `[-1/√fan_in, 1/√fan_in]` for biases). C
draws each layer's initial values from that layer's `INIT` key
(`trainingRngInitKey`), through the factory's `initWeightTensor`/
`initBiasTensor` and `initDistribution`.

## Shuffle ordering

The C data loader orders a `shuffle = true` loader with `dataLoaderShuffle`
under one key, so an order is a function of its key alone. `trainingRun`
shuffles the train loader at the start of every epoch, under that epoch's
`SHUFFLE` key when the loader reshuffles per epoch
(`dataLoaderSetReshufflePerEpoch`; the HAR float32 trainer's `RESHUFFLE=1`, its
default) and under epoch 0's key otherwise, and a `shuffle = true` eval loader
once per run under its `SHUFFLE_EVAL` key. `examples/_shared/keyed_rng.py`
reproduces every order in Python:
`keyed_shuffle(n, fold(key_from_seed(SEED), TAGS["SHUFFLE"], e))` is epoch
`e`'s order of an `n`-sample train loader (`e = 0` in every epoch without the
per-epoch reshuffle). The PyTorch twins train in these orders:
`KeyedShuffleSampler(n, SEED, reshuffle)` (`examples/_shared/keyed_sampler.py`)
yields them, so both sides see the same samples in the same order.

## Resuming

`trainingRngSerialize` writes the training state as one 24-byte `ODTK` record
(root and both counters); `trainingRngDeserialize` reads it back. Saved between
epochs beside the model's ODTS file, it lets a stopped run go on with the keys
it would have used, and so with the same shuffle orders and draws. The record
holds no position inside an epoch, and optimizer and scheduler state are not
part of it (#350): an exact resume needs SGD without momentum and no
scheduler.

## Outside the key tree

- `SPLIT_SEED`'s numpy split in `prepare_data.py`.
- The PyTorch twins' own init and dropout (`torch.manual_seed(SEED)`).

## Parity assertion regime

Final-state only. After `N` epochs both sides should reach the same
test accuracy / loss within per-example tolerance tables in `compare.py`.
The framework's `parity.ParityCheck` either uses absolute or relative
tolerance, never both.

## The tests that hold this contract

- Keys: `test/unit/rng/UnitTestRNG.c` (`testRngFoldMatchesThreefryKnownAnswers`,
  `testRngKeyFromSeedMatchesReference`, `testRngFoldKeysDoNotDependOnDerivationOrder`);
  the helpers in `test/unit/userAPI/UnitTestTrainingLoopApi.c`
  (`testTrainingRngStepKeyIsTheStepFoldOfTheRoot` and the other
  `testTrainingRng…` tests).
- Streams: `testRngNextU32IsTheXorshift32Recurrence`,
  `testRngNextUnitFloatIsTheTop24BitsOfTheDraw`,
  `testRngNextBoundedShufflesLikeTheReference`,
  `testRngReopenedStreamReplaysItsValues` (`UnitTestRNG.c`).
- Order independence: the `…IsAPureFunctionOfItsKey` and
  `…IsAPureFunctionOfItsForwardOpKey` layer tests,
  `testReplayDrawsDependOnTheStepKeyAlone`,
  `testLinearInitIsAFunctionOfItsKeyAlone`, and the recompute checks
  `testDropoutMatchesTheReferenceOnArena` and
  `testSrForwardMatchesTheReferenceOnArena`
  (`UnitTestCalculateGradsConformance.c`).
- Shuffle: `testDataLoaderShuffleIsFisherYatesFromTheIdentityUnderItsKey`
  (`UnitTestDataLoader.c`), `testTrainingRunShufflesEachEpochUnderItsOwnKey`
  (`UnitTestTrainingLoopApi.c`).
- Resume: `testTrainingRngDeserializeRestoresWhatSerializeSaved`
  (`UnitTestTrainingLoopApi.c`),
  `testResumingAtAnEpochBoundaryContinuesTheRunBitForBit`
  (`UnitTestDropoutIntegration.c`).
- The Python mirror: `python/tests/test_keyed_rng.py` compiles a harness
  against `src/rng/RNG.c` and checks `fold`, `key_from_seed`, every stream
  function and `keyed_shuffle` against it (`test_mirror_matches_c_*`), and
  `TAGS` against the header's tag table (`test_tags_mirror_the_c_tag_table`,
  `test_tags_are_nonzero_and_unique`); `python/tests/test_keyed_sampler.py`
  checks `KeyedShuffleSampler` against the orders the C training-loop tests
  expect.
