# Determinism Contract

Both PyTorch and C training programs in `examples/` must seed every
source of randomness from `examples/_shared/seeds.py` constants:

- `SEED = 42` — used for any deterministic init (`torch.manual_seed`, etc.)
  and, on both sides, for the training sample order (see "Shuffle ordering").
- `SPLIT_SEED = 42` — seeds the numpy train/validation split of every
  `prepare_data.py`; neither trainer draws with it.

## Init weights

Each side uses its own native init. PyTorch uses `torch.nn.init`
defaults (Kaiming uniform for `Conv1d`/`Linear` weights, uniform
`[-1/√fan_in, 1/√fan_in]` for biases). C uses the framework's native
parameter init via each layer factory's `initTensor` + `initWeightTensor`/
`initBiasTensor` (which route through `initDistribution`).
Init weights are NOT exchanged between implementations — that's the
design choice that keeps the parity story robust to known C-side init
drift.

## Shuffle ordering

The C data loader orders a `shuffle = true` loader with `dataLoaderShuffle`:
Fisher-Yates from the identity order over the stream of one key, so an order
is a function of its key alone and no other draw can change it. `trainingRun`
takes the keys from the trainer's `trainingRng_t` (seeded with `SEED`): it
shuffles the train loader at the start of every epoch, under that epoch's key
when the loader reshuffles per epoch (`dataLoaderSetReshufflePerEpoch`; the
HAR float32 trainer's `RESHUFFLE=1`, its default) and under epoch 0's key
otherwise, and a `shuffle = true` eval loader once per run under its own key.
`examples/_shared/keyed_rng.py` reproduces every order in Python:
`keyed_shuffle(n, fold(key_from_seed(SEED), 21, e))` is epoch `e`'s order of
an `n`-sample train loader (21 is the shuffle tag; `e = 0` in every epoch
without the per-epoch reshuffle). The PyTorch twins train in these orders:
`KeyedShuffleSampler(n, SEED, reshuffle)` (`examples/_shared/keyed_sampler.py`)
yields them from `SEED`, the seed the C trainers hand to `trainingRngInit` by
default, so both sides see the same samples in the same order.

## Parity assertion regime

Final-state only. After `N` epochs both sides should reach the same
test accuracy / loss within per-example tolerance tables in `compare.py`.
The framework's `parity.ParityCheck` either uses absolute or relative
tolerance, never both.

## Verifying the shuffle mirrors

`python/tests/test_keyed_rng.py::test_mirror_matches_c_keyed_shuffle`
compiles a C harness against `src/rng/RNG.c` at test time and checks
`keyed_shuffle` against the same Fisher-Yates loop on the keyed stream; the C
unit tests check `dataLoaderShuffle` and `trainingRun` against orders derived
with `keyed_shuffle`. The module-global generator's shuffle
(`rngSetSeed` + `rngShuffleIndices`, mirrored by `xorshift32.py`) is checked
the same way by `python/tests/test_xorshift32.py`.
