"""Python mirror of the framework's XorShift32 RNG (src/rng/RNG.c).

This module exists so PyTorch DataLoaders shuffle samples in the same order
as the C framework's DataLoader: one Fisher-Yates shuffle at init
(dataLoaderInit: rngSetSeed(shuffleSeed) + rngShuffleIndices) and — for
examples that opt in via dataLoaderSetReshufflePerEpoch — one more
Fisher-Yates on the CURRENT permutation at the start of every epoch > 0
(dataLoaderReshuffle: no reseed, no identity reset, continuing the global
stream). `shuffle_indices_with_state` + `reshuffle_indices` mirror that pair;
the mirror is bit-exact only while nothing else draws from the C global RNG
between the shuffles (true for the float32 HAR binary; see DETERMINISM.md).
"""
from __future__ import annotations

UINT32_MAX = 0xFFFFFFFF


def xorshift32_next(state: int) -> int:
    """One step of Marsaglia 32-bit XorShift with shifts (13, 17, 5).

    All XOR-shifts are masked to 32 bits because Python ints are
    arbitrary-precision; without the mask, drift appears within ~30
    iterations.
    """
    x = state
    x ^= (x << 13) & UINT32_MAX
    x ^= x >> 17
    x ^= (x << 5) & UINT32_MAX
    return x


def _bounded(state: int, bound: int) -> tuple[int, int]:
    """Rejection-sampled uniform in [0, bound), mirroring rngBounded.

    Returns (new_state, sample).
    """
    limit = UINT32_MAX - (UINT32_MAX % bound)
    while True:
        state = xorshift32_next(state)
        if state < limit:
            return state, state % bound


def shuffle_indices(n: int, seed: int) -> list[int]:
    """Fisher-Yates shuffle mirroring rngSetSeed + rngShuffleIndices.

    Returns a permutation of [0, n). For n < 2 returns the identity
    permutation (matches the C early-return).
    """
    return shuffle_indices_with_state(n, seed)[0]


def shuffle_indices_with_state(n: int, seed: int) -> tuple[list[int], int]:
    """As shuffle_indices, but also returns the RNG state after the shuffle.

    The seed aliasing (0 -> UINT32_MAX) is rngSetSeed's and happens BEFORE
    rngShuffleIndices' n < 2 early return, so the returned state is the
    aliased seed even for a tiny dataset.
    """
    state = seed if seed != 0 else UINT32_MAX
    indices = list(range(n))
    state = reshuffle_indices(indices, state)
    return indices, state


def reshuffle_indices(indices: list[int], state: int) -> int:
    """Fisher-Yates on the CURRENT list in place, continuing `state`; returns the new state.

    Mirrors dataLoaderReshuffle -> rngShuffleIndices: no reseed, no identity
    reset, n < 2 is a no-op (state unchanged).
    """
    n = len(indices)
    if n < 2:
        return state
    for i in range(n - 1, 0, -1):
        state, j = _bounded(state, i + 1)
        indices[i], indices[j] = indices[j], indices[i]
    return state
