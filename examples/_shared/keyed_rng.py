"""Python mirror of the framework's keyed random-number core (src/rng/RNG.c).

Keys form a tree: key_from_seed(seed) is the root and fold(parent, tag, index)
one edge (one Threefry-2x32-20 block). Every random value comes from the stream
that open_stream(key) starts, so it depends on its key's path only, never on
draw order. Backend v1 is today's xorshift32 on a keyed state. A stream is its
state (an int); each draw returns (new_state, value), like xorshift32._bounded.
python/tests/test_keyed_rng.py checks every function against the C code
through a compiled harness.
"""
from __future__ import annotations

from examples._shared.xorshift32 import UINT32_MAX, xorshift32_next

UINT64_MAX = 0xFFFFFFFFFFFFFFFF

# Threefry-2x32 rotation constants; round i uses entry i % 8.
_ROTATIONS = (13, 15, 26, 6, 17, 29, 16, 24)
# Threefry's key-schedule parity constant.
_PARITY = 0x1BD11BDA


def _word(name: str, value: int) -> int:
    """C would truncate a value outside 32 bits silently; the mirror refuses it."""
    if not 0 <= value <= UINT32_MAX:
        raise ValueError(f"{name} must be in [0, UINT32_MAX], got {value}")
    return value


def _state(state: int) -> int:
    if not 1 <= state <= UINT32_MAX:
        raise ValueError(f"a stream state is in [1, UINT32_MAX], got {state}")
    return state


def _rotl32(x: int, r: int) -> int:
    return ((x << r) | (x >> (32 - r))) & UINT32_MAX


def fold(parent: tuple[int, int], tag: int, index: int) -> tuple[int, int]:
    """One edge of the key tree, mirroring rngFold: one Threefry-2x32-20 block
    with the parent as the key and (tag, index) as the counter."""
    k0, k1 = _word("parent[0]", parent[0]), _word("parent[1]", parent[1])
    ks = (k0, k1, k0 ^ k1 ^ _PARITY)
    x0 = (_word("tag", tag) + ks[0]) & UINT32_MAX
    x1 = (_word("index", index) + ks[1]) & UINT32_MAX
    for step in range(20):
        x0 = (x0 + x1) & UINT32_MAX
        x1 = _rotl32(x1, _ROTATIONS[step % 8]) ^ x0
        if step % 4 == 3:
            injection = (step + 1) // 4
            x0 = (x0 + ks[injection % 3]) & UINT32_MAX
            x1 = (x1 + ks[(injection + 1) % 3] + injection) & UINT32_MAX
    return x0, x1


def key_from_seed(seed: int) -> tuple[int, int]:
    """The root key, mirroring rngKeyFromSeed: the fold of (low word, high word)
    under the all-zero key. Seed 0 is an ordinary seed."""
    if not 0 <= seed <= UINT64_MAX:
        raise ValueError(f"seed must be in [0, 2**64 - 1], got {seed}")
    return fold((0, 0), seed & UINT32_MAX, seed >> 32)


def open_stream(key: tuple[int, int]) -> int:
    """The v1 state, mirroring rngOpen: w[0], else w[1], else 1. Not the seed
    rule of xorshift32.py (0 -> UINT32_MAX)."""
    w0, w1 = _word("key[0]", key[0]), _word("key[1]", key[1])
    return w0 if w0 else (w1 if w1 else 1)


def next_u32(state: int) -> tuple[int, int]:
    """Mirrors rngNextU32: one xorshift32 step; the draw is the new state."""
    state = xorshift32_next(_state(state))
    return state, state


def next_unit_float(state: int) -> tuple[int, float]:
    """Mirrors rngNextUnitFloat: (x >> 8) / 2**24 is exact in a double and
    equals the C float."""
    state = xorshift32_next(_state(state))
    return state, (state >> 8) / 2**24


def next_bounded(state: int, bound: int) -> tuple[int, int]:
    """Mirrors rngNextBounded: today's rejection rule, its >= included."""
    if bound == 0:
        raise ValueError("rngNextBounded: bound is 0")
    if bound > UINT32_MAX:
        raise ValueError("rngNextBounded: bound exceeds UINT32_MAX")
    if bound < 0:
        raise ValueError(f"bound must be in [1, UINT32_MAX], got {bound}")
    reject = UINT32_MAX - (UINT32_MAX % bound)
    state = _state(state)
    while True:
        state = xorshift32_next(state)
        if state < reject:
            return state, state % bound
