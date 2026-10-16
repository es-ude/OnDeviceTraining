"""Training-order sampler of the PyTorch twins: the C trainers' keyed shuffle.

A C trainer's trainingRun orders epoch e of an n-sample train loader as
keyed_shuffle(n, fold(key_from_seed(seed), TAGS["SHUFFLE"], e)) when the loader
reshuffles per epoch, and with epoch 0's key in every epoch otherwise, where
seed is the seed it hands to trainingRngInit. A twin that passes the same seed
sees the same sample order. torch's DataLoader accepts any iterable with a
length as its sampler, so no torch import is needed.
"""
from __future__ import annotations

from collections.abc import Iterator

from examples._shared.keyed_rng import TAGS, fold, key_from_seed, keyed_shuffle


class KeyedShuffleSampler:
    """Each iteration yields the next epoch's order of indices 0 .. n-1."""

    def __init__(self, n: int, seed: int, reshuffle: bool) -> None:
        self.n = n
        self.root = key_from_seed(seed)
        self.reshuffle = reshuffle
        self.epochs_started = 0

    def __len__(self) -> int:
        return self.n

    def __iter__(self) -> Iterator[int]:
        epoch = self.epochs_started if self.reshuffle else 0
        self.epochs_started += 1
        return iter(keyed_shuffle(self.n, fold(self.root, TAGS["SHUFFLE"], epoch)))
