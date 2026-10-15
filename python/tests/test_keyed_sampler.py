"""Test examples/_shared/keyed_sampler.py: the PyTorch twins' training order."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from examples._shared.keyed_rng import fold, key_from_seed, keyed_shuffle  # noqa: E402
from examples._shared.keyed_sampler import KeyedShuffleSampler  # noqa: E402


def test_reshuffling_sampler_yields_each_epochs_keyed_order():
    sampler = KeyedShuffleSampler(10, 42, reshuffle=True)
    root = key_from_seed(42)
    orders = [list(sampler) for _ in range(3)]
    assert orders == [keyed_shuffle(10, fold(root, 21, e)) for e in range(3)]
    assert len(sampler) == 10


def test_shuffle_once_sampler_repeats_epoch_zeros_order():
    sampler = KeyedShuffleSampler(10, 42, reshuffle=False)
    first = keyed_shuffle(10, fold(key_from_seed(42), 21, 0))
    assert [list(sampler) for _ in range(3)] == [first] * 3


def test_sampler_matches_the_c_training_loop_orders():
    """The orders the C trainingRun tests expect for 4 samples under run seed 42
    (UnitTestTrainingLoopApi.c, kTrainOrderGold): epochs 0, 1, 2 with reshuffle."""
    sampler = KeyedShuffleSampler(4, 42, reshuffle=True)
    assert [list(sampler) for _ in range(3)] == [[2, 1, 3, 0], [2, 0, 1, 3], [1, 3, 0, 2]]
