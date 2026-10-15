"""Single source of truth for example seeds.

Both PyTorch and C training programs read SEED for any deterministic init;
the C trainers and the PyTorch twins also derive the sample order from it.
SPLIT_SEED seeds the numpy train/validation split of every prepare_data.py.
"""

SEED: int = 42
SPLIT_SEED: int = 42
