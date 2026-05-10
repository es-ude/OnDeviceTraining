"""PyTorch reference implementation of the ECG5000 reconstruction AE.

Input: train/val/test .npy files produced by prepare_data.py (no labels
used for training; test labels used only at evaluation time, downstream).
Output:
  logs/pytorch.json
  outputs/pytorch_reconstructions.npy        [N_test,         1, 140]
  outputs/pytorch_train_recons.npy           [N_train_normal, 1, 140]

train_recons is needed by compare.py to derive the anomaly threshold
(mean + 3 sigma on training-set normals). test_reconstructions are needed
for both the anomaly score histogram and per-sample MSE.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from examples._shared.log_schema import RunLog, dump_log  # noqa: E402
from examples._shared.seeds import SEED, SHUFFLE_SEED  # noqa: E402
from examples._shared.xorshift32 import shuffle_indices  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
LOGS = HERE / "logs"
OUTPUTS = HERE / "outputs"

EPOCHS = 200
BATCH = 32
LR = 0.005
MOMENTUM = 0.9


class EcgDataset(torch.utils.data.Dataset):
    """Self-supervised AE dataset: target == input (no label argument)."""

    def __init__(self, x: np.ndarray) -> None:
        self.x = torch.from_numpy(x.astype(np.float32))

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.x[idx]


class XorShift32Sampler(torch.utils.data.Sampler[int]):
    """Single-shot shuffle, no per-epoch reshuffle, matches DataLoader.c."""

    def __init__(self, n: int, seed: int) -> None:
        self.indices = shuffle_indices(n, seed)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


class EcgAutoencoder(nn.Module):
    """ECG5000 reconstruction AE.

    Decoder uses kernel_size=2 (not the spec section 4.2 K=4 with PyTorch
    padding=1) because our C framework's Conv1dTransposed has no
    integer-padding parameter; the only kernel/stride combination
    that hits length 70 from 35 (and 140 from 70) without input-side
    padding is K=2, S=2, op=0. PyTorch matches the same K=2 layout
    here for parity. Receptive field is smaller; AE still trains.
    """

    def __init__(self) -> None:
        super().__init__()
        # Encoder
        self.enc1 = nn.Conv1d(1, 8, kernel_size=7, stride=2, padding=3)   # [8, 70]
        # MaxPool1d(K=2, S=2)                                             # [8, 35]
        self.enc2 = nn.Conv1d(8, 16, kernel_size=5, padding=2)            # [16, 35]
        # AvgPool1d(K=5, S=5)                                             # [16, 7]
        # Decoder (K=2, S=2 substitution for the K=4-pad=1 layers)
        self.dec1 = nn.ConvTranspose1d(16, 8, kernel_size=5, stride=5)    # [8, 35]
        self.dec2 = nn.ConvTranspose1d(8, 4, kernel_size=2, stride=2)     # [4, 70]
        self.dec3 = nn.ConvTranspose1d(4, 1, kernel_size=2, stride=2)     # [1, 140]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.enc1(x))
        x = F.max_pool1d(x, kernel_size=2, stride=2)
        x = F.relu(self.enc2(x))
        x = F.avg_pool1d(x, kernel_size=5, stride=5)
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        return self.dec3(x)


def evaluate_mse(model: nn.Module, x: np.ndarray, batch: int) -> float:
    """Mean-per-element MSE over the dataset (matches MSE-mean reduction in C)."""
    model.eval()
    total_loss = 0.0
    total_elems = 0
    with torch.no_grad():
        for i in range(0, len(x), batch):
            xb = torch.from_numpy(x[i : i + batch].astype(np.float32))
            recon = model(xb)
            total_loss += F.mse_loss(recon, xb, reduction="sum").item()
            total_elems += xb.numel()
    return total_loss / total_elems


def reconstruct_all(model: nn.Module, x: np.ndarray, batch: int) -> np.ndarray:
    model.eval()
    recons = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            xb = torch.from_numpy(x[i : i + batch].astype(np.float32))
            recons.append(model(xb).numpy())
    return np.concatenate(recons, axis=0)


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)

    train_x = np.load(DATA / "train_x.npy")
    val_x = np.load(DATA / "val_x.npy")
    test_x = np.load(DATA / "test_x.npy")

    train_ds = EcgDataset(train_x)
    sampler = XorShift32Sampler(len(train_ds), SHUFFLE_SEED)
    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH, sampler=sampler, drop_last=True,
    )

    model = EcgAutoencoder()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)

    epoch_records = []
    for epoch in range(EPOCHS):
        t0 = time.time()
        model.train()
        step_losses: list[float] = []
        for xb in loader:
            optimizer.zero_grad()
            recon = model(xb)
            loss = F.mse_loss(recon, xb)
            loss.backward()
            optimizer.step()
            step_losses.append(loss.item())
        train_loss = float(np.mean(step_losses)) if step_losses else 0.0
        val_loss = evaluate_mse(model, val_x, BATCH)
        epoch_records.append({
            "epoch": epoch,
            "step_losses": step_losses,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": None,
            "wall_s": time.time() - t0,
        })
        print(
            f"epoch {epoch:2d}: train_loss={train_loss:.6f} val_loss={val_loss:.6f}",
            flush=True,
        )

    test_loss = evaluate_mse(model, test_x, BATCH)
    log: RunLog = {
        "impl": "pytorch",
        "example": "ecg_anomaly_ae",
        "config": {
            "epochs": EPOCHS, "batch": BATCH, "lr": LR, "momentum": MOMENTUM,
            "seed": SEED, "shuffle_seed": SHUFFLE_SEED,
        },
        "epochs": epoch_records,  # type: ignore[typeddict-item]
        "final": {"test_loss": test_loss, "test_acc": None, "test_auc": None},
    }
    LOGS.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    dump_log(LOGS / "pytorch.json", log)

    pt_test_recons = reconstruct_all(model, test_x, BATCH)
    pt_train_recons = reconstruct_all(model, train_x, BATCH)
    np.save(OUTPUTS / "pytorch_reconstructions.npy", pt_test_recons.astype(np.float32))
    np.save(OUTPUTS / "pytorch_train_recons.npy", pt_train_recons.astype(np.float32))
    print(f"FINAL test_loss={test_loss:.6f}", flush=True)


if __name__ == "__main__":
    main()
