"""PyTorch reference implementation of the kws_mfcc 1D-CNN classifier.

Input: MFCC [40,32] from prepare_data.py. Output: logs/<n>class/pytorch.json +
outputs/<n>class/pytorch_predictions.npy + weights/<n>class/{conv1,conv2,fc}.{weight,bias}.npy
for the C-side BIT_PARITY mode. num_classes from KWS_CLASSES (default 6).
"""
from __future__ import annotations

import os
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
NUM_CLASSES = int(os.environ.get("KWS_CLASSES", "6"))
assert NUM_CLASSES in (6, 35), NUM_CLASSES
TAG = f"{NUM_CLASSES}class"
DATA = HERE / "data" / TAG
LOGS = HERE / "logs" / TAG
OUTPUTS = HERE / "outputs" / TAG
WEIGHTS = HERE / "weights" / TAG

EPOCHS = 15
BATCH = 32
LR = 0.001
MOMENTUM = 0.9


class KwsDataset(torch.utils.data.Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class XorShift32Sampler(torch.utils.data.Sampler[int]):
    """Single-shot shuffle, no per-epoch reshuffle, matching framework DataLoader.c."""
    def __init__(self, n: int, seed: int) -> None:
        self.indices = shuffle_indices(n, seed)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


class KwsMfccCnn(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(40, 32, kernel_size=3, padding=1)  # SAME (K odd, stride 1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))              # [B,32,32]
        x = F.max_pool1d(x, 2)                 # [B,32,16]
        x = F.relu(self.conv2(x))              # [B,64,16]
        x = F.max_pool1d(x, 2)                 # [B,64,8]
        x = F.adaptive_avg_pool1d(x, 1)        # [B,64,1]
        x = x.flatten(start_dim=1)             # [B,64]
        return self.fc(x)


def evaluate(model: nn.Module, x: np.ndarray, y: np.ndarray, batch: int) -> tuple[float, float]:
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    with torch.no_grad():
        for i in range(0, len(x), batch):
            xb = torch.from_numpy(x[i : i + batch].astype(np.float32))
            yb = torch.from_numpy(y[i : i + batch].astype(np.int64))
            logits = model(xb)
            loss = F.cross_entropy(logits, yb, reduction="sum")
            total_loss += loss.item()
            total_correct += (logits.argmax(dim=1) == yb).sum().item()
            total += yb.shape[0]
    return total_loss / total, total_correct / total


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)

    train_x = np.load(DATA / "train_x.npy")
    train_y = np.load(DATA / "train_y.npy")
    val_x = np.load(DATA / "val_x.npy")
    val_y = np.load(DATA / "val_y.npy")
    test_x = np.load(DATA / "test_x.npy")
    test_y = np.load(DATA / "test_y.npy")

    train_ds = KwsDataset(train_x, train_y)
    sampler = XorShift32Sampler(len(train_ds), SHUFFLE_SEED)
    loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH, sampler=sampler, drop_last=True)

    model = KwsMfccCnn(NUM_CLASSES)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)

    epoch_records = []
    for epoch in range(EPOCHS):
        t0 = time.time()
        model.train()
        step_losses: list[float] = []
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            optimizer.step()
            step_losses.append(loss.item())
        train_loss = float(np.mean(step_losses)) if step_losses else 0.0
        val_loss, val_acc = evaluate(model, val_x, val_y, BATCH)
        epoch_records.append({
            "epoch": epoch, "step_losses": step_losses, "train_loss": train_loss,
            "val_loss": val_loss, "val_acc": val_acc, "wall_s": time.time() - t0,
        })
        print(f"epoch {epoch:2d}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}", flush=True)

    test_loss, test_acc = evaluate(model, test_x, test_y, BATCH)
    log: RunLog = {
        "impl": "pytorch", "example": "kws_mfcc",
        "config": {"epochs": EPOCHS, "batch": BATCH, "lr": LR, "momentum": MOMENTUM,
                   "seed": SEED, "shuffle_seed": SHUFFLE_SEED},
        "epochs": epoch_records,  # type: ignore[typeddict-item]
        "final": {"test_loss": test_loss, "test_acc": test_acc, "test_auc": None},
    }
    LOGS.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    dump_log(LOGS / "pytorch.json", log)

    model.eval()
    with torch.no_grad():
        preds = model(torch.from_numpy(test_x.astype(np.float32))).argmax(dim=1).numpy().astype(np.int32)
    np.save(OUTPUTS / "pytorch_predictions.npy", preds)
    print(f"FINAL test_loss={test_loss:.4f} test_acc={test_acc:.4f}", flush=True)

    WEIGHTS.mkdir(parents=True, exist_ok=True)
    layer_map = {"conv1": model.conv1, "conv2": model.conv2, "fc": model.fc}
    print("Saving per-layer weights:", flush=True)
    for name, layer in layer_map.items():
        w = layer.weight.detach().cpu().numpy().astype(np.float32)
        np.save(WEIGHTS / f"{name}.weight.npy", w)
        if layer.bias is not None:
            b = layer.bias.detach().cpu().numpy().astype(np.float32)
            np.save(WEIGHTS / f"{name}.bias.npy", b)
        print(f"  wrote {name}.weight.npy shape={w.shape}", flush=True)


if __name__ == "__main__":
    main()
