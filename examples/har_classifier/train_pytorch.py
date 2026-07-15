"""PyTorch reference implementation of the HAR classifier.

Input: train/val/test .npy files produced by prepare_data.py.
Output: logs/pytorch.json + outputs/pytorch_predictions.npy
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

EPOCHS = 20
BATCH = 64
LR = 0.01
MOMENTUM = 0.9
NUM_CLASSES = 6

# Optimizer mirror of the C binaries (#328): "sgd" -> train_c.c (LR=0.01,
# MOMENTUM), "adamw" -> train_c_adamw.c (set LR = 0.001 to mirror its
# default; MOMENTUM is ignored). foreach=False pins torch's single-tensor
# kernel sequence, the one the C optimizer is bit-modeled on.
OPTIMIZER: str = "sgd"
WEIGHT_DECAY = 0.01

# LR schedule mirror of the C SYM binary's LR_SCHEDULE/LR_MIN env knobs
# (#327). None = constant LR (the FLOAT32 parity baseline stays constant-LR).
SCHEDULER: str | None = None
LR_MIN = 0.0


class HarDataset(torch.utils.data.Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))  # CrossEntropy wants int64 in PyTorch

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


class HarModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(9, 16, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.fc = nn.Linear(64, NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.avg_pool1d(x, kernel_size=32)  # global avg pool over remaining length
        x = x.flatten(start_dim=1)
        return self.fc(x)  # logits, CrossEntropyLoss applies log_softmax internally


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

    train_ds = HarDataset(train_x, train_y)
    sampler = XorShift32Sampler(len(train_ds), SHUFFLE_SEED)
    loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH, sampler=sampler, drop_last=True)

    model = HarModel()
    if OPTIMIZER not in ("sgd", "adamw"):
        raise ValueError(f"OPTIMIZER={OPTIMIZER!r} not supported ('sgd' or 'adamw')")
    optimizer = (
        torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
        if OPTIMIZER == "sgd"
        else torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-8,
                               weight_decay=WEIGHT_DECAY, foreach=False)
    )
    if SCHEDULER not in (None, "cosine"):
        raise ValueError(f"SCHEDULER={SCHEDULER!r} not supported (None or 'cosine')")
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR_MIN)
        if SCHEDULER == "cosine"
        else None
    )

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
        epoch_lr = float(optimizer.param_groups[0]["lr"])
        val_loss, val_acc = evaluate(model, val_x, val_y, BATCH)
        epoch_records.append({
            "epoch": epoch,
            "step_losses": step_losses,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "wall_s": time.time() - t0,
            "lr": epoch_lr,
        })
        print(f"epoch {epoch:2d}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}", flush=True)
        if scheduler is not None:
            scheduler.step()

    test_loss, test_acc = evaluate(model, test_x, test_y, BATCH)
    config = {
        "epochs": EPOCHS, "batch": BATCH, "lr": LR, "momentum": MOMENTUM,
        "seed": SEED, "shuffle_seed": SHUFFLE_SEED,
        "lr_schedule": SCHEDULER or "none", "lr_min": LR_MIN,
        "optimizer": OPTIMIZER,
    }
    if OPTIMIZER == "adamw":
        config["weight_decay"] = WEIGHT_DECAY
    log: RunLog = {
        "impl": "pytorch",
        "example": "har_classifier",
        "config": config,  # type: ignore[typeddict-item]
        "epochs": epoch_records,  # type: ignore[typeddict-item]
        "final": {"test_loss": test_loss, "test_acc": test_acc, "test_auc": None},
    }
    LOGS.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    dump_log(LOGS / "pytorch.json", log)

    # Predictions
    model.eval()
    with torch.no_grad():
        preds = model(torch.from_numpy(test_x.astype(np.float32))).argmax(dim=1).numpy().astype(np.int32)
    np.save(OUTPUTS / "pytorch_predictions.npy", preds)
    print(f"FINAL test_loss={test_loss:.4f} test_acc={test_acc:.4f}", flush=True)

    # Save per-layer weights for the C-side BIT_PARITY mode.
    # C-side expects: examples/har_classifier/weights/<name>.{weight,bias}.npy
    # Where <name> in {conv1, conv2, conv3, fc} matches the order in v2's buildModel.
    import os

    weights_dir = HERE / "weights"
    os.makedirs(weights_dir, exist_ok=True)

    layer_map = {
        "conv1": model.conv1,
        "conv2": model.conv2,
        "conv3": model.conv3,
        "fc": model.fc,
    }

    print("Saving per-layer weights:", flush=True)
    for name, layer in layer_map.items():
        w = layer.weight.detach().cpu().numpy().astype(np.float32)
        np.save(weights_dir / f"{name}.weight.npy", w)
        if layer.bias is not None:
            b = layer.bias.detach().cpu().numpy().astype(np.float32)
            np.save(weights_dir / f"{name}.bias.npy", b)
        has_bias = f" + {name}.bias.npy" if layer.bias is not None else ""
        print(f"  wrote {name}.weight.npy shape={w.shape}{has_bias}", flush=True)


if __name__ == "__main__":
    main()
