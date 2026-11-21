"""PyTorch twin of the residual HAR classifier (one skip around conv2-conv3).

Input: train/val/test .npy files produced by prepare_data.py.
Output: logs/pytorch_residual.json + outputs/pytorch_residual_predictions.npy
+ weights_residual/<name>.{weight,bias}.npy for the C side's BIT_PARITY mode.
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
from examples.har_classifier.train_pytorch import (  # noqa: E402
    BATCH,
    DATA,
    EPOCHS,
    HERE,
    LOGS,
    LR,
    MOMENTUM,
    NUM_CLASSES,
    OUTPUTS,
    RESHUFFLE,
    SEED,
    HarDataset,
    KeyedShuffleSampler,
    evaluate,
)


class HarResidualModel(nn.Module):
    """conv2 and conv3 share one shape on purpose: a swapped weight order fails on values."""

    def __init__(self) -> None:
        super().__init__()
        # Construction order fixes the random init and the state_dict order; it is the C node order.
        self.conv1 = nn.Conv1d(9, 16, kernel_size=7, padding=3)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(16, 16, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv1d(16, 16, kernel_size=5, padding=2)
        self.relu3 = nn.ReLU()
        self.gap = nn.AvgPool1d(64)  # global avg pool over the remaining length
        self.flatten = nn.Flatten(start_dim=1)
        self.fc = nn.Linear(16, NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.relu1(self.conv1(x)))
        y = self.conv3(self.relu2(self.conv2(x)))
        x = self.relu3(x + y)
        return self.fc(self.flatten(self.gap(x)))  # logits


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
    sampler = KeyedShuffleSampler(len(train_ds), SEED, reshuffle=RESHUFFLE != 0)
    loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH, sampler=sampler, drop_last=True)

    model = HarResidualModel()
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
            "epoch": epoch,
            "step_losses": step_losses,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "wall_s": time.time() - t0,
            "lr": LR,
        })
        print(f"epoch {epoch:2d}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}", flush=True)

    test_loss, test_acc = evaluate(model, test_x, test_y, BATCH)
    log: RunLog = {
        "impl": "pytorch",
        "example": "har_classifier_residual",
        "config": {  # type: ignore[typeddict-item]
            "epochs": EPOCHS, "batch": BATCH, "lr": LR, "momentum": MOMENTUM,
            "seed": SEED, "lr_schedule": "none", "optimizer": "sgd",
            "reshuffle": int(RESHUFFLE != 0),
        },
        "epochs": epoch_records,  # type: ignore[typeddict-item]
        "final": {"test_loss": test_loss, "test_acc": test_acc, "test_auc": None},
    }
    LOGS.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    dump_log(LOGS / "pytorch_residual.json", log)

    model.eval()
    with torch.no_grad():
        preds = model(torch.from_numpy(test_x.astype(np.float32))).argmax(dim=1).numpy().astype(np.int32)
    np.save(OUTPUTS / "pytorch_residual_predictions.npy", preds)
    print(f"FINAL test_loss={test_loss:.4f} test_acc={test_acc:.4f}", flush=True)

    # The C side loads these in its node order, so the list is explicit, not state_dict order.
    weights_dir = HERE / "weights_residual"
    weights_dir.mkdir(parents=True, exist_ok=True)
    for name, layer in [("conv1", model.conv1), ("conv2", model.conv2),
                        ("conv3", model.conv3), ("fc", model.fc)]:
        np.save(weights_dir / f"{name}.weight.npy", layer.weight.detach().numpy().astype(np.float32))
        np.save(weights_dir / f"{name}.bias.npy", layer.bias.detach().numpy().astype(np.float32))


if __name__ == "__main__":
    main()
