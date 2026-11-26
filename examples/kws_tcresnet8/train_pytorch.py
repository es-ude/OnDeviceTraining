"""PyTorch twin of the TC-ResNet8 keyword spotter (Choi et al. 2019).

Written from the paper and hyperconnect/TC-ResNet (audio_nets/tc_resnet.py), not from train_c.c;
only the run interface follows the C program. Input: MFCC [40, T] from
examples/kws_mfcc/prepare_data.py. Knobs (the C program's): KWS_DATA_DIR (required),
TCR_WIDTH full|half, TCR_SEED, TCR_MICROBATCH, TCR_EPOCHS, TCR_OUT. Output: TCR_OUT/pytorch.json
in the schema of c.json.

`--export-init DIR` writes the initial weights and BN buffers in C node order (the files the C
program's TCR_SAME_START mode loads) and exits; same_start.py drives the same-start check.
"""
from __future__ import annotations

import argparse
import math
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
from examples._shared.keyed_sampler import KeyedShuffleSampler  # noqa: E402
from examples._shared.log_schema import RunLog, dump_log  # noqa: E402
from examples._shared.seeds import SEED  # noqa: E402

IN_CHANNELS = 40
FULL_STEM = 16
FULL_BLOCK_OUT = (24, 32, 48)
BATCH = 100
BASE_LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-3
LR_GAMMA = 0.1
PAPER_ITERS_PER_DROP = 10_000
DROPOUT_P = 0.5
# TF's BN decay 0.997 is PyTorch momentum 0.003
BN_MOMENTUM = 0.003
BN_EPS = 1e-3


def tf_same_pad(length: int, kernel: int, stride: int) -> tuple[int, int]:
    """TF 'SAME' padding (left, right) for one conv: the output has ceil(length / stride) frames
    and the odd pad goes on the right."""
    total = max((math.ceil(length / stride) - 1) * stride + kernel - length, 0)
    return total // 2, total - total // 2


class SameConv1d(nn.Conv1d):
    """Bias-free Conv1d with TF 'SAME' padding; nn.Conv1d(padding='same') rejects stride > 1."""

    def __init__(self, cin: int, cout: int, kernel: int, stride: int) -> None:
        super().__init__(cin, cout, kernel, stride=stride, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        left, right = tf_same_pad(x.shape[-1], self.kernel_size[0], self.stride[0])
        return super().forward(F.pad(x, (left, right)))


def _bn(channels: int) -> nn.BatchNorm1d:
    return nn.BatchNorm1d(channels, eps=BN_EPS, momentum=BN_MOMENTUM)


class ResidualBlock(nn.Module):
    """main: conv k9 s2 -> BN -> ReLU -> conv k9 s1 -> BN; shortcut: conv k1 s2 -> BN -> ReLU;
    out = ReLU(main + shortcut)."""

    def __init__(self, cin: int, cout: int) -> None:
        super().__init__()
        self.conv_a = SameConv1d(cin, cout, 9, 2)
        self.bn_a = _bn(cout)
        self.conv_b = SameConv1d(cout, cout, 9, 1)
        self.bn_b = _bn(cout)
        self.conv_short = SameConv1d(cin, cout, 1, 2)
        self.bn_short = _bn(cout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        main = self.bn_b(self.conv_b(F.relu(self.bn_a(self.conv_a(x)))))
        short = F.relu(self.bn_short(self.conv_short(x)))
        return F.relu(main + short)


class TcResNet8(nn.Module):
    """Returns logits; the softmax lives in the loss."""

    def __init__(self, width: str, classes: int, dropout_p: float = DROPOUT_P) -> None:
        super().__init__()
        div = {"full": 1, "half": 2}[width]
        stem = FULL_STEM // div
        self.stem = nn.Sequential(SameConv1d(IN_CHANNELS, stem, 3, 1), _bn(stem), nn.ReLU())
        blocks, cin = [], stem
        for full_out in FULL_BLOCK_OUT:
            blocks.append(ResidualBlock(cin, full_out // div))
            cin = full_out // div
        self.blocks = nn.Sequential(*blocks)
        self.dropout = nn.Dropout(dropout_p)
        self.fc = nn.Linear(cin, classes, bias=False)
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.blocks(self.stem(x))
        return self.fc(self.dropout(h.mean(dim=-1)))  # global average pool over time

    def c_node_order(self) -> list[nn.Module]:
        """The parameter layers in the C graph's node order (modelLoadStateDict's order)."""
        order: list[nn.Module] = [self.stem[0], self.stem[1]]
        for b in self.blocks:
            order += [b.conv_a, b.bn_a, b.conv_b, b.bn_b, b.conv_short, b.bn_short]
        return order + [self.fc]


def export_init(model: TcResNet8, out_dir: Path) -> None:
    """p<i>.weight.npy (+ p<i>.bias.npy = BN beta) for param layer i, bn<j>.running_{mean,var}.npy
    for BN layer j, both counted in C node order."""
    out_dir.mkdir(parents=True, exist_ok=True)
    bn_index = 0
    for i, layer in enumerate(model.c_node_order()):
        np.save(out_dir / f"p{i:02d}.weight.npy", layer.weight.detach().numpy().astype(np.float32))
        if isinstance(layer, nn.BatchNorm1d):
            np.save(out_dir / f"p{i:02d}.bias.npy", layer.bias.detach().numpy().astype(np.float32))
            np.save(out_dir / f"bn{bn_index:02d}.running_mean.npy",
                    layer.running_mean.numpy().astype(np.float32))
            np.save(out_dir / f"bn{bn_index:02d}.running_var.npy",
                    layer.running_var.numpy().astype(np.float32))
            bn_index += 1


def make_optimizer(model: TcResNet8) -> torch.optim.SGD:
    """The paper's weight decay hits conv/fc weights only; BN scale and shift get none."""
    bn_params = [p for m in model.modules() if isinstance(m, nn.BatchNorm1d) for p in m.parameters()]
    bn_ids = {id(p) for p in bn_params}
    decayed = [p for p in model.parameters() if id(p) not in bn_ids]
    return torch.optim.SGD(
        [{"params": decayed, "weight_decay": WEIGHT_DECAY}, {"params": bn_params, "weight_decay": 0.0}],
        lr=BASE_LR, momentum=MOMENTUM)


def paper_step_size_epochs(train_size: int) -> int:
    """Whole epochs closest to the paper's 10k iterations per LR drop; integer round-half-up,
    as the C program computes it (Python's round() is banker's)."""
    iters = train_size // BATCH
    return max(1, (PAPER_ITERS_PER_DROP + iters // 2) // iters)


def train_step(model: TcResNet8, optimizer: torch.optim.Optimizer, xb: torch.Tensor,
               yb: torch.Tensor, micro_batch: int) -> float:
    """One optimizer step over a batch, forward/backward in chunks of micro_batch rows (BN batch
    statistics per chunk, as the C side computes them); returns the batch-mean loss."""
    optimizer.zero_grad()
    total = 0.0
    for first in range(0, xb.shape[0], micro_batch):
        loss = F.cross_entropy(model(xb[first:first + micro_batch]), yb[first:first + micro_batch])
        (loss * (micro_batch / xb.shape[0])).backward()
        total += loss.item() * micro_batch
    optimizer.step()
    return total / xb.shape[0]


def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
    model.eval()
    total_loss, correct = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(x), BATCH):
            logits = model(x[i:i + BATCH])
            total_loss += F.cross_entropy(logits, y[i:i + BATCH], reduction="sum").item()
            correct += (logits.argmax(dim=1) == y[i:i + BATCH]).sum().item()
    return total_loss / len(x), correct / len(x)


def load_split(data_dir: Path, split: str) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.from_numpy(np.load(data_dir / f"{split}_x.npy").astype(np.float32, copy=False))
    y = torch.from_numpy(np.load(data_dir / f"{split}_y.npy").astype(np.int64))
    if x.ndim != 3 or x.shape[1] != IN_CHANNELS or len(x) != len(y):
        raise SystemExit(f"ERROR: {data_dir}/{split}: expected [N, {IN_CHANNELS}, T] items and N labels")
    return x, y


def train_loader(x: torch.Tensor, y: torch.Tensor, seed: int) -> torch.utils.data.DataLoader:
    """Batches in the C trainer's keyed order (reshuffled per epoch), last partial batch dropped."""
    ds = torch.utils.data.TensorDataset(x, y)
    return torch.utils.data.DataLoader(ds, batch_size=BATCH, drop_last=True,
                                       sampler=KeyedShuffleSampler(len(ds), seed, reshuffle=True))


def same_start_model(width: str, classes: int, seed: int,
                     dtype: torch.dtype = torch.float32) -> tuple[TcResNet8, torch.optim.SGD]:
    """The seeded start every same-start run shares, Dropout off. Built and initialised in float32
    and only then cast, so a float64 twin starts from exactly the values export_init writes."""
    torch.manual_seed(seed)
    model = TcResNet8(width, classes, dropout_p=0.0).to(dtype)
    return model, make_optimizer(model)


def same_start_train(model: TcResNet8, optimizer: torch.optim.Optimizer, x: torch.Tensor,
                     y: torch.Tensor, seed: int, steps: int) -> np.ndarray:
    """The first `steps` steps of epoch 0 at micro-batch = batch, in the C trainer's keyed order;
    returns each step's loss. The batches are cast to the model's dtype."""
    dtype = next(model.parameters()).dtype
    model.train()
    losses = []
    for xb, yb in train_loader(x, y, seed):
        if len(losses) == steps:
            break
        losses.append(train_step(model, optimizer, xb.to(dtype), yb, BATCH))
    if len(losses) < steps:
        raise SystemExit(f"ERROR: epoch 0 holds only {len(losses)} of the {steps} same-start steps")
    return np.asarray(losses, dtype=np.float64)


def training_state(model: TcResNet8, optimizer: torch.optim.Optimizer) -> dict[str, list[np.ndarray]]:
    """The full training state as float64 arrays, in the order the C program's TCR_SAME_START
    mode writes it: `param` and `momentum` per parameter tensor in C node order (BN: scale, then
    shift), `running_mean` and `running_var` per BN layer. Call after at least one step (SGD
    creates its momentum buffers on the first step)."""
    state: dict[str, list[np.ndarray]] = {"param": [], "momentum": [], "running_mean": [],
                                          "running_var": []}
    for layer in model.c_node_order():
        params = [layer.weight] + ([layer.bias] if isinstance(layer, nn.BatchNorm1d) else [])
        for p in params:
            state["param"].append(p.detach().double().numpy().ravel())
            state["momentum"].append(optimizer.state[p]["momentum_buffer"].double().numpy().ravel())
        if isinstance(layer, nn.BatchNorm1d):
            state["running_mean"].append(layer.running_mean.double().numpy())
            state["running_var"].append(layer.running_var.double().numpy())
    return state


def eval_probs(model: TcResNet8, x: torch.Tensor) -> np.ndarray:
    """Eval-mode softmax (running BN statistics, no Dropout), as float64 [N, classes]."""
    model.eval()
    with torch.no_grad():
        logits = model(x.to(next(model.parameters()).dtype))
    return torch.softmax(logits, dim=1).double().numpy()


def _env_int(name: str, default: int, lo: int, hi: int) -> int:
    v = os.environ.get(name, "")
    if v == "":
        return default
    if not v.isdigit() or not lo <= int(v) <= hi:
        raise SystemExit(f"ERROR: {name}={v}: expected a whole number in [{lo}, {hi}]")
    return int(v)


def read_config() -> dict:
    data_dir = os.environ.get("KWS_DATA_DIR", "")
    if data_dir == "":
        raise SystemExit("ERROR: KWS_DATA_DIR is required (a directory with {train,val,test}_{x,y}.npy)")
    width = os.environ.get("TCR_WIDTH", "") or "full"
    if width not in ("full", "half"):
        raise SystemExit(f"ERROR: TCR_WIDTH={width}: expected full or half")
    micro = _env_int("TCR_MICROBATCH", BATCH, 1, BATCH)
    if BATCH % micro != 0:
        raise SystemExit(f"ERROR: TCR_MICROBATCH={micro}: the micro-batch must divide the batch of 100")
    return {
        "data_dir": Path(data_dir),
        "width": width,
        "seed": _env_int("TCR_SEED", SEED, 0, 2**32 - 1),
        "micro_batch": micro,
        "epochs_override": _env_int("TCR_EPOCHS", 0, 1, 100_000),
        "out_dir": Path(os.environ.get("TCR_OUT", "") or "examples/kws_tcresnet8/logs"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--export-init", type=Path, metavar="DIR",
                        help="write the initial weights and BN buffers in C node order, then exit")
    args = parser.parse_args()
    cfg = read_config()
    # One thread per run: the sweep runs several twins side by side, like the C binaries.
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.manual_seed(cfg["seed"])

    train_x, train_y = load_split(cfg["data_dir"], "train")
    if len(train_x) < BATCH:
        raise SystemExit(f"ERROR: train holds {len(train_x)} samples, fewer than one batch of {BATCH}")
    classes = int(train_y.max()) + 1
    model = TcResNet8(cfg["width"], classes)
    if args.export_init is not None:
        export_init(model, args.export_init)
        return
    val_x, val_y = load_split(cfg["data_dir"], "val")
    test_x, test_y = load_split(cfg["data_dir"], "test")
    num_params = sum(p.numel() for p in model.parameters())

    step_size = paper_step_size_epochs(len(train_x))
    epochs = cfg["epochs_override"] or 3 * step_size
    optimizer = make_optimizer(model)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=LR_GAMMA)
    loader = train_loader(train_x, train_y, cfg["seed"])
    print(f"data: train {len(train_x)}, val {len(val_x)}, test {len(test_x)}, "
          f"[{IN_CHANNELS}, {train_x.shape[2]}], {classes} classes; parameters: {num_params}; "
          f"epochs={epochs} step_size={step_size} micro_batch={cfg['micro_batch']} seed={cfg['seed']}",
          flush=True)

    records = []
    diverged = False
    run_t0 = time.time()
    for epoch in range(epochs):
        t0 = time.time()
        lr = optimizer.param_groups[0]["lr"]
        model.train()
        losses = [train_step(model, optimizer, xb, yb, cfg["micro_batch"]) for xb, yb in loader]
        train_loss = float(np.mean(losses))
        val_loss, val_acc = evaluate(model, val_x, val_y)
        records.append({
            "epoch": epoch, "step_losses": [], "train_loss": train_loss, "val_loss": val_loss,
            "val_acc": val_acc, "wall_s": time.time() - t0, "lr": lr, "batch_size": BATCH,
            "parameter_updates": len(losses),
        })
        print(f"epoch {epoch}: lr={lr:.6g} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"val_acc={val_acc:.4f} wall_s={time.time() - t0:.2f}", flush=True)
        if not (math.isfinite(train_loss) and math.isfinite(val_loss)):
            diverged = True
            break
        scheduler.step()

    test_loss, test_acc = evaluate(model, test_x, test_y)
    log: RunLog = {
        "impl": "pytorch",
        "example": "kws_tcresnet8",
        "config": {
            "epochs": epochs, "batch": BATCH, "lr": BASE_LR, "momentum": MOMENTUM,
            "seed": cfg["seed"], "lr_schedule": "step", "gamma": LR_GAMMA, "step_size": step_size,
            "optimizer": "sgd", "weight_decay": WEIGHT_DECAY, "reshuffle": 1,
            "width": cfg["width"], "micro_batch": cfg["micro_batch"], "num_params": num_params,
        },
        "epochs": records,  # type: ignore[typeddict-item]
        "final": {"test_loss": test_loss, "test_acc": test_acc, "test_auc": None,
                  "diverged": int(diverged), "epochs_completed": len(records)},
    }
    cfg["out_dir"].mkdir(parents=True, exist_ok=True)
    dump_log(cfg["out_dir"] / "pytorch.json", log)
    print(f"FINAL test_loss={test_loss:.4f} test_acc={test_acc:.4f} wall_s={time.time() - run_t0:.1f}",
          flush=True)


if __name__ == "__main__":
    main()
