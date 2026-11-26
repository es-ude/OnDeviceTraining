"""The TC-ResNet8 PyTorch twin (examples/kws_tcresnet8/train_pytorch.py) has the C model's shape."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import pytest  # noqa: E402
import torch  # noqa: E402

from examples.kws_tcresnet8.train_pytorch import TcResNet8, tf_same_pad  # noqa: E402


def _num_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@pytest.mark.parametrize(
    ("width", "classes", "expected"),
    # the counts train_c_kws_tcresnet8 prints for data A (12 classes) and B (6 classes)
    [("full", 12, 65_168), ("full", 6, 64_880), ("half", 12, 17_080), ("half", 6, 16_936)],
)
def test_param_count_matches_paper_scale(width, classes, expected):
    assert _num_params(TcResNet8(width, classes)) == expected


@pytest.mark.parametrize(("frames", "lengths"), [(98, [49, 25, 13]), (32, [16, 8, 4])])
def test_tf_same_pad_lengths(frames, lengths):
    model = TcResNet8("half", 6).eval()
    h = model.stem(torch.zeros(2, 40, frames))
    assert h.shape[-1] == frames
    seen = []
    for block in model.blocks:
        h = block(h)
        seen.append(h.shape[-1])
    assert seen == lengths


def test_tf_same_pad_puts_extra_on_right():
    assert tf_same_pad(98, 9, 2) == (3, 4)



def _run_twin(tmp_path, monkeypatch, train_rows):
    """One epoch of the half-width twin on zero inputs with `train_rows` training rows."""
    import numpy as np

    from examples.kws_tcresnet8 import train_pytorch

    for split, rows in (("train", train_rows), ("val", 10), ("test", 10)):
        np.save(tmp_path / f"{split}_x.npy", np.zeros((rows, 40, 32), dtype=np.float32))
        np.save(tmp_path / f"{split}_y.npy", np.arange(rows, dtype=np.int32) % 6)
    for name in ("TCR_SEED", "TCR_MICROBATCH"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("KWS_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("TCR_WIDTH", "half")
    monkeypatch.setenv("TCR_EPOCHS", "1")
    monkeypatch.setenv("TCR_OUT", str(tmp_path / "out"))
    monkeypatch.setattr(sys, "argv", ["train_pytorch.py"])
    train_pytorch.main()


def test_a_train_split_below_one_batch_stops_with_the_c_message(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="train holds 99 samples, fewer than one batch of 100"):
        _run_twin(tmp_path, monkeypatch, 99)


def test_one_batch_of_train_rows_trains(tmp_path, monkeypatch):
    _run_twin(tmp_path, monkeypatch, 100)
    assert (tmp_path / "out" / "pytorch.json").exists()
