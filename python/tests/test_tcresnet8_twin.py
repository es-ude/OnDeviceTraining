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
