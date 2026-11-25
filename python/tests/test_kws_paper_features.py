"""Test the 12-class Speech Commands data for the TC-ResNet8 paper setup:
the paper MFCC features (40 x 98) and the 12-class label layout."""
import sys
from pathlib import Path

import numpy as np
import torch
from torchaudio.transforms import MFCC

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from examples._shared import speechcommands_data as scd  # noqa: E402
from examples.kws_mfcc.prepare_data import paper_mfcc  # noqa: E402


def test_paper_mfcc_shape():
    x = np.random.default_rng(0).uniform(-1, 1, size=(3, 1, 16000)).astype(np.float32)
    feats = paper_mfcc(x)
    assert feats.shape == (3, 40, 98)
    assert feats.dtype == np.float32


def test_paper_mfcc_matches_torchaudio_reference():
    x = np.random.default_rng(1).uniform(-0.5, 0.5, size=(2, 1, 16000)).astype(np.float32)
    reference = MFCC(
        sample_rate=16000,
        n_mfcc=40,
        melkwargs={
            "n_fft": 480,
            "win_length": 480,
            "hop_length": 160,
            "n_mels": 40,
            "center": False,
        },
    )
    with torch.no_grad():
        expected = reference(torch.from_numpy(x[1])).squeeze(0).numpy()
    assert np.allclose(paper_mfcc(x)[1], expected, rtol=1e-5, atol=1e-4)


PAPER_KEYWORDS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]


def test_split_12_labels(monkeypatch):
    """Keyword k (paper order) gets label k, silence 10, unknown 11. Each fake
    clip is filled with its keyword's paper index (unknown words: -1), so the
    label each clip lands on shows where it came from."""
    others = ["bed", "bird", "cat", "dog"]
    code = {kw: float(i) for i, kw in enumerate(PAPER_KEYWORDS)} | {kw: -1.0 for kw in others}
    monkeypatch.setattr(
        scd, "_decode", lambda paths: [np.full(16000, code[p.parent.name], np.float32) for p in paths]
    )
    paths_by_label = {kw: [Path(f"{kw}/{i}.wav") for i in range(3)] for kw in PAPER_KEYWORDS}
    paths_by_label.update({kw: [Path(f"{kw}/{i}.wav") for i in range(2)] for kw in others})

    x, y = scd._build_split_keywords(paths_by_label, 0, scd.KEYWORDS_12)

    assert x.shape == (36, 1, 16000) and y.dtype == np.int32
    assert sorted(set(y.tolist())) == list(range(12))
    for label_id in range(10):
        assert np.all(x[y == label_id] == float(label_id))
    silence = x[y == 10]
    assert silence.shape[0] == 3 and np.all(np.std(silence, axis=-1) > 0.0)
    unknown = x[y == 11]
    assert unknown.shape[0] == 3 and np.all(unknown == -1.0)
