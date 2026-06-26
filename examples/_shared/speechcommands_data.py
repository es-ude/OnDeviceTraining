"""Shared SpeechCommands loader for the kws_mfcc and kws_raw examples.

Wraps torchaudio.datasets.SPEECHCOMMANDS (v0.02) so both KWS examples download
the ~2.3 GB corpus once into a shared raw root and deliver identical waveform
arrays. Output is the native 16 kHz mono waveform (float32 in [-1, 1], the range
torchaudio yields from the int16 PCM), pad/truncated to exactly 16000 samples.
Feature extraction (MFCC) and downsampling are the model's job, not the loader's,
per the repo's data-shape convention.

    load_speechcommands(root, num_classes) -> dict
        num_classes in {6, 35}
        returns {"train": (x, y), "val": (x, y), "test": (x, y)}
            x: float32 [N, 1, 16000]
            y: int32   [N]  (0..num_classes-1)

6-class config (labels 0..5, fixed order):
    0 yes  1 no  2 up  3 down
    4 silence  -- synthetic low-amplitude Gaussian noise (fixed per-split seed)
    5 unknown  -- random clips drawn from the other 31 keywords (fixed per-split seed)
35-class config (labels 0..34): the 35 natural keywords, alphabetical. No synthetic classes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torchaudio.datasets import SPEECHCOMMANDS

SAMPLE_RATE = 16000
CLIP_LEN = 16000  # 1 s
KEYWORDS_6 = ["yes", "no", "up", "down"]
SILENCE_STD = 0.05
SHUFFLE_SEED = 42  # mirrors examples/_shared/seeds.py; kept local to avoid an import cycle
_SUBSETS = {"train": "training", "val": "validation", "test": "testing"}


def _fix_length(wav: np.ndarray) -> np.ndarray:
    """Pad with zeros / truncate a mono waveform to exactly CLIP_LEN samples."""
    n = wav.shape[0]
    if n == CLIP_LEN:
        return wav
    if n > CLIP_LEN:
        return wav[:CLIP_LEN]
    out = np.zeros(CLIP_LEN, dtype=np.float32)
    out[:n] = wav
    return out


def _collect_by_label(ds) -> dict[str, list[np.ndarray]]:
    """Group every clip in a subset by its label string, length-fixed float32."""
    by_label: dict[str, list[np.ndarray]] = {}
    for waveform, sample_rate, label, *_ in ds:
        assert sample_rate == SAMPLE_RATE, sample_rate
        wav = _fix_length(waveform.squeeze(0).numpy().astype(np.float32))
        by_label.setdefault(label, []).append(wav)
    return by_label


def _stack(clips: list[np.ndarray], label_id: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.stack(clips).astype(np.float32)[:, None, :]  # [N, 1, 16000]
    y = np.full((x.shape[0],), label_id, dtype=np.int32)
    return x, y


def _build_split_6(by_label, split_index: int) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for label_id, kw in enumerate(KEYWORDS_6):
        clips = by_label.get(kw, [])
        x, y = _stack(clips, label_id)
        xs.append(x)
        ys.append(y)
    n_per = int(round(np.mean([len(by_label.get(kw, [])) for kw in KEYWORDS_6])))

    rng = np.random.default_rng(SHUFFLE_SEED + split_index)
    # silence (label 4): synthetic low-amplitude Gaussian noise
    silence = rng.normal(0.0, SILENCE_STD, size=(n_per, CLIP_LEN)).astype(np.float32)
    silence = np.clip(silence, -1.0, 1.0)
    xs.append(silence[:, None, :])
    ys.append(np.full((n_per,), 4, dtype=np.int32))
    # unknown (label 5): random draw from the other 31 keywords in THIS split
    pool = [w for lab, clips in by_label.items() if lab not in KEYWORDS_6 for w in clips]
    idx = rng.choice(len(pool), size=min(n_per, len(pool)), replace=False)
    unknown = np.stack([pool[i] for i in idx]).astype(np.float32)
    xs.append(unknown[:, None, :])
    ys.append(np.full((unknown.shape[0],), 5, dtype=np.int32))

    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def _build_split_35(by_label, keywords_35) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for label_id, kw in enumerate(keywords_35):
        clips = by_label.get(kw, [])
        if not clips:
            continue
        x, y = _stack(clips, label_id)
        xs.append(x)
        ys.append(y)
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def load_speechcommands(root, num_classes: int) -> dict:
    assert num_classes in (6, 35), num_classes
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    grouped = {}
    for split, subset in _SUBSETS.items():
        ds = SPEECHCOMMANDS(root=str(root), download=True, subset=subset)
        grouped[split] = _collect_by_label(ds)

    if num_classes == 35:
        keywords_35 = sorted({lab for g in grouped.values() for lab in g})
        assert len(keywords_35) == 35, (len(keywords_35), keywords_35)

    out = {}
    for split_index, split in enumerate(("train", "val", "test")):
        if num_classes == 6:
            out[split] = _build_split_6(grouped[split], split_index)
        else:
            out[split] = _build_split_35(grouped[split], keywords_35)
    return out
