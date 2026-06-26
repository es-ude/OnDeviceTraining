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

import wave
from pathlib import Path

import numpy as np
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


def _read_wav_int16(path) -> np.ndarray:
    """Read a 16 kHz mono 16-bit PCM .wav as float32 in [-1, 1] (stdlib only).

    torchaudio 2.11 (maintenance mode) routes its dataset decode through
    torchcodec, which needs a system FFmpeg. We sidestep that with the stdlib
    `wave` reader the spec blessed as the fallback: int16 PCM / 32768 reproduces
    exactly what torchaudio/torchcodec would yield from these clips.
    """
    with wave.open(str(path), "rb") as w:
        frames = w.readframes(w.getnframes())
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0


def _paths_by_label(ds) -> dict[str, list[Path]]:
    """Map each label string to its list of absolute .wav paths for a subset.

    Uses ds.get_metadata (which does NOT decode audio, so no torchcodec / FFmpeg
    dependency); the metadata path is relative to ds._archive (pinned to
    torchaudio 2.11's SPEECHCOMMANDS layout). Returning paths instead of decoded
    waveforms lets the 6-class build decode only the clips it keeps, bounding
    peak memory (the CI runner has ~7 GB; decoding all 35 words would exceed it).
    """
    by_label: dict[str, list[Path]] = {}
    archive = Path(ds._archive)
    for i in range(len(ds)):
        relpath, sample_rate, label, *_ = ds.get_metadata(i)
        assert sample_rate == SAMPLE_RATE, sample_rate
        by_label.setdefault(label, []).append(archive / relpath)
    return by_label


def _decode(paths: list[Path]) -> list[np.ndarray]:
    """Decode + length-fix a list of .wav paths to float32 [16000] waveforms."""
    return [_fix_length(_read_wav_int16(p)) for p in paths]


def _stack(clips: list[np.ndarray], label_id: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.stack(clips).astype(np.float32)[:, None, :]  # [N, 1, 16000]
    y = np.full((x.shape[0],), label_id, dtype=np.int32)
    return x, y


def _build_split_6(paths_by_label, split_index: int) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for label_id, kw in enumerate(KEYWORDS_6):
        x, y = _stack(_decode(paths_by_label.get(kw, [])), label_id)
        xs.append(x)
        ys.append(y)
    n_per = int(round(np.mean([len(paths_by_label.get(kw, [])) for kw in KEYWORDS_6])))

    rng = np.random.default_rng(SHUFFLE_SEED + split_index)
    # silence (label 4): synthetic low-amplitude Gaussian noise
    silence = rng.normal(0.0, SILENCE_STD, size=(n_per, CLIP_LEN)).astype(np.float32)
    silence = np.clip(silence, -1.0, 1.0)
    xs.append(silence[:, None, :])
    ys.append(np.full((n_per,), 4, dtype=np.int32))
    # unknown (label 5): random draw of paths from the other 31 keywords in THIS
    # split, decoding only the selected clips (memory-bounded).
    pool = [p for lab, ps in paths_by_label.items() if lab not in KEYWORDS_6 for p in ps]
    idx = rng.choice(len(pool), size=min(n_per, len(pool)), replace=False)
    unknown = np.stack(_decode([pool[i] for i in idx])).astype(np.float32)
    xs.append(unknown[:, None, :])
    ys.append(np.full((unknown.shape[0],), 5, dtype=np.int32))

    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def _build_split_35(paths_by_label, keywords_35) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for label_id, kw in enumerate(keywords_35):
        paths = paths_by_label.get(kw, [])
        if not paths:
            continue
        x, y = _stack(_decode(paths), label_id)
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
        grouped[split] = _paths_by_label(ds)

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
