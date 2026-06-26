"""Prepare SpeechCommands MFCC features for the kws_mfcc example.

For each clip: log-MFCC via torchaudio (n_mfcc=40, n_fft=400, hop=512, n_mels=40)
over the native 16 kHz waveform -> [40, 32] frames (T=32 exact, no trim).

Output (under examples/kws_mfcc/data/<n>class/, n = KWS_CLASSES in {6,35}, default 6):
  {train,val,test}_x.npy  [N,40,32] f32
  {train,val,test}_y.npy  [N] i32  (0..n-1)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
from torchaudio.transforms import MFCC

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from examples._shared.speechcommands_data import load_speechcommands  # noqa: E402

HERE = Path(__file__).resolve().parent
RAW_ROOT = REPO_ROOT / "examples" / "_shared" / "data" / "speech_commands"
N_MFCC = 40
T_FRAMES = 32


def _mfcc_features(x: np.ndarray) -> np.ndarray:
    """x: [N,1,16000] f32 waveform -> [N,40,32] f32 MFCC (frame axis fixed to 32)."""
    mfcc = MFCC(
        sample_rate=16000,
        n_mfcc=N_MFCC,
        melkwargs={"n_fft": 400, "hop_length": 512, "n_mels": N_MFCC},
    )
    feats = np.empty((x.shape[0], N_MFCC, T_FRAMES), dtype=np.float32)
    with torch.no_grad():
        for i in range(x.shape[0]):
            m = mfcc(torch.from_numpy(x[i]))  # [1,40,frames]
            m = m.squeeze(0).numpy().astype(np.float32)  # [40,frames]
            if m.shape[1] >= T_FRAMES:
                m = m[:, :T_FRAMES]
            else:
                pad = np.zeros((N_MFCC, T_FRAMES), dtype=np.float32)
                pad[:, : m.shape[1]] = m
                m = pad
            feats[i] = m
    return feats


def main() -> None:
    num_classes = int(os.environ.get("KWS_CLASSES", "6"))
    assert num_classes in (6, 35), num_classes
    data_dir = HERE / "data" / f"{num_classes}class"
    data_dir.mkdir(parents=True, exist_ok=True)

    splits = load_speechcommands(RAW_ROOT, num_classes)
    for split in ("train", "val", "test"):
        x_wav, y = splits[split]
        x = _mfcc_features(x_wav)
        np.save(data_dir / f"{split}_x.npy", x)
        np.save(data_dir / f"{split}_y.npy", y.astype(np.int32))
        print(f"{split}: x={x.shape} y={y.shape} classes={num_classes}", flush=True)


if __name__ == "__main__":
    main()
