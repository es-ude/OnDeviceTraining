"""Prepare raw SpeechCommands waveforms for the kws_raw example.

Writes the native 16 kHz waveform directly — no resampling, no feature
extraction. Downsampling (16 kHz → 1 kHz via AvgPool1d) is the model's first
layer, so PyTorch and C read identical raw .npy.

Output (under examples/kws_raw/data/<n>class/, n = KWS_CLASSES in {6,35}, default 6):
  {train,val,test}_x.npy  [N,1,16000] f32
  {train,val,test}_y.npy  [N] i32  (0..n-1)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from examples._shared.speechcommands_data import load_speechcommands  # noqa: E402

HERE = Path(__file__).resolve().parent
RAW_ROOT = REPO_ROOT / "examples" / "_shared" / "data" / "speech_commands"


def main() -> None:
    num_classes = int(os.environ.get("KWS_CLASSES", "6"))
    assert num_classes in (6, 35), num_classes
    data_dir = HERE / "data" / f"{num_classes}class"
    data_dir.mkdir(parents=True, exist_ok=True)

    splits = load_speechcommands(RAW_ROOT, num_classes)
    for split in ("train", "val", "test"):
        x, y = splits[split]
        np.save(data_dir / f"{split}_x.npy", x.astype(np.float32))
        np.save(data_dir / f"{split}_y.npy", y.astype(np.int32))
        print(f"{split}: x={x.shape} y={y.shape} classes={num_classes}", flush=True)


if __name__ == "__main__":
    main()
