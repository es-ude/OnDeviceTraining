"""Same-start check: the C program and the PyTorch twin from one initial state, step by step.

The twin exports its initial weights and BN buffers, the C program loads them (TCR_SAME_START),
and both train the first 100 steps of epoch 0 in the same keyed sample order with Dropout off and
micro-batch = batch = 100. Every step's loss must agree to a relative 1e-4; exits 1 otherwise.

    uv run examples/kws_tcresnet8/same_start.py --data A --width full
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))
from examples._shared.seeds import SEED  # noqa: E402
from examples.kws_tcresnet8.train_pytorch import same_start_losses  # noqa: E402

STEPS = 100
REL_TOL = 1e-4
DATA = {"A": REPO_ROOT / "examples/kws_mfcc/data/12class_paper",
        "B": REPO_ROOT / "examples/kws_mfcc/data/6class"}
C_BINARY = REPO_ROOT / "build/examples/examples/kws_tcresnet8/train_c_kws_tcresnet8"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", choices=sorted(DATA), required=True)
    parser.add_argument("--width", choices=("full", "half"), required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--c-binary", type=Path, default=C_BINARY)
    parser.add_argument("--out", type=Path, help="default: examples/kws_tcresnet8/logs/same_start_<data>_<width>")
    args = parser.parse_args()
    out = args.out or HERE / "logs" / f"same_start_{args.data}_{args.width}"
    init_dir = out / "init"

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=True)
    py = same_start_losses(DATA[args.data], args.width, args.seed, STEPS, init_dir)

    env = dict(os.environ, KWS_DATA_DIR=str(DATA[args.data]), TCR_WIDTH=args.width,
               TCR_SEED=str(args.seed), TCR_MICROBATCH="100", TCR_OUT=str(out),
               TCR_SAME_START=str(init_dir))
    subprocess.run([str(args.c_binary)], env=env, check=True, cwd=REPO_ROOT)
    c = np.load(out / "step_losses.npy").astype(np.float64)

    rel = np.abs(c - py) / np.abs(py)
    for step in range(STEPS):
        print(f"step {step + 1:3d}: pytorch {py[step]:.7f}  c {c[step]:.7f}  rel {rel[step]:.2e}")
    bad = np.flatnonzero(rel > REL_TOL)
    summary = {"data": args.data, "width": args.width, "seed": args.seed, "steps": STEPS,
               "rel_tol": REL_TOL, "max_rel": float(rel.max()), "argmax_step": int(rel.argmax()) + 1,
               "first_failing_step": int(bad[0]) + 1 if bad.size else None,
               "pytorch": py.tolist(), "c": c.tolist()}
    (out / "same_start.json").write_text(json.dumps(summary, indent=2))
    print(f"max relative |dLoss| = {rel.max():.3e} at step {rel.argmax() + 1} (tolerance {REL_TOL:g})")
    if bad.size:
        print(f"FAIL: {bad.size} steps above tolerance, the first at step {bad[0] + 1}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
