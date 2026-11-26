"""Same-start check: the C program and the PyTorch twin from one initial state.

The twin exports its initial weights and BN buffers, the C program loads them (TCR_SAME_START),
and both train the first steps of epoch 0 in the same keyed sample order with Dropout off and
micro-batch = batch = 100. Passes (exit 0) only when all three checks hold:

1. step 1: |C - PyTorch fp32| / |PyTorch fp32| <= 1e-4 on the first step's loss;
2. one update: per kind (param, momentum, running_mean, running_var),
   d(X) = max over tensors of max|X - t64| / max|t64| against a float64 twin (t64) of the same
   step, and d(C) <= 10 * max(d(t32), 1e-6); a non-finite distance fails;
3. eval: after that update, C's probabilities are finite, both sides hold exactly the first 200
   validation samples, and C and PyTorch fp32 agree on the argmax of >= 199 of them (running BN
   statistics, no Dropout).

The 100-step loss curve is printed and saved for information only: float32 rounding at near-zero
ReLU inputs and in BN's batch sums makes any two float32 implementations part within a few steps.

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
from examples.kws_tcresnet8.train_pytorch import (  # noqa: E402
    eval_probs, export_init, load_split, same_start_model, same_start_train, training_state)

KINDS = ("param", "momentum", "running_mean", "running_var")
STEP1_REL_TOL = 1e-4
STATE_FACTOR = 10.0
# ~8 float32 ulp: t32 can sit 1 ulp from t64, where a pure 10x bound would be sub-ulp.
STATE_FLOOR = 1e-6
EVAL_ROWS = 200
EVAL_MIN_AGREE = 199
CURVE_STEPS = 100
CURVE_REL_TOL = 1e-4
DATA = {"A": REPO_ROOT / "examples/kws_mfcc/data/12class_paper",
        "B": REPO_ROOT / "examples/kws_mfcc/data/6class"}
C_BINARY = REPO_ROOT / "build/examples/examples/kws_tcresnet8/train_c_kws_tcresnet8"


def kind_distance(x: list[np.ndarray], ref: list[np.ndarray]) -> float:
    """max over tensors of max|x - ref| / max|ref|: each tensor is scaled by its own largest
    value, since an element-wise relative error is dominated by values near 0. NaN propagates
    (from x, or 0/0 from an all-zero ref), so the caller must reject a non-finite distance."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return float(np.max([np.abs(a - r).max() / np.abs(r).max()
                             for a, r in zip(x, ref, strict=True)]))


def same_start_verdict(step1_c: float, step1_t32: float, state_c: dict, state_t32: dict,
                       state_t64: dict, probs_c: np.ndarray, probs_t32: np.ndarray) -> dict:
    """The rule of the module docstring over plain arrays. state_*: kind -> list of tensors in one
    order on all three sides; probs_*: [EVAL_ROWS, classes], any other shape fails the eval
    check."""
    step1_rel = float(abs(step1_c - step1_t32) / abs(step1_t32))
    state = {}
    for k in KINDS:
        d_c = kind_distance(state_c[k], state_t64[k])
        d_t32 = kind_distance(state_t32[k], state_t64[k])
        bound = STATE_FACTOR * max(d_t32, STATE_FLOOR)
        state[k] = {"d_c": d_c, "d_t32": d_t32, "bound": bound, "ratio": d_c / bound,
                    "pass": bool(np.isfinite(d_c) and np.isfinite(d_t32) and d_c <= bound)}
    shape_ok = probs_c.shape == probs_t32.shape and probs_c.ndim == 2 and len(probs_c) == EVAL_ROWS
    agree = int((probs_c.argmax(axis=1) == probs_t32.argmax(axis=1)).sum()) if shape_ok else 0
    probs_finite = bool(np.isfinite(probs_c).all())
    verdict = {
        "step1": {"c": float(step1_c), "t32": float(step1_t32), "rel": step1_rel, "tol": STEP1_REL_TOL,
                  "pass": bool(step1_rel <= STEP1_REL_TOL)},
        "state": state,
        "eval": {"agree": agree, "rows": EVAL_ROWS, "min_agree": EVAL_MIN_AGREE,
                 "shape_c": list(probs_c.shape), "shape_t32": list(probs_t32.shape),
                 "shape_ok": shape_ok, "finite": probs_finite,
                 "pass": shape_ok and probs_finite and agree >= EVAL_MIN_AGREE},
    }
    verdict["pass"] = (verdict["step1"]["pass"] and all(s["pass"] for s in state.values())
                       and verdict["eval"]["pass"])
    return verdict


def run_c(binary: Path, data_dir: Path, width: str, seed: int, init_dir: Path, steps: int,
          out: Path) -> None:
    env = dict(os.environ, KWS_DATA_DIR=str(data_dir), TCR_WIDTH=width, TCR_SEED=str(seed),
               TCR_MICROBATCH="100", TCR_OUT=str(out), TCR_SAME_START=str(init_dir),
               TCR_SAME_START_STEPS=str(steps))
    subprocess.run([str(binary)], env=env, check=True, cwd=REPO_ROOT)


def load_c_state(state_dir: Path, like: dict) -> dict:
    """The C program's state files, as many per kind as the twin holds."""
    return {k: [np.load(state_dir / f"{k}{i:02d}.npy").astype(np.float64).ravel()
                for i in range(len(like[k]))] for k in KINDS}


def print_verdict(v: dict) -> None:
    s1 = v["step1"]
    print(f"1. step-1 loss: pytorch {s1['t32']:.7f}  c {s1['c']:.7f}  rel {s1['rel']:.2e} "
          f"(tol {s1['tol']:g})  {'PASS' if s1['pass'] else 'FAIL'}")
    print("2. one-update state vs float64 twin:")
    print(f"   {'kind':<13} {'d(C)':>9} {'d(t32)':>9} {'bound':>9} {'d(C)/bound':>10}")
    for k, s in v["state"].items():
        print(f"   {k:<13} {s['d_c']:9.2e} {s['d_t32']:9.2e} {s['bound']:9.2e} {s['ratio']:10.3f}"
              f"  {'PASS' if s['pass'] else 'FAIL'}")
    e = v["eval"]
    finite = "" if e["finite"] else ", C probabilities not finite"
    shape = "" if e["shape_ok"] else (f", shapes C {e['shape_c']} / PyTorch {e['shape_t32']} "
                                      f"(need [{e['rows']}, classes] on both)")
    print(f"3. eval argmax: {e['agree']}/{e['rows']} agree (need {e['min_agree']}){finite}{shape}  "
          f"{'PASS' if e['pass'] else 'FAIL'}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """The command line with --out and --c-binary absolute: the C child runs from the repo root,
    so a relative path would name a different directory there."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", choices=sorted(DATA), required=True)
    parser.add_argument("--width", choices=("full", "half"), required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--c-binary", type=Path, default=C_BINARY)
    parser.add_argument("--out", type=Path, help="default: examples/kws_tcresnet8/logs/same_start_<data>_<width>")
    args = parser.parse_args(argv)
    args.out = (args.out or HERE / "logs" / f"same_start_{args.data}_{args.width}").resolve()
    args.c_binary = args.c_binary.resolve()
    return args


def main() -> int:
    args = parse_args()
    out = args.out
    init_dir = out / "init"
    data_dir = DATA[args.data]

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=True)
    x, y = load_split(data_dir, "train")
    val_x, _ = load_split(data_dir, "val")
    val_x = val_x[:EVAL_ROWS]
    classes = int(y.max()) + 1

    model, opt = same_start_model(args.width, classes, args.seed)
    export_init(model, init_dir)
    curve_py = same_start_train(model, opt, x, y, args.seed, CURVE_STEPS)
    twin = {}
    for name, dtype in (("t32", torch.float32), ("t64", torch.float64)):
        model, opt = same_start_model(args.width, classes, args.seed, dtype)
        loss = same_start_train(model, opt, x, y, args.seed, 1)[0]
        twin[name] = (loss, training_state(model, opt), eval_probs(model, val_x))

    run_c(args.c_binary, data_dir, args.width, args.seed, init_dir, 1, out / "one_step")
    run_c(args.c_binary, data_dir, args.width, args.seed, init_dir, CURVE_STEPS, out / "curve")
    c_loss = float(np.load(out / "one_step" / "step_losses.npy")[0])
    c_state = load_c_state(out / "one_step" / "state", twin["t32"][1])
    c_probs = np.load(out / "one_step" / "val_probs.npy").astype(np.float64)
    curve_c = np.load(out / "curve" / "step_losses.npy").astype(np.float64)

    rel = np.abs(curve_c - curve_py) / np.abs(curve_py)
    for step in range(CURVE_STEPS):
        print(f"step {step + 1:3d}: pytorch {curve_py[step]:.7f}  c {curve_c[step]:.7f}  rel {rel[step]:.2e}")
    departed = np.flatnonzero(rel > CURVE_REL_TOL)
    first_departure = int(departed[0]) + 1 if departed.size else None
    print(f"curve (information only): first step with rel > {CURVE_REL_TOL:g}: {first_departure}; "
          f"max rel {rel.max():.3e} at step {rel.argmax() + 1}")

    verdict = same_start_verdict(c_loss, twin["t32"][0], c_state, twin["t32"][1], twin["t64"][1],
                                 c_probs, twin["t32"][2])
    print_verdict(verdict)
    verdict["curve"] = {"steps": CURVE_STEPS, "rel_tol": CURVE_REL_TOL,
                        "first_departure_step": first_departure, "max_rel": float(rel.max()),
                        "argmax_step": int(rel.argmax()) + 1,
                        "pytorch": curve_py.tolist(), "c": curve_c.tolist()}
    summary = {"data": args.data, "width": args.width, "seed": args.seed, **verdict}
    (out / "same_start.json").write_text(json.dumps(summary, indent=2))
    print("PASS" if verdict["pass"] else "FAIL")
    return 0 if verdict["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
