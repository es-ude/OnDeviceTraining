"""Offline sweep driver for the HAR FLOAT32-vs-packed-SYM memory/accuracy study.

Runs the config matrix {float, sym12, sym10, sym8, sym6, sym4, sym8cos, sym4cos,
adamw} x seed 1..10 = 90 runs. Each run invokes the memory-profiling build of the
appropriate trainer binary with per-run env and collects one extended RunLog
JSON under ``logs/``.

THREE BINARIES, not one MODE-switched binary: ``train_c_har_classifier`` (FLOAT32),
``train_c_har_classifier_sym`` (packed SYM@x weights, x = SYM_BITS), and
``train_c_har_classifier_adamw`` (AdamW, ignores MOMENTUM). The SYM configs are
the SAME binary at different packed widths. LR is left to each binary's
per-config default (float/SYM default LR=0.01, adamw defaults LR=0.001);
momentum 0.9 (adamw ignores it).

This is a LONG offline job (~40-63 s/epoch x 50 epochs x 90 runs ~= 60+ h) and is
deliberately NOT wired into CI — CI keeps only the fast FLOAT32 BIT_PARITY gate.
Use --configs / --seeds / --epochs to smoke a subset before committing to the
full run, e.g.::

    uv run examples/har_classifier/run_matrix.py --configs float sym8 --seeds 1 2 --epochs 2

Requires the memory-profiling build::

    cmake --preset examples_memprofile
    cmake --build --preset examples_memprofile --target \
        train_c_har_classifier train_c_har_classifier_sym
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BUILD_DIR = ROOT / "build" / "examples_memprofile" / "examples" / "har_classifier"

# config name -> (binary, extra per-run env). The FLOAT32 binary ignores SYM_BITS;
# each SYM config is the same binary at a different packed weight width.
CONFIGS: dict[str, tuple[str, dict[str, str]]] = {
    "float": ("train_c_har_classifier", {}),
    "sym12": ("train_c_har_classifier_sym", {"SYM_BITS": "12"}),
    "sym10": ("train_c_har_classifier_sym", {"SYM_BITS": "10"}),
    "sym8": ("train_c_har_classifier_sym", {"SYM_BITS": "8"}),
    "sym6": ("train_c_har_classifier_sym", {"SYM_BITS": "6"}),
    "sym4": ("train_c_har_classifier_sym", {"SYM_BITS": "4"}),
    "sym8cos": ("train_c_har_classifier_sym", {"SYM_BITS": "8", "LR_SCHEDULE": "cosine"}),
    "sym4cos": ("train_c_har_classifier_sym", {"SYM_BITS": "4", "LR_SCHEDULE": "cosine"}),
    "adamw": ("train_c_har_classifier_adamw", {}),
}


def parse_final_acc(stdout: str) -> float | None:
    """Parse the 'test_acc=<float>' token from the trainer's FINAL output line."""
    for token in stdout.split():
        if token.startswith("test_acc="):
            try:
                return float(token[len("test_acc="):])
            except ValueError:
                return None
    return None


def run_one(config: str, seed: int, epochs: int, logs_dir: Path) -> tuple[float | None, float]:
    """Run one (config, seed) training job; return (final test_acc, wall seconds).

    Raises on a genuine binary error (missing data, crash). Non-convergence is
    NOT an error — the SYM binary records it as an advisory diagnostic and exits
    0, so a coarse config that fails to descend still writes its JSON.
    """
    binary_name, extra_env = CONFIGS[config]
    binary = BUILD_DIR / binary_name
    if not binary.exists():
        raise FileNotFoundError(
            f"trainer binary not found: {binary}\n"
            "Build the memory-profiling preset first:\n"
            "  cmake --preset examples_memprofile && \\\n"
            "  cmake --build --preset examples_memprofile --target "
            f"{binary_name}"
        )
    log_path = logs_dir / f"{config}_seed{seed}.json"
    env = {
        **os.environ,
        **extra_env,
        "SEED": str(seed),
        "EPOCHS": str(epochs),
        "LOG_PATH": str(log_path),
    }
    t0 = time.monotonic()
    try:
        result = subprocess.run(
            [str(binary)], cwd=ROOT, env=env, capture_output=True, text=True, check=True
        )
    except subprocess.CalledProcessError as exc:
        print(
            f"  {config} seed{seed} FAILED (exit {exc.returncode}):\n{exc.stderr}",
            file=sys.stderr,
        )
        raise
    wall = time.monotonic() - t0
    return parse_final_acc(result.stdout), wall


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS),
        help="subset of the config matrix (default: all eight)",
    )
    ap.add_argument(
        "--seeds", nargs="+", type=int, default=list(range(1, 11)),
        help="seeds to run (default: 1..10)",
    )
    ap.add_argument("--epochs", type=int, default=50, help="epochs per run (default: 50)")
    ap.add_argument(
        "--logs", default=str(HERE / "logs"), help="output dir for per-run JSON"
    )
    args = ap.parse_args()

    logs_dir = Path(args.logs)
    logs_dir.mkdir(parents=True, exist_ok=True)

    total = len(args.configs) * len(args.seeds)
    print(
        f"HAR sweep: {len(args.configs)} configs x {len(args.seeds)} seeds "
        f"x {args.epochs} epochs = {total} runs -> {logs_dir}",
        flush=True,
    )

    done = 0
    t_start = time.monotonic()
    for seed in args.seeds:
        for config in args.configs:
            done += 1
            acc, wall = run_one(config, seed, args.epochs, logs_dir)
            acc_str = f"{acc:.4f}" if acc is not None else "N/A"
            print(
                f"[{done:3d}/{total}] {config:6s} seed{seed:<3d} "
                f"test_acc={acc_str} wall={wall:6.1f}s",
                flush=True,
            )

    elapsed = time.monotonic() - t_start
    print(f"done: {done} runs in {elapsed/60:.1f} min -> {logs_dir}", flush=True)


if __name__ == "__main__":
    main()
