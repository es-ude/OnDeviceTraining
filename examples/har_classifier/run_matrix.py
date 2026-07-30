"""Offline sweep driver for the HAR FLOAT32-vs-packed-SYM memory/accuracy study.

Runs a config matrix x seeds. Each run invokes the appropriate trainer binary
with per-run env and collects one extended RunLog JSON under ``logs/``.

THREE BINARIES, not one MODE-switched binary: ``train_c_har_classifier`` (FLOAT32),
``train_c_har_classifier_sym`` (packed SYM@x weights, x = SYM_BITS), and
``train_c_har_classifier_adamw`` (AdamW, ignores MOMENTUM). The SYM configs are
the SAME binary at different packed widths. LR is left to each binary's
per-config default (float/SYM default LR=0.01, adamw defaults LR=0.001);
momentum 0.9 (adamw ignores it).

#279 dead-zone A/B: the ``sym<N>`` configs default to seeded SR_HALF_AWAY on the
packed-SYM param write-back (the dead-zone escape); the ``sym<N>det`` configs
force deterministic HALF_AWAY (the dead-zone baseline). Same binary + width, so
symN vs symNdet isolates exactly the write-back rounding mode.

This is a LONG offline job (~40-63 s/epoch on the single-threaded C trainers).
Each (config, seed) pair is independent (distinct LOG_PATH), so ``--jobs N`` fans
them out across N processes -- on an M-series Mac (6P+2E) ``--jobs 8`` is the
knee. It is deliberately NOT wired into CI. Smoke a subset first, e.g.::

    uv run examples/har_classifier/run_matrix.py \
        --configs float sym8 sym8det --seeds 1 2 --epochs 2 --jobs 8

Requires a trainer build (pick the matching --build-subdir)::

    cmake --preset examples && cmake --build --preset examples --target \
        train_c_har_classifier train_c_har_classifier_sym train_c_har_classifier_adamw
    # ...then --build-subdir examples  (default: examples_memprofile)

Caveat (parallel): fixed-path trainer artifacts (e.g. outputs/*.npy) are
clobbered by concurrent runs -- they are gitignored/regenerable and this study
reads only the per-run JSON, so it does not matter here.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

# config name -> (binary, extra per-run env). The FLOAT32 binary ignores SYM_BITS;
# each SYM config is the same binary at a different packed weight width.
CONFIGS: dict[str, tuple[str, dict[str, str]]] = {
    "float": ("train_c_har_classifier", {}),
    "sym12": ("train_c_har_classifier_sym", {"SYM_BITS": "12"}),
    "sym10": ("train_c_har_classifier_sym", {"SYM_BITS": "10"}),
    "sym8": ("train_c_har_classifier_sym", {"SYM_BITS": "8"}),
    # Full-SYM wires (#206 acceptance): SYM_INT32 activation + dx wires, SYM
    # dx compute, SYM softmax output -> fake-quant CE + SYM metrics argmax.
    "sym8w": ("train_c_har_classifier_sym", {"SYM_BITS": "8", "SYM_WIRES": "1"}),
    "sym6": ("train_c_har_classifier_sym", {"SYM_BITS": "6"}),
    "sym4": ("train_c_har_classifier_sym", {"SYM_BITS": "4"}),
    "sym8cos": ("train_c_har_classifier_sym", {"SYM_BITS": "8", "LR_SCHEDULE": "cosine"}),
    "sym4cos": ("train_c_har_classifier_sym", {"SYM_BITS": "4", "LR_SCHEDULE": "cosine"}),
    "adamw": ("train_c_har_classifier_adamw", {}),
    # #279 dead-zone baseline arm: deterministic HALF_AWAY write-back (the plain
    # sym<N> above default to seeded SR_HALF_AWAY). Only the rounding mode differs.
    "sym8det": ("train_c_har_classifier_sym", {"SYM_BITS": "8", "SYM_ROUNDING": "det"}),
    "sym6det": ("train_c_har_classifier_sym", {"SYM_BITS": "6", "SYM_ROUNDING": "det"}),
    "sym4det": ("train_c_har_classifier_sym", {"SYM_BITS": "4", "SYM_ROUNDING": "det"}),
    # Group-granular quantization (#300 axis, group-quant PR5): {per-channel, G64,
    # G32} x {SYM, ASYM} at the two coarse widths where the accuracy-per-byte
    # frontier is most interesting (sym4/sym6 -- see the design spec's §8
    # acceptance grid, docs/superpowers/specs/2026-07-28-group-quantization-design.md).
    # "pc" = per-channel (one scale/zero-point per output channel, GROUP_MODE=channel);
    # "g64"/"g32" = fixed group size via GROUP_MODE=size. Bare "sym{W}"/"asym{W}"
    # (no pc/g64/g32 suffix) are per-TENSOR (today's default GROUP_MODE) -- the
    # asym twin of the existing sym4/sym6 arms above, added here since ASYM only
    # exists as an axis from this PR on.
    #
    # DIVISIBILITY FALLBACK: a GROUP_MODE=size request only takes if GROUP_SIZE
    # evenly divides a layer's own element count N; otherwise that layer alone
    # falls back to its per-channel groupSize (see resolveGroupShape in
    # train_c_sym.c). On HAR, conv1's weight is N=1008=2^4*3^2*7 over 16 output
    # channels (pc=63=3^2*7) -- neither 64 nor 32 divides 1008, so EVERY g64/g32
    # arm below silently runs conv1 at per-channel while conv2/conv3/linear get
    # their requested group size. This is not a bug to chase; it is recorded
    # per-run in the log's `groups_resolved` field (per-layer [numGroups,
    # groupSize]) -- read that field, don't infer group shapes from the config
    # name alone. See README.md's "Group-granular quantization" section for the
    # full per-layer resolved-shape table and the ODTS round-trip demo.
    "sym4pc": ("train_c_har_classifier_sym", {"SYM_BITS": "4", "GROUP_MODE": "channel"}),
    "sym4g64": ("train_c_har_classifier_sym",
                {"SYM_BITS": "4", "GROUP_MODE": "size", "GROUP_SIZE": "64"}),
    "sym4g32": ("train_c_har_classifier_sym",
                {"SYM_BITS": "4", "GROUP_MODE": "size", "GROUP_SIZE": "32"}),
    "asym4": ("train_c_har_classifier_sym", {"SYM_BITS": "4", "WEIGHT_DTYPE": "asym"}),
    "asym4pc": ("train_c_har_classifier_sym",
                {"SYM_BITS": "4", "WEIGHT_DTYPE": "asym", "GROUP_MODE": "channel"}),
    "asym4g64": ("train_c_har_classifier_sym",
                 {"SYM_BITS": "4", "WEIGHT_DTYPE": "asym", "GROUP_MODE": "size",
                  "GROUP_SIZE": "64"}),
    "asym4g32": ("train_c_har_classifier_sym",
                 {"SYM_BITS": "4", "WEIGHT_DTYPE": "asym", "GROUP_MODE": "size",
                  "GROUP_SIZE": "32"}),
    "sym6pc": ("train_c_har_classifier_sym", {"SYM_BITS": "6", "GROUP_MODE": "channel"}),
    "sym6g64": ("train_c_har_classifier_sym",
                {"SYM_BITS": "6", "GROUP_MODE": "size", "GROUP_SIZE": "64"}),
    "sym6g32": ("train_c_har_classifier_sym",
                {"SYM_BITS": "6", "GROUP_MODE": "size", "GROUP_SIZE": "32"}),
    "asym6": ("train_c_har_classifier_sym", {"SYM_BITS": "6", "WEIGHT_DTYPE": "asym"}),
    "asym6pc": ("train_c_har_classifier_sym",
                {"SYM_BITS": "6", "WEIGHT_DTYPE": "asym", "GROUP_MODE": "channel"}),
    "asym6g64": ("train_c_har_classifier_sym",
                 {"SYM_BITS": "6", "WEIGHT_DTYPE": "asym", "GROUP_MODE": "size",
                  "GROUP_SIZE": "64"}),
    "asym6g32": ("train_c_har_classifier_sym",
                 {"SYM_BITS": "6", "WEIGHT_DTYPE": "asym", "GROUP_MODE": "size",
                  "GROUP_SIZE": "32"}),
}


def is_complete(log_path: Path, epochs: int, seed: int) -> bool:
    """A run is complete if its JSON log exists, carries a `final` block, and its
    persisted config matches the requested run identity. Lets --resume skip
    finished (config, seed) pairs so a sweep survives an interruption / kill and
    continues where it left off on relaunch.

    Epochs and seed are the two identity parameters not already bound by the
    log filename (config name determines binary + env), so compare them against
    the log's `config` block: without this, a 2-epoch smoke log would silently
    survive into a 50-epoch sweep's aggregate. Missing/malformed config counts
    as incomplete -- the trainers open the log with mode "w", so a rerun
    overwrites cleanly."""
    if not log_path.exists():
        return False
    try:
        data = json.loads(log_path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    config = data.get("config", {})
    return "final" in data and config.get("epochs") == epochs and config.get("seed") == seed


def parse_final_acc(stdout: str) -> float | None:
    """Parse the 'test_acc=<float>' token from the trainer's FINAL output line."""
    for token in stdout.split():
        if token.startswith("test_acc="):
            try:
                return float(token[len("test_acc="):])
            except ValueError:
                return None
    return None


def run_one(
    config: str, seed: int, epochs: int, logs_dir: Path, build_dir: Path
) -> tuple[float | None, float]:
    """Run one (config, seed) training job; return (final test_acc, wall seconds).

    Raises FileNotFoundError if the binary is missing (a setup error). Non-zero
    exit propagates as CalledProcessError; non-convergence is NOT an error (the
    SYM binary records it as an advisory diagnostic and exits 0).
    """
    binary_name, extra_env = CONFIGS[config]
    binary = build_dir / binary_name
    if not binary.exists():
        raise FileNotFoundError(
            f"trainer binary not found: {binary}\n"
            "Build a trainer preset first, e.g.:\n"
            "  cmake --preset examples && \\\n"
            f"  cmake --build --preset examples --target {binary_name}\n"
            "(and pass the matching --build-subdir; default is examples_memprofile)"
        )
    log_path = logs_dir / f"{config}_seed{seed}.json"
    env = {
        **os.environ,
        **extra_env,
        "SEED": str(seed),
        "EPOCHS": str(epochs),
        "LOG_PATH": str(log_path),
    }
    # A shell-exported ODTS_ROUNDTRIP=1 would leak into every parallel run and
    # race the fixed outputs/har_sym_group.odts path (spurious MISMATCH exits).
    # The round-trip demo is a standalone check, never a sweep dimension.
    env.pop("ODTS_ROUNDTRIP", None)
    t0 = time.monotonic()
    result = subprocess.run(
        [str(binary)], cwd=ROOT, env=env, capture_output=True, text=True, check=True
    )
    wall = time.monotonic() - t0
    return parse_final_acc(result.stdout), wall


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS),
        help="subset of the config matrix (default: all)",
    )
    ap.add_argument(
        "--seeds", nargs="+", type=int, default=list(range(1, 11)),
        help="seeds to run (default: 1..10)",
    )
    ap.add_argument("--epochs", type=int, default=50, help="epochs per run (default: 50)")
    ap.add_argument(
        "--logs", default=str(HERE / "logs"), help="output dir for per-run JSON"
    )
    ap.add_argument(
        "--jobs", type=int, default=1,
        help="number of (config, seed) runs to execute in parallel (default: 1). "
             "The C trainers are single-threaded, so ~all cores is the knee (8 on "
             "an M-series 6P+2E).",
    )
    ap.add_argument(
        "--build-subdir", default="examples_memprofile",
        help="build/<subdir>/examples/har_classifier holds the trainer binaries "
             "(default: examples_memprofile; use 'examples' for the plain build)",
    )
    ap.add_argument(
        "--resume", action="store_true",
        help="skip (config, seed) pairs whose log already has a `final` block AND "
             "matching config (epochs, seed), so an interrupted/killed sweep "
             "continues where it left off on relaunch; mismatched logs rerun",
    )
    args = ap.parse_args()

    logs_dir = Path(args.logs)
    logs_dir.mkdir(parents=True, exist_ok=True)
    build_dir = ROOT / "build" / args.build_subdir / "examples" / "har_classifier"

    pairs = [(config, seed) for seed in args.seeds for config in args.configs]
    if args.resume:
        before = len(pairs)
        pairs = [
            (c, s)
            for c, s in pairs
            if not is_complete(logs_dir / f"{c}_seed{s}.json", args.epochs, s)
        ]
        skipped = before - len(pairs)
        if skipped:
            print(f"resume: skipping {skipped} already-complete run(s)", flush=True)
    total = len(pairs)

    # Preflight every selected binary BEFORE submitting jobs: a missing binary
    # discovered mid-sweep would otherwise surface only after the executor
    # drains its whole queue (shutdown(wait=True) keeps STARTING queued jobs),
    # i.e. hours of unrelated training runs for a setup error.
    missing = {
        build_dir / CONFIGS[config][0]
        for config, _ in pairs
        if not (build_dir / CONFIGS[config][0]).exists()
    }
    if missing:
        sys.exit(
            "trainer binaries not found:\n"
            + "\n".join(f"  {path}" for path in sorted(missing))
            + "\nBuild the trainer preset first, e.g.:\n"
            "  cmake --preset examples && \\\n"
            "  cmake --build --preset examples --target "
            + " ".join(sorted(path.name for path in missing))
            + "\n(and pass the matching --build-subdir; default is examples_memprofile)"
        )

    print(
        f"HAR sweep: {len(args.configs)} configs x {len(args.seeds)} seeds "
        f"x {args.epochs} epochs = {total} runs, {args.jobs}-way parallel -> {logs_dir}",
        flush=True,
    )

    done = 0
    failures = 0
    t_start = time.monotonic()
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = {
            pool.submit(run_one, config, seed, args.epochs, logs_dir, build_dir): (config, seed)
            for config, seed in pairs
        }
        for future in as_completed(futures):
            config, seed = futures[future]
            done += 1
            try:
                acc, wall = future.result()
            except FileNotFoundError as exc:
                # Binary vanished mid-sweep (preflight passed at launch): a setup
                # error common to every run. Cancel the queued jobs -- otherwise
                # the executor's exit shutdown(wait=True) keeps starting them --
                # and let the in-flight ones finish writing their logs.
                print(f"\n{exc}", file=sys.stderr)
                pool.shutdown(wait=False, cancel_futures=True)
                raise
            except subprocess.CalledProcessError as exc:
                failures += 1
                print(
                    f"[{done:3d}/{total}] {config:8s} seed{seed:<3d} FAILED "
                    f"(exit {exc.returncode})\n{exc.stderr}",
                    file=sys.stderr, flush=True,
                )
                continue
            acc_str = f"{acc:.4f}" if acc is not None else "N/A"
            print(
                f"[{done:3d}/{total}] {config:8s} seed{seed:<3d} "
                f"test_acc={acc_str} wall={wall:6.1f}s",
                flush=True,
            )

    elapsed = time.monotonic() - t_start
    tail = f" ({failures} FAILED)" if failures else ""
    print(f"done: {done} runs in {elapsed/60:.1f} min{tail} -> {logs_dir}", flush=True)
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
