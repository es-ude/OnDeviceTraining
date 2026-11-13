"""Offline sweep driver for the HAR FLOAT32-vs-packed-SYM memory/accuracy study.

Runs a config matrix x seeds. Each run invokes the appropriate trainer binary
with per-run env and collects one extended RunLog JSON under ``logs/``.

FOUR BINARIES, not one MODE-switched binary: ``train_c_har_classifier`` (FLOAT32),
``train_c_har_classifier_sym`` (packed SYM@x weights, x = SYM_BITS),
``train_c_har_classifier_adamw`` (AdamW, ignores MOMENTUM), and
``train_c_har_classifier_bfp`` (block floating point, epic #410 PR7). The SYM
configs are the SAME binary at different packed widths. LR is left to each
binary's per-config default (float/SYM default LR=0.01, adamw defaults
LR=0.001); momentum 0.9 (adamw ignores it).

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
        train_c_har_classifier train_c_har_classifier_sym train_c_har_classifier_adamw \
        train_c_har_classifier_bfp
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


def bfp_config_name(env: dict[str, str]) -> str:
    """The BFP name grammar (#410 PR7): EVERY knob is encoded so --resume
    can never confuse two numerically different runs:
      bfp_wb{t|pc|N}_ab{f|t|N}_m{M}_e{E}_x{nat|fq}_g{0|1}_s{0|1}_r{sr|det}_l{const|cos}_h{flt|bfp}
    (h = BFP_HEAD, the required model-head knob)."""
    wb = {"tensor": "t", "channel": "pc"}.get(env["BFP_WEIGHT_BLOCK"], env["BFP_WEIGHT_BLOCK"])
    ab = {"float": "f", "tensor": "t"}.get(env["BFP_WIRE_BLOCK"], env["BFP_WIRE_BLOCK"])
    x = {"native": "nat", "fq": "fq"}[env["BFP_MATH"]]
    sched = {"none": "const", "cosine": "cos"}[env["LR_SCHEDULE"]]
    head = {"float": "flt", "bfp": "bfp"}[env["BFP_HEAD"]]
    return (f"bfp_wb{wb}_ab{ab}_m{env['BFP_MANTISSA_BITS']}_e{env['BFP_EXPONENT_BITS']}"
            f"_x{x}_g{env['BFP_GRADS']}_s{env['BFP_STATE']}_r{env['BFP_ROUNDING']}_l{sched}"
            f"_h{head}")


def _bfp(**knobs: str) -> tuple[str, dict[str, str]]:
    env = {"BFP_WEIGHT_BLOCK": "32", "BFP_WIRE_BLOCK": "16", "BFP_MANTISSA_BITS": "6",
           "BFP_EXPONENT_BITS": "8", "BFP_MATH": "native", "BFP_GRADS": "0", "BFP_STATE": "0",
           "BFP_ROUNDING": "sr", "LR_SCHEDULE": "none", "BFP_HEAD": "bfp"}
    env.update(knobs)
    return ("train_c_har_classifier_bfp", env)


# BFP epic PR7 (#410) stage-1 arms: the anchor wb32/ab16/m6/e8 native/sr, then one
# axis varied per row. abf = FLOAT32 wires (arm A, activations staged per-tensor at
# the weight widths -- NOT a float arm); xfq = GEMM fake-quant reference (NOT a
# whole-model twin, see README); g1/s1 = per-tensor BFP grad / momentum storage;
# h = the model head (BFP_HEAD, required): hbfp = integer Softmax + integer p - y,
# hflt = float Softmax + float p - y with one conversion onto the loss-grad wire
# (the only head FLOAT32 wires take).
_BFP_STAGE1 = [
    _bfp(),
    _bfp(BFP_HEAD="float"),
    _bfp(BFP_WEIGHT_BLOCK="tensor"),
    _bfp(BFP_WEIGHT_BLOCK="channel"),
    _bfp(BFP_WEIGHT_BLOCK="8"),
    _bfp(BFP_WIRE_BLOCK="float", BFP_HEAD="float"),
    _bfp(BFP_WIRE_BLOCK="tensor"),
    _bfp(BFP_WIRE_BLOCK="32"),
    _bfp(BFP_MANTISSA_BITS="4"),
    _bfp(BFP_MANTISSA_BITS="8"),
    _bfp(BFP_EXPONENT_BITS="4"),
    _bfp(BFP_ROUNDING="det"),
    _bfp(BFP_MATH="fq"),
    _bfp(BFP_GRADS="1"),
    _bfp(BFP_GRADS="1", BFP_STATE="1"),
]

# The full-integer training path (bfp8full): bfp8 (m8 e8, per-channel weights, ab16,
# native, sr) with BFP grads AND BFP momentum, so the forward, the backward, the
# gradient accumulation, the mean scaling, the momentum store and the parameter
# update all run the integer BFP kernels. The trainer derives the same predicate
# (native math, BFP wires, BFP_GRADS=1, BFP_STATE=1) and checks every layer at
# init (FULL GATE). The head has no default here either: one alias per BFP_HEAD.
_BFP8FULL = {"BFP_WEIGHT_BLOCK": "channel", "BFP_MANTISSA_BITS": "8", "BFP_GRADS": "1",
             "BFP_STATE": "1"}
_BFP8FULL_CONFIGS = {
    "bfp8full_hflt": _bfp(**_BFP8FULL, BFP_HEAD="float"),
    "bfp8full_hbfp": _bfp(**_BFP8FULL, BFP_HEAD="bfp"),
}


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
    # E0 linear-quant baseline (host twins of the board firmware arms): SYM_BWD=1
    # = integer dW/db/dx (SYM_INT32 grad math, FLOAT32 grad storage) under
    # FLOAT32 wires (sym8b), + SYM_INT32 wires (sym8bw), + per-output-channel
    # weight groups (sym8bwg). asym8 = ASYM@8 weights, FLOAT32 backward.
    "sym8b": ("train_c_har_classifier_sym", {"SYM_BITS": "8", "SYM_BWD": "1"}),
    "sym8bw": ("train_c_har_classifier_sym", {"SYM_BITS": "8", "SYM_BWD": "1", "SYM_WIRES": "1"}),
    "sym8bwg": ("train_c_har_classifier_sym",
                {"SYM_BITS": "8", "SYM_BWD": "1", "SYM_WIRES": "1", "GROUP_MODE": "channel"}),
    # Fixed-size weight groups on top of sym8bw (twins of the firmware arms of the
    # same name); conv1 falls back to per-channel, see DIVISIBILITY FALLBACK below.
    "sym8bwg64": ("train_c_har_classifier_sym",
                  {"SYM_BITS": "8", "SYM_BWD": "1", "SYM_WIRES": "1", "GROUP_MODE": "size",
                   "GROUP_SIZE": "64"}),
    "sym8bwg32": ("train_c_har_classifier_sym",
                  {"SYM_BITS": "8", "SYM_BWD": "1", "SYM_WIRES": "1", "GROUP_MODE": "size",
                   "GROUP_SIZE": "32"}),
    "asym8": ("train_c_har_classifier_sym", {"SYM_BITS": "8", "WEIGHT_DTYPE": "asym"}),
    # ASYM@8 weights on the integer backward path: twins of sym8b / sym8bw / sym8bwg
    # (and of the firmware arms of the same name) with WEIGHT_DTYPE=asym.
    "asym8b": ("train_c_har_classifier_sym",
               {"SYM_BITS": "8", "WEIGHT_DTYPE": "asym", "SYM_BWD": "1"}),
    "asym8bw": ("train_c_har_classifier_sym",
                {"SYM_BITS": "8", "WEIGHT_DTYPE": "asym", "SYM_BWD": "1", "SYM_WIRES": "1"}),
    "asym8bwg": ("train_c_har_classifier_sym",
                 {"SYM_BITS": "8", "WEIGHT_DTYPE": "asym", "SYM_BWD": "1", "SYM_WIRES": "1",
                  "GROUP_MODE": "channel"}),
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
    # frontier is most interesting (sym4/sym6 -- the #300 acceptance grid).
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
    **{bfp_config_name(env): (binary, env) for binary, env in _BFP_STAGE1},
    **_BFP8FULL_CONFIGS,
}


# Every variable an arm can set (union over CONFIGS) plus the per-run knobs this
# script sets itself. ODTS_ROUNDTRIP is the standalone round-trip demo, never a
# sweep dimension: inherited into parallel runs it races the fixed
# outputs/har_sym_group.odts path (spurious MISMATCH exits).
ARM_KNOBS: frozenset[str] = frozenset(
    {key for _, env in CONFIGS.values() for key in env}
    | {"SEED", "EPOCHS", "LOG_PATH", "ODTS_ROUNDTRIP"}
)


def child_env(config_env: dict[str, str], **run_knobs: str) -> dict[str, str]:
    """The trainer environment for one arm: the parent environment without any
    arm knob, then exactly this arm's config and per-run knobs. Without the
    strip, a knob exported in the parent shell (e.g. SYM_BWD=1) would leak into
    every arm that does not set it and mislabel those runs."""
    inherited = {k: v for k, v in os.environ.items() if k not in ARM_KNOBS}
    return {**inherited, **config_env, **run_knobs}


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
    """Parse the 'test_acc=<float>' token from the trainer's FINAL output line.

    Anchored to the line starting with 'FINAL ' (a space): the harness also
    prints an optional 'FINAL@best-val ...' line that carries its own
    test_acc= token (best-val-loss snapshot + graceful divergence, #446),
    which must never be picked up here.
    """
    for line in stdout.splitlines():
        if not line.startswith("FINAL "):
            continue
        for token in line.split():
            if token.startswith("test_acc="):
                try:
                    return float(token[len("test_acc="):])
                except ValueError:
                    return None
    return None


def needs_bfp_data(env: dict[str, str]) -> bool:
    """The trainer's full-integer predicate (native math, BFP wires, BFP grads AND
    momentum): such a run trains on data pre-converted by `--write-bfp-data`."""
    return (env.get("BFP_GRADS") == "1" and env.get("BFP_STATE") == "1"
            and env.get("BFP_MATH", "native") == "native"
            and env.get("BFP_WIRE_BLOCK") not in (None, "float"))


def bfp_data_key(env: dict[str, str]) -> tuple[str, str, str]:
    """What the pre-converted data depends on: (m, e, wire block)."""
    return (env["BFP_MANTISSA_BITS"], env["BFP_EXPONENT_BITS"], env["BFP_WIRE_BLOCK"])


def write_bfp_data(configs, build_dir: Path) -> None:
    """Runs `<trainer> --write-bfp-data` once per data format the selected
    full-integer configs need, serially, before any job starts (the parallel
    runs then only read the files)."""
    done = set()
    for config in configs:
        binary_name, env = CONFIGS[config]
        if not needs_bfp_data(env) or bfp_data_key(env) in done:
            continue
        done.add(bfp_data_key(env))
        subprocess.run([str(build_dir / binary_name), "--write-bfp-data"], cwd=ROOT,
                       env=child_env(env), capture_output=True, text=True, check=True)
        print(f"pre-converted BFP data written for {config}", flush=True)


def run_one(
    config: str, seed: int, epochs: int, logs_dir: Path, build_dir: Path
) -> tuple[float | None, float]:
    """Run one (config, seed) training job; return (final test_acc, wall seconds).

    Raises FileNotFoundError if the binary is missing (a setup error). Non-zero
    exit propagates as CalledProcessError (run_pair counts it as diverged);
    non-convergence is NOT an error (the SYM binary records it as an advisory
    diagnostic and exits 0).
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
    env = child_env(extra_env, SEED=str(seed), EPOCHS=str(epochs), LOG_PATH=str(log_path))
    t0 = time.monotonic()
    result = subprocess.run(
        [str(binary)], cwd=ROOT, env=env, capture_output=True, text=True, check=True
    )
    wall = time.monotonic() - t0
    return parse_final_acc(result.stdout), wall


def log_diverged(log_path: Path) -> bool:
    """True iff the run's JSON log records `final.diverged == 1` (graceful stop at
    the first non-finite loss, #446). A missing/unreadable log is not divergence."""
    try:
        final = json.loads(log_path.read_text()).get("final", {})
    except (json.JSONDecodeError, OSError):
        return False
    return bool(final.get("diverged"))


def run_pair(
    config: str, seed: int, epochs: int, logs_dir: Path, build_dir: Path
) -> dict:
    """Run one (config, seed) and return its result row, never dropping the seed.

    A trainer that exits non-zero (e.g. the SYM scale fail-fast on a non-finite
    fit, which is how a diverging SYM run ends) counts as DIVERGED with no
    accuracy, so per-config averages are never silently taken over fewer seeds.
    A zero exit reads `diverged` from the run's own log. A missing binary is a
    setup error and still raises FileNotFoundError.
    """
    try:
        acc, wall = run_one(config, seed, epochs, logs_dir, build_dir)
    except subprocess.CalledProcessError as exc:
        return {"config": config, "seed": seed, "test_acc": None, "wall": None,
                "diverged": 1, "returncode": exc.returncode, "stderr": exc.stderr}
    diverged = int(log_diverged(logs_dir / f"{config}_seed{seed}.json"))
    return {"config": config, "seed": seed, "test_acc": acc, "wall": wall,
            "diverged": diverged, "returncode": 0, "stderr": ""}


def resumed_row(config: str, seed: int, log_path: Path) -> dict:
    """The result row of a pair --resume skipped as complete, read from its log
    (`final.test_acc`, `final.diverged`), so the summary covers every selected
    (config, seed), not only the pairs run in this invocation."""
    final = json.loads(log_path.read_text())["final"]
    return {"config": config, "seed": seed, "test_acc": final.get("test_acc"), "wall": None,
            "diverged": int(bool(final.get("diverged"))), "returncode": 0, "stderr": ""}


def format_summary(rows: list[dict]) -> str:
    """Per-config table (one line per seed, crashed seeds included with
    diverged=1 and no accuracy) plus the total diverged count."""
    lines = ["per-config results (diverged=1: stopped at a non-finite loss or crashed):"]
    for config in sorted({r["config"] for r in rows}):
        by_seed = sorted((r for r in rows if r["config"] == config), key=lambda r: r["seed"])
        n_div = sum(r["diverged"] for r in by_seed)
        lines.append(f"  {config}: {len(by_seed)} seeds, {n_div} diverged")
        for r in by_seed:
            acc = f"{r['test_acc']:.4f}" if r["test_acc"] is not None else "N/A"
            lines.append(f"    seed{r['seed']:<3d} test_acc={acc} diverged={r['diverged']}")
    lines.append(f"diverged: {sum(r['diverged'] for r in rows)}/{len(rows)} runs")
    return "\n".join(lines)


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
    rows: list[dict] = []
    if args.resume:
        complete = [
            (c, s) for c, s in pairs
            if is_complete(logs_dir / f"{c}_seed{s}.json", args.epochs, s)
        ]
        rows = [resumed_row(c, s, logs_dir / f"{c}_seed{s}.json") for c, s in complete]
        done_pairs = set(complete)
        pairs = [pair for pair in pairs if pair not in done_pairs]
        if complete:
            print(f"resume: skipping {len(complete)} already-complete run(s)", flush=True)
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
    write_bfp_data(dict.fromkeys(c for c, _ in pairs), build_dir)

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
            pool.submit(run_pair, config, seed, args.epochs, logs_dir, build_dir): (config, seed)
            for config, seed in pairs
        }
        for future in as_completed(futures):
            config, seed = futures[future]
            done += 1
            try:
                row = future.result()
            except FileNotFoundError as exc:
                # Binary vanished mid-sweep (preflight passed at launch): a setup
                # error common to every run. Cancel the queued jobs -- otherwise
                # the executor's exit shutdown(wait=True) keeps starting them --
                # and let the in-flight ones finish writing their logs.
                print(f"\n{exc}", file=sys.stderr)
                pool.shutdown(wait=False, cancel_futures=True)
                raise
            rows.append(row)
            acc, wall = row["test_acc"], row["wall"]
            if row["returncode"]:
                failures += 1
                print(
                    f"[{done:3d}/{total}] {config:42s} seed{seed:<3d} FAILED "
                    f"(exit {row['returncode']}), counted as diverged\n{row['stderr']}",
                    file=sys.stderr, flush=True,
                )
                continue
            acc_str = f"{acc:.4f}" if acc is not None else "N/A"
            print(
                f"[{done:3d}/{total}] {config:42s} seed{seed:<3d} "
                f"test_acc={acc_str} wall={wall:6.1f}s",
                flush=True,
            )

    elapsed = time.monotonic() - t_start
    tail = f" ({failures} FAILED)" if failures else ""
    print(f"done: {done} runs in {elapsed/60:.1f} min{tail} -> {logs_dir}", flush=True)
    print(format_summary(rows), flush=True)
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
