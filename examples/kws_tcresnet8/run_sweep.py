"""TC-ResNet8 sweep: every C and PyTorch run that compare.py reads, several at a time.

The grid is data {A, B} x width {full, half} x seeds 0-9 for three arms: C and the PyTorch twin
at micro-batch 100 (BN statistics over the whole batch, as in PyTorch), and C at micro-batch 4
(the Pico-sized memory setting). On top come 8 one-epoch memory runs (4 configurations x
micro-batch {100, 4}) on the ODT_MEM_PROFILE build; the heap peak is reached in the first
training step, so one epoch measures it.

C runs use optimized builds (the `examples` preset is -O0, about 6x slower on A full width): `examples_release`
for training and `examples_release_memprofile` for the memory runs. Both are configured and
built here before any run, so a stale binary is never timed.

A run whose log is complete (it loads and holds every epoch its config names) is skipped, so a
killed sweep resumes where it stopped; a partial log is rerun.

    uv run examples/kws_tcresnet8/run_sweep.py --dry-run
    uv run examples/kws_tcresnet8/run_sweep.py --jobs 6 [--only 'A_full_*']
    uv run examples/kws_tcresnet8/run_sweep.py --time-one
"""
from __future__ import annotations

import argparse
import fnmatch
import heapq
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))
from examples._shared.log_schema import load_log  # noqa: E402
from examples.kws_tcresnet8.train_pytorch import paper_step_size_epochs  # noqa: E402

DATA = {"A": REPO_ROOT / "examples/kws_mfcc/data/12class_paper",
        "B": REPO_ROOT / "examples/kws_mfcc/data/6class"}
WIDTHS = ("full", "half")
CONFIGS = tuple((d, w) for d in DATA for w in WIDTHS)
# (side, micro-batch): the C-vs-PyTorch pair at m = batch, and C at the Pico-sized m
ARMS = (("c", 100), ("pytorch", 100), ("c", 4))
SEEDS = tuple(range(10))
MEM_MICRO_BATCHES = (100, 4)
TIMING_EPOCHS = 2

TRAIN_PRESET = "examples_release"
MEM_PRESET = "examples_release_memprofile"
C_TARGET = "train_c_kws_tcresnet8"
TWIN = HERE / "train_pytorch.py"
DEFAULT_LOGS = HERE / "logs" / "sweep"
TIMING_LOGS = HERE / "logs" / "timing"


def c_binary(preset: str) -> Path:
    return REPO_ROOT / "build" / preset / "examples" / "kws_tcresnet8" / C_TARGET


def train_size(data: str) -> int:
    return np.load(DATA[data] / "train_y.npy", mmap_mode="r").shape[0]


def paper_epochs(data: str) -> int:
    """The epoch count both trainers run without TCR_EPOCHS: 3 x the STEP_LR step size."""
    return 3 * paper_step_size_epochs(train_size(data))


@dataclass(frozen=True)
class Run:
    data: str
    width: str
    side: str  # "c" | "pytorch"
    micro_batch: int
    seed: int
    memory: bool = False
    epochs: int | None = None  # None: the paper's count

    @property
    def name(self) -> str:
        if self.memory:
            return f"{self.data}_{self.width}_mem_m{self.micro_batch}"
        return f"{self.data}_{self.width}_{self.side}_m{self.micro_batch}_s{self.seed}"

    def out_dir(self, root: Path) -> Path:
        cfg = root / f"{self.data}_{self.width}"
        if self.memory:
            return cfg / f"mem_m{self.micro_batch}"
        return cfg / f"{self.side}_m{self.micro_batch}" / f"seed{self.seed}"

    def log_path(self, root: Path) -> Path:
        return self.out_dir(root) / ("c.json" if self.side == "c" else "pytorch.json")

    def expected_epochs(self) -> int:
        return self.epochs if self.epochs is not None else paper_epochs(self.data)

    def command(self) -> list[str]:
        if self.side == "pytorch":
            return ["uv", "run", str(TWIN)]
        return [str(c_binary(MEM_PRESET if self.memory else TRAIN_PRESET))]

    def env(self, root: Path) -> dict[str, str]:
        env = dict(os.environ, KWS_DATA_DIR=str(DATA[self.data]), TCR_WIDTH=self.width,
                   TCR_SEED=str(self.seed), TCR_MICROBATCH=str(self.micro_batch),
                   TCR_OUT=str(self.out_dir(root)))
        # one thread per run: the sweep already runs one job per performance core
        env["OMP_NUM_THREADS"] = "1"
        env.pop("TCR_SAME_START", None)
        env.pop("TCR_EPOCHS", None)
        if self.epochs is not None:
            env["TCR_EPOCHS"] = str(self.epochs)
        return env

    def work(self) -> float:
        """Relative cost for longest-first ordering: epochs x samples x frames x width."""
        frames = 98 if self.data == "A" else 32
        return self.expected_epochs() * train_size(self.data) * frames * (4 if self.width == "full" else 1)


def sweep_grid() -> list[Run]:
    runs = [Run(d, w, side, m, s) for d, w in CONFIGS for side, m in ARMS for s in SEEDS]
    runs += [Run(d, w, "c", m, 0, memory=True, epochs=1)
             for d, w in CONFIGS for m in MEM_MICRO_BATCHES]
    return runs


def timing_grid() -> list[Run]:
    """One 2-epoch seed-0 run per configuration and arm, A/full/C first (the reference)."""
    return [Run(d, w, side, m, 0, epochs=TIMING_EPOCHS) for d, w in CONFIGS for side, m in ARMS]


def is_complete(run: Run, root: Path) -> bool:
    path = run.log_path(root)
    if not path.exists():
        return False
    try:
        log = load_log(path)
    except (ValueError, KeyError):
        return False  # a run killed mid-way leaves C's incrementally written log unparseable
    if "final" not in log:
        return False
    # a run stopped on a non-finite loss is a recorded result, not one to rerun
    return log["final"].get("diverged", 0) == 1 or len(log["epochs"]) == run.expected_epochs()


def select(runs: list[Run], only: str | None) -> list[Run]:
    if not only:
        return runs
    patterns = [p for p in only.split(",") if p]
    return [r for r in runs if any(fnmatch.fnmatchcase(r.name, p) for p in patterns)]


def build_c(presets: set[str]) -> None:
    for preset in sorted(presets):
        if not (REPO_ROOT / "build" / preset / "CMakeCache.txt").exists():
            subprocess.run(["cmake", "--preset", preset], cwd=REPO_ROOT, check=True)
        subprocess.run(["cmake", "--build", "--preset", preset, "--target", C_TARGET],
                       cwd=REPO_ROOT, check=True)


def execute(run: Run, root: Path) -> tuple[Run, int, float]:
    out = run.out_dir(root)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    with open(out / f"{run.side}.txt", "w") as console:
        rc = subprocess.run(run.command(), env=run.env(root), cwd=REPO_ROOT, stdout=console,
                            stderr=subprocess.STDOUT).returncode
    return run, rc, time.monotonic() - t0


def run_all(runs: list[Run], root: Path, jobs: int) -> list[tuple[Run, int, float]]:
    c_presets = {MEM_PRESET if r.memory else TRAIN_PRESET for r in runs if r.side == "c"}
    build_c(c_presets)
    results = []
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(execute, r, root) for r in runs]
        for fut in as_completed(futures):
            run, rc, wall = fut.result()
            status = "ok" if rc == 0 and is_complete(run, root) else f"FAILED rc={rc}"
            if status == "ok" and load_log(run.log_path(root))["final"].get("diverged", 0):
                status = "ok (diverged, recorded)"
            print(f"[{len(results) + 1}/{len(runs)}] {run.name}: {status} ({wall:.0f} s)",
                  flush=True)
            results.append((run, rc, wall))
    return results


def lpt_makespan(durations: list[float], workers: int) -> float:
    """Wall time of longest-first greedy scheduling on `workers` slots (what run_all does)."""
    slots = [0.0] * workers
    for d in sorted(durations, reverse=True):
        heapq.heapreplace(slots, slots[0] + d)
    return max(slots)


def time_one(only: str | None, jobs: int) -> int:
    """Time each configuration and arm solo, then project the full sweep: each run costs its
    load + test overhead once plus epochs x s/epoch. A probe with a complete log and timing.json
    is reused, so the projection can be redone without rerunning it.

    s/epoch is the fastest logged epoch: epoch 0 carries warm-up, and on macOS the C program's
    CLOCK_MONOTONIC also counts time the host spent asleep."""
    probes = select(timing_grid(), only)
    measured: dict[tuple[str, str, str, int], tuple[float, float]] = {}
    pending = [p for p in probes
               if not (is_complete(p, TIMING_LOGS) and (p.out_dir(TIMING_LOGS) / "timing.json").exists())]
    build_c({TRAIN_PRESET} if any(p.side == "c" for p in pending) else set())
    for probe in probes:  # serial: a solo number per cell
        timing = probe.out_dir(TIMING_LOGS) / "timing.json"
        if probe in pending:
            run, rc, wall = execute(probe, TIMING_LOGS)
            if rc != 0 or not is_complete(run, TIMING_LOGS):
                print(f"{run.name}: FAILED rc={rc}, see {run.out_dir(TIMING_LOGS)}")
                return 1
            timing.write_text(json.dumps({"process_wall_s": wall}))
        wall = json.loads(timing.read_text())["process_wall_s"]
        epoch_s = [e["wall_s"] for e in load_log(probe.log_path(TIMING_LOGS))["epochs"]]
        per_epoch = float(min(epoch_s))
        overhead = max(wall - len(epoch_s) * per_epoch, 0.0)
        measured[(probe.data, probe.width, probe.side, probe.micro_batch)] = (per_epoch, overhead)
        print(f"{probe.name}: wall {wall:.1f} s for {len(epoch_s)} epochs (logged "
              f"{', '.join(f'{s:.1f}' for s in epoch_s)} s), {per_epoch:.2f} s/epoch, "
              f"load+test {overhead:.1f} s", flush=True)

    print(f"\n{'configuration':<22} {'epochs':>6} {'s/epoch':>8} {'h/run':>7} {'runs':>5} "
          f"{'run-h':>7}")
    durations = []
    for run in sweep_grid():
        key = (run.data, run.width, run.side, run.micro_batch)
        if key not in measured:
            continue
        per_epoch, overhead = measured[key]
        durations.append(overhead + run.expected_epochs() * per_epoch)
    for key, (per_epoch, overhead) in measured.items():
        data, width, side, mb = key
        epochs = paper_epochs(data)
        per_run = overhead + epochs * per_epoch
        print(f"{f'{data}/{width}/{side}/m{mb}':<22} {epochs:>6} {per_epoch:>8.2f} "
              f"{per_run / 3600:>7.2f} {len(SEEDS):>5} {len(SEEDS) * per_run / 3600:>7.2f}")
    total = sum(durations)
    print(f"\ntotal {total / 3600:.1f} run-hours over {len(durations)} runs; "
          f"total/{jobs} = {total / jobs / 3600:.1f} h; "
          f"longest-first on {jobs} slots = {lpt_makespan(durations, jobs) / 3600:.1f} h "
          f"(solo s/epoch: optimistic for {jobs} concurrent jobs)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--only", help="comma-separated globs over run names, e.g. 'A_full_*'")
    parser.add_argument("--logs", type=Path, default=DEFAULT_LOGS)
    parser.add_argument("--time-one", action="store_true",
                        help=f"time {TIMING_EPOCHS} epochs per configuration and arm (A/full/c "
                             f"first) into {TIMING_LOGS.relative_to(REPO_ROOT)} and project "
                             "the sweep; runs nothing else")
    parser.add_argument("--dry-run", action="store_true", help="list the pending runs")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be >= 1")
    args.logs = args.logs.resolve()  # the runs work from the repo root

    if args.time_one:
        return time_one(args.only, args.jobs)

    runs = select(sweep_grid(), args.only)
    pending = sorted((r for r in runs if not is_complete(r, args.logs)), key=Run.work,
                     reverse=True)
    print(f"{len(runs)} runs selected, {len(runs) - len(pending)} complete, "
          f"{len(pending)} to run with {args.jobs} jobs into {args.logs}")
    if args.dry_run:
        for r in pending:
            print(f"  {r.name}")
        return 0
    results = run_all(pending, args.logs, args.jobs)
    failed = [r.name for r, rc, _ in results if rc != 0 or not is_complete(r, args.logs)]
    if failed:
        print(f"{len(failed)} runs failed: {', '.join(sorted(failed))}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
