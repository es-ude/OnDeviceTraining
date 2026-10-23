"""Every frontier plan trains like STORE_ALL (run log and test predictions equal): local, not CI.

    uv run examples/_shared/remat_conformance.py --work <new dir> [--example mnist_cnn]
        [--samples 512] [--epochs 2] [--jobs 8]

Run from the repository root after `cmake --build --preset examples` (the plain preset, never a
build with other optimisation flags: the predictions check compares file bytes, which a
build with other flags changes), with the data prepared. Per
example (default: har_classifier and mnist_cnn) the planner's frontier with and without
in-place dx gives the plans, each once. One reference run uses REMAT_PLAN=store_all; each
plan's run uses REMAT_STORAGE=arena, REMAT_PLAN=<plan> and REMAT_BUDGET_B=<its arena>, so C
also checks the budget in a real training run. Every run has its own directory under --work,
because the trainers read and write fixed paths below their working directory; its data is
the first --samples samples of each split, which keeps a run short (mnist_cnn always trains
10 epochs; har_classifier trains --epochs). A plan conforms when its run log has the
reference's numbers in config, every epoch and final (wall-clock times and the remat config
keys aside) and its test predictions file has the same bytes. One line per plan; the exit is
1 if any plan differs or a run fails. The log holds six decimals, so this is not a byte-level
check: the byte-level proof that a retain plan computes STORE_ALL's gradients is the C
conformance suite (remat D9 in docs/conventions/remat-scheduler.md).
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples._shared.log_schema import load_log  # noqa: E402
from examples._shared.remat_frontier import FrontierError, problem_of, run_trainer  # noqa: E402
from odt.planner import PlanSpace  # noqa: E402  (remat_frontier put the package on the path)
from odt.planner.frontier import frontier  # noqa: E402

EXAMPLES = ("har_classifier", "mnist_cnn")
SPLITS = [f"{part}_{xy}.npy" for part in ("train", "val", "test") for xy in ("x", "y")]
# The scheme a run chose; the reference and the plans differ in these by design.
REMAT_CONFIG_KEYS = ("remat_storage", "remat_plan", "wire_budget_b", "wire_cost_model")


class ConformanceError(Exception):
    """The reference run failed, so no plan can be compared; the message names it."""


class MissingOutput(Exception):
    """A run that exited 0 left out the file (relative path) in the message; `check` words it
    for the reference or the plan."""


def data_dir_of(example: str) -> Path:
    return ROOT / "examples" / example / "data"


def plans_of(example: str, runner=run_trainer) -> list[tuple[str, int]]:
    """(plan, its arena) for every frontier point, in-place allowed first, each plan once."""
    problem, plans = problem_of(example, runner), {}
    for allow_inplace in (True, False):
        for point in frontier(PlanSpace.build(problem, 1, allow_inplace)):
            plans.setdefault(point.plan.to_remat_plan(), point.budget_lo_b)
    return list(plans.items())


def differences(reference: dict, run: dict) -> list[str]:
    """Every number of `run` that is not the reference's, named by its place in the log."""
    out = []

    def compare(where: str, a: dict, b: dict, skip: tuple[str, ...] = ()) -> None:
        for key in sorted((a.keys() | b.keys()) - set(skip)):
            if a.get(key) != b.get(key):
                out.append(f"{where}{key}: {a.get(key)} (STORE_ALL) vs {b.get(key)}")

    compare("config.", reference["config"], run["config"], REMAT_CONFIG_KEYS)
    if len(reference["epochs"]) != len(run["epochs"]):
        out.append(f"epochs: {len(reference['epochs'])} (STORE_ALL) vs {len(run['epochs'])}")
    else:
        for i, (a, b) in enumerate(zip(reference["epochs"], run["epochs"])):
            compare(f"epochs[{i}].", a, b, ("wall_s",))
    compare("final.", reference["final"], run["final"])
    return out


def _prepare(example: str, work: Path, samples: int, data_dir: Path, runs: int) -> list[Path]:
    """The shortened data once, then one directory per run that sees it as its data/."""
    data = work / example / "data"
    data.mkdir(parents=True)
    for name in SPLITS:
        first = np.load(data_dir / name, mmap_mode="r")[:samples]
        np.save(data / name, np.ascontiguousarray(first))
    dirs = []
    for k in range(runs):
        cwd = work / example / f"run-{k}"
        for sub in ("logs", "outputs"):
            (cwd / "examples" / example / sub).mkdir(parents=True)
        (cwd / "examples" / example / "data").symlink_to(data)
        dirs.append(cwd)
    return dirs


def check(example: str, work: Path, samples: int, epochs: int, jobs: int, runner=run_trainer,
          data_dir: Path | None = None) -> list[tuple[str, str, list[str]]]:
    """(example, plan, differences) per plan; no differences = run log and test predictions
    equal STORE_ALL's."""
    plans = plans_of(example, lambda e, v: runner(e, v, ROOT))
    extra = {"EPOCHS": str(epochs)} if example == "har_classifier" else {}
    variables = [{"REMAT_PLAN": "store_all", **extra}] + [
        {"REMAT_STORAGE": "arena", "REMAT_PLAN": plan, "REMAT_BUDGET_B": str(budget), **extra}
        for plan, budget in plans]
    # Absolute, so each run's data/ symlink and working directory hold wherever it starts.
    work, data_dir = work.absolute(), (data_dir or data_dir_of(example)).absolute()
    dirs = _prepare(example, work, samples, data_dir, len(variables))
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        done = list(pool.map(lambda kv: runner(example, *kv), zip(variables, dirs)))

    def outputs(k: int) -> tuple[dict, bytes]:
        base = dirs[k] / "examples" / example
        for relpath in ("logs/c.json", "outputs/c_predictions.npy"):
            if not (base / relpath).is_file():
                raise MissingOutput(relpath)
        return (load_log(base / "logs/c.json"),
                (base / "outputs/c_predictions.npy").read_bytes())

    if done[0][0] != 0:
        raise ConformanceError(f"{example}: the STORE_ALL run exited {done[0][0]}: "
                               f"{done[0][2].strip()}")
    try:
        ref_log, ref_predictions = outputs(0)
    except MissingOutput as e:
        raise ConformanceError(f"{example}: the STORE_ALL run exited 0 but wrote no {e}") from None
    report = []
    for k, (plan, _) in enumerate(plans, start=1):
        rc, _, err = done[k]
        if rc != 0:
            report.append((example, plan, [f"the run exited {rc}: {err.strip()}"]))
            continue
        try:
            log, predictions = outputs(k)
        except MissingOutput as e:
            report.append((example, plan, [f"the run exited 0 but wrote no {e}"]))
            continue
        found = differences(ref_log, log)
        if predictions != ref_predictions:
            found.append("outputs/c_predictions.npy: the bytes differ")
        report.append((example, plan, found))
    return report


def _at_least_one(text: str) -> int:
    """A count for argparse: below one, no plan would be tested (or the pool cannot start)."""
    try:
        n = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a whole number, not {text!r}") from None
    if n < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, not {n}")
    return n


def main(argv: list[str] | None = None, runner=run_trainer) -> int:
    parser = argparse.ArgumentParser(prog="remat_conformance.py",
                                     description=__doc__.split("\n\n")[0])
    parser.add_argument("--work", required=True, type=Path, help="a new directory for the runs")
    parser.add_argument("--example", action="append", choices=EXAMPLES,
                        help="repeatable; default: both")
    parser.add_argument("--samples", type=_at_least_one, default=512,
                        help="samples per split (512)")
    parser.add_argument("--epochs", type=_at_least_one, default=2,
                        help="har_classifier epochs (2)")
    parser.add_argument("--jobs", type=_at_least_one, default=1, help="runs at once (1)")
    args = parser.parse_args(argv)
    if args.work.exists():
        print(f"remat_conformance: --work {args.work} exists: give a new directory, so no "
              f"earlier run's files are compared", file=sys.stderr)
        return 2
    report = []
    try:
        for example in args.example or EXAMPLES:
            report += check(example, args.work, args.samples, args.epochs, args.jobs, runner)
    except (ConformanceError, FrontierError) as e:
        print(f"remat_conformance: {e}", file=sys.stderr)
        return 1
    for example, plan, found in report:
        print(f"DIFFERS {example} {plan}: {'; '.join(found)}" if found
              else f"conforms {example} {plan}")
    bad = sum(1 for *_, found in report if found)
    print(f"remat_conformance: {bad} of {len(report)} plans differ from STORE_ALL" if bad else
          f"remat_conformance: all {len(report)} plans: run log and test predictions equal "
          f"STORE_ALL's")
    (args.work / "report.json").write_text(json.dumps(report, indent=1))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
