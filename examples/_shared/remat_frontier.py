"""The frontier table of an example trainer: every arena budget at which the planner's best
retain plan changes, each row measured by C.

    uv run examples/_shared/remat_frontier.py --example har_classifier --out frontier.csv

Run from the repository root after `cmake --build --preset examples`, with the example's data
prepared. The trainer writes its problem (DUMP_PROBLEM=1), the planner walks the frontier, and
the trainer runs every frontier plan under DRY_PLAN=1 with REMAT_BUDGET_B set to the plan's
arena, so C checks that the arena fits. The C columns are C's numbers; the py_ columns are the
planner's prediction for the same plan, and any difference, or a plan C refuses, stops the
script naming the row. Nothing is trained.
"""
import argparse
import csv
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))  # the planner package, when it is not installed

from odt.planner import Plan, PlanSpace, Problem, ProblemError, UnsupportedProblem  # noqa: E402
from odt.planner import parse_problem  # noqa: E402
from odt.planner.frontier import frontier  # noqa: E402

BUILD = ROOT / "build/examples/examples"
TRAINERS = {
    "har_classifier": "har_classifier/train_c_har_classifier",
    "har_classifier_finetune": "har_classifier/train_c_har_classifier_finetune",
    "mnist_cnn": "mnist_cnn/train_c_mnist_cnn",
}
# Every C number of a row; CHECKED are those the planner predicts and must match.
C_COLUMNS = ("arena_b", "arena_gap_b", "wires_peak_b", "wire_metadata_b", "recompute_cost",
             "refwd_steps", "steps")
CHECKED = ("arena_b", "wires_peak_b", "recompute_cost", "refwd_steps", "steps")
COLUMNS = ("budget_b", "plan", *C_COLUMNS, *(f"py_{k}" for k in CHECKED))

# (example, variables) -> (exit code, stdout, stderr) of one trainer run.
Runner = Callable[[str, dict[str, str]], tuple[int, str, str]]


class FrontierError(Exception):
    """A row C does not confirm, or a trainer run that failed; the message names it."""


def run_trainer(example: str, variables: dict[str, str], cwd: Path = ROOT) -> tuple[int, str, str]:
    """One trainer run with only PATH, HOME and `variables` set, so a variable left in the
    caller's shell (EPOCHS, REMAT_PLAN, ...) cannot change the run."""
    binary = BUILD / TRAINERS[example]
    if not binary.is_file():
        raise FrontierError(f"{binary} does not exist: build the examples preset first "
                            f"(cmake --build --preset examples)")
    env = {"PATH": os.environ.get("PATH", ""), "HOME": os.environ.get("HOME", ""), **variables}
    done = subprocess.run([str(binary)], cwd=cwd, env=env, capture_output=True, timeout=3600)
    return done.returncode, done.stdout.decode(), done.stderr.decode()


def problem_of(example: str, runner: Runner = run_trainer) -> Problem:
    """The trainer's problem, as DUMP_PROBLEM=1 writes it; its hash must be the one C printed."""
    rc, out, err = runner(example, {"DUMP_PROBLEM": "1"})
    if rc != 0:
        raise FrontierError(f"{example}: DUMP_PROBLEM=1 exited {rc}: {err.strip()}")
    problem = parse_problem(out.encode())
    if err.strip() != f"PROBLEM_HASH fnv1a64={problem.hash}":
        raise FrontierError(f"{example}: the problem file hashes to {problem.hash}, but C "
                            f"printed {err.strip()!r}")
    return problem


def dry_plan_fields(line: str) -> dict[str, str]:
    """`DRY_PLAN key=value key=value ...` as a dict of strings."""
    return dict(field.split("=", 1) for field in line.split()[1:])


def measure_point(example: str, row: int, budget: int, plan: Plan, runner: Runner) -> dict:
    name = plan.to_remat_plan()
    where = f"{example} row {row} ({name}, REMAT_BUDGET_B={budget})"
    rc, out, err = runner(example, {"REMAT_STORAGE": "arena", "REMAT_PLAN": name,
                                    "REMAT_BUDGET_B": str(budget), "DRY_PLAN": "1"})
    if rc != 0:
        raise FrontierError(f"{where}: C exited {rc}: {err.strip()}")
    lines = out.splitlines()
    if len(lines) != 1 or not lines[0].startswith("DRY_PLAN "):
        raise FrontierError(f"{where}: expected one DRY_PLAN line, C printed {out!r}")
    c = dry_plan_fields(lines[0])
    if (c.get("example"), c.get("plan")) != (example, name):
        raise FrontierError(f"{where}: C ran example={c.get('example')} plan={c.get('plan')}")
    for key in CHECKED:
        if int(c[key]) != getattr(plan, key):
            raise FrontierError(f"{where}: C {key}={c[key]}, the planner predicted "
                                f"{getattr(plan, key)}")
    return {"budget_b": budget, "plan": name, **{k: int(c[k]) for k in C_COLUMNS},
            **{f"py_{k}": getattr(plan, k) for k in CHECKED}}


def measure(example: str, allow_inplace: bool = True, runner: Runner = run_trainer) -> list[dict]:
    """One row per frontier point, largest budget first; budget_b is the point's arena."""
    space = PlanSpace.build(problem_of(example, runner), 1, allow_inplace)
    return [measure_point(example, row, point.budget_lo_b, point.plan, runner)
            for row, point in enumerate(frontier(space))]


def write_csv(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None, runner: Runner = run_trainer) -> int:
    parser = argparse.ArgumentParser(prog="remat_frontier.py", description=__doc__.split("\n\n")[0])
    parser.add_argument("--example", required=True, choices=sorted(TRAINERS))
    parser.add_argument("--no-inplace", action="store_true", help="only plans without in-place dx")
    parser.add_argument("--out", required=True, type=Path, help="the CSV file to write")
    args = parser.parse_args(argv)
    try:
        rows = measure(args.example, not args.no_inplace, runner)
    except (FrontierError, ProblemError, UnsupportedProblem) as e:
        print(f"remat_frontier: {e}", file=sys.stderr)
        return 1
    write_csv(rows, args.out)
    print(f"remat_frontier: {args.example}: frontier points {len(rows)}, each measured by C, "
          f"written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
