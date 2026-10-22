"""Rewrites the planner's C reference outputs (goldens) from the built example trainers.

Run after `cmake --build --preset examples`, with the HAR and mnist_cnn data prepared (the
trainers load their dataset before they dump or plan; they run from the repository root):

    uv run python -B python/tests/planner_goldens/regenerate.py

Writes, per model, `<model>.problem.json` (the trainer's stdout under DUMP_PROBLEM=1),
`<model>.problem.hash` (its stderr line) and `<model>.dry_plan.txt` (one DRY_PLAN line per
plan of a fixed sample and of the planner's frontier, REMAT_STORAGE=arena). These files are
C output: never edit them by hand.
"""
import json
import os
import random
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BUILD = ROOT / "build/examples/examples"
TRAINERS = {
    "har": BUILD / "har_classifier/train_c_har_classifier",
    "har_finetune": BUILD / "har_classifier/train_c_har_classifier_finetune",
    "mnist_cnn": BUILD / "mnist_cnn/train_c_mnist_cnn",
}


def run_trainer(model: str, variables: dict[str, str]) -> subprocess.CompletedProcess:
    """One trainer run with only PATH, HOME and the given variables set."""
    env = {"PATH": os.environ.get("PATH", ""), "HOME": os.environ.get("HOME", ""), **variables}
    done = subprocess.run([str(TRAINERS[model])], cwd=ROOT, env=env, capture_output=True,
                          timeout=600)
    if done.returncode != 0:
        sys.exit(f"regenerate: {model} {variables} exited {done.returncode}: "
                 f"{done.stderr.decode(errors='replace').strip()}")
    return done


def write_problem(model: str) -> None:
    done = run_trainer(model, {"DUMP_PROBLEM": "1"})
    (HERE / f"{model}.problem.json").write_bytes(done.stdout)
    (HERE / f"{model}.problem.hash").write_bytes(done.stderr)


def sample_plans(model: str, n: int) -> list[str]:
    """Per in-place setting: keep all, keep only ACT n, each inner ACT kept alone, and 32
    seeded random masks (ACT n is always kept)."""
    plans = []
    for suffix in ("", "+inplace"):
        rng = random.Random(f"{model}{suffix}")
        masks = [set(range(1, n + 1)), {n}] + [{j, n} for j in range(1, n)]
        masks += [{j for j in range(1, n) if rng.random() < 0.5} | {n} for _ in range(32)]
        for kept in masks:
            plan = "retain:" + ",".join(map(str, sorted(kept))) + suffix
            if plan not in plans:
                plans.append(plan)
    return plans


def write_dry_plans(model: str, plans: list[str]) -> None:
    lines = []
    for plan in plans:
        done = run_trainer(model, {"REMAT_STORAGE": "arena", "REMAT_PLAN": plan, "DRY_PLAN": "1"})
        line = done.stdout.decode()
        if not line.startswith("DRY_PLAN ") or line.count("\n") != 1:
            sys.exit(f"regenerate: {model} {plan}: expected one DRY_PLAN line, got {line!r}")
        lines.append(line)
    (HERE / f"{model}.dry_plan.txt").write_text("".join(lines))


def frontier_plans(model: str) -> list[str]:
    """Every frontier plan the planner computes from the problem just written, with and
    without in-place dx: the parity tests need C's line for each. The planner chooses these
    plans itself; the FRONTIER table in test_planner_frontier.py, pinned from a second
    derivation, is what keeps that honest, so never replace it with this function's output."""
    sys.path.insert(0, str(ROOT / "python"))
    from odt.planner import PlanSpace, load_problem
    from odt.planner.frontier import frontier

    problem = load_problem(HERE / f"{model}.problem.json")
    return [point.plan.to_remat_plan() for allow in (True, False)
            for point in frontier(PlanSpace.build(problem, 1, allow))]


def main() -> None:
    for model in TRAINERS:
        write_problem(model)
        n = len(json.loads((HERE / f"{model}.problem.json").read_bytes())["layers"])
        plans = sample_plans(model, n)
        plans += [p for p in frontier_plans(model) if p not in plans]
        write_dry_plans(model, plans)
        print(f"regenerate: {model}: problem and {len(plans)} DRY_PLAN lines written")


if __name__ == "__main__":
    main()
