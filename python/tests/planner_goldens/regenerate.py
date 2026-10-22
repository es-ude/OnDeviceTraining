"""Rewrites the planner's C reference outputs (goldens) from the built example trainers.

Run after `cmake --build --preset examples`, with the HAR and mnist_cnn data prepared (the
trainers load their dataset before they dump or plan; they run from the repository root):

    uv run python -B python/tests/planner_goldens/regenerate.py

Writes, per model, `<model>.problem.json` (the trainer's stdout under DUMP_PROBLEM=1) and
`<model>.problem.hash` (its stderr line). These files are C output: never edit them by hand.
"""
import os
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


def main() -> None:
    for model in TRAINERS:
        write_problem(model)
        print(f"regenerate: {model}: problem written")


if __name__ == "__main__":
    main()
