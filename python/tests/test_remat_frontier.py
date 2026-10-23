"""examples/_shared/remat_frontier.py: the frontier table of an example, each row measured by C.

A stub stands in for the trainers: it answers DUMP_PROBLEM=1 and DRY_PLAN=1 with the committed
C goldens (python/tests/planner_goldens/, written by running the trainers), so these tests
never run C and still compare the planner against C's own lines.
"""
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from examples._shared import remat_frontier  # noqa: E402
from examples._shared.remat_frontier import (COLUMNS, FrontierError, main, measure,  # noqa: E402
                                             run_trainer)

GOLDENS = REPO_ROOT / "python/tests/planner_goldens"
GOLDEN_OF = {"har_classifier": "har", "har_classifier_finetune": "har_finetune",
             "mnist_cnn": "mnist_cnn"}


class GoldenTrainer:
    """Answers each run the way the trainer did when the goldens were written; `edit` changes
    a DRY_PLAN line, `refuse` answers DRY_PLAN runs with C's over-budget exit."""

    def __init__(self, example: str, edit=None, refuse: str | None = None, hash_line=None):
        self.golden, self.calls = GOLDEN_OF[example], []
        self.edit, self.refuse, self.hash_line = edit or (lambda line: line), refuse, hash_line

    def __call__(self, example: str, variables: dict[str, str]) -> tuple[int, str, str]:
        self.calls.append((example, variables))
        if variables == {"DUMP_PROBLEM": "1"}:
            problem = (GOLDENS / f"{self.golden}.problem.json").read_text()
            hash_line = self.hash_line or (GOLDENS / f"{self.golden}.problem.hash").read_text()
            return 0, problem, hash_line
        if self.refuse is not None:
            return 1, "", self.refuse
        for line in (GOLDENS / f"{self.golden}.dry_plan.txt").read_text().splitlines(True):
            if f" plan={variables['REMAT_PLAN']} " in line:
                return 0, self.edit(line), ""
        raise AssertionError(f"no golden DRY_PLAN line for {variables}")


# HAR with in-place allowed: (plan, budget_b = arena_b, arena_gap_b, wires_peak_b,
# wire_metadata_b, recompute_cost, refwd_steps, steps). Derived twice: C's DRY_PLAN lines
# (the goldens) and the planner's own numbers, which the script requires to agree.
HAR = [
    ("retain:1,3,4,6,7,10,12+inplace", 41216, 0, 41216, 5576, 0, 0, 25),
    ("retain:1,4,6,7,10,12+inplace", 37120, 0, 37120, 5616, 4096, 2, 27),
    ("retain:1,4,7,10,12+inplace", 36864, 0, 36864, 5656, 8192, 4, 29),
    ("retain:3,4,6,7,10,12+inplace", 33024, 0, 33024, 5596, 129024, 1, 26),
    ("retain:4,6,7,10,12+inplace", 28928, 0, 28928, 5636, 133120, 3, 28),
    ("retain:4,7,10,12+inplace", 28672, 0, 28672, 5676, 137216, 5, 30),
    ("retain:3,6,7,10,12+inplace", 24832, 0, 24832, 5616, 292864, 2, 27),
    ("retain:3,6,10,12+inplace", 24576, 0, 24576, 5636, 489472, 3, 28),
]


def test_every_row_is_cs_dry_plan_line_with_the_planners_numbers_alongside():
    trainer = GoldenTrainer("har_classifier")
    rows = measure("har_classifier", True, trainer)
    assert [(r["plan"], r["budget_b"], r["arena_gap_b"], r["wires_peak_b"], r["wire_metadata_b"],
             r["recompute_cost"], r["refwd_steps"], r["steps"]) for r in rows] == HAR
    for r in rows:
        assert r["arena_b"] == r["budget_b"]
        assert [r[f"py_{k}"] for k in ("arena_b", "wires_peak_b", "recompute_cost",
                                       "refwd_steps", "steps")] == [
            r[k] for k in ("arena_b", "wires_peak_b", "recompute_cost", "refwd_steps", "steps")]
    assert trainer.calls[0] == ("har_classifier", {"DUMP_PROBLEM": "1"})
    assert trainer.calls[1:] == [("har_classifier", {
        "REMAT_STORAGE": "arena", "REMAT_PLAN": plan, "REMAT_BUDGET_B": str(budget),
        "DRY_PLAN": "1"}) for plan, budget, *_ in HAR]


@pytest.mark.parametrize(("example", "allow_inplace", "points"), [
    ("har_classifier", True, 8), ("har_classifier", False, 5),
    ("har_classifier_finetune", True, 1), ("har_classifier_finetune", False, 1),
    ("mnist_cnn", True, 4), ("mnist_cnn", False, 3)])
def test_c_places_every_frontier_plan_of_every_example_at_its_peak(example, allow_inplace,
                                                                   points):
    rows = measure(example, allow_inplace, GoldenTrainer(example))
    assert len(rows) == points
    assert all(r["arena_b"] == r["budget_b"] and r["arena_gap_b"] == 0 for r in rows)
    assert all(r["plan"].endswith("+inplace") == allow_inplace for r in rows) or (
        example == "har_classifier_finetune")


def test_a_c_number_the_planner_did_not_predict_stops_naming_the_row():
    trainer = GoldenTrainer("har_classifier",
                            edit=lambda line: line.replace(" arena_b=37120 ", " arena_b=37128 "))
    with pytest.raises(FrontierError, match=re.escape(
            "har_classifier row 1 (retain:1,4,6,7,10,12+inplace, REMAT_BUDGET_B=37120): "
            "C arena_b=37128, the planner predicted 37120")):
        measure("har_classifier", True, trainer)


def test_a_plan_c_refuses_at_its_budget_stops_naming_the_row_and_cs_message():
    refusal = "ERROR: remat arena 41224 B exceeds REMAT_BUDGET_B=41216 B\n"
    with pytest.raises(FrontierError, match=re.escape(
            "har_classifier row 0 (retain:1,3,4,6,7,10,12+inplace, REMAT_BUDGET_B=41216): "
            "C exited 1: ERROR: remat arena 41224 B exceeds REMAT_BUDGET_B=41216 B")):
        measure("har_classifier", True, GoldenTrainer("har_classifier", refuse=refusal))


def test_a_problem_hash_c_did_not_print_stops_before_any_plan_runs():
    trainer = GoldenTrainer("mnist_cnn", hash_line="PROBLEM_HASH fnv1a64=0000000000000000\n")
    with pytest.raises(FrontierError, match=re.escape(
            "mnist_cnn: the problem file hashes to 6a814ff6da6cae50, but C printed "
            "'PROBLEM_HASH fnv1a64=0000000000000000'")):
        measure("mnist_cnn", True, trainer)
    assert len(trainer.calls) == 1


def test_the_csv_has_cs_columns_then_the_planners(tmp_path):
    assert COLUMNS == ("budget_b", "plan", "arena_b", "arena_gap_b", "wires_peak_b",
                       "wire_metadata_b", "recompute_cost", "refwd_steps", "steps",
                       "py_arena_b", "py_wires_peak_b", "py_recompute_cost", "py_refwd_steps",
                       "py_steps")
    out = tmp_path / "har.csv"
    assert main(["--example", "har_classifier", "--out", str(out)],
                GoldenTrainer("har_classifier")) == 0
    lines = out.read_text().splitlines()
    assert lines[0] == ",".join(COLUMNS)
    # The plan holds commas, so the CSV quotes it.
    assert lines[1] == ('41216,"retain:1,3,4,6,7,10,12+inplace",41216,0,41216,5576,0,0,25,'
                        "41216,41216,0,0,25")
    assert len(lines) == 1 + len(HAR)


def test_a_failure_is_one_stderr_line_and_exit_1_with_no_csv(tmp_path, capsys):
    out = tmp_path / "har.csv"
    trainer = GoldenTrainer("har_classifier", refuse="ERROR: remat arena 41224 B exceeds "
                            "REMAT_BUDGET_B=41216 B\n")
    assert main(["--example", "har_classifier", "--out", str(out)], trainer) == 1
    assert capsys.readouterr().err == (
        "remat_frontier: har_classifier row 0 (retain:1,3,4,6,7,10,12+inplace, "
        "REMAT_BUDGET_B=41216): C exited 1: ERROR: remat arena 41224 B exceeds "
        "REMAT_BUDGET_B=41216 B\n")
    assert not out.exists()


def test_a_missing_trainer_names_the_binary_and_the_build_command(tmp_path, monkeypatch):
    monkeypatch.setattr(remat_frontier, "BUILD", tmp_path)
    with pytest.raises(FrontierError, match=r"does not exist: build the examples preset first "
                                           r"\(cmake --build --preset examples\)$"):
        run_trainer("har_classifier", {})


def test_a_trainer_past_the_timeout_is_one_named_error(tmp_path, monkeypatch):
    monkeypatch.setattr(remat_frontier, "BUILD", tmp_path)
    binary = tmp_path / remat_frontier.TRAINERS["har_classifier"]
    binary.parent.mkdir(parents=True)
    binary.touch()

    def hang(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", hang)
    with pytest.raises(FrontierError, match=rf"{re.escape(str(binary))} ran past 3600 s"):
        run_trainer("har_classifier", {})
