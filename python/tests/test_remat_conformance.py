"""examples/_shared/remat_conformance.py: every frontier plan's run log and test predictions
equal STORE_ALL's.

A fake trainer stands in for C: it writes the files a training run writes into its working
directory, so these tests check the plan list, the per-run directories, the environment of
each run and the comparison, never C itself (the script's recorded run does that).
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from examples._shared.remat_conformance import check, differences, main, plans_of  # noqa: E402
from tests.test_remat_frontier import GoldenTrainer  # noqa: E402

LOG = {"impl": "c", "example": "har_classifier",
       "config": {"epochs": 2, "batch": 64, "seed": 42, "remat_storage": "heap",
                  "remat_plan": "store_all"},
       "epochs": [{"epoch": 0, "step_losses": [], "train_loss": 1.660617, "val_loss": 1.303707,
                   "val_acc": 0.54966, "wall_s": 30.927},
                  {"epoch": 1, "step_losses": [], "train_loss": 1.2, "val_loss": 1.1,
                   "val_acc": 0.6, "wall_s": 31.5}],
       "final": {"test_loss": 1.344172, "test_acc": 0.514082, "test_auc": None}}


def changed(**edits) -> dict:
    log = json.loads(json.dumps(LOG))
    for path, value in edits.items():
        node, *keys = path.split("__")
        target = log[node]
        for key in keys[:-1]:
            target = target[int(key)] if key.isdigit() else target[key]
        target[int(keys[-1]) if keys[-1].isdigit() else keys[-1]] = value
    return log


def test_runs_that_differ_only_in_wall_time_and_the_remat_keys_conform():
    run = changed(epochs__0__wall_s=29.1, epochs__1__wall_s=33.0)
    run["config"].update({"remat_storage": "arena", "remat_plan": "retain:3,6,10,12+inplace",
                          "wire_budget_b": 24576, "wire_cost_model": "mac"})
    assert differences(LOG, run) == []


def test_every_number_that_differs_is_named_with_its_place():
    run = changed(epochs__1__val_loss=1.100001, final__test_acc=0.514083, config__seed=7)
    assert differences(LOG, run) == [
        "config.seed: 42 (STORE_ALL) vs 7", "epochs[1].val_loss: 1.1 (STORE_ALL) vs 1.100001",
        "final.test_acc: 0.514082 (STORE_ALL) vs 0.514083"]
    assert differences(LOG, {**LOG, "epochs": LOG["epochs"][:1]}) == [
        "epochs: 2 (STORE_ALL) vs 1"]


def test_the_plans_are_both_frontiers_each_once_with_their_arenas():
    plans = plans_of("har_classifier", GoldenTrainer("har_classifier"))
    assert len(plans) == 13 and len(set(plans)) == 13
    assert plans[0] == ("retain:1,3,4,6,7,10,12+inplace", 41216)
    assert plans[8] == ("retain:1,3,4,6,7,10,12", 49152)
    assert [p for p, _ in plans_of("mnist_cnn", GoldenTrainer("mnist_cnn"))] == [
        "retain:1,3,4,8,10+inplace", "retain:1,4,8,10+inplace", "retain:3,4,8,10+inplace",
        "retain:3,8,10+inplace", "retain:1,3,4,8,10", "retain:1,4,8,10", "retain:3,4,8,10"]


class FakeTraining:
    """Writes the run log and the predictions file into the run's working directory; the run
    of plan `drift` computes one different loss, that of `other_predictions` one other
    prediction."""

    def __init__(self, drift: str | None = None, other_predictions: str | None = None):
        self.drift, self.other_predictions = drift, other_predictions
        self.runs, self.golden = [], GoldenTrainer("mnist_cnn")

    def __call__(self, example, variables, cwd):
        if variables == {"DUMP_PROBLEM": "1"}:
            return self.golden(example, variables)
        self.runs.append((variables, cwd))
        assert np.load(cwd / "examples/mnist_cnn/data/train_x.npy").shape == (4, 1, 28, 28)
        log = changed(epochs__0__wall_s=random.random())
        if variables.get("REMAT_PLAN") == self.drift:
            log["epochs"][0]["val_loss"] = 1.303708
        (cwd / "examples/mnist_cnn/logs/c.json").write_text(json.dumps(log))
        predictions = b"NPY 3 1 4 2" if variables.get("REMAT_PLAN") == self.other_predictions \
            else b"NPY 3 1 4 1"
        (cwd / "examples/mnist_cnn/outputs/c_predictions.npy").write_bytes(predictions)
        return 0, "", ""


@pytest.fixture
def data(tmp_path):
    src = tmp_path / "data"
    src.mkdir()
    for part in ("train", "val", "test"):
        np.save(src / f"{part}_x.npy", np.zeros((10, 1, 28, 28), np.float32))
        np.save(src / f"{part}_y.npy", np.arange(10, dtype=np.int32))
    return src


def test_each_run_has_its_own_directory_and_every_plan_runs_at_its_budget(tmp_path, data):
    trainer = FakeTraining()
    report = check("mnist_cnn", tmp_path / "work", 4, 2, 1, trainer, data)
    assert report == [("mnist_cnn", plan, []) for plan, _ in plans_of(
        "mnist_cnn", GoldenTrainer("mnist_cnn"))]
    assert trainer.runs[0][0] == {"REMAT_PLAN": "store_all"}
    assert [v for v, _ in trainer.runs[1:]] == [
        {"REMAT_STORAGE": "arena", "REMAT_PLAN": plan, "REMAT_BUDGET_B": str(budget)}
        for plan, budget in plans_of("mnist_cnn", GoldenTrainer("mnist_cnn"))]
    assert len({cwd for _, cwd in trainer.runs}) == 8


def test_a_plan_whose_test_predictions_differ_does_not_conform(tmp_path, data):
    trainer = FakeTraining(other_predictions="retain:3,8,10+inplace")
    report = check("mnist_cnn", tmp_path / "work", 4, 2, 1, trainer, data)
    assert [found for _, plan, found in report if found] == [
        ["outputs/c_predictions.npy: the bytes differ"]]
    assert report[3][1] == "retain:3,8,10+inplace"


def test_a_plan_whose_training_differs_is_reported_and_the_exit_is_1(tmp_path, data, capsys,
                                                                    monkeypatch):
    monkeypatch.setattr("examples._shared.remat_conformance.data_dir_of", lambda example: data)
    argv = ["--example", "mnist_cnn", "--work", str(tmp_path / "work"), "--samples", "4"]
    assert main(argv, FakeTraining(drift="retain:1,4,8,10")) == 1
    out = capsys.readouterr().out.splitlines()
    assert "DIFFERS mnist_cnn retain:1,4,8,10: epochs[0].val_loss: 1.303707 (STORE_ALL) vs " \
           "1.303708" in out
    assert out[-1] == "remat_conformance: 1 of 7 plans differ from STORE_ALL"
    assert main(argv, FakeTraining()) == 2  # the work directory exists now
    assert capsys.readouterr().err == (
        f"remat_conformance: --work {tmp_path / 'work'} exists: give a new directory, so no "
        f"earlier run's files are compared\n")


def test_a_relative_work_directory_conforms_and_says_what_was_compared(tmp_path, data, capsys,
                                                                        monkeypatch):
    monkeypatch.setattr("examples._shared.remat_conformance.data_dir_of", lambda example: data)
    monkeypatch.chdir(tmp_path)
    trainer = FakeTraining()
    assert main(["--example", "mnist_cnn", "--work", "work", "--samples", "4"], trainer) == 0
    assert all(cwd.is_absolute() for _, cwd in trainer.runs)
    assert capsys.readouterr().out.splitlines()[-1] == (
        "remat_conformance: all 7 plans: run log and test predictions equal STORE_ALL's")


@pytest.mark.parametrize("flag, value", [("--jobs", "0"), ("--samples", "0"),
                                         ("--samples", "-1"), ("--epochs", "0")])
def test_a_count_below_one_is_refused_before_any_work(tmp_path, capsys, flag, value):
    def runner(*_):
        pytest.fail("a trainer ran although an argument was refused")

    with pytest.raises(SystemExit) as stop:
        main(["--example", "mnist_cnn", "--work", str(tmp_path / "work"), flag, value], runner)
    assert stop.value.code == 2
    assert not (tmp_path / "work").exists()
    assert flag in capsys.readouterr().err
