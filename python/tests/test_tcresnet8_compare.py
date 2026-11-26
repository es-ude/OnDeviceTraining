"""The TC-ResNet8 hypothesis test (examples/kws_tcresnet8/compare.py): the C and PyTorch
test-accuracy means over 10 seeds differ by less than one pooled standard deviation."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from examples.kws_tcresnet8.compare import gap_within_pooled_std  # noqa: E402


def test_gap_verdict():
    # Both sets share these deviations, so the pooled standard deviation is their std alone.
    deviations = np.array([-0.9, -0.6, -0.4, -0.2, -0.1, 0.1, 0.2, 0.4, 0.6, 0.9]) * 0.01
    s = float(np.std(deviations, ddof=1))
    py = 0.95 + deviations
    assert gap_within_pooled_std(py + 0.5 * s, py)
    assert not gap_within_pooled_std(py + 1.5 * s, py)
    assert not gap_within_pooled_std(py - 1.5 * s, py)


def _write_log(root, run, epochs_logged, diverged):
    from examples._shared.log_schema import dump_log

    epoch = {"epoch": 0, "step_losses": [], "train_loss": 1.0, "val_loss": 1.0, "val_acc": 0.5,
             "wall_s": 1.0}
    run.out_dir(root).mkdir(parents=True)
    dump_log(run.log_path(root), {
        "impl": "c", "example": "kws_tcresnet8", "config": {"epochs": 5},
        "epochs": [dict(epoch, epoch=i) for i in range(epochs_logged)],
        "final": {"test_loss": 1.0, "test_acc": 0.5, "test_auc": None, "diverged": diverged,
                  "epochs_completed": epochs_logged}})


def test_diverged_run_is_complete_but_a_short_one_is_not(tmp_path):
    # A run stopped on a non-finite loss is a recorded result: the sweep must not rerun it.
    from examples.kws_tcresnet8.run_sweep import Run, is_complete

    diverged = Run("B", "half", "c", 100, 0, epochs=5)
    short = Run("B", "half", "c", 100, 1, epochs=5)
    _write_log(tmp_path, diverged, epochs_logged=2, diverged=1)
    _write_log(tmp_path, short, epochs_logged=2, diverged=0)
    assert is_complete(diverged, tmp_path)
    assert not is_complete(short, tmp_path)


def _final_logs(accs, diverged_accs=()):
    """(seed, log) pairs: finished runs with these test accuracies, then diverged ones."""
    logs = [{"final": {"test_acc": a, "diverged": 0}} for a in accs]
    logs += [{"final": {"test_acc": a, "diverged": 1}} for a in diverged_accs]
    return list(enumerate(logs))


_DEV = np.array([-0.9, -0.6, -0.4, -0.2, -0.1, 0.1, 0.2, 0.4, 0.6, 0.9]) * 0.01
_S = float(np.std(_DEV, ddof=1))


def test_diverged_runs_stay_out_of_the_statistics():
    # A 1.5-std gap fails; averaging in one collapsed run per side would inflate the pooled
    # std enough to call it a pass.
    from examples.kws_tcresnet8.compare import cell_stats, hypothesis_verdict

    c_accs, c_div = cell_stats(_final_logs(0.95 + _DEV + 1.5 * _S, diverged_accs=[0.1]))
    py_accs, py_div = cell_stats(_final_logs(0.95 + _DEV, diverged_accs=[0.1]))
    assert len(c_accs) == len(py_accs) == 10
    assert c_div == py_div == [10]
    assert hypothesis_verdict(c_accs, c_div, py_accs, py_div) == "FAIL"


def test_divergence_on_one_side_only_fails():
    from examples.kws_tcresnet8.compare import cell_stats, hypothesis_verdict

    c_accs, c_div = cell_stats(_final_logs(0.95 + _DEV[:9] + 0.5 * _S, diverged_accs=[float("nan")]))
    py_accs, py_div = cell_stats(_final_logs(0.95 + _DEV))
    assert hypothesis_verdict(c_accs, [], py_accs, py_div) == "pass"
    assert hypothesis_verdict(c_accs, c_div, py_accs, py_div) == "FAIL"
