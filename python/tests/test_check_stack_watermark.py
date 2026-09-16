"""Tests for the stack-watermark report/gate script (hardening item 6, Stage 1)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples" / "_shared"))
import check_stack_watermark as csw


def _log(tmp_path: Path, name: str, sym_bits: int, stack_peak_b) -> str:
    payload = {"memory": {"sym_bits": sym_bits, "stack_peak_b": stack_peak_b}}
    if stack_peak_b is None:
        del payload["memory"]["stack_peak_b"]
    p = tmp_path / name
    p.write_text(json.dumps(payload))
    return str(p)


def test_under_budget_passes(tmp_path, capsys):
    log = _log(tmp_path, "f.json", -1, 27768)  # #296 Stage 2 measured peak
    assert csw.main(["--platform", "darwin", log]) == 0
    assert "OK" in capsys.readouterr().out


def test_over_budget_warns_but_exits_zero_by_default(tmp_path, capsys):
    log = _log(tmp_path, "f.json", -1, 200000)
    assert csw.main(["--platform", "darwin", log]) == 0
    assert "WARN" in capsys.readouterr().out


def test_over_budget_fails_under_enforce(tmp_path):
    log = _log(tmp_path, "f.json", -1, 200000)
    assert csw.main(["--enforce", "--platform", "darwin", log]) == 1


def test_sym_config_uses_sym_budget(tmp_path):
    ok = _log(tmp_path, "s8.json", 8, 51968)  # #296 Stage 2 measured peak
    over = _log(tmp_path, "s4.json", 4, 158000)
    assert csw.main(["--enforce", "--platform", "darwin", ok]) == 0
    assert csw.main(["--enforce", "--platform", "darwin", over]) == 1


def test_uncalibrated_platform_reports_and_passes(tmp_path, capsys):
    log = _log(tmp_path, "f.json", -1, 123456)
    assert csw.main(["--enforce", "--platform", "linux", log]) == 0
    assert "UNCALIBRATED" in capsys.readouterr().out


def test_missing_stack_peak_is_a_hard_error_even_without_enforce(tmp_path):
    log = _log(tmp_path, "f.json", -1, None)
    assert csw.main(["--platform", "darwin", log]) == 2


def test_zero_stack_peak_is_a_hard_error(tmp_path):
    log = _log(tmp_path, "f.json", -1, 0)
    assert csw.main(["--platform", "darwin", log]) == 2


def _log_impl(tmp_path: Path, name: str, impl: str, memory: dict) -> str:
    p = tmp_path / name
    p.write_text(json.dumps({"impl": impl, "memory": memory}))
    return str(p)


def test_bfp_impl_routes_to_the_uncalibrated_bfp_bucket(tmp_path, capsys):
    log = _log_impl(tmp_path, "b.json", "c-bfp",
                    {"storage_dtype": "bfp", "stack_peak_b": 40000})  # no sym_bits at all
    assert csw.main(["--enforce", "--platform", "darwin", log]) == 0
    out = capsys.readouterr().out
    assert "UNCALIBRATED" in out and "bfp" in out
    assert csw.BUDGETS_B["darwin"]["bfp"] is None
    assert csw.BUDGETS_B["linux"]["bfp"] is None


def test_legacy_logs_without_impl_still_key_on_sym_bits(tmp_path):
    ok = _log(tmp_path, "s8.json", 8, 51968)
    assert csw._config_of(json.loads(Path(ok).read_text())) == "sym"


def test_finetune_impl_wins_over_sym_bits(tmp_path):
    log = _log_impl(tmp_path, "f.json", "c-finetune", {"sym_bits": -1, "stack_peak_b": 4016})
    assert csw._config_of(json.loads(Path(log).read_text())) == "finetune"
