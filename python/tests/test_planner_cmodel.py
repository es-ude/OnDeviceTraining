"""The planner's copy of C (wire table, retain steps, ranges, costs, peaks) equals C.

The goldens are C output: `planner_goldens/<model>.dry_plan.txt` holds raw DRY_PLAN lines
(REMAT_STORAGE=arena) for a fixed sample of retain plans, written by regenerate.py.
"""
import dataclasses
import re
from functools import cache
from pathlib import Path

import pytest

from odt.planner import UnsupportedProblem, load_problem
from odt.planner.cmodel import (
    BACKWARD,
    FORWARD,
    LOSS_BACKWARD,
    LOSS_FORWARD,
    REFORWARD,
    plan_numbers,
    retain_steps,
    scan_ranges,
    wire_table,
)

GOLDENS = Path(__file__).resolve().parent / "planner_goldens"
MODELS = ("har", "har_finetune", "mnist_cnn")
NUMBER_FIELDS = ("steps", "wires_peak_b", "activations_peak_b", "arena_pad_b", "refwd_steps",
                 "forward_cost", "recompute_cost")


@cache
def table(model: str):
    return wire_table(load_problem(GOLDENS / f"{model}.problem.json"), 1)


def retain_of(n: int, kept) -> tuple[bool, ...]:
    return tuple(j in kept for j in range(n + 1))


def parse_plan(name: str, n: int) -> tuple[tuple[bool, ...], bool]:
    body, plus, _ = name.partition("+")
    assert body.startswith("retain:") and plus in ("", "+")
    return retain_of(n, {int(j) for j in body[len("retain:"):].split(",")}), plus == "+"


@cache
def golden_rows(model: str) -> list[dict]:
    rows = []
    for line in (GOLDENS / f"{model}.dry_plan.txt").read_text().splitlines():
        assert line.startswith("DRY_PLAN ")
        row = {k: int(v) for k, v in re.findall(r"(\w+)=(\d+)\b", line)}
        row["plan"] = re.search(r" plan=(\S+)", line).group(1)
        rows.append(row)
    return rows


def test_the_wire_tables_number_acts_then_the_seed_then_the_dx_wires_top_down():
    t = table("har")
    assert (t.n, t.deepest, t.top, t.has_backward) == (12, 0, 10, True)
    assert [(w.kind, w.index) for w in t.wires] == (
        [("ACT", j) for j in range(13)] + [("GRAD", 12)] + [("GRAD", l) for l in range(10, 0, -1)])
    assert [w.bytes for w in t.wires[:13]] == [4608, 8192, 8192, 4096, 8192, 8192, 4096, 8192,
                                               8192, 256, 256, 24, 24]
    assert all(t.wires[t.grad_id(l)].bytes == t.wires[l].bytes for l in range(1, 11))
    assert t.wires[t.grad_id(12)].bytes == 24
    assert (table("har_finetune").deepest, table("har_finetune").top) == (10, 10)
    assert (table("mnist_cnn").n, table("mnist_cnn").deepest, table("mnist_cnn").top) == (10, 0, 8)


def test_reads_and_costs_follow_c_per_layer():
    t = table("har")
    assert t.reads_input == (True, True, False, True, True, False, True, True, False, False,
                             True, True)
    assert table("har_finetune").reads_input[:7] == (False, True, False, False, True, False, False)
    # conv: macPerElement x out elements; relu, softmax: out elements; pools: kernel x out
    # elements; flatten: 0. Their sum is DRY_PLAN's forward_cost, 502150.
    assert t.layer_cost == (129024, 2048, 2048, 163840, 2048, 2048, 196608, 2048, 2048, 0, 384,
                            6)


def test_a_retain_list_recomputes_each_dropped_segment_once_from_the_act_below_it():
    t = table("har")
    steps = retain_steps(t, retain_of(12, {3, 7, 12}))
    assert steps == ([(FORWARD, l) for l in range(12)] + [(LOSS_FORWARD, 12), (LOSS_BACKWARD, 12)]
                     + [(REFORWARD, 7), (REFORWARD, 8), (REFORWARD, 9), (BACKWARD, 10),
                        (BACKWARD, 9), (BACKWARD, 8), (BACKWARD, 7), (REFORWARD, 3), (REFORWARD, 4),
                        (REFORWARD, 5), (BACKWARD, 6), (BACKWARD, 5), (BACKWARD, 4), (BACKWARD, 3),
                        (BACKWARD, 2), (REFORWARD, 0), (BACKWARD, 1), (BACKWARD, 0)])
    numbers = plan_numbers(t, retain_of(12, {3, 7, 12}), False)
    assert (numbers.steps, numbers.refwd_steps, numbers.recompute_cost) == (32, 7, 301056)


def test_in_place_dx_folds_the_relu_and_flatten_gradients_into_their_upstream_range():
    t = table("har")
    steps = retain_steps(t, retain_of(12, set(range(1, 13))))
    assert len(scan_ranges(t, steps, False)) == 23
    folded = scan_ranges(t, steps, True)
    assert len(folded) == 19
    assert not {t.grad_id(l) for l in (9, 7, 4, 1)} & {r.wire for r in folded}


@pytest.mark.parametrize("model", MODELS)
def test_every_golden_dry_plan_line_matches_the_copy_of_c(model):
    t = table(model)
    for row in golden_rows(model):
        retain, in_place = parse_plan(row["plan"], t.n)
        numbers = plan_numbers(t, retain, in_place)
        got = {f: getattr(numbers, f) for f in NUMBER_FIELDS}
        assert (numbers.plan, got) == (row["plan"], {f: row[f] for f in NUMBER_FIELDS})


def test_the_goldens_cover_both_settings_and_the_fixed_sample():
    for model in MODELS:
        plans = [row["plan"] for row in golden_rows(model)]
        n = table(model).n
        for suffix in ("", "+inplace"):
            assert f"retain:{','.join(map(str, range(1, n + 1)))}{suffix}" in plans
            assert f"retain:{n}{suffix}" in plans
            assert all(f"retain:{j},{n}{suffix}" in plans for j in range(1, n))


def test_the_copy_of_c_refuses_a_problem_the_planner_does_not_accept():
    p = load_problem(GOLDENS / "har.problem.json")
    with pytest.raises(UnsupportedProblem, match=r"^loss: MSE has no C reference run"):
        wire_table(dataclasses.replace(p, loss="MSE"), 1)
