"""The planner's copy of C (wire table, retain steps, ranges, costs, peaks) equals C.

The goldens are C output: `planner_goldens/<model>.dry_plan.txt` holds raw DRY_PLAN lines
(REMAT_STORAGE=arena) for a fixed sample of retain plans, written by regenerate.py.
"""
import dataclasses
import random
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
    MEMBERS,
    REFORWARD,
    Range,
    Wire,
    WireTable,
    extent,
    place_portfolio,
    place_two_ended,
    plan_numbers,
    retain_steps,
    scan_ranges,
    step_live,
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


# ---- the placer: C's portfolio, compared on arena bytes ----

def hand_plan(spec):
    """spec: (bytes, begin, end, kind) per range in begin order; range r is wire r + 1."""
    wires = (Wire("ACT", 0, 0, "FLOAT32"),) + tuple(
        Wire(kind, r + 1, nbytes, "FLOAT32") for r, (nbytes, _, _, kind) in enumerate(spec))
    t = WireTable(0, 0, 0, True, wires, (), (), ())
    return t, [Range(r + 1, begin, end) for r, (_, begin, end, _) in enumerate(spec)]


def check_layout(t, ranges, offsets) -> int:
    """Independent of the placer: aligned, non-negative, co-live ranges disjoint."""
    sizes = [(t.wires[r.wire].bytes + 7) // 8 * 8 for r in ranges]
    assert all(o >= 0 and o % 8 == 0 for o in offsets)
    for a in range(len(ranges)):
        for b in range(a):
            if ranges[a].begin <= ranges[b].end and ranges[b].begin <= ranges[a].end:
                assert offsets[a] + sizes[a] <= offsets[b] or offsets[b] + sizes[b] <= offsets[a]
    return max(o + s for o, s in zip(offsets, sizes))


def random_plan(rng):
    steps = rng.randint(1, 24)
    begins = sorted(rng.randrange(steps) for _ in range(rng.randint(1, 20)))
    return hand_plan([(rng.randint(1, 4096), b, rng.randint(b, steps - 1),
                       rng.choice(("ACT", "GRAD"))) for b in begins])


ACT, GRAD = "ACT", "GRAD"
HAND_PLANS = {  # the C unit test's hand plans: each reaches its peak through one member only
    "size-then-end": ([(16, 0, 3, ACT), (16, 1, 4, ACT), (8, 3, 5, ACT), (8, 4, 5, ACT),
                       (24, 5, 5, GRAD)], [16, 0, 32, 24, 0], 40),
    "ping-pong": ([(24, 0, 2, ACT), (16, 1, 3, ACT), (16, 2, 2, GRAD), (32, 3, 3, ACT)],
                  [0, 40, 24, 0], 56),
    "breadth/area": ([(16, 0, 3, ACT), (24, 1, 2, GRAD), (24, 2, 4, GRAD), (8, 3, 4, GRAD),
                      (32, 4, 4, GRAD)], [24, 40, 0, 56, 24], 64),
    "two-ended": ([(32, 0, 1, ACT), (16, 1, 3, GRAD), (16, 2, 3, GRAD), (24, 3, 5, GRAD),
                   (32, 4, 4, GRAD)], [0, 40, 24, 0, 24], 56),
}


@pytest.mark.parametrize("model", MODELS)
def test_every_golden_dry_plan_line_has_the_arena_of_the_copy_of_c(model):
    t = table(model)
    for row in golden_rows(model):
        numbers = plan_numbers(t, *parse_plan(row["plan"], t.n))
        placement = place_portfolio(t, numbers.ranges)
        assert (row["plan"], placement.bytes, placement.bytes - numbers.placed_peak_b) == (
            row["plan"], row["arena_b"], row["arena_gap_b"])


@pytest.mark.parametrize("name", list(HAND_PLANS))
def test_each_member_alone_places_its_hand_plan_at_the_peak(name):
    spec, offsets, nbytes = HAND_PLANS[name]
    t, ranges = hand_plan(spec)
    member = {m: f for m, f, _ in MEMBERS}[name]
    assert member(t, ranges, nbytes) == offsets
    placement = place_portfolio(t, ranges)
    assert (placement.member, placement.bytes, list(placement.offsets)) == (name, nbytes, offsets)
    for other, f, needs_grad in MEMBERS:
        if other != name:
            assert check_layout(t, ranges, f(t, ranges, nbytes)) > nbytes


def test_every_member_layout_passes_the_overlap_check_and_the_portfolio_keeps_the_best():
    rng = random.Random(4)
    for _ in range(500):
        t, ranges = random_plan(rng)
        peak = max(step_live(t, ranges, max(r.end for r in ranges) + 1, True))
        got = {}
        for name, member, needs_grad in MEMBERS:
            if needs_grad and not any(t.wires[r.wire].kind == "GRAD" for r in ranges):
                continue
            offsets = member(t, ranges, peak)
            got[name] = check_layout(t, ranges, offsets)
            assert got[name] == extent(t, ranges, offsets) >= peak
        placement = place_portfolio(t, ranges)
        assert placement.bytes == min(got.values()) <= got["first-fit-decreasing"]
        assert placement.member == next(n for n in got if got[n] == placement.bytes)
        assert check_layout(t, ranges, placement.offsets) == placement.bytes


def test_two_ended_needs_a_grad_range_and_the_portfolio_skips_it_without_one():
    t, ranges = hand_plan([(16, 0, 1, ACT), (8, 1, 2, ACT)])
    with pytest.raises(ValueError, match=r"^remat\[arena\]: two-ended placement needs a GRAD "
                                         r"range$"):
        place_two_ended(t, ranges, 24)
    assert place_portfolio(t, ranges).bytes == 24
