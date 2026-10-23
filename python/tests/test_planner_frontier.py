"""The frontier walk, the CLI, and the solver at REMAT_SELECT_MAX_LAYERS (64) layers."""
import json
import os
import re
import subprocess
import sys
import time
from functools import cache
from pathlib import Path

import pytest

from odt.planner import (ExactRetainStrategy, NothingFits, PlanSpace, UnsupportedProblem,
                         load_problem, peak_floor)
from odt.planner.__main__ import nothing_fits_message
from odt.planner.frontier import FrontierPoint, frontier
from tests.planner_oracle import (all_plans, chain_doc, chain_problem, frontier as oracle_frontier,
                                  random_chain, random_chain_spec)

GOLDENS = Path(__file__).resolve().parent / "planner_goldens"
PYTHON_DIR = Path(__file__).resolve().parents[1]
MODELS = ("har", "har_finetune", "mnist_cnn")
# (plan, arena) per breakpoint, largest budget first. Derived twice: the brute-force oracle
# over every mask (the walk test below), and C DRY_PLAN runs of every mask of both settings
# walked with the same tie order; the C lines of these plans are goldens.
FRONTIER = {
    ("har", True): [("retain:1,3,4,6,7,10,12+inplace", 41216),
                    ("retain:1,4,6,7,10,12+inplace", 37120), ("retain:1,4,7,10,12+inplace", 36864), ("retain:3,4,6,7,10,12+inplace", 33024),
                    ("retain:4,6,7,10,12+inplace", 28928), ("retain:4,7,10,12+inplace", 28672),
                    ("retain:3,6,7,10,12+inplace", 24832), ("retain:3,6,10,12+inplace", 24576)],
    ("har", False): [("retain:1,3,4,6,7,10,12", 49152), ("retain:1,3,4,7,10,12", 45056),
                     ("retain:1,4,7,10,12", 40960), ("retain:3,4,7,10,12", 36864),
                     ("retain:4,7,10,12", 32768)],
    ("har_finetune", True): [("retain:10,12", 16384)],
    ("har_finetune", False): [("retain:10,12", 16384)],
    ("mnist_cnn", True): [("retain:1,3,4,8,10+inplace", 100352), ("retain:1,4,8,10+inplace", 87808),
                          ("retain:3,4,8,10+inplace", 75264), ("retain:3,8,10+inplace", 62720)],
    ("mnist_cnn", False): [("retain:1,3,4,8,10", 112896), ("retain:1,4,8,10", 100352),
                           ("retain:3,4,8,10", 87808)],
}


@cache
def space(model: str, allow_inplace: bool) -> PlanSpace:
    return PlanSpace.build(load_problem(GOLDENS / f"{model}.problem.json"), 1, allow_inplace)


def golden(model: str) -> dict[str, dict]:
    rows = {}
    for line in (GOLDENS / f"{model}.dry_plan.txt").read_text().splitlines():
        row = {k: int(v) for k, v in re.findall(r"(\w+)=(\d+)\b", line)}
        rows[re.search(r" plan=(\S+)", line).group(1)] = row
    return rows


def ramp64():
    """32 x (LINEAR, RELU), widths 24 + 13 i: no two layers alike, so no large ties."""
    widths = [24 + 13 * i for i in range(33)]
    layers = []
    for i in range(32):
        layers += [("LINEAR", [1, widths[i + 1]], False, {"macPerElement": widths[i]}),
                   ("RELU", [1, widths[i + 1]], False, None)]
    return chain_problem([1, widths[0]], layers)


def wide_chain(width: int) -> list[tuple]:
    """LINEAR 4 -> width, RELU, LINEAR width -> 2, SOFTMAX: four FLOAT32 wires of 4 x width B
    (ACT 1, ACT 2, GRAD 2, GRAD 1) beside three small ones."""
    return [("LINEAR", [1, width], False, {"macPerElement": 4}),
            ("RELU", [1, width], False, None),
            ("LINEAR", [1, 2], False, {"macPerElement": width}),
            ("SOFTMAX", [1, 2], False, None)]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("allow_inplace", (True, False))
def test_the_frontier_is_every_breakpoint_of_brute_force(model, allow_inplace):
    points = frontier(space(model, allow_inplace))
    assert [p.plan for p in points] == oracle_frontier(all_plans(space(model, allow_inplace)))
    assert all(p.budget_lo_b == p.plan.arena_b for p in points)
    assert [(p.plan.to_remat_plan(), p.budget_lo_b) for p in points] == FRONTIER[
        (model, allow_inplace)]


@pytest.mark.parametrize("seed", range(12))
def test_the_frontier_walk_equals_brute_force_on_random_chains(seed):
    sp = PlanSpace.build(random_chain(seed, 3 + seed % 10), 1)
    assert [p.plan for p in frontier(sp)] == oracle_frontier(all_plans(sp))


def test_the_walk_starts_with_no_budget_limit():
    """So a plan whose arena is above sys.maxsize, still a budget C takes (at most
    2^64 - 1), is a point: here about 1.1 x 2^63 B. The four wide wires total about
    0.75 x 2^64 B, within what C's wire table adds up, so C accepts this problem."""
    problem = chain_problem([1, 4], wide_chain(3 * 2**58))
    first = frontier(PlanSpace.build(problem, 1))[0]
    assert first.plan.recompute_cost == 0
    assert sys.maxsize < first.budget_lo_b < 2**64


@pytest.mark.parametrize("model", MODELS)
def test_c_measured_every_frontier_plan_at_its_arena_with_no_gap(model):
    rows = golden(model)
    for allow_inplace in (True, False):
        for point in frontier(space(model, allow_inplace)):
            p, row = point.plan, rows[point.plan.to_remat_plan()]
            assert (row["arena_b"], row["arena_gap_b"], row["wires_peak_b"], row["recompute_cost"],
                    row["refwd_steps"], row["steps"]) == (
                p.arena_b, 0, p.wires_peak_b, p.recompute_cost, p.refwd_steps, p.steps)


def test_every_plan_prints_ascending_and_keeps_act_n():
    for seed in range(12):
        sp = PlanSpace.build(random_chain(seed, 3 + seed % 10), 1)
        for plan in all_plans(sp):
            body = plan.to_remat_plan().removesuffix("+inplace")
            kept = [int(j) for j in body.removeprefix("retain:").split(",")]
            assert kept == sorted(set(kept)) and kept[-1] == sp.n and kept[0] >= 1


def test_the_solver_handles_64_layers():
    t0 = time.perf_counter()
    sp = PlanSpace.build(ramp64(), 1)
    solver = ExactRetainStrategy()
    top = solver.solve(sp, sys.maxsize)
    below = solver.solve(sp, 1)
    assert isinstance(below, NothingFits) and below.budget_b == 1
    mid_budget = (top.arena_b + below.peak_floor_b) // 2
    mid = solver.solve(sp, mid_budget)
    for plan, budget in ((top, sys.maxsize), (mid, mid_budget)):
        assert plan == sp.plan(plan.mask, plan.in_place)
        assert below.peak_floor_b <= plan.placed_peak_b <= plan.arena_b <= budget
        assert sp.decomposed_peak(sp.effective_path(plan.mask), plan.in_place) == plan.placed_peak_b
    assert (sp.n, top.recompute_cost) == (64, 0)
    assert 0 < mid.recompute_cost < sum(sp.table.layer_cost)
    print(f"64-layer ramp: space and three solves in {time.perf_counter() - t0:.1f} s")


def run_cli(*args):
    env = dict(os.environ, PYTHONPATH=str(PYTHON_DIR))
    return subprocess.run([sys.executable, "-m", "odt.planner", *args], env=env,
                          capture_output=True, text=True, timeout=120)


def test_the_cli_prints_the_variables_that_run_the_plan_in_c():
    done = run_cli("solve", str(GOLDENS / "har.problem.json"), "--m", "1", "--budget", "41216")
    assert (done.returncode, done.stderr) == (0, "")
    assert done.stdout == ("# problem fnv1a64=26b4a7babc5f10bf\n"
                           "REMAT_PLAN=retain:1,3,4,6,7,10,12+inplace REMAT_BUDGET_B=41216\n")
    done = run_cli("solve", str(GOLDENS / "har.problem.json"), "--m", "1", "--budget", "41216",
                   "--no-inplace")
    assert done.stdout.splitlines()[1] == "REMAT_PLAN=retain:1,4,7,10,12 REMAT_BUDGET_B=41216"


def test_the_cli_names_a_budget_nothing_fits_and_its_peak_floor():
    done = run_cli("solve", str(GOLDENS / "har.problem.json"), "--m", "1", "--budget", "24575")
    assert (done.returncode, done.stdout) == (1, "")
    assert done.stderr == ("odt.planner: nothing fits REMAT_BUDGET_B=24575: no retain plan's arena "
                           "is at most 24575 B (the smallest placed peak of any plan is 24576 B; "
                           "the frontier's last point is the smallest arena)\n")


def test_the_nothing_fits_message_names_a_placement_gap_at_or_above_the_peak_floor():
    """The solver also answers "nothing fits" when the budget covers some plan's placed peak
    but placement gaps lift every such plan's arena above it. No golden
    problem file reaches it; the generated chain of the CLI test below does. Here the two
    wordings are checked on the message itself."""
    assert nothing_fits_message(NothingFits(budget_b=63, peak_floor_b=64)) == (
        "odt.planner: nothing fits REMAT_BUDGET_B=63: no retain plan's arena is at most 63 B "
        "(the smallest placed peak of any plan is 64 B; the frontier's last point is the "
        "smallest arena)")
    for budget_b in (64, 100):
        assert nothing_fits_message(NothingFits(budget_b=budget_b, peak_floor_b=64)) == (
            f"odt.planner: nothing fits REMAT_BUDGET_B={budget_b}: no retain plan's arena is at "
            f"most {budget_b} B (the smallest placed peak of any plan, 64 B, is within the "
            "budget, but a placement gap lifts the arena of every such plan above it; the "
            "frontier's last point is the smallest arena)")


def test_the_cli_names_a_placement_gap_at_or_above_the_peak_floor(tmp_path):
    # First seed-major hit of random_chain(seed, n), n in 3..12, where solve at the peak floor
    # answers nothing-fits.
    SEED, N, FLOOR = 7, 7, 392
    path = tmp_path / "gap.problem.json"
    path.write_text(json.dumps(chain_doc(*random_chain_spec(SEED, N))))
    assert peak_floor(PlanSpace.build(load_problem(path), 1)) == FLOOR
    done = run_cli("solve", str(path), "--m", "1", "--budget", str(FLOOR))
    assert (done.returncode, done.stdout, done.stderr) == (
        1, "", nothing_fits_message(NothingFits(FLOOR, FLOOR)) + "\n")
    assert "is within the budget, but a placement gap" in done.stderr


def test_the_cli_refuses_an_unsupported_problem_by_name():
    done = run_cli("solve", str(GOLDENS / "har.problem.json"), "--m", "2", "--budget", "41216")
    assert (done.returncode, done.stdout, done.stderr) == (
        1, "", "odt.planner: m: 2 has no C reference run; the planner accepts 1\n")
    done = run_cli("solve", str(GOLDENS / "har.problem.json"), "--m", "1", "--budget", "0")
    assert done.returncode == 2
    assert done.stderr.endswith("error: --budget 0: expected the arena budget in bytes, "
                                "1 <= value <= 18446744073709551615\n")


def test_a_problem_whose_wires_total_more_than_size_max_is_refused_by_name(tmp_path):
    """C's wire table adds up every wire's bytes in id order (ACT 1 .. n, then the GRAD wires)
    and exits when the total passes size_t, so no plan for such a problem could run. Four
    wires of 2^62 B: the total passes 2^64 - 1 at the last of them, GRAD 1."""
    layers = wide_chain(2**60)
    message = ("wires: more than 18446744073709551615 B in total (passed at wire GRAD 1) has no "
               "C reference run; the planner accepts wires that total at most "
               "18446744073709551615 B, as C's wire table does")
    with pytest.raises(UnsupportedProblem, match=f"^{re.escape(message)}$"):
        PlanSpace.build(chain_problem([1, 4], layers), 1)
    path = tmp_path / "wide.problem.json"
    path.write_text(json.dumps(chain_doc([1, 4], layers)))
    done = run_cli("frontier", str(path), "--m", "1")
    assert (done.returncode, done.stdout, done.stderr) == (1, "", f"odt.planner: {message}\n")


def assert_refused_by_name(tmp_path, input_shape: list[int], layers: list[tuple], message: str):
    """Through PlanSpace.build and through the CLI, with the same message."""
    with pytest.raises(UnsupportedProblem, match=f"^{re.escape(message)}$"):
        PlanSpace.build(chain_problem(input_shape, layers), 1)
    path = tmp_path / "huge.problem.json"
    path.write_text(json.dumps(chain_doc(input_shape, layers)))
    done = run_cli("frontier", str(path), "--m", "1")
    assert (done.returncode, done.stdout, done.stderr) == (1, "", f"odt.planner: {message}\n")


SMALL_HEAD = [("LINEAR", [1, 2], False, {"macPerElement": 1}), ("SOFTMAX", [1, 2], False, None)]


@pytest.mark.parametrize("input_shape, layers, message", [
    pytest.param([1, 2**32, 2**32], SMALL_HEAD,
                 "wire ACT 0: 18446744073709551616 elements has no C reference run; the planner "
                 "accepts a wire of at most 18446744073709551615 elements, as C's wire table does",
                 id="act0-elements"),
    # (2^32 - 1)(2^32 + 1) = SIZE_MAX elements is allowed, so the refusal is for the bytes
    pytest.param([1, 2**32 - 1, 2**32 + 1], SMALL_HEAD,
                 "wire ACT 0: 73786976294838206460 B has no C reference run; the planner accepts a "
                 "wire of at most 18446744073709551615 B, as C's wire table does",
                 id="act0-elements-at-size-max"),
    pytest.param([1, 2**62], SMALL_HEAD,
                 "wire ACT 0: 18446744073709551616 B has no C reference run; the planner accepts a "
                 "wire of at most 18446744073709551615 B, as C's wire table does",
                 id="act0-bytes"),
    pytest.param([1, 4], [("LINEAR", [1, 2**62], False, {"macPerElement": 1})] + SMALL_HEAD,
                 "wire ACT 1: 18446744073709551616 B has no C reference run; the planner accepts a "
                 "wire of at most 18446744073709551615 B, as C's wire table does",
                 id="act1-bytes"),
])
def test_a_wire_of_more_than_size_max_elements_or_bytes_is_refused_by_name(
        tmp_path, input_shape, layers, message):
    """C's wire table derives every ACT wire in id order, ACT 0 (the borrowed input) first:
    elementsOf multiplies its dimensions and wireBytes its bytes, each checked, and exits on the
    first product that passes size_t -- before it adds up the total of ACT 1 .. n and the GRAD
    wires. A GRAD wire has its ACT's elements and bytes, so it cannot pass first."""
    assert_refused_by_name(tmp_path, input_shape, layers, message)


@pytest.mark.parametrize("layers, message", [
    pytest.param([("LINEAR", [1, 2], False, {"macPerElement": 2**63}),
                  ("SOFTMAX", [1, 2], False, None)],
                 "layers[0].cost: a forward cost of 18446744073709551616 has no C reference run; "
                 "the planner accepts a layer forward cost of at most 18446744073709551615, as "
                 "C's layerForwardCost does",
                 id="layer-cost"),
    # a layer cost of exactly SIZE_MAX is allowed, so the refusal is for the sum
    pytest.param([("LINEAR", [1, 1], False, {"macPerElement": 2**64 - 1}),
                  ("SOFTMAX", [1, 1], False, None)],
                 "layers: more than 18446744073709551615 forward cost in total (passed at "
                 "layers[1]) has no C reference run; the planner accepts layers whose forward "
                 "costs total at most 18446744073709551615, as C's plan build does",
                 id="layer-cost-at-size-max"),
    pytest.param([("LINEAR", [1, 1], False, {"macPerElement": 2**63}),
                  ("LINEAR", [1, 1], False, {"macPerElement": 2**63}),
                  ("SOFTMAX", [1, 1], False, None)],
                 "layers: more than 18446744073709551615 forward cost in total (passed at "
                 "layers[1]) has no C reference run; the planner accepts layers whose forward "
                 "costs total at most 18446744073709551615, as C's plan build does",
                 id="cost-total"),
])
def test_a_forward_cost_past_size_max_is_refused_by_name(tmp_path, layers, message):
    """Every C plan build sums the layers' forward costs in layer order (recordRecomputeNumbers):
    layerForwardCost exits when a layer's checked product passes size_t, and the checked running
    sum exits when the total does, so no plan for such a problem could run. The wires are small:
    only the costs are past the limit."""
    assert_refused_by_name(tmp_path, [1, 4], layers, message)


@pytest.mark.parametrize("input_shape, layers", [
    # LINEAR costs 1 x (2^64 - 2), SOFTMAX costs its 1 output element: SIZE_MAX in total
    pytest.param([1, 4], [("LINEAR", [1, 1], False, {"macPerElement": 2**64 - 2}),
                          ("SOFTMAX", [1, 1], False, None)], id="cost-total-at-size-max"),
    # ACT 0 of 2^62 - 1 elements, 2^64 - 4 B; ACT 0 is not in the total
    pytest.param([1, 2**62 - 1], SMALL_HEAD, id="act0-bytes-at-size-max-minus-3"),
    # ACT 1, ACT 2 and GRAD 2 of (2^62 - 1) / 3 elements each: 2^64 - 4 B in total
    pytest.param([1, 1], [("LINEAR", [1, (2**62 - 1) // 3], False, {"macPerElement": 1}),
                          ("SOFTMAX", [1, (2**62 - 1) // 3], False, None)],
                 id="wire-total-at-size-max-minus-3"),
])
def test_a_problem_at_the_size_max_limits_is_accepted(tmp_path, input_shape, layers):
    """C refuses only a size past SIZE_MAX, so a problem right at the limit builds and has a
    frontier, through PlanSpace.build and through the CLI. Wire bytes are 4 x elements, and
    SIZE_MAX is not a multiple of 4, so a wire or a wire total can come no closer than
    SIZE_MAX - 3 B; a forward cost total can be SIZE_MAX exactly."""
    assert frontier(PlanSpace.build(chain_problem(input_shape, layers), 1))
    path = tmp_path / "limit.problem.json"
    path.write_text(json.dumps(chain_doc(input_shape, layers)))
    done = run_cli("frontier", str(path), "--m", "1")
    assert (done.returncode, done.stderr) == (0, "")
    assert done.stdout


def test_the_cli_names_a_missing_problem_file(tmp_path):
    missing = tmp_path / "nofile.json"
    done = run_cli("solve", str(missing), "--m", "1", "--budget", "5")
    assert (done.returncode, done.stdout) == (1, "")
    assert done.stderr == f"odt.planner: [Errno 2] No such file or directory: '{missing}'\n"


def test_the_cli_frontier_prints_one_line_per_point():
    done = run_cli("frontier", str(GOLDENS / "mnist_cnn.problem.json"), "--m", "1")
    assert done.returncode == 0
    lines = done.stdout.splitlines()
    assert lines[0] == "# problem fnv1a64=6a814ff6da6cae50"
    assert lines[1] == ("budget_lo_b=100352 plan=retain:1,3,4,8,10+inplace arena_b=100352 "
                        "wires_peak_b=100352 placed_peak_b=100352 recompute_cost=0 refwd_steps=0")
    assert [re.search(r" plan=(\S+)", line).group(1) for line in lines[1:]] == [
        p for p, _ in FRONTIER[("mnist_cnn", True)]]


def test_a_frontier_point_names_the_smallest_budget_its_plan_answers():
    point = frontier(space("har_finetune", True))[0]
    assert point == FrontierPoint(budget_lo_b=16384, plan=point.plan)
