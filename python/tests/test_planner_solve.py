"""The exact retain-list solver equals brute force, and its plan object runs in C."""
import dataclasses
import math
from functools import cache
from pathlib import Path

import pytest

from odt.planner import (ExactRetainStrategy, NothingFits, Plan, PlanSpace, UnsupportedProblem,
                         load_problem, peak_floor)
from odt.planner.cmodel import plan_numbers, retain_steps
from tests.planner_oracle import all_plans, best, frontier, random_chain

GOLDENS = Path(__file__).resolve().parent / "planner_goldens"
MODELS = ("har", "har_finetune", "mnist_cnn")
SOLVER = ExactRetainStrategy()
# The peak floors: the DP's binary search here, the brute-force minimum placed peak in
# test_the_peak_floor_is_the_smallest_placed_peak_of_any_plan, and the minimum
# wires_peak_b + arena_pad_b over C DRY_PLAN runs of every mask.
PEAK_FLOOR = {("har", True): 24576, ("har", False): 32768, ("har_finetune", True): 16384,
              ("har_finetune", False): 16384, ("mnist_cnn", True): 62720,
              ("mnist_cnn", False): 87808}


@cache
def space(model: str, allow_inplace: bool) -> PlanSpace:
    return PlanSpace.build(load_problem(GOLDENS / f"{model}.problem.json"), 1, allow_inplace)


@cache
def oracle(model: str, allow_inplace: bool) -> list[Plan]:
    return all_plans(space(model, allow_inplace))


def cost_of(answer) -> float:
    return math.inf if isinstance(answer, NothingFits) else answer.recompute_cost


def same_answer(got, want: Plan | None, budget: int) -> bool:
    return got == (NothingFits(budget, got.peak_floor_b) if want is None else want)


def test_a_plan_prints_as_the_retain_list_c_runs():
    plan = Plan(n=12, mask=(1 << 2) | (1 << 6) | (1 << 11), in_place=True, arena_b=32768,
                wires_peak_b=32768, placed_peak_b=32768, recompute_cost=301056, refwd_steps=7,
                forward_cost=502150, steps=32)
    assert (plan.policy, plan.kept(), plan.metadata_b) == ("retain", (3, 7, 12), None)
    assert plan.to_remat_plan() == "retain:3,7,12+inplace"
    assert dataclasses.replace(plan, in_place=False).to_remat_plan() == "retain:3,7,12"
    assert dataclasses.replace(plan, in_place=False).tie_key() < plan.tie_key()


def test_the_plan_space_refuses_an_unsupported_problem_before_any_work():
    p = load_problem(GOLDENS / "har.problem.json")
    with pytest.raises(UnsupportedProblem, match=r"^m: 2 has no C reference run"):
        PlanSpace.build(p, 2)


def test_a_retain_mask_keeps_act_n_and_no_higher_bit():
    sp = space("har", True)
    for mask, shown in ((1 << 10, "0x400"), (1 << 12 | 1 << 11, "0x1800")):
        with pytest.raises(ValueError, match=rf"^mask {shown}: a retain mask of 12 layers keeps "
                                             rf"ACT 12 \(bit 11\) and no higher bit$"):
            sp.retain_of(mask)


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("allow_inplace", (True, False))
def test_the_solver_equals_brute_force_at_every_arena_and_one_byte_below(model, allow_inplace):
    plans = oracle(model, allow_inplace)
    for budget in sorted({p.arena_b for p in plans} | {p.arena_b - 1 for p in plans}):
        assert same_answer(SOLVER.solve(space(model, allow_inplace), budget), best(plans, budget),
                           budget), (model, allow_inplace, budget)


@pytest.mark.parametrize("seed", range(12))
def test_the_solver_equals_brute_force_on_random_chains(seed):
    problem = random_chain(seed, 3 + seed % 10)
    for allow_inplace in (True, False):
        sp = PlanSpace.build(problem, 1, allow_inplace)
        plans = all_plans(sp)
        for budget in sorted({p.arena_b for p in plans} | {p.arena_b - 1 for p in plans}):
            assert same_answer(SOLVER.solve(sp, budget), best(plans, budget), budget), (
                seed, allow_inplace, budget)


@pytest.mark.parametrize("model", MODELS)
def test_the_segment_numbers_reproduce_every_mask(model):
    sp = space(model, True)
    top = 1 << (sp.n - 1)
    by_program = {}
    for low in range(top):
        mask = top | low
        path = sp.effective_path(mask)
        assert all(sp.allowed(k, k2) for k, k2 in zip(path, path[1:]))
        for ip in sp.settings:
            raw = plan_numbers(sp.table, sp.retain_of(mask), ip)
            reduced = plan_numbers(sp.table, sp.retain_of(sp.plan(mask, ip).mask), ip)
            assert raw.ranges == reduced.ranges
            assert sp.decomposed_peak(path, ip) == raw.placed_peak_b
        program = tuple(retain_steps(sp.table, sp.retain_of(mask)))
        by_program[program] = min(by_program.get(program, mask), mask)
    for program, smallest in by_program.items():
        assert sp.plan(smallest, False).mask == smallest


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("allow_inplace", (True, False))
def test_the_peak_floor_is_the_smallest_placed_peak_of_any_plan(model, allow_inplace):
    floor = min(p.placed_peak_b for p in oracle(model, allow_inplace))
    assert peak_floor(space(model, allow_inplace)) == floor == PEAK_FLOOR[(model, allow_inplace)]


@pytest.mark.parametrize("model", MODELS)
def test_below_the_peak_floor_nothing_fits_and_the_floor_is_reported(model):
    floor = PEAK_FLOOR[(model, True)]
    assert SOLVER.solve(space(model, True), floor - 1) == NothingFits(floor - 1, floor)
    assert SOLVER.solve(space(model, True), 0) == NothingFits(0, floor)


@pytest.mark.parametrize("model", MODELS)
def test_a_budget_equal_to_a_plans_arena_returns_that_plan(model):
    for p in frontier(oracle(model, True)):
        assert SOLVER.solve(space(model, True), p.arena_b) == p
        assert cost_of(SOLVER.solve(space(model, True), p.arena_b - 1)) > p.recompute_cost


def test_one_plan_space_answers_each_budget_on_its_own():
    """The solver's peak table depends on the budget: one space serves budgets in any order."""
    plans = oracle("har", True)
    budgets = sorted({p.arena_b for p in plans})
    for order in (budgets, budgets[::-1]):
        shared = PlanSpace.build(load_problem(GOLDENS / "har.problem.json"), 1)
        for budget in order:
            assert same_answer(SOLVER.solve(shared, budget), best(plans, budget), budget), budget


@pytest.mark.parametrize("model", MODELS)
def test_the_best_cost_never_rises_as_the_budget_grows(model):
    shared = PlanSpace.build(load_problem(GOLDENS / f"{model}.problem.json"), 1)
    costs = [cost_of(SOLVER.solve(shared, b))
             for b in sorted({p.arena_b for p in oracle(model, True)})]
    assert costs == sorted(costs, reverse=True)
    assert costs[-1] == 0


def test_without_in_place_the_solver_returns_no_in_place_plan():
    for budget in sorted({p.arena_b for p in oracle("har", False)}):
        assert not SOLVER.solve(space("har", False), budget).in_place
