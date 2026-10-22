"""The budget frontier: every arena budget at which the best plan changes, with that plan."""
from dataclasses import dataclass

from odt.planner.plan import NothingFits, Plan
from odt.planner.solve import ExactRetainStrategy, Strategy
from odt.planner.space import PlanSpace


@dataclass(frozen=True)
class FrontierPoint:
    budget_lo_b: int  # the smallest budget this plan answers: its arena
    plan: Plan


def frontier(space: PlanSpace, strategy: Strategy | None = None) -> list[FrontierPoint]:
    """Largest budget first. Point i's plan is the answer for every budget from its arena up
    to the previous point's arena minus one: below an answer's arena the plans that fit are a
    subset of those above it, and the answer stays among them down to its own arena. The last
    point's arena is the smallest arena of any plan."""
    strategy = strategy or ExactRetainStrategy()
    points, budget = [], None
    while not isinstance(answer := strategy.solve(space, budget), NothingFits):
        points.append(FrontierPoint(answer.arena_b, answer))
        budget = answer.arena_b - 1
    return points
