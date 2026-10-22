"""Solver strategies: (plan space, arena budget) -> the plan C runs, or "nothing fits".

ExactRetainStrategy is exact for the retain-list space: over all 2^(n-1) masks keeping ACT
n, times the allowed in-place settings, it returns the plan of least (recompute cost,
REFORWARD count) whose arena, as C's placer packs it, is at most the budget; ties go to the
lower raw peak, then without in-place, then the smaller canonical mask (Plan.tie_key).

How. Cost and REFORWARD count add up over segments, and the placed peak of a plan is the
max over its segments of S + local (space.py). So a dynamic program h(k, S, ip) gives the
least (cost, REFORWARD count) of any completion from kept ACT k, with S bytes living
through, whose every segment's placed peak fits the budget; None when none does. A
best-first search over path prefixes keyed by g + h pops complete plans in nondecreasing
(cost, REFORWARD count) tiers. Each complete plan is placed with the copy of C's placer;
the first tier holding a plan whose arena fits is the answer tier, and the answer is its
least plan by the remaining tie keys.

Why it is exact. (i) Every plan whose placed peak fits is an effective path all of whose
edges the program accepts, and every popped prefix has a peak-feasible completion of
exactly its key. (ii) An arena is never below its placed peak (C's placement verifier
proves co-live ranges disjoint inside the arena), so arena <= budget implies placed peak
<= budget: no fitting plan is pruned, and tiers pop in order, so the first tier with a fit
holds the least (cost, REFORWARD count). (iii) That tier is searched to its end, so the
remaining tie keys range over the whole tie class. Fit is decided by the placer copy, whose
arena equals C's on every plan the parity tests run.

The tie key uses C's raw peak (DRY_PLAN's wires_peak_b); the program and the prune use the
8-rounded placed peak, the one the arena is bounded by. Neither replaces the other.

h depends on the budget (its edges test S + local <= budget): it is built for each solve.
Reusing one built for another budget prunes plans that fit or keeps ones that do not.
"""
import heapq
import math
from typing import Protocol

from odt.planner.plan import NothingFits, Plan
from odt.planner.space import PlanSpace


class Strategy(Protocol):
    """budget_b None means no limit: every plan fits (the frontier's first solve)."""

    def solve(self, space: PlanSpace, budget_b: int | None) -> Plan | NothingFits: ...


class _PeakFeasibleCost:
    """h(k, S, ip) for one budget."""

    def __init__(self, space: PlanSpace, budget_b: int):
        self.space, self.budget_b, self.memo = space, budget_b, {}

    def edges(self, k: int, live_below: int, ip: bool):
        """(k2, S below k2, segment cost, segment REFORWARDs, h from k2) of each accepted
        edge out of k."""
        below = live_below + self.space.act_placed[k]
        for k2, local, cost, refwd in self.space.out_edges[(k, ip)]:
            if live_below + local <= self.budget_b:
                rest = self(k2, below, ip)
                if rest is not None:
                    yield k2, below, cost, refwd, rest

    def __call__(self, k: int, live_below: int, ip: bool):
        key = (k, live_below, ip)
        if key not in self.memo:
            if k == self.space.n:
                self.memo[key] = (0, 0)
            else:
                self.memo[key] = min(((c + r[0], f + r[1]) for _, _, c, f, r
                                      in self.edges(k, live_below, ip)), default=None)
        return self.memo[key]


def peak_floor(space: PlanSpace) -> int:
    """The smallest budget some plan's placed peak fits: a lower bound on any arena."""
    keep_all = space.plan((1 << space.n) - 1, False).placed_peak_b
    lo, hi = 0, keep_all
    while lo < hi:
        mid = (lo + hi) // 2
        h = _PeakFeasibleCost(space, mid)
        if any(h(0, 0, ip) is not None for ip in space.settings):
            hi = mid
        else:
            lo = mid + 1
    return lo


def _best_fit(tier: list[Plan], budget_b: int) -> Plan | None:
    return min((p for p in tier if p.arena_b <= budget_b), key=Plan.tie_key, default=None)


class ExactRetainStrategy:
    def solve(self, space: PlanSpace, budget_b: int | None) -> Plan | NothingFits:
        budget_b = math.inf if budget_b is None else budget_b
        h = _PeakFeasibleCost(space, budget_b)
        heap = []
        for ip in space.settings:
            start = h(0, 0, ip)
            if start is not None:
                heap.append((start[0], start[1], ip, 0, 0, 0, 0, (0,)))
        heapq.heapify(heap)
        tier_key, tier = None, []
        while heap:
            cost, refwd, ip, k, below, g_cost, g_refwd, path = heapq.heappop(heap)
            if (cost, refwd) != tier_key:
                found = _best_fit(tier, budget_b)
                if found is not None:
                    return found
                tier_key, tier = (cost, refwd), []
            if k == space.n:
                tier.append(space.plan(sum(1 << (j - 1) for j in path[1:]), ip))
                continue
            for k2, below2, c, f, rest in h.edges(k, below, ip):
                heapq.heappush(heap, (g_cost + c + rest[0], g_refwd + f + rest[1], ip, k2, below2,
                                      g_cost + c, g_refwd + f, path + (k2,)))
        found = _best_fit(tier, budget_b)
        return found if found is not None else NothingFits(budget_b, peak_floor(space))
