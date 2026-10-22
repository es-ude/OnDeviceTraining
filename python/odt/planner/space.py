"""The retain-list plan space of one (problem, m): every mask keeping ACT n, with and without
in-place dx, and the per-segment numbers the exact solver searches with.

A segment is the run of dropped ACTs between two consecutive kept ACTs k < k2 (ACT 0, the
input, counts as kept; ACT n is always kept). C recomputes a segment once, from ACT k up to
its highest ACT a backward reads, so recompute cost and REFORWARD count add up over
segments (`seg`). A kept ACT that no backward reads and that roots no recompute changes no
step: masks that differ only in such ACTs are one plan, named by the mask with those bits 0
(`effective_path`). On such a path every kept ACT below a segment is read again after it,
so every step of segment (k, k2] has live bytes S + local(k, k2, in_place), where S is the
placed bytes of the kept ACTs below k (`local`; the backward runs top-down, so a higher
segment's own wires are gone by then).
"""
from dataclasses import dataclass

from odt.planner.cmodel import (BACKWARD, LOSS_BACKWARD, LOSS_FORWARD, REFORWARD, WireTable,
                                place_portfolio, placed, plan_numbers, retain_steps,
                                scan_ranges, step_live, wire_table)
from odt.planner.plan import Plan
from odt.planner.schema import Problem


def _segment_live(t: WireTable, k: int, k2: int, ip: bool) -> int:
    """The segment's own peak, read off the plan that keeps k, k2 and every ACT above k2."""
    retain = tuple(j == k or j >= k2 for j in range(t.n + 1))
    steps = retain_steps(t, retain)
    live = step_live(t, scan_ranges(t, steps, ip), len(steps), True)
    return max(live[s] for s, (kind, l) in enumerate(steps)
               if (s < t.n and k <= l < k2) or (kind in (REFORWARD, BACKWARD) and k <= l < k2)
               or (k2 == t.n and kind in (LOSS_FORWARD, LOSS_BACKWARD)))


@dataclass(frozen=True)
class PlanSpace:
    problem: Problem
    m: int
    table: WireTable
    settings: tuple[bool, ...]  # the in-place dx settings the solver may choose
    needed: tuple[bool, ...]  # ACT j is read by a backward step
    act_placed: tuple[int, ...]  # placed bytes of ACT j; 0 for ACT 0, which has no range
    seg: dict  # (k, k2) -> (recompute cost, REFORWARD count, has a recompute)
    local: dict  # (k, k2, in_place) -> max over the segment's steps of its own live bytes
    out_edges: dict  # (k, in_place) -> ((k2, local, cost, REFORWARDs), ...) of allowed k2

    @classmethod
    def build(cls, problem: Problem, m: int, allow_inplace: bool = True) -> "PlanSpace":
        """Raises UnsupportedProblem before any work for input without a C reference run."""
        t = wire_table(problem, m)
        n = t.n
        needed = tuple(j < n and t.has_backward and t.deepest <= j <= t.top and t.reads_input[j]
                       for j in range(n + 1))
        settings = (False, True) if allow_inplace else (False,)
        seg, local, out_edges = {}, {}, {}
        for k in range(n):
            top_need = None
            for k2 in range(k + 1, n + 1):
                if k2 - 1 > k and needed[k2 - 1]:
                    top_need = k2 - 1
                seg[(k, k2)] = ((0, 0, False) if top_need is None else
                                (sum(t.layer_cost[k:top_need]), top_need - k, True))
            for ip in settings:
                for k2 in range(k + 1, n + 1):
                    if k == 0 or needed[k] or seg[(k, k2)][2]:
                        local[(k, k2, ip)] = _segment_live(t, k, k2, ip)
                out_edges[(k, ip)] = tuple((k2, local[(k, k2, ip)]) + seg[(k, k2)][:2]
                                           for k2 in range(k + 1, n + 1) if (k, k2, ip) in local)
        acts = (0,) + tuple(placed(t.wires[j].bytes) for j in range(1, n + 1))
        return cls(problem, m, t, settings, needed, acts, seg, local, out_edges)

    @property
    def n(self) -> int:
        return self.table.n

    def allowed(self, k: int, k2: int) -> bool:
        """k -> k2 can be a segment of a plan's effective path."""
        return k == 0 or self.needed[k] or self.seg[(k, k2)][2]

    def retain_of(self, mask: int) -> tuple[bool, ...]:
        if mask >> (self.n - 1) != 1:
            raise ValueError(f"mask {mask:#x}: a retain mask of {self.n} layers keeps ACT "
                             f"{self.n} (bit {self.n - 1}) and no higher bit")
        return (True,) + tuple(bool(mask >> (j - 1) & 1) for j in range(1, self.n + 1))

    def effective_path(self, mask: int) -> tuple[int, ...]:
        """0, then the kept ACTs that change a step, then n."""
        retain = self.retain_of(mask)
        path = [self.n]
        for j in range(self.n - 1, 0, -1):
            if retain[j] and (self.needed[j] or self.seg[(j, path[0])][2]):
                path.insert(0, j)
        return (0, *path)

    def decomposed_peak(self, path: tuple[int, ...], in_place: bool) -> int:
        """The placed peak of an effective path from the segment numbers alone."""
        live_below, peak = 0, 0
        for k, k2 in zip(path, path[1:]):
            peak = max(peak, live_below + self.local[(k, k2, in_place)])
            live_below += self.act_placed[k]
        return peak

    def plan(self, mask: int, in_place: bool) -> Plan:
        """Any mask keeping ACT n, as C would run and place it."""
        numbers = plan_numbers(self.table, self.retain_of(mask), in_place)
        canonical = sum(1 << (j - 1) for j in self.effective_path(mask)[1:])
        return Plan(n=self.n, mask=canonical, in_place=in_place,
                    arena_b=place_portfolio(self.table, numbers.ranges).bytes,
                    wires_peak_b=numbers.wires_peak_b, placed_peak_b=numbers.placed_peak_b,
                    recompute_cost=numbers.recompute_cost, refwd_steps=numbers.refwd_steps,
                    forward_cost=numbers.forward_cost, steps=numbers.steps)
