"""A retain plan's TRAIN program as C derives it: steps (rematFillRetainSteps) with the
operands the deriver attaches to each, ranges (rematScanRanges over those operands, with the
in-place dx rule), and the numbers DRY_PLAN prints. Chain models only."""
from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

from odt.planner.cmodel.table import WireTable, placed

FORWARD = "FORWARD"
LOSS_FORWARD = "LOSS_FORWARD"
LOSS_BACKWARD = "LOSS_BACKWARD"
BACKWARD = "BACKWARD"
REFORWARD = "REFORWARD"
IN_PLACE_TYPES = ("RELU", "DROPOUT", "FLATTEN")
IN_PLACE_DTYPES = ("FLOAT32", "SYM_INT32")

READ = "READ"
META = "META"  # the step needs the wire's header only, so the wire's range is not extended
WRITE = "WRITE"
NONE = 0xFFFF  # the wire of a write nothing receives: the dx of the deepest backward


class Operand(NamedTuple):
    wire: int
    mode: str


class Step(NamedTuple):
    kind: str
    layer: int
    reads: tuple[Operand, ...]
    writes: tuple[Operand, ...]


class Range(NamedTuple):
    wire: int
    begin: int  # the step that writes the wire
    end: int  # its last reader, inclusive


@dataclass(frozen=True)
class PlanNumbers:
    plan: str  # as DRY_PLAN prints it
    steps: int
    wires_peak_b: int  # C's peakLiveBytes: raw bytes, the tie key
    activations_peak_b: int
    placed_peak_b: int  # 8-rounded bytes: what the placer packs, and the solver's prune
    refwd_steps: int
    forward_cost: int
    recompute_cost: int
    ranges: tuple[Range, ...]

    @property
    def arena_pad_b(self) -> int:
        return self.placed_peak_b - self.wires_peak_b


def derive_steps(t: WireTable, retain: Sequence[bool]) -> list[Step]:
    """retain[j]: ACT j is kept (index 0 is the input, always present)."""
    n = t.n
    if len(retain) != n + 1:
        raise ValueError(f"retain: {len(retain)} entries for a {n}-layer model; "
                         f"expected {n + 1} (ACT 0 to ACT {n})")

    def forward(kind: str, l: int) -> Step:
        return Step(kind, l, (Operand(l, READ),), (Operand(l + 1, WRITE),))

    steps = [forward(FORWARD, l) for l in range(n)]
    steps.append(Step(LOSS_FORWARD, n, (Operand(n, READ),), ()))
    if not t.has_backward:
        return steps
    recomputed_from = n + 1

    def recompute_if_dropped(j: int) -> None:
        nonlocal recomputed_from
        if j == 0 or retain[j] or j >= recomputed_from:
            return
        k = j - 1
        while k > 0 and not retain[k]:
            k -= 1
        steps.extend(forward(REFORWARD, l) for l in range(k, j))
        recomputed_from = k + 1

    recompute_if_dropped(n)
    steps.append(Step(LOSS_BACKWARD, n, (Operand(n, READ),), (Operand(t.grad_id(n), WRITE),)))
    for l in range(t.top, t.deepest - 1, -1):
        if t.reads_input[l]:
            recompute_if_dropped(l)
        dy = Operand(t.grad_id(l + 1), READ)
        x = Operand(l, READ if t.reads_input[l] else META)
        dx = Operand(t.grad_id(l) if l > t.deepest else NONE, WRITE)
        steps.append(Step(BACKWARD, l, (dy, x), (dx,)))
    return steps


def in_place_member(t: WireTable, in_place: bool, w: int) -> bool:
    out = t.wires[w]
    if not in_place or out.kind != "GRAD" or out.index == t.n:
        return False
    if t.layer_types[out.index] not in IN_PLACE_TYPES:
        return False
    upstream = t.wires[w - 1]
    return (out.dtype == upstream.dtype and out.dtype in IN_PLACE_DTYPES
            and out.bytes == upstream.bytes)


def _root(t: WireTable, in_place: bool, w: int) -> int:
    while in_place_member(t, in_place, w):
        w -= 1
    return w


def _extend_newest(ranges: list[list[int]], w: int, s: int) -> None:
    for r in reversed(ranges):
        if r[0] == w:
            r[2] = s
            return


def scan_ranges(t: WireTable, steps: Sequence[Step], in_place: bool) -> list[Range]:
    """A READ extends the newest range of its wire's root; META extends nothing. A write opens
    a range when its wire is its own root; an in-place member's write extends its root's."""
    ranges: list[list[int]] = []
    for s, step in enumerate(steps):
        for op in step.reads:
            if op.mode == READ:
                _extend_newest(ranges, _root(t, in_place, op.wire), s)
        for op in step.writes:
            if op.wire == NONE:
                continue
            root = _root(t, in_place, op.wire)
            if root == op.wire and op.mode == WRITE:
                ranges.append([op.wire, s, s])
            else:
                _extend_newest(ranges, root, s)
    return [Range(*r) for r in ranges]


def step_live(t: WireTable, ranges: Sequence[Range], num_steps: int, rounded: bool,
              max_wire: int | None = None) -> list[int]:
    """Live bytes per step (both range ends inclusive), raw or 8-rounded."""
    live = [0] * num_steps
    for r in ranges:
        if max_wire is None or r.wire <= max_wire:
            size = placed(t.wires[r.wire].bytes) if rounded else t.wires[r.wire].bytes
            for s in range(r.begin, r.end + 1):
                live[s] += size
    return live


def retain_name(t: WireTable, retain: Sequence[bool], in_place: bool) -> str:
    kept = ",".join(str(j) for j in range(1, t.n + 1) if retain[j])
    return f"retain:{kept}" + ("+inplace" if in_place else "")


def plan_numbers(t: WireTable, retain: Sequence[bool], in_place: bool) -> PlanNumbers:
    steps = derive_steps(t, retain)
    ranges = scan_ranges(t, steps, in_place)
    refwd = [step.layer for step in steps if step.kind == REFORWARD]
    return PlanNumbers(
        plan=retain_name(t, retain, in_place), steps=len(steps),
        wires_peak_b=max(step_live(t, ranges, len(steps), False)),
        activations_peak_b=max(step_live(t, ranges, len(steps), False, max_wire=t.n)),
        placed_peak_b=max(step_live(t, ranges, len(steps), True)),
        refwd_steps=len(refwd), forward_cost=sum(t.layer_cost),
        recompute_cost=sum(t.layer_cost[l] for l in refwd), ranges=tuple(ranges))
