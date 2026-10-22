"""C's ARENA placement (RematArena.c): a portfolio of placement orders, each one offset per
range, the smallest arena kept, the earlier member winning ties. Parity with C is on the
arena bytes; the members keep C's tie keys so the chosen layouts are C's too.

`ranges` is a program's range list in C's order (begin-sorted, as scan_ranges returns it);
"r" below is the index into it. Sizes are 8-rounded placed bytes.
"""
from collections.abc import Callable, Sequence
from typing import NamedTuple

from odt.planner.cmodel.program import Range, step_live
from odt.planner.cmodel.table import WireTable, placed


class Placement(NamedTuple):
    bytes: int
    offsets: tuple[int, ...]
    member: str


def _sizes(t: WireTable, ranges: Sequence[Range]) -> list[int]:
    return [placed(t.wires[r.wire].bytes) for r in ranges]


def _co_live(a: Range, b: Range) -> bool:
    return a.begin <= b.end and b.begin <= a.end


def extent(t: WireTable, ranges: Sequence[Range], offsets: Sequence[int]) -> int:
    return max((o + s for o, s in zip(offsets, _sizes(t, ranges))), default=0)


def has_grad_range(t: WireTable, ranges: Sequence[Range]) -> bool:
    return any(t.wires[r.wire].kind == "GRAD" for r in ranges)


def _co_live_placed(ranges, offsets, placed_ids, r):
    """The placed ranges co-live with r, by ascending offset (C's byOffset sweep)."""
    return sorted((q for q in placed_ids if _co_live(ranges[q], ranges[r])),
                  key=lambda q: offsets[q])


def _lowest_fit(ranges, sizes, offsets, placed_ids, r) -> int:
    off = 0
    for q in _co_live_placed(ranges, offsets, placed_ids, r):
        if offsets[q] >= off + sizes[r]:
            break
        off = max(off, offsets[q] + sizes[q])
    return off


def _first_fit(ranges, sizes, order) -> list[int]:
    offsets = [0] * len(ranges)
    done: list[int] = []
    for r in order:
        offsets[r] = _lowest_fit(ranges, sizes, offsets, done, r)
        done.append(r)
    return offsets


def _stream_key(ranges, sizes):
    return lambda r: (ranges[r].begin, -ranges[r].end, -sizes[r], r)


def place_first_fit_decreasing(t, ranges, placed_peak) -> list[int]:
    s = _sizes(t, ranges)
    return _first_fit(ranges, s, sorted(range(len(ranges)),
                                        key=lambda r: (-s[r], ranges[r].begin, ranges[r].wire)))


def place_breadth_area(t, ranges, placed_peak) -> list[int]:
    """Busiest step first: each range is ordered at its step of most live placed bytes (the
    earliest on ties), larger placed x lifetime first, then wire id."""
    s = _sizes(t, ranges)
    live = step_live(t, ranges, max((r.end for r in ranges), default=-1) + 1, True)
    busiest = [max(range(r.begin, r.end + 1), key=lambda k: (live[k], -k)) for r in ranges]
    return _first_fit(ranges, s, sorted(
        range(len(ranges)), key=lambda r: (-live[busiest[r]], busiest[r],
                                           -s[r] * (ranges[r].end - ranges[r].begin + 1),
                                           ranges[r].wire)))


def _ping_pong_offset(ranges, sizes, offsets, placed_ids, r, peak) -> int:
    size, prev, low, high = sizes[r], 0, None, None
    co_live = _co_live_placed(ranges, offsets, placed_ids, r)
    for q in co_live:
        if offsets[q] > prev:
            if low is None and offsets[q] - prev >= size:
                low = prev
            if min(offsets[q], peak) - prev >= size:
                high = min(offsets[q], peak) - size
        prev = max(prev, offsets[q] + sizes[q])
    if low is None:
        low = prev
    if peak - prev >= size:
        high = peak - size
    if not co_live or high is None:
        return low
    latest = max(co_live, key=lambda q: (ranges[q].begin, q))
    return high if 2 * offsets[latest] + sizes[latest] < peak else low


def place_ping_pong(t, ranges, placed_peak) -> list[int]:
    """Stream order; a range whose latest-begun co-live range sits in the lower half of the
    placed peak takes the highest gap below the peak, else the lowest gap."""
    s = _sizes(t, ranges)
    offsets = [0] * len(ranges)
    done: list[int] = []
    for r in sorted(range(len(ranges)), key=_stream_key(ranges, s)):
        offsets[r] = _ping_pong_offset(ranges, s, offsets, done, r, placed_peak)
        done.append(r)
    return offsets


def place_two_ended(t, ranges, placed_peak) -> list[int]:
    """Ranges alive across the first backward step first fit from the bottom; the others
    first fit on their own and are mirrored down from the top."""
    if not has_grad_range(t, ranges):
        raise ValueError("remat[arena]: two-ended placement needs a GRAD range")
    s = _sizes(t, ranges)
    boundary = min(r.begin for r in ranges if t.wires[r.wire].kind == "GRAD")
    stored = [boundary > r.begin and boundary <= r.end for r in ranges]
    bottom = sorted((r for r in range(len(ranges)) if stored[r]),
                    key=lambda r: (ranges[r].begin, -ranges[r].end, r))
    others = sorted((r for r in range(len(ranges)) if not stored[r]), key=_stream_key(ranges, s))
    low, high = _first_fit(ranges, s, bottom), _first_fit(ranges, s, others)
    offsets = [low[r] if stored[r] else high[r] for r in range(len(ranges))]
    ends = [offsets[r] + s[r] for r in range(len(ranges))]
    height = max([0] + ends + [ends[a] + ends[b] for a in bottom for b in others
                               if _co_live(ranges[a], ranges[b])])
    return [offsets[r] if stored[r] else height - ends[r] for r in range(len(ranges))]


def place_size_then_end(t, ranges, placed_peak) -> list[int]:
    s = _sizes(t, ranges)
    return _first_fit(ranges, s, sorted(
        range(len(ranges)), key=lambda r: (-s[r], -ranges[r].end, ranges[r].begin, r)))


Member = Callable[[WireTable, Sequence[Range], int], list[int]]
# C's member array, in its order (the order decides ties): (name, member, needs a GRAD range).
MEMBERS: tuple[tuple[str, Member, bool], ...] = (
    ("first-fit-decreasing", place_first_fit_decreasing, False),
    ("breadth/area", place_breadth_area, False),
    ("ping-pong", place_ping_pong, False),
    ("two-ended", place_two_ended, True),
    ("size-then-end", place_size_then_end, False),
)


def place_portfolio(t: WireTable, ranges: Sequence[Range],
                    members: Sequence[tuple[str, Member, bool]] = MEMBERS) -> Placement:
    peak = max(step_live(t, ranges, max((r.end for r in ranges), default=-1) + 1, True),
               default=0)
    grad = has_grad_range(t, ranges)
    best = None
    for name, member, needs_grad in members:
        if needs_grad and not grad:
            continue
        offsets = member(t, ranges, peak)
        size = extent(t, ranges, offsets)
        if best is None or size < best.bytes:
            best = Placement(size, tuple(offsets), name)
    return best
