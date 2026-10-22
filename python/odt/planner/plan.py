"""What a solver strategy returns: a plan C can run, or "nothing fits" with a lower bound."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Plan:
    """A retain plan: `mask` bit j - 1 keeps ACT j (ACT n always kept; bits that change no
    step are 0), with its numbers as C reports them in DRY_PLAN."""
    n: int
    mask: int
    in_place: bool
    arena_b: int
    wires_peak_b: int
    placed_peak_b: int
    recompute_cost: int
    refwd_steps: int
    forward_cost: int
    steps: int
    policy: str = "retain"
    metadata_b: int | None = None  # C's wire_metadata_b: host struct sizes, known only to C

    def kept(self) -> tuple[int, ...]:
        return tuple(j for j in range(1, self.n + 1) if self.mask >> (j - 1) & 1)

    def tie_key(self) -> tuple:
        """Least recompute, then fewer REFORWARD steps, lower raw peak, without in-place
        first, smaller mask. No key depends on placement, so a better placer can only make
        more plans fit; it never changes which plan wins a budget."""
        return (self.recompute_cost, self.refwd_steps, self.wires_peak_b, self.in_place,
                self.mask)

    def to_remat_plan(self) -> str:
        """The REMAT_PLAN value that runs this plan in the example trainers."""
        return "retain:" + ",".join(map(str, self.kept())) + ("+inplace" if self.in_place else "")


@dataclass(frozen=True)
class NothingFits:
    """No plan's arena is at most `budget_b`. `peak_floor_b` is the smallest placed peak of
    any plan: a lower bound on the arena floor (the frontier's last point is the floor)."""
    budget_b: int
    peak_floor_b: int
