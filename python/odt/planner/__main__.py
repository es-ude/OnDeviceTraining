"""Command line: `uv run -m odt.planner solve <problem.json> --m 1 --budget <bytes>` prints the
REMAT_PLAN and REMAT_BUDGET_B that run the best plan in the example trainers;
`... frontier <problem.json> --m 1` prints one line per budget breakpoint."""
import argparse
import sys

from odt.planner.accept import SIZE_MAX, UnsupportedProblem
from odt.planner.frontier import frontier
from odt.planner.plan import NothingFits
from odt.planner.schema import ProblemError, load_problem
from odt.planner.solve import ExactRetainStrategy
from odt.planner.space import PlanSpace


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m odt.planner")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("solve", "frontier"):
        command = commands.add_parser(name)
        command.add_argument("problem", help="a problem file: odt.providers.problem_bytes or the C DUMP_PROBLEM=1 dump")
        command.add_argument("--m", type=int, required=True, help="the micro-batch")
        command.add_argument("--no-inplace", action="store_true",
                             help="only plans without in-place dx")
        if name == "solve":
            command.add_argument("--budget", type=int, required=True,
                                 help="the arena budget in bytes")
    return parser


def nothing_fits_message(answer: NothingFits) -> str:
    """Below the peak floor no plan can fit at all; at or above it, some plan's placed peak
    fits but C's placement of every such plan needs more than the budget."""
    head = (f"odt.planner: nothing fits REMAT_BUDGET_B={answer.budget_b}: no retain plan's "
            f"arena is at most {answer.budget_b} B (")
    tail = "; the frontier's last point is the smallest arena)"
    if answer.budget_b < answer.peak_floor_b:
        return head + f"the smallest placed peak of any plan is {answer.peak_floor_b} B" + tail
    return head + (f"the smallest placed peak of any plan, {answer.peak_floor_b} B, is within the "
                   "budget, but a placement gap lifts the arena of every such plan above it") + tail


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "solve" and not 1 <= args.budget <= SIZE_MAX:
        parser.error(f"--budget {args.budget}: expected the arena budget in bytes, "
                     f"1 <= value <= {SIZE_MAX}")
    try:
        problem = load_problem(args.problem)
        space = PlanSpace.build(problem, args.m, allow_inplace=not args.no_inplace)
    except (OSError, ProblemError, UnsupportedProblem) as e:
        print(f"odt.planner: {e}", file=sys.stderr)
        return 1
    if args.command == "solve":
        answer = ExactRetainStrategy().solve(space, args.budget)
        if isinstance(answer, NothingFits):
            print(nothing_fits_message(answer), file=sys.stderr)
            return 1
        print(f"# problem fnv1a64={problem.hash}")
        print(f"REMAT_PLAN={answer.to_remat_plan()} REMAT_BUDGET_B={args.budget}")
        return 0
    print(f"# problem fnv1a64={problem.hash}")
    for point in frontier(space):
        p = point.plan
        print(f"budget_lo_b={point.budget_lo_b} plan={p.to_remat_plan()} arena_b={p.arena_b} "
              f"wires_peak_b={p.wires_peak_b} placed_peak_b={p.placed_peak_b} "
              f"recompute_cost={p.recompute_cost} refwd_steps={p.refwd_steps}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
