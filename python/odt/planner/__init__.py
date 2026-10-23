"""Remat planner: for a model, a micro-batch and an arena budget, the retain list that
recomputes least and still fits the arena C will place. Host-only; C stays the truth."""
from odt.planner.accept import UnsupportedProblem, require_supported
from odt.planner.plan import NothingFits, Plan
from odt.planner.schema import (
    Dtype,
    InputSpec,
    LayerSpec,
    Problem,
    ProblemError,
    load_problem,
    parse_problem,
    problem_bytes,
    problem_hash,
)
from odt.planner.solve import ExactRetainStrategy, Strategy, peak_floor
from odt.planner.space import PlanSpace

__all__ = [
    "Dtype",
    "ExactRetainStrategy",
    "InputSpec",
    "LayerSpec",
    "NothingFits",
    "Plan",
    "PlanSpace",
    "Problem",
    "ProblemError",
    "Strategy",
    "UnsupportedProblem",
    "load_problem",
    "parse_problem",
    "peak_floor",
    "problem_bytes",
    "problem_hash",
    "require_supported",
]
