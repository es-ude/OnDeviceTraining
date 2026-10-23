"""Test-only brute force over every retain mask (ACT n kept) x in-place setting: the
reference the exact solver and the frontier walk are checked against (n <= 12); and the
synthetic problems the planner tests use."""
import json
import random

from odt.planner import Plan, PlanSpace, Problem, parse_problem


def all_plans(space: PlanSpace) -> list[Plan]:
    top = 1 << (space.n - 1)
    return [space.plan(top | low, ip) for ip in space.settings for low in range(top)]


def best(plans: list[Plan], budget_b: int) -> Plan | None:
    return min((p for p in plans if p.arena_b <= budget_b), key=Plan.tie_key, default=None)


def frontier(plans: list[Plan]) -> list[Plan]:
    """The best plan at every distinct arena, largest budget first, repeats dropped."""
    points = []
    for budget in sorted({p.arena_b for p in plans}, reverse=True):
        p = best(plans, budget)
        if not points or p != points[-1]:
            points.append(p)
    return points


def chain_doc(input_shape: list[int], layers: list[tuple]) -> dict:
    """A problem file's content. layers: (type, outShape, frozen, cost or None)."""
    f32 = {"kind": "FLOAT32"}
    doc = {"schema_version": 0, "input": {"shape": input_shape, "dtype": f32},
           "loss": "CROSS_ENTROPY", "layers": []}
    for kind, out_shape, frozen, cost in layers:
        layer = {"type": kind, "frozen": frozen, "trackRunningStats": False,
                 "outShape": out_shape, "outputDtype": f32, "propLossDtype": f32}
        if cost is not None:
            layer["cost"] = cost
        doc["layers"].append(layer)
    return doc


def chain_problem(input_shape: list[int], layers: list[tuple]) -> Problem:
    """A problem through the loader."""
    return parse_problem(json.dumps(chain_doc(input_shape, layers)).encode())


def random_chain_spec(seed: int, n: int) -> tuple[list[int], list[tuple]]:
    """The (input_shape, layers) of n layers ending in SOFTMAX: conv/relu/pool blocks, maybe a
    flatten into linear/relu.
    Up to two layers with parameters at the front are frozen (no backward reads their ACT),
    pools and flattens leave ACTs no backward reads, and one layer with parameters trains."""
    rng = random.Random(seed)
    input_shape = [1, rng.randint(1, 6), rng.choice((16, 32, 64))]
    shape, frozen_left, layers = input_shape, rng.randint(0, 2), []
    while len(layers) < n - 1:
        kind = rng.choice(("CONV1D", "CONV1D", "RELU", "RELU", "MAXPOOL1D", "AVGPOOL1D",
                           "FLATTEN") if len(shape) == 3 else ("LINEAR", "RELU"))
        if not layers:
            kind = "CONV1D"
        cost = None
        if kind == "CONV1D":
            cost, shape = {"macPerElement": shape[1] * rng.choice((1, 3, 5))}, [
                1, rng.randint(1, 12), shape[2]]
        elif kind in ("MAXPOOL1D", "AVGPOOL1D"):
            if shape[2] < 2:
                continue
            cost, shape = {"kernel": 2}, [1, shape[1], shape[2] // 2]
        elif kind == "FLATTEN":
            shape = [1, shape[1] * shape[2]]
        elif kind == "LINEAR":
            cost, shape = {"macPerElement": shape[1]}, [1, rng.randint(2, 48)]
        frozen = kind in ("CONV1D", "LINEAR") and frozen_left > 0
        frozen_left -= frozen
        layers.append((kind, list(shape), frozen, cost))
    trains = [j for j, layer in enumerate(layers) if layer[0] in ("CONV1D", "LINEAR")]
    if all(layers[j][2] for j in trains):
        kind, out_shape, _, cost = layers[trains[-1]]
        layers[trains[-1]] = (kind, out_shape, False, cost)
    layers.append(("SOFTMAX", list(shape), False, None))
    return input_shape, layers


def random_chain(seed: int, n: int) -> Problem:
    """The problem random_chain_spec describes."""
    return chain_problem(*random_chain_spec(seed, n))
