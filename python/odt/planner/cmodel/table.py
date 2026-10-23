"""The wire table and the per-layer rules, as C builds them (RematWireTable.c, RematPlan.c,
Layer.c, LayerConfigAccess.c). Changing one of those C rules means changing it here too:
the parity tests compare every number against C output."""
import math
from dataclasses import dataclass

from odt.planner.accept import SIZE_MAX, UnsupportedProblem, require_supported
from odt.planner.schema import LayerSpec, Problem

ALIGN = 8  # ODT_WIRE_ALIGN: every placed buffer is rounded up to it
PARAM_TYPES = ("LINEAR", "CONV1D")  # the accepted types that have parameters
READS_INPUT = {"LINEAR": None, "CONV1D": None, "RELU": True, "SOFTMAX": True,
               "MAXPOOL1D": False, "AVGPOOL1D": False, "FLATTEN": False}  # None: unless frozen
FLOAT32_BYTES = 4


@dataclass(frozen=True)
class Wire:
    kind: str  # "ACT" or "GRAD"
    index: int  # ACT j: the input of layer j (ACT n: the output); GRAD l: d loss / d ACT l
    bytes: int
    dtype: str


@dataclass(frozen=True)
class WireTable:
    n: int
    deepest: int  # the first layer with parameters that trains; n when none does
    top: int  # the last layer the backward runs: n - 2 under cross entropy
    has_backward: bool
    wires: tuple[Wire, ...]
    layer_types: tuple[str, ...]
    reads_input: tuple[bool, ...]  # BACKWARD(l) reads ACT l
    layer_cost: tuple[int, ...]  # the forward's multiply-accumulate count

    def grad_id(self, l: int) -> int:
        """The seed GRAD n is wire n + 1; GRAD top .. deepest + 1 follow in that order."""
        return self.n + 1 if l == self.n else self.n + 2 + (self.top - l)


def placed(nbytes: int) -> int:
    return (nbytes + ALIGN - 1) // ALIGN * ALIGN


def backward_reads_input(layer: LayerSpec) -> bool:
    reads = READS_INPUT[layer.type]
    return not layer.frozen if reads is None else reads


def forward_cost(layer: LayerSpec, in_elements: int, out_elements: int) -> int:
    """C's layerForwardCost: the switch is on the layer type, not on which operand is set."""
    t = layer.type
    if t in ("LINEAR", "CONV1D"):
        return out_elements * layer.mac_per_element
    if t == "CONV1D_TRANSPOSED":
        return in_elements * layer.mac_per_element
    if t in ("MAXPOOL1D", "AVGPOOL1D"):
        return out_elements * layer.kernel
    if t == "ADAPTIVE_AVGPOOL1D":
        return in_elements
    return 0 if t == "FLATTEN" else out_elements


def wire_table(problem: Problem, m: int) -> WireTable:
    """Raises UnsupportedProblem for anything without a C reference run."""
    require_supported(problem, m)
    layers = problem.layers
    n = len(layers)
    shapes = [problem.input.shape] + [layer.out_shape for layer in layers]
    elements = [m * math.prod(shape[1:]) for shape in shapes]
    act_dtype = [problem.input.dtype.kind] + [layer.output_dtype.kind for layer in layers]
    deepest = next((j for j, layer in enumerate(layers)
                    if layer.type in PARAM_TYPES and not layer.frozen), n)
    top = n - 1 - (1 if problem.loss == "CROSS_ENTROPY" else 0)
    has_backward = deepest < n
    wires = [Wire("ACT", j, elements[j] * FLOAT32_BYTES, act_dtype[j]) for j in range(n + 1)]
    for w in wires:  # C's elementsOf, then wireBytes, per ACT wire, ACT 0 included
        for size, unit in ((elements[w.index], "elements"), (w.bytes, "B")):
            if size > SIZE_MAX:
                raise UnsupportedProblem(
                    f"wire ACT {w.index}: {size} {unit} has no C reference run; the planner "
                    f"accepts a wire of at most {SIZE_MAX} {unit}, as C's wire table does")
    if has_backward:
        wires.append(Wire("GRAD", n, wires[n].bytes, wires[n].dtype))
        wires += [Wire("GRAD", l, elements[l] * FLOAT32_BYTES, layers[l].prop_loss_dtype.kind)
                  for l in range(top, deepest, -1)]
    total = 0
    for w in wires[1:]:  # C's id order: ACT 1 .. ACT n, then the GRAD wires
        total += w.bytes
        if total > SIZE_MAX:
            raise UnsupportedProblem(
                f"wires: more than {SIZE_MAX} B in total (passed at wire {w.kind} {w.index}) has "
                f"no C reference run; the planner accepts wires that total at most {SIZE_MAX} B, "
                "as C's wire table does")
    costs = tuple(forward_cost(layer, elements[j], elements[j + 1])
                  for j, layer in enumerate(layers))
    cost_total = 0
    for j, cost in enumerate(costs):  # every C plan build: layerForwardCost, then the sum
        if cost > SIZE_MAX:
            raise UnsupportedProblem(
                f"layers[{j}].cost: a forward cost of {cost} has no C reference run; the "
                f"planner accepts a layer forward cost of at most {SIZE_MAX}, as C's "
                "layerForwardCost does")
        cost_total += cost
        if cost_total > SIZE_MAX:
            raise UnsupportedProblem(
                f"layers: more than {SIZE_MAX} forward cost in total (passed at layers[{j}]) "
                f"has no C reference run; the planner accepts layers whose forward costs total "
                f"at most {SIZE_MAX}, as C's plan build does")
    return WireTable(n, deepest, top, has_backward, tuple(wires),
                     tuple(layer.type for layer in layers),
                     tuple(backward_reads_input(layer) for layer in layers), costs)
