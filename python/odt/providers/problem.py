"""Build the remat memory-problem (schema v0) from an annotated creator IR."""

import copy
from collections.abc import Mapping

from .annotate import Annotated

SCHEMA_VERSION = 0
LOSSES = ("CROSS_ENTROPY", "MSE")


def _plain(value: object) -> object:
    """Recursively turn mappings and tuples into plain dicts and lists."""
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _layer(block: Mapping[str, object]) -> dict[str, object]:
    layer: dict[str, object] = {
        "type": block["odt_type"],
        "frozen": bool(block["frozen"]),
        "trackRunningStats": False,
        "outShape": _plain(block["out_shape"]),
        "outputDtype": _plain(block["output_q"]),
        "propLossDtype": _plain(block["prop_loss_q"]),
    }
    if "cost" in block:
        layer["cost"] = _plain(block["cost"])
    return layer


def problem_from_annotated(annotated: Annotated, loss: str) -> dict:
    """Emit the plain-JSON problem for a chain-shaped annotated IR."""
    if loss not in LOSSES:
        raise ValueError(f"loss must be one of {LOSSES}, got {loss!r}")

    nodes = annotated.root.nodes
    (input_node,) = [n for n in nodes.values() if n.type == "input"]
    input_block = input_node.attributes["training"]

    layers = [_layer(nodes[name].attributes["training"]) for name in annotated.order]
    if loss == "CROSS_ENTROPY":
        last = layers[-1]
        layers.append(
            {
                "type": "SOFTMAX",
                "frozen": False,
                "trackRunningStats": False,
                # copies, so editing one layer never changes the other
                "outShape": copy.deepcopy(last["outShape"]),
                "outputDtype": copy.deepcopy(last["outputDtype"]),
                "propLossDtype": copy.deepcopy(last["propLossDtype"]),
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "input": {
            "shape": _plain(input_block["out_shape"]),
            "dtype": _plain(input_block["output_q"]),
        },
        "loss": loss,
        "layers": layers,
    }
