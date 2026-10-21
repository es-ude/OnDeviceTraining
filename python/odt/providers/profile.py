"""Quantization profiles: the dtype objects written into the training annotation."""

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class QuantProfile:
    """Dtype objects for the model input, every layer output and the propagated loss."""

    input: Mapping[str, object]
    output: Mapping[str, object]
    prop_loss: Mapping[str, object]


FLOAT32_UNIFORM = QuantProfile(
    {"kind": "FLOAT32"}, {"kind": "FLOAT32"}, {"kind": "FLOAT32"}
)
