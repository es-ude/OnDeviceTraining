"""Turn a PyTorch model into the on-device-training problem description."""

from torch import nn

from .annotate import annotate
from .errors import NotAChainError, UnsupportedDtypeError, UnsupportedLayerError
from .problem import problem_from_annotated
from .profile import FLOAT32_UNIFORM, QuantProfile
from .trace import to_shaped_ir

__all__ = [
    "FLOAT32_UNIFORM",
    "NotAChainError",
    "QuantProfile",
    "UnsupportedDtypeError",
    "UnsupportedLayerError",
    "build_problem",
]


def build_problem(
    model: nn.Module,
    input_shape: tuple[int, int, int],
    loss: str,
    profile: QuantProfile = FLOAT32_UNIFORM,
) -> dict:
    """Trace ``model`` at ``input_shape`` and describe it as a plain-JSON problem."""
    root, registry = to_shaped_ir(model, input_shape)
    return problem_from_annotated(annotate(root, registry, model, profile), loss)
