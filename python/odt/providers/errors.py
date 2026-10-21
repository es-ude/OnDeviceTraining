"""Errors raised while turning a PyTorch model into the training problem."""


class UnsupportedLayerError(ValueError):
    """A module type outside the supported set was found."""

    def __init__(self, layer_type: str, where: str) -> None:
        super().__init__(f"unsupported layer type {layer_type!r} at {where!r}")
        self.layer_type = layer_type
        self.where = where


class UnsupportedDtypeError(ValueError):
    """A parameter or floating-point buffer whose dtype is not float32 was found.

    Integer buffers are allowed.
    """


class NotAChainError(ValueError):
    """The model graph is not a single linear chain of layers."""

    def __init__(self, node: str) -> None:
        super().__init__(f"model is not a linear chain at node {node!r}")
        self.node = node
