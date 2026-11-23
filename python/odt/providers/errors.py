"""Errors raised while turning a PyTorch model into the training problem."""


class UnsupportedLayerError(ValueError):
    """A module type outside the supported set was found."""

    def __init__(self, layer_type: str, where: str, reason: str = "") -> None:
        message = f"unsupported layer type {layer_type!r} at {where!r}"
        super().__init__(f"{message}: {reason}" if reason else message)
        self.layer_type = layer_type
        self.where = where


class UnsupportedDtypeError(ValueError):
    """A parameter or floating-point buffer whose dtype is not float32 was found.

    Integer buffers are allowed.
    """


class UnsupportedGraphError(ValueError):
    """The model graph is not one the C framework runs: one input, one output,
    every value read, every node reachable from the input, each module with
    parameters or buffers called once, no parameter or buffer held by two
    modules, and an in-place ReLU only in a chain."""

    def __init__(self, node: str, reason: str) -> None:
        super().__init__(f"unsupported graph at node {node!r}: {reason}")
        self.node = node
        self.reason = reason
