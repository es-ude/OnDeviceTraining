"""What the planner accepts: only inputs a C reference run checks (the three example trainers).

Widening it is one rule here plus its C reference case and its copy-of-C rule, never a
schema change: the schema already holds every dtype, loss and layer type.
"""
from odt.planner.schema import Problem, is_int

ACCEPTED_TYPES = ("CONV1D", "RELU", "MAXPOOL1D", "AVGPOOL1D", "FLATTEN", "LINEAR", "SOFTMAX")
# REMAT_SELECT_MAX_LAYERS in examples/_shared/remat_select.h: the harness refuses a retain plan
# for a longer model, so no plan for one could run.
MAX_LAYERS = 64
# REMAT_DUMP_MAX_RANK in examples/_shared/remat_select.c and BATCH_VIEW_MAX_RANK in
# src/userApi/tensor/include/BatchView.h: C can neither dump nor train an input of a higher rank
# (the batch axis counted), so no plan for one could run.
MAX_INPUT_RANK = 8
# size_t on the 64-bit host: examples/_shared/remat_select.c reads REMAT_BUDGET_B into one, and
# src/userApi/training_loop/remat/RematWireTable.c adds every wire's bytes into one and exits by
# name when the total passes it, so no plan for a heavier problem could run.
SIZE_MAX = 2**64 - 1


class UnsupportedProblem(ValueError):
    """A valid problem the planner has no C reference run for; the message names it."""


def _refuse(field: str, value, allowed: str) -> None:
    raise UnsupportedProblem(f"{field}: {value} has no C reference run; the planner accepts "
                             f"{allowed}")


def require_supported(problem: Problem, m: int) -> None:
    if not is_int(m) or m != 1:
        _refuse("m", m, "1")
    if problem.loss != "CROSS_ENTROPY":
        _refuse("loss", problem.loss, "CROSS_ENTROPY")
    if problem.input.dtype.kind != "FLOAT32":
        _refuse("input.dtype", problem.input.dtype.kind, "FLOAT32")
    if len(problem.input.shape) > MAX_INPUT_RANK:
        _refuse("input.shape", f"rank {len(problem.input.shape)}",
                f"ranks 1 to {MAX_INPUT_RANK} (the example harness's REMAT_DUMP_MAX_RANK)")
    if len(problem.layers) > MAX_LAYERS:
        _refuse("layers", len(problem.layers),
                f"at most {MAX_LAYERS} layers (the example harness's REMAT_SELECT_MAX_LAYERS)")
    for j, layer in enumerate(problem.layers):
        if layer.type not in ACCEPTED_TYPES:
            _refuse(f"layers[{j}].type", layer.type, ", ".join(ACCEPTED_TYPES))
        for field, dtype in (("outputDtype", layer.output_dtype),
                             ("propLossDtype", layer.prop_loss_dtype)):
            if dtype.kind != "FLOAT32":
                _refuse(f"layers[{j}].{field}", dtype.kind, "FLOAT32")
