"""JSON log schema shared by train_pytorch.py and train_c.c.

The C side writes this format by hand using printf, so the schema here
is also the contract C must follow. Keep keys ASCII and avoid nested
constructs that complicate hand-written JSON emission.
"""
# Deliberately NO `from __future__ import annotations`: stringized annotations
# make typing.TypedDict put every NotRequired key into __required_keys__, so the
# optional-key declarations below become runtime lies (pinned by
# python/tests/test_log_schema.py).
import json
import re
from pathlib import Path
from typing import NotRequired, TypedDict


class TrainConfig(TypedDict):
    epochs: int
    batch: int
    lr: float
    momentum: NotRequired[float]  # absent for optimizers without momentum (#328)
    seed: int
    shuffle_seed: int
    lr_schedule: NotRequired[str]  # "none" | "cosine" (#327) | "step" | "exp" (batch-size scheduler port); absent = constant LR
    lr_min: NotRequired[float]
    optimizer: NotRequired[str]  # "sgd" | "adamw" (#328); absent = sgd
    weight_decay: NotRequired[float]
    weight_dtype: NotRequired[str]  # "sym" | "asym" (#300); absent = sym
    group_mode: NotRequired[str]  # "tensor" | "channel" | "size" (#300); absent = tensor
    group_size: NotRequired[int]  # GROUP_SIZE when group_mode == "size"; 0 otherwise
    groups_resolved: NotRequired[dict[str, list[int]]]  # per-layer [numGroups, groupSize] (#300)
    group_overhead_b: NotRequired[int]  # Σ per-tensor numGroups·(4 + asym?2:0), all 8 param tensors (#300)
    odts_roundtrip: NotRequired[str]  # "ok" iff ODTS_ROUNDTRIP=1 demo passed; absent otherwise (#300)
    bs_schedule: NotRequired[str]  # "none" | "step" | "exp": batch DIVIDED by gamma per epoch; absent = constant batch
    bs_lr_compensation: NotRequired[int]  # 0/1: 1 iff a batch schedule is active AND compensation was requested (the effective BC arm)
    gamma: NotRequired[float]  # shared factor of lr_schedule step/exp (LR x gamma) and bs_schedule (batch / gamma)
    step_size: NotRequired[int]  # step schedules only
    max_batch_size: NotRequired[int]  # cap of the batch scheduler; `batch` stays the INITIAL batch
    reshuffle: NotRequired[int]  # 0/1: per-epoch reshuffle of the train loader (#381); HAR default 1
    toolchain: NotRequired[str]  # the C compiler's __VERSION__ (provenance; docs/conventions/toolchain-parity.md)


class EpochLog(TypedDict):
    epoch: int
    step_losses: list[float]
    train_loss: float
    val_loss: float
    val_acc: float | None
    wall_s: float
    lr: NotRequired[float]  # LR this epoch trained with (#327)
    batch_size: NotRequired[int]  # train batch this epoch trained with (epochInfo_t)
    parameter_updates: NotRequired[int]  # optimizer steps this epoch = datasetSize // batch_size


class FinalLog(TypedDict):
    test_loss: float
    test_acc: float | None
    test_auc: float | None


class MemoryLog(TypedDict):
    """Per-run memory breakdown (bytes unless noted), emitted only by C runs
    built with -DODT_MEM_PROFILE. All keys are written verbatim by the C side
    (see examples/har_classifier/mem_instrument.c). ``reconciliation_gap_b`` is
    ``heap_peak_b - mcu_total_b`` and is RECORDED, never massaged.
    """
    sym_bits: int  # SYM weight width for the sym run; -1 for the float run
    dataset_b: int  # instrumented phase mark: live bytes after initDataSets
    params_grads_b: int  # instrumented phase-mark delta: buildModel (+requantize)
    optstate_b: int  # instrumented phase-mark delta: optimizer creation
    params_b: int  # analytic: weight+bias tensor bytes (dtype-aware)
    grads_b: int  # analytic: grad tensor bytes
    optstate_analytic_b: int  # analytic: optimizer momentum-buffer bytes
    activations_b: int  # analytic: forward-wire bytes only (NOT the true peak — see dx_peak_b)
    io_b: int  # analytic: batched input + one-hot label bytes
    pool_backward_b: int  # analytic: persistent MaxPool argmax-index buffers (#321)
    dx_peak_b: int  # analytic: worst concurrent dx ping-pong pair during backprop (#321)
    mcu_total_b: int  # params+grads+optstate+activations+io+pool_backward+dx_peak
    heap_peak_b: int  # instrumented: memProfilePeakBytes()
    stack_peak_b: int  # instrumented: measurePeakStackBytes() on one step
    rss_peak_kb: int  # instrumented: memProfileRssPeakKb() (KiB)
    reconciliation_gap_b: int  # heap_peak_b - mcu_total_b (signed)


class RunLog(TypedDict, total=False):
    impl: str  # "pytorch" or "c"
    example: str
    config: TrainConfig
    epochs: list[EpochLog]
    final: FinalLog
    memory: MemoryLog  # optional; present only in -DODT_MEM_PROFILE C runs


_REQUIRED_TOP = ("impl", "config", "epochs", "final", "example")


# C printf("%.6f", x) emits bare nan/-nan/inf/-inf for non-finite values; json
# rejects those tokens but accepts NaN/Infinity/-Infinity. Bound the match to JSON
# value positions (preceded/followed by a structural char) so it never touches a
# token inside a string.
_NONFINITE_RE = re.compile(r"(?<=[:\s,\[])(-?)(nan|inf)(?=[,\s}\]])", re.IGNORECASE)


def _sanitize_nonfinite(text: str) -> str:
    """Rewrite a C emitter's bare nan/-nan/inf/-inf into json-parseable
    NaN/Infinity/-Infinity. A divergent training run (loss -> inf/nan) must stay a
    RECORDED finding, not a JSONDecodeError that drops the whole sweep."""
    def repl(m: "re.Match[str]") -> str:
        sign, word = m.group(1), m.group(2).lower()
        return "NaN" if word == "nan" else f"{sign}Infinity"

    return _NONFINITE_RE.sub(repl, text)


def dump_log(path: Path | str, log: RunLog) -> None:
    Path(path).write_text(json.dumps(log, indent=2))


def load_log(path: Path | str) -> RunLog:
    data = json.loads(_sanitize_nonfinite(Path(path).read_text()))
    for key in _REQUIRED_TOP:
        if key not in data:
            raise KeyError(f"log file {path}: missing required key {key!r}")
    return data  # type: ignore[return-value]
