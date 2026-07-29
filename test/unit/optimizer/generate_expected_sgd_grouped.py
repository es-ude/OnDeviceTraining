#!/usr/bin/env python3
"""Generate expected_sgd_grouped.h for UnitTestSgd (group-quant PR3 Task 4 --
spec docs/superpowers/specs/2026-07-28-group-quantization-design.md,
task-4-brief.md).

Pins sgdStepM's grouped-SYM param update: the update opSpecs declare the
param's groupedSymOperandPos, so the executeOp funnel's EXISTING FLOAT32
prologue/epilogue does the requant -- per-group dequant on the way in,
fresh per-group absmax requant (packFloatBufferAsSym's grouped path) on the
way out. No new C-side conversion code; this generator's job is to emulate
that funnel path bit-for-bit in float32 (never float64 -- see
sym_gold.dequant_sym_grouped_f32/requant_absmax_grouped_f32 docstrings).

Two fixtures, mirroring sgdStepM's two DISTINCT code paths (Sgd.c):

  step0     momentumFactor == 0 -- the single-op fast path (sgdUpdateKernel
            {param, grad} -> groupedSymOperandPos==1): g = grad + wd*paramDeq;
            new = paramDeq - lr*g; per-group requant. Emulated by
            sym_gold.sgd_grouped_step_ref.

  momentum  momentumFactor > 0 -- the TWO-op path, in the EXACT sequence
            sgdStepM issues (read Sgd.c:144-174 before touching this):
              op1 sgdMStateKernel {state, grad, param} (param declared at
                  groupedSymOperandPos==3): newState = momentum*state +
                  (grad + wd*paramDeq). state/grad are per-tensor FLOAT32
                  (the momentum-state carrier gate, PR2 -- states never
                  group), so this op has NO quantization/rounding of its
                  own: a straight float32 write-back.
              op2 sgdMParamKernel {param, state} (param declared at
                  groupedSymOperandPos==1): newParam = paramDeq - lr*newState
                  (paramDeq is RE-DERIVED here -- op1 never touched param,
                  same mantissas/scales both times), then the per-group
                  absmax requant.
            Emulated directly below (sgd_grouped_momentum_step_ref) by
            composing the SAME two float32-precise primitives
            sgd_grouped_step_ref uses internally, in this exact order --
            kept local (not promoted into sym_gold.py) since the two-op
            SEQUENCE is SGD-specific, not a reusable cross-generator
            primitive like the dequant/requant helpers themselves.

Both fixtures use qBits=8 (byte-aligned, numGroups=2, groupSize=3, N=6) and
HALF_AWAY write-back rounding (the #279 explicit opt-out -- the C test must
call optimizerSetWriteBackRounding(optim, HALF_AWAY) for these golds to hold;
factories default to seeded SR_HALF_AWAY, which this generator does NOT
emulate, consistent with every other goldgen script in this tree, see
matmul_grouped_ref's docstring).

Self-checks (mutation-discriminating fixture properties, asserted here so a
broken fixture aborts generation rather than silently passing a vacuous
test):
  (i)  post-step scales are pairwise distinct across groups -- a fixture
       where both groups land on the same fresh scale could not tell a
       correct per-group requant from an accidental single-scale one.
  (ii) the per-group result's mantissas differ from a WHOLE-TENSOR requant
       emulation (group_size == N, i.e. numGroups collapsed to 1) of the
       SAME float32 update -- pins the group-collapse mutation (a bug that
       forgets per-group boundaries and derives one absmax over the entire
       tensor).

Run via `uv run` (CMake wires this automatically, see CMakeLists.txt).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "goldgen"))

from sym_gold import (dequant_sym_grouped_f32, emit_float_array, emit_float_scalar,
                      emit_int32_array, emit_int32_scalar, requant_absmax_grouped_f32,
                      sgd_grouped_step_ref)

Q_BITS = 8
GROUP_SIZE = 3
NUM_GROUPS = 2
N = GROUP_SIZE * NUM_GROUPS  # 6


def sgd_grouped_momentum_step_ref(param_mantissas, param_scales, group_size: int, q_bits: int,
                                  grad, state_prev, momentum: float, lr: float,
                                  weight_decay: float = 0.0):
    """SGD momentum>0 grouped-SYM param update -- see this module's docstring
    for the exact two-op sequence this mirrors (sgdMStateKernel then
    sgdMParamKernel, Sgd.c:144-174). Returns (new_mantissas: list[int],
    new_scales: list[float], new_state: torch.Tensor [per-tensor FLOAT32,
    no further encoding])."""
    param_deq = dequant_sym_grouped_f32(param_mantissas, param_scales, group_size)
    g = torch.as_tensor(grad, dtype=torch.float32)
    state = torch.as_tensor(state_prev, dtype=torch.float32)
    wd_t = torch.tensor(weight_decay, dtype=torch.float32)

    # op1 sgdMStateKernel: g = grad + wd*paramDeq; newState = momentum*state + g.
    combined = g + wd_t * param_deq
    momentum_t = torch.tensor(momentum, dtype=torch.float32)
    new_state = momentum_t * state + combined

    # op2 sgdMParamKernel: newParam = paramDeq - lr*newState; per-group requant.
    lr_t = torch.tensor(lr, dtype=torch.float32)
    new_param_float = param_deq - lr_t * new_state
    new_mant, new_scales = requant_absmax_grouped_f32(new_param_float, q_bits, group_size)
    return new_mant, new_scales, new_state


def check_no_collapse(name, new_mantissas, new_scales, new_param_float, q_bits, group_size, n):
    """Self-checks (i)-(ii) from the module docstring; abort on failure."""
    assert len(set(new_scales)) == len(new_scales), (
        f"{name}: post-step scales are not pairwise distinct -- fixture cannot "
        "discriminate a correct per-group requant from an accidental single scale")

    collapsed_mantissas, _ = requant_absmax_grouped_f32(new_param_float, q_bits, n)
    assert new_mantissas != collapsed_mantissas, (
        f"{name}: per-group result is indistinguishable from a whole-tensor "
        "(collapsed) requant -- fixture is vacuous against the group-collapse mutation")


def fixture_step0():
    param_mantissas = [40, -90, 100, -30, 70, -10]
    param_scales = [0.05, 0.02]
    grad = [0.3, -0.15, 0.05, 0.2, -0.25, 0.1]
    lr = 0.1
    weight_decay = 0.01

    new_mantissas, new_scales = sgd_grouped_step_ref(
        param_mantissas, param_scales, GROUP_SIZE, Q_BITS, grad, lr, weight_decay)

    # Recompute the float32 update (pre-requant) for the collapse self-check.
    param_deq = dequant_sym_grouped_f32(param_mantissas, param_scales, GROUP_SIZE)
    g = torch.as_tensor(grad, dtype=torch.float32)
    combined = g + torch.tensor(weight_decay, dtype=torch.float32) * param_deq
    new_param_float = param_deq - torch.tensor(lr, dtype=torch.float32) * combined
    check_no_collapse("step0", new_mantissas, new_scales, new_param_float, Q_BITS, GROUP_SIZE, N)

    return {
        "paramMantissas": param_mantissas, "paramScales": param_scales, "grad": grad,
        "lr": lr, "weightDecay": weight_decay,
        "newMantissas": new_mantissas, "newScales": new_scales,
    }


def fixture_momentum():
    param_mantissas = [60, -20, 110, -50, 30, -80]
    param_scales = [0.03, 0.04]
    grad = [0.2, 0.1, -0.3, 0.15, -0.05, 0.25]
    state_prev = [0.5, -0.3, 0.2, -0.4, 0.1, -0.6]
    momentum = 0.9
    lr = 0.05
    weight_decay = 0.0

    new_mantissas, new_scales, new_state = sgd_grouped_momentum_step_ref(
        param_mantissas, param_scales, GROUP_SIZE, Q_BITS, grad, state_prev, momentum, lr,
        weight_decay)

    param_deq = dequant_sym_grouped_f32(param_mantissas, param_scales, GROUP_SIZE)
    g = torch.as_tensor(grad, dtype=torch.float32)
    state_t = torch.as_tensor(state_prev, dtype=torch.float32)
    combined = g + torch.tensor(weight_decay, dtype=torch.float32) * param_deq
    new_state_check = torch.tensor(momentum, dtype=torch.float32) * state_t + combined
    assert torch.equal(new_state_check, new_state), "momentum: new_state recompute mismatch"
    new_param_float = param_deq - torch.tensor(lr, dtype=torch.float32) * new_state
    check_no_collapse("momentum", new_mantissas, new_scales, new_param_float, Q_BITS, GROUP_SIZE,
                      N)

    return {
        "paramMantissas": param_mantissas, "paramScales": param_scales, "grad": grad,
        "statePrev": state_prev, "momentum": momentum, "lr": lr, "weightDecay": weight_decay,
        "newMantissas": new_mantissas, "newScales": new_scales,
        "newState": new_state.tolist(),
    }


def emit_fixture(parts, prefix, fx, momentum: bool):
    parts.append(emit_int32_array(f"{prefix}ParamMantissas", torch.tensor(fx["paramMantissas"])))
    parts.append(emit_float_array(f"{prefix}ParamScales", torch.tensor(fx["paramScales"])))
    parts.append(emit_float_array(f"{prefix}Grad", torch.tensor(fx["grad"])))
    parts.append(emit_float_scalar(f"{prefix}Lr", fx["lr"]))
    parts.append(emit_float_scalar(f"{prefix}WeightDecay", fx["weightDecay"]))
    if momentum:
        parts.append(emit_float_array(f"{prefix}StatePrev", torch.tensor(fx["statePrev"])))
        parts.append(emit_float_scalar(f"{prefix}Momentum", fx["momentum"]))
        parts.append(emit_float_array(f"{prefix}NewState", torch.tensor(fx["newState"])))
    parts.append(emit_int32_array(f"{prefix}NewMantissas", torch.tensor(fx["newMantissas"])))
    parts.append(emit_float_array(f"{prefix}NewScales", torch.tensor(fx["newScales"])))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    parts = [
        "// AUTOGENERATED by generate_expected_sgd_grouped.py - DO NOT EDIT\n",
        "#ifndef ODT_EXPECTED_SGD_GROUPED_H\n",
        "#define ODT_EXPECTED_SGD_GROUPED_H\n",
        "#include <stdint.h>\n",
        "#include <stdlib.h>\n\n",
    ]
    parts.append(emit_int32_scalar("sgdGroupedQBits", Q_BITS))
    parts.append(emit_int32_scalar("sgdGroupedGroupSize", GROUP_SIZE))
    parts.append(emit_int32_scalar("sgdGroupedNumGroups", NUM_GROUPS))
    parts.append("\n")

    emit_fixture(parts, "sgdGroupedStep0", fixture_step0(), momentum=False)
    parts.append("\n")
    emit_fixture(parts, "sgdGroupedMomentum", fixture_momentum(), momentum=True)

    parts.append("\n#endif // ODT_EXPECTED_SGD_GROUPED_H\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(parts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
