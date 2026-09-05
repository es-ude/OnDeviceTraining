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

Group-quant PR4 Task 3 adds the grouped-ASYM twin of step0 (asymStep0): the
SAME single-op funnel path, only the param's dtype differs -- the FLOAT32
prologue dequants per-group AFFINE ((code - zp[g]) * scale[g],
convertAsymTensorToFloatTensor's grouped path, PR4 Task 2), the kernel is the
identical float32 sgdUpdateKernel, and the OUT_WRITE epilogue re-derives a
fresh NUDGED code-domain grid PER GROUP (quantizeFloatToAsym's grouped path):
scales AND zeroPoints both move every step, so the C test pins post-step
codes AND all scales AND all zps exactly. Extra self-checks for the affine
fixture: pre- AND post-step zps pairwise distinct (a zp[g] -> zp[0] shift
bug must change the result), post-step scales pairwise distinct, and the
group-collapse discriminability check as above.

BFP epic PR3 Task 7 adds the per-tensor-BFP momentum twin (bfpMomentum): the
SAME two-op sequence as the `momentum` fixture, but BOTH the param AND the
momentum state are per-tensor BFP -- so op1's OUT_WRITE now REQUANTIZES the
state (fresh absmax exponent + HALF_AWAY codes, packFloatBufferAsBfp) and
op2 reads the state from those freshly requantized codes, never op1's raw
float result. Emulated by sym_gold.sgd_bfp_step_ref, whose docstring
discloses the exact funnel mirror and whose self-checks (canonical inputs,
param-exponent-moves, both-repacks-change-values) abort generation rather
than emit a fixture that cannot observe the quantization. weight_decay is
deliberately NONZERO here (unlike the `momentum` fixture): with wd=0 a
wd-placement bug in the mirror or the kernel would be invisible.

#420 C3 adds a SECOND BFP momentum fixture (bfpMomV2) alongside the first.
In the original fixture the param and the state both land on stored exponent
121 after the step, so the C test's two exponent asserts are interchangeable:
a param/state exponent CROSS-WIRING is invisible there and the
state-exponent assert only ever kills transitively (through the codes). V2 is
chosen so the two final exponents are FAR apart (120 vs 123) while the param
exponent still moves off its input value, which makes each exponent assert
independently load-bearing. The generator asserts that separation (and that
V2's exponent pair is not V1's) rather than trusting it.

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

from sym_gold import (bfp_dequant_f32, dequant_asym_grouped, dequant_sym_grouped_f32,
                      emit_float_array, emit_float_scalar, emit_int32_array, emit_int32_scalar,
                      quantize_asym_grouped, requant_absmax_grouped_f32, sgd_bfp_step_ref,
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


def fixture_asym_step0():
    """Grouped-ASYM twin of fixture_step0 (PR4 Task 3) -- see the module
    docstring for the exact funnel path this mirrors. The update composes the
    two Task-2 affine primitives (dequant_asym_grouped in,
    quantize_asym_grouped out) around the SAME float32 kernel arithmetic as
    sgd_grouped_step_ref."""
    param_codes = [40, 190, 100, 30, 210, 120]
    param_scales = [0.05, 0.02]
    param_zps = [90, 150]
    grad = [0.3, -0.15, 0.05, 0.2, -0.25, 0.1]
    lr = 0.1
    weight_decay = 0.01

    # Pre-step zp distinctness: a zp[g] -> zp[0] dequant bug must not be able
    # to reproduce the gold.
    assert len(set(param_zps)) == len(param_zps), (
        "asymStep0: pre-step zps must be pairwise distinct")

    # prologue: per-group affine dequant, float32 throughout.
    param_deq = torch.tensor(dequant_asym_grouped(param_codes, param_scales, param_zps,
                                                  GROUP_SIZE), dtype=torch.float32)
    # kernel: g = grad + wd*paramDeq; new = paramDeq - lr*g (sgdUpdateKernel's
    # exact float32 op order, Sgd.c).
    g = torch.as_tensor(grad, dtype=torch.float32)
    combined = g + torch.tensor(weight_decay, dtype=torch.float32) * param_deq
    new_param_float = param_deq - torch.tensor(lr, dtype=torch.float32) * combined
    # epilogue: per-group NUDGED code-domain requant (scales AND zps re-derived).
    new_codes, new_scales, new_zps = quantize_asym_grouped(new_param_float.tolist(), Q_BITS,
                                                           GROUP_SIZE)

    # Self-checks (module docstring): post-step grid discriminability.
    assert len(set(new_zps)) == len(new_zps), (
        "asymStep0: post-step zps are not pairwise distinct -- fixture cannot "
        "discriminate a per-group zp re-derivation from a shared one")
    assert len(set(new_scales)) == len(new_scales), (
        "asymStep0: post-step scales are not pairwise distinct")
    collapsed_codes, _, _ = quantize_asym_grouped(new_param_float.tolist(), Q_BITS, N)
    assert new_codes != collapsed_codes, (
        "asymStep0: per-group result is indistinguishable from a whole-tensor "
        "(collapsed) requant -- fixture is vacuous against the group-collapse mutation")

    return {
        "paramCodes": param_codes, "paramScales": param_scales, "paramZps": param_zps,
        "grad": grad, "lr": lr, "weightDecay": weight_decay,
        "newCodes": new_codes, "newScales": new_scales, "newZps": new_zps,
    }


BFP_MANTISSA_BITS = 8
BFP_EXPONENT_BITS = 8


def fixture_bfp_momentum():
    """Per-tensor-BFP momentum twin of fixture_momentum (BFP epic PR3 Task 7)
    -- see the module docstring. Input codes/exponents are chosen so that
    (a) each absmax code sits in (qMax/2, qMax] (canonical: requantizing the
    exact dequant reproduces codes AND exponents bit-for-bit -- the C test
    builds both tensors via FLOAT32 init + requantizeTensorInPlace from the
    emitted float values), and (b) the param absmax crosses a binade during
    the update (127/128 = 0.992 grows past 1.0 at index 2, where the state
    push is negative), so a write-back that forgets to re-derive the shared
    exponent is observable. sgd_bfp_step_ref aborts if either property (or
    repack-changes-values) fails to hold."""
    qc = {"mantissa_bits": BFP_MANTISSA_BITS, "exponent_bits": BFP_EXPONENT_BITS,
          "group_size": 0}
    param_codes = [100, -50, 127, -30, 60, -90]
    param_exps = [120]  # stored; bias 127 -> scale 2^-7
    state_codes = [-64, 96, -120, 45, -80, 30]
    state_exps = [121]  # scale 2^-6
    grad = [0.31, -0.47, 0.11, 0.26, -0.33, 0.18]
    momentum = 0.9
    lr = 0.05
    weight_decay = 0.05  # NONZERO -- pins the wd placement (module docstring)

    new_param_codes, new_param_exps, new_state_codes, new_state_exps = sgd_bfp_step_ref(
        param_codes, param_exps, qc, grad, lr, momentum, state_codes, state_exps, qc,
        weight_decay=weight_decay)

    param_deq = bfp_dequant_f32(param_codes, param_exps, qc)
    state_deq = bfp_dequant_f32(state_codes, state_exps, qc)
    return {
        "paramValues": param_deq.tolist(), "statePrev": state_deq.tolist(), "grad": grad,
        "lr": lr, "momentum": momentum, "weightDecay": weight_decay,
        "newParamCodes": new_param_codes, "newParamExp": new_param_exps[0],
        "newStateCodes": new_state_codes, "newStateExp": new_state_exps[0],
    }


def fixture_bfp_momentum_v2():
    """#420 C3: the state-binade-cross BFP momentum fixture. Same two-op
    sequence and same self-checks as fixture_bfp_momentum, but the operand
    values are picked so the PARAM and the STATE land on DIFFERENT stored
    exponents after the step (120 vs 123) instead of coinciding on 121.

    Why that matters: with coincident final exponents the C test's
    `paramExp`/`stateExp` asserts are interchangeable, so a cross-wiring that
    binds the state's assert to the param's grid (or vice versa) passes, and
    the state-exponent assert can only ever fail transitively via the codes.
    With the exponents three binades apart each assert kills on its own.

    Geometry of the choice: the param's dequantized absmax (125/2^8 =
    0.48828125 at stored 119) is pushed just past 0.5 by the -lr*state term,
    so the param grid moves 119 -> 120 (fixture_bfp_momentum's
    param-exponent-moves self-check still holds), while the state's absmax
    (127/2^5 = 3.96875 at stored 122) is amplified by momentum*state + g into
    the next binade, 122 -> 123. Both operands stay canonical (absmax code in
    (qMax/2, qMax]) so requantizeTensorInPlace on the emitted floats
    reproduces these exact codes and exponents in C."""
    qc = {"mantissa_bits": BFP_MANTISSA_BITS, "exponent_bits": BFP_EXPONENT_BITS,
          "group_size": 0}
    param_codes = [108, -4, 42, -74, -125, 59]
    param_exps = [119]  # stored; bias 127 -> scale 2^-8
    state_codes = [-99, 12, 127, 73, -120, 97]
    state_exps = [122]  # scale 2^-5
    grad = [0.12, 0.2, 0.46, -0.08, -0.12, -0.48]
    momentum = 0.9
    lr = 0.05
    weight_decay = 0.05

    new_param_codes, new_param_exps, new_state_codes, new_state_exps = sgd_bfp_step_ref(
        param_codes, param_exps, qc, grad, lr, momentum, state_codes, state_exps, qc,
        weight_decay=weight_decay)

    # The discrimination this fixture exists for -- abort rather than emit a
    # second fixture that is no stronger than the first.
    assert new_param_exps[0] != new_state_exps[0], (
        f"fixture_bfp_momentum_v2: final param exponent {new_param_exps[0]} equals the final "
        f"state exponent -- a param/state exponent cross-wiring stays unobservable; pick "
        "values whose state absmax crosses a binade the param's does not")
    assert new_state_exps[0] != state_exps[0], (
        "fixture_bfp_momentum_v2: the state exponent did not move -- the state write-back's "
        "own exponent derivation stays unpinned")

    param_deq = bfp_dequant_f32(param_codes, param_exps, qc)
    state_deq = bfp_dequant_f32(state_codes, state_exps, qc)
    return {
        "paramValues": param_deq.tolist(), "statePrev": state_deq.tolist(), "grad": grad,
        "lr": lr, "momentum": momentum, "weightDecay": weight_decay,
        "newParamCodes": new_param_codes, "newParamExp": new_param_exps[0],
        "newStateCodes": new_state_codes, "newStateExp": new_state_exps[0],
    }


def emit_bfp_fixture(parts, prefix, fx):
    parts.append(emit_int32_scalar(f"{prefix}MantissaBits", BFP_MANTISSA_BITS))
    parts.append(emit_int32_scalar(f"{prefix}ExponentBits", BFP_EXPONENT_BITS))
    parts.append(emit_float_array(f"{prefix}ParamValues", torch.tensor(fx["paramValues"])))
    parts.append(emit_float_array(f"{prefix}StatePrev", torch.tensor(fx["statePrev"])))
    parts.append(emit_float_array(f"{prefix}Grad", torch.tensor(fx["grad"])))
    parts.append(emit_float_scalar(f"{prefix}Lr", fx["lr"]))
    parts.append(emit_float_scalar(f"{prefix}Momentum", fx["momentum"]))
    parts.append(emit_float_scalar(f"{prefix}WeightDecay", fx["weightDecay"]))
    parts.append(emit_int32_array(f"{prefix}NewParamCodes", torch.tensor(fx["newParamCodes"])))
    parts.append(emit_int32_scalar(f"{prefix}NewParamExp", fx["newParamExp"]))
    parts.append(emit_int32_array(f"{prefix}NewStateCodes", torch.tensor(fx["newStateCodes"])))
    parts.append(emit_int32_scalar(f"{prefix}NewStateExp", fx["newStateExp"]))


def emit_asym_fixture(parts, prefix, fx):
    parts.append(emit_int32_array(f"{prefix}ParamCodes", torch.tensor(fx["paramCodes"])))
    parts.append(emit_float_array(f"{prefix}ParamScales", torch.tensor(fx["paramScales"])))
    parts.append(emit_int32_array(f"{prefix}ParamZps", torch.tensor(fx["paramZps"])))
    parts.append(emit_float_array(f"{prefix}Grad", torch.tensor(fx["grad"])))
    parts.append(emit_float_scalar(f"{prefix}Lr", fx["lr"]))
    parts.append(emit_float_scalar(f"{prefix}WeightDecay", fx["weightDecay"]))
    parts.append(emit_int32_array(f"{prefix}NewCodes", torch.tensor(fx["newCodes"])))
    parts.append(emit_float_array(f"{prefix}NewScales", torch.tensor(fx["newScales"])))
    parts.append(emit_int32_array(f"{prefix}NewZps", torch.tensor(fx["newZps"])))


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
    parts.append("\n")
    emit_asym_fixture(parts, "sgdGroupedAsymStep0", fixture_asym_step0())
    parts.append("\n")
    bfp_v1 = fixture_bfp_momentum()
    bfp_v2 = fixture_bfp_momentum_v2()
    assert (bfp_v1["newParamExp"], bfp_v1["newStateExp"]) != (bfp_v2["newParamExp"],
                                                              bfp_v2["newStateExp"]), (
        "bfpMomV2 lands on the SAME (param, state) exponent pair as bfpMom -- the second "
        "fixture adds no discrimination over the first")
    emit_bfp_fixture(parts, "sgdBfpMom", bfp_v1)
    parts.append("\n")
    emit_bfp_fixture(parts, "sgdBfpMomV2", bfp_v2)

    parts.append("\n#endif // ODT_EXPECTED_SGD_GROUPED_H\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(parts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
