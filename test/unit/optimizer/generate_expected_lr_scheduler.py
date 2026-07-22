"""Emit gold LR sequences from torch.optim.lr_scheduler as a C header.

For each fixture: build a dummy SGD optimizer at base LR, construct the
scheduler, record param_groups[0]["lr"] after construction (index 0) and
after each of NUM_STEPS scheduler.step() calls (indices 1..NUM_STEPS),
rounded to float32 — the C side computes in double and casts to float at
setLr, so float32 is the comparison precision.

Self-check: every recorded value must equal the float32-rounded closed form
(StepLR: base*gamma**(e//s); Exp: base*gamma**e; Cosine:
eta_min + (base-eta_min)*(1+cos(pi*e/T_max))/2). This pins the
"recursive get_lr() collapses to the closed form under strict once-per-epoch
stepping" property the C implementation relies on — a fixture that violates
it must fail generation loudly, not ship a lying gold.
"""
from __future__ import annotations

import argparse
import math

import numpy as np
import torch

NUM_STEPS = 20


def _format_float_literal(v: float) -> str:
    s = repr(v)
    if s in ("inf", "-inf", "nan"):
        raise ValueError(f"non-finite gold value: {v!r}")
    return s + "f"


def emit_float_array(name: str, values: list[float]) -> str:
    body = ", ".join(_format_float_literal(v) for v in values)
    return (
        f"static const float {name}[] = {{ {body} }};\n"
        f"static const size_t {name}_len = {len(values)};\n"
    )


def f32(v: float) -> float:
    """Round a hyperparameter through float32.

    The C side stores baseLr/gamma/etaMin as float and computes the closed
    form from their double-widened values. Feeding torch the raw python
    double (e.g. 0.1 instead of double(0.1f), ~2^-29 apart) would make
    ~2-3% of emitted values land on the other side of a float32 rounding
    boundary. Rounding HERE makes generator and C bit-identical by
    construction."""
    return float(np.float32(v))


def record_sequence(
    make_scheduler,
    base_lr: float,
    closed_form,
    boundary_tolerant: bool = False,
) -> list[float]:
    """Run the scheduler and self-check each step against the closed form.

    boundary_tolerant: for a fixture deliberately placed on a float32
    rounding boundary (see the discriminator fixture below), the gold
    values come from the closed form itself rather than torch's lr, and
    the self-check against torch is a <=1-float32-ulp tolerance instead of
    bit equality. Rationale: the generator's python `math.cos` and the C
    test's `cos` are the SAME libm on the same build host, and the double
    expression is transcribed identically -- so closed-form golds are
    bit-identical to the C computation BY CONSTRUCTION on every platform,
    which is exactly what the discriminator exists to pin (the C
    association order). Torch's recursive get_lr() can drift from the
    closed form by a sub-ulp double difference that lands on either side
    of a float32 rounding boundary depending on libm (observed: Apple
    libm and glibc `cos` disagree at exactly this boundary) -- that drift
    is a torch/libm implementation detail unrelated to what this fixture
    tests, so it must not gate generation.
    """
    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([param], lr=f32(base_lr))
    scheduler = make_scheduler(optimizer)
    seq: list[float] = []
    for epoch in range(NUM_STEPS + 1):
        lr32 = float(np.float32(optimizer.param_groups[0]["lr"]))
        expected32 = float(np.float32(closed_form(epoch)))
        if boundary_tolerant:
            ulp = float(np.spacing(np.float32(expected32)))
            diff = abs(lr32 - expected32)
            assert diff <= ulp, (
                f"epoch {epoch}: torch lr {lr32!r} vs closed form {expected32!r} "
                f"differ by {diff!r}, more than one float32 ulp ({ulp!r}) -- "
                "this exceeds the expected torch-recursion/libm drift"
            )
            seq.append(expected32)
        else:
            assert lr32 == expected32, (
                f"epoch {epoch}: torch lr {lr32!r} != closed form {expected32!r}"
            )
            seq.append(lr32)
        scheduler.step()
    return seq


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    fixtures: list[tuple[str, list[float]]] = []

    for name, base, step_size, gamma in [
        ("step_b01_s5_g05", 0.1, 5, 0.5),
        ("step_b001_s1_g09", 0.01, 1, 0.9),
        ("step_b005_s7_g01", 0.05, 7, 0.1),
    ]:
        fixtures.append((name, record_sequence(
            lambda opt, s=step_size, g=f32(gamma):
                torch.optim.lr_scheduler.StepLR(opt, step_size=s, gamma=g),
            base,
            lambda e, b=f32(base), s=step_size, g=f32(gamma): b * g ** (e // s),
        )))

    for name, base, gamma in [
        ("exp_b01_g095", 0.1, 0.95),
        ("exp_b001_g05", 0.01, 0.5),
    ]:
        fixtures.append((name, record_sequence(
            lambda opt, g=f32(gamma):
                torch.optim.lr_scheduler.ExponentialLR(opt, gamma=g),
            base,
            lambda e, b=f32(base), g=f32(gamma): b * g ** e,
        )))

    for name, base, t_max, eta_min, boundary_tolerant in [
        ("cos_b01_t20_e0", 0.1, 20, 0.0, False),
        ("cos_b001_t10_e0001", 0.01, 10, 0.001, False),  # runs PAST T_max: pins periodicity
        # Discriminates the two float64 associations of pi*e/T_max:
        # (pi*e)/T_max (torch's, correct) vs pi*(e/T_max) (the bug fixed in
        # #327 fix-wave-1) diverge at epoch 13 for this combo
        # (0.029999999329447746f vs 0.030000001192092896f) - found by a grid
        # sweep, see #327. Without this fixture the
        # other two cosine fixtures are bit-identical under both
        # associations and don't enforce the constraint.
        # This combo also sits ON a float32 rounding boundary at epoch 13,
        # which makes it sensitive to sub-ulp double drift between torch's
        # recursive get_lr() and the closed form across libms (Apple libm
        # agrees with the closed form here; glibc's cos lands one ulp to
        # the other side) - hence boundary_tolerant=True: gold comes from
        # the closed form (bit-identical to the C computation by
        # construction), torch is only sanity-checked within 1 ulp. See
        # record_sequence's docstring.
        ("cos_discrim_b005_t26_e001", 0.05, 26, 0.01, True),
    ]:
        fixtures.append((name, record_sequence(
            lambda opt, t=t_max, e=f32(eta_min):
                torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t, eta_min=e),
            base,
            lambda ep, b=f32(base), t=t_max, em=f32(eta_min):
                em + (b - em) * (1 + math.cos(math.pi * ep / t)) / 2,
            boundary_tolerant=boundary_tolerant,
        )))

    # #383: LINEAR_WARMUP_LR. torch reference is SequentialLR(opt,
    # schedulers=[LinearLR(start_factor=S, end_factor=1.0, total_iters=W), main],
    # milestones=[W]). Surprise worth pinning: main's OWN base_lrs equals the
    # true pre-warmup base LR regardless of whether LinearLR or main is
    # constructed first in torch -- the very first scheduler ever attached to
    # an optimizer locks group["initial_lr"] to the then-pristine lr, and every
    # later scheduler on the same optimizer inherits that value via
    # `setdefault` even though LinearLR's own construction already mutated the
    # visible `lr`. The C API has no such per-optimizer "initial_lr" memory, so
    # linearWarmupLrInit's doc comment requires callers to init `mainSched`
    # BEFORE calling linearWarmupLrInit -- that ordering is what makes
    # mainSched capture the pristine baseLr via getLr(), mirroring torch.
    for name, base, sf, warmup, t_max, eta_min in [
        ("warmup_cos_b01_w5_sf03_t10_e0", 0.1, 0.3, 5, 10, 0.0),
        ("warmup_cos_b005_w3_sf01_t15_e0005", 0.05, 0.1, 3, 15, 0.005),
    ]:

        def make_warmup_cos_scheduler(opt, w=warmup, s=f32(sf), t=t_max, e=f32(eta_min)):
            lin = torch.optim.lr_scheduler.LinearLR(opt, start_factor=s, end_factor=1.0,
                                                     total_iters=w)
            main = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t, eta_min=e)
            return torch.optim.lr_scheduler.SequentialLR(opt, schedulers=[lin, main],
                                                          milestones=[w])

        def warmup_cos_closed_form(epoch, b=f32(base), s=f32(sf), w=warmup, t=t_max,
                                    em=f32(eta_min)):
            if epoch < w:
                return b * (s + (1.0 - s) * epoch / w)
            local = epoch - w
            return em + (b - em) * (1 + math.cos(math.pi * local / t)) / 2

        fixtures.append((name, record_sequence(make_warmup_cos_scheduler, base,
                                               warmup_cos_closed_form)))

    for name, base, sf, warmup in [
        ("warmup_null_b005_w4_sf02", 0.05, 0.2, 4),
        ("warmup_null_b02_w6_sf05", 0.2, 0.5, 6),
    ]:
        fixtures.append((name, record_sequence(
            lambda opt, w=warmup, s=f32(sf):
                torch.optim.lr_scheduler.LinearLR(opt, start_factor=s, end_factor=1.0,
                                                  total_iters=w),
            base,
            lambda e, b=f32(base), s=f32(sf), w=warmup: b * (s + (1.0 - s) * min(e, w) / w),
        )))

    parts = [
        "// AUTOGENERATED by generate_expected_lr_scheduler.py - DO NOT EDIT\n",
        "#ifndef ODT_EXPECTED_LR_SCHEDULER_H\n#define ODT_EXPECTED_LR_SCHEDULER_H\n",
        "#include <stdlib.h>\n\n",
    ]
    for name, seq in fixtures:
        parts.append(emit_float_array(f"lr_gold_{name}", seq))
        parts.append("\n")
    parts.append("#endif // ODT_EXPECTED_LR_SCHEDULER_H\n")
    with open(args.out, "w") as f:
        f.write("".join(parts))


if __name__ == "__main__":
    main()
