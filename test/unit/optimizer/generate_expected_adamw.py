"""Emit gold AdamW step values as a C header (#328).

GOLD SOURCE = documented-order float32 replication of the C kernel sequence
(K1 lerp / K2 mul+addcmul / K3 mul+addcdivDenom), with per-step scalars
composed in python double exactly like AdamW.c and cast to float32 once:

- numpy float32 ops give the separate roundings, bit-identical to the C
  helpers on every IEEE host BY CONSTRUCTION;
- the ONE fused rounding -- lerp's fmaf -- comes from torch.Tensor.lerp_
  (bit-stable across both CI hosts, pinned by the PR B lerp golds).

torch.optim.AdamW(foreach=False) is a SANITY CHECK (m bit-equal; v and
param within 1 float32 ulp), NOT the gold source: torch's addcdiv
vectorization fuses the final mul-add on Linux-AVX2 but not on macOS-ARM
(#353, docs/conventions/testing.md worked example 2), so torch's own bits
are platform-dependent in regimes an optimizer trajectory visits (zero
accumulators at t=1, boundary-adjacent sums later).
"""
from __future__ import annotations

import argparse
import math

import numpy as np
import torch


def format_float_literal(v: float) -> str:
    s = repr(float(v))
    if s in ("inf", "-inf", "nan"):
        raise ValueError(f"non-finite gold value: {v!r}")
    return s + "f"


def emit_float_array(name: str, values: np.ndarray) -> str:
    flat = np.asarray(values, dtype=np.float32).ravel()
    body = ", ".join(format_float_literal(v) for v in flat)
    return (
        f"static const float {name}[] = {{ {body} }};\n"
        f"static const size_t {name}_len = {len(flat)};\n"
    )


def f32(v: float) -> float:
    return float(np.float32(v))


def step_scalars(lr_f32: float, beta1: float, beta2: float, eps: float, wd: float, t: int):
    """Per-step scalars, python double composition == AdamW.c, then ONE
    float32 cast each (returned as np.float32 so numpy never upcasts)."""
    lr = float(np.float32(lr_f32))  # C computes (double)(float)learningRate
    w1 = 1.0 - beta1
    s2 = 1.0 - beta2
    bc1 = 1.0 - beta1 ** t
    bc2sqrt = math.sqrt(1.0 - beta2 ** t)
    step_scale = -(lr / bc1)
    decay = 1.0 - lr * wd
    return {
        "W1": np.float32(w1),
        "BETA2": np.float32(beta2),
        "S2": np.float32(s2),
        "DECAY": np.float32(decay),
        "BC2SQRT": np.float32(bc2sqrt),
        "EPS": np.float32(eps),
        "STEPSCALE": np.float32(step_scale),
    }


def adamw_step_replica(p, m, v, g, sc):
    """One documented-order AdamW step on float32 arrays. Every operation
    below is one C-side rounding; order is the contract."""
    # K1: m = lerp(m, g, w1) -- torch's fmaf CPU kernel (small-weight branch).
    mt = torch.from_numpy(m.copy())
    mt.lerp_(torch.from_numpy(g.copy()), float(sc["W1"]))
    m = mt.numpy().astype(np.float32)
    # K2: v = BETA2*v; v += (S2*g)*g  (mul_ then addcmul_, left-assoc).
    v = (v * sc["BETA2"]).astype(np.float32)
    v = (v + ((sc["S2"] * g).astype(np.float32) * g).astype(np.float32)).astype(np.float32)
    # K3: p = DECAY*p; d=sqrt(v); d/=BC2SQRT; d+=EPS; n=STEPSCALE*m; n/=d; p+=n.
    p = (p * sc["DECAY"]).astype(np.float32)
    d = np.sqrt(v, dtype=np.float32)
    d = (d / sc["BC2SQRT"]).astype(np.float32)
    d = (d + sc["EPS"]).astype(np.float32)
    n = (sc["STEPSCALE"] * m).astype(np.float32)
    n = (n / d).astype(np.float32)
    p = (p + n).astype(np.float32)
    return p, m, v


def torch_adamw_reference(p0, g_seq, lr_f32, beta1, beta2, eps, wd, steps):
    """torch.optim.AdamW single-tensor path, for the sanity check."""
    param = torch.nn.Parameter(torch.from_numpy(p0.copy()))
    opt = torch.optim.AdamW([param], lr=f32(lr_f32), betas=(beta1, beta2), eps=eps,
                            weight_decay=wd, foreach=False)
    for t in range(steps):
        param.grad = torch.from_numpy(g_seq[t].copy())
        opt.step()
    state = opt.state[param]
    return (param.detach().numpy().astype(np.float32),
            state["exp_avg"].numpy().astype(np.float32),
            state["exp_avg_sq"].numpy().astype(np.float32))


def sanity_check(p_r, m_r, v_r, p_t, m_t, v_t, name):
    assert np.array_equal(m_r.view(np.uint32), m_t.view(np.uint32)), \
        f"{name}: replica m != torch m (lerp kernel drifted -- re-derive, see testing.md)"
    for label, a, b in (("v", v_r, v_t), ("param", p_r, p_t)):
        ulp = np.spacing(np.abs(a).astype(np.float32))
        assert np.all(np.abs(a - b) <= ulp), \
            f"{name}: replica {label} differs from torch by more than 1 ulp"


def run_fixture(name, p0, g_seq, lr, beta1, beta2, eps, wd, steps, parts,
                emit_per_step_param=False):
    p, m, v = p0.copy(), np.zeros_like(p0), np.zeros_like(p0)
    per_step_p = []
    for t in range(1, steps + 1):
        sc = step_scalars(lr, beta1, beta2, eps, wd, t)
        p, m, v = adamw_step_replica(p, m, v, g_seq[t - 1], sc)
        per_step_p.append(p.copy())
    p_t, m_t, v_t = torch_adamw_reference(p0, g_seq, lr, beta1, beta2, eps, wd, steps)
    sanity_check(p, m, v, p_t, m_t, v_t, name)
    parts.append(emit_float_array(f"adamw_{name}_p0", p0))
    parts.append(emit_float_array(f"adamw_{name}_g", np.concatenate(g_seq)))
    if emit_per_step_param:
        parts.append(emit_float_array(f"adamw_{name}_p_steps", np.concatenate(per_step_p)))
    else:
        parts.append(emit_float_array(f"adamw_{name}_p1", p))
        parts.append(emit_float_array(f"adamw_{name}_m1", m))
        parts.append(emit_float_array(f"adamw_{name}_v1", v))
    if emit_per_step_param:
        parts.append(emit_float_array(f"adamw_{name}_m_final", m))
        parts.append(emit_float_array(f"adamw_{name}_v_final", v))


def warm_values(rng, n):
    """Magnitudes in [0.1, 1) with random sign: keeps every accumulator out
    of the absorption-free zero regime (constraint: warm fixtures)."""
    return (rng.uniform(0.1, 1.0, n).astype(np.float32)
            * rng.choice(np.array([-1.0, 1.0], dtype=np.float32), n))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    parts = [
        "// AUTOGENERATED by generate_expected_adamw.py - DO NOT EDIT\n",
        "#ifndef ODT_EXPECTED_ADAMW_H\n#define ODT_EXPECTED_ADAMW_H\n",
        "#include <stdlib.h>\n\n",
    ]

    rng = np.random.default_rng(1)

    # --- Task 2: step-1 fixtures (m0 = v0 = 0; small-accumulator regime =
    # AdamW-step-1-realistic AND mutation-discriminating). n = 32.
    p0 = warm_values(rng, 32)
    g0 = [warm_values(rng, 32)]
    run_fixture("step1_default", p0, g0, 0.001, 0.9, 0.999, 1e-8, 0.01, 1, parts)
    run_fixture("step1_wd0", p0, g0, 0.001, 0.9, 0.999, 1e-8, 0.0, 1, parts)
    # Exaggerated lr/wd: decay = 1 - 0.05, update term ~1e-1 -- nothing is
    # absorbed, so K3 helper-ORDER mutations (decay after addcdiv) and
    # coupled-L2 mutations go RED here even where the default fixture
    # absorbs them (constraint 4's absorption trap).
    run_fixture("step1_orderdiscrim", p0, g0, 0.1, 0.9, 0.999, 1e-8, 0.5, 1, parts)

    # --- Trajectories: 8 steps, warm params, fresh grads per step.
    # Multi-step is load-bearing: step-1 alone cannot see lerp's fmaf
    # (m0 = 0 makes fused and separate rounding identical), nor the
    # v-decay term (v0 = 0).
    for name, wd in (("traj_default", 0.01), ("traj_wd0", 0.0)):
        p0 = warm_values(rng, 32)
        g_seq = [warm_values(rng, 32) for _ in range(8)]
        run_fixture(name, p0, g_seq, 0.001, 0.9, 0.999, 1e-8, wd, 8, parts,
                    emit_per_step_param=True)

    # --- Scheduler composition: CosineAnnealingLR(T_max=5, eta_min=0.001),
    # base lr 0.01, 5 epochs, one AdamW step per epoch. Epoch e trains with
    # the float32-cast closed form after e scheduler steps (PR A semantics:
    # construction leaves lr = base; step() advances). The generator feeds
    # the float32-ROUNDED lr into the double scalar composition, exactly
    # like C's (double)(float)learningRate after setLr.
    base, t_max, eta_min, epochs = 0.01, 5, 0.001, 5
    lr_seq = [f32(f32(eta_min) + (f32(base) - f32(eta_min))
                  * (1 + math.cos(math.pi * e / t_max)) / 2) for e in range(epochs)]
    lr_seq[0] = f32(base)  # lastEpoch=0: lr == baseLr exactly (PR A contract)
    p0 = warm_values(rng, 32)
    g_seq = [warm_values(rng, 32) for _ in range(epochs)]
    p, m, v = p0.copy(), np.zeros_like(p0), np.zeros_like(p0)
    for t in range(1, epochs + 1):
        sc = step_scalars(lr_seq[t - 1], 0.9, 0.999, 1e-8, 0.01, t)
        p, m, v = adamw_step_replica(p, m, v, g_seq[t - 1], sc)
    parts.append(emit_float_array("adamw_sched_cosine_lr", np.array(lr_seq, dtype=np.float32)))
    parts.append(emit_float_array("adamw_sched_cosine_p0", p0))
    parts.append(emit_float_array("adamw_sched_cosine_g", np.concatenate(g_seq)))
    parts.append(emit_float_array("adamw_sched_cosine_p_final", p))

    parts.append("#endif // ODT_EXPECTED_ADAMW_H\n")
    with open(args.out, "w") as f:
        f.write("".join(parts))


if __name__ == "__main__":
    main()
