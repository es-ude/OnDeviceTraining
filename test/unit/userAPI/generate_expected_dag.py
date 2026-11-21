#!/usr/bin/env python3
"""Generate expected_dag.h for UnitTestDagTraining.

Residual block R (one sample, float64):

    h1 = Linear 4->4 (x); h3 = Linear 4->4 (relu(h1)); out = Linear 4->2 (h1 + h3)

under MSE, and R-CE: the same trunk (same x, W0, B0, W2, B2) into a Softmax
sink under cross-entropy with the one-hot label of class 2. The C grads are raw
per-element sums (the optimizer applies the mean scale), so the reference
backward is ((out - y)**2).sum() for MSE and -(y * log_softmax(h1 + h3)).sum()
for CE. The emitted loss is F.mse_loss(out, y) (mean) for MSE and the same CE
sum for CE (batch 1). Grads in C parameter order: W0, B0, W2, B2[, W4, B4].

Self-checks before writing: the raw MSE grads equal size(y) times the autograd
of F.mse_loss; the CE seed equals softmax - y; and cutting the ReLU branch's
contribution to h1's gradient (relu(h1.detach())) moves the grads by more than
1e-2 for both models, so a dropped fan-out contribution is visible. Seed 329
passes it.

Fan-out fixtures under MSE, label [2], each with its own weights, x and y,
drawn after R's from the same generator so R's golds do not move:

    SELF: h1 = Linear 4->4 (x); out = Linear 4->2 (h1 + h1)
    IN0:  out = Linear 4->2 (relu(Linear 4->4 (x)) + x), x needs no grad
    FAN3: h1 = Linear 4->4 (x); h2 = Linear 4->4 (h1); out = Linear 4->2 ((h1 + h2) + h1)

Each passes the same > 1e-2 vacuity check: SELF against h1 + h1.detach(), IN0
against the ReLU branch cut, FAN3 against the ACC contribution cut
(h1.detach() + h2) and, separately, the PART contribution cut
(Linear(h1.detach())).

Freezing fixtures under MSE, drawn after FAN3; F marks a frozen Linear
(requires_grad False). Grads are emitted for the trainable Linears only, in C
node order:

    FRZ: h1 = Linear F (x); out = Linear 4->2 (h1 + Linear (relu(h1)))
    GAP: h1 = Linear F (x); out = Linear 4->2 (Linear (h1) + Linear F (h1))

Nothing trainable sits upstream of h1, so a detach cut would change nothing;
instead the check is that every frozen leaf gets no grad and every trainable
weight grad exceeds 1e-2 somewhere (FRZ's ReLU could be all dead).

Literals: repr(v)+"f". Run via `uv run` (CMake wires this automatically).
"""
import argparse

import torch
import torch.nn.functional as F

F64 = torch.float64
SEED = 329
CE_CLASS = 2


def lit(v):
    s = repr(float(v))
    if s in ("inf", "-inf", "nan"):
        raise ValueError(v)
    return s + "f"


def emit(name, t):
    flat = t.detach().to(torch.float32).flatten().tolist()
    return f"static const float {name}[{len(flat)}] = {{{', '.join(lit(v) for v in flat)}}};\n"


def emit_scalar(name, v):
    return f"static const float {name} = {lit(v)};\n"


def leaves(ts):
    return [t.clone().requires_grad_(True) for t in ts]


def trunk(x, w0, b0, w2, b2, cut):
    h1 = F.linear(x, w0, b0)
    r = torch.relu(h1.detach() if cut else h1)
    h3 = F.linear(r, w2, b2)
    return h1 + h3


def res_grads(x, y, params, cut):
    p = leaves(params)
    out = F.linear(trunk(x, *p[:4], cut), p[4], p[5])
    ((out - y) ** 2).sum().backward()
    return out.detach(), torch.cat([t.grad.flatten() for t in p])


def res_ce_grads(x, y, params, cut):
    p = leaves(params)
    logits = trunk(x, *p, cut)
    loss = -(y * F.log_softmax(logits, dim=-1)).sum()
    loss.backward()
    return torch.softmax(logits.detach(), dim=-1), loss.detach(), torch.cat(
        [t.grad.flatten() for t in p])


def self_fwd(x, p, cut):
    h1 = F.linear(x, p[0], p[1])
    return F.linear(h1 + (h1.detach() if cut else h1), p[2], p[3])


def in0_fwd(x, p, cut):
    r = torch.relu(F.linear(x, p[0], p[1]))
    return F.linear((r.detach() if cut else r) + x, p[2], p[3])


def fan3_fwd(x, p, cut):
    h1 = F.linear(x, p[0], p[1])
    h2 = F.linear(h1.detach() if cut == "part" else h1, p[2], p[3])
    h3 = (h1.detach() if cut == "acc" else h1) + h2
    return F.linear(h3 + h1, p[4], p[5])


def mse_grads(fwd, x, y, params, cut):
    p = leaves(params)
    out = fwd(x, p, cut)
    ((out - y) ** 2).sum().backward()
    # A leaf a cut disconnects gets no .grad: its gradient is 0.
    grads = [torch.zeros_like(t) if t.grad is None else t.grad for t in p]
    return out.detach(), torch.cat([t.flatten() for t in grads])


def fan_out_fixture(name, fwd, x, y, params, cuts):
    out, grads = mse_grads(fwd, x, y, params, None)
    for cut in cuts:
        _, grads_cut = mse_grads(fwd, x, y, params, cut)
        assert (grads - grads_cut).abs().max() > 1e-2, f"{name}: cutting {cut} changes nothing"
    return grads, F.mse_loss(out, y), out


def frz_fwd(x, p):
    h1 = F.linear(x, p[0], p[1])
    return F.linear(h1 + F.linear(torch.relu(h1), p[2], p[3]), p[4], p[5])


def gap_fwd(x, p):
    h1 = F.linear(x, p[0], p[1])
    return F.linear(F.linear(h1, p[2], p[3]) + F.linear(h1, p[4], p[5]), p[6], p[7])


def frozen_fixture(name, fwd, x, y, params, frozen):
    p = [t.clone().requires_grad_(not f) for t, f in zip(params, frozen)]
    out = fwd(x, p)
    ((out - y) ** 2).sum().backward()
    for i, (t, f) in enumerate(zip(p, frozen)):
        if f:
            assert t.grad is None, f"{name}: frozen leaf {i} got a grad"
        elif t.dim() == 2:
            assert t.grad.abs().max() > 1e-2, f"{name}: trainable weight {i} has no signal"
    grads = torch.cat([t.grad.flatten() for t, f in zip(p, frozen) if not f])
    return grads, F.mse_loss(out.detach(), y), out.detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    g = torch.Generator().manual_seed(SEED)

    def linear_params(fan_in, fan_out):
        return (torch.randn(fan_out, fan_in, generator=g, dtype=F64) * 0.5,
                torch.randn(fan_out, generator=g, dtype=F64) * 0.1)

    w0, b0 = linear_params(4, 4)
    w2, b2 = linear_params(4, 4)
    w4, b4 = linear_params(4, 2)
    x = torch.randn(4, generator=g, dtype=F64)
    y = torch.randn(2, generator=g, dtype=F64)
    y_ce = F.one_hot(torch.tensor(CE_CLASS), 4).to(F64)
    res_params = (w0, b0, w2, b2, w4, b4)
    ce_params = (w0, b0, w2, b2)

    out, grads = res_grads(x, y, res_params, cut=False)
    _, grads_cut = res_grads(x, y, res_params, cut=True)
    assert (grads - grads_cut).abs().max() > 1e-2, "R: the ReLU branch adds nothing to h1's grad"
    p = leaves(res_params)
    F.mse_loss(F.linear(trunk(x, *p[:4], False), p[4], p[5]), y).backward()
    mean_grads = torch.cat([t.grad.flatten() for t in p])
    assert torch.allclose(grads, mean_grads * y.numel(), atol=1e-12), "raw != size * mean MSE"
    loss = F.mse_loss(out, y)

    probs, ce_loss, ce_grads = res_ce_grads(x, y_ce, ce_params, cut=False)
    _, _, ce_grads_cut = res_ce_grads(x, y_ce, ce_params, cut=True)
    assert (ce_grads - ce_grads_cut).abs().max() > 1e-2, "R-CE: the ReLU branch adds nothing"
    logits = trunk(x, *ce_params, False).requires_grad_(True)
    (-(y_ce * F.log_softmax(logits, dim=-1)).sum()).backward()
    assert torch.allclose(logits.grad, probs - y_ce, atol=1e-12), "CE seed != softmax - y"

    fan_out = []
    for name, fwd, widths, cuts in (("dagSelf", self_fwd, ((4, 4, "0"), (4, 2, "2")), [True]),
                                    ("dagIn0", in0_fwd, ((4, 4, "0"), (4, 2, "3")), [True]),
                                    ("dagFan3", fan3_fwd,
                                     ((4, 4, "0"), (4, 4, "1"), (4, 2, "4")), ["acc", "part"])):
        params = []
        names = []
        for fan_in, width_out, idx in widths:
            params += linear_params(fan_in, width_out)
            names += [f"{name}W{idx}", f"{name}B{idx}"]
        fx = torch.randn(4, generator=g, dtype=F64)
        fy = torch.randn(2, generator=g, dtype=F64)
        f_grads, f_loss, f_out = fan_out_fixture(name, fwd, fx, fy, params, cuts)
        fan_out += [emit(f"{name}X", fx), emit(f"{name}Y", fy)]
        fan_out += [emit(n, t) for n, t in zip(names, params)]
        fan_out += [emit(f"{name}Grads", f_grads), emit_scalar(f"{name}Loss", f_loss),
                    emit(f"{name}Out", f_out)]

    frozen_out = []
    for name, fwd, widths in (("dagFrz", frz_fwd, ((4, 4, "0", True), (4, 4, "2", False),
                                                    (4, 2, "4", False))),
                              ("dagGap", gap_fwd, ((4, 4, "0", True), (4, 4, "1", False),
                                                   (4, 4, "2", True), (4, 2, "4", False)))):
        params = []
        names = []
        frozen = []
        for fan_in, width_out, idx, f in widths:
            params += linear_params(fan_in, width_out)
            names += [f"{name}W{idx}", f"{name}B{idx}"]
            frozen += [f, f]
        fx = torch.randn(4, generator=g, dtype=F64)
        fy = torch.randn(2, generator=g, dtype=F64)
        f_grads, f_loss, f_out = frozen_fixture(name, fwd, fx, fy, params, frozen)
        frozen_out += [emit(f"{name}X", fx), emit(f"{name}Y", fy)]
        frozen_out += [emit(n, t) for n, t in zip(names, params)]
        frozen_out += [emit(f"{name}Grads", f_grads), emit_scalar(f"{name}Loss", f_loss),
                       emit(f"{name}Out", f_out)]

    parts = ["// AUTOGENERATED by generate_expected_dag.py -- DO NOT EDIT\n",
             "#ifndef ODT_EXPECTED_DAG_H\n#define ODT_EXPECTED_DAG_H\n",
             emit("dagResX", x), emit("dagResY", y),
             emit("dagResW0", w0), emit("dagResB0", b0), emit("dagResW2", w2),
             emit("dagResB2", b2), emit("dagResW4", w4), emit("dagResB4", b4),
             emit("dagResGrads", grads), emit_scalar("dagResLoss", loss), emit("dagResOut", out),
             emit("dagResCeY", y_ce), emit("dagResCeGrads", ce_grads),
             emit_scalar("dagResCeLoss", ce_loss), emit("dagResCeOut", probs),
             *fan_out,
             *frozen_out,
             "#endif\n"]
    with open(args.out, "w") as f:
        f.write("".join(parts))


if __name__ == "__main__":
    main()
