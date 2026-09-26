#!/usr/bin/env python3
"""Generate expected_batchnorm1d.h for UnitTestBatchNorm1d (FLOAT32, #460).

PyTorch nn.BatchNorm1d ground truth (torch 2.11, float64 then cast):
training-mode forward y, backward dx/dgamma/dbeta, and the running buffers
+ num_batches_tracked after the forward, for rank-2 [m, C] and rank-3
[m, C, T] fixtures and each option variant (affine off, track off,
momentum 0.3, momentum=None over three forwards, n = 2), plus one
eval-mode fixture with non-trivial running stats (forward AND backward).
gamma/beta/running stats are randn (not 1/0/0/1) so the affine and the
running-stat arithmetic are non-vacuous; lossGrad is randn (non-uniform).

Self-checks before emitting (else the build fails): y against a hand
biased-variance formula (mean/var over dims (0, 2..)), and the running
buffers against the hand EMA with the UNBIASED variance n/(n-1)
(cumulative: the plain mean of the per-step batch means / unbiased vars).

Literals: repr(v)+"f". Run via `uv run` (CMake wires this automatically).
"""
import argparse
import sys
from pathlib import Path

import torch

F64 = torch.float64


def lit(v: float) -> str:
    s = repr(float(v))
    if s in ("inf", "-inf", "nan"):
        raise ValueError(f"non-finite gold value: {v!r}")
    return s + "f"


def emit(name: str, t: torch.Tensor) -> str:
    flat = t.detach().to(torch.float32).flatten().tolist()
    body = ", ".join(lit(v) for v in flat)
    return (f"static const float {name}[] = {{ {body} }};\n"
            f"static const size_t {name}_len = {len(flat)};\n")


def reduce_dims(x):
    return [0] + list(range(2, x.dim()))


def hand_forward(x, mean, var, gamma, beta, eps):
    shape = [1, x.shape[1]] + [1] * (x.dim() - 2)
    y = (x - mean.reshape(shape)) / torch.sqrt(var.reshape(shape) + eps)
    if gamma is not None:
        y = y * gamma.reshape(shape) + beta.reshape(shape)
    return y


def batch_stats(x):
    dims = reduce_dims(x)
    mean = x.mean(dim=dims)
    var = ((x - mean.reshape([1, -1] + [1] * (x.dim() - 2))) ** 2).mean(dim=dims)
    n = x.numel() // x.shape[1]
    return mean, var, n


def train_fixture(name, shape, *, seed, affine=True, track=True,
                  momentum=0.1, steps=1, eps=1e-5):
    torch.manual_seed(seed)
    C = shape[1]
    bn = torch.nn.BatchNorm1d(C, eps=eps, momentum=momentum, affine=affine,
                              track_running_stats=track).to(F64)
    gamma = beta = None
    if affine:
        with torch.no_grad():
            bn.weight.copy_(torch.randn(C, dtype=F64))
            bn.bias.copy_(torch.randn(C, dtype=F64))
        gamma, beta = bn.weight.detach().clone(), bn.bias.detach().clone()
    bn.train()
    xs = [torch.randn(*shape, dtype=F64) for _ in range(steps)]
    rm_ref = torch.zeros(C, dtype=F64)
    rv_ref = torch.ones(C, dtype=F64)
    means, uvars = [], []
    for k, xk in enumerate(xs):
        mean, var, n = batch_stats(xk)
        uvar = var * n / (n - 1)
        means.append(mean)
        uvars.append(uvar)
        f = (1.0 / (k + 1)) if momentum is None else momentum
        rm_ref = (1 - f) * rm_ref + f * mean
        rv_ref = (1 - f) * rv_ref + f * uvar
        if k < steps - 1:
            with torch.no_grad():
                bn(xk)
    x = xs[-1].clone().requires_grad_(True)
    y = bn(x)
    mean, var, _ = batch_stats(x.detach())
    assert torch.allclose(y.detach(), hand_forward(x.detach(), mean, var, gamma, beta, eps),
                          atol=1e-9), f"{name}: forward disagrees with hand formula"
    if track:
        assert torch.allclose(bn.running_mean, rm_ref, atol=1e-12), f"{name}: running_mean"
        assert torch.allclose(bn.running_var, rv_ref, atol=1e-12), f"{name}: running_var"
        if momentum is None:
            assert torch.allclose(bn.running_mean, torch.stack(means).mean(0), atol=1e-12)
            assert torch.allclose(bn.running_var, torch.stack(uvars).mean(0), atol=1e-12)
    gy = torch.randn_like(y)
    y.backward(gy)
    fx = {"input": torch.cat([xk.flatten() for xk in xs]), "expectedForward": y,
          "lossGrad": gy, "expectedPropLoss": x.grad}
    if affine:
        fx.update(gamma=gamma, beta=beta, expectedDgamma=bn.weight.grad,
                  expectedDbeta=bn.bias.grad)
    if track:
        fx.update(expectedRunningMean=bn.running_mean, expectedRunningVar=bn.running_var)
        fx["numBatchesTracked"] = int(bn.num_batches_tracked)
    return name, fx


def eval_fixture(name, shape, *, seed, eps=1e-5):
    torch.manual_seed(seed)
    C = shape[1]
    bn = torch.nn.BatchNorm1d(C, eps=eps).to(F64)
    with torch.no_grad():
        bn.weight.copy_(torch.randn(C, dtype=F64))
        bn.bias.copy_(torch.randn(C, dtype=F64))
        bn.running_mean.copy_(torch.randn(C, dtype=F64))
        bn.running_var.copy_(torch.rand(C, dtype=F64) + 0.5)
    rm0, rv0 = bn.running_mean.clone(), bn.running_var.clone()
    bn.eval()
    x = torch.randn(*shape, dtype=F64, requires_grad=True)
    y = bn(x)
    assert torch.allclose(y.detach(), hand_forward(x.detach(), rm0, rv0, bn.weight.detach(),
                                                   bn.bias.detach(), eps), atol=1e-9)
    gy = torch.randn_like(y)
    y.backward(gy)
    return name, {"input": x.detach(), "gamma": bn.weight.detach(), "beta": bn.bias.detach(),
                  "runningMeanInit": rm0, "runningVarInit": rv0, "expectedForward": y,
                  "lossGrad": gy, "expectedPropLoss": x.grad,
                  "expectedDgamma": bn.weight.grad, "expectedDbeta": bn.bias.grad}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    fixtures = [
        train_fixture("trainRank2", (4, 3), seed=461),
        train_fixture("trainRank3", (3, 2, 5), seed=462),
        train_fixture("minimalRank2", (2, 3), seed=463),          # n = 2: n/(n-1) = 2
        train_fixture("noAffineRank3", (3, 2, 5), seed=464, affine=False),
        train_fixture("noTrackRank2", (4, 3), seed=465, track=False),
        train_fixture("momentum03Rank2", (4, 3), seed=466, momentum=0.3),
        train_fixture("momentumOneRank2", (4, 3), seed=467, momentum=1.0),
        train_fixture("cumulativeRank3", (3, 2, 5), seed=468, momentum=None, steps=3),
        eval_fixture("evalRank3", (3, 2, 5), seed=469),
    ]
    parts = ["// AUTOGENERATED by generate_expected_batchnorm1d.py — DO NOT EDIT\n",
             "#ifndef ODT_EXPECTED_BATCHNORM1D_H\n#define ODT_EXPECTED_BATCHNORM1D_H\n",
             "#include <stddef.h>\n#include <stdint.h>\n\n"]
    for name, fx in fixtures:
        for key, val in fx.items():
            if key == "numBatchesTracked":
                parts.append(f"static const uint64_t {key}_bn_{name} = {val}u;\n")
            else:
                parts.append(emit(f"{key}_bn_{name}", val))
    parts.append("\n#endif // ODT_EXPECTED_BATCHNORM1D_H\n")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(parts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
