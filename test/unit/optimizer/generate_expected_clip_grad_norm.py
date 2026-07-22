"""Emit gold global-norm gradient-clipping values as a C header (#382).

GOLD SOURCE = flat-concatenation replication of `torch.nn.utils.clip_grad_norm_`:
every grad tensor's elements are concatenated into ONE logical vector, summed
as squares in a double accumulator, one sqrt at the end (float32 cast once) --
this is the exact order `optimizerClipGradNorm` computes in (joint norm, not
per-tensor). It is mathematically identical to torch's own per-tensor
`vector_norm` -> `vector_norm`-of-the-stack (Minkowski/L2-of-L2 identity:
||[||g1||,||g2||]||_2 == sqrt(||g1||^2 + ||g2||^2) == sqrt(sum over every
element across every tensor)), so the two AGREE, but via a different
float32/double rounding PATH -- hence torch is a sanity check here, not the
gold source (same posture as generate_expected_adamw.py).

clip_coef = max_norm / (total_norm + 1e-6), clamped to <= 1.0 (torch's
`_clip_grads_with_norm_`); the exactly-at-norm fixture below pins that this
epsilon is NOT dropped (max_norm == total_norm still clips by a hair, it does
not no-op) and that the clamp direction is right (max_norm far above total_norm
must not amplify).

The exactly-at-norm fixture uses grads ~1000x SMALLER than the no-clip/clip
pair on purpose: the epsilon's relative effect on clip_coef is
`1e-6 / total_norm`, so shrinking total_norm by 1000x turns a
sub-Unity-tolerance ~5e-7 relative discrepancy (invisible to a normal
`TEST_ASSERT_EQUAL_FLOAT_ARRAY`, and easily swallowed by ordinary
FMA-contraction/toolchain float noise) into a ~5e-4 relative one -- 40x+ above
both the ~1e-5 Unity default tolerance AND any plausible cross-toolchain
rounding-path noise, so the C-side test can use the SAME tolerant comparison
idiom as every other gold fixture in this file instead of a toolchain-fragile
bit-exact assertion.
"""
from __future__ import annotations

import argparse

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


def emit_float_scalar(name: str, value: float) -> str:
    return f"static const float {name} = {format_float_literal(value)};\n"


def manual_total_norm(*grads: np.ndarray) -> np.float32:
    """GOLD SOURCE: double-accumulator flat sum of squares, one sqrt, cast
    to float32 once -- bit-for-bit the order optimizerClipGradNorm computes
    in (joint over ALL tensors, not per-tensor norm-then-combine)."""
    sum_sq = 0.0
    for g in grads:
        for v in np.asarray(g, dtype=np.float32).ravel():
            d = float(v)
            sum_sq += d * d
    return np.float32(np.sqrt(sum_sq))


def manual_clip(grads: list[np.ndarray], max_norm: float,
                total_norm: np.float32) -> tuple[list[np.ndarray], np.float32]:
    """GOLD SOURCE for post-clip values: replicates torch's
    `_clip_grads_with_norm_` float32 arithmetic exactly (max_norm and the
    1e-6 epsilon both float32, clamp to <= 1.0)."""
    max_norm32 = np.float32(max_norm)
    denom = np.float32(np.float32(total_norm) + np.float32(1e-6))
    clip_coef = np.float32(max_norm32 / denom)
    clip_coef_clamped = np.float32(min(clip_coef, np.float32(1.0)))
    clipped = [(g.astype(np.float32) * clip_coef_clamped).astype(np.float32) for g in grads]
    return clipped, clip_coef_clamped


def torch_sanity_check(grads: list[np.ndarray], max_norm: float,
                       expected_total_norm: np.float32,
                       expected_clipped: list[np.ndarray], name: str) -> None:
    """Sanity check ONLY (not the gold source, see module docstring):
    torch.nn.utils.clip_grad_norm_ on cloned tensors must agree with the
    manual replication within a tight float32 tolerance."""
    tensors = [torch.tensor(g.astype(np.float32).copy()) for g in grads]
    for t in tensors:
        t.grad = t.clone()
    total_norm_t = torch.nn.utils.clip_grad_norm_(tensors, max_norm)

    ulp_norm = float(np.spacing(np.float32(expected_total_norm)))
    assert abs(float(total_norm_t.item()) - float(expected_total_norm)) <= max(ulp_norm, 1e-6), (
        f"{name}: torch total_norm {total_norm_t.item()!r} vs manual "
        f"{expected_total_norm!r} differ by more than 1 float32 ulp"
    )
    for i, (t, expected) in enumerate(zip(tensors, expected_clipped)):
        got = t.grad.numpy()
        assert np.allclose(got, expected, atol=1e-6, rtol=1e-5), (
            f"{name}: torch post-clip grad[{i}] {got!r} vs manual {expected!r} "
            "differ by more than tolerance"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    # Two grad tensors, deliberately different (non-multiple-of-32, no fused
    # kernels touch this path) sizes with non-trivial (mixed-sign, varied
    # magnitude) values.
    grad_a = np.array([0.6, -0.3, 0.9, -1.2, 0.15], dtype=np.float32)
    grad_b = np.array([0.2, -0.5, 0.35, 0.8, -0.1, 0.65, -0.25], dtype=np.float32)

    total_norm = manual_total_norm(grad_a, grad_b)

    # Case (i): no-clip -- max_norm clearly above total_norm (clip_coef > 1,
    # clamped to 1 -- optimizerClipGradNorm must skip the scale call entirely,
    # leaving grads byte-identical; no gold arrays needed for this branch,
    # only the norm and the max_norm value itself).
    noclip_max_norm = 3.0

    # Case (ii): clip -- max_norm clearly below total_norm (clip_coef << 1).
    clip_max_norm = 1.0
    clip_a, clip_b = manual_clip([grad_a, grad_b], clip_max_norm, total_norm)[0]
    torch_sanity_check([grad_a, grad_b], clip_max_norm, total_norm, [clip_a, clip_b], "clip")

    # Case (iii): exactly-at-norm -- pins the +1e-6/clamp semantics: even
    # though max_norm == total_norm exactly, torch still clips by a hair
    # (clip_coef = total_norm/(total_norm+1e-6) < 1), it does not no-op.
    # Deliberately ~1000x smaller magnitude than grad_a/grad_b (see module
    # docstring): makes the epsilon's relative effect on the post-clip values
    # large enough to survive ordinary float tolerance, not just bit-exact
    # comparison.
    eps_grad_a = (grad_a * 1e-3).astype(np.float32)
    eps_grad_b = (grad_b * 1e-3).astype(np.float32)
    eps_total_norm = manual_total_norm(eps_grad_a, eps_grad_b)
    exact_max_norm = float(eps_total_norm)
    exact_a, exact_b = manual_clip([eps_grad_a, eps_grad_b], exact_max_norm, eps_total_norm)[0]
    torch_sanity_check([eps_grad_a, eps_grad_b], exact_max_norm, eps_total_norm,
                       [exact_a, exact_b], "exact")

    # no-clip sanity check too (confirms clip_coef really does clamp to 1 and
    # torch leaves the grads untouched, matching optimizerClipGradNorm's
    # skip-the-call behavior).
    torch_sanity_check([grad_a, grad_b], noclip_max_norm, total_norm, [grad_a, grad_b], "noclip")

    parts = [
        "// AUTOGENERATED by generate_expected_clip_grad_norm.py - DO NOT EDIT\n",
        "#ifndef ODT_EXPECTED_CLIP_GRAD_NORM_H\n#define ODT_EXPECTED_CLIP_GRAD_NORM_H\n",
        "#include <stdlib.h>\n\n",
    ]
    parts.append(emit_float_array("clip_grad_a", grad_a))
    parts.append(emit_float_array("clip_grad_b", grad_b))
    parts.append(emit_float_scalar("clip_total_norm", float(total_norm)))
    parts.append("\n")
    parts.append(emit_float_scalar("clip_noclip_max_norm", noclip_max_norm))
    parts.append("\n")
    parts.append(emit_float_scalar("clip_clip_max_norm", clip_max_norm))
    parts.append(emit_float_array("clip_clip_a", clip_a))
    parts.append(emit_float_array("clip_clip_b", clip_b))
    parts.append("\n")
    parts.append(emit_float_array("clip_eps_grad_a", eps_grad_a))
    parts.append(emit_float_array("clip_eps_grad_b", eps_grad_b))
    parts.append(emit_float_scalar("clip_eps_total_norm", float(eps_total_norm)))
    parts.append(emit_float_scalar("clip_exact_max_norm", exact_max_norm))
    parts.append(emit_float_array("clip_exact_a", exact_a))
    parts.append(emit_float_array("clip_exact_b", exact_b))
    parts.append("#endif // ODT_EXPECTED_CLIP_GRAD_NORM_H\n")
    with open(args.out, "w") as f:
        f.write("".join(parts))


if __name__ == "__main__":
    main()
