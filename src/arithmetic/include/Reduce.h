#ifndef ENV5_RUNTIME_REDUCE_H
#define ENV5_RUNTIME_REDUCE_H

#include "Tensor.h"
#include <stddef.h>

/* Axis reductions over the TRAILING k logical dims of `in`.
 *
 * The reduction collapses the last k logical dims (element count N = product of
 * those k dims) and yields one scalar per block, where a block is a fixed choice
 * of the leading rank-k logical dims (K = product of the leading dims). The
 * outputs (meanOut/meanIn/varOut) are FLOAT32 tensors the caller pre-sizes to
 * the leading dims; results are written FLAT in row-major order over the leading
 * dims (index b = 0..K-1). k == rank reduces the whole tensor to a single scalar
 * (K = 1).
 *
 * Reads honor `in`'s orderOfDimensions (permutation-aware): a transposed/viewed
 * input is gathered through the logical index map, never as a contiguous block.
 * Variance is BIASED (÷N). SYM_INT32 inputs read raw int32 mantissas + the
 * per-tensor qConfig->scale; the SYM entry points fail fast if `in` is not
 * SYM_INT32 or its qMaxBits exceeds 16. */
void meanOverTrailingAxesFloat32(tensor_t *in, size_t k, tensor_t *meanOut);
void meanOverTrailingAxesSymInt32(tensor_t *in, size_t k, tensor_t *meanOut); /* meanOut FLOAT32 */
void varianceBiasedOverTrailingAxesFloat32(tensor_t *in, size_t k, tensor_t *meanIn,
                                           tensor_t *varOut);
void varianceBiasedOverTrailingAxesSymInt32(tensor_t *in, size_t k, tensor_t *meanIn,
                                            tensor_t *varOut); /* varOut FLOAT32 */

/* Sum of squares over the trailing k logical dims: one FLOAT32 scalar per
 * leading-dims block (same block/permutation contract as the reductions
 * above). Serves PPCA row norms and CCIPCA ||v_i|| (#326). */
void sumSquaresOverTrailingAxesFloat32(tensor_t *in, size_t k, tensor_t *ssqOut);

/* BFP arms (epic PR5). `in` is the executeOp prologue's UNPACKED-BFP scratch
 * form: data = sign-extended int32 mantissa codes under a live bfpQConfig_t
 * (docs/conventions/arithmetic-bfp.md §5.3) -- never packed bytes. meanOut /
 * varOut are FLOAT32 [K]. The mean is a pure mantissa VALUE-sum: one int32
 * partial per same-exponent segment, folded via ldexpf on every group
 * crossing plus the tail (the R-P4/AvgPool sum contract; sum-headroom guard
 * applies). The variance dequants per element (exact mantissa*2^E), centers
 * against the float mean and accumulates float32 -- the mean is not on the
 * grid, so no mantissa-domain variance exists (mirrors the SYM_INT32
 * variance's dequant-center-square). */
void meanOverTrailingAxesBfp(tensor_t *in, size_t k, tensor_t *meanOut);
void varianceBiasedOverTrailingAxesBfp(tensor_t *in, size_t k, tensor_t *meanIn, tensor_t *varOut);

/* 1/sqrt(x + eps): eps INSIDE the sqrt guards x == 0 (matches LayerNorm). */
float rsqrtFloat32(float x, float eps);

/* sqrt(x), x >= 0: no guard -- callers (PPCA) clamp before calling, so a
 * negative input is a caller bug, not something this function should mask.
 * Fail-fast is wrong here; NaN propagates loudly instead. */
float sqrtFloat32(float x);

/* Elementwise 1/sqrt over a SYM_INT32 tensor: dequant each mantissa via in's
 * scale, rsqrtFloat32, then requant the whole result into `out` with a dynamic
 * absmax-derived scale (absMax == 0 -> zero mantissas + scale 1.0), writing
 * out's qConfig->scale. `in` and `out` are SYM_INT32 of the same element count;
 * both are validated (qMaxBits <= 16). */
void rsqrtSymInt32(tensor_t *in, float eps, tensor_t *out);

#endif // ENV5_RUNTIME_REDUCE_H
