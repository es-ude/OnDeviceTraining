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

/* 1/sqrt(x + eps): eps INSIDE the sqrt guards x == 0 (matches LayerNorm). */
float rsqrtFloat32(float x, float eps);

/* Elementwise 1/sqrt over a SYM_INT32 tensor: dequant each mantissa via in's
 * scale, rsqrtFloat32, then requant the whole result into `out` with a dynamic
 * absmax-derived scale (absMax == 0 -> zero mantissas + scale 1.0), writing
 * out's qConfig->scale. `in` and `out` are SYM_INT32 of the same element count;
 * both are validated (qMaxBits <= 16). */
void rsqrtSymInt32(tensor_t *in, float eps, tensor_t *out);

#endif // ENV5_RUNTIME_REDUCE_H
