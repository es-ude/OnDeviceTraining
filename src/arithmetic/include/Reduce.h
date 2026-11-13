#ifndef ENV5_RUNTIME_REDUCE_H
#define ENV5_RUNTIME_REDUCE_H

#include "Tensor.h"
#include <stddef.h>

/* Reductions over the trailing k logical dims of `in` (caller guarantees k <= rank),
 * one result per block of the leading dims: K = product of the leading dims,
 * N = product of the trailing k. The caller picks the arm matching in's dtype.
 * Outputs are FLOAT32 [K], written row-major over the leading dims; k == rank
 * gives one scalar. Reads honor orderOfDimensions, so permuted views are fine.
 * Variance is biased (divides by N). The SYM arms need 1 <= qMaxBits <= 16 and
 * N < 2^(32-qMaxBits) so the int32 value-sum cannot overflow. */
void meanOverTrailingAxesFloat32(tensor_t *in, size_t k, tensor_t *meanOut);
void meanOverTrailingAxesSymInt32(tensor_t *in, size_t k, tensor_t *meanOut);
void varianceBiasedOverTrailingAxesFloat32(tensor_t *in, size_t k, tensor_t *meanIn,
                                           tensor_t *varOut);
void varianceBiasedOverTrailingAxesSymInt32(tensor_t *in, size_t k, tensor_t *meanIn,
                                            tensor_t *varOut);
void sumSquaresOverTrailingAxesFloat32(tensor_t *in, size_t k, tensor_t *ssqOut);

/* BFP arms. `in` is the executeOp prologue's unpacked scratch (sign-extended int32
 * mantissas under a live bfpQConfig_t), never packed bytes; N > 0. The mean sums
 * mantissas in int32 per same-exponent segment and folds each segment with ldexpf
 * (sum-headroom guard applies). The variance dequantizes each element and
 * accumulates in float, because the mean is not on the grid. */
void meanOverTrailingAxesBfp(tensor_t *in, size_t k, tensor_t *meanOut);
void varianceBiasedOverTrailingAxesBfp(tensor_t *in, size_t k, tensor_t *meanIn, tensor_t *varOut);

/* 1/sqrt(x + eps): eps inside the sqrt keeps zero variance finite (as LayerNorm). */
float rsqrtFloat32(float x, float eps);

/* Caller guarantees x >= 0 (PPCA clamps first); a negative x yields NaN, not an exit. */
float sqrtFloat32(float x);

#endif // ENV5_RUNTIME_REDUCE_H
