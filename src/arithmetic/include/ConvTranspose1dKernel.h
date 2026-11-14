#ifndef ODT_CONV_TRANSPOSE_1D_KERNEL_H
#define ODT_CONV_TRANSPOSE_1D_KERNEL_H

#include <stdlib.h>

#include "Kernel.h"
#include "SymGroupedWeight.h"
#include "Tensor.h"

/* Pointer-only use: the identical typedef lives in RNG.h (C11 allows the
 * repetition), so this header needs no RNG include path. */
typedef struct rngStream rngStream_t;

/* Shared contract: these are executeOp kernels. The funnel prologue hands them
 * operands in the arithmetic's compute representation, so the kernels do not
 * re-check dtypes or group shapes. Layouts:
 *   input  [batch, in_channels, input_length]
 *   weight [in_channels, out_channels/groups, kernel_size]
 *   bias   [out_channels] or NULL
 *   output [batch, out_channels, output_length], pre-allocated
 * `groups` (conv groups, independent of quantization groups) must divide
 * in_channels and out_channels.
 *
 * Geometry. VALID: Lout = (Lin - 1) * stride + dilation * (K - 1)
 * + outputPadding + 1, with outputPadding == 0 || outputPadding <
 * max(stride, dilation) (PyTorch). SAME/EXPLICIT is the Conv1d dx adjoint:
 * Lout is the forward input length, padding comes from
 * windowGeometry1dCalc(Lout, kernel), whose output length must equal Lin, and
 * outputPadding must be 0. The kernels check the length equations. */

/*! FLOAT32 forward (scatter). */
void convTranspose1dKernelFloat32(tensor_t const *input, tensor_t const *weight,
                                  tensor_t const *bias, kernel_t const *kernel, size_t groups,
                                  size_t outputPadding, tensor_t *output);

/*! SYM_INT32 forward (scatter): int32 accumulator; the bias seed, rescaled into
 *  s_in*s_w, is added in a separate pass after the scatter. The output holds raw
 *  accumulator codes and its qMaxBits is set to the proven width including the
 *  seeds (up to 32; Matmul.h: requantize before you clone). Conv1d's dx uses it
 *  with bias == NULL. */
void convTranspose1dKernelSymInt32(tensor_t const *input, tensor_t const *weight,
                                   tensor_t const *bias, kernel_t const *kernel, size_t groups,
                                   size_t outputPadding, tensor_t *output, rngStream_t *rs);

/*! Grouped-weight SYM_INT32 forward (scatter). `weight` is the funnel's
 *  unpacked scratch (raw int32 codes, scale 1.0f, roundingMode = the op's
 *  mode); per-group scales and the code domain come from `groupedWeight`.
 *  Groups bind to flat weight storage. A scatter has no per-output run to carry
 *  a group partial across, so the INPUT code is folded into each group it meets
 *  (SymGroupFold.h): xv_g = round(xv * m_g * 2^-(E - e_g + 15 - F)), once per
 *  (input element, group run), and the products xv_g * w are plain int32 MACs
 *  at the accumulator scale A = {s_in.m, s_in.e + E + 15 - F} (E = the largest
 *  group exponent, F in [1, 8] the largest fraction count the accumulator
 *  guard proves). Per output element |err| <= 0.5 * A * sum|w| over its
 *  contributing products (HALF_AWAY; SR is unbiased); an input code of 0 is
 *  never folded and takes no SR draw. Bias is seeded at A after the scatter.
 *  Output scale = A, qMaxBits = the fold site's width (x widened to
 *  [xMin * 2^F - 1, xMax * 2^F + 1]) with the bias seeds.
 *
 *  Legacy float path (inputs wider than 16 bits, no provable F >= 1, or A
 *  outside the canonical exponent range): each product is rescaled into
 *  s_acc = s_in * max_g(scales[g]) on its own, so an output element receiving
 *  C products carries up to 0.5*C*s_acc rounding error; bias refolded into
 *  s_acc; output scale = s_acc.
 *
 *  Conv1d's grouped dx uses it with bias == NULL and the Conv1d weight layout
 *  (the flat-storage binding holds either way). */
void convTranspose1dKernelSymInt32Grouped(tensor_t const *input, tensor_t const *weight,
                                          tensor_t const *bias, kernel_t const *kernel,
                                          size_t groups, size_t outputPadding, tensor_t *output,
                                          const symGroupedWeight_t *groupedWeight, rngStream_t *rs);

/*! BFP forward, gather form: each output element is a dot product over its
 *  contributors (convTranspose1dTapsAt), because the scatter form cannot carry
 *  an int32 block partial. Operand contract as conv1dKernelBfp: unpacked BFP
 *  scratch in, raw FLOAT32 out, no rounding. Taps outer, icOffset inner; both
 *  operands are group-tracked per step (bfpGroupAt) and each finished int32
 *  segment folds into a float accumulator via bfpLdexpInt32. Bias is a float
 *  seed per output element, so outputPadding tail positions hold only the bias.
 *  bfpValidateBlockHeadroom over in_channels/groups * kernel_size rules out
 *  int32 overflow before the loop. */
void convTranspose1dKernelBfpGather(tensor_t const *input, tensor_t const *weight,
                                    tensor_t const *bias, kernel_t const *kernel, size_t groups,
                                    size_t outputPadding, tensor_t *output);

/*! Conv1d dx under ARITH_BFP: the ConvT1d adjoint on the SAME scatter loop as
 *  convTranspose1dKernelFloat32 (same loops, order, MAC count and index math),
 *  with an int32 raw accumulator per output element instead of a float.
 *
 *  Every product a * b (codes, |product| <= 2^(ma+mb-2)) enters its output
 *  element as product * 2^-(E_acc - Ea - Eb) (bfpAlignPartial: exact left
 *  shift, or half away from zero right shift). E_acc is fixed before the walk
 *  per (batch, conv group): the largest input-row exponent plus weight-row
 *  exponent over the group's input channels (row maxima over the groups
 *  holding a NONZERO code; a pair with an all-zero row sets nothing), plus
 *  the guard g of
 *  bfpFoldGuardBits over N = in_channels/groups * (taps that can land on one
 *  output position: k, k' with (k - k') * dilation a multiple of stride). So
 *  no accumulator can leave int32 whatever the operands; every product is its
 *  own one-product segment. The accumulators live in `output`'s buffer.
 *
 *  Output: the BFP_RAW (opSpec_t.bfpRawOut = BFP_RAW_BACKWARD_DX) on the dx
 *  wire's geometry. Each raw target group's exponent is the max E_acc over
 *  the output rows it covers that have a nonzero product (a group may
 *  straddle channels, conv groups and the batch; none: 0 + g); a sum on a row with a lower E_acc is
 * re-gridded to it by ONE HALF_AWAY right shift of the finished sum (bfpAlignPartial; a shift only
 *  shrinks, so the headroom proof holds). Error per element <= N/2 units of
 *  its row's 2^E_acc plus 1/2 unit of the group's 2^E_t. Every group's
 *  exponent is written, the empty output included. Integer only: rows and
 *  groups advance with cursors, no division. The epilogue re-blocks the raw
 *  into the BFP dx wire by the op's rounding mode (arithmetic-bfp.md §14).
 *
 *  Operands as convTranspose1dKernelBfpGather (unpacked BFP scratch), no bias
 *  (the Conv1d dx has none). Geometry resolves through the same helper as
 *  the other ConvT kernels.
 *
 *  @param input          [batch, in_channels, input_length], BFP scratch
 *  @param weight         [in_channels, out_channels/groups, kernel_size], BFP scratch
 *  @param kernel         kernel_t (paddingType VALID or SAME/EXPLICIT adjoint)
 *  @param groups         must divide in_channels and out_channels
 *  @param outputPadding  trailing zeros at output end (VALID only); 0 for SAME
 *  @param output         [batch, out_channels, output_length], BFP_RAW with a
 *                        valid bfpRawQConfig_t, pre-allocated
 */
void convTranspose1dKernelBfpScatter(tensor_t const *input, tensor_t const *weight,
                                     kernel_t const *kernel, size_t groups, size_t outputPadding,
                                     tensor_t *output);

#endif // ODT_CONV_TRANSPOSE_1D_KERNEL_H
