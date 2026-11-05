#ifndef ODT_CONV_TRANSPOSE_1D_KERNEL_H
#define ODT_CONV_TRANSPOSE_1D_KERNEL_H

#include <stdlib.h>

#include "Kernel.h"
#include "SymGroupedWeight.h"
#include "Tensor.h"

/* Pointer-only use: the identical typedef lives in RNG.h (C11 allows the
 * repetition), so this header needs no RNG include path. */
typedef struct rngStream rngStream_t;

/*! Transposed 1D convolution forward (FLOAT32). Scatter form: each input
 *  value is multiplied into a window region of the output.
 *
 *  Supports both VALID and SAME paddingType. SAME is used by Conv1d-backward
 *  via the adjoint identity (dL/dx of Conv1d(SAME) = convTranspose1d(lossGrad,
 *  W, NULL, kernel-with-SAME, groups, 0, propLoss) where propLoss takes the
 *  shape of the original Conv1d's input). In SAME mode, padLeft/padRight are
 *  recovered from windowGeometry1dCalc(outputLength, kernel) — that is, the
 *  forward Conv1d's geometry on the adjoint output shape.
 *
 *  VALID output length:
 *    Lout = (Lin - 1) * stride + dilation * (K - 1) + outputPadding + 1
 *  SAME output length: caller-determined; the kernel asserts that
 *    windowGeometry1dCalc(Lout, kernel).outputLength == Lin.
 *
 *  PyTorch convention: outputPadding must satisfy
 *    outputPadding == 0  ||  outputPadding < max(stride, dilation)
 *  outputPadding must be 0 in SAME mode.
 *
 *  @param input          [batch, in_channels, input_length], FLOAT32
 *  @param weight         [in_channels, out_channels/groups, kernel_size], FLOAT32
 *  @param bias           [out_channels] or NULL, FLOAT32
 *  @param kernel         kernel_t (paddingType VALID or SAME)
 *  @param groups         must divide in_channels and out_channels
 *  @param outputPadding  trailing zeros at output end (VALID only); 0 for SAME
 *  @param output         [batch, out_channels, output_length], FLOAT32, pre-allocated
 */
void convTranspose1dKernelFloat32(tensor_t const *input, tensor_t const *weight,
                                  tensor_t const *bias, kernel_t const *kernel, size_t groups,
                                  size_t outputPadding, tensor_t *output);

/*! Transposed 1D convolution forward (SYM_INT32). Integer scatter sibling of
 *  convTranspose1dKernelFloat32: int32 accumulator, mulInt32s, per-output-channel
 *  bias seed (refold) added in a separate pass. Output mantissas are raw
 *  accumulator range at scale s_in*s_w; the output's qMaxBits is overwritten
 *  with the static accumulator width including the bias-pass seed, up to 32
 *  (#516; Matmul.h has the requant-before-clone rule). Same VALID/SAME/EXPLICIT geometry; used
 *  as Conv1d's dx adjoint with bias==NULL (SAME/EXPLICIT branch), and as
 *  Conv1dTransposed's forward in PR3.
 *
 *  @param input  [batch, in_channels, input_length], SYM_INT32
 *  @param weight [in_channels, out_channels/groups, kernel_size], SYM_INT32
 *  @param bias   [out_channels] or NULL, SYM_INT32
 */
void convTranspose1dKernelSymInt32(tensor_t const *input, tensor_t const *weight,
                                   tensor_t const *bias, kernel_t const *kernel, size_t groups,
                                   size_t outputPadding, tensor_t *output, rngStream_t *rs);

/*! Group-quant PR3 (Task 2): grouped-weight sibling of
 *  convTranspose1dKernelSymInt32 -- the SCATTER core. The weight's
 *  quantization groups bind to FLAT STORAGE (g = wStorageIdx /
 *  weightGroups->groupSize), and the scatter's weight read at
 *  (ic, ocOffset, k) is the direct flat index
 *  (ic*outChPerGroup + ocOffset)*kernelSize + k -- the SAME index
 *  convTranspose1dKernelSymInt32 computes, never a transposed logical view.
 *
 *  Unlike the gather cores (matmulIntCoreGrouped /
 *  conv1dKernelSymInt32Grouped), there is NO running group-partial here:
 *  consecutive products of a k-run land in DIFFERENT output elements, so no
 *  per-(target, group) run exists across which a raw int32 partial could be
 *  carried. Each product is rescaled into the accumulator scale
 *  INDIVIDUALLY: yAcc[outIdx] += rescaleIntoAccumulatorScale(x*w,
 *  s_in*scales[g], sAcc, op rounding), with sAcc = s_in *
 *  max_g(weightGroups->scales[g]) -- never scales[0]. Error consequence: an
 *  output element that C products scatter into carries |err| <= 0.5*C*sAcc
 *  worst case. Bias is refolded into sAcc AFTER the scatter (same pass order
 *  and primitive as the scalar sibling). Output scale = sAcc.
 *
 *  Same VALID/SAME/EXPLICIT geometry as the scalar sibling; Conv1d's grouped
 *  dx adjoint (PR3 Task 3) reuses this entry with bias==NULL and the
 *  [out_channels, in_channels/groups, K] layout (the group binding is to
 *  flat storage either way).
 *
 *  @param input         [batch, in_channels, input_length], SYM_INT32
 *  @param weight        [in_channels, out_channels/groups, kernel_size],
 *                       grouped-SYM scratch (raw int32 mantissas, scale
 *                       poisoned to 1.0f; roundingMode carries the OP's mode)
 *  @param bias          [out_channels] or NULL, SYM_INT32
 *  @param kernel        kernel_t (paddingType VALID or SAME/EXPLICIT adjoint)
 *  @param groups        must divide in_channels and out_channels (CONV groups
 *                       -- independent of the QUANTIZATION groups)
 *  @param outputPadding trailing zeros at output end (VALID only); 0 for SAME
 *  @param output        [batch, out_channels, output_length], SYM_INT32,
 *                       pre-allocated
 *  @param groupedWeight the stored weight's OWN symQConfig_t (group shape/
 *                       scales; must be grouped, numGroups > 1) plus its code
 *                       domain, which fixes the operand range the accumulator
 *                       guard proves and the output width declares
 *                       (SymGroupedWeight.h, SymAccumGuard.h)
 */
void convTranspose1dKernelSymInt32Grouped(tensor_t const *input, tensor_t const *weight,
                                          tensor_t const *bias, kernel_t const *kernel,
                                          size_t groups, size_t outputPadding, tensor_t *output,
                                          const symGroupedWeight_t *groupedWeight, rngStream_t *rs);

/*! BFP epic PR2 (Task 5, D9): output-centric (gather) ConvT1d forward under
 *  ARITH_BFP -- every output element is a dot product over its contributors
 *  (convTranspose1dTapsAt), giving the int32 block-partial contract
 *  (same-exponent segments) that a scatter cannot (consecutive scatter
 *  products land in DIFFERENT output elements; the Conv1d dx scatter below
 *  aligns every product on its own instead); the SYM scatter cores above
 *  are untouched.
 *
 *  Same operand contract as matmulBfpTensors / conv1dKernelBfp: input/weight/
 *  bias arrive in the funnel's UNPACKED-BFP scratch form (->data int32
 *  sign-extended mantissa codes, ->quantization a live bfpQConfig_t), output
 *  is raw FLOAT32, the kernel never rounds. The reduction walks taps OUTER /
 *  icOffset INNER; BOTH operands' storage indices map to group ids per step
 *  (bfpGroupAt -- per-element lookup, tap hops make both index sequences
 *  non-contiguous; it divides only when an index leaves the cached current
 *  group); when EITHER id changes, the finished segment's int32
 *  partial folds into the float accumulator via bfpLdexpInt32, plus a tail fold per
 *  output element. `bias` is seeded per output element (value-seed
 *  dequantized to float BEFORE the reduction -- no separate bias pass,
 *  unlike the scatter cores); outputPadding tail positions have no
 *  contributors and stay at the bias seed. int32 overflow is excluded up
 *  front by bfpValidateBlockHeadroom over in_channels/groups * kernel_size;
 *  every operand's group shape is fail-fast-checked via
 *  validateBfpQConfigShape. Geometry (VALID, and the SAME/EXPLICIT adjoint)
 *  resolves through the same shared helper as the scatter kernels.
 *
 *  @param input          [batch, in_channels, input_length], BFP scratch
 *  @param weight         [in_channels, out_channels/groups, kernel_size], BFP scratch
 *  @param bias           [out_channels] or NULL, BFP scratch
 *  @param kernel         kernel_t (paddingType VALID or SAME/EXPLICIT adjoint)
 *  @param groups         must divide in_channels and out_channels (CONV groups
 *                        -- independent of the QUANTIZATION groups)
 *  @param outputPadding  trailing zeros at output end (VALID only); 0 for SAME
 *  @param output         [batch, out_channels, output_length], FLOAT32, pre-allocated
 */
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
