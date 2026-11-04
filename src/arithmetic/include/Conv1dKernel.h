#ifndef ODT_CONV1D_KERNEL_H
#define ODT_CONV1D_KERNEL_H

#include <stdlib.h>

#include "Kernel.h"
#include "SymGroupedWeight.h"
#include "Tensor.h"

/* Pointer-only use: the identical typedef lives in RNG.h (C11 allows the
 * repetition), so this header needs no RNG include path. */
typedef struct rngStream rngStream_t;

/*! 1D convolution forward (FLOAT32 only). Correlation, not flipped.
 *
 *  @param input  [batch, in_channels, input_length], FLOAT32
 *  @param weight [out_channels, in_channels/groups, kernel_size], FLOAT32
 *  @param bias   [out_channels] or NULL, FLOAT32
 *  @param kernel kernel_t with size/stride/dilation/paddingType
 *  @param groups must divide in_channels and out_channels
 *  @param output [batch, out_channels, output_length], FLOAT32, pre-allocated
 */
void conv1dKernelFloat32(tensor_t const *input, tensor_t const *weight, tensor_t const *bias,
                         kernel_t const *kernel, size_t groups, tensor_t *output);

/*! 1D convolution forward (SYM_INT32). Integer correlation: int32 accumulator,
 *  mulInt32s per MAC, per-output-channel bias seed refolded into the product
 *  scale via rescaleIntoAccumulatorScale. Output mantissas are raw accumulator
 *  range at scale s_in*s_w, and the output's qMaxBits is overwritten with the
 *  static accumulator width, up to 32 (#516; Matmul.h has the requant-before-
 *  clone rule). The funnel epilogue restores the declared operand width.
 *
 *  @param input  [batch, in_channels, input_length], SYM_INT32
 *  @param weight [out_channels, in_channels/groups, kernel_size], SYM_INT32
 *  @param bias   [out_channels] or NULL, SYM_INT32
 *  @param kernel kernel_t with size/stride/dilation/paddingType
 *  @param groups must divide in_channels and out_channels
 *  @param output [batch, out_channels, output_length], SYM_INT32, pre-allocated
 */
void conv1dKernelSymInt32(tensor_t const *input, tensor_t const *weight, tensor_t const *bias,
                          kernel_t const *kernel, size_t groups, tensor_t *output, rngStream_t *rs);

/*! Group-quant PR2 (Task 4): grouped-weight Conv1d forward gather core.
 *  Sibling of conv1dKernelSymInt32, adding the running group-partial
 *  rescale-combine (mirrors matmulIntCoreGrouped's idiom exactly).
 *
 *  `weight` holds the grouped weight's UNPACKED mantissas as raw int32 (the
 *  same scratch shape executeOp's grouped-operand prologue branch produces:
 *  SYM_INT32 dtype, scale poisoned to 1.0f -- never read here, the real
 *  per-group scales live in the separate `groupedWeight` argument, and its
 *  qConfig->roundingMode carries the OP's rounding mode, exactly like
 *  matmulSymInt32TensorsGroupedWeight's `bQC`). `groupedWeight`
 *  (SymGroupedWeight.h) carries the ORIGINAL symQConfig_t of the stored
 *  weight tensor (validated: non-NULL, numGroups > 1 -- the per-tensor
 *  sentinel {1,0} is rejected, shape matches weight's element count via
 *  validateSymQConfigShape; any qBits is accepted) and the weight's code
 *  domain, which fixes the operand range the accumulator guard proves and
 *  the output width declares (SymAccumGuard.h). `bias` is NULL-able,
 *  SYM_INT32 like the scalar entry's bias.
 *
 *  Reduction order (per (batch, outChannel, outPos)): the weight's flat
 *  storage index for (oc, icOffset, kernelIdx) is
 *  (oc*inChannelsPerGroup + icOffset)*kernelSize + kernelIdx -- the SAME
 *  index conv1dKernelSymInt32 already computes for its own weight reads --
 *  so walking (icOffset, kernelIdx) in that same nested-loop order visits
 *  weight storage MONOTONICALLY increasing, exactly like
 *  matmulIntCoreGrouped's reduction axis (no separate contiguity check is
 *  needed here: unlike Matmul's tensor_t, which may expose a transposed
 *  logical view over its storage, this kernel always indexes `weight`
 *  through the SAME direct flat-array arithmetic conv1dKernelSymInt32 uses,
 *  never through calcElementIndexByIndices/orderOfDimensions). `g` binds to
 *  that index; a running int32 partial folds into the sAcc accumulator via
 *  rescaleIntoAccumulatorScale at every group-boundary crossing AND at the
 *  end of each reduction (GGUF rescale-combine pattern, #189): s_acc = s_in *
 *  max_g(weightGroups->scales[g]) -- never scales[0].
 *
 *  @param input  [batch, in_channels, input_length], SYM_INT32
 *  @param weight [out_channels, in_channels/groups, kernel_size], grouped-SYM
 *                scratch (raw int32 mantissas, scale poisoned to 1.0f)
 *  @param bias   [out_channels] or NULL, SYM_INT32
 *  @param kernel kernel_t with size/stride/dilation/paddingType
 *  @param groups must divide in_channels and out_channels (CONV groups --
 *                independent of the QUANTIZATION groups in groupedWeight)
 *  @param output [batch, out_channels, output_length], SYM_INT32, pre-allocated
 *  @param groupedWeight the stored weight's OWN symQConfig_t (group
 *                shape/scales) plus its code domain
 */
void conv1dKernelSymInt32Grouped(tensor_t const *input, tensor_t const *weight,
                                 tensor_t const *bias, kernel_t const *kernel, size_t groups,
                                 tensor_t *output, const symGroupedWeight_t *groupedWeight,
                                 rngStream_t *rs);

/*! BFP epic PR2 (Task 4): block-floating-point Conv1d forward. Same operand
 *  contract as matmulBfpTensors (see Matmul.h for the shared BFP GEMM-family
 *  kernel contract): input/weight/bias arrive in the funnel's UNPACKED-BFP
 *  scratch form (->data int32 sign-extended mantissa codes, ->quantization a
 *  live bfpQConfig_t), output is raw FLOAT32, the kernel never rounds.
 *  Unlike the SYM grouped kernel (which tracks only the weight's groups),
 *  BOTH operands' storage indices map to group ids per visited tap
 *  (bfpGroupAt -- per-element lookup, because padding clips taps and
 *  leaves GAPS in the visited weight/input index sequences; it divides only
 *  when an index leaves the cached current group); when EITHER id
 *  changes, the finished segment's int32 partial folds into the float
 *  accumulator via ldexpf, plus a tail fold per output element. `bias` is a
 *  value-seed dequantized to float BEFORE the reduction (headroom-exempt).
 *  int32 overflow is excluded up front by bfpValidateBlockHeadroom over the
 *  full reduction length inChannels/groups * kernelSize; every operand's
 *  group shape is fail-fast-checked via validateBfpQConfigShape.
 *
 *  @param input  [batch, in_channels, input_length], BFP scratch
 *  @param weight [out_channels, in_channels/groups, kernel_size], BFP scratch
 *  @param bias   [out_channels] or NULL, BFP scratch
 *  @param kernel kernel_t with size/stride/dilation/paddingType
 *  @param groups must divide in_channels and out_channels (CONV groups --
 *                independent of the QUANTIZATION groups in the qConfigs)
 *  @param output [batch, out_channels, output_length], FLOAT32, pre-allocated
 */
void conv1dKernelBfp(tensor_t const *input, tensor_t const *weight, tensor_t const *bias,
                     kernel_t const *kernel, size_t groups, tensor_t *output);

/*! Integer-only BFP Conv1d forward (docs/conventions/arithmetic-bfp.md §13,
 *  "The integer fold"): conv1dKernelBfp's walk -- same loop skeleton, loop
 *  order, MAC count, windowSlice1dAt and index math, same per-tap group
 *  tracking on BOTH operands -- with the float fold replaced by an int32
 *  accumulator per output in units of 2^E_acc, and the output a BFP_RAW
 *  tensor (int32 mantissas + one UNBIASED int16 exponent per raw target
 *  group, bfpRawQConfig_t) that the funnel re-blocks or dequantizes. No
 *  float, no division per MAC, no int64. conv1dKernelBfp (FLOAT32 raw)
 *  stays for the backward (the ConvT1d dx adjoint uses it).
 *
 *  Pre-pass (O(groups)): per output row (b, oc) E_top = max over the conv
 *  group's input rows of their group-max exponent + the weight row's
 *  group-max exponent, max'ed with oc's bias exponent; every raw target
 *  group's exponent is E_acc = max E_top over the rows it covers +
 *  bfpFoldGuardBits(inChannels/groups * kernelSize, widths) -- the static
 *  headroom proof (BfpKernelSupport.h). Per output: acc starts at the bias
 *  code aligned to E_acc (bfpAlignPartial, hoisted per channel and group);
 *  a same-exponent segment closes on EITHER operand's group change AND when
 *  it holds bfpSegmentLimit products (over-long runs are SPLIT, not
 *  rejected -- there is no headroom fail-fast here), its partial enters the
 *  accumulator as bfpAlignPartial(partial, E_acc - (Ein + Ew)) -- exact left
 *  shift or HALF_AWAY right shift; tail fold. SR never happens here: the
 *  re-block owns it.
 *
 *  @param input  [batch, in_channels, input_length], BFP scratch
 *  @param weight [out_channels, in_channels/groups, kernel_size], BFP scratch
 *  @param bias   [out_channels] or NULL, BFP scratch
 *  @param kernel kernel_t with size/stride/dilation/paddingType
 *  @param groups must divide in_channels and out_channels
 *  @param output [batch, out_channels, output_length], BFP_RAW with a valid
 *                {numGroups, groupSize} for the element count; every group's
 *                exponent is written
 */
void conv1dKernelBfpRaw(tensor_t const *input, tensor_t const *weight, tensor_t const *bias,
                        kernel_t const *kernel, size_t groups, tensor_t *output);

#endif // ODT_CONV1D_KERNEL_H
