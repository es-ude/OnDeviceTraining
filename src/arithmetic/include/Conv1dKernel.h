#ifndef ODT_CONV1D_KERNEL_H
#define ODT_CONV1D_KERNEL_H

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
 *   weight [out_channels, in_channels/groups, kernel_size]
 *   bias   [out_channels] or NULL
 *   output [batch, out_channels, output_length], pre-allocated
 * `groups` (conv groups, independent of quantization groups) must divide
 * in_channels and out_channels. Correlation, not flipped. */

/*! FLOAT32 forward. */
void conv1dKernelFloat32(tensor_t const *input, tensor_t const *weight, tensor_t const *bias,
                         kernel_t const *kernel, size_t groups, tensor_t *output);

/*! SYM_INT32 forward: int32 accumulator, per-output-channel bias seed rescaled
 *  into the product scale s_in*s_w. The output holds raw accumulator codes and
 *  its qMaxBits is set to the proven accumulator width (up to 32); the funnel
 *  epilogue restores the declared wire width (Matmul.h: requantize before you
 *  clone such a tensor). */
void conv1dKernelSymInt32(tensor_t const *input, tensor_t const *weight, tensor_t const *bias,
                          kernel_t const *kernel, size_t groups, tensor_t *output, rngStream_t *rs);

/*! Grouped-weight SYM_INT32 forward (gather). `weight` is the funnel's unpacked
 *  scratch (raw int32 codes, scale 1.0f, roundingMode = the op's mode); the real
 *  per-group scales and the code domain come from `groupedWeight`. Quantization
 *  groups bind to flat weight storage, which the (icOffset, kernelIdx) walk
 *  visits in increasing order, so the running int32 partial of each group run
 *  is combined once, at each group boundary and at the end of each reduction,
 *  into the accumulator scale A = {s_in.m, s_in.e + E + 15 - F} (SymGroupFold.h;
 *  E = the largest group exponent, F in [1, 8] the largest fraction count the
 *  accumulator guard proves with the bias seed):
 *      acc += round(partial * m_g * 2^-(E - e_g + 15 - F))   (symMulShiftRound,
 *  the op's rounding mode). Per output element |err| <= 0.5 * A * (number of
 *  combines) (HALF_AWAY; SR is unbiased); a window with no valid tap combines
 *  nothing and takes no SR draw. The bias is seeded at A before the reduction.
 *  Output scale = A, qMaxBits = the width of the site with x widened to
 *  [xMin * 2^F - 1, xMax * 2^F + 1] plus the bias seeds.
 *
 *  Legacy float path (inputs wider than 16 bits, no provable F >= 1, or A
 *  outside the canonical exponent range): each partial folds into
 *  s_acc = s_in * max_g(scales[g]) via rescaleIntoAccumulatorScale; the bias is
 *  refolded into s_acc; output scale = s_acc. */
void conv1dKernelSymInt32Grouped(tensor_t const *input, tensor_t const *weight,
                                 tensor_t const *bias, kernel_t const *kernel, size_t groups,
                                 tensor_t *output, const symGroupedWeight_t *groupedWeight,
                                 rngStream_t *rs);

/*! BFP forward with the FLOAT32 raw; since conv1dKernelBfpRaw it serves the
 *  backward only (the ConvT1d dx adjoint). Operands arrive as the funnel's
 *  unpacked BFP scratch (int32 mantissa codes plus a live bfpQConfig_t); the
 *  output is raw FLOAT32 and the kernel never rounds (operand contract:
 *  matmulBfpTensors in Matmul.h). Both operands are group-tracked per visited
 *  tap (bfpGroupAt), because padding leaves gaps in the index sequences; when
 *  either group changes, the finished int32 segment folds into a float
 *  accumulator via ldexpf. Bias is a float value seed. bfpValidateBlockHeadroom
 *  over in_channels/groups * kernel_size rules out int32 overflow before the
 *  loop. */
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
 *  Pre-pass (one read per operand code): per output row (b, oc) E_top = max
 *  over the conv group's input rows of their group-max exponent + the weight
 *  row's group-max exponent, max'ed with oc's bias exponent -- every maximum
 *  over the groups holding a NONZERO code (a zero code or zero bias sets
 *  nothing; nothing set: 0); every raw target group's exponent is E_acc =
 *  max E_top over the rows it covers +
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
