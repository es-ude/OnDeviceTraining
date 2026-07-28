#ifndef ODT_CONV1D_KERNEL_H
#define ODT_CONV1D_KERNEL_H

#include <stdlib.h>

#include "Kernel.h"
#include "Tensor.h"

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
 *  range at scale s_in*s_w (a downstream Quantization layer restores int16).
 *
 *  @param input  [batch, in_channels, input_length], SYM_INT32
 *  @param weight [out_channels, in_channels/groups, kernel_size], SYM_INT32
 *  @param bias   [out_channels] or NULL, SYM_INT32
 *  @param kernel kernel_t with size/stride/dilation/paddingType
 *  @param groups must divide in_channels and out_channels
 *  @param output [batch, out_channels, output_length], SYM_INT32, pre-allocated
 */
void conv1dKernelSymInt32(tensor_t const *input, tensor_t const *weight, tensor_t const *bias,
                          kernel_t const *kernel, size_t groups, tensor_t *output);

/*! Group-quant PR2 (Task 4): grouped-weight Conv1d forward gather core.
 *  Sibling of conv1dKernelSymInt32, adding the running group-partial
 *  rescale-combine (mirrors matmulIntCoreGrouped's idiom exactly).
 *
 *  `weight` holds the grouped weight's UNPACKED mantissas as raw int32 (the
 *  same scratch shape executeOp's grouped-operand prologue branch produces:
 *  SYM_INT32 dtype, scale poisoned to 1.0f -- never read here, the real
 *  per-group scales live in the separate `weightGroups` argument, and its
 *  qConfig->roundingMode carries the OP's rounding mode, exactly like
 *  matmulSymInt32TensorsGroupedWeight's `bQC`). `weightGroups` is the
 *  ORIGINAL symQConfig_t of the stored weight tensor (validated: non-NULL,
 *  qBits <= ODT_SYM_OPERAND_QMAXBITS, numGroups > 1 -- the per-tensor
 *  sentinel {1,0} is rejected, shape matches weight's element count via
 *  validateSymQConfigShape). `bias` is NULL-able, SYM_INT32 like the scalar
 *  entry's bias.
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
 *                independent of the QUANTIZATION groups in weightGroups)
 *  @param output [batch, out_channels, output_length], SYM_INT32, pre-allocated
 *  @param weightGroups the stored weight's OWN symQConfig_t (group shape/scales)
 */
void conv1dKernelSymInt32Grouped(tensor_t const *input, tensor_t const *weight,
                                 tensor_t const *bias, kernel_t const *kernel, size_t groups,
                                 tensor_t *output, const symQConfig_t *weightGroups);

#endif // ODT_CONV1D_KERNEL_H
