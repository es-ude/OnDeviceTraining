#ifndef ODT_CONV_TRANSPOSE_1D_KERNEL_H
#define ODT_CONV_TRANSPOSE_1D_KERNEL_H

#include <stdlib.h>

#include "Kernel.h"
#include "Tensor.h"

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
 *  accumulator range at scale s_in*s_w. Same VALID/SAME/EXPLICIT geometry; used
 *  as Conv1d's dx adjoint with bias==NULL (SAME/EXPLICIT branch), and as
 *  Conv1dTransposed's forward in PR3.
 *
 *  @param input  [batch, in_channels, input_length], SYM_INT32
 *  @param weight [in_channels, out_channels/groups, kernel_size], SYM_INT32
 *  @param bias   [out_channels] or NULL, SYM_INT32
 */
void convTranspose1dKernelSymInt32(tensor_t const *input, tensor_t const *weight,
                                   tensor_t const *bias, kernel_t const *kernel, size_t groups,
                                   size_t outputPadding, tensor_t *output);

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
 *  @param weightGroups  the stored weight's OWN symQConfig_t (group shape/
 *                       scales; must be grouped, numGroups > 1)
 */
void convTranspose1dKernelSymInt32Grouped(tensor_t const *input, tensor_t const *weight,
                                          tensor_t const *bias, kernel_t const *kernel,
                                          size_t groups, size_t outputPadding, tensor_t *output,
                                          const symQConfig_t *weightGroups);

#endif // ODT_CONV_TRANSPOSE_1D_KERNEL_H
