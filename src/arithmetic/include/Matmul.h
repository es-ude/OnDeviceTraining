#ifndef ENV5_RUNTIME_MATMUL_H
#define ENV5_RUNTIME_MATMUL_H

#include <stddef.h>

#include "Tensor.h"

typedef void (*matmulFunc_t)(tensor_t *aTensor, tensor_t *bTensor, tensor_t outputTensor);

void matmulInt32Tensors(tensor_t *aTensor, tensor_t *bTensor, tensor_t *outputTensor);

void matmulFloat32Tensors(tensor_t *aTensor, tensor_t *bTensor, tensor_t *outputTensor);

void matmulSymInt32Tensors(tensor_t *aTensor, tensor_t *bTensor, tensor_t *outputTensor);

void matmulFloat32TensorsWithBias(tensor_t *aTensor, tensor_t *bTensor, tensor_t *outputTensor,
                                  tensor_t *bias);

void matmulSymInt32TensorsWithBias(tensor_t *aTensor, tensor_t *bTensor, tensor_t *outputTensor,
                                   tensor_t *bias);

/*! Group-quant PR2 (Task 3): GGUF-style grouped-weight matmul. `a` is a plain
 *  (per-tensor) SYM_INT32 operand, validated exactly like the scalar entries
 *  above. `b` holds the grouped weight's UNPACKED mantissas as raw int32 (the
 *  same scratch shape the executeOp prologue's grouped-operand branch
 *  produces: SYM_INT32 dtype, scale poisoned to 1.0f — never read here).
 *  `weightGroups` is the ORIGINAL symQConfig_t carrying the real per-group
 *  scales/qBits/groupSize (validated: non-NULL, qBits <=
 *  ODT_SYM_OPERAND_QMAXBITS, shape matches `b`'s element count via
 *  validateSymQConfigShape). `bias` is NULL-able, SYM_INT32 like the scalar
 *  entries' bias.
 *
 *  Accumulator-scale rule (GGUF rescale-combine pattern): s_acc = s_in *
 *  max_g(weightGroups->scales[g]) — NEVER scales[0] or any other single
 *  group's scale, since every group's rescale factor scales[g]/s_wmax <= 1
 *  only when s_wmax is the true max (a smaller denominator would grow some
 *  group's rescaled mantissa past its accumulator headroom, #189). Each
 *  group-boundary crossing (and the reduction's end) folds that group's raw
 *  int32 partial into the running accumulator via ONE
 *  rescaleIntoAccumulatorScale call (one rounding per boundary, honoring the
 *  weight scratch operand's own roundingMode — the same plumbing the funnel
 *  prologue already uses to carry the op's rounding mode into SYM kernels).
 *  Group membership binds to each visited element's actual STORAGE index
 *  (PR3 Task 1), so ANY weight orientation of `b` works: in the contiguous
 *  forward orientation (Linear.c's transposeTensor(w,0,1)) per-channel
 *  weights fold exactly ONE combine per output element, while the strided
 *  Linear-dx orientation (RAW [outFeatures, inFeatures] view, reduction over
 *  logical dim 0) hops groups on every reduction step and folds one combine
 *  per visited-group change. */
void matmulSymInt32TensorsGroupedWeight(tensor_t *aTensor, tensor_t *bTensor, tensor_t *bias,
                                        tensor_t *outputTensor, const symQConfig_t *weightGroups);

/*! BFP epic PR2 (Task 3): block-floating-point GEMM forward. Operands are in
 *  the executeOp funnel's UNPACKED-BFP scratch form: ->data holds int32
 *  sign-extended mantissa codes, ->quantization is BFP with a live
 *  bfpQConfig_t (the form exists only between funnel prologue and kernel).
 *  `bias` is NULL-able, same form, element count == output columns; it is a
 *  VALUE-seed dequantized to float BEFORE the reduction ((float)mantissa *
 *  bfpGroupScale), so it is exempt from the product-operand headroom bound.
 *  Output is raw FLOAT32 -- no width-restore here (that is the funnel
 *  epilogue's OUT_WRITE job, never the kernel's).
 *
 *  Kernel contract (shared across the BFP GEMM family): per output element
 *  ONE int32 partial; per reduction step both operands' STORAGE indices map
 *  to group ids (bfpGroupOf -- gap-robust on strided walks, exactly like
 *  matmulIntCoreGrouped's per-element division); when EITHER id changes, the
 *  finished segment folds via acc += ldexpf((float)partial, Ea + Eb - biasA
 *  - biasB) and the partial resets; tail-fold after the loop. The kernel
 *  never rounds (rounding lives at staging and the OUT_WRITE epilogue).
 *  int32 overflow is excluded up front by bfpValidateBlockHeadroom
 *  (BfpKernelSupport.h): a same-exponent segment never accumulates more than
 *  min(runA, runB, K) products, which must stay within
 *  bfpSegmentLimit(ma, mb) == INT32_MAX >> (ma+mb-2) (#227 headroom, no
 *  int64 anywhere). */
void matmulBfpTensors(tensor_t *aTensor, tensor_t *bTensor, tensor_t *bias, tensor_t *outputTensor);

size_t getMatmulInstructionCounter();

#endif // ENV5_RUNTIME_MATMUL_H
