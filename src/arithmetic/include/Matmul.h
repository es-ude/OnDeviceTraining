#ifndef ENV5_RUNTIME_MATMUL_H
#define ENV5_RUNTIME_MATMUL_H

#include <stddef.h>

#include "SymGroupedWeight.h"
#include "Tensor.h"

/* Pointer-only use: the identical typedef lives in RNG.h (C11 allows the
 * repetition), so this header needs no RNG include path. */
typedef struct rngStream rngStream_t;

typedef void (*matmulFunc_t)(tensor_t *aTensor, tensor_t *bTensor, tensor_t outputTensor);

void matmulInt32Tensors(tensor_t *aTensor, tensor_t *bTensor, tensor_t *outputTensor);

void matmulFloat32Tensors(tensor_t *aTensor, tensor_t *bTensor, tensor_t *outputTensor);

/* SYM_INT32 entries (this one, WithBias, GroupedWeight): the output's
 * symInt32QConfig_t is overwritten -- `scale` with the product scale and
 * `qMaxBits` with the static accumulator width of the raw codes (#516:
 * symAccumResultBits, the exact two's-complement width of the accumulator's
 * range, up to 32 = the raw int32 range). A 32-wide
 * output is an operand or a requant source only: requantize it into a <= 31-bit
 * tensor before cloning it (getQLike) or requantizing into it. */
void matmulSymInt32Tensors(tensor_t *aTensor, tensor_t *bTensor, tensor_t *outputTensor);

void matmulFloat32TensorsWithBias(tensor_t *aTensor, tensor_t *bTensor, tensor_t *outputTensor,
                                  tensor_t *bias);

void matmulSymInt32TensorsWithBias(tensor_t *aTensor, tensor_t *bTensor, tensor_t *outputTensor,
                                   tensor_t *bias, rngStream_t *rs);

/*! Group-quant PR2 (Task 3): GGUF-style grouped-weight matmul. `a` is a plain
 *  (per-tensor) SYM_INT32 operand, validated exactly like the scalar entries
 *  above. `b` holds the grouped weight's UNPACKED mantissas as raw int32 (the
 *  same scratch shape the executeOp prologue's grouped-operand branch
 *  produces: SYM_INT32 dtype, scale poisoned to 1.0f — never read here).
 *  `groupedWeight` (SymGroupedWeight.h) carries the ORIGINAL symQConfig_t
 *  with the real per-group scales/qBits/groupSize (validated: non-NULL,
 *  numGroups > 1, shape matches `b`'s element count via
 *  validateSymQConfigShape; any qBits is accepted) and the weight's code
 *  domain, which fixes the operand range the accumulator guard proves and
 *  the output width declares (SymAccumGuard.h). `bias` is NULL-able,
 *  SYM_INT32 like the scalar entries' bias.
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
                                        tensor_t *outputTensor,
                                        const symGroupedWeight_t *groupedWeight, rngStream_t *rs);

/*! BFP epic PR2 (Task 3): block-floating-point GEMM with the FLOAT32 raw --
 *  since the integer forward and backward (matmulBfpRawTensors below) it
 *  serves only the Linear weightGrad into FLOAT32 grads and under the fixed
 *  grid (the float bridge). Operands are in
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
 *  to group ids (bfpGroupAt -- gap-robust on strided walks like
 *  matmulIntCoreGrouped's per-element division, but the division runs only
 *  when an index leaves the cached current group); when EITHER id changes, the
 *  finished segment folds via acc += bfpLdexpInt32(partial, Ea + Eb - biasA
 *  - biasB) and the partial resets; tail-fold after the loop. The kernel
 *  never rounds (rounding lives at staging and the OUT_WRITE epilogue).
 *  Every operand's group shape is fail-fast-checked against its element
 *  count (validateBfpQConfigShape -- the SYM grouped entry's precedent),
 *  since bfpGroupOf's division knows nothing of numGroups. int32 overflow
 *  is excluded up front by bfpValidateBlockHeadroom
 *  (BfpKernelSupport.h): a same-exponent segment never accumulates more than
 *  min(runA, runB, K) products, which must stay within
 *  bfpSegmentLimit(ma, mb) == INT32_MAX >> (ma+mb-2) (#227 headroom, no
 *  int64 anywhere). */
void matmulBfpTensors(tensor_t *aTensor, tensor_t *bTensor, tensor_t *bias, tensor_t *outputTensor);

/*! Integer-only BFP GEMM (docs/conventions/arithmetic-bfp.md §13: the Linear
 *  forward; §14: the Linear dx and the weightGrad into BFP grads, through the
 *  same entry on the backward operands' views).
 *  Same operands, walk and group tracking as matmulBfpTensors; the output is
 *  BFP_RAW (int32 mantissas + one unbiased exponent per raw target group,
 *  geometry from the output's own bfpRawQConfig_t, every exponent written)
 *  and the fold is integer: ONE int32 accumulator per output in units of
 *  2^E_acc, E_acc = (max over the group's outputs of max(rowTop(a) +
 *  colTop(b), E_bias)) + g, g from bfpFoldGuardBits; every segment enters
 *  through bfpAlignPartial (exact left shift / HALF_AWAY right shift, never
 *  SR) and the bias is one more segment. Over-long same-exponent runs SPLIT
 *  at bfpSegmentLimit (no fail-fast). rowTop / colTop are per-line maxima when
 *  the reduction axis is physically innermost (Linear's input and transposed
 *  weight), else the operand-wide max (coarser, still sound) -- every
 *  maximum over the groups holding a NONZERO code, and E_bias only for a
 *  nonzero bias code (zero groups never set E_acc; nothing set: 0). */
void matmulBfpRawTensors(tensor_t *aTensor, tensor_t *bTensor, tensor_t *bias,
                         tensor_t *outputTensor);

size_t getMatmulInstructionCounter();

#endif // ENV5_RUNTIME_MATMUL_H
