#ifndef ENV5_RUNTIME_MATMUL_H
#define ENV5_RUNTIME_MATMUL_H

#include <stddef.h>

#include "SymGroupedWeight.h"
#include "Tensor.h"

typedef struct rngStream rngStream_t;

/* All entries: operands of rank <= 2 in the matching compute representation (the
 * executeOp prologue stages them); an optional bias has one value per output
 * column. A contraction-dimension mismatch exits. */

void matmulFloat32Tensors(tensor_t *aTensor, tensor_t *bTensor, tensor_t *outputTensor);

/* SYM_INT32 entries (this one, WithBias, GroupedWeight) overwrite the output's
 * scale and qMaxBits: the static width of the raw accumulator codes, up to 32. A
 * 32-bit output may be used as an operand or requantized from; requantize it to
 * at most 31 bits before cloning it or requantizing into it. */
void matmulSymInt32Tensors(tensor_t *aTensor, tensor_t *bTensor, tensor_t *outputTensor);

void matmulFloat32TensorsWithBias(tensor_t *aTensor, tensor_t *bTensor, tensor_t *outputTensor,
                                  tensor_t *bias);

void matmulSymInt32TensorsWithBias(tensor_t *aTensor, tensor_t *bTensor, tensor_t *outputTensor,
                                   tensor_t *bias, rngStream_t *rs);

/*! Grouped-weight matmul (GGUF style). `a` is per-tensor SYM_INT32; `b` is the
 *  rank-2 unpacked weight scratch (its scale is ignored, its rounding mode drives
 *  the combines); `groupedWeight` comes from symGroupedWeightOrNull and matches
 *  b's storage geometry. The accumulator scale is a's scale times the maximum
 *  group scale, so every rescale factor is <= 1. Each visited-group change and
 *  the tail fold the group's int32 partial with one rounding. Groups follow
 *  storage indices, so both the contiguous forward and the strided Linear-dx
 *  orientation work. */
void matmulSymInt32TensorsGroupedWeight(tensor_t *aTensor, tensor_t *bTensor, tensor_t *bias,
                                        tensor_t *outputTensor,
                                        const symGroupedWeight_t *groupedWeight, rngStream_t *rs);

/*! Block-floating-point GEMM with the FLOAT32 raw: since the integer GEMM
 *  (matmulBfpRawTensors below) it serves only the Linear weightGrad into
 *  FLOAT32 grads and under the fixed grid. Operands and the optional bias are
 *  the funnel's unpacked BFP scratch (int32 mantissas, live bfpQConfig_t); the
 *  output is raw FLOAT32 and the funnel epilogue packs it. The bias is
 *  dequantized into the float accumulator before the reduction. One int32
 *  partial per output element folds via ldexpf whenever either operand
 *  changes storage group (bfpGroupAt), and at the tail; the kernel never
 *  rounds. bfpValidateBlockHeadroom rejects a reduction whose segment could
 *  overflow int32. */
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
