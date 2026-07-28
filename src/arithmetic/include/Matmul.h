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
 *  Per-channel weights (groupSize == the full reduction length per output
 *  channel) never cross a boundary mid-reduction, so exactly ONE combine
 *  runs per output element. */
void matmulSymInt32TensorsGroupedWeight(tensor_t *aTensor, tensor_t *bTensor, tensor_t *bias,
                                        tensor_t *outputTensor, const symQConfig_t *weightGroups);

size_t getMatmulInstructionCounter();

#endif // ENV5_RUNTIME_MATMUL_H
