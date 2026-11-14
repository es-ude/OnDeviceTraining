#ifndef ODT_MAX_POOL_1D_H
#define ODT_MAX_POOL_1D_H

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#include "ArithmeticType.h"
#include "Kernel.h"
#include "Layer.h"
#include "Tensor.h"

/* argmaxIndices: INT32 [B, C, Lout] index state written by every forward and
 * read by the matching backward. The factory sizes it for B = 1; the forward
 * grows it to the largest B seen, keeps capacity on smaller batches and
 * rewrites dims[0] only when B changes. Its ->data must come from reserveMemory
 * because a growing forward frees and replaces it.
 * argmaxCapacity is valid only while argmaxCapacityData == argmaxIndices->data;
 * any other buffer (zero-initialised config, argmax swapped in by hand) is
 * adopted at its own element count on the next forward.
 * When replacing argmaxIndices by hand, set argmaxCapacityData = NULL (or re-run
 * initMaxPool1dConfig): a replacement that reuses the freed address would
 * otherwise inherit the stale capacity.
 *
 * Serialize access per layer instance: no two concurrent forwards, and no
 * forward concurrent with its backward. Concurrent forwards need separate
 * layer instances. */
typedef struct maxPool1dConfig {
    kernel_t *kernel;
    tensor_t *argmaxIndices;           // INT32 [B, C, Lout]; see the block comment above
    size_t argmaxCapacity;             // elements the argmax allocation holds
    const uint8_t *argmaxCapacityData; // the allocation argmaxCapacity describes
    arithmetic_t forwardMath;
    arithmetic_t propLossMath;
    quantization_t *outputQ;
    quantization_t *propLossQ;
    bool ownsQuantizations;
} maxPool1dConfig_t;

void initMaxPool1dConfig(maxPool1dConfig_t *cfg, kernel_t *kernel, tensor_t *argmaxIndices,
                         quantization_t *forwardQ, quantization_t *propLossQ);

void maxPool1dForward(layer_t *layer, tensor_t *input, tensor_t *output, const layerExec_t *exec);

void maxPool1dBackward(layer_t *layer, tensor_t *forwardInput, tensor_t *lossGrad,
                       tensor_t *propLoss, const layerExec_t *exec);

struct rngStream;

/* The integer-only ARITH_BFP forward kernel (executeOp adapter, ctx = the
 * config): ops[0] = the unpacked-BFP input, rawOut = BFP_RAW, auxOut = the
 * INT32 argmax. Exposed for kernel-level tests; layers call maxPool1dForward. */
void maxPool1dForwardKernelBfpRaw(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                                  const void *ctx, struct rngStream *rs);

/* The integer-only ARITH_BFP dx kernel (executeOp adapter, ctx = the config,
 * whose argmaxIndices the forward recorded; no auxOut): ops[0] = the
 * unpacked-BFP loss gradient, rawOut = BFP_RAW on the dx geometry
 * (opSpec_t.bfpRawOut = BFP_RAW_BACKWARD_DX). Exposed for kernel-level
 * tests; layers call maxPool1dBackward. */
void maxPool1dBackwardKernelBfpRaw(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                                   const void *ctx, struct rngStream *rs);

void maxPool1dCalcOutputShape(layer_t *layer, shape_t *inputShape, shape_t *outputShape);

#endif // ODT_MAX_POOL_1D_H
