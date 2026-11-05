#ifndef ODT_AVG_POOL_1D_H
#define ODT_AVG_POOL_1D_H

#include <stdbool.h>
#include <stdlib.h>

#include "ArithmeticType.h"
#include "Kernel.h"
#include "Layer.h"
#include "Tensor.h"

typedef struct avgPool1dConfig {
    kernel_t *kernel;
    arithmetic_t forwardMath;
    arithmetic_t propLossMath;
    quantization_t *outputQ;
    quantization_t *propLossQ;
    bool ownsQuantizations;
} avgPool1dConfig_t;

void initAvgPool1dConfig(avgPool1dConfig_t *cfg, kernel_t *kernel, quantization_t *forwardQ,
                         quantization_t *propLossQ);

void avgPool1dForward(layer_t *layer, tensor_t *input, tensor_t *output, const layerExec_t *exec);

void avgPool1dBackward(layer_t *layer, tensor_t *forwardInput, tensor_t *lossGrad,
                       tensor_t *propLoss, const layerExec_t *exec);
void avgPool1dBackwardFloat(layer_t *layer, tensor_t *forwardInput, tensor_t *lossGrad,
                            tensor_t *propLoss);

struct rngStream;

/* The integer-only ARITH_BFP forward kernel (executeOp adapter, ctx = the
 * config, no auxOut): ops[0] = the unpacked-BFP input, rawOut = BFP_RAW.
 * Exposed for kernel-level tests; layers call avgPool1dForward. */
void avgPool1dForwardKernelBfpRaw(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                                  const void *ctx, struct rngStream *rs);

/* The integer-only ARITH_BFP dx kernel (executeOp adapter, ctx = the config,
 * no auxOut): ops[0] = the unpacked-BFP loss gradient, rawOut = BFP_RAW on
 * the dx geometry (opSpec_t.bfpRawOut = BFP_RAW_BACKWARD_DX). Exposed for
 * kernel-level tests; layers call avgPool1dBackward. */
void avgPool1dBackwardKernelBfpRaw(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                                   const void *ctx, struct rngStream *rs);

void avgPool1dCalcOutputShape(layer_t *layer, shape_t *inputShape, shape_t *outputShape);

#endif // ODT_AVG_POOL_1D_H
