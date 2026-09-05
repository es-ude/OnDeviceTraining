#ifndef ODT_CONV1D_TRANSPOSED_H
#define ODT_CONV1D_TRANSPOSED_H

#include <stdbool.h>
#include <stdlib.h>

#include "ArithmeticType.h"
#include "ExecuteOp.h"
#include "Kernel.h"
#include "Layer.h"
#include "Tensor.h"

typedef struct conv1dTransposedConfig {
    kernel_t *kernel;
    parameter_t *weights; // [Cin, Cout/groups, K]
    parameter_t *bias;    // [Cout] or NULL
    size_t groups;        // must divide Cin and Cout
    size_t outputPadding; // PyTorch parameter; default 0; < max(stride, dilation)

    arithmetic_t forwardMath;    /* declared forward compute representation */
    arithmetic_t weightGradMath; /* declared weight-grad ARITHMETIC */
    arithmetic_t biasGradMath;   /* declared bias-grad ARITHMETIC */
    arithmetic_t propLossMath;   /* declared dx-wire ARITHMETIC (kernel selection) */

    quantization_t *outputQ;   /* produced forward-wire storage config */
    quantization_t *propLossQ; /* storage config of the produced dx-wire buffer */

    outputMode_t weightGradAccMode; /* weight-grad executeOp accumulate mode (PR3 spec D1) */
    outputMode_t biasGradAccMode;   /* bias-grad executeOp accumulate mode (PR3 spec D1) */

    bool ownsQuantizations; /* true -> free* will tear down outputQ/propLossQ and their
                               qConfigs */

    bool frozen; /* create-time freeze (#380): no grad buffers; optimizer collection,
                    state allocation and backward weight/bias-grad ops skip this layer */
} conv1dTransposedConfig_t;

void initConv1dTransposedConfigWithWeightsAndBias(
    conv1dTransposedConfig_t *cfg, kernel_t *kernel, parameter_t *weights, parameter_t *bias,
    size_t groups, size_t outputPadding, quantization_t *forwardQ, quantization_t *weightGradQ,
    quantization_t *biasGradQ, quantization_t *propLossQ);

void conv1dTransposedForward(layer_t *layer, tensor_t *input, tensor_t *output);

void conv1dTransposedBackward(layer_t *layer, tensor_t *forwardInput, tensor_t *lossGrad,
                              tensor_t *propLoss);

void conv1dTransposedCalcWeightGradsSymInt32(conv1dTransposedConfig_t *cfg, tensor_t *forwardInput,
                                             tensor_t *lossGrad);
void conv1dTransposedCalcBiasGradsSymInt32(conv1dTransposedConfig_t *cfg, tensor_t *lossGrad);

/*! BFP epic PR3 (Task 4): ARITH_BFP grad twins (executeOp-funneled, raw
 *  FLOAT32 intermediates, rounding-free ldexpf folds). Both REQUIRE
 *  BFP-stored weights (fail-fast) -- the width anchor FLOAT32-stored
 *  operands stage at; BFP-stored operands are borrowed zero-copy. */
void conv1dTransposedCalcWeightGradsBfp(conv1dTransposedConfig_t *cfg, tensor_t *forwardInput,
                                        tensor_t *lossGrad);
void conv1dTransposedCalcBiasGradsBfp(conv1dTransposedConfig_t *cfg, tensor_t *lossGrad);

void conv1dTransposedCalcOutputShape(layer_t *layer, shape_t *inputShape, shape_t *outputShape);

#endif // ODT_CONV1D_TRANSPOSED_H
