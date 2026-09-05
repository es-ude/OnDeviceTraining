#ifndef ODT_CONV1D_H
#define ODT_CONV1D_H

#include <stdbool.h>
#include <stdlib.h>

#include "ArithmeticType.h"
#include "ExecuteOp.h"
#include "Kernel.h"
#include "Layer.h"
#include "Tensor.h"

typedef struct conv1dConfig {
    kernel_t *kernel;
    parameter_t *weights; // [Cout, Cin/groups, K]
    parameter_t *bias;    // [Cout] or NULL
    size_t groups;        // must divide Cin and Cout

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
} conv1dConfig_t;

void initConv1dConfigWithWeightsAndBias(conv1dConfig_t *conv1dConfig, kernel_t *kernel,
                                        parameter_t *weights, parameter_t *bias, size_t groups,
                                        quantization_t *forwardQ, quantization_t *weightGradQ,
                                        quantization_t *biasGradQ, quantization_t *propLossQ);

void conv1dForward(layer_t *layer, tensor_t *input, tensor_t *output);

void conv1dBackward(layer_t *layer, tensor_t *forwardInput, tensor_t *lossGrad, tensor_t *propLoss);

void conv1dCalcWeightGradsSymInt32(conv1dConfig_t *cfg, tensor_t *forwardInput, tensor_t *lossGrad);
void conv1dCalcBiasGradsSymInt32(conv1dConfig_t *cfg, tensor_t *lossGrad);

/* BFP epic PR3 (Task 3): native ARITH_BFP grad twins of the SYM publics
 * above (executeOp wrappers; accumulate into cfg->weights->grad /
 * cfg->bias->grad under the layer's acc modes). REQUIRE BFP-stored weights
 * -- the width anchor FLOAT32-stored operands stage at (fail-fast
 * otherwise); BFP-stored operands are borrowed zero-copy. */
void conv1dCalcWeightGradsBfp(conv1dConfig_t *cfg, tensor_t *forwardInput, tensor_t *lossGrad);
void conv1dCalcBiasGradsBfp(conv1dConfig_t *cfg, tensor_t *lossGrad);

void conv1dCalcOutputShape(layer_t *layer, shape_t *inputShape, shape_t *outputShape);

#endif // ODT_CONV1D_H
