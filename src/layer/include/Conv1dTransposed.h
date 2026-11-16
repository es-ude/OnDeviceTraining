#ifndef ODT_CONV1D_TRANSPOSED_H
#define ODT_CONV1D_TRANSPOSED_H

#include <stdbool.h>
#include <stdlib.h>

#include "ArithmeticType.h"
#include "Common.h"
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

    outputMode_t weightGradAccMode; /* weight-grad executeOp accumulate mode */
    outputMode_t biasGradAccMode;   /* bias-grad executeOp accumulate mode */

    bool ownsQuantizations; /* true -> free* will tear down outputQ/propLossQ and their
                               qConfigs */

    bool frozen; /* no parameter-gradient buffers; optimizer collection, state allocation
                    and backward weight/bias-grad ops skip this layer, but input gradients
                    still propagate */
} conv1dTransposedConfig_t;

/*! The geometry contract forward/backward rely on: VALID padding (the only geometry the
 *  ConvTranspose kernels implement), stride and dilation >= 1, groups dividing inChannels,
 *  outputPadding < max(stride, dilation), kernel size matching the weight, and a bias of
 *  outChannels elements. Call once where a config enters -- the factory, the config
 *  initialiser and the deserializer -- never per step. `who` prefixes the message. */
void conv1dTransposedValidateConfig(const conv1dTransposedConfig_t *cfg, const char *who);

void initConv1dTransposedConfigWithWeightsAndBias(
    conv1dTransposedConfig_t *cfg, kernel_t *kernel, parameter_t *weights, parameter_t *bias,
    size_t groups, size_t outputPadding, quantization_t *forwardQ, quantization_t *weightGradQ,
    quantization_t *biasGradQ, quantization_t *propLossQ);

void conv1dTransposedForward(layer_t *layer, tensor_t *input, tensor_t *output,
                             const layerExec_t *exec);

void conv1dTransposedBackward(layer_t *layer, tensor_t *forwardInput, tensor_t *lossGrad,
                              tensor_t *propLoss, const layerExec_t *exec);

void conv1dTransposedCalcWeightGradsSymInt32(conv1dTransposedConfig_t *cfg, tensor_t *forwardInput,
                                             tensor_t *lossGrad, const rngKey_t *key);
void conv1dTransposedCalcBiasGradsSymInt32(conv1dTransposedConfig_t *cfg, tensor_t *lossGrad,
                                           const rngKey_t *key);

/*! ARITH_BFP gradients with rounding-free ldexpf folds into FLOAT32
 *  intermediates. Requires BFP-stored weights (the caller guarantees it;
 *  conv1dTransposedBackward checks); FLOAT32 operands stage at their widths. */
void conv1dTransposedCalcWeightGradsBfp(conv1dTransposedConfig_t *cfg, tensor_t *forwardInput,
                                        tensor_t *lossGrad, const rngKey_t *key);
void conv1dTransposedCalcBiasGradsBfp(conv1dTransposedConfig_t *cfg, tensor_t *lossGrad,
                                      const rngKey_t *key);

void conv1dTransposedCalcOutputShape(layer_t *layer, shape_t *inputShape, shape_t *outputShape);

#endif // ODT_CONV1D_TRANSPOSED_H
