#ifndef ODT_CONV1D_H
#define ODT_CONV1D_H

#include <stdlib.h>

#include "Kernel.h"
#include "Layer.h"
#include "Tensor.h"

typedef struct conv1dConfig {
    kernel_t *kernel;
    parameter_t *weights;
    parameter_t *bias;

    quantization_t *forwardQ;
    quantization_t *weightGradQ;
    quantization_t *biasGradQ;
    quantization_t *propLossQ;
} conv1dConfig_t;

/*! Sets conv1dConfig attributes.
 *
 * @param conv1dConfig Pointer to config
 * @param kernel Pointer to kernel
 * @param weights Pointer to weights
 * @param bias Pointer to bias
 * @param forwardQ Quantization for forward pass
 * @param weightGradQ Quantization for weight gradient calculation
 * @param biasGradQ Quantization for bias gradient calculation
 * @param propLossQ Quantization for prop loss calculation
 */
void initConv1dConfigWithWeightsAndBias(conv1dConfig_t *conv1dConfig, kernel_t *kernel,
                                        parameter_t *weights, parameter_t *bias,
                                        quantization_t *forwardQ, quantization_t *weightGradQ,
                                        quantization_t *biasGradQ, quantization_t *propLossQ);

/*! Calculates output length per channel by given input length per channel and kernel.
 *
 * @param inputLengthPerChannel Input length per channel
 * @param kernel Pointer to kernel_t
 * @returns Output length per channel
 */
size_t calcOutputLengthPerChannel(size_t inputLengthPerChannel, kernel_t *kernel);

/*! Calculates 1d convolution forward from given input, kernel, weights and bias. \n
 * Bias is optional.\n
 * The result is written to the given output tensor.
 *
 * @param conv1d Pointer to conv1d layer
 * @param input Pointer to input
 * @param output Pointer to output
 */
void conv1dForwardFloat(layer_t *conv1d, tensor_t *input, tensor_t *output);

void conv1dForward(layer_t *conv1d, tensor_t *input, tensor_t *output);

/*! Calculates 1d convolution backward from given input, kernel, weights and bias. \n
 * Prop loss will be written into the given prop loss tensor. \n
 * Weight gradients and (if provided) bias gradients will be updated accordingly.
 *
 * @param input Pointer to input
 * @param propLoss Pointer to prop loss
 * @param lossGrad dL/dy
 * @param kernel Pointer to kernel
 * @param weights Pointer to weights
 * @param bias Pointer to bias (NULL if you don't want to use bias)
 */
void conv1dBackwardFloat(tensor_t *input, tensor_t *propLoss, tensor_t *lossGrad, kernel_t *kernel,
                         parameter_t *weights, parameter_t *bias);

void conv1dBackward(layer_t *conv1d, tensor_t *input, tensor_t *lossGrad, tensor_t *propLoss);

void conv1dCalcOutputShape(layer_t *conv1dLayer, shape_t *inputShape, shape_t *outputShape);

#endif // ODT_CONV1D_H
