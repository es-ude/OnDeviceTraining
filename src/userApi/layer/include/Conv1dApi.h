#ifndef CONV1DAPI_H
#define CONV1DAPI_H

#include "Kernel.h"
#include "Layer.h"

/*! Initializes a 1D convolution layer with given parameters.
 *
 * @param weights Weights with gradients
 * @param bias Optional bias parameter with gradients
 * @param kernel Kernel to be used for convolution
 * @param forwardQ Quantization for forward pass
 * @param weightGradQ Quantization for weight gradient calculation
 * @param biasGradQ Quantization for bias gradient calculation
 * @param propLossQ Quantization for prop loss calculation
 *
 * @returns Pointer to initializes layer_t
 */
layer_t *conv1dLayerInit(parameter_t *weights, parameter_t *bias, kernel_t *kernel,
                         quantization_t *forwardQ, quantization_t *weightGradQ,
                         quantization_t *biasGradQ, quantization_t *propLossQ);

/*! Frees 1D convolutional layer and all contained data structures recursively
 *
 * @param conv1dLayer Pointer to layer_t
 */
void freeConv1dLayer(layer_t *conv1dLayer);

#endif // CONV1DAPI_H
