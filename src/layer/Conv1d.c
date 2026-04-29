#define SOURCE_FILE "ODT_CONV1D"

#include <string.h>

#include "Add.h"
#include "Common.h"
#include "Conv1d.h"
#include "Layer.h"
#include "Mul.h"
#include "Tensor.h"

void initConv1dConfigWithWeightsAndBias(conv1dConfig_t *conv1dConfig, kernel_t *kernel,
                                        parameter_t *weights, parameter_t *bias,
                                        quantization_t *forwardQ, quantization_t *weightGradQ,
                                        quantization_t *biasGradQ, quantization_t *propLossQ) {
    conv1dConfig->kernel = kernel;
    conv1dConfig->weights = weights;
    conv1dConfig->bias = bias;

    conv1dConfig->forwardQ = forwardQ;
    conv1dConfig->weightGradQ = weightGradQ;
    conv1dConfig->biasGradQ = biasGradQ;
    conv1dConfig->propLossQ = propLossQ;
}

size_t calcOutputLengthPerChannel(size_t inputLengthPerChannel, kernel_t *kernel) {
    // effective kernel size = dilation * (kernelSize - 1) + 1
    size_t effectiveKernelSize = 0;
    size_t dilation = kernel->dilation;
    size_t kernelSize = kernel->size;

    effectiveKernelSize = dilation * (kernelSize - 1) + 1;
    size_t outputLengthPerChannel = 0;
    size_t stride = kernel->stride;

    while ((int)inputLengthPerChannel - (int)effectiveKernelSize >= 0) {
        outputLengthPerChannel += 1;
        inputLengthPerChannel -= stride;
    }

    return outputLengthPerChannel;
}

void conv1dForwardFloat(layer_t *conv1dLayer, tensor_t *input, tensor_t *output) {

    conv1dConfig_t *conv1dConfig = conv1dLayer->config->conv1d;
    kernel_t *kernel = conv1dConfig->kernel;
    parameter_t *weights = conv1dConfig->weights;
    parameter_t *bias = conv1dConfig->bias;

    if (input->shape->numberOfDimensions != 3) {
        PRINT_ERROR("Conv1d expects 3 dimensional input: [batch, channel, length]");
        exit(1);
    }

    if (output->shape->numberOfDimensions != 3) {
        PRINT_ERROR("Conv1d expects 3 dimensional output: [batch, channel, length]");
        exit(1);
    }

    size_t channel = input->shape->dimensions[1];
    size_t inputLengthPerChannel = input->shape->dimensions[2];

    size_t outputLengthPerChannel = 0;
    switch (kernel->paddingType) {
    case VALID:
        outputLengthPerChannel = calcOutputLengthPerChannel(inputLengthPerChannel, kernel);
        break;
    case SAME:
        outputLengthPerChannel = inputLengthPerChannel;
        break;
    }

    size_t kernelSize = kernel->size;

    float *x = (float *)input->data;
    float *w = (float *)weights->param->data;
    float *b;
    if (bias) {
        b = (float *)bias->param->data;
    }
    float *y = (float *)output->data;

    size_t paddingSize = kernelCalculatePaddingSize1d(inputLengthPerChannel, kernel);
    size_t padLeft = paddingSize / 2;

    for (size_t channelIndex = 0; channelIndex < channel; channelIndex++) {
        size_t outputBaseIndex = channelIndex * outputLengthPerChannel;

        for (size_t outputIndex = 0; outputIndex < outputLengthPerChannel; outputIndex++) {
            float sum = 0;

            size_t windowIndices[kernelSize];
            size_t inputStart = outputIndex * kernel->stride;
            kernelGetWindowIndices1d(kernel, inputStart, windowIndices);

            for (size_t kernelIndex = 0; kernelIndex < kernelSize; kernelIndex++) {
                size_t currentWindowIndex = windowIndices[kernelIndex];

                float inputValue = 0;
                int paddedIndex = (int)currentWindowIndex - padLeft;
                if (paddedIndex < 0 || paddedIndex >= inputLengthPerChannel) {
                    inputValue = 0;
                } else {
                    inputValue = x[paddedIndex];
                }

                sum = addFloat32s(sum, inputValue * w[kernelIndex]);
                PRINT_DEBUG("%f * %f\n", inputValue, w[kernelIndex]);
            }
            PRINT_DEBUG("________\n");

            if (bias) {
                sum = addFloat32s(sum, b[channelIndex]);
            }

            y[outputBaseIndex + outputIndex] = sum;
        }
    }
}

void conv1dForward(layer_t *conv1d, tensor_t *input, tensor_t *output) {

    conv1dConfig_t *conv1dConfig = conv1d->config->conv1d;

    switch (conv1dConfig->forwardQ->type) {
    case FLOAT32:
        conv1dForwardFloat(conv1d, input, output);
        break;
    default:
        PRINT_ERROR("Quantization not implemented!");
        exit(1);
    }
}

void conv1dCalcWeightGradsFloat32(kernel_t *kernel, tensor_t *input, tensor_t *lossGrad,
                                  parameter_t *weights) {
    if (input->shape->numberOfDimensions != 3) {
        PRINT_ERROR("Conv1d expects 3 dimensional input: [batch, channel, length]");
        exit(1);
    }

    if (lossGrad->shape->numberOfDimensions != 3) {
        PRINT_ERROR("Conv1d expects 3 dimensional loss gradient: [batch, channel, length]");
        exit(1);
    }

    size_t inputLengthPerChannel = input->shape->dimensions[2];
    size_t kernelSize = kernel->size;

    float *inputArr = (float *)input->data;
    float *lossGradArr = (float *)lossGrad->data;
    float *weightGradArr = (float *)weights->grad->data;

    size_t paddingSize = kernelCalculatePaddingSize1d(inputLengthPerChannel, kernel);
    size_t padLeft = paddingSize / 2;

    size_t outputSize = calcNumberOfElementsByTensor(lossGrad);

    for (size_t outputIndex = 0; outputIndex < outputSize; outputIndex++) {

        size_t windowIndices[kernelSize];
        kernelGetWindowIndices1d(kernel, outputIndex, windowIndices);

        for (size_t kernelIndex = 0; kernelIndex < kernelSize; kernelIndex++) {

            size_t currentWindowIndex = windowIndices[kernelIndex];

            int paddedIndex = (int)currentWindowIndex - padLeft;

            PRINT_DEBUG("paddedIndex: %i\n", paddedIndex);
            if (paddedIndex < 0 || paddedIndex >= inputLengthPerChannel) {
                continue;
            }

            float inputValue = inputArr[paddedIndex];

            weightGradArr[kernelIndex] = addFloat32s(
                weightGradArr[kernelIndex], mulFloat32s(inputValue, lossGradArr[outputIndex]));
        }
        PRINT_DEBUG("___________\n");
    }
}

void conv1dCalcPropLossFloat32(kernel_t *kernel, tensor_t *input, tensor_t *lossGrad,
                               parameter_t *weights, tensor_t *propLoss) {
    if (input->shape->numberOfDimensions != 3) {
        PRINT_ERROR("Conv1d expects 3 dimensional input: [batch, channel, length]");
        exit(1);
    }

    if (lossGrad->shape->numberOfDimensions != 3) {
        PRINT_ERROR("Conv1d expects 3 dimensional loss gradient: [batch, channel, length]");
        exit(1);
    }

    size_t inputLengthPerChannel = input->shape->dimensions[2];
    size_t kernelSize = kernel->size;

    float *propLossArr = (float *)propLoss->data;
    float *lossGradArr = (float *)lossGrad->data;
    float *weightArr = (float *)weights->param->data;

    size_t paddingSize = kernelCalculatePaddingSize1d(inputLengthPerChannel, kernel);
    size_t padLeft = paddingSize / 2;

    size_t outputSize = calcNumberOfElementsByTensor(lossGrad);

    for (size_t outputIndex = 0; outputIndex < outputSize; outputIndex++) {

        size_t windowIndices[kernelSize];
        kernelGetWindowIndices1d(kernel, outputIndex, windowIndices);

        for (size_t kernelIndex = 0; kernelIndex < kernelSize; kernelIndex++) {

            size_t currentWindowIndex = windowIndices[kernelIndex];

            int paddedIndex = (int)currentWindowIndex - padLeft;

            PRINT_DEBUG("paddedIndex: %i\n", paddedIndex);
            if (paddedIndex < 0 || paddedIndex >= inputLengthPerChannel) {
                continue;
            }

            propLossArr[paddedIndex] =
                addFloat32s(propLossArr[paddedIndex],
                            mulFloat32s(lossGradArr[outputIndex], weightArr[kernelIndex]));
        }
        PRINT_DEBUG("___________\n");
    }
}

void conv1dCalcBiasGradsFloat32(tensor_t *lossGrad, parameter_t *bias) {
    float *lossGradArr = (float *)lossGrad->data;
    float *biasGradArr = (float *)bias->grad->data;

    size_t outChannels = bias->param->shape->dimensions[0];
    size_t outputSize = calcNumberOfElementsByTensor(lossGrad);

    for (size_t biasIndex = 0; biasIndex < outChannels; biasIndex++) {
        for (size_t lossGradIndex = 0; lossGradIndex < outputSize; lossGradIndex++) {
            biasGradArr[biasIndex] =
                addFloat32s(biasGradArr[biasIndex], lossGradArr[lossGradIndex]);
        }
    }
}

void conv1dBackwardFloat(tensor_t *input, tensor_t *propLoss, tensor_t *lossGrad, kernel_t *kernel,
                         parameter_t *weights, parameter_t *bias) {

    conv1dCalcWeightGradsFloat32(kernel, input, lossGrad, weights);

    conv1dCalcPropLossFloat32(kernel, input, lossGrad, weights, propLoss);

    if (bias) {
        conv1dCalcBiasGradsFloat32(lossGrad, bias);
    }
}

void conv1dBackward(layer_t *conv1d, tensor_t *input, tensor_t *lossGrad, tensor_t *propLoss) {

    conv1dConfig_t *conv1dConfig = conv1d->config->conv1d;

    switch (conv1dConfig->weightGradQ->type) {
    case FLOAT32:
        conv1dCalcWeightGradsFloat32(conv1dConfig->kernel, input, lossGrad, conv1dConfig->weights);
        break;
    default:
        PRINT_ERROR("Weight Grad Quantization not implemented!");
        exit(1);
    }

    switch (conv1dConfig->propLossQ->type) {
    case FLOAT32:
        conv1dCalcPropLossFloat32(conv1dConfig->kernel, input, lossGrad, conv1dConfig->weights,
                                  propLoss);
        break;
    default:
        PRINT_ERROR("Prop Loss Quantization not implemented!");
        exit(1);
    }

    if (conv1dConfig->bias) {
        switch (conv1dConfig->biasGradQ->type) {
        case FLOAT32:
            conv1dCalcBiasGradsFloat32(lossGrad, conv1dConfig->bias);
            break;
        default:
            PRINT_ERROR("Weight Grad Quantization not implemented!");
            exit(1);
        }
    }
}

void conv1dCalcOutputShape(layer_t *conv1dLayer, shape_t *inputShape, shape_t *outputShape) {
    if (inputShape->numberOfDimensions != 3) {
        PRINT_ERROR("Conv1d layer expects 3D input [batch, channel, length], got %luD\n",
                    inputShape->numberOfDimensions);
    }

    conv1dConfig_t *cfg = conv1dLayer->config->conv1d;
    kernel_t *kernel = cfg->kernel;

    size_t batchSize = inputShape->dimensions[0];
    size_t inputLength = inputShape->dimensions[2];

    // out_channels is the first dimension of the weight tensor: [out_ch, in_ch, kernel_size]
    size_t outChannels = cfg->weights->param->shape->dimensions[0];

    size_t outputLength;
    switch (kernel->paddingType) {
    case VALID:
        outputLength = calcOutputLengthPerChannel(inputLength, kernel);
        break;
    case SAME:
        outputLength = inputLength;
        break;
    default:
        PRINT_ERROR("Unknown padding type\n");
        exit(1);
    }

    outputShape->dimensions[0] = batchSize;
    outputShape->dimensions[1] = outChannels;
    outputShape->dimensions[2] = outputLength;

    outputShape->numberOfDimensions = inputShape->numberOfDimensions;

    setOrderOfDimsForNewTensor(inputShape->numberOfDimensions, outputShape->orderOfDimensions);
}
