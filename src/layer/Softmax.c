#define SOURCE_FILE "SOFTMAX"

#define EULER_APPROX = 2.71828

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "Common.h"
#include "Softmax.h"
#include "TensorConversion.h"

void softmaxInitConfig(softmaxConfig_t *softmaxConfig, quantization_t *forwardQ,
                       quantization_t *backwardQ) {
    softmaxConfig->forwardQ = forwardQ;
    softmaxConfig->backwardQ = backwardQ;
}

void softmaxInitLayer(layerConfig_t *softmaxConfig, layer_t *softmaxLayer) {
    softmaxLayer->type = SOFTMAX;
    softmaxLayer->config = softmaxConfig;
}

static void softmaxForwardFloat(tensor_t *input, tensor_t *output) {
    size_t n = calcNumberOfElementsByTensor(input);

    float *x = (float *)input->data;
    float *y = (float *)output->data;

    // 1. find max
    float max = x[0];
    for (size_t i = 1; i < n; i++) {
        if (x[i] > max) {
            max = x[i];
        }
    }

    // 2. exp and sum
    float sum = 0.f;
    for (size_t i = 0; i < n; i++) {
        float e = expf(x[i] - max);
        y[i] = e;
        sum += e;
    }

    // 3. normalize
    for (size_t i = 0; i < n; i++) {
        y[i] /= sum;
    }
}

/*static void softmaxForwardFloat(tensor_t *input, tensor_t *output) {
    size_t inputSize = calcNumberOfElementsByTensor(input);

    float *inputFloat = (float *)input->data;
    float *outputFloat = (float *)output->data;

    float sum = 0;
    for (size_t i = 0; i < inputSize; i++) {
        sum += expf(inputFloat[i]);
    }

    for (size_t i = 0; i < inputSize; i++) {
        outputFloat[i] = expf(inputFloat[i]) / sum;
    }
}*/

static void softmaxForwardSymInt32(tensor_t *input, tensor_t *output) {
    size_t inputSize = calcNumberOfElementsByTensor(input);

    tensor_t inputFloat;
    quantization_t inputFloatQ;
    initFloat32Quantization(&inputFloatQ);
    uint8_t inputFloatData[inputSize * sizeof(float)];
    setTensorValuesForConversion(inputFloatData, &inputFloatQ, input, &inputFloat);
    convertTensor(input, &inputFloat);

    tensor_t outputFloat;
    quantization_t outputFloatQ;
    initFloat32Quantization(&outputFloatQ);
    uint8_t outputFloatData[inputSize * sizeof(float)];
    setTensorValuesForConversion(outputFloatData, &outputFloatQ, output, &outputFloat);
    convertTensor(output, &outputFloat);

    float *inputFloatArr = (float *)inputFloat.data;
    float *outputFloatArr = (float *)outputFloat.data;

    float sum = 0;
    for (size_t i = 0; i < inputSize; i++) {
        sum += expf(inputFloatArr[i]);
    }

    for (size_t i = 0; i < inputSize; i++) {
        outputFloatArr[i] = expf(inputFloatArr[i]) / sum;
    }

    convertTensor(&outputFloat, output);
}

void softmaxForward(layer_t *softmaxLayer, tensor_t *input, tensor_t *output) {
    switch (softmaxLayer->config->softmax->forwardQ->type) {
    case FLOAT32:
        softmaxForwardFloat(input, output);
        break;
    case SYM_INT32:
        softmaxForwardSymInt32(input, output);
        break;
    default:
        PRINT_ERROR("Unknown QType!");
        exit(1);
    }
}

static void softmaxBackwardFloat(tensor_t *input, tensor_t *loss, tensor_t *propLoss) {
    size_t n = calcNumberOfElementsByTensor(input);

    float *s = (float *)input->data;
    float *dLds = (float *)loss->data;
    float *dLdx = (float *)propLoss->data;

    float dot = 0.0f;
    for (size_t i = 0; i < n; i++) {
        dot += s[i] * dLds[i];
    }

    for (size_t i = 0; i < n; i++) {
        dLdx[i] = s[i] * (dLds[i] - dot);
    }
}

static void softmaxBackwardSymInt32(tensor_t *input, tensor_t *loss, tensor_t *propLoss) {
    size_t inputSize = calcNumberOfElementsByTensor(input);

    tensor_t inputFloat;
    quantization_t inputFloatQ;
    initFloat32Quantization(&inputFloatQ);
    uint8_t inputFloatData[inputSize * sizeof(float)];
    setTensorValuesForConversion(inputFloatData, &inputFloatQ, input, &inputFloat);
    convertTensor(input, &inputFloat);

    tensor_t lossFloat;
    quantization_t lossFloatQ;
    initFloat32Quantization(&lossFloatQ);
    uint8_t lossFloatData[inputSize * sizeof(float)];
    setTensorValuesForConversion(lossFloatData, &lossFloatQ, loss, &lossFloat);
    convertTensor(loss, &lossFloat);

    tensor_t propLossFloat;
    quantization_t propLossFloatQ;
    initFloat32Quantization(&propLossFloatQ);
    uint8_t propLossFloatData[inputSize * sizeof(float)];
    setTensorValuesForConversion(propLossFloatData, &propLossFloatQ, propLoss, &propLossFloat);
    convertTensor(propLoss, &propLossFloat);

    float *s = (float *)inputFloat.data;
    float *dLds = (float *)lossFloat.data;
    float *dLdx = (float *)propLossFloat.data;

    float dot = 0.0f;
    for (size_t i = 0; i < inputSize; i++) {
        dot += s[i] * dLds[i];
    }

    for (size_t i = 0; i < inputSize; i++) {
        dLdx[i] = s[i] * (dLds[i] - dot);
    }

    convertTensor(&propLossFloat, propLoss);
}

void softmaxBackward(layer_t *softmaxLayer, tensor_t *input, tensor_t *loss, tensor_t *propLoss) {
    switch (softmaxLayer->config->softmax->backwardQ->type) {
    case FLOAT32:
        softmaxBackwardFloat(input, loss, propLoss);
        break;
    case SYM_INT32:
        softmaxBackwardSymInt32(input, loss, propLoss);
        break;
    default:
        PRINT_ERROR("Unknown QType!");
        exit(1);
    }
}

void softmaxCalcOutputShape(layer_t *softmaxLayer, shape_t *inputShape, shape_t *outputShape) {
    memcpy(outputShape->dimensions, inputShape->dimensions,
           inputShape->numberOfDimensions * sizeof(size_t));
    memcpy(outputShape->orderOfDimensions, inputShape->orderOfDimensions,
           inputShape->numberOfDimensions * sizeof(size_t));
    outputShape->numberOfDimensions = inputShape->numberOfDimensions;
}
