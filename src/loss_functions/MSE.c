#define SOURCE_FILE "MSE"

#include <Mul.h>
#include <stdio.h>
#include <stdlib.h>

#include "Common.h"
#include "MSE.h"
#include "Sub.h"
#include "Tensor.h"
#include "TensorConversion.h"

float mseLossForwardFloat(tensor_t *output, tensor_t *label, reduction_t reduction) {
    size_t size = calcNumberOfElementsByTensor(output);

    float *outputFloat = (float *)output->data;
    float *labelFloat = (float *)label->data;

    float sum = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        float delta = outputFloat[i] - labelFloat[i];
        sum += delta * delta;
    }

    if (reduction == REDUCTION_MEAN) {
        return sum / (float)size;
    }
    return sum;
}

float mseLossForwardSymInt32(tensor_t *output, tensor_t *label, reduction_t reduction) {
    size_t size = calcNumberOfElementsByTensor(output);

    tensor_t outputFloat;
    float outputFloatData[size];
    quantization_t outputFloatQ;
    initFloat32Quantization(&outputFloatQ);
    setTensorValuesForConversion((uint8_t *)outputFloatData, &outputFloatQ, output, &outputFloat);
    convertTensor(output, &outputFloat);

    tensor_t labelFloat;
    float labelFloatData[size];
    quantization_t labelFloatQ;
    initFloat32Quantization(&labelFloatQ);
    setTensorValuesForConversion((uint8_t *)labelFloatData, &labelFloatQ, label, &labelFloat);
    convertTensor(label, &labelFloat);

    float *outputFloatArr = (float *)outputFloat.data;
    float *labelFloatArr = (float *)labelFloat.data;

    float sum = 0.0f;

    for (size_t i = 0; i < size; ++i) {
        float delta = outputFloatArr[i] - labelFloatArr[i];
        sum += delta * delta;
    }

    if (reduction == REDUCTION_MEAN) {
        return sum / (float)size;
    }
    return sum;
}

float mseLossForward(tensor_t *output, tensor_t *label, reduction_t reduction) {
    switch (output->quantization->type) {
    case FLOAT32:
        return mseLossForwardFloat(output, label, reduction);
    case SYM_INT32:
        return mseLossForwardSymInt32(output, label, reduction);
    default:
        PRINT_ERROR("Unknown QType!");
        exit(1);
    }
}

void mseLossBackwardFloat(tensor_t *modelOutput, tensor_t *label, tensor_t *result) {
    size_t numberOfElements = calcNumberOfElementsByTensor(modelOutput);
    float *modelOutputArray = (float *)modelOutput->data;
    float *labelArray = (float *)label->data;
    float *resultArray = (float *)result->data;

    for (size_t i = 0; i < numberOfElements; i++) {
        resultArray[i] = 2.0f * (modelOutputArray[i] - labelArray[i]);
    }
}

void mseLossBackwardSymInt32(tensor_t *modelOutput, tensor_t *label, tensor_t *result) {
    size_t numberOfElements = calcNumberOfElementsByTensor(modelOutput);

    tensor_t modelOutputFloat;
    quantization_t modelOutputFloatQ;
    initFloat32Quantization(&modelOutputFloatQ);
    uint8_t modelOutputFloatData[numberOfElements * sizeof(float)];
    setTensorValuesForConversion(modelOutputFloatData, &modelOutputFloatQ, modelOutput,
                                 &modelOutputFloat);
    convertTensor(modelOutput, &modelOutputFloat);

    tensor_t labelFloat;
    quantization_t labelFloatQ;
    initFloat32Quantization(&labelFloatQ);
    uint8_t labelFloatData[numberOfElements * sizeof(float)];
    setTensorValuesForConversion(labelFloatData, &labelFloatQ, label, &labelFloat);
    convertTensor(label, &labelFloat);

    tensor_t resultFloat;
    quantization_t resultFloatQ;
    initFloat32Quantization(&resultFloatQ);
    uint8_t resultFloatData[numberOfElements * sizeof(float)];
    setTensorValuesForConversion(resultFloatData, &resultFloatQ, result, &resultFloat);

    float *modelOutputArray = (float *)modelOutputFloat.data;
    float *labelArray = (float *)labelFloat.data;
    float *resultArray = (float *)resultFloat.data;

    for (size_t i = 0; i < numberOfElements; i++) {
        resultArray[i] = 2.0f * (modelOutputArray[i] - labelArray[i]);
    }

    convertTensor(&resultFloat, result);
}

void mseLossBackward(tensor_t *modelOutput, tensor_t *label, tensor_t *result) {
    qtype_t modelOutputQType = modelOutput->quantization->type;

    switch (modelOutputQType) {
    case FLOAT32:
        mseLossBackwardFloat(modelOutput, label, result);
        break;
    case SYM_INT32:
        mseLossBackwardSymInt32(modelOutput, label, result);
        break;
    default:
        PRINT_ERROR("Unknown QType!");
        exit(1);
    }
}

float computeMeanScaleMSE(size_t totalSamples, tensor_t *modelOutput) {
    size_t microbatch = modelOutput->shape->dimensions[0];
    size_t numFeaturesPerSample = calcNumberOfElementsByTensor(modelOutput) / microbatch;
    return 1.0f / (float)(totalSamples * numFeaturesPerSample);
}
