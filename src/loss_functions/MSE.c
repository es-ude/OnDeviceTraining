#include <Mul.h>
#include <stdio.h>

#include "MSE.h"
#include "Tensor.h"
#include "TensorConversion.h"
#include "Sub.h"

float mseLossForwardFloat(tensor_t *output, tensor_t *label) {
    size_t size = calcNumberOfElementsByTensor(output);

    float *outputFloat = (float *)output->data;
    float *labelFloat = (float *)label->data;

    float sum = 0.0f;

    for (size_t i = 0; i < size; ++i) {
        float delta = outputFloat[i] - labelFloat[i];
        sum += delta * delta;
    }

    return sum / (float)size;
}

float mseLossForwardAsym(tensor_t *output, tensor_t *label) {
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

    return sum / (float)size;
}

float mseLossForward(tensor_t *output, tensor_t *label) {
    switch (output->quantization->type) {
    case FLOAT32:
        return mseLossForwardFloat(output, label);
    case ASYM:
        return mseLossForwardAsym(output, label);
    default:
        // TODO this is shit
        return 0.f;
    }
}

void mseLossBackwardFloat(tensor_t *modelOutput, tensor_t *label, tensor_t *result) {
    size_t numberOfElements = calcNumberOfElementsByTensor(modelOutput);

    float mean = 2.f / (float)numberOfElements;

    float *modelOutputArray = (float *)modelOutput->data;
    float *labelArray = (float *)label->data;
    float *resultArray = (float *)result->data;

    for (size_t i = 0; i < numberOfElements; i++) {
        resultArray[i] = mulFloat32s(mean, subFloat32s(modelOutputArray[i], labelArray[i]));
    }
}

void mseLossBackwardAsym(tensor_t *modelOutput, tensor_t *label, tensor_t *result) {
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
    convertTensor(result, &resultFloat);

    float *modelOutputArray = (float *)modelOutputFloat.data;
    float *labelArray = (float *)labelFloat.data;
    float *resultArray = (float *)resultFloat.data;

    float mean = 2.f / (float)numberOfElements;

    for (size_t i = 0; i < numberOfElements; i++) {
        resultArray[i] = mulFloat32s(mean, subFloat32s(modelOutputArray[i], labelArray[i]));
    }

    convertTensor(&resultFloat, result);
}

void mseLossBackward(tensor_t *modelOutput, tensor_t *label, tensor_t *result) {
    qtype_t modelOutputQType = modelOutput->quantization->type;

    switch (modelOutputQType) {
    case FLOAT32:
        mseLossBackwardFloat(modelOutput, label, result);
        break;
    case ASYM:
        mseLossBackwardAsym(modelOutput, label, result);
        break;
    default:
        printf("Error in MSE Backward: qtype not supported\n");
        break;
    }
}
