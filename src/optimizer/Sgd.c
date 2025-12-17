#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <tgmath.h>

#include "Tensor.h"
#include "Layer.h"
#include "Linear.h"
#include "TensorConversion.h"
#include "Sgd.h"

void sgdInit(sgd_t *sgd, float learningRate, float momentumFactor, float weightDecay) {
    sgd->learningRate = learningRate;
    sgd->momentumFactor = momentumFactor;
    sgd->weightDecay = weightDecay;
}

static void sgdStepFloat(optimizer_t *optim) {
    sgd_t *sgd = (sgd_t *)optim->impl;

    for (size_t stateIndex = 0; stateIndex < optim->sizeStates; stateIndex++) {
        parameter_t *param = optim->parameter[stateIndex];
        size_t numberOfValues = calcNumberOfElementsByParameter(param);
        float *gradArr = (float *)param->grad->data;
        float *dataArr = (float *)param->param->data;

        for (size_t elementIndex = 0; elementIndex < numberOfValues; ++elementIndex) {
            float grad = gradArr[elementIndex] + sgd->weightDecay * dataArr[elementIndex];
            dataArr[elementIndex] -= sgd->learningRate * grad;
        }
    }
}

static void sgdStepSymInt32(optimizer_t *optim) {
    sgd_t *sgd = (sgd_t *)optim->impl;

    for (size_t stateIndex = 0; stateIndex < optim->sizeStates; stateIndex++) {
        parameter_t *param = optim->parameter[stateIndex];
        size_t numberOfValues = calcNumberOfElementsByParameter(param);

        tensor_t paramFloat;
        quantization_t paramFloatQ;
        initFloat32Quantization(&paramFloatQ);
        uint8_t paramFloatData[numberOfValues * sizeof(float)];
        setTensorValuesForConversion(paramFloatData, &paramFloatQ, param->param, &paramFloat);
        convertTensor(param->param, &paramFloat);

        float *paramFloatArr = (float *)paramFloat.data;

        tensor_t gradFloat;
        quantization_t gradFloatQ;
        initFloat32Quantization(&gradFloatQ);
        float gradFloatData[numberOfValues];
        uint8_t *gradFloatDataBytes = (uint8_t *)gradFloatData;
        setTensorValuesForConversion(gradFloatDataBytes, &gradFloatQ, param->grad, &gradFloat);
        convertTensor(param->grad, &gradFloat);

        float *gradFloatArr = (float *)gradFloat.data;

        for (size_t j = 0; j < numberOfValues; ++j) {
            float grad = gradFloatArr[j] + sgd->weightDecay * paramFloatArr[j];
            paramFloatArr[j] -= sgd->learningRate * grad;
        }

        convertTensor(&paramFloat, param->param);
        convertTensor(&gradFloat, param->grad);
    }
}

void sgdStep(optimizer_t *optimizer) {
    switch (optimizer->qtype) {
    case FLOAT32:
        sgdStepFloat(optimizer);
        break;
    case SYM_INT32:
        sgdStepSymInt32(optimizer);
        break;
    default:
        break;
    }
}

static void sgdStepMFloat(optimizer_t *optim) {
    sgd_t *sgd = optim->impl->sgd;
    for (size_t i = 0; i < optim->sizeStates; i++) {
        parameter_t *param = optim->parameter[i];
        size_t numberOfValues = calcNumberOfElementsByParameter(param);
        float *gradArr = (float *)param->grad->data;
        float *dataArr = (float *)param->param->data;

        states_t *states = optim->states[i];
        tensor_t *state = states->stateBuffers[0];

        float *stateArr = (float *)state->data;

        for (size_t elementIndex = 0; elementIndex < numberOfValues; ++elementIndex) {
            float grad = gradArr[elementIndex] + sgd->weightDecay * dataArr[elementIndex];
            stateArr[elementIndex] = sgd->momentumFactor * stateArr[elementIndex] + grad;
            dataArr[elementIndex] -= sgd->learningRate * stateArr[elementIndex];
        }
    }
}

static void sgdStepMSymInt32(optimizer_t *optim) {
    sgd_t *sgd = optim->impl->sgd;

    for (size_t i = 0; i < optim->sizeStates; i++) {
        parameter_t *param = optim->parameter[i];
        size_t numberOfValues = calcNumberOfElementsByParameter(param);

        tensor_t paramFloat;
        quantization_t paramFloatQ;
        initFloat32Quantization(&paramFloatQ);
        uint8_t paramFloatData[numberOfValues * sizeof(float)];
        setTensorValuesForConversion(paramFloatData, &paramFloatQ, param->param, &paramFloat);
        convertTensor(param->param, &paramFloat);

        float *paramFloatArr = (float *)paramFloat.data;

        tensor_t gradFloat;
        quantization_t gradFloatQ;
        initFloat32Quantization(&gradFloatQ);
        float gradFloatData[numberOfValues];
        uint8_t *gradFloatDataBytes = (uint8_t *)gradFloatData;
        setTensorValuesForConversion(gradFloatDataBytes, &gradFloatQ, param->grad, &gradFloat);
        convertTensor(param->grad, &gradFloat);

        float *gradFloatArr = (float *)gradFloat.data;

        states_t *states = optim->states[i];

        tensor_t *state = states->stateBuffers[0];

        tensor_t stateFloat;
        quantization_t stateFloatQ;
        initFloat32Quantization(&stateFloatQ);
        uint8_t stateFloatData[numberOfValues * sizeof(float)];
        setTensorValuesForConversion(stateFloatData, &stateFloatQ, state,
                                     &stateFloat);
        convertTensor(state, &stateFloat);


        float *stateFloatArr = (float *)stateFloat.data;

        for (size_t j = 0; j < numberOfValues; ++j) {
            float grad = gradFloatArr[j] + sgd->weightDecay * paramFloatArr[j];
            stateFloatArr[j] = sgd->momentumFactor * stateFloatArr[j] + grad;
            paramFloatArr[j] -= sgd->learningRate * stateFloatArr[j];
        }

        convertTensor(&stateFloat, state);
        convertTensor(&paramFloat, param->param);
    }
}

void sgdStepM(optimizer_t *optimizer) {
    switch (optimizer->qtype) {
    case FLOAT32:
        sgdStepMFloat(optimizer);
        break;
    case SYM_INT32:
        sgdStepMSymInt32(optimizer);
        break;
    default:
        break;
    }
}

void sgdZeroGrad(optimizer_t *optimizer) {
    for (size_t i = 0; i < optimizer->sizeStates; i++) {
        parameter_t *param = optimizer->parameter[i];
        size_t paramSize = calcNumberOfElementsByParameter(param);
        size_t bitsPerElement = calcBitsPerElement(param->grad->quantization);
        size_t totalNumberOfBytes = ceil(paramSize * bitsPerElement / 8);

        memset(param->grad->data, 0, totalNumberOfBytes);

        if(param->grad->quantization->type == SYM_INT32) {
            symInt32QConfig_t *symIntQ = param->grad->quantization->qConfig;
            symIntQ->scale = 0.f;
        }
    }
}
