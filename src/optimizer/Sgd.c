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

typedef struct numParams {
    size_t numParams;
    parameter_t *parameters;
} numParams_t;

void initMomentumBuffer(momentumBuffer_t *momentumBuffer, parameter_t *parameter,
                        tensor_t *momentums) {
    momentumBuffer->parameter = parameter;
    momentumBuffer->momentums = momentums;
}

uint32_t calcNumberOfMomentumBuffersByLayerType(const layerType_t type) {
    switch (type) {
    case LINEAR:
        return 2;
    case RELU:
        return 0;
    case CONV1D:
        return 2;
    default:
        return 0;
    }
}

uint32_t calcTotalNumberOfMomentumBuffers(layer_t *model, size_t sizeModel) {
    uint32_t number = 0;
    for (size_t i = 0; i < sizeModel; i++) {
        number += calcNumberOfMomentumBuffersByLayerType(model[i].type);
    }
    return number;
}

void initSGDConfig(sgd_t *sgd, float learningRate, float momentumFactor, float weightDecay,
                   momentumBuffer_t **momentumBuffers, size_t sizeMomentumBuffers) {
    sgd->learningRate = learningRate;
    sgd->momentumFactor = momentumFactor;
    sgd->weightDecay = weightDecay;
    sgd->momentumBuffers = momentumBuffers;
    sgd->sizeMomentumBuffers = sizeMomentumBuffers;
}


void SGDStepFloat(sgd_t *sgd) {
    for (size_t i = 0; i < sgd->sizeMomentumBuffers; i++) {
        parameter_t *param = sgd->momentumBuffers[i]->parameter;
        float *momentums = (float *)sgd->momentumBuffers[i]->momentums->data;

        size_t paramSize = calcNumberOfElementsByParameter(param);
        float *gradFloat = (float *)param->grad->data;
        float *dataFloat = (float *)param->param->data;

        for (size_t j = 0; j < paramSize; ++j) {
            float grad = gradFloat[j] + sgd->weightDecay * dataFloat[j];
            momentums[j] = sgd->momentumFactor * momentums[j] + grad;
            dataFloat[j] -= sgd->learningRate * momentums[j];
        }
    }
}

void SGDStepAsym(sgd_t *sgd) {
    for (size_t i = 0; i < sgd->sizeMomentumBuffers; i++) {
        parameter_t *param = sgd->momentumBuffers[i]->parameter;
        tensor_t *momentums = sgd->momentumBuffers[i]->momentums;

        size_t numberOfValues = calcNumberOfElementsByShape(momentums->shape);

        tensor_t momentumsFloat;
        quantization_t momentumsFloatQ;
        initFloat32Quantization(&momentumsFloatQ);
        uint8_t momentumsFloatData[numberOfValues * sizeof(float)];
        setTensorValuesForConversion(momentumsFloatData, &momentumsFloatQ, momentums,
                                     &momentumsFloat);
        convertTensor(momentums, &momentumsFloat);

        tensor_t paramFloat;
        quantization_t paramFloatQ;
        initFloat32Quantization(&paramFloatQ);
        uint8_t paramFloatData[numberOfValues * sizeof(float)];
        setTensorValuesForConversion(paramFloatData, &paramFloatQ, param->param, &paramFloat);
        convertTensor(param->param, &paramFloat);

        tensor_t gradFloat;
        quantization_t gradFloatQ;
        initFloat32Quantization(&gradFloatQ);
        float gradFloatData[numberOfValues];
        uint8_t *gradFloatDataBytes = (uint8_t *)gradFloatData;
        setTensorValuesForConversion(gradFloatDataBytes, &gradFloatQ, param->grad, &gradFloat);
        convertTensor(param->grad, &gradFloat);

        float *momentumsFloatArr = (float *)momentumsFloat.data;
        float *paramFloatArr = (float *)paramFloat.data;
        float *gradFloatArr = (float *)gradFloat.data;

        for (size_t j = 0; j < numberOfValues; ++j) {
            float grad = gradFloatArr[j] + sgd->weightDecay * paramFloatArr[j];
            momentumsFloatArr[j] = sgd->momentumFactor * momentumsFloatArr[j] + grad;
            paramFloatArr[j] -= sgd->learningRate * momentumsFloatArr[j];
        }

        convertTensor(&momentumsFloat, momentums);
        convertTensor(&paramFloat, param->param);
        convertTensor(&gradFloat, param->grad);

    }
}

void SGDZeroGrad(sgd_t *sgd) {
    for (size_t i = 0; i < sgd->sizeMomentumBuffers; i++) {
        parameter_t *param = sgd->momentumBuffers[i]->parameter;
        size_t paramSize = calcNumberOfElementsByParameter(param);
        size_t bitsPerElement = calcBitsPerElement(param->grad->quantization);
        size_t totalNumberOfBytes = ceil(paramSize * bitsPerElement / 8);

        memset(param->grad->data, 0, totalNumberOfBytes);
        qtype_t currentQType = param->grad->quantization->type;
        if (currentQType == ASYM) {
            asymQConfig_t *currentAsymQC = param->grad->quantization->qConfig;
            currentAsymQC->zeroPoint = 0;
        }
    }
}
