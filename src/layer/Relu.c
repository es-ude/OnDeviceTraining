#include <string.h>

#include "Relu.h"
#include "Tensor.h"
#include "Comparison.h"
#include "DTypes.h"
#include "TensorConversion.h"
#include "Layer.h"

#include <stdio.h>

void reluInitConfig(reluConfig_t *reluConfig, quantization_t *forwardQ, quantization_t *backwardQ) {
    reluConfig->forwardQ = forwardQ;
    reluConfig->backwardQ = backwardQ;
}

void reluForwardFloat(tensor_t *input, tensor_t *output) {
    gteFloatValue(input, 0, 0, output);
}

void reluForwardSymInt32(tensor_t *input, tensor_t *output) {
    symInt32QConfig_t *inputSymInt32QC = input->quantization->qConfig;
    symInt32QConfig_t *outputSymInt32QC = output->quantization->qConfig;
    gteSymInt32Zero(input, 0, output);
    outputSymInt32QC->scale = inputSymInt32QC->scale;
}

void reluForward(layer_t *reluLayer, tensor_t *input, tensor_t *output) {
    reluConfig_t *reluConfig = reluLayer->config->relu;

    switch (reluConfig->forwardQ->type) {
    case FLOAT32:
        reluForwardFloat(input, output);
        break;
    case SYM_INT32:
        reluForwardSymInt32(input, output);
        break;
    default:
        break;
    }
}

void reluBackwardFloat(tensor_t *forwardInput, tensor_t *loss, tensor_t *propLoss) {
    size_t numberOfElements = calcNumberOfElementsByTensor(forwardInput);

    float *inputArray = (float *)forwardInput->data;
    float *gradOutArray = (float *)loss->data;
    float *gradInArray = (float *)propLoss->data;

    for (size_t i = 0; i < numberOfElements; i++) {
        if (inputArray[i] <= 0) {
            gradInArray[i] = 0;
        } else {
            gradInArray[i] = gradOutArray[i];
        }
    }
}

void reluBackwardSymInt32(tensor_t *forwardInput, tensor_t *loss, tensor_t *propLoss) {
    size_t numberOfElements = calcNumberOfElementsByTensor(forwardInput);

    int32_t *inputArray = (int32_t *)forwardInput->data;
    int32_t *gradOutputArray = (int32_t *)loss->data;
    int32_t *gradInputArray = (int32_t *)propLoss->data;

    for (size_t i = 0; i < numberOfElements; i++) {
        if (inputArray[i] <= 0) {
            gradInputArray[i] = 0;
        } else {
            gradInputArray[i] = gradOutputArray[i];
        }
    }

    symInt32QConfig_t *lossQC = loss->quantization->qConfig;
    symInt32QConfig_t *propLossQC = propLoss->quantization->qConfig;
    propLossQC->scale = lossQC->scale;
}

void reluBackward(layer_t *reluLayer, tensor_t *forwardInput, tensor_t *loss,
                  tensor_t *propLoss) {
    reluConfig_t *reluConfig = reluLayer->config->relu;

    switch (reluConfig->backwardQ->type) {
    case FLOAT32:
        reluBackwardFloat(forwardInput, loss, propLoss);
        break;
    case SYM_INT32:
        reluBackwardSymInt32(forwardInput, loss, propLoss);
        break;
    default:
        break;
    }
}

void reluCalcOutputShape(layer_t *reluLayer, shape_t *inputShape, shape_t *outputShape) {
    memcpy(outputShape->dimensions, inputShape->dimensions,
           inputShape->numberOfDimensions * sizeof(size_t));
    memcpy(outputShape->orderOfDimensions, inputShape->orderOfDimensions,
           inputShape->numberOfDimensions * sizeof(size_t));
    outputShape->numberOfDimensions = inputShape->numberOfDimensions;
}
