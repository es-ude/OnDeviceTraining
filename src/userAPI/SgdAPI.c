#include "Tensor.h"
#include "Layer.h"
#include "StorageAPI.h"
#include "SgdAPI.h"
#include "TensorAPI.h"

#include <stdio.h>

// IMPORTANT: Currently, the quantization for states are the same as the corresponding parameter
optimizer_t *sgdMCreateOptim(float learningRate, float momentumFactor, float weightDecay,
                             layer_t **model, size_t sizeModel, qtype_t qType) {
    optimizer_t *optim = *reserveMemory(sizeof(optimizer_t));
    optim->type = SGD_M;
    optim->qtype = qType;

    optimImpl_t *sgdImpl = *reserveMemory(sizeof(optimImpl_t));
    sgd_t *sgd = *reserveMemory(sizeof(sgd_t));
    sgdInit(sgd, learningRate, momentumFactor, weightDecay);
    sgdImpl->sgd = sgd;
    optim->impl = sgdImpl;

    size_t sizeStates = calcTotalNumberOfStates(model, sizeModel);
    optim->sizeStates = sizeStates;
    states_t **states = *reserveMemory(sizeStates * sizeof(states_t *));
    optim->states = states;
    parameter_t **parameter = *reserveMemory(sizeStates * sizeof(parameter_t *));
    optim->parameter = parameter;
    size_t statesPerParam = 1;

    for (size_t i = 0; i < sizeModel; i++) {
        layer_t *currentLayer = model[i];
        layerConfig_t *layerConfig = currentLayer->config;

        switch (currentLayer->type) {
        case LINEAR:
            linearConfig_t *linearConfig = layerConfig->linear;

            parameter_t *weights = linearConfig->weights;

            optim->parameter[i] = weights;
            tensor_t *weightStateBuffer = getTensorLike(weights->param);

            parameter_t *bias = linearConfig->bias;
            optim->parameter[i + 1] = bias;
            tensor_t *biasStateBuffer = getTensorLike(bias->param);

            states_t *weightStates = *reserveMemory(sizeof(states_t));
            weightStates->statesPerParameter = statesPerParam;
            weightStates->stateBuffers = *reserveMemory(sizeof(tensor_t));
            weightStates->stateBuffers[0] = weightStateBuffer;

            states_t *biasStates = *reserveMemory(sizeof(states_t));
            biasStates->statesPerParameter = statesPerParam;
            biasStates->stateBuffers = *reserveMemory(sizeof(tensor_t));
            biasStates->stateBuffers[0] = biasStateBuffer;

            states[i] = weightStates;
            states[i + 1] = biasStates;

            break;
        default:
            break;
        }
    }
    return optim;
}

void freeState(states_t *state) {
    for(size_t i = 0; i < state->statesPerParameter; i++) {
        freeTensor(state->stateBuffers[i]);
    }
    freeReservedMemory(state);
}

void freeOptimSgdM(optimizer_t *sgdM) {
    for(size_t i = 0; i < sgdM->sizeStates; i++) {
        freeParameter(sgdM->parameter[i]);
        freeState(sgdM->states[i]);
    }
    sgd_t *sgdImpl = sgdM->impl->sgd;
    freeReservedMemory(sgdImpl);
    freeReservedMemory(sgdM);
}
