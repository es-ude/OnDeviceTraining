#define SOURCE_FILE "SGD_API"

#include <stdio.h>
#include <stdlib.h>

#include "Common.h"
#include "Conv1d.h"
#include "Conv1dTransposed.h"
#include "Layer.h"
#include "LayerNorm.h"
#include "Linear.h"
#include "SgdApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"

// IMPORTANT: Currently, the quantization for states are the same as the corresponding parameter
optimizer_t *sgdMCreateOptim(float learningRate, float momentumFactor, float weightDecay,
                             layer_t **model, size_t sizeModel, qtype_t qType) {
    optimizer_t *optim = reserveMemory(sizeof(optimizer_t));
    optim->type = SGD_M;
    optim->qtype = qType;

    optimImpl_t *sgdImpl = reserveMemory(sizeof(optimImpl_t));
    sgd_t *sgd = reserveMemory(sizeof(sgd_t));
    sgdInit(sgd, learningRate, momentumFactor, weightDecay);
    sgdImpl->sgd = sgd;
    optim->impl = sgdImpl;

    size_t sizeStates = calcTotalNumberOfStates(model, sizeModel);
    optim->sizeStates = sizeStates;
    states_t **states = reserveMemory(sizeStates * sizeof(states_t *));
    optim->states = states;
    parameter_t **parameter = reserveMemory(sizeStates * sizeof(parameter_t *));
    optim->parameter = parameter;
    size_t statesPerParam = 1;

    size_t paramSlot = 0;
    for (size_t i = 0; i < sizeModel; i++) {
        layer_t *currentLayer = model[i];
        layerConfig_t *layerConfig = currentLayer->config;

        switch (currentLayer->type) {
        case LINEAR:
            linearConfig_t *linearConfig = layerConfig->linear;

            parameter_t *weights = linearConfig->weights;

            optim->parameter[paramSlot] = weights;
            tensor_t *weightStateBuffer = getTensorLike(weights->param);

            parameter_t *bias = linearConfig->bias;
            optim->parameter[paramSlot + 1] = bias;
            tensor_t *biasStateBuffer = getTensorLike(bias->param);

            states_t *weightStates = reserveMemory(sizeof(states_t));
            weightStates->statesPerParameter = statesPerParam;
            weightStates->stateBuffers = reserveMemory(sizeof(tensor_t *));
            weightStates->stateBuffers[0] = weightStateBuffer;

            states_t *biasStates = reserveMemory(sizeof(states_t));
            biasStates->statesPerParameter = statesPerParam;
            biasStates->stateBuffers = reserveMemory(sizeof(tensor_t *));
            biasStates->stateBuffers[0] = biasStateBuffer;

            states[paramSlot] = weightStates;
            states[paramSlot + 1] = biasStates;

            paramSlot += 2;
            break;
        case CONV1D: {
            conv1dConfig_t *conv1dCfg = layerConfig->conv1d;

            parameter_t *cWeights = conv1dCfg->weights;
            optim->parameter[paramSlot] = cWeights;
            tensor_t *cWeightStateBuffer = getTensorLike(cWeights->param);

            parameter_t *cBias = conv1dCfg->bias;
            optim->parameter[paramSlot + 1] = cBias;
            tensor_t *cBiasStateBuffer = getTensorLike(cBias->param);

            states_t *cWeightStates = reserveMemory(sizeof(states_t));
            cWeightStates->statesPerParameter = statesPerParam;
            cWeightStates->stateBuffers = reserveMemory(sizeof(tensor_t *));
            cWeightStates->stateBuffers[0] = cWeightStateBuffer;

            states_t *cBiasStates = reserveMemory(sizeof(states_t));
            cBiasStates->statesPerParameter = statesPerParam;
            cBiasStates->stateBuffers = reserveMemory(sizeof(tensor_t *));
            cBiasStates->stateBuffers[0] = cBiasStateBuffer;

            states[paramSlot] = cWeightStates;
            states[paramSlot + 1] = cBiasStates;

            paramSlot += 2;
            break;
        }
        case CONV1D_TRANSPOSED: {
            conv1dTransposedConfig_t *ctCfg = layerConfig->conv1dTransposed;

            parameter_t *ctWeights = ctCfg->weights;
            optim->parameter[paramSlot] = ctWeights;
            tensor_t *ctWeightStateBuffer = getTensorLike(ctWeights->param);

            parameter_t *ctBias = ctCfg->bias;
            optim->parameter[paramSlot + 1] = ctBias;
            tensor_t *ctBiasStateBuffer = getTensorLike(ctBias->param);

            states_t *ctWeightStates = reserveMemory(sizeof(states_t));
            ctWeightStates->statesPerParameter = statesPerParam;
            ctWeightStates->stateBuffers = reserveMemory(sizeof(tensor_t *));
            ctWeightStates->stateBuffers[0] = ctWeightStateBuffer;

            states_t *ctBiasStates = reserveMemory(sizeof(states_t));
            ctBiasStates->statesPerParameter = statesPerParam;
            ctBiasStates->stateBuffers = reserveMemory(sizeof(tensor_t *));
            ctBiasStates->stateBuffers[0] = ctBiasStateBuffer;

            states[paramSlot] = ctWeightStates;
            states[paramSlot + 1] = ctBiasStates;

            paramSlot += 2;
            break;
        }
        case LAYERNORM: {
            layerNormConfig_t *lnCfg = layerConfig->layerNorm;

            parameter_t *lnGamma = lnCfg->gamma;
            optim->parameter[paramSlot] = lnGamma;
            tensor_t *lnGammaStateBuffer = getTensorLike(lnGamma->param);

            parameter_t *lnBeta = lnCfg->beta;
            optim->parameter[paramSlot + 1] = lnBeta;
            tensor_t *lnBetaStateBuffer = getTensorLike(lnBeta->param);

            states_t *lnGammaStates = reserveMemory(sizeof(states_t));
            lnGammaStates->statesPerParameter = statesPerParam;
            lnGammaStates->stateBuffers = reserveMemory(sizeof(tensor_t *));
            lnGammaStates->stateBuffers[0] = lnGammaStateBuffer;

            states_t *lnBetaStates = reserveMemory(sizeof(states_t));
            lnBetaStates->statesPerParameter = statesPerParam;
            lnBetaStates->stateBuffers = reserveMemory(sizeof(tensor_t *));
            lnBetaStates->stateBuffers[0] = lnBetaStateBuffer;

            states[paramSlot] = lnGammaStates;
            states[paramSlot + 1] = lnBetaStates;

            paramSlot += 2;
            break;
        }
        case RELU:
        case SOFTMAX:
        case FLATTEN:
        case MAXPOOL1D:
        case AVGPOOL1D:
        case ADAPTIVE_AVGPOOL1D:
        case DROPOUT:
        case QUANTIZATION:
            break;
        default:
            PRINT_ERROR("Unknown Layer Type");
            exit(1);
        }
    }
    return optim;
}

void freeState(states_t *state) {
    for (size_t i = 0; i < state->statesPerParameter; i++) {
        freeTensor(state->stateBuffers[i]);
    }
    freeReservedMemory(state->stateBuffers);
    freeReservedMemory(state);
}

void freeOptimSgdM(optimizer_t *sgdM) {
    for (size_t i = 0; i < sgdM->sizeStates; i++) {
        freeParameter(sgdM->parameter[i]);
        freeState(sgdM->states[i]);
    }
    freeReservedMemory(sgdM->parameter);
    freeReservedMemory(sgdM->states);
    sgd_t *sgdImpl = sgdM->impl->sgd;
    freeReservedMemory(sgdImpl);
    freeReservedMemory(sgdM->impl);
    freeReservedMemory(sgdM);
}
