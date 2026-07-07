#define SOURCE_FILE "SGD_API"

#include <stdio.h>
#include <stdlib.h>

#include "Common.h"
#include "Conv1d.h"
#include "Conv1dTransposed.h"
#include "GroupNorm.h"
#include "Layer.h"
#include "LayerNorm.h"
#include "Linear.h"
#include "SgdApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"

/*! Builds a momentum-state tensor at `param`'s shape but with its OWN
 * quantization (`momentumQuant`, deep-cloned via getQLike) -- decouples the
 * accumulator's dtype from the parameter's storage dtype (#277 Task 2). */
static tensor_t *momentumStateInit(tensor_t *param, quantization_t *momentumQuant) {
    return initTensor(getShapeLike(param->shape), getQLike(momentumQuant), NULL);
}

optimizer_t *sgdMCreateOptim(float learningRate, float momentumFactor, float weightDecay,
                             layer_t **model, size_t sizeModel, qtype_t qType,
                             quantization_t *momentumQuant) {
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
            tensor_t *weightStateBuffer = momentumStateInit(weights->param, momentumQuant);

            states_t *weightStates = reserveMemory(sizeof(states_t));
            weightStates->statesPerParameter = statesPerParam;
            weightStates->stateBuffers = reserveMemory(sizeof(tensor_t *));
            weightStates->stateBuffers[0] = weightStateBuffer;

            states[paramSlot] = weightStates;

            /* BIAS_FALSE (header-sanctioned): no bias parameter to collect. */
            if (linearConfig->bias != NULL) {
                parameter_t *bias = linearConfig->bias;
                optim->parameter[paramSlot + 1] = bias;
                tensor_t *biasStateBuffer = momentumStateInit(bias->param, momentumQuant);

                states_t *biasStates = reserveMemory(sizeof(states_t));
                biasStates->statesPerParameter = statesPerParam;
                biasStates->stateBuffers = reserveMemory(sizeof(tensor_t *));
                biasStates->stateBuffers[0] = biasStateBuffer;

                states[paramSlot + 1] = biasStates;

                paramSlot += 2;
            } else {
                paramSlot += 1;
            }
            break;
        case CONV1D: {
            conv1dConfig_t *conv1dCfg = layerConfig->conv1d;

            parameter_t *cWeights = conv1dCfg->weights;
            optim->parameter[paramSlot] = cWeights;
            tensor_t *cWeightStateBuffer = momentumStateInit(cWeights->param, momentumQuant);

            states_t *cWeightStates = reserveMemory(sizeof(states_t));
            cWeightStates->statesPerParameter = statesPerParam;
            cWeightStates->stateBuffers = reserveMemory(sizeof(tensor_t *));
            cWeightStates->stateBuffers[0] = cWeightStateBuffer;

            states[paramSlot] = cWeightStates;

            /* BIAS_FALSE (header-sanctioned): no bias parameter to collect. */
            if (conv1dCfg->bias != NULL) {
                parameter_t *cBias = conv1dCfg->bias;
                optim->parameter[paramSlot + 1] = cBias;
                tensor_t *cBiasStateBuffer = momentumStateInit(cBias->param, momentumQuant);

                states_t *cBiasStates = reserveMemory(sizeof(states_t));
                cBiasStates->statesPerParameter = statesPerParam;
                cBiasStates->stateBuffers = reserveMemory(sizeof(tensor_t *));
                cBiasStates->stateBuffers[0] = cBiasStateBuffer;

                states[paramSlot + 1] = cBiasStates;

                paramSlot += 2;
            } else {
                paramSlot += 1;
            }
            break;
        }
        case CONV1D_TRANSPOSED: {
            conv1dTransposedConfig_t *ctCfg = layerConfig->conv1dTransposed;

            parameter_t *ctWeights = ctCfg->weights;
            optim->parameter[paramSlot] = ctWeights;
            tensor_t *ctWeightStateBuffer = momentumStateInit(ctWeights->param, momentumQuant);

            states_t *ctWeightStates = reserveMemory(sizeof(states_t));
            ctWeightStates->statesPerParameter = statesPerParam;
            ctWeightStates->stateBuffers = reserveMemory(sizeof(tensor_t *));
            ctWeightStates->stateBuffers[0] = ctWeightStateBuffer;

            states[paramSlot] = ctWeightStates;

            /* BIAS_FALSE (header-sanctioned): no bias parameter to collect. */
            if (ctCfg->bias != NULL) {
                parameter_t *ctBias = ctCfg->bias;
                optim->parameter[paramSlot + 1] = ctBias;
                tensor_t *ctBiasStateBuffer = momentumStateInit(ctBias->param, momentumQuant);

                states_t *ctBiasStates = reserveMemory(sizeof(states_t));
                ctBiasStates->statesPerParameter = statesPerParam;
                ctBiasStates->stateBuffers = reserveMemory(sizeof(tensor_t *));
                ctBiasStates->stateBuffers[0] = ctBiasStateBuffer;

                states[paramSlot + 1] = ctBiasStates;

                paramSlot += 2;
            } else {
                paramSlot += 1;
            }
            break;
        }
        case LAYERNORM: {
            layerNormConfig_t *lnCfg = layerConfig->layerNorm;

            parameter_t *lnGamma = lnCfg->gamma;
            optim->parameter[paramSlot] = lnGamma;
            tensor_t *lnGammaStateBuffer = momentumStateInit(lnGamma->param, momentumQuant);

            parameter_t *lnBeta = lnCfg->beta;
            optim->parameter[paramSlot + 1] = lnBeta;
            tensor_t *lnBetaStateBuffer = momentumStateInit(lnBeta->param, momentumQuant);

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
        case GROUPNORM: {
            groupNormConfig_t *gnCfg = layerConfig->groupNorm;

            parameter_t *gnGamma = gnCfg->gamma;
            optim->parameter[paramSlot] = gnGamma;
            tensor_t *gnGammaStateBuffer = momentumStateInit(gnGamma->param, momentumQuant);

            parameter_t *gnBeta = gnCfg->beta;
            optim->parameter[paramSlot + 1] = gnBeta;
            tensor_t *gnBetaStateBuffer = momentumStateInit(gnBeta->param, momentumQuant);

            states_t *gnGammaStates = reserveMemory(sizeof(states_t));
            gnGammaStates->statesPerParameter = statesPerParam;
            gnGammaStates->stateBuffers = reserveMemory(sizeof(tensor_t *));
            gnGammaStates->stateBuffers[0] = gnGammaStateBuffer;

            states_t *gnBetaStates = reserveMemory(sizeof(states_t));
            gnBetaStates->statesPerParameter = statesPerParam;
            gnBetaStates->stateBuffers = reserveMemory(sizeof(tensor_t *));
            gnBetaStates->stateBuffers[0] = gnBetaStateBuffer;

            states[paramSlot] = gnGammaStates;
            states[paramSlot + 1] = gnBetaStates;

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
    /* #261, PR3: grads may be stored FLOAT32 (default), SYM_INT32 (explicit
     * low-level knob), or packed SYM/ASYM (explicit grad-storage knob,
     * memory-constrained targets). INT32/BOOL grad storage remains
     * unimplemented - fail fast rather than silently misread bytes in an
     * unsupported layout. Non-trainable params carry no grad: skip. */
    for (size_t s = 0; s < optim->sizeStates; s++) {
        tensor_t *grad = optim->parameter[s]->grad;
        if (grad == NULL) {
            continue;
        }
        qtype_t gradType = grad->quantization->type;
        if (gradType != FLOAT32 && gradType != SYM_INT32 && gradType != SYM && gradType != ASYM) {
            PRINT_ERROR("sgdMCreateOptim: gradient storage dtype %d not supported "
                        "(accepted: FLOAT32, SYM_INT32, SYM, ASYM; INT32/BOOL grad "
                        "storage remains unsupported, #261)",
                        (int)gradType);
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
