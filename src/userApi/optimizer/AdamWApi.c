#define SOURCE_FILE "ADAM-W-API"

#include "AdamWApi.h"
#include "AdamW.h"
#include "OptimizerApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"

/*! Moment buffer at `param`'s shape with its OWN quantization (deep clone
 * of momentQuant) -- accumulator dtype decoupled from the parameter's
 * storage dtype (SgdApi momentumStateInit precedent, #277). */
static tensor_t *momentStateInit(tensor_t *param, quantization_t *momentQuant) {
    return initTensor(getShapeLike(param->shape), getQLike(momentQuant), NULL);
}

optimizer_t *adamWCreateOptim(float learningRate, double beta1, double beta2, double eps,
                              double weightDecay, layer_t **model, size_t sizeModel,
                              quantization_t *momentQuant, arithmetic_t updateMath) {
    optimizer_t *optim = reserveMemory(sizeof(optimizer_t));
    optim->type = ADAM_W;

    optimImpl_t *impl = reserveMemory(sizeof(optimImpl_t));
    adamW_t *adamW = reserveMemory(sizeof(adamW_t));
    adamWInit(adamW, learningRate, beta1, beta2, eps, weightDecay, updateMath);
    impl->adamW = adamW;
    optim->impl = impl;

    size_t sizeStates = calcTotalNumberOfStates(model, sizeModel);
    optim->sizeStates = sizeStates;
    parameter_t **parameter = reserveMemory(sizeStates * sizeof(parameter_t *));
    optim->parameter = parameter;
    collectTrainableParameters(model, sizeModel, parameter);

    /* AdamW always carries both moments (m = stateBuffers[0], v = [1]);
     * there is no SGD-style momentum==0 stateless arm. reserveMemory
     * zero-fills, so both moments start at torch's init (zeros). */
    states_t **states = reserveMemory(sizeStates * sizeof(states_t *));
    for (size_t s = 0; s < sizeStates; s++) {
        states_t *paramStates = reserveMemory(sizeof(states_t));
        paramStates->statesPerParameter = 2;
        paramStates->stateBuffers = reserveMemory(2 * sizeof(tensor_t *));
        paramStates->stateBuffers[0] = momentStateInit(optim->parameter[s]->param, momentQuant);
        paramStates->stateBuffers[1] = momentStateInit(optim->parameter[s]->param, momentQuant);
        states[s] = paramStates;
    }
    optim->states = states;

    validateOptimizerGradStorage(optim, "adamWCreateOptim");
    return optim;
}
