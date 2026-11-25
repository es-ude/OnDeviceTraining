#define SOURCE_FILE "ADAM-W-API"

#include <stdlib.h>
#include <string.h>

#include "AdamW.h"
#include "AdamWApi.h"
#include "Common.h"
#include "OptimizerApi.h"
#include "StorageApi.h"
#include "Tensor.h"

optimizer_t *adamWCreateOptim(float learningRate, double beta1, double beta2, double eps,
                              double weightDecay, const model_t *m, quantization_t *momentQuant,
                              arithmetic_t updateMath) {
    optimizer_t *optim = reserveMemory(sizeof(optimizer_t));
    optim->type = ADAM_W;
    /* Seeded SR write-back by default (dead-zone escape);
     * optimizerSetWriteBackRounding(optim, HALF_AWAY) is the explicit opt-out. */
    optim->writeBackRounding = SR_HALF_AWAY;

    optimImpl_t *impl = reserveMemory(sizeof(optimImpl_t));
    adamW_t *adamW = reserveMemory(sizeof(adamW_t));
    adamWInit(adamW, learningRate, beta1, beta2, eps, weightDecay, updateMath);
    impl->adamW = adamW;
    optim->impl = impl;

    size_t sizeStates = calcTotalNumberOfStates(m);
    /* Reject all-frozen models while allowing parameterless models. */
    if (sizeStates == 0 && modelHasFrozenLayer(m)) {
        PRINT_ERROR("adamWCreateOptim: model has no trainable parameters "
                    "(every parameter-bearing layer is frozen) - nothing to optimize");
        exit(1);
    }
    optim->sizeStates = sizeStates;
    parameter_t **parameter = reserveMemory(sizeStates * sizeof(parameter_t *));
    optim->parameter = parameter;
    collectTrainableParameters(m, parameter);

    /* m = stateBuffers[0], v = [1], both zero-initialized like torch. */
    optim->states = initMomentStates(optim, 2, momentQuant, "adamWCreateOptim");

    validateOptimizerGradStorage(optim, "adamWCreateOptim");
    optim->numGroups = 1;
    return optim;
}

size_t adamWAddGroup(optimizer_t *optim, const model_t *m, paramSelectFn_t select, void *ctx,
                     float lrMultiplier, double weightDecay) {
    if (optim->type != ADAM_W) {
        PRINT_ERROR("adamWAddGroup: the optimizer is not ADAM_W");
        exit(1);
    }
    const size_t group = assignParameterGroup(optim, m, select, ctx, lrMultiplier, "adamWAddGroup");

    adamW_t *groups = reserveMemory((group + 1) * sizeof(adamW_t));
    memcpy(groups, optim->impl->adamW, group * sizeof(adamW_t));
    adamWInitGroup(&groups[group], &groups[0], lrMultiplier, weightDecay);
    freeReservedMemory(optim->impl->adamW);
    optim->impl->adamW = groups;
    optim->numGroups = group + 1;
    return group;
}
