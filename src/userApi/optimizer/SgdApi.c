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
#include "OptimizerApi.h"
#include "SgdApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"

/*! Builds a momentum-state tensor at `param`'s shape but with its OWN
 * quantization (`momentumQuant`, deep-cloned via getQLike) -- decouples the
 * accumulator's dtype from the parameter's storage dtype (#277 Task 2). */
static tensor_t *momentumStateInit(tensor_t *param, quantization_t *momentumQuant) {
    /* Group-quant PR2 final-review Fix 3(c) carrier gate (mirrors gradInit,
     * TensorApi.c): groups are legal ONLY on GEMM-family weight tensors --
     * momentum states stay per-tensor unconditionally until PR3. Without
     * this, getQLike would silently clone a grouped SYM momentumQuant
     * template into a grouped momentum buffer -- a state PR2's contract
     * never intended to support. */
    if (momentumQuant->type == SYM) {
        symQConfig_t *symQC = momentumQuant->qConfig;
        if (symQC->numGroups > 1) {
            PRINT_ERROR("momentumStateInit: grouped SYM momentum templates are unsupported -- "
                        "grouped momentum is a future #300 axis (spec §3)");
            exit(1);
        }
    }
    return initTensor(getShapeLike(param->shape), getQLike(momentumQuant), NULL);
}

optimizer_t *sgdMCreateOptim(float learningRate, float momentumFactor, float weightDecay,
                             layer_t **model, size_t sizeModel, quantization_t *momentumQuant,
                             arithmetic_t updateMath) {
    optimizer_t *optim = reserveMemory(sizeof(optimizer_t));
    optim->type = SGD_M;
    /* #279 ratified default: seeded-SR training write-back (dead-zone escape);
     * optimizerSetWriteBackRounding(optim, HALF_AWAY) is the explicit opt-out. */
    optim->writeBackRounding = SR_HALF_AWAY;

    optimImpl_t *sgdImpl = reserveMemory(sizeof(optimImpl_t));
    sgd_t *sgd = reserveMemory(sizeof(sgd_t));
    sgdInit(sgd, learningRate, momentumFactor, weightDecay, updateMath);
    sgdImpl->sgd = sgd;
    optim->impl = sgdImpl;

    size_t sizeStates = calcTotalNumberOfStates(model, sizeModel);
    /* #380: a zero-state model is only an error when it's DEGENERATE -- every
     * parameter-bearing layer got frozen (nothing to optimize). A model that
     * simply has no parameter-bearing layer types at all (e.g. a lone
     * Dropout/pooling layer) is a pre-existing, deliberately-supported
     * zero-state configuration (UnitTestDropoutIntegration/
     * UnitTestAdaptiveAvgPool1dIntegration) and must keep working. */
    if (sizeStates == 0 && modelHasFrozenLayer(model, sizeModel)) {
        PRINT_ERROR("sgdMCreateOptim: model has no trainable parameters "
                    "(every parameter-bearing layer is frozen) - nothing to optimize");
        exit(1);
    }
    optim->sizeStates = sizeStates;
    parameter_t **parameter = reserveMemory(sizeStates * sizeof(parameter_t *));
    optim->parameter = parameter;
    collectTrainableParameters(model, sizeModel, parameter);

    /* momentumFactor == 0: momentum state is semantically nonexistent --
     * allocate none (sgdStepM's momentum==0 path never reads states, #308).
     * Every trainable parameter otherwise gets one state buffer at the
     * param's shape with the caller's momentumQuant (dtype-decoupled, #277). */
    if (momentumFactor == 0.0f) {
        optim->states = NULL;
    } else {
        states_t **states = reserveMemory(sizeStates * sizeof(states_t *));
        for (size_t s = 0; s < sizeStates; s++) {
            states_t *paramStates = reserveMemory(sizeof(states_t));
            paramStates->statesPerParameter = 1;
            paramStates->stateBuffers = reserveMemory(sizeof(tensor_t *));
            paramStates->stateBuffers[0] =
                momentumStateInit(optim->parameter[s]->param, momentumQuant);
            states[s] = paramStates;
        }
        optim->states = states;
    }

    validateOptimizerGradStorage(optim, "sgdMCreateOptim");
    return optim;
}
