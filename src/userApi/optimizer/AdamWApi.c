#define SOURCE_FILE "ADAM-W-API"

#include <stdlib.h>

#include "AdamW.h"
#include "AdamWApi.h"
#include "Common.h"
#include "OptimizerApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"

/*! Moment buffer at `param`'s shape with its OWN quantization (deep clone
 * of momentQuant) -- accumulator dtype decoupled from the parameter's
 * storage dtype (SgdApi momentumStateInit precedent, #277). */
static tensor_t *momentStateInit(tensor_t *param, quantization_t *momentQuant) {
    /* Group-quant PR2 final-review Fix 3(c) carrier gate (mirrors gradInit,
     * TensorApi.c, and SgdApi's momentumStateInit twin): groups are legal
     * ONLY on GEMM-family weight tensors -- moment states stay per-tensor
     * unconditionally until PR3. Without this, getQLike would silently
     * clone a grouped SYM momentQuant template into a grouped moment
     * buffer -- a state PR2's contract never intended to support. */
    if (momentQuant->type == SYM) {
        symQConfig_t *symQC = momentQuant->qConfig;
        if (symQC->numGroups > 1) {
            PRINT_ERROR("momentStateInit: grouped SYM moment templates are unsupported -- "
                        "grouped moments are a future #300 axis (spec §3)");
            exit(1);
        }
    }
    /* Group-quant PR4 (Task 3): ASYM twin of the gate above (SgdApi's
     * momentumStateInit mirror) -- getQLike's ASYM arm deep-clones grouped
     * grids, so without this a grouped template would silently become a
     * grouped moment buffer. */
    if (momentQuant->type == ASYM) {
        asymQConfig_t *asymQC = momentQuant->qConfig;
        if (asymQC->numGroups > 1) {
            PRINT_ERROR("momentStateInit: grouped ASYM moment templates are unsupported -- "
                        "grouped moments are a future #300 axis (spec §3)");
            exit(1);
        }
    }
    /* BFP epic PR1 carrier gate (mirrors gradInit, TensorApi.c, and SgdApi's
     * momentumStateInit twin): BFP grad/state storage is out of scope for
     * this epic PR -- reject any BFP moment template outright. */
    if (momentQuant->type == BFP) {
        PRINT_ERROR("momentStateInit: BFP moment templates are unsupported -- "
                    "BFP grad/state storage arrives with BFP epic PR3");
        exit(1);
    }
    return initTensor(getShapeLike(param->shape), getQLike(momentQuant), NULL);
}

optimizer_t *adamWCreateOptim(float learningRate, double beta1, double beta2, double eps,
                              double weightDecay, layer_t **model, size_t sizeModel,
                              quantization_t *momentQuant, arithmetic_t updateMath) {
    optimizer_t *optim = reserveMemory(sizeof(optimizer_t));
    optim->type = ADAM_W;
    /* #279 ratified default: seeded-SR training write-back (dead-zone escape);
     * optimizerSetWriteBackRounding(optim, HALF_AWAY) is the explicit opt-out. */
    optim->writeBackRounding = SR_HALF_AWAY;

    optimImpl_t *impl = reserveMemory(sizeof(optimImpl_t));
    adamW_t *adamW = reserveMemory(sizeof(adamW_t));
    adamWInit(adamW, learningRate, beta1, beta2, eps, weightDecay, updateMath);
    impl->adamW = adamW;
    optim->impl = impl;

    size_t sizeStates = calcTotalNumberOfStates(model, sizeModel);
    /* #380: a zero-state model is only an error when it's DEGENERATE -- every
     * parameter-bearing layer got frozen (nothing to optimize). A model that
     * simply has no parameter-bearing layer types at all (e.g. a lone
     * Dropout/pooling layer) is a pre-existing, deliberately-supported
     * zero-state configuration (SgdApi sibling contract) and must keep
     * working. */
    if (sizeStates == 0 && modelHasFrozenLayer(model, sizeModel)) {
        PRINT_ERROR("adamWCreateOptim: model has no trainable parameters "
                    "(every parameter-bearing layer is frozen) - nothing to optimize");
        exit(1);
    }
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
