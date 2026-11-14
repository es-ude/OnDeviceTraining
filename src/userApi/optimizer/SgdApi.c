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

/* The integer SGD ops on BFP storage (Sgd.c) have no weight-decay term: a BFP
 * state on a BFP gradient is the integer momentum store (needs 0 <
 * momentumFactor <= 1), and a BFP parameter whose update is integer -- the
 * stored BFP momentum on a BFP gradient, or the BFP gradient itself without
 * momentum -- is the integer parameter update (needs a finite learningRate).
 * Refuse such a configuration at creation rather than at the first step
 * (sgdStepM checks again for hand-built optimizers). */
static void rejectUnimplementedIntegerSgd(const optimizer_t *optim, const sgd_t *sgd,
                                          const quantization_t *momentumQuant) {
    for (size_t i = 0; i < optim->sizeStates; i++) {
        const parameter_t *p = optim->parameter[i];
        if (p->grad->quantization->type != BFP) {
            continue;
        }
        const bool paramBfp = p->param->quantization->type == BFP;
        const bool integerState = optim->states != NULL && momentumQuant->type == BFP;
        const bool integerParam = paramBfp && (optim->states == NULL || integerState);
        if (!integerState && !integerParam) {
            continue;
        }
        if (sgd->weightDecay != 0.0f) {
            PRINT_ERROR("sgdMCreateOptim: the integer SGD update on BFP storage has no "
                        "weight-decay term (weightDecay = %g must be 0)",
                        (double)sgd->weightDecay);
            exit(1);
        }
        if (integerState && !sgd->momentumIntValid) {
            PRINT_ERROR("sgdMCreateOptim: BFP momentum on BFP gradients is the integer "
                        "momentum store, which needs 0 < momentumFactor <= 1 (got %g)",
                        (double)sgd->momentumFactor);
            exit(1);
        }
        if (integerParam && !sgd->learningRateIntValid) {
            PRINT_ERROR("sgdMCreateOptim: the integer parameter update on BFP parameters "
                        "needs a finite learningRate (got %g)",
                        (double)sgd->learningRate);
            exit(1);
        }
    }
}

optimizer_t *sgdMCreateOptim(float learningRate, float momentumFactor, float weightDecay,
                             const model_t *m, quantization_t *momentumQuant,
                             arithmetic_t updateMath) {
    optimizer_t *optim = reserveMemory(sizeof(optimizer_t));
    optim->type = SGD_M;
    /* Seeded SR write-back by default (dead-zone escape);
     * optimizerSetWriteBackRounding(optim, HALF_AWAY) is the explicit opt-out. */
    optim->writeBackRounding = SR_HALF_AWAY;

    optimImpl_t *sgdImpl = reserveMemory(sizeof(optimImpl_t));
    sgd_t *sgd = reserveMemory(sizeof(sgd_t));
    sgdInit(sgd, learningRate, momentumFactor, weightDecay, updateMath);
    sgdImpl->sgd = sgd;
    optim->impl = sgdImpl;

    size_t sizeStates = calcTotalNumberOfStates(m);
    /* Reject all-frozen models while allowing parameterless models. */
    if (sizeStates == 0 && modelHasFrozenLayer(m)) {
        PRINT_ERROR("sgdMCreateOptim: model has no trainable parameters "
                    "(every parameter-bearing layer is frozen) - nothing to optimize");
        exit(1);
    }
    optim->sizeStates = sizeStates;
    parameter_t **parameter = reserveMemory(sizeStates * sizeof(parameter_t *));
    optim->parameter = parameter;
    collectTrainableParameters(m, parameter);

    /* momentumFactor == 0 has no momentum state, so momentumQuant is ignored. */
    optim->states = (momentumFactor == 0.0f)
                        ? NULL
                        : initMomentStates(optim, 1, momentumQuant, "sgdMCreateOptim");

    validateOptimizerGradStorage(optim, "sgdMCreateOptim");
    rejectUnimplementedIntegerSgd(optim, sgd, momentumQuant);
    return optim;
}
