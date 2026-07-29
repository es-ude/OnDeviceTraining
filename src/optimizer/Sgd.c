#define SOURCE_FILE "SGD"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "ArithmeticType.h"
#include "Common.h"
#include "ExecuteOp.h"
#include "Sgd.h"
#include "Tensor.h"

void sgdInit(sgd_t *sgd, float learningRate, float momentumFactor, float weightDecay,
             arithmetic_t updateMath) {
    /* Only ARITH_FLOAT32 update arithmetic is implemented; fail at
     * construction so a training run cannot start on an unimplemented
     * arm (#310). */
    if (updateMath.type != ARITH_FLOAT32) {
        PRINT_ERROR("SGD updateMath: only ARITH_FLOAT32 is implemented "
                    "(integer-arithmetic update numerics not yet designed, #310)");
        exit(1);
    }
    sgd->learningRate = learningRate;
    sgd->momentumFactor = momentumFactor;
    sgd->weightDecay = weightDecay;
    sgd->updateMath = updateMath;
}

typedef struct {
    float lr, weightDecay;
} sgdUpdateCtx_t; /* plain SGD */
typedef struct {
    float momentum, weightDecay;
} sgdMStateCtx_t; /* momentum: state */
typedef struct {
    float lr;
} sgdMParamCtx_t; /* momentum: param */

/* executeOp update kernels. The update arithmetic comes from the sgd_t's
 * updateMath knob (#310); only ARITH_FLOAT32 is implemented (guards in
 * sgdInit + sgdStepM), so the funnel prologue dequants every operand to
 * float and the OUT_WRITE epilogue requants rawOut into the target's own
 * dtype -- per-parameter dtype dispatch without a qtype switch here.
 * Rounding ownership (#282): the OUT_WRITE epilogue rounds by the op's
 * arithmetic.roundingMode, so each step function overrides it with
 * optim->writeBackRounding on every param/state write-back -- that is where
 * a SYM param's SR_HALF_AWAY dead-zone escape lives (#279, PR #284). The
 * tensors' own qConfig roundingMode stays untouched (storage/serialization). */

/* operands {param, grad} -> rawOut = param - lr*(grad + wd*param) */
static void sgdUpdateKernel(tensor_t **op, size_t n, tensor_t *rawOut, tensor_t *aux,
                            const void *ctxv) {
    (void)n;
    (void)aux;
    const sgdUpdateCtx_t *ctx = ctxv;
    size_t numberOfValues = calcNumberOfElementsByTensor(rawOut);
    float *param = (float *)op[0]->data;
    float *grad = (float *)op[1]->data;
    float *out = (float *)rawOut->data;
    for (size_t i = 0; i < numberOfValues; i++) {
        float g = grad[i] + ctx->weightDecay * param[i];
        out[i] = param[i] - ctx->lr * g;
    }
}

/* operands {state, grad, param} -> rawOut = mom*state + grad + wd*param */
static void sgdMStateKernel(tensor_t **op, size_t n, tensor_t *rawOut, tensor_t *aux,
                            const void *ctxv) {
    (void)n;
    (void)aux;
    const sgdMStateCtx_t *ctx = ctxv;
    size_t numberOfValues = calcNumberOfElementsByTensor(rawOut);
    float *state = (float *)op[0]->data;
    float *grad = (float *)op[1]->data;
    float *param = (float *)op[2]->data;
    float *out = (float *)rawOut->data;
    for (size_t i = 0; i < numberOfValues; i++) {
        float g = grad[i] + ctx->weightDecay * param[i];
        out[i] = ctx->momentum * state[i] + g;
    }
}

/* operands {param, state} -> rawOut = param - lr*state */
static void sgdMParamKernel(tensor_t **op, size_t n, tensor_t *rawOut, tensor_t *aux,
                            const void *ctxv) {
    (void)n;
    (void)aux;
    const sgdMParamCtx_t *ctx = ctxv;
    size_t numberOfValues = calcNumberOfElementsByTensor(rawOut);
    float *param = (float *)op[0]->data;
    float *state = (float *)op[1]->data;
    float *out = (float *)rawOut->data;
    for (size_t i = 0; i < numberOfValues; i++) {
        out[i] = param[i] - ctx->lr * state[i];
    }
}

void sgdStepM(optimizer_t *optim) {
    sgd_t *sgd = optim->impl->sgd;

    /* Re-checked here (not only in sgdInit): the update kernels raw-cast
     * operand data to float*, so a non-FLOAT32 prologue would be silently
     * misread -- hand-assembled optimizers must hit the same wall (#310). */
    if (sgd->updateMath.type != ARITH_FLOAT32) {
        PRINT_ERROR("SGD updateMath: only ARITH_FLOAT32 is implemented "
                    "(integer-arithmetic update numerics not yet designed, #310)");
        exit(1);
    }

    /* #282: training write-backs round by the OPTIMIZER's knob -- the update
     * ops run with writeBackRounding as their operation-owned rounding. */
    arithmetic_t updateMath = sgd->updateMath;
    updateMath.roundingMode = optim->writeBackRounding;

    /* momentumFactor == 0: momentum state is semantically nonexistent --
     * the factory allocates no state buffers in this mode (optim->states
     * may be NULL), so run the stateless single-op update instead of the
     * two-op momentum path. Mathematically identical at momentum == 0:
     * state would be exactly grad + wd*param, and param -= lr*state
     * collapses to param -= lr*(grad + wd*param). (#308) */
    if (sgd->momentumFactor == 0.0f) {
        for (size_t i = 0; i < optim->sizeStates; i++) {
            parameter_t *p = optim->parameter[i];
            sgdUpdateCtx_t ctx = {.lr = sgd->learningRate, .weightDecay = sgd->weightDecay};
            /* #296 Stage 1: all three update kernels are elementwise (read i before
             * write i), so the FLOAT32 fast paths may write params/state in place
             * instead of staging through rawData. */
            executeOp(
                &(opSpec_t){
                    .kernel = sgdUpdateKernel,
                    .ctx = &ctx,
                    .inputs = (tensor_t *[]){p->param, p->grad},
                    .nInputs = 2,
                    .arithmetic = updateMath,
                    .mode = OUT_WRITE,
                    .auxOut = NULL,
                    .writesInPlaceSafe = true,
                    /* Group-quant PR3 Task 4: param is inputs[0] -- the only
                     * operand allowed to be a grouped-SYM tensor here. */
                    .groupedSymOperandPos = 1,
                },
                p->param);
        }
        return;
    }

    for (size_t i = 0; i < optim->sizeStates; i++) {
        parameter_t *p = optim->parameter[i];
        tensor_t *state = optim->states[i]->stateBuffers[0];

        sgdMStateCtx_t sc = {.momentum = sgd->momentumFactor, .weightDecay = sgd->weightDecay};
        executeOp(
            &(opSpec_t){
                .kernel = sgdMStateKernel,
                .ctx = &sc,
                .inputs = (tensor_t *[]){state, p->grad, p->param},
                .nInputs = 3,
                .arithmetic = updateMath,
                .mode = OUT_WRITE,
                .auxOut = NULL,
                .writesInPlaceSafe = true,
                /* Group-quant PR3 Task 4: param is inputs[2] here (state,
                 * then grad, then param) -- the momentum-state carrier gate
                 * (PR2) keeps state/grad per-tensor, so param is the only
                 * operand allowed to be grouped-SYM. */
                .groupedSymOperandPos = 3,
            },
            state);

        sgdMParamCtx_t pc = {.lr = sgd->learningRate};
        executeOp(
            &(opSpec_t){
                .kernel = sgdMParamKernel,
                .ctx = &pc,
                .inputs = (tensor_t *[]){p->param, state},
                .nInputs = 2,
                .arithmetic = updateMath,
                .mode = OUT_WRITE,
                .auxOut = NULL,
                .writesInPlaceSafe = true,
                /* Group-quant PR3 Task 4: param is inputs[0] here. */
                .groupedSymOperandPos = 1,
            },
            p->param);
    }
}

float sgdGetLr(optimizer_t *optimizer) {
    return optimizer->impl->sgd->learningRate;
}

void sgdSetLr(optimizer_t *optimizer, float learningRate) {
    optimizer->impl->sgd->learningRate = learningRate;
}
