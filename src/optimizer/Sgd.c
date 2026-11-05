#define SOURCE_FILE "SGD"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "ArithmeticType.h"
#include "BfpKernelSupport.h"
#include "Common.h"
#include "ExecuteOp.h"
#include "OptimizerKeys.h"
#include "RNG.h"
#include "Sgd.h"
#include "Tensor.h"
#include "TensorConversion.h"

/* Config time: the learning rate and its integer encoding for the integer
 * parameter update on BFP parameters (non-finite: no encoding, that update
 * fails fast; the float update ops do not use it). */
static void sgdSetLearningRate(sgd_t *sgd, float learningRate) {
    sgd->learningRate = learningRate;
    sgd->learningRateQ = 0;
    sgd->learningRateE = 0;
    sgd->learningRateIntValid = false;
    if (isfinite(learningRate)) {
        const bfpScalar_t lr = bfpScalarFromFloat(learningRate, "SGD learningRate");
        sgd->learningRateQ = lr.q;
        sgd->learningRateE = lr.e;
        sgd->learningRateIntValid = true;
    }
}

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
    sgdSetLearningRate(sgd, learningRate);
    sgd->momentumFactor = momentumFactor;
    sgd->weightDecay = weightDecay;
    sgd->updateMath = updateMath;
    sgd->momentumIsZero = momentumFactor == 0.0f;
    sgd->weightDecayIsNonzero = weightDecay != 0.0f;
    /* Config time (float is allowed here, never in the training-step roots):
     * the momentum for the integer store on BFP state, encoded once. */
    sgd->momentumQ = 0;
    sgd->momentumE = 0;
    sgd->momentumIntValid = false;
    if (momentumFactor > 0.0f && momentumFactor <= 1.0f) {
        const bfpScalar_t mu = bfpScalarFromFloat(momentumFactor, "SGD momentumFactor");
        sgd->momentumQ = mu.q;
        sgd->momentumE = mu.e;
        sgd->momentumIntValid = true;
    }
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
                            const void *ctxv, rngStream_t *rs) {
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
                            const void *ctxv, rngStream_t *rs) {
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
                            const void *ctxv, rngStream_t *rs) {
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

/* The update ops' OP(k) under (PARAM, p), one numbering for both paths (the
 * stateless path runs only the param write). Part of the value contract:
 * append, never renumber. */
enum { SGD_OP_STATE = 0, SGD_OP_PARAM = 1 };

/* The OUT stream of update op `op` of parameter `paramIndex` -- the stream the
 * funnel's OUT_WRITE re-pack of that op opens -- or NULL without an update
 * key. */
static rngStream_t *sgdOpOutStream(const rngKey_t *updateKey, uint32_t paramIndex, uint32_t op,
                                   rngStream_t *storage) {
    rngKey_t opKey;
    const rngKey_t *key = optimizerUpdateOpKey(updateKey, paramIndex, op, &opKey);
    if (key == NULL) {
        return NULL;
    }
    *storage = rngOpen(rngFold(*key, RNG_TAG_OUT, 0u));
    return storage;
}

/* Cold and out of line: keeps the formatted float of the message out of the
 * step's own frame (the census reads sgdStepM's own call sites). */
#if defined(__GNUC__) || defined(__clang__)
__attribute__((noinline, cold))
#endif
static void sgdRejectIntegerWeightDecay(const sgd_t *sgd) {
    PRINT_ERROR("SGD: nonzero weightDecay (%g) is not implemented by the integer update on "
                "BFP storage -- fails fast instead of dropping it",
                (double)sgd->weightDecay);
    exit(1);
}

/* The state op when the gradient AND the state are BFP: the integer momentum
 * store v' = mu * v + g (momentumStoreBfpInPlace; arithmetic-bfp.md §15.7) in
 * place of the float kernel, the same formula and order (no dampening, no
 * Nesterov; coupled weight decay is rejected by the caller). Same RNG
 * contract as the funnel op it replaces: the OUT stream of the state op's
 * key, one draw per element in element order -- the draw count of the
 * OUT_WRITE re-pack -- so every other op's stream is untouched. Rounds by
 * the optimizer's write-back mode (#282). Out of line: its own census root,
 * integer only. */
#if defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#endif
static void sgdMomentumStoreBfp(const sgd_t *sgd, tensor_t *state, const tensor_t *grad,
                                uint32_t paramIndex, roundingMode_t rm, const rngKey_t *updateKey) {
    if (!sgd->momentumIntValid) {
        PRINT_ERROR("SGD: the integer momentum store on BFP state needs 0 < momentumFactor "
                    "<= 1 (the float momentum op is not a fallback for BFP state)");
        exit(1);
    }
    rngStream_t stream;
    momentumStoreBfpInPlace(state, grad, (bfpScalar_t){.q = sgd->momentumQ, .e = sgd->momentumE},
                            rm, sgdOpOutStream(updateKey, paramIndex, SGD_OP_STATE, &stream));
}

/* The param op when the parameter and its update are BFP: the integer update
 * param' = param - lr * update (paramUpdateBfpInPlace; arithmetic-bfp.md
 * §15.7) in place of the float kernel. `update` is the STORED new momentum
 * (after the state op's own rounding: the float op's order) or, without
 * momentum, the gradient. Same RNG contract as the funnel op it replaces: the
 * OUT stream of the param op's key, one draw per element in element order.
 * Rounds by the optimizer's write-back mode (#282). Out of line: its own
 * census root, integer only. */
#if defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#endif
static void sgdParamUpdateBfp(const sgd_t *sgd, tensor_t *param, const tensor_t *update,
                              uint32_t paramIndex, roundingMode_t rm, const rngKey_t *updateKey) {
    if (!sgd->learningRateIntValid) {
        PRINT_ERROR("SGD: the integer parameter update on BFP parameters needs a finite "
                    "learningRate (the float update op is not a fallback for BFP parameters)");
        exit(1);
    }
    rngStream_t stream;
    paramUpdateBfpInPlace(param, update,
                          (bfpScalar_t){.q = sgd->learningRateQ, .e = sgd->learningRateE}, rm,
                          sgdOpOutStream(updateKey, paramIndex, SGD_OP_PARAM, &stream));
}

void sgdStepM(optimizer_t *optim, const rngKey_t *updateKey) {
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
    if (sgd->momentumIsZero) {
        for (size_t i = 0; i < optim->sizeStates; i++) {
            parameter_t *p = optim->parameter[i];
            /* BFP parameter on a BFP gradient: the integer update w - lr * g */
            if (p->param->quantization->type == BFP && p->grad->quantization->type == BFP) {
                if (sgd->weightDecayIsNonzero) {
                    sgdRejectIntegerWeightDecay(sgd);
                }
                sgdParamUpdateBfp(sgd, p->param, p->grad, (uint32_t)i, optim->writeBackRounding,
                                  updateKey);
                continue;
            }
            sgdUpdateCtx_t ctx = {.lr = sgd->learningRate, .weightDecay = sgd->weightDecay};
            rngKey_t opKey;
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
                    .key = optimizerUpdateOpKey(updateKey, (uint32_t)i, SGD_OP_PARAM, &opKey),
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

        /* BFP state on a BFP gradient: the integer momentum store, then -- for a
         * BFP parameter -- the integer parameter update from the STORED v' */
        const bool integerState =
            state->quantization->type == BFP && p->grad->quantization->type == BFP;
        if (integerState) {
            if (sgd->weightDecayIsNonzero) {
                sgdRejectIntegerWeightDecay(sgd);
            }
            sgdMomentumStoreBfp(sgd, state, p->grad, (uint32_t)i, optim->writeBackRounding,
                                updateKey);
        } else {
            sgdMStateCtx_t sc = {.momentum = sgd->momentumFactor, .weightDecay = sgd->weightDecay};
            rngKey_t stateKey;
            executeOp(
                &(opSpec_t){
                    .kernel = sgdMStateKernel,
                    .ctx = &sc,
                    .inputs = (tensor_t *[]){state, p->grad, p->param},
                    .nInputs = 3,
                    .arithmetic = updateMath,
                    .key = optimizerUpdateOpKey(updateKey, (uint32_t)i, SGD_OP_STATE, &stateKey),
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
        }

        if (integerState && p->param->quantization->type == BFP) {
            sgdParamUpdateBfp(sgd, p->param, state, (uint32_t)i, optim->writeBackRounding,
                              updateKey);
            continue;
        }
        sgdMParamCtx_t pc = {.lr = sgd->learningRate};
        rngKey_t paramKey;
        executeOp(
            &(opSpec_t){
                .kernel = sgdMParamKernel,
                .ctx = &pc,
                .inputs = (tensor_t *[]){p->param, state},
                .nInputs = 2,
                .arithmetic = updateMath,
                .key = optimizerUpdateOpKey(updateKey, (uint32_t)i, SGD_OP_PARAM, &paramKey),
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
    sgdSetLearningRate(optimizer->impl->sgd, learningRate);
}
