#define SOURCE_FILE "CALCULATE_GRADS_SEQUENTIAL"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "BatchNorm1d.h"
#include "CalculateGradsSequential.h"
#include "Common.h"
#include "Dropout.h"
#include "Layer.h"
#include "LayerConfigAccess.h"
#include "LossFunction.h"
#include "OdtHook.h"
#include "RematCheck.h"
#include "RematScheduler.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "TraceApi.h"

/* Dropout and BatchNorm1d run in training mode exactly for one grads call
 * (forward + backward, #460): inference, inferenceWithLoss and evaluation
 * never flip it, so they see eval mode. */
static void setLayersTrainingMode(layer_t **model, size_t modelSize, bool training) {
    for (size_t i = 0; i < modelSize; i++) {
        switch (model[i]->type) {
        case DROPOUT:
            model[i]->config->dropout->training = training;
            break;
        case BATCHNORM1D:
            model[i]->config->batchNorm1d->training = training;
            break;
        default:
            break;
        }
    }
}

/* The one per-call allocation the driver keeps: the caller-owned result. */
static trainingStats_t *initTrainingStats(tensor_t *output) {
    trainingStats_t *trainingStats = reserveMemory(sizeof(trainingStats_t));

    tensor_t *o = getTensorLike(output);
    trainingStats->output = o;

    return trainingStats;
}

/* What a training call without a scheduler runs on (remat D30). The policy
 * of this use case, not of the scheduler library, whose NULL spec keeps
 * meaning STORE_ALL. */
static const rematPlanSpec_t defaultPlanSpec = {.policy = REMAT_PLAN_STORE_ALL};

const rematPlanSpec_t *calculateGradsDefaultPlanSpec(void) {
    return &defaultPlanSpec;
}

/* The validating interpreter: a row hands out each step, the checker
 * validates it and resolves its operands before anything runs, and only then
 * does the driver execute it. Without a caller scheduler every call builds an
 * ephemeral HEAP one on the default plan, one block per wire range. */
static trainingStats_t *calculateGradsImpl(layer_t **model, size_t modelSize,
                                           lossConfig_t lossConfig, reduction_t forwardReduction,
                                           tensor_t *input, tensor_t *label, traceSink_t sink,
                                           void *sinkCtx, const trainingCall_t *call) {
    /* Phase hook (OdtHook.h): FORWARD and BACKWARD tile this whole call. */
    odtHookFire(ODT_EVENT_FORWARD_BEGIN);
    setLayersTrainingMode(model, modelSize, true);

    rematScheduler_t ephemeral;
    rematScheduler_t *s = (call != NULL) ? call->remat : NULL;
    if (s == NULL) {
        if (!rematHeapInit(&ephemeral, model, modelSize, lossConfig, input, &defaultPlanSpec)) {
            PRINT_ERROR("calculateGrads: ephemeral HEAP scheduler: reserveMemory failed");
            exit(1);
        }
        s = &ephemeral;
    }
    uint32_t producedGen[rematCheckNumWires(s)];
    rematCheck_t chk;
    rematCheckInit(&chk, s, model, modelSize, lossConfig.funcType, REMAT_MODE_TRAIN, producedGen);

    rematBegin(s, model, modelSize, lossConfig, input);
    lossFunctions_t lossFns = lossFunctions[lossConfig.funcType];
    trainingStats_t *trainingStats = NULL;
    rematStep_t st;
    while (rematNext(s, &st)) {
        rematOperands_t op;
        rematCheckStep(&chk, &st, &op);
        layer_t *layer = (st.layer < modelSize) ? model[st.layer] : NULL;
        switch (st.kind) {
        case REMAT_STEP_FORWARD:
            layerFunctions[layer->type].forward(layer, op.in, op.out);
            if (sink != NULL) {
                sink(sinkCtx, st.layer, layer->type, "fwd", op.out);
            }
            break;
        case REMAT_STEP_LOSS_FORWARD:
            trainingStats = initTrainingStats(op.in);
            copyTensor(trainingStats->output, op.in);
            trainingStats->loss = lossFns.forward(op.in, label, forwardReduction);
            odtHookFire(ODT_EVENT_FORWARD_END);
            /* BACKWARD fires unconditionally -- also around a truncated or
             * skipped backward (all-frozen model) -- so the per-call event
             * count stays a constant an external occurrence counter can rely
             * on. */
            odtHookFire(ODT_EVENT_BACKWARD_BEGIN);
            break;
        case REMAT_STEP_LOSS_BACKWARD:
            lossFns.backward(op.in, label, op.out);
            if (sink != NULL) {
                sink(sinkCtx, modelSize, model[modelSize - 1]->type, "lossgrad", op.out);
            }
            break;
        case REMAT_STEP_BACKWARD: {
            /* agrad@l = the gradient w.r.t. layer l's OUTPUT (the wire grad
             * entering its backward), matching the PyTorch forward-hook
             * activation.grad; so it fires before the backward. */
            if (sink != NULL) {
                sink(sinkCtx, st.layer, layer->type, "agrad", op.gradIn);
            }
            tensor_t *x = op.in;
#ifdef ODT_REMAT_VERIFY
            /* Strict W_dead: a backward that does not read its input gets a
             * header without bytes on every plan, so a read the read-set table
             * denies crashes here instead of passing on the NULL path and
             * breaking under LIVENESS. */
            tensor_t dead;
            if (!layerBackwardReadsInput(layer)) {
                dead = *op.in;
                dead.data = NULL;
                x = &dead;
            }
#endif
            /* op.out is NULL at deepest: grads only, nothing below consumes dx. */
            layerFunctions[layer->type].backward(layer, x, op.gradIn, op.out);
            break;
        }
        default:
            break; /* unreachable: rematCheckStep exits on an unknown kind */
        }
        rematDone(s, &st);
    }
    rematCheckFinish(&chk);
    rematEnd(s);
    rematCheckReleased(&chk);
    /* A caller's scheduler is borrowed: it must survive for the next call. */
    if (s == &ephemeral) {
        rematSchedulerDeinit(&ephemeral);
    }

    setLayersTrainingMode(model, modelSize, false);
    odtHookFire(ODT_EVENT_BACKWARD_END);
    return trainingStats;
}

trainingStats_t *calculateGradsSequential(layer_t **model, size_t modelSize,
                                          lossConfig_t lossConfig, reduction_t forwardReduction,
                                          tensor_t *input, tensor_t *label,
                                          const trainingCall_t *call) {
    return calculateGradsImpl(model, modelSize, lossConfig, forwardReduction, input, label, NULL,
                              NULL, call);
}

trainingStats_t *tracedGrads(layer_t **model, size_t modelSize, lossConfig_t lossConfig,
                             reduction_t forwardReduction, tensor_t *input, tensor_t *label,
                             traceSink_t sink, void *ctx, const trainingCall_t *call) {
    return calculateGradsImpl(model, modelSize, lossConfig, forwardReduction, input, label, sink,
                              ctx, call);
}

static void traceModelParams(layer_t **model, size_t modelSize, const char *tag, bool wantGrad,
                             traceSink_t sink, void *ctx) {
    char phase[64];
    for (size_t i = 0; i < modelSize; i++) {
        parameter_t *w = NULL, *b = NULL;
        if (!layerParameters(model[i], &w, &b)) {
            continue;
        }
        /* Frozen layers (#380): getGradFromParameter returns NULL (Task 1 elides
         * the grad tensor). The TraceApi contract promises sinks a borrowed VALID
         * tensor -- skip the call entirely rather than handing them NULL. */
        tensor_t *wt = wantGrad ? getGradFromParameter(w) : getParamFromParameter(w);
        if (wt != NULL) {
            snprintf(phase, sizeof(phase), "%s.weight", tag);
            sink(ctx, i, model[i]->type, phase, wt);
        }
        if (b != NULL) {
            tensor_t *bt = wantGrad ? getGradFromParameter(b) : getParamFromParameter(b);
            if (bt != NULL) {
                snprintf(phase, sizeof(phase), "%s.bias", tag);
                sink(ctx, i, model[i]->type, phase, bt);
            }
        }
    }
}

void traceModelWeights(layer_t **model, size_t modelSize, const char *tag, traceSink_t sink,
                       void *ctx) {
    traceModelParams(model, modelSize, tag, /*wantGrad=*/false, sink, ctx);
}

void traceModelGrads(layer_t **model, size_t modelSize, const char *tag, traceSink_t sink,
                     void *ctx) {
    traceModelParams(model, modelSize, tag, /*wantGrad=*/true, sink, ctx);
}
