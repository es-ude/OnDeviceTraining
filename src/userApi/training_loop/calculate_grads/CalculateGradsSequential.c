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
#include "LayerExec.h"
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
static void setLayersTrainingMode(const model_t *m, bool training) {
    for (size_t i = 0; i < m->numLayers; i++) {
        switch (m->layers[i]->type) {
        case DROPOUT:
            m->layers[i]->config->dropout->training = training;
            break;
        case BATCHNORM1D:
            m->layers[i]->config->batchNorm1d->training = training;
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

/* What a training call without a scheduler runs on (remat D15, D30):
 * LIVENESS frees each activation after its last reader, with values
 * identical to STORE_ALL. The policy of this use case, not of the scheduler
 * library, whose NULL spec keeps meaning STORE_ALL. */
static const rematPlanSpec_t defaultPlanSpec = {.policy = REMAT_PLAN_LIVENESS};

const rematPlanSpec_t *calculateGradsDefaultPlanSpec(void) {
    return &defaultPlanSpec;
}

/* The validating interpreter: a row hands out each step, the checker
 * validates it and looks up its operands before anything runs, and only then
 * does the driver execute it. Without a caller scheduler every call builds an
 * ephemeral HEAP one on the default plan, one block per wire range. */
static trainingStats_t *calculateGradsImpl(const model_t *m, lossConfig_t lossConfig,
                                           reduction_t forwardReduction, tensor_t *input,
                                           tensor_t *label, traceSink_t sink, void *sinkCtx,
                                           const trainingCall_t *call) {
    /* Phase hook (OdtHook.h): FORWARD and BACKWARD tile this whole call. */
    odtHookFire(ODT_EVENT_FORWARD_BEGIN);
    setLayersTrainingMode(m, true);

    rematScheduler_t ephemeral;
    rematScheduler_t *s = (call != NULL) ? call->remat : NULL;
    const rngKey_t *callKey = (call != NULL) ? call->key : NULL;
    if (s == NULL) {
        if (!rematHeapInit(&ephemeral, m, lossConfig, input, &defaultPlanSpec)) {
            PRINT_ERROR("calculateGrads: ephemeral HEAP scheduler: reserveMemory failed");
            exit(1);
        }
        s = &ephemeral;
    }
    uint32_t producedGen[rematCheckNumWires(s)];
    rematCheck_t chk;
    rematCheckInit(&chk, s, m, lossConfig.funcType, REMAT_MODE_TRAIN, producedGen);

    rematBegin(s, m, lossConfig, input);
    lossFunctions_t lossFns = lossFunctions[lossConfig.funcType];
    trainingStats_t *trainingStats = NULL;
    rematStep_t st;
    while (rematNext(s, &st)) {
        rematOperands_t op;
        rematCheckStep(&chk, &st, &op);
        layer_t *layer = (st.layer < m->numLayers) ? m->layers[st.layer] : NULL;
        /* From the checked step's layer index only, so the scheduler cannot
         * change which draws a layer sees. */
        layerExec_t exStorage;
        const layerExec_t *ex = layerExecForLayer(callKey, st.layer, &exStorage);
        switch (st.kind) {
        case REMAT_STEP_FORWARD:
            layerFunctions[layer->type].forward(layer, op.in[0], op.out, ex);
            if (sink != NULL) {
                sink(sinkCtx, st.layer, layer->type, "fwd", op.out);
            }
            break;
        case REMAT_STEP_REFORWARD:
            /* FORWARD's arm under FORWARD's key (ex derives from the layer
             * index alone): the recomputed output is the first one's bytes. */
            layerFunctions[layer->type].forward(layer, op.in[0], op.out, ex);
            if (sink != NULL) {
                sink(sinkCtx, st.layer, layer->type, "refwd", op.out);
            }
            break;
        case REMAT_STEP_LOSS_FORWARD:
            trainingStats = initTrainingStats(op.in[0]);
            copyTensor(trainingStats->output, op.in[0]);
            trainingStats->loss = lossFns.forward(op.in[0], label, forwardReduction, ex);
            odtHookFire(ODT_EVENT_FORWARD_END);
            /* BACKWARD fires unconditionally -- also around a truncated or
             * skipped backward (all-frozen model) -- so the per-call event
             * count stays a constant an external occurrence counter can rely
             * on. */
            odtHookFire(ODT_EVENT_BACKWARD_BEGIN);
            break;
        case REMAT_STEP_LOSS_BACKWARD:
            lossFns.backward(op.in[0], label, op.out, ex);
            if (sink != NULL) {
                sink(sinkCtx, m->numLayers, m->layers[m->numLayers - 1]->type, "lossgrad", op.out);
            }
            break;
        case REMAT_STEP_BACKWARD: {
            /* agrad@l = the gradient w.r.t. layer l's OUTPUT (the wire grad
             * entering its backward), matching the PyTorch forward-hook
             * activation.grad; so it fires before the backward. */
            if (sink != NULL) {
                sink(sinkCtx, st.layer, layer->type, "agrad", op.dy);
            }
            tensor_t *x = op.in[0];
#ifdef ODT_REMAT_VERIFY
            /* Strict W_dead: an input the step marks META (its backward does
             * not read it) gets a header without bytes on every plan, so a
             * read the read-set table denies crashes here instead of passing
             * on the NULL path and breaking under LIVENESS. */
            tensor_t dead;
            if (op.inMeta[0]) {
                dead = *op.in[0];
                dead.data = NULL;
                x = &dead;
            }
#endif
            /* op.dx[0] is NULL where the input needs no gradient: grads only. */
            layerFunctions[layer->type].backward(layer, x, op.dy, op.dx[0], ex);
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

    setLayersTrainingMode(m, false);
    odtHookFire(ODT_EVENT_BACKWARD_END);
    return trainingStats;
}

trainingStats_t *calculateGradsSequential(const model_t *m, lossConfig_t lossConfig,
                                          reduction_t forwardReduction, tensor_t *input,
                                          tensor_t *label, const trainingCall_t *call) {
    return calculateGradsImpl(m, lossConfig, forwardReduction, input, label, NULL, NULL, call);
}

trainingStats_t *tracedGrads(const model_t *m, lossConfig_t lossConfig,
                             reduction_t forwardReduction, tensor_t *input, tensor_t *label,
                             traceSink_t sink, void *ctx, const trainingCall_t *call) {
    return calculateGradsImpl(m, lossConfig, forwardReduction, input, label, sink, ctx, call);
}

static void traceModelParams(const model_t *m, const char *tag, bool wantGrad, traceSink_t sink,
                             void *ctx) {
    char phase[64];
    for (size_t i = 0; i < m->numLayers; i++) {
        parameter_t *w = NULL, *b = NULL;
        if (!layerParameters(m->layers[i], &w, &b)) {
            continue;
        }
        /* Frozen layers (#380): getGradFromParameter returns NULL (Task 1 elides
         * the grad tensor). The TraceApi contract promises sinks a borrowed VALID
         * tensor -- skip the call entirely rather than handing them NULL. */
        tensor_t *wt = wantGrad ? getGradFromParameter(w) : getParamFromParameter(w);
        if (wt != NULL) {
            snprintf(phase, sizeof(phase), "%s.weight", tag);
            sink(ctx, i, m->layers[i]->type, phase, wt);
        }
        if (b != NULL) {
            tensor_t *bt = wantGrad ? getGradFromParameter(b) : getParamFromParameter(b);
            if (bt != NULL) {
                snprintf(phase, sizeof(phase), "%s.bias", tag);
                sink(ctx, i, m->layers[i]->type, phase, bt);
            }
        }
    }
}

void traceModelWeights(const model_t *m, const char *tag, traceSink_t sink, void *ctx) {
    traceModelParams(m, tag, /*wantGrad=*/false, sink, ctx);
}

void traceModelGrads(const model_t *m, const char *tag, traceSink_t sink, void *ctx) {
    traceModelParams(m, tag, /*wantGrad=*/true, sink, ctx);
}
