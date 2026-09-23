#define SOURCE_FILE "har_mem_instrument"

#include "mem_instrument.h"

#include <stdlib.h>
#include <string.h>

#include "BatchView.h"
#include "Common.h"
#include "Layer.h"
#include "MaxPool1d.h"
#include "MemProfile.h"
#include "Optimizer.h"
#include "Tensor.h"
#include "param_gate.h"

/* ---- Analytic sums over the optimizer's trainable-parameter array --------- */

size_t memInstrumentParamBytes(optimizer_t *optim) {
    size_t total = 0;
    for (size_t i = 0; i < optim->sizeStates; i++) {
        total += calcBytesPerTensor(optim->parameter[i]->param);
    }
    return total;
}

size_t memInstrumentGradBytes(optimizer_t *optim) {
    size_t total = 0;
    for (size_t i = 0; i < optim->sizeStates; i++) {
        total += calcBytesPerTensor(optim->parameter[i]->grad);
    }
    return total;
}

size_t memInstrumentOptStateBytes(optimizer_t *optim) {
    if (optim->states == NULL) { /* momentumFactor == 0: no state exists (#308) */
        return 0;
    }
    size_t total = 0;
    for (size_t i = 0; i < optim->sizeStates; i++) {
        states_t *s = optim->states[i];
        for (size_t j = 0; j < s->statesPerParameter; j++) {
            total += calcBytesPerTensor(s->stateBuffers[j]);
        }
    }
    return total;
}

/* ---- HAR-specific analytic activation / IO model -------------------------- */

/* Per-sample output element count of every layer of the fixed HAR topology
 * (train_c.c / train_c_sym.c):
 *   input [C=9, L=128]
 *     conv1 9->16 K7 SAME   -> [16,128]   relu -> [16,128]   maxpool/2 -> [16,64]
 *     conv2 16->32 K5 SAME  -> [32,64]    relu -> [32,64]    maxpool/2 -> [32,32]
 *     conv3 32->64 K3 SAME  -> [64,32]    relu -> [64,32]    avgpool/32 -> [64,1]
 *     flatten -> [64]       linear 64->6 -> [6]   softmax -> [6]
 * Wire dtypes come from the caller's harWireProfile_t (spec §7.2); NULL =
 * FLOAT32 everywhere. SYM_WIRES=1 wires are SYM_INT32 = 4 B/elem, so the SYM
 * trainer also passes NULL. */
static const size_t HAR_LAYER_OUT_ELEMS_PER_SAMPLE[HAR_NUM_LAYERS] = {
    16 * 128, 16 * 128, 16 * 64, /* conv1, relu1, maxpool1 */
    32 * 64,  32 * 64,  32 * 32, /* conv2, relu2, maxpool2 */
    64 * 32,  64 * 32,  64 * 1,  /* conv3, relu3, avgpool  */
    64,       6,        6,       /* flatten, linear, softmax */
};

/* dx PRODUCED by layer k = layer k's INPUT elements. Slot 0 (conv1) is never
 * allocated (deepest trainable layer, #380 PR2); slot 11 stands for the CE
 * loss grad (6 elements) that coexists with linear's dx at the first
 * backward step (softmax's own dx is never produced under CrossEntropy). */
static const size_t HAR_LAYER_DX_ELEMS_PER_SAMPLE[HAR_NUM_LAYERS] = {
    9 * 128, 16 * 128, 16 * 128, /* conv1 (unused), relu1, pool1 */
    16 * 64, 32 * 64,  32 * 64,  /* conv2, relu2, pool2 */
    32 * 32, 64 * 32,  64 * 32,  /* conv3, relu3, pool3 */
    64,      64,       6,        /* flatten, linear, (loss grad) */
};

static const harWireProfile_t FLOAT_WIRE = {
    .present = true, .type = FLOAT32, .bits = 32, .numGroups = 0};

static harWireProfile_t profileAt(const harWireProfile_t *arr, size_t i) {
    return (arr == NULL) ? FLOAT_WIRE : arr[i];
}

static size_t wirePayload(harWireProfile_t p, size_t elems) {
    return p.present ? packedPayloadBytes(p.type, p.bits, elems) : 0;
}

static size_t wireMetadata(harWireProfile_t p) {
    return p.present ? packedMetadataBytes(p.type, p.numGroups) : 0;
}

size_t memInstrumentHarActivationBytes(size_t microBatch,
                                       const harWireProfile_t out[HAR_NUM_LAYERS]) {
    /* Sum of EVERY layer's forward output-tensor bytes for ONE micro-batch.
     * calculateGradsSequential allocates all forward activations up front
     * (initLayerOutputs) and frees them only AFTER the full backward pass
     * (deInitLayerOutputs), so every forward activation is concurrently live
     * during backprop.
     *
     * #321: this is the forward-wire sum ONLY — NOT the true activation peak.
     * During backprop the dx ping-pong (gradNext + gradCurr) coexists with these
     * wires; that transient is reported separately as dx_peak_b (see
     * memInstrumentHarDxPeakBytes), so the true concurrent wire peak is
     * activations_b + dx_peak_b. The per-op conversion scratch executeOp
     * allocates is on the STACK (VLAs), already captured by stack_peak_b — it does
     * NOT under-count this heap sum.
     *
     * microBatch is the CONCURRENT sample count, NOT the macro-batch:
     * trainingBatchDefault loops the macro-batch one sample at a time (loss.md:
     * dimensions[0]=B, today B=1) and accumulates grads at the optimizer, so
     * macro-batching does NOT multiply activation memory. Passing the macro-batch
     * here would over-count activations by that factor. */
    size_t bytes = 0;
    for (size_t i = 0; i < HAR_NUM_LAYERS; i++) {
        bytes += wirePayload(profileAt(out, i), HAR_LAYER_OUT_ELEMS_PER_SAMPLE[i] * microBatch);
    }
    return bytes;
}

size_t memInstrumentHarIoBytes(size_t microBatch) {
    /* One micro-batch of input [microBatch, 9, 128] float + one-hot labels
     * [microBatch, 6] float. The macro-batch is streamed one sample at a time
     * (see the activation note), so this is the micro-batch, not the loader batch. */
    return (size_t)(9 * 128 + 6) * microBatch * sizeof(float);
}

size_t memInstrumentPoolBackwardBytes(layer_t **model, size_t modelSize) {
    /* #321: each MaxPool layer pre-allocates an INT32 argmax-index tensor (shape ==
     * output shape) at build time; it lives the whole run and is required for the
     * backward pass, but lands in no params/grads/optstate/activations category —
     * it was hidden inside the measured params_grads_b delta. Walk the model and
     * sum it, dtype-aware via calcBytesPerTensor. */
    size_t total = 0;
    for (size_t i = 0; i < modelSize; i++) {
        if (model[i]->type == MAXPOOL1D) {
            total += calcBytesPerTensor(model[i]->config->maxPool1d->argmaxIndices);
        }
    }
    return total;
}

/* The dx ping-pong: at backward step i (linear=10 down to relu1=1) gradNext is
 * the dx produced by layer i+1 (the loss grad for i == 10) and gradCurr the dx
 * layer i produces; the two coexist. Peak = the pair with the largest payload
 * (FLOAT32: relu1/pool1 = 2 x 2048 x 4 = 16384 B, the pre-PR7 number). */
static size_t dxPairIndexOfPeak(size_t microBatch, const harWireProfile_t dx[HAR_NUM_LAYERS]) {
    size_t best = 1, bestBytes = 0;
    for (size_t i = 1; i <= 10; i++) {
        size_t bytes =
            wirePayload(profileAt(dx, i + 1), HAR_LAYER_DX_ELEMS_PER_SAMPLE[i + 1] * microBatch) +
            wirePayload(profileAt(dx, i), HAR_LAYER_DX_ELEMS_PER_SAMPLE[i] * microBatch);
        if (bytes > bestBytes) {
            bestBytes = bytes;
            best = i;
        }
    }
    return best;
}

size_t memInstrumentHarDxPeakBytes(size_t microBatch, const harWireProfile_t dx[HAR_NUM_LAYERS]) {
    /* #321: the transient dx ping-pong during backprop. CalculateGradsSequential
     * allocates gradCurr before freeing gradNext, so the two dx wires coexist with
     * all forward wires; this is the payload of the worst concurrent pair. */
    size_t i = dxPairIndexOfPeak(microBatch, dx);
    return wirePayload(profileAt(dx, i + 1), HAR_LAYER_DX_ELEMS_PER_SAMPLE[i + 1] * microBatch) +
           wirePayload(profileAt(dx, i), HAR_LAYER_DX_ELEMS_PER_SAMPLE[i] * microBatch);
}

size_t memInstrumentHarWireOverheadBytes(size_t microBatch,
                                         const harWireProfile_t out[HAR_NUM_LAYERS],
                                         const harWireProfile_t dx[HAR_NUM_LAYERS]) {
    size_t bytes = 0;
    for (size_t i = 0; i < HAR_NUM_LAYERS; i++) {
        bytes += wireMetadata(profileAt(out, i));
    }
    size_t i = dxPairIndexOfPeak(microBatch, dx);
    return bytes + wireMetadata(profileAt(dx, i + 1)) + wireMetadata(profileAt(dx, i));
}

/* ---- Stack high-water of one training step -------------------------------- */

static void memOneStepThunk(void *p) {
    memStepCtx_t *c = (memStepCtx_t *)p;
    optimizerFunctions_t fns = optimizerFunctions[c->optim->type];

    fns.zero(c->optim);
    /* The two [1, ...] views live in this frame exactly as they live in
     * trainingBatchDefault's on the real path, so the probe still measures
     * trainingRun's step (the sample itself is natural-shape, #152 PR3a). */
    batchView_t itemView;
    batchView_t labelView;
    trainingStats_t *stats = calculateGradsSequential(
        c->model, c->modelSize, c->lossConfig, REDUCTION_MEAN, batchViewOf(&itemView, c->input),
        batchViewOf(&labelView, c->label));
    freeTrainingStats(stats);
    /* No scaleOptimizerGradients: the macro-batch mean scale is a scalar grad
     * multiply that does not deepen the call stack; the step itself does.
     * optimizerStep, not fns.step: the probe measures trainingRun's path, and
     * trainingEpochDefault steps through that wrapper (one extra frame; #432). */
    optimizerStep(c->optim);
    fns.zero(c->optim);
}

size_t memInstrumentStackPeakBytes(memStepCtx_t *ctx, size_t stackBytes) {
    return measurePeakStackBytes(memOneStepThunk, ctx, stackBytes);
}

/* ---- Reconciliation + emit ------------------------------------------------ */

void memInstrumentFinalize(memReport_t *r) {
    if (r->storage_dtype == NULL) {
        PRINT_ERROR("memInstrumentFinalize: storage_dtype was never set by the trainer");
        exit(1);
    }
    r->mcu_total_b = r->params_b + r->group_overhead_b + r->grads_b + r->grad_overhead_b +
                     r->optstate_analytic_b + r->optstate_overhead_b + r->activations_b +
                     r->wire_overhead_b + r->io_b + r->pool_backward_b + r->dx_peak_b;
    /* Signed on purpose: a positive gap = unaccounted heap (dataset, dataloaders,
     * per-op scratch, bookkeeping); a negative gap would mean the analytic model
     * over-counts. RECORD it — never tune the categories to shrink it. */
    r->reconciliation_gap_b = (long)r->heap_peak_b - (long)r->mcu_total_b;
}

void memInstrumentEmitJson(FILE *f, const memReport_t *r) {
    fprintf(f, "{\"storage_dtype\": \"%s\", ", r->storage_dtype);
    if (strcmp(r->storage_dtype, "bfp") != 0) {
        fprintf(f, "\"sym_bits\": %d, ", r->sym_bits);
    }
    fprintf(
        f,
        "\"dataset_b\": %zu, \"params_grads_b\": %zu, \"optstate_b\": %zu, "
        "\"params_b\": %zu, \"group_overhead_b\": %zu, \"grads_b\": %zu, \"grad_overhead_b\": %zu, "
        "\"optstate_analytic_b\": %zu, \"optstate_overhead_b\": %zu, "
        "\"activations_b\": %zu, \"wire_overhead_b\": %zu, \"io_b\": %zu, "
        "\"pool_backward_b\": %zu, \"dx_peak_b\": %zu, \"mcu_total_b\": %zu, "
        "\"heap_peak_b\": %zu, \"stack_peak_b\": %zu, \"rss_peak_kb\": %zu, "
        "\"reconciliation_gap_b\": %ld}",
        r->dataset_b, r->params_grads_b, r->optstate_b, r->params_b, r->group_overhead_b,
        r->grads_b, r->grad_overhead_b, r->optstate_analytic_b, r->optstate_overhead_b,
        r->activations_b, r->wire_overhead_b, r->io_b, r->pool_backward_b, r->dx_peak_b,
        r->mcu_total_b, r->heap_peak_b, r->stack_peak_b, r->rss_peak_kb, r->reconciliation_gap_b);
}

void memInstrumentPrintReconciliation(const memReport_t *r) {
    fprintf(stdout, "RECONCILIATION heap_peak=%zu mcu_total=%zu gap=%ld\n", r->heap_peak_b,
            r->mcu_total_b, r->reconciliation_gap_b);
    fflush(stdout);
}
