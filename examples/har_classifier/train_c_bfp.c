#define SOURCE_FILE "har_classifier_train_c_bfp"

/* BFP epic #410 PR7 -- block-floating-point HAR conv classifier for the sweep
 * (spec docs/superpowers/specs/2026-09-14-bfp-pr7-sweep-integration-design.md).
 *
 *   - params: FLOAT32-init (#270) then requantizeTensorInPlace to BFP at
 *     (BFP_MANTISSA_BITS, BFP_EXPONENT_BITS); weights blocked per
 *     BFP_WEIGHT_BLOCK via resolveGroupShape (per-channel fallback when the
 *     size does not divide N -- conv1 at 32/64), biases per-tensor {1,0}.
 *   - wires: BFP_WIRE_BLOCK=float keeps FLOAT32 wires (arm A: the GEMM math
 *     still runs ARITH_BFP and stages the float operand per-tensor at the
 *     weight widths, spec §3.4); tensor|N builds ONE BFP template PER WIRE,
 *     forward and dx resolved independently by resolveWireShape (6-element
 *     head wires always fall back to per-tensor -- recorded in wires_resolved).
 *   - math: native = ARITH_BFP on the four GEMM slots; fq = those slots and
 *     Softmax's forward pinned ARITH_FLOAT32 (the GEMM fake-quant reference,
 *     NOT a whole-model twin: Relu/pools keep their BFP arms).
 *   - rounding (§3.5): OUT_WRITE rounds by the OPERATION's mode, so the
 *     carriers are the math slots: forward/weightGrad/biasGrad HALF_AWAY
 *     (deterministic inference + staging), propLossMath SR_HALF_AWAY under
 *     BFP_ROUNDING=sr (dx packs), grad/state templates and the optimizer
 *     write-back likewise. The CE loss-grad pack has no carrier and stays
 *     HALF_AWAY (documented exception §3.5.1).
 *   - grads / momentum: FLOAT32 by default; BFP_GRADS=1 / BFP_STATE=1 opt
 *     into per-tensor BFP storage (grouped BFP grads are rejected upstream).
 *   - optimizer: SGD-M via optimizerStep; updateMath FLOAT32 (#310).
 *
 * Knobs: BFP_MANTISSA_BITS BFP_EXPONENT_BITS BFP_WEIGHT_BLOCK BFP_WIRE_BLOCK
 * BFP_MATH BFP_GRADS BFP_STATE BFP_ROUNDING (examples/_shared/param_gate.h)
 * + LR MOMENTUM EPOCHS SEED SHUFFLE_SEED LR_SCHEDULE LR_MIN LOG_PATH
 * LOG_CODE_MOVEMENT (inherited). The SYM-only knobs are ignored with a WARN. */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "ArithmeticType.h"
#include "CalculateGradsSequential.h"
#include "Common.h"
#include "Conv1d.h"
#include "Conv1dApi.h"
#include "DataLoader.h"
#include "DataLoaderApi.h"
#include "FlattenApi.h"
#include "InferenceApi.h"
#include "Layer.h"
#include "LayerCommon.h"
#include "LayerQuant.h"
#include "Linear.h"
#include "LinearApi.h"
#include "LossFunction.h"
#include "LrScheduler.h"
#include "NPYLoaderApi.h"
#include "Optimizer.h"
#include "Pool1dApi.h"
#include "Quantization.h"
#include "QuantizationApi.h"
#include "RNG.h"
#include "ReluApi.h"
#include "Rounding.h"
#include "SgdApi.h"
#include "SoftmaxApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "TensorConversion.h"
#include "TraceApi.h"
#include "TrainingLoopApi.h"

#include "mem_instrument.h"
#include "param_gate.h"

#define BATCH 64 /* macro-batch: loader groups 64 samples per optimizer step */
/* Micro-batch = concurrent samples per forward/backward. The training loop
 * streams the macro-batch one sample at a time (loss.md B=1), so peak activation
 * memory is ONE sample's worth — this is what the analytic footprint must use. */
#define MICRO_BATCH 1
#define NUM_CLASSES 6

#define IN_CHANNELS 9
#define LEN_INPUT 128

#define C1_OUT 16
#define C1_K 7
#define C2_OUT 32
#define C2_K 5
#define C3_OUT 64
#define C3_K 3

/* 3 x (Conv1d + ReLU + Pool) + Flatten + Linear + Softmax = 12 layers */
#define MODEL_SIZE 12
_Static_assert(MODEL_SIZE == HAR_NUM_LAYERS, "HAR topology");

static dataset_t g_trainDataset;
static dataset_t g_valDataset;
static dataset_t g_testDataset;

/* Runtime config (env-overridable). */
static float g_lr = 0.01f;
static float g_momentum = 0.9f;
static int g_epochs = 50;
static unsigned g_seed = 1;
static unsigned g_shuffleSeed = 1;
static int g_useCosine = 0;         /* LR_SCHEDULE=cosine */
static float g_lrMin = 0.0f;        /* LR_MIN (cosine floor) */
static optimizer_t *g_optim = NULL; /* for per-epoch LR logging in epochCallback */

static bfpSweepConfig_t g_cfg;

/* Wire templates (spec §3.3): trainer-owned, one per forward wire (g_outQ)
 * and one per dx wire (g_dxQ, NULL where the run never allocates one);
 * layers never own them. Under BFP_WIRE_BLOCK=float every slot is g_floatQ. */
static quantization_t *g_floatQ = NULL;
static quantization_t *g_gradQ = NULL; /* per-tensor BFP grad template when BFP_GRADS=1 */
static quantization_t *g_outQ[MODEL_SIZE];
static quantization_t *g_dxQ[MODEL_SIZE];
static groupShape_t g_outShape[MODEL_SIZE];
static groupShape_t g_dxShape[MODEL_SIZE];

static const char *const kLayerNames[MODEL_SIZE] = {"conv1", "relu1",   "pool1",  "conv2",
                                                    "relu2", "pool2",   "conv3",  "relu3",
                                                    "pool3", "flatten", "linear", "softmax"};
/* Per-sample forward output elements of the fixed topology (mem_instrument.c). */
static const size_t kOutElems[MODEL_SIZE] = {2048, 2048, 1024, 2048, 2048, 1024,
                                             2048, 2048, 64,   64,   6,    6};
/* dx PRODUCED by layer k = its input elements; 0 = never allocated (conv1 is
 * the deepest trainable layer, #380 PR2; softmax is skipped by CrossEntropy's
 * fused backward). The CE loss grad (6 elements) clones softmax.out's template. */
static const size_t kDxElems[MODEL_SIZE] = {0,    2048, 2048, 1024, 2048, 2048,
                                            1024, 2048, 2048, 64,   64,   0};
static const bool kIsGemm[MODEL_SIZE] = {true, false, false, true,  false, false,
                                         true, false, false, false, true,  false};

static roundingMode_t trainingSideRounding(void) {
    return (g_cfg.rounding == BFP_ROUNDING_SR) ? SR_HALF_AWAY : HALF_AWAY;
}

static quantization_t *makeWireTemplate(groupShape_t gs) {
    if (g_cfg.wireMode == WIRE_BLOCK_FLOAT) {
        return g_floatQ;
    }
    if (gs.numGroups == 1) {
        return quantizationInitBfp(g_cfg.mantissaBits, g_cfg.exponentBits, HALF_AWAY);
    }
    return quantizationInitBfpGrouped(g_cfg.mantissaBits, g_cfg.exponentBits, HALF_AWAY,
                                      gs.numGroups, gs.groupSize);
}

static void buildWireTemplates(void) {
    g_floatQ = quantizationInitFloat();
    for (size_t i = 0; i < MODEL_SIZE; i++) {
        g_outShape[i] = resolveWireShape(kOutElems[i], g_cfg.wireMode, g_cfg.wireSize);
        g_outQ[i] = makeWireTemplate(g_outShape[i]);
        if (kDxElems[i] == 0) {
            g_dxShape[i] = (groupShape_t){.numGroups = 1, .groupSize = 0};
            g_dxQ[i] = NULL;
        } else {
            g_dxShape[i] = resolveWireShape(kDxElems[i], g_cfg.wireMode, g_cfg.wireSize);
            g_dxQ[i] = makeWireTemplate(g_dxShape[i]);
        }
    }
    if (g_cfg.bfpGrads) {
        g_gradQ =
            quantizationInitBfp(g_cfg.mantissaBits, g_cfg.exponentBits, trainingSideRounding());
    }
}

static void freeWireTemplates(void) {
    for (size_t i = 0; i < MODEL_SIZE; i++) {
        if (g_outQ[i] != g_floatQ) {
            freeQuantization(g_outQ[i]);
        }
        if (g_dxQ[i] != NULL && g_dxQ[i] != g_floatQ) {
            freeQuantization(g_dxQ[i]);
        }
    }
    if (g_gradQ != NULL) {
        freeQuantization(g_gradQ);
    }
    freeQuantization(g_floatQ);
}

/* One layer's profile (spec §3.3-3.5). layerQuantInitUniform derives the math
 * type AND copies the template's HALF_AWAY into every slot; the rounding
 * carriers are then set explicitly: forward/staging deterministic, dx packs
 * training-side. GEMM math is pinned by BFP_MATH regardless of wire dtype
 * (under float wires the derivation would silently yield FLOAT32). */
static layerQuant_t layerQuantFor(size_t i) {
    layerQuant_t lq;
    layerQuantInitUniform(&lq, g_outQ[i]);
    /* A layer with no dx wire never reads propLossQ; point it at the out
     * template so no float/BFP mix exists on paper. Flatten has no config at
     * all: its dx clones pool3.out's config (same N=64 -> same resolution). */
    lq.propLossQ = (g_dxQ[i] != NULL) ? g_dxQ[i] : g_outQ[i];
    lq.forwardMath.roundingMode = HALF_AWAY;
    lq.weightGradMath.roundingMode = HALF_AWAY;
    lq.biasGradMath.roundingMode = HALF_AWAY;
    lq.propLossMath.roundingMode = trainingSideRounding();
    if (kIsGemm[i]) {
        bool fq = (g_cfg.math == BFP_MATH_FQ);
        lq.forwardMath =
            (arithmetic_t){.type = fq ? ARITH_FLOAT32 : ARITH_BFP, .roundingMode = HALF_AWAY};
        lq.weightGradMath = lq.forwardMath;
        lq.biasGradMath = lq.forwardMath;
        lq.propLossMath = (arithmetic_t){.type = fq ? ARITH_FLOAT32 : ARITH_BFP,
                                         .roundingMode = trainingSideRounding()};
        lq.weightStorage = g_floatQ; /* #270: FLOAT32 init, requantized post-build */
        lq.biasStorage = g_floatQ;
        lq.weightGradStorage = g_cfg.bfpGrads ? g_gradQ : NULL;
        lq.biasGradStorage = g_cfg.bfpGrads ? g_gradQ : NULL;
        /* DYNAMIC_RESCALE: the only ACC mode that can never abort on a BFP
         * grad target (FIXED_SCALE aborts on mantissa overflow -- a coarse
         * config's FINDING must not become a crash). Also the required value
         * for a FLOAT32 target (executeOpValidateAccMode). */
        lq.weightGradAccMode = OUT_ACC_DYNAMIC_RESCALE;
        lq.biasGradAccMode = OUT_ACC_DYNAMIC_RESCALE;
    }
    if (i == 11 && g_cfg.math == BFP_MATH_FQ) {
        /* fq pins Softmax's FORWARD only (funnel-routed, accepts a BFP input);
         * its propLossMath stays derived -- CrossEntropy's fused backward
         * skips the softmax layer, so it is never reached. */
        lq.forwardMath = (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY};
    }
    return lq;
}

static float envFloat(const char *name, float dflt) {
    const char *v = getenv(name);
    return (v != NULL && v[0] != '\0') ? strtof(v, NULL) : dflt;
}
static int envInt(const char *name, int dflt) {
    const char *v = getenv(name);
    return (v != NULL && v[0] != '\0') ? (int)strtol(v, NULL, 10) : dflt;
}

static void reshapeItemsAddBatchDim(tensorArray_t *items) {
    for (size_t i = 0; i < items->size; ++i) {
        tensor_t *t = items->array[i];
        size_t oldRank = t->shape->numberOfDimensions;
        size_t newRank = oldRank + 1;

        size_t *newDims = reserveMemory(newRank * sizeof(size_t));
        size_t *newOrder = reserveMemory(newRank * sizeof(size_t));
        newDims[0] = 1;
        for (size_t d = 0; d < oldRank; ++d) {
            newDims[d + 1] = t->shape->dimensions[d];
        }
        for (size_t d = 0; d < newRank; ++d) {
            newOrder[d] = d;
        }

        freeReservedMemory(t->shape->dimensions);
        freeReservedMemory(t->shape->orderOfDimensions);
        t->shape->dimensions = newDims;
        t->shape->orderOfDimensions = newOrder;
        t->shape->numberOfDimensions = newRank;
    }
}

static tensorArray_t *buildOneHotLabels(tensorArray_t *intLabels) {
    tensorArray_t *out = reserveMemory(sizeof(tensorArray_t));
    tensor_t **arr = reserveMemory(intLabels->size * sizeof(tensor_t *));
    out->array = arr;
    out->size = intLabels->size;

    for (size_t i = 0; i < intLabels->size; ++i) {
        size_t *dims = reserveMemory(1 * sizeof(size_t));
        size_t *order = reserveMemory(1 * sizeof(size_t));
        dims[0] = NUM_CLASSES;
        order[0] = 0;
        shape_t *shape = reserveMemory(sizeof(shape_t));
        shape->dimensions = dims;
        shape->orderOfDimensions = order;
        shape->numberOfDimensions = 1;

        quantization_t *q = quantizationInitFloat();
        tensor_t *t = initTensor(shape, q, NULL);

        int32_t cls = ((int32_t *)intLabels->array[i]->data)[0];
        float *data = (float *)t->data;
        for (size_t c = 0; c < NUM_CLASSES; ++c) {
            data[c] = (c == (size_t)cls) ? 1.0f : 0.0f;
        }
        arr[i] = t;
    }
    return out;
}

static void initDataSets(void) {
    tensorArray_t *trainItems = npyLoad("examples/har_classifier/data/train_x.npy");
    tensorArray_t *trainLabelsRaw = npyLoad("examples/har_classifier/data/train_y.npy");
    reshapeItemsAddBatchDim(trainItems);
    g_trainDataset.items = trainItems;
    g_trainDataset.labels = buildOneHotLabels(trainLabelsRaw);

    tensorArray_t *valItems = npyLoad("examples/har_classifier/data/val_x.npy");
    tensorArray_t *valLabelsRaw = npyLoad("examples/har_classifier/data/val_y.npy");
    reshapeItemsAddBatchDim(valItems);
    g_valDataset.items = valItems;
    g_valDataset.labels = buildOneHotLabels(valLabelsRaw);

    tensorArray_t *testItems = npyLoad("examples/har_classifier/data/test_x.npy");
    tensorArray_t *testLabelsRaw = npyLoad("examples/har_classifier/data/test_y.npy");
    reshapeItemsAddBatchDim(testItems);
    g_testDataset.items = testItems;
    g_testDataset.labels = buildOneHotLabels(testLabelsRaw);
}

static sample_t *getTrainSample(size_t id) {
    return npyGetSample(&g_trainDataset, id);
}
static sample_t *getValSample(size_t id) {
    return npyGetSample(&g_valDataset, id);
}
static sample_t *getTestSample(size_t id) {
    return npyGetSample(&g_testDataset, id);
}
static size_t getTrainSize(void) {
    return g_trainDataset.items->size;
}
static size_t getValSize(void) {
    return g_valDataset.items->size;
}
static size_t getTestSize(void) {
    return g_testDataset.items->size;
}

static void buildModel(layer_t **model) {
    layerQuant_t lq[MODEL_SIZE];
    for (size_t i = 0; i < MODEL_SIZE; i++) {
        lq[i] = layerQuantFor(i);
    }
    model[0] = conv1dLayerInit(
        &(conv1dInit_t){
            .inChannels = IN_CHANNELS, .outChannels = C1_OUT, .kernelSize = C1_K, .padding = SAME},
        &lq[0]);
    model[1] = reluLayerInit(&lq[1]);
    model[2] = maxPool1dLayerInit(
        &(maxPool1dInit_t){
            .kernelSize = 2, .stride = 2, .inputChannels = C1_OUT, .inputLength = LEN_INPUT},
        &lq[2]);
    model[3] = conv1dLayerInit(
        &(conv1dInit_t){
            .inChannels = C1_OUT, .outChannels = C2_OUT, .kernelSize = C2_K, .padding = SAME},
        &lq[3]);
    model[4] = reluLayerInit(&lq[4]);
    model[5] = maxPool1dLayerInit(
        &(maxPool1dInit_t){
            .kernelSize = 2, .stride = 2, .inputChannels = C2_OUT, .inputLength = LEN_INPUT / 2},
        &lq[5]);
    model[6] = conv1dLayerInit(
        &(conv1dInit_t){
            .inChannels = C2_OUT, .outChannels = C3_OUT, .kernelSize = C3_K, .padding = SAME},
        &lq[6]);
    model[7] = reluLayerInit(&lq[7]);
    model[8] = avgPool1dLayerInit(
        &(avgPool1dInit_t){.kernelSize = LEN_INPUT / 4, .stride = LEN_INPUT / 4}, &lq[8]);
    model[9] = flattenLayerInit();
    model[10] =
        linearLayerInit(&(linearInit_t){.inFeatures = C3_OUT, .outFeatures = NUM_CLASSES}, &lq[10]);
    model[11] = softmaxLayerInit(&lq[11]);
}

static quantization_t *buildBfpWeightQuant(groupShape_t gs) {
    if (gs.numGroups == 1) {
        return quantizationInitBfp(g_cfg.mantissaBits, g_cfg.exponentBits, HALF_AWAY);
    }
    return quantizationInitBfpGrouped(g_cfg.mantissaBits, g_cfg.exponentBits, HALF_AWAY,
                                      gs.numGroups, gs.groupSize);
}

/* Weights -> BFP grouped per BFP_WEIGHT_BLOCK (resolveGroupShape on the
 * tensor's own N/outCh), biases -> per-tensor BFP; HALF_AWAY templates
 * (deterministic initial encode, §3.5). Every template is freed right after
 * its one requantizeTensorInPlace (which deep-clones it). */
static void requantizeParamsToBfp(layer_t **model) {
    quantization_t *biasQ = quantizationInitBfp(g_cfg.mantissaBits, g_cfg.exponentBits, HALF_AWAY);
    const size_t convIdx[3] = {0, 3, 6};
    for (size_t k = 0; k < 3; k++) {
        conv1dConfig_t *cfg = model[convIdx[k]]->config->conv1d;
        tensor_t *w = cfg->weights->param;
        groupShape_t gs =
            resolveGroupShape(calcNumberOfElementsByTensor(w), w->shape->dimensions[0],
                              g_cfg.weightMode, g_cfg.weightSize);
        quantization_t *weightQ = buildBfpWeightQuant(gs);
        requantizeTensorInPlace(w, weightQ);
        freeQuantization(weightQ);
        if (cfg->bias != NULL) {
            requantizeTensorInPlace(cfg->bias->param, biasQ);
        }
    }
    linearConfig_t *fc = model[10]->config->linear;
    tensor_t *lw = fc->weights->param;
    groupShape_t linGs =
        resolveGroupShape(calcNumberOfElementsByTensor(lw), lw->shape->dimensions[0],
                          g_cfg.weightMode, g_cfg.weightSize);
    quantization_t *linWeightQ = buildBfpWeightQuant(linGs);
    requantizeTensorInPlace(lw, linWeightQ);
    freeQuantization(linWeightQ);
    requantizeTensorInPlace(fc->bias->param, biasQ);
    freeQuantization(biasQ);
}

/* ---- Gates -------------------------------------------------------------- */

typedef struct paramGateCtx {
    bool isGrad;
    int count;
    int fails;
} paramGateCtx_t;

/* PARAM gate (spec §3.6 gate 3): every trainable weight/bias tensor is BFP at
 * (m, e), in the group SHAPE BFP_WEIGHT_BLOCK should have produced for that
 * specific tensor (weights: resolveGroupShape on the tensor's own N/outCh;
 * biases: always per-tensor {1,0}) -- an independent recheck of what
 * requantizeParamsToBfp built, via the same shared helper. GRAD gate (gate 4):
 * every trainable grad tensor is FLOAT32 (BFP_GRADS=0) or per-tensor BFP at
 * (m, e) (BFP_GRADS=1). The per-tensor check itself is
 * examples/_shared/param_gate.c (#417); this sink only filters, counts and
 * reports. */
static void paramGateSink(void *ctxVoid, size_t layerIdx, layerType_t layerType, const char *phase,
                          tensor_t *tensor) {
    if (layerType != LINEAR && layerType != CONV1D) {
        return;
    }
    paramGateCtx_t *ctx = ctxVoid;

    paramGateExpect_t expect;
    if (ctx->isGrad) {
        expect = g_cfg.bfpGrads ? (paramGateExpect_t){.type = BFP,
                                                      .bits = g_cfg.mantissaBits,
                                                      .exponentBits = g_cfg.exponentBits,
                                                      .shape = {.numGroups = 1, .groupSize = 0}}
                                : (paramGateExpect_t){.type = FLOAT32};
    } else {
        bool isBias = strstr(phase, ".bias") != NULL;
        groupShape_t shape;
        if (isBias) {
            shape = (groupShape_t){.numGroups = 1, .groupSize = 0};
        } else {
            size_t N = calcNumberOfElementsByTensor(tensor);
            size_t outCh = tensor->shape->dimensions[0];
            shape = resolveGroupShape(N, outCh, g_cfg.weightMode, g_cfg.weightSize);
        }
        expect = (paramGateExpect_t){.type = BFP,
                                     .bits = g_cfg.mantissaBits,
                                     .exponentBits = g_cfg.exponentBits,
                                     .shape = shape};
    }

    char msg[160];
    if (!paramGateCheck(tensor, &expect, msg, sizeof(msg))) {
        fprintf(stderr, "GATE FAIL: layer %zu %s %s\n", layerIdx, phase, msg);
        ctx->fails++;
        return;
    }
    ctx->count++;
}

/* STATE gate (new for BFP: state dtype is a sweep axis). Every momentum buffer
 * is FLOAT32 (BFP_STATE=0) or per-tensor BFP at (m, e) (BFP_STATE=1); 8 total. */
static int stateGate(optimizer_t *optim) {
    if (optim->states == NULL) {
        fprintf(stderr, "GATE FAIL: optimizer has no momentum state (MOMENTUM must be > 0)\n");
        return 2;
    }
    int count = 0, fails = 0;
    for (size_t i = 0; i < optim->sizeStates; i++) {
        states_t *s = optim->states[i];
        for (size_t j = 0; j < s->statesPerParameter; j++) {
            paramGateExpect_t expect =
                g_cfg.bfpState ? (paramGateExpect_t){.type = BFP,
                                                     .bits = g_cfg.mantissaBits,
                                                     .exponentBits = g_cfg.exponentBits,
                                                     .shape = {.numGroups = 1, .groupSize = 0}}
                               : (paramGateExpect_t){.type = FLOAT32};
            char msg[160];
            if (!paramGateCheck(s->stateBuffers[j], &expect, msg, sizeof(msg))) {
                fprintf(stderr, "GATE FAIL: momentum state %zu/%zu %s\n", i, j, msg);
                fails++;
            } else {
                count++;
            }
        }
    }
    if (fails != 0 || count != 8) {
        fprintf(stderr, "STATE GATE FAILED (ok=%d fails=%d, expected 8)\n", count, fails);
        return 2;
    }
    fprintf(stdout, "STATE GATE PASS: momentum=%s (8 checks)\n",
            g_cfg.bfpState ? "BFP per-tensor" : "FLOAT32");
    return 0;
}

typedef struct groupLogInfo {
    groupShape_t conv1, conv2, conv3, linear;
    size_t overheadBytes;
} groupLogInfo_t;

/* Reads the ACTUAL post-requantize group shape off each of the 8 trainable
 * param tensors -- single source of truth, no separate bookkeeping to drift
 * from what requantizeParamsToBfp actually built -- for the log's
 * "groups_resolved" (the 4 weight tensors) and "group_overhead_b" (all 8:
 * weights AND biases, since even a per-tensor {1,0} tensor carries one u8
 * exponent of metadata; spec-§7-mandatory honest accuracy-per-byte
 * accounting). */
static groupLogInfo_t computeGroupLogInfo(layer_t **model) {
    const size_t convIdx[3] = {0, 3, 6};
    groupShape_t convShapes[3];
    size_t totalGroups = 0;

    for (size_t k = 0; k < 3; k++) {
        conv1dConfig_t *cfg = model[convIdx[k]]->config->conv1d;
        qShapeView_t wv = viewQShape(cfg->weights->param->quantization);
        convShapes[k] = (groupShape_t){.numGroups = wv.numGroups, .groupSize = wv.groupSize};
        totalGroups += wv.numGroups;
        if (cfg->bias != NULL) {
            totalGroups += viewQShape(cfg->bias->param->quantization).numGroups;
        }
    }

    linearConfig_t *fc = model[10]->config->linear;
    qShapeView_t lwv = viewQShape(fc->weights->param->quantization);
    groupShape_t linearShape = {.numGroups = lwv.numGroups, .groupSize = lwv.groupSize};
    totalGroups += lwv.numGroups;
    totalGroups += viewQShape(fc->bias->param->quantization).numGroups;

    return (groupLogInfo_t){
        .conv1 = convShapes[0],
        .conv2 = convShapes[1],
        .conv3 = convShapes[2],
        .linear = linearShape,
        .overheadBytes = packedMetadataBytes(BFP, totalGroups),
    };
}

static void emitWiresResolved(FILE *f) {
    fputs("{", f);
    for (size_t i = 0; i < MODEL_SIZE; i++) {
        fprintf(f, "%s\"%s.out\": [%zu, %zu]", i == 0 ? "" : ", ", kLayerNames[i],
                g_outShape[i].numGroups, g_outShape[i].groupSize);
        if (kDxElems[i] != 0) {
            fprintf(f, ", \"%s.dx\": [%zu, %zu]", kLayerNames[i], g_dxShape[i].numGroups,
                    g_dxShape[i].groupSize);
        }
    }
    fputs("}", f);
}

#ifdef ODT_MEM_PROFILE
static size_t gradMetadataBytes(optimizer_t *o) {
    size_t bytes = 0;
    for (size_t i = 0; i < o->sizeStates; i++) {
        quantization_t *q = o->parameter[i]->grad->quantization;
        if (q->type != FLOAT32) {
            bytes += packedMetadataBytes(q->type, viewQShape(q).numGroups);
        }
    }
    return bytes;
}

static size_t stateMetadataBytes(optimizer_t *o) {
    if (o->states == NULL) {
        return 0;
    }
    size_t bytes = 0;
    for (size_t i = 0; i < o->sizeStates; i++) {
        for (size_t j = 0; j < o->states[i]->statesPerParameter; j++) {
            quantization_t *q = o->states[i]->stateBuffers[j]->quantization;
            if (q->type != FLOAT32) {
                bytes += packedMetadataBytes(q->type, viewQShape(q).numGroups);
            }
        }
    }
    return bytes;
}

/* Wire profiles for mem_instrument (spec §7.2), read off the templates the
 * trainer actually built; numGroups is derived from the wire's N exactly as
 * the allocators do. Slot dx[11] stands for the CE loss grad (6 elements),
 * which clones softmax.out's template. */
static harWireProfile_t profileFrom(const quantization_t *q, size_t elems) {
    if (q->type == FLOAT32) {
        return (harWireProfile_t){.present = true, .type = FLOAT32, .bits = 32, .numGroups = 0};
    }
    qShapeView_t v = viewQShape(q);
    return (harWireProfile_t){.present = true,
                              .type = BFP,
                              .bits = v.qBits,
                              .numGroups = (v.groupSize == 0) ? 1 : elems / v.groupSize};
}

static void fillWireProfiles(harWireProfile_t out[HAR_NUM_LAYERS],
                             harWireProfile_t dx[HAR_NUM_LAYERS]) {
    for (size_t i = 0; i < MODEL_SIZE; i++) {
        out[i] = profileFrom(g_outQ[i], kOutElems[i]);
        dx[i] = (kDxElems[i] == 0) ? (harWireProfile_t){.present = false}
                                   : profileFrom(g_dxQ[i], kDxElems[i]);
    }
    dx[11] = profileFrom(g_outQ[11], kOutElems[11]);
}
#endif /* ODT_MEM_PROFILE */

static FILE *g_log_file = NULL;
static int g_first_epoch = 1;
static struct timespec g_epoch_t0;
static float g_firstTrainLoss = -1.0f;
static float g_lastTrainLoss = -1.0f;

/* #279 code-movement instrumentation (opt-in via LOG_CODE_MOVEMENT, off by
 * default so the mem study / bit-parity runs are untouched). */
static int g_trackMovement = 0;
static uint8_t *g_wsnap = NULL; /* previous-epoch snapshot of all packed-BFP param bytes */
static size_t g_wsnapLen = 0;

/* Fraction of trainable packed-BFP param STORAGE bytes that changed since the
 * previous epoch. This is the direct #279 dead-zone signal: HALF_AWAY freezes
 * the codes once the FLOAT32 grad step goes sub-ULP (-> collapses to 0), while
 * seeded SR_HALF_AWAY keeps dithering them (-> stays > 0). Byte-granular (a
 * packed byte holds several sub-byte mantissas) so it answers "did any code in this
 * byte move" -- exact for the frozen-vs-moving question, which is all the
 * mechanism claim needs. Returns -1 on the first call (baseline epoch, no prior
 * snapshot to diff against). */
static double codeMovementFraction(void) {
    size_t total = 0;
    for (size_t i = 0; i < g_optim->sizeStates; i++) {
        tensor_t *w = g_optim->parameter[i]->param;
        total += calcNumberOfBytesForData(w->quantization, calcNumberOfElementsByTensor(w));
    }
    if (total == 0) {
        return -1.0;
    }
    int firstCall = (g_wsnap == NULL);
    if (firstCall) {
        g_wsnap = reserveMemory(total);
        g_wsnapLen = total;
    }
    size_t changed = 0, off = 0;
    for (size_t i = 0; i < g_optim->sizeStates; i++) {
        tensor_t *w = g_optim->parameter[i]->param;
        size_t nb = calcNumberOfBytesForData(w->quantization, calcNumberOfElementsByTensor(w));
        const uint8_t *cur = (const uint8_t *)w->data;
        for (size_t b = 0; b < nb && off + b < g_wsnapLen; b++) {
            if (!firstCall && cur[b] != g_wsnap[off + b]) {
                changed++;
            }
            g_wsnap[off + b] = cur[b];
        }
        off += nb;
    }
    return firstCall ? -1.0 : (double)changed / (double)total;
}

static void epochCallback(epochInfo_t info, epochStats_t evalStats) {
    struct timespec t1;
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double wall_s =
        (double)(t1.tv_sec - g_epoch_t0.tv_sec) + (double)(t1.tv_nsec - g_epoch_t0.tv_nsec) * 1e-9;

    if (g_firstTrainLoss < 0.0f) {
        g_firstTrainLoss = info.trainLoss;
    }
    g_lastTrainLoss = info.trainLoss;

    if (g_log_file != NULL) {
        if (!g_first_epoch) {
            fprintf(g_log_file, ",\n");
        }
        fprintf(g_log_file,
                "    {\"epoch\": %zu, \"step_losses\": [], \"train_loss\": %.6f, "
                "\"val_loss\": %.6f, \"val_acc\": %.6f, \"wall_s\": %.4f, \"lr\": %.8f",
                info.epoch, (double)info.trainLoss, (double)evalStats.loss,
                (double)evalStats.accuracy, wall_s,
                (double)optimizerFunctions[g_optim->type].getLr(g_optim));
        if (g_trackMovement) {
            fprintf(g_log_file, ", \"codes_changed_frac\": %.6f", codeMovementFraction());
        }
        fprintf(g_log_file, "}");
        fflush(g_log_file);
    }
    g_first_epoch = 0;

    fprintf(stdout, "epoch %zu: train_loss=%.4f val_loss=%.4f val_acc=%.4f wall_s=%.2f\n",
            info.epoch, (double)info.trainLoss, (double)evalStats.loss, (double)evalStats.accuracy,
            wall_s);
    fflush(stdout);

    clock_gettime(CLOCK_MONOTONIC, &g_epoch_t0);
}

/* Spec §3.6 gate 2: every GEMM product op and every BFP sum op of the fixed
 * topology, checked with the kernels' own segment rule BEFORE the dataset
 * loads. run 0 = per-tensor operand; a FLOAT32 operand stages per-tensor at
 * the weight widths (§5.4 Decision 1) -> run 0 as well. Skipped under fq (no
 * int32 partials). The kernel guards stay authoritative (defense in depth). */
static size_t runOf(groupShape_t gs) {
    return gs.groupSize; /* 0 for per-tensor, the guards' convention */
}

static void headroomPreflight(void) {
    if (g_cfg.math == BFP_MATH_FQ) {
        return;
    }
    uint8_t m = g_cfg.mantissaBits;
    /* Weight runs from the same policy the requantize will apply. */
    size_t wRun[4] = {runOf(resolveGroupShape(1008, 16, g_cfg.weightMode, g_cfg.weightSize)),
                      runOf(resolveGroupShape(2560, 32, g_cfg.weightMode, g_cfg.weightSize)),
                      runOf(resolveGroupShape(6144, 64, g_cfg.weightMode, g_cfg.weightSize)),
                      runOf(resolveGroupShape(384, 6, g_cfg.weightMode, g_cfg.weightSize))};
    bool bfpWires = (g_cfg.wireMode != WIRE_BLOCK_FLOAT);
    /* input-wire run per GEMM: dataset input (always staged), pool1.out, pool2.out, flatten.out */
    size_t inRun[4] = {0, bfpWires ? runOf(g_outShape[2]) : 0, bfpWires ? runOf(g_outShape[5]) : 0,
                       bfpWires ? runOf(g_outShape[9]) : 0};
    /* dY entering each GEMM's backward = dx produced by the layer above:
     * relu1.dx, relu2.dx, relu3.dx, and the loss grad (clones softmax.out). */
    size_t dyRun[4] = {bfpWires ? runOf(g_dxShape[1]) : 0, bfpWires ? runOf(g_dxShape[4]) : 0,
                       bfpWires ? runOf(g_dxShape[7]) : 0, bfpWires ? runOf(g_outShape[11]) : 0};
    static const size_t fwdK[4] = {63, 80, 96, 64};   /* inCh*K, inFeatures */
    static const size_t wgradK[4] = {128, 64, 32, 1}; /* L of the output, batch 1 for linear */
    static const size_t dxK[4] = {0, 160, 192, 6};    /* outCh*K; conv1 has no dx */
    static const char *const names[4] = {"conv1", "conv2", "conv3", "linear"};
    for (size_t g = 0; g < 4; g++) {
        if (!bfpBlockHeadroomFits(m, m, wRun[g], inRun[g], fwdK[g])) {
            fprintf(stderr,
                    "PREFLIGHT FAIL: %s forward -- (m=%u, weight block, input block) would "
                    "overflow the int32 block partial over K=%zu (#227 headroom)\n",
                    names[g], m, fwdK[g]);
            exit(1);
        }
        if (!bfpBlockHeadroomFits(m, m, dyRun[g], inRun[g], wgradK[g])) {
            fprintf(stderr, "PREFLIGHT FAIL: %s weightGrad -- headroom over L=%zu\n", names[g],
                    wgradK[g]);
            exit(1);
        }
        if (dxK[g] != 0 && !bfpBlockHeadroomFits(m, m, dyRun[g], wRun[g], dxK[g])) {
            fprintf(stderr, "PREFLIGHT FAIL: %s dx -- headroom over outCh*K=%zu\n", names[g],
                    dxK[g]);
            exit(1);
        }
        if (bfpWires && !bfpSumHeadroomFits(m, dyRun[g], wgradK[g])) {
            fprintf(stderr, "PREFLIGHT FAIL: %s biasGrad -- sum headroom over L=%zu\n", names[g],
                    wgradK[g]);
            exit(1);
        }
    }
    if (bfpWires && !bfpSumHeadroomFits(m, runOf(g_outShape[7]), 32)) {
        fprintf(stderr, "PREFLIGHT FAIL: avgpool forward -- sum headroom over K=32\n");
        exit(1);
    }
    fprintf(stdout, "PREFLIGHT PASS: headroom ok for m=%u at every HAR reduction\n", m);
}

int main(void) {
    const char *cfgErr = bfpSweepConfigFromEnv(&g_cfg);
    if (cfgErr != NULL) {
        fprintf(stderr, "%s\n", cfgErr);
        exit(1);
    }
    static const struct {
        unsigned bit;
        const char *name;
    } legacyKnobs[] = {
        {LEGACY_KNOB_SYM_BITS, "SYM_BITS"},
        {LEGACY_KNOB_SYM_WIRES, "SYM_WIRES"},
        {LEGACY_KNOB_WEIGHT_DTYPE, "WEIGHT_DTYPE"},
        {LEGACY_KNOB_GROUP_MODE, "GROUP_MODE"},
        {LEGACY_KNOB_GROUP_SIZE, "GROUP_SIZE"},
        {LEGACY_KNOB_SYM_ROUNDING, "SYM_ROUNDING"},
        {LEGACY_KNOB_ODTS_ROUNDTRIP, "ODTS_ROUNDTRIP"},
    };
    for (size_t i = 0; i < sizeof(legacyKnobs) / sizeof(legacyKnobs[0]); i++) {
        if (g_cfg.ignoredLegacyKnobs & legacyKnobs[i].bit) {
            fprintf(stderr, "WARN: %s is a SYM-trainer knob and is IGNORED by the BFP trainer\n",
                    legacyKnobs[i].name);
        }
    }
    g_lr = envFloat("LR", g_lr);
    g_momentum = envFloat("MOMENTUM", g_momentum);
    g_epochs = envInt("EPOCHS", g_epochs);
    g_seed = (unsigned)envInt("SEED", (int)g_seed);
    g_shuffleSeed = (unsigned)envInt("SHUFFLE_SEED", (int)g_shuffleSeed);
    g_trackMovement = envInt("LOG_CODE_MOVEMENT", 0);
    /* run_matrix passes LR_SCHEDULE=none explicitly (every knob in the name);
     * the SYM trainer only knew "cosine", so accept "none" here. */
    const char *schedEnv = getenv("LR_SCHEDULE");
    if (schedEnv != NULL && schedEnv[0] != '\0' && strcmp(schedEnv, "none") != 0) {
        if (strcmp(schedEnv, "cosine") != 0) {
            fprintf(stderr, "LR_SCHEDULE=%s not supported (only: none, cosine)\n", schedEnv);
            exit(1);
        }
        g_useCosine = 1;
    }
    g_lrMin = envFloat("LR_MIN", g_lrMin);
    const char *logPath = getenv("LOG_PATH");
    const char *mathStr = (g_cfg.math == BFP_MATH_FQ) ? "fq" : "native";
    const char *roundingStr = (g_cfg.rounding == BFP_ROUNDING_DET) ? "det" : "sr";

    fprintf(stdout,
            "CONFIG mantissa_bits=%u exponent_bits=%u weight_block=%s wire_block=%s bfp_math=%s "
            "bfp_grads=%d bfp_state=%d bfp_rounding=%s lr=%.5f momentum=%.3f epochs=%d seed=%u "
            "shuffle_seed=%u\n",
            g_cfg.mantissaBits, g_cfg.exponentBits, g_cfg.weightBlockStr, g_cfg.wireBlockStr,
            mathStr, (int)g_cfg.bfpGrads, (int)g_cfg.bfpState, roundingStr, (double)g_lr,
            (double)g_momentum, g_epochs, g_seed, g_shuffleSeed);

#ifdef ODT_MEM_PROFILE
    /* Reset the heap counter to 0 BEFORE the first allocation so dataset_b is
     * counted from a clean baseline; env reads / printf above allocate nothing
     * via reserveMemory. */
    memProfileReset();
#endif

    buildWireTemplates(); /* resolves every wire; the preflight reads the shapes */
    headroomPreflight();

    initDataSets();

#ifdef ODT_MEM_PROFILE
    size_t markDataset = memProfileMark(); /* dataset_b */
#endif

    layer_t *model[MODEL_SIZE];
    rngSetSeed(g_seed);
#ifdef ODT_MEM_PROFILE
    size_t markBeforeModel = memProfileMark();
#endif
    buildModel(model);
    requantizeParamsToBfp(model);
#ifdef ODT_MEM_PROFILE
    size_t markAfterModel = memProfileMark(); /* params_grads_b = delta */
#endif

    /* ---- Gate: param + grad storage dtypes -------------------------------- */
    paramGateCtx_t wCtx = {.isGrad = false, .count = 0, .fails = 0};
    traceModelWeights(model, MODEL_SIZE, "gate", paramGateSink, &wCtx);
    paramGateCtx_t gCtx = {.isGrad = true, .count = 0, .fails = 0};
    traceModelGrads(model, MODEL_SIZE, "gate", paramGateSink, &gCtx);
    /* 4 trainable layers x (weight + bias) = 8 each. */
    if (wCtx.fails != 0 || gCtx.fails != 0 || wCtx.count != 8 || gCtx.count != 8) {
        fprintf(stderr, "GATES FAILED (weight ok=%d fails=%d; grad ok=%d fails=%d)\n", wCtx.count,
                wCtx.fails, gCtx.count, gCtx.fails);
        return 2;
    }
    fprintf(stdout, "GATES PASS: weights+bias=BFP m%u/e%u grads=%s (8 param + 8 grad checks)\n",
            g_cfg.mantissaBits, g_cfg.exponentBits, g_cfg.bfpGrads ? "BFP per-tensor" : "FLOAT32");
    fflush(stdout);

    dataLoader_t *trainLoader = dataLoaderInit(getTrainSample, getTrainSize, BATCH, NULL, NULL,
                                               /*shuffle*/ true, g_shuffleSeed, /*dropLast*/ true);
    dataLoader_t *valLoader = dataLoaderInit(getValSample, getValSize, 1, NULL, NULL,
                                             /*shuffle*/ false, 0, /*dropLast*/ true);
    dataLoader_t *testLoader = dataLoaderInit(getTestSample, getTestSize, 1, NULL, NULL,
                                              /*shuffle*/ false, 0, /*dropLast*/ true);

    /* ---- Gate: sane initial loss (~ln(6)=1.7918 for 6-class near-uniform) -- */
    epochStats_t initStats = evaluationEpochWithMetrics(model, MODEL_SIZE, CROSS_ENTROPY, valLoader,
                                                        inferenceWithLoss, REDUCTION_MEAN);
    fprintf(stdout, "initial_val_loss=%.6f initial_val_acc=%.6f (expected ~%.4f)\n",
            (double)initStats.loss, (double)initStats.accuracy, log(6.0));
    fflush(stdout);
    if (!(fabs((double)initStats.loss - log(6.0)) < 0.25)) {
        fprintf(stderr, "GATE FAIL: initial val loss %.6f not near ln(6)=%.4f\n",
                (double)initStats.loss, log(6.0));
        return 2;
    }

    /* SGD-M routes through executeOp (#284): per-target dtype dispatch is a
     * funnel property, so the BFP weights round-trip BFP<->FLOAT32 each step,
     * write-back rounding OPTIMIZER-owned (BFP_ROUNDING -> setter below, #279),
     * while grad and momentum state are read/written in their own storage
     * dtype (BFP_GRADS / BFP_STATE). Update arithmetic itself is FLOAT32 via
     * updateMath. */
#ifdef ODT_MEM_PROFILE
    size_t markBeforeOpt = memProfileMark();
#endif
    quantization_t *momentumQ =
        g_cfg.bfpState
            ? quantizationInitBfp(g_cfg.mantissaBits, g_cfg.exponentBits, trainingSideRounding())
            : quantizationInitFloat();
    optimizer_t *sgd =
        sgdMCreateOptim(g_lr, g_momentum, /*weightDecay*/ 0.0f, model, MODEL_SIZE, momentumQ,
                        (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});
    /* ONE call covers the param write-back AND the momentum-state write-back
     * (Sgd.c runs both update ops with writeBackRounding as the op rounding). */
    optimizerSetWriteBackRounding(sgd, trainingSideRounding());
#ifdef ODT_MEM_PROFILE
    size_t markAfterOpt = memProfileMark(); /* optstate_b = delta */
#endif
    if (stateGate(sgd) != 0) {
        return 2;
    }
    g_optim = sgd;
    lrScheduler_t cosineSched;
    lrScheduler_t *sched = NULL;
    if (g_useCosine && g_epochs < 1) {
        fprintf(stderr, "LR_SCHEDULE=cosine requires EPOCHS >= 1\n");
        exit(1);
    }
    if (g_useCosine) {
        /* tMax = full run length: one half-cosine from g_lr down to g_lrMin.
         * The tail deliberately drives per-step updates below one BFP level —
         * SR_HALF_AWAY keeps them alive in expectation (#279); see README. */
        cosineAnnealingLrInit(&cosineSched, sgd, (size_t)g_epochs, g_lrMin);
        sched = &cosineSched;
    }

    groupLogInfo_t groupInfo = computeGroupLogInfo(model);

    if (logPath != NULL && logPath[0] != '\0') {
        g_log_file = fopen(logPath, "w");
        if (g_log_file != NULL) {
            fprintf(g_log_file,
                    "{\n  \"impl\": \"c-bfp\", \"example\": \"har_classifier\",\n"
                    "  \"config\": {\"epochs\": %d, \"batch\": %d, \"lr\": %.6f, "
                    "\"momentum\": %.6f, \"seed\": %u, \"shuffle_seed\": %u, "
                    "\"lr_schedule\": \"%s\", \"lr_min\": %.6f, "
                    "\"weight_dtype\": \"bfp\", \"mantissa_bits\": %u, \"exponent_bits\": %u, "
                    "\"weight_block\": \"%s\", \"wire_block\": \"%s\", \"bfp_math\": \"%s\", "
                    "\"bfp_grads\": %d, \"bfp_state\": %d, \"bfp_rounding\": \"%s\", "
                    "\"groups_resolved\": {\"conv1\": [%zu, %zu], \"conv2\": [%zu, %zu], "
                    "\"conv3\": [%zu, %zu], \"linear\": [%zu, %zu]}, ",
                    g_epochs, BATCH, (double)g_lr, (double)g_momentum, g_seed, g_shuffleSeed,
                    g_useCosine ? "cosine" : "none", (double)g_lrMin, g_cfg.mantissaBits,
                    g_cfg.exponentBits, g_cfg.weightBlockStr, g_cfg.wireBlockStr, mathStr,
                    (int)g_cfg.bfpGrads, (int)g_cfg.bfpState, roundingStr,
                    groupInfo.conv1.numGroups, groupInfo.conv1.groupSize, groupInfo.conv2.numGroups,
                    groupInfo.conv2.groupSize, groupInfo.conv3.numGroups, groupInfo.conv3.groupSize,
                    groupInfo.linear.numGroups, groupInfo.linear.groupSize);
            if (g_cfg.wireMode != WIRE_BLOCK_FLOAT) {
                fputs("\"wires_resolved\": ", g_log_file);
                emitWiresResolved(g_log_file);
                fputs(", ", g_log_file);
            }
            fprintf(g_log_file, "\"group_overhead_b\": %zu},\n  \"epochs\": [\n",
                    groupInfo.overheadBytes);
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &g_epoch_t0);

    trainingRunResult_t result = trainingRun(
        model, MODEL_SIZE,
        (lossConfig_t){
            .funcType = CROSS_ENTROPY, .backwardReduction = REDUCTION_MEAN, .classWeights = NULL},
        trainLoader, valLoader, sgd, g_epochs, calculateGradsSequential, inferenceWithLoss,
        &(trainingRunOptions_t){.lrScheduler = sched, .callback = epochCallback});
    (void)result;

    epochStats_t testStats = evaluationEpochWithMetrics(
        model, MODEL_SIZE, CROSS_ENTROPY, testLoader, inferenceWithLoss, REDUCTION_MEAN);

    fprintf(stdout, "FINAL test_loss=%.4f test_acc=%.4f\n", (double)testStats.loss,
            (double)testStats.accuracy);
    fflush(stdout);

#ifdef ODT_MEM_PROFILE
    /* Honest per-run memory breakdown. The stack probe below runs one REAL
     * training step (mutating model + momentum), but no evaluation / output
     * follows, so the mutation is inert. */
    memReport_t report = {0};
    report.sym_bits = -1;
    report.storage_dtype = "bfp";
    report.dataset_b = markDataset;
    report.params_grads_b = markAfterModel - markBeforeModel;
    report.optstate_b = markAfterOpt - markBeforeOpt;
    report.params_b = memInstrumentParamBytes(sgd);
    report.grads_b = memInstrumentGradBytes(sgd);
    report.optstate_analytic_b = memInstrumentOptStateBytes(sgd);
    report.io_b = memInstrumentHarIoBytes(MICRO_BATCH);
    report.pool_backward_b = memInstrumentPoolBackwardBytes(model, MODEL_SIZE);
    harWireProfile_t outProf[HAR_NUM_LAYERS], dxProf[HAR_NUM_LAYERS];
    fillWireProfiles(outProf, dxProf);
    report.activations_b = memInstrumentHarActivationBytes(MICRO_BATCH, outProf);
    report.dx_peak_b = memInstrumentHarDxPeakBytes(MICRO_BATCH, dxProf);
    report.wire_overhead_b = memInstrumentHarWireOverheadBytes(MICRO_BATCH, outProf, dxProf);
    report.group_overhead_b = groupInfo.overheadBytes;
    report.grad_overhead_b = gradMetadataBytes(sgd);
    report.optstate_overhead_b = stateMetadataBytes(sgd);

    sample_t *stepSample = getTrainSample(0);
    memStepCtx_t stepCtx = {
        .model = model,
        .modelSize = MODEL_SIZE,
        .lossConfig = (lossConfig_t){.funcType = CROSS_ENTROPY,
                                     .backwardReduction = REDUCTION_MEAN,
                                     .classWeights = NULL},
        .input = stepSample->item,
        .label = stepSample->label,
        .optim = sgd,
    };
    report.stack_peak_b = memInstrumentStackPeakBytes(&stepCtx, 1u << 20);
    freeSample(stepSample);

    report.heap_peak_b = memProfilePeakBytes();
    report.rss_peak_kb = memProfileRssPeakKb();
    memInstrumentFinalize(&report);
    memInstrumentPrintReconciliation(&report);
#endif

    if (g_log_file != NULL) {
        fprintf(g_log_file,
                "\n  ],\n  \"final\": {\"test_loss\": %.6f, \"test_acc\": %.6f, "
                "\"test_auc\": null}",
                (double)testStats.loss, (double)testStats.accuracy);
#ifdef ODT_MEM_PROFILE
        fprintf(g_log_file, ",\n  \"memory\": ");
        memInstrumentEmitJson(g_log_file, &report);
#endif
        fprintf(g_log_file, "\n}\n");
        fclose(g_log_file);
    }

    /* ---- Convergence DIAGNOSTIC (advisory, never fatal) ------------------- */
    /* Whether train loss descends is the very quantity this sweep MEASURES: at
     * coarse widths (e.g. m=4) a config may legitimately fail to descend, and
     * that is a FINDING to record, not a run to crash (integrity rule). This is
     * a printed diagnostic only — per-epoch train_loss is already in the JSON,
     * and compare_memory.py reports convergence as k/N across seeds. The fatal
     * build-sanity gates are the dtype checks + the initial-loss gate above;
     * those still hard-fail on a genuinely broken build. */
    fprintf(stdout, "train_loss first=%.6f last=%.6f\n", (double)g_firstTrainLoss,
            (double)g_lastTrainLoss);
    if (g_epochs < 2) {
        /* first/last are per-EPOCH means: with one epoch they are the same
         * number by construction, so the comparison below would always WARN
         * (bit the CI stack-watermark probe, which runs EPOCHS=1). */
        fprintf(stdout, "CONVERGENCE SKIP: single-epoch run — first/last epoch means are "
                        "identical by construction\n");
    } else if (g_lastTrainLoss < g_firstTrainLoss) {
        fprintf(stdout, "CONVERGENCE OK: train loss decreased (%.6f -> %.6f)\n",
                (double)g_firstTrainLoss, (double)g_lastTrainLoss);
    } else {
        fprintf(stderr,
                "CONVERGENCE WARN: train loss did not decrease (first=%.6f last=%.6f) — BFP "
                "m%u/e%u may be too coarse to descend at base_lr=%.4f lr_schedule=%s (recorded, "
                "not fatal)\n",
                (double)g_firstTrainLoss, (double)g_lastTrainLoss, g_cfg.mantissaBits,
                g_cfg.exponentBits, (double)g_lr, g_useCosine ? "cosine" : "none");
    }

    freeWireTemplates();
    return 0;
}
