#define SOURCE_FILE "har_classifier_train_c_sym"

/* SPIKE — SYM-quantized-WEIGHTS HAR conv classifier (de-risking a config the
 * framework has never run: trainable params stored as packed sub-byte SYM).
 *
 * Config (see .superpowers/sdd/sym-spike-report.md):
 *   - weight storage + bias storage = packed SYM@SYM_BITS (the sub-byte `SYM`
 *     dtype, NOT the SYM_INT32 int32 container). Factories can only init
 *     FLOAT32-native param storage (#270 requireFloat32 gate), so the params
 *     are built FLOAT32/Kaiming and requantized in place to SYM@SYM_BITS after
 *     construction (requantizeTensorInPlace — the mixed_width_mlp dance,
 *     retargeted from SYM_INT32 to packed SYM).
 *   - conv/linear forward compute = ARITH_SYM_INT32: the executeOp prologue
 *     converts each operand to the SYM_INT32 arithmetic form. For the packed
 *     SYM weight/bias that is conversionMatrix[SYM][SYM_INT32]
 *     (convertSymTensorToSymInt32Tensor); for the FLOAT32 activation wire it is
 *     convertFloatTensorToSymInt32Tensor. Restored to FLOAT32 at the producer
 *     (outputQ) by the OUT_WRITE epilogue.
 *   - ALL backward compute FLOAT32 (weightGradMath/biasGradMath/propLossMath)
 *     and grad storage FLOAT32 (weightGradStorage/biasGradStorage left NULL =
 *     per-layer FLOAT32 default). No sub-byte packing on the grad side here.
 *   - all activation + dx wires FLOAT32.
 *   - optimizer: SGD-M, routed through executeOp like every other op (#284) --
 *     per-target dtype dispatch is a funnel property, not an optimizer-level
 *     switch: the prologue converts operands to the update arithmetic and the
 *     OUT_WRITE epilogue requants rawOut into each TARGET tensor's own dtype,
 *     so the packed-SYM weight round-trips SYM->FLOAT32->SYM every step.
 *     Update arithmetic itself = FLOAT32 via the `updateMath` knob (fail-fast
 *     on anything else, #310); write-back rounding is OPTIMIZER-owned (#279):
 *     the factory defaults to seeded SR_HALF_AWAY, `SYM_ROUNDING=det` opts
 *     out to deterministic HALF_AWAY via optimizerSetWriteBackRounding.
 *     Param storage qConfigs stay HALF_AWAY (deterministic init encode +
 *     serialization truth).
 *
 * SYM_BITS / SEED / LR / MOMENTUM / EPOCHS / LOG_PATH are env-overridable. */

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
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
#include "StateDictApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "TensorConversion.h"
#include "TraceApi.h"
#include "TrainingLoopApi.h"

#include "mem_instrument.h"

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

static dataset_t g_trainDataset;
static dataset_t g_valDataset;
static dataset_t g_testDataset;

/* Runtime config (env-overridable). */
static float g_lr = 0.01f;
static float g_momentum = 0.9f;
static int g_epochs = 50;
static unsigned g_seed = 1;
static unsigned g_shuffleSeed = 1;
static int g_symBits = 12;
static int g_symWires = 0;          /* SYM_WIRES=1 -> full-SYM wires (#206 acceptance) */
static int g_useCosine = 0;         /* LR_SCHEDULE=cosine */
static float g_lrMin = 0.0f;        /* LR_MIN (cosine floor) */
static optimizer_t *g_optim = NULL; /* for per-epoch LR logging in epochCallback */

/* Group-quant PR5 axis (#300): GROUP_MODE/GROUP_SIZE/WEIGHT_DTYPE env knobs.
 * Enum names carry a _SWEEP/_MODE_/_DTYPE_ prefix to avoid colliding with
 * qtype_t's own SYM/ASYM enumerators (Quantization.h) -- C enums share one
 * flat namespace. */
typedef enum groupModeSweep {
    GROUP_MODE_TENSOR, /* default: per-tensor, today's behavior (byte-identical) */
    GROUP_MODE_CHANNEL,
    GROUP_MODE_SIZE
} groupModeSweep_t;
typedef enum weightDtypeSweep { WEIGHT_DTYPE_SYM, WEIGHT_DTYPE_ASYM } weightDtypeSweep_t;

/* Resolved {numGroups, groupSize} for one weight tensor -- the shape grammar
 * from Quantization.h: per-tensor = {1,0}; grouped = {k>1, g>0, k*g==N};
 * {1,N} is never valid and must never be emitted by resolveGroupShape below. */
typedef struct groupShape {
    size_t numGroups;
    size_t groupSize;
} groupShape_t;

/* dtype-generic read of a SYM or ASYM tensor's {qBits, numGroups, groupSize}.
 * symQConfig_t and asymQConfig_t share these three field NAMES but differ in
 * layout (asymQConfig_t also carries zeroPoints), so the qtype decides which
 * struct qConfig actually points to. */
typedef struct qShapeView {
    uint8_t qBits;
    size_t numGroups;
    size_t groupSize;
} qShapeView_t;

static groupModeSweep_t g_groupMode = GROUP_MODE_TENSOR;    /* GROUP_MODE=tensor|channel|size */
static int g_groupSize = 0;                                 /* GROUP_SIZE (mode=size only) */
static weightDtypeSweep_t g_weightDtype = WEIGHT_DTYPE_SYM; /* WEIGHT_DTYPE=sym|asym */

static float envFloat(const char *name, float dflt) {
    const char *v = getenv(name);
    return (v != NULL && v[0] != '\0') ? strtof(v, NULL) : dflt;
}
static int envInt(const char *name, int dflt) {
    const char *v = getenv(name);
    return (v != NULL && v[0] != '\0') ? (int)strtol(v, NULL, 10) : dflt;
}

/* Group-quant PR5 (#300): per-layer group-shape resolution.
 *
 * HAR weight tensors (N = element count, outCh = dim-0 size, pc = N/outCh =
 * elements per output channel; Conv weight = [outCh, inCh/groups, K], Linear
 * = [outFeatures, inFeatures], so outCh is always dim 0):
 *   layer   shape         N      outCh  pc   G64                G32
 *   conv1   [16, 9, 7]    1008   16     63   FALLBACK->pc=63    FALLBACK->pc=63
 *   conv2   [32,16, 5]    2560   32     80   ->40 groups        ->80 groups
 *   conv3   [64,32, 3]    6144   64     96   ->96 groups        ->192 groups
 *   linear  [6, 64]       384    6      64   ->6 groups (==pc)  ->12 groups
 * (conv1's N=1008=2^4*3^2*7: neither 64 nor 32 divide it, so both G64 and G32
 * fall back to per-channel (pc=63=3^2*7) for that layer only.)
 *
 * mode=tensor:  always per-tensor {1,0} (today's behavior).
 * mode=channel: one group per output channel -- groupSize = N/outCh, which
 *               is always an exact divisor of N by construction.
 * mode=size:    groupSize = GROUP_SIZE if it evenly divides N, else FALL
 *               BACK to the per-channel groupSize (N/outCh) for that layer.
 * Either grouped branch collapses to the canonical per-tensor shape {1,0} if
 * the resolved groupSize would leave numGroups <= 1 (a single-output-channel
 * layer, not present in HAR but handled safely) -- {1,N} is never valid. */
static groupShape_t resolveGroupShape(size_t N, size_t outCh, groupModeSweep_t mode,
                                      int groupSizeEnv) {
    if (mode == GROUP_MODE_TENSOR) {
        return (groupShape_t){.numGroups = 1, .groupSize = 0};
    }
    size_t groupSize;
    if (mode == GROUP_MODE_CHANNEL) {
        groupSize = N / outCh;
    } else { /* GROUP_MODE_SIZE */
        size_t requested = (size_t)groupSizeEnv;
        groupSize = (requested > 0 && N % requested == 0) ? requested : N / outCh;
    }
    size_t numGroups = N / groupSize;
    if (numGroups <= 1) {
        return (groupShape_t){.numGroups = 1, .groupSize = 0};
    }
    return (groupShape_t){.numGroups = numGroups, .groupSize = groupSize};
}

static qShapeView_t viewQShape(quantization_t *q) {
    if (q->type == ASYM) {
        asymQConfig_t *ac = q->qConfig;
        return (qShapeView_t){
            .qBits = ac->qBits, .numGroups = ac->numGroups, .groupSize = ac->groupSize};
    }
    symQConfig_t *sc = q->qConfig;
    return (qShapeView_t){
        .qBits = sc->qBits, .numGroups = sc->numGroups, .groupSize = sc->groupSize};
}

/* Builds a fresh weight-quantization template matching the resolved group
 * shape: per-tensor (numGroups==1) uses the plain quantizationInitSym/Asym,
 * grouped uses the *Grouped variant. Caller owns the returned template and
 * must freeQuantization it after the one requantizeTensorInPlace call it is
 * built for (requantizeTensorInPlace clones via getQLike; caller keeps
 * ownership of the original). */
static quantization_t *buildWeightQuant(groupShape_t gs, bool isAsym) {
    if (gs.numGroups == 1) {
        return isAsym ? quantizationInitAsym((uint8_t)g_symBits, HALF_AWAY)
                      : quantizationInitSym((uint8_t)g_symBits, HALF_AWAY);
    }
    return isAsym ? quantizationInitAsymGrouped((uint8_t)g_symBits, HALF_AWAY, gs.numGroups,
                                                gs.groupSize)
                  : quantizationInitSymGrouped((uint8_t)g_symBits, HALF_AWAY, gs.numGroups,
                                               gs.groupSize);
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

/* Shared quantization templates (borrowed by every layer for the whole run). */
typedef struct symQuant {
    quantization_t *floatQ;   /* FLOAT32 wire + temp weight/bias init storage */
    quantization_t *symWireQ; /* SYM_INT32 int12 wires (SYM_WIRES=1, #206) */
} symQuant_t;

/* SYM layerQuant for conv/linear: SYM_INT32 forward compute, FLOAT32 grad
 * math, FLOAT32 grad storage (weightGradStorage/biasGradStorage NULL =>
 * per-layer FLOAT32 default). Wires are FLOAT32 by default; SYM_WIRES=1
 * (#206 acceptance) switches the activation + dx wires to SYM_INT32 and the
 * dx COMPUTE to the native SYM arms — param grads stay FLOAT32 either way
 * (#261). weightStorage/biasStorage are FLOAT32 during construction (factory
 * only inits FLOAT32-native param storage, #270); main() requantizes them to
 * packed SYM@SYM_BITS post-build. */
static layerQuant_t symLayerQuant(symQuant_t *sq) {
    quantization_t *wireQ = g_symWires ? sq->symWireQ : sq->floatQ;
    arithmetic_t dxMath = g_symWires ? (arithmetic_t){ARITH_SYM_INT32, HALF_AWAY}
                                     : (arithmetic_t){ARITH_FLOAT32, HALF_AWAY};
    return (layerQuant_t){
        .forwardMath = (arithmetic_t){ARITH_SYM_INT32, HALF_AWAY},
        .weightGradMath = (arithmetic_t){ARITH_FLOAT32, HALF_AWAY},
        .biasGradMath = (arithmetic_t){ARITH_FLOAT32, HALF_AWAY},
        .propLossMath = dxMath,
        .outputQ = wireQ,
        .propLossQ = wireQ,
        .weightStorage = sq->floatQ, /* temp; requantized to SYM@SYM_BITS post-build */
        .biasStorage = sq->floatQ,   /* temp; requantized to SYM@SYM_BITS post-build */
        .weightGradStorage = NULL,   /* FLOAT32 grad default */
        .biasGradStorage = NULL,     /* FLOAT32 grad default */
        /* FLOAT32 grad target: accumulateOut adds in float regardless of mode,
         * but the mode must still be a valid ACC value (executeOpValidateAccMode
         * rejects the OUT_WRITE==0 zero-init on a hand-wired config). */
        .weightGradAccMode = OUT_ACC_DYNAMIC_RESCALE,
        .biasGradAccMode = OUT_ACC_DYNAMIC_RESCALE,
    };
}

static void buildModel(layer_t **model, symQuant_t *sq, layerQuant_t *lqNontrain) {
    layerQuant_t lqSym = symLayerQuant(sq);

    /* Block 1: Conv1d(9->16, K=7, SAME), ReLU, MaxPool(K=2, S=2). */
    model[0] = conv1dLayerInit(
        &(conv1dInit_t){
            .inChannels = IN_CHANNELS, .outChannels = C1_OUT, .kernelSize = C1_K, .padding = SAME},
        &lqSym);
    model[1] = reluLayerInit(lqNontrain);
    model[2] = maxPool1dLayerInit(
        &(maxPool1dInit_t){
            .kernelSize = 2, .stride = 2, .inputChannels = C1_OUT, .inputLength = LEN_INPUT},
        lqNontrain);

    /* Block 2 */
    model[3] = conv1dLayerInit(
        &(conv1dInit_t){
            .inChannels = C1_OUT, .outChannels = C2_OUT, .kernelSize = C2_K, .padding = SAME},
        &lqSym);
    model[4] = reluLayerInit(lqNontrain);
    model[5] = maxPool1dLayerInit(
        &(maxPool1dInit_t){
            .kernelSize = 2, .stride = 2, .inputChannels = C2_OUT, .inputLength = LEN_INPUT / 2},
        lqNontrain);

    /* Block 3 */
    model[6] = conv1dLayerInit(
        &(conv1dInit_t){
            .inChannels = C2_OUT, .outChannels = C3_OUT, .kernelSize = C3_K, .padding = SAME},
        &lqSym);
    model[7] = reluLayerInit(lqNontrain);
    model[8] = avgPool1dLayerInit(
        &(avgPool1dInit_t){.kernelSize = LEN_INPUT / 4, .stride = LEN_INPUT / 4}, lqNontrain);

    /* Head */
    model[9] = flattenLayerInit();
    model[10] =
        linearLayerInit(&(linearInit_t){.inFeatures = C3_OUT, .outFeatures = NUM_CLASSES}, &lqSym);
    model[11] = softmaxLayerInit(lqNontrain);
}

/* Requantize weights -> per-layer SYM/ASYM (grouped per GROUP_MODE) and bias
 * -> per-tensor SYM/ASYM (SAME dtype as the weights, but a SEPARATE template:
 * a grouped weight template has the wrong element count for the bias tensor)
 * for the 4 trainable layers. Weight groups are resolved from each tensor's
 * OWN shape (N = element count, outCh = dim-0 size) via resolveGroupShape --
 * see the HAR table on that function. Grad tensors are untouched (FLOAT32,
 * from the NULL grad knob at construction). Every weight template is a
 * throwaway built for exactly one requantizeTensorInPlace call (which
 * deep-clones it; caller keeps ownership) and is freed immediately after;
 * the bias template is shared across all 4 biases (identical dtype/bits/
 * per-tensor shape for every layer) and freed once at the end. */
static void requantizeParamsToSym(layer_t **model) {
    bool isAsym = (g_weightDtype == WEIGHT_DTYPE_ASYM);
    quantization_t *biasQ = isAsym ? quantizationInitAsym((uint8_t)g_symBits, HALF_AWAY)
                                   : quantizationInitSym((uint8_t)g_symBits, HALF_AWAY);

    const size_t convIdx[3] = {0, 3, 6};
    for (size_t k = 0; k < 3; k++) {
        conv1dConfig_t *cfg = model[convIdx[k]]->config->conv1d;
        tensor_t *w = cfg->weights->param;
        groupShape_t gs = resolveGroupShape(calcNumberOfElementsByTensor(w),
                                            w->shape->dimensions[0], g_groupMode, g_groupSize);
        quantization_t *weightQ = buildWeightQuant(gs, isAsym);
        requantizeTensorInPlace(w, weightQ);
        freeQuantization(weightQ);
        if (cfg->bias != NULL) {
            requantizeTensorInPlace(cfg->bias->param, biasQ);
        }
    }

    linearConfig_t *fc = model[10]->config->linear;
    tensor_t *lw = fc->weights->param;
    groupShape_t linGs = resolveGroupShape(calcNumberOfElementsByTensor(lw),
                                           lw->shape->dimensions[0], g_groupMode, g_groupSize);
    quantization_t *linWeightQ = buildWeightQuant(linGs, isAsym);
    requantizeTensorInPlace(lw, linWeightQ);
    freeQuantization(linWeightQ);
    requantizeTensorInPlace(fc->bias->param, biasQ);

    freeQuantization(biasQ);
}

/* ---- Gates -------------------------------------------------------------- */

typedef struct paramGateCtx {
    bool isGrad;
    int expectBits;
    int count;
    int fails;
} paramGateCtx_t;

/* PARAM gate: every trainable weight/bias tensor is packed SYM-or-ASYM (per
 * WEIGHT_DTYPE) @SYM_BITS, in the group SHAPE GROUP_MODE/GROUP_SIZE should
 * have produced for that specific tensor (weights: resolveGroupShape on the
 * tensor's own N/outCh; biases: always per-tensor {1,0}) -- an independent
 * recheck of what requantizeParamsToSym built, via the same shared helper.
 * GRAD gate: every trainable grad tensor is FLOAT32 (the NULL grad-knob
 * default). */
static void paramGateSink(void *ctxVoid, size_t layerIdx, layerType_t layerType, const char *phase,
                          tensor_t *tensor) {
    if (layerType != LINEAR && layerType != CONV1D) {
        return;
    }
    paramGateCtx_t *ctx = ctxVoid;

    if (ctx->isGrad) {
        if (tensor->quantization->type != FLOAT32) {
            fprintf(stderr, "GATE FAIL: layer %zu %s expected FLOAT32 grad, got qtype %d\n",
                    layerIdx, phase, (int)tensor->quantization->type);
            ctx->fails++;
            return;
        }
        ctx->count++;
        return;
    }

    qtype_t expectType = (g_weightDtype == WEIGHT_DTYPE_ASYM) ? ASYM : SYM;
    if (tensor->quantization->type != expectType) {
        fprintf(stderr, "GATE FAIL: layer %zu %s expected %s param, got qtype %d\n", layerIdx,
                phase, expectType == ASYM ? "ASYM" : "SYM", (int)tensor->quantization->type);
        ctx->fails++;
        return;
    }

    qShapeView_t actual = viewQShape(tensor->quantization);
    if ((int)actual.qBits != ctx->expectBits) {
        fprintf(stderr, "GATE FAIL: layer %zu %s expected qBits %d, got %u\n", layerIdx, phase,
                ctx->expectBits, (unsigned)actual.qBits);
        ctx->fails++;
        return;
    }

    bool isBias = strstr(phase, ".bias") != NULL;
    groupShape_t expectShape;
    if (isBias) {
        expectShape = (groupShape_t){.numGroups = 1, .groupSize = 0};
    } else {
        size_t N = calcNumberOfElementsByTensor(tensor);
        size_t outCh = tensor->shape->dimensions[0];
        expectShape = resolveGroupShape(N, outCh, g_groupMode, g_groupSize);
    }
    if (actual.numGroups != expectShape.numGroups || actual.groupSize != expectShape.groupSize) {
        fprintf(stderr, "GATE FAIL: layer %zu %s expected group shape {%zu,%zu}, got {%zu,%zu}\n",
                layerIdx, phase, expectShape.numGroups, expectShape.groupSize, actual.numGroups,
                actual.groupSize);
        ctx->fails++;
        return;
    }

    ctx->count++;
}

typedef struct groupLogInfo {
    groupShape_t conv1, conv2, conv3, linear;
    size_t overheadBytes;
} groupLogInfo_t;

/* Reads the ACTUAL post-requantize group shape off each of the 8 trainable
 * param tensors -- single source of truth, no separate bookkeeping to drift
 * from what requantizeParamsToSym actually built -- for the log's
 * "groups_resolved" (the 4 weight tensors) and "group_overhead_b" (all 8:
 * weights AND biases, since even a per-tensor {1,0} tensor carries one float
 * scale [+ one uint16 zero-point if ASYM] of metadata; spec-§8-mandatory
 * honest accuracy-per-byte accounting). */
static groupLogInfo_t computeGroupLogInfo(layer_t **model, bool isAsym) {
    size_t bytesPerGroup = sizeof(float) + (isAsym ? sizeof(uint16_t) : 0);
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
        .overheadBytes = totalGroups * bytesPerGroup,
    };
}

static FILE *g_log_file = NULL;
static int g_first_epoch = 1;
static struct timespec g_epoch_t0;
static float g_firstTrainLoss = -1.0f;
static float g_lastTrainLoss = -1.0f;

/* #279 code-movement instrumentation (opt-in via LOG_CODE_MOVEMENT, off by
 * default so the mem study / bit-parity runs are untouched). */
static int g_trackMovement = 0;
static uint8_t *g_wsnap = NULL; /* previous-epoch snapshot of all packed-SYM param bytes */
static size_t g_wsnapLen = 0;

/* Fraction of trainable packed-SYM param STORAGE bytes that changed since the
 * previous epoch. This is the direct #279 dead-zone signal: HALF_AWAY freezes
 * the codes once the FLOAT32 grad step goes sub-ULP (-> collapses to 0), while
 * seeded SR_HALF_AWAY keeps dithering them (-> stays > 0). Byte-granular (a
 * packed byte holds several sub-byte codes) so it answers "did any code in this
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

static void epochCallback(size_t epoch, float trainLoss, epochStats_t evalStats) {
    struct timespec t1;
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double wall_s =
        (double)(t1.tv_sec - g_epoch_t0.tv_sec) + (double)(t1.tv_nsec - g_epoch_t0.tv_nsec) * 1e-9;

    if (g_firstTrainLoss < 0.0f) {
        g_firstTrainLoss = trainLoss;
    }
    g_lastTrainLoss = trainLoss;

    if (g_log_file != NULL) {
        if (!g_first_epoch) {
            fprintf(g_log_file, ",\n");
        }
        fprintf(g_log_file,
                "    {\"epoch\": %zu, \"step_losses\": [], \"train_loss\": %.6f, "
                "\"val_loss\": %.6f, \"val_acc\": %.6f, \"wall_s\": %.4f, \"lr\": %.8f",
                epoch, (double)trainLoss, (double)evalStats.loss, (double)evalStats.accuracy,
                wall_s, (double)optimizerFunctions[g_optim->type].getLr(g_optim));
        if (g_trackMovement) {
            fprintf(g_log_file, ", \"codes_changed_frac\": %.6f", codeMovementFraction());
        }
        fprintf(g_log_file, "}");
        fflush(g_log_file);
    }
    g_first_epoch = 0;

    fprintf(stdout, "epoch %zu: train_loss=%.4f val_loss=%.4f val_acc=%.4f wall_s=%.2f\n", epoch,
            (double)trainLoss, (double)evalStats.loss, (double)evalStats.accuracy, wall_s);
    fflush(stdout);

    clock_gettime(CLOCK_MONOTONIC, &g_epoch_t0);
}

int main(void) {
    g_symBits = envInt("SYM_BITS", g_symBits);
    g_symWires = envInt("SYM_WIRES", g_symWires);
    g_lr = envFloat("LR", g_lr);
    g_momentum = envFloat("MOMENTUM", g_momentum);
    g_epochs = envInt("EPOCHS", g_epochs);
    g_seed = (unsigned)envInt("SEED", (int)g_seed);
    g_shuffleSeed = (unsigned)envInt("SHUFFLE_SEED", (int)g_shuffleSeed);
    g_trackMovement = envInt("LOG_CODE_MOVEMENT", 0);
    const char *schedEnv = getenv("LR_SCHEDULE");
    if (schedEnv != NULL && schedEnv[0] != '\0') {
        if (strcmp(schedEnv, "cosine") != 0) {
            fprintf(stderr, "LR_SCHEDULE=%s not supported (only: cosine)\n", schedEnv);
            exit(1);
        }
        g_useCosine = 1;
    }
    g_lrMin = envFloat("LR_MIN", g_lrMin);

    /* Group-quant PR5 axis (#300): GROUP_MODE/GROUP_SIZE/WEIGHT_DTYPE. */
    const char *groupModeEnv = getenv("GROUP_MODE");
    if (groupModeEnv != NULL && groupModeEnv[0] != '\0') {
        if (strcmp(groupModeEnv, "tensor") == 0) {
            g_groupMode = GROUP_MODE_TENSOR;
        } else if (strcmp(groupModeEnv, "channel") == 0) {
            g_groupMode = GROUP_MODE_CHANNEL;
        } else if (strcmp(groupModeEnv, "size") == 0) {
            g_groupMode = GROUP_MODE_SIZE;
        } else {
            fprintf(stderr, "GROUP_MODE=%s not supported (only: tensor, channel, size)\n",
                    groupModeEnv);
            exit(1);
        }
    }
    if (g_groupMode == GROUP_MODE_SIZE) {
        g_groupSize = envInt("GROUP_SIZE", 0);
        if (g_groupSize <= 0) {
            fprintf(stderr, "GROUP_SIZE must be > 0 when GROUP_MODE=size (got %d)\n", g_groupSize);
            exit(1);
        }
    }
    const char *weightDtypeEnv = getenv("WEIGHT_DTYPE");
    if (weightDtypeEnv != NULL && weightDtypeEnv[0] != '\0') {
        if (strcmp(weightDtypeEnv, "sym") == 0) {
            g_weightDtype = WEIGHT_DTYPE_SYM;
        } else if (strcmp(weightDtypeEnv, "asym") == 0) {
            g_weightDtype = WEIGHT_DTYPE_ASYM;
        } else {
            fprintf(stderr, "WEIGHT_DTYPE=%s not supported (only: sym, asym)\n", weightDtypeEnv);
            exit(1);
        }
    }

    const char *logPath = getenv("LOG_PATH");

    /* Packed-SYM/ASYM STORAGE supports wide widths, but this example's forward
     * is ARITH_SYM_INT32: it multiplies the packed weights as integer operands
     * accumulated in an int32 accumulator, and matmul/conv cap the SYM_INT32
     * operand at 12 bits (#227) — wider weights (e.g. 16) would overflow int32
     * over the conv accumulation length, so the kernel rejects them mid-forward.
     * Fail fast here with the reason instead of crashing deep in the matmul.
     * This ceiling applies to WEIGHT_DTYPE=asym too (ASYM's own D6 ceiling is
     * [1,16], looser): the ASYM-stored weight still converts to the same
     * SYM_INT32 forward operand every step (conversionMatrix[ASYM][SYM_INT32]). */
    if (g_symBits < 1 || g_symBits > 12) {
        fprintf(stderr,
                "SYM_BITS must be in [1, 12]: the SYM_INT32 forward operand contract caps at 12 "
                "bits (#227; int32 accumulator would overflow at wider widths). Got %d.\n",
                g_symBits);
        return 2;
    }

    const char *weightDtypeStr = (g_weightDtype == WEIGHT_DTYPE_ASYM) ? "asym" : "sym";
    const char *groupModeStr = (g_groupMode == GROUP_MODE_CHANNEL) ? "channel"
                               : (g_groupMode == GROUP_MODE_SIZE)  ? "size"
                                                                   : "tensor";
    int groupSizeForLog = (g_groupMode == GROUP_MODE_SIZE) ? g_groupSize : 0;

    fprintf(stdout,
            "CONFIG sym_bits=%d lr=%.5f momentum=%.3f epochs=%d seed=%u shuffle_seed=%u "
            "group_mode=%s group_size=%d weight_dtype=%s\n",
            g_symBits, (double)g_lr, (double)g_momentum, g_epochs, g_seed, g_shuffleSeed,
            groupModeStr, groupSizeForLog, weightDtypeStr);

#ifdef ODT_MEM_PROFILE
    /* Reset the heap counter to 0 BEFORE the first allocation so dataset_b is
     * counted from a clean baseline; env reads / printf above allocate nothing
     * via reserveMemory. */
    memProfileReset();
#endif

    initDataSets();

#ifdef ODT_MEM_PROFILE
    size_t markDataset = memProfileMark(); /* dataset_b */
#endif

    /* Stochastic rounding (SR_HALF_AWAY) on the training write-back is the
     * dead-zone escape (#279, sweep-ratified default): a FLOAT32 grad step
     * smaller than one SYM level would deterministically round back to the
     * same int under HALF_AWAY and vanish; SR_HALF_AWAY rounds it up with
     * probability equal to the fractional part, so sub-ULP updates move the
     * weight IN EXPECTATION. Since #279 the mode is OPTIMIZER-owned (set on
     * the optimizer below); the param storage qConfig stays HALF_AWAY so the
     * initial requantize + any storage encode stay deterministic.
     * SYM_ROUNDING=det forces deterministic HALF_AWAY write-backs to A/B the
     * dead-zone claim (the numerics hypothesis this example exists to test). */
    const char *roundingEnv = getenv("SYM_ROUNDING");
    roundingMode_t symRounding =
        (roundingEnv != NULL && strcmp(roundingEnv, "det") == 0) ? HALF_AWAY : SR_HALF_AWAY;

    symQuant_t sq = {
        .floatQ = quantizationInitFloat(),
        .symWireQ = quantizationInitSymInt32(HALF_AWAY),
    };

    /* Non-trainable layers (ReLU/pools/Softmax): FLOAT32 wires by default;
     * under SYM_WIRES=1 the uniform lq carries the SYM wire config, so their
     * forward/propLoss dispatch selects the native SYM arms (#205) and Softmax
     * runs its funnel dequant->stable-float->requant path. */
    layerQuant_t lqNontrain;
    layerQuantInitUniform(&lqNontrain, g_symWires ? sq.symWireQ : quantizationInitFloat());

    layer_t *model[MODEL_SIZE];
    rngSetSeed(g_seed);
#ifdef ODT_MEM_PROFILE
    size_t markBeforeModel = memProfileMark();
#endif
    buildModel(model, &sq, &lqNontrain);
    requantizeParamsToSym(model);
#ifdef ODT_MEM_PROFILE
    size_t markAfterModel = memProfileMark(); /* params_grads_b = delta */
#endif

    /* ---- Gate: param + grad storage dtypes -------------------------------- */
    paramGateCtx_t wCtx = {.isGrad = false, .expectBits = g_symBits, .count = 0, .fails = 0};
    traceModelWeights(model, MODEL_SIZE, "gate", paramGateSink, &wCtx);
    paramGateCtx_t gCtx = {.isGrad = true, .expectBits = 0, .count = 0, .fails = 0};
    traceModelGrads(model, MODEL_SIZE, "gate", paramGateSink, &gCtx);
    /* 4 trainable layers x (weight + bias) = 8 each. */
    if (wCtx.fails != 0 || gCtx.fails != 0 || wCtx.count != 8 || gCtx.count != 8) {
        fprintf(stderr, "GATES FAILED (weight ok=%d fails=%d; grad ok=%d fails=%d)\n", wCtx.count,
                wCtx.fails, gCtx.count, gCtx.fails);
        return 2;
    }
    fprintf(stdout, "GATES PASS: weights+bias=%s@%d grads=FLOAT32 (8 param + 8 grad checks)\n",
            weightDtypeStr, g_symBits);
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
     * funnel property, so the packed-SYM weights round-trip SYM<->FLOAT32 each
     * step, write-back rounding OPTIMIZER-owned (SYM_ROUNDING env -> setter
     * below, #279), while the FLOAT32 grad and momentum state (below) are
     * read/written directly. Update arithmetic itself is FLOAT32 via
     * updateMath. */
#ifdef ODT_MEM_PROFILE
    size_t markBeforeOpt = memProfileMark();
#endif
    /* Momentum accumulator is FLOAT32, DECOUPLED from the packed-SYM params via
     * the epic's own-config knob. A packed-SYM momentum would requantize the
     * running velocity to the same coarse levels as the weights and reinstate a
     * dead-zone in the accumulator; a FLOAT32 accumulator keeps velocity precise
     * so ONLY the weights carry the memory win. */
    quantization_t *momentumQ = quantizationInitFloat();
    optimizer_t *sgd =
        sgdMCreateOptim(g_lr, g_momentum, /*weightDecay*/ 0.0f, model, MODEL_SIZE, momentumQ,
                        (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});
    /* Set explicitly in BOTH directions (factory already defaults SR): the
     * A/B arms of the #279 sweep should read symmetrically from the code. */
    optimizerSetWriteBackRounding(sgd, symRounding);
#ifdef ODT_MEM_PROFILE
    size_t markAfterOpt = memProfileMark(); /* optstate_b = delta */
#endif
    g_optim = sgd;
    lrScheduler_t cosineSched;
    lrScheduler_t *sched = NULL;
    if (g_useCosine && g_epochs < 1) {
        fprintf(stderr, "LR_SCHEDULE=cosine requires EPOCHS >= 1\n");
        exit(1);
    }
    if (g_useCosine) {
        /* tMax = full run length: one half-cosine from g_lr down to g_lrMin.
         * The tail deliberately drives per-step updates below one SYM level —
         * SR_HALF_AWAY keeps them alive in expectation (#279); see README. */
        cosineAnnealingLrInit(&cosineSched, sgd, (size_t)g_epochs, g_lrMin);
        sched = &cosineSched;
    }

    groupLogInfo_t groupInfo = computeGroupLogInfo(model, g_weightDtype == WEIGHT_DTYPE_ASYM);

    if (logPath != NULL && logPath[0] != '\0') {
        g_log_file = fopen(logPath, "w");
        if (g_log_file != NULL) {
            fprintf(g_log_file,
                    "{\n  \"impl\": \"c-sym-weights\", \"example\": \"har_classifier\",\n"
                    "  \"config\": {\"sym_bits\": %d, \"epochs\": %d, \"batch\": %d, \"lr\": %.6f, "
                    "\"momentum\": %.6f, \"seed\": %u, \"shuffle_seed\": %u, "
                    "\"lr_schedule\": \"%s\", \"lr_min\": %.6f, "
                    "\"weight_dtype\": \"%s\", \"group_mode\": \"%s\", \"group_size\": %d, "
                    "\"groups_resolved\": {\"conv1\": [%zu, %zu], \"conv2\": [%zu, %zu], "
                    "\"conv3\": [%zu, %zu], \"linear\": [%zu, %zu]}, "
                    "\"group_overhead_b\": %zu},\n  \"epochs\": [\n",
                    g_symBits, g_epochs, BATCH, (double)g_lr, (double)g_momentum, g_seed,
                    g_shuffleSeed, g_useCosine ? "cosine" : "none", (double)g_lrMin, weightDtypeStr,
                    groupModeStr, groupSizeForLog, groupInfo.conv1.numGroups,
                    groupInfo.conv1.groupSize, groupInfo.conv2.numGroups, groupInfo.conv2.groupSize,
                    groupInfo.conv3.numGroups, groupInfo.conv3.groupSize,
                    groupInfo.linear.numGroups, groupInfo.linear.groupSize,
                    groupInfo.overheadBytes);
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &g_epoch_t0);

    trainingRunResult_t result = trainingRun(
        model, MODEL_SIZE,
        (lossConfig_t){
            .funcType = CROSS_ENTROPY, .backwardReduction = REDUCTION_MEAN, .classWeights = NULL},
        trainLoader, valLoader, sgd, sched, g_epochs, calculateGradsSequential, inferenceWithLoss,
        epochCallback);
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
    report.sym_bits = g_symBits;
    report.dataset_b = markDataset;
    report.params_grads_b = markAfterModel - markBeforeModel;
    report.optstate_b = markAfterOpt - markBeforeOpt;
    report.params_b = memInstrumentParamBytes(sgd);
    report.grads_b = memInstrumentGradBytes(sgd);
    report.optstate_analytic_b = memInstrumentOptStateBytes(sgd);
    report.activations_b = memInstrumentHarActivationBytes(MICRO_BATCH);
    report.io_b = memInstrumentHarIoBytes(MICRO_BATCH);
    report.pool_backward_b = memInstrumentPoolBackwardBytes(model, MODEL_SIZE);
    report.dx_peak_b = memInstrumentHarDxPeakBytes(MICRO_BATCH);

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
     * coarse widths (e.g. SYM@4) a config may legitimately fail to descend, and
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
                "CONVERGENCE WARN: train loss did not decrease (first=%.6f last=%.6f) — SYM@%d "
                "may be too coarse to descend at base_lr=%.4f lr_schedule=%s (recorded, not "
                "fatal)\n",
                (double)g_firstTrainLoss, (double)g_lastTrainLoss, g_symBits, (double)g_lr,
                g_useCosine ? "cosine" : "none");
    }

    return 0;
}
