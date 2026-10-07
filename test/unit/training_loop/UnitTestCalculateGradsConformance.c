#define SOURCE_FILE "UNIT_TEST_CALCULATE_GRADS_CONFORMANCE"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "AdaptivePool1dApi.h"
#include "AsanDeath.h"
#include "BatchNorm1dApi.h"
#include "CalculateGradsSequential.h"
#include "Common.h"
#include "Conv1dTransposedApi.h"
#include "DeathTest.h"
#include "DropoutApi.h"
#include "GroupNormApi.h"
#include "InferenceApi.h"
#include "Layer.h"
#include "LayerQuant.h"
#include "LegacyCalculateGrads.h"
#include "LossFunction.h"
#include "OdtHook.h"
#include "Quantization.h"
#include "QuantizationApi.h"
#include "RNG.h"
#include "RematPlan.h"
#include "RematTestDecorators.h"
#include "RematTestFixtures.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "TraceApi.h"
#include "TrainingLoopApi.h"
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

/* ---- the fixture zoo: one model per layer, wire and loss class ---- */

/* A tensor with real, deterministic bytes: the driver's checker requires the
 * input to be resident, and remat P1 needs values that exercise every layer. */
static tensor_t *makeFloatTensor(const size_t *dims, size_t rank, float seed) {
    size_t *d = reserveMemory(rank * sizeof(size_t));
    size_t *order = reserveMemory(rank * sizeof(size_t));
    size_t elements = 1;
    for (size_t k = 0; k < rank; k++) {
        d[k] = dims[k];
        elements *= dims[k];
    }
    setOrderOfDimsForNewTensor(rank, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, d, rank, order);
    tensor_t *t = initTensor(shape, quantizationInitFloat(), NULL);
    float *v = (float *)t->data;
    for (size_t i = 0; i < elements; i++) {
        v[i] = seed * (float)((int)((i * 37u + 11u) % 17u) - 8) / 8.0f;
    }
    return t;
}

/* A one-hot [1, classes] label for the CE fixtures. */
static tensor_t *makeOneHot(size_t classes, size_t hot) {
    tensor_t *t = makeFloatTensor((size_t[]){1, classes}, 2, 0.0f);
    ((float *)t->data)[hot] = 1.0f;
    return t;
}

static tensor_t *makeBoolMask(size_t elements) {
    size_t *dims = reserveMemory(sizeof(size_t));
    size_t *order = reserveMemory(sizeof(size_t));
    dims[0] = elements;
    setOrderOfDimsForNewTensor(1, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 1, order);
    return initTensor(shape, quantizationInitBool(), NULL);
}

#define ZOO_MAX_LAYERS 12
#define ZOO_MAX_TEMPLATES 4
#define ZOO_SEED 4242u
typedef struct zooFixture {
    layer_t *model[ZOO_MAX_LAYERS];
    size_t n;
    lossConfig_t loss;
    tensor_t *x;
    tensor_t *y;
    /* Wire templates and the Dropout mask the layers borrow. */
    quantization_t *templates[ZOO_MAX_TEMPLATES];
    size_t numTemplates;
    tensor_t *mask;
} zooFixture_t;

static void beginZoo(zooFixture_t *f) {
    memset(f, 0, sizeof *f);
    rngSetSeed(7u); /* deterministic random init */
}

static quantization_t *keepTemplate(zooFixture_t *f, quantization_t *q) {
    TEST_ASSERT_TRUE(f->numTemplates < ZOO_MAX_TEMPLATES);
    f->templates[f->numTemplates++] = q;
    return q;
}

static void freeZoo(zooFixture_t *f) {
    for (size_t i = 0; i < f->n; i++) {
        switch (f->model[i]->type) {
        case GROUPNORM:
            freeGroupNormLayer(f->model[i]);
            break;
        case DROPOUT:
            freeDropoutLayer(f->model[i]);
            break;
        case BATCHNORM1D:
            freeBatchNorm1dLayer(f->model[i]);
            break;
        case CONV1D_TRANSPOSED:
            freeConv1dTransposedLayer(f->model[i]);
            break;
        case ADAPTIVE_AVGPOOL1D:
            freeAdaptiveAvgPool1dLayer(f->model[i]);
            break;
        default:
            freeModel(&f->model[i], 1);
        }
    }
    for (size_t k = 0; k < f->numTemplates; k++) {
        freeQuantization(f->templates[k]);
    }
    if (f->mask != NULL) {
        freeTensor(f->mask);
    }
    freeTensor(f->x);
    freeTensor(f->y);
}

static void mseLabel(zooFixture_t *f, size_t outFeatures) {
    f->loss = defaultLossConfig(MSE);
    f->y = makeFloatTensor((size_t[]){1, outFeatures}, 2, 0.5f);
}

/* Linear 4 -> 3 -> ReLU -> Linear 3 -> 2 under MSE: forward wires, the seed,
 * one dx wire, grads-only at deepest. */
static void buildMlp(zooFixture_t *f) {
    beginZoo(f);
    f->model[0] = makeLinear(4, 3, false);
    f->model[1] = makeRelu(&g_floatQ);
    f->model[2] = makeLinear(3, 2, false);
    f->n = 3;
    f->x = makeFloatTensor((size_t[]){1, 4}, 2, 1.0f);
    mseLabel(f, 2);
}

/* HAR (RematTestFixtures.h) under CE; freezeConvs is the #380-truncated
 * stage-2 backbone (deepest 10). */
static void buildHarZoo(zooFixture_t *f, bool freezeConvs) {
    beginZoo(f);
    buildHar(f->model, freezeConvs);
    f->n = HAR_N;
    f->loss = defaultLossConfig(CROSS_ENTROPY);
    f->x = makeFloatTensor((size_t[]){1, 9, 128}, 3, 1.0f);
    f->y = makeOneHot(6, 2);
}

static void buildHarCnn(zooFixture_t *f) {
    buildHarZoo(f, false);
}

static void buildTruncatedHar(zooFixture_t *f) {
    buildHarZoo(f, true);
}

/* MSE after Softmax: the Softmax backward runs and reads its logits (no CE
 * positional skip). */
static void buildSoftmaxMse(zooFixture_t *f) {
    beginZoo(f);
    f->model[0] = makeLinear(4, 3, false);
    f->model[1] = makeSoftmax();
    f->n = 2;
    f->x = makeFloatTensor((size_t[]){1, 4}, 2, 1.0f);
    mseLabel(f, 3);
}

/* A frozen LayerNorm and a frozen Linear above deepest 0: the norm's backward
 * reads its input, the frozen GEMM's does not. */
static void buildFrozenLayerNorm(zooFixture_t *f) {
    beginZoo(f);
    f->model[0] = makeLinear(4, 4, false);
    f->model[1] = makeLayerNorm(4, true);
    f->model[2] = makeLinear(4, 4, true);
    f->model[3] = makeLinear(4, 2, false);
    f->n = 4;
    f->x = makeFloatTensor((size_t[]){1, 4}, 2, 1.0f);
    mseLabel(f, 2);
}

/* Conv1d -> frozen GroupNorm -> Flatten -> Linear: a frozen norm above
 * deepest on the [B, C, L] path. */
static void buildFrozenGroupNorm(zooFixture_t *f) {
    beginZoo(f);
    layerQuant_t lq;
    layerQuantInitUniform(&lq, &g_floatQ);
    f->model[0] =
        conv1dLayerInit(&(conv1dInit_t){.inChannels = 4, .outChannels = 4, .kernelSize = 1}, &lq);
    f->model[1] = groupNormLayerInit(
        &(groupNormInit_t){.numGroups = 2, .numChannels = 4, .trainable = TRAINABLE_FALSE}, &lq);
    f->model[2] = flattenLayerInit();
    f->model[3] = makeLinear(12, 2, false);
    f->n = 4;
    f->x = makeFloatTensor((size_t[]){1, 4, 3}, 3, 1.0f);
    mseLabel(f, 2);
}

/* Nothing trains: FORWARD x n and LOSS_FORWARD only, four events still. */
static void buildAllFrozen(zooFixture_t *f) {
    beginZoo(f);
    f->model[0] = makeLinear(4, 2, true);
    f->model[1] = makeRelu(&g_floatQ);
    f->n = 2;
    f->x = makeFloatTensor((size_t[]){1, 4}, 2, 1.0f);
    mseLabel(f, 2);
}

/* Dropout draws its mask from the global stream: remat P2 compares the stream
 * position after the call. */
static void buildDropout(zooFixture_t *f) {
    beginZoo(f);
    f->mask = makeBoolMask(4);
    f->model[0] = makeLinear(4, 4, false);
    f->model[1] = dropoutLayerInit(0.5f, f->mask, &g_floatQ, &g_floatQ);
    f->model[2] = makeLinear(4, 2, false);
    f->n = 3;
    f->x = makeFloatTensor((size_t[]){1, 4}, 2, 1.0f);
    mseLabel(f, 2);
}

/* Two Quantization layers: ACT 2 is SYM_INT32 at qMaxBits 8, GRAD 2 and
 * GRAD 1 at qMaxBits 10 (a SYM -> SYM requant in the first layer's
 * backward), so every SYM width differs from the int12 default. */
static void buildQuantSym(zooFixture_t *f) {
    beginZoo(f);
    quantization_t *act = keepTemplate(f, quantizationInitSymInt32WithBits(HALF_AWAY, 8));
    quantization_t *grad = keepTemplate(f, quantizationInitSymInt32WithBits(HALF_AWAY, 10));
    f->model[0] = makeLinear(4, 4, false);
    f->model[1] = makeQuant(act, grad);
    f->model[2] = makeQuant(&g_floatQ, grad);
    f->model[3] = makeLinear(4, 2, false);
    f->n = 4;
    f->x = makeFloatTensor((size_t[]){1, 4}, 2, 1.0f);
    mseLabel(f, 2);
}

/* Flatten at 0: ACT 1 takes the caller's input config. */
static void buildFlattenAt0(zooFixture_t *f) {
    beginZoo(f);
    f->model[0] = flattenLayerInit();
    f->model[1] = makeLinear(6, 2, false);
    f->n = 2;
    f->x = makeFloatTensor((size_t[]){1, 2, 3}, 3, 1.0f);
    mseLabel(f, 2);
}

/* The F1 alignment model: ACT 1 is a 5-byte per-tensor BFP wire. */
static void buildF1Zoo(zooFixture_t *f) {
    beginZoo(f);
    quantization_t *bfp = keepTemplate(f, quantizationInitBfp(8, 8, HALF_AWAY));
    f->model[0] = makeQuant(bfp, &g_floatQ);
    f->model[1] = makeLinear(5, 1, false);
    f->n = 2;
    f->x = makeFloatTensor((size_t[]){1, 5}, 2, 1.0f);
    mseLabel(f, 1);
}

/* A grouped BFP wire: 8 elements in groups of 4 (the numberOfValues /
 * groupSize arm); the template's own numGroups is ignored. */
static void buildGroupedBfp(zooFixture_t *f) {
    beginZoo(f);
    quantization_t *bfp = keepTemplate(f, quantizationInitBfpGrouped(8, 8, HALF_AWAY, 2, 4));
    f->model[0] = makeQuant(bfp, &g_floatQ);
    f->model[1] = makeLinear(8, 2, false);
    f->n = 2;
    f->x = makeFloatTensor((size_t[]){1, 8}, 2, 1.0f);
    mseLabel(f, 2);
}

/* n = 1 under CE: top = -1 < deepest = 0 (remat D20), LOSS_BACKWARD and no BACKWARD. */
static void buildSingleLayerCe(zooFixture_t *f) {
    beginZoo(f);
    f->model[0] = makeLinear(4, 3, false);
    f->n = 1;
    f->loss = defaultLossConfig(CROSS_ENTROPY);
    f->x = makeFloatTensor((size_t[]){1, 4}, 2, 1.0f);
    f->y = makeOneHot(3, 1);
}

/* CE with only the last Linear trainable: deepest 2 = top. */
static void buildCeLastLinearOnly(zooFixture_t *f) {
    beginZoo(f);
    f->model[0] = makeLinear(4, 4, true);
    f->model[1] = makeRelu(&g_floatQ);
    f->model[2] = makeLinear(4, 3, false);
    f->model[3] = makeSoftmax();
    f->n = 4;
    f->loss = defaultLossConfig(CROSS_ENTROPY);
    f->x = makeFloatTensor((size_t[]){1, 4}, 2, 1.0f);
    f->y = makeOneHot(3, 0);
}

/* CE over a model whose last layer is not a Softmax: the positional rule
 * still skips layer n - 1's backward, exactly as the old driver did. */
static void buildCeWithoutSoftmax(zooFixture_t *f) {
    beginZoo(f);
    f->model[0] = makeLinear(4, 3, false);
    f->model[1] = makeRelu(&g_floatQ);
    f->n = 2;
    f->loss = defaultLossConfig(CROSS_ENTROPY);
    f->x = makeFloatTensor((size_t[]){1, 4}, 2, 1.0f);
    f->y = makeOneHot(3, 2);
}

/* BatchNorm1d (#460): its backward reads the training flag, and its training
 * forward moves the running stats. Four rows, so the batch variance is not 0. */
static void buildBatchNorm(zooFixture_t *f) {
    beginZoo(f);
    layerQuant_t lq;
    layerQuantInitUniform(&lq, &g_floatQ);
    f->model[0] = batchNorm1dLayerInit(&(batchNorm1dInit_t){.numChannels = 2}, &lq);
    f->model[1] = makeLinear(2, 2, false);
    f->n = 2;
    f->loss = defaultLossConfig(MSE);
    f->x = makeFloatTensor((size_t[]){4, 2}, 2, 1.0f);
    f->y = makeFloatTensor((size_t[]){4, 2}, 2, 0.5f);
}

/* Conv1d -> ConvTranspose1d -> Flatten -> Linear under MSE. ConvTranspose1d
 * reads its input only for the weight grad. The frozen one sits above the
 * trainable Conv1d: at the deepest position the frozen-layer cut would drop
 * its backward, and its "does not read" case would never run. */
static void buildConvTransposed(zooFixture_t *f, bool frozen) {
    beginZoo(f);
    layerQuant_t lq;
    layerQuantInitUniform(&lq, &g_floatQ);
    f->model[0] =
        conv1dLayerInit(&(conv1dInit_t){.inChannels = 2, .outChannels = 2, .kernelSize = 1}, &lq);
    f->model[1] = conv1dTransposedLayerInit(
        &(conv1dTransposedInit_t){.inChannels = 2,
                                  .outChannels = 2,
                                  .kernelSize = 2,
                                  .trainable = frozen ? TRAINABLE_FALSE : TRAINABLE_DEFAULT},
        &lq);
    f->model[2] = flattenLayerInit();
    f->model[3] = makeLinear(8, 2, false);
    f->n = 4;
    f->x = makeFloatTensor((size_t[]){1, 2, 3}, 3, 1.0f);
    mseLabel(f, 2);
}

static void buildConvTransposedTrainable(zooFixture_t *f) {
    buildConvTransposed(f, false);
}

static void buildFrozenConvTransposed(zooFixture_t *f) {
    buildConvTransposed(f, true);
}

/* Conv1d -> AdaptiveAvgPool1d (length 5 -> 2) -> Flatten -> Linear under
 * MSE: the pool's backward reads no input, so under LIVENESS its input dies
 * at its forward. */
static void buildAdaptiveAvgPool(zooFixture_t *f) {
    beginZoo(f);
    layerQuant_t lq;
    layerQuantInitUniform(&lq, &g_floatQ);
    f->model[0] =
        conv1dLayerInit(&(conv1dInit_t){.inChannels = 2, .outChannels = 2, .kernelSize = 1}, &lq);
    f->model[1] = adaptiveAvgPool1dLayerInit(&(adaptiveAvgPool1dInit_t){.outputSize = 2}, &lq);
    f->model[2] = flattenLayerInit();
    f->model[3] = makeLinear(4, 2, false);
    f->n = 4;
    f->x = makeFloatTensor((size_t[]){1, 2, 5}, 3, 1.0f);
    mseLabel(f, 2);
}

/* ---- capturing one call (remat P1-P5, P9) ---- */

/* Static, never reserved, so a capture cannot move the memory-profile
 * counters it sits beside (remat P5). */
#define BLOB_MAX_BYTES 262144u
typedef struct blob {
    size_t used;
    uint8_t bytes[BLOB_MAX_BYTES];
} blob_t;

#define CAPTURE_MAX_EVENTS 8
typedef struct runCapture {
    float loss;
    uint32_t seedAfter;                    /* remat P2 */
    size_t numEvents;                      /* remat P3 */
    odtEvent_t events[CAPTURE_MAX_EVENTS]; /* remat P3 */
    size_t memBefore;                      /* remat P5 */
    size_t memAfter;                       /* remat P5 */
    blob_t values;                         /* remat P1 */
    blob_t trace;                          /* remat P4 */
} runCapture_t;
static runCapture_t g_legacy;
static runCapture_t g_driver;

static void appendBytes(blob_t *b, const void *src, size_t n) {
    TEST_ASSERT_TRUE_MESSAGE(b->used + n <= BLOB_MAX_BYTES, "raise BLOB_MAX_BYTES");
    memcpy(b->bytes + b->used, src, n);
    b->used += n;
}

/* A tensor's shape (rank, dims, dim order) and bytes with its dynamic quantization state: the SYM
 * scale and the BFP exponents are values too (remat D9). */
static void captureTensor(blob_t *b, tensor_t *t) {
    size_t elements = calcNumberOfElementsByTensor(t);
    int type = (int)t->quantization->type;
    appendBytes(b, &type, sizeof type);
    appendBytes(b, &elements, sizeof elements);
    appendBytes(b, &t->shape->numberOfDimensions, sizeof t->shape->numberOfDimensions);
    appendBytes(b, t->shape->dimensions, t->shape->numberOfDimensions * sizeof(size_t));
    appendBytes(b, t->shape->orderOfDimensions, t->shape->numberOfDimensions * sizeof(size_t));
    appendBytes(b, t->data, calcNumberOfBytesForData(t->quantization, elements));
    if (t->quantization->type == SYM_INT32) {
        const symInt32QConfig_t *qc = t->quantization->qConfig;
        appendBytes(b, &qc->scale, sizeof qc->scale);
    } else if (t->quantization->type == BFP) {
        const bfpQConfig_t *qc = t->quantization->qConfig;
        appendBytes(b, &qc->numGroups, sizeof qc->numGroups);
        appendBytes(b, qc->exponents, qc->numGroups);
    }
}

/* Every parameter grad, into b; or, with b == NULL, zeroed. The zoo's grads
 * are FLOAT32, whose zero is all-zero bytes. */
static void forEachGrad(layer_t **model, size_t n, blob_t *b) {
    for (size_t i = 0; i < n; i++) {
        parameter_t *w = NULL;
        parameter_t *bias = NULL;
        if (!layerParameters(model[i], &w, &bias)) {
            continue;
        }
        parameter_t *params[2] = {w, bias};
        for (size_t p = 0; p < 2; p++) {
            tensor_t *g = (params[p] != NULL) ? getGradFromParameter(params[p]) : NULL;
            if (g == NULL) {
                continue; /* no bias, or a frozen layer (#380: no grad tensor) */
            }
            if (b != NULL) {
                captureTensor(b, g);
            } else {
                TEST_ASSERT_EQUAL_INT(FLOAT32, g->quantization->type);
                memset(g->data, 0, calcNumberOfElementsByTensor(g) * sizeof(float));
            }
        }
    }
}

/* BatchNorm1d's running stats: a training call moves them, so they are
 * captured as values and restored before the second run. */
#define BN_SAVE_MAX 16
typedef struct bnState {
    float mean[BN_SAVE_MAX];
    float var[BN_SAVE_MAX];
    uint64_t tracked;
} bnState_t;
static bnState_t g_bnSaved[ZOO_MAX_LAYERS];

static void bnStateIo(zooFixture_t *f, bool save, blob_t *b) {
    for (size_t i = 0; i < f->n; i++) {
        if (f->model[i]->type != BATCHNORM1D) {
            continue;
        }
        batchNorm1dConfig_t *c = f->model[i]->config->batchNorm1d;
        size_t bytes = calcNumberOfElementsByTensor(c->runningMean) * sizeof(float);
        TEST_ASSERT_TRUE(bytes <= sizeof g_bnSaved[i].mean);
        if (b != NULL) {
            captureTensor(b, c->runningMean);
            captureTensor(b, c->runningVar);
            appendBytes(b, &c->numBatchesTracked, sizeof c->numBatchesTracked);
        } else if (save) {
            memcpy(g_bnSaved[i].mean, c->runningMean->data, bytes);
            memcpy(g_bnSaved[i].var, c->runningVar->data, bytes);
            g_bnSaved[i].tracked = c->numBatchesTracked;
        } else {
            memcpy(c->runningMean->data, g_bnSaved[i].mean, bytes);
            memcpy(c->runningVar->data, g_bnSaved[i].var, bytes);
            c->numBatchesTracked = g_bnSaved[i].tracked;
        }
    }
}

static void countingHook(void *ctx, odtEvent_t event) {
    runCapture_t *cap = ctx;
    if (cap->numEvents < CAPTURE_MAX_EVENTS) {
        cap->events[cap->numEvents] = event;
    }
    cap->numEvents++;
}

/* remat P4: the (idx, phase) sequence and every traced tensor's bytes. */
static void recordingSink(void *ctx, size_t layerIdx, layerType_t layerType, const char *phase,
                          tensor_t *tensor) {
    blob_t *b = ctx;
    char name[16] = {0};
    strncpy(name, phase, sizeof name - 1);
    appendBytes(b, &layerIdx, sizeof layerIdx);
    appendBytes(b, name, sizeof name);
    appendBytes(b, &layerType, sizeof layerType);
    captureTensor(b, tensor);
}

/* remat P9: the caller's input header and bytes. */
#define INPUT_MAX_BYTES 8192u
typedef struct inputSnapshot {
    tensor_t header;
    shape_t shape;
    size_t dims[TEST_MAX_RANK];
    size_t order[TEST_MAX_RANK];
    size_t bytes;
    uint8_t data[INPUT_MAX_BYTES];
} inputSnapshot_t;

static void snapshotInput(inputSnapshot_t *s, const tensor_t *x) {
    s->header = *x;
    s->shape = *x->shape;
    TEST_ASSERT_TRUE(x->shape->numberOfDimensions <= TEST_MAX_RANK);
    memcpy(s->dims, x->shape->dimensions, x->shape->numberOfDimensions * sizeof(size_t));
    memcpy(s->order, x->shape->orderOfDimensions, x->shape->numberOfDimensions * sizeof(size_t));
    s->bytes =
        calcNumberOfBytesForData(x->quantization, calcNumberOfElementsByTensor((tensor_t *)x));
    TEST_ASSERT_TRUE(s->bytes <= INPUT_MAX_BYTES);
    memcpy(s->data, x->data, s->bytes);
}

static void assertInputUnchanged(const inputSnapshot_t *s, const tensor_t *x) {
    TEST_ASSERT_EQUAL_MEMORY_MESSAGE(&s->header, x, sizeof *x, "remat P9: input header");
    TEST_ASSERT_EQUAL_MEMORY_MESSAGE(&s->shape, x->shape, sizeof s->shape, "remat P9: input shape");
    TEST_ASSERT_EQUAL_MEMORY_MESSAGE(s->dims, x->shape->dimensions,
                                     x->shape->numberOfDimensions * sizeof(size_t),
                                     "remat P9: input dims");
    TEST_ASSERT_EQUAL_MEMORY_MESSAGE(s->order, x->shape->orderOfDimensions,
                                     x->shape->numberOfDimensions * sizeof(size_t),
                                     "remat P9: input dim order");
    TEST_ASSERT_EQUAL_MEMORY_MESSAGE(s->data, x->data, s->bytes, "remat P9: input bytes");
}
static inputSnapshot_t g_input;

typedef enum { RUN_LEGACY, RUN_DRIVER } runner_t;

/* One call from the same starting state: zeroed grads (they accumulate,
 * OUT_ACC), restored BN running stats, the global stream at ZOO_SEED. The
 * counting hook is installed around the call only. */
static void captureRun(runner_t runner, zooFixture_t *f, const trainingCall_t *call,
                       runCapture_t *cap) {
    forEachGrad(f->model, f->n, NULL);
    bnStateIo(f, runner == RUN_LEGACY, NULL);
    cap->numEvents = 0;
    cap->values.used = 0;
    cap->trace.used = 0;
    snapshotInput(&g_input, f->x);
    rngSetSeed(ZOO_SEED);

    cap->memBefore = memProfileCurrentBytes();
    odtHookSet(countingHook, cap);
    trainingStats_t *stats = (runner == RUN_LEGACY)
                                 ? legacyCalculateGrads(f->model, f->n, f->loss, REDUCTION_MEAN,
                                                        f->x, f->y, recordingSink, &cap->trace)
                                 : tracedGrads(f->model, f->n, f->loss, REDUCTION_MEAN, f->x, f->y,
                                               recordingSink, &cap->trace, call);
    odtHookSet(NULL, NULL);
    cap->seedAfter = rngGetSeed();
    cap->loss = stats->loss;
    captureTensor(&cap->values, stats->output);
    freeTrainingStats(stats);
    cap->memAfter = memProfileCurrentBytes();

    forEachGrad(f->model, f->n, &cap->values);
    bnStateIo(f, false, &cap->values);
    assertInputUnchanged(&g_input, f->x);
}

/* A failure message that names the matrix cell; cell == NULL is the NULL
 * path, whose messages stay bare. */
static char g_message[128];
static const char *say(const char *cell, const char *what) {
    if (cell == NULL) {
        return what;
    }
    (void)snprintf(g_message, sizeof g_message, "%s: %s", cell, what);
    return g_message;
}

static void assertSameValues(const char *cell) {
    TEST_ASSERT_TRUE(g_legacy.values.used > 0);
    TEST_ASSERT_EQUAL_MEMORY_MESSAGE(&g_legacy.loss, &g_driver.loss, sizeof g_legacy.loss,
                                     say(cell, "remat P1: loss"));
    TEST_ASSERT_EQUAL_size_t_MESSAGE(g_legacy.values.used, g_driver.values.used,
                                     say(cell, "remat P1: captured byte count"));
    TEST_ASSERT_EQUAL_MEMORY_MESSAGE(
        g_legacy.values.bytes, g_driver.values.bytes, g_legacy.values.used,
        say(cell, "remat P1: output snapshot, parameter grads, BN running stats"));
}

static void assertConforms(const char *cell) {
    assertSameValues(cell);
    TEST_ASSERT_EQUAL_HEX32_MESSAGE(g_legacy.seedAfter, g_driver.seedAfter,
                                    say(cell, "remat P2: the global stream after the call"));
    TEST_ASSERT_EQUAL_size_t_MESSAGE(4, g_driver.numEvents,
                                     say(cell, "remat P3: four hook events"));
    TEST_ASSERT_EQUAL_size_t_MESSAGE(g_legacy.numEvents, g_driver.numEvents,
                                     say(cell, "remat P3: event count"));
    TEST_ASSERT_EQUAL_MEMORY_MESSAGE(g_legacy.events, g_driver.events,
                                     g_driver.numEvents * sizeof(odtEvent_t),
                                     say(cell, "remat P3: event order"));
    TEST_ASSERT_EQUAL_size_t_MESSAGE(g_legacy.trace.used, g_driver.trace.used,
                                     say(cell, "remat P4: trace byte count"));
    TEST_ASSERT_EQUAL_MEMORY_MESSAGE(g_legacy.trace.bytes, g_driver.trace.bytes,
                                     g_legacy.trace.used,
                                     say(cell, "remat P4: trace (idx, phase, tensor)"));
    TEST_ASSERT_EQUAL_size_t_MESSAGE(g_legacy.memBefore, g_legacy.memAfter,
                                     "remat P5: Legacy releases everything but the stats");
    TEST_ASSERT_EQUAL_size_t_MESSAGE(
        g_driver.memBefore, g_driver.memAfter,
        say(cell, "remat P5: the driver releases everything but the stats"));
}

static void assertNullPathMatchesLegacy(void (*build)(zooFixture_t *)) {
    zooFixture_t f;
    build(&f);
    captureRun(RUN_LEGACY, &f, NULL, &g_legacy);
    captureRun(RUN_DRIVER, &f, NULL, &g_driver);
    freeZoo(&f);
    assertConforms(NULL);
}

static trainingStats_t *runOnce(runner_t runner, zooFixture_t *f, const trainingCall_t *call) {
    return (runner == RUN_LEGACY) ? legacyCalculateGrads(f->model, f->n, f->loss, REDUCTION_MEAN,
                                                         f->x, f->y, NULL, NULL)
                                  : calculateGradsSequential(f->model, f->n, f->loss,
                                                             REDUCTION_MEAN, f->x, f->y, call);
}

/* A caller that does not zero between calls: the grads accumulate over both,
 * BN's running stats move twice and Dropout's stream advances twice. */
static void captureTwoCalls(runner_t runner, zooFixture_t *f, runCapture_t *cap) {
    forEachGrad(f->model, f->n, NULL);
    bnStateIo(f, runner == RUN_LEGACY, NULL);
    cap->values.used = 0;
    rngSetSeed(ZOO_SEED);
    freeTrainingStats(runOnce(runner, f, NULL));
    trainingStats_t *stats = runOnce(runner, f, NULL);
    cap->seedAfter = rngGetSeed();
    cap->loss = stats->loss;
    captureTensor(&cap->values, stats->output);
    freeTrainingStats(stats);
    forEachGrad(f->model, f->n, &cap->values);
    bnStateIo(f, false, &cap->values);
}

static void assertTwoCallsMatchLegacy(void (*build)(zooFixture_t *)) {
    zooFixture_t f;
    build(&f);
    captureTwoCalls(RUN_LEGACY, &f, &g_legacy);
    captureTwoCalls(RUN_DRIVER, &f, &g_driver);
    freeZoo(&f);
    assertSameValues(NULL);
    TEST_ASSERT_EQUAL_HEX32_MESSAGE(g_legacy.seedAfter, g_driver.seedAfter,
                                    "remat P2: the global stream after two calls");
}

/* ---- the NULL scheduler against the Legacy oracle, fixture by fixture ---- */

/* The plain wrapper as well: calculateGradsSequential and tracedGrads share
 * one body, and the sink only observes. */
void testNullPathMatchesLegacyOnTheMlp(void) {
    zooFixture_t f;
    buildMlp(&f);
    forEachGrad(f.model, f.n, NULL);
    trainingStats_t *legacy =
        legacyCalculateGrads(f.model, f.n, f.loss, REDUCTION_MEAN, f.x, f.y, NULL, NULL);
    g_legacy.values.used = 0;
    g_legacy.loss = legacy->loss;
    captureTensor(&g_legacy.values, legacy->output);
    freeTrainingStats(legacy);
    forEachGrad(f.model, f.n, &g_legacy.values);

    forEachGrad(f.model, f.n, NULL);
    trainingStats_t *driver =
        calculateGradsSequential(f.model, f.n, f.loss, REDUCTION_MEAN, f.x, f.y, NULL);
    g_driver.values.used = 0;
    g_driver.loss = driver->loss;
    captureTensor(&g_driver.values, driver->output);
    freeTrainingStats(driver);
    forEachGrad(f.model, f.n, &g_driver.values);
    freeZoo(&f);
    assertSameValues(NULL);
}

void testTracedMlpConforms(void) {
    assertNullPathMatchesLegacy(buildMlp);
}

void testHarCnnUnderCeConforms(void) {
    assertNullPathMatchesLegacy(buildHarCnn);
}

void testSoftmaxUnderMseConforms(void) {
    assertNullPathMatchesLegacy(buildSoftmaxMse);
}

void testFrozenLayerNormAboveDeepestConforms(void) {
    assertNullPathMatchesLegacy(buildFrozenLayerNorm);
}

void testFrozenGroupNormAboveDeepestConforms(void) {
    assertNullPathMatchesLegacy(buildFrozenGroupNorm);
}

void testTruncatedHarConforms(void) {
    assertNullPathMatchesLegacy(buildTruncatedHar);
}

void testAllFrozenModelConforms(void) {
    assertNullPathMatchesLegacy(buildAllFrozen);
}

void testDropoutConforms(void) {
    assertNullPathMatchesLegacy(buildDropout);
}

void testQuantizationToSymInt32WiresConforms(void) {
    assertNullPathMatchesLegacy(buildQuantSym);
}

void testFlattenAt0Conforms(void) {
    assertNullPathMatchesLegacy(buildFlattenAt0);
}

void testPerTensorBfpF1ModelConforms(void) {
    assertNullPathMatchesLegacy(buildF1Zoo);
}

void testGroupedBfpWireConforms(void) {
    assertNullPathMatchesLegacy(buildGroupedBfp);
}

void testSingleLayerUnderCeConforms(void) {
    assertNullPathMatchesLegacy(buildSingleLayerCe);
}

void testCeWithOnlyTheLastLinearTrainableConforms(void) {
    assertNullPathMatchesLegacy(buildCeLastLinearOnly);
}

void testBatchNormConforms(void) {
    assertNullPathMatchesLegacy(buildBatchNorm);
}

void testCeWithoutATrailingSoftmaxConforms(void) {
    assertNullPathMatchesLegacy(buildCeWithoutSoftmax);
}

void testConvTransposedConforms(void) {
    assertNullPathMatchesLegacy(buildConvTransposedTrainable);
}

void testFrozenConvTransposedAboveDeepestConforms(void) {
    assertNullPathMatchesLegacy(buildFrozenConvTransposed);
}

void testAdaptiveAvgPoolConforms(void) {
    assertNullPathMatchesLegacy(buildAdaptiveAvgPool);
}

void testTwoCallsInARowMatchLegacyOnBatchNorm(void) {
    assertTwoCallsMatchLegacy(buildBatchNorm);
}

void testTwoCallsInARowMatchLegacyOnDropout(void) {
    assertTwoCallsMatchLegacy(buildDropout);
}

/* The "agrad" trace phase fires BEFORE the layer's backward. No layer writes
 * its gradIn before #4 PR7 (in-place dx), so the traced bytes cannot tell
 * before from after; the layer's own weight grad can: zeroed before the
 * call, written by its backward. */
typedef struct agradProbe {
    zooFixture_t *f;
    size_t agrads;
    size_t writtenAtAgrad;
} agradProbe_t;

static void agradOrderSink(void *ctx, size_t layerIdx, layerType_t layerType, const char *phase,
                           tensor_t *tensor) {
    (void)layerType;
    (void)tensor;
    agradProbe_t *p = ctx;
    parameter_t *w = NULL;
    parameter_t *b = NULL;
    if (strcmp(phase, "agrad") != 0) {
        return;
    }
    p->agrads++;
    if (!layerParameters(p->f->model[layerIdx], &w, &b) || getGradFromParameter(w) == NULL) {
        return;
    }
    tensor_t *g = getGradFromParameter(w);
    const uint8_t *bytes = g->data;
    size_t n = calcNumberOfBytesForData(g->quantization, calcNumberOfElementsByTensor(g));
    for (size_t i = 0; i < n; i++) {
        if (bytes[i] != 0) {
            p->writtenAtAgrad++;
            return;
        }
    }
}

static agradProbe_t probeAgradOrder(runner_t runner) {
    zooFixture_t f;
    buildMlp(&f);
    forEachGrad(f.model, f.n, NULL);
    agradProbe_t p = {.f = &f};
    trainingStats_t *stats =
        (runner == RUN_LEGACY)
            ? legacyCalculateGrads(f.model, f.n, f.loss, REDUCTION_MEAN, f.x, f.y, agradOrderSink,
                                   &p)
            : tracedGrads(f.model, f.n, f.loss, REDUCTION_MEAN, f.x, f.y, agradOrderSink, &p, NULL);
    freeTrainingStats(stats);
    freeZoo(&f);
    return p;
}

void testAgradFiresBeforeTheLayersBackward(void) {
    agradProbe_t legacy = probeAgradOrder(RUN_LEGACY);
    agradProbe_t driver = probeAgradOrder(RUN_DRIVER);
    TEST_ASSERT_EQUAL_size_t_MESSAGE(3, legacy.agrads, "Legacy: agrad at layers 2, 1, 0");
    TEST_ASSERT_EQUAL_size_t_MESSAGE(0, legacy.writtenAtAgrad, "Legacy: agrad before backward");
    TEST_ASSERT_EQUAL_size_t_MESSAGE(3, driver.agrads, "driver: agrad at layers 2, 1, 0");
    TEST_ASSERT_EQUAL_size_t_MESSAGE(0, driver.writtenAtAgrad, "driver: agrad before backward");
}

/* ---- what the validating interpreter changes: checked inputs, named exits ---- */

/* ACT 0 is borrowed, so it is exempt from the bind generation but not from
 * residency. */
void testAnInputWithoutBytesExitsBeforeLayer0Runs(void) {
    zooFixture_t f;
    buildMlp(&f);
    uint8_t *bytes = f.x->data;
    f.x->data = NULL;
    ASSERT_EXITS_WITH_OUTPUT(
        1, "remat[heap]: step #0 FORWARD(layer 0) violates 'operand not resident: in ACT 0'",
        freeTrainingStats(
            calculateGradsSequential(f.model, f.n, f.loss, REDUCTION_MEAN, f.x, f.y, NULL)));
    f.x->data = bytes;
    freeZoo(&f);
}

/* The wire table cannot describe an empty model. The label has the input's
 * shape, so only the model size keeps the call from running. */
void testAnEmptyModelExitsNamingIt(void) {
    zooFixture_t f;
    buildMlp(&f);
    tensor_t *inputShapedLabel = makeFloatTensor((size_t[]){1, 4}, 2, 0.5f);
    ASSERT_EXITS_WITH_OUTPUT(1, "rematWireTableInit: modelSize == 0: nothing to schedule",
                             freeTrainingStats(calculateGradsSequential(
                                 f.model, 0, f.loss, REDUCTION_MEAN, f.x, inputShapedLabel, NULL)));
    freeTensor(inputShapedLabel);
    freeZoo(&f);
}

/* Wire headers carry no sparsity marker (an input's marker is not
 * propagated), so a marked input yields an unmarked output snapshot and
 * nothing is reserved for markers. The values still match Legacy. The memory
 * baseline is taken after the Legacy run, which leaks its markers
 * (freeSparsity is a no-op). */
void testAMarkedInputYieldsAnUnmarkedOutputAndLeaksNothing(void) {
    zooFixture_t f;
    buildMlp(&f);
    sparsity_t marker = {0};
    f.x->sparsity = &marker;
    captureRun(RUN_LEGACY, &f, NULL, &g_legacy);

    forEachGrad(f.model, f.n, NULL);
    g_driver.values.used = 0;
    size_t before = memProfileCurrentBytes();
    trainingStats_t *stats =
        calculateGradsSequential(f.model, f.n, f.loss, REDUCTION_MEAN, f.x, f.y, NULL);
    bool outputMarked = stats->output->sparsity != NULL;
    g_driver.loss = stats->loss;
    captureTensor(&g_driver.values, stats->output);
    freeTrainingStats(stats);
    size_t after = memProfileCurrentBytes();
    forEachGrad(f.model, f.n, &g_driver.values);

    f.x->sparsity = NULL;
    freeZoo(&f);
    TEST_ASSERT_FALSE_MESSAGE(outputMarked, "the output snapshot carries no sparsity marker");
    TEST_ASSERT_EQUAL_size_t_MESSAGE(before, after, "no marker is reserved, so none leaks");
    assertSameValues(NULL);
}

#ifndef ODT_TEST_ASAN
/* A failed ephemeral build exits naming it. HEAP init reserves no wire data,
 * only the table slab and the plan block, so the slab must fail: one
 * Quantization layer to a BFP wire grouped by 2, over a borrowed [1, 2^60]
 * input (never read at init), puts 2^59 exponent bytes of ACT 1 in the slab,
 * which no 64-bit host can reserve. Host-only (LP64); skipped under ASan,
 * which aborts on oversized requests (the #4 PR1 tests that pin remat R8's
 * recoverable init failure do the same). macOS malloc prints a "can't
 * allocate region" warning to stderr here. */
void testAFailedEphemeralBuildExitsNamingIt(void) {
    quantization_t *bfp = quantizationInitBfpGrouped(8, 8, HALF_AWAY, 2, 2);
    layer_t *model[1] = {makeQuant(bfp, &g_floatQ)};
    inputLike_t in;
    tensor_t *x = makeInput(&in, (size_t[]){1, (size_t)1 << 60}, 2, &g_floatQ);
    tensor_t *y = makeFloatTensor((size_t[]){1, 1}, 2, 0.5f);
    ASSERT_EXITS_WITH_OUTPUT(1, "calculateGrads: ephemeral HEAP scheduler: reserveMemory failed",
                             freeTrainingStats(calculateGradsSequential(
                                 model, 1, defaultLossConfig(MSE), REDUCTION_MEAN, x, y, NULL)));
    freeTensor(y);
    freeModel(model, 1);
    freeQuantization(bfp);
}
#endif

/* ---- a caller's scheduler through the call struct: borrowed, never torn down ---- */

/* remat P1's values of one call from zeroed grads: the loss, the output snapshot and
 * every parameter grad. */
static void captureValues(runner_t runner, zooFixture_t *f, const trainingCall_t *call,
                          runCapture_t *cap) {
    forEachGrad(f->model, f->n, NULL);
    cap->values.used = 0;
    trainingStats_t *stats =
        (runner == RUN_LEGACY)
            ? legacyCalculateGrads(f->model, f->n, f->loss, REDUCTION_MEAN, f->x, f->y, NULL, NULL)
            : calculateGradsSequential(f->model, f->n, f->loss, REDUCTION_MEAN, f->x, f->y, call);
    cap->loss = stats->loss;
    captureTensor(&cap->values, stats->output);
    freeTrainingStats(stats);
    forEachGrad(f->model, f->n, &cap->values);
}

/* Built from the fixture's own model and input, so the first bind's key check
 * passes. */
static void initFixtureHeap(rematScheduler_t *s, zooFixture_t *f) {
    TEST_ASSERT_TRUE(rematHeapInit(s, f->model, f->n, f->loss, f->x, NULL));
}

static rematReport_t reportOf(const rematScheduler_t *s) {
    rematReport_t r;
    rematSchedulerReport(s, &r);
    return r;
}

/* A bind resets the observed peak and every bound wire raises it; nothing
 * lowers it before the next bind. So a scheduler that ran the call reports the
 * planned peak afterwards, and one the driver ignored keeps the 0 of a fresh
 * init. */
void testCalculateGradsSequentialRunsOnTheCallersScheduler(void) {
    zooFixture_t f;
    buildMlp(&f);
    captureValues(RUN_LEGACY, &f, NULL, &g_legacy);
    rematScheduler_t s;
    initFixtureHeap(&s, &f);
    size_t observedBefore = reportOf(&s).observedPeakLiveBytes;
    captureValues(RUN_DRIVER, &f, &(trainingCall_t){.remat = &s}, &g_driver);
    rematReport_t after = reportOf(&s);
    rematSchedulerDeinit(&s);
    freeZoo(&f);
    TEST_ASSERT_EQUAL_size_t(0, observedBefore);
    TEST_ASSERT_TRUE(after.peakLiveBytes > 0);
    TEST_ASSERT_EQUAL_size_t_MESSAGE(after.peakLiveBytes, after.observedPeakLiveBytes,
                                     "the call ran on the caller's scheduler");
    assertSameValues(NULL);
}

void testTracedGradsRunsOnTheCallersScheduler(void) {
    zooFixture_t f;
    buildMlp(&f);
    rematScheduler_t s;
    initFixtureHeap(&s, &f);
    g_driver.trace.used = 0;
    freeTrainingStats(tracedGrads(f.model, f.n, f.loss, REDUCTION_MEAN, f.x, f.y, recordingSink,
                                  &g_driver.trace, &(trainingCall_t){.remat = &s}));
    rematReport_t after = reportOf(&s);
    rematSchedulerDeinit(&s);
    freeZoo(&f);
    TEST_ASSERT_TRUE(g_driver.trace.used > 0);
    TEST_ASSERT_TRUE(after.peakLiveBytes > 0);
    TEST_ASSERT_EQUAL_size_t_MESSAGE(after.peakLiveBytes, after.observedPeakLiveBytes,
                                     "the traced call ran on the caller's scheduler");
}

static void twoCallsOnOneScheduler(zooFixture_t *f, rematScheduler_t *s) {
    const trainingCall_t call = {.remat = s};
    freeTrainingStats(
        calculateGradsSequential(f->model, f->n, f->loss, REDUCTION_MEAN, f->x, f->y, &call));
    freeTrainingStats(
        calculateGradsSequential(f->model, f->n, f->loss, REDUCTION_MEAN, f->x, f->y, &call));
}

/* The driver borrows a caller's scheduler and tears down only its own
 * ephemeral one, so a second call on the same instance runs. */
void testTheCallersSchedulerSurvivesTheCall(void) {
    zooFixture_t f;
    buildMlp(&f);
    rematScheduler_t s;
    initFixtureHeap(&s, &f);
    ASSERT_EXITS_WITH_OUTPUT(0, "", twoCallsOnOneScheduler(&f, &s));
    rematSchedulerDeinit(&s);
    freeZoo(&f);
}

/* A NULL call and a zero-initialised one mean the same: the ephemeral
 * scheduler. The death-test wrapper turns a NULL dereference into a readable
 * failure instead of a crashed binary. */
void testAZeroInitialisedCallIsTheNullScheduler(void) {
    zooFixture_t f;
    buildMlp(&f);
    ASSERT_EXITS_WITH_OUTPUT(
        0, "",
        freeTrainingStats(calculateGradsSequential(f.model, f.n, f.loss, REDUCTION_MEAN, f.x, f.y,
                                                   &(trainingCall_t){0})));
    captureValues(RUN_LEGACY, &f, NULL, &g_legacy);
    captureValues(RUN_DRIVER, &f, &(trainingCall_t){0}, &g_driver);
    freeZoo(&f);
    assertSameValues(NULL);
}

/* ---- the default plan of a training call without a scheduler ---- */

/* What a NULL scheduler resolves to is the training call's rule, visible to
 * its callers: LIVENESS (remat D15). The scheduler library's own NULL spec
 * stays STORE_ALL. */
void testANullSchedulerResolvesToTheNamedDefaultPlan(void) {
    const rematPlanSpec_t *spec = calculateGradsDefaultPlanSpec();
    TEST_ASSERT_NOT_NULL(spec);
    TEST_ASSERT_EQUAL_INT(REMAT_PLAN_LIVENESS, spec->policy);
}

#ifdef ODT_MEM_PROFILE
/* HAR under CE: BACKWARD 10..0 (the Softmax at 11 has none). */
#define HAR_CNN_AGRADS 11u

/* Live bytes at every "agrad" event of one call, relative to a mark taken
 * right before the call. The sink asserts nothing: a Unity failure would
 * longjmp out of the driver mid-call. */
typedef struct liveSampler {
    size_t mark;
    size_t agrads;
    bool underflow;
    size_t at[ZOO_MAX_LAYERS];
} liveSampler_t;

static void liveBytesSink(void *ctx, size_t layerIdx, layerType_t layerType, const char *phase,
                          tensor_t *tensor) {
    (void)layerType;
    (void)tensor;
    liveSampler_t *s = ctx;
    if (strcmp(phase, "agrad") != 0 || layerIdx >= ZOO_MAX_LAYERS) {
        return;
    }
    size_t cur = memProfileCurrentBytes();
    if (cur < s->mark) {
        s->underflow = true;
        return;
    }
    s->at[layerIdx] = cur - s->mark;
    s->agrads++;
}

/* One traced HAR CNN call, on the NULL path or on an explicit HEAP scheduler
 * built from spec before the mark. The NULL path reserves its table and plan
 * inside the call, after the mark, so the explicit scheduler's metadata is
 * added back: both then count the same blocks. */
static liveSampler_t sampleHarCnn(bool explicitScheduler, const rematPlanSpec_t *spec) {
    zooFixture_t f;
    buildHarCnn(&f);
    rematScheduler_t s = {0};
    size_t meta = 0;
    if (explicitScheduler) {
        TEST_ASSERT_TRUE(rematHeapInit(&s, f.model, f.n, f.loss, f.x, spec));
        meta = reportOf(&s).metadataBytes;
    }
    const trainingCall_t call = {.remat = explicitScheduler ? &s : NULL};
    liveSampler_t smp = {0};
    smp.mark = memProfileCurrentBytes();
    freeTrainingStats(
        tracedGrads(f.model, f.n, f.loss, REDUCTION_MEAN, f.x, f.y, liveBytesSink, &smp, &call));
    rematSchedulerDeinit(&s);
    freeZoo(&f);
    for (size_t l = 0; l < HAR_CNN_AGRADS; l++) {
        smp.at[l] += meta;
    }
    return smp;
}

/* The NULL path holds exactly what a scheduler built from the named default
 * holds, at every backward step: the default is the whole difference. */
void testANullSchedulerHoldsWhatTheDefaultPlansSchedulerHolds(void) {
    liveSampler_t nullPath = sampleHarCnn(false, NULL);
    liveSampler_t viaDefault = sampleHarCnn(true, calculateGradsDefaultPlanSpec());
    TEST_ASSERT_FALSE_MESSAGE(nullPath.underflow, "NULL path: live bytes below the mark");
    TEST_ASSERT_FALSE_MESSAGE(viaDefault.underflow, "default plan: live bytes below the mark");
    TEST_ASSERT_EQUAL_size_t_MESSAGE(HAR_CNN_AGRADS, nullPath.agrads, "NULL path: agrads");
    TEST_ASSERT_EQUAL_size_t_MESSAGE(HAR_CNN_AGRADS, viaDefault.agrads, "default plan: agrads");
    for (size_t l = 0; l < HAR_CNN_AGRADS; l++) {
        TEST_ASSERT_EQUAL_size_t_MESSAGE(viaDefault.at[l], nullPath.at[l],
                                         "live bytes at the agrad of this layer index");
    }
}

/* What the default buys: mid-backward the NULL path holds less than a
 * STORE_ALL scheduler at every backward step (remat D15). */
void testANullSchedulerHoldsLessThanStoreAllMidBackward(void) {
    liveSampler_t nullPath = sampleHarCnn(false, NULL);
    liveSampler_t storeAll = sampleHarCnn(true, &(rematPlanSpec_t){.policy = REMAT_PLAN_STORE_ALL});
    TEST_ASSERT_FALSE_MESSAGE(nullPath.underflow, "NULL path: live bytes below the mark");
    TEST_ASSERT_FALSE_MESSAGE(storeAll.underflow, "STORE_ALL: live bytes below the mark");
    TEST_ASSERT_EQUAL_size_t_MESSAGE(HAR_CNN_AGRADS, nullPath.agrads, "NULL path: agrads");
    TEST_ASSERT_EQUAL_size_t_MESSAGE(HAR_CNN_AGRADS, storeAll.agrads, "STORE_ALL: agrads");
    for (size_t l = 0; l < HAR_CNN_AGRADS; l++) {
        TEST_ASSERT_LESS_THAN_size_t_MESSAGE(storeAll.at[l], nullPath.at[l],
                                             "live bytes at the agrad of this layer index");
    }
}
#endif

/* ---- the conformance matrix: {ARENA, HEAP} x {STORE_ALL, LIVENESS} ---- */

typedef struct matrixCell {
    const char *name;
    rematSchedulerType_t row;
    const rematPlanSpec_t *spec;
} matrixCell_t;

#define MATRIX_CELLS 4
static const matrixCell_t g_cells[MATRIX_CELLS] = {
    {"arena/store-all", REMAT_ARENA, NULL},
    {"arena/liveness", REMAT_ARENA, &g_liveness},
    {"heap/store-all", REMAT_HEAP, NULL},
    {"heap/liveness", REMAT_HEAP, &g_liveness},
};

/* A persistent scheduler of the cell, built from the fixture's own model and
 * input. */
static void initCell(rematScheduler_t *s, const matrixCell_t *cell, zooFixture_t *f) {
    bool built = (cell->row == REMAT_ARENA)
                     ? rematArenaInit(s, f->model, f->n, f->loss, f->x, cell->spec)
                     : rematHeapInit(s, f->model, f->n, f->loss, f->x, cell->spec);
    TEST_ASSERT_TRUE_MESSAGE(built, say(cell->name, "init"));
}

/* One cell's own observations; the call's remat P1-P5 and P9 land in g_driver. */
typedef struct cellRun {
    rematReport_t report;
    size_t memBeforeInit;
    size_t memAfterDeinit;
} cellRun_t;

/* A fresh fixture per cell, so every cell starts from the state Legacy saw.
 * remat P5's call bracket starts after the init, whose blocks the caller owns, and
 * the deinit must return them. */
static cellRun_t captureCell(void (*build)(zooFixture_t *), const matrixCell_t *cell) {
    zooFixture_t f;
    build(&f);
    captureRun(RUN_LEGACY, &f, NULL, &g_legacy);
    cellRun_t c;
    c.memBeforeInit = memProfileCurrentBytes();
    rematScheduler_t s;
    initCell(&s, cell, &f);
    captureRun(RUN_DRIVER, &f, &(trainingCall_t){.remat = &s}, &g_driver);
    c.report = reportOf(&s);
    rematSchedulerDeinit(&s);
    c.memAfterDeinit = memProfileCurrentBytes();
    freeZoo(&f);
    return c;
}

/* One test per fixture and row, so a failing row cannot mask the other: the
 * poison mutation must turn both red. */
static void assertRowMatchesLegacy(void (*build)(zooFixture_t *), rematSchedulerType_t row) {
    for (size_t k = 0; k < MATRIX_CELLS; k++) {
        const matrixCell_t *cell = &g_cells[k];
        if (cell->row != row) {
            continue;
        }
        cellRun_t c = captureCell(build, cell);
        TEST_ASSERT_EQUAL_INT_MESSAGE(cell->row, c.report.type, say(cell->name, "the cell's row"));
        TEST_ASSERT_EQUAL_INT_MESSAGE(cell->spec == NULL ? REMAT_PLAN_STORE_ALL
                                                         : cell->spec->policy,
                                      c.report.policy, say(cell->name, "the cell's plan"));
        assertConforms(cell->name);
        TEST_ASSERT_TRUE_MESSAGE(c.report.peakLiveBytes > 0,
                                 say(cell->name, "remat P8: a planned peak"));
        TEST_ASSERT_EQUAL_size_t_MESSAGE(
            c.report.peakLiveBytes, c.report.observedPeakLiveBytes,
            say(cell->name, "remat P8: the call ran on the cell's scheduler"));
        TEST_ASSERT_EQUAL_size_t_MESSAGE(
            c.memBeforeInit, c.memAfterDeinit,
            say(cell->name, "remat P5: the deinit returns what the init reserved"));
    }
}

void testMlpMatchesLegacyOnArena(void) {
    assertRowMatchesLegacy(buildMlp, REMAT_ARENA);
}

void testMlpMatchesLegacyOnHeap(void) {
    assertRowMatchesLegacy(buildMlp, REMAT_HEAP);
}

void testHarCnnMatchesLegacyOnArena(void) {
    assertRowMatchesLegacy(buildHarCnn, REMAT_ARENA);
}

void testHarCnnMatchesLegacyOnHeap(void) {
    assertRowMatchesLegacy(buildHarCnn, REMAT_HEAP);
}

void testTruncatedHarMatchesLegacyOnArena(void) {
    assertRowMatchesLegacy(buildTruncatedHar, REMAT_ARENA);
}

void testTruncatedHarMatchesLegacyOnHeap(void) {
    assertRowMatchesLegacy(buildTruncatedHar, REMAT_HEAP);
}

void testSoftmaxMseMatchesLegacyOnArena(void) {
    assertRowMatchesLegacy(buildSoftmaxMse, REMAT_ARENA);
}

void testSoftmaxMseMatchesLegacyOnHeap(void) {
    assertRowMatchesLegacy(buildSoftmaxMse, REMAT_HEAP);
}

void testFrozenLayerNormMatchesLegacyOnArena(void) {
    assertRowMatchesLegacy(buildFrozenLayerNorm, REMAT_ARENA);
}

void testFrozenLayerNormMatchesLegacyOnHeap(void) {
    assertRowMatchesLegacy(buildFrozenLayerNorm, REMAT_HEAP);
}

void testFrozenGroupNormMatchesLegacyOnArena(void) {
    assertRowMatchesLegacy(buildFrozenGroupNorm, REMAT_ARENA);
}

void testFrozenGroupNormMatchesLegacyOnHeap(void) {
    assertRowMatchesLegacy(buildFrozenGroupNorm, REMAT_HEAP);
}

void testAllFrozenMatchesLegacyOnArena(void) {
    assertRowMatchesLegacy(buildAllFrozen, REMAT_ARENA);
}

void testAllFrozenMatchesLegacyOnHeap(void) {
    assertRowMatchesLegacy(buildAllFrozen, REMAT_HEAP);
}

void testDropoutMatchesLegacyOnArena(void) {
    assertRowMatchesLegacy(buildDropout, REMAT_ARENA);
}

void testDropoutMatchesLegacyOnHeap(void) {
    assertRowMatchesLegacy(buildDropout, REMAT_HEAP);
}

void testQuantSymMatchesLegacyOnArena(void) {
    assertRowMatchesLegacy(buildQuantSym, REMAT_ARENA);
}

void testQuantSymMatchesLegacyOnHeap(void) {
    assertRowMatchesLegacy(buildQuantSym, REMAT_HEAP);
}

void testFlattenAt0MatchesLegacyOnArena(void) {
    assertRowMatchesLegacy(buildFlattenAt0, REMAT_ARENA);
}

void testFlattenAt0MatchesLegacyOnHeap(void) {
    assertRowMatchesLegacy(buildFlattenAt0, REMAT_HEAP);
}

void testF1ZooMatchesLegacyOnArena(void) {
    assertRowMatchesLegacy(buildF1Zoo, REMAT_ARENA);
}

void testF1ZooMatchesLegacyOnHeap(void) {
    assertRowMatchesLegacy(buildF1Zoo, REMAT_HEAP);
}

void testGroupedBfpMatchesLegacyOnArena(void) {
    assertRowMatchesLegacy(buildGroupedBfp, REMAT_ARENA);
}

void testGroupedBfpMatchesLegacyOnHeap(void) {
    assertRowMatchesLegacy(buildGroupedBfp, REMAT_HEAP);
}

void testSingleLayerCeMatchesLegacyOnArena(void) {
    assertRowMatchesLegacy(buildSingleLayerCe, REMAT_ARENA);
}

void testSingleLayerCeMatchesLegacyOnHeap(void) {
    assertRowMatchesLegacy(buildSingleLayerCe, REMAT_HEAP);
}

void testCeLastLinearOnlyMatchesLegacyOnArena(void) {
    assertRowMatchesLegacy(buildCeLastLinearOnly, REMAT_ARENA);
}

void testCeLastLinearOnlyMatchesLegacyOnHeap(void) {
    assertRowMatchesLegacy(buildCeLastLinearOnly, REMAT_HEAP);
}

void testCeWithoutSoftmaxMatchesLegacyOnArena(void) {
    assertRowMatchesLegacy(buildCeWithoutSoftmax, REMAT_ARENA);
}

void testCeWithoutSoftmaxMatchesLegacyOnHeap(void) {
    assertRowMatchesLegacy(buildCeWithoutSoftmax, REMAT_HEAP);
}

void testBatchNormMatchesLegacyOnArena(void) {
    assertRowMatchesLegacy(buildBatchNorm, REMAT_ARENA);
}

void testBatchNormMatchesLegacyOnHeap(void) {
    assertRowMatchesLegacy(buildBatchNorm, REMAT_HEAP);
}

void testConvTransposedMatchesLegacyOnArena(void) {
    assertRowMatchesLegacy(buildConvTransposedTrainable, REMAT_ARENA);
}

void testConvTransposedMatchesLegacyOnHeap(void) {
    assertRowMatchesLegacy(buildConvTransposedTrainable, REMAT_HEAP);
}

void testFrozenConvTransposedMatchesLegacyOnArena(void) {
    assertRowMatchesLegacy(buildFrozenConvTransposed, REMAT_ARENA);
}

void testFrozenConvTransposedMatchesLegacyOnHeap(void) {
    assertRowMatchesLegacy(buildFrozenConvTransposed, REMAT_HEAP);
}

void testAdaptiveAvgPoolMatchesLegacyOnArena(void) {
    assertRowMatchesLegacy(buildAdaptiveAvgPool, REMAT_ARENA);
}

void testAdaptiveAvgPoolMatchesLegacyOnHeap(void) {
    assertRowMatchesLegacy(buildAdaptiveAvgPool, REMAT_HEAP);
}

/* ---- one persistent instance across calls (remat P6, P9) ---- */

typedef void (*fixtureEdit_t)(zooFixture_t *f);

/* Two calls from zeroed grads, without zeroing between them, and `edit`
 * applied to the live fixture between them. */
static void captureEditedTwoCalls(runner_t runner, zooFixture_t *f, const trainingCall_t *call,
                                  fixtureEdit_t edit, runCapture_t *cap) {
    forEachGrad(f->model, f->n, NULL);
    cap->values.used = 0;
    rngSetSeed(ZOO_SEED);
    freeTrainingStats(runOnce(runner, f, call));
    edit(f);
    trainingStats_t *stats = runOnce(runner, f, call);
    cap->seedAfter = rngGetSeed();
    cap->loss = stats->loss;
    captureTensor(&cap->values, stats->output);
    freeTrainingStats(stats);
    forEachGrad(f->model, f->n, &cap->values);
}

/* Legacy reads every config live; each cell's scheduler is built on the
 * unedited fixture before the first call, so its second bind must re-derive
 * the edited wire (a key-preserving edit, remat P6). */
static void assertEditedTwoCallsMatchLegacy(void (*build)(zooFixture_t *), fixtureEdit_t edit) {
    for (size_t k = 0; k < MATRIX_CELLS; k++) {
        const matrixCell_t *cell = &g_cells[k];
        zooFixture_t f;
        build(&f);
        captureEditedTwoCalls(RUN_LEGACY, &f, NULL, edit, &g_legacy);
        freeZoo(&f);
        build(&f);
        rematScheduler_t s;
        initCell(&s, cell, &f);
        captureEditedTwoCalls(RUN_DRIVER, &f, &(trainingCall_t){.remat = &s}, edit, &g_driver);
        rematSchedulerDeinit(&s);
        freeZoo(&f);
        assertSameValues(cell->name);
        TEST_ASSERT_EQUAL_HEX32_MESSAGE(
            g_legacy.seedAfter, g_driver.seedAfter,
            say(cell->name, "remat P2: the global stream after two calls"));
    }
}

/* buildQuantSym's ACT template is its first kept template. */
static symInt32QConfig_t *quantSymActConfig(zooFixture_t *f) {
    return f->templates[0]->qConfig;
}

static void raiseTheActWidthTo16(zooFixture_t *f) {
    quantSymActConfig(f)->qMaxBits = 16;
}

static void roundTheActStochastically(zooFixture_t *f) {
    quantSymActConfig(f)->roundingMode = SR_HALF_AWAY;
}

/* remat P6: qMaxBits 8 -> 16 between two calls on one instance. */
void testTwoCallsAcrossAWidthEditMatchLegacy(void) {
    assertEditedTwoCallsMatchLegacy(buildQuantSym, raiseTheActWidthTo16);
}

/* The draw-count half of testBindRederivesRoundingModeAndDrawCount: after the
 * switch to stochastic rounding the second call draws from the global stream,
 * as many times as Legacy's (remat P2). */
void testTwoCallsAcrossARoundingEditMatchLegacy(void) {
    assertEditedTwoCallsMatchLegacy(buildQuantSym, roundTheActStochastically);
}

/* A [1, 4, 4] input quantized to BFP in numGroups groups of groupSize. */
static tensor_t *makeBfpInput(size_t numGroups, size_t groupSize) {
    tensor_t *x = makeFloatTensor((size_t[]){1, 4, 4}, 3, 1.0f);
    quantization_t *q = quantizationInitBfpGrouped(8, 8, HALF_AWAY, numGroups, groupSize);
    requantizeTensorInPlace(x, q);
    freeQuantization(q);
    return x;
}

/* Flatten at 0 inherits the input's config: ACT 1 is BFP in the input's
 * grouping, then back to FLOAT32 for the trained Linear. */
static void buildBfpFlattenAt0(zooFixture_t *f) {
    beginZoo(f);
    f->model[0] = flattenLayerInit();
    f->model[1] = makeQuant(&g_floatQ, &g_floatQ);
    f->model[2] = makeLinear(16, 2, false);
    f->n = 3;
    f->x = makeBfpInput(4, 4);
    mseLabel(f, 2);
}

/* The caller re-quantizes its input between calls: {4 x 4} -> {2 x 8} shrinks
 * within the built exponent capacity (testBindRederivesFlattenBfpGroupingFromTheLiveInput). */
static void regroupTheInput(zooFixture_t *f) {
    freeTensor(f->x);
    f->x = makeBfpInput(2, 8);
}

void testTwoCallsAcrossAnInputRegroupMatchLegacy(void) {
    assertEditedTwoCallsMatchLegacy(buildBfpFlattenAt0, regroupTheInput);
}

/* remat P9, second half: a scheduler built on sample A runs sample B (same shape,
 * other bytes) exactly as Legacy runs B, and leaves B untouched. */
void testASchedulerBuiltOnSampleARunsSampleB(void) {
    for (size_t k = 0; k < MATRIX_CELLS; k++) {
        const matrixCell_t *cell = &g_cells[k];
        zooFixture_t f;
        buildMlp(&f);
        tensor_t *sampleA = f.x;
        tensor_t *sampleB = makeFloatTensor((size_t[]){1, 4}, 2, -0.75f);
        f.x = sampleB;
        captureRun(RUN_LEGACY, &f, NULL, &g_legacy);
        f.x = sampleA;
        rematScheduler_t s;
        initCell(&s, cell, &f);
        f.x = sampleB;
        captureRun(RUN_DRIVER, &f, &(trainingCall_t){.remat = &s}, &g_driver);
        rematSchedulerDeinit(&s);
        f.x = sampleA;
        freeTensor(sampleB);
        freeZoo(&f);
        assertConforms(cell->name);
    }
}

/* ---- remat P2 has teeth: a row that draws is caught ---- */

/* A row that draws from the global stream in next(): every value still
 * matches, only remat P2 can see it. */
static bool drawingNext(rematScheduler_t *s, rematStep_t *st) {
    (void)rngNextFloat();
    return decoratedNext(s, st);
}

static const rematSchedulerFunctions_t g_drawing = {.name = "drawing",
                                                    .begin = decoratedBegin,
                                                    .next = drawingNext,
                                                    .done = decoratedDone,
                                                    .end = decoratedEnd,
                                                    .deinit = decoratedDeinit};

void testADrawingRowFailsP2AgainstLegacy(void) {
    for (size_t k = 0; k < MATRIX_CELLS; k++) {
        const matrixCell_t *cell = &g_cells[k];
        zooFixture_t f;
        buildMlp(&f);
        captureRun(RUN_LEGACY, &f, NULL, &g_legacy);
        rematScheduler_t s;
        initCell(&s, cell, &f);
        s.fns = &g_drawing;
        captureRun(RUN_DRIVER, &f, &(trainingCall_t){.remat = &s}, &g_driver);
        s.fns = &rematSchedulerFunctions[s.type];
        rematSchedulerDeinit(&s);
        freeZoo(&f);
        assertSameValues(cell->name);
        TEST_ASSERT_NOT_EQUAL_HEX32_MESSAGE(
            g_legacy.seedAfter, g_driver.seedAfter,
            say(cell->name, "remat P2: a drawing row moves the stream"));
    }
}

/* ---- evaluation on the matrix (remat D19) ---- */

typedef struct zooEntry {
    const char *name;
    void (*build)(zooFixture_t *f);
} zooEntry_t;

/* The returned output is SYM_INT32: its scale is compared. */
static void buildSymOutput(zooFixture_t *f) {
    beginZoo(f);
    quantization_t *act = keepTemplate(f, quantizationInitSymInt32WithBits(HALF_AWAY, 8));
    f->model[0] = makeLinear(4, 2, false);
    f->model[1] = makeQuant(act, &g_floatQ);
    f->n = 2;
    f->x = makeFloatTensor((size_t[]){1, 4}, 2, 1.0f);
    mseLabel(f, 2);
}

/* The returned output is grouped BFP: its exponents are compared. */
static void buildBfpOutput(zooFixture_t *f) {
    beginZoo(f);
    quantization_t *bfp = keepTemplate(f, quantizationInitBfpGrouped(8, 8, HALF_AWAY, 2, 4));
    f->model[0] = makeLinear(8, 8, false);
    f->model[1] = makeQuant(bfp, &g_floatQ);
    f->n = 2;
    f->x = makeFloatTensor((size_t[]){1, 8}, 2, 1.0f);
    mseLabel(f, 8);
}

#define ZOO_FIXTURES 21
static const zooEntry_t g_zoo[ZOO_FIXTURES] = {
    {"mlp", buildMlp},
    {"har-cnn", buildHarCnn},
    {"softmax-mse", buildSoftmaxMse},
    {"frozen-layernorm", buildFrozenLayerNorm},
    {"frozen-groupnorm", buildFrozenGroupNorm},
    {"truncated-har", buildTruncatedHar},
    {"all-frozen", buildAllFrozen},
    {"dropout", buildDropout},
    {"quant-sym", buildQuantSym},
    {"flatten-at-0", buildFlattenAt0},
    {"f1-bfp", buildF1Zoo},
    {"grouped-bfp", buildGroupedBfp},
    {"single-layer-ce", buildSingleLayerCe},
    {"ce-last-linear-only", buildCeLastLinearOnly},
    {"batchnorm", buildBatchNorm},
    {"ce-without-softmax", buildCeWithoutSoftmax},
    {"sym-output", buildSymOutput},
    {"bfp-output", buildBfpOutput},
    {"conv-transposed", buildConvTransposedTrainable},
    {"frozen-conv-transposed", buildFrozenConvTransposed},
    {"adaptive-avgpool", buildAdaptiveAvgPool},
};

/* One inferenceWithLoss call: the loss, the returned output with its shape
 * and dynamic state (SYM scale, BFP exponents), and the memory bracket. */
typedef struct evalCapture {
    float loss;
    size_t memBefore;
    size_t memAfter;
    blob_t output;
    blob_t state;
} evalCapture_t;
static evalCapture_t g_evalNull;
static evalCapture_t g_evalCell;

static void captureEval(zooFixture_t *f, const trainingCall_t *call, evalCapture_t *cap) {
    snapshotInput(&g_input, f->x);
    cap->output.used = 0;
    cap->memBefore = memProfileCurrentBytes();
    inferenceStats_t *stats =
        inferenceWithLoss(f->model, f->n, f->x, f->y, f->loss.funcType, REDUCTION_MEAN, call);
    cap->loss = stats->loss;
    captureTensor(&cap->output, stats->output);
    freeInferenceStats(stats);
    cap->memAfter = memProfileCurrentBytes();
    cap->state.used = 0;
    forEachGrad(f->model, f->n, &cap->state);
    bnStateIo(f, false, &cap->state);
    assertInputUnchanged(&g_input, f->x);
}

static char g_evalMessage[160];
static const char *sayEval(const char *fixture, const char *cell, const char *what) {
    (void)snprintf(g_evalMessage, sizeof g_evalMessage, "%s on %s: %s", fixture, cell, what);
    return g_evalMessage;
}

static void assertEvalCaptureMatches(const char *fixture, const char *cell, const char *what) {
    TEST_ASSERT_EQUAL_MEMORY_MESSAGE(&g_evalNull.loss, &g_evalCell.loss, sizeof(float),
                                     sayEval(fixture, cell, what));
    TEST_ASSERT_EQUAL_size_t_MESSAGE(g_evalNull.output.used, g_evalCell.output.used,
                                     sayEval(fixture, cell, what));
    TEST_ASSERT_EQUAL_MEMORY_MESSAGE(g_evalNull.output.bytes, g_evalCell.output.bytes,
                                     g_evalNull.output.used, sayEval(fixture, cell, what));
    TEST_ASSERT_EQUAL_size_t_MESSAGE(g_evalNull.state.used, g_evalCell.state.used,
                                     sayEval(fixture, cell, "model state after eval"));
    /* all-frozen without BatchNorm has no state; Unity rejects a 0-byte compare */
    if (g_evalNull.state.used > 0) {
        TEST_ASSERT_EQUAL_MEMORY_MESSAGE(g_evalNull.state.bytes, g_evalCell.state.bytes,
                                         g_evalNull.state.used,
                                         sayEval(fixture, cell, "model state after eval"));
    }
}

/* Every zoo fixture on every cell of one row: output, loss and the model state
 * after the call memcmp-equal to the NULL path (remat D9), the input untouched
 * (remat P9, in captureEval), the observed peak the EVAL program's (remat P8),
 * and every byte the call reserved returned with the stats (remat P5). */
static void assertEvalMatchesTheNullPath(rematSchedulerType_t row) {
    for (size_t z = 0; z < ZOO_FIXTURES; z++) {
        for (size_t k = 0; k < MATRIX_CELLS; k++) {
            const matrixCell_t *cell = &g_cells[k];
            if (cell->row != row) {
                continue;
            }
            zooFixture_t f;
            g_zoo[z].build(&f);
            captureEval(&f, NULL, &g_evalNull);
            rematScheduler_t s;
            initCell(&s, cell, &f);
            captureEval(&f, &(trainingCall_t){.remat = &s}, &g_evalCell);
            rematReport_t report = reportOf(&s);
            size_t evalPeak = rematPlanProgram(s.plan, REMAT_MODE_EVAL)->peakLiveBytes;
            rematSchedulerDeinit(&s);
            freeZoo(&f);
            const char *name = g_zoo[z].name;
            assertEvalCaptureMatches(name, cell->name, "the output");
            TEST_ASSERT_EQUAL_size_t_MESSAGE(evalPeak, report.observedPeakLiveBytes,
                                             sayEval(name, cell->name, "remat P8 in eval"));
            TEST_ASSERT_EQUAL_size_t_MESSAGE(g_evalCell.memBefore, g_evalCell.memAfter,
                                             sayEval(name, cell->name, "remat P5 in eval"));
        }
    }
}

/* Evaluation on a scheduler drops an input's sparsity marker, as training
 * does: the wire headers carry none, so the output is unmarked and nothing is
 * reserved for a marker, the values equal the unmarked NULL path's, and the
 * marked input stays unchanged (remat P9). The NULL path runs unmarked here:
 * it would hand the output a fresh marker, and freeSparsity is a no-op. */
void testAMarkedInputEvaluatesToAnUnmarkedOutputOnEitherRow(void) {
    for (size_t k = 0; k < MATRIX_CELLS; k++) {
        const matrixCell_t *cell = &g_cells[k];
        zooFixture_t f;
        buildMlp(&f);
        captureEval(&f, NULL, &g_evalNull);
        rematScheduler_t s;
        initCell(&s, cell, &f);
        sparsity_t marker = {0};
        f.x->sparsity = &marker;
        snapshotInput(&g_input, f.x);
        size_t before = memProfileCurrentBytes();
        inferenceStats_t *stats = inferenceWithLoss(f.model, f.n, f.x, f.y, f.loss.funcType,
                                                    REDUCTION_MEAN, &(trainingCall_t){.remat = &s});
        assertInputUnchanged(&g_input, f.x);
        bool outputMarked = stats->output->sparsity != NULL;
        g_evalCell.loss = stats->loss;
        g_evalCell.output.used = 0;
        captureTensor(&g_evalCell.output, stats->output);
        freeInferenceStats(stats);
        size_t after = memProfileCurrentBytes();
        f.x->sparsity = NULL;
        rematSchedulerDeinit(&s);
        freeZoo(&f);
        TEST_ASSERT_FALSE_MESSAGE(outputMarked,
                                  sayEval("mlp", cell->name, "the output carries no marker"));
        TEST_ASSERT_EQUAL_size_t_MESSAGE(before, after,
                                         sayEval("mlp", cell->name, "no marker is reserved"));
        TEST_ASSERT_EQUAL_MEMORY_MESSAGE(&g_evalNull.loss, &g_evalCell.loss, sizeof(float),
                                         sayEval("mlp", cell->name, "the loss"));
        TEST_ASSERT_EQUAL_size_t_MESSAGE(g_evalNull.output.used, g_evalCell.output.used,
                                         sayEval("mlp", cell->name, "the output's size"));
        TEST_ASSERT_EQUAL_MEMORY_MESSAGE(g_evalNull.output.bytes, g_evalCell.output.bytes,
                                         g_evalNull.output.used,
                                         sayEval("mlp", cell->name, "the output"));
    }
}

void testEvalOverTheZooMatchesTheNullPathOnArena(void) {
    assertEvalMatchesTheNullPath(REMAT_ARENA);
}

void testEvalOverTheZooMatchesTheNullPathOnHeap(void) {
    assertEvalMatchesTheNullPath(REMAT_HEAP);
}

/* One persistent scheduler across modes: train, evaluate, train again equals
 * Legacy train, the NULL path's eval, Legacy train (remat P1 on both training
 * calls; the eval call leaves nothing the next training call sees), and an
 * eval call on sample B of a scheduler built on sample A equals the NULL path
 * on B (remat P9). BatchNorm carries running stats from training into eval.
 * Each eval call leaves the parameter grads and BatchNorm running stats as the
 * NULL path's eval does; it is compared right there, because the training
 * captures reset both. */
static void assertTrainEvalTrainMatches(rematSchedulerType_t row) {
    const zooEntry_t fixtures[2] = {{"har-cnn", buildHarCnn}, {"batchnorm", buildBatchNorm}};
    for (size_t z = 0; z < 2u; z++) {
        for (size_t k = 0; k < MATRIX_CELLS; k++) {
            const matrixCell_t *cell = &g_cells[k];
            if (cell->row != row) {
                continue;
            }
            zooFixture_t f;
            fixtures[z].build(&f);
            captureRun(RUN_LEGACY, &f, NULL, &g_legacy);
            captureEval(&f, NULL, &g_evalNull);
            rematScheduler_t s;
            initCell(&s, cell, &f);
            const trainingCall_t call = {.remat = &s};
            captureRun(RUN_DRIVER, &f, &call, &g_driver);
            assertSameValues(cell->name);
            captureEval(&f, &call, &g_evalCell);
            assertEvalCaptureMatches(fixtures[z].name, cell->name, "eval between trainings");
            captureRun(RUN_DRIVER, &f, &call, &g_driver);
            assertSameValues(cell->name);
            tensor_t *sampleA = f.x;
            f.x = makeFloatTensor(sampleA->shape->dimensions, sampleA->shape->numberOfDimensions,
                                  2.5f);
            captureEval(&f, NULL, &g_evalNull);
            captureEval(&f, &call, &g_evalCell);
            assertEvalCaptureMatches(fixtures[z].name, cell->name, "remat P9: eval on sample B");
            freeTensor(f.x);
            f.x = sampleA;
            rematSchedulerDeinit(&s);
            freeZoo(&f);
        }
    }
}

void testTrainEvalTrainOnOneArenaMatchesLegacy(void) {
    assertTrainEvalTrainMatches(REMAT_ARENA);
}

void testTrainEvalTrainOnOneHeapMatchesLegacy(void) {
    assertTrainEvalTrainMatches(REMAT_HEAP);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testNullPathMatchesLegacyOnTheMlp);
    RUN_TEST(testTracedMlpConforms);
    RUN_TEST(testHarCnnUnderCeConforms);
    RUN_TEST(testSoftmaxUnderMseConforms);
    RUN_TEST(testFrozenLayerNormAboveDeepestConforms);
    RUN_TEST(testFrozenGroupNormAboveDeepestConforms);
    RUN_TEST(testTruncatedHarConforms);
    RUN_TEST(testAllFrozenModelConforms);
    RUN_TEST(testDropoutConforms);
    RUN_TEST(testQuantizationToSymInt32WiresConforms);
    RUN_TEST(testFlattenAt0Conforms);
    RUN_TEST(testPerTensorBfpF1ModelConforms);
    RUN_TEST(testGroupedBfpWireConforms);
    RUN_TEST(testSingleLayerUnderCeConforms);
    RUN_TEST(testCeWithOnlyTheLastLinearTrainableConforms);
    RUN_TEST(testBatchNormConforms);
    RUN_TEST(testCeWithoutATrailingSoftmaxConforms);
    RUN_TEST(testConvTransposedConforms);
    RUN_TEST(testFrozenConvTransposedAboveDeepestConforms);
    RUN_TEST(testAdaptiveAvgPoolConforms);
    RUN_TEST(testTwoCallsInARowMatchLegacyOnBatchNorm);
    RUN_TEST(testTwoCallsInARowMatchLegacyOnDropout);
    RUN_TEST(testAgradFiresBeforeTheLayersBackward);
    RUN_TEST(testAnInputWithoutBytesExitsBeforeLayer0Runs);
    RUN_TEST(testAnEmptyModelExitsNamingIt);
    RUN_TEST(testAMarkedInputYieldsAnUnmarkedOutputAndLeaksNothing);
#ifndef ODT_TEST_ASAN
    RUN_TEST(testAFailedEphemeralBuildExitsNamingIt);
#endif
    RUN_TEST(testCalculateGradsSequentialRunsOnTheCallersScheduler);
    RUN_TEST(testTracedGradsRunsOnTheCallersScheduler);
    RUN_TEST(testTheCallersSchedulerSurvivesTheCall);
    RUN_TEST(testAZeroInitialisedCallIsTheNullScheduler);
    RUN_TEST(testANullSchedulerResolvesToTheNamedDefaultPlan);
#ifdef ODT_MEM_PROFILE
    RUN_TEST(testANullSchedulerHoldsWhatTheDefaultPlansSchedulerHolds);
    RUN_TEST(testANullSchedulerHoldsLessThanStoreAllMidBackward);
#endif
    RUN_TEST(testMlpMatchesLegacyOnArena);
    RUN_TEST(testMlpMatchesLegacyOnHeap);
    RUN_TEST(testHarCnnMatchesLegacyOnArena);
    RUN_TEST(testHarCnnMatchesLegacyOnHeap);
    RUN_TEST(testTruncatedHarMatchesLegacyOnArena);
    RUN_TEST(testTruncatedHarMatchesLegacyOnHeap);
    RUN_TEST(testSoftmaxMseMatchesLegacyOnArena);
    RUN_TEST(testSoftmaxMseMatchesLegacyOnHeap);
    RUN_TEST(testFrozenLayerNormMatchesLegacyOnArena);
    RUN_TEST(testFrozenLayerNormMatchesLegacyOnHeap);
    RUN_TEST(testFrozenGroupNormMatchesLegacyOnArena);
    RUN_TEST(testFrozenGroupNormMatchesLegacyOnHeap);
    RUN_TEST(testAllFrozenMatchesLegacyOnArena);
    RUN_TEST(testAllFrozenMatchesLegacyOnHeap);
    RUN_TEST(testDropoutMatchesLegacyOnArena);
    RUN_TEST(testDropoutMatchesLegacyOnHeap);
    RUN_TEST(testQuantSymMatchesLegacyOnArena);
    RUN_TEST(testQuantSymMatchesLegacyOnHeap);
    RUN_TEST(testFlattenAt0MatchesLegacyOnArena);
    RUN_TEST(testFlattenAt0MatchesLegacyOnHeap);
    RUN_TEST(testF1ZooMatchesLegacyOnArena);
    RUN_TEST(testF1ZooMatchesLegacyOnHeap);
    RUN_TEST(testGroupedBfpMatchesLegacyOnArena);
    RUN_TEST(testGroupedBfpMatchesLegacyOnHeap);
    RUN_TEST(testSingleLayerCeMatchesLegacyOnArena);
    RUN_TEST(testSingleLayerCeMatchesLegacyOnHeap);
    RUN_TEST(testCeLastLinearOnlyMatchesLegacyOnArena);
    RUN_TEST(testCeLastLinearOnlyMatchesLegacyOnHeap);
    RUN_TEST(testCeWithoutSoftmaxMatchesLegacyOnArena);
    RUN_TEST(testCeWithoutSoftmaxMatchesLegacyOnHeap);
    RUN_TEST(testBatchNormMatchesLegacyOnArena);
    RUN_TEST(testBatchNormMatchesLegacyOnHeap);
    RUN_TEST(testConvTransposedMatchesLegacyOnArena);
    RUN_TEST(testConvTransposedMatchesLegacyOnHeap);
    RUN_TEST(testFrozenConvTransposedMatchesLegacyOnArena);
    RUN_TEST(testFrozenConvTransposedMatchesLegacyOnHeap);
    RUN_TEST(testAdaptiveAvgPoolMatchesLegacyOnArena);
    RUN_TEST(testAdaptiveAvgPoolMatchesLegacyOnHeap);
    RUN_TEST(testTwoCallsAcrossAWidthEditMatchLegacy);
    RUN_TEST(testTwoCallsAcrossARoundingEditMatchLegacy);
    RUN_TEST(testTwoCallsAcrossAnInputRegroupMatchLegacy);
    RUN_TEST(testASchedulerBuiltOnSampleARunsSampleB);
    RUN_TEST(testADrawingRowFailsP2AgainstLegacy);
    RUN_TEST(testAMarkedInputEvaluatesToAnUnmarkedOutputOnEitherRow);
    RUN_TEST(testEvalOverTheZooMatchesTheNullPathOnArena);
    RUN_TEST(testEvalOverTheZooMatchesTheNullPathOnHeap);
    RUN_TEST(testTrainEvalTrainOnOneArenaMatchesLegacy);
    RUN_TEST(testTrainEvalTrainOnOneHeapMatchesLegacy);
    return UNITY_END();
}
