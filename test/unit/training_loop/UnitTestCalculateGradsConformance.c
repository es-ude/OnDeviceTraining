#define SOURCE_FILE "UNIT_TEST_CALCULATE_GRADS_CONFORMANCE"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "AsanDeath.h"
#include "BatchNorm1dApi.h"
#include "CalculateGradsSequential.h"
#include "Common.h"
#include "DeathTest.h"
#include "DropoutApi.h"
#include "GroupNormApi.h"
#include "Layer.h"
#include "LayerQuant.h"
#include "LegacyCalculateGrads.h"
#include "LossFunction.h"
#include "OdtHook.h"
#include "Quantization.h"
#include "QuantizationApi.h"
#include "RNG.h"
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
 * (remat D20) still skips layer n - 1's backward, exactly as the old driver did. */
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
static void captureRun(runner_t runner, zooFixture_t *f, runCapture_t *cap) {
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
                                               recordingSink, &cap->trace);
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

static void assertSameValues(void) {
    TEST_ASSERT_TRUE(g_legacy.values.used > 0);
    TEST_ASSERT_EQUAL_MEMORY_MESSAGE(&g_legacy.loss, &g_driver.loss, sizeof g_legacy.loss,
                                     "P1: loss");
    TEST_ASSERT_EQUAL_size_t_MESSAGE(g_legacy.values.used, g_driver.values.used,
                                     "P1: captured byte count");
    TEST_ASSERT_EQUAL_MEMORY_MESSAGE(g_legacy.values.bytes, g_driver.values.bytes,
                                     g_legacy.values.used,
                                     "P1: output snapshot, parameter grads, BN running stats");
}

static void assertConforms(void) {
    assertSameValues();
    TEST_ASSERT_EQUAL_HEX32_MESSAGE(g_legacy.seedAfter, g_driver.seedAfter,
                                    "P2: the global stream after the call");
    TEST_ASSERT_EQUAL_size_t_MESSAGE(4, g_driver.numEvents, "P3: four hook events");
    TEST_ASSERT_EQUAL_size_t_MESSAGE(g_legacy.numEvents, g_driver.numEvents, "P3: event count");
    TEST_ASSERT_EQUAL_MEMORY_MESSAGE(g_legacy.events, g_driver.events,
                                     g_driver.numEvents * sizeof(odtEvent_t), "P3: event order");
    TEST_ASSERT_EQUAL_size_t_MESSAGE(g_legacy.trace.used, g_driver.trace.used,
                                     "P4: trace byte count");
    TEST_ASSERT_EQUAL_MEMORY_MESSAGE(g_legacy.trace.bytes, g_driver.trace.bytes,
                                     g_legacy.trace.used, "P4: trace (idx, phase, tensor)");
    TEST_ASSERT_EQUAL_size_t_MESSAGE(g_legacy.memBefore, g_legacy.memAfter,
                                     "remat P5: Legacy releases everything but the stats");
    TEST_ASSERT_EQUAL_size_t_MESSAGE(g_driver.memBefore, g_driver.memAfter,
                                     "P5: the driver releases everything but the stats");
}

static void assertNullPathMatchesLegacy(void (*build)(zooFixture_t *)) {
    zooFixture_t f;
    build(&f);
    captureRun(RUN_LEGACY, &f, &g_legacy);
    captureRun(RUN_DRIVER, &f, &g_driver);
    freeZoo(&f);
    assertConforms();
}

static trainingStats_t *runOnce(runner_t runner, zooFixture_t *f) {
    return (runner == RUN_LEGACY)
               ? legacyCalculateGrads(f->model, f->n, f->loss, REDUCTION_MEAN, f->x, f->y, NULL,
                                      NULL)
               : calculateGradsSequential(f->model, f->n, f->loss, REDUCTION_MEAN, f->x, f->y);
}

/* A caller that does not zero between calls: the grads accumulate over both,
 * BN's running stats move twice and Dropout's stream advances twice. */
static void captureTwoCalls(runner_t runner, zooFixture_t *f, runCapture_t *cap) {
    forEachGrad(f->model, f->n, NULL);
    bnStateIo(f, runner == RUN_LEGACY, NULL);
    cap->values.used = 0;
    rngSetSeed(ZOO_SEED);
    freeTrainingStats(runOnce(runner, f));
    trainingStats_t *stats = runOnce(runner, f);
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
    assertSameValues();
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
        calculateGradsSequential(f.model, f.n, f.loss, REDUCTION_MEAN, f.x, f.y);
    g_driver.values.used = 0;
    g_driver.loss = driver->loss;
    captureTensor(&g_driver.values, driver->output);
    freeTrainingStats(driver);
    forEachGrad(f.model, f.n, &g_driver.values);
    freeZoo(&f);
    assertSameValues();
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
            : tracedGrads(f.model, f.n, f.loss, REDUCTION_MEAN, f.x, f.y, agradOrderSink, &p);
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
            calculateGradsSequential(f.model, f.n, f.loss, REDUCTION_MEAN, f.x, f.y)));
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
                                 f.model, 0, f.loss, REDUCTION_MEAN, f.x, inputShapedLabel)));
    freeTensor(inputShapedLabel);
    freeZoo(&f);
}

/* Wire headers carry no sparsity marker (an input's marker is not
 * propagated), so a marked input yields an unmarked output snapshot and
 * nothing is reserved for markers. The values still match Legacy. The memory baseline is taken
 * after the Legacy run, which leaks its markers (freeSparsity is a no-op). */
void testAMarkedInputYieldsAnUnmarkedOutputAndLeaksNothing(void) {
    zooFixture_t f;
    buildMlp(&f);
    sparsity_t marker = {0};
    f.x->sparsity = &marker;
    captureRun(RUN_LEGACY, &f, &g_legacy);

    forEachGrad(f.model, f.n, NULL);
    g_driver.values.used = 0;
    size_t before = memProfileCurrentBytes();
    trainingStats_t *stats =
        calculateGradsSequential(f.model, f.n, f.loss, REDUCTION_MEAN, f.x, f.y);
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
    assertSameValues();
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
                                 model, 1, defaultLossConfig(MSE), REDUCTION_MEAN, x, y)));
    freeTensor(y);
    freeModel(model, 1);
    freeQuantization(bfp);
}
#endif

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
    RUN_TEST(testTwoCallsInARowMatchLegacyOnBatchNorm);
    RUN_TEST(testTwoCallsInARowMatchLegacyOnDropout);
    RUN_TEST(testAgradFiresBeforeTheLayersBackward);
    RUN_TEST(testAnInputWithoutBytesExitsBeforeLayer0Runs);
    RUN_TEST(testAnEmptyModelExitsNamingIt);
    RUN_TEST(testAMarkedInputYieldsAnUnmarkedOutputAndLeaksNothing);
#ifndef ODT_TEST_ASAN
    RUN_TEST(testAFailedEphemeralBuildExitsNamingIt);
#endif
    return UNITY_END();
}
