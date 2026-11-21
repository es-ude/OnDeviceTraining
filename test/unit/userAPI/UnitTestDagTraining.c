#define SOURCE_FILE "UNIT_TEST_DAG_TRAINING"

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "AddApi.h"
#include "BatchNorm1d.h"
#include "BatchNorm1dApi.h"
#include "BatchView.h"
#include "CalculateGradsSequential.h"
#include "Conv1dApi.h"
#include "DagFixtures.h"
#include "DataLoaderApi.h"
#include "FlattenApi.h"
#include "InferenceApi.h"
#include "Layer.h"
#include "LayerQuant.h"
#include "LayerWeightsApi.h"
#include "LinearApi.h"
#include "LossFunction.h"
#include "Model.h"
#include "ModelGraphApi.h"
#include "Optimizer.h"
#include "OptimizerApi.h"
#include "QuantizationApi.h"
#include "RNG.h"
#include "ReluApi.h"
#include "RematScheduler.h"
#include "SoftmaxApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "TraceApi.h"
#include "TrainingBatchDefault.h"
#include "TrainingLoopApi.h"
#include "expected_dag.h"
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

/* ---- shared harness ------------------------------------------------------------------------ */

/* Natural-shape FLOAT32 tensor: batchViewOf adds axis 0. */
static tensor_t *buildFloatTensor(const size_t *dims, size_t rank, const float *src) {
    size_t *ownedDims = reserveMemory(rank * sizeof(size_t));
    for (size_t i = 0; i < rank; i++) {
        ownedDims[i] = dims[i];
    }
    size_t *order = reserveMemory(rank * sizeof(size_t));
    setOrderOfDimsForNewTensor(rank, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, ownedDims, rank, order);
    tensor_t *t = initTensor(shape, quantizationInitFloat(), NULL);
    if (src != NULL) {
        tensorFillFromFloatBuffer(t, src, calcNumberOfElementsByTensor(t));
    }
    return t;
}

#define GOLD_MAX_WEIGHTS 64

/* A FLOAT32 Linear holding the gold weights. The init key only has to be valid: the drawn
 * values are overwritten. The local copy keeps the gold arrays const. */
static layer_t *goldLinear(size_t in, size_t out, trainable_t tr, const float *w, const float *b,
                           size_t node) {
    TEST_ASSERT_TRUE(in * out <= GOLD_MAX_WEIGHTS);
    layerQuant_t lq;
    layerQuantInitUniform(&lq, dagFloatQ());
    const rngKey_t key = rngFold(rngKeyFromSeed(1u), RNG_TAG_INIT, (uint32_t)node);
    layer_t *layer = linearLayerInit(
        &(linearInit_t){
            .inFeatures = in, .outFeatures = out, .trainable = tr, .weightInit.key = &key},
        &lq);
    float wCopy[GOLD_MAX_WEIGHTS];
    float bCopy[GOLD_MAX_WEIGHTS];
    memcpy(wCopy, w, in * out * sizeof(float));
    memcpy(bCopy, b, out * sizeof(float));
    layerLoadWeights(layer, wCopy, bCopy);
    return layer;
}

static layer_t *goldSoftmax(void) {
    layerQuant_t lq;
    layerQuantInitUniform(&lq, dagFloatQ());
    return softmaxLayerInit(&lq);
}

#define MAX_PARAMS 16

static size_t trainableSlots(const model_t *m, parameter_t **slots) {
    size_t n = calcTotalNumberOfStates(m);
    TEST_ASSERT_TRUE_MESSAGE(n <= MAX_PARAMS, "raise MAX_PARAMS");
    collectTrainableParameters(m, slots);
    return n;
}

/* The factories hand out zeroed grads, but every call accumulates: clear before each one. */
static void zeroGrads(const model_t *m) {
    parameter_t *slots[MAX_PARAMS];
    size_t n = trainableSlots(m, slots);
    for (size_t p = 0; p < n; p++) {
        memset(slots[p]->grad->data, 0, calcBytesPerTensor(slots[p]->grad));
    }
}

/* Every trainable FLOAT32 grad times scale, in model order; returns the element count, which
 * callers assert after teardown. */
static size_t snapshotGrads(const model_t *m, float scale, float *out, size_t cap) {
    parameter_t *slots[MAX_PARAMS];
    size_t n = trainableSlots(m, slots);
    size_t k = 0;
    for (size_t p = 0; p < n; p++) {
        const float *g = (const float *)slots[p]->grad->data;
        size_t count = calcNumberOfElementsByTensor(slots[p]->grad);
        TEST_ASSERT_TRUE_MESSAGE(k + count <= cap, "raise the grad cap");
        for (size_t i = 0; i < count; i++) {
            out[k++] = g[i] * scale;
        }
    }
    return k;
}

/* allclose: |ref - got| <= atol + rtol * |ref|. Returns the first failing index, or SIZE_MAX. */
static size_t firstMismatch(const float *ref, const float *got, size_t n, float atol, float rtol) {
    for (size_t i = 0; i < n; i++) {
        if (!(fabsf(ref[i] - got[i]) <= atol + rtol * fabsf(ref[i]))) {
            return i;
        }
    }
    return SIZE_MAX;
}

/* Build in place and never copy: the model points into the struct's arrays. */
typedef struct dagGraph {
    layer_t *layers[8];
    uint16_t inputStart[9];
    uint16_t inputIds[10];
    graphBuilder_t g;
    model_t m;
} dagGraph_t;

static void dagGraphBegin(dagGraph_t *f) {
    graphBegin(&f->g, f->layers, f->inputStart, f->inputIds, 8, 10);
}

static void dagGraphFree(dagGraph_t *f) {
    for (size_t j = 0; j < f->m.numLayers; j++) {
        layer_t *layer = f->layers[j];
        switch (layer->type) {
        case LINEAR:
            freeLinearLayer(layer);
            break;
        case RELU:
            freeReluLayer(layer);
            break;
        case SOFTMAX:
            freeSoftmaxLayer(layer);
            break;
        case ADD:
            freeAddLayer(layer);
            break;
        case CONV1D:
            freeConv1dLayer(layer);
            break;
        case BATCHNORM1D:
            freeBatchNorm1dLayer(layer);
            break;
        case FLATTEN:
            freeFlattenLayer(layer);
            break;
        default:
            TEST_FAIL_MESSAGE("dagGraphFree: a layer type the harness does not build");
        }
    }
}

/* The remat plans a DAG admits. */
static const struct {
    bool arena;
    rematPlanPolicy_t policy;
    const char *name;
} kCells[4] = {
    {false, REMAT_PLAN_STORE_ALL, "HEAP/STORE_ALL"},
    {false, REMAT_PLAN_LIVENESS, "HEAP/LIVENESS"},
    {true, REMAT_PLAN_STORE_ALL, "ARENA/STORE_ALL"},
    {true, REMAT_PLAN_LIVENESS, "ARENA/LIVENESS"},
};

#define DAG_MAX_GRADS 128
#define DAG_MAX_OUT 8

/* One parity run: the NULL-call reference (loss, output, grads), then the same call on each
 * caller-owned cell scheduler, compared byte for byte against the reference, then loss-free
 * inference. Everything it builds is freed before it returns. */
typedef struct parityRun {
    size_t count;
    float loss;
    float ref[DAG_MAX_GRADS];
    float out[DAG_MAX_OUT];
    float outInf[DAG_MAX_OUT];
    int cellDiff[4];
} parityRun_t;

static void runParity(const model_t *m, lossConfig_t loss, tensor_t *x, tensor_t *y, size_t numOut,
                      parityRun_t *r) {
    TEST_ASSERT_TRUE_MESSAGE(numOut <= DAG_MAX_OUT, "raise DAG_MAX_OUT");
    zeroGrads(m);
    trainingStats_t *st = calculateGradsSequential(m, loss, REDUCTION_MEAN, x, y, NULL);
    r->loss = st->loss;
    memcpy(r->out, st->output->data, numOut * sizeof(float));
    freeTrainingStats(st);
    r->count = snapshotGrads(m, 1.0f, r->ref, DAG_MAX_GRADS);

    for (size_t c = 0; c < 4; c++) {
        float got[DAG_MAX_GRADS];
        zeroGrads(m);
        rematScheduler_t s;
        const rematPlanSpec_t spec = {.policy = kCells[c].policy};
        bool ok = kCells[c].arena ? rematArenaInit(&s, m, loss, x, &spec)
                                  : rematHeapInit(&s, m, loss, x, &spec);
        TEST_ASSERT_TRUE_MESSAGE(ok, kCells[c].name);
        freeTrainingStats(calculateGradsSequential(m, loss, REDUCTION_MEAN, x, y,
                                                   &(trainingCall_t){.remat = &s}));
        rematSchedulerDeinit(&s);
        snapshotGrads(m, 1.0f, got, DAG_MAX_GRADS);
        r->cellDiff[c] = memcmp(r->ref, got, r->count * sizeof(float));
    }

    tensor_t *inf = inference(m, x);
    memcpy(r->outInf, inf->data, numOut * sizeof(float));
    freeTensor(inf);
}

/* ---- the residual block -------------------------------------------------------------------- */

/* n0 = Linear 4->4 (b0); n1 = ReLU (b1); n2 = Linear 4->4 (b2); n3 = ADD (b1, b3). b1 fans out
 * to the ReLU and the ADD, so its gradient is the SUM of two contributions. */
static uint16_t buildResidualTrunk(dagGraph_t *f) {
    dagGraphBegin(f);
    uint16_t h1 =
        graphNode(&f->g, goldLinear(4, 4, TRAINABLE_DEFAULT, dagResW0, dagResB0, 0), GRAPH_INPUT);
    uint16_t h2 = graphNode(&f->g, dagRelu(), h1);
    uint16_t h3 = graphNode(&f->g, goldLinear(4, 4, TRAINABLE_DEFAULT, dagResW2, dagResB2, 2), h2);
    return graphMerge(&f->g, addLayerInit(NULL), h1, h3);
}

#define RES_GRADS 50 /* (16 + 4) + (16 + 4) + (8 + 2) */
#define RES_CE_GRADS 40

void testResidualBlockMatchesPyTorchOnEveryAdmittedPlan(void) {
    dagGraph_t f;
    uint16_t h4 = buildResidualTrunk(&f);
    (void)graphNode(&f.g, goldLinear(4, 2, TRAINABLE_DEFAULT, dagResW4, dagResB4, 4), h4);
    f.m = graphFinish(&f.g);
    tensor_t *x = buildFloatTensor((size_t[]){4}, 1, dagResX);
    tensor_t *y = buildFloatTensor((size_t[]){2}, 1, dagResY);
    batchView_t xv;
    batchView_t yv;

    parityRun_t r;
    runParity(&f.m, defaultLossConfig(MSE), batchViewOf(&xv, x), batchViewOf(&yv, y), 2, &r);

    freeTensor(y);
    freeTensor(x);
    dagGraphFree(&f);

    TEST_ASSERT_EQUAL_size_t(RES_GRADS, r.count);
    TEST_ASSERT_EQUAL_size_t_MESSAGE(SIZE_MAX,
                                     firstMismatch(dagResGrads, r.ref, RES_GRADS, 1e-6f, 1e-4f),
                                     "grads diverge from PyTorch");
    TEST_ASSERT_FLOAT_WITHIN(1e-5f, dagResLoss, r.loss);
    TEST_ASSERT_EQUAL_size_t(SIZE_MAX, firstMismatch(dagResOut, r.out, 2, 1e-6f, 1e-5f));
    TEST_ASSERT_EQUAL_MEMORY(r.out, r.outInf, 2 * sizeof(float));
    for (size_t c = 0; c < 4; c++) {
        TEST_ASSERT_EQUAL_INT_MESSAGE(0, r.cellDiff[c], kCells[c].name);
    }
}

/* The Softmax sink couples with CE: the ADD output's gradient is the seed p - y, and the
 * Softmax itself never runs backward. */
void testResidualIntoSoftmaxMatchesPyTorchUnderCrossEntropy(void) {
    dagGraph_t f;
    uint16_t h4 = buildResidualTrunk(&f);
    (void)graphNode(&f.g, goldSoftmax(), h4);
    f.m = graphFinish(&f.g);
    tensor_t *x = buildFloatTensor((size_t[]){4}, 1, dagResX);
    tensor_t *y = buildFloatTensor((size_t[]){4}, 1, dagResCeY);
    batchView_t xv;
    batchView_t yv;

    parityRun_t r;
    runParity(&f.m, defaultLossConfig(CROSS_ENTROPY), batchViewOf(&xv, x), batchViewOf(&yv, y), 4,
              &r);

    freeTensor(y);
    freeTensor(x);
    dagGraphFree(&f);

    TEST_ASSERT_EQUAL_size_t(RES_CE_GRADS, r.count);
    TEST_ASSERT_EQUAL_size_t_MESSAGE(
        SIZE_MAX, firstMismatch(dagResCeGrads, r.ref, RES_CE_GRADS, 1e-6f, 1e-4f),
        "grads diverge from PyTorch");
    TEST_ASSERT_FLOAT_WITHIN(1e-5f, dagResCeLoss, r.loss);
    TEST_ASSERT_EQUAL_size_t(SIZE_MAX, firstMismatch(dagResCeOut, r.out, 4, 1e-6f, 1e-5f));
    TEST_ASSERT_EQUAL_MEMORY(r.out, r.outInf, 4 * sizeof(float));
    for (size_t c = 0; c < 4; c++) {
        TEST_ASSERT_EQUAL_INT_MESSAGE(0, r.cellDiff[c], kCells[c].name);
    }
}

/* ---- fan-out mechanics: 4 -> 2 MSE fixtures -------------------------------------------------- */

/* Finishes the wiring, runs the parity on x[4] / y[2], copies the caller's input bytes out after
 * the run into xAfter, and frees everything. */
static void runMseParity(dagGraph_t *f, const float *xData, const float *yData, float *xAfter,
                         parityRun_t *r) {
    f->m = graphFinish(&f->g);
    tensor_t *x = buildFloatTensor((size_t[]){4}, 1, xData);
    tensor_t *y = buildFloatTensor((size_t[]){2}, 1, yData);
    batchView_t xv;
    batchView_t yv;
    runParity(&f->m, defaultLossConfig(MSE), batchViewOf(&xv, x), batchViewOf(&yv, y), 2, r);
    memcpy(xAfter, x->data, 4 * sizeof(float));
    freeTensor(y);
    freeTensor(x);
    dagGraphFree(f);
}

static void assertMseParity(const parityRun_t *r, size_t count, const float *grads, float loss,
                            const float *out) {
    TEST_ASSERT_EQUAL_size_t(count, r->count);
    TEST_ASSERT_EQUAL_size_t_MESSAGE(SIZE_MAX, firstMismatch(grads, r->ref, count, 1e-6f, 1e-4f),
                                     "grads diverge from PyTorch");
    TEST_ASSERT_FLOAT_WITHIN(1e-5f, loss, r->loss);
    TEST_ASSERT_EQUAL_size_t(SIZE_MAX, firstMismatch(out, r->out, 2, 1e-6f, 1e-5f));
    TEST_ASSERT_EQUAL_MEMORY(r->out, r->outInf, 2 * sizeof(float));
    for (size_t c = 0; c < 4; c++) {
        TEST_ASSERT_EQUAL_INT_MESSAGE(0, r->cellDiff[c], kCells[c].name);
    }
}

/* Both ADD slots read b1: the first writes g(b1), the second accumulates onto it. */
void testAddOfOneBufferWithItselfDoublesItsGradient(void) {
    dagGraph_t f;
    dagGraphBegin(&f);
    uint16_t h1 =
        graphNode(&f.g, goldLinear(4, 4, TRAINABLE_DEFAULT, dagSelfW0, dagSelfB0, 0), GRAPH_INPUT);
    uint16_t h2 = graphMerge(&f.g, addLayerInit(NULL), h1, h1);
    (void)graphNode(&f.g, goldLinear(4, 2, TRAINABLE_DEFAULT, dagSelfW2, dagSelfB2, 2), h2);

    parityRun_t r;
    float xAfter[4];
    runMseParity(&f, dagSelfX, dagSelfY, xAfter, &r);

    assertMseParity(&r, 30, dagSelfGrads, dagSelfLoss, dagSelfOut);
}

/* The model input needs no gradient, so the ADD's second slot has no dx; the input bytes the
 * caller owns stay untouched. */
void testSkipFromTheModelInputGetsNoGradient(void) {
    dagGraph_t f;
    dagGraphBegin(&f);
    uint16_t h1 =
        graphNode(&f.g, goldLinear(4, 4, TRAINABLE_DEFAULT, dagIn0W0, dagIn0B0, 0), GRAPH_INPUT);
    uint16_t h2 = graphNode(&f.g, dagRelu(), h1);
    uint16_t h3 = graphMerge(&f.g, addLayerInit(NULL), h2, GRAPH_INPUT);
    (void)graphNode(&f.g, goldLinear(4, 2, TRAINABLE_DEFAULT, dagIn0W3, dagIn0B3, 3), h3);

    parityRun_t r;
    float xAfter[4];
    runMseParity(&f, dagIn0X, dagIn0Y, xAfter, &r);

    TEST_ASSERT_EQUAL_MEMORY(dagIn0X, xAfter, 4 * sizeof(float));
    assertMseParity(&r, 30, dagIn0Grads, dagIn0Loss, dagIn0Out);
}

/* b1 has three consumers: the second ADD writes g(b1), the first ADD accumulates onto it, and
 * L1's dx arrives as a partial that is summed in. */
void testFanOutThreeWritesAccumulatesAndSums(void) {
    dagGraph_t f;
    dagGraphBegin(&f);
    uint16_t h1 =
        graphNode(&f.g, goldLinear(4, 4, TRAINABLE_DEFAULT, dagFan3W0, dagFan3B0, 0), GRAPH_INPUT);
    uint16_t h2 = graphNode(&f.g, goldLinear(4, 4, TRAINABLE_DEFAULT, dagFan3W1, dagFan3B1, 1), h1);
    uint16_t h3 = graphMerge(&f.g, addLayerInit(NULL), h1, h2);
    uint16_t h4 = graphMerge(&f.g, addLayerInit(NULL), h3, h1);
    (void)graphNode(&f.g, goldLinear(4, 2, TRAINABLE_DEFAULT, dagFan3W4, dagFan3B4, 4), h4);

    parityRun_t r;
    float xAfter[4];
    runMseParity(&f, dagFan3X, dagFan3Y, xAfter, &r);

    assertMseParity(&r, 50, dagFan3Grads, dagFan3Loss, dagFan3Out);
}

/* ---- freezing: which nodes run backward --------------------------------------------------- */

typedef struct agradLog {
    size_t idx[8];
    size_t n;
} agradLog_t;

/* Counts every "agrad" event but stores at most 8, so an overflow shows up as a wrong count. */
static void recordAgrad(void *ctx, size_t layerIdx, layerType_t t, const char *phase, tensor_t *x) {
    (void)t;
    (void)x;
    agradLog_t *log = ctx;
    if (strcmp(phase, "agrad") == 0) {
        if (log->n < 8) {
            log->idx[log->n] = layerIdx;
        }
        log->n++;
    }
}

/* runMseParity with a traced NULL-call run first: its agrad order and grads land in log/got. */
static void runTracedMseParity(dagGraph_t *f, const float *xData, const float *yData,
                               agradLog_t *log, float *got, size_t *gotCount, parityRun_t *r) {
    f->m = graphFinish(&f->g);
    tensor_t *x = buildFloatTensor((size_t[]){4}, 1, xData);
    tensor_t *y = buildFloatTensor((size_t[]){2}, 1, yData);
    batchView_t xv;
    batchView_t yv;
    tensor_t *xView = batchViewOf(&xv, x);
    tensor_t *yView = batchViewOf(&yv, y);
    zeroGrads(&f->m);
    freeTrainingStats(tracedGrads(&f->m, defaultLossConfig(MSE), REDUCTION_MEAN, xView, yView,
                                  recordAgrad, log, NULL));
    *gotCount = snapshotGrads(&f->m, 1.0f, got, DAG_MAX_GRADS);
    runParity(&f->m, defaultLossConfig(MSE), xView, yView, 2, r);
    freeTensor(y);
    freeTensor(x);
    dagGraphFree(f);
}

/* The skip carries b1 out of the frozen prefix: the ADD gives b1 no gradient, and node 2, whose
 * input needs none either, runs grads-only. The prefix never runs backward. */
void testFrozenPrefixWithSkipAcrossTheBoundary(void) {
    dagGraph_t f;
    dagGraphBegin(&f);
    uint16_t h1 =
        graphNode(&f.g, goldLinear(4, 4, TRAINABLE_FALSE, dagFrzW0, dagFrzB0, 0), GRAPH_INPUT);
    uint16_t h2 = graphNode(&f.g, dagRelu(), h1);
    uint16_t h3 = graphNode(&f.g, goldLinear(4, 4, TRAINABLE_DEFAULT, dagFrzW2, dagFrzB2, 2), h2);
    uint16_t h4 = graphMerge(&f.g, addLayerInit(NULL), h1, h3);
    (void)graphNode(&f.g, goldLinear(4, 2, TRAINABLE_DEFAULT, dagFrzW4, dagFrzB4, 4), h4);

    agradLog_t log = {0};
    float got[DAG_MAX_GRADS];
    size_t count;
    parityRun_t r;
    runTracedMseParity(&f, dagFrzX, dagFrzY, &log, got, &count, &r);

    TEST_ASSERT_EQUAL_size_t(3, log.n);
    TEST_ASSERT_EQUAL_size_t_ARRAY(((size_t[]){4, 3, 2}), log.idx, 3);
    TEST_ASSERT_EQUAL_size_t(30, count);
    TEST_ASSERT_EQUAL_size_t_MESSAGE(SIZE_MAX, firstMismatch(dagFrzGrads, got, 30, 1e-6f, 1e-4f),
                                     "grads diverge from PyTorch");
    assertMseParity(&r, 30, dagFrzGrads, dagFrzLoss, dagFrzOut);
}

/* Node 2 is frozen and reads b1, which needs no gradient, so the backward skips it between
 * nodes 3 and 1: the ADD gives b3 no gradient, and node 1 runs grads-only. */
void testFrozenNodeOnTheSkipPathLeavesAGapInTheBackward(void) {
    dagGraph_t f;
    dagGraphBegin(&f);
    uint16_t h1 =
        graphNode(&f.g, goldLinear(4, 4, TRAINABLE_FALSE, dagGapW0, dagGapB0, 0), GRAPH_INPUT);
    uint16_t h2 = graphNode(&f.g, goldLinear(4, 4, TRAINABLE_DEFAULT, dagGapW1, dagGapB1, 1), h1);
    uint16_t h3 = graphNode(&f.g, goldLinear(4, 4, TRAINABLE_FALSE, dagGapW2, dagGapB2, 2), h1);
    uint16_t h4 = graphMerge(&f.g, addLayerInit(NULL), h2, h3);
    (void)graphNode(&f.g, goldLinear(4, 2, TRAINABLE_DEFAULT, dagGapW4, dagGapB4, 4), h4);

    agradLog_t log = {0};
    float got[DAG_MAX_GRADS];
    size_t count;
    parityRun_t r;
    runTracedMseParity(&f, dagGapX, dagGapY, &log, got, &count, &r);

    TEST_ASSERT_EQUAL_size_t(3, log.n);
    TEST_ASSERT_EQUAL_size_t_ARRAY(((size_t[]){4, 3, 1}), log.idx, 3);
    TEST_ASSERT_EQUAL_size_t(30, count);
    TEST_ASSERT_EQUAL_size_t_MESSAGE(SIZE_MAX, firstMismatch(dagGapGrads, got, 30, 1e-6f, 1e-4f),
                                     "grads diverge from PyTorch");
    assertMseParity(&r, 30, dagGapGrads, dagGapLoss, dagGapOut);
}

/* ---- BatchNorm inside the residual block, per micro-batch --------------------------------- */

/* A FLOAT32 Conv1d k3 SAME holding the gold weights ([out, in, k], as in PyTorch). */
static layer_t *goldConv1d(size_t in, size_t out, const float *w, const float *b, size_t node) {
    TEST_ASSERT_TRUE(in * out * 3 <= GOLD_MAX_WEIGHTS);
    layerQuant_t lq;
    layerQuantInitUniform(&lq, dagFloatQ());
    const rngKey_t key = rngFold(rngKeyFromSeed(1u), RNG_TAG_INIT, (uint32_t)node);
    layer_t *layer = conv1dLayerInit(&(conv1dInit_t){.inChannels = in,
                                                     .outChannels = out,
                                                     .kernelSize = 3,
                                                     .padding = SAME,
                                                     .weightInit.key = &key},
                                     &lq);
    float wCopy[GOLD_MAX_WEIGHTS];
    float bCopy[GOLD_MAX_WEIGHTS];
    memcpy(wCopy, w, in * out * 3 * sizeof(float));
    memcpy(bCopy, b, out * sizeof(float));
    layerLoadWeights(layer, wCopy, bCopy);
    return layer;
}

static layer_t *goldBatchNorm1d(const float *gamma, const float *beta) {
    layerQuant_t lq;
    layerQuantInitUniform(&lq, dagFloatQ());
    layer_t *layer = batchNorm1dLayerInit(&(batchNorm1dInit_t){.numChannels = 3}, &lq);
    float g[3];
    float b[3];
    memcpy(g, gamma, sizeof g);
    memcpy(b, beta, sizeof b);
    layerLoadWeights(layer, g, b);
    return layer;
}

/* trainingBatchDefault frees every sample_t it consumes, so each run needs fresh samples over
 * the same tensors; freeBatch releases the rest afterwards. */
static batch_t *buildBatch(tensor_t **items, tensor_t **labels, size_t n) {
    batch_t *batch = reserveMemory(sizeof(batch_t));
    batch->samples = reserveMemory(n * sizeof(sample_t *));
    batch->size = n;
    for (size_t i = 0; i < n; i++) {
        sample_t *s = reserveMemory(sizeof(sample_t));
        s->item = items[i];
        s->label = labels[i];
        batch->samples[i] = s;
    }
    return batch;
}

#define BN_B 4
#define BN_GRADS 95 /* (18 + 3) + (3 + 3) + (27 + 3) + (36 + 2) */

typedef struct bnRun {
    size_t count;
    float grads[DAG_MAX_GRADS];
    float mean[3];
    float var[3];
    uint64_t nbt;
    float loss;
} bnRun_t;

/* n0 = Conv1d 2->3 (b0); n1 = BN (b1); n2 = ReLU (b2); n3 = Conv1d 3->3 (b3); n4 = ADD(b1, b4);
 * n5 = Flatten (b5); n6 = Linear 18->2 (b6). One macro batch of 4 in chunks of m rows; the
 * grads are snapshot times the MSE mean scale, as the optimizer would see them. */
static void runBnResidual(size_t m, bnRun_t *r) {
    dagGraph_t f;
    dagGraphBegin(&f);
    layer_t *bn = goldBatchNorm1d(dagBnGamma, dagBnBeta);
    uint16_t h1 = graphNode(&f.g, goldConv1d(2, 3, dagBnW0, dagBnB0, 0), GRAPH_INPUT);
    uint16_t h2 = graphNode(&f.g, bn, h1);
    uint16_t h3 = graphNode(&f.g, dagRelu(), h2);
    uint16_t h4 = graphNode(&f.g, goldConv1d(3, 3, dagBnW3, dagBnB3, 3), h3);
    uint16_t h5 = graphMerge(&f.g, addLayerInit(NULL), h1, h4);
    uint16_t h6 = graphNode(&f.g, flattenLayerInit(), h5);
    (void)graphNode(&f.g, goldLinear(18, 2, TRAINABLE_DEFAULT, dagBnW6, dagBnB6, 6), h6);
    f.m = graphFinish(&f.g);

    tensor_t *items[BN_B];
    tensor_t *labels[BN_B];
    for (size_t s = 0; s < BN_B; s++) {
        items[s] = buildFloatTensor((size_t[]){2, 6}, 2, dagBnX + 12 * s);
        labels[s] = buildFloatTensor((size_t[]){2}, 1, dagBnY + 2 * s);
    }
    batchView_t view;
    float scale = lossFunctions[MSE].computeMeanScale(BN_B, batchViewOf(&view, labels[0]));
    batch_t *batch = buildBatch(items, labels, BN_B);
    r->loss = trainingBatchDefault(&f.m, defaultLossConfig(MSE), batch, calculateGradsSequential,
                                   REDUCTION_MEAN, m, NULL);
    freeBatch(batch);
    r->count = snapshotGrads(&f.m, scale, r->grads, DAG_MAX_GRADS);
    const batchNorm1dConfig_t *c = bn->config->batchNorm1d;
    memcpy(r->mean, c->runningMean->data, sizeof r->mean);
    memcpy(r->var, c->runningVar->data, sizeof r->var);
    r->nbt = c->numBatchesTracked;

    for (size_t s = 0; s < BN_B; s++) {
        freeTensor(labels[s]);
        freeTensor(items[s]);
    }
    dagGraphFree(&f);
}

/* Ghost BN over each chunk of m rows: the running stats update b/m times, and b1's gradient
 * sums the ADD's skip contribution with the BN branch's partial. */
void testBatchNormInsideTheBlockMatchesPyTorchPerMicroBatch(void) {
    static const size_t ms[3] = {1, 2, 4};
    static const char *const label[3] = {"m=1", "m=2", "m=4"};
    static const char *const diverge[3] = {"m=1: grads diverge from PyTorch",
                                           "m=2: grads diverge from PyTorch",
                                           "m=4: grads diverge from PyTorch"};
    const float *goldGrads[3] = {dagBnGrads_m1, dagBnGrads_m2, dagBnGrads_m4};
    const float *goldMean[3] = {dagBnRunningMean_m1, dagBnRunningMean_m2, dagBnRunningMean_m4};
    const float *goldVar[3] = {dagBnRunningVar_m1, dagBnRunningVar_m2, dagBnRunningVar_m4};
    const uint64_t goldNbt[3] = {dagBnNbt_m1, dagBnNbt_m2, dagBnNbt_m4};
    const float goldLoss[3] = {dagBnLoss_m1, dagBnLoss_m2, dagBnLoss_m4};
    bnRun_t r[3];
    for (size_t k = 0; k < 3; k++) {
        runBnResidual(ms[k], &r[k]);
    }

    for (size_t k = 0; k < 3; k++) {
        TEST_ASSERT_EQUAL_size_t_MESSAGE(BN_GRADS, r[k].count, label[k]);
        TEST_ASSERT_EQUAL_size_t_MESSAGE(
            SIZE_MAX, firstMismatch(goldGrads[k], r[k].grads, BN_GRADS, 1e-6f, 1e-4f), diverge[k]);
        TEST_ASSERT_EQUAL_size_t_MESSAGE(
            SIZE_MAX, firstMismatch(goldMean[k], r[k].mean, 3, 1e-6f, 1e-5f), label[k]);
        TEST_ASSERT_EQUAL_size_t_MESSAGE(
            SIZE_MAX, firstMismatch(goldVar[k], r[k].var, 3, 1e-6f, 1e-5f), label[k]);
        TEST_ASSERT_EQUAL_UINT64_MESSAGE(goldNbt[k], r[k].nbt, label[k]);
        TEST_ASSERT_FLOAT_WITHIN_MESSAGE(1e-5f, goldLoss[k], r[k].loss, label[k]);
    }
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testResidualBlockMatchesPyTorchOnEveryAdmittedPlan);
    RUN_TEST(testResidualIntoSoftmaxMatchesPyTorchUnderCrossEntropy);
    RUN_TEST(testAddOfOneBufferWithItselfDoublesItsGradient);
    RUN_TEST(testSkipFromTheModelInputGetsNoGradient);
    RUN_TEST(testFanOutThreeWritesAccumulatesAndSums);
    RUN_TEST(testFrozenPrefixWithSkipAcrossTheBoundary);
    RUN_TEST(testFrozenNodeOnTheSkipPathLeavesAGapInTheBackward);
    RUN_TEST(testBatchNormInsideTheBlockMatchesPyTorchPerMicroBatch);
    return UNITY_END();
}
