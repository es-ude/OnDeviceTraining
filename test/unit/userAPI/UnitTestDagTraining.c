#define SOURCE_FILE "UNIT_TEST_DAG_TRAINING"

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "AddApi.h"
#include "BatchView.h"
#include "CalculateGradsSequential.h"
#include "DagFixtures.h"
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

/* The factories hand out zeroed grads, but every call accumulates: clear before each one. */
static void zeroGrads(const model_t *m) {
    parameter_t *slots[MAX_PARAMS];
    size_t n = calcTotalNumberOfStates(m);
    collectTrainableParameters(m, slots);
    for (size_t p = 0; p < n; p++) {
        memset(slots[p]->grad->data, 0, calcBytesPerTensor(slots[p]->grad));
    }
}

/* Every trainable FLOAT32 grad times scale, in model order; returns the element count, which
 * callers assert after teardown. */
static size_t snapshotGrads(const model_t *m, float scale, float *out, size_t cap) {
    parameter_t *slots[MAX_PARAMS];
    size_t n = calcTotalNumberOfStates(m);
    collectTrainableParameters(m, slots);
    size_t k = 0;
    for (size_t p = 0; p < n; p++) {
        const float *g = (const float *)slots[p]->grad->data;
        size_t count = calcNumberOfElementsByTensor(slots[p]->grad);
        for (size_t i = 0; i < count; i++) {
            if (k < cap) {
                out[k] = g[i] * scale;
            }
            k++;
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

#define DAG_MAX_GRADS 64
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

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testResidualBlockMatchesPyTorchOnEveryAdmittedPlan);
    RUN_TEST(testResidualIntoSoftmaxMatchesPyTorchUnderCrossEntropy);
    return UNITY_END();
}
