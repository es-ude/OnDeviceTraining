#ifndef ODT_TEST_REMAT_TEST_FIXTURES_H
#define ODT_TEST_REMAT_TEST_FIXTURES_H

/* Fixture builders shared by the remat test binaries (UnitTestRematPlan,
 * UnitTestRematScheduler, UnitTestRematCheck) and the driver's conformance
 * harness, so all four build the same models. Header-only statics, like
 * BorrowedLayer.h: each binary compiles the helpers it calls. A consumer
 * links the layer API targets of every type freeModel frees (LinearApi,
 * ReluApi, SoftmaxApi, Conv1dApi, Pool1dApi, FlattenApi, QuantLayerApi,
 * LayerNormApi) and RematScheduler for initArena/initHeap. */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "Conv1dApi.h"
#include "FlattenApi.h"
#include "Layer.h"
#include "LayerNormApi.h"
#include "LayerQuant.h"
#include "LinearApi.h"
#include "LossFunction.h"
#include "Pool1dApi.h"
#include "QuantLayerApi.h"
#include "Quantization.h"
#include "ReluApi.h"
#include "RematScheduler.h"
#include "SoftmaxApi.h"
#include "Tensor.h"
#include "unity.h"

/* Fixture layers borrow their wire templates, so one FLOAT32 template outlives
 * every fixture model. Tests that edit a template use a local one. */
static quantization_t g_floatQ = {.type = FLOAT32, .qConfig = NULL};

static layer_t *makeLinear(size_t in, size_t out, bool frozen) {
    layerQuant_t lq;
    layerQuantInitUniform(&lq, &g_floatQ);
    return linearLayerInit(
        &(linearInit_t){.inFeatures = in,
                        .outFeatures = out,
                        .trainable = frozen ? TRAINABLE_FALSE : TRAINABLE_DEFAULT},
        &lq);
}

static layer_t *makeRelu(quantization_t *q) {
    return reluLayerInit(&(layerQuant_t){.outputQ = q, .propLossQ = q});
}

static layer_t *makeSoftmax(void) {
    layerQuant_t lq;
    layerQuantInitUniform(&lq, &g_floatQ);
    return softmaxLayerInit(&lq);
}

static layer_t *makeQuant(quantization_t *outputQ, quantization_t *propLossQ) {
    return quantLayerInit(&(layerQuant_t){.outputQ = outputQ, .propLossQ = propLossQ});
}

static layer_t *makeLayerNorm(size_t features, bool frozen) {
    layerQuant_t lq;
    layerQuantInitUniform(&lq, &g_floatQ);
    return layerNormLayerInit(
        &(layerNormInit_t){.normalizedShape = (size_t[]){features},
                           .numNormDims = 1,
                           .trainable = frozen ? TRAINABLE_FALSE : TRAINABLE_DEFAULT},
        &lq);
}

static void freeModel(layer_t **model, size_t n) {
    for (size_t i = 0; i < n; i++) {
        switch (model[i]->type) {
        case LINEAR:
            freeLinearLayer(model[i]);
            break;
        case RELU:
            freeReluLayer(model[i]);
            break;
        case SOFTMAX:
            freeSoftmaxLayer(model[i]);
            break;
        case CONV1D:
            freeConv1dLayer(model[i]);
            break;
        case MAXPOOL1D:
            freeMaxPool1dLayer(model[i]);
            break;
        case AVGPOOL1D:
            freeAvgPool1dLayer(model[i]);
            break;
        case FLATTEN:
            freeFlattenLayer(model[i]);
            break;
        case QUANTIZATION:
            freeQuantLayer(model[i]);
            break;
        case LAYERNORM:
            freeLayerNormLayer(model[i]);
            break;
        default:
            TEST_FAIL_MESSAGE("freeModel: extend the switch for this layer type");
        }
    }
}

/* A borrowed input header on the caller's stack, with room for the bytes of
 * up to the HAR sample. makeInput leaves data NULL (table init and bind never
 * read it); makeResidentInput points it at the FLOAT32 buffer, because the
 * checker requires every operand it reads to be resident, ACT 0 included. */
#define TEST_MAX_RANK 4
#define TEST_MAX_INPUT_FLOATS 1152u
typedef struct inputLike {
    size_t dims[TEST_MAX_RANK];
    size_t order[TEST_MAX_RANK];
    shape_t shape;
    tensor_t tensor;
    float data[TEST_MAX_INPUT_FLOATS];
} inputLike_t;

static tensor_t *makeInput(inputLike_t *in, const size_t *dims, size_t rank, quantization_t *q) {
    TEST_ASSERT_TRUE(rank <= TEST_MAX_RANK);
    for (size_t d = 0; d < rank; d++) {
        in->dims[d] = dims[d];
        in->order[d] = d;
    }
    in->shape = (shape_t){
        .numberOfDimensions = rank, .dimensions = in->dims, .orderOfDimensions = in->order};
    in->tensor = (tensor_t){.data = NULL, .shape = &in->shape, .quantization = q, .sparsity = NULL};
    return &in->tensor;
}

static tensor_t *makeResidentInput(inputLike_t *in, const size_t *dims, size_t rank) {
    tensor_t *x = makeInput(in, dims, rank, &g_floatQ);
    TEST_ASSERT_TRUE(calcNumberOfElementsByShape(&in->shape) <= TEST_MAX_INPUT_FLOATS);
    x->data = (uint8_t *)in->data;
    return x;
}

/* examples/har_classifier/train_c.c:178-216 (B = 1); freezeConvs gives the
 * stage-2 backbone of train_c_finetune.c:171-216. n = 12, CE, deepest 0, top
 * 10; 25 steps; ACT 0..12 plus GRAD {12, 10, 9, ..., 1} = 24 wires. */
#define HAR_N 12
static void buildHar(layer_t **model, bool freezeConvs) {
    layerQuant_t lq;
    layerQuantInitUniform(&lq, &g_floatQ);
    trainable_t conv = freezeConvs ? TRAINABLE_FALSE : TRAINABLE_DEFAULT;
    model[0] = conv1dLayerInit(&(conv1dInit_t){.inChannels = 9,
                                               .outChannels = 16,
                                               .kernelSize = 7,
                                               .padding = SAME,
                                               .trainable = conv},
                               &lq);
    model[1] = reluLayerInit(&lq);
    model[2] = maxPool1dLayerInit(
        &(maxPool1dInit_t){.kernelSize = 2, .stride = 2, .inputChannels = 16, .inputLength = 128},
        &lq);
    model[3] = conv1dLayerInit(&(conv1dInit_t){.inChannels = 16,
                                               .outChannels = 32,
                                               .kernelSize = 5,
                                               .padding = SAME,
                                               .trainable = conv},
                               &lq);
    model[4] = reluLayerInit(&lq);
    model[5] = maxPool1dLayerInit(
        &(maxPool1dInit_t){.kernelSize = 2, .stride = 2, .inputChannels = 32, .inputLength = 64},
        &lq);
    model[6] = conv1dLayerInit(&(conv1dInit_t){.inChannels = 32,
                                               .outChannels = 64,
                                               .kernelSize = 3,
                                               .padding = SAME,
                                               .trainable = conv},
                               &lq);
    model[7] = reluLayerInit(&lq);
    model[8] = avgPool1dLayerInit(&(avgPool1dInit_t){.kernelSize = 32, .stride = 32}, &lq);
    model[9] = flattenLayerInit();
    model[10] = linearLayerInit(&(linearInit_t){.inFeatures = 64, .outFeatures = 6}, &lq);
    model[11] = softmaxLayerInit(&lq);
}

static tensor_t *makeHarInput(inputLike_t *in) {
    return makeInput(in, (size_t[]){1, 9, 128}, 3, &g_floatQ);
}

static uint32_t nextRandom(uint32_t *state) { /* xorshift32, test-local */
    uint32_t v = *state;
    v ^= v << 13;
    v ^= v >> 17;
    v ^= v << 5;
    *state = v;
    return v;
}

static const rematPlanSpec_t g_liveness = {.policy = REMAT_PLAN_LIVENESS};

/* One fixture model and its resident input. The F1 model's BFP template lives
 * here too, because its layers borrow it: the fixture must not move while the
 * model is alive. */
#define FIXTURE_MAX_LAYERS 12
typedef struct fixture {
    layer_t *model[FIXTURE_MAX_LAYERS];
    size_t n;
    lossFuncType_t lt;
    inputLike_t in;
    tensor_t *x;
    uint8_t bfpExponent[1];
    bfpQConfig_t bfpQc;
    quantization_t bfpQ;
} fixture_t;

static void buildHarFixture(fixture_t *f, bool freezeConvs) {
    buildHar(f->model, freezeConvs);
    f->n = HAR_N;
    f->lt = CROSS_ENTROPY;
    f->x = makeResidentInput(&f->in, (size_t[]){1, 9, 128}, 3);
}

static void buildHarModel(fixture_t *f) {
    buildHarFixture(f, false);
}

/* The F1 alignment model: FLOAT32 [1,5] -> Quantization to
 * BFP m = 8 -> Linear 5 -> 1 under MSE. n = 2, deepest 1, top 1. Wires: ACT 0,
 * ACT 1 (BFP, 5 B), ACT 2 (4 B), the seed GRAD 2 (id 3, 4 B). Steps: FORWARD 0
 * (#0), FORWARD 1 (#1), LOSS_FORWARD (#2), LOSS_BACKWARD (#3), BACKWARD(1)
 * (#4, grads-only). Unaligned FFD would place the wires at {0, 5, 9}. */
static void buildF1Model(fixture_t *f) {
    initBfpQConfigInto(8, 8, HALF_AWAY, f->bfpExponent, &f->bfpQc);
    f->bfpQ = (quantization_t){.type = BFP, .qConfig = &f->bfpQc};
    f->model[0] = makeQuant(&f->bfpQ, &g_floatQ);
    f->model[1] = makeLinear(5, 1, false);
    f->n = 2;
    f->lt = MSE;
    f->x = makeResidentInput(&f->in, (size_t[]){1, 5}, 2);
}

/* Both inits share one shape, so a test can run on either row. */
typedef rematScheduler_t (*rowInit_t)(fixture_t *f, const rematPlanSpec_t *spec);

static rematScheduler_t initArena(fixture_t *f, const rematPlanSpec_t *spec) {
    rematScheduler_t s;
    TEST_ASSERT_TRUE(rematArenaInit(&s, f->model, f->n, defaultLossConfig(f->lt), f->x, spec));
    TEST_ASSERT_NOT_NULL(s.wires);
    TEST_ASSERT_NOT_NULL(s.plan);
    return s;
}

static rematScheduler_t initHeap(fixture_t *f, const rematPlanSpec_t *spec) {
    rematScheduler_t s;
    TEST_ASSERT_TRUE(rematHeapInit(&s, f->model, f->n, defaultLossConfig(f->lt), f->x, spec));
    TEST_ASSERT_NOT_NULL(s.wires);
    TEST_ASSERT_NOT_NULL(s.plan);
    return s;
}

static void freeFixture(fixture_t *f, rematScheduler_t *s) {
    rematSchedulerDeinit(s);
    freeModel(f->model, f->n);
}

#endif // ODT_TEST_REMAT_TEST_FIXTURES_H
