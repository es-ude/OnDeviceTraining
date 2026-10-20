#ifndef ODT_TEST_REMAT_TEST_FIXTURES_H
#define ODT_TEST_REMAT_TEST_FIXTURES_H

/* Fixture builders shared by the remat test binaries (UnitTestRematPlan,
 * UnitTestRematScheduler, UnitTestRematCheck) and the driver's conformance
 * and death binaries, so all of them build the same models. Header-only
 * statics, like BorrowedLayer.h: each binary compiles the helpers it calls.
 * A consumer links the layer API targets of every type freeModel frees
 * (LinearApi, ReluApi, SoftmaxApi, Conv1dApi, Pool1dApi, FlattenApi,
 * QuantLayerApi, LayerNormApi) and RematScheduler for initArena/initHeap. */

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
#include "RNG.h"
#include "ReluApi.h"
#include "RematScheduler.h"
#include "SoftmaxApi.h"
#include "Tensor.h"
#include "unity.h"

/* Fixture layers borrow their wire templates, so one FLOAT32 template outlives
 * every fixture model. Tests that edit a template use a local one. */
static quantization_t g_floatQ = {.type = FLOAT32, .qConfig = NULL};

/* position = the layer's index in its model; it picks the layer's init key, so
 * same-shape layers of one model do not share a key. */
static layer_t *makeLinear(size_t in, size_t out, bool frozen, size_t position) {
    layerQuant_t lq;
    layerQuantInitUniform(&lq, &g_floatQ);
    const rngKey_t initKey = rngFold(rngKeyFromSeed(1u), RNG_TAG_INIT, (uint32_t)position);
    return linearLayerInit(
        &(linearInit_t){.inFeatures = in,
                        .outFeatures = out,
                        .trainable = frozen ? TRAINABLE_FALSE : TRAINABLE_DEFAULT,
                        .weightInit.key = &initKey},
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

/* examples/har_classifier/train_c.c buildModel (B = 1); freezeConvs gives the
 * stage-2 backbone of train_c_finetune.c buildModel. n = 12, CE, deepest 0, top
 * 10; 25 steps; ACT 0..12 plus GRAD {12, 10, 9, ..., 1} = 24 wires. */
#define HAR_N 12
static void buildHar(layer_t **model, bool freezeConvs) {
    layerQuant_t lq;
    layerQuantInitUniform(&lq, &g_floatQ);
    trainable_t conv = freezeConvs ? TRAINABLE_FALSE : TRAINABLE_DEFAULT;
    const rngKey_t initKey0 = rngFold(rngKeyFromSeed(1u), RNG_TAG_INIT, 0u);
    const rngKey_t initKey3 = rngFold(rngKeyFromSeed(1u), RNG_TAG_INIT, 3u);
    const rngKey_t initKey6 = rngFold(rngKeyFromSeed(1u), RNG_TAG_INIT, 6u);
    const rngKey_t initKey10 = rngFold(rngKeyFromSeed(1u), RNG_TAG_INIT, 10u);
    model[0] = conv1dLayerInit(&(conv1dInit_t){.inChannels = 9,
                                               .outChannels = 16,
                                               .kernelSize = 7,
                                               .padding = SAME,
                                               .trainable = conv,
                                               .weightInit.key = &initKey0},
                               &lq);
    model[1] = reluLayerInit(&lq);
    model[2] = maxPool1dLayerInit(
        &(maxPool1dInit_t){.kernelSize = 2, .stride = 2, .inputChannels = 16, .inputLength = 128},
        &lq);
    model[3] = conv1dLayerInit(&(conv1dInit_t){.inChannels = 16,
                                               .outChannels = 32,
                                               .kernelSize = 5,
                                               .padding = SAME,
                                               .trainable = conv,
                                               .weightInit.key = &initKey3},
                               &lq);
    model[4] = reluLayerInit(&lq);
    model[5] = maxPool1dLayerInit(
        &(maxPool1dInit_t){.kernelSize = 2, .stride = 2, .inputChannels = 32, .inputLength = 64},
        &lq);
    model[6] = conv1dLayerInit(&(conv1dInit_t){.inChannels = 32,
                                               .outChannels = 64,
                                               .kernelSize = 3,
                                               .padding = SAME,
                                               .trainable = conv,
                                               .weightInit.key = &initKey6},
                               &lq);
    model[7] = reluLayerInit(&lq);
    model[8] = avgPool1dLayerInit(&(avgPool1dInit_t){.kernelSize = 32, .stride = 32}, &lq);
    model[9] = flattenLayerInit();
    model[10] = linearLayerInit(
        &(linearInit_t){.inFeatures = 64, .outFeatures = 6, .weightInit.key = &initKey10}, &lq);
    model[11] = softmaxLayerInit(&lq);
}

static tensor_t *makeHarInput(inputLike_t *in) {
    return makeInput(in, (size_t[]){1, 9, 128}, 3, &g_floatQ);
}

/* examples/mnist_cnn/train_c.c buildModel (B = 1; the loop feeds [1, 1, 784]). */
#define MNIST_N 10
static void buildMnistCnn(layer_t **model) {
    layerQuant_t lq;
    layerQuantInitUniform(&lq, &g_floatQ);
    const rngKey_t initKey0 = rngFold(rngKeyFromSeed(1u), RNG_TAG_INIT, 0u);
    const rngKey_t initKey3 = rngFold(rngKeyFromSeed(1u), RNG_TAG_INIT, 3u);
    const rngKey_t initKey8 = rngFold(rngKeyFromSeed(1u), RNG_TAG_INIT, 8u);
    model[0] = conv1dLayerInit(&(conv1dInit_t){.inChannels = 1,
                                               .outChannels = 8,
                                               .kernelSize = 3,
                                               .padding = SAME,
                                               .weightInit.key = &initKey0},
                               &lq);
    model[1] = reluLayerInit(&lq);
    model[2] = maxPool1dLayerInit(
        &(maxPool1dInit_t){.kernelSize = 2, .stride = 2, .inputChannels = 8, .inputLength = 784},
        &lq);
    model[3] = conv1dLayerInit(&(conv1dInit_t){.inChannels = 8,
                                               .outChannels = 16,
                                               .kernelSize = 3,
                                               .padding = SAME,
                                               .weightInit.key = &initKey3},
                               &lq);
    model[4] = reluLayerInit(&lq);
    model[5] = maxPool1dLayerInit(
        &(maxPool1dInit_t){.kernelSize = 2, .stride = 2, .inputChannels = 16, .inputLength = 392},
        &lq);
    model[6] = avgPool1dLayerInit(&(avgPool1dInit_t){.kernelSize = 196, .stride = 196}, &lq);
    model[7] = flattenLayerInit();
    model[8] = linearLayerInit(
        &(linearInit_t){.inFeatures = 16, .outFeatures = 10, .weightInit.key = &initKey8}, &lq);
    model[9] = softmaxLayerInit(&lq);
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

/* Step and range records for hand-written programs and sequences. */
#define STEP_F(l) {.kind = REMAT_STEP_FORWARD, .layer = (l)}
#define STEP_LF(n) {.kind = REMAT_STEP_LOSS_FORWARD, .layer = (n)}
#define STEP_LB(n) {.kind = REMAT_STEP_LOSS_BACKWARD, .layer = (n)}
#define STEP_B(l) {.kind = REMAT_STEP_BACKWARD, .layer = (l)}
#define STEP_RF(l) {.kind = REMAT_STEP_REFORWARD, .layer = (l)}
#define RANGE_OF(w, b, e) {.wire = (w), .begin = (b), .end = (e)}
#define COUNT_OF(a) (sizeof(a) / sizeof((a)[0]))

/* The recompute tests' 3-layer MLP: Linear(2->4), ReLU, Linear(4->2), every
 * layer trainable, input [1, 2] FLOAT32. Under MSE deepest = 0 and top = 2;
 * ACT 0..3 are wires 0..3, GRAD 3 (the seed) is 4, GRAD 2 is 5, GRAD 1 is 6. */
#define MLP3_N 3
static void buildMlp3(layer_t **model) {
    model[0] = makeLinear(2, 4, false, 0);
    model[1] = makeRelu(&g_floatQ);
    model[2] = makeLinear(4, 2, false, 2);
}

/* MLP3 under MSE with ACT 3 kept and ACT 1, ACT 2 dropped after their last
 * forward reads: the REFORWARD run 0..1 regenerates both before BACKWARD(2),
 * which reads ACT 2; BACKWARD(1) (ReLU) then reads the regenerated ACT 1. */
#define MLP3_RECOMPUTE_STEPS                                                                       \
    {STEP_F(0),  STEP_F(1),  STEP_F(2), STEP_LF(3), STEP_LB(3),                                    \
     STEP_RF(0), STEP_RF(1), STEP_B(2), STEP_B(1),  STEP_B(0)}
/* Its ranges, derived by hand from the steps above: each write opens a range
 * that ends at the last read of that wire before its next write. */
#define MLP3_RECOMPUTE_RANGES                                                                      \
    {RANGE_OF(1, 0, 1), RANGE_OF(2, 1, 2), RANGE_OF(3, 2, 4), RANGE_OF(4, 4, 7),                   \
     RANGE_OF(1, 5, 8), RANGE_OF(2, 6, 7), RANGE_OF(5, 7, 8), RANGE_OF(6, 8, 9)}
/* The same MLP3 recomputing ACT 1 twice: the run 0..1 before BACKWARD(2), and
 * REFORWARD(0) once more before BACKWARD(1). 11 steps, 9 writes. */
#define MLP3_RECOMPUTE_TWICE_STEPS                                                                 \
    {STEP_F(0),  STEP_F(1), STEP_F(2),  STEP_LF(3), STEP_LB(3), STEP_RF(0),                        \
     STEP_RF(1), STEP_B(2), STEP_RF(0), STEP_B(1),  STEP_B(0)}

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
    f->model[1] = makeLinear(5, 1, false, 1);
    f->n = 2;
    f->lt = MSE;
    f->x = makeResidentInput(&f->in, (size_t[]){1, 5}, 2);
}

static void buildMnistCnnModel(fixture_t *f) {
    buildMnistCnn(f->model);
    f->n = MNIST_N;
    f->lt = CROSS_ENTROPY;
    f->x = makeResidentInput(&f->in, (size_t[]){1, 1, 784}, 3);
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
