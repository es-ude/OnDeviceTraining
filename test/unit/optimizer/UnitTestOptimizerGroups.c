#define SOURCE_FILE "UNIT_TEST_OPTIMIZER_GROUPS"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "AdamW.h"
#include "AdamWApi.h"
#include "ArithmeticType.h"
#include "BorrowedLayer.h"
#include "DeathTest.h"
#include "DyadicScale.h"
#include "Linear.h"
#include "LrScheduler.h"
#include "Optimizer.h"
#include "OptimizerApi.h"
#include "ParameterWalk.h"
#include "QuantizationApi.h"
#include "RNG.h"
#include "Sgd.h"
#include "SgdApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

static const arithmetic_t FLOAT_MATH = {.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY};

static const float W0[4] = {0.5f, -1.f, 2.f, 1.f};
static const float B0[2] = {0.1f, -0.2f};
static const float GW[4] = {1.f, 2.f, 3.f, 4.f};
static const float GB[2] = {0.5f, 0.5f};

static parameter_t *makeParam(quantization_t *q, const size_t *dims, size_t numDims,
                              const float *values, const float *grads, size_t count) {
    size_t *d = reserveMemory(numDims * sizeof(size_t));
    memcpy(d, dims, numDims * sizeof(size_t));
    size_t *order = reserveMemory(numDims * sizeof(size_t));
    setOrderOfDimsForNewTensor(numDims, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, d, numDims, order);
    tensor_t *p = initTensor(shape, q, NULL);
    tensorFillFromFloatBuffer(p, values, count);
    tensor_t *g = gradInitFloat(p, NULL);
    tensorFillFromFloatBuffer(g, grads, count);
    return parameterInit(p, g);
}

/* Linear(2->2, bias) with W0/B0 and grads GW/GB, borrowing its parameters. */
static layer_t *makeLinear(quantization_t *layerQ) {
    parameter_t *w = makeParam(quantizationInitFloat(), (size_t[]){2, 2}, 2, W0, GW, 4);
    parameter_t *b = makeParam(quantizationInitFloat(), (size_t[]){2}, 1, B0, GB, 2);
    return buildBorrowedLinearLayer(w, b, layerQ);
}

static float paramAt(const parameter_t *p, size_t i) {
    return ((const float *)p->param->data)[i];
}

static uint8_t groupOfSlot(const optimizer_t *optim, size_t k) {
    return optim->groupOf == NULL ? 0 : optim->groupOf[k];
}

static bool selectNothing(const paramRef_t *ref, void *ctx) {
    (void)ref;
    (void)ctx;
    return false;
}

void testSgdTwoGroupsOneStepMomentumZero(void) {
    quantization_t *layerQ = quantizationInitFloat();
    quantization_t *momentumQ = quantizationInitFloat();
    layer_t *lin = makeLinear(layerQ);
    layer_t *layers[] = {lin};
    const model_t *model = &MODEL_CHAIN(layers, 1);
    optimizer_t *optim = sgdMCreateOptim(0.1f, 0.0f, 0.01f, model, momentumQ, FLOAT_MATH);

    size_t group = sgdMAddGroup(optim, model, paramSelectNormOrBias, NULL, 0.5f, 0.0f);
    optimizerStep(optim, NULL);

    float w[4], b[2];
    for (size_t i = 0; i < 4; i++) {
        w[i] = paramAt(lin->config->linear->weights, i);
    }
    for (size_t i = 0; i < 2; i++) {
        b[i] = paramAt(lin->config->linear->bias, i);
    }
    freeOptim(optim);
    freeLinearLayerShellOnly(lin);
    freeQuantization(momentumQ);
    freeQuantization(layerQ);

    TEST_ASSERT_EQUAL_size_t(1, group);
    for (size_t i = 0; i < 4; i++) {
        TEST_ASSERT_EQUAL_FLOAT(W0[i] - 0.1f * (GW[i] + 0.01f * W0[i]), w[i]);
    }
    for (size_t i = 0; i < 2; i++) {
        TEST_ASSERT_EQUAL_FLOAT(B0[i] - 0.05f * GB[i], b[i]);
    }
}

void testSgdTwoGroupsOneStepMomentum(void) {
    quantization_t *layerQ = quantizationInitFloat();
    quantization_t *momentumQ = quantizationInitFloat();
    layer_t *lin = makeLinear(layerQ);
    layer_t *layers[] = {lin};
    const model_t *model = &MODEL_CHAIN(layers, 1);
    optimizer_t *optim = sgdMCreateOptim(0.1f, 0.9f, 0.01f, model, momentumQ, FLOAT_MATH);

    sgdMAddGroup(optim, model, paramSelectNormOrBias, NULL, 0.5f, 0.0f);
    optimizerStep(optim, NULL);
    optimizerStep(optim, NULL);

    float w[4], b[2];
    for (size_t i = 0; i < 4; i++) {
        w[i] = paramAt(lin->config->linear->weights, i);
    }
    for (size_t i = 0; i < 2; i++) {
        b[i] = paramAt(lin->config->linear->bias, i);
    }
    freeOptim(optim);
    freeLinearLayerShellOnly(lin);
    freeQuantization(momentumQ);
    freeQuantization(layerQ);

    /* state = 0.9*state + g + wd*w; param -= lr_g*state; grads stay set across both steps */
    for (size_t i = 0; i < 4; i++) {
        float p = W0[i];
        float s = 0.f;
        for (int step = 0; step < 2; step++) {
            s = 0.9f * s + (GW[i] + 0.01f * p);
            p = p - 0.1f * s;
        }
        TEST_ASSERT_EQUAL_FLOAT(p, w[i]);
    }
    for (size_t i = 0; i < 2; i++) {
        float p = B0[i];
        float s = 0.f;
        for (int step = 0; step < 2; step++) {
            s = 0.9f * s + GB[i];
            p = p - 0.05f * s;
        }
        TEST_ASSERT_EQUAL_FLOAT(p, b[i]);
    }
}

/* The AdamW.c per-step scalars (double), with the group's own lr and weight decay. */
static float adamWFirstStepExpected(float p, float g, double lr, double wd) {
    const double beta1 = 0.9, beta2 = 0.999, eps = 1e-8, t = 1.0;
    double w1 = 1.0 - beta1;
    double s2 = 1.0 - beta2;
    double bc1 = 1.0 - pow(beta1, t);
    double bc2sqrt = sqrt(1.0 - pow(beta2, t));
    double stepScale = -(lr / bc1);
    double decay = 1.0 - lr * wd;
    double m = w1 * (double)g;
    double v = s2 * (double)g * (double)g;
    return (float)(decay * (double)p + stepScale * m / (sqrt(v) / bc2sqrt + eps));
}

void testAdamWTwoGroupsFirstStep(void) {
    quantization_t *layerQ = quantizationInitFloat();
    quantization_t *momentQ = quantizationInitFloat();
    layer_t *lin = makeLinear(layerQ);
    layer_t *layers[] = {lin};
    const model_t *model = &MODEL_CHAIN(layers, 1);
    optimizer_t *optim = adamWCreateOptim(0.01f, 0.9, 0.999, 1e-8, 0.1, model, momentQ, FLOAT_MATH);

    size_t group = adamWAddGroup(optim, model, paramSelectNormOrBias, NULL, 2.0f, 0.0);
    optimizerStep(optim, NULL);

    float w[4], b[2];
    for (size_t i = 0; i < 4; i++) {
        w[i] = paramAt(lin->config->linear->weights, i);
    }
    for (size_t i = 0; i < 2; i++) {
        b[i] = paramAt(lin->config->linear->bias, i);
    }
    freeOptim(optim);
    freeLinearLayerShellOnly(lin);
    freeQuantization(momentQ);
    freeQuantization(layerQ);

    TEST_ASSERT_EQUAL_size_t(1, group);
    for (size_t i = 0; i < 4; i++) {
        TEST_ASSERT_EQUAL_FLOAT(adamWFirstStepExpected(W0[i], GW[i], (double)0.01f, 0.1), w[i]);
    }
    for (size_t i = 0; i < 2; i++) {
        TEST_ASSERT_EQUAL_FLOAT(adamWFirstStepExpected(B0[i], GB[i], (double)(0.01f * 2.0f), 0.0),
                                b[i]);
    }
}

void testSetLrScalesEveryGroupAndGetLrReturnsBase(void) {
    quantization_t *layerQ = quantizationInitFloat();
    quantization_t *momentumQ = quantizationInitFloat();
    layer_t *lin = makeLinear(layerQ);
    layer_t *layers[] = {lin};
    const model_t *model = &MODEL_CHAIN(layers, 1);
    optimizer_t *optim = sgdMCreateOptim(0.1f, 0.9f, 0.0f, model, momentumQ, FLOAT_MATH);
    sgdMAddGroup(optim, model, paramSelectNormOrBias, NULL, 0.5f, 0.0f);

    optimizerFunctions[SGD_M].setLr(optim, 0.2f);
    float baseAfterSet = optimizerFunctions[SGD_M].getLr(optim);
    float group1AfterSet = optim->impl->sgd[1].learningRate;

    /* The scheduler captures its base at init, so restore 0.1 first. */
    optimizerFunctions[SGD_M].setLr(optim, 0.1f);
    lrScheduler_t sched;
    stepLrInit(&sched, optim, 1, 0.1f);
    lrSchedulerStep(&sched);
    float baseAfterSched = optimizerFunctions[SGD_M].getLr(optim);
    float group1AfterSched = optim->impl->sgd[1].learningRate;

    freeOptim(optim);
    freeLinearLayerShellOnly(lin);
    freeQuantization(momentumQ);
    freeQuantization(layerQ);

    TEST_ASSERT_EQUAL_FLOAT(0.2f, baseAfterSet);
    TEST_ASSERT_EQUAL_FLOAT(0.1f, group1AfterSet);
    TEST_ASSERT_EQUAL_FLOAT(0.01f, baseAfterSched);
    TEST_ASSERT_EQUAL_FLOAT(0.005f, group1AfterSched);
}

void testAddGroupWithFrozenLayerSelectsRightParameters(void) {
    quantization_t *layerQ = quantizationInitFloat();
    quantization_t *momentumQ = quantizationInitFloat();
    layer_t *first = makeLinear(layerQ);
    layer_t *frozen = makeLinear(layerQ);
    frozen->config->linear->frozen = true;
    layer_t *last = makeLinear(layerQ);
    layer_t *layers[] = {first, frozen, last};
    const model_t *model = &MODEL_CHAIN(layers, 3);
    optimizer_t *optim = sgdMCreateOptim(0.1f, 0.9f, 0.01f, model, momentumQ, FLOAT_MATH);

    sgdMAddGroup(optim, model, paramSelectNormOrBias, NULL, 1.0f, 0.0f);

    size_t sizeStates = optim->sizeStates;
    bool grouped[4] = {false};
    bool biasSlot[4] = {false};
    for (size_t k = 0; k < sizeStates && k < 4; k++) {
        grouped[k] = groupOfSlot(optim, k) == 1;
        biasSlot[k] = optim->parameter[k] == first->config->linear->bias ||
                      optim->parameter[k] == last->config->linear->bias;
    }

    freeOptim(optim);
    freeParameter(frozen->config->linear->weights);
    freeParameter(frozen->config->linear->bias);
    freeLinearLayerShellOnly(first);
    freeLinearLayerShellOnly(frozen);
    freeLinearLayerShellOnly(last);
    freeQuantization(momentumQ);
    freeQuantization(layerQ);

    TEST_ASSERT_EQUAL_size_t(4, sizeStates);
    for (size_t k = 0; k < 4; k++) {
        TEST_ASSERT_EQUAL(biasSlot[k], grouped[k]);
    }
    TEST_ASSERT_TRUE(grouped[1]);
    TEST_ASSERT_TRUE(grouped[3]);
}

#define SR_LEN 8u

/* SYM_INT32 bias: an anchor that pins the scale, then targets whose per-step update is a
 * quarter ULP, so every code that moves was moved by an SR draw. */
static layer_t *makeSrLinear(quantization_t *layerQ) {
    const float lr = 0.1f;
    const float scale = 100.f / 2047.f;
    float bias[SR_LEN] = {100.f};
    float gradB[SR_LEN] = {0.f};
    for (size_t i = 1; i < SR_LEN; i++) {
        gradB[i] = 0.25f * scale / lr;
    }
    parameter_t *w = makeParam(quantizationInitFloat(), (size_t[]){2, 2}, 2, W0, GW, 4);
    parameter_t *b =
        makeParam(quantizationInitSymInt32(HALF_AWAY), (size_t[]){SR_LEN}, 1, bias, gradB, SR_LEN);
    return buildBorrowedLinearLayer(w, b, layerQ);
}

/* Steps 20 times under keys seed+i; copies the bias codes and scale out. */
static void runSrSteps(bool addGroup, uint64_t seed, int32_t *codes, float *scale) {
    quantization_t *layerQ = quantizationInitFloat();
    quantization_t *momentumQ = quantizationInitFloat();
    layer_t *lin = makeSrLinear(layerQ);
    layer_t *layers[] = {lin};
    const model_t *model = &MODEL_CHAIN(layers, 1);
    optimizer_t *optim = sgdMCreateOptim(0.1f, 0.0f, 0.01f, model, momentumQ, FLOAT_MATH);
    if (addGroup) {
        sgdMAddGroup(optim, model, paramSelectNormOrBias, NULL, 1.0f, 0.01f);
    }
    for (uint64_t i = 0; i < 20; i++) {
        const rngKey_t key = rngKeyFromSeed(seed + i);
        optimizerStep(optim, &key);
    }
    const tensor_t *b = lin->config->linear->bias->param;
    memcpy(codes, b->data, SR_LEN * sizeof(int32_t));
    *scale = odtScaleToFloat(((symInt32QConfig_t *)b->quantization->qConfig)->scale);
    freeOptim(optim);
    freeLinearLayerShellOnly(lin);
    freeQuantization(momentumQ);
    freeQuantization(layerQ);
}

void testSingleGroupSrStreamUnchanged(void) {
    int32_t plain[SR_LEN], grouped[SR_LEN], otherKeys[SR_LEN];
    float plainScale, groupedScale, otherScale;
    runSrSteps(false, 1000u, plain, &plainScale);
    runSrSteps(true, 1000u, grouped, &groupedScale);
    runSrSteps(true, 5000u, otherKeys, &otherScale);

    TEST_ASSERT_EQUAL_INT32_ARRAY(plain, grouped, SR_LEN);
    TEST_ASSERT_EQUAL_MEMORY(&plainScale, &groupedScale, sizeof(float));
    /* Control: the codes do depend on the draws. */
    TEST_ASSERT_TRUE(memcmp(plain, otherKeys, sizeof plain) != 0);
}

void testAddGroupRejectsWrongModel(void) {
    ASSERT_EXITS_WITH_FAILURE({
        quantization_t *layerQ = quantizationInitFloat();
        layer_t *a[] = {makeLinear(layerQ)};
        layer_t *b[] = {makeLinear(layerQ)};
        optimizer_t *optim = sgdMCreateOptim(0.1f, 0.0f, 0.0f, &MODEL_CHAIN(a, 1),
                                             quantizationInitFloat(), FLOAT_MATH);
        sgdMAddGroup(optim, &MODEL_CHAIN(b, 1), paramSelectNormOrBias, NULL, 1.0f, 0.0f);
    });
}

void testAddGroupRejectsParameterAlreadyGrouped(void) {
    ASSERT_EXITS_WITH_FAILURE({
        quantization_t *layerQ = quantizationInitFloat();
        layer_t *layers[] = {makeLinear(layerQ)};
        const model_t *model = &MODEL_CHAIN(layers, 1);
        optimizer_t *optim =
            adamWCreateOptim(0.01f, 0.9, 0.999, 1e-8, 0.1, model, layerQ, FLOAT_MATH);
        adamWAddGroup(optim, model, paramSelectNormOrBias, NULL, 1.0f, 0.0);
        adamWAddGroup(optim, model, paramSelectNormOrBias, NULL, 2.0f, 0.0);
    });
}

void testAddGroupRejectsEmptySelection(void) {
    ASSERT_EXITS_WITH_FAILURE({
        quantization_t *layerQ = quantizationInitFloat();
        layer_t *layers[] = {makeLinear(layerQ)};
        const model_t *model = &MODEL_CHAIN(layers, 1);
        optimizer_t *optim = sgdMCreateOptim(0.1f, 0.0f, 0.0f, model, layerQ, FLOAT_MATH);
        sgdMAddGroup(optim, model, selectNothing, NULL, 1.0f, 0.0f);
    });
}

void testAddGroupRejectsNonPositiveMultiplier(void) {
    ASSERT_EXITS_WITH_FAILURE({
        quantization_t *layerQ = quantizationInitFloat();
        layer_t *layers[] = {makeLinear(layerQ)};
        const model_t *model = &MODEL_CHAIN(layers, 1);
        optimizer_t *optim = sgdMCreateOptim(0.1f, 0.0f, 0.0f, model, layerQ, FLOAT_MATH);
        sgdMAddGroup(optim, model, paramSelectNormOrBias, NULL, 0.0f, 0.0f);
    });
    ASSERT_EXITS_WITH_FAILURE({
        quantization_t *layerQ = quantizationInitFloat();
        layer_t *layers[] = {makeLinear(layerQ)};
        const model_t *model = &MODEL_CHAIN(layers, 1);
        optimizer_t *optim = sgdMCreateOptim(0.1f, 0.0f, 0.0f, model, layerQ, FLOAT_MATH);
        sgdMAddGroup(optim, model, paramSelectNormOrBias, NULL, NAN, 0.0f);
    });
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testSgdTwoGroupsOneStepMomentumZero);
    RUN_TEST(testSgdTwoGroupsOneStepMomentum);
    RUN_TEST(testAdamWTwoGroupsFirstStep);
    RUN_TEST(testSetLrScalesEveryGroupAndGetLrReturnsBase);
    RUN_TEST(testAddGroupWithFrozenLayerSelectsRightParameters);
    RUN_TEST(testSingleGroupSrStreamUnchanged);
    RUN_TEST(testAddGroupRejectsWrongModel);
    RUN_TEST(testAddGroupRejectsParameterAlreadyGrouped);
    RUN_TEST(testAddGroupRejectsEmptySelection);
    RUN_TEST(testAddGroupRejectsNonPositiveMultiplier);
    return UNITY_END();
}
