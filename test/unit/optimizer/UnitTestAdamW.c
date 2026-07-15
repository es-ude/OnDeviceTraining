#define SOURCE_FILE "ADAM-W-UTEST"

#include <math.h>
#include <stdint.h>
#include <string.h>

#include "AdamW.h"
#include "AdamWApi.h"
#include "ArithmeticType.h"
#include "DeathTest.h"
#include "ExecuteOp.h"
#include "Linear.h"
#include "LrScheduler.h"
#include "Optimizer.h"
#include "OptimizerApi.h"
#include "QuantizationApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "unity.h"

#include "expected_adamw.h"

/* #328 signature contract (compile-time pin). */
_Static_assert(_Generic(&adamWInit,
                   void (*)(adamW_t *, float, double, double, double, double, arithmetic_t): 1,
                   default: 0),
               "#328: adamWInit must be (adamW, lr, beta1, beta2, eps, weightDecay, updateMath)");

/* #328 PR C factory signature contract. */
_Static_assert(_Generic(&adamWCreateOptim,
                   optimizer_t *(*)(float, double, double, double, double, layer_t **, size_t,
                                    quantization_t *, arithmetic_t): 1,
                   default: 0),
               "#328: adamWCreateOptim must be (lr, beta1, beta2, eps, weightDecay, model, "
               "sizeModel, momentQuant, updateMath)");

void setUp() {}
void tearDown() {}

static const arithmetic_t FLOAT_MATH = {.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY};

void testAdamWInitStoresDoubleHyperparamsAndZeroStepCount(void) {
    adamW_t adamW;
    adamWInit(&adamW, 0.001f, 0.9, 0.999, 1e-8, 0.01, FLOAT_MATH);
    TEST_ASSERT_EQUAL_FLOAT(0.001f, adamW.learningRate);
    /* Exact == on purpose: DOUBLE storage of the betas is the numerics
     * contract (float-stored betas lose 1-beta unrecoverably). A float
     * field would round 0.999 to 0.99900001287460327 and fail here. */
    TEST_ASSERT_TRUE(adamW.beta1 == 0.9);
    TEST_ASSERT_TRUE(adamW.beta2 == 0.999);
    TEST_ASSERT_TRUE(adamW.eps == 1e-8);
    TEST_ASSERT_TRUE(adamW.weightDecay == 0.01);
    TEST_ASSERT_EQUAL_size_t(0, adamW.stepCount);
    TEST_ASSERT_EQUAL_INT(ARITH_FLOAT32, adamW.updateMath.type);
}

void testAdamWGetSetLrRoundTripThroughImpl(void) {
    adamW_t adamW;
    adamWInit(&adamW, 0.001f, 0.9, 0.999, 1e-8, 0.01, FLOAT_MATH);
    optimImpl_t impl = {.adamW = &adamW};
    optimizer_t optim = {.type = ADAM_W, .impl = &impl};
    TEST_ASSERT_EQUAL_FLOAT(0.001f, adamWGetLr(&optim));
    adamWSetLr(&optim, 0.5f);
    TEST_ASSERT_EQUAL_FLOAT(0.5f, adamWGetLr(&optim));
    TEST_ASSERT_EQUAL_FLOAT(0.5f, adamW.learningRate);
}

void testAdamWInitRejectsNonFloat32UpdateMath(void) {
    adamW_t adamW;
    ASSERT_EXITS_WITH_FAILURE(
        adamWInit(&adamW, 0.001f, 0.9, 0.999, 1e-8, 0.01,
                  (arithmetic_t){.type = ARITH_SYM_INT32, .roundingMode = HALF_AWAY}));
}

void testAdamWInitRejectsBeta1AtOne(void) {
    adamW_t adamW;
    ASSERT_EXITS_WITH_FAILURE(adamWInit(&adamW, 0.001f, 1.0, 0.999, 1e-8, 0.01, FLOAT_MATH));
}

void testAdamWInitRejectsNegativeBeta2(void) {
    adamW_t adamW;
    ASSERT_EXITS_WITH_FAILURE(adamWInit(&adamW, 0.001f, 0.9, -0.1, 1e-8, 0.01, FLOAT_MATH));
}

void testAdamWInitRejectsNanBeta1(void) {
    adamW_t adamW;
    ASSERT_EXITS_WITH_FAILURE(adamWInit(&adamW, 0.001f, NAN, 0.999, 1e-8, 0.01, FLOAT_MATH));
}

void testAdamWInitRejectsZeroEps(void) {
    adamW_t adamW;
    ASSERT_EXITS_WITH_FAILURE(adamWInit(&adamW, 0.001f, 0.9, 0.999, 0.0, 0.01, FLOAT_MATH));
}

/* Heap-tier Rule-1 fixture: 1-D FLOAT32 tensor filled from src (NULL src =
 * zero-filled; initTensor's reserveMemory data is calloc-zeroed). */
static tensor_t *makeFloatTensor1D(const float *src, size_t n) {
    size_t *dims = reserveMemory(1 * sizeof(size_t));
    dims[0] = n;
    size_t *order = reserveMemory(1 * sizeof(size_t));
    setOrderOfDimsForNewTensor(1, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 1, order);
    tensor_t *t = initTensor(shape, quantizationInitFloat(), NULL);
    if (src != NULL) {
        tensorFillFromFloatBuffer(t, src, n);
    }
    return t;
}

static void assertBitsEqualAt(const float *expected, const tensor_t *actual, size_t n,
                              const char *what) {
    const float *act = (const float *)actual->data;
    for (size_t i = 0; i < n; i++) {
        uint32_t e, a;
        memcpy(&e, &expected[i], sizeof e);
        memcpy(&a, &act[i], sizeof a);
        char msg[96];
        snprintf(msg, sizeof msg, "%s[%zu]", what, i);
        TEST_ASSERT_EQUAL_HEX32_MESSAGE(e, a, msg);
    }
}

/* Single-parameter hand-assembled AdamW optimizer (no factory until Task 4).
 * Caller frees: freeParameter(par); freeTensor(m); freeTensor(v). */
typedef struct {
    adamW_t adamW;
    optimImpl_t impl;
    parameter_t *par;
    tensor_t *m, *v;
    states_t st;
    tensor_t *stateBuffers[2];
    parameter_t *parArr[1];
    states_t *stArr[1];
    optimizer_t optim;
} adamWHarness_t;

static optimizer_t *harnessInit(adamWHarness_t *h, const float *p0, const float *g0, size_t n,
                                float lr, double beta1, double beta2, double eps, double wd) {
    adamWInit(&h->adamW, lr, beta1, beta2, eps, wd,
              (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});
    h->impl = (optimImpl_t){.adamW = &h->adamW};
    tensor_t *p = makeFloatTensor1D(p0, n);
    tensor_t *g = makeFloatTensor1D(g0, n);
    h->par = parameterInit(p, g);
    h->m = makeFloatTensor1D(NULL, n);
    h->v = makeFloatTensor1D(NULL, n);
    h->stateBuffers[0] = h->m;
    h->stateBuffers[1] = h->v;
    h->st = (states_t){.stateBuffers = h->stateBuffers, .statesPerParameter = 2};
    h->parArr[0] = h->par;
    h->stArr[0] = &h->st;
    h->optim = (optimizer_t){.type = ADAM_W,
                             .impl = &h->impl,
                             .parameter = h->parArr,
                             .states = h->stArr,
                             .sizeStates = 1};
    return &h->optim;
}

static void harnessFree(adamWHarness_t *h) {
    freeParameter(h->par);
    freeTensor(h->m);
    freeTensor(h->v);
}

void testAdamWStepMatchesGoldStep1Defaults(void) {
    adamWHarness_t h;
    optimizer_t *optim = harnessInit(&h, adamw_step1_default_p0, adamw_step1_default_g, 32, 0.001f,
                                     0.9, 0.999, 1e-8, 0.01);
    adamWStep(optim);
    float pOut[32], mOut[32], vOut[32];
    memcpy(pOut, h.par->param->data, sizeof pOut);
    memcpy(mOut, h.m->data, sizeof mOut);
    memcpy(vOut, h.v->data, sizeof vOut);
    size_t capturedStepCount = h.adamW.stepCount;
    harnessFree(&h);
    tensor_t pT = {.data = (uint8_t *)pOut}, mT = {.data = (uint8_t *)mOut},
             vT = {.data = (uint8_t *)vOut};
    assertBitsEqualAt(adamw_step1_default_p1, &pT, 32, "step1 default p");
    assertBitsEqualAt(adamw_step1_default_m1, &mT, 32, "step1 default m");
    assertBitsEqualAt(adamw_step1_default_v1, &vT, 32, "step1 default v");
    TEST_ASSERT_EQUAL_size_t(1, capturedStepCount);
}

void testAdamWStepMatchesGoldStep1WdZero(void) {
    /* wd=0 -> decay == 1.0 exactly; discriminates any wd contamination of
     * the moment path (coupled-L2 mutant changes m/v here too). */
    adamWHarness_t h;
    optimizer_t *optim =
        harnessInit(&h, adamw_step1_wd0_p0, adamw_step1_wd0_g, 32, 0.001f, 0.9, 0.999, 1e-8, 0.0);
    adamWStep(optim);
    float pOut[32], mOut[32], vOut[32];
    memcpy(pOut, h.par->param->data, sizeof pOut);
    memcpy(mOut, h.m->data, sizeof mOut);
    memcpy(vOut, h.v->data, sizeof vOut);
    harnessFree(&h);
    tensor_t pT = {.data = (uint8_t *)pOut}, mT = {.data = (uint8_t *)mOut},
             vT = {.data = (uint8_t *)vOut};
    assertBitsEqualAt(adamw_step1_wd0_p1, &pT, 32, "step1 wd0 p");
    assertBitsEqualAt(adamw_step1_wd0_m1, &mT, 32, "step1 wd0 m");
    assertBitsEqualAt(adamw_step1_wd0_v1, &vT, 32, "step1 wd0 v");
}

void testAdamWStepMatchesGoldStep1OrderDiscrim(void) {
    /* lr=0.1, wd=0.5: decay=0.95, update ~1e-1 -- K3 order mutations and
     * coupled-L2 are NOT absorbed here (absorption-trap fixture). */
    adamWHarness_t h;
    optimizer_t *optim = harnessInit(&h, adamw_step1_orderdiscrim_p0, adamw_step1_orderdiscrim_g,
                                     32, 0.1f, 0.9, 0.999, 1e-8, 0.5);
    adamWStep(optim);
    float pOut[32];
    memcpy(pOut, h.par->param->data, sizeof pOut);
    harnessFree(&h);
    tensor_t pT = {.data = (uint8_t *)pOut};
    assertBitsEqualAt(adamw_step1_orderdiscrim_p1, &pT, 32, "step1 orderdiscrim p");
}

void testAdamWVtableRowIsFullyPopulated(void) {
    TEST_ASSERT_EQUAL_PTR(adamWStep, optimizerFunctions[ADAM_W].step);
    TEST_ASSERT_EQUAL_PTR(optimizerZeroGrad, optimizerFunctions[ADAM_W].zero);
    TEST_ASSERT_EQUAL_PTR(adamWGetLr, optimizerFunctions[ADAM_W].getLr);
    TEST_ASSERT_EQUAL_PTR(adamWSetLr, optimizerFunctions[ADAM_W].setLr);
}

void testAdamWStepIncrementsStepCountPerCall(void) {
    adamWHarness_t h;
    optimizer_t *optim = harnessInit(&h, adamw_step1_default_p0, adamw_step1_default_g, 32, 0.001f,
                                     0.9, 0.999, 1e-8, 0.01);
    adamWStep(optim);
    adamWStep(optim);
    size_t captured = h.adamW.stepCount;
    harnessFree(&h);
    TEST_ASSERT_EQUAL_size_t(2, captured);
}

void testAdamWStepRejectsTamperedUpdateMath(void) {
    /* #310 pattern: hand-assembled optimizers must hit the same wall at
     * step time, not only at init. */
    adamWHarness_t h;
    optimizer_t *optim = harnessInit(&h, adamw_step1_default_p0, adamw_step1_default_g, 32, 0.001f,
                                     0.9, 0.999, 1e-8, 0.01);
    h.adamW.updateMath.type = ARITH_SYM_INT32;
    ASSERT_EXITS_WITH_FAILURE(adamWStep(optim));
    h.adamW.updateMath.type = ARITH_FLOAT32; /* restore for clean teardown */
    harnessFree(&h);
}

void testAdamWTrajectoryMatchesGoldEveryStep(void) {
    adamWHarness_t h;
    optimizer_t *optim = harnessInit(&h, adamw_traj_default_p0, adamw_traj_default_g, 32, 0.001f,
                                     0.9, 0.999, 1e-8, 0.01);
    float pSteps[8][32];
    for (size_t s = 0; s < 8; s++) {
        tensorFillFromFloatBuffer(h.par->grad, &adamw_traj_default_g[s * 32], 32);
        adamWStep(optim);
        memcpy(pSteps[s], h.par->param->data, sizeof pSteps[s]);
    }
    float mOut[32], vOut[32];
    memcpy(mOut, h.m->data, sizeof mOut);
    memcpy(vOut, h.v->data, sizeof vOut);
    harnessFree(&h);
    for (size_t s = 0; s < 8; s++) {
        char what[32];
        snprintf(what, sizeof what, "traj step %zu p", s + 1);
        tensor_t pT = {.data = (uint8_t *)pSteps[s]};
        assertBitsEqualAt(&adamw_traj_default_p_steps[s * 32], &pT, 32, what);
    }
    tensor_t mT = {.data = (uint8_t *)mOut}, vT = {.data = (uint8_t *)vOut};
    assertBitsEqualAt(adamw_traj_default_m_final, &mT, 32, "traj m final");
    assertBitsEqualAt(adamw_traj_default_v_final, &vT, 32, "traj v final");
}

void testAdamWTrajectoryWdZeroMatchesGold(void) {
    /* same shape as above with the traj_wd0 arrays; asserts the final
     * step's p plus m/v finals (intermediate steps covered by default). */
    adamWHarness_t h;
    optimizer_t *optim =
        harnessInit(&h, adamw_traj_wd0_p0, adamw_traj_wd0_g, 32, 0.001f, 0.9, 0.999, 1e-8, 0.0);
    for (size_t s = 0; s < 8; s++) {
        tensorFillFromFloatBuffer(h.par->grad, &adamw_traj_wd0_g[s * 32], 32);
        adamWStep(optim);
    }
    float pOut[32], mOut[32], vOut[32];
    memcpy(pOut, h.par->param->data, sizeof pOut);
    memcpy(mOut, h.m->data, sizeof mOut);
    memcpy(vOut, h.v->data, sizeof vOut);
    harnessFree(&h);
    tensor_t pT = {.data = (uint8_t *)pOut}, mT = {.data = (uint8_t *)mOut},
             vT = {.data = (uint8_t *)vOut};
    assertBitsEqualAt(&adamw_traj_wd0_p_steps[7 * 32], &pT, 32, "traj wd0 p final");
    assertBitsEqualAt(adamw_traj_wd0_m_final, &mT, 32, "traj wd0 m final");
    assertBitsEqualAt(adamw_traj_wd0_v_final, &vT, 32, "traj wd0 v final");
}

void testAdamWWithCosineSchedulerMatchesGold(void) {
    /* Pins the ADAM_W getLr/setLr row through the REAL scheduler plus the
     * (double)(float)lr composition: epoch e trains at the float32-cast
     * closed form, scheduler steps AFTER the epoch (PR A contract). */
    adamWHarness_t h;
    optimizer_t *optim = harnessInit(&h, adamw_sched_cosine_p0, adamw_sched_cosine_g, 32, 0.01f,
                                     0.9, 0.999, 1e-8, 0.01);
    lrScheduler_t sched;
    cosineAnnealingLrInit(&sched, optim, 5, 0.001f);
    float lrSeen[5];
    for (size_t e = 0; e < 5; e++) {
        lrSeen[e] = optimizerFunctions[ADAM_W].getLr(optim);
        tensorFillFromFloatBuffer(h.par->grad, &adamw_sched_cosine_g[e * 32], 32);
        optimizerFunctions[ADAM_W].step(optim);
        lrSchedulerStep(&sched);
    }
    float pOut[32];
    memcpy(pOut, h.par->param->data, sizeof pOut);
    harnessFree(&h);
    for (size_t e = 0; e < 5; e++) {
        char what[32];
        snprintf(what, sizeof what, "sched lr epoch %zu", e);
        uint32_t exp, act;
        memcpy(&exp, &adamw_sched_cosine_lr[e], sizeof exp);
        memcpy(&act, &lrSeen[e], sizeof act);
        TEST_ASSERT_EQUAL_HEX32_MESSAGE(exp, act, what);
    }
    tensor_t pT = {.data = (uint8_t *)pOut};
    assertBitsEqualAt(adamw_sched_cosine_p_final, &pT, 32, "sched p final");
}

/*! Borrows already-built weight/bias parameter_t and a single quantization
 *  for forward + all backward math (UnitTestSgd.c precedent, copied
 *  file-locally: the 3+-file cross-file-helper threshold is not met). */
static layer_t *buildBorrowedLinearLayer(parameter_t *weights, parameter_t *bias,
                                         quantization_t *q) {
    linearConfig_t *cfg = reserveMemory(sizeof(linearConfig_t));
    cfg->weights = weights;
    cfg->bias = bias;
    cfg->forwardMath = arithmeticFromQuantization(q);
    cfg->weightGradMath = arithmeticFromQuantization(q);
    cfg->biasGradMath = arithmeticFromQuantization(q);
    cfg->propLossMath = arithmeticFromQuantization(q);
    cfg->outputQ = q;
    cfg->propLossQ = q;
    cfg->ownsQuantizations = false;
    layerConfig_t *layerCfg = reserveMemory(sizeof(layerConfig_t));
    layerCfg->linear = cfg;
    layer_t *layer = reserveMemory(sizeof(layer_t));
    initLayer(layer, LINEAR, layerCfg);
    return layer;
}

/*! Frees only the layer_t + layerConfig_t + linearConfig_t shells — NOT the
 *  weight/bias parameters. Needed after freeOptim, which already frees
 *  every parameter it registered (freeLinearLayer would double-free them). */
static void freeLinearLayerShellOnly(layer_t *layer) {
    freeReservedMemory(layer->config->linear);
    freeReservedMemory(layer->config);
    freeReservedMemory(layer);
}

void testAdamWCreateOptimAllocatesTwoZeroMomentBuffersPerParameter(void) {
    quantization_t *layerQ = quantizationInitFloat();
    /* weights {1,32} + bias {32}: both multiples of 32. */
    size_t *wDims = reserveMemory(2 * sizeof(size_t));
    wDims[0] = 1;
    wDims[1] = 32;
    size_t *wOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, wOrder);
    shape_t *wShape = reserveMemory(sizeof(shape_t));
    setShape(wShape, wDims, 2, wOrder);
    tensor_t *wParam = initTensor(wShape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(wParam, adamw_step1_default_p0, 32);
    tensor_t *wGrad = gradInitFloat(wParam, NULL);
    tensorFillFromFloatBuffer(wGrad, adamw_step1_default_g, 32);
    parameter_t *weights = parameterInit(wParam, wGrad);

    size_t *bDims = reserveMemory(1 * sizeof(size_t));
    bDims[0] = 32;
    size_t *bOrder = reserveMemory(1 * sizeof(size_t));
    setOrderOfDimsForNewTensor(1, bOrder);
    shape_t *bShape = reserveMemory(sizeof(shape_t));
    setShape(bShape, bDims, 1, bOrder);
    tensor_t *bParam = initTensor(bShape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(bParam, adamw_step1_default_p0, 32);
    tensor_t *bGrad = gradInitFloat(bParam, NULL);
    tensorFillFromFloatBuffer(bGrad, adamw_step1_default_g, 32);
    parameter_t *bias = parameterInit(bParam, bGrad);

    layer_t *linear = buildBorrowedLinearLayer(weights, bias, layerQ);
    layer_t *model[] = {linear};

    quantization_t *momentQ = quantizationInitFloat();
    optimizer_t *optim =
        adamWCreateOptim(0.001f, 0.9, 0.999, 1e-8, 0.01, model, 1, momentQ,
                         (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});

    /* capture */
    size_t capturedSizeStates = optim->sizeStates;
    optimizerType_t capturedType = optim->type;
    size_t capturedStatesPer0 = optim->states[0]->statesPerParameter;
    size_t capturedStatesPer1 = optim->states[1]->statesPerParameter;
    bool buffersDistinct = optim->states[0]->stateBuffers[0] != optim->states[0]->stateBuffers[1];
    qtype_t capturedM0Type = optim->states[0]->stateBuffers[0]->quantization->type;
    qtype_t capturedV0Type = optim->states[0]->stateBuffers[1]->quantization->type;
    bool momentQuantCloned = optim->states[0]->stateBuffers[0]->quantization != momentQ &&
                             optim->states[0]->stateBuffers[1]->quantization != momentQ;
    float mSum = 0.f, vSum = 0.f; /* zero-init check over buffer 0 of param 0 */
    const float *m0 = (const float *)optim->states[0]->stateBuffers[0]->data;
    const float *v0 = (const float *)optim->states[0]->stateBuffers[1]->data;
    for (size_t i = 0; i < 32; i++) {
        mSum += fabsf(m0[i]);
        vSum += fabsf(v0[i]);
    }

    /* teardown (freeOptim owns params + states; layer shell freed separately;
     * momentQ template stays caller-owned) */
    freeOptim(optim);
    freeLinearLayerShellOnly(linear);
    freeQuantization(momentQ);
    freeQuantization(layerQ);

    /* assert */
    TEST_ASSERT_EQUAL_size_t(2, capturedSizeStates);
    TEST_ASSERT_EQUAL_INT(ADAM_W, capturedType);
    TEST_ASSERT_EQUAL_size_t(2, capturedStatesPer0);
    TEST_ASSERT_EQUAL_size_t(2, capturedStatesPer1);
    TEST_ASSERT_TRUE(buffersDistinct);
    TEST_ASSERT_EQUAL_INT(FLOAT32, capturedM0Type);
    TEST_ASSERT_EQUAL_INT(FLOAT32, capturedV0Type);
    TEST_ASSERT_TRUE(momentQuantCloned);
    TEST_ASSERT_EQUAL_FLOAT(0.f, mSum);
    TEST_ASSERT_EQUAL_FLOAT(0.f, vSum);
}

void testAdamWCreateOptimStepMatchesHandAssembledGold(void) {
    /* Factory-built optimizer over one bias-less borrowed linear layer with
     * weights {1,32} filled from adamw_step1_default_p0 and grad from
     * adamw_step1_default_g -> one adamWStep -> param/m/v bits must equal
     * the step-1 default gold (proves factory wiring feeds the same kernels
     * as the hand-assembled harness; kills an m/v buffer-index swap: [0]
     * seeded as m by lerp semantics). */
    quantization_t *layerQ = quantizationInitFloat();

    size_t *wDims = reserveMemory(2 * sizeof(size_t));
    wDims[0] = 1;
    wDims[1] = 32;
    size_t *wOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, wOrder);
    shape_t *wShape = reserveMemory(sizeof(shape_t));
    setShape(wShape, wDims, 2, wOrder);
    tensor_t *wParam = initTensor(wShape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(wParam, adamw_step1_default_p0, 32);
    tensor_t *wGrad = gradInitFloat(wParam, NULL);
    tensorFillFromFloatBuffer(wGrad, adamw_step1_default_g, 32);
    parameter_t *weights = parameterInit(wParam, wGrad);

    layer_t *linear = buildBorrowedLinearLayer(weights, NULL, layerQ);
    layer_t *model[] = {linear};

    quantization_t *momentQ = quantizationInitFloat();
    optimizer_t *optim =
        adamWCreateOptim(0.001f, 0.9, 0.999, 1e-8, 0.01, model, 1, momentQ,
                         (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});

    optimizerFunctions[optim->type].step(optim);

    /* capture */
    float pOut[32], mOut[32], vOut[32];
    memcpy(pOut, weights->param->data, sizeof pOut);
    memcpy(mOut, optim->states[0]->stateBuffers[0]->data, sizeof mOut);
    memcpy(vOut, optim->states[0]->stateBuffers[1]->data, sizeof vOut);

    /* free */
    freeOptim(optim);
    freeLinearLayerShellOnly(linear);
    freeQuantization(momentQ);
    freeQuantization(layerQ);

    /* assert */
    tensor_t pT = {.data = (uint8_t *)pOut}, mT = {.data = (uint8_t *)mOut},
             vT = {.data = (uint8_t *)vOut};
    assertBitsEqualAt(adamw_step1_default_p1, &pT, 32, "factory step1 default p");
    assertBitsEqualAt(adamw_step1_default_m1, &mT, 32, "factory step1 default m");
    assertBitsEqualAt(adamw_step1_default_v1, &vT, 32, "factory step1 default v");
}

void testAdamWCreateOptimSymMomentSmoke(void) {
    /* Pins the funnel's staged (non-aliased) path for quantized moment
     * storage -- momentQuant = SYM@12 forces op[0]/rawOut/target into three
     * distinct buffers, exercising seedRawOutFromFirstOperand's memcpy arm,
     * which a FLOAT32 momentQuant (aliased fast path) never reaches. No
     * bit-parity claim: SYM moment divergence is by-design (#279 dead-zone
     * semantics apply to moments exactly as to params). Two independent
     * factory-built optimizers over the SAME single-param fixture (32
     * elems, step-1 gold p0/g arrays reused as plain data -- not for bit
     * parity), one momentQuant=FLOAT32, one momentQuant=SYM@12; 2
     * adamWSteps each, grad refilled with the same values before step 2. */
    quantization_t *layerQ = quantizationInitFloat();

    /* FLOAT32-moment run. */
    size_t *wDimsF = reserveMemory(2 * sizeof(size_t));
    wDimsF[0] = 1;
    wDimsF[1] = 32;
    size_t *wOrderF = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, wOrderF);
    shape_t *wShapeF = reserveMemory(sizeof(shape_t));
    setShape(wShapeF, wDimsF, 2, wOrderF);
    tensor_t *wParamF = initTensor(wShapeF, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(wParamF, adamw_step1_default_p0, 32);
    tensor_t *wGradF = gradInitFloat(wParamF, NULL);
    tensorFillFromFloatBuffer(wGradF, adamw_step1_default_g, 32);
    parameter_t *weightsF = parameterInit(wParamF, wGradF);
    layer_t *linearF = buildBorrowedLinearLayer(weightsF, NULL, layerQ);
    layer_t *modelF[] = {linearF};
    quantization_t *momentQF = quantizationInitFloat();
    optimizer_t *optimF =
        adamWCreateOptim(0.001f, 0.9, 0.999, 1e-8, 0.01, modelF, 1, momentQF,
                         (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});

    /* SYM@12-moment run: separate parameters (freeOptim frees each
     * optimizer's own registered parameters -- sharing across two
     * optimizers would double-free). */
    size_t *wDimsS = reserveMemory(2 * sizeof(size_t));
    wDimsS[0] = 1;
    wDimsS[1] = 32;
    size_t *wOrderS = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, wOrderS);
    shape_t *wShapeS = reserveMemory(sizeof(shape_t));
    setShape(wShapeS, wDimsS, 2, wOrderS);
    tensor_t *wParamS = initTensor(wShapeS, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(wParamS, adamw_step1_default_p0, 32);
    tensor_t *wGradS = gradInitFloat(wParamS, NULL);
    tensorFillFromFloatBuffer(wGradS, adamw_step1_default_g, 32);
    parameter_t *weightsS = parameterInit(wParamS, wGradS);
    layer_t *linearS = buildBorrowedLinearLayer(weightsS, NULL, layerQ);
    layer_t *modelS[] = {linearS};
    quantization_t *momentQS = quantizationInitSym(12, HALF_AWAY);
    optimizer_t *optimS =
        adamWCreateOptim(0.001f, 0.9, 0.999, 1e-8, 0.01, modelS, 1, momentQS,
                         (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});

    /* Step 1 (both runs). */
    optimizerFunctions[optimF->type].step(optimF);
    optimizerFunctions[optimS->type].step(optimS);

    tensor_t *mSDecoded = makeFloatTensor1D(NULL, 32);
    tensor_t *vSDecoded = makeFloatTensor1D(NULL, 32);
    executeConvert(optimS->states[0]->stateBuffers[0], mSDecoded);
    executeConvert(optimS->states[0]->stateBuffers[1], vSDecoded);

    float mF1Sum = 0.f, vF1Sum = 0.f, mS1Sum = 0.f, vS1Sum = 0.f;
    const float *mF1 = (const float *)optimF->states[0]->stateBuffers[0]->data;
    const float *vF1 = (const float *)optimF->states[0]->stateBuffers[1]->data;
    const float *mS1 = (const float *)mSDecoded->data;
    const float *vS1 = (const float *)vSDecoded->data;
    for (size_t i = 0; i < 32; i++) {
        mF1Sum += fabsf(mF1[i]);
        vF1Sum += fabsf(vF1[i]);
        mS1Sum += fabsf(mS1[i]);
        vS1Sum += fabsf(vS1[i]);
    }

    /* Step 2 (both runs), grad refilled with the same values. */
    tensorFillFromFloatBuffer(weightsF->grad, adamw_step1_default_g, 32);
    tensorFillFromFloatBuffer(weightsS->grad, adamw_step1_default_g, 32);
    optimizerFunctions[optimF->type].step(optimF);
    optimizerFunctions[optimS->type].step(optimS);

    executeConvert(optimS->states[0]->stateBuffers[0], mSDecoded);
    executeConvert(optimS->states[0]->stateBuffers[1], vSDecoded);

    float mFFinal[32], vFFinal[32], mSFinal[32], vSFinal[32], pSFinal[32];
    memcpy(mFFinal, optimF->states[0]->stateBuffers[0]->data, sizeof mFFinal);
    memcpy(vFFinal, optimF->states[0]->stateBuffers[1]->data, sizeof vFFinal);
    memcpy(mSFinal, mSDecoded->data, sizeof mSFinal);
    memcpy(vSFinal, vSDecoded->data, sizeof vSFinal);
    memcpy(pSFinal, weightsS->param->data, sizeof pSFinal);
    float finalScale =
        ((symQConfig_t *)optimS->states[0]->stateBuffers[0]->quantization->qConfig)->scale;

    int allFinite = 1;
    for (size_t i = 0; i < 32; i++) {
        if (!isfinite(mSFinal[i]) || !isfinite(vSFinal[i])) {
            allFinite = 0;
        }
    }

    float maxAbsDiffM = 0.f;
    for (size_t i = 0; i < 32; i++) {
        float d = fabsf(mSFinal[i] - mFFinal[i]);
        if (d > maxAbsDiffM) {
            maxAbsDiffM = d;
        }
    }

    int paramMovedS = 0;
    for (size_t i = 0; i < 32; i++) {
        if (pSFinal[i] != adamw_step1_default_p0[i]) {
            paramMovedS = 1;
        }
    }

    /* free */
    freeTensor(mSDecoded);
    freeTensor(vSDecoded);
    freeOptim(optimF);
    freeOptim(optimS);
    freeLinearLayerShellOnly(linearF);
    freeLinearLayerShellOnly(linearS);
    freeQuantization(momentQF);
    freeQuantization(momentQS);
    freeQuantization(layerQ);

    /* assert */
    TEST_ASSERT_TRUE_MESSAGE(mF1Sum > 0.f, "FLOAT32 m must be non-zero after step 1");
    TEST_ASSERT_TRUE_MESSAGE(vF1Sum > 0.f, "FLOAT32 v must be non-zero after step 1");
    TEST_ASSERT_TRUE_MESSAGE(mS1Sum > 0.f, "SYM-decoded m must be non-zero after step 1");
    TEST_ASSERT_TRUE_MESSAGE(vS1Sum > 0.f, "SYM-decoded v must be non-zero after step 1");
    TEST_ASSERT_TRUE_MESSAGE(allFinite, "SYM-decoded m/v must be finite");
    TEST_ASSERT_TRUE_MESSAGE(maxAbsDiffM <= 2.f * finalScale,
                             "SYM-decoded m must be within 2*finalScale of the FLOAT32 run's m");
    TEST_ASSERT_TRUE_MESSAGE(paramMovedS, "SYM-momentum run's param must have moved from p0");
}

void testAdamWCreateOptimRejectsInt32GradStorage(void) {
    /* validateOptimizerGradStorage wiring: the weight grad tensor carries a
     * raw-stack quantization_t {.type = INT32} (UnitTestMatmul.c/
     * UnitTestAdd.c idiom) -- an unsupported grad dtype the factory must
     * reject before ever touching grad->data. The grad tensor and its
     * quantization/shape backing are entirely stack-local (no *Init* call),
     * so nothing but the Rule-1 weight tensor needs freeing afterward. */
    quantization_t *layerQ = quantizationInitFloat();

    size_t *wDims = reserveMemory(2 * sizeof(size_t));
    wDims[0] = 1;
    wDims[1] = 32;
    size_t *wOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, wOrder);
    shape_t *wShape = reserveMemory(sizeof(shape_t));
    setShape(wShape, wDims, 2, wOrder);
    tensor_t *wParam = initTensor(wShape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(wParam, adamw_step1_default_p0, 32);

    quantization_t int32GradQ = {.type = INT32, .qConfig = NULL};
    size_t gDims[2] = {1, 32};
    size_t gOrder[2];
    setOrderOfDimsForNewTensor(2, gOrder);
    shape_t gShape;
    setShape(&gShape, gDims, 2, gOrder);
    tensor_t gradTensor = {
        .data = NULL, .shape = &gShape, .quantization = &int32GradQ, .sparsity = NULL};
    parameter_t weightsParam = {.param = wParam, .grad = &gradTensor};

    layer_t *linear = buildBorrowedLinearLayer(&weightsParam, NULL, layerQ);
    layer_t *model[] = {linear};

    quantization_t *momentQ = quantizationInitFloat();

    ASSERT_EXITS_WITH_FAILURE(
        adamWCreateOptim(0.001f, 0.9, 0.999, 1e-8, 0.01, model, 1, momentQ,
                         (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY}));

    /* Teardown (parent continues after the fork-based assert; file
     * convention). weightsParam/gradTensor/int32GradQ/gShape are
     * stack-local: only wParam (Rule-1 heap tensor) needs freeing. */
    freeTensor(wParam);
    freeLinearLayerShellOnly(linear);
    freeQuantization(momentQ);
    freeQuantization(layerQ);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testAdamWInitStoresDoubleHyperparamsAndZeroStepCount);
    RUN_TEST(testAdamWGetSetLrRoundTripThroughImpl);
    RUN_TEST(testAdamWInitRejectsNonFloat32UpdateMath);
    RUN_TEST(testAdamWInitRejectsBeta1AtOne);
    RUN_TEST(testAdamWInitRejectsNegativeBeta2);
    RUN_TEST(testAdamWInitRejectsNanBeta1);
    RUN_TEST(testAdamWInitRejectsZeroEps);
    RUN_TEST(testAdamWStepMatchesGoldStep1Defaults);
    RUN_TEST(testAdamWStepMatchesGoldStep1WdZero);
    RUN_TEST(testAdamWStepMatchesGoldStep1OrderDiscrim);
    RUN_TEST(testAdamWVtableRowIsFullyPopulated);
    RUN_TEST(testAdamWStepIncrementsStepCountPerCall);
    RUN_TEST(testAdamWStepRejectsTamperedUpdateMath);
    RUN_TEST(testAdamWTrajectoryMatchesGoldEveryStep);
    RUN_TEST(testAdamWTrajectoryWdZeroMatchesGold);
    RUN_TEST(testAdamWWithCosineSchedulerMatchesGold);
    RUN_TEST(testAdamWCreateOptimAllocatesTwoZeroMomentBuffersPerParameter);
    RUN_TEST(testAdamWCreateOptimStepMatchesHandAssembledGold);
    RUN_TEST(testAdamWCreateOptimSymMomentSmoke);
    RUN_TEST(testAdamWCreateOptimRejectsInt32GradStorage);
    return UNITY_END();
}
