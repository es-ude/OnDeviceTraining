#define SOURCE_FILE "ADAM-W-UTEST"

#include <math.h>
#include <stdint.h>
#include <string.h>

#include "AdamW.h"
#include "AdamWApi.h"
#include "ArithmeticType.h"
#include "BorrowedLayer.h"
#include "DeathTest.h"
#include "ExecuteOp.h"
#include "LayerQuant.h"
#include "Linear.h"
#include "LinearApi.h"
#include "LrScheduler.h"
#include "Optimizer.h"
#include "OptimizerApi.h"
#include "QuantizationApi.h"
#include "RNG.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "TensorConversion.h"
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
    /* #279: this smoke pins the DETERMINISTIC HALF_AWAY moment quantization
     * (the 2*scale bounds were margin-audited for it, PR #367) -- opt out of
     * the factory's seeded-SR training default. */
    optimizerSetWriteBackRounding(optimS, HALF_AWAY);

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
    float finalMScale =
        ((symQConfig_t *)optimS->states[0]->stateBuffers[0]->quantization->qConfig)->scales[0];
    float finalVScale =
        ((symQConfig_t *)optimS->states[0]->stateBuffers[1]->quantization->qConfig)->scales[0];

    int allFinite = 1;
    for (size_t i = 0; i < 32; i++) {
        if (!isfinite(mSFinal[i]) || !isfinite(vSFinal[i])) {
            allFinite = 0;
        }
    }

    float maxAbsDiffM = 0.f;
    float maxAbsDiffV = 0.f;
    for (size_t i = 0; i < 32; i++) {
        float d = fabsf(mSFinal[i] - mFFinal[i]);
        if (d > maxAbsDiffM) {
            maxAbsDiffM = d;
        }
        d = fabsf(vSFinal[i] - vFFinal[i]);
        if (d > maxAbsDiffV) {
            maxAbsDiffV = d;
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
    TEST_ASSERT_TRUE_MESSAGE(maxAbsDiffM <= 2.f * finalMScale,
                             "SYM-decoded m must be within 2*finalMScale of the FLOAT32 run's m");
    TEST_ASSERT_TRUE_MESSAGE(maxAbsDiffV <= 2.f * finalVScale,
                             "SYM-decoded v must be within 2*finalVScale of the FLOAT32 run's v");
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

void testAdamWCreateOptimDefaultsWriteBackRoundingToSr(void) {
    /* #279 ratified default, AdamW side: same param-storage write-back seam
     * as SGD, same silent non-learning footgun without it. */
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
    parameter_t *weights = parameterInit(wParam, wGrad);

    layer_t *linear = buildBorrowedLinearLayer(weights, NULL, layerQ);
    layer_t *model[] = {linear};

    quantization_t *momentQ = quantizationInitFloat();
    optimizer_t *optim =
        adamWCreateOptim(0.001f, 0.9, 0.999, 1e-8, 0.01, model, 1, momentQ,
                         (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});

    /* CAPTURE -> free -> assert. */
    roundingMode_t capturedDefault = optim->writeBackRounding;

    freeOptim(optim);
    freeLinearLayerShellOnly(linear);
    freeQuantization(momentQ);
    freeQuantization(layerQ);

    TEST_ASSERT_EQUAL_INT_MESSAGE(SR_HALF_AWAY, capturedDefault,
                                  "#279: adamWCreateOptim must default writeBackRounding "
                                  "to seeded SR_HALF_AWAY");
}

void testAdamWStepOptimizerSrWriteBackEscapesSymDeadZone(void) {
    /* #279: AdamW's param write-back (K3's OUT_WRITE epilogue requant) is the
     * same training write-back seam as SGD's -- the optimizer's SR_HALF_AWAY
     * must beat the param's own HALF_AWAY qConfig here too. With a constant
     * grad the bias-corrected update magnitude is ~lr (mhat/sqrt(vhat) ~ 1),
     * so lr = 0.25*scale is a persistent sub-ULP step: deterministic rounding
     * freezes the code forever, SR escapes with per-step probability ~0.25.
     * Anchor element (grad 0 -> m = v = 0 -> delta 0 at wd = 0) pins the
     * re-derived scale, same construction as UnitTestSgd's dead-zone
     * fixture. */
    rngSetSeed(13579u);

    size_t *pDims = reserveMemory(1 * sizeof(size_t));
    pDims[0] = 2;
    size_t *pOrder = reserveMemory(1 * sizeof(size_t));
    setOrderOfDimsForNewTensor(1, pOrder);
    shape_t *pShape = reserveMemory(sizeof(shape_t));
    setShape(pShape, pDims, 1, pOrder);
    tensor_t *p = initTensor(pShape, quantizationInitSymInt32(HALF_AWAY), NULL);
    tensorFillFromFloatBuffer(p, (float[]){100.f, 0.f}, 2);

    const float qMax = 2047.f; /* 2^(ODT_SYM_OPERAND_QMAXBITS-1) - 1 */
    const float scale = 100.f / qMax;
    const float lr = 0.25f * scale;

    tensor_t *g = makeFloatTensor1D((float[]){0.f, 1.f}, 2);
    parameter_t *par = parameterInit(p, g);

    int32_t initialTargetCode = ((int32_t *)p->data)[1];
    TEST_ASSERT_EQUAL_INT(0, initialTargetCode); /* fixture guard */

    adamW_t adamW;
    adamWInit(&adamW, lr, 0.9, 0.999, 1e-8, 0.0,
              (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});
    optimImpl_t impl = {.adamW = &adamW};
    tensor_t *m = makeFloatTensor1D(NULL, 2);
    tensor_t *v = makeFloatTensor1D(NULL, 2);
    tensor_t *stateBuffers[2] = {m, v};
    states_t st = {.stateBuffers = stateBuffers, .statesPerParameter = 2};
    parameter_t *parArr[1] = {par};
    states_t *stArr[1] = {&st};
    optimizer_t optim = {.type = ADAM_W,
                         .impl = &impl,
                         .parameter = parArr,
                         .states = stArr,
                         .sizeStates = 1,
                         .writeBackRounding = SR_HALF_AWAY};

    int codeEverMoved = 0;
    for (int i = 0; i < 500; i++) {
        adamWStep(&optim);
        if (((int32_t *)p->data)[1] != initialTargetCode) {
            codeEverMoved = 1;
        }
    }
    roundingMode_t storageModeAfter = ((symInt32QConfig_t *)p->quantization->qConfig)->roundingMode;

    freeTensor(m);
    freeTensor(v);
    freeParameter(par);

    TEST_ASSERT_TRUE_MESSAGE(codeEverMoved, "#279: AdamW param write-back must honor the "
                                            "optimizer's SR_HALF_AWAY and escape the dead-zone");
    TEST_ASSERT_EQUAL_INT_MESSAGE(HALF_AWAY, storageModeAfter,
                                  "AdamW write-back swap must restore the param's storage mode");
}

/* Three hand-assembled AdamW steps over packed-SYM@12 m/v moments with the
 * given optimizer write-back rounding; decodes both moments into the caller's
 * float buffers. Fixed p/g so two calls differ ONLY in the rounding mode; 3
 * steps x 6 elements gives 18 SR draws per moment over step-evolving requant
 * fractions, so an SR run cannot stay bit-identical to the deterministic
 * one by fraction luck (a single step can: seed-dependent ~15%). */
static void adamWStepWithSymMoments(roundingMode_t writeBackRounding, float *mOut, float *vOut) {
    const float p0[6] = {0.5f, -0.25f, 0.125f, 0.75f, -0.5f, 0.3f};
    const float g0[6] = {0.11f, 0.23f, -0.37f, 0.41f, -0.53f, 0.61f};

    adamW_t adamW;
    adamWInit(&adamW, 0.001f, 0.9, 0.999, 1e-8, 0.0,
              (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});
    optimImpl_t impl = {.adamW = &adamW};
    tensor_t *p = makeFloatTensor1D(p0, 6);
    tensor_t *g = makeFloatTensor1D(g0, 6);
    parameter_t *par = parameterInit(p, g);
    tensor_t *m = initTensor(getShapeLike(p->shape), quantizationInitSym(12, HALF_AWAY), NULL);
    tensor_t *v = initTensor(getShapeLike(p->shape), quantizationInitSym(12, HALF_AWAY), NULL);
    tensor_t *stateBuffers[2] = {m, v};
    states_t st = {.stateBuffers = stateBuffers, .statesPerParameter = 2};
    parameter_t *parArr[1] = {par};
    states_t *stArr[1] = {&st};
    optimizer_t optim = {.type = ADAM_W,
                         .impl = &impl,
                         .parameter = parArr,
                         .states = stArr,
                         .sizeStates = 1,
                         .writeBackRounding = writeBackRounding};

    adamWStep(&optim);
    adamWStep(&optim);
    adamWStep(&optim);

    tensor_t *mDecoded = makeFloatTensor1D(NULL, 6);
    tensor_t *vDecoded = makeFloatTensor1D(NULL, 6);
    executeConvert(m, mDecoded);
    executeConvert(v, vDecoded);
    memcpy(mOut, mDecoded->data, 6 * sizeof(float));
    memcpy(vOut, vDecoded->data, 6 * sizeof(float));

    freeTensor(mDecoded);
    freeTensor(vDecoded);
    freeTensor(m);
    freeTensor(v);
    freeParameter(par);
}

void testAdamWMomentWriteBacksHonorOptimizerSrRounding(void) {
    /* #279 (params + states): the m/v moment write-backs (K1/K2 OUT_WRITE
     * requants into packed-SYM storage) must also run under the optimizer's
     * writeBackRounding. Two otherwise-identical single steps -- one SR, one
     * deterministic -- must decode to different m AND different v: the m/v
     * requant fractions are generic values, so seeded-SR jitter almost surely
     * moves at least one of the 6 codes per moment (and with this seed,
     * reproducibly does). If either kernel's write-back ignored the optimizer
     * mode, its decoded moments would be bit-identical across the two runs. */
    rngSetSeed(11111u);
    float mSr[6], vSr[6];
    adamWStepWithSymMoments(SR_HALF_AWAY, mSr, vSr);

    float mDet[6], vDet[6];
    adamWStepWithSymMoments(HALF_AWAY, mDet, vDet);

    TEST_ASSERT_TRUE_MESSAGE(memcmp(mSr, mDet, sizeof mSr) != 0,
                             "#279: the m write-back must honor the optimizer's SR rounding "
                             "(decoded m identical to the deterministic run)");
    TEST_ASSERT_TRUE_MESSAGE(memcmp(vSr, vDet, sizeof vSr) != 0,
                             "#279: the v write-back must honor the optimizer's SR rounding "
                             "(decoded v identical to the deterministic run)");
}

/* Local copy of the Task-1 helper (test/unit/layer/UnitTestLinear.c:355) --
 * tests are independent binaries, so this file builds its own frozen/
 * trainable Linear layer instead of including across test directories. */
static layer_t *buildFloatLinearWithTrainable(trainable_t trainable) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);
    layer_t *layer = linearLayerInitOwning(
        &(linearInit_t){.inFeatures = 3, .outFeatures = 2, .trainable = trainable}, &lq);
    freeQuantization(q);
    return layer;
}

/* #380 PR1 Task 4: AdamW twin of the SGD count/collection test -- frozen
 * layers must contribute zero states and never be collected. */
void testAdamWOptimizerSkipsFrozenLayerInCountAndCollection(void) {
    layer_t *frozenL = buildFloatLinearWithTrainable(TRAINABLE_FALSE);
    layer_t *trainL = buildFloatLinearWithTrainable(TRAINABLE_DEFAULT);
    layer_t *model[] = {frozenL, trainL};
    size_t count = calcTotalNumberOfStates(model, 2);

    quantization_t *momentQ = quantizationInitFloat();
    optimizer_t *optim =
        adamWCreateOptim(0.001f, 0.9, 0.999, 1e-8, 0.01, model, 2, momentQ,
                         (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});
    size_t sizeStates = optim->sizeStates;
    bool slot0IsTrainWeights = optim->parameter[0] == trainL->config->linear->weights;
    bool slot1IsTrainBias = optim->parameter[1] == trainL->config->linear->bias;

    freeOptim(optim);                 /* frees ONLY the collected (trainable) params */
    freeLinearLayerShellOnly(trainL); /* BorrowedLayer.h helper — params already freed */
    freeLinearLayer(frozenL);         /* frozen params NOT collected — full free */
    freeQuantization(momentQ);

    TEST_ASSERT_EQUAL_size_t(2, count);
    TEST_ASSERT_EQUAL_size_t(2, sizeStates);
    TEST_ASSERT_TRUE(slot0IsTrainWeights);
    TEST_ASSERT_TRUE(slot1IsTrainBias);
}

/* #380 PR1 Task 4: AdamW twin of the SGD all-frozen death test. */
void testAdamWCreateAllFrozenModelExits(void) {
    ASSERT_EXITS_WITH(1, {
        layer_t *frozenL = buildFloatLinearWithTrainable(TRAINABLE_FALSE);
        layer_t *model[] = {frozenL};
        quantization_t *momentQ = quantizationInitFloat();
        adamWCreateOptim(0.001f, 0.9, 0.999, 1e-8, 0.01, model, 1, momentQ,
                         (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});
    });
}

/* Final-review Fix 3(c): AdamW twin of the SGD momentum-carrier-gate death
 * test (UnitTestSgd.c's testSgdCreateGroupedSymMomentumQuantExits) --
 * momentStateInit (AdamWApi.c) builds BOTH moment buffers via
 * getQLike(momentQuant) with no carrier gate of its own, mirroring SgdApi's
 * momentumStateInit exactly (#277 precedent noted in both files' comments).
 * A grouped SYM momentQuant template must fail-fast just like gradInit's
 * gate (TensorApi.c) -- grouped moments are a future #300 axis, not
 * something PR2's optimizer paths support. The weight tensor's OWN element
 * count (2*4=8) deliberately EQUALS the momentQuant template's
 * numGroups*groupSize (2*4=8), so the death observed here cannot be the
 * UNRELATED, pre-existing initTensor/validateSymQConfigShape guard
 * (TensorApi.c) tripping on a coincidental shape mismatch instead of the new
 * gate under test (mutation-vacuity guard, same reasoning as the SGD twin).
 * bias=NULL keeps the model to exactly one trainable state. */
void testAdamWCreateGroupedSymMomentQuantExits(void) {
    ASSERT_EXITS_WITH(1, {
        quantization_t *layerQ = quantizationInitFloat();
        size_t *wDims = reserveMemory(2 * sizeof(size_t));
        wDims[0] = 2;
        wDims[1] = 4;
        size_t *wOrder = reserveMemory(2 * sizeof(size_t));
        setOrderOfDimsForNewTensor(2, wOrder);
        shape_t *wShape = reserveMemory(sizeof(shape_t));
        setShape(wShape, wDims, 2, wOrder);
        tensor_t *wParam = initTensor(wShape, quantizationInitFloat(), NULL);
        tensorFillFromFloatBuffer(wParam, (float[]){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f}, 8);
        tensor_t *wGrad = gradInitFloat(wParam, NULL);
        parameter_t *weights = parameterInit(wParam, wGrad);

        layer_t *linear = buildBorrowedLinearLayer(weights, NULL, layerQ);
        layer_t *model[] = {linear};

        quantization_t *momentQ = quantizationInitSymGrouped(4, HALF_AWAY, 2, 4);
        adamWCreateOptim(0.001f, 0.9, 0.999, 1e-8, 0.01, model, 1, momentQ,
                         (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});
    });
}

/* Group-quant PR4 (Task 3): ASYM twin of the gate test above -- a grouped
 * ASYM momentQuant template must fail-fast identically (getQLike's ASYM arm
 * deep-clones grouped grids, scales AND zeroPoints, so without the gate the
 * grouped template would silently become a grouped moment buffer). Same
 * shape-coincidence discipline (weight element count == numGroups*groupSize
 * == 8) as the SYM twin. */
void testAdamWCreateGroupedAsymMomentQuantExits(void) {
    ASSERT_EXITS_WITH(1, {
        quantization_t *layerQ = quantizationInitFloat();
        size_t *wDims = reserveMemory(2 * sizeof(size_t));
        wDims[0] = 2;
        wDims[1] = 4;
        size_t *wOrder = reserveMemory(2 * sizeof(size_t));
        setOrderOfDimsForNewTensor(2, wOrder);
        shape_t *wShape = reserveMemory(sizeof(shape_t));
        setShape(wShape, wDims, 2, wOrder);
        tensor_t *wParam = initTensor(wShape, quantizationInitFloat(), NULL);
        tensorFillFromFloatBuffer(wParam, (float[]){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f}, 8);
        tensor_t *wGrad = gradInitFloat(wParam, NULL);
        parameter_t *weights = parameterInit(wParam, wGrad);

        layer_t *linear = buildBorrowedLinearLayer(weights, NULL, layerQ);
        layer_t *model[] = {linear};

        quantization_t *momentQ = quantizationInitAsymGrouped(4, HALF_AWAY, 2, 4);
        adamWCreateOptim(0.001f, 0.9, 0.999, 1e-8, 0.01, model, 1, momentQ,
                         (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});
    });
}

/* BFP epic PR3 Task 7: the PR1 unconditional carrier gate lifts --
 * momentStateInit now admits a PER-TENSOR BFP moment template (SgdApi
 * momentumStateInit twin; grads and states share the per-tensor-only rule,
 * #300 axis). BOTH moment buffers (m and v) must be getQLike clones at
 * zero-state: BFP-typed, per-tensor, exponent at bias (e=8 -> 127),
 * all-zero packed codes. */
void testAdamWCreateAdmitsPerTensorBfpMoments(void) {
    quantization_t *layerQ = quantizationInitFloat();
    size_t *wDims = reserveMemory(2 * sizeof(size_t));
    wDims[0] = 2;
    wDims[1] = 4;
    size_t *wOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, wOrder);
    shape_t *wShape = reserveMemory(sizeof(shape_t));
    setShape(wShape, wDims, 2, wOrder);
    tensor_t *wParam = initTensor(wShape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(wParam, (float[]){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f}, 8);
    tensor_t *wGrad = gradInitFloat(wParam, NULL);
    parameter_t *weights = parameterInit(wParam, wGrad);

    layer_t *linear = buildBorrowedLinearLayer(weights, NULL, layerQ);
    layer_t *model[] = {linear};

    quantization_t *momentQ = quantizationInitBfp(8, 8, HALF_AWAY);
    optimizer_t *optim =
        adamWCreateOptim(0.001f, 0.9, 0.999, 1e-8, 0.01, model, 1, momentQ,
                         (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});

    /* CAPTURE -> free -> assert (file convention). */
    qtype_t momentTypes[2];
    size_t momentNumGroups[2];
    uint8_t momentExps[2];
    bool codesAllZero = true;
    for (size_t b = 0; b < 2; b++) {
        tensor_t *state = optim->states[0]->stateBuffers[b];
        momentTypes[b] = state->quantization->type;
        bfpQConfig_t *stateQC = state->quantization->qConfig;
        momentNumGroups[b] = stateQC->numGroups;
        momentExps[b] = stateQC->exponents[0];
        for (size_t i = 0; i < 8; i++) { /* 8 elements x 8-bit mantissas = 8 packed bytes */
            if (((uint8_t *)state->data)[i] != 0) {
                codesAllZero = false;
            }
        }
    }
    freeOptim(optim);
    freeLinearLayerShellOnly(linear);
    freeQuantization(momentQ);
    freeQuantization(layerQ);

    for (size_t b = 0; b < 2; b++) {
        TEST_ASSERT_EQUAL_INT(BFP, momentTypes[b]);
        TEST_ASSERT_EQUAL_size_t(1, momentNumGroups[b]);
        TEST_ASSERT_EQUAL_UINT8(127, momentExps[b]); /* bias = 2^(8-1)-1: zero-state */
    }
    TEST_ASSERT_TRUE_MESSAGE(codesAllZero,
                             "fresh BFP moment buffers must have all-zero packed codes");
}

/* Grouped-BFP twin of testAdamWCreateGroupedSymMomentQuantExits above: the
 * lifted gate keeps rejecting GROUPED templates (per-tensor only, #300
 * axis). Same shape-coincidence discipline as the SYM twin: weight element
 * count (2*4=8) EQUALS numGroups*groupSize (2*4=8), so the death cannot come
 * from the unrelated initTensor/validateBfpQConfigShape guard. */
void testAdamWCreateRejectsGroupedBfpMoments(void) {
    ASSERT_EXITS_WITH(1, {
        quantization_t *layerQ = quantizationInitFloat();
        size_t *wDims = reserveMemory(2 * sizeof(size_t));
        wDims[0] = 2;
        wDims[1] = 4;
        size_t *wOrder = reserveMemory(2 * sizeof(size_t));
        setOrderOfDimsForNewTensor(2, wOrder);
        shape_t *wShape = reserveMemory(sizeof(shape_t));
        setShape(wShape, wDims, 2, wOrder);
        tensor_t *wParam = initTensor(wShape, quantizationInitFloat(), NULL);
        tensorFillFromFloatBuffer(wParam, (float[]){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f}, 8);
        tensor_t *wGrad = gradInitFloat(wParam, NULL);
        parameter_t *weights = parameterInit(wParam, wGrad);

        layer_t *linear = buildBorrowedLinearLayer(weights, NULL, layerQ);
        layer_t *model[] = {linear};

        quantization_t *momentQ = quantizationInitBfpGrouped(8, 8, HALF_AWAY, 2, 4);
        adamWCreateOptim(0.001f, 0.9, 0.999, 1e-8, 0.01, model, 1, momentQ,
                         (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});
    });
}

/* ---- Group-quant PR3 Task 4: AdamW updates a grouped-SYM param ----------
 *
 * Same funnel wiring as SGD (UnitTestSgd.c's *MatchesGold tests): the param
 * opSpec now declares groupedSymOperandPos, so the FLOAT32 prologue dequants
 * the grouped param per-group and the OUT_WRITE epilogue re-derives fresh
 * per-group absmax scales -- no new conversion code. m/v are per-tensor
 * FLOAT32 (the momentum-state carrier gate, PR2) and NEVER see the param's
 * grouping at all -- adamWMomentKernel/adamWVarianceKernel take {state,
 * grad} only, no param operand -- so with m0=v0=0 (fresh states) they
 * collapse to the trivial m1=w1*grad, v1=s2*grad^2 and are asserted with a
 * tight tolerance.
 *
 * GOLD CHOICE (disclosed, task-4-brief.md): the param path chains six
 * float32 roundings through a sqrt/div (adamWParamKernel ->
 * addcdivDenomFloat32TensorsInplace, PointwiseFused.c) on top of the
 * per-group dequant/requant. Reproducing that exact float32 rounding
 * sequence in an independent second implementation -- the bar the SGD gold
 * generator hits for SGD's much simpler single-mul-add kernels -- is
 * disproportionate scaffolding for one test's worth of coverage, so this
 * test takes the brief's documented minimum bar instead:
 *   (1) the two post-step group scales must be DISTINCT -- proves a real
 *       per-group requant happened (not a collapse onto one shared scale);
 *   (2) a DOUBLE-precision float reference of the param update, computed
 *       from the SAME formula/inputs (not hand-transcribed constants -- see
 *       the loop below, which mirrors AdamW.c:98-111's t=1 scalars and
 *       addcdivDenomFloat32TensorsInplace's exact op order: d=sqrt(v);
 *       d/=bc2sqrt; d+=eps; numer=stepScale*m; quotient=numer/d;
 *       out=decay*paramDeq+quotient), compared against the actual
 *       per-group-dequantized output within a tolerance DERIVED from that
 *       group's own re-derived scale (0.5*scale = the requant's own
 *       rounding-error bound) plus a small slack for the float32-vs-double
 *       arithmetic gap through the sqrt/div chain (negligible at this
 *       magnitude, included for rigor, not because it is expected to bite). */
void testAdamWStepGroupedSymParamRunsAndRequantsPerGroup(void) {
    const size_t n = 6, groupSize = 3, numGroups = 2;
    const uint8_t qBits = 8;
    const int32_t paramMantissas[6] = {50, -80, 20, -60, 90, -40};
    const float paramScales[2] = {0.02f, 0.015f};
    const float gradVals[6] = {0.2f, -0.1f, 0.3f, -0.2f, 0.1f, -0.3f};
    const float lr = 0.1f;
    const double beta1 = 0.5, beta2 = 0.75, eps = 0.01, wd = 0.1;

    size_t *pDims = reserveMemory(1 * sizeof(size_t));
    pDims[0] = n;
    size_t *pOrder = reserveMemory(1 * sizeof(size_t));
    setOrderOfDimsForNewTensor(1, pOrder);
    shape_t *pShape = reserveMemory(sizeof(shape_t));
    setShape(pShape, pDims, 1, pOrder);
    tensor_t *p = initTensor(
        pShape, quantizationInitSymGrouped(qBits, HALF_AWAY, numGroups, groupSize), NULL);
    byteConversion((uint8_t *)paramMantissas, 32, p->data, qBits, n);
    symQConfig_t *paramQC = p->quantization->qConfig;
    for (size_t grp = 0; grp < numGroups; grp++) {
        paramQC->scales[grp] = paramScales[grp];
    }
    tensor_t *g = gradInitFloat(p, NULL);
    tensorFillFromFloatBuffer(g, gradVals, n);
    parameter_t *par = parameterInit(p, g);

    adamW_t adamW;
    adamWInit(&adamW, lr, beta1, beta2, eps, wd,
              (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});
    optimImpl_t impl = {.adamW = &adamW};
    tensor_t *m = makeFloatTensor1D(NULL, n);
    tensor_t *v = makeFloatTensor1D(NULL, n);
    tensor_t *stateBuffers[2] = {m, v};
    states_t st = {.stateBuffers = stateBuffers, .statesPerParameter = 2};
    parameter_t *parArr[1] = {par};
    states_t *stArr[1] = {&st};
    optimizer_t optim = {.type = ADAM_W,
                         .impl = &impl,
                         .parameter = parArr,
                         .states = stArr,
                         .sizeStates = 1,
                         /* #279 explicit opt-out: deterministic write-back so
                          * the requant's rounding is reproducible. */
                         .writeBackRounding = HALF_AWAY};

    adamWStep(&optim);

    /* CAPTURE -> free -> assert (file convention). */
    int32_t mant[6];
    unpackSignExtend(p->data, qBits, 0, mant, n);
    float scale0 = ((symQConfig_t *)p->quantization->qConfig)->scales[0];
    float scale1 = ((symQConfig_t *)p->quantization->qConfig)->scales[1];
    float mOut[6], vOut[6];
    memcpy(mOut, m->data, sizeof mOut);
    memcpy(vOut, v->data, sizeof vOut);
    freeTensor(m);
    freeTensor(v);
    freeParameter(par);

    /* (1) per-group requant actually happened (not collapsed onto one scale). */
    TEST_ASSERT_TRUE_MESSAGE(scale0 != scale1,
                             "post-step group scales must differ -- a real per-group "
                             "requant, not an accidental single scale");

    /* m/v never see the param's grouping (no param operand in their
     * kernels); m0=v0=0 makes the exact formula trivial. */
    const double w1 = 1.0 - beta1, s2 = 1.0 - beta2;
    for (size_t i = 0; i < n; i++) {
        double expectedM = w1 * (double)gradVals[i];
        double expectedV = s2 * (double)gradVals[i] * (double)gradVals[i];
        TEST_ASSERT_FLOAT_WITHIN(1e-6f, (float)expectedM, mOut[i]);
        TEST_ASSERT_FLOAT_WITHIN(1e-6f, (float)expectedV, vOut[i]);
    }

    /* (2) double-precision float reference of the param update, from the
     * SAME formula/inputs -- t=1 so bc1==w1 and bc2sqrt==sqrt(1-beta2). */
    const double bc1 = 1.0 - beta1, bc2sqrt = sqrt(1.0 - beta2);
    const double stepScale = -((double)lr / bc1);
    const double decay = 1.0 - (double)lr * wd;
    double paramNew[6];
    for (size_t i = 0; i < n; i++) {
        size_t grp = i / groupSize;
        double paramDeq = (double)paramMantissas[i] * (double)paramScales[grp];
        double mI = w1 * (double)gradVals[i];
        double vI = s2 * (double)gradVals[i] * (double)gradVals[i];
        double d = sqrt(vI) / bc2sqrt + eps;
        paramNew[i] = decay * paramDeq + stepScale * mI / d;
    }

    for (size_t i = 0; i < n; i++) {
        size_t grp = i / groupSize;
        float scale = (grp == 0) ? scale0 : scale1;
        float actualDeq = (float)mant[i] * scale;
        float tolerance = 0.5f * scale + 1e-3f;
        TEST_ASSERT_FLOAT_WITHIN(tolerance, (float)paramNew[i], actualDeq);
    }
}

/* BFP epic PR3 Task 7: BEHAVIORAL pin for BFP moment storage through the
 * full factory path (adamWCreateOptim admits the per-tensor BFP template
 * since the gate lift above). AdamWApi.h's no-bit-parity disclaimer for
 * quantized moments rules out a gold here -- the param path chains six
 * float32 roundings through a sqrt/div (see the grouped-SYM test's GOLD
 * CHOICE note above) -- so this pins the observable storage contract
 * instead: after 5 steps on a constant nonzero grad, (a) every param stays
 * finite, (b) BOTH moment buffers are still BFP-typed per-tensor, and (c)
 * BOTH stored exponents have moved off the zero-state bias (e=8 -> 127)
 * with nonzero packed codes -- i.e. every step's OUT_WRITE really
 * re-derived a fresh absmax exponent and repacked, so the moments live on
 * the BFP grid rather than in a float buffer a broken momentStateInit could
 * have handed out. writeBackRounding opts out to HALF_AWAY (#279) for
 * determinism, though these assertions would hold under seeded SR too. */
void testAdamWStepBfpMomentsRunAndRequantize(void) {
    quantization_t *layerQ = quantizationInitFloat();
    size_t *wDims = reserveMemory(2 * sizeof(size_t));
    wDims[0] = 2;
    wDims[1] = 4;
    size_t *wOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, wOrder);
    shape_t *wShape = reserveMemory(sizeof(shape_t));
    setShape(wShape, wDims, 2, wOrder);
    tensor_t *wParam = initTensor(wShape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(wParam,
                              (float[]){0.5f, -0.25f, 0.75f, -0.5f, 0.3f, -0.8f, 0.6f, -0.4f}, 8);
    tensor_t *wGrad = gradInitFloat(wParam, NULL);
    tensorFillFromFloatBuffer(wGrad,
                              (float[]){0.2f, -0.1f, 0.3f, -0.2f, 0.1f, -0.3f, 0.25f, -0.15f}, 8);
    parameter_t *weights = parameterInit(wParam, wGrad);

    layer_t *linear = buildBorrowedLinearLayer(weights, NULL, layerQ);
    layer_t *model[] = {linear};

    quantization_t *momentQ = quantizationInitBfp(8, 8, HALF_AWAY);
    optimizer_t *optim =
        adamWCreateOptim(0.01f, 0.9, 0.999, 1e-8, 0.01, model, 1, momentQ,
                         (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});
    optimizerSetWriteBackRounding(optim, HALF_AWAY);

    for (int s = 0; s < 5; s++) {
        adamWStep(optim);
    }

    /* CAPTURE -> free -> assert (file convention; freeOptim's cascade frees
     * the registered parameters, so params are copied out first). */
    float params[8];
    memcpy(params, wParam->data, sizeof params);
    qtype_t momentTypes[2];
    size_t momentNumGroups[2];
    uint8_t momentExps[2];
    bool codesAllZero[2];
    for (size_t b = 0; b < 2; b++) {
        tensor_t *state = optim->states[0]->stateBuffers[b];
        momentTypes[b] = state->quantization->type;
        bfpQConfig_t *stateQC = state->quantization->qConfig;
        momentNumGroups[b] = stateQC->numGroups;
        momentExps[b] = stateQC->exponents[0];
        codesAllZero[b] = true;
        for (size_t i = 0; i < 8; i++) { /* 8 elements x 8-bit mantissas = 8 packed bytes */
            if (((uint8_t *)state->data)[i] != 0) {
                codesAllZero[b] = false;
            }
        }
    }
    freeOptim(optim);
    freeLinearLayerShellOnly(linear);
    freeQuantization(momentQ);
    freeQuantization(layerQ);

    for (size_t i = 0; i < 8; i++) {
        TEST_ASSERT_TRUE_MESSAGE(isfinite(params[i]),
                                 "param must stay finite across 5 BFP-moment steps");
    }
    for (size_t b = 0; b < 2; b++) {
        TEST_ASSERT_EQUAL_INT(BFP, momentTypes[b]);
        TEST_ASSERT_EQUAL_size_t(1, momentNumGroups[b]);
        TEST_ASSERT_TRUE_MESSAGE(momentExps[b] != 127,
                                 "moment exponent must move off the zero-state bias -- the "
                                 "OUT_WRITE repack re-derives it from the running moment");
        TEST_ASSERT_FALSE_MESSAGE(codesAllZero[b],
                                  "moment codes must be nonzero after 5 nonzero-grad steps");
    }
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testAdamWOptimizerSkipsFrozenLayerInCountAndCollection);
    RUN_TEST(testAdamWCreateAllFrozenModelExits);
    RUN_TEST(testAdamWCreateGroupedSymMomentQuantExits);
    RUN_TEST(testAdamWCreateGroupedAsymMomentQuantExits);
    RUN_TEST(testAdamWCreateAdmitsPerTensorBfpMoments);
    RUN_TEST(testAdamWCreateRejectsGroupedBfpMoments);
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
    RUN_TEST(testAdamWCreateOptimDefaultsWriteBackRoundingToSr);
    RUN_TEST(testAdamWStepOptimizerSrWriteBackEscapesSymDeadZone);
    RUN_TEST(testAdamWMomentWriteBacksHonorOptimizerSrRounding);
    RUN_TEST(testAdamWCreateOptimRejectsInt32GradStorage);
    RUN_TEST(testAdamWStepGroupedSymParamRunsAndRequantsPerGroup);
    RUN_TEST(testAdamWStepBfpMomentsRunAndRequantize);
    return UNITY_END();
}
