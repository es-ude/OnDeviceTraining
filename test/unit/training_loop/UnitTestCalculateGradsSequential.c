#define SOURCE_FILE "UnitTestCalculateGradsSequential"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "ArithmeticType.h"
#include "BorrowedLayer.h"
#include "CalculateGradsSequential.h"
#include "Common.h"
#include "Layer.h"
#include "LayerCommon.h"
#include "LayerExec.h"
#include "LayerQuant.h"
#include "Linear.h"
#include "LinearApi.h"
#include "OptimizerApi.h"
#include "Pool1dApi.h"
#include "QuantizationApi.h"
#include "QuantizationLayer.h"
#include "RNG.h"
#include "ReluApi.h"
#include "SgdApi.h"
#include "SoftmaxApi.h"
#include "StateDictApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "TraceApi.h"
#include "unity.h"

void setUp() {}
void tearDown() {}

/* Build a [1,2] float32 tensor from a stack buffer (data is copied into the tensor). */
static tensor_t *makeRowVec2(float a, float b) {
    size_t *dims = reserveMemory(2 * sizeof(size_t));
    size_t *order = reserveMemory(2 * sizeof(size_t));
    dims[0] = 1;
    dims[1] = 2;
    order[0] = 0;
    order[1] = 1;
    shape_t *shape = reserveMemory(sizeof(shape_t));
    shape->dimensions = dims;
    shape->orderOfDimensions = order;
    shape->numberOfDimensions = 2;
    tensor_t *t = initTensor(shape, quantizationInitFloat(), NULL);
    float vals[2] = {a, b};
    tensorFillFromFloatBuffer(t, vals, 2);
    return t;
}

/* Build a [1,n] float32 tensor from a caller-owned buffer (data is copied into the tensor). */
static tensor_t *makeRowVecN(const float *vals, size_t n) {
    size_t *dims = reserveMemory(2 * sizeof(size_t));
    size_t *order = reserveMemory(2 * sizeof(size_t));
    dims[0] = 1;
    dims[1] = n;
    setOrderOfDimsForNewTensor(2, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 2, order);
    tensor_t *t = initTensor(shape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(t, vals, n);
    return t;
}

/* Structural note: tracedGrads and calculateGradsSequential both call calculateGradsImpl
 * internally; npyDumpSink (and any other sink) observes tensors but does not mutate them.
 * This means the closed-form characterisation test pins both paths simultaneously. */
void testCalculateGradsSequentialClosedForm() {
    layerQuant_t lq;
    layerQuantInitUniform(&lq, quantizationInitFloat());

    layer_t *model[2];
    const rngKey_t initKey0 = rngFold(rngKeyFromSeed(1u), RNG_TAG_INIT, 0u);
    model[0] = linearLayerInit(
        &(linearInit_t){.inFeatures = 2, .outFeatures = 2, .weightInit.key = &initKey0}, &lq);
    model[1] = softmaxLayerInit(&lq);

    /* Set known weights/bias: W = {{0.1,0.2},{0.3,0.4}}, b = {0,0}. */
    float W[4] = {0.1f, 0.2f, 0.3f, 0.4f};
    float B[2] = {0.0f, 0.0f};
    modelLoadStateDict(model, 2,
                       (stateDictEntry_t[]){{.name = "fc", .weightData = W, .biasData = B}}, 1);

    tensor_t *x = makeRowVec2(1.0f, 1.0f);
    tensor_t *label = makeRowVec2(1.0f, 0.0f); /* one-hot class 0 */

    trainingStats_t *stats = calculateGradsSequential(
        model, 2,
        (lossConfig_t){
            .funcType = CROSS_ENTROPY, .backwardReduction = REDUCTION_MEAN, .classWeights = NULL},
        REDUCTION_MEAN, x, label, NULL);

    TEST_ASSERT_FLOAT_WITHIN(1e-4f, 0.91300f, stats->loss);

    float *wg = (float *)getGradFromParameter(model[0]->config->linear->weights)->data;
    TEST_ASSERT_FLOAT_WITHIN(1e-4f, -0.59869f, wg[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-4f, -0.59869f, wg[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-4f, 0.59869f, wg[2]);
    TEST_ASSERT_FLOAT_WITHIN(1e-4f, 0.59869f, wg[3]);

    float *bg = (float *)getGradFromParameter(model[0]->config->linear->bias)->data;
    TEST_ASSERT_FLOAT_WITHIN(1e-4f, -0.59869f, bg[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-4f, 0.59869f, bg[1]);

    freeTrainingStats(stats);
    freeTensor(x);
    freeTensor(label);
    freeLinearLayer(model[0]);
    freeSoftmaxLayer(model[1]);
}

#define MAX_EVENTS 64
typedef struct {
    size_t idx;
    char phase[32];
    size_t ndim;
} traceEvent_t;
static traceEvent_t g_events[MAX_EVENTS];
static size_t g_eventCount;

static void recordingSink(void *ctx, size_t layerIdx, layerType_t type, const char *phase,
                          tensor_t *t) {
    (void)ctx;
    (void)type;
    if (g_eventCount >= MAX_EVENTS) {
        return;
    }
    g_events[g_eventCount].idx = layerIdx;
    snprintf(g_events[g_eventCount].phase, sizeof(g_events[g_eventCount].phase), "%s", phase);
    g_events[g_eventCount].ndim = t->shape->numberOfDimensions;
    g_eventCount++;
}

void testTracedGradsFiresInOrder() {
    g_eventCount = 0;
    layerQuant_t lq;
    layerQuantInitUniform(&lq, quantizationInitFloat());
    layer_t *model[2];
    const rngKey_t initKey0 = rngFold(rngKeyFromSeed(1u), RNG_TAG_INIT, 0u);
    model[0] = linearLayerInit(
        &(linearInit_t){.inFeatures = 2, .outFeatures = 2, .weightInit.key = &initKey0}, &lq);
    model[1] = softmaxLayerInit(&lq);
    float W[4] = {0.1f, 0.2f, 0.3f, 0.4f}, B[2] = {0};
    modelLoadStateDict(model, 2,
                       (stateDictEntry_t[]){{.name = "fc", .weightData = W, .biasData = B}}, 1);
    tensor_t *x = makeRowVec2(1.0f, 1.0f);
    tensor_t *label = makeRowVec2(1.0f, 0.0f);

    trainingStats_t *stats = tracedGrads(model, 2,
                                         (lossConfig_t){.funcType = CROSS_ENTROPY,
                                                        .backwardReduction = REDUCTION_MEAN,
                                                        .classWeights = NULL},
                                         REDUCTION_MEAN, x, label, recordingSink, NULL, NULL);

    /* fwd L0, fwd L1, lossgrad@2, agrad L0  (Softmax skipped under CE) */
    TEST_ASSERT_EQUAL_size_t(4, g_eventCount);
    TEST_ASSERT_EQUAL_size_t(0, g_events[0].idx);
    TEST_ASSERT_EQUAL_STRING("fwd", g_events[0].phase);
    TEST_ASSERT_EQUAL_size_t(2, g_events[0].ndim);
    TEST_ASSERT_EQUAL_size_t(1, g_events[1].idx);
    TEST_ASSERT_EQUAL_STRING("fwd", g_events[1].phase);
    TEST_ASSERT_EQUAL_size_t(2, g_events[1].ndim);
    TEST_ASSERT_EQUAL_size_t(2, g_events[2].idx);
    TEST_ASSERT_EQUAL_STRING("lossgrad", g_events[2].phase);
    TEST_ASSERT_EQUAL_size_t(2, g_events[2].ndim);
    TEST_ASSERT_EQUAL_size_t(0, g_events[3].idx);
    TEST_ASSERT_EQUAL_STRING("agrad", g_events[3].phase);
    TEST_ASSERT_EQUAL_size_t(2, g_events[3].ndim);

    freeTrainingStats(stats);
    freeTensor(x);
    freeTensor(label);
    freeLinearLayer(model[0]);
    freeSoftmaxLayer(model[1]);
}

void testTraceModelParamsFiresPerTrainableParam() {
    g_eventCount = 0;
    layerQuant_t lq;
    layerQuantInitUniform(&lq, quantizationInitFloat());
    layer_t *model[2];
    const rngKey_t initKey0 = rngFold(rngKeyFromSeed(1u), RNG_TAG_INIT, 0u);
    model[0] = linearLayerInit(
        &(linearInit_t){.inFeatures = 2, .outFeatures = 2, .weightInit.key = &initKey0}, &lq);
    model[1] = softmaxLayerInit(&lq);
    float W[4] = {0.1f, 0.2f, 0.3f, 0.4f}, B[2] = {0};
    modelLoadStateDict(model, 2,
                       (stateDictEntry_t[]){{.name = "fc", .weightData = W, .biasData = B}}, 1);
    tensor_t *x = makeRowVec2(1.0f, 1.0f), *label = makeRowVec2(1.0f, 0.0f);
    trainingStats_t *stats = calculateGradsSequential(
        model, 2,
        (lossConfig_t){
            .funcType = CROSS_ENTROPY, .backwardReduction = REDUCTION_MEAN, .classWeights = NULL},
        REDUCTION_MEAN, x, label, NULL);

    traceModelWeights(model, 2, "w_before", recordingSink, NULL);
    traceModelGrads(model, 2, "grad_raw", recordingSink, NULL);

    /* weight+bias for the one Linear, then wgrad+bgrad */
    TEST_ASSERT_EQUAL_size_t(4, g_eventCount);
    TEST_ASSERT_EQUAL_STRING("w_before.weight", g_events[0].phase);
    TEST_ASSERT_EQUAL_STRING("w_before.bias", g_events[1].phase);
    TEST_ASSERT_EQUAL_STRING("grad_raw.weight", g_events[2].phase);
    TEST_ASSERT_EQUAL_STRING("grad_raw.bias", g_events[3].phase);

    freeTrainingStats(stats);
    freeTensor(x);
    freeTensor(label);
    freeLinearLayer(model[0]);
    freeSoftmaxLayer(model[1]);
}

/* #380: traceModelGrads must never hand a NULL tensor to
 * the sink. A frozen layer's parameter_t carries grad == NULL (Task 1 elides
 * it), so the pre-fix traceModelParams unconditionally dereferenced it inside
 * sink(); every real sink (npyDumpSink, paramGateSink) dereferences
 * unconditionally too, so this is a hard crash, not a soft no-op. Model:
 * [frozen Linear, trainable Linear] -- the sink must fire ONLY for the
 * trainable layer's weight+bias, and never with a NULL tensor pointer. */
typedef struct {
    size_t callCount;
    size_t layerIdx[MAX_EVENTS];
    bool sawNoNull;
} gradTraceCounts_t;

static void countingSink(void *ctx, size_t layerIdx, layerType_t type, const char *phase,
                         tensor_t *t) {
    (void)type;
    (void)phase;
    gradTraceCounts_t *c = (gradTraceCounts_t *)ctx;
    if (t == NULL) {
        c->sawNoNull = false;
        return;
    }
    if (c->callCount < MAX_EVENTS) {
        c->layerIdx[c->callCount] = layerIdx;
    }
    c->callCount++;
}

void testTraceModelGradsSkipsFrozenLayerNeverPassesNull(void) {
    layerQuant_t lq;
    layerQuantInitUniform(&lq, quantizationInitFloat());
    layer_t *model[2];
    const rngKey_t initKey0 = rngFold(rngKeyFromSeed(1u), RNG_TAG_INIT, 0u);
    const rngKey_t initKey1 = rngFold(rngKeyFromSeed(1u), RNG_TAG_INIT, 1u);
    model[0] = linearLayerInit(&(linearInit_t){.inFeatures = 2,
                                               .outFeatures = 2,
                                               .trainable = TRAINABLE_FALSE,
                                               .weightInit.key = &initKey0},
                               &lq);
    model[1] = linearLayerInit(
        &(linearInit_t){.inFeatures = 2, .outFeatures = 2, .weightInit.key = &initKey1}, &lq);

    gradTraceCounts_t counts = {.callCount = 0, .sawNoNull = true};
    traceModelGrads(model, 2, "grad", countingSink, &counts);

    /* CAPTURE (before any free touches the layers). */
    size_t callCount = counts.callCount;
    bool sawNoNull = counts.sawNoNull;
    size_t firstIdx = counts.callCount > 0 ? counts.layerIdx[0] : (size_t)-1;
    size_t secondIdx = counts.callCount > 1 ? counts.layerIdx[1] : (size_t)-1;

    freeLinearLayer(model[0]);
    freeLinearLayer(model[1]);

    TEST_ASSERT_TRUE_MESSAGE(sawNoNull, "sink must never receive a NULL tensor");
    TEST_ASSERT_EQUAL_size_t_MESSAGE(
        2, callCount, "sink must fire exactly twice (trainable layer's weight+bias only)");
    TEST_ASSERT_EQUAL_size_t_MESSAGE(1, firstIdx,
                                     "sink must fire for the trainable layer (index 1), not the "
                                     "frozen layer (index 0)");
    TEST_ASSERT_EQUAL_size_t_MESSAGE(1, secondIdx,
                                     "sink must fire for the trainable layer (index 1) twice "
                                     "(weight+bias)");
}

/* ── #221 regression: dx wire must honor the producer's declared propLossQ ── */

typedef struct {
    roundingMode_t agradRoundingMode;
    bool capturedAgrad;
} agradCapCtx_t;

static void agradCaptureSink(void *ctx, size_t layerIdx, layerType_t type, const char *phase,
                             tensor_t *t) {
    (void)type;
    agradCapCtx_t *c = (agradCapCtx_t *)ctx;
    /* agrad@1 = dx wire produced by Linear1 (idx 2) via backwardWireQ → propLossQ (SR_HALF_AWAY) */
    if (layerIdx == 1 && strcmp(phase, "agrad") == 0) {
        c->agradRoundingMode = ((symInt32QConfig_t *)t->quantization->qConfig)->roundingMode;
        c->capturedAgrad = true;
    }
}

/*! Two-Q variant of buildBorrowedLinearLayer (BorrowedLayer.h): forwardMath/
 *  weightGradMath/biasGradMath/outputQ derive from mathQ, propLossMath/
 *  propLossQ derive from the (possibly divergent) propLossQ — replicates the
 *  deleted linearLayerInitLegacy(weights, bias, mathQ, mathQ, mathQ,
 *  propLossQ) shape. Needed because the weight/bias tensors here are
 *  SYM_INT32-native (makeSymParam), which the factory does not allocate
 *  (LayerCommon.c requireFloat32, by design — #270). */
static layer_t *buildSymLinearLayer(parameter_t *weights, parameter_t *bias, quantization_t *mathQ,
                                    quantization_t *propLossQ) {
    linearConfig_t *cfg = reserveMemory(sizeof(linearConfig_t));
    cfg->weights = weights;
    cfg->bias = bias;
    cfg->forwardMath = arithmeticFromQuantization(mathQ);
    cfg->weightGradMath = arithmeticFromQuantization(mathQ);
    cfg->biasGradMath = arithmeticFromQuantization(mathQ);
    cfg->propLossMath = arithmeticFromQuantization(propLossQ);
    cfg->outputQ = mathQ;
    cfg->propLossQ = propLossQ;
    /* packed-grad D1: today's per-callsite hardcodes (linearBackward); hand-wired
     * here since this helper builds the config directly instead of going
     * through linearInitConfig/a layerQuant_t factory. */
    cfg->weightGradAccMode = OUT_ACC_DYNAMIC_RESCALE;
    cfg->biasGradAccMode = OUT_ACC_FIXED_SCALE;
    cfg->ownsQuantizations = false;
    layerConfig_t *layerCfg = reserveMemory(sizeof(layerConfig_t));
    layerCfg->linear = cfg;
    layer_t *layer = reserveMemory(sizeof(layer_t));
    initLayer(layer, LINEAR, layerCfg);
    return layer;
}

/* Create a SYM_INT32 parameter (param tensor + optional grad tensor) from float values.
 * dims[0..ndim-1] describe the shape; vals[0..n-1] are loaded via tensorFillFromFloatBuffer. */
static parameter_t *makeSymParam(float *vals, size_t n, size_t *dims, size_t ndim,
                                 roundingMode_t rm, bool needsGrad) {
    size_t *d = reserveMemory(ndim * sizeof(size_t));
    size_t *o = reserveMemory(ndim * sizeof(size_t));
    for (size_t i = 0; i < ndim; i++) {
        d[i] = dims[i];
    }
    setOrderOfDimsForNewTensor(ndim, o);
    shape_t *s = reserveMemory(sizeof(shape_t));
    setShape(s, d, ndim, o);
    tensor_t *param = initTensor(s, quantizationInitSymInt32(rm), NULL);
    tensorFillFromFloatBuffer(param, vals, n);
    tensor_t *grad = needsGrad ? gradInitSymInt32(param, rm, NULL) : NULL;
    return parameterInit(param, grad);
}

void testDxWireHonorsProducerPropLossQ(void) {
    /* 4-layer SYM_INT32 chain: [Linear0 (idx 0), Quant0 (idx 1), Linear1 (idx 2), Quant1 (idx 3)].
     * All forward/backwardMath use HALF_AWAY; Linear1's propLossQ = SR_HALF_AWAY.
     * Quantization layers are mandatory between SYM producers: the forward wire carries raw
     * accumulator-range mantissas until a Quant layer restores int12 range (a direct
     * Linear-Linear chain overflows int32 under UBSan).
     *
     * Pre-fix: the agrad@1 buffer carried HALF_AWAY (derived from Quant0's forward-output
     * quantization by the training dx-wire allocator, ignoring Linear1's propLossQ).
     * Post-fix: it carries SR_HALF_AWAY (from Linear1's propLossQ). */
    quantization_t *symQ = quantizationInitSymInt32(HALF_AWAY);
    quantization_t *symQSr = quantizationInitSymInt32(SR_HALF_AWAY);

    float wVals[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    float bVals[2] = {0.0f, 0.0f};
    size_t wDims[2] = {2, 2};
    size_t bDims[1] = {2};
    parameter_t *w0 = makeSymParam(wVals, 4, wDims, 2, HALF_AWAY, /*needsGrad=*/true);
    parameter_t *b0 = makeSymParam(bVals, 2, bDims, 1, HALF_AWAY, /*needsGrad=*/true);
    parameter_t *w1 = makeSymParam(wVals, 4, wDims, 2, HALF_AWAY, /*needsGrad=*/true);
    parameter_t *b1 = makeSymParam(bVals, 2, bDims, 1, HALF_AWAY, /*needsGrad=*/true);

    layer_t *linear0 = buildSymLinearLayer(w0, b0, symQ, symQ);
    /* linear1: forwardQ/backwardMath = HALF_AWAY, propLossQ = SR_HALF_AWAY */
    layer_t *linear1 = buildSymLinearLayer(w1, b1, symQ, symQSr);

    /* Borrowing Quantization layers between SYM producers (ownsQuantizations=false). */
    quantizationConfig_t *qCfg0 = reserveMemory(sizeof(quantizationConfig_t));
    qCfg0->outputQ = symQ;
    qCfg0->propLossQ = symQ;
    qCfg0->ownsQuantizations = false;
    layerConfig_t *lc0 = reserveMemory(sizeof(layerConfig_t));
    lc0->quantization = qCfg0;
    layer_t *quant0 = reserveMemory(sizeof(layer_t));
    initLayer(quant0, QUANTIZATION, lc0);

    quantizationConfig_t *qCfg1 = reserveMemory(sizeof(quantizationConfig_t));
    qCfg1->outputQ = symQ;
    qCfg1->propLossQ = symQ;
    qCfg1->ownsQuantizations = false;
    layerConfig_t *lc1 = reserveMemory(sizeof(layerConfig_t));
    lc1->quantization = qCfg1;
    layer_t *quant1 = reserveMemory(sizeof(layer_t));
    initLayer(quant1, QUANTIZATION, lc1);

    layer_t *model[4] = {linear0, quant0, linear1, quant1};

    size_t *xd = reserveMemory(2 * sizeof(size_t));
    size_t *xo = reserveMemory(2 * sizeof(size_t));
    xd[0] = 1;
    xd[1] = 2;
    setOrderOfDimsForNewTensor(2, xo);
    shape_t *xShape = reserveMemory(sizeof(shape_t));
    setShape(xShape, xd, 2, xo);
    tensor_t *x = initTensor(xShape, quantizationInitSymInt32(HALF_AWAY), NULL);
    tensorFillFromFloatBuffer(x, (float[]){1.0f, 2.0f}, 2);

    size_t *ld = reserveMemory(2 * sizeof(size_t));
    size_t *lo = reserveMemory(2 * sizeof(size_t));
    ld[0] = 1;
    ld[1] = 2;
    setOrderOfDimsForNewTensor(2, lo);
    shape_t *lShape = reserveMemory(sizeof(shape_t));
    setShape(lShape, ld, 2, lo);
    tensor_t *label = initTensor(lShape, quantizationInitSymInt32(HALF_AWAY), NULL);
    tensorFillFromFloatBuffer(label, (float[]){0.0f, 0.0f}, 2);

    /* The SR dx wire makes the backward draw, so the call carries a key. */
    const rngKey_t callKey = rngKeyFromSeed(0x221u);
    agradCapCtx_t ctx = {.capturedAgrad = false};
    trainingStats_t *stats = tracedGrads(
        model, 4,
        (lossConfig_t){.funcType = MSE, .backwardReduction = REDUCTION_SUM, .classWeights = NULL},
        REDUCTION_SUM, x, label, agradCaptureSink, &ctx, &(trainingCall_t){.key = &callKey});

    /* Capture before teardown (ASSERT LAST convention). */
    bool captured = ctx.capturedAgrad;
    roundingMode_t mode = ctx.agradRoundingMode;

    freeTrainingStats(stats);
    freeTensor(x);
    freeTensor(label);
    freeLinearLayerShellOnly(linear0);
    freeLinearLayerShellOnly(linear1);
    freeReservedMemory(qCfg0);
    freeReservedMemory(lc0);
    freeReservedMemory(quant0);
    freeReservedMemory(qCfg1);
    freeReservedMemory(lc1);
    freeReservedMemory(quant1);
    freeParameter(w0);
    freeParameter(b0);
    freeParameter(w1);
    freeParameter(b1);
    freeQuantization(symQ);
    freeQuantization(symQSr);

    TEST_ASSERT_TRUE_MESSAGE(captured, "sink never captured agrad for layer 1");
    TEST_ASSERT_EQUAL_INT_MESSAGE(SR_HALF_AWAY, mode,
                                  "dx wire must carry the PRODUCER's declared propLossQ "
                                  "roundingMode, not the upstream forward roundingMode (#221)");
}

/* ── Task 10 regression: allocators honor the producer's declared qMaxBits ── */

typedef struct {
    uint8_t agradQMaxBits;
    bool capturedAgrad;
} agradBitsCapCtx_t;

static void agradBitsCaptureSink(void *ctx, size_t layerIdx, layerType_t type, const char *phase,
                                 tensor_t *t) {
    (void)type;
    agradBitsCapCtx_t *c = (agradBitsCapCtx_t *)ctx;
    /* agrad@1 = dx wire produced by Linear1 (idx 2) via backwardWireQ -> propLossQ (qMaxBits=8). */
    if (layerIdx == 1 && strcmp(phase, "agrad") == 0) {
        c->agradQMaxBits = ((symInt32QConfig_t *)t->quantization->qConfig)->qMaxBits;
        c->capturedAgrad = true;
    }
}

void testDxWireHonorsProducerPropLossQMaxBits(void) {
    /* Same 4-layer SYM_INT32 chain as testDxWireHonorsProducerPropLossQ. This time
     * Linear1's propLossQ declares qMaxBits=8 (narrower than the int12 operand
     * default) instead of a divergent roundingMode.
     * Pre-fix: the training dx-wire allocator re-defaulted the dx wire to 12 regardless
     * of the declared width. Post-fix: it carries the declared 8. */
    quantization_t *symQ = quantizationInitSymInt32(HALF_AWAY);
    quantization_t *symQ8 = quantizationInitSymInt32WithBits(HALF_AWAY, 8);

    float wVals[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    float bVals[2] = {0.0f, 0.0f};
    size_t wDims[2] = {2, 2};
    size_t bDims[1] = {2};
    parameter_t *w0 = makeSymParam(wVals, 4, wDims, 2, HALF_AWAY, /*needsGrad=*/true);
    parameter_t *b0 = makeSymParam(bVals, 2, bDims, 1, HALF_AWAY, /*needsGrad=*/true);
    parameter_t *w1 = makeSymParam(wVals, 4, wDims, 2, HALF_AWAY, /*needsGrad=*/true);
    parameter_t *b1 = makeSymParam(bVals, 2, bDims, 1, HALF_AWAY, /*needsGrad=*/true);

    layer_t *linear0 = buildSymLinearLayer(w0, b0, symQ, symQ);
    /* linear1: forwardQ/backwardMath = HALF_AWAY@12, propLossQ = HALF_AWAY@8 */
    layer_t *linear1 = buildSymLinearLayer(w1, b1, symQ, symQ8);

    quantizationConfig_t *qCfg0 = reserveMemory(sizeof(quantizationConfig_t));
    qCfg0->outputQ = symQ;
    qCfg0->propLossQ = symQ;
    qCfg0->ownsQuantizations = false;
    layerConfig_t *lc0 = reserveMemory(sizeof(layerConfig_t));
    lc0->quantization = qCfg0;
    layer_t *quant0 = reserveMemory(sizeof(layer_t));
    initLayer(quant0, QUANTIZATION, lc0);

    quantizationConfig_t *qCfg1 = reserveMemory(sizeof(quantizationConfig_t));
    qCfg1->outputQ = symQ;
    qCfg1->propLossQ = symQ;
    qCfg1->ownsQuantizations = false;
    layerConfig_t *lc1 = reserveMemory(sizeof(layerConfig_t));
    lc1->quantization = qCfg1;
    layer_t *quant1 = reserveMemory(sizeof(layer_t));
    initLayer(quant1, QUANTIZATION, lc1);

    layer_t *model[4] = {linear0, quant0, linear1, quant1};

    size_t *xd = reserveMemory(2 * sizeof(size_t));
    size_t *xo = reserveMemory(2 * sizeof(size_t));
    xd[0] = 1;
    xd[1] = 2;
    setOrderOfDimsForNewTensor(2, xo);
    shape_t *xShape = reserveMemory(sizeof(shape_t));
    setShape(xShape, xd, 2, xo);
    tensor_t *x = initTensor(xShape, quantizationInitSymInt32(HALF_AWAY), NULL);
    tensorFillFromFloatBuffer(x, (float[]){1.0f, 2.0f}, 2);

    size_t *ld = reserveMemory(2 * sizeof(size_t));
    size_t *lo = reserveMemory(2 * sizeof(size_t));
    ld[0] = 1;
    ld[1] = 2;
    setOrderOfDimsForNewTensor(2, lo);
    shape_t *lShape = reserveMemory(sizeof(shape_t));
    setShape(lShape, ld, 2, lo);
    tensor_t *label = initTensor(lShape, quantizationInitSymInt32(HALF_AWAY), NULL);
    tensorFillFromFloatBuffer(label, (float[]){0.0f, 0.0f}, 2);

    agradBitsCapCtx_t ctx = {.capturedAgrad = false};
    trainingStats_t *stats = tracedGrads(
        model, 4,
        (lossConfig_t){.funcType = MSE, .backwardReduction = REDUCTION_SUM, .classWeights = NULL},
        REDUCTION_SUM, x, label, agradBitsCaptureSink, &ctx, NULL);

    /* Capture before teardown (ASSERT LAST convention). */
    bool captured = ctx.capturedAgrad;
    uint8_t bits = ctx.agradQMaxBits;

    freeTrainingStats(stats);
    freeTensor(x);
    freeTensor(label);
    freeLinearLayerShellOnly(linear0);
    freeLinearLayerShellOnly(linear1);
    freeReservedMemory(qCfg0);
    freeReservedMemory(lc0);
    freeReservedMemory(quant0);
    freeReservedMemory(qCfg1);
    freeReservedMemory(lc1);
    freeReservedMemory(quant1);
    freeParameter(w0);
    freeParameter(b0);
    freeParameter(w1);
    freeParameter(b1);
    freeQuantization(symQ);
    freeQuantization(symQ8);

    TEST_ASSERT_TRUE_MESSAGE(captured, "sink never captured agrad for layer 1");
    TEST_ASSERT_EQUAL_UINT8_MESSAGE(8, bits,
                                    "dx wire must carry the PRODUCER's declared propLossQ "
                                    "qMaxBits, not the re-defaulted int12 operand width");
}

typedef struct {
    uint8_t fwdQMaxBits;
    bool capturedFwd;
} fwdBitsCapCtx_t;

static void fwdBitsCaptureSink(void *ctx, size_t layerIdx, layerType_t type, const char *phase,
                               tensor_t *t) {
    (void)type;
    fwdBitsCapCtx_t *c = (fwdBitsCapCtx_t *)ctx;
    if (layerIdx == 0 && strcmp(phase, "fwd") == 0) {
        c->fwdQMaxBits = ((symInt32QConfig_t *)t->quantization->qConfig)->qMaxBits;
        c->capturedFwd = true;
    }
}

void testForwardWireHonorsDeclaredOutputQMaxBits(void) {
    /* Single SYM_INT32 Linear layer whose outputQ declares qMaxBits=8 (narrower
     * than the int12 operand default). Pre-fix: the training forward-wire
     * allocator re-defaulted the wire to 12, discarding the declared width.
     * Post-fix: the forward wire carries the declared 8. */
    quantization_t *symQ = quantizationInitSymInt32(HALF_AWAY);
    quantization_t *symQ8 = quantizationInitSymInt32WithBits(HALF_AWAY, 8);

    float wVals[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    float bVals[2] = {0.0f, 0.0f};
    size_t wDims[2] = {2, 2};
    size_t bDims[1] = {2};
    parameter_t *w0 = makeSymParam(wVals, 4, wDims, 2, HALF_AWAY, /*needsGrad=*/true);
    parameter_t *b0 = makeSymParam(bVals, 2, bDims, 1, HALF_AWAY, /*needsGrad=*/true);

    layer_t *linear0 = buildSymLinearLayer(w0, b0, symQ, symQ);
    /* Override just the forward-wire storage config to a declared width of 8. */
    linear0->config->linear->outputQ = symQ8;

    layer_t *model[1] = {linear0};

    size_t *xd = reserveMemory(2 * sizeof(size_t));
    size_t *xo = reserveMemory(2 * sizeof(size_t));
    xd[0] = 1;
    xd[1] = 2;
    setOrderOfDimsForNewTensor(2, xo);
    shape_t *xShape = reserveMemory(sizeof(shape_t));
    setShape(xShape, xd, 2, xo);
    tensor_t *x = initTensor(xShape, quantizationInitSymInt32(HALF_AWAY), NULL);
    tensorFillFromFloatBuffer(x, (float[]){1.0f, 2.0f}, 2);

    size_t *ld = reserveMemory(2 * sizeof(size_t));
    size_t *lo = reserveMemory(2 * sizeof(size_t));
    ld[0] = 1;
    ld[1] = 2;
    setOrderOfDimsForNewTensor(2, lo);
    shape_t *lShape = reserveMemory(sizeof(shape_t));
    setShape(lShape, ld, 2, lo);
    tensor_t *label = initTensor(lShape, quantizationInitSymInt32(HALF_AWAY), NULL);
    tensorFillFromFloatBuffer(label, (float[]){0.0f, 0.0f}, 2);

    fwdBitsCapCtx_t ctx = {.capturedFwd = false};
    trainingStats_t *stats = tracedGrads(
        model, 1,
        (lossConfig_t){.funcType = MSE, .backwardReduction = REDUCTION_SUM, .classWeights = NULL},
        REDUCTION_SUM, x, label, fwdBitsCaptureSink, &ctx, NULL);

    /* Capture before teardown (ASSERT LAST convention). */
    bool captured = ctx.capturedFwd;
    uint8_t bits = ctx.fwdQMaxBits;

    freeTrainingStats(stats);
    freeTensor(x);
    freeTensor(label);
    freeLinearLayerShellOnly(linear0);
    freeParameter(w0);
    freeParameter(b0);
    freeQuantization(symQ);
    freeQuantization(symQ8);

    TEST_ASSERT_TRUE_MESSAGE(captured, "sink never captured fwd for layer 0");
    TEST_ASSERT_EQUAL_UINT8_MESSAGE(8, bits,
                                    "forward wire must carry the layer's declared outputQ "
                                    "qMaxBits, not the re-defaulted int12 operand width");
}

/* ── #380 PR1 Task 8: end-to-end frozen-layer integration gate ──
 * model: frozen Linear(4->4) -> RELU -> trainable Linear(4->2), MSE loss.
 * The frozen layer is wired as an identity map (W=I4, b=0) and fed a
 * strictly-positive input, so ReLU is a no-op -- this isolates the freeze
 * mechanism from any incidental dead-ReLU zero-gradient effect on the
 * trainable layer's weight grad (x=0 through a dead ReLU would zero the
 * outer-product weight grad regardless of freezing). Both Linear layers use
 * the Borrowing factory (linearLayerInit, not …Owning): they share one
 * long-lived quantization_t (freed once at the end), which is what lets the
 * frozen layer take a FULL freeLinearLayer teardown while the trainable
 * layer takes freeOptim + freeLinearLayerShellOnly (freeOptim already frees
 * its param/bias tensors) without leaking an Owning-cloned outputQ/propLossQ.
 */
void testFrozenLayerSurvivesTrainingUntouched(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);

    const rngKey_t initKey0 = rngFold(rngKeyFromSeed(1u), RNG_TAG_INIT, 0u);
    const rngKey_t initKey2 = rngFold(rngKeyFromSeed(1u), RNG_TAG_INIT, 2u);
    layer_t *frozenL = linearLayerInit(&(linearInit_t){.inFeatures = 4,
                                                       .outFeatures = 4,
                                                       .trainable = TRAINABLE_FALSE,
                                                       .weightInit.key = &initKey0},
                                       &lq);
    layer_t *reluL = reluLayerInit(&lq);
    layer_t *trainL = linearLayerInit(
        &(linearInit_t){.inFeatures = 4, .outFeatures = 2, .weightInit.key = &initKey2}, &lq);
    layer_t *model[3] = {frozenL, reluL, trainL};

    float frozenW[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    float frozenB[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float trainW[8] = {0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f};
    float trainB[2] = {0.0f, 0.0f};
    modelLoadStateDict(
        model, 3,
        (stateDictEntry_t[]){{.name = "frozen", .weightData = frozenW, .biasData = frozenB},
                             {.name = "trainable", .weightData = trainW, .biasData = trainB}},
        2);

    size_t *inDims = reserveMemory(2 * sizeof(size_t));
    inDims[0] = 1;
    inDims[1] = 4;
    size_t *inOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, inOrder);
    shape_t *inShape = reserveMemory(sizeof(shape_t));
    setShape(inShape, inDims, 2, inOrder);
    tensor_t *input = initTensor(inShape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(input, (float[]){1.0f, 1.0f, 1.0f, 1.0f}, 4);

    size_t *labelDims = reserveMemory(2 * sizeof(size_t));
    labelDims[0] = 1;
    labelDims[1] = 2;
    size_t *labelOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, labelOrder);
    shape_t *labelShape = reserveMemory(sizeof(shape_t));
    setShape(labelShape, labelDims, 2, labelOrder);
    tensor_t *label = initTensor(labelShape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(label, (float[]){1.0f, 0.0f}, 2);

    /* Snapshot frozen + trainable weight/bias bytes BEFORE training. */
    size_t frozenWBytes = calcBytesPerTensor(frozenL->config->linear->weights->param);
    uint8_t *beforeFrozenW = reserveMemory(frozenWBytes);
    memcpy(beforeFrozenW, frozenL->config->linear->weights->param->data, frozenWBytes);

    size_t frozenBBytes = calcBytesPerTensor(frozenL->config->linear->bias->param);
    uint8_t *beforeFrozenB = reserveMemory(frozenBBytes);
    memcpy(beforeFrozenB, frozenL->config->linear->bias->param->data, frozenBBytes);

    size_t trainWBytes = calcBytesPerTensor(trainL->config->linear->weights->param);
    uint8_t *beforeTrainW = reserveMemory(trainWBytes);
    memcpy(beforeTrainW, trainL->config->linear->weights->param->data, trainWBytes);

    quantization_t *momentumQ = quantizationInitFloat();
    optimizer_t *optim =
        sgdMCreateOptim(0.05f, 0.0f, 0.0f, model, 3, momentumQ,
                        (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});
    optimizerFunctions_t optimFns = optimizerFunctions[optim->type];

    lossConfig_t lossConfig = {
        .funcType = MSE, .backwardReduction = REDUCTION_SUM, .classWeights = NULL};

    float firstLoss = 0.0f, lastLoss = 0.0f;
    for (size_t step = 0; step < 5; step++) {
        optimFns.zero(optim);
        trainingStats_t *stats =
            calculateGradsSequential(model, 3, lossConfig, REDUCTION_SUM, input, label, NULL);
        if (step == 0) {
            firstLoss = stats->loss;
        }
        lastLoss = stats->loss;
        freeTrainingStats(stats);
        optimFns.step(optim, NULL);
    }

    /* CAPTURE (before any free touches the parameter data). */
    bool frozenWUnchanged =
        memcmp(beforeFrozenW, frozenL->config->linear->weights->param->data, frozenWBytes) == 0;
    bool frozenBUnchanged =
        memcmp(beforeFrozenB, frozenL->config->linear->bias->param->data, frozenBBytes) == 0;
    bool trainWChanged =
        memcmp(beforeTrainW, trainL->config->linear->weights->param->data, trainWBytes) != 0;
    float capturedFirstLoss = firstLoss;
    float capturedLastLoss = lastLoss;

    /* FREE. freeOptim frees only the COLLECTED (trainable-layer) parameters
     * (trainL's weights+bias) -- frozenL was never collected (#380), so it
     * needs a full teardown here; trainL gets shell-only (its param/bias
     * parameter_t's are already gone via freeOptim's cascade). */
    freeReservedMemory(beforeFrozenW);
    freeReservedMemory(beforeFrozenB);
    freeReservedMemory(beforeTrainW);
    freeTensor(input);
    freeTensor(label);
    freeOptim(optim);
    freeLinearLayerShellOnly(trainL);
    freeLinearLayer(frozenL);
    freeReluLayer(reluL);
    freeQuantization(momentumQ);
    freeQuantization(q);

    TEST_ASSERT_TRUE_MESSAGE(frozenWUnchanged,
                             "frozen layer's weights must survive training untouched");
    TEST_ASSERT_TRUE_MESSAGE(frozenBUnchanged,
                             "frozen layer's bias must survive training untouched");
    TEST_ASSERT_TRUE_MESSAGE(trainWChanged, "trainable layer's weights must change under training");
    TEST_ASSERT_TRUE_MESSAGE(capturedLastLoss < capturedFirstLoss,
                             "loss must decrease over training");
}

/* ── #380 PR2 Task 2: backward truncation at the deepest trainable layer ──
 * model: [frozen Linear(4->4), RELU, trainable Linear(4->2)], MSE loss.
 * Below the deepest trainable layer (index 2, the only trainable one — index
 * 0 is frozen, index 1 has no params) no dx is consumed, so the backward
 * loop must truncate there: "agrad" fires exactly once (i == 2), never for
 * i == 1 or i == 0, and exactly one "lossgrad" fires. The trainable layer's
 * weight grad must still be non-zero -- backward really ran for it. */
void testBackwardStopsAtDeepestTrainableLayer(void) {
    g_eventCount = 0;
    layerQuant_t lq;
    layerQuantInitUniform(&lq, quantizationInitFloat());

    const rngKey_t initKey0 = rngFold(rngKeyFromSeed(1u), RNG_TAG_INIT, 0u);
    const rngKey_t initKey2 = rngFold(rngKeyFromSeed(1u), RNG_TAG_INIT, 2u);
    layer_t *frozenL = linearLayerInit(&(linearInit_t){.inFeatures = 4,
                                                       .outFeatures = 4,
                                                       .trainable = TRAINABLE_FALSE,
                                                       .weightInit.key = &initKey0},
                                       &lq);
    layer_t *reluL = reluLayerInit(&lq);
    layer_t *trainL = linearLayerInit(
        &(linearInit_t){.inFeatures = 4, .outFeatures = 2, .weightInit.key = &initKey2}, &lq);
    layer_t *model[3] = {frozenL, reluL, trainL};

    float frozenW[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    float frozenB[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float trainW[8] = {0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f};
    float trainB[2] = {0.0f, 0.0f};
    modelLoadStateDict(
        model, 3,
        (stateDictEntry_t[]){{.name = "frozen", .weightData = frozenW, .biasData = frozenB},
                             {.name = "trainable", .weightData = trainW, .biasData = trainB}},
        2);

    tensor_t *x = makeRowVecN((float[]){1.0f, 1.0f, 1.0f, 1.0f}, 4);
    tensor_t *label = makeRowVecN((float[]){1.0f, 0.0f}, 2);

    trainingStats_t *stats = tracedGrads(
        model, 3,
        (lossConfig_t){.funcType = MSE, .backwardReduction = REDUCTION_SUM, .classWeights = NULL},
        REDUCTION_SUM, x, label, recordingSink, NULL, NULL);

    size_t agradAt0 = 0, agradAt1 = 0, agradAt2 = 0, lossgradCount = 0;
    for (size_t i = 0; i < g_eventCount; i++) {
        if (strcmp(g_events[i].phase, "agrad") == 0) {
            if (g_events[i].idx == 0) {
                agradAt0++;
            } else if (g_events[i].idx == 1) {
                agradAt1++;
            } else if (g_events[i].idx == 2) {
                agradAt2++;
            }
        } else if (strcmp(g_events[i].phase, "lossgrad") == 0) {
            lossgradCount++;
        }
    }

    float *wg = (float *)getGradFromParameter(trainL->config->linear->weights)->data;
    bool weightGradNonZero = false;
    for (size_t i = 0; i < 8; i++) {
        if (wg[i] != 0.0f) {
            weightGradNonZero = true;
        }
    }

    /* CAPTURE (before any free touches the layers). */
    size_t capturedAgradAt0 = agradAt0;
    size_t capturedAgradAt1 = agradAt1;
    size_t capturedAgradAt2 = agradAt2;
    size_t capturedLossgrad = lossgradCount;
    bool capturedWeightGradNonZero = weightGradNonZero;

    freeTrainingStats(stats);
    freeTensor(x);
    freeTensor(label);
    freeLinearLayer(frozenL);
    freeReluLayer(reluL);
    freeLinearLayer(trainL);

    TEST_ASSERT_EQUAL_size_t_MESSAGE(0, capturedAgradAt0,
                                     "agrad must not fire below the deepest "
                                     "trainable layer (index 0)");
    TEST_ASSERT_EQUAL_size_t_MESSAGE(0, capturedAgradAt1,
                                     "agrad must not fire below the deepest "
                                     "trainable layer (index 1)");
    TEST_ASSERT_EQUAL_size_t_MESSAGE(1, capturedAgradAt2,
                                     "agrad must fire exactly once, at the deepest trainable "
                                     "layer (index 2)");
    TEST_ASSERT_EQUAL_size_t_MESSAGE(1, capturedLossgrad, "lossgrad must fire exactly once");
    TEST_ASSERT_TRUE_MESSAGE(capturedWeightGradNonZero,
                             "trainable top layer's weight grad must be non-zero (backward ran)");
}

/* ── #380 PR2 Task 2: upper-layer grads don't depend on lower dx ──
 * M1: [frozen Linear(seed S), RELU, trainable Linear(seed T)] -- top layer is
 * the deepest trainable one, so its backward runs via the grads-only
 * (propLoss == NULL) path.
 * M2: [trainable Linear(seed S), RELU, trainable Linear(seed T)] -- the
 * bottom layer is now the deepest trainable one, so the top layer's backward
 * runs via the normal (propLoss != NULL) path and the loop continues to
 * compute dx all the way down.
 * Same input/label, same weights for both layers across both models. The
 * top layer's own weight+bias grads must be byte-identical either way --
 * truncating the loop below it must not perturb its own computation. */
void testTruncationPreservesUpperLayerGrads(void) {
    layerQuant_t lq;
    layerQuantInitUniform(&lq, quantizationInitFloat());

    float seedS[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    float seedSBias[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float seedT[8] = {0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f};
    float seedTBias[2] = {0.0f, 0.0f};

    const rngKey_t initKey0 = rngFold(rngKeyFromSeed(1u), RNG_TAG_INIT, 0u);
    const rngKey_t initKey2 = rngFold(rngKeyFromSeed(1u), RNG_TAG_INIT, 2u);
    layer_t *m1Bottom = linearLayerInit(&(linearInit_t){.inFeatures = 4,
                                                        .outFeatures = 4,
                                                        .trainable = TRAINABLE_FALSE,
                                                        .weightInit.key = &initKey0},
                                        &lq);
    layer_t *m1Relu = reluLayerInit(&lq);
    layer_t *m1Top = linearLayerInit(
        &(linearInit_t){.inFeatures = 4, .outFeatures = 2, .weightInit.key = &initKey2}, &lq);
    layer_t *m1[3] = {m1Bottom, m1Relu, m1Top};

    layer_t *m2Bottom = linearLayerInit(
        &(linearInit_t){.inFeatures = 4, .outFeatures = 4, .weightInit.key = &initKey0}, &lq);
    layer_t *m2Relu = reluLayerInit(&lq);
    layer_t *m2Top = linearLayerInit(
        &(linearInit_t){.inFeatures = 4, .outFeatures = 2, .weightInit.key = &initKey2}, &lq);
    layer_t *m2[3] = {m2Bottom, m2Relu, m2Top};

    modelLoadStateDict(
        m1, 3,
        (stateDictEntry_t[]){{.name = "bottom", .weightData = seedS, .biasData = seedSBias},
                             {.name = "top", .weightData = seedT, .biasData = seedTBias}},
        2);
    modelLoadStateDict(
        m2, 3,
        (stateDictEntry_t[]){{.name = "bottom", .weightData = seedS, .biasData = seedSBias},
                             {.name = "top", .weightData = seedT, .biasData = seedTBias}},
        2);

    tensor_t *x1 = makeRowVecN((float[]){1.0f, 1.0f, 1.0f, 1.0f}, 4);
    tensor_t *label1 = makeRowVecN((float[]){1.0f, 0.0f}, 2);
    tensor_t *x2 = makeRowVecN((float[]){1.0f, 1.0f, 1.0f, 1.0f}, 4);
    tensor_t *label2 = makeRowVecN((float[]){1.0f, 0.0f}, 2);

    lossConfig_t lossConfig = {
        .funcType = MSE, .backwardReduction = REDUCTION_SUM, .classWeights = NULL};
    trainingStats_t *stats1 =
        calculateGradsSequential(m1, 3, lossConfig, REDUCTION_SUM, x1, label1, NULL);
    trainingStats_t *stats2 =
        calculateGradsSequential(m2, 3, lossConfig, REDUCTION_SUM, x2, label2, NULL);

    float *wg1 = (float *)getGradFromParameter(m1Top->config->linear->weights)->data;
    float *wg2 = (float *)getGradFromParameter(m2Top->config->linear->weights)->data;
    float *bg1 = (float *)getGradFromParameter(m1Top->config->linear->bias)->data;
    float *bg2 = (float *)getGradFromParameter(m2Top->config->linear->bias)->data;

    /* CAPTURE (before any free touches the layers). */
    bool weightGradsIdentical = memcmp(wg1, wg2, 8 * sizeof(float)) == 0;
    bool biasGradsIdentical = memcmp(bg1, bg2, 2 * sizeof(float)) == 0;

    freeTrainingStats(stats1);
    freeTrainingStats(stats2);
    freeTensor(x1);
    freeTensor(label1);
    freeTensor(x2);
    freeTensor(label2);
    freeLinearLayer(m1Bottom);
    freeReluLayer(m1Relu);
    freeLinearLayer(m1Top);
    freeLinearLayer(m2Bottom);
    freeReluLayer(m2Relu);
    freeLinearLayer(m2Top);

    TEST_ASSERT_TRUE_MESSAGE(weightGradsIdentical,
                             "top layer's weight grad must be byte-identical regardless of "
                             "whether the layer below it is frozen (truncated) or trainable "
                             "(full backward)");
    TEST_ASSERT_TRUE_MESSAGE(biasGradsIdentical,
                             "top layer's bias grad must be byte-identical regardless of "
                             "whether the layer below it is frozen (truncated) or trainable "
                             "(full backward)");
}

/* ── #380 PR2 Task 2: all-frozen model skips backward entirely ──
 * model: [frozen Linear(4->2)], MSE loss. No layer trains, so
 * deepestTrainableIndex returns the sentinel (modelSize) and the entire
 * backward block -- including the loss-gradient seed -- must be skipped: no
 * "lossgrad", no "agrad". The forward pass and loss must still run (finite
 * loss value), and nothing may crash. */
void testAllFrozenModelSkipsBackwardEntirely(void) {
    g_eventCount = 0;
    layerQuant_t lq;
    layerQuantInitUniform(&lq, quantizationInitFloat());

    const rngKey_t initKey = rngFold(rngKeyFromSeed(1u), RNG_TAG_INIT, 0u);
    layer_t *frozenL = linearLayerInit(&(linearInit_t){.inFeatures = 4,
                                                       .outFeatures = 2,
                                                       .trainable = TRAINABLE_FALSE,
                                                       .weightInit.key = &initKey},
                                       &lq);
    layer_t *model[1] = {frozenL};

    float W[8] = {0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f};
    float B[2] = {0.0f, 0.0f};
    modelLoadStateDict(model, 1,
                       (stateDictEntry_t[]){{.name = "frozen", .weightData = W, .biasData = B}}, 1);

    tensor_t *x = makeRowVecN((float[]){1.0f, 1.0f, 1.0f, 1.0f}, 4);
    tensor_t *label = makeRowVecN((float[]){1.0f, 0.0f}, 2);

    trainingStats_t *stats = tracedGrads(
        model, 1,
        (lossConfig_t){.funcType = MSE, .backwardReduction = REDUCTION_SUM, .classWeights = NULL},
        REDUCTION_SUM, x, label, recordingSink, NULL, NULL);

    size_t backwardEvents = 0;
    for (size_t i = 0; i < g_eventCount; i++) {
        if (strcmp(g_events[i].phase, "agrad") == 0 || strcmp(g_events[i].phase, "lossgrad") == 0) {
            backwardEvents++;
        }
    }

    /* CAPTURE (before any free touches the layers). */
    bool lossFinite = isfinite(stats->loss);
    size_t capturedBackwardEvents = backwardEvents;

    freeTrainingStats(stats);
    freeTensor(x);
    freeTensor(label);
    freeLinearLayer(frozenL);

    TEST_ASSERT_TRUE_MESSAGE(lossFinite, "loss must still be computed for an all-frozen model");
    TEST_ASSERT_EQUAL_size_t_MESSAGE(0, capturedBackwardEvents,
                                     "all-frozen model must fire zero backward events (no "
                                     "lossgrad, no agrad)");
}

static tensor_t *makeFloatTensor3D(size_t d0, size_t d1, size_t d2, const float *vals) {
    size_t *dims = reserveMemory(3 * sizeof(size_t));
    dims[0] = d0;
    dims[1] = d1;
    dims[2] = d2;
    size_t *order = reserveMemory(3 * sizeof(size_t));
    setOrderOfDimsForNewTensor(3, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 3, order);
    tensor_t *t = initTensor(shape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(t, vals, d0 * d1 * d2);
    return t;
}

/* #152 PR1/PR3b: the training entry point allocates its wires from the
 * runtime input (the remat wire table), but the MaxPool argmax is the factory's
 * config-owned [1, C, Lout] buffer. PR 1 made a stacked batch-2 input die in
 * the forward's argmax shape guard; PR 3b grows the argmax on demand instead.
 * The batch-1 call on the same model runs first. */
void testCalculateGradsFactoryMaxPoolGrowsToBatch2(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);
    layer_t *pool = maxPool1dLayerInit(
        &(maxPool1dInit_t){.kernelSize = 2, .stride = 2, .inputChannels = 2, .inputLength = 4},
        &lq);
    layer_t *model[1] = {pool};
    lossConfig_t lossConfig = defaultLossConfig(MSE);
    tensor_t *x1 =
        makeFloatTensor3D(1, 2, 4, (float[]){0.5f, -1.f, 2.f, 0.25f, -0.75f, 1.5f, 0.f, -2.f});
    tensor_t *label1 = makeFloatTensor3D(1, 2, 2, (float[]){0.f, 0.f, 0.f, 0.f});
    tensor_t *x2 = makeFloatTensor3D(2, 2, 4,
                                     (float[]){0.5f, -1.f, 2.f, 0.25f, -0.75f, 1.5f, 0.f, -2.f, 3.f,
                                               1.f, -0.5f, -0.25f, 0.75f, 0.5f, -1.5f, 1.25f});
    tensor_t *label2 =
        makeFloatTensor3D(2, 2, 2, (float[]){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f});

    trainingStats_t *stats =
        calculateGradsSequential(model, 1, lossConfig, REDUCTION_MEAN, x1, label1, NULL);

    /* CAPTURE. */
    float capturedLoss = stats->loss;
    freeTrainingStats(stats);

    /* #152 PR3b: the stacked batch-2 input grows the argmax. */
    trainingStats_t *stats2 =
        calculateGradsSequential(model, 1, lossConfig, REDUCTION_MEAN, x2, label2, NULL);
    float capturedLoss2 = stats2->loss;
    freeTrainingStats(stats2);

    /* FREE. */
    freeTensor(label2);
    freeTensor(x2);
    freeTensor(label1);
    freeTensor(x1);
    freeMaxPool1dLayer(pool);
    freeQuantization(q);

    /* ASSERT: pooled {0.5, 2, 1.5, 0} vs zeros -> (0.25 + 4 + 2.25 + 0) / 4. */
    TEST_ASSERT_EQUAL_FLOAT(1.625f, capturedLoss);
    /* Batch 2 adds row 1 {3, -0.25, 0.75, 1.25}: (6.5 + 11.1875) / 8. */
    TEST_ASSERT_EQUAL_FLOAT(2.2109375f, capturedLoss2);
}

/* ---- the layer keys the driver hands out (#4) ---- */

/* Expected words from the Python mirror (LAYER spelled 3), repository root:
 *   uv run python -c "from examples._shared.keyed_rng import (fold,
 *   key_from_seed as kfs); c = kfs(0xd21f); print([[hex(w) for w in fold(c, 3, i)]
 *   for i in (0, 1, 2)])"
 * Each check also compares with the explicit rngFold chain. */
#define DRIVER_CALL_KEY_SEED 0xd21fu
#define DRIVER_EXEC_RECORDS 12

typedef struct {
    bool hasExec;
    rngKey_t key;
} execRecord_t;

static execRecord_t g_execRecords[DRIVER_EXEC_RECORDS];
static size_t g_execRecordCount;
static layerFunctions_t g_savedLinear;
static layerFunctions_t g_savedSoftmax;
static lossFunctions_t g_savedCrossEntropy;

static void recordExec(const layerExec_t *exec) {
    if (g_execRecordCount < DRIVER_EXEC_RECORDS) {
        g_execRecords[g_execRecordCount].hasExec = exec != NULL;
        if (exec != NULL) {
            g_execRecords[g_execRecordCount].key = exec->key;
        }
    }
    g_execRecordCount++;
}

static void spyForward(layer_t *layer, tensor_t *in, tensor_t *out, const layerExec_t *exec) {
    recordExec(exec);
    (layer->type == LINEAR ? g_savedLinear : g_savedSoftmax).forward(layer, in, out, exec);
}

static void spyBackward(layer_t *layer, tensor_t *forwardInput, tensor_t *loss, tensor_t *propLoss,
                        const layerExec_t *exec) {
    recordExec(exec);
    g_savedLinear.backward(layer, forwardInput, loss, propLoss, exec);
}

static float spyLossForward(tensor_t *out, tensor_t *label, reduction_t reduction,
                            const layerExec_t *exec) {
    recordExec(exec);
    return g_savedCrossEntropy.forward(out, label, reduction, exec);
}

static void spyLossBackward(tensor_t *out, tensor_t *label, tensor_t *result,
                            const layerExec_t *exec) {
    recordExec(exec);
    g_savedCrossEntropy.backward(out, label, result, exec);
}

static void installExecSpies(void) {
    g_savedLinear = layerFunctions[LINEAR];
    g_savedSoftmax = layerFunctions[SOFTMAX];
    g_savedCrossEntropy = lossFunctions[CROSS_ENTROPY];
    layerFunctions[LINEAR].forward = spyForward;
    layerFunctions[LINEAR].backward = spyBackward;
    layerFunctions[SOFTMAX].forward = spyForward;
    lossFunctions[CROSS_ENTROPY].forward = spyLossForward;
    lossFunctions[CROSS_ENTROPY].backward = spyLossBackward;
    g_execRecordCount = 0;
}

static void restoreExecSpies(void) {
    layerFunctions[LINEAR] = g_savedLinear;
    layerFunctions[SOFTMAX] = g_savedSoftmax;
    lossFunctions[CROSS_ENTROPY] = g_savedCrossEntropy;
}

static void assertLayerExec(size_t record, rngKey_t callKey, uint32_t layer, uint32_t w0,
                            uint32_t w1) {
    rngKey_t expected = rngFold(callKey, RNG_TAG_LAYER, layer);
    TEST_ASSERT_TRUE(g_execRecords[record].hasExec);
    TEST_ASSERT_EQUAL_HEX32(expected.w[0], g_execRecords[record].key.w[0]);
    TEST_ASSERT_EQUAL_HEX32(expected.w[1], g_execRecords[record].key.w[1]);
    TEST_ASSERT_EQUAL_HEX32(w0, g_execRecords[record].key.w[0]);
    TEST_ASSERT_EQUAL_HEX32(w1, g_execRecords[record].key.w[1]);
}

/* testCalculateGradsSequentialClosedForm's Linear + Softmax under CE, run
 * once with `call`; the spies see every forward, backward and loss call. */
static void runSpiedGradsCall(const trainingCall_t *call) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);
    layer_t *model[2];
    const rngKey_t initKey0 = rngFold(rngKeyFromSeed(1u), RNG_TAG_INIT, 0u);
    model[0] = linearLayerInit(
        &(linearInit_t){.inFeatures = 2, .outFeatures = 2, .weightInit.key = &initKey0}, &lq);
    model[1] = softmaxLayerInit(&lq);
    float W[4] = {0.1f, 0.2f, 0.3f, 0.4f}, B[2] = {0};
    modelLoadStateDict(model, 2,
                       (stateDictEntry_t[]){{.name = "fc", .weightData = W, .biasData = B}}, 1);
    tensor_t *x = makeRowVec2(1.0f, 1.0f);
    tensor_t *label = makeRowVec2(1.0f, 0.0f);
    installExecSpies();

    trainingStats_t *stats = calculateGradsSequential(model, 2, defaultLossConfig(CROSS_ENTROPY),
                                                      REDUCTION_MEAN, x, label, call);

    restoreExecSpies();
    freeTrainingStats(stats);
    freeTensor(x);
    freeTensor(label);
    freeLinearLayer(model[0]);
    freeSoftmaxLayer(model[1]);
    freeQuantization(q);
}

/* Every step gets the call key folded with its checked layer index; the loss
 * steps are layer n. Order: forward 0, forward 1, loss forward, loss
 * backward, backward 0 (the Softmax backward is skipped under CE). */
void testCalculateGradsHandsEachStepTheLayerKeyOfItsIndex(void) {
    const rngKey_t callKey = rngKeyFromSeed(DRIVER_CALL_KEY_SEED);

    runSpiedGradsCall(&(trainingCall_t){.key = &callKey});

    TEST_ASSERT_EQUAL_size_t(5, g_execRecordCount);
    assertLayerExec(0, callKey, 0u, 0x6842c650u, 0xaf1109a3u);
    assertLayerExec(1, callKey, 1u, 0x47477cedu, 0xab1bacc3u);
    assertLayerExec(2, callKey, 2u, 0x626020c6u, 0x288f25e1u);
    assertLayerExec(3, callKey, 2u, 0x626020c6u, 0x288f25e1u);
    assertLayerExec(4, callKey, 0u, 0x6842c650u, 0xaf1109a3u);
}

/* Without a key (a NULL call, or a call without .key) no step gets a context. */
void testCalculateGradsWithoutAKeyHandsNoLayerContext(void) {
    runSpiedGradsCall(NULL);
    TEST_ASSERT_EQUAL_size_t(5, g_execRecordCount);
    for (size_t r = 0; r < 5; r++) {
        TEST_ASSERT_FALSE(g_execRecords[r].hasExec);
    }
    runSpiedGradsCall(&(trainingCall_t){.remat = NULL});
    TEST_ASSERT_EQUAL_size_t(5, g_execRecordCount);
    for (size_t r = 0; r < 5; r++) {
        TEST_ASSERT_FALSE(g_execRecords[r].hasExec);
    }
}

#define ORDER_HIDDEN 16
#define ORDER_SEED 0x0dd5u

typedef struct {
    quantization_t *floatQ;
    quantization_t *wireQ;
    layer_t *model[2];
    tensor_t *x;
    tensor_t *label;
} orderModel_t;

static tensor_t *makeSymRow(size_t n) {
    size_t *dims = reserveMemory(2 * sizeof(size_t));
    dims[0] = 1;
    dims[1] = n;
    size_t *order = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 2, order);
    return initTensor(shape, quantizationInitSymInt32WithBits(HALF_AWAY, 8), NULL);
}

/* Linear(3 -> 16) rounding its forward SR into an 8-bit SYM_INT32 wire, then
 * Linear(16 -> 2) rounding its dx SR into the same kind of wire: one draw per
 * element in layer 0's forward and in layer 1's backward. */
static void buildOrderModel(orderModel_t *m) {
    m->floatQ = quantizationInitFloat();
    m->wireQ = quantizationInitSymInt32WithBits(HALF_AWAY, 8);
    const arithmetic_t sr = {.type = ARITH_FLOAT32, .roundingMode = SR_HALF_AWAY};
    layerQuant_t lq0;
    layerQuantInitUniform(&lq0, m->floatQ);
    lq0.forwardMath = sr;
    lq0.outputQ = m->wireQ;
    const rngKey_t initKey0 = rngFold(rngKeyFromSeed(ORDER_SEED), RNG_TAG_INIT, 0u);
    const rngKey_t initKey1 = rngFold(rngKeyFromSeed(ORDER_SEED), RNG_TAG_INIT, 1u);
    m->model[0] = linearLayerInit(
        &(linearInit_t){.inFeatures = 3, .outFeatures = ORDER_HIDDEN, .weightInit.key = &initKey0},
        &lq0);
    layerQuant_t lq1;
    layerQuantInitUniform(&lq1, m->floatQ);
    lq1.propLossMath = sr;
    lq1.propLossQ = m->wireQ;
    m->model[1] = linearLayerInit(
        &(linearInit_t){.inFeatures = ORDER_HIDDEN, .outFeatures = 2, .weightInit.key = &initKey1},
        &lq1);
    m->x = makeRowVecN((float[]){0.5f, -1.f, 2.f}, 3);
    m->label = makeRowVec2(0.3f, -0.2f);
}

static void freeOrderModel(orderModel_t *m) {
    freeTensor(m->label);
    freeTensor(m->x);
    freeLinearLayer(m->model[1]);
    freeLinearLayer(m->model[0]);
    freeQuantization(m->wireQ);
    freeQuantization(m->floatQ);
}

#define ORDER_BYTES 512

static size_t appendTensorBytes(uint8_t *out, size_t used, tensor_t *t) {
    size_t n = calcBytesPerTensor(t);
    TEST_ASSERT_TRUE(used + n <= ORDER_BYTES);
    memcpy(out + used, t->data, n);
    return used + n;
}

/* The loss, the output and every grad of one call. */
static size_t captureCallResult(orderModel_t *m, trainingStats_t *stats, uint8_t *out) {
    memcpy(out, &stats->loss, sizeof(float));
    size_t used = appendTensorBytes(out, sizeof(float), stats->output);
    for (size_t l = 0; l < 2; l++) {
        linearConfig_t *cfg = m->model[l]->config->linear;
        used = appendTensorBytes(out, used, getGradFromParameter(cfg->weights));
        used = appendTensorBytes(out, used, getGradFromParameter(cfg->bias));
    }
    return used;
}

/* Copies of the wires a standalone run needs, plus an unrelated keyed draw at
 * every trace event. */
typedef struct {
    tensor_t *forward0; /* layer 0's output = layer 1's forward input */
    tensor_t *gradIn1;  /* the loss grad entering layer 1's backward */
    tensor_t *dx1;      /* layer 1's dx = agrad@0 */
    rngStream_t unrelated;
} orderCapture_t;

static void interposingSink(void *ctx, size_t layerIdx, layerType_t type, const char *phase,
                            tensor_t *t) {
    (void)type;
    orderCapture_t *c = ctx;
    (void)rngNextUnitFloat(&c->unrelated);
    if (layerIdx == 0 && strcmp(phase, "fwd") == 0) {
        copyTensor(c->forward0, t);
    } else if (layerIdx == 1 && strcmp(phase, "agrad") == 0) {
        copyTensor(c->gradIn1, t);
    } else if (layerIdx == 0 && strcmp(phase, "agrad") == 0) {
        copyTensor(c->dx1, t);
    }
}

static void assertSameSymWire(tensor_t *expected, tensor_t *actual, const char *what) {
    TEST_ASSERT_EQUAL_MEMORY_MESSAGE(expected->data, actual->data, calcBytesPerTensor(expected),
                                     what);
    TEST_ASSERT_EQUAL_MEMORY_MESSAGE(&((symInt32QConfig_t *)expected->quantization->qConfig)->scale,
                                     &((symInt32QConfig_t *)actual->quantization->qConfig)->scale,
                                     sizeof(float), what);
}

/* Draws depend on the call key and the layer's index only: draws interposed
 * between the steps change nothing, and layer 0's forward and layer 1's
 * backward, each run standalone under its layer key, reproduce the in-call
 * wires. */
void testCalculateGradsIsIndependentOfDrawsAroundItsSteps(void) {
    const rngKey_t callKey = rngKeyFromSeed(DRIVER_CALL_KEY_SEED);
    uint8_t plain[ORDER_BYTES];
    uint8_t interposed[ORDER_BYTES];

    orderModel_t m;
    buildOrderModel(&m);
    trainingStats_t *stats =
        calculateGradsSequential(m.model, 2, defaultLossConfig(MSE), REDUCTION_MEAN, m.x, m.label,
                                 &(trainingCall_t){.key = &callKey});
    size_t plainBytes = captureCallResult(&m, stats, plain);
    freeTrainingStats(stats);
    freeOrderModel(&m);

    buildOrderModel(&m);
    orderCapture_t cap = {.forward0 = makeSymRow(ORDER_HIDDEN),
                          .gradIn1 = makeRowVec2(0.f, 0.f),
                          .dx1 = makeSymRow(ORDER_HIDDEN),
                          .unrelated = rngOpen(rngKeyFromSeed(0xfeedu))};
    stats = tracedGrads(m.model, 2, defaultLossConfig(MSE), REDUCTION_MEAN, m.x, m.label,
                        interposingSink, &cap, &(trainingCall_t){.key = &callKey});
    size_t interposedBytes = captureCallResult(&m, stats, interposed);
    freeTrainingStats(stats);

    layerExec_t exStorage;
    tensor_t *forward0 = makeSymRow(ORDER_HIDDEN);
    layerFunctions[LINEAR].forward(m.model[0], m.x, forward0,
                                   layerExecForLayer(&callKey, 0, &exStorage));
    tensor_t *dx1 = makeSymRow(ORDER_HIDDEN);
    layerFunctions[LINEAR].backward(m.model[1], cap.forward0, cap.gradIn1, dx1,
                                    layerExecForLayer(&callKey, 1, &exStorage));

    TEST_ASSERT_EQUAL_size_t(plainBytes, interposedBytes);
    TEST_ASSERT_EQUAL_MEMORY_MESSAGE(plain, interposed, plainBytes,
                                     "draws around the steps must not move the call");
    assertSameSymWire(cap.forward0, forward0, "layer 0's standalone forward");
    assertSameSymWire(cap.dx1, dx1, "layer 1's standalone backward");
    freeTensor(dx1);
    freeTensor(forward0);
    freeTensor(cap.dx1);
    freeTensor(cap.gradIn1);
    freeTensor(cap.forward0);
    freeOrderModel(&m);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testCalculateGradsSequentialClosedForm);
    RUN_TEST(testTracedGradsFiresInOrder);
    RUN_TEST(testTraceModelParamsFiresPerTrainableParam);
    RUN_TEST(testTraceModelGradsSkipsFrozenLayerNeverPassesNull);
    RUN_TEST(testDxWireHonorsProducerPropLossQ);
    RUN_TEST(testDxWireHonorsProducerPropLossQMaxBits);
    RUN_TEST(testForwardWireHonorsDeclaredOutputQMaxBits);
    RUN_TEST(testFrozenLayerSurvivesTrainingUntouched);
    RUN_TEST(testBackwardStopsAtDeepestTrainableLayer);
    RUN_TEST(testTruncationPreservesUpperLayerGrads);
    RUN_TEST(testAllFrozenModelSkipsBackwardEntirely);
    RUN_TEST(testCalculateGradsFactoryMaxPoolGrowsToBatch2);
    RUN_TEST(testCalculateGradsHandsEachStepTheLayerKeyOfItsIndex);
    RUN_TEST(testCalculateGradsWithoutAKeyHandsNoLayerContext);
    RUN_TEST(testCalculateGradsIsIndependentOfDrawsAroundItsSteps);
    return UNITY_END();
}
