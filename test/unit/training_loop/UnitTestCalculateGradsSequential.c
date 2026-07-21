#define SOURCE_FILE "UnitTestCalculateGradsSequential"

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
#include "LayerQuant.h"
#include "Linear.h"
#include "LinearApi.h"
#include "OptimizerApi.h"
#include "QuantizationApi.h"
#include "QuantizationLayer.h"
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

/* Structural note: tracedGrads and calculateGradsSequential both call calculateGradsImpl
 * internally; npyDumpSink (and any other sink) observes tensors but does not mutate them.
 * This means the closed-form characterisation test pins both paths simultaneously. */
void testCalculateGradsSequentialClosedForm() {
    layerQuant_t lq;
    layerQuantInitUniform(&lq, quantizationInitFloat());

    layer_t *model[2];
    model[0] = linearLayerInit(&(linearInit_t){.inFeatures = 2, .outFeatures = 2}, &lq);
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
        REDUCTION_MEAN, x, label);

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
    model[0] = linearLayerInit(&(linearInit_t){.inFeatures = 2, .outFeatures = 2}, &lq);
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
                                         REDUCTION_MEAN, x, label, recordingSink, NULL);

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
    model[0] = linearLayerInit(&(linearInit_t){.inFeatures = 2, .outFeatures = 2}, &lq);
    model[1] = softmaxLayerInit(&lq);
    float W[4] = {0.1f, 0.2f, 0.3f, 0.4f}, B[2] = {0};
    modelLoadStateDict(model, 2,
                       (stateDictEntry_t[]){{.name = "fc", .weightData = W, .biasData = B}}, 1);
    tensor_t *x = makeRowVec2(1.0f, 1.0f), *label = makeRowVec2(1.0f, 0.0f);
    trainingStats_t *stats = calculateGradsSequential(
        model, 2,
        (lossConfig_t){
            .funcType = CROSS_ENTROPY, .backwardReduction = REDUCTION_MEAN, .classWeights = NULL},
        REDUCTION_MEAN, x, label);

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

/* #380 final-review Fix 1: traceModelGrads must never hand a NULL tensor to
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
    model[0] = linearLayerInit(
        &(linearInit_t){.inFeatures = 2, .outFeatures = 2, .trainable = TRAINABLE_FALSE}, &lq);
    model[1] = linearLayerInit(&(linearInit_t){.inFeatures = 2, .outFeatures = 2}, &lq);

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
    /* PR3 spec D1: today's per-callsite hardcodes (linearBackward); hand-wired
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
     * quantization via initGradTensor, ignoring Linear1's propLossQ).
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

    agradCapCtx_t ctx = {.capturedAgrad = false};
    trainingStats_t *stats = tracedGrads(
        model, 4,
        (lossConfig_t){.funcType = MSE, .backwardReduction = REDUCTION_SUM, .classWeights = NULL},
        REDUCTION_SUM, x, label, agradCaptureSink, &ctx);

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
     * Pre-fix: initGradTensor's SYM arm re-defaults the dx wire to 12 regardless
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
        REDUCTION_SUM, x, label, agradBitsCaptureSink, &ctx);

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
     * than the int12 operand default). Pre-fix: initLayerOutputs' SYM arm
     * re-defaults the forward wire to 12, discarding the declared width.
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
        REDUCTION_SUM, x, label, fwdBitsCaptureSink, &ctx);

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

    layer_t *frozenL = linearLayerInit(
        &(linearInit_t){.inFeatures = 4, .outFeatures = 4, .trainable = TRAINABLE_FALSE}, &lq);
    layer_t *reluL = reluLayerInit(&lq);
    layer_t *trainL = linearLayerInit(&(linearInit_t){.inFeatures = 4, .outFeatures = 2}, &lq);
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
            calculateGradsSequential(model, 3, lossConfig, REDUCTION_SUM, input, label);
        if (step == 0) {
            firstLoss = stats->loss;
        }
        lastLoss = stats->loss;
        freeTrainingStats(stats);
        optimFns.step(optim);
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
    return UNITY_END();
}
