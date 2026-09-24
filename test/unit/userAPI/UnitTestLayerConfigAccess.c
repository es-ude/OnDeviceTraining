#define SOURCE_FILE "UNIT_TEST_LAYER_CONFIG_ACCESS"

#include <stdbool.h>
#include <stddef.h>
#include <string.h>

#include "AdaptiveAvgPool1d.h"
#include "AdaptivePool1dApi.h"
#include "ArithmeticType.h"
#include "AvgPool1d.h"
#include "Conv1d.h"
#include "Conv1dApi.h"
#include "Conv1dTransposed.h"
#include "Conv1dTransposedApi.h"
#include "Dropout.h"
#include "DropoutApi.h"
#include "FlattenApi.h"
#include "GroupNorm.h"
#include "GroupNormApi.h"
#include "Layer.h"
#include "LayerConfigAccess.h"
#include "LayerNorm.h"
#include "LayerNormApi.h"
#include "LayerQuant.h"
#include "Linear.h"
#include "LinearApi.h"
#include "MaxPool1d.h"
#include "Pool1dApi.h"
#include "QuantLayerApi.h"
#include "QuantizationApi.h"
#include "Relu.h"
#include "ReluApi.h"
#include "Softmax.h"
#include "SoftmaxApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

/* Default arithmetic for layers with no consumed arithmetic (Flatten,
 * Quantization — D4) and the shared arithmetic every uniform-FLOAT32-config
 * layer below derives (all uniform-profile fixtures in this file — every
 * fixture but LINEAR, which uses a divergent SYM_INT32 profile to make the
 * accessors discriminating, see testLinearAccessorsMatchConfig — use a
 * single float quantization_t, so forwardMath is always this same value for
 * real arithmetic-bearing layers too). */
static void assertUniformArithmetic(arithmetic_t a) {
    TEST_ASSERT_EQUAL(ARITH_FLOAT32, a.type);
    TEST_ASSERT_EQUAL(HALF_AWAY, a.roundingMode);
}

static tensor_t *buildBoolMask(size_t n) {
    size_t *dims = reserveMemory(sizeof(size_t));
    dims[0] = n;
    size_t *order = reserveMemory(sizeof(size_t));
    setOrderOfDimsForNewTensor(1, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 1, order);
    return initTensor(shape, quantizationInitBool(), NULL);
}

/* Discriminating fixture (mirrors #221's testDxWireHonorsProducerPropLossQ in
 * UnitTestCalculateGradsSequential.c): forwardMath (-> outputQ) and
 * backwardMath (-> propLossQ) are two DISTINCT SYM_INT32 objects, so this
 * test only passes if layerOutputQ/backwardWireQ each read their own arm —
 * every other fixture below shares one quantization_t across both arms and
 * would pass even if the LINEAR case of one accessor returned the other's
 * config. */
void testLinearAccessorsMatchConfig(void) {
    /* weightStorage/biasStorage stay FLOAT32: initWeightTensor/initBiasTensor
     * (PyTorch-parity random init, called unconditionally by linearLayerInit)
     * currently require FLOAT32 storage (LAYER_COMMON: requireFloat32) — a
     * pre-existing constraint orthogonal to what this fixture exercises
     * (outputQ/propLossQ pointer routing), so weight/bias storage is kept
     * real but out of the way of the forwardMath/backwardMath divergence. */
    quantization_t *qStorage = quantizationInitFloat();
    quantization_t *qA = quantizationInitSymInt32(HALF_AWAY);
    quantization_t *qB = quantizationInitSymInt32(SR_HALF_AWAY);
    layerQuant_t lq = {
        .forwardMath = arithmeticFromQuantization(qA),
        .weightGradMath = arithmeticFromQuantization(qB),
        .biasGradMath = arithmeticFromQuantization(qB),
        .propLossMath = arithmeticFromQuantization(qB),
        .outputQ = qA,
        .propLossQ = qB,
        .weightStorage = qStorage,
        .biasStorage = qStorage,
    };
    layer_t *layer = linearLayerInit(&(linearInit_t){.inFeatures = 1, .outFeatures = 1}, &lq);

    linearConfig_t *cfg = layer->config->linear;
    TEST_ASSERT_EQUAL_PTR(cfg->outputQ, layerOutputQ(layer));
    TEST_ASSERT_EQUAL_PTR(cfg->propLossQ, backwardWireQ(layer));
    TEST_ASSERT_EQUAL_PTR(qA, layerOutputQ(layer));
    TEST_ASSERT_EQUAL_PTR(qB, backwardWireQ(layer));
    arithmetic_t fm = layerForwardMath(layer);
    TEST_ASSERT_EQUAL(ARITH_SYM_INT32, fm.type);
    TEST_ASSERT_EQUAL(HALF_AWAY, fm.roundingMode);

    freeLinearLayer(layer);
    freeQuantization(qStorage);
    freeQuantization(qA);
    freeQuantization(qB);
}

void testReluAccessorsMatchConfig(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);
    layer_t *layer = reluLayerInit(&lq);

    reluConfig_t *cfg = layer->config->relu;
    TEST_ASSERT_EQUAL_PTR(cfg->outputQ, layerOutputQ(layer));
    TEST_ASSERT_EQUAL_PTR(cfg->propLossQ, backwardWireQ(layer));
    assertUniformArithmetic(layerForwardMath(layer));

    freeReluLayer(layer);
    freeQuantization(q);
}

void testConv1dAccessorsMatchConfig(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);
    layer_t *layer =
        conv1dLayerInit(&(conv1dInit_t){.inChannels = 1, .outChannels = 1, .kernelSize = 1}, &lq);

    conv1dConfig_t *cfg = layer->config->conv1d;
    TEST_ASSERT_EQUAL_PTR(cfg->outputQ, layerOutputQ(layer));
    TEST_ASSERT_EQUAL_PTR(cfg->propLossQ, backwardWireQ(layer));
    assertUniformArithmetic(layerForwardMath(layer));

    freeConv1dLayer(layer);
    freeQuantization(q);
}

void testConv1dTransposedAccessorsMatchConfig(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);
    layer_t *layer = conv1dTransposedLayerInit(
        &(conv1dTransposedInit_t){.inChannels = 1, .outChannels = 1, .kernelSize = 1, .stride = 1},
        &lq);

    conv1dTransposedConfig_t *cfg = layer->config->conv1dTransposed;
    TEST_ASSERT_EQUAL_PTR(cfg->outputQ, layerOutputQ(layer));
    TEST_ASSERT_EQUAL_PTR(cfg->propLossQ, backwardWireQ(layer));
    assertUniformArithmetic(layerForwardMath(layer));

    freeConv1dTransposedLayer(layer);
    freeQuantization(q);
}

void testMaxPool1dAccessorsMatchConfig(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);
    layer_t *layer = maxPool1dLayerInit(
        &(maxPool1dInit_t){.kernelSize = 1, .inputChannels = 1, .inputLength = 1}, &lq);

    maxPool1dConfig_t *cfg = layer->config->maxPool1d;
    TEST_ASSERT_EQUAL_PTR(cfg->outputQ, layerOutputQ(layer));
    TEST_ASSERT_EQUAL_PTR(cfg->propLossQ, backwardWireQ(layer));
    assertUniformArithmetic(layerForwardMath(layer));

    freeMaxPool1dLayer(layer);
    freeQuantization(q);
}

void testAvgPool1dAccessorsMatchConfig(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);
    layer_t *layer = avgPool1dLayerInit(&(avgPool1dInit_t){.kernelSize = 1}, &lq);

    avgPool1dConfig_t *cfg = layer->config->avgPool1d;
    TEST_ASSERT_EQUAL_PTR(cfg->outputQ, layerOutputQ(layer));
    TEST_ASSERT_EQUAL_PTR(cfg->propLossQ, backwardWireQ(layer));
    assertUniformArithmetic(layerForwardMath(layer));

    freeAvgPool1dLayer(layer);
    freeQuantization(q);
}

void testAdaptiveAvgPool1dAccessorsMatchConfig(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);
    layer_t *layer = adaptiveAvgPool1dLayerInit(&(adaptiveAvgPool1dInit_t){.outputSize = 1}, &lq);

    adaptiveAvgPool1dConfig_t *cfg = layer->config->adaptiveAvgPool1d;
    TEST_ASSERT_EQUAL_PTR(cfg->outputQ, layerOutputQ(layer));
    TEST_ASSERT_EQUAL_PTR(cfg->propLossQ, backwardWireQ(layer));
    assertUniformArithmetic(layerForwardMath(layer));

    freeAdaptiveAvgPool1dLayer(layer);
    freeQuantization(q);
}

void testSoftmaxAccessorsMatchConfig(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);
    layer_t *layer = softmaxLayerInit(&lq);

    softmaxConfig_t *cfg = layer->config->softmax;
    TEST_ASSERT_EQUAL_PTR(cfg->outputQ, layerOutputQ(layer));
    TEST_ASSERT_EQUAL_PTR(cfg->propLossQ, backwardWireQ(layer));
    assertUniformArithmetic(layerForwardMath(layer));

    freeSoftmaxLayer(layer);
    freeQuantization(q);
}

void testDropoutAccessorsMatchConfig(void) {
    quantization_t *q = quantizationInitFloat();
    tensor_t *mask = buildBoolMask(4);
    layer_t *layer = dropoutLayerInit(0.5f, mask, q, q);

    dropoutConfig_t *cfg = layer->config->dropout;
    TEST_ASSERT_EQUAL_PTR(cfg->outputQ, layerOutputQ(layer));
    TEST_ASSERT_EQUAL_PTR(cfg->propLossQ, backwardWireQ(layer));
    assertUniformArithmetic(layerForwardMath(layer));

    freeDropoutLayer(layer);
    freeTensor(mask);
    freeQuantization(q);
}

void testLayerNormAccessorsMatchConfig(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);
    layer_t *layer = layerNormLayerInit(
        &(layerNormInit_t){.normalizedShape = (size_t[]){1}, .numNormDims = 1}, &lq);

    layerNormConfig_t *cfg = layer->config->layerNorm;
    TEST_ASSERT_EQUAL_PTR(cfg->outputQ, layerOutputQ(layer));
    TEST_ASSERT_EQUAL_PTR(cfg->propLossQ, backwardWireQ(layer));
    assertUniformArithmetic(layerForwardMath(layer));

    freeLayerNormLayer(layer);
    freeQuantization(q);
}

void testQuantizationAccessorsMatchConfig(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);
    layer_t *layer = quantLayerInit(&lq);

    quantizationConfig_t *cfg = layer->config->quantization;
    TEST_ASSERT_EQUAL_PTR(cfg->outputQ, layerOutputQ(layer));
    TEST_ASSERT_EQUAL_PTR(cfg->propLossQ, backwardWireQ(layer));
    /* Quantization is a pure conversion node (D4) — no consumed arithmetic. */
    assertUniformArithmetic(layerForwardMath(layer));

    freeQuantLayer(layer);
    freeQuantization(q);
}

void testFlattenAccessorsAreNullAndDefaultArithmetic(void) {
    layer_t *layer = flattenLayerInit();

    TEST_ASSERT_NULL(layerOutputQ(layer));
    TEST_ASSERT_NULL(backwardWireQ(layer));
    assertUniformArithmetic(layerForwardMath(layer));

    freeFlattenLayer(layer);
}

/* ---- #152 PR3b: FLOAT32-only gate for stacked training (spec §6.6) ---- */

_Static_assert(_Generic(&layerIsFloat32Only, bool (*)(layer_t *): 1, default: 0),
               "layerIsFloat32Only must be bool (layer_t *layer) (#152)");
_Static_assert(_Generic(&layerNonFloat32Field, const char *(*)(layer_t *): 1, default: 0),
               "layerNonFloat32Field must be const char *(layer_t *layer) (#152)");

/* The gate's field capture helpers flip ONE slot, record what the gate names,
 * and restore the slot before anything else runs -- capture only, assertions
 * after teardown (testing.md Rule 3). The returned names are string literals,
 * valid after every free. */
static void captureMathFlips(layer_t *layer, arithmetic_t *const *slots, size_t n,
                             arithmeticType_t flipTo, const char **got) {
    for (size_t i = 0; i < n; i++) {
        arithmeticType_t saved = slots[i]->type;
        slots[i]->type = flipTo;
        got[i] = layerNonFloat32Field(layer);
        slots[i]->type = saved;
    }
}

static void captureWireFlips(layer_t *layer, quantization_t **const *slots, size_t n,
                             quantization_t *nonFloatQ, const char **got) {
    for (size_t i = 0; i < n; i++) {
        quantization_t *saved = *slots[i];
        *slots[i] = nonFloatQ;
        got[i] = layerNonFloat32Field(layer);
        *slots[i] = saved;
    }
}

/* got[0] = param storage flipped, got[1] = grad storage flipped. The dtype
 * TAG is flipped in place (FLOAT32 carries no qConfig, so nothing else is
 * read) and restored before any teardown touches the tensor. */
static void captureParamFlips(layer_t *layer, parameter_t *p, const char **got) {
    qtype_t saved = p->param->quantization->type;
    p->param->quantization->type = SYM_INT32;
    got[0] = layerNonFloat32Field(layer);
    p->param->quantization->type = saved;
    saved = p->grad->quantization->type;
    p->grad->quantization->type = SYM_INT32;
    got[1] = layerNonFloat32Field(layer);
    p->grad->quantization->type = saved;
}

static void assertFieldNames(const char *const *expected, const char *const *got, size_t n) {
    for (size_t i = 0; i < n; i++) {
        TEST_ASSERT_NOT_NULL_MESSAGE(got[i], expected[i]);
        TEST_ASSERT_EQUAL_STRING(expected[i], got[i]);
    }
}

static const char *const GEMM_FIELDS[10] = {
    "forwardMath", "weightGradMath", "biasGradMath", "propLossMath", "outputQ",
    "propLossQ",   "weights.param",  "weights.grad", "bias.param",   "bias.grad"};

static const char *const NORM_FIELDS[8] = {"forwardMath", "propLossMath", "outputQ",
                                           "propLossQ",   "gamma.param",  "gamma.grad",
                                           "beta.param",  "beta.grad"};

static const char *const WIRE_FIELDS[4] = {"forwardMath", "propLossMath", "outputQ", "propLossQ"};

void testLayerIsFloat32OnlyAcceptsEveryUniformFloat32LayerType(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);
    tensor_t *mask = buildBoolMask(4);
    layer_t *layers[12] = {
        linearLayerInit(&(linearInit_t){.inFeatures = 2, .outFeatures = 2}, &lq),
        conv1dLayerInit(&(conv1dInit_t){.inChannels = 1, .outChannels = 1, .kernelSize = 1}, &lq),
        conv1dTransposedLayerInit(
            &(conv1dTransposedInit_t){.inChannels = 1, .outChannels = 1, .kernelSize = 1}, &lq),
        layerNormLayerInit(&(layerNormInit_t){.normalizedShape = (size_t[]){2}, .numNormDims = 1},
                           &lq),
        groupNormLayerInit(&(groupNormInit_t){.numGroups = 1, .numChannels = 2}, &lq),
        reluLayerInit(&lq),
        softmaxLayerInit(&lq),
        maxPool1dLayerInit(
            &(maxPool1dInit_t){.kernelSize = 1, .inputChannels = 1, .inputLength = 1}, &lq),
        avgPool1dLayerInit(&(avgPool1dInit_t){.kernelSize = 1}, &lq),
        adaptiveAvgPool1dLayerInit(&(adaptiveAvgPool1dInit_t){.outputSize = 1}, &lq),
        dropoutLayerInit(0.5f, mask, q, q),
        flattenLayerInit()};
    bool accepted[12];
    const char *field[12];
    for (size_t i = 0; i < 12; i++) {
        accepted[i] = layerIsFloat32Only(layers[i]);
        field[i] = layerNonFloat32Field(layers[i]);
    }

    freeFlattenLayer(layers[11]);
    freeDropoutLayer(layers[10]);
    freeAdaptiveAvgPool1dLayer(layers[9]);
    freeAvgPool1dLayer(layers[8]);
    freeMaxPool1dLayer(layers[7]);
    freeSoftmaxLayer(layers[6]);
    freeReluLayer(layers[5]);
    freeGroupNormLayer(layers[4]);
    freeLayerNormLayer(layers[3]);
    freeConv1dTransposedLayer(layers[2]);
    freeConv1dLayer(layers[1]);
    freeLinearLayer(layers[0]);
    freeTensor(mask);
    freeQuantization(q);

    for (size_t i = 0; i < 12; i++) {
        TEST_ASSERT_TRUE_MESSAGE(accepted[i], "uniform FLOAT32 layer must pass the gate");
        TEST_ASSERT_NULL(field[i]);
    }
}

void testLayerIsFloat32OnlyRejectsQuantizationLayer(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);
    layer_t *layer = quantLayerInit(&lq);
    bool accepted = layerIsFloat32Only(layer);
    const char *field = layerNonFloat32Field(layer);
    freeQuantLayer(layer);
    freeQuantization(q);
    TEST_ASSERT_FALSE(accepted);
    TEST_ASSERT_NOT_NULL(field);
    TEST_ASSERT_NOT_NULL_MESSAGE(strstr(field, "QUANTIZATION"),
                                 "the field must name the layer type");
}

void testLayerNonFloat32FieldNamesEveryGemmFamilyField(void) {
    quantization_t *q = quantizationInitFloat();
    quantization_t *symQ = quantizationInitSymInt32(HALF_AWAY);
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);
    layer_t *layers[3] = {
        linearLayerInit(&(linearInit_t){.inFeatures = 2, .outFeatures = 2}, &lq),
        conv1dLayerInit(&(conv1dInit_t){.inChannels = 1, .outChannels = 1, .kernelSize = 1}, &lq),
        conv1dTransposedLayerInit(
            &(conv1dTransposedInit_t){.inChannels = 1, .outChannels = 1, .kernelSize = 1}, &lq)};
    const char *got[3][10];

    linearConfig_t *lin = layers[0]->config->linear;
    captureMathFlips(layers[0],
                     (arithmetic_t *const[]){&lin->forwardMath, &lin->weightGradMath,
                                             &lin->biasGradMath, &lin->propLossMath},
                     4, ARITH_SYM_INT32, got[0]);
    captureWireFlips(layers[0], (quantization_t * *const[]){&lin->outputQ, &lin->propLossQ}, 2,
                     symQ, got[0] + 4);
    captureParamFlips(layers[0], lin->weights, got[0] + 6);
    captureParamFlips(layers[0], lin->bias, got[0] + 8);

    conv1dConfig_t *conv = layers[1]->config->conv1d;
    captureMathFlips(layers[1],
                     (arithmetic_t *const[]){&conv->forwardMath, &conv->weightGradMath,
                                             &conv->biasGradMath, &conv->propLossMath},
                     4, ARITH_SYM_INT32, got[1]);
    captureWireFlips(layers[1], (quantization_t * *const[]){&conv->outputQ, &conv->propLossQ}, 2,
                     symQ, got[1] + 4);
    captureParamFlips(layers[1], conv->weights, got[1] + 6);
    captureParamFlips(layers[1], conv->bias, got[1] + 8);

    conv1dTransposedConfig_t *convT = layers[2]->config->conv1dTransposed;
    captureMathFlips(layers[2],
                     (arithmetic_t *const[]){&convT->forwardMath, &convT->weightGradMath,
                                             &convT->biasGradMath, &convT->propLossMath},
                     4, ARITH_SYM_INT32, got[2]);
    captureWireFlips(layers[2], (quantization_t * *const[]){&convT->outputQ, &convT->propLossQ}, 2,
                     symQ, got[2] + 4);
    captureParamFlips(layers[2], convT->weights, got[2] + 6);
    captureParamFlips(layers[2], convT->bias, got[2] + 8);

    freeConv1dTransposedLayer(layers[2]);
    freeConv1dLayer(layers[1]);
    freeLinearLayer(layers[0]);
    freeQuantization(symQ);
    freeQuantization(q);

    for (size_t l = 0; l < 3; l++) {
        assertFieldNames(GEMM_FIELDS, got[l], 10);
    }
}

void testLayerNonFloat32FieldNamesEveryNormField(void) {
    quantization_t *q = quantizationInitFloat();
    quantization_t *symQ = quantizationInitSymInt32(HALF_AWAY);
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);
    layer_t *ln = layerNormLayerInit(
        &(layerNormInit_t){.normalizedShape = (size_t[]){2}, .numNormDims = 1}, &lq);
    layer_t *gn = groupNormLayerInit(&(groupNormInit_t){.numGroups = 1, .numChannels = 2}, &lq);
    const char *got[2][8];

    layerNormConfig_t *lc = ln->config->layerNorm;
    captureMathFlips(ln, (arithmetic_t *const[]){&lc->forwardMath, &lc->propLossMath}, 2,
                     ARITH_SYM_INT32, got[0]);
    captureWireFlips(ln, (quantization_t * *const[]){&lc->outputQ, &lc->propLossQ}, 2, symQ,
                     got[0] + 2);
    captureParamFlips(ln, lc->gamma, got[0] + 4);
    captureParamFlips(ln, lc->beta, got[0] + 6);

    groupNormConfig_t *gc = gn->config->groupNorm;
    captureMathFlips(gn, (arithmetic_t *const[]){&gc->forwardMath, &gc->propLossMath}, 2,
                     ARITH_SYM_INT32, got[1]);
    captureWireFlips(gn, (quantization_t * *const[]){&gc->outputQ, &gc->propLossQ}, 2, symQ,
                     got[1] + 2);
    captureParamFlips(gn, gc->gamma, got[1] + 4);
    captureParamFlips(gn, gc->beta, got[1] + 6);

    freeGroupNormLayer(gn);
    freeLayerNormLayer(ln);
    freeQuantization(symQ);
    freeQuantization(q);

    assertFieldNames(NORM_FIELDS, got[0], 8);
    assertFieldNames(NORM_FIELDS, got[1], 8);
}

void testLayerNonFloat32FieldNamesEveryWireOnlyLayerField(void) {
    quantization_t *q = quantizationInitFloat();
    quantization_t *symQ = quantizationInitSymInt32(HALF_AWAY);
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);
    tensor_t *mask = buildBoolMask(4);
    layer_t *relu = reluLayerInit(&lq);
    layer_t *softmax = softmaxLayerInit(&lq);
    layer_t *maxPool = maxPool1dLayerInit(
        &(maxPool1dInit_t){.kernelSize = 1, .inputChannels = 1, .inputLength = 1}, &lq);
    layer_t *avgPool = avgPool1dLayerInit(&(avgPool1dInit_t){.kernelSize = 1}, &lq);
    layer_t *adaptive =
        adaptiveAvgPool1dLayerInit(&(adaptiveAvgPool1dInit_t){.outputSize = 1}, &lq);
    layer_t *dropout = dropoutLayerInit(0.5f, mask, q, q);
    const char *got[6][4];

    reluConfig_t *rc = relu->config->relu;
    captureMathFlips(relu, (arithmetic_t *const[]){&rc->forwardMath, &rc->propLossMath}, 2,
                     ARITH_SYM_INT32, got[0]);
    captureWireFlips(relu, (quantization_t * *const[]){&rc->outputQ, &rc->propLossQ}, 2, symQ,
                     got[0] + 2);
    softmaxConfig_t *sc = softmax->config->softmax;
    captureMathFlips(softmax, (arithmetic_t *const[]){&sc->forwardMath, &sc->propLossMath}, 2,
                     ARITH_SYM_INT32, got[1]);
    captureWireFlips(softmax, (quantization_t * *const[]){&sc->outputQ, &sc->propLossQ}, 2, symQ,
                     got[1] + 2);
    maxPool1dConfig_t *mc = maxPool->config->maxPool1d;
    captureMathFlips(maxPool, (arithmetic_t *const[]){&mc->forwardMath, &mc->propLossMath}, 2,
                     ARITH_SYM_INT32, got[2]);
    captureWireFlips(maxPool, (quantization_t * *const[]){&mc->outputQ, &mc->propLossQ}, 2, symQ,
                     got[2] + 2);
    avgPool1dConfig_t *ac = avgPool->config->avgPool1d;
    captureMathFlips(avgPool, (arithmetic_t *const[]){&ac->forwardMath, &ac->propLossMath}, 2,
                     ARITH_SYM_INT32, got[3]);
    captureWireFlips(avgPool, (quantization_t * *const[]){&ac->outputQ, &ac->propLossQ}, 2, symQ,
                     got[3] + 2);
    adaptiveAvgPool1dConfig_t *dc = adaptive->config->adaptiveAvgPool1d;
    captureMathFlips(adaptive, (arithmetic_t *const[]){&dc->forwardMath, &dc->propLossMath}, 2,
                     ARITH_SYM_INT32, got[4]);
    captureWireFlips(adaptive, (quantization_t * *const[]){&dc->outputQ, &dc->propLossQ}, 2, symQ,
                     got[4] + 2);
    dropoutConfig_t *pc = dropout->config->dropout;
    captureMathFlips(dropout, (arithmetic_t *const[]){&pc->forwardMath, &pc->propLossMath}, 2,
                     ARITH_SYM_INT32, got[5]);
    captureWireFlips(dropout, (quantization_t * *const[]){&pc->outputQ, &pc->propLossQ}, 2, symQ,
                     got[5] + 2);

    freeDropoutLayer(dropout);
    freeAdaptiveAvgPool1dLayer(adaptive);
    freeAvgPool1dLayer(avgPool);
    freeMaxPool1dLayer(maxPool);
    freeSoftmaxLayer(softmax);
    freeReluLayer(relu);
    freeTensor(mask);
    freeQuantization(symQ);
    freeQuantization(q);

    for (size_t l = 0; l < 6; l++) {
        assertFieldNames(WIRE_FIELDS, got[l], 4);
    }
}

void testLayerIsFloat32OnlyRejectsBfpNotJustSym(void) {
    /* The gate is "== FLOAT32", not "!= SYM_INT32": BFP math and a BFP wire
     * are rejected too (D3). */
    quantization_t *q = quantizationInitFloat();
    quantization_t *bfpQ = quantizationInitBfp(8, 8, HALF_AWAY);
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);
    layer_t *relu = reluLayerInit(&lq);
    reluConfig_t *rc = relu->config->relu;
    const char *gotMath[1];
    const char *gotWire[1];
    captureMathFlips(relu, (arithmetic_t *const[]){&rc->forwardMath}, 1, ARITH_BFP, gotMath);
    captureWireFlips(relu, (quantization_t * *const[]){&rc->outputQ}, 1, bfpQ, gotWire);
    rc->forwardMath.type = ARITH_BFP;
    bool accepted = layerIsFloat32Only(relu);
    rc->forwardMath.type = ARITH_FLOAT32;

    freeReluLayer(relu);
    freeQuantization(bfpQ);
    freeQuantization(q);

    TEST_ASSERT_EQUAL_STRING("forwardMath", gotMath[0]);
    TEST_ASSERT_EQUAL_STRING("outputQ", gotWire[0]);
    TEST_ASSERT_FALSE(accepted);
}

void testLayerIsFloat32OnlyAcceptsFrozenBiaslessAndPassthroughLayers(void) {
    /* Absent storage is not non-FLOAT32 storage: a frozen layer carries no
     * grad tensors (#380), a bias-less conv no bias parameter, and a NULL
     * wire config is the upstream-dtype passthrough (initLayerOutputs). */
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);
    layer_t *frozen = linearLayerInit(
        &(linearInit_t){.inFeatures = 2, .outFeatures = 2, .trainable = TRAINABLE_FALSE}, &lq);
    layer_t *biasless = conv1dLayerInit(
        &(conv1dInit_t){.inChannels = 1, .outChannels = 1, .kernelSize = 1, .bias = BIAS_FALSE},
        &lq);
    layer_t *relu = reluLayerInit(&lq);
    relu->config->relu->outputQ = NULL;
    bool frozenGradIsNull = frozen->config->linear->weights->grad == NULL;
    bool biasIsNull = biasless->config->conv1d->bias == NULL;
    bool acceptedFrozen = layerIsFloat32Only(frozen);
    bool acceptedBiasless = layerIsFloat32Only(biasless);
    bool acceptedPassthrough = layerIsFloat32Only(relu);
    relu->config->relu->outputQ = q;

    freeReluLayer(relu);
    freeConv1dLayer(biasless);
    freeLinearLayer(frozen);
    freeQuantization(q);

    TEST_ASSERT_TRUE(frozenGradIsNull);
    TEST_ASSERT_TRUE(biasIsNull);
    TEST_ASSERT_TRUE(acceptedFrozen);
    TEST_ASSERT_TRUE(acceptedBiasless);
    TEST_ASSERT_TRUE(acceptedPassthrough);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testLinearAccessorsMatchConfig);
    RUN_TEST(testReluAccessorsMatchConfig);
    RUN_TEST(testConv1dAccessorsMatchConfig);
    RUN_TEST(testConv1dTransposedAccessorsMatchConfig);
    RUN_TEST(testMaxPool1dAccessorsMatchConfig);
    RUN_TEST(testAvgPool1dAccessorsMatchConfig);
    RUN_TEST(testAdaptiveAvgPool1dAccessorsMatchConfig);
    RUN_TEST(testSoftmaxAccessorsMatchConfig);
    RUN_TEST(testDropoutAccessorsMatchConfig);
    RUN_TEST(testLayerNormAccessorsMatchConfig);
    RUN_TEST(testQuantizationAccessorsMatchConfig);
    RUN_TEST(testFlattenAccessorsAreNullAndDefaultArithmetic);
    RUN_TEST(testLayerIsFloat32OnlyAcceptsEveryUniformFloat32LayerType);
    RUN_TEST(testLayerIsFloat32OnlyRejectsQuantizationLayer);
    RUN_TEST(testLayerNonFloat32FieldNamesEveryGemmFamilyField);
    RUN_TEST(testLayerNonFloat32FieldNamesEveryNormField);
    RUN_TEST(testLayerNonFloat32FieldNamesEveryWireOnlyLayerField);
    RUN_TEST(testLayerIsFloat32OnlyRejectsBfpNotJustSym);
    RUN_TEST(testLayerIsFloat32OnlyAcceptsFrozenBiaslessAndPassthroughLayers);
    return UNITY_END();
}
