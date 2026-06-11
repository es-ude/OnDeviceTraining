#define SOURCE_FILE "UNIT_TEST_MODEL_VALIDATION_API"

#include <stdbool.h>

#include "Layer.h"
#include "LayerNorm.h"
#include "Linear.h"
#include "LinearApi.h"
#include "ModelValidationApi.h"
#include "QuantizationApi.h"
#include "QuantizationLayer.h"
#include "Rounding.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

/* The validator reads ONLY layer->type + config->...->forwardQ, so
 * parameter-free stubs suffice (linearLayerInitLegacy stores forwardQ without
 * dereferencing weights/bias — UnitTestLinear roundtrip precedent). */
static layer_t *buildLinearStub(quantization_t *fq) {
    return linearLayerInitLegacy(NULL, NULL, fq, fq, fq, fq);
}

static layer_t *buildLayerNormStub(quantization_t *fq) {
    layerNormConfig_t *cfg = reserveMemory(sizeof(layerNormConfig_t));
    cfg->gamma = NULL;
    cfg->beta = NULL;
    cfg->normalizedShape = NULL;
    cfg->numNormDims = 0;
    cfg->eps = 1e-5f;
    cfg->forwardQ = fq;
    cfg->backwardQ = fq;
    cfg->ownsQuantizations = false;
    layerConfig_t *lc = reserveMemory(sizeof(layerConfig_t));
    lc->layerNorm = cfg;
    layer_t *layer = reserveMemory(sizeof(layer_t));
    initLayer(layer, LAYERNORM, lc);
    return layer;
}

static layer_t *buildQuantStub(quantization_t *fq) {
    quantizationConfig_t *cfg = reserveMemory(sizeof(quantizationConfig_t));
    cfg->forwardQ = fq;
    cfg->backwardQ = fq;
    cfg->ownsQuantizations = false;
    layerConfig_t *lc = reserveMemory(sizeof(layerConfig_t));
    lc->quantization = cfg;
    layer_t *layer = reserveMemory(sizeof(layer_t));
    initLayer(layer, QUANTIZATION, lc);
    return layer;
}

static void freeLayerNormStub(layer_t *layer) {
    freeReservedMemory(layer->config->layerNorm);
    freeReservedMemory(layer->config);
    freeReservedMemory(layer);
}

static void freeQuantStub(layer_t *layer) {
    freeReservedMemory(layer->config->quantization);
    freeReservedMemory(layer->config);
    freeReservedMemory(layer);
}

void testValidatorRejectsSymProducerFollowedByNonQuantLayer(void) {
    quantization_t *symQ = quantizationInitSymInt32(HALF_AWAY);
    layer_t *linear = buildLinearStub(symQ);
    layer_t *norm = buildLayerNormStub(symQ);
    layer_t *model[2] = {linear, norm};

    bool valid = validateModelQuantization(model, 2);

    freeLayerNormStub(norm);
    freeLinearLayerLegacy(linear);
    freeQuantization(symQ);

    TEST_ASSERT_FALSE_MESSAGE(valid, "Linear-SYM feeding LayerNorm-SYM without a Quant layer "
                                     "violates the int16 inter-layer contract");
}

void testValidatorAcceptsChainWithQuantLayersBetweenProducers(void) {
    quantization_t *symQ = quantizationInitSymInt32(HALF_AWAY);
    layer_t *linear0 = buildLinearStub(symQ);
    layer_t *quant1 = buildQuantStub(symQ);
    layer_t *norm = buildLayerNormStub(symQ);
    layer_t *quant2 = buildQuantStub(symQ);
    layer_t *linear1 = buildLinearStub(symQ); /* producer in last position: allowed */
    layer_t *model[5] = {linear0, quant1, norm, quant2, linear1};

    bool valid = validateModelQuantization(model, 5);

    freeLinearLayerLegacy(linear1);
    freeQuantStub(quant2);
    freeLayerNormStub(norm);
    freeQuantStub(quant1);
    freeLinearLayerLegacy(linear0);
    freeQuantization(symQ);

    TEST_ASSERT_TRUE(valid);
}

void testValidatorAcceptsSymProducerInLastPosition(void) {
    quantization_t *symQ = quantizationInitSymInt32(HALF_AWAY);
    layer_t *linear = buildLinearStub(symQ);
    layer_t *model[1] = {linear};

    bool valid = validateModelQuantization(model, 1);

    freeLinearLayerLegacy(linear);
    freeQuantization(symQ);

    TEST_ASSERT_TRUE_MESSAGE(valid, "producer at the loss boundary is allowed");
}

void testValidatorAcceptsFloatOnlyModel(void) {
    quantization_t *fQ = quantizationInitFloat();
    layer_t *linear = buildLinearStub(fQ);
    layer_t *norm = buildLayerNormStub(fQ);
    layer_t *model[2] = {linear, norm};

    bool valid = validateModelQuantization(model, 2);

    freeLayerNormStub(norm);
    freeLinearLayerLegacy(linear);
    freeQuantization(fQ);

    TEST_ASSERT_TRUE(valid);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testValidatorRejectsSymProducerFollowedByNonQuantLayer);
    RUN_TEST(testValidatorAcceptsChainWithQuantLayersBetweenProducers);
    RUN_TEST(testValidatorAcceptsSymProducerInLastPosition);
    RUN_TEST(testValidatorAcceptsFloatOnlyModel);
    return UNITY_END();
}
