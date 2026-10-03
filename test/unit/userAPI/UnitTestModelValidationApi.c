#define SOURCE_FILE "UNIT_TEST_MODEL_VALIDATION_API"

#include <stdbool.h>

#include "ArithmeticType.h"
#include "Layer.h"
#include "LayerNorm.h"
#include "Linear.h"
#include "ModelValidationApi.h"
#include "QuantizationApi.h"
#include "Rounding.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

/* The validator only walks layer->type + array shape now (the SYM-producer
 * rule below buildLayerNormStub was retired, PR1b.2 D3), so a
 * parameter-free stub suffices — same shape as buildLayerNormStub below (the
 * deleted linearLayerInitLegacy used to store forwardMath without
 * dereferencing weights/bias). */
static layer_t *buildLinearStub(quantization_t *fq) {
    linearConfig_t *cfg = reserveMemory(sizeof(linearConfig_t));
    cfg->weights = NULL;
    cfg->bias = NULL;
    cfg->forwardMath = arithmeticFromQuantization(fq);
    cfg->weightGradMath = arithmeticFromQuantization(fq);
    cfg->biasGradMath = arithmeticFromQuantization(fq);
    cfg->propLossMath = arithmeticFromQuantization(fq);
    cfg->outputQ = fq;
    cfg->propLossQ = fq;
    cfg->ownsQuantizations = false;
    layerConfig_t *lc = reserveMemory(sizeof(layerConfig_t));
    lc->linear = cfg;
    layer_t *layer = reserveMemory(sizeof(layer_t));
    initLayer(layer, LINEAR, lc);
    return layer;
}

static void freeLinearStub(layer_t *layer) {
    freeReservedMemory(layer->config->linear);
    freeReservedMemory(layer->config);
    freeReservedMemory(layer);
}

static layer_t *buildLayerNormStub(quantization_t *fq) {
    layerNormConfig_t *cfg = reserveMemory(sizeof(layerNormConfig_t));
    cfg->gamma = NULL;
    cfg->beta = NULL;
    cfg->normalizedShape = NULL;
    cfg->numNormDims = 0;
    cfg->eps = 1e-5f;
    cfg->forwardMath = arithmeticFromQuantization(fq);
    cfg->propLossMath = arithmeticFromQuantization(fq);
    cfg->outputQ = fq;
    cfg->propLossQ = fq;
    cfg->ownsQuantizations = false;
    layerConfig_t *lc = reserveMemory(sizeof(layerConfig_t));
    lc->layerNorm = cfg;
    layer_t *layer = reserveMemory(sizeof(layer_t));
    initLayer(layer, LAYERNORM, lc);
    return layer;
}

static void freeLayerNormStub(layer_t *layer) {
    freeReservedMemory(layer->config->layerNorm);
    freeReservedMemory(layer->config);
    freeReservedMemory(layer);
}

void testValidatorAcceptsEmptyModel(void) {
    /* modelSize == 0: loop never runs, valid stays true. */
    layer_t *model[1] = {NULL}; /* pointer needed; element never read */
    bool valid = validateModelQuantization(model, 0);
    TEST_ASSERT_TRUE_MESSAGE(valid, "empty model (size 0) should be accepted");
}

void testValidatorRejectsNullModel(void) {
    /* model == NULL: guarded at entry, returns false. */
    bool valid = validateModelQuantization(NULL, 0);
    TEST_ASSERT_FALSE_MESSAGE(valid, "NULL model pointer should be rejected");
}

void testValidatorRejectsNullElementMidArray(void) {
    /* NULL element mid-array: guarded in loop body (model[i]==NULL branch),
     * sets valid=false and continues without crashing. */
    quantization_t *symQ = quantizationInitSymInt32(HALF_AWAY);
    layer_t *linear = buildLinearStub(symQ);
    layer_t *model[2] = {linear, NULL};

    bool valid = validateModelQuantization(model, 2);

    freeLinearStub(linear);
    freeQuantization(symQ);

    TEST_ASSERT_FALSE_MESSAGE(valid, "NULL element in model array should be rejected");
}

/* Documents the retired rule's replacement contract (PR1b.2 D3): a SYM
 * accumulator-range producer (Linear) feeding a non-QUANTIZATION layer
 * (LayerNorm) used to be REJECTED (no chained Quant layer restoring width);
 * the forward funnel now restores width at the producer's own wire, so this
 * is a perfectly ordinary, ACCEPTED chain — a Quant layer here would just be
 * a redundant identical-config requant (the double-requant anti-pattern
 * `docs/conventions/arithmetic-sym.md` warns about), not a requirement. */
void testValidatorAcceptsSymProducerFollowedByNonQuantLayer(void) {
    quantization_t *symQ = quantizationInitSymInt32(HALF_AWAY);
    layer_t *linear = buildLinearStub(symQ);
    layer_t *norm = buildLayerNormStub(symQ);
    layer_t *model[2] = {linear, norm};

    bool valid = validateModelQuantization(model, 2);

    freeLayerNormStub(norm);
    freeLinearStub(linear);
    freeQuantization(symQ);

    TEST_ASSERT_TRUE_MESSAGE(valid, "Linear-SYM feeding LayerNorm-SYM directly is accepted: "
                                    "the forward funnel already restored width at the wire, "
                                    "a Quant layer here is optional, not required");
}

void testValidatorAcceptsFloatOnlyModel(void) {
    quantization_t *fQ = quantizationInitFloat();
    layer_t *linear = buildLinearStub(fQ);
    layer_t *norm = buildLayerNormStub(fQ);
    layer_t *model[2] = {linear, norm};

    bool valid = validateModelQuantization(model, 2);

    freeLayerNormStub(norm);
    freeLinearStub(linear);
    freeQuantization(fQ);

    TEST_ASSERT_TRUE(valid);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testValidatorAcceptsEmptyModel);
    RUN_TEST(testValidatorRejectsNullModel);
    RUN_TEST(testValidatorRejectsNullElementMidArray);
    RUN_TEST(testValidatorAcceptsSymProducerFollowedByNonQuantLayer);
    RUN_TEST(testValidatorAcceptsFloatOnlyModel);
    return UNITY_END();
}
