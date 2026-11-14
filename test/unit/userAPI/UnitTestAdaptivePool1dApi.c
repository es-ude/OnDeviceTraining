#define SOURCE_FILE "UNIT_TEST_ADAPTIVE_POOL1D_API"

#include "AdaptiveAvgPool1d.h"
#include "AdaptivePool1dApi.h"
#include "ArithmeticType.h"
#include "DeathTest.h"
#include "Layer.h"
#include "LayerQuant.h"
#include "QuantizationApi.h"
#include "TensorApi.h"
#include "unity.h"

void setUp() {}
void tearDown() {}

void testBorrowingBuildsLayer(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);

    layer_t *layer = adaptiveAvgPool1dLayerInit(&(adaptiveAvgPool1dInit_t){.outputSize = 8}, &lq);

    TEST_ASSERT_NOT_NULL(layer);
    TEST_ASSERT_EQUAL_INT(ADAPTIVE_AVGPOOL1D, layer->type);

    adaptiveAvgPool1dConfig_t *cfg = layer->config->adaptiveAvgPool1d;
    TEST_ASSERT_NOT_NULL(cfg);
    TEST_ASSERT_FALSE(cfg->ownsQuantizations);
    TEST_ASSERT_EQUAL_UINT(8, cfg->outputSize);
    TEST_ASSERT_EQUAL_PTR(q, cfg->outputQ);
    TEST_ASSERT_EQUAL_PTR(q, cfg->propLossQ);
    TEST_ASSERT_EQUAL_INT(ARITH_FLOAT32, cfg->forwardMath.type);
    TEST_ASSERT_EQUAL_INT(ARITH_FLOAT32, cfg->propLossMath.type);

    freeAdaptiveAvgPool1dLayer(layer);
    freeQuantization(q);
}

void testOwningDeepCopiesTwoQuantizations(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);

    layer_t *layer =
        adaptiveAvgPool1dLayerInitOwning(&(adaptiveAvgPool1dInit_t){.outputSize = 4}, &lq);

    adaptiveAvgPool1dConfig_t *cfg = layer->config->adaptiveAvgPool1d;
    TEST_ASSERT_NOT_EQUAL(q, cfg->outputQ);
    TEST_ASSERT_NOT_EQUAL(q, cfg->propLossQ);
    /* The two deep copies must be distinct allocations, else freeAdaptive...
       would double-free and the borrowing/owning contract would be broken. */
    TEST_ASSERT_NOT_EQUAL(cfg->outputQ, cfg->propLossQ);
    TEST_ASSERT_EQUAL_INT(q->type, cfg->outputQ->type);
    TEST_ASSERT_EQUAL_INT(ARITH_FLOAT32, cfg->forwardMath.type);
    TEST_ASSERT_TRUE(cfg->ownsQuantizations);

    freeAdaptiveAvgPool1dLayer(layer);
    freeQuantization(q);
}

void testRejectsUnsupportedArithmetic(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t badForward;
    layerQuantInitUniform(&badForward, q);
    badForward.forwardMath.type = (arithmeticType_t)7;
    layerQuant_t badPropLoss;
    layerQuantInitUniform(&badPropLoss, q);
    badPropLoss.propLossMath.type = (arithmeticType_t)7;
    const char *msg = "adaptiveAvgPool1dLayerInit: layerQuant.forwardMath and propLossMath must be";

    ASSERT_EXITS_WITH_OUTPUT(
        1, msg,
        (void)adaptiveAvgPool1dLayerInit(&(adaptiveAvgPool1dInit_t){.outputSize = 2}, &badForward));
    ASSERT_EXITS_WITH_OUTPUT(1, msg,
                             (void)adaptiveAvgPool1dLayerInitOwning(
                                 &(adaptiveAvgPool1dInit_t){.outputSize = 2}, &badPropLoss));
    freeQuantization(q);
}

/* An ARITH_BFP op stages FLOAT32 operands at its produced-wire widths, so that
 * wire config must be BFP. */
void testRejectsBfpMathWithoutBfpWire(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t bfpForward;
    layerQuantInitUniform(&bfpForward, q);
    bfpForward.forwardMath.type = ARITH_BFP;
    layerQuant_t bfpPropLoss;
    layerQuantInitUniform(&bfpPropLoss, q);
    bfpPropLoss.propLossMath.type = ARITH_BFP;

    ASSERT_EXITS_WITH_OUTPUT(
        1, "adaptiveAvgPool1dLayerInit: ARITH_BFP forwardMath requires a BFP-typed outputQ",
        (void)adaptiveAvgPool1dLayerInit(&(adaptiveAvgPool1dInit_t){.outputSize = 2}, &bfpForward));
    ASSERT_EXITS_WITH_OUTPUT(
        1, "adaptiveAvgPool1dLayerInit: ARITH_BFP propLossMath requires a BFP-typed propLossQ",
        (void)adaptiveAvgPool1dLayerInitOwning(&(adaptiveAvgPool1dInit_t){.outputSize = 2},
                                               &bfpPropLoss));
    freeQuantization(q);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testBorrowingBuildsLayer);
    RUN_TEST(testOwningDeepCopiesTwoQuantizations);
    RUN_TEST(testRejectsUnsupportedArithmetic);
    RUN_TEST(testRejectsBfpMathWithoutBfpWire);
    return UNITY_END();
}
