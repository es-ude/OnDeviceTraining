#define SOURCE_FILE "UNIT_TEST_ADD_API"

#include <stddef.h>

#include "AddApi.h"
#include "ArithmeticType.h"
#include "DeathTest.h"
#include "Layer.h"
#include "LayerConfigAccess.h"
#include "LayerQuant.h"
#include "Model.h"
#include "Optimizer.h"
#include "Quantization.h"
#include "QuantizationApi.h"
#include "TensorApi.h"
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

void testAddLayerInitDefaultsToFloat32(void) {
    layer_t *a = addLayerInit(NULL);
    TEST_ASSERT_EQUAL_INT(ADD, a->type);
    TEST_ASSERT_EQUAL_INT(FLOAT32, layerOutputQ(a)->type);
    TEST_ASSERT_EQUAL_INT(FLOAT32, backwardWireQ(a)->type);
    TEST_ASSERT_EQUAL_INT(ARITH_FLOAT32, layerForwardMath(a).type);
    TEST_ASSERT_NULL(layerNonFloat32Field(a));
    TEST_ASSERT_FALSE(layerForwardUpdatesState(a));
    TEST_ASSERT_EQUAL_size_t(6u, layerForwardCost(a, 6u, 6u));
    TEST_ASSERT_FALSE(layerParameters(a, &(parameter_t *){0}, &(parameter_t *){0}));
    TEST_ASSERT_EQUAL_size_t(0u, calcTotalNumberOfStates(&MODEL_CHAIN((layer_t *[]){a}, 1)));
    freeAddLayer(a);
}

/* The caller keeps ownership: freeAddLayer must leave the borrowed configs alive. */
void testAddLayerInitBorrowsTheCallersWires(void) {
    quantization_t *out = quantizationInitFloat(), *grad = quantizationInitFloat();
    layer_t *a = addLayerInit(&(layerQuant_t){.outputQ = out, .propLossQ = grad});
    TEST_ASSERT_EQUAL_PTR(out, layerOutputQ(a));
    TEST_ASSERT_EQUAL_PTR(grad, backwardWireQ(a));
    freeAddLayer(a);
    TEST_ASSERT_EQUAL_INT(FLOAT32, out->type);
    freeQuantization(out);
    freeQuantization(grad);
}

void testAddLayerInitRefusesANonFloat32OutputQ(void) {
    quantization_t *f = quantizationInitFloat(),
                   *s = quantizationInitSymInt32WithBits(HALF_AWAY, 12);
    ASSERT_EXITS_WITH_OUTPUT(1, "addLayerInit: layerQuant.outputQ is not FLOAT32",
                             addLayerInit(&(layerQuant_t){.outputQ = s, .propLossQ = f}));
    freeQuantization(f);
    freeQuantization(s);
}

void testAddLayerInitRefusesEveryOtherNonFloat32Slot(void) {
    quantization_t *f = quantizationInitFloat(),
                   *s = quantizationInitSymInt32WithBits(HALF_AWAY, 12);
    const arithmetic_t sym = {ARITH_SYM_INT32, HALF_AWAY};
    ASSERT_EXITS_WITH_OUTPUT(1, "addLayerInit: layerQuant.propLossQ is not FLOAT32",
                             addLayerInit(&(layerQuant_t){.outputQ = f, .propLossQ = s}));
    ASSERT_EXITS_WITH_OUTPUT(
        1, "addLayerInit: layerQuant.forwardMath is not FLOAT32",
        addLayerInit(&(layerQuant_t){.forwardMath = sym, .outputQ = f, .propLossQ = f}));
    ASSERT_EXITS_WITH_OUTPUT(
        1, "addLayerInit: layerQuant.propLossMath is not FLOAT32",
        addLayerInit(&(layerQuant_t){.propLossMath = sym, .outputQ = f, .propLossQ = f}));
    freeQuantization(f);
    freeQuantization(s);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testAddLayerInitDefaultsToFloat32);
    RUN_TEST(testAddLayerInitBorrowsTheCallersWires);
    RUN_TEST(testAddLayerInitRefusesANonFloat32OutputQ);
    RUN_TEST(testAddLayerInitRefusesEveryOtherNonFloat32Slot);
    return UNITY_END();
}
