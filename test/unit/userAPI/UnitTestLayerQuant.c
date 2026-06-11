#define SOURCE_FILE "UNIT_TEST_LAYER_QUANT"

#include "LayerQuant.h"
#include "QuantizationApi.h"
#include "StorageApi.h"
#include "unity.h"

void setUp() {}
void tearDown() {}

void testLayerQuantInitUniformSetsAllFourSlotsToTheSamePointer(void) {
    quantization_t *q = quantizationInitFloat();

    layerQuant_t lq = {0};
    layerQuantInitUniform(&lq, q);

    TEST_ASSERT_EQUAL_PTR(q, lq.forwardMath);
    TEST_ASSERT_EQUAL_PTR(q, lq.backwardMath);
    TEST_ASSERT_EQUAL_PTR(q, lq.weightStorage);
    TEST_ASSERT_EQUAL_PTR(q, lq.biasStorage);
}

void testLayerQuantInitUniformDoesNotMutateTheQuantization(void) {
    quantization_t *q = quantizationInitFloat();
    qtype_t typeBefore = q->type;
    void *configBefore = q->qConfig;

    layerQuant_t lq = {0};
    layerQuantInitUniform(&lq, q);

    TEST_ASSERT_EQUAL_INT(typeBefore, q->type);
    TEST_ASSERT_EQUAL_PTR(configBefore, q->qConfig);
}

void testDeepCopyQuantizationReturnsNullForNullInput(void) {
    TEST_ASSERT_NULL(deepCopyQuantization(NULL));
}

void testDeepCopyQuantizationFloat32ReturnsFreshAllocationWithNullQConfig(void) {
    quantization_t *src = quantizationInitFloat();
    quantization_t *dst = deepCopyQuantization(src);

    TEST_ASSERT_NOT_NULL(dst);
    TEST_ASSERT_NOT_EQUAL(src, dst); /* fresh allocation */
    TEST_ASSERT_EQUAL_INT(FLOAT32, dst->type);
    TEST_ASSERT_NULL(dst->qConfig);

    freeReservedMemory(dst->qConfig);
    freeReservedMemory(dst);
}

void testDeepCopyQuantizationSymInt32DuplicatesQConfigBytes(void) {
    quantization_t *src = quantizationInitSymInt32(HALF_AWAY);
    quantization_t *dst = deepCopyQuantization(src);

    TEST_ASSERT_NOT_NULL(dst);
    TEST_ASSERT_NOT_EQUAL(src, dst);
    TEST_ASSERT_EQUAL_INT(SYM_INT32, dst->type);
    TEST_ASSERT_NOT_NULL(dst->qConfig);
    TEST_ASSERT_NOT_EQUAL(src->qConfig, dst->qConfig);

    symInt32QConfig_t *srcCfg = (symInt32QConfig_t *)src->qConfig;
    symInt32QConfig_t *dstCfg = (symInt32QConfig_t *)dst->qConfig;
    TEST_ASSERT_EQUAL_MEMORY(srcCfg, dstCfg, sizeof(symInt32QConfig_t));

    freeReservedMemory(dst->qConfig);
    freeReservedMemory(dst);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testLayerQuantInitUniformSetsAllFourSlotsToTheSamePointer);
    RUN_TEST(testLayerQuantInitUniformDoesNotMutateTheQuantization);
    RUN_TEST(testDeepCopyQuantizationReturnsNullForNullInput);
    RUN_TEST(testDeepCopyQuantizationFloat32ReturnsFreshAllocationWithNullQConfig);
    RUN_TEST(testDeepCopyQuantizationSymInt32DuplicatesQConfigBytes);
    return UNITY_END();
}
