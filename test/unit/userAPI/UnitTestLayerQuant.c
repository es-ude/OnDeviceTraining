#define SOURCE_FILE "UNIT_TEST_LAYER_QUANT"

#include "ArithmeticType.h"
#include "LayerQuant.h"
#include "QuantizationApi.h"
#include "StorageApi.h"
#include "TensorApi.h"
#include "unity.h"

void setUp() {}
void tearDown() {}

void testLayerQuantInitUniformFloat32DerivesFloatArithmeticAndSharesStorage(void) {
    quantization_t *q = quantizationInitFloat();

    layerQuant_t lq = {0};
    layerQuantInitUniform(&lq, q);

    TEST_ASSERT_EQUAL_INT(ARITH_FLOAT32, lq.forwardMath.type);
    TEST_ASSERT_EQUAL_INT(HALF_AWAY, lq.forwardMath.roundingMode);
    TEST_ASSERT_EQUAL_INT(ARITH_FLOAT32, lq.weightGradMath.type);
    TEST_ASSERT_EQUAL_INT(ARITH_FLOAT32, lq.biasGradMath.type);
    TEST_ASSERT_EQUAL_INT(ARITH_FLOAT32, lq.propLossMath.type);

    TEST_ASSERT_EQUAL_PTR(q, lq.outputQ);
    TEST_ASSERT_EQUAL_PTR(q, lq.propLossQ);
    TEST_ASSERT_EQUAL_PTR(q, lq.weightStorage);
    TEST_ASSERT_EQUAL_PTR(q, lq.biasStorage);

    TEST_ASSERT_NULL(lq.weightGradStorage);
    TEST_ASSERT_NULL(lq.biasGradStorage);
}

void testLayerQuantInitUniformSymInt32DerivesSymArithmeticWithProfileRoundingMode(void) {
    quantization_t *q = quantizationInitSymInt32(SR_HALF_AWAY);

    layerQuant_t lq = {0};
    layerQuantInitUniform(&lq, q);

    TEST_ASSERT_EQUAL_INT(ARITH_SYM_INT32, lq.forwardMath.type);
    TEST_ASSERT_EQUAL_INT(SR_HALF_AWAY, lq.forwardMath.roundingMode);
    TEST_ASSERT_EQUAL_INT(ARITH_SYM_INT32, lq.weightGradMath.type);
    TEST_ASSERT_EQUAL_INT(SR_HALF_AWAY, lq.weightGradMath.roundingMode);
    TEST_ASSERT_EQUAL_INT(ARITH_SYM_INT32, lq.biasGradMath.type);
    TEST_ASSERT_EQUAL_INT(SR_HALF_AWAY, lq.biasGradMath.roundingMode);
    TEST_ASSERT_EQUAL_INT(ARITH_SYM_INT32, lq.propLossMath.type);
    TEST_ASSERT_EQUAL_INT(SR_HALF_AWAY, lq.propLossMath.roundingMode);

    TEST_ASSERT_EQUAL_PTR(q, lq.outputQ);
    TEST_ASSERT_EQUAL_PTR(q, lq.propLossQ);
    TEST_ASSERT_EQUAL_PTR(q, lq.weightStorage);
    TEST_ASSERT_EQUAL_PTR(q, lq.biasStorage);

    TEST_ASSERT_NULL(lq.weightGradStorage);
    TEST_ASSERT_NULL(lq.biasGradStorage);
}

void testLayerQuantInitUniformAsymBridgesThroughFloatArithmeticButKeepsAsymStorage(void) {
    /* Storage-only dtype (spec D5): arithmetic bridges through ARITH_FLOAT32,
     * but the storage slots keep the real ASYM quantization untouched. */
    quantization_t *q = quantizationInitAsym(8, HALF_AWAY);

    layerQuant_t lq = {0};
    layerQuantInitUniform(&lq, q);

    TEST_ASSERT_EQUAL_INT(ARITH_FLOAT32, lq.forwardMath.type);
    TEST_ASSERT_EQUAL_INT(ARITH_FLOAT32, lq.weightGradMath.type);
    TEST_ASSERT_EQUAL_INT(ARITH_FLOAT32, lq.biasGradMath.type);
    TEST_ASSERT_EQUAL_INT(ARITH_FLOAT32, lq.propLossMath.type);

    TEST_ASSERT_EQUAL_PTR(q, lq.outputQ);
    TEST_ASSERT_EQUAL_PTR(q, lq.propLossQ);
    TEST_ASSERT_EQUAL_PTR(q, lq.weightStorage);
    TEST_ASSERT_EQUAL_PTR(q, lq.biasStorage);
    TEST_ASSERT_EQUAL_INT(ASYM, lq.outputQ->type);
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

void testDeepCopyQuantizationSymDeepCopiesScalesArray(void) {
    /* Group-quant PR1: symQConfig_t carries a heap `scales` pointer, so a
     * blind memcpy of the qConfig struct (the SYM_INT32/ASYM pattern above)
     * would alias dst's scales onto src's -- a double-free once both sides
     * are freed, and a value assertion alone wouldn't catch it (the aliased
     * array reads the same float). Mutation guard: reverting the SYM branch
     * in deepCopyQuantization to the generic byte-memcpy path makes the
     * pointer-independence assertion below FAIL (and a subsequent
     * freeQuantization of both configs double-frees under ASan). */
    quantization_t *src = quantizationInitSym(6, SR_HALF_AWAY);
    quantization_t *dst = deepCopyQuantization(src);

    TEST_ASSERT_NOT_NULL(dst);
    TEST_ASSERT_NOT_EQUAL(src, dst);
    TEST_ASSERT_EQUAL_INT(SYM, dst->type);
    TEST_ASSERT_NOT_NULL(dst->qConfig);
    TEST_ASSERT_NOT_EQUAL(src->qConfig, dst->qConfig);

    symQConfig_t *srcCfg = (symQConfig_t *)src->qConfig;
    symQConfig_t *dstCfg = (symQConfig_t *)dst->qConfig;
    TEST_ASSERT_NOT_EQUAL(srcCfg->scales, dstCfg->scales); /* independent arrays */
    TEST_ASSERT_EQUAL_FLOAT(srcCfg->scales[0], dstCfg->scales[0]);
    TEST_ASSERT_EQUAL_UINT8(srcCfg->qBits, dstCfg->qBits);
    TEST_ASSERT_EQUAL_INT(srcCfg->roundingMode, dstCfg->roundingMode);
    TEST_ASSERT_EQUAL_size_t(srcCfg->numGroups, dstCfg->numGroups);
    TEST_ASSERT_EQUAL_size_t(srcCfg->groupSize, dstCfg->groupSize);

    freeQuantization(dst);
    freeQuantization(src);
}

void testDeepCopyQuantizationBfpGrouped(void) {
    /* BFP epic PR1 twin of testDeepCopyQuantizationSymDeepCopiesScalesArray:
     * bfpQConfig_t carries a heap `exponents` pointer, so a blind memcpy of
     * the qConfig struct would alias dst's exponents onto src's -- deep-copy
     * the array instead. Mutation guard: reverting the BFP branch in
     * deepCopyQuantization to the generic byte-memcpy path makes the
     * pointer-independence assertion below FAIL (and a subsequent
     * freeQuantization of both configs double-frees under ASan). */
    quantization_t *src = quantizationInitBfpGrouped(4, 8, HALF_AWAY, 2, 4);
    bfpQConfig_t *srcCfg = (bfpQConfig_t *)src->qConfig;
    srcCfg->exponents[0] = 130;
    srcCfg->exponents[1] = 140;

    quantization_t *dst = deepCopyQuantization(src);

    TEST_ASSERT_NOT_NULL(dst);
    TEST_ASSERT_NOT_EQUAL(src, dst);
    TEST_ASSERT_EQUAL_INT(BFP, dst->type);
    TEST_ASSERT_NOT_NULL(dst->qConfig);
    TEST_ASSERT_NOT_EQUAL(src->qConfig, dst->qConfig);

    bfpQConfig_t *dstCfg = (bfpQConfig_t *)dst->qConfig;
    TEST_ASSERT_NOT_EQUAL(srcCfg->exponents, dstCfg->exponents); /* independent arrays */
    TEST_ASSERT_EQUAL_UINT8(srcCfg->exponents[0], dstCfg->exponents[0]);
    TEST_ASSERT_EQUAL_UINT8(srcCfg->exponents[1], dstCfg->exponents[1]);
    TEST_ASSERT_EQUAL_UINT8(srcCfg->mantissaBits, dstCfg->mantissaBits);
    TEST_ASSERT_EQUAL_UINT8(srcCfg->exponentBits, dstCfg->exponentBits);
    TEST_ASSERT_EQUAL_INT(srcCfg->roundingMode, dstCfg->roundingMode);
    TEST_ASSERT_EQUAL_size_t(srcCfg->numGroups, dstCfg->numGroups);
    TEST_ASSERT_EQUAL_size_t(srcCfg->groupSize, dstCfg->groupSize);

    freeQuantization(dst);
    freeQuantization(src);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testLayerQuantInitUniformFloat32DerivesFloatArithmeticAndSharesStorage);
    RUN_TEST(testLayerQuantInitUniformSymInt32DerivesSymArithmeticWithProfileRoundingMode);
    RUN_TEST(testLayerQuantInitUniformAsymBridgesThroughFloatArithmeticButKeepsAsymStorage);
    RUN_TEST(testLayerQuantInitUniformDoesNotMutateTheQuantization);
    RUN_TEST(testDeepCopyQuantizationReturnsNullForNullInput);
    RUN_TEST(testDeepCopyQuantizationFloat32ReturnsFreshAllocationWithNullQConfig);
    RUN_TEST(testDeepCopyQuantizationSymInt32DuplicatesQConfigBytes);
    RUN_TEST(testDeepCopyQuantizationSymDeepCopiesScalesArray);
    RUN_TEST(testDeepCopyQuantizationBfpGrouped);
    return UNITY_END();
}
