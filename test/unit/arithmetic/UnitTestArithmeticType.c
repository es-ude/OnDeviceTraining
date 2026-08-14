#define SOURCE_FILE "UNIT-TEST-ARITHMETIC-TYPE"

#include "ArithmeticType.h"
#include "Quantization.h"
#include "Rounding.h"
#include "unity.h"

static void testFloat32QuantizationDerivesFloatArithmeticWithHalfAway(void) {
    quantization_t q;
    initFloat32Quantization(&q);
    arithmetic_t a = arithmeticFromQuantization(&q);
    TEST_ASSERT_EQUAL(ARITH_FLOAT32, a.type);
    TEST_ASSERT_EQUAL(HALF_AWAY, a.roundingMode);
}

static void testInt32QuantizationDerivesFloatArithmeticWithHalfAway(void) {
    /* INT32 is a spec-named storage-only dtype (raw 32-bit integer, no
     * qConfig) — bridges through float like the other storage-only types. */
    quantization_t q;
    initInt32Quantization(&q);
    arithmetic_t a = arithmeticFromQuantization(&q);
    TEST_ASSERT_EQUAL(ARITH_FLOAT32, a.type);
    TEST_ASSERT_EQUAL(HALF_AWAY, a.roundingMode);
}

static void testSymInt32QuantizationDerivesSymArithmeticWithItsRoundingMode(void) {
    symInt32QConfig_t qc;
    initSymInt32QConfig(SR_HALF_AWAY, &qc);
    quantization_t q;
    initSymInt32Quantization(&qc, &q);
    arithmetic_t a = arithmeticFromQuantization(&q);
    TEST_ASSERT_EQUAL(ARITH_SYM_INT32, a.type);
    TEST_ASSERT_EQUAL(SR_HALF_AWAY, a.roundingMode);
}

/* BFP epic PR2: the D5 float-bridge staging rule is RETIRED -- BFP storage
 * derives NATIVE ARITH_BFP, like SYM_INT32 derives ARITH_SYM_INT32, and its
 * OWN roundingMode seeds the derived arithmetic. Fake-quant over BFP storage
 * stays available, but is now EXPLICIT: pin the math slots to ARITH_FLOAT32
 * instead of letting them derive (see
 * testBfpWireFakeQuantTrainingLossDecreasesAndWirePacks). */
static void testDerivationBfpIsNativeArithBfpWithConfigRounding(void) {
    /* Stack-fixture idiom (docs/conventions/testing.md): avoid initBfpQConfig's
     * heap allocation for a fixture never passed to freeQuantization. */
    uint8_t exponents[1] = {127};
    bfpQConfig_t qc = {.exponents = exponents,
                       .numGroups = 1,
                       .groupSize = 0,
                       .roundingMode = SR_HALF_AWAY,
                       .mantissaBits = 8,
                       .exponentBits = 8};
    quantization_t q;
    initBfpQuantization(&qc, &q);
    arithmetic_t a = arithmeticFromQuantization(&q);
    TEST_ASSERT_EQUAL(ARITH_BFP, a.type);
    TEST_ASSERT_EQUAL(SR_HALF_AWAY, a.roundingMode);
}

static void testStorageOnlyDtypesDeriveFloatArithmetic(void) {
    /* ASYM/SYM/BOOL/INT32 are storage formats; compute bridges through float
     * (spec D5, project ASYM design: conversion between native ops). */
    /* Stack-fixture idiom (PR4): initAsymQConfig would heap-allocate two
     * arrays this test never frees. */
    float aqcScales[1] = {1.f};
    uint16_t aqcZps[1] = {0};
    asymQConfig_t aqc = {.scales = aqcScales,
                         .zeroPoints = aqcZps,
                         .numGroups = 1,
                         .groupSize = 0,
                         .qBits = 8,
                         .roundingMode = SR_HALF_AWAY};
    quantization_t asymQ;
    initAsymQuantization(&aqc, &asymQ);
    arithmetic_t a = arithmeticFromQuantization(&asymQ);
    TEST_ASSERT_EQUAL(ARITH_FLOAT32, a.type);
    TEST_ASSERT_EQUAL(SR_HALF_AWAY, a.roundingMode); /* roundingMode carried over */

    /* Stack-fixture idiom (docs/conventions/testing.md): a heap-allocating
     * initSymQConfig call here would leak its scales array. */
    float symScales[1] = {1.f};
    symQConfig_t sqc = {.scales = symScales,
                        .numGroups = 1,
                        .groupSize = 0,
                        .roundingMode = SR_HALF_AWAY,
                        .qBits = 8};
    quantization_t symQ;
    initSymQuantization(&sqc, &symQ);
    arithmetic_t s = arithmeticFromQuantization(&symQ);
    TEST_ASSERT_EQUAL(ARITH_FLOAT32, s.type);
    TEST_ASSERT_EQUAL(SR_HALF_AWAY, s.roundingMode); /* roundingMode carried over */

    quantization_t boolQ;
    initBoolQuantization(&boolQ);
    arithmetic_t b = arithmeticFromQuantization(&boolQ);
    TEST_ASSERT_EQUAL(ARITH_FLOAT32, b.type);
    TEST_ASSERT_EQUAL(HALF_AWAY, b.roundingMode); /* no qConfig -> default HALF_AWAY */
}

static void testOrDefaultReturnsFloat32HalfAwayForNull(void) {
    arithmetic_t a = arithmeticFromQuantizationOrDefault(NULL);
    TEST_ASSERT_EQUAL(ARITH_FLOAT32, a.type);
    TEST_ASSERT_EQUAL(HALF_AWAY, a.roundingMode);
}

static void testOrDefaultMatchesArithmeticFromQuantizationForNonNull(void) {
    symInt32QConfig_t qc;
    initSymInt32QConfig(SR_HALF_AWAY, &qc);
    quantization_t q;
    initSymInt32Quantization(&qc, &q);

    arithmetic_t expected = arithmeticFromQuantization(&q);
    arithmetic_t actual = arithmeticFromQuantizationOrDefault(&q);
    TEST_ASSERT_EQUAL(expected.type, actual.type);
    TEST_ASSERT_EQUAL(expected.roundingMode, actual.roundingMode);
}

void setUp() {}
void tearDown() {}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testFloat32QuantizationDerivesFloatArithmeticWithHalfAway);
    RUN_TEST(testInt32QuantizationDerivesFloatArithmeticWithHalfAway);
    RUN_TEST(testSymInt32QuantizationDerivesSymArithmeticWithItsRoundingMode);
    RUN_TEST(testDerivationBfpIsNativeArithBfpWithConfigRounding);
    RUN_TEST(testStorageOnlyDtypesDeriveFloatArithmetic);
    RUN_TEST(testOrDefaultReturnsFloat32HalfAwayForNull);
    RUN_TEST(testOrDefaultMatchesArithmeticFromQuantizationForNonNull);

    return UNITY_END();
}
