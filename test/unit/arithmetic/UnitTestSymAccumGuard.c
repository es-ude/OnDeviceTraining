#define SOURCE_FILE "UNIT_TEST_SYM_ACCUM_GUARD"

#include <stdint.h>

#include "DeathTest.h"
#include "SymAccumGuard.h"
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

/* int12 x int12: |code| <= 2^11 (two's-complement magnitude, -2^11 reachable),
 * product <= 2^22; 511 * 2^22 = 2^31 - 2^22 fits INT32_MAX, 512 * 2^22 = 2^31 does not. */
void testTermLimitInt12Seed0Is511(void) {
    TEST_ASSERT_EQUAL_size_t(511, symAccumTermLimit(12, 12, 0));
}

void testProvenSafeBoundaryAt511And512(void) {
    symAccumSite_t s511 = symAccumSiteInit("t", 511, 12, 12, 0);
    symAccumSite_t s512 = symAccumSiteInit("t", 512, 12, 12, 0);
    TEST_ASSERT_TRUE(symAccumProvenSafe(&s511));
    TEST_ASSERT_FALSE(symAccumProvenSafe(&s512));
}

void testSeedThatTipsTheBoundShrinksTheLimit(void) {
    /* room for exactly 511 products leaves 2^22 - 1 of seed; one more tips it */
    uint64_t roomFor511 = (uint64_t)INT32_MAX - 511ull * (1ull << 22);
    TEST_ASSERT_EQUAL_size_t(511, symAccumTermLimit(12, 12, roomFor511));
    TEST_ASSERT_EQUAL_size_t(510, symAccumTermLimit(12, 12, roomFor511 + 1));
}

void testMaxAbsSeedHandlesInt32Min(void) {
    int32_t seeds[] = {5, -7, INT32_MIN, 3};
    TEST_ASSERT_EQUAL_UINT64(2147483648ull, symAccumMaxAbsSeed(seeds, 4));
    int32_t small[] = {5, -7, 3};
    TEST_ASSERT_EQUAL_UINT64(7ull, symAccumMaxAbsSeed(small, 3));
    TEST_ASSERT_EQUAL_UINT64(0ull, symAccumMaxAbsSeed(NULL, 0));
}

void testInt32MinSeedProvesOnlyZeroTerms(void) {
    /* |INT32_MIN| = 2^31 > INT32_MAX: any product can already overflow */
    TEST_ASSERT_EQUAL_size_t(0, symAccumTermLimit(12, 12, 2147483648ull));
}

void testInt16By16AdmitsExactlyOneTerm(void) {
    TEST_ASSERT_EQUAL_size_t(1, symAccumTermLimit(16, 16, 0));
}

void testWidthsSummingPastTheInt32ProductProveNothing(void) {
    /* 2^15 * 2^16 = 2^31 > INT32_MAX */
    TEST_ASSERT_EQUAL_size_t(0, symAccumTermLimit(16, 17, 0));
    TEST_ASSERT_EQUAL_size_t(0, symAccumTermLimit(20, 20, 0));
    TEST_ASSERT_EQUAL_size_t(0, symAccumTermLimit(32, 32, 0));
    /* absurd widths clamp instead of shifting past the bound arithmetic */
    TEST_ASSERT_EQUAL_size_t(0, symAccumTermLimit(255, 255, 0));
}

void testZeroTermsAreAlwaysProvenSafe(void) {
    symAccumSite_t s = symAccumSiteInit("t", 0, 32, 32, 2147483648ull);
    TEST_ASSERT_TRUE(symAccumProvenSafe(&s));
}

void testSizeMaxDoesNotWrapTheBound(void) {
    symAccumSite_t s = symAccumSiteInit("t", SIZE_MAX, 12, 12, 0);
    TEST_ASSERT_FALSE(symAccumProvenSafe(&s));
    symAccumSite_t narrow = symAccumSiteInit("t", SIZE_MAX, 1, 1, 0);
    TEST_ASSERT_FALSE(symAccumProvenSafe(&narrow));
}

void testWideProductFlagFollowsOperandWidths(void) {
    TEST_ASSERT_FALSE(symAccumSiteInit("t", 1, 16, 16, 0).wideProduct);
    TEST_ASSERT_TRUE(symAccumSiteInit("t", 1, 12, 17, 0).wideProduct);
    TEST_ASSERT_TRUE(symAccumSiteInit("t", 1, 17, 12, 0).wideProduct);
}

void testCheckedAddReturnsExactSumWithoutOverflow(void) {
    symAccumSite_t s = symAccumSiteInit("t", 600, 12, 12, 0);
    TEST_ASSERT_EQUAL_INT32(INT32_MAX, symAccumAddChecked(INT32_MAX - 5, 5, &s));
    TEST_ASSERT_EQUAL_INT32(INT32_MIN, symAccumAddChecked(INT32_MIN + 5, -5, &s));
}

void testCheckedAddAbortsOnPositiveAndNegativeOverflow(void) {
    symAccumSite_t s = symAccumSiteInit("testop", 600, 12, 12, 0);
    ASSERT_EXITS_WITH_OUTPUT(1, "testop", (void)symAccumAddChecked(INT32_MAX, 1, &s));
    ASSERT_EXITS_WITH_OUTPUT(1, "testop", (void)symAccumAddChecked(INT32_MIN, -1, &s));
}

void testBannerNamesNWidthsSeedAndRemedies(void) {
    symAccumSite_t s = symAccumSiteInit("bannerop", 600, 12, 13, 77);
    ASSERT_EXITS_WITH_OUTPUT(1, "N=600", (void)symAccumAddChecked(INT32_MAX, 1, &s));
    ASSERT_EXITS_WITH_OUTPUT(1, "widths (12, 13)", (void)symAccumAddChecked(INT32_MAX, 1, &s));
    ASSERT_EXITS_WITH_OUTPUT(1, "max|seed|=77", (void)symAccumAddChecked(INT32_MAX, 1, &s));
    ASSERT_EXITS_WITH_OUTPUT(1, "ODT_SYM_OPERAND_QMAXBITS",
                             (void)symAccumAddChecked(INT32_MAX, 1, &s));
    ASSERT_EXITS_WITH_OUTPUT(1, "ARITH_FLOAT32", (void)symAccumAddChecked(INT32_MAX, 1, &s));
}

void testMacCheckedExactResultAndWideProductAbort(void) {
    symAccumSite_t narrow = symAccumSiteInit("t", 600, 12, 12, 0);
    TEST_ASSERT_EQUAL_INT32(100 + 6, symAccumMacChecked(100, 2, 3, &narrow));
    /* wide operands: the PRODUCT itself overflows int32 (65536 * 65536) */
    symAccumSite_t wide = symAccumSiteInit("widemac", 2, 20, 20, 0);
    ASSERT_EXITS_WITH_OUTPUT(1, "widemac", (void)symAccumMacChecked(0, 65536, 65536, &wide));
    /* wide operands, fitting product: exact */
    TEST_ASSERT_EQUAL_INT32(1000 + 40000, symAccumMacChecked(1000, 200, 200, &wide));
}

void testProductCheckedAbortsOnlyWhenWideAndOverflowing(void) {
    symAccumSite_t wide = symAccumSiteInit("widemul", 1, 20, 20, 0);
    TEST_ASSERT_EQUAL_INT32(40000, symProduct(200, 200, true, &wide));
    TEST_ASSERT_EQUAL_INT32(12, symProduct(3, 4, false, &wide));
    ASSERT_EXITS_WITH_OUTPUT(1, "widemul", (void)symProduct(65536, 65536, true, &wide));
}

/* A rescaled partial (grouped kernels) is (p * s_g) / sAcc in float32: two
 * roundings can land one ulp (2^-23 relative) above |p|, and a stochastic round
 * can add up to 1 more. Per term the bound is mp * (1 + 2^-22) + 1, i.e.
 * N * (mp + 1) + N * mp / 2^22 <= room. For int12 x int12, mp = 2^22, so
 * N * (2^22 + 2) must fit: exactly room == 511 * (2^22 + 2) admits 511. */
void testRescaledTermLimitBoundaryIsTheSlackedBound(void) {
    uint64_t room511 = 511ull * ((1ull << 22) + 2ull);
    uint64_t seed = (uint64_t)INT32_MAX - room511;
    TEST_ASSERT_EQUAL_size_t(511, symAccumTermLimitRescaled(12, 12, seed));
    TEST_ASSERT_EQUAL_size_t(510, symAccumTermLimitRescaled(12, 12, seed + 1));
    /* the unslacked bound still admits 511 there: the slack is what tightens it */
    TEST_ASSERT_EQUAL_size_t(511, symAccumTermLimit(12, 12, seed + 1));
}

void testRescaledTermLimitIsNeverLooserThanTheExactLimit(void) {
    for (uint8_t a = 1; a <= 20; a++) {
        for (uint8_t b = 1; b <= 20; b++) {
            TEST_ASSERT_TRUE(symAccumTermLimitRescaled(a, b, 0) <= symAccumTermLimit(a, b, 0));
        }
    }
    TEST_ASSERT_EQUAL_size_t(0, symAccumTermLimitRescaled(16, 17, 0));
    TEST_ASSERT_EQUAL_size_t(0, symAccumTermLimitRescaled(12, 12, 2147483648ull));
    TEST_ASSERT_EQUAL_size_t(0, symAccumTermLimitRescaled(32, 32, 0));
}

void testRescaledSiteUsesTheSlackedLimit(void) {
    uint64_t seed = (uint64_t)INT32_MAX - 511ull * ((1ull << 22) + 2ull) + 1ull;
    symAccumSite_t plain = symAccumSiteInit("t", 511, 12, 12, seed);
    symAccumSite_t rescaled = symAccumSiteInitRescaled("t", 511, 12, 12, seed);
    TEST_ASSERT_TRUE(symAccumProvenSafe(&plain));
    TEST_ASSERT_FALSE(symAccumProvenSafe(&rescaled));
    symAccumSite_t zero = symAccumSiteInitRescaled("t", 0, 32, 32, 2147483648ull);
    TEST_ASSERT_TRUE(symAccumProvenSafe(&zero));
}

/* A value sum is a product site with b = 1 bit (|term| <= 2^(q-1)): the proof
 * boundary of a 16-bit wire is N = 65535, of a 24-bit wire N = 255. */
void testSumSiteProofBoundaryFollowsWireWidth(void) {
    symAccumSite_t s16 = symAccumSiteInitSum("sumop", 65535, 16);
    TEST_ASSERT_TRUE(symAccumProvenSafe(&s16));
    s16.nMax = 65536;
    TEST_ASSERT_FALSE(symAccumProvenSafe(&s16));
    symAccumSite_t s24 = symAccumSiteInitSum("sumop", 255, 24);
    TEST_ASSERT_TRUE(symAccumProvenSafe(&s24));
    s24.nMax = 256;
    TEST_ASSERT_FALSE(symAccumProvenSafe(&s24));
}

/* The sum banner names the wire width and its own remedies, never the operand
 * clamp knob that does not govern a stored wire. */
void testSumSiteBannerNamesWireWidthNotOperandKnob(void) {
    symAccumSite_t s = symAccumSiteInitSum("sumop", 3, 31);
    ASSERT_EXITS_WITH_OUTPUT(1, "sumop: SYM int32 accumulator overflow -- N=3 terms",
                             (void)symAccumAddChecked(INT32_MAX, 1, &s));
    ASSERT_EXITS_WITH_OUTPUT(1, "value sum of 31-bit codes",
                             (void)symAccumAddChecked(INT32_MAX, 1, &s));
    ASSERT_EXITS_WITH_OUTPUT(1, "ARITH_FLOAT32", (void)symAccumAddChecked(INT32_MAX, 1, &s));
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testRescaledTermLimitBoundaryIsTheSlackedBound);
    RUN_TEST(testRescaledTermLimitIsNeverLooserThanTheExactLimit);
    RUN_TEST(testRescaledSiteUsesTheSlackedLimit);
    RUN_TEST(testTermLimitInt12Seed0Is511);
    RUN_TEST(testProvenSafeBoundaryAt511And512);
    RUN_TEST(testSeedThatTipsTheBoundShrinksTheLimit);
    RUN_TEST(testMaxAbsSeedHandlesInt32Min);
    RUN_TEST(testInt32MinSeedProvesOnlyZeroTerms);
    RUN_TEST(testInt16By16AdmitsExactlyOneTerm);
    RUN_TEST(testWidthsSummingPastTheInt32ProductProveNothing);
    RUN_TEST(testZeroTermsAreAlwaysProvenSafe);
    RUN_TEST(testSizeMaxDoesNotWrapTheBound);
    RUN_TEST(testWideProductFlagFollowsOperandWidths);
    RUN_TEST(testCheckedAddReturnsExactSumWithoutOverflow);
    RUN_TEST(testCheckedAddAbortsOnPositiveAndNegativeOverflow);
    RUN_TEST(testBannerNamesNWidthsSeedAndRemedies);
    RUN_TEST(testMacCheckedExactResultAndWideProductAbort);
    RUN_TEST(testProductCheckedAbortsOnlyWhenWideAndOverflowing);
    RUN_TEST(testSumSiteProofBoundaryFollowsWireWidth);
    RUN_TEST(testSumSiteBannerNamesWireWidthNotOperandKnob);
    return UNITY_END();
}
