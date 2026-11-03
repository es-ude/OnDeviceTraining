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

/* ---- The declared result width is the exact two's-complement range -------- */

/* Smallest w with maxPos <= 2^(w-1) - 1 and maxNegMag <= 2^(w-1): the
 * negative side holds one more magnitude than the positive one; 32 caps. */
void testWidthForRangeIsTheSmallestTwosComplementWidth(void) {
    TEST_ASSERT_EQUAL_UINT8(1, symAccumWidthForRange(0, 0));
    TEST_ASSERT_EQUAL_UINT8(1, symAccumWidthForRange(0, 1));
    TEST_ASSERT_EQUAL_UINT8(2, symAccumWidthForRange(1, 0));
    TEST_ASSERT_EQUAL_UINT8(2, symAccumWidthForRange(1, 2));
    TEST_ASSERT_EQUAL_UINT8(3, symAccumWidthForRange(2, 0));
    TEST_ASSERT_EQUAL_UINT8(3, symAccumWidthForRange(0, 3));
    TEST_ASSERT_EQUAL_UINT8(12, symAccumWidthForRange(2047, 2048));
    TEST_ASSERT_EQUAL_UINT8(13, symAccumWidthForRange(2048, 0));
    TEST_ASSERT_EQUAL_UINT8(13, symAccumWidthForRange(0, 2049));
    TEST_ASSERT_EQUAL_UINT8(31, symAccumWidthForRange((1ull << 30) - 1, 1ull << 30));
    TEST_ASSERT_EQUAL_UINT8(32, symAccumWidthForRange(1ull << 30, 0));
    TEST_ASSERT_EQUAL_UINT8(32, symAccumWidthForRange(0, (1ull << 30) + 1));
    TEST_ASSERT_EQUAL_UINT8(32, symAccumWidthForRange(UINT64_MAX, UINT64_MAX));
}

/* A value sum adds the codes themselves: N * (2^(q-1) - 1) on the positive
 * side, N * 2^(q-1) on the negative -- not a product with a one-bit code in
 * {-1, 0}, which mirrors the sides and costs a bit. int12: N=1 -> 12 (one code
 * adds no bit), N=2 -> 13, N=3 -> 14; an unprovable sum -> 32. */
void testResultBitsOfAValueSumIsTheCodeRange(void) {
    symAccumSite_t one = symAccumSiteInitSum("t", 1, 12);
    symAccumSite_t two = symAccumSiteInitSum("t", 2, 12);
    symAccumSite_t three = symAccumSiteInitSum("t", 3, 12);
    symAccumSite_t unprovable = symAccumSiteInitSum("t", 2, 31);
    TEST_ASSERT_EQUAL_UINT8(12, symAccumResultBits(&one));
    TEST_ASSERT_EQUAL_UINT8(13, symAccumResultBits(&two));
    TEST_ASSERT_EQUAL_UINT8(14, symAccumResultBits(&three));
    TEST_ASSERT_EQUAL_UINT8(32, symAccumResultBits(&unprovable));
}

/* A product's positive corner (-2^(qa-1)) * (-2^(qb-1)) = 2^(qa+qb-2) is one
 * past what the negative side -2^(qa-1) * (2^(qb-1) - 1) reaches, so a product
 * site keeps the corner's bit: int12 x int12 -> 24, int4 x int4 -> 8. Zero
 * terms and no seed -> 1; an unprovable site -> 32. */
void testResultBitsOfAProductKeepsThePositiveCorner(void) {
    symAccumSite_t int12 = symAccumSiteInit("t", 1, 12, 12, 0);
    symAccumSite_t int4 = symAccumSiteInit("t", 1, 4, 4, 0);
    symAccumSite_t none = symAccumSiteInit("t", 0, 12, 12, 0);
    symAccumSite_t unprovable = symAccumSiteInit("t", 512, 12, 12, 0);
    TEST_ASSERT_EQUAL_UINT8(24, symAccumResultBits(&int12));
    TEST_ASSERT_EQUAL_UINT8(8, symAccumResultBits(&int4));
    TEST_ASSERT_EQUAL_UINT8(1, symAccumResultBits(&none));
    TEST_ASSERT_EQUAL_UINT8(32, symAccumResultBits(&unprovable));
}

/* The seed shifts the range, so its SIGN picks the side it widens. One
 * int4 x int4 term spans [-56, 64]: seed +8 -> [-48, 72] -> 8 bits, seed -8 ->
 * [-64, 56] -> 7. Four terms with seed -1: [-225, 255] -> 9, where the
 * seedless site (256) and a magnitude-only seed say 10. Three terms with seed
 * -1870: [-2038, -1678] -> 12, where max|seed| on both sides (2062) says 13. */
void testResultBitsSeedSignPicksTheSideItWidens(void) {
    int32_t plus8 = 8, minus8 = -8, minus1 = -1, minus1870 = -1870;
    symAccumSite_t one = symAccumSiteInit("t", 1, 4, 4, 0);
    symAccumSiteSetSeeds(&one, &plus8, 1);
    TEST_ASSERT_EQUAL_UINT8(8, symAccumResultBits(&one));
    symAccumSiteSetSeeds(&one, &minus8, 1);
    TEST_ASSERT_EQUAL_UINT8(7, symAccumResultBits(&one));
    symAccumSite_t four = symAccumSiteInit("t", 4, 4, 4, 0);
    TEST_ASSERT_EQUAL_UINT8(10, symAccumResultBits(&four));
    symAccumSiteSetSeeds(&four, &minus1, 1);
    TEST_ASSERT_EQUAL_UINT8(9, symAccumResultBits(&four));
    symAccumSite_t three = symAccumSiteInit("t", 3, 4, 4, 1870);
    TEST_ASSERT_EQUAL_UINT8(13, symAccumResultBits(&three));
    symAccumSiteSetSeeds(&three, &minus1870, 1);
    TEST_ASSERT_EQUAL_UINT8(12, symAccumResultBits(&three));
}

/* The seed fields: the array's signed extremes AND their max magnitude (the
 * proof's seed room, INT32_MIN -> 2^31); NULL / no seeds -> [0, 0], the value
 * a seedless kernel starts from. A magnitude-only site reads its seed as
 * [-M, +M] within int32. */
void testSetSeedsFillsTheSignedRangeAndTheProofMagnitude(void) {
    int32_t seeds[] = {5, -7, INT32_MIN, 3};
    symAccumSite_t s = symAccumSiteInit("t", 1, 4, 4, 0);
    symAccumSiteSetSeeds(&s, seeds, 4);
    TEST_ASSERT_EQUAL_INT32(5, s.seedMax);
    TEST_ASSERT_EQUAL_INT32(INT32_MIN, s.seedMin);
    TEST_ASSERT_EQUAL_UINT64(2147483648ull, s.maxAbsSeed);
    symAccumSiteSetSeeds(&s, NULL, 0);
    TEST_ASSERT_EQUAL_INT32(0, s.seedMax);
    TEST_ASSERT_EQUAL_INT32(0, s.seedMin);
    TEST_ASSERT_EQUAL_UINT64(0ull, s.maxAbsSeed);
    int32_t positive[] = {5, 3};
    symAccumSiteSetSeeds(&s, positive, 2);
    TEST_ASSERT_EQUAL_INT32(5, s.seedMax);
    TEST_ASSERT_EQUAL_INT32(3, s.seedMin);
    TEST_ASSERT_EQUAL_UINT64(5ull, s.maxAbsSeed);
    symAccumSite_t magnitude = symAccumSiteInit("t", 1, 4, 4, 1000);
    TEST_ASSERT_EQUAL_INT32(1000, magnitude.seedMax);
    TEST_ASSERT_EQUAL_INT32(-1000, magnitude.seedMin);
    TEST_ASSERT_EQUAL_UINT64(1000ull, magnitude.maxAbsSeed);
    symAccumSite_t int32Min = symAccumSiteInit("t", 0, 32, 32, 2147483648ull);
    TEST_ASSERT_EQUAL_INT32(INT32_MAX, int32Min.seedMax);
    TEST_ASSERT_EQUAL_INT32(INT32_MIN, int32Min.seedMin);
}

/* The rescaled slack (nMax + ceil(nMax * t / 2^22)) applies on each side.
 * (4, 5) bits, 8 terms: the positive side 8 * 128 = 1024 slacks to 1033, the
 * negative 8 * 120 = 960 to 969. Seed +1023: 2056 vs 2047 exact (13 vs 12);
 * seed -1080: the negative side 2049 vs 2040 exact (13 vs 12), the positive
 * side gone. */
void testResultBitsRescaledSlackAppliesOnBothSides(void) {
    int32_t minus1080 = -1080;
    symAccumSite_t exactPos = symAccumSiteInit("t", 8, 4, 5, 1023);
    symAccumSite_t rescaledPos = symAccumSiteInitRescaled("t", 8, 4, 5, 1023);
    TEST_ASSERT_EQUAL_UINT8(12, symAccumResultBits(&exactPos));
    TEST_ASSERT_EQUAL_UINT8(13, symAccumResultBits(&rescaledPos));
    symAccumSite_t exactNeg = symAccumSiteInit("t", 8, 4, 5, 0);
    symAccumSite_t rescaledNeg = symAccumSiteInitRescaled("t", 8, 4, 5, 0);
    symAccumSiteSetSeeds(&exactNeg, &minus1080, 1);
    symAccumSiteSetSeeds(&rescaledNeg, &minus1080, 1);
    TEST_ASSERT_EQUAL_UINT8(12, symAccumResultBits(&exactNeg));
    TEST_ASSERT_EQUAL_UINT8(13, symAccumResultBits(&rescaledNeg));
}

/* ---- Per-operand code ranges (the grouped kernels' weight, SymGroupedWeight.h) ---- */

/* A site built from explicit signed operand ranges: one term's range is the
 * product's four corners, and the proof reads the larger side's magnitude.
 * int12 [-2048, 2047] against a stored ASYM weight's image [-4095, 4095] is
 * +-(2048 * 4095) = +-8386560 per term: 256 terms (2146959360) are proven and
 * 257 are not, where the image's holding width 13 ([-4096, 4095], 2^23 per
 * term) proves only 255 -- the rescaled site too. The width-based inits are
 * the two's-complement ranges. */
void testRangeSiteProofReadsTheOperandRanges(void) {
    symAccumSite_t n256 = symAccumSiteInitRanges("t", 256, -2048, 2047, -4095, 4095, 0);
    symAccumSite_t n257 = symAccumSiteInitRanges("t", 257, -2048, 2047, -4095, 4095, 0);
    TEST_ASSERT_TRUE(symAccumProvenSafe(&n256));
    TEST_ASSERT_FALSE(symAccumProvenSafe(&n257));
    TEST_ASSERT_EQUAL_size_t(256, symAccumSiteTermLimit(&n256));
    symAccumSite_t width13 = symAccumSiteInit("t", 256, 12, 13, 0);
    TEST_ASSERT_FALSE(symAccumProvenSafe(&width13));
    TEST_ASSERT_EQUAL_size_t(255, symAccumSiteTermLimit(&width13));
    symAccumSite_t rescaled256 =
        symAccumSiteInitRangesRescaled("t", 256, -2048, 2047, -4095, 4095, 0);
    symAccumSite_t rescaled257 =
        symAccumSiteInitRangesRescaled("t", 257, -2048, 2047, -4095, 4095, 0);
    TEST_ASSERT_TRUE(symAccumProvenSafe(&rescaled256));
    TEST_ASSERT_FALSE(symAccumProvenSafe(&rescaled257));
    symAccumSite_t byWidth = symAccumSiteInit("t", 511, 12, 12, 0);
    symAccumSite_t byRange = symAccumSiteInitRanges("t", 511, -2048, 2047, -2048, 2047, 0);
    TEST_ASSERT_EQUAL_INT32(-2048, byWidth.aMin);
    TEST_ASSERT_EQUAL_INT32(2047, byWidth.bMax);
    TEST_ASSERT_TRUE(symAccumProvenSafe(&byWidth));
    TEST_ASSERT_TRUE(symAccumProvenSafe(&byRange));
    byWidth.nMax = byRange.nMax = 512;
    TEST_ASSERT_FALSE(symAccumProvenSafe(&byWidth));
    TEST_ASSERT_FALSE(symAccumProvenSafe(&byRange));
}

/* The declared width follows the corners too. int4 [-8, 7] against the image
 * [-15, 15] is +-120 per term (one term -> 8 bits); 8 rescaled terms are 969
 * per side (slack 9). Seed +54 -> 1023 -> 11, where the image's holding width
 * 5 ([-16, 15], 128 per term, 1033 + 54) says 12; seed +500 -> 1469 -> 12,
 * where the true SYM int4 weight (64 per term, 521 + 500) says 11; seed -500
 * lands on the negative side, 1469 -> 12 as well. */
void testRangeSiteResultBitsFollowTheCorners(void) {
    int32_t minus500 = -500;
    symAccumSite_t one = symAccumSiteInitRanges("t", 1, -8, 7, -15, 15, 0);
    TEST_ASSERT_EQUAL_UINT8(8, symAccumResultBits(&one));
    symAccumSite_t image54 = symAccumSiteInitRangesRescaled("t", 8, -8, 7, -15, 15, 54);
    symAccumSite_t width54 = symAccumSiteInitRescaled("t", 8, 4, 5, 54);
    TEST_ASSERT_EQUAL_UINT8(11, symAccumResultBits(&image54));
    TEST_ASSERT_EQUAL_UINT8(12, symAccumResultBits(&width54));
    symAccumSite_t image500 = symAccumSiteInitRangesRescaled("t", 8, -8, 7, -15, 15, 500);
    symAccumSite_t sym500 = symAccumSiteInitRescaled("t", 8, 4, 4, 500);
    TEST_ASSERT_EQUAL_UINT8(12, symAccumResultBits(&image500));
    TEST_ASSERT_EQUAL_UINT8(11, symAccumResultBits(&sym500));
    symAccumSite_t imageNeg = symAccumSiteInitRangesRescaled("t", 8, -8, 7, -15, 15, 0);
    symAccumSiteSetSeeds(&imageNeg, &minus500, 1);
    TEST_ASSERT_EQUAL_UINT8(12, symAccumResultBits(&imageNeg));
}

/* The banner names each range by its holding width: the image +-4095 is 13
 * bits, +-65535 is 17 (which also turns on the wide-product check). */
void testRangeSiteHoldingWidthsNameTheBanner(void) {
    symAccumSite_t image = symAccumSiteInitRanges("t", 1, -2048, 2047, -4095, 4095, 0);
    TEST_ASSERT_EQUAL_UINT8(12, image.aBits);
    TEST_ASSERT_EQUAL_UINT8(13, image.bBits);
    TEST_ASSERT_FALSE(image.wideProduct);
    symAccumSite_t wide = symAccumSiteInitRanges("bannerop", 600, -2048, 2047, -65535, 65535, 0);
    TEST_ASSERT_EQUAL_UINT8(17, wide.bBits);
    TEST_ASSERT_TRUE(wide.wideProduct);
    ASSERT_EXITS_WITH_OUTPUT(1, "widths (12, 17)", (void)symAccumAddChecked(INT32_MAX, 1, &wide));
}

/* A value sum is the code range against the constant 1, never a mirrored
 * one-bit code; the banner keeps the one-bit second width. */
void testSumSiteIsTheCodeRangeAgainstOne(void) {
    symAccumSite_t s = symAccumSiteInitSum("t", 3, 12);
    TEST_ASSERT_EQUAL_INT32(-2048, s.aMin);
    TEST_ASSERT_EQUAL_INT32(2047, s.aMax);
    TEST_ASSERT_EQUAL_INT32(1, s.bMin);
    TEST_ASSERT_EQUAL_INT32(1, s.bMax);
    TEST_ASSERT_EQUAL_UINT8(12, s.aBits);
    TEST_ASSERT_EQUAL_UINT8(1, s.bBits);
    TEST_ASSERT_TRUE(s.valueSum);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testRangeSiteProofReadsTheOperandRanges);
    RUN_TEST(testRangeSiteResultBitsFollowTheCorners);
    RUN_TEST(testRangeSiteHoldingWidthsNameTheBanner);
    RUN_TEST(testSumSiteIsTheCodeRangeAgainstOne);
    RUN_TEST(testWidthForRangeIsTheSmallestTwosComplementWidth);
    RUN_TEST(testResultBitsOfAValueSumIsTheCodeRange);
    RUN_TEST(testResultBitsOfAProductKeepsThePositiveCorner);
    RUN_TEST(testResultBitsSeedSignPicksTheSideItWidens);
    RUN_TEST(testSetSeedsFillsTheSignedRangeAndTheProofMagnitude);
    RUN_TEST(testResultBitsRescaledSlackAppliesOnBothSides);
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
