#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "BfpSoftmaxExp.h"
#include "DeathTest.h"
#include "RNG.h"
#include "expected_bfp_softmax_core.h"
#include "unity.h"

/* Pins the three derived constants TWICE: against an independent C double
 * recomputation (a goldgen bug would emit the same wrong number twice; the
 * in-C derivation cannot follow it) AND against the goldgen-emitted
 * EXPECTED_* values (which catch a #define typo the C derivation shares). */
void testShiftConstantsMatchDerivation() {
    const int32_t qln2C = (int32_t)floor(log(2.0) * 16384.0);
    const int32_t qbC = (int32_t)floor(1.353 * 16384.0);
    const int32_t qcC = (int32_t)floor(0.344 * 268435456.0 / 0.3585);

    TEST_ASSERT_EQUAL_INT32(qln2C, BFP_SOFTMAX_QLN2);
    TEST_ASSERT_EQUAL_INT32(EXPECTED_QLN2, BFP_SOFTMAX_QLN2);
    TEST_ASSERT_EQUAL_INT32(qbC, BFP_SOFTMAX_QB);
    TEST_ASSERT_EQUAL_INT32(EXPECTED_QB, BFP_SOFTMAX_QB);
    TEST_ASSERT_EQUAL_INT32(qcC, BFP_SOFTMAX_QC);
    TEST_ASSERT_EQUAL_INT32(EXPECTED_QC, BFP_SOFTMAX_QC);

    TEST_ASSERT_EQUAL_INT32(14, BFP_SOFTMAX_EXP_FRAC_BITS);
    TEST_ASSERT_EQUAL_INT32(31, BFP_SOFTMAX_ZMAX);
    TEST_ASSERT_EQUAL_INT32(-32 * qln2C, BFP_SOFTMAX_QFLOOR);
    TEST_ASSERT_EQUAL_INT32(EXPECTED_QFLOOR, BFP_SOFTMAX_QFLOOR);
    TEST_ASSERT_FLOAT_WITHIN(0.0f, 0.3585f, BFP_SOFTMAX_A);
}

void testShiftTruncVectors() {
    char msg[64];
    for (int i = 0; i < SHIFT_NUM_V; i++) {
        for (int j = 0; j < SHIFT_NUM_K; j++) {
            snprintf(msg, sizeof msg, "v=%d k=%u", (int)kShiftV[i], (unsigned)kShiftK[j]);
            TEST_ASSERT_EQUAL_INT32_MESSAGE(
                kShiftTruncExpected[i][j],
                bfpShiftRightRounded(kShiftV[i], kShiftK[j], BFP_SHIFT_TRUNC, NULL), msg);
        }
        /* k >= 31 clamps to 31 (column SHIFT_NUM_K-1 is k=31) */
        TEST_ASSERT_EQUAL_INT32(kShiftTruncExpected[i][SHIFT_NUM_K - 1],
                                bfpShiftRightRounded(kShiftV[i], 32u, BFP_SHIFT_TRUNC, NULL));
        TEST_ASSERT_EQUAL_INT32(kShiftTruncExpected[i][SHIFT_NUM_K - 1],
                                bfpShiftRightRounded(kShiftV[i], 100u, BFP_SHIFT_TRUNC, NULL));
    }
}

void testShiftHalfAwayVectors() {
    char msg[64];
    for (int i = 0; i < SHIFT_NUM_V; i++) {
        for (int j = 0; j < SHIFT_NUM_K; j++) {
            snprintf(msg, sizeof msg, "v=%d k=%u", (int)kShiftV[i], (unsigned)kShiftK[j]);
            TEST_ASSERT_EQUAL_INT32_MESSAGE(
                kShiftHalfAwayExpected[i][j],
                bfpShiftRightRounded(kShiftV[i], kShiftK[j], BFP_SHIFT_HALF_AWAY, NULL), msg);
        }
        TEST_ASSERT_EQUAL_INT32(kShiftHalfAwayExpected[i][SHIFT_NUM_K - 1],
                                bfpShiftRightRounded(kShiftV[i], 32u, BFP_SHIFT_HALF_AWAY, NULL));
        TEST_ASSERT_EQUAL_INT32(kShiftHalfAwayExpected[i][SHIFT_NUM_K - 1],
                                bfpShiftRightRounded(kShiftV[i], 100u, BFP_SHIFT_HALF_AWAY, NULL));
    }
}

/* rem == 0 must never increment: SR of (v << k) by k returns v EXACTLY, on
 * every draw (10 repeats per pair; the Bernoulli threshold compare is
 * `< rem` with rem == 0, unsatisfiable for the non-negative cast draw). */
void testShiftSrExactWhenRemainderZero() {
    static const int32_t base[] = {13, -13, 22167, -22167, 5, -5, 0};
    static const uint32_t ks[] = {1u, 3u, 14u};
    rngStream_t rs = rngOpen(rngKeyFromSeed(9001u));
    for (size_t b = 0; b < sizeof base / sizeof base[0]; b++) {
        for (size_t j = 0; j < sizeof ks / sizeof ks[0]; j++) {
            const int32_t v = base[b] * (int32_t)(1u << ks[j]);
            for (int rep = 0; rep < 10; rep++) {
                TEST_ASSERT_EQUAL_INT32(base[b], bfpShiftRightRounded(v, ks[j], BFP_SHIFT_SR, &rs));
            }
        }
    }
}

/* SR is floor-domain: every draw lands in {floor, floor+1}, and for a half
 * remainder both values must occur (kills an "always floor" mutant). */
void testShiftSrBracketsFloor() {
    struct {
        int32_t v;
        uint32_t k;
        int32_t floorQ;
    } cases[] = {
        {13, 1u, 6},   /* rem = 1 of 2: exact half     */
        {-13, 1u, -7}, /* rem = 1 of 2: exact half     */
        {5, 3u, 0},    /* rem = 5 of 8: p = 0.625      */
    };
    rngStream_t rs = rngOpen(rngKeyFromSeed(4242u));
    for (size_t c = 0; c < sizeof cases / sizeof cases[0]; c++) {
        int sawFloor = 0;
        int sawFloorPlusOne = 0;
        for (int draw = 0; draw < 100; draw++) {
            const int32_t r = bfpShiftRightRounded(cases[c].v, cases[c].k, BFP_SHIFT_SR, &rs);
            if (r == cases[c].floorQ) {
                sawFloor++;
            } else if (r == cases[c].floorQ + 1) {
                sawFloorPlusOne++;
            } else {
                TEST_FAIL_MESSAGE("SR result outside {floor, floor+1}");
            }
        }
        TEST_ASSERT_TRUE_MESSAGE(sawFloor > 0, "floor never drawn");
        TEST_ASSERT_TRUE_MESSAGE(sawFloorPlusOne > 0, "floor+1 never drawn");
    }
}

static void shiftSrSequence(rngKey_t key, int32_t out[32]) {
    rngStream_t rs = rngOpen(key);
    for (int i = 0; i < 32; i++) {
        out[i] =
            bfpShiftRightRounded(kShiftV[i % SHIFT_NUM_V], kShiftK[1 + (i % 4)], BFP_SHIFT_SR, &rs);
    }
}

/* SR shifts are a function of their stream: the same key gives the same 32
 * results, another key different ones. */
void testShiftSrIsAFunctionOfItsStream() {
    int32_t first[32];
    int32_t second[32];
    int32_t other[32];
    shiftSrSequence(rngKeyFromSeed(77u), first);
    shiftSrSequence(rngKeyFromSeed(77u), second);
    shiftSrSequence(rngKeyFromSeed(78u), other);
    TEST_ASSERT_EQUAL_INT32_ARRAY(first, second, 32);
    TEST_ASSERT_FALSE(memcmp(first, other, sizeof first) == 0);
}

static void shiftSrWithoutAStream(void) {
    (void)bfpShiftRightRounded(13, 1u, BFP_SHIFT_SR, NULL);
}

/* No key, no randomness: an SR shift by k > 0 without a stream stops at the
 * draw (a shift by 0 and the underflow floor never draw, and never stop). */
void testShiftSrWithoutAStreamExits() {
    ASSERT_EXITS_WITH_OUTPUT(1, "bfpShiftRightRounded: SR shift without a stream",
                             shiftSrWithoutAStream());
}

void testIExpVectorsTrunc() {
    char msg[64];
    for (int i = 0; i < IEXP_NUM_QW; i++) {
        snprintf(msg, sizeof msg, "qW=%ld", (long)kIExpQw[i]);
        TEST_ASSERT_EQUAL_INT32_MESSAGE(kIExpTruncExpected[i],
                                        bfpIExpQ(kIExpQw[i], BFP_SHIFT_TRUNC, NULL), msg);
    }
}

void testIExpVectorsHalfAway() {
    char msg[64];
    for (int i = 0; i < IEXP_NUM_QW; i++) {
        snprintf(msg, sizeof msg, "qW=%ld", (long)kIExpQw[i]);
        TEST_ASSERT_EQUAL_INT32_MESSAGE(kIExpHalfAwayExpected[i],
                                        bfpIExpQ(kIExpQw[i], BFP_SHIFT_HALF_AWAY, NULL), msg);
    }
}

/* The QFLOOR gate: qW <= QFLOOR is an EXACT zero in every mode, with no RNG
 * draw (the SR stream must stay untouched -- a dropped gate would advance it
 * and occasionally return 1 from the Bernoulli on the clamped 31-shift), so a
 * NULL stream never stops there either. */
void testIExpUnderflowFloorIsExactZero() {
    static const int32_t inputs[] = {BFP_SOFTMAX_QFLOOR, BFP_SOFTMAX_QFLOOR - 5, INT32_MIN};
    static const bfpShiftRounding_t modes[] = {BFP_SHIFT_TRUNC, BFP_SHIFT_HALF_AWAY, BFP_SHIFT_SR};
    for (size_t i = 0; i < sizeof inputs / sizeof inputs[0]; i++) {
        for (size_t m = 0; m < sizeof modes / sizeof modes[0]; m++) {
            rngStream_t rs = rngOpen(rngKeyFromSeed(42u));
            const uint32_t before = rs.state;
            for (int rep = 0; rep < 32; rep++) {
                TEST_ASSERT_EQUAL_INT32(0, bfpIExpQ(inputs[i], modes[m], NULL));
                TEST_ASSERT_EQUAL_INT32(0, bfpIExpQ(inputs[i], modes[m], &rs));
            }
            TEST_ASSERT_EQUAL_UINT32(before, rs.state);
        }
    }
}

/* 64-point sweep on [-20, 0]: the integer result is pinned bit-exact against
 * the goldgen, the dequantized float against the emitted float32 value, and
 * the value against libm exp() within the 2.5e-3 acceptance bound (worst
 * measured TRUNC gap is 2.089e-3 -- numerics spec). */
void testIExpAccuracyEnvelope() {
    char msg[96];
    for (int i = 0; i < IEXP_ACC_N; i++) {
        const int32_t q = bfpIExpQ(kIExpAccQw[i], BFP_SHIFT_TRUNC, NULL);
        snprintf(msg, sizeof msg, "x=%f", kIExpAccX[i]);
        TEST_ASSERT_EQUAL_INT32_MESSAGE(kIExpAccTruncExpected[i], q, msg);

        const float value = ldexpf((float)q * BFP_SOFTMAX_A, -28);
        TEST_ASSERT_FLOAT_WITHIN_MESSAGE(1e-6f, kIExpAccValue[i], value, msg);

        const double gap = fabs((double)value - exp(kIExpAccX[i]));
        snprintf(msg, sizeof msg, "x=%f gap=%e", kIExpAccX[i], gap);
        TEST_ASSERT_TRUE_MESSAGE(gap < 2.5e-3, msg);
    }
}

/* A shift by 0 draws nothing in any mode, so it needs no stream: even in SR mode
 * with a NULL stream it returns v unchanged. */
void testShiftByZeroNeedsNoStreamInAnyMode() {
    static const bfpShiftRounding_t modes[] = {BFP_SHIFT_TRUNC, BFP_SHIFT_HALF_AWAY, BFP_SHIFT_SR};
    for (size_t m = 0; m < sizeof modes / sizeof modes[0]; m++) {
        TEST_ASSERT_EQUAL_INT32(12345, bfpShiftRightRounded(12345, 0u, modes[m], NULL));
        TEST_ASSERT_EQUAL_INT32(-7, bfpShiftRightRounded(-7, 0u, modes[m], NULL));
    }
}

void setUp() {}
void tearDown() {}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testShiftConstantsMatchDerivation);
    RUN_TEST(testShiftTruncVectors);
    RUN_TEST(testShiftHalfAwayVectors);
    RUN_TEST(testShiftSrExactWhenRemainderZero);
    RUN_TEST(testShiftSrBracketsFloor);
    RUN_TEST(testShiftSrIsAFunctionOfItsStream);
    RUN_TEST(testShiftSrWithoutAStreamExits);
    RUN_TEST(testIExpVectorsTrunc);
    RUN_TEST(testIExpVectorsHalfAway);
    RUN_TEST(testIExpUnderflowFloorIsExactZero);
    RUN_TEST(testIExpAccuracyEnvelope);
    RUN_TEST(testShiftByZeroNeedsNoStreamInAnyMode);
    return UNITY_END();
}
