#ifndef ODT_TEST_BFP_RAW_FIXTURE_H
#define ODT_TEST_BFP_RAW_FIXTURE_H

/*
 * Stack fixtures for the integer-only BFP kernels (test-only): the forward
 * and the Conv1d dx scatter.
 *
 * A kernel under ARITH_BFP sees its operands in the funnel's UNPACKED scratch
 * form (int32 codes under a live bfpQConfig_t) and, when the op opts in
 * (opSpec_t.bfpRawOut), writes a BFP_RAW output: int32 mantissas plus one
 * UNBIASED int16 exponent per raw target group. Tests that call such a kernel
 * directly build both forms here. Include after unity.h: the asserts are
 * Unity's.
 */

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "Quantization.h"
#include "Tensor.h"
#include "unity.h"

/* Unpacked-BFP scratch operand on the stack; `codes` is borrowed, the
 * exponents are copied so a kernel bug cannot corrupt a gold table. */
typedef struct bfpScratchFixture {
    tensor_t tensor;
    shape_t shape;
    size_t dims[3];
    size_t order[3];
    uint8_t exps[64];
    bfpQConfig_t qc;
    quantization_t q;
} bfpScratchFixture_t;

static inline void bfpScratchInit(bfpScratchFixture_t *f, size_t ndim, const size_t *dims,
                                  const int32_t *codes, const uint8_t *exps, size_t numGroups,
                                  size_t groupSize, uint8_t m, uint8_t e) {
    TEST_ASSERT_TRUE(ndim <= 3 && numGroups <= sizeof(f->exps));
    for (size_t i = 0; i < ndim; i++) {
        f->dims[i] = dims[i];
        f->order[i] = i;
    }
    setShape(&f->shape, f->dims, ndim, f->order);
    memcpy(f->exps, exps, numGroups);
    f->qc = (bfpQConfig_t){.exponents = f->exps,
                           .numGroups = numGroups,
                           .groupSize = groupSize,
                           .roundingMode = HALF_AWAY,
                           .mantissaBits = m,
                           .exponentBits = e};
    initBfpQuantization(&f->qc, &f->q);
    setTensorValues(&f->tensor, (uint8_t *)codes, &f->shape, &f->q, NULL);
}

#define BFP_RAW_FIXTURE_MAX_OUT 128
#define BFP_RAW_FIXTURE_MAX_GROUPS 16
#define BFP_RAW_SENTINEL_MANT (INT32_MIN + 7)
#define BFP_RAW_SENTINEL_EXP ((int16_t)0x7ABC)

/* BFP_RAW output on the stack, sentinel-prefilled: every mantissa and every
 * group exponent must be kernel-written. */
typedef struct bfpRawFixture {
    tensor_t tensor;
    shape_t shape;
    size_t dims[3];
    size_t order[3];
    int32_t mant[BFP_RAW_FIXTURE_MAX_OUT];
    int16_t exps[BFP_RAW_FIXTURE_MAX_GROUPS];
    bfpRawQConfig_t qc;
    quantization_t q;
} bfpRawFixture_t;

static inline void bfpRawInit(bfpRawFixture_t *f, size_t ndim, const size_t *dims, size_t numGroups,
                              size_t groupSize) {
    size_t n = 1;
    TEST_ASSERT_TRUE(ndim <= 3);
    for (size_t i = 0; i < ndim; i++) {
        f->dims[i] = dims[i];
        f->order[i] = i;
        n *= dims[i];
    }
    /* strict: assertRawEquals reads one sentinel past the output and the groups */
    TEST_ASSERT_TRUE(n < BFP_RAW_FIXTURE_MAX_OUT && numGroups < BFP_RAW_FIXTURE_MAX_GROUPS);
    setShape(&f->shape, f->dims, ndim, f->order);
    for (size_t i = 0; i < BFP_RAW_FIXTURE_MAX_OUT; i++) {
        f->mant[i] = BFP_RAW_SENTINEL_MANT;
    }
    for (size_t g = 0; g < BFP_RAW_FIXTURE_MAX_GROUPS; g++) {
        f->exps[g] = BFP_RAW_SENTINEL_EXP;
    }
    f->qc = (bfpRawQConfig_t){.exponents = f->exps, .numGroups = numGroups, .groupSize = groupSize};
    initBfpRawQuantization(&f->qc, &f->q);
    setTensorValues(&f->tensor, (uint8_t *)f->mant, &f->shape, &f->q, NULL);
}

static inline void assertRawEquals(const bfpRawFixture_t *f, const int32_t *mant, size_t n,
                                   const int16_t *exps, size_t numGroups) {
    TEST_ASSERT_EQUAL_INT32_ARRAY(mant, f->mant, n);
    TEST_ASSERT_EQUAL_INT16_ARRAY(exps, f->exps, numGroups);
    TEST_ASSERT_EQUAL_INT32(BFP_RAW_SENTINEL_MANT, f->mant[n]); /* nothing past the output */
    TEST_ASSERT_EQUAL_INT16(BFP_RAW_SENTINEL_EXP, f->exps[numGroups]);
}

/* a * 2^ea == b * 2^eb exactly (|a|, |b| < 2^62): an output value against the
 * test's own exact int64 sum, independent of how the kernel split it into
 * mantissa and exponent. */
static inline bool bfpValueExactlyEqual(int64_t a, int ea, int64_t b, int eb) {
    if (ea < eb) {
        int64_t t = a;
        a = b;
        b = t;
        int te = ea;
        ea = eb;
        eb = te;
    }
    if (a == 0) {
        return b == 0;
    }
    int d = ea - eb;
    if (d > 61) {
        return false;
    }
    int64_t lim = (int64_t)1 << (62 - d);
    if (a >= lim || a <= -lim) {
        return false;
    }
    return a * ((int64_t)1 << d) == b;
}

static inline void assertBfpValueExact(int64_t mant, int e, int64_t exactSum, int exactE,
                                       size_t idx) {
    char msg[160];
    snprintf(msg, sizeof(msg), "output %zu: %lld * 2^%d != exact %lld * 2^%d", idx, (long long)mant,
             e, (long long)exactSum, exactE);
    TEST_ASSERT_TRUE_MESSAGE(bfpValueExactlyEqual(mant, e, exactSum, exactE), msg);
}

#endif /* ODT_TEST_BFP_RAW_FIXTURE_H */
