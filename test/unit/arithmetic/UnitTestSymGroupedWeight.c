#define SOURCE_FILE "UNIT_TEST_SYM_GROUPED_WEIGHT"

#include <stdint.h>
#include <string.h>

#include "DeathTest.h"
#include "SymGroupedWeight.h"
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

static float gScales[2] = {0.5f, 0.25f};
static uint16_t gZps[2] = {3, 9};

static symQConfig_t symGroups(size_t numGroups, size_t groupSize, uint8_t qBits) {
    return (symQConfig_t){.scales = gScales,
                          .numGroups = numGroups,
                          .groupSize = groupSize,
                          .roundingMode = HALF_AWAY,
                          .qBits = qBits};
}

static asymQConfig_t asymGroups(size_t numGroups, size_t groupSize, uint8_t qBits) {
    return (asymQConfig_t){.scales = gScales,
                           .zeroPoints = gZps,
                           .numGroups = numGroups,
                           .groupSize = groupSize,
                           .qBits = qBits,
                           .roundingMode = SR_HALF_AWAY};
}

static void assertRange(symCodeDomain_t domain, uint8_t qBits, int32_t min, int32_t max) {
    symQConfig_t groups = symGroups(2, 4, qBits);
    symGroupedWeight_t w = {.groups = &groups, .domain = domain};
    int32_t lo = 1, hi = -1;
    symGroupedWeightRange(&w, &lo, &hi);
    TEST_ASSERT_EQUAL_INT32(min, lo);
    TEST_ASSERT_EQUAL_INT32(max, hi);
}

/* A stored SYM weight unpacks to two's-complement qBits codes; a stored ASYM
 * weight to code - zp[g], which reaches +-(2^qBits - 1): one code wider on
 * each side than the same qBits read as two's complement. */
void testRangeFollowsTheCodeDomain(void) {
    assertRange(SYM_CODES_TWOS_COMPLEMENT, 4, -8, 7);
    assertRange(SYM_CODES_ASYM_IMAGE, 4, -15, 15);
    assertRange(SYM_CODES_TWOS_COMPLEMENT, 12, -2048, 2047);
    assertRange(SYM_CODES_ASYM_IMAGE, 12, -4095, 4095);
    assertRange(SYM_CODES_ASYM_IMAGE, 16, -65535, 65535);
    assertRange(SYM_CODES_TWOS_COMPLEMENT, 31, -1073741824, 1073741823);
}

void testUnknownCodeDomainAborts(void) {
    symQConfig_t groups = symGroups(2, 4, 4);
    symGroupedWeight_t w = {.groups = &groups, .domain = (symCodeDomain_t)7};
    int32_t lo, hi;
    ASSERT_EXITS_WITH_OUTPUT(1, "symGroupedWeightRange", symGroupedWeightRange(&w, &lo, &hi));
}

/* No valid tag is 0: a ctx whose .domain was never set (designated initializer
 * without it, memset, calloc) must fail fast naming the tag, never read as a
 * SYM weight -- the SYM range is the unsafe one for an ASYM image. */
void testZeroedCodeDomainAbortsInsteadOfReadingAsSym(void) {
    symQConfig_t groups = symGroups(2, 4, 4);
    symGroupedWeight_t omitted = {.groups = &groups};
    int32_t lo, hi;
    ASSERT_EXITS_WITH_OUTPUT(1, "symGroupedWeightRange: unknown code domain tag 0",
                             symGroupedWeightRange(&omitted, &lo, &hi));
    symGroupedWeight_t zeroed;
    memset(&zeroed, 0, sizeof zeroed);
    zeroed.groups = &groups;
    ASSERT_EXITS_WITH_OUTPUT(1, "code domain tag 0", symGroupedWeightRange(&zeroed, &lo, &hi));
    TEST_ASSERT_NOT_EQUAL(0, SYM_CODES_TWOS_COMPLEMENT);
    TEST_ASSERT_NOT_EQUAL(0, SYM_CODES_ASYM_IMAGE);
}

/* The builder tags a stored grouped SYM weight with its OWN config ... */
void testBuilderTagsAStoredGroupedSymWeight(void) {
    symQConfig_t groups = symGroups(2, 4, 4);
    quantization_t q = {.type = SYM, .qConfig = &groups};
    tensor_t t = {.quantization = &q};
    symQConfig_t view;
    symGroupedWeight_t out;
    const symGroupedWeight_t *w = symGroupedWeightOrNull(&t, &view, &out);
    TEST_ASSERT_EQUAL_PTR(&out, w);
    TEST_ASSERT_EQUAL_PTR(&groups, w->groups);
    TEST_ASSERT_EQUAL_INT(SYM_CODES_TWOS_COMPLEMENT, w->domain);
}

/* ... and a stored grouped ASYM weight with a symQConfig-shaped VIEW of its
 * config (scales borrowed, shape and qBits copied) under the image domain. */
void testBuilderViewsAStoredGroupedAsymWeightAsAnImage(void) {
    asymQConfig_t groups = asymGroups(2, 4, 6);
    quantization_t q = {.type = ASYM, .qConfig = &groups};
    tensor_t t = {.quantization = &q};
    symQConfig_t view;
    symGroupedWeight_t out;
    const symGroupedWeight_t *w = symGroupedWeightOrNull(&t, &view, &out);
    TEST_ASSERT_EQUAL_PTR(&out, w);
    TEST_ASSERT_EQUAL_PTR(&view, w->groups);
    TEST_ASSERT_EQUAL_INT(SYM_CODES_ASYM_IMAGE, w->domain);
    TEST_ASSERT_EQUAL_PTR(gScales, view.scales);
    TEST_ASSERT_EQUAL_size_t(2, view.numGroups);
    TEST_ASSERT_EQUAL_size_t(4, view.groupSize);
    TEST_ASSERT_EQUAL_UINT8(6, view.qBits);
    TEST_ASSERT_EQUAL_INT(SR_HALF_AWAY, view.roundingMode);
}

/* Per-tensor SYM / ASYM and every other dtype are not grouped weights. */
void testBuilderReturnsNullForEveryUngroupedWeight(void) {
    symQConfig_t perTensorSym = symGroups(1, 0, 4);
    asymQConfig_t perTensorAsym = asymGroups(1, 0, 4);
    symInt32QConfig_t symInt32 = {.scale = 1.f, .roundingMode = HALF_AWAY, .qMaxBits = 12};
    quantization_t qs[] = {{.type = SYM, .qConfig = &perTensorSym},
                           {.type = ASYM, .qConfig = &perTensorAsym},
                           {.type = SYM_INT32, .qConfig = &symInt32},
                           {.type = FLOAT32, .qConfig = NULL}};
    for (size_t i = 0; i < sizeof qs / sizeof qs[0]; i++) {
        tensor_t t = {.quantization = &qs[i]};
        symQConfig_t view;
        symGroupedWeight_t out;
        TEST_ASSERT_NULL(symGroupedWeightOrNull(&t, &view, &out));
    }
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testRangeFollowsTheCodeDomain);
    RUN_TEST(testUnknownCodeDomainAborts);
    RUN_TEST(testZeroedCodeDomainAbortsInsteadOfReadingAsSym);
    RUN_TEST(testBuilderTagsAStoredGroupedSymWeight);
    RUN_TEST(testBuilderViewsAStoredGroupedAsymWeightAsAnImage);
    RUN_TEST(testBuilderReturnsNullForEveryUngroupedWeight);
    return UNITY_END();
}
