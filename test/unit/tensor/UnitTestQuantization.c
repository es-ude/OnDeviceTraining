#include <stdbool.h>

#include "DeathTest.h"
#include "Quantization.h"
#include "Rounding.h"
#include "StorageApi.h"
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

/* Group-quant PR1 (docs/superpowers/plans/2026-07-28-groupquant-pr1-always-array.md):
 * symQConfig_t's scalar `scale` is replaced by an always-array `scales`
 * representation, behavior-identical for PR1 (numGroups always 1, groupSize
 * always 0 -- the "whole tensor" sentinel). initSymQConfig is the one PR1
 * producer of symQConfig_t and must uphold the invariant
 * numGroups == 1 <=> groupSize == 0 by construction.
 *
 * No death-test/validation choke point exists in PR1: initTensor (the only
 * candidate attach point) does not validate its quantization_t argument's
 * internal qConfig at all today, for any dtype, and adding that generic
 * hook is out of scope for a wire-stable mechanical migration. A hand-built
 * violating config (e.g. numGroups=2, groupSize=0) is therefore NOT rejected
 * anywhere in PR1 -- that fail-fast choke point is deferred to PR2, which
 * introduces the grouped-creation APIs the invariant actually guards.
 * Disclosed per the plan's Step 1(c) instruction. */

void testInitSymQConfigProducesPerTensorSentinel(void) {
    /* Mutation guard: hardcoding numGroups/groupSize to any other constant
     * pair, or forgetting to set groupSize to the sentinel, fails this. */
    symQConfig_t qc;
    initSymQConfig(6, SR_HALF_AWAY, &qc);

    size_t numGroups = qc.numGroups;
    size_t groupSize = qc.groupSize;
    uint8_t qBits = qc.qBits;
    roundingMode_t roundingMode = qc.roundingMode;
    float scale0 = qc.scales[0];

    freeReservedMemory(qc.scales);

    TEST_ASSERT_EQUAL_size_t(1, numGroups);
    TEST_ASSERT_EQUAL_size_t(0, groupSize);
    TEST_ASSERT_EQUAL_UINT8(6, qBits);
    TEST_ASSERT_EQUAL_INT(SR_HALF_AWAY, roundingMode);
    TEST_ASSERT_EQUAL_FLOAT(1.f, scale0);
}

void testInitSymQConfigAllocatesIndependentScalesArrays(void) {
    /* Two separate calls must not share a backing array -- each SYM config
     * owns its own heap block (no accidental static/shared storage).
     * Mutation guard: a `static float scales[1]` inside initSymQConfig
     * would make both pointers equal and this FAILS. */
    symQConfig_t a;
    symQConfig_t b;
    initSymQConfig(4, HALF_AWAY, &a);
    initSymQConfig(4, HALF_AWAY, &b);

    float *aScales = a.scales;
    float *bScales = b.scales;

    freeReservedMemory(a.scales);
    freeReservedMemory(b.scales);

    TEST_ASSERT_NOT_NULL(aScales);
    TEST_ASSERT_NOT_NULL(bScales);
    TEST_ASSERT_NOT_EQUAL(aScales, bScales);
}

/* Group-quant PR2 (docs/superpowers/specs/2026-07-28-group-quantization-design.md
 * §2/D3, Task 1): the grouped-creation API this PR2 introduces, plus the
 * shape-validation choke point PR1 deferred (see the disclosure above --
 * initTensor is the attach point; PR2 wires validateSymQConfigShape there). */

void testInitSymQConfigGroupedAllocatesPerGroupScales(void) {
    symQConfig_t qc;
    initSymQConfigGrouped(4, SR_HALF_AWAY, 3, 5, &qc);
    size_t numGroups = qc.numGroups;
    size_t groupSize = qc.groupSize;
    float s0 = qc.scales[0], s2 = qc.scales[2];
    qc.scales[2] = 7.f; /* writable per-group slot */
    float s2w = qc.scales[2];
    freeReservedMemory(qc.scales);
    TEST_ASSERT_EQUAL_size_t(3, numGroups);
    TEST_ASSERT_EQUAL_size_t(5, groupSize);
    TEST_ASSERT_EQUAL_FLOAT(1.f, s0);
    TEST_ASSERT_EQUAL_FLOAT(1.f, s2);
    TEST_ASSERT_EQUAL_FLOAT(7.f, s2w);
}

void testInitSymQConfigGroupedRejectsSentinelViolations(void) {
    symQConfig_t qc;
    /* numGroups>1 with groupSize==0 violates the invariant */
    ASSERT_EXITS_WITH(1, { initSymQConfigGrouped(4, HALF_AWAY, 2, 0, &qc); });
    /* numGroups==1 with groupSize!=0 is the non-canonical {1,N} form */
    ASSERT_EXITS_WITH(1, { initSymQConfigGrouped(4, HALF_AWAY, 1, 8, &qc); });
    ASSERT_EXITS_WITH(1, { initSymQConfigGrouped(4, HALF_AWAY, 0, 0, &qc); });
}

void testValidateSymQConfigShapeDivisibility(void) {
    float s[2] = {1.f, 1.f};
    symQConfig_t ok = {
        .scales = s, .numGroups = 2, .groupSize = 5, .roundingMode = HALF_AWAY, .qBits = 4};
    validateSymQConfigShape(&ok, 10); /* must NOT exit */
    symQConfig_t bad = {
        .scales = s, .numGroups = 2, .groupSize = 4, .roundingMode = HALF_AWAY, .qBits = 4};
    ASSERT_EXITS_WITH(1, { validateSymQConfigShape(&bad, 10); });
    TEST_ASSERT_TRUE(true); /* reached ⇒ ok-case did not exit */
}

/* Group-quant PR4 (Task 1, spec D6): asymQConfig_t goes always-array
 * (scales[numGroups] + CODE-domain uint16 zeroPoints[numGroups]) with the
 * same {1,0}-per-tensor / {>1,>0}-grouped shape grammar as symQConfig_t.
 * ASYM's qBits ceiling drops 30 -> 16 (uint16 zp domain, D6). */

void testInitAsymQConfigProducesPerTensorSentinelWithZeroZp(void) {
    /* Mutation guard: hardcoding a different shape pair, or seeding
     * zeroPoints[0] != 0 / scales[0] != 1, fails this. */
    asymQConfig_t qc;
    initAsymQConfig(6, SR_HALF_AWAY, &qc);

    size_t numGroups = qc.numGroups;
    size_t groupSize = qc.groupSize;
    uint8_t qBits = qc.qBits;
    roundingMode_t roundingMode = qc.roundingMode;
    float scale0 = qc.scales[0];
    uint16_t zp0 = qc.zeroPoints[0];

    freeReservedMemory(qc.zeroPoints);
    freeReservedMemory(qc.scales);

    TEST_ASSERT_EQUAL_size_t(1, numGroups);
    TEST_ASSERT_EQUAL_size_t(0, groupSize);
    TEST_ASSERT_EQUAL_UINT8(6, qBits);
    TEST_ASSERT_EQUAL_INT(SR_HALF_AWAY, roundingMode);
    TEST_ASSERT_EQUAL_FLOAT(1.f, scale0);
    TEST_ASSERT_EQUAL_UINT16(0, zp0);
}

void testInitAsymQConfigAllocatesIndependentArrays(void) {
    /* Both owned blocks (scales AND zeroPoints) must be per-config heap
     * arrays -- no shared/static storage, and the two blocks are distinct
     * allocations (SYM ownership pattern, one block per array). */
    asymQConfig_t a;
    asymQConfig_t b;
    initAsymQConfig(4, HALF_AWAY, &a);
    initAsymQConfig(4, HALF_AWAY, &b);

    float *aScales = a.scales;
    float *bScales = b.scales;
    uint16_t *aZps = a.zeroPoints;
    uint16_t *bZps = b.zeroPoints;

    freeReservedMemory(a.zeroPoints);
    freeReservedMemory(a.scales);
    freeReservedMemory(b.zeroPoints);
    freeReservedMemory(b.scales);

    TEST_ASSERT_NOT_NULL(aScales);
    TEST_ASSERT_NOT_NULL(aZps);
    TEST_ASSERT_NOT_EQUAL(aScales, bScales);
    TEST_ASSERT_NOT_EQUAL(aZps, bZps);
    TEST_ASSERT_NOT_EQUAL((void *)aScales, (void *)aZps);
}

void testInitAsymQConfigGroupedAllocatesPerGroupArrays(void) {
    asymQConfig_t qc;
    initAsymQConfigGrouped(4, SR_HALF_AWAY, 3, 5, &qc);
    size_t numGroups = qc.numGroups;
    size_t groupSize = qc.groupSize;
    float s0 = qc.scales[0], s2 = qc.scales[2];
    uint16_t z0 = qc.zeroPoints[0], z2 = qc.zeroPoints[2];
    qc.scales[2] = 7.f; /* writable per-group slots */
    qc.zeroPoints[2] = 9;
    float s2w = qc.scales[2];
    uint16_t z2w = qc.zeroPoints[2];
    freeReservedMemory(qc.zeroPoints);
    freeReservedMemory(qc.scales);
    TEST_ASSERT_EQUAL_size_t(3, numGroups);
    TEST_ASSERT_EQUAL_size_t(5, groupSize);
    TEST_ASSERT_EQUAL_FLOAT(1.f, s0);
    TEST_ASSERT_EQUAL_FLOAT(1.f, s2);
    TEST_ASSERT_EQUAL_UINT16(0, z0);
    TEST_ASSERT_EQUAL_UINT16(0, z2);
    TEST_ASSERT_EQUAL_FLOAT(7.f, s2w);
    TEST_ASSERT_EQUAL_UINT16(9, z2w);
}

void testInitAsymQConfigGroupedRejectsSentinelViolations(void) {
    asymQConfig_t qc;
    /* numGroups>1 with groupSize==0 violates the invariant */
    ASSERT_EXITS_WITH(1, { initAsymQConfigGrouped(4, HALF_AWAY, 2, 0, &qc); });
    /* numGroups==1 with groupSize!=0 is the non-canonical {1,N} form */
    ASSERT_EXITS_WITH(1, { initAsymQConfigGrouped(4, HALF_AWAY, 1, 8, &qc); });
    ASSERT_EXITS_WITH(1, { initAsymQConfigGrouped(4, HALF_AWAY, 0, 0, &qc); });
}

void testInitAsymQConfigGroupedRejectsQBitsOutside1To16(void) {
    /* D6: the code-domain zeroPoint is uint16, so qBits > 16 has no zp
     * representation (supersedes the old [1, 30] #246 ceiling); 0 would
     * underflow the sub-byte packer, as before. */
    asymQConfig_t qc;
    ASSERT_EXITS_WITH(1, { initAsymQConfigGrouped(17, HALF_AWAY, 1, 0, &qc); });
    ASSERT_EXITS_WITH(1, { initAsymQConfigGrouped(0, HALF_AWAY, 1, 0, &qc); });
}

void testValidateAsymQConfigShapeDivisibilityAndQBits(void) {
    float s[2] = {1.f, 1.f};
    uint16_t z[2] = {0, 0};
    asymQConfig_t ok = {.scales = s,
                        .zeroPoints = z,
                        .numGroups = 2,
                        .groupSize = 5,
                        .roundingMode = HALF_AWAY,
                        .qBits = 4};
    validateAsymQConfigShape(&ok, 10); /* must NOT exit */
    asymQConfig_t bad = {.scales = s,
                         .zeroPoints = z,
                         .numGroups = 2,
                         .groupSize = 4,
                         .roundingMode = HALF_AWAY,
                         .qBits = 4};
    ASSERT_EXITS_WITH(1, { validateAsymQConfigShape(&bad, 10); });
    /* {1,N} is not a per-tensor spelling (same grammar as SYM) */
    asymQConfig_t oneN = {.scales = s,
                          .zeroPoints = z,
                          .numGroups = 1,
                          .groupSize = 10,
                          .roundingMode = HALF_AWAY,
                          .qBits = 4};
    ASSERT_EXITS_WITH(1, { validateAsymQConfigShape(&oneN, 10); });
    /* the attach-time validator re-checks the D6 width ceiling for
     * field-assigned configs that never went through the init funnel */
    asymQConfig_t wide = {.scales = s,
                          .zeroPoints = z,
                          .numGroups = 2,
                          .groupSize = 5,
                          .roundingMode = HALF_AWAY,
                          .qBits = 17};
    ASSERT_EXITS_WITH(1, { validateAsymQConfigShape(&wide, 10); });
    TEST_ASSERT_TRUE(true); /* reached ⇒ ok-case did not exit */
}

/* BFP epic PR1 (docs/superpowers/specs/2026-07-29-block-floating-point-design.md,
 * Task 1; the group shape itself mirrors the group-quant design's
 * docs/superpowers/specs/2026-07-28-group-quantization-design.md): bfpQConfig_t
 * mirrors symQConfig_t's always-array group shape
 * exactly ({1,0} per-tensor sentinel or {>1,>0} grouped), swapping the SYM
 * per-group float scale for a per-group biased exponent byte -- the group
 * grid itself (numGroups/groupSize/validate) is identical machinery. */

void testInitBfpQConfigPerTensorZeroState(void) {
    bfpQConfig_t qc;
    initBfpQConfig(8, 8, HALF_AWAY, &qc);
    TEST_ASSERT_EQUAL_size_t(1, qc.numGroups);
    TEST_ASSERT_EQUAL_size_t(0, qc.groupSize);
    TEST_ASSERT_EQUAL_UINT8(8, qc.mantissaBits);
    TEST_ASSERT_EQUAL_UINT8(8, qc.exponentBits);
    TEST_ASSERT_EQUAL_INT32(127, bfpExponentBias(&qc));
    TEST_ASSERT_EQUAL_UINT8(127, qc.exponents[0]); /* zero-state = bias, scale 1.0 */
    TEST_ASSERT_EQUAL_FLOAT(1.0f, bfpGroupScale(&qc, 0));
    freeReservedMemory(qc.exponents);
}

void testInitBfpQConfigGroupedAllocatesPerGroup(void) {
    bfpQConfig_t qc;
    initBfpQConfigGrouped(4, 5, SR_HALF_AWAY, 3, 8, &qc);
    TEST_ASSERT_EQUAL_size_t(3, qc.numGroups);
    TEST_ASSERT_EQUAL_size_t(8, qc.groupSize);
    TEST_ASSERT_EQUAL_INT32(15, bfpExponentBias(&qc)); /* 2^(5-1)-1 */
    for (size_t g = 0; g < 3; g++) {
        TEST_ASSERT_EQUAL_UINT8(15, qc.exponents[g]);
    }
    freeReservedMemory(qc.exponents);
}

void testInitBfpQConfigGroupedRejectsInvalidShape(void) {
    bfpQConfig_t qc;
    ASSERT_EXITS_WITH_FAILURE(initBfpQConfigGrouped(8, 8, HALF_AWAY, 1, 8, &qc)); /* {1,N>0} */
    ASSERT_EXITS_WITH_FAILURE(initBfpQConfigGrouped(8, 8, HALF_AWAY, 0, 0, &qc)); /* {0,*} */
    ASSERT_EXITS_WITH_FAILURE(initBfpQConfigGrouped(8, 8, HALF_AWAY, 4, 0, &qc)); /* {>1,0} */
}

void testInitBfpQConfigRejectsWidthCaps(void) {
    bfpQConfig_t qc;
    ASSERT_EXITS_WITH_FAILURE(initBfpQConfig(1, 8, HALF_AWAY, &qc));  /* mantissa < 2 */
    ASSERT_EXITS_WITH_FAILURE(initBfpQConfig(17, 8, HALF_AWAY, &qc)); /* mantissa > 16 */
    ASSERT_EXITS_WITH_FAILURE(initBfpQConfig(8, 1, HALF_AWAY, &qc));  /* exponent < 2 */
    ASSERT_EXITS_WITH_FAILURE(initBfpQConfig(8, 9, HALF_AWAY, &qc));  /* exponent > 8 */
}

void testValidateBfpQConfigShapeEnforcesElementIdentity(void) {
    bfpQConfig_t qc;
    initBfpQConfigGrouped(8, 8, HALF_AWAY, 3, 8, &qc);
    validateBfpQConfigShape(&qc, 24); /* 3*8 == 24: passes */
    ASSERT_EXITS_WITH_FAILURE(validateBfpQConfigShape(&qc, 23));
    freeReservedMemory(qc.exponents);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testInitSymQConfigProducesPerTensorSentinel);
    RUN_TEST(testInitSymQConfigAllocatesIndependentScalesArrays);
    RUN_TEST(testInitSymQConfigGroupedAllocatesPerGroupScales);
    RUN_TEST(testInitSymQConfigGroupedRejectsSentinelViolations);
    RUN_TEST(testValidateSymQConfigShapeDivisibility);
    RUN_TEST(testInitAsymQConfigProducesPerTensorSentinelWithZeroZp);
    RUN_TEST(testInitAsymQConfigAllocatesIndependentArrays);
    RUN_TEST(testInitAsymQConfigGroupedAllocatesPerGroupArrays);
    RUN_TEST(testInitAsymQConfigGroupedRejectsSentinelViolations);
    RUN_TEST(testInitAsymQConfigGroupedRejectsQBitsOutside1To16);
    RUN_TEST(testValidateAsymQConfigShapeDivisibilityAndQBits);
    RUN_TEST(testInitBfpQConfigPerTensorZeroState);
    RUN_TEST(testInitBfpQConfigGroupedAllocatesPerGroup);
    RUN_TEST(testInitBfpQConfigGroupedRejectsInvalidShape);
    RUN_TEST(testInitBfpQConfigRejectsWidthCaps);
    RUN_TEST(testValidateBfpQConfigShapeEnforcesElementIdentity);
    return UNITY_END();
}
