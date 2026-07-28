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

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testInitSymQConfigProducesPerTensorSentinel);
    RUN_TEST(testInitSymQConfigAllocatesIndependentScalesArrays);
    RUN_TEST(testInitSymQConfigGroupedAllocatesPerGroupScales);
    RUN_TEST(testInitSymQConfigGroupedRejectsSentinelViolations);
    RUN_TEST(testValidateSymQConfigShapeDivisibility);
    return UNITY_END();
}
