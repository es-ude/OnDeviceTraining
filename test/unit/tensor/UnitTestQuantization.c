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

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testInitSymQConfigProducesPerTensorSentinel);
    RUN_TEST(testInitSymQConfigAllocatesIndependentScalesArrays);
    return UNITY_END();
}
