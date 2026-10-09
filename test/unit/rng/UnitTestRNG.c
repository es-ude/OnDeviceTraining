#include <string.h>

#include "RNG.h"
#include "unity.h"

void setUp(void) {}

void tearDown(void) {}

void testRngNextFloatInRange(void) {
    rngSetSeed(42);
    for (size_t i = 0; i < 10000; i++) {
        float val = rngNextFloat();
        TEST_ASSERT_TRUE(val >= 0.0f);
        TEST_ASSERT_TRUE(val < 1.0f);
    }
}

void testRngNextFloatDistribution(void) {
    rngSetSeed(42);
    const size_t n = 10000;
    const size_t buckets = 10;
    size_t counts[10] = {0};

    for (size_t i = 0; i < n; i++) {
        float val = rngNextFloat();
        size_t bucket = (size_t)(val * buckets);
        if (bucket >= buckets) {
            bucket = buckets - 1;
        }
        counts[bucket]++;
    }

    // Each bucket should have ~1000 samples; allow 200 deviation
    for (size_t b = 0; b < buckets; b++) {
        TEST_ASSERT_INT_WITHIN(200, 1000, (int)counts[b]);
    }
}

void testRngNextFloatReproducible(void) {
    float first[100];
    float second[100];

    rngSetSeed(123);
    for (size_t i = 0; i < 100; i++) {
        first[i] = rngNextFloat();
    }

    rngSetSeed(123);
    for (size_t i = 0; i < 100; i++) {
        second[i] = rngNextFloat();
    }

    for (size_t i = 0; i < 100; i++) {
        TEST_ASSERT_EQUAL_FLOAT(first[i], second[i]);
    }
}

void testRngShuffleUsesGlobalState(void) {
    size_t indices1[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    size_t indices2[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

    rngSetSeed(42);
    rngShuffleIndices(indices1, 10);

    rngSetSeed(42);
    rngShuffleIndices(indices2, 10);

    for (size_t i = 0; i < 10; i++) {
        TEST_ASSERT_EQUAL_UINT(indices1[i], indices2[i]);
    }
}

void testRngSetSeedZeroNotAliasedToOne(void) {
    rngSetSeed(0);
    float a0 = rngNextFloat();

    rngSetSeed(1);
    float b0 = rngNextFloat();

    TEST_ASSERT_TRUE(a0 != b0);

    rngSetSeed(0);
    float a1 = rngNextFloat();
    TEST_ASSERT_EQUAL_FLOAT(a0, a1);
}

void testRngNextFloatCtxReproducible(void) {
    rng32_t a = {.state = 123};
    rng32_t b = {.state = 123};
    for (size_t i = 0; i < 100; i++) {
        float val_a = rngNextFloatCtx(&a);
        float val_b = rngNextFloatCtx(&b);
        TEST_ASSERT_EQUAL_FLOAT(val_a, val_b);
    }
}

void testRngNextFloatCtxMatchesGlobalStream(void) {
    /* The global rngNextFloat() must be a bit-identical delegate of the ctx
     * variant: same seed -> same sequence. Seed 123 maps to state 123. */
    rng32_t ctx = {.state = 123};
    rngSetSeed(123);
    for (size_t i = 0; i < 100; i++) {
        float global_val = rngNextFloat();
        float ctx_val = rngNextFloatCtx(&ctx);
        TEST_ASSERT_EQUAL_FLOAT(global_val, ctx_val);
    }
}

void testRngNextFloatCtxDoesNotTouchGlobalStream(void) {
    rngSetSeed(77);
    uint32_t before = rngGetSeed();
    rng32_t ctx = {.state = 999};
    for (size_t i = 0; i < 50; i++) {
        (void)rngNextFloatCtx(&ctx);
    }
    TEST_ASSERT_EQUAL_UINT32(before, rngGetSeed());
}

/* ---- keyed core: rngFold ---- */

static void assertKeyEquals(uint32_t w0, uint32_t w1, rngKey_t key) {
    TEST_ASSERT_EQUAL_HEX32(w0, key.w[0]);
    TEST_ASSERT_EQUAL_HEX32(w1, key.w[1]);
}

static int keysEqual(rngKey_t a, rngKey_t b) {
    return a.w[0] == b.w[0] && a.w[1] == b.w[1];
}

/* Threefry-2x32 known answers at 20 rounds, read as
 * rngFold({KEY0, KEY1}, CTR0, CTR1) == {OUT0, OUT1}. Source: tests/kat_vectors of
 * https://github.com/DEShawResearch/random123 at commit 9545ff64
 * (Random123, Copyright 2010-2011 D. E. Shaw Research, BSD-3-Clause). */
void testRngFoldMatchesThreefryKnownAnswers(void) {
    assertKeyEquals(0x6b200159u, 0x99ba4efeu,
                    rngFold((rngKey_t){{0x00000000u, 0x00000000u}}, 0x00000000u, 0x00000000u));
    assertKeyEquals(0x1cb996fcu, 0xbb002be7u,
                    rngFold((rngKey_t){{0xffffffffu, 0xffffffffu}}, 0xffffffffu, 0xffffffffu));
    /* The only row whose words all differ: it fixes which input word is the
     * tag, which the index, and which key word comes first. */
    assertKeyEquals(0xc4923a9cu, 0x483df7a0u,
                    rngFold((rngKey_t){{0x13198a2eu, 0x03707344u}}, 0x243f6a88u, 0x85a308d3u));
}

void testRngFoldIsNotSymmetricInTagAndIndex(void) {
    const rngKey_t parent = {{0x13198a2eu, 0x03707344u}};
    TEST_ASSERT_FALSE(keysEqual(rngFold(parent, 7u, 42u), rngFold(parent, 42u, 7u)));
}

/* Siblings that differ in the tag only or in the index only are different
 * nodes: 8 tags x 8 indices give 64 pairwise distinct children. */
void testRngFoldDistinctEdgesGiveDistinctKeys(void) {
    const rngKey_t parent = {{0x13198a2eu, 0x03707344u}};
    rngKey_t child[64];
    for (uint32_t tag = 1u; tag <= 8u; ++tag) {
        for (uint32_t index = 0u; index < 8u; ++index) {
            child[(tag - 1u) * 8u + index] = rngFold(parent, tag, index);
        }
    }
    for (size_t a = 0; a < 64; ++a) {
        for (size_t b = a + 1; b < 64; ++b) {
            TEST_ASSERT_FALSE(keysEqual(child[a], child[b]));
        }
    }
}

/* A two-edge path is the nested fold; the values come from an independent
 * implementation of the published algorithm (the Python mirror repeats them). */
void testRngFoldTwoEdgePathMatchesReference(void) {
    const rngKey_t first = rngFold((rngKey_t){{0x13198a2eu, 0x03707344u}}, 7u, 0u);
    assertKeyEquals(0x627945fdu, 0xa1f54519u, first);
    assertKeyEquals(0x2b6fc7a8u, 0x5d69d71du, rngFold(first, 3u, 5u));
}

/* A key depends on its path only, never on which keys were derived before
 * it: three two-edge paths, derived path by path and interleaved. */
void testRngFoldKeysDoNotDependOnDerivationOrder(void) {
    const rngKey_t root = {{0x13198a2eu, 0x03707344u}};
    const rngKey_t a1 = rngFold(rngFold(root, 7u, 1u), 9u, 2u);
    const rngKey_t b1 = rngFold(rngFold(root, 7u, 1u), 9u, 3u);
    const rngKey_t c1 = rngFold(rngFold(root, 8u, 0u), 5u, 5u);
    const rngKey_t mid7 = rngFold(root, 7u, 1u);
    const rngKey_t mid8 = rngFold(root, 8u, 0u);
    const rngKey_t c2 = rngFold(mid8, 5u, 5u);
    const rngKey_t b2 = rngFold(mid7, 9u, 3u);
    const rngKey_t a2 = rngFold(mid7, 9u, 2u);
    TEST_ASSERT_TRUE(keysEqual(a1, a2));
    TEST_ASSERT_TRUE(keysEqual(b1, b2));
    TEST_ASSERT_TRUE(keysEqual(c1, c2));
}

/* The keyed core keeps no module state: deriving keys never moves the global
 * stream that the not yet migrated consumers draw from. */
void testRngFoldDoesNotMoveTheGlobalStream(void) {
    rngSetSeed(77);
    const uint32_t before = rngGetSeed();
    rngKey_t key = {{1u, 2u}};
    for (uint32_t i = 0; i < 50u; ++i) {
        key = rngFold(rngFold(key, 7u, i), 3u, i);
    }
    TEST_ASSERT_EQUAL_UINT32(before, rngGetSeed());
}

/* ---- keyed core: rngKeyFromSeed ---- */

/* Seed 0 is the all-zero counter under the all-zero key, Random123's first
 * 20-round row (credited above); 42 and 2^32 (low word 0) come from the
 * reference implementation. */
void testRngKeyFromSeedMatchesReference(void) {
    assertKeyEquals(0x6b200159u, 0x99ba4efeu, rngKeyFromSeed(0u));
    assertKeyEquals(0xa608025au, 0x0ef6f004u, rngKeyFromSeed(42u));
    assertKeyEquals(0x375f238fu, 0xcddb151du, rngKeyFromSeed(UINT64_C(1) << 32));
}

/* The seed's low word is the tag position of the counter, its high word the
 * index position; distinct words make a swap visible. */
void testRngKeyFromSeedIsTheFoldOfItsSeedWords(void) {
    const rngKey_t zero = {{0u, 0u}};
    TEST_ASSERT_TRUE(keysEqual(rngFold(zero, 0x89abcdefu, 0x01234567u),
                               rngKeyFromSeed(UINT64_C(0x0123456789abcdef))));
    TEST_ASSERT_TRUE(
        keysEqual(rngFold(zero, 0xffffffffu, 0xffffffffu), rngKeyFromSeed(UINT64_MAX)));
}

/* Unlike rngSetSeed, seed 0 is not an alias of UINT32_MAX. */
void testRngKeyFromSeedTreatsZeroAsAnOrdinarySeed(void) {
    TEST_ASSERT_FALSE(keysEqual(rngKeyFromSeed(0u), rngKeyFromSeed(0xFFFFFFFFu)));
    TEST_ASSERT_FALSE(keysEqual(rngKeyFromSeed(0u), rngKeyFromSeed(1u)));
}

void testRngKeyFromSeedDoesNotMoveTheGlobalStream(void) {
    rngSetSeed(77);
    const uint32_t before = rngGetSeed();
    for (uint64_t seed = 0; seed < 50u; ++seed) {
        (void)rngKeyFromSeed(seed);
    }
    TEST_ASSERT_EQUAL_UINT32(before, rngGetSeed());
}

/* ---- keyed core: rngOpen and rngNextU32 (backend v1, keyed xorshift32) ---- */

/* xorshift32 never leaves state 0, so the reduction is total and non-zero:
 * w[0], else w[1], else 1. */
void testRngOpenReducesTheKeyToANonZeroState(void) {
    const rngStream_t full = rngOpen((rngKey_t){{0x929c8a01u, 0xee4a7510u}});
    const rngStream_t lowWordZero = rngOpen((rngKey_t){{0u, 0xee4a7510u}});
    const rngStream_t bothWordsZero = rngOpen((rngKey_t){{0u, 0u}});
    TEST_ASSERT_EQUAL_HEX32(0x929c8a01u, full.state);
    TEST_ASSERT_EQUAL_HEX32(0xee4a7510u, lowWordZero.state);
    TEST_ASSERT_EQUAL_HEX32(1u, bothWordsZero.state);
}

/* Backend v1 is today's generator on a keyed state: from state 1 the first
 * draws are the known xorshift32 vectors, and from any state s each draw is
 * the state that the global step leaves in an rng32_t holding s. */
void testRngNextU32ReproducesTheGlobalGenerator(void) {
    rngStream_t one = rngOpen((rngKey_t){{1u, 0xdeadbeefu}});
    TEST_ASSERT_EQUAL_UINT32(270369u, rngNextU32(&one));
    TEST_ASSERT_EQUAL_UINT32(67634689u, rngNextU32(&one));
    TEST_ASSERT_EQUAL_UINT32(2647435461u, rngNextU32(&one));
    TEST_ASSERT_EQUAL_UINT32(307599695u, rngNextU32(&one));
    const uint32_t states[4] = {1u, 42u, 0x9e3779b9u, 0xffffffffu};
    for (size_t k = 0; k < 4; ++k) {
        rng32_t ctx = {.state = states[k]};
        rngStream_t stream = rngOpen((rngKey_t){{states[k], 0u}});
        for (size_t i = 0; i < 100; ++i) {
            (void)rngNextFloatCtx(&ctx);
            TEST_ASSERT_EQUAL_HEX32(ctx.state, rngNextU32(&stream));
        }
    }
}

/* A derived key's stream end to end; the Python mirror repeats these values. */
void testRngNextU32MatchesReferenceOnADerivedKey(void) {
    const rngKey_t key = rngFold(rngFold(rngKeyFromSeed(42u), 7u, 0u), 3u, 5u);
    assertKeyEquals(0x929c8a01u, 0xee4a7510u, key);
    rngStream_t stream = rngOpen(key);
    TEST_ASSERT_EQUAL_HEX32(0x7849d60fu, rngNextU32(&stream));
    TEST_ASSERT_EQUAL_HEX32(0x138afe2bu, rngNextU32(&stream));
    TEST_ASSERT_EQUAL_HEX32(0xc5b8b98cu, rngNextU32(&stream));
    TEST_ASSERT_EQUAL_HEX32(0x83a349c8u, rngNextU32(&stream));
}

void testRngOpenedZeroKeysNeverStickAtZero(void) {
    const rngKey_t keys[2] = {{{0u, 0xee4a7510u}}, {{0u, 0u}}};
    for (size_t k = 0; k < 2; ++k) {
        rngStream_t stream = rngOpen(keys[k]);
        for (size_t i = 0; i < 100; ++i) {
            TEST_ASSERT_NOT_EQUAL_UINT32(0u, rngNextU32(&stream));
        }
    }
}

/* Drawing from stream A between the draws of stream B leaves B's values
 * unchanged: a stream's state lives in its caller's struct only. */
void testRngStreamsDrawIndependently(void) {
    const rngKey_t keyA = rngFold(rngKeyFromSeed(1u), 7u, 0u);
    const rngKey_t keyB = rngFold(rngKeyFromSeed(1u), 7u, 1u);
    uint32_t alone[50];
    rngStream_t b = rngOpen(keyB);
    for (size_t i = 0; i < 50; ++i) {
        alone[i] = rngNextU32(&b);
    }
    rngStream_t a = rngOpen(keyA);
    b = rngOpen(keyB);
    for (size_t i = 0; i < 50; ++i) {
        (void)rngNextU32(&a);
        (void)rngNextU32(&a);
        TEST_ASSERT_EQUAL_HEX32(alone[i], rngNextU32(&b));
    }
}

/* Reopening a key replays its stream and a copied stream continues
 * identically: a recompute reopens the key of the first computation. */
void testRngReopenedStreamReplaysItsValues(void) {
    const rngKey_t key = rngFold(rngKeyFromSeed(3u), 7u, 2u);
    uint32_t drawn[20];
    rngStream_t first = rngOpen(key);
    for (size_t i = 0; i < 20; ++i) {
        drawn[i] = rngNextU32(&first);
    }
    rngStream_t again = rngOpen(key);
    for (size_t i = 0; i < 10; ++i) {
        TEST_ASSERT_EQUAL_HEX32(drawn[i], rngNextU32(&again));
    }
    rngStream_t copy = again;
    for (size_t i = 10; i < 20; ++i) {
        TEST_ASSERT_EQUAL_HEX32(drawn[i], rngNextU32(&copy));
        TEST_ASSERT_EQUAL_HEX32(drawn[i], rngNextU32(&again));
    }
}

void testRngKeyedStreamsDoNotMoveTheGlobalStream(void) {
    rngSetSeed(77);
    const uint32_t before = rngGetSeed();
    rngStream_t stream = rngOpen(rngKeyFromSeed(5u));
    for (size_t i = 0; i < 50; ++i) {
        (void)rngNextU32(&stream);
    }
    TEST_ASSERT_EQUAL_UINT32(before, rngGetSeed());
}

/* ---- keyed core: rngNextUnitFloat ---- */

static uint32_t floatBits(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof bits);
    return bits;
}

/* Today's mapping bit for bit, and in [0, 1). Unity's float compare has a
 * tolerance, so the identity compares bit patterns. */
void testRngNextUnitFloatIsBitIdenticalToTheGlobalMapping(void) {
    const uint32_t states[3] = {1u, 0x929c8a01u, 0xffffffffu};
    for (size_t k = 0; k < 3; ++k) {
        rng32_t ctx = {.state = states[k]};
        rngStream_t stream = rngOpen((rngKey_t){{states[k], 0u}});
        for (size_t i = 0; i < 1000; ++i) {
            const float expected = rngNextFloatCtx(&ctx);
            const float actual = rngNextUnitFloat(&stream);
            TEST_ASSERT_EQUAL_HEX32(floatBits(expected), floatBits(actual));
            TEST_ASSERT_TRUE(actual >= 0.0f && actual < 1.0f);
        }
    }
}

/* The derived key of the U32 reference test; the Python mirror repeats it. */
void testRngNextUnitFloatMatchesReferenceOnADerivedKey(void) {
    rngStream_t stream = rngOpen((rngKey_t){{0x929c8a01u, 0xee4a7510u}});
    TEST_ASSERT_EQUAL_HEX32(0x3ef093acu, floatBits(rngNextUnitFloat(&stream)));
    TEST_ASSERT_EQUAL_HEX32(0x3d9c57f0u, floatBits(rngNextUnitFloat(&stream)));
    TEST_ASSERT_EQUAL_HEX32(0x3f45b8b9u, floatBits(rngNextUnitFloat(&stream)));
    TEST_ASSERT_EQUAL_HEX32(0x3f03a349u, floatBits(rngNextUnitFloat(&stream)));
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testRngNextFloatInRange);
    RUN_TEST(testRngNextFloatDistribution);
    RUN_TEST(testRngNextFloatReproducible);
    RUN_TEST(testRngShuffleUsesGlobalState);
    RUN_TEST(testRngSetSeedZeroNotAliasedToOne);
    RUN_TEST(testRngNextFloatCtxReproducible);
    RUN_TEST(testRngNextFloatCtxMatchesGlobalStream);
    RUN_TEST(testRngNextFloatCtxDoesNotTouchGlobalStream);
    RUN_TEST(testRngFoldMatchesThreefryKnownAnswers);
    RUN_TEST(testRngFoldIsNotSymmetricInTagAndIndex);
    RUN_TEST(testRngFoldDistinctEdgesGiveDistinctKeys);
    RUN_TEST(testRngFoldTwoEdgePathMatchesReference);
    RUN_TEST(testRngFoldKeysDoNotDependOnDerivationOrder);
    RUN_TEST(testRngFoldDoesNotMoveTheGlobalStream);
    RUN_TEST(testRngKeyFromSeedMatchesReference);
    RUN_TEST(testRngKeyFromSeedIsTheFoldOfItsSeedWords);
    RUN_TEST(testRngKeyFromSeedTreatsZeroAsAnOrdinarySeed);
    RUN_TEST(testRngKeyFromSeedDoesNotMoveTheGlobalStream);
    RUN_TEST(testRngOpenReducesTheKeyToANonZeroState);
    RUN_TEST(testRngNextU32ReproducesTheGlobalGenerator);
    RUN_TEST(testRngNextU32MatchesReferenceOnADerivedKey);
    RUN_TEST(testRngOpenedZeroKeysNeverStickAtZero);
    RUN_TEST(testRngStreamsDrawIndependently);
    RUN_TEST(testRngReopenedStreamReplaysItsValues);
    RUN_TEST(testRngKeyedStreamsDoNotMoveTheGlobalStream);
    RUN_TEST(testRngNextUnitFloatIsBitIdenticalToTheGlobalMapping);
    RUN_TEST(testRngNextUnitFloatMatchesReferenceOnADerivedKey);
    return UNITY_END();
}
