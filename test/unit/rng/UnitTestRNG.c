#include <string.h>

#include "DeathTest.h"
#include "RNG.h"
#include "unity.h"

void setUp(void) {}

void tearDown(void) {}

/* ---- test-local reference, written from the definitions, not from RNG.c ---- */

/* One xorshift32 step (shifts 13, 17, 5). */
static uint32_t referenceXorshift32(uint32_t x) {
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return x;
}

/* Fisher-Yates from the identity order, from state `state` (n >= 1): from the
 * last slot down, slot i swaps with the next draw below UINT32_MAX -
 * UINT32_MAX % (i + 1), taken modulo i + 1; a draw at or above it is redrawn. */
static void referenceShuffle(size_t *order, size_t n, uint32_t state) {
    for (size_t i = 0; i < n; ++i) {
        order[i] = i;
    }
    for (size_t i = n - 1; i > 0; --i) {
        const uint32_t bound = (uint32_t)(i + 1);
        do {
            state = referenceXorshift32(state);
        } while (state >= UINT32_MAX - UINT32_MAX % bound);
        const size_t j = state % bound;
        const size_t tmp = order[i];
        order[i] = order[j];
        order[j] = tmp;
    }
}

/* The same loop and swap with rngNextBounded on the stream of `key`. */
static void keyedShuffle(size_t *order, size_t n, rngKey_t key) {
    rngStream_t stream = rngOpen(key);
    for (size_t i = 0; i < n; ++i) {
        order[i] = i;
    }
    for (size_t i = n - 1; i > 0; --i) {
        const size_t j = rngNextBounded(&stream, i + 1);
        const size_t tmp = order[i];
        order[i] = order[j];
        order[j] = tmp;
    }
}

void testRngNextUnitFloatInRange(void) {
    /* Raw key {S, 0}: same draws as the former global stream seeded with S != 0. */
    rngStream_t stream = rngOpen((rngKey_t){{42u, 0u}});
    for (size_t i = 0; i < 10000; i++) {
        float val = rngNextUnitFloat(&stream);
        TEST_ASSERT_TRUE(val >= 0.0f);
        TEST_ASSERT_TRUE(val < 1.0f);
    }
}

void testRngNextUnitFloatDistribution(void) {
    /* Raw key {S, 0}: same draws as the former global stream seeded with S != 0. */
    rngStream_t stream = rngOpen((rngKey_t){{42u, 0u}});
    const size_t n = 10000;
    const size_t buckets = 10;
    size_t counts[10] = {0};

    for (size_t i = 0; i < n; i++) {
        float val = rngNextUnitFloat(&stream);
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

/* Seed 0 is an ordinary seed: its key is neither that of 0xFFFFFFFF nor that
 * of 1. */
void testRngKeyFromSeedTreatsZeroAsAnOrdinarySeed(void) {
    TEST_ASSERT_FALSE(keysEqual(rngKeyFromSeed(0u), rngKeyFromSeed(0xFFFFFFFFu)));
    TEST_ASSERT_FALSE(keysEqual(rngKeyFromSeed(0u), rngKeyFromSeed(1u)));
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

/* Backend v1 is xorshift32 on a keyed state: from state 1 the first draws are
 * the known xorshift32 vectors, and from any state each draw is one step of
 * the reference recurrence, the draw being the new state. */
void testRngNextU32IsTheXorshift32Recurrence(void) {
    rngStream_t one = rngOpen((rngKey_t){{1u, 0xdeadbeefu}});
    TEST_ASSERT_EQUAL_UINT32(270369u, rngNextU32(&one));
    TEST_ASSERT_EQUAL_UINT32(67634689u, rngNextU32(&one));
    TEST_ASSERT_EQUAL_UINT32(2647435461u, rngNextU32(&one));
    TEST_ASSERT_EQUAL_UINT32(307599695u, rngNextU32(&one));
    const uint32_t states[4] = {1u, 42u, 0x9e3779b9u, 0xffffffffu};
    for (size_t k = 0; k < 4; ++k) {
        uint32_t expected = states[k];
        rngStream_t stream = rngOpen((rngKey_t){{states[k], 0u}});
        for (size_t i = 0; i < 100; ++i) {
            expected = referenceXorshift32(expected);
            TEST_ASSERT_EQUAL_HEX32(expected, rngNextU32(&stream));
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

/* ---- keyed core: rngNextUnitFloat ---- */

static uint32_t floatBits(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof bits);
    return bits;
}

/* The unit float is the draw's top 24 bits times 2^-24, which a float holds
 * exactly, so it lies in [0, 1). Unity's float compare has a tolerance, so the
 * check compares bit patterns. */
void testRngNextUnitFloatIsTheTop24BitsOfTheDraw(void) {
    const uint32_t states[3] = {1u, 0x929c8a01u, 0xffffffffu};
    for (size_t k = 0; k < 3; ++k) {
        uint32_t draw = states[k];
        rngStream_t stream = rngOpen((rngKey_t){{states[k], 0u}});
        for (size_t i = 0; i < 1000; ++i) {
            draw = referenceXorshift32(draw);
            const float expected = (float)(draw >> 8) * 0x1p-24f;
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

/* ---- keyed core: rngNextBounded ---- */

/* Fisher-Yates with the keyed bounded draw on the stream of {seed, 0} gives
 * the reference permutation of state seed (seed != 0, n <= 1000). */
static void assertShuffleMatchesTheReference(size_t n, uint32_t seed) {
    size_t expected[1000];
    size_t actual[1000];
    referenceShuffle(expected, n, seed);
    keyedShuffle(actual, n, (rngKey_t){{seed, 0u}});
    for (size_t i = 0; i < n; ++i) {
        TEST_ASSERT_EQUAL_size_t(expected[i], actual[i]);
    }
}

/* Recorded values, from the repository root:
 *   uv run python -c "from examples._shared.keyed_rng import (keyed_shuffle as ks,
 *   open_stream as o, next_u32 as u, next_bounded as nb); s = o((0xFC7D0B53, 0)); print(
 *   ks(10, (1, 0)), hex(u(s)[1]), nb(s, 1000)[1])"
 * prints [3, 1, 0, 2, 6, 7, 8, 5, 4, 9] 0xfffffed8 218. 0xfc7d0b53's first draw
 * is exactly the rejection limit for bound 1000, so the limit itself must be
 * rejected and the second draw gives 218. */
void testRngNextBoundedShufflesLikeTheReference(void) {
    const size_t recorded[10] = {3, 1, 0, 2, 6, 7, 8, 5, 4, 9};
    size_t order[10];
    keyedShuffle(order, 10, (rngKey_t){{1u, 0u}});
    for (size_t i = 0; i < 10; ++i) {
        TEST_ASSERT_EQUAL_size_t(recorded[i], order[i]);
    }
    TEST_ASSERT_EQUAL_HEX32(UINT32_MAX - UINT32_MAX % 1000u, referenceXorshift32(0xfc7d0b53u));
    rngStream_t atTheLimit = rngOpen((rngKey_t){{0xfc7d0b53u, 0u}});
    TEST_ASSERT_EQUAL_size_t(218, rngNextBounded(&atTheLimit, 1000));
    assertShuffleMatchesTheReference(10, 42u);
    assertShuffleMatchesTheReference(100, 99u);
    assertShuffleMatchesTheReference(1000, 1u);
    assertShuffleMatchesTheReference(1000, 0xfc7d0b53u);
}

/* Bounds 10, 1000, then the smallest and the largest valid bound on one
 * stream of a derived key; the Python mirror repeats these values. */
void testRngNextBoundedMatchesReferenceOnADerivedKey(void) {
    rngStream_t stream = rngOpen((rngKey_t){{0x929c8a01u, 0xee4a7510u}});
    const size_t bounds[8] = {10, 10, 10, 10, 1000, 1000, 1, UINT32_MAX};
    const size_t expected[8] = {7, 9, 6, 2, 485, 721, 0, 1286403703};
    for (size_t k = 0; k < 8; ++k) {
        TEST_ASSERT_EQUAL_size_t(expected[k], rngNextBounded(&stream, bounds[k]));
    }
}

/* Without its exit, bound 0 is a remainder by zero and a bound above
 * UINT32_MAX makes the rejection limit 0: either traps or never returns. The
 * child arms the alarm, so a hang becomes a signal the parent reports. */
static void drawWithBoundZero(void) {
    rngStream_t stream = rngOpen((rngKey_t){{1u, 2u}});
    alarm(5);
    (void)rngNextBounded(&stream, 0);
}

void testRngNextBoundedExitsOnBoundZero(void) {
    ASSERT_EXITS_WITH_OUTPUT(1, "rngNextBounded: bound is 0", drawWithBoundZero());
}

#if SIZE_MAX > UINT32_MAX
static void drawWithBoundAboveUint32Max(void) {
    rngStream_t stream = rngOpen((rngKey_t){{1u, 2u}});
    alarm(5);
    (void)rngNextBounded(&stream, (size_t)UINT32_MAX + 1u);
}

void testRngNextBoundedExitsAboveUint32Max(void) {
    ASSERT_EXITS_WITH_OUTPUT(1, "rngNextBounded: bound exceeds UINT32_MAX",
                             drawWithBoundAboveUint32Max());
}
#endif

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testRngNextUnitFloatInRange);
    RUN_TEST(testRngNextUnitFloatDistribution);
    RUN_TEST(testRngFoldMatchesThreefryKnownAnswers);
    RUN_TEST(testRngFoldIsNotSymmetricInTagAndIndex);
    RUN_TEST(testRngFoldDistinctEdgesGiveDistinctKeys);
    RUN_TEST(testRngFoldTwoEdgePathMatchesReference);
    RUN_TEST(testRngFoldKeysDoNotDependOnDerivationOrder);
    RUN_TEST(testRngKeyFromSeedMatchesReference);
    RUN_TEST(testRngKeyFromSeedIsTheFoldOfItsSeedWords);
    RUN_TEST(testRngKeyFromSeedTreatsZeroAsAnOrdinarySeed);
    RUN_TEST(testRngOpenReducesTheKeyToANonZeroState);
    RUN_TEST(testRngNextU32IsTheXorshift32Recurrence);
    RUN_TEST(testRngNextU32MatchesReferenceOnADerivedKey);
    RUN_TEST(testRngOpenedZeroKeysNeverStickAtZero);
    RUN_TEST(testRngStreamsDrawIndependently);
    RUN_TEST(testRngReopenedStreamReplaysItsValues);
    RUN_TEST(testRngNextUnitFloatIsTheTop24BitsOfTheDraw);
    RUN_TEST(testRngNextUnitFloatMatchesReferenceOnADerivedKey);
    RUN_TEST(testRngNextBoundedShufflesLikeTheReference);
    RUN_TEST(testRngNextBoundedMatchesReferenceOnADerivedKey);
    RUN_TEST(testRngNextBoundedExitsOnBoundZero);
#if SIZE_MAX > UINT32_MAX
    RUN_TEST(testRngNextBoundedExitsAboveUint32Max);
#endif
    return UNITY_END();
}
