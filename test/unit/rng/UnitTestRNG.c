#include "unity.h"
#include "RNG.h"

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
        if (bucket >= buckets) bucket = buckets - 1;
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

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testRngNextFloatInRange);
    RUN_TEST(testRngNextFloatDistribution);
    RUN_TEST(testRngNextFloatReproducible);
    RUN_TEST(testRngShuffleUsesGlobalState);
    return UNITY_END();
}
