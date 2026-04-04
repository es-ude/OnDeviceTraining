#include <stdio.h>
#include <math.h>

#include "unity.h"
#include "Distributions.h"
#include "RNG.h"
#include "MinMax.h"
#include "Common.h"


static float calculateMean(float *samples, size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        sum += samples[i];
    }
    return sum / (float)n;
}

static float calculateStd(float *samples, size_t n) {
    float mean = calculateMean(samples, n);
    float variance = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float diff = samples[i] - mean;
        variance += diff * diff;
    }
    variance /= (float)n;
    return sqrtf(variance);
}

void testRandomUniform() {
    const size_t n = 10000;
    float samples[n];

    rngSetSeed(42);
    float min_val = -0.5f;
    float max_val = 0.5f;

    for (size_t i = 0; i < n; i++) {
        samples[i] = randomUniform(min_val, max_val);
    }

    float mean = calculateMean(samples, n);
    float min = findMinFloat((uint8_t *)samples, n);
    float max = findMaxFloat((uint8_t *)samples, n);

    float expected_mean = (min_val + max_val) / 2.0f;
    float expected_std = (max_val - min_val) / sqrtf(12.0f);
    float std = calculateStd(samples, n);

    TEST_ASSERT_FLOAT_WITHIN(0.01f, expected_mean, mean);
    TEST_ASSERT_FLOAT_WITHIN(0.01f, expected_std, std);
    TEST_ASSERT_TRUE(min >= min_val && min <= min_val + 0.01f);
    TEST_ASSERT_TRUE(max <= max_val && max >= max_val - 0.01f);
}

void testRandomNormal() {
    const size_t n = 10000;
    float samples[n];

    rngSetSeed(42);
    float expected_mean = 0.0f;
    float expected_std = 0.1f;

    for (size_t i = 0; i < n; i++) {
        samples[i] = randomNormal(expected_mean, expected_std);
    }

    float mean = calculateMean(samples, n);
    float std = calculateStd(samples, n);

    TEST_ASSERT_FLOAT_WITHIN(0.01f, expected_mean, mean);
    TEST_ASSERT_FLOAT_WITHIN(0.01f, expected_std, std);

    int in_1sigma = 0, in_2sigma = 0, in_3sigma = 0;
    for (size_t i = 0; i < n; i++) {
        float z = fabsf((samples[i] - expected_mean) / expected_std);
        if (z <= 1.0f)
            in_1sigma++;
        if (z <= 2.0f)
            in_2sigma++;
        if (z <= 3.0f)
            in_3sigma++;
    }

    TEST_ASSERT_INT_WITHIN(100, 6827, in_1sigma); // ~68.27%
    TEST_ASSERT_INT_WITHIN(100, 9545, in_2sigma); // ~95.45%
    TEST_ASSERT_INT_WITHIN(50, 9973, in_3sigma); // ~99.73%
}

void testKaimingNormal() {
    const size_t n = 10000;
    const size_t fan_in = 784;
    float samples[n];

    rngSetSeed(42);
    for (size_t i = 0; i < n; i++) {
        samples[i] = kaimingNormal(1, fan_in);
    }

    float mean = calculateMean(samples, n);
    float std = calculateStd(samples, n);
    float min = findMinFloat((uint8_t *)samples, n);
    float max = findMaxFloat((uint8_t *)samples, n);

    float expected_std = 1 / sqrtf(fan_in);

    TEST_ASSERT_FLOAT_WITHIN(0.01f, 0.0f, mean);
    TEST_ASSERT_FLOAT_WITHIN(0.01f, expected_std, std);

    float range_5sigma = 5.0f * expected_std;
    TEST_ASSERT_TRUE(min > -range_5sigma);
    TEST_ASSERT_TRUE(max < range_5sigma);
}

void testKaimingUniform() {
    const size_t n = 10000;
    const size_t fan_in = 784;
    float samples[n];

    rngSetSeed(42);
    for (size_t i = 0; i < n; i++) {
        samples[i] = kaimingUniform(1, fan_in);
    }

    float mean = calculateMean(samples, n);
    float std = calculateStd(samples, n);
    float min = findMinFloat((uint8_t *)samples, n);
    float max = findMaxFloat((uint8_t *)samples, n);

    float limit = 1 * sqrtf(3.0f / fan_in);
    float expected_std = limit / sqrtf(3.0f);

    TEST_ASSERT_FLOAT_WITHIN(0.01f, 0.0f, mean);
    TEST_ASSERT_FLOAT_WITHIN(0.01f, expected_std, std);

    TEST_ASSERT_TRUE(min >= -limit && min <= -limit * 0.9f);
    TEST_ASSERT_TRUE(max <= limit && max >= limit * 0.9f);
}

void testXavierNormal() {
    const size_t n = 10000;
    const size_t fan_in = 784;
    const size_t fan_out = 128;
    float samples[n];

    rngSetSeed(42);
    for (size_t i = 0; i < n; i++) {
        samples[i] = xavierNormal(1, fan_in, fan_out);
    }

    float mean = calculateMean(samples, n);
    float std = calculateStd(samples, n);
    float min = findMinFloat((uint8_t *)samples, n);
    float max = findMaxFloat((uint8_t *)samples, n);

    float expected_std = 1 * sqrtf(2.0f / (fan_in + fan_out));

    TEST_ASSERT_FLOAT_WITHIN(0.01f, 0.0f, mean);
    TEST_ASSERT_FLOAT_WITHIN(0.01f, expected_std, std);

    float range_5sigma = 5.0f * expected_std;
    TEST_ASSERT_TRUE(min > -range_5sigma);
    TEST_ASSERT_TRUE(max < range_5sigma);
}

void testXavierUniform() {
    const size_t n = 10000;
    const size_t fan_in = 784;
    const size_t fan_out = 128;
    float samples[n];

    rngSetSeed(42);
    for (size_t i = 0; i < n; i++) {
        samples[i] = xavierUniform(1, fan_in, fan_out);
    }

    float mean = calculateMean(samples, n);
    float std = calculateStd(samples, n);
    float min = findMinFloat((uint8_t *)samples, n);
    float max = findMaxFloat((uint8_t *)samples, n);

    float limit = 1 * sqrtf(6.0f / (fan_in + fan_out));
    float expected_std = limit / sqrtf(3.0f);

    TEST_ASSERT_FLOAT_WITHIN(0.01f, 0.0f, mean);
    TEST_ASSERT_FLOAT_WITHIN(0.01f, expected_std, std);

    TEST_ASSERT_TRUE(min >= -limit && min <= -limit * 0.9f);
    TEST_ASSERT_TRUE(max <= limit && max >= limit * 0.9f);
}

void setUp(void) {}

void tearDown(void) {}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testRandomUniform);
    RUN_TEST(testRandomNormal);

    RUN_TEST(testKaimingNormal);
    RUN_TEST(testKaimingUniform);
    RUN_TEST(testXavierNormal);
    RUN_TEST(testXavierUniform);

    return UNITY_END();
}
