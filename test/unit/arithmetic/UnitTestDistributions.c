#include <math.h>
#include <stdio.h>

#include "Common.h"
#include "DeathTest.h"
#include "Distributions.h"
#include "MinMax.h"
#include "QuantizationApi.h"
#include "RNG.h"
#include "StorageApi.h"
#include "TensorApi.h"
#include "unity.h"

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

void testRandomUniformCtxStatistics(void) {
    const size_t n = 10000;
    float samples[n];

    /* Raw key {S, 0}: same draws as the former global stream seeded with S != 0. */
    rngStream_t rs = rngOpen((rngKey_t){{42u, 0u}});
    float min_val = -0.5f;
    float max_val = 0.5f;

    for (size_t i = 0; i < n; i++) {
        samples[i] = randomUniformCtx(&rs, min_val, max_val);
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

void testRandomNormalCtxStatistics(void) {
    const size_t n = 10000;
    float samples[n];

    /* Raw key {S, 0}: same draws as the former global stream seeded with S != 0. */
    rngStream_t rs = rngOpen((rngKey_t){{42u, 0u}});
    float expected_mean = 0.0f;
    float expected_std = 0.1f;

    for (size_t i = 0; i < n; i++) {
        samples[i] = randomNormalCtx(&rs, expected_mean, expected_std);
    }

    float mean = calculateMean(samples, n);
    float std = calculateStd(samples, n);

    TEST_ASSERT_FLOAT_WITHIN(0.01f, expected_mean, mean);
    TEST_ASSERT_FLOAT_WITHIN(0.01f, expected_std, std);

    int in_1sigma = 0, in_2sigma = 0, in_3sigma = 0;
    for (size_t i = 0; i < n; i++) {
        float z = fabsf((samples[i] - expected_mean) / expected_std);
        if (z <= 1.0f) {
            in_1sigma++;
        }
        if (z <= 2.0f) {
            in_2sigma++;
        }
        if (z <= 3.0f) {
            in_3sigma++;
        }
    }

    TEST_ASSERT_INT_WITHIN(100, 6827, in_1sigma); // ~68.27%
    TEST_ASSERT_INT_WITHIN(100, 9545, in_2sigma); // ~95.45%
    TEST_ASSERT_INT_WITHIN(50, 9973, in_3sigma);  // ~99.73%
}

void testKaimingNormal() {
    const size_t n = 10000;
    const size_t fan_in = 784;
    float samples[n];

    rngStream_t rs = rngOpen(rngKeyFromSeed(42u));
    for (size_t i = 0; i < n; i++) {
        samples[i] = kaimingNormal(1, fan_in, &rs);
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

    rngStream_t rs = rngOpen(rngKeyFromSeed(42u));
    for (size_t i = 0; i < n; i++) {
        samples[i] = kaimingUniform(1, fan_in, &rs);
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

    rngStream_t rs = rngOpen(rngKeyFromSeed(42u));
    for (size_t i = 0; i < n; i++) {
        samples[i] = xavierNormal(1, fan_in, fan_out, &rs);
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

    rngStream_t rs = rngOpen(rngKeyFromSeed(42u));
    for (size_t i = 0; i < n; i++) {
        samples[i] = xavierUniform(1, fan_in, fan_out, &rs);
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

void testFillNormalFloat32TensorStats(void) {
    size_t *dims = reserveMemory(sizeof(size_t));
    size_t *order = reserveMemory(sizeof(size_t));
    dims[0] = 10000;
    order[0] = 0;
    shape_t *shape = reserveMemory(sizeof(shape_t));
    shape->dimensions = dims;
    shape->orderOfDimensions = order;
    shape->numberOfDimensions = 1;
    tensor_t *t = initTensor(shape, quantizationInitFloat(), NULL);

    rngStream_t rs = rngOpen(rngKeyFromSeed(42u));
    fillNormalFloat32Tensor(t, &rs, 0.5f, 2.0f);

    float *v = (float *)t->data;
    float sum = 0.0f;
    for (size_t i = 0; i < 10000; i++) {
        sum += v[i];
    }
    float mean = sum / 10000.0f;
    float sq = 0.0f;
    for (size_t i = 0; i < 10000; i++) {
        sq += (v[i] - mean) * (v[i] - mean);
    }
    float std = sqrtf(sq / 10000.0f);
    TEST_ASSERT_FLOAT_WITHIN(0.05f, 0.5f, mean);
    TEST_ASSERT_FLOAT_WITHIN(0.05f, 2.0f, std);
    freeTensor(t);
}

void testFillNormalFloat32TensorDeterministic(void) {
    size_t *dims = reserveMemory(sizeof(size_t));
    size_t *order = reserveMemory(sizeof(size_t));
    dims[0] = 16;
    order[0] = 0;
    shape_t *shape = reserveMemory(sizeof(shape_t));
    shape->dimensions = dims;
    shape->orderOfDimensions = order;
    shape->numberOfDimensions = 1;
    tensor_t *t1 = initTensor(shape, quantizationInitFloat(), NULL);
    size_t *dims2 = reserveMemory(sizeof(size_t));
    size_t *order2 = reserveMemory(sizeof(size_t));
    dims2[0] = 16;
    order2[0] = 0;
    shape_t *shape2 = reserveMemory(sizeof(shape_t));
    shape2->dimensions = dims2;
    shape2->orderOfDimensions = order2;
    shape2->numberOfDimensions = 1;
    tensor_t *t2 = initTensor(shape2, quantizationInitFloat(), NULL);

    rngStream_t a = rngOpen(rngKeyFromSeed(7u));
    rngStream_t b = rngOpen(rngKeyFromSeed(7u));
    fillNormalFloat32Tensor(t1, &a, 0.0f, 1.0f);
    fillNormalFloat32Tensor(t2, &b, 0.0f, 1.0f);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY((float *)t1->data, (float *)t2->data, 16);
    freeTensor(t2);
    freeTensor(t1);
}

void testFillNormalRejectsNonFloat(void) {
    size_t *dims = reserveMemory(sizeof(size_t));
    size_t *order = reserveMemory(sizeof(size_t));
    dims[0] = 8;
    order[0] = 0;
    shape_t *shape = reserveMemory(sizeof(shape_t));
    shape->dimensions = dims;
    shape->orderOfDimensions = order;
    shape->numberOfDimensions = 1;
    quantization_t *q = reserveMemory(sizeof(quantization_t));
    initInt32Quantization(q);
    tensor_t *t = initTensor(shape, q, NULL);
    rngStream_t rs = rngOpen(rngKeyFromSeed(1u));
    ASSERT_EXITS_WITH_FAILURE(fillNormalFloat32Tensor(t, &rs, 0.0f, 1.0f));
    freeTensor(t);
}

/* The keyed uniform scales the stream's unit floats, u -> min + u * (max - min),
 * and leaves the global generator alone. Golds from the Python mirror (the
 * stream of the root of seed 42; -1 + 2u is exact in float32), from the
 * repository root:
 *   uv run python -c "from examples._shared.keyed_rng import (key_from_seed,
 *   open_stream, next_unit_float as nuf); t = (open_stream(key_from_seed(42)), 0.0); d = [
 *   (t := nuf(t[0])) for _ in range(3)]; print([repr(-1 + 2 * u) for _, u in d])"
 */
void testRandomUniformCtxScalesTheStreamsUnitFloats(void) {
    const float gold[3] = {-0.13961052894592285f, -0.08065330982208252f, 0.9558695554733276f};
    rngStream_t rs = rngOpen(rngKeyFromSeed(42u));
    const uint32_t globalBefore = rngGetSeed();
    float draws[3];
    for (size_t i = 0; i < 3; i++) {
        draws[i] = randomUniformCtx(&rs, -1.0f, 1.0f);
    }
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(gold, draws, 3);
    TEST_ASSERT_EQUAL_UINT32(globalBefore, rngGetSeed());
}

static void uniformDrawWithoutAStream(void) {
    (void)randomUniformCtx(NULL, 0.0f, 1.0f);
}

void testRandomUniformCtxExitsWithoutAStream(void) {
    ASSERT_EXITS_WITH_OUTPUT(1, "randomUniformCtx: draw without a stream",
                             uniformDrawWithoutAStream());
}

/* The uniform init helpers draw from the stream they are handed — the keyed
 * uniform over their bound — and leave the global generator alone. */
void testUniformInitHelpersDrawFromTheirStream(void) {
    const rngKey_t key = rngKeyFromSeed(42u);
    rngStream_t got = rngOpen(key);
    rngStream_t want = rngOpen(key);
    const float kaimingLimit = 1.0f * sqrtf(3.0f / 784.0f);
    const float xavierLimit = 1.0f * sqrtf(6.0f / (float)(784 + 128));
    const uint32_t globalBefore = rngGetSeed();
    for (size_t i = 0; i < 4; i++) {
        /* Locals: Unity evaluates a macro argument more than once. */
        const float k = kaimingUniform(1.0f, 784, &got);
        const float kWant = randomUniformCtx(&want, -kaimingLimit, kaimingLimit);
        TEST_ASSERT_EQUAL_FLOAT(kWant, k);
        const float x = xavierUniform(1.0f, 784, 128, &got);
        const float xWant = randomUniformCtx(&want, -xavierLimit, xavierLimit);
        TEST_ASSERT_EQUAL_FLOAT(xWant, x);
    }
    TEST_ASSERT_EQUAL_UINT32(globalBefore, rngGetSeed());
}

static void kaimingDrawWithoutAStream(void) {
    (void)kaimingUniform(1.0f, 784, NULL);
}

void testKaimingUniformExitsWithoutAStream(void) {
    ASSERT_EXITS_WITH_OUTPUT(1, "randomUniformCtx: draw without a stream",
                             kaimingDrawWithoutAStream());
}

/* The normal init helpers draw from the stream they are handed -- the keyed
 * normal at their standard deviation -- and leave the global generator alone. */
void testNormalInitHelpersDrawFromTheirStream(void) {
    const rngKey_t key = rngKeyFromSeed(42u);
    rngStream_t got = rngOpen(key);
    rngStream_t want = rngOpen(key);
    const float kaimingStd = 1.0f / sqrtf(784.0f);
    const float xavierStd = 1.0f * sqrtf(2.0f / (float)(784 + 128));
    const uint32_t globalBefore = rngGetSeed();
    for (size_t i = 0; i < 4; i++) {
        /* Locals: Unity evaluates a macro argument more than once. */
        const float k = kaimingNormal(1.0f, 784, &got);
        const float kWant = randomNormalCtx(&want, 0.0f, kaimingStd);
        TEST_ASSERT_EQUAL_FLOAT(kWant, k);
        const float x = xavierNormal(1.0f, 784, 128, &got);
        const float xWant = randomNormalCtx(&want, 0.0f, xavierStd);
        TEST_ASSERT_EQUAL_FLOAT(xWant, x);
    }
    TEST_ASSERT_EQUAL_UINT32(globalBefore, rngGetSeed());
}

static void normalDrawWithoutAStream(void) {
    (void)randomNormalCtx(NULL, 0.0f, 1.0f);
}

void testRandomNormalCtxExitsWithoutAStream(void) {
    ASSERT_EXITS_WITH_OUTPUT(1, "randomNormalCtx: draw without a stream",
                             normalDrawWithoutAStream());
}

static void kaimingNormalDrawWithoutAStream(void) {
    (void)kaimingNormal(1.0f, 784, NULL);
}

void testKaimingNormalExitsWithoutAStream(void) {
    ASSERT_EXITS_WITH_OUTPUT(1, "randomNormalCtx: draw without a stream",
                             kaimingNormalDrawWithoutAStream());
}

/* An empty tensor draws nothing, so it needs no stream. Host only: a
 * zero-element allocation may be NULL on newlib (#160). */
static void normalFillOfAnEmptyTensor(void) {
    size_t *dims = reserveMemory(sizeof(size_t));
    size_t *order = reserveMemory(sizeof(size_t));
    dims[0] = 0;
    order[0] = 0;
    shape_t *shape = reserveMemory(sizeof(shape_t));
    shape->dimensions = dims;
    shape->orderOfDimensions = order;
    shape->numberOfDimensions = 1;
    tensor_t *t = initTensor(shape, quantizationInitFloat(), NULL);
    fillNormalFloat32Tensor(t, NULL, 0.0f, 1.0f);
    freeTensor(t);
}

void testFillNormalOfAnEmptyTensorNeedsNoStream(void) {
    ASSERT_EXITS_WITH(0, normalFillOfAnEmptyTensor());
}

void setUp(void) {}

void tearDown(void) {}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testRandomUniformCtxStatistics);
    RUN_TEST(testRandomNormalCtxStatistics);

    RUN_TEST(testKaimingNormal);
    RUN_TEST(testKaimingUniform);
    RUN_TEST(testXavierNormal);
    RUN_TEST(testXavierUniform);

    RUN_TEST(testFillNormalFloat32TensorStats);
    RUN_TEST(testFillNormalFloat32TensorDeterministic);
    RUN_TEST(testFillNormalRejectsNonFloat);
    RUN_TEST(testRandomUniformCtxScalesTheStreamsUnitFloats);
    RUN_TEST(testRandomUniformCtxExitsWithoutAStream);
    RUN_TEST(testUniformInitHelpersDrawFromTheirStream);
    RUN_TEST(testKaimingUniformExitsWithoutAStream);
    RUN_TEST(testNormalInitHelpersDrawFromTheirStream);
    RUN_TEST(testRandomNormalCtxExitsWithoutAStream);
    RUN_TEST(testKaimingNormalExitsWithoutAStream);
    RUN_TEST(testFillNormalOfAnEmptyTensorNeedsNoStream);

    return UNITY_END();
}
