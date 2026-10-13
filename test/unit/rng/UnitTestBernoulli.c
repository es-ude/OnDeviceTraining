#define SOURCE_FILE "UNIT_TEST_BERNOULLI"

#include <stdbool.h>
#include <stdlib.h>

#include "Bernoulli.h"
#include "DeathTest.h"
#include "QuantizationApi.h"
#include "RNG.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

static tensor_t *buildBoolTensor(size_t n) {
    size_t *dims = reserveMemory(sizeof(size_t));
    dims[0] = n;
    size_t *order = reserveMemory(sizeof(size_t));
    setOrderOfDimsForNewTensor(1, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 1, order);
    return initTensor(shape, quantizationInitBool(), NULL);
}

static size_t countSetBits(tensor_t *mask, size_t n) {
    size_t set = 0;
    for (size_t i = 0; i < n; i++) {
        if (tensorBoolGet(mask, i)) {
            set++;
        }
    }
    return set;
}

void testReferenceRateApproxProbTrue(void) {
    size_t n = 10000;
    tensor_t *mask = buildBoolTensor(n);
    rngStream_t rs = rngOpen(rngKeyFromSeed(12345u));

    bernoulliFillMask(mask, 0.3f, &rs);
    size_t set = countSetBits(mask, n);

    freeTensor(mask);

    // ~30% of bits set; tolerate sampling noise on 10k draws.
    TEST_ASSERT_FLOAT_WITHIN(0.03f, 0.3f, (float)set / (float)n);
}

void testReferenceEdgeAllZero(void) {
    size_t n = 256;
    tensor_t *mask = buildBoolTensor(n);
    rngStream_t rs = rngOpen(rngKeyFromSeed(1u));

    bernoulliFillMask(mask, 0.0f, &rs);
    size_t set = countSetBits(mask, n);

    freeTensor(mask);
    TEST_ASSERT_EQUAL_UINT(0, set);
}

void testReferenceEdgeAllOne(void) {
    size_t n = 256;
    tensor_t *mask = buildBoolTensor(n);
    rngStream_t rs = rngOpen(rngKeyFromSeed(1u));

    bernoulliFillMask(mask, 1.0f, &rs);
    size_t set = countSetBits(mask, n);

    freeTensor(mask);
    TEST_ASSERT_EQUAL_UINT(256, set);
}

static bool sameMask(tensor_t *a, tensor_t *b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (tensorBoolGet(a, i) != tensorBoolGet(b, i)) {
            return false;
        }
    }
    return true;
}

/* The reference sampler is a function of its stream: two streams opened from
 * one key fill the same mask, a stream from another key a different one. */
void testReferenceIsAFunctionOfItsStream(void) {
    size_t n = 512;
    tensor_t *a = buildBoolTensor(n);
    tensor_t *b = buildBoolTensor(n);
    tensor_t *c = buildBoolTensor(n);
    rngStream_t first = rngOpen(rngKeyFromSeed(999u));
    rngStream_t second = rngOpen(rngKeyFromSeed(999u));
    rngStream_t other = rngOpen(rngKeyFromSeed(1000u));

    bernoulliFillMask(a, 0.5f, &first);
    bernoulliFillMask(b, 0.5f, &second);
    bernoulliFillMask(c, 0.5f, &other);
    bool identical = sameMask(a, b, n);
    bool otherDiffers = !sameMask(a, c, n);

    freeTensor(c);
    freeTensor(b);
    freeTensor(a);
    TEST_ASSERT_TRUE(identical);
    TEST_ASSERT_TRUE(otherDiffers);
}

static void fillWithoutAStream(void) {
    tensor_t *mask = buildBoolTensor(8);
    bernoulliFillMask(mask, 0.5f, NULL);
}

/* No key, no randomness: the reference sampler stops at its first draw. */
void testReferenceWithoutAStreamExits(void) {
    ASSERT_EXITS_WITH_OUTPUT(1, "bernoulliFillMaskReference: Bernoulli draw without a stream",
                             fillWithoutAStream());
}

static void fillAnEmptyMaskWithoutAStream(void) {
    size_t dims[1] = {0};
    size_t order[1] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};
    quantization_t boolQ = {.type = BOOL, .qConfig = NULL};
    uint8_t byte = 0;
    tensor_t mask;
    setTensorValues(&mask, &byte, &shape, &boolQ, NULL);
    bernoulliFillMask(&mask, 0.5f, NULL);
}

/* An empty mask draws nothing, so a missing stream never stops it. */
void testReferenceFillsAnEmptyMaskWithoutAStream(void) {
    ASSERT_EXITS_WITH_OUTPUT(0, "", fillAnEmptyMaskWithoutAStream());
}

static void stubFillAllSet(tensor_t *mask, float probTrue, rngStream_t *rs) {
    (void)probTrue;
    (void)rs;
    size_t n = calcNumberOfElementsByTensor(mask);
    for (size_t i = 0; i < n; i++) {
        tensorBoolSet(mask, i, true);
    }
}

void testSwapMechanismInstallsAndRestores(void) {
    size_t n = 64;
    tensor_t *mask = buildBoolTensor(n);

    bernoulliFillMaskFn_t saved = bernoulliGetFillMaskFn();
    bernoulliSetFillMaskFn(stubFillAllSet);

    bool active = (bernoulliGetFillMaskFn() == stubFillAllSet);
    bernoulliFillMask(mask, 0.0f, NULL); // probTrue 0 but stub ignores it → all set
    size_t setAfterStub = countSetBits(mask, n);

    bernoulliSetFillMaskFn(saved);
    bool restored = (bernoulliGetFillMaskFn() == saved);

    freeTensor(mask);
    TEST_ASSERT_TRUE(active);
    TEST_ASSERT_EQUAL_UINT(64, setAfterStub);
    TEST_ASSERT_TRUE(restored);
}

static void fillThroughANonDrawingSamplerWithoutAStream(void) {
    tensor_t *mask = buildBoolTensor(8);
    bernoulliSetFillMaskFn(stubFillAllSet);
    bernoulliFillMask(mask, 0.5f, NULL);
}

/* The stop lives in the reference sampler, not in the dispatcher: an
 * installed sampler that draws nothing takes a NULL stream. */
void testANonDrawingSamplerTakesNoStream(void) {
    ASSERT_EXITS_WITH_OUTPUT(0, "", fillThroughANonDrawingSamplerWithoutAStream());
}

static rngStream_t *recordedStream;
static float recordedProbTrue;

static void recordingFill(tensor_t *mask, float probTrue, rngStream_t *rs) {
    (void)mask;
    recordedProbTrue = probTrue;
    recordedStream = rs;
}

/* The dispatcher hands its caller's stream to the installed sampler unchanged:
 * the stream is the only randomness an installed sampler may use. */
void testFillMaskHandsTheStreamToTheInstalledSampler(void) {
    tensor_t *mask = buildBoolTensor(8);
    rngStream_t stream = rngOpen(rngKeyFromSeed(1u));
    bernoulliFillMaskFn_t saved = bernoulliGetFillMaskFn();
    bernoulliSetFillMaskFn(recordingFill);
    recordedStream = NULL;
    recordedProbTrue = -1.0f;

    bernoulliFillMask(mask, 0.25f, &stream);

    bernoulliSetFillMaskFn(saved);
    freeTensor(mask);
    TEST_ASSERT_EQUAL_PTR(&stream, recordedStream);
    TEST_ASSERT_EQUAL_FLOAT(0.25f, recordedProbTrue);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testReferenceRateApproxProbTrue);
    RUN_TEST(testReferenceEdgeAllZero);
    RUN_TEST(testReferenceEdgeAllOne);
    RUN_TEST(testReferenceIsAFunctionOfItsStream);
    RUN_TEST(testReferenceWithoutAStreamExits);
    RUN_TEST(testReferenceFillsAnEmptyMaskWithoutAStream);
    RUN_TEST(testSwapMechanismInstallsAndRestores);
    RUN_TEST(testANonDrawingSamplerTakesNoStream);
    RUN_TEST(testFillMaskHandsTheStreamToTheInstalledSampler);
    return UNITY_END();
}
