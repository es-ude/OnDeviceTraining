#include <stdbool.h>
#include <stdlib.h>

#include "DataLoader.h"
#include "DataLoaderApi.h"
#include "Dataset.h"
#include "DeathTest.h"
#include "NPYLoaderApi.h"
#include "OdtAssert.h"
#include "QuantizationApi.h"
#include "RNG.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "unity.h"

/* Compile-time contract: a loader takes no seed of its own. Its order comes
 * from the key handed to dataLoaderShuffle, so one root seed drives the
 * order, the initial values and every other draw of a run. */
_Static_assert(_Generic((&initDataLoader),
                   void (*)(dataLoader_t *, getSampleFn_t, getDatasetSizeFn_t, getBatchFn_t,
                            uint16_t, transformFn_t, transformFn_t, bool, size_t *, bool): 1,
                   default: 0),
               "initDataLoader takes no shuffle seed");
_Static_assert(_Generic((&dataLoaderInit),
                   dataLoader_t *(*)(getSampleFn_t, getDatasetSizeFn_t, uint16_t, transformFn_t,
                                     transformFn_t, bool, bool): 1,
                   default: 0),
               "dataLoaderInit takes no shuffle seed");

/* Compile-time contract: a batch fetch is handed the key of the update the
 * batch feeds, so a loader that draws (the replay wrapper) draws under it. */
_Static_assert(_Generic((getBatchFn_t)0,
                   batch_t *(*)(dataLoader_t *, size_t, const rngKey_t *): 1,
                   default: 0),
               "getBatchFn_t takes the step key");

#define PROXY_DATASET_SIZE 4
#define PROXY_FEATURES 2

/* Per-test fixture handle. Each test calls makeProxyDataset() at the start
 * and freeProxyDataset() once it's captured every assertion value, in line
 * with docs/CONVENTIONS.md "Test memory discipline" (heap-tier idiom). */
typedef struct proxyFixture {
    tensor_t *items[PROXY_DATASET_SIZE];
    tensor_t *labels[PROXY_DATASET_SIZE];
    tensorArray_t itemsArr;
    tensorArray_t labelsArr;
    dataset_t dataset;
} proxyFixture_t;

static proxyFixture_t g_proxy;

static tensor_t *makeProxyTensor(const float *src, size_t count) {
    /* Heap-tier construction per CONVENTIONS Rule 1: shape via reserveMemory,
     * data populated by tensorFillFromFloatBuffer (caller-buffer-safe). */
    size_t *dims = reserveMemory(2 * sizeof(size_t));
    dims[0] = 1;
    dims[1] = PROXY_FEATURES;
    size_t *order = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 2, order);

    tensor_t *t = initTensor(shape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(t, src, count);
    return t;
}

static void makeProxyDataset(void) {
    static const float in0[] = {1.f, 2.f};
    static const float in1[] = {3.f, 4.f};
    static const float in2[] = {5.f, 6.f};
    static const float in3[] = {7.f, 8.f};
    static const float lb0[] = {0.f, 1.f};
    static const float lb1[] = {1.f, 0.f};
    static const float lb2[] = {0.f, 1.f};
    static const float lb3[] = {1.f, 0.f};

    g_proxy.items[0] = makeProxyTensor(in0, PROXY_FEATURES);
    g_proxy.items[1] = makeProxyTensor(in1, PROXY_FEATURES);
    g_proxy.items[2] = makeProxyTensor(in2, PROXY_FEATURES);
    g_proxy.items[3] = makeProxyTensor(in3, PROXY_FEATURES);

    g_proxy.labels[0] = makeProxyTensor(lb0, PROXY_FEATURES);
    g_proxy.labels[1] = makeProxyTensor(lb1, PROXY_FEATURES);
    g_proxy.labels[2] = makeProxyTensor(lb2, PROXY_FEATURES);
    g_proxy.labels[3] = makeProxyTensor(lb3, PROXY_FEATURES);

    g_proxy.itemsArr.array = g_proxy.items;
    g_proxy.itemsArr.size = PROXY_DATASET_SIZE;
    g_proxy.labelsArr.array = g_proxy.labels;
    g_proxy.labelsArr.size = PROXY_DATASET_SIZE;

    g_proxy.dataset.items = &g_proxy.itemsArr;
    g_proxy.dataset.labels = &g_proxy.labelsArr;
}

static void freeProxyDataset(void) {
    /* Release each heap proxy tensor; the tensorArray_t and dataset_t
     * containers themselves are stack/static (members of g_proxy) and
     * need no free. */
    for (size_t i = 0; i < PROXY_DATASET_SIZE; i++) {
        freeTensor(g_proxy.items[i]);
        freeTensor(g_proxy.labels[i]);
    }
}

static sample_t *getProxySample(size_t id) {
    sample_t *s = reserveMemory(sizeof(sample_t));
    s->item = g_proxy.dataset.items->array[id];
    s->label = g_proxy.dataset.labels->array[id];
    return s;
}

static size_t getProxyDatasetSize(void) {
    return PROXY_DATASET_SIZE;
}

/* NPY-backed dataset. Each test that uses it loads at start and frees with
 * freeTensorArray (added in this batch) before exiting. */
static dataset_t g_npyDataset;

static void makeNpyDataset(void) {
    g_npyDataset.items = npyLoad(PROXY_X_PATH);
    g_npyDataset.labels = npyLoad(PROXY_Y_PATH);
}

static void freeNpyDataset(void) {
    freeTensorArray(g_npyDataset.items);
    freeTensorArray(g_npyDataset.labels);
}

static sample_t *getNpyProxySample(size_t id) {
    return npyGetSample(&g_npyDataset, id);
}

static size_t getNpyProxyDatasetSize(void) {
    return g_npyDataset.items->size;
}

void setUp() {}

void tearDown() {}

void testGetSample() {
    makeProxyDataset();

    dataLoader_t *dl =
        dataLoaderInit(getProxySample, getProxyDatasetSize, 1, NULL, NULL, false, true);

    sample_t *s = dl->getSample(0);

    /* CAPTURE: copy data referenced by sample so we can assert after free. */
    float capturedItem[PROXY_FEATURES];
    float capturedLabel[PROXY_FEATURES];
    for (size_t i = 0; i < PROXY_FEATURES; i++) {
        capturedItem[i] = ((float *)s->item->data)[i];
        capturedLabel[i] = ((float *)s->label->data)[i];
    }
    float expectedItem[PROXY_FEATURES];
    float expectedLabel[PROXY_FEATURES];
    for (size_t i = 0; i < PROXY_FEATURES; i++) {
        expectedItem[i] = ((float *)g_proxy.dataset.items->array[0]->data)[i];
        expectedLabel[i] = ((float *)g_proxy.dataset.labels->array[0]->data)[i];
    }

    /* FREE in reverse-init order. */
    freeSample(s);
    freeDataLoader(dl);
    freeProxyDataset();

    /* ASSERT on captured. */
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expectedItem, capturedItem, PROXY_FEATURES);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expectedLabel, capturedLabel, PROXY_FEATURES);
}

void testGetBatch() {
    makeProxyDataset();

    size_t batchSize = 2;
    size_t numBatches = PROXY_DATASET_SIZE / batchSize;
    dataLoader_t *dl =
        dataLoaderInit(getProxySample, getProxyDatasetSize, batchSize, NULL, NULL, false, true);

    /* CAPTURE: walk every batch, copy each sample's data side-by-side with
     * the expected dataset values. Frees happen along the way (samples,
     * batch struct) but assertions wait until everything is freed. */
    float capturedItems[PROXY_DATASET_SIZE][PROXY_FEATURES];
    float capturedLabels[PROXY_DATASET_SIZE][PROXY_FEATURES];
    float expectedItems[PROXY_DATASET_SIZE][PROXY_FEATURES];
    float expectedLabels[PROXY_DATASET_SIZE][PROXY_FEATURES];

    for (size_t b = 0; b < numBatches; b++) {
        batch_t *batch = dl->getBatch(dl, b, NULL);

        for (size_t i = 0; i < batchSize; i++) {
            size_t k = b * batchSize + i;
            for (size_t f = 0; f < PROXY_FEATURES; f++) {
                capturedItems[k][f] = ((float *)batch->samples[i]->item->data)[f];
                capturedLabels[k][f] = ((float *)batch->samples[i]->label->data)[f];
                expectedItems[k][f] = ((float *)g_proxy.dataset.items->array[k]->data)[f];
                expectedLabels[k][f] = ((float *)g_proxy.dataset.labels->array[k]->data)[f];
            }
            freeSample(batch->samples[i]);
        }
        freeBatch(batch);
    }

    freeDataLoader(dl);
    freeProxyDataset();

    /* ASSERT on captured. */
    for (size_t k = 0; k < PROXY_DATASET_SIZE; k++) {
        TEST_ASSERT_EQUAL_FLOAT_ARRAY(expectedItems[k], capturedItems[k], PROXY_FEATURES);
        TEST_ASSERT_EQUAL_FLOAT_ARRAY(expectedLabels[k], capturedLabels[k], PROXY_FEATURES);
    }
}

void testGetBatch_ShuffledMinibatchCoversAllSamples() {
    makeProxyDataset();

    size_t batchSize = 2;
    size_t numBatches = PROXY_DATASET_SIZE / batchSize;
    dataLoader_t *dl =
        dataLoaderInit(getProxySample, getProxyDatasetSize, batchSize, NULL, NULL, true, true);
    const rngKey_t key = rngFold(rngKeyFromSeed(42u), RNG_TAG_SHUFFLE, 0u);
    dataLoaderShuffle(dl, &key);

    /* CAPTURE: collect first feature value from every sample across all
     * batches; assertions wait until everything is freed. */
    float seen[PROXY_DATASET_SIZE];
    for (size_t b = 0; b < numBatches; b++) {
        batch_t *batch = dl->getBatch(dl, b, NULL);
        for (size_t i = 0; i < batchSize; i++) {
            seen[b * batchSize + i] = ((float *)batch->samples[i]->item->data)[0];
            freeSample(batch->samples[i]);
        }
        freeBatch(batch);
    }

    /* Sort the captured values so they can be compared against {1,3,5,7}. */
    for (size_t i = 0; i < PROXY_DATASET_SIZE; i++) {
        for (size_t j = i + 1; j < PROXY_DATASET_SIZE; j++) {
            if (seen[j] < seen[i]) {
                float tmp = seen[i];
                seen[i] = seen[j];
                seen[j] = tmp;
            }
        }
    }

    freeDataLoader(dl);
    freeProxyDataset();

    /* ASSERT on captured. */
    float expected[] = {1.f, 3.f, 5.f, 7.f};
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expected, seen, PROXY_DATASET_SIZE);
}

void testGetBatch_NpyBackedDataset() {
    makeProxyDataset();
    makeNpyDataset();

    size_t capturedNpySize = getNpyProxyDatasetSize();

    size_t batchSize = 2;
    size_t numBatches = PROXY_DATASET_SIZE / batchSize;
    dataLoader_t *dl = dataLoaderInit(getNpyProxySample, getNpyProxyDatasetSize, batchSize, NULL,
                                      NULL, false, true);

    float capturedItems[PROXY_DATASET_SIZE][PROXY_FEATURES];
    float capturedLabels[PROXY_DATASET_SIZE][PROXY_FEATURES];
    float expectedItems[PROXY_DATASET_SIZE][PROXY_FEATURES];
    float expectedLabels[PROXY_DATASET_SIZE][PROXY_FEATURES];

    for (size_t b = 0; b < numBatches; b++) {
        batch_t *batch = dl->getBatch(dl, b, NULL);

        for (size_t i = 0; i < batchSize; i++) {
            size_t k = b * batchSize + i;
            for (size_t f = 0; f < PROXY_FEATURES; f++) {
                capturedItems[k][f] = ((float *)batch->samples[i]->item->data)[f];
                capturedLabels[k][f] = ((float *)batch->samples[i]->label->data)[f];
                expectedItems[k][f] = ((float *)g_proxy.dataset.items->array[k]->data)[f];
                expectedLabels[k][f] = ((float *)g_proxy.dataset.labels->array[k]->data)[f];
            }
            freeSample(batch->samples[i]);
        }
        freeBatch(batch);
    }

    freeDataLoader(dl);
    freeNpyDataset();
    freeProxyDataset();

    /* ASSERT on captured. */
    TEST_ASSERT_EQUAL((size_t)PROXY_DATASET_SIZE, capturedNpySize);
    for (size_t k = 0; k < PROXY_DATASET_SIZE; k++) {
        TEST_ASSERT_EQUAL_FLOAT_ARRAY(expectedItems[k], capturedItems[k], PROXY_FEATURES);
        TEST_ASSERT_EQUAL_FLOAT_ARRAY(expectedLabels[k], capturedLabels[k], PROXY_FEATURES);
    }
}

/* Regression for issue #177: npyLoadFlat must load an ENTIRE .npy file as one
 * contiguous tensor (full shape, dim0 included) — unlike npyLoad, which slices
 * dim0 into N row tensors. The v2 BIT_PARITY weight loader needs every output
 * channel, not just channel 0; the previous npyLoad()+array[0] path saw only the
 * first dim0-slice and then memcpy'd past it into heap garbage. proxy_w.npy is
 * arange(24) reshaped to [3, 2, 4], so flat element i must equal i — and the
 * elements at i >= 8 (beyond row 0 of size 2*4) are exactly what the bug
 * corrupted. */
void testNpyLoadFlat_LoadsFullMultiDimTensor() {
    tensor_t *w = npyLoadFlat(PROXY_W_PATH);

    /* CAPTURE shape + all data before free, per this file's memory discipline. */
    size_t capturedRank = w->shape->numberOfDimensions;
    size_t capturedDims[3] = {0, 0, 0};
    for (size_t i = 0; i < capturedRank && i < 3; i++) {
        capturedDims[i] = w->shape->dimensions[i];
    }
    size_t n = calcNumberOfElementsByTensor(w);
    float captured[24];
    for (size_t i = 0; i < n && i < 24; i++) {
        captured[i] = ((float *)w->data)[i];
    }

    freeTensor(w);

    /* ASSERT on captured. */
    TEST_ASSERT_EQUAL_size_t(3, capturedRank);
    size_t expectedDims[3] = {3, 2, 4};
    ODT_ASSERT_EQUAL_size_t_ARRAY(expectedDims, capturedDims, 3);
    TEST_ASSERT_EQUAL_size_t(24, n);
    float expected[24];
    for (size_t i = 0; i < 24; i++) {
        expected[i] = (float)i;
    }
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expected, captured, 24);
}

/* Order-only fixtures: dataLoaderShuffle and the init checks touch only the
 * indices (never getSample/getBatch), so a stack-only dummy dataset (no
 * tensors, no heap) is enough. */
#define ORDER_DATASET_SIZE 8

static sample_t *orderDummyGetSample(size_t id) {
    (void)id;
    return NULL; /* never invoked: these tests only exercise indices/RNG */
}

static size_t orderDummyGetDatasetSize(void) {
    return ORDER_DATASET_SIZE;
}

static batch_t *orderDummyGetBatch(dataLoader_t *dataLoader, size_t index,
                                   const rngKey_t *stepKey) {
    (void)dataLoader;
    (void)index;
    (void)stepKey;
    return NULL; /* never invoked */
}

void testInitDataLoaderRejectsZeroBatchSize(void) {
    /* Every consumer divides datasetSize / batchSize; 0 must die at init. */
    dataLoader_t dl;
    size_t indices[ORDER_DATASET_SIZE];
    ASSERT_EXITS_WITH_FAILURE(initDataLoader(&dl, orderDummyGetSample, orderDummyGetDatasetSize,
                                             orderDummyGetBatch, 0, NULL, NULL, true, indices,
                                             true));
}

void testDataLoaderInitRejectsZeroBatchSize(void) {
    /* The API wrapper reaches the same guard. */
    ASSERT_EXITS_WITH_FAILURE(
        dataLoaderInit(getProxySample, getProxyDatasetSize, 0, NULL, NULL, false, true));
}

/* ---- the keyed shuffle (#4) ---- */

static size_t singleSampleDatasetSize(void) {
    return 1;
}

/* A shuffle = true loader over the proxy dataset whose order nobody has set yet. */
static dataLoader_t *unshuffledProxyLoader(size_t batchSize) {
    return dataLoaderInit(getProxySample, getProxyDatasetSize, batchSize, NULL, NULL, true, true);
}

/* Fisher-Yates from the identity order under the key: the order depends on the
 * key alone, never on the order before the call. Golds from the Python mirror,
 * tag value spelled as a number (SHUFFLE 21). From the repository root:
 *   uv run python -c "from examples._shared.keyed_rng import (fold,
 *   key_from_seed as kfs, keyed_shuffle as ks); r = kfs(42); print([ks(n,
 *   fold(r, 21, e)) for (n, e) in ((8, 0), (8, 1), (4, 0))])"
 */
void testDataLoaderShuffleIsFisherYatesFromTheIdentityUnderItsKey(void) {
    size_t indices[ORDER_DATASET_SIZE];
    dataLoader_t dl;
    initDataLoader(&dl, orderDummyGetSample, orderDummyGetDatasetSize, orderDummyGetBatch, 1, NULL,
                   NULL, true, indices, true);
    const rngKey_t epoch0 = rngFold(rngKeyFromSeed(42u), RNG_TAG_SHUFFLE, 0u);
    const rngKey_t epoch1 = rngFold(rngKeyFromSeed(42u), RNG_TAG_SHUFFLE, 1u);

    dataLoaderShuffle(&dl, &epoch0);
    size_t first[ORDER_DATASET_SIZE];
    for (size_t i = 0; i < ORDER_DATASET_SIZE; i++) {
        first[i] = indices[i];
    }
    dataLoaderShuffle(&dl, &epoch0);
    size_t again[ORDER_DATASET_SIZE];
    for (size_t i = 0; i < ORDER_DATASET_SIZE; i++) {
        again[i] = indices[i];
    }
    dataLoaderShuffle(&dl, &epoch1);

    size_t expected0[ORDER_DATASET_SIZE] = {1, 0, 7, 3, 2, 5, 6, 4};
    size_t expected1[ORDER_DATASET_SIZE] = {4, 5, 1, 6, 3, 0, 2, 7};
    ODT_ASSERT_EQUAL_size_t_ARRAY(expected0, first, ORDER_DATASET_SIZE);
    ODT_ASSERT_EQUAL_size_t_ARRAY(expected0, again, ORDER_DATASET_SIZE);
    ODT_ASSERT_EQUAL_size_t_ARRAY(expected1, indices, ORDER_DATASET_SIZE);
}

/* The default getBatch serves the order the shuffle set: (4, 0) above is
 * [2, 1, 3, 0], and proxy sample k starts with the value 2k + 1. */
void testGetBatchServesTheOrderOfTheKeyedShuffle(void) {
    makeProxyDataset();
    dataLoader_t *dl = unshuffledProxyLoader(2);
    const rngKey_t key = rngFold(rngKeyFromSeed(42u), RNG_TAG_SHUFFLE, 0u);
    dataLoaderShuffle(dl, &key);

    float seen[PROXY_DATASET_SIZE];
    for (size_t b = 0; b < 2; b++) {
        batch_t *batch = dl->getBatch(dl, b, NULL);
        for (size_t i = 0; i < 2; i++) {
            seen[b * 2 + i] = ((float *)batch->samples[i]->item->data)[0];
            freeSample(batch->samples[i]);
        }
        freeBatch(batch);
    }
    freeDataLoader(dl);
    freeProxyDataset();

    float expected[PROXY_DATASET_SIZE] = {5.f, 3.f, 7.f, 1.f};
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expected, seen, PROXY_DATASET_SIZE);
}

/* Only a draw needs a key: a loader with shuffle = false keeps its order, and
 * one sample is its own shuffled order. */
void testDataLoaderShuffleWithoutADrawNeedsNoKey(void) {
    size_t indices[ORDER_DATASET_SIZE];
    dataLoader_t plain;
    initDataLoader(&plain, orderDummyGetSample, orderDummyGetDatasetSize, orderDummyGetBatch, 1,
                   NULL, NULL, false, indices, true);
    dataLoaderShuffle(&plain, NULL);

    size_t single[1];
    dataLoader_t one;
    initDataLoader(&one, orderDummyGetSample, singleSampleDatasetSize, orderDummyGetBatch, 1, NULL,
                   NULL, true, single, true);
    dataLoaderShuffle(&one, NULL);

    size_t identity[ORDER_DATASET_SIZE] = {0, 1, 2, 3, 4, 5, 6, 7};
    ODT_ASSERT_EQUAL_size_t_ARRAY(identity, indices, ORDER_DATASET_SIZE);
    TEST_ASSERT_EQUAL_size_t(0, single[0]);
    TEST_ASSERT_TRUE(one.shuffled);
}

static void shuffleWithoutAKey(void) {
    size_t indices[ORDER_DATASET_SIZE];
    dataLoader_t dl;
    initDataLoader(&dl, orderDummyGetSample, orderDummyGetDatasetSize, orderDummyGetBatch, 1, NULL,
                   NULL, true, indices, true);
    dataLoaderShuffle(&dl, NULL);
}

void testDataLoaderShuffleWithoutAKeyExits(void) {
    ASSERT_EXITS_WITH_OUTPUT(1, "dataLoaderShuffle: shuffle without a key", shuffleWithoutAKey());
}

static void fetchFromAnUnshuffledLoader(void) {
    makeProxyDataset();
    dataLoader_t *dl = unshuffledProxyLoader(2);
    (void)dl->getBatch(dl, 0, NULL);
}

void testGetBatchOfANeverShuffledLoaderExits(void) {
    ASSERT_EXITS_WITH_OUTPUT(
        1, "getBatch: shuffle = true loader was never shuffled (call dataLoaderShuffle)",
        fetchFromAnUnshuffledLoader());
}

int main() {
    UNITY_BEGIN();
    RUN_TEST(testGetSample);
    RUN_TEST(testGetBatch);
    RUN_TEST(testGetBatch_ShuffledMinibatchCoversAllSamples);
    RUN_TEST(testGetBatch_NpyBackedDataset);
    RUN_TEST(testNpyLoadFlat_LoadsFullMultiDimTensor);
    RUN_TEST(testInitDataLoaderRejectsZeroBatchSize);
    RUN_TEST(testDataLoaderInitRejectsZeroBatchSize);
    RUN_TEST(testDataLoaderShuffleIsFisherYatesFromTheIdentityUnderItsKey);
    RUN_TEST(testGetBatchServesTheOrderOfTheKeyedShuffle);
    RUN_TEST(testDataLoaderShuffleWithoutADrawNeedsNoKey);
    RUN_TEST(testDataLoaderShuffleWithoutAKeyExits);
    RUN_TEST(testGetBatchOfANeverShuffledLoaderExits);
    return UNITY_END();
}
