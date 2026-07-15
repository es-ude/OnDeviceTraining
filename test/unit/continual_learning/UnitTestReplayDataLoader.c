#define SOURCE_FILE "UNIT_TEST_REPLAY_DATA_LOADER"

#include <stdlib.h>

#include "DataLoaderApi.h"
#include "DeathTest.h"
#include "PpcaReplay.h"
#include "PpcaReplayApi.h"
#include "Quantization.h"
#include "QuantizationApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

static tensor_t *buildFloat32TensorND(size_t rank, const size_t *srcDims, const float *fill) {
    size_t *dims = reserveMemory(rank * sizeof(size_t));
    size_t *order = reserveMemory(rank * sizeof(size_t));
    for (size_t i = 0; i < rank; i++) {
        dims[i] = srcDims[i];
        order[i] = i;
    }
    shape_t *shape = reserveMemory(sizeof(shape_t));
    shape->dimensions = dims;
    shape->orderOfDimensions = order;
    shape->numberOfDimensions = rank;
    tensor_t *t = initTensor(shape, quantizationInitFloat(), NULL);
    if (fill != NULL) {
        float *d = (float *)t->data;
        size_t n = calcNumberOfElementsByTensor(t);
        for (size_t i = 0; i < n; i++) {
            d[i] = fill[i];
        }
    }
    return t;
}

static ppcaReplayConfig_t floatConfig(size_t dim, size_t rank, size_t maxM) {
    static quantization_t floatQ; /* static: outlives the call; cloned by create */
    initFloat32Quantization(&floatQ);
    ppcaReplayConfig_t cfg = {
        .dim = dim,
        .rank = rank,
        .maxSessionSamples = maxM,
        .mergeMath = {.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY},
        .streamMath = {.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY},
        .sampleMath = {.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY},
        .meanQ = &floatQ,
        .basisQ = &floatQ,
        .eigvalsQ = &floatQ,
        .sigma2Floor = 1e-6f,
        .shrinkageGamma = 0.0f,
    };
    return cfg;
}

/* Fixed fake dataset: 4 samples, d=6 items, 2 classes one-hot labels. */
#define FAKE_N 4
static tensor_t *g_items[FAKE_N];
static tensor_t *g_labels[FAKE_N];

static void buildFakeDataset(void) {
    for (size_t i = 0; i < FAKE_N; i++) {
        float vals[6];
        for (size_t j = 0; j < 6; j++) {
            vals[j] = (float)(i * 10 + j);
        }
        g_items[i] = buildFloat32TensorND(1, (size_t[]){6}, vals);
        float onehot[2] = {i % 2 == 0 ? 1.0f : 0.0f, i % 2 == 0 ? 0.0f : 1.0f};
        g_labels[i] = buildFloat32TensorND(1, (size_t[]){2}, onehot);
    }
}

static void freeFakeDataset(void) {
    for (size_t i = 0; i < FAKE_N; i++) {
        freeTensor(g_items[i]);
        freeTensor(g_labels[i]);
    }
}

static sample_t *fakeGetSample(size_t id) {
    sample_t *s = reserveMemory(sizeof(sample_t));
    s->item = g_items[id];
    s->label = g_labels[id];
    return s;
}

static size_t fakeGetSize(void) {
    return FAKE_N;
}

/* Seed both class generators to sampling-ready state (count high, simple
 * identity-ish basis) — mirrors buildKnownGenerator but for d=6, k=2. */
static void makeSetSampleable(ppcaReplaySet_t *set) {
    for (size_t c = 0; c < set->numClasses; c++) {
        ppcaReplay_t *g = set->generators[c];
        float *b = (float *)g->basis->data;
        b[0] = 1.0f;         /* row0 = e0 */
        b[1 * 6 + 1] = 1.0f; /* row1 = e1 */
        float *lam = (float *)g->eigvals->data;
        lam[0] = 2.0f;
        lam[1] = 1.0f;
        g->sigma2 = 0.01f;
        g->count = 50;
    }
}

void testWrappedBatchAppendsSyntheticSamples(void) {
    buildFakeDataset();
    dataLoader_t *base = dataLoaderInit(fakeGetSample, fakeGetSize, 2, NULL, NULL, false, 0, true);
    ppcaReplayConfig_t cfg = floatConfig(6, 2, 8);
    ppcaReplaySet_t *set = ppcaReplaySetCreate(2, &cfg);
    makeSetSampleable(set);
    rng32_t stream = {.state = 99};
    replayLoaderConfig_t rcfg = {
        .set = set, .samplesPerClass = 1, .minCount = 1, .stream = &stream};
    dataLoader_t *wrapped = replayDataLoaderWrap(base, &rcfg);

    batch_t *batch = wrapped->getBatch(wrapped, 0);

    TEST_ASSERT_EQUAL_size_t(4, batch->size); /* 2 real + 2 classes * r=1 */
    /* real first: pointer identity with the dataset tensors */
    TEST_ASSERT_EQUAL_PTR(g_items[0], batch->samples[0]->item);
    TEST_ASSERT_EQUAL_PTR(g_items[1], batch->samples[1]->item);
    /* synthetic labels are one-hot with exactly one 1.0 */
    for (size_t s = 2; s < 4; s++) {
        float *lab = (float *)batch->samples[s]->label->data;
        TEST_ASSERT_EQUAL_FLOAT(1.0f, lab[0] + lab[1]);
        TEST_ASSERT_EQUAL_size_t(6, calcNumberOfElementsByTensor(batch->samples[s]->item));
    }
    /* emulate the training loop's ownership protocol */
    for (size_t s = 0; s < batch->size; s++) {
        freeSample(batch->samples[s]);
    }
    freeBatch(batch);

    /* pools are REUSED: a second batch must not crash and stays sized */
    batch_t *batch2 = wrapped->getBatch(wrapped, 1);
    TEST_ASSERT_EQUAL_size_t(4, batch2->size);
    for (size_t s = 0; s < batch2->size; s++) {
        freeSample(batch2->samples[s]);
    }
    freeBatch(batch2);

    freeReplayDataLoader(wrapped);
    freeDataLoader(base);
    freePpcaReplaySet(set);
    freeFakeDataset();
}

void testMinCountGatesClasses(void) {
    buildFakeDataset();
    dataLoader_t *base = dataLoaderInit(fakeGetSample, fakeGetSize, 2, NULL, NULL, false, 0, true);
    ppcaReplayConfig_t cfg = floatConfig(6, 2, 8);
    ppcaReplaySet_t *set = ppcaReplaySetCreate(2, &cfg);
    makeSetSampleable(set);
    set->generators[1]->count = 0; /* class 1 not eligible */
    rng32_t stream = {.state = 5};
    replayLoaderConfig_t rcfg = {
        .set = set, .samplesPerClass = 2, .minCount = 1, .stream = &stream};
    dataLoader_t *wrapped = replayDataLoaderWrap(base, &rcfg);

    batch_t *batch = wrapped->getBatch(wrapped, 0);
    TEST_ASSERT_EQUAL_size_t(2 + 2, batch->size); /* only class 0 adds r=2 */
    for (size_t s = 0; s < batch->size; s++) {
        freeSample(batch->samples[s]);
    }
    freeBatch(batch);
    freeReplayDataLoader(wrapped);
    freeDataLoader(base);
    freePpcaReplaySet(set);
    freeFakeDataset();
}

void testNoEligibleClassPassesBaseBatchThrough(void) {
    buildFakeDataset();
    dataLoader_t *base = dataLoaderInit(fakeGetSample, fakeGetSize, 2, NULL, NULL, false, 0, true);
    ppcaReplayConfig_t cfg = floatConfig(6, 2, 8);
    ppcaReplaySet_t *set = ppcaReplaySetCreate(2, &cfg); /* counts all 0 */
    rng32_t stream = {.state = 5};
    replayLoaderConfig_t rcfg = {
        .set = set, .samplesPerClass = 1, .minCount = 1, .stream = &stream};
    dataLoader_t *wrapped = replayDataLoaderWrap(base, &rcfg);
    batch_t *batch = wrapped->getBatch(wrapped, 0);
    TEST_ASSERT_EQUAL_size_t(2, batch->size);
    for (size_t s = 0; s < batch->size; s++) {
        freeSample(batch->samples[s]);
    }
    freeBatch(batch);
    freeReplayDataLoader(wrapped);
    freeDataLoader(base);
    freePpcaReplaySet(set);
    freeFakeDataset();
}

void testSyntheticSamplingDeterministic(void) {
    buildFakeDataset();
    for (int run = 0; run < 2; run++) {
        static float firstRun[6];
        dataLoader_t *base =
            dataLoaderInit(fakeGetSample, fakeGetSize, 2, NULL, NULL, false, 0, true);
        ppcaReplayConfig_t cfg = floatConfig(6, 2, 8);
        ppcaReplaySet_t *set = ppcaReplaySetCreate(2, &cfg);
        makeSetSampleable(set);
        rng32_t stream = {.state = 12345};
        replayLoaderConfig_t rcfg = {
            .set = set, .samplesPerClass = 1, .minCount = 1, .stream = &stream};
        dataLoader_t *wrapped = replayDataLoaderWrap(base, &rcfg);
        batch_t *batch = wrapped->getBatch(wrapped, 0);
        float *synth = (float *)batch->samples[2]->item->data;
        if (run == 0) {
            for (size_t j = 0; j < 6; j++) {
                firstRun[j] = synth[j];
            }
        } else {
            TEST_ASSERT_EQUAL_FLOAT_ARRAY(firstRun, synth, 6);
        }
        for (size_t s = 0; s < batch->size; s++) {
            freeSample(batch->samples[s]);
        }
        freeBatch(batch);
        freeReplayDataLoader(wrapped);
        freeDataLoader(base);
        freePpcaReplaySet(set);
    }
    freeFakeDataset();
}

void testMeanModeReplaysClassCentroidsWithoutStream(void) {
    /* REPLAY_MODE_CLASS_MEAN (#326 fair-comparison baseline): pool items are
     * the running class centroids, exactly; mean replay is deterministic, so
     * no RNG stream is required (NULL is legal in this mode only). */
    buildFakeDataset();
    dataLoader_t *base = dataLoaderInit(fakeGetSample, fakeGetSize, 2, NULL, NULL, false, 0, true);
    ppcaReplayConfig_t cfg = floatConfig(6, 2, 8);
    ppcaReplaySet_t *set = ppcaReplaySetCreate(2, &cfg);
    makeSetSampleable(set);
    for (size_t c = 0; c < 2; c++) {
        float *mu = (float *)set->generators[c]->mean->data;
        for (size_t j = 0; j < 6; j++) {
            mu[j] = (float)(100 * (c + 1) + j);
        }
    }
    replayLoaderConfig_t rcfg = {.set = set,
                                 .samplesPerClass = 2,
                                 .minCount = 1,
                                 .stream = NULL,
                                 .mode = REPLAY_MODE_CLASS_MEAN};
    dataLoader_t *wrapped = replayDataLoaderWrap(base, &rcfg);

    batch_t *batch = wrapped->getBatch(wrapped, 0);
    TEST_ASSERT_EQUAL_size_t(2 + 2 * 2, batch->size);
    for (size_t s = 2; s < 6; s++) {
        size_t c = (s - 2) / 2; /* synthetic samples appended class-major */
        TEST_ASSERT_EQUAL_FLOAT_ARRAY((float *)set->generators[c]->mean->data,
                                      (float *)batch->samples[s]->item->data, 6);
        float *lab = (float *)batch->samples[s]->label->data;
        TEST_ASSERT_EQUAL_FLOAT(1.0f, lab[c]);
    }
    for (size_t s = 0; s < batch->size; s++) {
        freeSample(batch->samples[s]);
    }
    freeBatch(batch);
    freeReplayDataLoader(wrapped);
    freeDataLoader(base);
    freePpcaReplaySet(set);
    freeFakeDataset();
}

void testPpcaModeStillRequiresStream(void) {
    /* The relaxed NULL-stream rule is mode-scoped: default (zero-init) PPCA
     * mode must keep failing fast without a stream. */
    buildFakeDataset();
    dataLoader_t *base = dataLoaderInit(fakeGetSample, fakeGetSize, 2, NULL, NULL, false, 0, true);
    ppcaReplayConfig_t cfg = floatConfig(6, 2, 8);
    ppcaReplaySet_t *set = ppcaReplaySetCreate(2, &cfg);
    replayLoaderConfig_t rcfg = {.set = set, .samplesPerClass = 1, .minCount = 1, .stream = NULL};
    ASSERT_EXITS_WITH_FAILURE(replayDataLoaderWrap(base, &rcfg));
    freeDataLoader(base);
    freePpcaReplaySet(set);
    freeFakeDataset();
}

void testExemplarBufferKeepsFirstKAndCopies(void) {
    /* First-K policy: adds beyond capacity are ignored; stored exemplars are
     * COPIES (mutating the source afterwards must not reach the buffer). */
    exemplarBuffer_t *buf = exemplarBufferCreate(2, 2);
    TEST_ASSERT_EQUAL_size_t(2, buf->numClasses);
    TEST_ASSERT_EQUAL_size_t(2, buf->capacity);
    float v0[6] = {0, 1, 2, 3, 4, 5};
    float v1[6] = {10, 11, 12, 13, 14, 15};
    float v2[6] = {20, 21, 22, 23, 24, 25};
    tensor_t *src = buildFloat32TensorND(1, (size_t[]){6}, v0);
    exemplarBufferAdd(buf, src, 0);
    float *d = (float *)src->data;
    for (size_t j = 0; j < 6; j++) {
        d[j] = v1[j];
    }
    exemplarBufferAdd(buf, src, 0);
    for (size_t j = 0; j < 6; j++) {
        d[j] = v2[j]; /* over capacity: must be dropped, and must not alias */
    }
    exemplarBufferAdd(buf, src, 0);
    TEST_ASSERT_EQUAL_UINT32(2, buf->counts[0]);
    TEST_ASSERT_EQUAL_UINT32(0, buf->counts[1]);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(v0, (float *)buf->items[0]->data, 6);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(v1, (float *)buf->items[1]->data, 6);
    freeTensor(src);
    freeExemplarBuffer(buf);
}

void testExemplarBufferRejectsBadClassAndShape(void) {
    exemplarBuffer_t *buf = exemplarBufferCreate(2, 2);
    tensor_t *ok = buildFloat32TensorND(1, (size_t[]){6}, NULL);
    tensor_t *bad = buildFloat32TensorND(1, (size_t[]){5}, NULL);
    ASSERT_EXITS_WITH_FAILURE(exemplarBufferAdd(buf, ok, 2)); /* class OOB */
    exemplarBufferAdd(buf, ok, 0);
    ASSERT_EXITS_WITH_FAILURE(exemplarBufferAdd(buf, bad, 0)); /* element mismatch */
    freeTensor(bad);
    freeTensor(ok);
    freeExemplarBuffer(buf);
}

void testExemplarModeReplaysStoredSamplesZeroCopy(void) {
    /* REPLAY_MODE_EXEMPLAR: slots point INTO the buffer (no copy); classes
     * with an empty buffer are skipped; the PPCA set is not required. */
    buildFakeDataset();
    dataLoader_t *base = dataLoaderInit(fakeGetSample, fakeGetSize, 2, NULL, NULL, false, 0, true);
    exemplarBuffer_t *buf = exemplarBufferCreate(2, 2);
    float e0[6] = {1, 1, 1, 1, 1, 1};
    float e1[6] = {2, 2, 2, 2, 2, 2};
    tensor_t *t0 = buildFloat32TensorND(1, (size_t[]){6}, e0);
    tensor_t *t1 = buildFloat32TensorND(1, (size_t[]){6}, e1);
    exemplarBufferAdd(buf, t0, 0);
    exemplarBufferAdd(buf, t1, 0); /* class 1 stays empty -> not eligible */
    freeTensor(t1);
    freeTensor(t0);
    rng32_t stream = {.state = 77};
    replayLoaderConfig_t rcfg = {.exemplars = buf,
                                 .samplesPerClass = 2,
                                 .minCount = 1,
                                 .stream = &stream,
                                 .mode = REPLAY_MODE_EXEMPLAR};
    dataLoader_t *wrapped = replayDataLoaderWrap(base, &rcfg);

    /* over several batches the uniform pick must reach BOTH stored
     * exemplars of class 0 (kills a fixed-index pick) */
    int seen[2] = {0, 0};
    for (size_t b = 0; b < 6; b++) {
        batch_t *batch = wrapped->getBatch(wrapped, b % 2);
        TEST_ASSERT_EQUAL_size_t(2 + 2, batch->size); /* only class 0 adds r=2 */
        for (size_t s = 2; s < 4; s++) {
            tensor_t *item = batch->samples[s]->item;
            if (item == buf->items[0]) {
                seen[0] = 1;
            } else if (item == buf->items[1]) {
                seen[1] = 1;
            } else {
                TEST_FAIL_MESSAGE("synthetic item is not a buffer tensor (copy?)");
            }
            float *lab = (float *)batch->samples[s]->label->data;
            TEST_ASSERT_EQUAL_FLOAT(1.0f, lab[0]);
        }
        for (size_t s = 0; s < batch->size; s++) {
            freeSample(batch->samples[s]);
        }
        freeBatch(batch);
    }
    TEST_ASSERT_TRUE(seen[0]);
    TEST_ASSERT_TRUE(seen[1]);
    freeReplayDataLoader(wrapped);
    freeDataLoader(base);
    freeExemplarBuffer(buf);
    freeFakeDataset();
}

void testExemplarModePicksDeterministic(void) {
    buildFakeDataset();
    static size_t firstRun[4];
    for (int run = 0; run < 2; run++) {
        dataLoader_t *base =
            dataLoaderInit(fakeGetSample, fakeGetSize, 2, NULL, NULL, false, 0, true);
        exemplarBuffer_t *buf = exemplarBufferCreate(2, 3);
        tensor_t *t = buildFloat32TensorND(1, (size_t[]){6}, NULL);
        for (size_t c = 0; c < 2; c++) {
            for (size_t i = 0; i < 3; i++) {
                exemplarBufferAdd(buf, t, c);
            }
        }
        freeTensor(t);
        rng32_t stream = {.state = 4242};
        replayLoaderConfig_t rcfg = {.exemplars = buf,
                                     .samplesPerClass = 2,
                                     .minCount = 1,
                                     .stream = &stream,
                                     .mode = REPLAY_MODE_EXEMPLAR};
        dataLoader_t *wrapped = replayDataLoaderWrap(base, &rcfg);
        batch_t *batch = wrapped->getBatch(wrapped, 0);
        for (size_t s = 2; s < 6; s++) {
            /* compare picked buffer SLOT indices across runs via pointer
             * offsets (tensors are per-run heap objects) */
            size_t slot = SIZE_MAX;
            for (size_t k = 0; k < 6; k++) {
                if (batch->samples[s]->item == buf->items[k]) {
                    slot = k;
                }
            }
            TEST_ASSERT_TRUE(slot != SIZE_MAX);
            if (run == 0) {
                firstRun[s - 2] = slot;
            } else {
                TEST_ASSERT_EQUAL_size_t(firstRun[s - 2], slot);
            }
        }
        for (size_t s = 0; s < batch->size; s++) {
            freeSample(batch->samples[s]);
        }
        freeBatch(batch);
        freeReplayDataLoader(wrapped);
        freeDataLoader(base);
        freeExemplarBuffer(buf);
    }
    freeFakeDataset();
}

void testExemplarModeRequiresBufferAndStream(void) {
    buildFakeDataset();
    dataLoader_t *base = dataLoaderInit(fakeGetSample, fakeGetSize, 2, NULL, NULL, false, 0, true);
    exemplarBuffer_t *buf = exemplarBufferCreate(2, 2);
    rng32_t stream = {.state = 1};
    replayLoaderConfig_t noBuf = {
        .samplesPerClass = 1, .minCount = 1, .stream = &stream, .mode = REPLAY_MODE_EXEMPLAR};
    ASSERT_EXITS_WITH_FAILURE(replayDataLoaderWrap(base, &noBuf));
    replayLoaderConfig_t noStream = {
        .exemplars = buf, .samplesPerClass = 1, .minCount = 1, .mode = REPLAY_MODE_EXEMPLAR};
    ASSERT_EXITS_WITH_FAILURE(replayDataLoaderWrap(base, &noStream));
    freeExemplarBuffer(buf);
    freeDataLoader(base);
    freeFakeDataset();
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testWrappedBatchAppendsSyntheticSamples);
    RUN_TEST(testMinCountGatesClasses);
    RUN_TEST(testNoEligibleClassPassesBaseBatchThrough);
    RUN_TEST(testSyntheticSamplingDeterministic);
    RUN_TEST(testMeanModeReplaysClassCentroidsWithoutStream);
    RUN_TEST(testPpcaModeStillRequiresStream);
    RUN_TEST(testExemplarBufferKeepsFirstKAndCopies);
    RUN_TEST(testExemplarBufferRejectsBadClassAndShape);
    RUN_TEST(testExemplarModeReplaysStoredSamplesZeroCopy);
    RUN_TEST(testExemplarModePicksDeterministic);
    RUN_TEST(testExemplarModeRequiresBufferAndStream);
    return UNITY_END();
}
