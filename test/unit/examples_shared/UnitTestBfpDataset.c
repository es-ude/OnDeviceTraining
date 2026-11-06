#define SOURCE_FILE "UNIT_TEST_BFP_DATASET"

#include "bfp_dataset.h"

#include "DeathTest.h"
#include "Layer.h"
#include "LayerExec.h"
#include "NPYLoaderApi.h"
#include "Quantization.h"
#include "QuantizationApi.h"
#include "QuantizationLayer.h"
#include "RNG.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "TensorConversion.h"
#include "unity.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void setUp(void) {}
void tearDown(void) {}

_Static_assert(_Generic((&bfpDatasetConvert),
                   tensorArray_t *(*)(const tensorArray_t *, const quantization_t *): 1,
                   default: 0),
               "bfpDatasetConvert(const tensorArray_t *, const quantization_t *)");
_Static_assert(_Generic((&bfpOneHotClass), bool (*)(const tensor_t *, size_t *): 1, default: 0),
               "bfpOneHotClass(const tensor_t *, size_t *)");

#define BD_ROWS 2
#define BD_COLS 24
#define BD_N (BD_ROWS * BD_COLS)

static const char *bdPath(const char *name) {
    static char buf[512];
    snprintf(buf, sizeof(buf), "%s/%s", BFP_DATASET_TEST_DIR, name);
    return buf;
}

static tensor_t *bdTensor2D(size_t rows, size_t cols, quantization_t *q) {
    size_t *dims = reserveMemory(2 * sizeof(size_t));
    dims[0] = rows;
    dims[1] = cols;
    size_t *order = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 2, order);
    return initTensor(shape, q, NULL);
}

/* Three raw-signal-like samples: wide dynamic range across groups, an
 * all-zero group, negative values, ties of the half-away rounding. */
static tensorArray_t *bdFloatSamples(void) {
    tensorArray_t *arr = reserveMemory(sizeof(tensorArray_t));
    arr->size = 3;
    arr->array = reserveMemory(arr->size * sizeof(tensor_t *));
    for (size_t s = 0; s < arr->size; s++) {
        tensor_t *t = bdTensor2D(BD_ROWS, BD_COLS, quantizationInitFloat());
        float *v = (float *)t->data;
        for (size_t i = 0; i < BD_N; i++) {
            float x = (float)((int)((i * 37u + s * 11u) % 29u) - 14) * 0.0371f;
            v[i] = (i / 8 == 2) ? 0.0f : x * (float)(1u << ((i / 8) % 5)) * (s == 1 ? 1e-3f : 1.f);
        }
        v[3] = 0.5f * 0.0625f; /* a tie on some grid */
        arr->array[s] = t;
    }
    return arr;
}

static quantization_t *bdTemplate(roundingMode_t mode) {
    return quantizationInitBfpGrouped(8, 8, mode, BD_N / 8, 8);
}

static void bdAssertSameBfp(const tensor_t *a, const tensor_t *b) {
    TEST_ASSERT_EQUAL_INT(BFP, a->quantization->type);
    TEST_ASSERT_EQUAL_INT(BFP, b->quantization->type);
    const bfpQConfig_t *qa = a->quantization->qConfig;
    const bfpQConfig_t *qb = b->quantization->qConfig;
    TEST_ASSERT_EQUAL_UINT8(qa->mantissaBits, qb->mantissaBits);
    TEST_ASSERT_EQUAL_UINT8(qa->exponentBits, qb->exponentBits);
    TEST_ASSERT_EQUAL_size_t(qa->numGroups, qb->numGroups);
    TEST_ASSERT_EQUAL_size_t(qa->groupSize, qb->groupSize);
    TEST_ASSERT_EQUAL_UINT8_ARRAY(qa->exponents, qb->exponents, qa->numGroups);
    size_t n = calcNumberOfElementsByTensor((tensor_t *)a);
    TEST_ASSERT_EQUAL_size_t(n, calcNumberOfElementsByTensor((tensor_t *)b));
    TEST_ASSERT_EQUAL_UINT8_ARRAY(a->data, b->data, calcNumberOfBytesForData(a->quantization, n));
}

/* The bit-identity contract of the offline path: convert -> file -> read ->
 * the input layer's straight-through gives byte for byte what the input
 * Quantization layer produces online from the float sample. */
void testOfflineConversionThroughTheFileIsBitIdenticalToTheOnlineInputLayer(void) {
    tensorArray_t *floats = bdFloatSamples();
    quantization_t *tmpl = bdTemplate(HALF_AWAY);
    tensorArray_t *bfp = bfpDatasetConvert(floats, tmpl);
    tensorArrayWriteFile(bdPath("bd_items.odta"), bfp);
    tensorArray_t *back = tensorArrayReadFile(bdPath("bd_items.odta"));
    TEST_ASSERT_EQUAL_size_t(floats->size, back->size);

    layer_t qLayer = {.type = QUANTIZATION, .config = NULL};
    for (size_t s = 0; s < floats->size; s++) {
        layerExec_t exec = {.key = rngKeyFromSeed(100u + s)};
        tensor_t *online = bdTensor2D(BD_ROWS, BD_COLS, bdTemplate(HALF_AWAY));
        quantizationForward(&qLayer, floats->array[s], online, &exec);
        tensor_t *offline = bdTensor2D(BD_ROWS, BD_COLS, bdTemplate(HALF_AWAY));
        quantizationForward(&qLayer, back->array[s], offline, &exec);
        bdAssertSameBfp(online, back->array[s]);
        bdAssertSameBfp(online, offline);
        TEST_ASSERT_EQUAL_size_t(2, back->array[s]->shape->numberOfDimensions);
        TEST_ASSERT_EQUAL_size_t(BD_COLS, back->array[s]->shape->dimensions[1]);
        freeTensor(offline);
        freeTensor(online);
    }
    freeTensorArray(back);
    freeTensorArray(bfp);
    freeQuantization(tmpl);
    freeTensorArray(floats);
}

/* An SR template would round with a draw only the online layer's per-call
 * stream can make: the offline conversion takes a deterministic one only. */
static void bdConvertWithSr(void) {
    tensorArray_t *floats = bdFloatSamples();
    quantization_t *tmpl = bdTemplate(SR_HALF_AWAY);
    (void)bfpDatasetConvert(floats, tmpl);
}
static void bdConvertIntoFloat(void) {
    tensorArray_t *floats = bdFloatSamples();
    quantization_t *tmpl = quantizationInitFloat();
    (void)bfpDatasetConvert(floats, tmpl);
}

void testOfflineConversionRejectsANonDeterministicOrNonBfpTemplate(void) {
    ASSERT_EXITS_WITH_OUTPUT(1, "HALF_AWAY", bdConvertWithSr());
    ASSERT_EXITS_WITH_OUTPUT(1, "needs a BFP template", bdConvertIntoFloat());
}

/* The file round trip on its own: per-tensor and grouped BFP, sub-byte m 6,
 * and FLOAT32. */
void testTensorArrayFileRoundTripsBfpAndFloatTensors(void) {
    tensorArray_t *arr = reserveMemory(sizeof(tensorArray_t));
    arr->size = 3;
    arr->array = reserveMemory(3 * sizeof(tensor_t *));
    arr->array[0] = bdTensor2D(1, 6, quantizationInitBfp(8, 8, HALF_AWAY));
    arr->array[1] = bdTensor2D(2, 8, quantizationInitBfpGrouped(6, 5, SR_HALF_AWAY, 4, 4));
    arr->array[2] = bdTensor2D(1, 3, quantizationInitFloat());
    for (size_t k = 0; k < 2; k++) {
        tensor_t *t = arr->array[k];
        size_t bytes = calcNumberOfBytesForData(t->quantization, calcNumberOfElementsByTensor(t));
        for (size_t b = 0; b < bytes; b++) {
            ((uint8_t *)t->data)[b] = (uint8_t)(17u * b + 3u * k + 1u);
        }
        bfpQConfig_t *qc = t->quantization->qConfig;
        for (size_t g = 0; g < qc->numGroups; g++) {
            qc->exponents[g] = (uint8_t)(9u + 5u * g + k);
        }
    }
    ((float *)arr->array[2]->data)[0] = 1.5f;
    ((float *)arr->array[2]->data)[1] = -2.25f;
    ((float *)arr->array[2]->data)[2] = 0.f;

    tensorArrayWriteFile(bdPath("bd_roundtrip.odta"), arr);
    tensorArray_t *back = tensorArrayReadFile(bdPath("bd_roundtrip.odta"));
    TEST_ASSERT_EQUAL_size_t(3, back->size);
    bdAssertSameBfp(arr->array[0], back->array[0]);
    bdAssertSameBfp(arr->array[1], back->array[1]);
    TEST_ASSERT_EQUAL_INT(SR_HALF_AWAY,
                          ((bfpQConfig_t *)back->array[1]->quantization->qConfig)->roundingMode);
    TEST_ASSERT_EQUAL_INT(FLOAT32, back->array[2]->quantization->type);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY((float *)arr->array[2]->data, (float *)back->array[2]->data, 3);
    freeTensorArray(back);
    freeTensorArray(arr);
}

static void bdWriteRaw(const char *name, const char *bytes, size_t n) {
    FILE *f = fopen(bdPath(name), "wb");
    TEST_ASSERT_NOT_NULL(f);
    fwrite(bytes, 1, n, f);
    fclose(f);
}
static void bdReadBadMagic(void) {
    bdWriteRaw("bd_badmagic.odta", "ODTX\x05\x00\x00\x00\x00\x00\x00\x00", 12);
    (void)tensorArrayReadFile(bdPath("bd_badmagic.odta"));
}
static void bdReadBadVersion(void) {
    bdWriteRaw("bd_badversion.odta", "ODTA\x04\x00\x00\x00\x00\x00\x00\x00", 12);
    (void)tensorArrayReadFile(bdPath("bd_badversion.odta"));
}
static void bdReadTrailingBytes(void) {
    bdWriteRaw("bd_trailing.odta", "ODTA\x05\x00\x00\x00\x00\x00\x00\x00Z", 13);
    (void)tensorArrayReadFile(bdPath("bd_trailing.odta"));
}
static void bdReadMissing(void) {
    (void)tensorArrayReadFile(bdPath("bd_does_not_exist.odta"));
}

void testTensorArrayFileRejectsAForeignOrDamagedFile(void) {
    ASSERT_EXITS_WITH_OUTPUT(1, "not an ODTA tensor-array file", bdReadBadMagic());
    ASSERT_EXITS_WITH_OUTPUT(1, "record version", bdReadBadVersion());
    ASSERT_EXITS_WITH_OUTPUT(1, "trailing bytes", bdReadTrailingBytes());
    ASSERT_EXITS_WITH_OUTPUT(1, "cannot open", bdReadMissing());
}

static tensorArray_t *bdClasses(const int32_t *cls, size_t n) {
    tensorArray_t *arr = reserveMemory(sizeof(tensorArray_t));
    arr->size = n;
    arr->array = reserveMemory(n * sizeof(tensor_t *));
    for (size_t i = 0; i < n; i++) {
        size_t *dims = reserveMemory(sizeof(size_t));
        dims[0] = 1;
        size_t *order = reserveMemory(sizeof(size_t));
        order[0] = 0;
        shape_t *shape = reserveMemory(sizeof(shape_t));
        setShape(shape, dims, 1, order);
        arr->array[i] = initTensor(shape, quantizationInitInt32(), NULL);
        ((int32_t *)arr->array[i]->data)[0] = cls[i];
    }
    return arr;
}

/* One-hot labels as BFP: 1.0 is exact (m 8: code 64 on 2^-6). */
void testOneHotLabelsAreExactBfpOneHots(void) {
    const int32_t cls[4] = {0, 5, 2, 3};
    tensorArray_t *classes = bdClasses(cls, 4);
    quantization_t *labelQ = quantizationInitBfp(8, 8, HALF_AWAY);
    tensorArray_t *labels = bfpDatasetOneHotLabels(classes, 6, labelQ);
    TEST_ASSERT_EQUAL_size_t(4, labels->size);
    for (size_t i = 0; i < 4; i++) {
        tensor_t *t = labels->array[i];
        TEST_ASSERT_EQUAL_INT(BFP, t->quantization->type);
        TEST_ASSERT_EQUAL_size_t(1, t->shape->numberOfDimensions);
        TEST_ASSERT_EQUAL_size_t(6, t->shape->dimensions[0]);
        size_t got = 99;
        TEST_ASSERT_TRUE(bfpOneHotClass(t, &got));
        TEST_ASSERT_EQUAL_size_t((size_t)cls[i], got);
        int32_t codes[6];
        unpackSignExtend(t->data, 8, 0, codes, 6);
        TEST_ASSERT_EQUAL_INT32(64, codes[cls[i]]);
        TEST_ASSERT_EQUAL_UINT8(121, ((bfpQConfig_t *)t->quantization->qConfig)->exponents[0]);
    }
    freeTensorArray(labels);
    freeQuantization(labelQ);
    freeTensorArray(classes);
}

static bool bdIsOneHot(const int32_t *codes, uint8_t stored, uint8_t m) {
    tensor_t *t = bdTensor2D(1, 4, quantizationInitBfp(m, 8, HALF_AWAY));
    int32_t tmp[4];
    memcpy(tmp, codes, sizeof(tmp));
    byteConversion((uint8_t *)tmp, 32, t->data, m, 4);
    ((bfpQConfig_t *)t->quantization->qConfig)->exponents[0] = stored;
    size_t cls = 0;
    bool ok = bfpOneHotClass(t, &cls);
    freeTensor(t);
    return ok;
}

void testOneHotClassAcceptsExactlyOneExactOne(void) {
    TEST_ASSERT_TRUE(bdIsOneHot((int32_t[]){0, 64, 0, 0}, 121, 8));
    TEST_ASSERT_TRUE(bdIsOneHot((int32_t[]){0, 0, 1, 0}, 127, 8));    /* 1 on 2^0 */
    TEST_ASSERT_TRUE(bdIsOneHot((int32_t[]){16, 0, 0, 0}, 123, 6));   /* m 6 */
    TEST_ASSERT_FALSE(bdIsOneHot((int32_t[]){0, 64, 64, 0}, 121, 8)); /* two ones */
    TEST_ASSERT_FALSE(bdIsOneHot((int32_t[]){0, 0, 0, 0}, 121, 8));   /* none */
    TEST_ASSERT_FALSE(bdIsOneHot((int32_t[]){0, 65, 0, 0}, 121, 8));  /* 65/64 */
    TEST_ASSERT_FALSE(bdIsOneHot((int32_t[]){0, 32, 32, 0}, 121, 8)); /* soft */
    TEST_ASSERT_FALSE(bdIsOneHot((int32_t[]){0, -64, 0, 0}, 121, 8)); /* -1 */
    TEST_ASSERT_FALSE(bdIsOneHot((int32_t[]){0, 64, 0, 1}, 121, 8));  /* stray code */
    TEST_ASSERT_FALSE(bdIsOneHot((int32_t[]){0, 1, 0, 0}, 128, 8));   /* 2.0 */
    tensor_t *f = bdTensor2D(1, 2, quantizationInitFloat());
    ((float *)f->data)[0] = 1.f;
    ((float *)f->data)[1] = 0.f;
    size_t cls = 0;
    TEST_ASSERT_FALSE(bfpOneHotClass(f, &cls)); /* not BFP */
    freeTensor(f);
}

static void bdLabelOutOfRange(void) {
    const int32_t cls[1] = {6};
    quantization_t *labelQ = quantizationInitBfp(8, 8, HALF_AWAY);
    (void)bfpDatasetOneHotLabels(bdClasses(cls, 1), 6, labelQ);
}

void testOneHotLabelsRejectAClassOutOfRange(void) {
    ASSERT_EXITS_WITH_OUTPUT(1, "out of range", bdLabelOutOfRange());
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testOfflineConversionThroughTheFileIsBitIdenticalToTheOnlineInputLayer);
    RUN_TEST(testOfflineConversionRejectsANonDeterministicOrNonBfpTemplate);
    RUN_TEST(testTensorArrayFileRoundTripsBfpAndFloatTensors);
    RUN_TEST(testTensorArrayFileRejectsAForeignOrDamagedFile);
    RUN_TEST(testOneHotLabelsAreExactBfpOneHots);
    RUN_TEST(testOneHotClassAcceptsExactlyOneExactOne);
    RUN_TEST(testOneHotLabelsRejectAClassOutOfRange);
    return UNITY_END();
}
