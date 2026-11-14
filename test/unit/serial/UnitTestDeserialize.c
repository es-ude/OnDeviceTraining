#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "ArithmeticType.h"
#include "BatchNorm1d.h"
#include "BatchNorm1dApi.h"
#include "Conv1dTransposed.h"
#include "Conv1dTransposedApi.h"
#include "DagFixtures.h"
#include "DeathTest.h"
#include "Deserialize.h"
#include "Dropout.h"
#include "DropoutApi.h"
#include "Flatten.h"
#include "FlattenApi.h"
#include "GroupNorm.h"
#include "GroupNormApi.h"
#include "Layer.h"
#include "LayerNorm.h"
#include "LayerNormApi.h"
#include "LayerQuant.h"
#include "Model.h"
#include "OdtAssert.h"
#include "QuantizationApi.h"
#include "RNG.h"
#include "Relu.h"
#include "ReluApi.h"
#include "Serialize.h"
#include "Softmax.h"
#include "SoftmaxApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "TensorRecord.h"
#include "TestScale.h"
#include "unity.h"

/* SERIALIZE_TEST_FILE_PATH is injected by the CMake target_compile_definitions
 * in test/unit/serial/CMakeLists.txt as an absolute path so the test does not
 * depend on the working directory (which differs between host runs and Docker
 * LSan runs). */
#define FILE_PATH SERIALIZE_TEST_FILE_PATH

/* Current ODTS format version for hand-crafted fixtures. Bump together with
 * SERIALIZE_FORMAT_VERSION: a stale value makes every fixture below die at
 * the version check instead of at the guard it targets (a vacuous pass). */
#define ODTS_VERSION 9u

/* Fixture writer for hand-crafted files: explicit little-endian bytes, so
 * the fixtures stay valid even on a big-endian test host. */
static void writeU32LE(FILE *f, uint32_t value) {
    uint8_t bytes[4] = {(uint8_t)value, (uint8_t)(value >> 8), (uint8_t)(value >> 16),
                        (uint8_t)(value >> 24)};
    fwrite(bytes, 1, 4, f);
}

static void writeU8(FILE *f, uint8_t value) {
    fwrite(&value, 1, 1, f);
}

/* Companion to writeU32LE for FLOAT32 payload values. */
static void writeF32LE(FILE *f, float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    writeU32LE(f, bits);
}

/* group-quant PR4 (Task 4): companion for the v5 ASYM record's
 * zeroPoints[numGroups] u16 LE array. */
static void writeU16LE(FILE *f, uint16_t value) {
    uint8_t bytes[2] = {(uint8_t)value, (uint8_t)(value >> 8)};
    fwrite(bytes, 1, 2, f);
}

/* A v9 scale field: `u16 m`, `i16 e`, both LE. Takes the raw fields so a
 * fixture can write a non-canonical scale. */
static void writeScaleLE(FILE *f, uint16_t m, int16_t e) {
    writeU16LE(f, m);
    writeU16LE(f, (uint16_t)e);
}

/* A v9 scale field holding the canonical form of x (x must be canonical). */
static void writeCanonScaleLE(FILE *f, float x) {
    odtScale_t s = odtTestScale(x);
    writeScaleLE(f, s.m, s.e);
}

static tensor_t *makeFloatTensor2D(size_t d0, size_t d1, const float *src, size_t count) {
    /* Heap-tier construction per CONVENTIONS Rule 1. */
    size_t *dims = reserveMemory(2 * sizeof(size_t));
    dims[0] = d0;
    dims[1] = d1;
    size_t *order = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 2, order);

    tensor_t *t = initTensor(shape, quantizationInitFloat(), NULL);
    if (src != NULL) {
        tensorFillFromFloatBuffer(t, src, count);
    }
    return t;
}

void testSerializeAndDeserializeTensor() {
    size_t numberOfValues = 6;
    float data[] = {9, 9, 9, 4.5f, 2.1112f, 999.123f};

    tensor_t *serialTensor = makeFloatTensor2D(2, 3, data, numberOfValues);

    FILE *f = fopen(FILE_PATH, "wb");
    serializeTensor(serialTensor, f);
    fclose(f);

    /* Heap-allocated zero-init buffer destination via initTensor. */
    tensor_t *deserialTensor = makeFloatTensor2D(2, 3, NULL, 0);

    f = fopen(FILE_PATH, "rb");
    deserializeTensor(deserialTensor, f);
    fclose(f);

    /* CAPTURE every assertion value before any free. */
    float capturedDeserialData[6];
    for (size_t i = 0; i < numberOfValues; i++) {
        capturedDeserialData[i] = ((float *)deserialTensor->data)[i];
    }
    qtype_t capturedSerialQType = serialTensor->quantization->type;
    qtype_t capturedDeserialQType = deserialTensor->quantization->type;
    size_t capturedSerialNumDims = serialTensor->shape->numberOfDimensions;
    size_t capturedDeserialNumDims = deserialTensor->shape->numberOfDimensions;

    size_t capturedSerialDims[2];
    size_t capturedDeserialDims[2];
    size_t capturedSerialOrder[2];
    size_t capturedDeserialOrder[2];
    for (size_t i = 0; i < 2; i++) {
        capturedSerialDims[i] = serialTensor->shape->dimensions[i];
        capturedDeserialDims[i] = deserialTensor->shape->dimensions[i];
        capturedSerialOrder[i] = serialTensor->shape->orderOfDimensions[i];
        capturedDeserialOrder[i] = deserialTensor->shape->orderOfDimensions[i];
    }

    /* FREE in reverse-init order. */
    freeTensor(deserialTensor);
    freeTensor(serialTensor);

    /* ASSERT on captured. */
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(data, capturedDeserialData, numberOfValues);
    TEST_ASSERT_EQUAL(capturedSerialQType, capturedDeserialQType);
    TEST_ASSERT_EQUAL(capturedSerialNumDims, capturedDeserialNumDims);
    ODT_ASSERT_EQUAL_size_t_ARRAY(capturedSerialDims, capturedDeserialDims, 2);
    ODT_ASSERT_EQUAL_size_t_ARRAY(capturedSerialOrder, capturedDeserialOrder, 2);
}

/* Hand-crafted malformed files exercising deserializeModel's NEW validation
 * (bad magic / wrong version / layerCount mismatch / tag mismatch). A single
 * Flatten layer is the minimal pre-built mirror model — Flatten needs no
 * quantization setup, so these tests isolate the header/tag validation from
 * any per-layer-type record decoding. */

static void testDeserializeRejectsBadMagic(void) {
    FILE *f = fopen(FILE_PATH, "wb");
    fwrite("XXXX", 1, 4, f);
    writeU32LE(f, ODTS_VERSION); /* version (dead value: bad magic short-circuits first) */
    writeU32LE(f, 1);            /* layerCount */
    writeU8(f, 0);               /* wiring: chain */
    uint8_t tag = (uint8_t)FLATTEN;
    fwrite(&tag, sizeof(uint8_t), 1, f);
    fclose(f);

    layer_t *layer = flattenLayerInit();
    layer_t *model[] = {layer};

    f = fopen(FILE_PATH, "rb");
    ASSERT_EXITS_WITH_OUTPUT(1, "deserializeModel: bad magic bytes",
                             deserializeModel(&MODEL_CHAIN(model, 1), f));
    fclose(f);

    freeFlattenLayer(layer);
}

/* Only the current version is accepted: an older layout would misparse, a
 * newer one is unknown to this build. */
static void testDeserializeRejectsUnsupportedVersions(void) {
    const uint32_t versions[] = {ODTS_VERSION - 1u, ODTS_VERSION + 1u};
    layer_t *layer = flattenLayerInit();
    layer_t *model[] = {layer};
    for (size_t v = 0; v < sizeof(versions) / sizeof(versions[0]); v++) {
        FILE *f = fopen(FILE_PATH, "wb");
        fwrite("ODTS", 1, 4, f);
        writeU32LE(f, versions[v]);
        writeU32LE(f, 1); /* layerCount */
        fputc(FLATTEN, f);
        fclose(f);

        f = fopen(FILE_PATH, "rb");
        ASSERT_EXITS_WITH_OUTPUT(1, "unsupported format version",
                                 deserializeModel(&MODEL_CHAIN(model, 1), f));
        fclose(f);
    }
    freeFlattenLayer(layer);
}

/*! v7 = the pre-wiring format: no wiring-kind byte after the layer count, so
 *  a v7 file would read its first tag as the wiring kind. */
static void testDeserializeRejectsV7Version(void) {
    FILE *f = fopen(FILE_PATH, "wb");
    fwrite("ODTS", 1, 4, f);
    writeU32LE(f, 7); /* v7: no wiring byte */
    writeU32LE(f, 1); /* layerCount */
    writeU8(f, (uint8_t)FLATTEN);
    fclose(f);

    layer_t *layer = flattenLayerInit();
    layer_t *model[] = {layer};

    f = fopen(FILE_PATH, "rb");
    ASSERT_EXITS_WITH_OUTPUT(1, "unsupported format version 7 (expected 9)",
                             deserializeModel(&MODEL_CHAIN(model, 1), f));
    fclose(f);

    freeFlattenLayer(layer);
}

/*! v8 = the wiring format whose SYM / SYM_INT32 / ASYM records still carry
 *  `f32` scales where v9 (linq S4) carries `u16 m, i16 e`. Same 4 bytes, so a
 *  v8 record would misparse rather than truncate. No v8 compatibility shim. */
static void testDeserializeRejectsV8Version(void) {
    FILE *f = fopen(FILE_PATH, "wb");
    fwrite("ODTS", 1, 4, f);
    writeU32LE(f, 8); /* v8: f32 scales */
    writeU32LE(f, 1); /* layerCount */
    writeU8(f, 0);    /* wiring: chain */
    writeU8(f, (uint8_t)FLATTEN);
    fclose(f);

    layer_t *layer = flattenLayerInit();
    layer_t *model[] = {layer};

    f = fopen(FILE_PATH, "rb");
    ASSERT_EXITS_WITH_OUTPUT(1, "unsupported format version 8 (expected 9)",
                             deserializeModel(&MODEL_CHAIN(model, 1), f));
    fclose(f);

    freeFlattenLayer(layer);
}

/*! A file wired as a graph cannot load into a chain skeleton: the layers would
 *  read the wrong buffers. */
static void testDeserializeRejectsWiringKindMismatch(void) {
    FILE *f = fopen(FILE_PATH, "wb");
    fwrite("ODTS", 1, 4, f);
    writeU32LE(f, ODTS_VERSION);
    writeU32LE(f, 1); /* layerCount */
    writeU8(f, 1);    /* wiring: CSR */
    writeU8(f, (uint8_t)FLATTEN);
    fclose(f);

    layer_t *layer = flattenLayerInit();
    layer_t *model[] = {layer};

    f = fopen(FILE_PATH, "rb");
    ASSERT_EXITS_WITH_OUTPUT(1, "deserializeModel: wiring mismatch on 'kind' (file 1, skeleton 0)",
                             deserializeModel(&MODEL_CHAIN(model, 1), f));
    fclose(f);

    freeFlattenLayer(layer);
}

static void testDeserializeRejectsLayerCountMismatch(void) {
    FILE *f = fopen(FILE_PATH, "wb");
    fwrite("ODTS", 1, 4, f);
    writeU32LE(f, ODTS_VERSION); /* version */
    writeU32LE(f, 2);            /* layerCount; caller below passes sizeModel = 1 */
    writeU8(f, 0);               /* wiring: chain */
    uint8_t tag = (uint8_t)FLATTEN;
    fwrite(&tag, sizeof(uint8_t), 1, f);
    fclose(f);

    layer_t *layer = flattenLayerInit();
    layer_t *model[] = {layer};

    f = fopen(FILE_PATH, "rb");
    ASSERT_EXITS_WITH_OUTPUT(1,
                             "deserializeModel: layerCount mismatch (file has 2, caller expects 1)",
                             deserializeModel(&MODEL_CHAIN(model, 1), f));
    fclose(f);

    freeFlattenLayer(layer);
}

static void testDeserializeRejectsTagMismatch(void) {
    FILE *f = fopen(FILE_PATH, "wb");
    fwrite("ODTS", 1, 4, f);
    writeU32LE(f, ODTS_VERSION);   /* version */
    writeU32LE(f, 1);              /* layerCount */
    writeU8(f, 0);                 /* wiring: chain */
    uint8_t tag = (uint8_t)LINEAR; /* pre-built mirror layer below is FLATTEN */
    fwrite(&tag, sizeof(uint8_t), 1, f);
    fclose(f);

    layer_t *layer = flattenLayerInit();
    layer_t *model[] = {layer};

    f = fopen(FILE_PATH, "rb");
    ASSERT_EXITS_WITH_OUTPUT(
        1, "deserializeModel: record tag 0 does not match expected layer type 7 at index 0",
        deserializeModel(&MODEL_CHAIN(model, 1), f));
    fclose(f);

    freeFlattenLayer(layer);
}

static tensor_t *makeSymInt32Tensor2D(size_t d0, size_t d1) {
    size_t *dims = reserveMemory(2 * sizeof(size_t));
    dims[0] = d0;
    dims[1] = d1;
    size_t *order = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 2, order);
    return initTensor(shape, quantizationInitSymInt32(HALF_AWAY), NULL);
}

/* #316: a checkpoint whose per-tensor dtype differs from the pre-built skeleton
 * must be rejected before deserializeQConfig writes qConfig fields through a
 * mismatched (or NULL, for FLOAT32/INT32/BOOL) pointer, and before the payload
 * fread can overflow the skeleton's allocation. Here a FLOAT32 record is loaded
 * into a SYM_INT32-built skeleton — pre-fix it silently overwrites the dtype;
 * the reverse (SYM into a FLOAT32 skeleton) NULL-derefs. */
void testDeserializeTensorRejectsDtypeMismatch(void) {
    float data[] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    tensor_t *floatTensor = makeFloatTensor2D(2, 3, data, 6);
    FILE *f = fopen(FILE_PATH, "wb");
    serializeTensor(floatTensor, f);
    fclose(f);

    tensor_t *symSkeleton = makeSymInt32Tensor2D(2, 3);
    f = fopen(FILE_PATH, "rb");
    ASSERT_EXITS_WITH_FAILURE(deserializeTensor(symSkeleton, f));
    fclose(f);

    freeTensor(symSkeleton);
    freeTensor(floatTensor);
}

/* #316: a same-dtype record whose element count differs from the skeleton would
 * fread past tensor->data, which initTensor sized from the build-time shape. The
 * payload-size check catches size changes (shape- or packed-qBits-driven) that
 * the dtype check alone misses. */
void testDeserializeTensorRejectsPayloadSizeMismatch(void) {
    float data[] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    tensor_t *bigTensor = makeFloatTensor2D(2, 3, data, 6);
    FILE *f = fopen(FILE_PATH, "wb");
    serializeTensor(bigTensor, f);
    fclose(f);

    /* Same dtype (FLOAT32) and rank, but a smaller allocation (2x2 = 4 elems). */
    tensor_t *smallSkeleton = makeFloatTensor2D(2, 2, NULL, 0);
    f = fopen(FILE_PATH, "rb");
    ASSERT_EXITS_WITH_FAILURE(deserializeTensor(smallSkeleton, f));
    fclose(f);

    freeTensor(smallSkeleton);
    freeTensor(bigTensor);
}

/* #370: the issue's named dtype-mismatch direction — a SYM record loaded into a
 * FLOAT32-built skeleton (whose qConfig is NULL) must fail fast; pre-#316 this
 * NULL-derefed in deserializeQConfig. */
static void testDeserializeTensorRejectsSymRecordIntoFloatSkeleton(void) {
    size_t *dims = reserveMemory(2 * sizeof(size_t));
    dims[0] = 2;
    dims[1] = 3;
    size_t *order = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 2, order);
    tensor_t *symTensor = initTensor(shape, quantizationInitSym(4, HALF_AWAY), NULL);

    FILE *f = fopen(FILE_PATH, "wb");
    serializeTensor(symTensor, f);
    fclose(f);

    tensor_t *floatSkeleton = makeFloatTensor2D(2, 3, NULL, 0);
    f = fopen(FILE_PATH, "rb");
    ASSERT_EXITS_WITH_FAILURE(deserializeTensor(floatSkeleton, f));
    fclose(f);

    freeTensor(floatSkeleton);
    freeTensor(symTensor);
}

#if SIZE_MAX == UINT64_MAX
/* #394: dims whose product (2^64 + 64) wraps to the skeleton's 64 elements
 * would match the payload-size check and be accepted with a corrupt shape. */
static void testDeserializeTensorRejectsWrappingDims(void) {
    size_t *dims = reserveMemory(3 * sizeof(size_t));
    dims[0] = 4;
    dims[1] = 4;
    dims[2] = 4;
    size_t *order = reserveMemory(3 * sizeof(size_t));
    setOrderOfDimsForNewTensor(3, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 3, order);
    tensor_t *skeleton = initTensor(shape, quantizationInitFloat(), NULL);

    FILE *f = fopen(FILE_PATH, "wb");
    writeU32LE(f, 3u);
    writeU32LE(f, 320u);
    writeU32LE(f, 107367629u);
    writeU32LE(f, 536903681u);
    for (uint32_t d = 0; d < 3u; d++) {
        writeU32LE(f, d);
    }
    uint8_t dtype = (uint8_t)FLOAT32;
    fwrite(&dtype, 1, 1, f);
    uint8_t payload[256] = {0};
    fwrite(payload, 1, sizeof(payload), f);
    fclose(f);

    f = fopen(FILE_PATH, "rb");
    ASSERT_EXITS_WITH_OUTPUT(1, "element count overflows", deserializeTensor(skeleton, f));
    fclose(f);

    freeTensor(skeleton);
}
#endif

/* #370: a header cut mid-field (magic + half the version u32) must fail fast —
 * pre-v2 the unchecked fread left the version uninitialized/garbage. */
static void testDeserializeModelFailsFastOnTruncatedHeader(void) {
    FILE *f = fopen(FILE_PATH, "wb");
    fwrite("ODTS", 1, 4, f);
    uint8_t partialVersion[2] = {0x02, 0x00};
    fwrite(partialVersion, 1, 2, f);
    fclose(f);

    layer_t *layer = flattenLayerInit();
    layer_t *model[] = {layer};
    f = fopen(FILE_PATH, "rb");
    ASSERT_EXITS_WITH_OUTPUT(1, "short read (4 bytes)",
                             deserializeModel(&MODEL_CHAIN(model, 1), f));
    fclose(f);

    freeFlattenLayer(layer);
}

/* #370: a stream cut inside the DATA payload must fail fast at the payload
 * read — pre-v2 the unchecked fread deserialized the truncation as silent
 * garbage (the trailing elements simply kept their zero-init). */
static void testDeserializeTensorFailsFastOnTruncatedPayload(void) {
    float data[] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    tensor_t *src = makeFloatTensor2D(2, 3, data, 6);
    FILE *f = fopen(FILE_PATH, "wb");
    serializeTensor(src, f);
    long full = ftell(f);
    fclose(f);

    FILE *in = fopen(FILE_PATH, "rb");
    uint8_t *buf = reserveMemory((size_t)full);
    size_t got = fread(buf, 1, (size_t)full, in);
    fclose(in);
    f = fopen(FILE_PATH, "wb");
    fwrite(buf, 1, (size_t)full - 2, f);
    fclose(f);
    freeReservedMemory(buf);
    TEST_ASSERT_EQUAL_size_t((size_t)full, got);

    tensor_t *skeleton = makeFloatTensor2D(2, 3, NULL, 0);
    f = fopen(FILE_PATH, "rb");
    ASSERT_EXITS_WITH_FAILURE(deserializeTensor(skeleton, f));
    fclose(f);

    freeTensor(skeleton);
    freeTensor(src);
}

/* #370: file rank 1 x [6] into a rank-2 [6,1] skeleton keeps the element count
 * equal, so the #316 payload-size check alone cannot see it — pre-v2 the file
 * rank silently overwrote the skeleton's (and a LARGER file rank wrote dims
 * past the skeleton's arrays). The v2 rank guard must reject it. */
static void testDeserializeTensorRejectsRankMismatch(void) {
    float data[] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    size_t *dims = reserveMemory(1 * sizeof(size_t));
    dims[0] = 6;
    size_t *order = reserveMemory(1 * sizeof(size_t));
    setOrderOfDimsForNewTensor(1, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 1, order);
    tensor_t *src = initTensor(shape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(src, data, 6);

    FILE *f = fopen(FILE_PATH, "wb");
    serializeTensor(src, f);
    fclose(f);

    tensor_t *skeleton = makeFloatTensor2D(6, 1, NULL, 0);
    f = fopen(FILE_PATH, "rb");
    ASSERT_EXITS_WITH_FAILURE(deserializeTensor(skeleton, f));
    fclose(f);

    freeTensor(skeleton);
    freeTensor(src);
}

/* Well-formed FLOAT32 [2,3] record except for the two orderOfDimensions
 * entries, so only the permutation check can stop it. */
static void writeFloat2x3RecordWithOrder(uint32_t order0, uint32_t order1) {
    FILE *f = fopen(FILE_PATH, "wb");
    writeU32LE(f, 2);
    writeU32LE(f, 2);
    writeU32LE(f, 3);
    writeU32LE(f, order0);
    writeU32LE(f, order1);
    fputc(FLOAT32, f);
    for (int i = 0; i < 6; i++) {
        writeF32LE(f, (float)i);
    }
    fclose(f);
}

/* orderOfDimensions indexes dimensions[]: an entry >= rank reads past it, and
 * a repeated entry leaves a dimension unreachable, so the strides it implies
 * no longer cover the payload. */
static void testDeserializeTensorRejectsOutOfRangeOrderOfDimensions(void) {
    tensor_t *skeleton = makeFloatTensor2D(2, 3, NULL, 0);

    writeFloat2x3RecordWithOrder(0, 2);
    FILE *f = fopen(FILE_PATH, "rb");
    ASSERT_EXITS_WITH_OUTPUT(1, "deserializeShape: orderOfDimensions[1] = 2 is out of range",
                             deserializeTensor(skeleton, f));
    fclose(f);

    writeFloat2x3RecordWithOrder(0xFFFFFFFFu, 1);
    f = fopen(FILE_PATH, "rb");
    ASSERT_EXITS_WITH_OUTPUT(1,
                             "deserializeShape: orderOfDimensions[0] = 4294967295 is out of range",
                             deserializeTensor(skeleton, f));
    fclose(f);

    freeTensor(skeleton);
}

static void testDeserializeTensorRejectsDuplicateOrderOfDimensions(void) {
    tensor_t *skeleton = makeFloatTensor2D(2, 3, NULL, 0);

    writeFloat2x3RecordWithOrder(1, 1);
    FILE *f = fopen(FILE_PATH, "rb");
    ASSERT_EXITS_WITH_OUTPUT(1,
                             "deserializeShape: orderOfDimensions[1] = 1 repeats an earlier entry",
                             deserializeTensor(skeleton, f));
    fclose(f);

    freeTensor(skeleton);
}

/* The permutation check must not reject a real transpose. */
static void testDeserializeTensorAcceptsTransposedOrderOfDimensions(void) {
    tensor_t *skeleton = makeFloatTensor2D(2, 3, NULL, 0);

    writeFloat2x3RecordWithOrder(1, 0);
    FILE *f = fopen(FILE_PATH, "rb");
    deserializeTensor(skeleton, f);
    fclose(f);

    TEST_ASSERT_EQUAL_size_t(1, skeleton->shape->orderOfDimensions[0]);
    TEST_ASSERT_EQUAL_size_t(0, skeleton->shape->orderOfDimensions[1]);
    freeTensor(skeleton);
}

/* A grad record a frozen skeleton skips gets the same permutation check; the
 * record is otherwise complete, so without it the skip succeeds. */
static void testSkipSerializedTensorRejectsDuplicateOrderOfDimensions(void) {
    float paramData[] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    tensor_t *paramTensor = makeFloatTensor2D(2, 3, paramData, 6);
    FILE *f = fopen(FILE_PATH, "wb");
    fputc(1, f); /* hasGrad */
    serializeTensor(paramTensor, f);
    writeU32LE(f, 2);
    writeU32LE(f, 2);
    writeU32LE(f, 3);
    writeU32LE(f, 1);
    writeU32LE(f, 1);
    fputc(FLOAT32, f);
    for (int i = 0; i < 6; i++) {
        writeF32LE(f, (float)i);
    }
    fclose(f);

    tensor_t *skeletonParam = makeFloatTensor2D(2, 3, NULL, 0);
    parameter_t frozenSkeleton = {.param = skeletonParam, .grad = NULL};
    f = fopen(FILE_PATH, "rb");
    ASSERT_EXITS_WITH_OUTPUT(
        1, "readTensorRecordHeader: orderOfDimensions[1] = 1 repeats an earlier entry",
        deserializeParameter(&frozenSkeleton, f));
    fclose(f);

    freeTensor(skeletonParam);
    freeTensor(paramTensor);
}

/* The writer emits hasGrad 0 or 1; any other byte is a corrupt record, even
 * when every following tensor record is well-formed. */
static void testDeserializeParameterRejectsHasGradAboveOne(void) {
    float data[] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    tensor_t *paramTensor = makeFloatTensor2D(2, 3, data, 6);
    FILE *f = fopen(FILE_PATH, "wb");
    fputc(2, f); /* hasGrad */
    serializeTensor(paramTensor, f);
    serializeTensor(paramTensor, f);
    fclose(f);

    tensor_t *skeletonParam = makeFloatTensor2D(2, 3, NULL, 0);
    tensor_t *skeletonGrad = makeFloatTensor2D(2, 3, NULL, 0);
    parameter_t trainableSkeleton = {.param = skeletonParam, .grad = skeletonGrad};
    f = fopen(FILE_PATH, "rb");
    ASSERT_EXITS_WITH_OUTPUT(1, "deserializeParameter: hasGrad byte 2 is neither 0 nor 1",
                             deserializeParameter(&trainableSkeleton, f));
    fclose(f);

    freeTensor(skeletonGrad);
    freeTensor(skeletonParam);
    freeTensor(paramTensor);
}

static tensor_t *makeAsymTensor1D(size_t d0) {
    size_t *dims = reserveMemory(1 * sizeof(size_t));
    dims[0] = d0;
    size_t *order = reserveMemory(1 * sizeof(size_t));
    setOrderOfDimsForNewTensor(1, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 1, order);
    return initTensor(shape, quantizationInitAsym(8, HALF_AWAY), NULL);
}

/* #370, re-pinned for group-quant PR4 (group-quant D6): the zeroPoint slot is still i32
 * LE on the wire but now carries the code-domain uint16 zp; 40000 (> INT16_MAX,
 * < 2^16) is the widest-band class that must round-trip losslessly. The old
 * -72817 value-domain pin has no uint16 representation -- it re-derives as the
 * 65535 ceiling under the nudge (see UnitTestTensorConversion's
 * f2NegBand16 pin). */
static void testDeserializeTensorRoundTripsAsymZeroPoint(void) {
    tensor_t *src = makeAsymTensor1D(4);
    asymQConfig_t *srcQc = src->quantization->qConfig;
    srcQc->scales[0] = odtTestScale(0.5f);
    srcQc->zeroPoints[0] = 40000;

    FILE *f = fopen(FILE_PATH, "wb");
    serializeTensor(src, f);
    fclose(f);

    tensor_t *dst = makeAsymTensor1D(4);
    f = fopen(FILE_PATH, "rb");
    deserializeTensor(dst, f);
    fclose(f);

    asymQConfig_t *dstQc = dst->quantization->qConfig;
    float capturedScale = odtScaleToFloat(dstQc->scales[0]);
    uint16_t capturedZeroPoint = dstQc->zeroPoints[0];
    freeTensor(dst);
    freeTensor(src);

    TEST_ASSERT_EQUAL_FLOAT(0.5f, capturedScale);
    TEST_ASSERT_EQUAL_UINT16(40000, capturedZeroPoint);
}

/* #370/PR4, re-pinned for v5 (group-quant PR4, Task 4): hand-crafted record
 * pins the u16 LE zeroPoints[] wire slot -- a value near the top of the
 * uint16 range (40000, > INT16_MAX) must round-trip losslessly through the
 * now-native u16 field. Pre-v5 this test exercised the i32-slot-to-uint16
 * clamp (clampInt32); v5's wire field IS u16, so there is no clamp left to
 * exercise -- this test now simply pins that a wide-but-valid u16 value
 * survives the read intact. */
static void testDeserializeQConfigAcceptsWideZeroPoint(void) {
    FILE *f = fopen(FILE_PATH, "wb");
    writeU32LE(f, 1); /* numberOfDimensions */
    writeU32LE(f, 4); /* dimensions[0] */
    writeU32LE(f, 0); /* orderOfDimensions[0] */
    uint8_t asymType = (uint8_t)ASYM;
    fwrite(&asymType, 1, 1, f);
    writeU32LE(f, 1);           /* numGroups */
    writeU32LE(f, 0);           /* groupSize */
    writeCanonScaleLE(f, 0.5f); /* scales[0] */
    writeU16LE(f, 40000u);      /* zeroPoints[0] u16 LE, > INT16_MAX */
    uint8_t qBits = 8;
    fwrite(&qBits, 1, 1, f);
    uint8_t roundingMode = 0; /* HALF_AWAY */
    fwrite(&roundingMode, 1, 1, f);
    uint8_t payload[4] = {0, 0, 0, 0};
    fwrite(payload, 1, 4, f);
    fclose(f);

    tensor_t *skeleton = makeAsymTensor1D(4);
    f = fopen(FILE_PATH, "rb");
    deserializeTensor(skeleton, f);
    fclose(f);

    asymQConfig_t *skelQc = skeleton->quantization->qConfig;
    uint16_t capturedZeroPoint = skelQc->zeroPoints[0];
    freeTensor(skeleton);

    TEST_ASSERT_EQUAL_UINT16(40000, capturedZeroPoint);
}

static tensor_t *makeSymTensor1D(size_t d0, uint8_t qBits, roundingMode_t roundingMode) {
    size_t *dims = reserveMemory(1 * sizeof(size_t));
    dims[0] = d0;
    size_t *order = reserveMemory(1 * sizeof(size_t));
    setOrderOfDimsForNewTensor(1, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 1, order);
    return initTensor(shape, quantizationInitSym(qBits, roundingMode), NULL);
}

/*! Grouped-SYM twin of makeSymTensor1D; numGroups * groupSize must equal d0. */
static tensor_t *makeSymGroupedTensor1D(size_t d0, uint8_t qBits, roundingMode_t roundingMode,
                                        size_t numGroups, size_t groupSize) {
    size_t *dims = reserveMemory(1 * sizeof(size_t));
    dims[0] = d0;
    size_t *order = reserveMemory(1 * sizeof(size_t));
    setOrderOfDimsForNewTensor(1, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 1, order);
    return initTensor(shape, quantizationInitSymGrouped(qBits, roundingMode, numGroups, groupSize),
                      NULL);
}

/*! A grouped file record loads into a per-tensor skeleton: the scales array
 *  is reallocated to the file's group count, and every scale and payload byte
 *  arrives. */
static void testDeserializeQConfigAcceptsNumGroupsMismatchViaReallocation(void) {
    FILE *f = fopen(FILE_PATH, "wb");
    writeU32LE(f, 1); /* numberOfDimensions */
    writeU32LE(f, 4); /* dimensions[0] */
    writeU32LE(f, 0); /* orderOfDimensions[0] */
    uint8_t symType = (uint8_t)SYM;
    fwrite(&symType, 1, 1, f);
    writeU32LE(f, 2); /* numGroups = 2; the skeleton is per-tensor */
    writeU32LE(f, 2); /* groupSize = 2; 2*2 == 4 == the skeleton's element count */
    writeCanonScaleLE(f, 0.5f);
    writeCanonScaleLE(f, 0.25f);
    uint8_t qBits = 4;
    fwrite(&qBits, 1, 1, f);
    uint8_t roundingMode = 0; /* HALF_AWAY */
    fwrite(&roundingMode, 1, 1, f);
    uint8_t payload[2] = {0xAB, 0xCD}; /* qBits=4, 4 elems -> ceil(16/8) = 2 packed bytes */
    fwrite(payload, 1, 2, f);
    fclose(f);

    tensor_t *skeleton = makeSymTensor1D(4, 4, HALF_AWAY);
    f = fopen(FILE_PATH, "rb");
    deserializeTensor(skeleton, f);
    fclose(f);

    symQConfig_t *dstQc = skeleton->quantization->qConfig;
    /* CAPTURE before any free. */
    size_t capturedNumGroups = dstQc->numGroups;
    size_t capturedGroupSize = dstQc->groupSize;
    float capturedScale0 = odtScaleToFloat(dstQc->scales[0]);
    float capturedScale1 = odtScaleToFloat(dstQc->scales[1]);
    uint8_t capturedPayload[2];
    memcpy(capturedPayload, skeleton->data, 2);

    freeTensor(skeleton);

    /* ASSERT on captured. */
    TEST_ASSERT_EQUAL_size_t(2, capturedNumGroups);
    TEST_ASSERT_EQUAL_size_t(2, capturedGroupSize);
    TEST_ASSERT_EQUAL_FLOAT(0.5f, capturedScale0);
    TEST_ASSERT_EQUAL_FLOAT(0.25f, capturedScale1);
    TEST_ASSERT_EQUAL_UINT8(0xAB, capturedPayload[0]);
    TEST_ASSERT_EQUAL_UINT8(0xCD, capturedPayload[1]);
}

/*! A well-formed grouped SYM shape (3 x 2) whose product differs from the
 *  record's element count (4) dies at the exact-product check. */
static void testDeserializeGroupedSymRejectsBadDivisibility(void) {
    FILE *f = fopen(FILE_PATH, "wb");
    writeU32LE(f, 1); /* numberOfDimensions */
    writeU32LE(f, 4); /* dimensions[0] = 4 elements */
    writeU32LE(f, 0); /* orderOfDimensions[0] */
    uint8_t symType = (uint8_t)SYM;
    fwrite(&symType, 1, 1, f);
    writeU32LE(f, 3); /* numGroups = 3 */
    writeU32LE(f, 2); /* groupSize = 2 -> 3*2 == 6 != 4 elements */
    writeCanonScaleLE(f, 0.5f);
    writeCanonScaleLE(f, 0.5f);
    writeCanonScaleLE(f, 0.5f); /* three scales */
    uint8_t qBits = 4;
    fwrite(&qBits, 1, 1, f);
    uint8_t roundingMode = 0; /* HALF_AWAY */
    fwrite(&roundingMode, 1, 1, f);
    uint8_t payload[2] = {0xAB, 0xCD}; /* never reached: the validate dies first */
    fwrite(payload, 1, 2, f);
    fclose(f);

    tensor_t *skeleton = makeSymTensor1D(4, 4, HALF_AWAY);
    f = fopen(FILE_PATH, "rb");
    ASSERT_EXITS_WITH_FAILURE(deserializeTensor(skeleton, f));
    fclose(f);

    freeTensor(skeleton);
}

/*! group-quant PR1: numGroups==1 but groupSize!=0 violates the sentinel
 *  invariant documented in Quantization.h (numGroups==1 <=> groupSize==0).
 *  Checked on the FILE values, independent of the numGroups-mismatch
 *  relax above -- a file this corrupt (or written by a future format this
 *  build cannot fully interpret) must fail fast rather than silently
 *  accepting a nonsensical per-tensor "group size". STAYS a death test
 *  post-PR2 (Task 5): the sentinel check is untouched by the reallocation
 *  relax.
 *
 *  Well-formed-record rationale and the groupSize=3 choice: mirrors
 *  testSkipSerializedTensorRejectsRankAboveCap's rationale above -- an
 *  otherwise valid record isolates THIS guard from the unrelated #316
 *  payload-size check a stale parser's misaligned read would otherwise
 *  trip. */
static void testDeserializeQConfigRejectsSentinelViolation(void) {
    FILE *f = fopen(FILE_PATH, "wb");
    writeU32LE(f, 1); /* numberOfDimensions */
    writeU32LE(f, 4); /* dimensions[0] */
    writeU32LE(f, 0); /* orderOfDimensions[0] */
    uint8_t symType = (uint8_t)SYM;
    fwrite(&symType, 1, 1, f);
    writeU32LE(f, 1); /* numGroups = 1, matches the skeleton */
    writeU32LE(f, 3); /* groupSize = 3, violates numGroups==1 <=> groupSize==0 */
    writeCanonScaleLE(f, 0.5f);
    uint8_t qBits = 4;
    fwrite(&qBits, 1, 1, f);
    uint8_t roundingMode = 0; /* HALF_AWAY */
    fwrite(&roundingMode, 1, 1, f);
    uint8_t payload[2] = {0xAB, 0xCD}; /* qBits=4, 4 elems -> ceil(16/8) = 2 packed bytes */
    fwrite(payload, 1, 2, f);
    fclose(f);

    tensor_t *skeleton = makeSymTensor1D(4, 4, HALF_AWAY);
    f = fopen(FILE_PATH, "rb");
    ASSERT_EXITS_WITH_FAILURE(deserializeTensor(skeleton, f));
    fclose(f);

    freeTensor(skeleton);
}

/*! BFP epic PR1 (Task 7): per-tensor BFP tensor builder, mirroring
 *  makeSymTensor1D above. */
static tensor_t *makeBfpTensor1D(size_t d0, uint8_t mantissaBits, uint8_t exponentBits,
                                 roundingMode_t roundingMode) {
    size_t *dims = reserveMemory(1 * sizeof(size_t));
    dims[0] = d0;
    size_t *order = reserveMemory(1 * sizeof(size_t));
    setOrderOfDimsForNewTensor(1, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 1, order);
    return initTensor(shape, quantizationInitBfp(mantissaBits, exponentBits, roundingMode), NULL);
}

/*! Grouped-BFP twin of makeBfpTensor1D; numGroups * groupSize must equal d0. */
static tensor_t *makeBfpGroupedTensor1D(size_t d0, uint8_t mantissaBits, uint8_t exponentBits,
                                        roundingMode_t roundingMode, size_t numGroups,
                                        size_t groupSize) {
    size_t *dims = reserveMemory(1 * sizeof(size_t));
    dims[0] = d0;
    size_t *order = reserveMemory(1 * sizeof(size_t));
    setOrderOfDimsForNewTensor(1, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 1, order);
    return initTensor(
        shape,
        quantizationInitBfpGrouped(mantissaBits, exponentBits, roundingMode, numGroups, groupSize),
        NULL);
}

/*! Grouped-BFP round trip into a same-shape skeleton: every qConfig field and
 *  the packed payload (raw codes via byteConversion) arrive intact. */
static void testBfpRoundTripGrouped(void) {
    tensor_t *src = makeBfpGroupedTensor1D(6, 4, 8, HALF_AWAY, 3, 2);
    bfpQConfig_t *srcQc = src->quantization->qConfig;
    srcQc->exponents[0] = 120;
    srcQc->exponents[1] = 130;
    srcQc->exponents[2] = 140;
    int32_t mantissas[] = {1, -2, 3, -4, 5, -6};
    byteConversion((uint8_t *)mantissas, 32, src->data, 4, 6);

    FILE *f = fopen(FILE_PATH, "wb");
    serializeTensor(src, f);
    fclose(f);

    tensor_t *dst = makeBfpGroupedTensor1D(6, 4, 8, HALF_AWAY, 3, 2);

    f = fopen(FILE_PATH, "rb");
    deserializeTensor(dst, f);
    fclose(f);

    bfpQConfig_t *dstQc = dst->quantization->qConfig;
    /* CAPTURE before any free. */
    size_t capturedNumGroups = dstQc->numGroups;
    size_t capturedGroupSize = dstQc->groupSize;
    uint8_t capturedExponents[3];
    for (size_t g = 0; g < 3; g++) {
        capturedExponents[g] = dstQc->exponents[g];
    }
    uint8_t capturedMantissaBits = dstQc->mantissaBits;
    uint8_t capturedExponentBits = dstQc->exponentBits;
    roundingMode_t capturedRoundingMode = dstQc->roundingMode;
    size_t dataBytes = calcNumberOfBytesForData(dst->quantization, 6);
    uint8_t capturedSrcData[3];
    uint8_t capturedDstData[3];
    memcpy(capturedSrcData, src->data, dataBytes);
    memcpy(capturedDstData, dst->data, dataBytes);

    /* FREE in reverse-init order. */
    freeTensor(dst);
    freeTensor(src);

    /* ASSERT on captured. */
    TEST_ASSERT_EQUAL_size_t(3, capturedNumGroups);
    TEST_ASSERT_EQUAL_size_t(2, capturedGroupSize);
    TEST_ASSERT_EQUAL_UINT8(120, capturedExponents[0]);
    TEST_ASSERT_EQUAL_UINT8(130, capturedExponents[1]);
    TEST_ASSERT_EQUAL_UINT8(140, capturedExponents[2]);
    TEST_ASSERT_EQUAL_UINT8(4, capturedMantissaBits);
    TEST_ASSERT_EQUAL_UINT8(8, capturedExponentBits);
    TEST_ASSERT_EQUAL_INT(HALF_AWAY, capturedRoundingMode);
    TEST_ASSERT_EQUAL_HEX8_ARRAY(capturedSrcData, capturedDstData, dataBytes);
}

/*! BFP sibling of
 *  testDeserializeQConfigAcceptsNumGroupsMismatchViaReallocation -- a GROUPED file record
 *  (numGroups=4, groupSize=2) deserialized into a freshly-built PER-TENSOR
 *  skeleton (numGroups=1, groupSize=0) must REALLOCATE the skeleton's
 *  exponents[] and load every group's exponent, and the packed tensor DATA
 *  must round-trip too. */
static void testBfpDeserializeReallocatesExponentsOnShapeChange(void) {
    tensor_t *src = makeBfpGroupedTensor1D(8, 4, 8, HALF_AWAY, 4, 2);
    bfpQConfig_t *srcQc = src->quantization->qConfig;
    srcQc->exponents[0] = 100;
    srcQc->exponents[1] = 110;
    srcQc->exponents[2] = 120;
    srcQc->exponents[3] = 130;
    int32_t mantissas[] = {1, -2, 3, -4, 5, -6, 7, -8};
    byteConversion((uint8_t *)mantissas, 32, src->data, 4, 8);

    FILE *f = fopen(FILE_PATH, "wb");
    serializeTensor(src, f);
    fclose(f);

    tensor_t *skeleton = makeBfpTensor1D(8, 4, 8, HALF_AWAY); /* per-tensor: {1,0} */

    f = fopen(FILE_PATH, "rb");
    deserializeTensor(skeleton, f);
    fclose(f);

    bfpQConfig_t *dstQc = skeleton->quantization->qConfig;
    /* CAPTURE before any free. */
    size_t capturedNumGroups = dstQc->numGroups;
    size_t capturedGroupSize = dstQc->groupSize;
    uint8_t capturedExponents[4];
    for (size_t g = 0; g < 4; g++) {
        capturedExponents[g] = dstQc->exponents[g];
    }
    size_t dataBytes = calcNumberOfBytesForData(skeleton->quantization, 8);
    uint8_t capturedSrcData[4];
    uint8_t capturedDstData[4];
    memcpy(capturedSrcData, src->data, dataBytes);
    memcpy(capturedDstData, skeleton->data, dataBytes);

    /* FREE in reverse-init order. */
    freeTensor(skeleton);
    freeTensor(src);

    /* ASSERT on captured. */
    TEST_ASSERT_EQUAL_size_t(4, capturedNumGroups);
    TEST_ASSERT_EQUAL_size_t(2, capturedGroupSize);
    TEST_ASSERT_EQUAL_UINT8(100, capturedExponents[0]);
    TEST_ASSERT_EQUAL_UINT8(110, capturedExponents[1]);
    TEST_ASSERT_EQUAL_UINT8(120, capturedExponents[2]);
    TEST_ASSERT_EQUAL_UINT8(130, capturedExponents[3]);
    TEST_ASSERT_EQUAL_HEX8_ARRAY(capturedSrcData, capturedDstData, dataBytes);
}

/*! fileNumGroups == 0 dies at the group cap. A layer wire config has no
 *  geometry, so no shape validation can catch it instead; groupSize is 1 so
 *  the sentinel check does not fire first either. */
static void testBfpDeserializeRejectsZeroNumGroupsInWireConfig(void) {
    FILE *f = fopen(FILE_PATH, "wb");
    fwrite("ODTS", 1, 4, f);
    writeU32LE(f, ODTS_VERSION); /* version */
    writeU32LE(f, 1);            /* layerCount */
    writeU8(f, 0);               /* wiring: chain */
    uint8_t tag = (uint8_t)RELU;
    fwrite(&tag, 1, 1, f);
    uint8_t arithByte = 0; /* ARITH_FLOAT32, HALF_AWAY -- forwardMath */
    fwrite(&arithByte, 1, 1, f);
    fwrite(&arithByte, 1, 1, f);
    fwrite(&arithByte, 1, 1, f); /* propLossMath */
    fwrite(&arithByte, 1, 1, f);
    uint8_t bfpType = (uint8_t)BFP;
    fwrite(&bfpType, 1, 1, f); /* outputQ dtype */
    writeU32LE(f, 0);          /* fileNumGroups = 0 */
    writeU32LE(f, 1);          /* groupSize: nonzero, satisfies the sentinel */
    uint8_t mantissaBits = 4;
    fwrite(&mantissaBits, 1, 1, f);
    uint8_t exponentBits = 8;
    fwrite(&exponentBits, 1, 1, f);
    uint8_t roundingMode = 0; /* HALF_AWAY */
    fwrite(&roundingMode, 1, 1, f);
    uint8_t propLossFloatTag = (uint8_t)FLOAT32; /* matches the skeleton's default propLossQ */
    fwrite(&propLossFloatTag, 1, 1, f);
    fclose(f);

    quantization_t *floatQ = quantizationInitFloat();
    quantization_t *bfpOutputQ = quantizationInitBfp(4, 8, HALF_AWAY);
    layerQuant_t lq;
    layerQuantInitUniform(&lq, floatQ);
    lq.outputQ = bfpOutputQ;
    layer_t *layer = reluLayerInit(&lq);
    layer_t *model[] = {layer};

    f = fopen(FILE_PATH, "rb");
    ASSERT_EXITS_WITH_OUTPUT(1, "deserializeQConfig: BFP file numGroups 0 is zero",
                             deserializeModel(&MODEL_CHAIN(model, 1), f));
    fclose(f);

    freeReluLayer(layer);
    freeQuantization(bfpOutputQ);
    freeQuantization(floatQ);
}

/*! BFP epic PR1 (Task 7): sibling of testDeserializeQConfigRejectsSentinelViolation
 *  -- numGroups==1 but groupSize!=0 violates the sentinel invariant
 *  (Quantization.h), checked on the FILE values. Well-formed-record
 *  rationale mirrors the SYM sibling: an otherwise valid record isolates
 *  THIS guard from the unrelated #316 payload-size check a stale parser's
 *  misaligned read would otherwise trip. */
static void testBfpDeserializeRejectsSentinelViolation(void) {
    uint8_t bfpType = (uint8_t)BFP;
    uint8_t mantissaBits = 4;
    uint8_t exponentBits = 8;
    uint8_t roundingMode = 0;    /* HALF_AWAY */
    uint8_t payload[2] = {0, 0}; /* mantissaBits=4, 4 elems -> ceil(16/8) = 2 packed bytes */

    FILE *f = fopen(FILE_PATH, "wb");
    writeU32LE(f, 1); /* numberOfDimensions */
    writeU32LE(f, 4); /* dimensions[0] */
    writeU32LE(f, 0); /* orderOfDimensions[0] */
    fwrite(&bfpType, 1, 1, f);
    writeU32LE(f, 1); /* numGroups = 1, matches the skeleton */
    writeU32LE(f, 3); /* groupSize = 3, violates numGroups==1 <=> groupSize==0 */
    uint8_t exponentByte = 130;
    fwrite(&exponentByte, 1, 1, f);
    fwrite(&mantissaBits, 1, 1, f);
    fwrite(&exponentBits, 1, 1, f);
    fwrite(&roundingMode, 1, 1, f);
    fwrite(payload, 1, 2, f);
    fclose(f);

    tensor_t *skeleton = makeBfpTensor1D(4, 4, 8, HALF_AWAY);
    f = fopen(FILE_PATH, "rb");
    ASSERT_EXITS_WITH_FAILURE(deserializeTensor(skeleton, f));
    fclose(f);

    freeTensor(skeleton);
}

/*! A BFP wire config with mantissaBits outside [2, 16] dies. Use a layer wire config so
 *  payload-size rejection cannot mask the width check. */
static void testBfpDeserializeRejectsMantissaBitsOutOfRange(void) {
    FILE *f = fopen(FILE_PATH, "wb");
    fwrite("ODTS", 1, 4, f);
    writeU32LE(f, ODTS_VERSION); /* version */
    writeU32LE(f, 1);            /* layerCount */
    writeU8(f, 0);               /* wiring: chain */
    uint8_t tag = (uint8_t)RELU;
    fwrite(&tag, 1, 1, f);
    uint8_t arithByte = 0; /* ARITH_FLOAT32, HALF_AWAY -- forwardMath */
    fwrite(&arithByte, 1, 1, f);
    fwrite(&arithByte, 1, 1, f);
    fwrite(&arithByte, 1, 1, f); /* propLossMath */
    fwrite(&arithByte, 1, 1, f);
    uint8_t bfpType = (uint8_t)BFP;
    fwrite(&bfpType, 1, 1, f); /* outputQ dtype */
    writeU32LE(f, 1);          /* fileNumGroups = 1, matches the skeleton */
    writeU32LE(f, 0);          /* groupSize = 0, matches the sentinel */
    uint8_t exponentByte = 127;
    fwrite(&exponentByte, 1, 1, f);
    uint8_t mantissaBitsTooNarrow = 1; /* < 2, initBfpQConfigGrouped's own cap */
    fwrite(&mantissaBitsTooNarrow, 1, 1, f);
    uint8_t exponentBits = 8;
    fwrite(&exponentBits, 1, 1, f);
    uint8_t roundingMode = 0; /* HALF_AWAY */
    fwrite(&roundingMode, 1, 1, f);
    uint8_t propLossFloatTag = (uint8_t)FLOAT32; /* matches the skeleton's default propLossQ */
    fwrite(&propLossFloatTag, 1, 1, f);
    fclose(f);

    quantization_t *floatQ = quantizationInitFloat();
    quantization_t *bfpOutputQ = quantizationInitBfp(4, 8, HALF_AWAY);
    layerQuant_t lq;
    layerQuantInitUniform(&lq, floatQ);
    lq.outputQ = bfpOutputQ;
    layer_t *layer = reluLayerInit(&lq);
    layer_t *model[] = {layer};

    f = fopen(FILE_PATH, "rb");
    ASSERT_EXITS_WITH_OUTPUT(1, "deserializeQConfig: BFP file mantissaBits (1) outside [2, 16]",
                             deserializeModel(&MODEL_CHAIN(model, 1), f));
    fclose(f);

    freeReluLayer(layer);
    freeQuantization(bfpOutputQ);
    freeQuantization(floatQ);
}

/*! A BFP wire config with exponentBits outside [2, 8] dies. Use a layer wire config so
 *  payload-size rejection cannot mask the width check. */
static void testBfpDeserializeRejectsExponentBitsOutOfRange(void) {
    FILE *f = fopen(FILE_PATH, "wb");
    fwrite("ODTS", 1, 4, f);
    writeU32LE(f, ODTS_VERSION); /* version */
    writeU32LE(f, 1);            /* layerCount */
    writeU8(f, 0);               /* wiring: chain */
    uint8_t tag = (uint8_t)RELU;
    fwrite(&tag, 1, 1, f);
    uint8_t arithByte = 0; /* ARITH_FLOAT32, HALF_AWAY -- forwardMath */
    fwrite(&arithByte, 1, 1, f);
    fwrite(&arithByte, 1, 1, f);
    fwrite(&arithByte, 1, 1, f); /* propLossMath */
    fwrite(&arithByte, 1, 1, f);
    uint8_t bfpType = (uint8_t)BFP;
    fwrite(&bfpType, 1, 1, f); /* outputQ dtype */
    writeU32LE(f, 1);          /* fileNumGroups = 1, matches the skeleton */
    writeU32LE(f, 0);          /* groupSize = 0, matches the sentinel */
    uint8_t exponentByte = 127;
    fwrite(&exponentByte, 1, 1, f);
    uint8_t mantissaBits = 4;
    fwrite(&mantissaBits, 1, 1, f);
    uint8_t exponentBitsTooWide = 9; /* > 8, initBfpQConfigGrouped's own cap */
    fwrite(&exponentBitsTooWide, 1, 1, f);
    uint8_t roundingMode = 0; /* HALF_AWAY */
    fwrite(&roundingMode, 1, 1, f);
    uint8_t propLossFloatTag = (uint8_t)FLOAT32; /* matches the skeleton's default propLossQ */
    fwrite(&propLossFloatTag, 1, 1, f);
    fclose(f);

    quantization_t *floatQ = quantizationInitFloat();
    quantization_t *bfpOutputQ = quantizationInitBfp(4, 8, HALF_AWAY);
    layerQuant_t lq;
    layerQuantInitUniform(&lq, floatQ);
    lq.outputQ = bfpOutputQ;
    layer_t *layer = reluLayerInit(&lq);
    layer_t *model[] = {layer};

    f = fopen(FILE_PATH, "rb");
    ASSERT_EXITS_WITH_OUTPUT(1, "deserializeQConfig: BFP file exponentBits (9) outside [2, 8]",
                             deserializeModel(&MODEL_CHAIN(model, 1), f));
    fclose(f);

    freeReluLayer(layer);
    freeQuantization(bfpOutputQ);
    freeQuantization(floatQ);
}

/*! #420 G5, third sibling of the two width-cap tests above (same bare
 *  wire-config isolation): a stored exponent byte above bias + 127 has NO
 *  finite float32 scale -- ldexpf(1.f, stored - bias) is +inf, so the first
 *  dequant turns every zero code into 0 * inf == NaN, which walks past every
 *  runtime guard instead of saturating like an out-of-range VALUE would.
 *  deriveBfpStoredExponent caps its own output at exactly bias + 127 (BFP D6,
 *  docs/conventions/arithmetic-bfp.md §2), so a file byte above it is one
 *  this build's quantizer could never have written. 255 at exponentBits 8
 *  (bias 127) is the canonical corrupt value -- the natural top of the
 *  8-bit range and one past the largest finite scale, 2^127. */
static void testBfpDeserializeRejectsNonFiniteScaleExponent(void) {
    FILE *f = fopen(FILE_PATH, "wb");
    fwrite("ODTS", 1, 4, f);
    writeU32LE(f, ODTS_VERSION); /* version */
    writeU32LE(f, 1);            /* layerCount */
    writeU8(f, 0);               /* wiring: chain */
    uint8_t tag = (uint8_t)RELU;
    fwrite(&tag, 1, 1, f);
    uint8_t arithByte = 0; /* ARITH_FLOAT32, HALF_AWAY -- forwardMath */
    fwrite(&arithByte, 1, 1, f);
    fwrite(&arithByte, 1, 1, f);
    fwrite(&arithByte, 1, 1, f); /* propLossMath */
    fwrite(&arithByte, 1, 1, f);
    uint8_t bfpType = (uint8_t)BFP;
    fwrite(&bfpType, 1, 1, f);           /* outputQ dtype */
    writeU32LE(f, 1);                    /* fileNumGroups = 1, matches the skeleton */
    writeU32LE(f, 0);                    /* groupSize = 0, matches the sentinel */
    uint8_t exponentByteNonFinite = 255; /* > bias + 127 == 254 -> scale 2^128 == +inf */
    fwrite(&exponentByteNonFinite, 1, 1, f);
    uint8_t mantissaBits = 4;
    fwrite(&mantissaBits, 1, 1, f);
    uint8_t exponentBits = 8;
    fwrite(&exponentBits, 1, 1, f);
    uint8_t roundingMode = 0; /* HALF_AWAY */
    fwrite(&roundingMode, 1, 1, f);
    uint8_t propLossFloatTag = (uint8_t)FLOAT32; /* matches the skeleton's default propLossQ */
    fwrite(&propLossFloatTag, 1, 1, f);
    fclose(f);

    quantization_t *floatQ = quantizationInitFloat();
    quantization_t *bfpOutputQ = quantizationInitBfp(4, 8, HALF_AWAY);
    layerQuant_t lq;
    layerQuantInitUniform(&lq, floatQ);
    lq.outputQ = bfpOutputQ;
    layer_t *layer = reluLayerInit(&lq);
    layer_t *model[] = {layer};

    f = fopen(FILE_PATH, "rb");
    ASSERT_EXITS_WITH_OUTPUT(1, "deserializeQConfig: BFP file exponent[0] (255) exceeds",
                             deserializeModel(&MODEL_CHAIN(model, 1), f));
    fclose(f);

    freeReluLayer(layer);
    freeQuantization(bfpOutputQ);
    freeQuantization(floatQ);
}

/*! #380 PR3: a FROZEN-serialized parameter (hasGrad=0, no grad tensor in the
 *  file) deserialized into a TRAINABLE skeleton (parameter->grad != NULL)
 *  must load the param and leave the skeleton's grad untouched — it was
 *  zero-initialized at construction (reserveMemory is calloc-backed), and
 *  optimizerZeroGrad re-zeros it before every batch regardless, so
 *  "untouched" and "all-zero" coincide here. Exercises deserializeParameter
 *  directly, mirroring this file's tensor-level style. */
static void testDeserializeWeightsOnlyIntoTrainableSkeleton(void) {
    float data[] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    tensor_t *srcParamTensor = makeFloatTensor2D(2, 3, data, 6);
    parameter_t srcParameter = {.param = srcParamTensor, .grad = NULL};

    FILE *f = fopen(FILE_PATH, "wb");
    serializeParameter(&srcParameter, f);
    fclose(f);

    tensor_t *skeletonParamTensor = makeFloatTensor2D(2, 3, NULL, 0);
    tensor_t *skeletonGradTensor = makeFloatTensor2D(2, 3, NULL, 0);
    parameter_t skeletonParameter = {.param = skeletonParamTensor, .grad = skeletonGradTensor};

    f = fopen(FILE_PATH, "rb");
    deserializeParameter(&skeletonParameter, f);
    fclose(f);

    /* CAPTURE every assertion value before any free. */
    float capturedParam[6];
    float capturedGrad[6];
    for (size_t i = 0; i < 6; i++) {
        capturedParam[i] = ((float *)skeletonParamTensor->data)[i];
        capturedGrad[i] = ((float *)skeletonGradTensor->data)[i];
    }

    /* FREE in reverse-init order. */
    freeTensor(skeletonGradTensor);
    freeTensor(skeletonParamTensor);
    freeTensor(srcParamTensor);

    /* ASSERT on captured. */
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(data, capturedParam, 6);
    for (size_t i = 0; i < 6; i++) {
        TEST_ASSERT_EQUAL_FLOAT(0.0f, capturedGrad[i]);
    }
}

/*! #380 PR3 review: skipSerializedTensor's rank-cap guard (ODTS_TENSOR_RECORD_MAX_RANK
 *  == 8) must reject a grad record announcing a rank above that bound BEFORE
 *  reading a single dim value. The grad record is hand-crafted as an
 *  otherwise-well-formed rank-9 FLOAT32/1-element tensor (9 dims of 1, 9
 *  positional order values, FLOAT32 type byte, 4-byte payload) rather than a
 *  truncated stub: a record that is merely too-short-to-be-rank-9 would still
 *  fail downstream via the unrelated short-read guard in serialReadBytes even
 *  with the rank cap disabled, silently passing the death-test assertion
 *  (exit code 1 either way) without ever exercising the rank-cap PRINT_ERROR
 *  itself. Only a well-formed rank-9 record makes the two guards
 *  distinguishable: guarded code exits(1) immediately on the oversized rank;
 *  with the guard removed (verified by temporary mutation) the record parses
 *  to completion and the function returns normally. Only a frozen skeleton (parameter->grad ==
 * NULL) reaches skipSerializedTensor. */
static void testSkipSerializedTensorRejectsRankAboveCap(void) {
    float paramData[] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    tensor_t *paramTensor = makeFloatTensor2D(2, 3, paramData, 6);

    FILE *f = fopen(FILE_PATH, "wb");
    uint8_t hasGrad = 1;
    fwrite(&hasGrad, sizeof(uint8_t), 1, f);
    serializeTensor(paramTensor, f);

    const uint32_t oversizedRank = 9; /* > ODTS_TENSOR_RECORD_MAX_RANK (8) */
    writeU32LE(f, oversizedRank);
    for (uint32_t d = 0; d < oversizedRank; d++) {
        writeU32LE(f, 1); /* dims[d] = 1 -> 1 element total, well-formed */
    }
    for (uint32_t d = 0; d < oversizedRank; d++) {
        writeU32LE(f, d); /* orderOfDimensions: positional only */
    }
    uint8_t floatType = (uint8_t)FLOAT32;
    fwrite(&floatType, 1, 1, f);       /* FLOAT32 qConfig is empty (no bytes) */
    uint8_t payload[4] = {0, 0, 0, 0}; /* 1 FLOAT32 element */
    fwrite(payload, 1, 4, f);
    fclose(f);

    tensor_t *skeletonParamTensor = makeFloatTensor2D(2, 3, NULL, 0);
    parameter_t frozenSkeleton = {.param = skeletonParamTensor, .grad = NULL};

    f = fopen(FILE_PATH, "rb");
    ASSERT_EXITS_WITH_FAILURE(deserializeParameter(&frozenSkeleton, f));
    fclose(f);

    freeTensor(skeletonParamTensor);
    freeTensor(paramTensor);
}

/*! #380 PR3 review: skipSerializedTensor's post-skip truncation guard (the
 *  ftell/SEEK_END comparison after the payload fseek) is the only thing that
 *  catches a payload cut short -- fseek past a genuinely truncated file's real
 *  end succeeds silently (POSIX), so the earlier fseek-past-payload call
 *  cannot fail on its own. A genuine hasGrad=1 parameter record has nothing
 *  after the grad tensor, so
 *  shortening the whole file trims the grad tensor's payload tail; mirrors
 *  testDeserializeTensorFailsFastOnTruncatedPayload's file-shortening idiom. */
static void testSkipSerializedTensorRejectsTruncatedPayload(void) {
    float paramData[] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    tensor_t *paramTensor = makeFloatTensor2D(2, 3, paramData, 6);
    tensor_t *gradTensor = makeFloatTensor2D(2, 3, paramData, 6);
    parameter_t srcParameter = {.param = paramTensor, .grad = gradTensor};

    FILE *f = fopen(FILE_PATH, "wb");
    serializeParameter(&srcParameter, f);
    long full = ftell(f);
    fclose(f);

    FILE *in = fopen(FILE_PATH, "rb");
    uint8_t *buf = reserveMemory((size_t)full);
    size_t got = fread(buf, 1, (size_t)full, in);
    fclose(in);
    f = fopen(FILE_PATH, "wb");
    fwrite(buf, 1, (size_t)full - 2, f); /* cuts into the grad tensor's payload tail */
    fclose(f);
    freeReservedMemory(buf);
    TEST_ASSERT_EQUAL_size_t((size_t)full, got);

    tensor_t *skeletonParamTensor = makeFloatTensor2D(2, 3, NULL, 0);
    parameter_t frozenSkeleton = {.param = skeletonParamTensor, .grad = NULL};

    f = fopen(FILE_PATH, "rb");
    ASSERT_EXITS_WITH_FAILURE(deserializeParameter(&frozenSkeleton, f));
    fclose(f);

    freeTensor(skeletonParamTensor);
    freeTensor(gradTensor);
    freeTensor(paramTensor);
}

#if SIZE_MAX == UINT64_MAX
/*! #394: the grad-skip path must compute the element count through the
 *  checked tensor core -- these dims wrap a size_t product to 64 elements, so
 *  an unchecked product seeks 256 bytes forward into a file that has them and
 *  silently desyncs the stream. */
static void testSkipSerializedTensorRejectsWrappingDims(void) {
    float paramData[] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    tensor_t *paramTensor = makeFloatTensor2D(2, 3, paramData, 6);

    FILE *f = fopen(FILE_PATH, "wb");
    uint8_t hasGrad = 1;
    fwrite(&hasGrad, sizeof(uint8_t), 1, f);
    serializeTensor(paramTensor, f);
    writeU32LE(f, 3u);
    writeU32LE(f, 320u);
    writeU32LE(f, 107367629u);
    writeU32LE(f, 536903681u);
    for (uint32_t d = 0; d < 3u; d++) {
        writeU32LE(f, d);
    }
    uint8_t floatType = (uint8_t)FLOAT32;
    fwrite(&floatType, 1, 1, f);
    uint8_t payload[256] = {0};
    fwrite(payload, 1, sizeof(payload), f);
    fclose(f);

    tensor_t *skeletonParamTensor = makeFloatTensor2D(2, 3, NULL, 0);
    parameter_t frozenSkeleton = {.param = skeletonParamTensor, .grad = NULL};

    f = fopen(FILE_PATH, "rb");
    ASSERT_EXITS_WITH_OUTPUT(1, "element count overflows",
                             deserializeParameter(&frozenSkeleton, f));
    fclose(f);

    freeTensor(skeletonParamTensor);
    freeTensor(paramTensor);
}
#endif

#if SIZE_MAX == UINT64_MAX && LONG_MAX == INT64_MAX
/*! #394: 2^62 - 4 FLOAT32 elements need 2^64 - 16 payload bytes -- no size_t
 *  wrap, but the (long) seek offset becomes -16, so the skip would step
 *  backwards inside the file and pass the truncation check. */
static void testSkipSerializedTensorRejectsPayloadBeyondLongMax(void) {
    float paramData[] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    tensor_t *paramTensor = makeFloatTensor2D(2, 3, paramData, 6);

    FILE *f = fopen(FILE_PATH, "wb");
    uint8_t hasGrad = 1;
    fwrite(&hasGrad, sizeof(uint8_t), 1, f);
    serializeTensor(paramTensor, f);
    writeU32LE(f, 2u);
    writeU32LE(f, 4027518961u);
    writeU32LE(f, 1145043900u);
    writeU32LE(f, 0u);
    writeU32LE(f, 1u);
    uint8_t floatType = (uint8_t)FLOAT32;
    fwrite(&floatType, 1, 1, f);
    fclose(f);

    tensor_t *skeletonParamTensor = makeFloatTensor2D(2, 3, NULL, 0);
    parameter_t frozenSkeleton = {.param = skeletonParamTensor, .grad = NULL};

    f = fopen(FILE_PATH, "rb");
    ASSERT_EXITS_WITH_OUTPUT(1, "exceeds the seekable range",
                             deserializeParameter(&frozenSkeleton, f));
    fclose(f);

    freeTensor(skeletonParamTensor);
    freeTensor(paramTensor);
}
#endif

/*! Group-quant PR2 (Task 5) mutation guard: skipSerializedTensor's SYM branch
 *  must parse a GROUPED v4 record (numGroups=3, groupSize=2) exactly -- the
 *  payload byte count is independent of numGroups (sized by qBits x element
 *  count only), but the scratch qConfig's scales[] must grow past its
 *  1-element stack array (symScratchScale) to fileNumGroups and be freed
 *  again afterwards (the mutation this guards: dropping that free leaks the
 *  reallocated array, see the report for the ASan/LSan evidence) -- or the
 *  stream desyncs and the sibling record right after misparses.
 *
 *  Built directly via deserializeParameter/serializeTensor rather than
 *  through a real layer's gradInit: Quantization.h's "Carrier gate" note
 *  fail-fasts a grad TEMPLATE with numGroups > 1 (grouped grads are a future
 *  #300 axis), but skipSerializedTensor itself has no such restriction --
 *  it only consumes wire bytes, never attaches the qConfig to a live
 *  gradient tensor. The "grad" here is a real grouped-SYM tensor_t (v4's SYM
 *  arm has been group-general in Serialize.c since PR1) that happens to be
 *  discarded because the skeleton is frozen (parameter->grad == NULL); the
 *  trailing sibling FLOAT32 tensor's correct round-trip is the stream-sync
 *  proof, mirroring testDeserializeSkipsSymGradIntoFrozenSkeleton
 *  (UnitTestSerialize.c)'s bias-round-trips-after-the-skip idiom. */
static void testSkipSerializedGroupedSymGrad(void) {
    float paramData[] = {1.f, 2.f, 3.f, 4.f};
    tensor_t *paramTensor = makeFloatTensor2D(2, 2, paramData, 4);

    tensor_t *gradTensor = makeSymGroupedTensor1D(6, 8, HALF_AWAY, 3, 2);
    int32_t gradMantissas[] = {1, -2, 3, -4, 5, -6};
    byteConversion((uint8_t *)gradMantissas, 32, gradTensor->data, 8, 6);
    symQConfig_t *gradQc = gradTensor->quantization->qConfig;
    gradQc->scales[0] = odtScaleFromFloat(0.1f, ODT_SCALE_NEAREST);
    gradQc->scales[1] = odtScaleFromFloat(0.2f, ODT_SCALE_NEAREST);
    gradQc->scales[2] = odtScaleFromFloat(0.3f, ODT_SCALE_NEAREST);

    float siblingData[] = {9.f, 8.f, 7.f};
    tensor_t *siblingTensor = makeFloatTensor2D(1, 3, siblingData, 3);

    FILE *f = fopen(FILE_PATH, "wb");
    uint8_t hasGrad = 1;
    fwrite(&hasGrad, sizeof(uint8_t), 1, f);
    serializeTensor(paramTensor, f);
    serializeTensor(gradTensor, f);
    serializeTensor(siblingTensor, f);
    fclose(f);

    tensor_t *skeletonParamTensor = makeFloatTensor2D(2, 2, NULL, 0);
    parameter_t frozenSkeleton = {.param = skeletonParamTensor, .grad = NULL};
    tensor_t *skeletonSibling = makeFloatTensor2D(1, 3, NULL, 0);

    f = fopen(FILE_PATH, "rb");
    deserializeParameter(&frozenSkeleton, f);
    deserializeTensor(skeletonSibling, f);
    fclose(f);

    /* CAPTURE before any free. */
    float capturedParam[4];
    for (size_t i = 0; i < 4; i++) {
        capturedParam[i] = ((float *)skeletonParamTensor->data)[i];
    }
    float capturedSibling[3];
    for (size_t i = 0; i < 3; i++) {
        capturedSibling[i] = ((float *)skeletonSibling->data)[i];
    }

    /* FREE in reverse-init order. */
    freeTensor(skeletonSibling);
    freeTensor(skeletonParamTensor);
    freeTensor(siblingTensor);
    freeTensor(gradTensor);
    freeTensor(paramTensor);

    /* ASSERT on captured. */
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(paramData, capturedParam, 4);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(siblingData, capturedSibling, 3);
}

/*! Group-quant PR2 (Task 5): fileNumGroups is untrusted wire
 *  input read directly into an allocation size (fileNumGroups *
 *  sizeof(float)) with no bound before this fix -- on a 32-bit size_t (MCU
 *  target) a value like 0x40000001 makes that multiplication wrap to 4
 *  bytes (a 1-float buffer) while the scales-read loop still iterates the
 *  file's full unwrapped count, a heap overflow; on a 64-bit host the same
 *  value survives the multiply intact but makes reserveMemory's calloc
 *  return NULL, and the write loop dereferences it unconditionally.
 *  SERIAL_MAX_QCONFIG_GROUPS (Deserialize.c) rejects it before any allocation.
 *
 *  Exercised through a RELU layer's outputQ -- a layer wire-config call site
 *  (deserializeLayer passes numberOfElements=0 there, no live tensor backs
 *  it) -- rather than deserializeTensor, specifically to isolate
 *  SERIAL_MAX_QCONFIG_GROUPS from the sibling elements-bound guard (Finding
 *  1.2): that guard is gated on numberOfElements != 0 and would otherwise
 *  ALSO reject any oversized fileNumGroups against a small skeleton tensor,
 *  making a deserializeTensor-based test pass even with
 *  SERIAL_MAX_QCONFIG_GROUPS itself removed.
 *
 *  The record is otherwise COMPLETE and well-formed (real scale floats,
 *  qBits/rounding, a matching propLossQ tag) all the way through, on
 *  purpose: numberOfElements==0 here means no divisibility validate ever
 *  runs regardless, so an incomplete record would die at the next truncated
 *  read instead -- exit(1) for an unrelated reason that would make the
 *  mutation check below a false positive (verified empirically while
 *  designing this test, see the report). fileNumGroups is
 *  SERIAL_MAX_QCONFIG_GROUPS+1 -- the smallest value that is both over the
 *  cap and cheap to fully materialize on the wire. */
static void testDeserializeSymRejectsOversizedNumGroupsInWireConfig(void) {
    const uint32_t oversizedNumGroups = 65537u; /* SERIAL_MAX_QCONFIG_GROUPS (65536) + 1 */

    FILE *f = fopen(FILE_PATH, "wb");
    fwrite("ODTS", 1, 4, f);
    writeU32LE(f, ODTS_VERSION); /* version */
    writeU32LE(f, 1);            /* layerCount */
    writeU8(f, 0);               /* wiring: chain */
    uint8_t tag = (uint8_t)RELU;
    fwrite(&tag, 1, 1, f);
    uint8_t arithByte = 0; /* ARITH_FLOAT32, HALF_AWAY -- forwardMath */
    fwrite(&arithByte, 1, 1, f);
    fwrite(&arithByte, 1, 1, f);
    fwrite(&arithByte, 1, 1, f); /* propLossMath */
    fwrite(&arithByte, 1, 1, f);
    uint8_t symType = (uint8_t)SYM;
    fwrite(&symType, 1, 1, f);         /* outputQ dtype */
    writeU32LE(f, oversizedNumGroups); /* fileNumGroups: past the cap */
    writeU32LE(f, 1);                  /* groupSize: nonzero, satisfies the sentinel */
    for (uint32_t g = 0; g < oversizedNumGroups; g++) {
        writeCanonScaleLE(f, 0.5f);
    }
    uint8_t qBits = 6;
    fwrite(&qBits, 1, 1, f);
    uint8_t roundingMode = 0; /* HALF_AWAY */
    fwrite(&roundingMode, 1, 1, f);
    uint8_t propLossFloatTag = (uint8_t)FLOAT32; /* matches the skeleton's default propLossQ */
    fwrite(&propLossFloatTag, 1, 1, f);
    fclose(f);

    quantization_t *floatQ = quantizationInitFloat();
    quantization_t *symOutputQ = quantizationInitSym(6, HALF_AWAY);
    layerQuant_t lq;
    layerQuantInitUniform(&lq, floatQ);
    lq.outputQ = symOutputQ;
    layer_t *layer = reluLayerInit(&lq);
    layer_t *model[] = {layer};

    f = fopen(FILE_PATH, "rb");
    ASSERT_EXITS_WITH_OUTPUT(1, "deserializeQConfig: SYM file numGroups 65537 is zero or exceeds",
                             deserializeModel(&MODEL_CHAIN(model, 1), f));
    fclose(f);

    freeReluLayer(layer);
    freeQuantization(symOutputQ);
    freeQuantization(floatQ);
}

/*! group-quant PR4 (Task 4): the ASYM twin of the oversized-numGroups test
 *  above -- same isolation rationale (RELU's propLossQ wire-config call
 *  site, numberOfElements=0, SERIAL_MAX_QCONFIG_GROUPS is the only guard
 *  reachable there), same record-completeness discipline (real scale/
 *  zeroPoint values, qBits/rounding, a matching outputQ tag all present, so
 *  removing the cap would let the whole model parse instead of dying for an
 *  unrelated truncated-read reason). */
static void testDeserializeAsymRejectsOversizedNumGroupsInWireConfig(void) {
    const uint32_t oversizedNumGroups = 65537u; /* SERIAL_MAX_QCONFIG_GROUPS (65536) + 1 */

    FILE *f = fopen(FILE_PATH, "wb");
    fwrite("ODTS", 1, 4, f);
    writeU32LE(f, ODTS_VERSION); /* version */
    writeU32LE(f, 1);            /* layerCount */
    writeU8(f, 0);               /* wiring: chain */
    uint8_t tag = (uint8_t)RELU;
    fwrite(&tag, 1, 1, f);
    uint8_t arithByte = 0; /* ARITH_FLOAT32, HALF_AWAY -- forwardMath */
    fwrite(&arithByte, 1, 1, f);
    fwrite(&arithByte, 1, 1, f);
    fwrite(&arithByte, 1, 1, f); /* propLossMath */
    fwrite(&arithByte, 1, 1, f);
    uint8_t outputFloatTag = (uint8_t)FLOAT32; /* matches the skeleton's default outputQ */
    fwrite(&outputFloatTag, 1, 1, f);
    uint8_t asymType = (uint8_t)ASYM;
    fwrite(&asymType, 1, 1, f);        /* propLossQ dtype */
    writeU32LE(f, oversizedNumGroups); /* fileNumGroups: past the cap */
    writeU32LE(f, 1);                  /* groupSize: nonzero, satisfies the sentinel */
    for (uint32_t g = 0; g < oversizedNumGroups; g++) {
        writeCanonScaleLE(f, 0.5f); /* scales[g] */
    }
    for (uint32_t g = 0; g < oversizedNumGroups; g++) {
        writeU16LE(f, 0); /* zeroPoints[g] */
    }
    uint8_t qBits = 8;
    fwrite(&qBits, 1, 1, f);
    uint8_t roundingMode = 0; /* HALF_AWAY */
    fwrite(&roundingMode, 1, 1, f);
    fclose(f);

    quantization_t *floatQ = quantizationInitFloat();
    quantization_t *asymPropLossQ = quantizationInitAsym(8, HALF_AWAY);
    layerQuant_t lq;
    layerQuantInitUniform(&lq, floatQ);
    lq.propLossQ = asymPropLossQ;
    layer_t *layer = reluLayerInit(&lq);
    layer_t *model[] = {layer};

    f = fopen(FILE_PATH, "rb");
    ASSERT_EXITS_WITH_OUTPUT(1, "deserializeQConfig: ASYM file numGroups 65537 is zero or exceeds",
                             deserializeModel(&MODEL_CHAIN(model, 1), f));
    fclose(f);

    freeReluLayer(layer);
    freeQuantization(asymPropLossQ);
    freeQuantization(floatQ);
}

/*! fileNumGroups == 0 dies at the group cap. A layer wire config has no
 *  geometry, so no shape validation can catch it instead, and the record is
 *  complete, so without the cap the model parses. */
static void testDeserializeSymRejectsZeroNumGroupsInWireConfig(void) {
    FILE *f = fopen(FILE_PATH, "wb");
    fwrite("ODTS", 1, 4, f);
    writeU32LE(f, ODTS_VERSION); /* version */
    writeU32LE(f, 1);            /* layerCount */
    writeU8(f, 0);               /* wiring: chain */
    uint8_t tag = (uint8_t)RELU;
    fwrite(&tag, 1, 1, f);
    uint8_t arithByte = 0;
    fwrite(&arithByte, 1, 1, f);
    fwrite(&arithByte, 1, 1, f);
    fwrite(&arithByte, 1, 1, f);
    fwrite(&arithByte, 1, 1, f);
    uint8_t symType = (uint8_t)SYM;
    fwrite(&symType, 1, 1, f);
    writeU32LE(f, 0); /* fileNumGroups = 0 */
    writeU32LE(f, 1); /* groupSize: nonzero, satisfies the sentinel (numGroups
                       * != 1 requires groupSize != 0) */
    uint8_t qBits = 6;
    fwrite(&qBits, 1, 1, f);
    uint8_t roundingMode = 0;
    fwrite(&roundingMode, 1, 1, f);
    uint8_t propLossFloatTag = (uint8_t)FLOAT32;
    fwrite(&propLossFloatTag, 1, 1, f);
    fclose(f);

    quantization_t *floatQ = quantizationInitFloat();
    quantization_t *symOutputQ = quantizationInitSym(6, HALF_AWAY);
    layerQuant_t lq;
    layerQuantInitUniform(&lq, floatQ);
    lq.outputQ = symOutputQ;
    layer_t *layer = reluLayerInit(&lq);
    layer_t *model[] = {layer};

    f = fopen(FILE_PATH, "rb");
    ASSERT_EXITS_WITH_OUTPUT(1, "deserializeQConfig: SYM file numGroups 0 is zero",
                             deserializeModel(&MODEL_CHAIN(model, 1), f));
    fclose(f);

    freeReluLayer(layer);
    freeQuantization(symOutputQ);
    freeQuantization(floatQ);
}

/*! group-quant PR4 (Task 4): the ASYM twin of the zero-numGroups test above
 *  -- same isolation rationale, exercised through propLossQ instead of
 *  outputQ (mirrors testDeserializeAsymRejectsOversizedNumGroupsInWireConfig
 *  above). Record is complete (0 scale/zeroPoint entries follow, since
 *  fileNumGroups == 0) so removing the guard would let the whole model
 *  parse instead of dying for an unrelated reason. */
static void testDeserializeAsymRejectsZeroNumGroupsInWireConfig(void) {
    FILE *f = fopen(FILE_PATH, "wb");
    fwrite("ODTS", 1, 4, f);
    writeU32LE(f, ODTS_VERSION); /* version */
    writeU32LE(f, 1);            /* layerCount */
    writeU8(f, 0);               /* wiring: chain */
    uint8_t tag = (uint8_t)RELU;
    fwrite(&tag, 1, 1, f);
    uint8_t arithByte = 0;
    fwrite(&arithByte, 1, 1, f);
    fwrite(&arithByte, 1, 1, f);
    fwrite(&arithByte, 1, 1, f);
    fwrite(&arithByte, 1, 1, f);
    uint8_t outputFloatTag = (uint8_t)FLOAT32;
    fwrite(&outputFloatTag, 1, 1, f);
    uint8_t asymType = (uint8_t)ASYM;
    fwrite(&asymType, 1, 1, f);
    writeU32LE(f, 0); /* fileNumGroups = 0 */
    writeU32LE(f, 1); /* groupSize: nonzero, satisfies the sentinel (numGroups
                       * != 1 requires groupSize != 0) */
    uint8_t qBits = 8;
    fwrite(&qBits, 1, 1, f);
    uint8_t roundingMode = 0;
    fwrite(&roundingMode, 1, 1, f);
    fclose(f);

    quantization_t *floatQ = quantizationInitFloat();
    quantization_t *asymPropLossQ = quantizationInitAsym(8, HALF_AWAY);
    layerQuant_t lq;
    layerQuantInitUniform(&lq, floatQ);
    lq.propLossQ = asymPropLossQ;
    layer_t *layer = reluLayerInit(&lq);
    layer_t *model[] = {layer};

    f = fopen(FILE_PATH, "rb");
    ASSERT_EXITS_WITH_OUTPUT(1, "deserializeQConfig: ASYM file numGroups 0 is zero",
                             deserializeModel(&MODEL_CHAIN(model, 1), f));
    fclose(f);

    freeReluLayer(layer);
    freeQuantization(asymPropLossQ);
    freeQuantization(floatQ);
}

/*! group-quant PR4 (group-quant D6): a wire qBits outside [1, 16] must fail fast
 *  regardless of numGroups/numberOfElements -- the code-domain zp requires
 *  qBits <= 16 to fit u16 (see Quantization.h). Exercised through a real
 *  per-tensor ASYM tensor record (deserializeTensor, numberOfElements != 0)
 *  with an otherwise well-formed v5 record: numGroups=1, groupSize=0, a real
 *  scale/zeroPoint, THEN qBits=17. The check is unconditional (independent
 *  of numberOfElements), so this also pins the wire-config call site would
 *  reject it identically. */
static void testDeserializeAsymRejectsWireQBitsAboveSixteen(void) {
    FILE *f = fopen(FILE_PATH, "wb");
    writeU32LE(f, 1); /* numberOfDimensions */
    writeU32LE(f, 4); /* dimensions[0] */
    writeU32LE(f, 0); /* orderOfDimensions[0] */
    uint8_t asymType = (uint8_t)ASYM;
    fwrite(&asymType, 1, 1, f);
    writeU32LE(f, 1); /* numGroups */
    writeU32LE(f, 0); /* groupSize */
    writeCanonScaleLE(f, 0.5f);
    writeU16LE(f, 3);
    uint8_t qBits = 17; /* > 16, group-quant D6 ceiling */
    fwrite(&qBits, 1, 1, f);
    uint8_t roundingMode = 0; /* HALF_AWAY */
    fwrite(&roundingMode, 1, 1, f);
    uint8_t payload[4] = {0, 0, 0, 0}; /* never reached: the guard dies first */
    fwrite(payload, 1, 4, f);
    fclose(f);

    tensor_t *skeleton = makeAsymTensor1D(4);
    f = fopen(FILE_PATH, "rb");
    /* The message pins the width check: a wider record also fails the payload size. */
    ASSERT_EXITS_WITH_OUTPUT(1, "deserializeQConfig: ASYM file qBits",
                             deserializeTensor(skeleton, f));
    fclose(f);

    freeTensor(skeleton);
}

static void writeSymTensorRecordWithQBits(uint8_t qBits) {
    FILE *f = fopen(FILE_PATH, "wb");
    writeU32LE(f, 1); /* numberOfDimensions */
    writeU32LE(f, 4); /* dimensions[0] */
    writeU32LE(f, 0); /* orderOfDimensions[0] */
    uint8_t symType = (uint8_t)SYM;
    fwrite(&symType, 1, 1, f);
    writeU32LE(f, 1); /* numGroups */
    writeU32LE(f, 0); /* groupSize */
    writeCanonScaleLE(f, 0.5f);
    fwrite(&qBits, 1, 1, f);
    uint8_t roundingMode = 0; /* HALF_AWAY */
    fwrite(&roundingMode, 1, 1, f);
    uint8_t payload[16] = {0}; /* never reached: the guard dies first */
    fwrite(payload, 1, sizeof(payload), f);
    fclose(f);
}

/* A complete RELU record whose outputQ wire config carries qBits: without the guard the whole
 * model parses, because a wire config has no geometry to validate against. */
static void writeReluSymOutputWireRecord(uint32_t numGroups, uint32_t groupSize, uint8_t qBits) {
    FILE *f = fopen(FILE_PATH, "wb");
    fwrite("ODTS", 1, 4, f);
    writeU32LE(f, ODTS_VERSION);
    writeU32LE(f, 1); /* layerCount */
    writeU8(f, 0);    /* wiring: chain */
    uint8_t tag = (uint8_t)RELU;
    fwrite(&tag, 1, 1, f);
    uint8_t arithByte = 0;
    fwrite(&arithByte, 1, 1, f);
    fwrite(&arithByte, 1, 1, f);
    fwrite(&arithByte, 1, 1, f);
    fwrite(&arithByte, 1, 1, f);
    uint8_t symType = (uint8_t)SYM;
    fwrite(&symType, 1, 1, f);
    writeU32LE(f, numGroups);
    writeU32LE(f, groupSize);
    for (uint32_t g = 0; g < numGroups; g++) {
        writeCanonScaleLE(f, 0.5f);
    }
    fwrite(&qBits, 1, 1, f);
    uint8_t roundingMode = 0;
    fwrite(&roundingMode, 1, 1, f);
    uint8_t propLossFloatTag = (uint8_t)FLOAT32;
    fwrite(&propLossFloatTag, 1, 1, f);
    fclose(f);
}

static void writeReluSymOutputWireRecordWithQBits(uint8_t qBits) {
    writeReluSymOutputWireRecord(1, 0, qBits);
}

static void assertSymTensorRecordQBitsRejected(uint8_t qBits) {
    writeSymTensorRecordWithQBits(qBits);
    tensor_t *skeleton = makeSymTensor1D(4, 8, HALF_AWAY);
    FILE *f = fopen(FILE_PATH, "rb");
    ASSERT_EXITS_WITH_OUTPUT(1, "deserializeQConfig: SYM file qBits",
                             deserializeTensor(skeleton, f));
    fclose(f);
    freeTensor(skeleton);
}

static void assertSymWireConfigQBitsRejected(uint8_t qBits) {
    writeReluSymOutputWireRecordWithQBits(qBits);
    quantization_t *floatQ = quantizationInitFloat();
    quantization_t *symOutputQ = quantizationInitSym(8, HALF_AWAY);
    layerQuant_t lq;
    layerQuantInitUniform(&lq, floatQ);
    lq.outputQ = symOutputQ;
    layer_t *layer = reluLayerInit(&lq);
    layer_t *model[] = {layer};
    FILE *f = fopen(FILE_PATH, "rb");
    ASSERT_EXITS_WITH_OUTPUT(1, "deserializeQConfig: SYM file qBits",
                             deserializeModel(&MODEL_CHAIN(model, 1), f));
    fclose(f);
    freeReluLayer(layer);
    freeQuantization(symOutputQ);
    freeQuantization(floatQ);
}

/*! A wire config has no geometry, so only the unconditional sentinel check
 *  stops a {1, 3} or {2, 0} group shape there; the record is otherwise complete. */
static void testDeserializeSymRejectsSentinelViolationInWireConfig(void) {
    const uint32_t shapes[][2] = {{1, 3}, {2, 0}};
    for (size_t i = 0; i < 2; i++) {
        writeReluSymOutputWireRecord(shapes[i][0], shapes[i][1], 8);
        quantization_t *floatQ = quantizationInitFloat();
        quantization_t *symOutputQ = quantizationInitSym(8, HALF_AWAY);
        layerQuant_t lq;
        layerQuantInitUniform(&lq, floatQ);
        lq.outputQ = symOutputQ;
        layer_t *layer = reluLayerInit(&lq);
        layer_t *model[] = {layer};
        FILE *f = fopen(FILE_PATH, "rb");
        ASSERT_EXITS_WITH_OUTPUT(1, "sentinel invariant",
                                 deserializeModel(&MODEL_CHAIN(model, 1), f));
        fclose(f);
        freeReluLayer(layer);
        freeQuantization(symOutputQ);
        freeQuantization(floatQ);
    }
}

/*! SYM widths outside [2, 31] die at both call sites: a tensor record and a
 *  layer wire config, which has no geometry and skips the shape validation.
 *  qBits == 1 has qMax == 0, so its scale is inf and every dequant is NaN. */
static void testDeserializeSymRejectsWireQBitsOutsideRange(void) {
    const uint8_t outside[] = {0, 1, 32};
    for (size_t i = 0; i < sizeof(outside); i++) {
        assertSymTensorRecordQBitsRejected(outside[i]);
        assertSymWireConfigQBitsRejected(outside[i]);
    }
}

/*! Pins both inclusive ends of [2, 31], so an off-by-one tightening of the guard fails. */
static void testDeserializeSymAcceptsWireQBitsBoundsInWireConfig(void) {
    const uint8_t bounds[] = {2, 31};
    for (size_t i = 0; i < 2; i++) {
        writeReluSymOutputWireRecordWithQBits(bounds[i]);
        quantization_t *floatQ = quantizationInitFloat();
        quantization_t *symOutputQ = quantizationInitSym(8, HALF_AWAY);
        layerQuant_t lq;
        layerQuantInitUniform(&lq, floatQ);
        lq.outputQ = symOutputQ;
        layer_t *layer = reluLayerInit(&lq);
        layer_t *model[] = {layer};
        FILE *f = fopen(FILE_PATH, "rb");
        deserializeModel(&MODEL_CHAIN(model, 1), f);
        fclose(f);
        symQConfig_t *qC = layer->config->relu->outputQ->qConfig;
        uint8_t capturedQBits = qC->qBits;
        freeReluLayer(layer);
        freeQuantization(symOutputQ);
        freeQuantization(floatQ);
        TEST_ASSERT_EQUAL_UINT8(bounds[i], capturedQBits);
    }
}

static tensor_t *makeAsymGroupedTensor1D(size_t d0, uint8_t qBits, roundingMode_t roundingMode,
                                         size_t numGroups, size_t groupSize) {
    size_t *dims = reserveMemory(1 * sizeof(size_t));
    dims[0] = d0;
    size_t *order = reserveMemory(1 * sizeof(size_t));
    setOrderOfDimsForNewTensor(1, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 1, order);
    return initTensor(shape, quantizationInitAsymGrouped(qBits, roundingMode, numGroups, groupSize),
                      NULL);
}

/* Writes a parameter record whose grad is `fileGrad`, then loads it into a trainable skeleton
 * whose grad is the per-tensor `skeletonGrad` of the same payload size. */
static void loadGradRecordIntoPerTensorSkeleton(tensor_t *fileGrad, tensor_t *skeletonGrad) {
    tensor_t *param = makeFloatTensor2D(2, 3, (float[]){1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, 6);
    FILE *f = fopen(FILE_PATH, "wb");
    uint8_t hasGrad = 1;
    fwrite(&hasGrad, sizeof(uint8_t), 1, f);
    serializeTensor(param, f);
    serializeTensor(fileGrad, f);
    fclose(f);

    parameter_t skeleton = {.param = makeFloatTensor2D(2, 3, NULL, 0), .grad = skeletonGrad};
    f = fopen(FILE_PATH, "rb");
    deserializeParameter(&skeleton, f);
    fclose(f);
    freeTensor(skeleton.param);
    freeTensor(param);
}

/* The optimizer's gradient scaling folds into group 0 only, so a loaded gradient must stay
 * per-tensor even when its record regroups the skeleton's grid. */
static void testDeserializeParameterRejectsGroupedSymGrad(void) {
    ASSERT_EXITS_WITH_OUTPUT(
        1, "grouped SYM gradient record",
        loadGradRecordIntoPerTensorSkeleton(makeSymGroupedTensor1D(6, 8, HALF_AWAY, 3, 2),
                                            makeSymTensor1D(6, 8, HALF_AWAY)));
}

static void testDeserializeParameterRejectsGroupedAsymGrad(void) {
    ASSERT_EXITS_WITH_OUTPUT(
        1, "grouped ASYM gradient record",
        loadGradRecordIntoPerTensorSkeleton(makeAsymGroupedTensor1D(6, 8, HALF_AWAY, 3, 2),
                                            makeAsymTensor1D(6)));
}

/* A per-tensor grad record still loads into a trainable skeleton. */
static void testDeserializeParameterLoadsPerTensorSymGrad(void) {
    tensor_t *fileGrad = makeSymTensor1D(6, 8, HALF_AWAY);
    tensorFillFromFloatBuffer(fileGrad, (float[]){1.f, -2.f, 3.f, -4.f, 5.f, -6.f}, 6);
    tensor_t *skeletonGrad = makeSymTensor1D(6, 8, HALF_AWAY);
    loadGradRecordIntoPerTensorSkeleton(fileGrad, skeletonGrad);

    size_t bytes = calcNumberOfBytesForData(fileGrad->quantization, 6);
    TEST_ASSERT_EQUAL_UINT8_ARRAY(fileGrad->data, skeletonGrad->data, bytes);
    TEST_ASSERT_EQUAL_FLOAT(
        odtScaleToFloat(((symQConfig_t *)fileGrad->quantization->qConfig)->scales[0]),
        odtScaleToFloat(((symQConfig_t *)skeletonGrad->quantization->qConfig)->scales[0]));
    freeTensor(skeletonGrad);
    freeTensor(fileGrad);
}

/*! ASYM twin of testDeserializeGroupedSymRejectsBadDivisibility, with the
 *  zeroPoints array. */
static void testDeserializeGroupedAsymRejectsBadDivisibility(void) {
    FILE *f = fopen(FILE_PATH, "wb");
    writeU32LE(f, 1); /* numberOfDimensions */
    writeU32LE(f, 4); /* dimensions[0] = 4 elements */
    writeU32LE(f, 0); /* orderOfDimensions[0] */
    uint8_t asymType = (uint8_t)ASYM;
    fwrite(&asymType, 1, 1, f);
    writeU32LE(f, 3); /* numGroups = 3 */
    writeU32LE(f, 2); /* groupSize = 2 -> 3*2 == 6 != 4 elements */
    writeCanonScaleLE(f, 0.5f);
    writeCanonScaleLE(f, 0.5f);
    writeCanonScaleLE(f, 0.5f); /* three scales */
    writeU16LE(f, 1);
    writeU16LE(f, 2);
    writeU16LE(f, 3); /* three zeroPoints */
    uint8_t qBits = 4;
    fwrite(&qBits, 1, 1, f);
    uint8_t roundingMode = 0; /* HALF_AWAY */
    fwrite(&roundingMode, 1, 1, f);
    uint8_t payload[2] = {0xAB, 0xCD}; /* never reached: the validate dies first */
    fwrite(payload, 1, 2, f);
    fclose(f);

    tensor_t *skeleton = makeAsymTensor1D(4);
    f = fopen(FILE_PATH, "rb");
    ASSERT_EXITS_WITH_FAILURE(deserializeTensor(skeleton, f));
    fclose(f);

    freeTensor(skeleton);
}

/*! group-quant PR4 (Task 4): a GROUPED file record (numGroups=3, groupSize=2)
 *  deserialized into a freshly-built PER-TENSOR skeleton (numGroups=1,
 *  groupSize=0) must reallocate BOTH the skeleton's scales[] AND
 *  zeroPoints[] arrays and load every group's scale/zeroPoint, and the
 *  packed tensor DATA must round-trip too -- proving the payload read
 *  (sized by qBits x element count only, independent of numGroups) is
 *  unaffected. The SYM case is
 *  testDeserializeQConfigAcceptsNumGroupsMismatchViaReallocation; this one is built through the
 * real API (quantizationInitAsymGrouped + serializeTensor), not hand-crafted bytes. Run under ASan
 * (mutation ii: sizing only scales[] on realloc leaves zeroPoints[] stale -- an ASan-visible
 * heap-buffer-overflow the moment the zeroPoints loop writes past it). */
static void testDeserializeGroupedAsymIntoPerTensorSkeleton(void) {
    tensor_t *src = makeAsymGroupedTensor1D(6, 8, HALF_AWAY, 3, 2);
    int32_t codes[] = {10, 20, 30, 40, 50, 60};
    byteConversion((uint8_t *)codes, 32, src->data, 8, 6);
    asymQConfig_t *srcQc = src->quantization->qConfig;
    srcQc->scales[0] = odtTestScale(0.125f);
    srcQc->scales[1] = odtTestScale(0.25f);
    srcQc->scales[2] = odtTestScale(0.375f);
    srcQc->zeroPoints[0] = 5;
    srcQc->zeroPoints[1] = 120;
    srcQc->zeroPoints[2] = 250;

    FILE *f = fopen(FILE_PATH, "wb");
    serializeTensor(src, f);
    fclose(f);

    tensor_t *skeleton = makeAsymTensor1D(6); /* per-tensor: {1,0} */

    f = fopen(FILE_PATH, "rb");
    deserializeTensor(skeleton, f);
    fclose(f);

    asymQConfig_t *dstQc = skeleton->quantization->qConfig;
    /* CAPTURE before any free. */
    size_t capturedNumGroups = dstQc->numGroups;
    size_t capturedGroupSize = dstQc->groupSize;
    float capturedScales[3];
    uint16_t capturedZps[3];
    for (size_t g = 0; g < 3; g++) {
        capturedScales[g] = odtScaleToFloat(dstQc->scales[g]);
        capturedZps[g] = dstQc->zeroPoints[g];
    }
    size_t dataBytes = calcNumberOfBytesForData(skeleton->quantization, 6);
    uint8_t capturedSrcData[8];
    uint8_t capturedDstData[8];
    memcpy(capturedSrcData, src->data, dataBytes);
    memcpy(capturedDstData, skeleton->data, dataBytes);

    /* FREE in reverse-init order. */
    freeTensor(skeleton);
    freeTensor(src);

    /* ASSERT on captured. */
    TEST_ASSERT_EQUAL_size_t(3, capturedNumGroups);
    TEST_ASSERT_EQUAL_size_t(2, capturedGroupSize);
    TEST_ASSERT_EQUAL_FLOAT(0.125f, capturedScales[0]);
    TEST_ASSERT_EQUAL_FLOAT(0.25f, capturedScales[1]);
    TEST_ASSERT_EQUAL_FLOAT(0.375f, capturedScales[2]);
    TEST_ASSERT_EQUAL_UINT16(5, capturedZps[0]);
    TEST_ASSERT_EQUAL_UINT16(120, capturedZps[1]);
    TEST_ASSERT_EQUAL_UINT16(250, capturedZps[2]);
    TEST_ASSERT_EQUAL_HEX8_ARRAY(capturedSrcData, capturedDstData, dataBytes);
}

/*! group-quant PR4 (Task 4) mutation guard: skipSerializedTensor's ASYM
 *  branch must parse a GROUPED v5 record (numGroups=3, groupSize=2) exactly
 *  -- the payload byte count is independent of numGroups, but the scratch
 *  qConfig's scales[]/zeroPoints[] must both grow past their 1-element
 *  initial arrays to fileNumGroups and be freed again afterwards (mutation
 *  iii: dropping the zeroPoints free leaks the reallocated array -- LSan/
 *  ASan-visible, see the report) -- or the stream desyncs and the sibling
 *  record right after misparses. Mirrors testSkipSerializedGroupedSymGrad
 *  exactly. */
static void testSkipSerializedGroupedAsymGrad(void) {
    float paramData[] = {1.f, 2.f, 3.f, 4.f};
    tensor_t *paramTensor = makeFloatTensor2D(2, 2, paramData, 4);

    tensor_t *gradTensor = makeAsymGroupedTensor1D(6, 8, HALF_AWAY, 3, 2);
    int32_t gradCodes[] = {10, 20, 30, 40, 50, 60};
    byteConversion((uint8_t *)gradCodes, 32, gradTensor->data, 8, 6);
    asymQConfig_t *gradQc = gradTensor->quantization->qConfig;
    gradQc->scales[0] = odtScaleFromFloat(0.1f, ODT_SCALE_NEAREST);
    gradQc->scales[1] = odtScaleFromFloat(0.2f, ODT_SCALE_NEAREST);
    gradQc->scales[2] = odtScaleFromFloat(0.3f, ODT_SCALE_NEAREST);
    gradQc->zeroPoints[0] = 5;
    gradQc->zeroPoints[1] = 120;
    gradQc->zeroPoints[2] = 250;

    float siblingData[] = {9.f, 8.f, 7.f};
    tensor_t *siblingTensor = makeFloatTensor2D(1, 3, siblingData, 3);

    FILE *f = fopen(FILE_PATH, "wb");
    uint8_t hasGrad = 1;
    fwrite(&hasGrad, sizeof(uint8_t), 1, f);
    serializeTensor(paramTensor, f);
    serializeTensor(gradTensor, f);
    serializeTensor(siblingTensor, f);
    fclose(f);

    tensor_t *skeletonParamTensor = makeFloatTensor2D(2, 2, NULL, 0);
    parameter_t frozenSkeleton = {.param = skeletonParamTensor, .grad = NULL};
    tensor_t *skeletonSibling = makeFloatTensor2D(1, 3, NULL, 0);

    f = fopen(FILE_PATH, "rb");
    deserializeParameter(&frozenSkeleton, f);
    deserializeTensor(skeletonSibling, f);
    fclose(f);

    /* CAPTURE before any free. */
    float capturedParam[4];
    for (size_t i = 0; i < 4; i++) {
        capturedParam[i] = ((float *)skeletonParamTensor->data)[i];
    }
    float capturedSibling[3];
    for (size_t i = 0; i < 3; i++) {
        capturedSibling[i] = ((float *)skeletonSibling->data)[i];
    }

    /* FREE in reverse-init order. */
    freeTensor(skeletonSibling);
    freeTensor(skeletonParamTensor);
    freeTensor(siblingTensor);
    freeTensor(gradTensor);
    freeTensor(paramTensor);

    /* ASSERT on captured. */
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(paramData, capturedParam, 4);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(siblingData, capturedSibling, 3);
}

/*! ASYM twin of testSkipSerializedGroupedSymGradRejectsBadDivisibility. */
static void testSkipSerializedGroupedAsymGradRejectsBadDivisibility(void) {
    float paramData[] = {1.f, 2.f, 3.f, 4.f};
    tensor_t *paramTensor = makeFloatTensor2D(2, 2, paramData, 4);

    FILE *f = fopen(FILE_PATH, "wb");
    uint8_t hasGrad = 1;
    fwrite(&hasGrad, sizeof(uint8_t), 1, f);
    serializeTensor(paramTensor, f);

    writeU32LE(f, 1); /* grad numberOfDimensions */
    writeU32LE(f, 6); /* grad dimensions[0] = 6 elements */
    writeU32LE(f, 0); /* grad orderOfDimensions[0] */
    uint8_t asymType = (uint8_t)ASYM;
    fwrite(&asymType, 1, 1, f);
    writeU32LE(f, 3); /* numGroups = 3 */
    writeU32LE(f, 3); /* groupSize = 3 -> 3*3 == 9 != 6 elements */
    writeCanonScaleLE(f, 0.5f);
    writeCanonScaleLE(f, 0.5f);
    writeCanonScaleLE(f, 0.5f); /* three scales */
    writeU16LE(f, 1);
    writeU16LE(f, 2);
    writeU16LE(f, 3); /* three zeroPoints */
    uint8_t qBits = 4;
    fwrite(&qBits, 1, 1, f);
    uint8_t roundingMode = 0; /* HALF_AWAY */
    fwrite(&roundingMode, 1, 1, f);
    uint8_t payload[3] = {0xAB, 0xCD, 0xEF}; /* never reached: the validate dies first */
    fwrite(payload, 1, 3, f);
    fclose(f);

    tensor_t *skeletonParamTensor = makeFloatTensor2D(2, 2, NULL, 0);
    parameter_t frozenSkeleton = {.param = skeletonParamTensor, .grad = NULL};

    f = fopen(FILE_PATH, "rb");
    ASSERT_EXITS_WITH_FAILURE(deserializeParameter(&frozenSkeleton, f));
    fclose(f);

    freeTensor(skeletonParamTensor);
    freeTensor(paramTensor);
}

/*! A skipped grouped grad record is validated against its own element count:
 *  numGroups * groupSize (9) != 6 dies even though the record is discarded. */
static void testSkipSerializedGroupedSymGradRejectsBadDivisibility(void) {
    float paramData[] = {1.f, 2.f, 3.f, 4.f};
    tensor_t *paramTensor = makeFloatTensor2D(2, 2, paramData, 4);

    FILE *f = fopen(FILE_PATH, "wb");
    uint8_t hasGrad = 1;
    fwrite(&hasGrad, sizeof(uint8_t), 1, f);
    serializeTensor(paramTensor, f);

    writeU32LE(f, 1); /* grad numberOfDimensions */
    writeU32LE(f, 6); /* grad dimensions[0] = 6 elements */
    writeU32LE(f, 0); /* grad orderOfDimensions[0] */
    uint8_t symType = (uint8_t)SYM;
    fwrite(&symType, 1, 1, f);
    writeU32LE(f, 3); /* numGroups = 3 */
    writeU32LE(f, 3); /* groupSize = 3 -> 3*3 == 9 != 6 elements */
    writeCanonScaleLE(f, 0.5f);
    writeCanonScaleLE(f, 0.5f);
    writeCanonScaleLE(f, 0.5f); /* three scales */
    uint8_t qBits = 4;
    fwrite(&qBits, 1, 1, f);
    uint8_t roundingMode = 0; /* HALF_AWAY */
    fwrite(&roundingMode, 1, 1, f);
    uint8_t payload[3] = {0xAB, 0xCD, 0xEF}; /* never reached: the validate dies first */
    fwrite(payload, 1, 3, f);
    fclose(f);

    tensor_t *skeletonParamTensor = makeFloatTensor2D(2, 2, NULL, 0);
    parameter_t frozenSkeleton = {.param = skeletonParamTensor, .grad = NULL};

    f = fopen(FILE_PATH, "rb");
    ASSERT_EXITS_WITH_FAILURE(deserializeParameter(&frozenSkeleton, f));
    fclose(f);

    freeTensor(skeletonParamTensor);
    freeTensor(paramTensor);
}

/*! BFP epic PR2 (Task 1): arithmetic wire-tag round trip -- a layer's
 *  forwardMath hand-set to {ARITH_BFP, SR_HALF_AWAY} on the SERIAL side only
 *  (mirroring testRoundTripLinear's grad-seed rationale in
 *  UnitTestSerialize.c: a round trip that actually moves bytes is
 *  distinguishable from one that leaves the deserial mirror's uniform
 *  FLOAT32/HALF_AWAY default untouched) survives serializeModel /
 *  deserializeModel intact. RELU is the minimal layer with a forwardMath
 *  field. The derivation flip (arithmeticFromQuantization producing
 *  ARITH_BFP) is Task 9 -- this test only proves the wire tag itself
 *  round-trips once a caller hand-sets it. */
static void testDeserializeArithmeticRoundTripsBfp(void) {
    quantization_t *floatQ = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, floatQ);

    layer_t *serialLayer = reluLayerInit(&lq);
    layer_t *deserialLayer = reluLayerInit(&lq);

    serialLayer->config->relu->forwardMath =
        (arithmetic_t){.type = ARITH_BFP, .roundingMode = SR_HALF_AWAY};

    layer_t *serialModel[] = {serialLayer};
    layer_t *deserialModel[] = {deserialLayer};

    FILE *f = fopen(FILE_PATH, "wb");
    serializeModel(&MODEL_CHAIN(serialModel, 1), f);
    fclose(f);

    f = fopen(FILE_PATH, "rb");
    deserializeModel(&MODEL_CHAIN(deserialModel, 1), f);
    fclose(f);

    /* CAPTURE before any free. */
    arithmetic_t loadedForwardMath = deserialLayer->config->relu->forwardMath;

    freeReluLayer(deserialLayer);
    freeReluLayer(serialLayer);
    freeQuantization(floatQ);

    /* ASSERT on captured. */
    TEST_ASSERT_EQUAL_INT(ARITH_BFP, loadedForwardMath.type);
    TEST_ASSERT_EQUAL_INT(SR_HALF_AWAY, loadedForwardMath.roundingMode);
}

/*! BFP epic PR2 (Task 1): corruption guard sibling -- an arithmetic-type wire
 *  tag one past ARITH_BFP (3) has no matching enum member yet. Hand-crafted
 *  ODTS v5 bytes, mirroring the neighboring BFP width-rejection tests'
 *  well-formed-except-one-byte idiom (testBfpDeserializeRejectsMantissaBitsOutOfRange
 *  et al. above): every other byte in the record is what a real serialize of
 *  the RELU skeleton below would produce, EXCEPT the forwardMath type byte,
 *  which is patched to the out-of-range value 3. deserializeArithmetic reads
 *  and range-checks the type byte before the roundingMode byte, so this dies
 *  before the rest of the record is even read -- an unvalidated read here
 *  would instead alias to whatever future 4th arithmeticType_t member gets
 *  appended, silently misinterpreting the compute representation. */
static void testDeserializeArithmeticRejectsUnknownTypeTag(void) {
    FILE *f = fopen(FILE_PATH, "wb");
    fwrite("ODTS", 1, 4, f);
    writeU32LE(f, ODTS_VERSION); /* version */
    writeU32LE(f, 1);            /* layerCount */
    writeU8(f, 0);               /* wiring: chain */
    uint8_t tag = (uint8_t)RELU;
    fwrite(&tag, 1, 1, f);
    uint8_t badArithType = 3;       /* one past ARITH_BFP (2) -- unknown wire tag */
    fwrite(&badArithType, 1, 1, f); /* forwardMath.type */
    uint8_t halfAway = (uint8_t)HALF_AWAY;
    fwrite(&halfAway, 1, 1, f); /* forwardMath.roundingMode */
    uint8_t floatType = (uint8_t)ARITH_FLOAT32;
    fwrite(&floatType, 1, 1, f); /* propLossMath.type */
    fwrite(&halfAway, 1, 1, f);  /* propLossMath.roundingMode */
    uint8_t floatQType = (uint8_t)FLOAT32;
    fwrite(&floatQType, 1, 1, f); /* outputQ dtype */
    fwrite(&floatQType, 1, 1, f); /* propLossQ dtype */
    fclose(f);

    quantization_t *floatQ = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, floatQ);
    layer_t *layer = reluLayerInit(&lq);
    layer_t *model[] = {layer};

    f = fopen(FILE_PATH, "rb");
    ASSERT_EXITS_WITH_OUTPUT(1, "deserializeArithmetic: unknown arithmetic wire tag 3",
                             deserializeModel(&MODEL_CHAIN(model, 1), f));
    fclose(f);

    freeReluLayer(layer);
    freeQuantization(floatQ);
}

/*! BFP epic PR2 (Task 1): roundingMode-tag sibling of
 *  testDeserializeArithmeticRejectsUnknownTypeTag above -- an unvalidated
 *  roundingMode byte one past SR_HALF_AWAY (2) would alias to whatever future
 *  3rd roundingMode_t member gets appended, silently misinterpreting the
 *  op's rounding policy. The type tag is left VALID (ARITH_FLOAT32) so this
 *  isolates the second guard from the first -- if only the type-tag guard
 *  existed, this record would parse the type byte fine and only misbehave
 *  on the untested roundingMode byte. */
static void testDeserializeArithmeticRejectsUnknownRoundingModeTag(void) {
    FILE *f = fopen(FILE_PATH, "wb");
    fwrite("ODTS", 1, 4, f);
    writeU32LE(f, ODTS_VERSION); /* version */
    writeU32LE(f, 1);            /* layerCount */
    writeU8(f, 0);               /* wiring: chain */
    uint8_t tag = (uint8_t)RELU;
    fwrite(&tag, 1, 1, f);
    uint8_t floatType = (uint8_t)ARITH_FLOAT32;
    fwrite(&floatType, 1, 1, f);       /* forwardMath.type -- valid */
    uint8_t badRoundingMode = 2;       /* one past SR_HALF_AWAY (1) -- unknown wire tag */
    fwrite(&badRoundingMode, 1, 1, f); /* forwardMath.roundingMode */
    fwrite(&floatType, 1, 1, f);       /* propLossMath.type */
    uint8_t halfAway = (uint8_t)HALF_AWAY;
    fwrite(&halfAway, 1, 1, f); /* propLossMath.roundingMode */
    uint8_t floatQType = (uint8_t)FLOAT32;
    fwrite(&floatQType, 1, 1, f); /* outputQ dtype */
    fwrite(&floatQType, 1, 1, f); /* propLossQ dtype */
    fclose(f);

    quantization_t *floatQ = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, floatQ);
    layer_t *layer = reluLayerInit(&lq);
    layer_t *model[] = {layer};

    f = fopen(FILE_PATH, "rb");
    ASSERT_EXITS_WITH_OUTPUT(1, "deserializeArithmetic: unknown roundingMode wire tag 2",
                             deserializeModel(&MODEL_CHAIN(model, 1), f));
    fclose(f);

    freeReluLayer(layer);
    freeQuantization(floatQ);
}

/* BATCHNORM1D reader tests (#460, wire format v6). Byte offsets below are
 * named constants derived from the record layout for a FROZEN (no grad
 * tensors -- `.trainable = TRAINABLE_FALSE`), C = 2, affine, tracking
 * BATCHNORM1D record (see Serialize.c's BATCHNORM1D arm and
 * testGoldenBytesModelBatchNorm1dV6 in UnitTestSerialize.c for the same
 * layout pinned byte-for-byte):
 *   header (13: magic, version, layerCount, wiring) + tag (1) = 14
 *                                     -> numChannels u32 LE
 *   14 + 4 = 18                       -> eps f32 LE
 *   18 + 4 = 22                       -> momentumMode u8
 *   22 + 1 = 23                       -> momentum f32 LE
 *   23 + 4 = 27                       -> affine u8
 *   27 + 1 = 28                       -> track u8
 *   29                                -> gamma parameter record (hasGrad=0,
 *                                        rank u32=1, dims[0] u32=2,
 *                                        order[0] u32=0, qtype u8=FLOAT32,
 *                                        2 x f32 data = 1+4+4+4+1+8 = 22
 *                                        bytes for C=2)
 *   29 + 22 = 51                      -> beta parameter record (22 bytes)
 *   51 + 22 = 73                      -> runningMean[0] f32 LE
 *   73 + 8 = 81                       -> runningVar[0] f32 LE
 * Frozen (no grad) keeps these fixed regardless of the seeded param values,
 * exactly as the golden byte test's own comment notes. */
#define BN_OFFSET_EPS 18
#define BN_OFFSET_MODE 22
#define BN_OFFSET_MOMENTUM 23
#define BN_OFFSET_RUNNING_MEAN 73
#define BN_OFFSET_RUNNING_VAR 81

/*! Writes a otherwise-valid v6 BATCHNORM1D record to FILE_PATH: C = 2,
 *  frozen, affine, tracking, eps 0.5, VALUE momentum 0.25 -- the fixture the
 *  reader-rejection tests below patch one field of via fseek + fwrite. */
static void writeValidBatchNorm1dFile(void) {
    quantization_t *floatQ = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, floatQ);
    layer_t *layer =
        batchNorm1dLayerInitOwning(&(batchNorm1dInit_t){.numChannels = 2,
                                                        .eps = 0.5f,
                                                        .momentumMode = BN_MOMENTUM_VALUE,
                                                        .momentum = 0.25f,
                                                        .trainable = TRAINABLE_FALSE},
                                   &lq);

    layer_t *model[] = {layer};
    FILE *f = fopen(FILE_PATH, "wb");
    serializeModel(&MODEL_CHAIN(model, 1), f);
    fclose(f);

    freeBatchNorm1dLayer(layer);
    freeQuantization(floatQ);
}

/*! Writes a MINIMAL (noAffine, noRunningStats) v6 BATCHNORM1D record: no
 *  gamma/beta/buffer bytes at all, so numChannels has NO effect on any other
 *  byte in the record (forwardMath/propLossMath/outputQ/propLossQ follow
 *  immediately after the header fields regardless of C). This isolates the
 *  numChannels equality check from #316-style downstream tensor-shape/dtype
 *  guards that would otherwise catch a mismatch via stream desync instead of
 *  the check itself -- see testDeserializeBatchNormRejectsChannelMismatch's
 *  mutation note. */
static void writeMinimalBatchNorm1dFile(size_t numChannels) {
    quantization_t *floatQ = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, floatQ);
    layer_t *layer = batchNorm1dLayerInitOwning(
        &(batchNorm1dInit_t){.numChannels = numChannels, .noAffine = true, .noRunningStats = true},
        &lq);

    layer_t *model[] = {layer};
    FILE *f = fopen(FILE_PATH, "wb");
    serializeModel(&MODEL_CHAIN(model, 1), f);
    fclose(f);

    freeBatchNorm1dLayer(layer);
    freeQuantization(floatQ);
}

/*! A fresh BATCHNORM1D skeleton to deserialize into -- frozen (matches the
 *  fixture's no-grad record shape) so a structural-mismatch test only ever
 *  exercises the ONE invariant it names. */
static layer_t *makeSkeletonBatchNorm1d(size_t numChannels, bool noAffine, bool noRunningStats) {
    quantization_t *floatQ = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, floatQ);
    layer_t *layer =
        batchNorm1dLayerInitOwning(&(batchNorm1dInit_t){.numChannels = numChannels,
                                                        .noAffine = noAffine,
                                                        .noRunningStats = noRunningStats,
                                                        .trainable = TRAINABLE_FALSE},
                                   &lq);
    freeQuantization(floatQ);
    return layer;
}

static void patchFileU8(long offset, uint8_t value) {
    FILE *f = fopen(FILE_PATH, "r+b");
    TEST_ASSERT_NOT_NULL_MESSAGE(f, "patchFileU8: fopen failed");
    TEST_ASSERT_EQUAL_INT_MESSAGE(0, fseek(f, offset, SEEK_SET), "patchFileU8: fseek failed");
    fwrite(&value, 1, 1, f);
    fclose(f);
}

static void patchFileU32Bits(long offset, uint32_t bits) {
    FILE *f = fopen(FILE_PATH, "r+b");
    TEST_ASSERT_NOT_NULL_MESSAGE(f, "patchFileU32Bits: fopen failed");
    TEST_ASSERT_EQUAL_INT_MESSAGE(0, fseek(f, offset, SEEK_SET), "patchFileU32Bits: fseek failed");
    writeU32LE(f, bits);
    fclose(f);
}

static uint32_t floatBitsOf(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

/*! noAffine + noRunningStats on both sides (see writeMinimalBatchNorm1dFile):
 *  numChannels is otherwise inert in this record shape, so this is the ONE
 *  scenario where removing the numChannels equality check produces a
 *  genuinely SILENT pass (no downstream tensor-shape/dtype guard happens to
 *  catch it) rather than an incidental exit(1) from stream desync -- see the
 *  report's mutation table. */
static void testDeserializeBatchNormRejectsChannelMismatch(void) {
    writeMinimalBatchNorm1dFile(2); /* file numChannels = 2 */
    layer_t *skeleton = makeSkeletonBatchNorm1d(3, true, true);
    layer_t *model[] = {skeleton};

    FILE *f = fopen(FILE_PATH, "rb");
    ASSERT_EXITS_WITH_OUTPUT(
        1, "BATCHNORM1D record (C 2, affine 0, track 0) does not match the skeleton (C 3",
        deserializeModel(&MODEL_CHAIN(model, 1), f));
    fclose(f);

    freeBatchNorm1dLayer(skeleton);
}

static void testDeserializeBatchNormRejectsAffineMismatch(void) {
    writeValidBatchNorm1dFile(); /* file affine = true */
    layer_t *skeleton = makeSkeletonBatchNorm1d(2, /*noAffine=*/true, /*noRunningStats=*/false);
    layer_t *model[] = {skeleton};

    FILE *f = fopen(FILE_PATH, "rb");
    ASSERT_EXITS_WITH_OUTPUT(
        1, "BATCHNORM1D record (C 2, affine 1, track 1) does not match the skeleton (C 2, affine 0",
        deserializeModel(&MODEL_CHAIN(model, 1), f));
    fclose(f);

    freeBatchNorm1dLayer(skeleton);
}

static void testDeserializeBatchNormRejectsNonFiniteEps(void) {
    writeValidBatchNorm1dFile();
    patchFileU32Bits(BN_OFFSET_EPS, 0x7FC00000u); /* quiet NaN */
    layer_t *skeleton = makeSkeletonBatchNorm1d(2, false, false);
    layer_t *model[] = {skeleton};

    FILE *f = fopen(FILE_PATH, "rb");
    ASSERT_EXITS_WITH_OUTPUT(1, "BATCHNORM1D eps must be finite",
                             deserializeModel(&MODEL_CHAIN(model, 1), f));
    fclose(f);

    freeBatchNorm1dLayer(skeleton);
}

static void testDeserializeBatchNormRejectsMomentumOutOfRange(void) {
    writeValidBatchNorm1dFile();
    patchFileU32Bits(BN_OFFSET_MOMENTUM, floatBitsOf(2.0f)); /* outside [0, 1] */
    layer_t *skeleton = makeSkeletonBatchNorm1d(2, false, false);
    layer_t *model[] = {skeleton};

    FILE *f = fopen(FILE_PATH, "rb");
    ASSERT_EXITS_WITH_OUTPUT(1, "BATCHNORM1D momentum must be finite",
                             deserializeModel(&MODEL_CHAIN(model, 1), f));
    fclose(f);

    freeBatchNorm1dLayer(skeleton);
}

static void testDeserializeBatchNormRejectsUnknownMomentumMode(void) {
    writeValidBatchNorm1dFile();
    patchFileU8(BN_OFFSET_MODE, 0); /* neither VALUE (1) nor CUMULATIVE (2) */
    layer_t *skeleton = makeSkeletonBatchNorm1d(2, false, false);
    layer_t *model[] = {skeleton};

    FILE *f = fopen(FILE_PATH, "rb");
    ASSERT_EXITS_WITH_OUTPUT(1, "BATCHNORM1D momentumMode 0 unknown",
                             deserializeModel(&MODEL_CHAIN(model, 1), f));
    fclose(f);

    freeBatchNorm1dLayer(skeleton);
}

static void testDeserializeBatchNormRejectsNaNRunningMean(void) {
    writeValidBatchNorm1dFile();
    patchFileU32Bits(BN_OFFSET_RUNNING_MEAN, 0x7FC00000u); /* quiet NaN */
    layer_t *skeleton = makeSkeletonBatchNorm1d(2, false, false);
    layer_t *model[] = {skeleton};

    FILE *f = fopen(FILE_PATH, "rb");
    ASSERT_EXITS_WITH_OUTPUT(1, "BATCHNORM1D runningMean[0] not finite",
                             deserializeModel(&MODEL_CHAIN(model, 1), f));
    fclose(f);

    freeBatchNorm1dLayer(skeleton);
}

static void testDeserializeBatchNormRejectsNegativeRunningVar(void) {
    writeValidBatchNorm1dFile();
    patchFileU32Bits(BN_OFFSET_RUNNING_VAR, floatBitsOf(-1.0f));
    layer_t *skeleton = makeSkeletonBatchNorm1d(2, false, false);
    layer_t *model[] = {skeleton};

    FILE *f = fopen(FILE_PATH, "rb");
    ASSERT_EXITS_WITH_OUTPUT(1, "BATCHNORM1D runningVar[0] must be finite",
                             deserializeModel(&MODEL_CHAIN(model, 1), f));
    fclose(f);

    freeBatchNorm1dLayer(skeleton);
}

/*! Review Focus 3: a skeleton built with DEFAULT hyperparameters (eps 0 ->
 *  1e-5, momentumMode DEFAULT -> VALUE 0.1) must end up with the FILE's own
 *  eps/momentum after a successful load -- the file's hyperparameters win
 *  over whatever the skeleton was constructed with, matching every other
 *  overwrite-from-file field. numChannels/affine/track are identical on
 *  both sides so the record loads cleanly. */
static void testDeserializeBatchNormFileHyperparametersWin(void) {
    quantization_t *floatQ = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, floatQ);

    layer_t *sourceLayer = batchNorm1dLayerInitOwning(
        &(batchNorm1dInit_t){
            .numChannels = 2, .eps = 1e-3f, .momentumMode = BN_MOMENTUM_VALUE, .momentum = 0.3f},
        &lq);
    layer_t *sourceModel[] = {sourceLayer};
    FILE *f = fopen(FILE_PATH, "wb");
    serializeModel(&MODEL_CHAIN(sourceModel, 1), f);
    fclose(f);
    freeBatchNorm1dLayer(sourceLayer);

    layer_t *skeletonLayer =
        batchNorm1dLayerInitOwning(&(batchNorm1dInit_t){.numChannels = 2}, &lq);
    batchNorm1dConfig_t *cfg = skeletonLayer->config->batchNorm1d;
    layer_t *skeletonModel[] = {skeletonLayer};

    f = fopen(FILE_PATH, "rb");
    deserializeModel(&MODEL_CHAIN(skeletonModel, 1), f);
    fclose(f);

    /* CAPTURE every assertion value before any free. */
    float capturedEps = cfg->eps;
    float capturedMomentum = cfg->momentum;

    freeBatchNorm1dLayer(skeletonLayer);
    freeQuantization(floatQ);

    /* ASSERT on captured. */
    TEST_ASSERT_EQUAL_FLOAT(1e-3f, capturedEps);
    TEST_ASSERT_EQUAL_FLOAT(0.3f, capturedMomentum);
}

/* The BATCHNORM1D reader decoded forwardMath/
 * propLossMath (deserializeArithmetic already range-checks the wire tag
 * against the arithmeticType_t enum) but never required FLOAT32 -- unlike
 * the factory (BatchNorm1dApi.c's validateLayerQuantForBatchNorm1d), which
 * rejects a non-FLOAT32 forwardMath/propLossMath at construction time. A
 * loaded file could carry a non-FLOAT32-tagged BN that only the forward/
 * backward FLOAT32-only guards would reject, at CALL time, instead of at
 * LOAD time (the deserialize-time trust boundary every other file-derived
 * BN field is checked at, per Deserialize.c's BATCHNORM1D arm).
 *
 * BN_OFFSET_FORWARD_MATH_TYPE derivation, continuing the layout comment
 * above (writeValidBatchNorm1dFile: C = 2, frozen, affine, tracking):
 *   BN_OFFSET_RUNNING_VAR (81) + runningVar[C=2] (2 x f32 = 8) -> 89
 *                                              numBatchesTracked u64 (8)
 *   89 + 8                                  -> 97  forwardMath.type u8
 *   97 + 1                                  -> 98  forwardMath.roundingMode u8
 *   98 + 1                                  -> 99  propLossMath.type u8
 * (serializeArithmetic/deserializeArithmetic: type u8 then roundingMode u8,
 * Serialize.c:143-146 / Deserialize.c:212-227). */
#define BN_OFFSET_FORWARD_MATH_TYPE 97

static void testDeserializeBatchNormRejectsNonFloat32ForwardMath(void) {
    writeValidBatchNorm1dFile();
    patchFileU8(BN_OFFSET_FORWARD_MATH_TYPE, (uint8_t)ARITH_SYM_INT32); /* valid tag, not FLOAT32 */
    layer_t *skeleton = makeSkeletonBatchNorm1d(2, false, false);
    layer_t *model[] = {skeleton};

    FILE *f = fopen(FILE_PATH, "rb");
    ASSERT_EXITS_WITH_OUTPUT(1, "BATCHNORM1D forwardMath must be ARITH_FLOAT32",
                             deserializeModel(&MODEL_CHAIN(model, 1), f));
    fclose(f);

    freeBatchNorm1dLayer(skeleton);
}

/* The GROUPNORM/LAYERNORM records overwrite the layer geometry, so the reader re-validates the
 * loaded config. Each fixture serializes a factory-built layer whose geometry was edited after
 * construction, then loads it into a clean skeleton of the original geometry. */
static void expectGroupNormLoadRejected(size_t numGroups, size_t numChannels) {
    quantization_t *floatQ = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, floatQ);
    groupNormInit_t init = {.numGroups = 2, .numChannels = 4, .trainable = TRAINABLE_FALSE};
    layer_t *layer = groupNormLayerInitOwning(&init, &lq);
    layer->config->groupNorm->numGroups = numGroups;
    layer->config->groupNorm->numChannels = numChannels;
    layer_t *model[] = {layer};
    FILE *f = fopen(FILE_PATH, "wb");
    serializeModel(&MODEL_CHAIN(model, 1), f);
    fclose(f);
    freeGroupNormLayer(layer);

    layer = groupNormLayerInitOwning(&init, &lq);
    model[0] = layer;
    f = fopen(FILE_PATH, "rb");
    ASSERT_EXITS_WITH_OUTPUT(1, "deserializeModel: GROUPNORM",
                             deserializeModel(&MODEL_CHAIN(model, 1), f));
    fclose(f);
    freeGroupNormLayer(layer);
    freeQuantization(floatQ);
}

static void testDeserializeGroupNormRejectsIncoherentGeometry(void) {
    expectGroupNormLoadRejected(3, 4); /* 3 groups do not divide 4 channels */
    expectGroupNormLoadRejected(2, 8); /* 8 channels against the skeleton's 4 gamma/beta values */
}

static void testDeserializeLayerNormRejectsNormalizedShapeGammaMismatch(void) {
    quantization_t *floatQ = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, floatQ);
    layerNormInit_t init = {
        .normalizedShape = (size_t[]){4}, .numNormDims = 1, .trainable = TRAINABLE_FALSE};
    layer_t *layer = layerNormLayerInitOwning(&init, &lq);
    layer->config->layerNorm->normalizedShape[0] = 2; /* gamma/beta keep 4 values */
    layer_t *model[] = {layer};
    FILE *f = fopen(FILE_PATH, "wb");
    serializeModel(&MODEL_CHAIN(model, 1), f);
    fclose(f);
    freeLayerNormLayer(layer);

    layer = layerNormLayerInitOwning(&init, &lq);
    model[0] = layer;
    f = fopen(FILE_PATH, "rb");
    ASSERT_EXITS_WITH_OUTPUT(1, "deserializeModel: LAYERNORM",
                             deserializeModel(&MODEL_CHAIN(model, 1), f));
    fclose(f);
    freeLayerNormLayer(layer);
    freeQuantization(floatQ);
}

/* The SOFTMAX record carries both maths but the skeleton fixes the wire dtypes, so the reader
 * re-validates their coherence: the file pairs ARITH_BFP math with the skeleton's FLOAT32 wire. */
static void expectSoftmaxLoadRejected(bool backward, const char *message) {
    quantization_t *floatQ = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, floatQ);
    layer_t *layer = softmaxLayerInitOwning(&lq);
    arithmetic_t *math =
        backward ? &layer->config->softmax->propLossMath : &layer->config->softmax->forwardMath;
    math->type = ARITH_BFP;
    layer_t *model[] = {layer};
    FILE *f = fopen(FILE_PATH, "wb");
    serializeModel(&MODEL_CHAIN(model, 1), f);
    fclose(f);
    freeSoftmaxLayer(layer);

    layer = softmaxLayerInitOwning(&lq);
    model[0] = layer;
    f = fopen(FILE_PATH, "rb");
    ASSERT_EXITS_WITH_OUTPUT(1, message, deserializeModel(&MODEL_CHAIN(model, 1), f));
    fclose(f);
    freeSoftmaxLayer(layer);
    freeQuantization(floatQ);
}

static void testDeserializeSoftmaxRejectsBfpMathOverANonBfpWire(void) {
    expectSoftmaxLoadRejected(false, "deserializeModel: SOFTMAX: ARITH_BFP forwardMath");
    expectSoftmaxLoadRejected(true, "deserializeModel: SOFTMAX: ARITH_BFP propLossMath");
}

/* The DROPOUT record overwrites p, which the scale 1/(1-p) needs in [0, 1); NaN included. */
static void testDeserializeDropoutRejectsAnOutOfRangeP(void) {
    const float bad[] = {1.0f, NAN};
    size_t dims[] = {4};
    size_t order[] = {0};
    shape_t shape = {.numberOfDimensions = 1, .dimensions = dims, .orderOfDimensions = order};
    quantization_t boolQ;
    initBoolQuantization(&boolQ);
    uint8_t maskBits[1] = {0};
    tensor_t mask;
    setTensorValues(&mask, maskBits, &shape, &boolQ, NULL);
    quantization_t *floatQ = quantizationInitFloat();
    for (size_t i = 0; i < 2; i++) {
        layer_t *layer = dropoutLayerInit(0.5f, &mask, floatQ, floatQ);
        layer->config->dropout->p = bad[i];
        layer_t *model[] = {layer};
        FILE *f = fopen(FILE_PATH, "wb");
        serializeModel(&MODEL_CHAIN(model, 1), f);
        fclose(f);
        layer->config->dropout->p = 0.5f;
        f = fopen(FILE_PATH, "rb");
        ASSERT_EXITS_WITH_OUTPUT(1, "deserializeModel: DROPOUT: p must be in [0, 1)",
                                 deserializeModel(&MODEL_CHAIN(model, 1), f));
        fclose(f);
        freeDropoutLayer(layer);
    }
    freeQuantization(floatQ);
}

/* The CONV1D_TRANSPOSED record overwrites the kernel, and the ConvTranspose kernels
 * implement VALID geometry only. */
static void testDeserializeConv1dTransposedRejectsNonValidPadding(void) {
    quantization_t *floatQ = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, floatQ);
    const rngKey_t initKey = rngFold(rngKeyFromSeed(1u), RNG_TAG_INIT, 0u);
    conv1dTransposedInit_t init = {
        .inChannels = 2, .outChannels = 2, .kernelSize = 3, .weightInit.key = &initKey};
    layer_t *layer = conv1dTransposedLayerInitOwning(&init, &lq);
    layer->config->conv1dTransposed->kernel->paddingType = SAME;
    layer_t *model[] = {layer};
    FILE *f = fopen(FILE_PATH, "wb");
    serializeModel(&MODEL_CHAIN(model, 1), f);
    fclose(f);
    layer->config->conv1dTransposed->kernel->paddingType = VALID;

    f = fopen(FILE_PATH, "rb");
    ASSERT_EXITS_WITH_OUTPUT(
        1, "deserializeModel: CONV1D_TRANSPOSED: only VALID paddingType is supported",
        deserializeModel(&MODEL_CHAIN(model, 1), f));
    fclose(f);
    freeConv1dTransposedLayer(layer);
    freeQuantization(floatQ);
}

#define BN_OFFSET_PROP_LOSS_MATH_TYPE (BN_OFFSET_FORWARD_MATH_TYPE + 2)

static void testDeserializeBatchNormRejectsNonFloat32PropLossMath(void) {
    writeValidBatchNorm1dFile();
    patchFileU8(BN_OFFSET_PROP_LOSS_MATH_TYPE, (uint8_t)ARITH_SYM_INT32);
    layer_t *skeleton = makeSkeletonBatchNorm1d(2, false, false);
    layer_t *model[] = {skeleton};

    FILE *f = fopen(FILE_PATH, "rb");
    ASSERT_EXITS_WITH_OUTPUT(1, "propLossMath must be ARITH_FLOAT32",
                             deserializeModel(&MODEL_CHAIN(model, 1), f));
    fclose(f);

    freeBatchNorm1dLayer(skeleton);
}

/* Serializes src to FILE_PATH and reopens it for reading; *fileLength gets
 * the on-disk size so a test can check readTensorRecordHeader stopped
 * exactly at the payload. */
static FILE *writeAndReopenRecord(tensor_t *src, long *fileLength) {
    FILE *f = fopen(FILE_PATH, "wb");
    serializeTensor(src, f);
    fclose(f);
    f = fopen(FILE_PATH, "rb");
    fseek(f, 0, SEEK_END);
    *fileLength = ftell(f);
    rewind(f);
    return f;
}

/* Review E #3: a stored BFP exponent must fit its own exponent width too, not
 * only the finite-scale cap bias + 127 (which is 128 at e2, far above the
 * 2-bit maximum 3). One-element per-tensor record written by hand:
 * rank 1, dim 1, order 0, BFP {1, 0}, the exponent byte, m 8, e, HALF_AWAY. */
static void writeBfpExponentRecord(uint8_t stored, uint8_t exponentBits) {
    FILE *f = fopen(FILE_PATH, "wb");
    writeU32LE(f, 1);
    writeU32LE(f, 1);
    writeU32LE(f, 0);
    uint8_t bytes[] = {(uint8_t)BFP};
    fwrite(bytes, 1, 1, f);
    writeU32LE(f, 1);
    writeU32LE(f, 0);
    uint8_t tail[] = {stored, 8, exponentBits, 0, 1 /* payload: code 1 */};
    fwrite(tail, 1, sizeof(tail), f);
    fclose(f);
}

static void readBfpExponentRecord(uint8_t stored, uint8_t exponentBits) {
    writeBfpExponentRecord(stored, exponentBits);
    FILE *f = fopen(FILE_PATH, "rb");
    tensorRecordHeader_t header;
    readTensorRecordHeader(&header, f);
    fclose(f);
    freeTensorRecordHeader(&header);
}

static void testBfpRecordRejectsAnExponentWiderThanItsWidth(void) {
    readBfpExponentRecord(3, 2);   /* the 2-bit maximum */
    readBfpExponentRecord(7, 3);   /* the 3-bit maximum */
    readBfpExponentRecord(254, 8); /* e8: bias + 127, the largest finite scale */
    const char *msg = "BFP file exponent[0]";
    ASSERT_EXITS_WITH_OUTPUT(1, msg, readBfpExponentRecord(4, 2));
    ASSERT_EXITS_WITH_OUTPUT(1, msg, readBfpExponentRecord(8, 3));
    ASSERT_EXITS_WITH_OUTPUT(1, msg, readBfpExponentRecord(16, 4));
    ASSERT_EXITS_WITH_OUTPUT(1, msg, readBfpExponentRecord(255, 8));
}

static void testReadTensorRecordHeaderFloat32(void) {
    float data[] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    tensor_t *src = makeFloatTensor2D(2, 3, data, 6);
    long fileLength = 0;
    FILE *f = writeAndReopenRecord(src, &fileLength);

    tensorRecordHeader_t header;
    readTensorRecordHeader(&header, f);
    long position = ftell(f);
    fclose(f);
    freeTensorRecordHeader(&header);
    freeTensor(src);

    TEST_ASSERT_EQUAL_size_t(2, header.numberOfDimensions);
    TEST_ASSERT_EQUAL_size_t(2, header.dimensions[0]);
    TEST_ASSERT_EQUAL_size_t(3, header.dimensions[1]);
    TEST_ASSERT_EQUAL_size_t(6, header.numberOfElements);
    TEST_ASSERT_EQUAL_INT(FLOAT32, header.quantization.type);
    TEST_ASSERT_EQUAL_size_t(24, header.payloadBytes);
    TEST_ASSERT_EQUAL_INT64(fileLength - 24, position);
}

static void testReadTensorRecordHeaderGroupedSym(void) {
    tensor_t *src = makeSymGroupedTensor1D(6, 4, HALF_AWAY, 3, 2);
    size_t expectedPayload = calcNumberOfBytesForData(src->quantization, 6);
    long fileLength = 0;
    FILE *f = writeAndReopenRecord(src, &fileLength);

    tensorRecordHeader_t header;
    readTensorRecordHeader(&header, f);
    long position = ftell(f);
    fclose(f);
    size_t numGroups = header.qConfigStorage.sym.numGroups;
    size_t groupSize = header.qConfigStorage.sym.groupSize;
    uint8_t qBits = header.qConfigStorage.sym.qBits;
    freeTensorRecordHeader(&header);
    freeTensor(src);

    TEST_ASSERT_EQUAL_INT(SYM, header.quantization.type);
    TEST_ASSERT_EQUAL_size_t(6, header.numberOfElements);
    TEST_ASSERT_EQUAL_size_t(3, numGroups);
    TEST_ASSERT_EQUAL_size_t(2, groupSize);
    TEST_ASSERT_EQUAL_UINT8(4, qBits);
    TEST_ASSERT_EQUAL_size_t(expectedPayload, header.payloadBytes);
    TEST_ASSERT_EQUAL_INT64(fileLength - (long)expectedPayload, position);
}

static void testReadTensorRecordHeaderGroupedAsym(void) {
    tensor_t *src = makeAsymGroupedTensor1D(6, 8, HALF_AWAY, 3, 2);
    size_t expectedPayload = calcNumberOfBytesForData(src->quantization, 6);
    long fileLength = 0;
    FILE *f = writeAndReopenRecord(src, &fileLength);

    tensorRecordHeader_t header;
    readTensorRecordHeader(&header, f);
    long position = ftell(f);
    fclose(f);
    size_t numGroups = header.qConfigStorage.asym.numGroups;
    size_t groupSize = header.qConfigStorage.asym.groupSize;
    uint8_t qBits = header.qConfigStorage.asym.qBits;
    freeTensorRecordHeader(&header);
    freeTensor(src);

    TEST_ASSERT_EQUAL_INT(ASYM, header.quantization.type);
    TEST_ASSERT_EQUAL_size_t(3, numGroups);
    TEST_ASSERT_EQUAL_size_t(2, groupSize);
    TEST_ASSERT_EQUAL_UINT8(8, qBits);
    TEST_ASSERT_EQUAL_size_t(expectedPayload, header.payloadBytes);
    TEST_ASSERT_EQUAL_INT64(fileLength - (long)expectedPayload, position);
}

static void testReadTensorRecordHeaderGroupedBfp(void) {
    tensor_t *src = makeBfpGroupedTensor1D(6, 4, 6, HALF_AWAY, 3, 2);
    size_t expectedPayload = calcNumberOfBytesForData(src->quantization, 6);
    long fileLength = 0;
    FILE *f = writeAndReopenRecord(src, &fileLength);

    tensorRecordHeader_t header;
    readTensorRecordHeader(&header, f);
    long position = ftell(f);
    fclose(f);
    size_t numGroups = header.qConfigStorage.bfp.numGroups;
    uint8_t mantissaBits = header.qConfigStorage.bfp.mantissaBits;
    uint8_t exponentBits = header.qConfigStorage.bfp.exponentBits;
    freeTensorRecordHeader(&header);
    freeTensor(src);

    TEST_ASSERT_EQUAL_INT(BFP, header.quantization.type);
    TEST_ASSERT_EQUAL_size_t(3, numGroups);
    TEST_ASSERT_EQUAL_UINT8(4, mantissaBits);
    TEST_ASSERT_EQUAL_UINT8(6, exponentBits);
    TEST_ASSERT_EQUAL_size_t(expectedPayload, header.payloadBytes);
    TEST_ASSERT_EQUAL_INT64(fileLength - (long)expectedPayload, position);
}

/*! Well-formed rank-9 record, so only the cap (not a short read) can stop it. */
static void testReadTensorRecordHeaderRejectsRankAboveCap(void) {
    FILE *f = fopen(FILE_PATH, "wb");
    const uint32_t oversizedRank = ODTS_TENSOR_RECORD_MAX_RANK + 1u;
    writeU32LE(f, oversizedRank);
    for (uint32_t d = 0; d < oversizedRank; d++) {
        writeU32LE(f, 1);
    }
    for (uint32_t d = 0; d < oversizedRank; d++) {
        writeU32LE(f, d);
    }
    uint8_t floatType = (uint8_t)FLOAT32;
    fwrite(&floatType, 1, 1, f);
    writeF32LE(f, 0.f);
    fclose(f);

    tensorRecordHeader_t header;
    f = fopen(FILE_PATH, "rb");
    ASSERT_EXITS_WITH_OUTPUT(1, "exceeds the 8-dim cap", readTensorRecordHeader(&header, f));
    fclose(f);
}

/* Two FLOAT32 ReLUs wired so buffer 1 has no consumer: a wiring the chain loop would misread. */
static const uint16_t g_skipStart[3] = {0, 1, 2}, g_skipIds[2] = {0, 0};

/* The stream is a valid chain serialization of the same layers, so only the wiring can refuse it.
 */
static void testDeserializeModelValidatesTheWiring(void) {
    quantization_t *floatQ = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, floatQ);
    layer_t *layers[2] = {reluLayerInit(&lq), reluLayerInit(&lq)};
    FILE *f = tmpfile();
    TEST_ASSERT_NOT_NULL(f);
    serializeModel(&MODEL_CHAIN(layers, 2), f);
    rewind(f);

    ASSERT_EXITS_WITH_OUTPUT(1, "deserializeModel: wiring: buffer 1 has no consumer",
                             deserializeModel(&(model_t){layers, 2, g_skipStart, g_skipIds}, f));

    fclose(f);
    freeReluLayer(layers[1]);
    freeReluLayer(layers[0]);
    freeQuantization(floatQ);
}

static size_t serializeToBuffer(const model_t *m, uint8_t *buffer, size_t capacity) {
    FILE *f = tmpfile();
    TEST_ASSERT_NOT_NULL(f);
    serializeModel(m, f);
    rewind(f);
    size_t length = fread(buffer, 1, capacity, f);
    fclose(f);
    TEST_ASSERT_TRUE(length < capacity);
    return length;
}

static FILE *streamOf(const uint8_t *bytes, size_t length) {
    FILE *f = tmpfile();
    TEST_ASSERT_NOT_NULL(f);
    fwrite(bytes, 1, length, f);
    rewind(f);
    return f;
}

static void testDeserializeRejectsADifferentlyWiredSkeleton(void) {
    dagFixture_t f3, f3r;
    dagBuildF3(&f3, false);
    dagBuildF3r(&f3r);
    FILE *f = tmpfile();
    TEST_ASSERT_NOT_NULL(f);
    serializeModel(&f3.m, f);
    rewind(f);

    ASSERT_EXITS_WITH_OUTPUT(
        1, "deserializeModel: wiring mismatch on 'inputIds[5]' (file 1, skeleton 2)",
        deserializeModel(&f3r.m, f));

    fclose(f);
    dagFree(&f3r);
    dagFree(&f3);
}

/* R's file (inputStart {0, 1, 3}) into ADD(b0, b0) -> ReLU(b1) (inputStart {0, 2, 3}). */
static void testDeserializeRejectsADifferentInputStart(void) {
    dagFixture_t r, swapped;
    dagBuildR(&r);
    uint8_t bytes[64];
    size_t length = serializeToBuffer(&r.m, bytes, sizeof(bytes));
    dagBegin(&swapped);
    uint16_t b1 = graphMerge(&swapped.g, addLayerInit(NULL), GRAPH_INPUT, GRAPH_INPUT);
    (void)graphNode(&swapped.g, dagRelu(), b1);
    swapped.m = graphFinish(&swapped.g);
    FILE *f = streamOf(bytes, length);

    ASSERT_EXITS_WITH_OUTPUT(
        1, "deserializeModel: wiring mismatch on 'inputStart[1]' (file 1, skeleton 2)",
        deserializeModel(&swapped.m, f));

    fclose(f);
    dagFree(&swapped);
    dagFree(&r);
}

/* Cut after magic, version, count, kind and one and a half inputStart entries. */
static void testDeserializeRejectsATruncatedWiringRecord(void) {
    dagFixture_t r;
    dagBuildR(&r);
    uint8_t bytes[64];
    (void)serializeToBuffer(&r.m, bytes, sizeof(bytes));
    FILE *cut = streamOf(bytes, 4 + 4 + 4 + 1 + 3);

    ASSERT_EXITS_WITH_OUTPUT(1, "short read (2 bytes): truncated or corrupt stream",
                             deserializeModel(&r.m, cut));

    fclose(cut);
    dagFree(&r);
}

/* The FLOAT32 merge gate trusts addLayerInit, so a file must not install what the factory refuses.
 * R's ADD record starts at byte 32: tag, forwardMath type, rounding, propLossMath type. */
static void testDeserializeAddRejectsNonFloat32Math(void) {
    dagFixture_t r;
    dagBuildR(&r);
    uint8_t bytes[64];
    size_t length = serializeToBuffer(&r.m, bytes, sizeof(bytes));
    TEST_ASSERT_EQUAL_HEX8((uint8_t)ADD, bytes[32]);

    bytes[33] = (uint8_t)ARITH_SYM_INT32;
    FILE *f = streamOf(bytes, length);
    ASSERT_EXITS_WITH_OUTPUT(1, "deserializeModel: ADD forwardMath must be ARITH_FLOAT32 (got 1)",
                             deserializeModel(&r.m, f));
    fclose(f);

    bytes[33] = (uint8_t)ARITH_FLOAT32;
    bytes[35] = (uint8_t)ARITH_BFP;
    f = streamOf(bytes, length);
    ASSERT_EXITS_WITH_OUTPUT(1, "deserializeModel: ADD propLossMath must be ARITH_FLOAT32 (got 2)",
                             deserializeModel(&r.m, f));
    fclose(f);

    dagFree(&r);
}

/* A complete rank-1, one-element SYM_INT32 record whose scale field is
 * {m, e}, read back through readTensorRecordHeader. */
static odtScale_t readSymInt32RecordScale(uint16_t m, int16_t e) {
    FILE *f = fopen(FILE_PATH, "wb");
    writeU32LE(f, 1); /* numberOfDimensions */
    writeU32LE(f, 1); /* dimensions[0] */
    writeU32LE(f, 0); /* orderOfDimensions[0] */
    uint8_t type = (uint8_t)SYM_INT32;
    fwrite(&type, 1, 1, f);
    writeScaleLE(f, m, e);
    uint8_t tail[2] = {(uint8_t)HALF_AWAY, 12}; /* roundingMode, qMaxBits */
    fwrite(tail, 1, 2, f);
    writeU32LE(f, 0); /* payload: one int32 */
    fclose(f);

    tensorRecordHeader_t header;
    f = fopen(FILE_PATH, "rb");
    readTensorRecordHeader(&header, f);
    fclose(f);
    odtScale_t s = odtScaleLoad(&header.qConfigStorage.symInt32.scale);
    freeTensorRecordHeader(&header);
    return s;
}

/*! A v9 scale field is accepted only in canonical form: m in [16384, 32767]
 *  and e in [ODT_SCALE_E_MIN, ODT_SCALE_E_MAX]. Each bound is probed one past
 *  its edge (rejected) and at its edge (accepted, field for field). */
static void testDeserializeRejectsInvalidDyadicScale(void) {
    ASSERT_EXITS_WITH_OUTPUT(1, "scale mantissa 16383", (void)readSymInt32RecordScale(16383, 0));
    ASSERT_EXITS_WITH_OUTPUT(1, "scale mantissa 32768", (void)readSymInt32RecordScale(32768, 0));
    ASSERT_EXITS_WITH_OUTPUT(1, "scale mantissa 0", (void)readSymInt32RecordScale(0, 0));
    ASSERT_EXITS_WITH_OUTPUT(1, "scale exponent -141",
                             (void)readSymInt32RecordScale(16384, ODT_SCALE_E_MIN - 1));
    ASSERT_EXITS_WITH_OUTPUT(1, "scale exponent 114",
                             (void)readSymInt32RecordScale(16384, ODT_SCALE_E_MAX + 1));

    odtScale_t low = readSymInt32RecordScale(ODT_SCALE_M_MIN, ODT_SCALE_E_MIN);
    odtScale_t high = readSymInt32RecordScale(ODT_SCALE_M_MAX, ODT_SCALE_E_MAX);
    TEST_ASSERT_EQUAL_UINT16(ODT_SCALE_M_MIN, low.m);
    TEST_ASSERT_EQUAL_INT16(ODT_SCALE_E_MIN, low.e);
    TEST_ASSERT_EQUAL_UINT16(ODT_SCALE_M_MAX, high.m);
    TEST_ASSERT_EQUAL_INT16(ODT_SCALE_E_MAX, high.e);
}

void setUp() {}
void tearDown() {}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testSerializeAndDeserializeTensor);
    RUN_TEST(testDeserializeRejectsBadMagic);
    RUN_TEST(testDeserializeRejectsUnsupportedVersions);
    RUN_TEST(testDeserializeRejectsV7Version);
    RUN_TEST(testDeserializeRejectsV8Version);
    RUN_TEST(testDeserializeRejectsWiringKindMismatch);
    RUN_TEST(testDeserializeRejectsLayerCountMismatch);
    RUN_TEST(testDeserializeRejectsTagMismatch);
    RUN_TEST(testDeserializeTensorRejectsDtypeMismatch);
    RUN_TEST(testDeserializeTensorRejectsPayloadSizeMismatch);
    RUN_TEST(testDeserializeTensorRejectsSymRecordIntoFloatSkeleton);
#if SIZE_MAX == UINT64_MAX
    RUN_TEST(testDeserializeTensorRejectsWrappingDims);
#endif
    RUN_TEST(testDeserializeModelFailsFastOnTruncatedHeader);
    RUN_TEST(testDeserializeTensorFailsFastOnTruncatedPayload);
    RUN_TEST(testDeserializeTensorRejectsRankMismatch);
    RUN_TEST(testDeserializeTensorRejectsOutOfRangeOrderOfDimensions);
    RUN_TEST(testDeserializeTensorRejectsDuplicateOrderOfDimensions);
    RUN_TEST(testDeserializeTensorAcceptsTransposedOrderOfDimensions);
    RUN_TEST(testSkipSerializedTensorRejectsDuplicateOrderOfDimensions);
    RUN_TEST(testDeserializeParameterRejectsHasGradAboveOne);
    RUN_TEST(testDeserializeTensorRoundTripsAsymZeroPoint);
    RUN_TEST(testDeserializeQConfigAcceptsWideZeroPoint);
    RUN_TEST(testDeserializeQConfigAcceptsNumGroupsMismatchViaReallocation);
    RUN_TEST(testDeserializeGroupedSymRejectsBadDivisibility);
    RUN_TEST(testDeserializeQConfigRejectsSentinelViolation);
    RUN_TEST(testBfpRoundTripGrouped);
    RUN_TEST(testBfpDeserializeReallocatesExponentsOnShapeChange);
    RUN_TEST(testBfpDeserializeRejectsZeroNumGroupsInWireConfig);
    RUN_TEST(testBfpDeserializeRejectsSentinelViolation);
    RUN_TEST(testBfpDeserializeRejectsMantissaBitsOutOfRange);
    RUN_TEST(testBfpDeserializeRejectsExponentBitsOutOfRange);
    RUN_TEST(testBfpDeserializeRejectsNonFiniteScaleExponent);
    RUN_TEST(testBfpRecordRejectsAnExponentWiderThanItsWidth);
    RUN_TEST(testDeserializeWeightsOnlyIntoTrainableSkeleton);
    RUN_TEST(testSkipSerializedTensorRejectsRankAboveCap);
    RUN_TEST(testSkipSerializedTensorRejectsTruncatedPayload);
#if SIZE_MAX == UINT64_MAX
    RUN_TEST(testSkipSerializedTensorRejectsWrappingDims);
#endif
#if SIZE_MAX == UINT64_MAX && LONG_MAX == INT64_MAX
    RUN_TEST(testSkipSerializedTensorRejectsPayloadBeyondLongMax);
#endif
    RUN_TEST(testSkipSerializedGroupedSymGrad);
    RUN_TEST(testDeserializeSymRejectsOversizedNumGroupsInWireConfig);
    RUN_TEST(testDeserializeAsymRejectsOversizedNumGroupsInWireConfig);
    RUN_TEST(testDeserializeSymRejectsZeroNumGroupsInWireConfig);
    RUN_TEST(testDeserializeAsymRejectsZeroNumGroupsInWireConfig);
    RUN_TEST(testDeserializeAsymRejectsWireQBitsAboveSixteen);
    RUN_TEST(testDeserializeSymRejectsSentinelViolationInWireConfig);
    RUN_TEST(testDeserializeSymRejectsWireQBitsOutsideRange);
    RUN_TEST(testDeserializeSymAcceptsWireQBitsBoundsInWireConfig);
    RUN_TEST(testDeserializeGroupedAsymRejectsBadDivisibility);
    RUN_TEST(testDeserializeParameterRejectsGroupedSymGrad);
    RUN_TEST(testDeserializeParameterRejectsGroupedAsymGrad);
    RUN_TEST(testDeserializeParameterLoadsPerTensorSymGrad);
    RUN_TEST(testDeserializeGroupedAsymIntoPerTensorSkeleton);
    RUN_TEST(testSkipSerializedGroupedAsymGrad);
    RUN_TEST(testSkipSerializedGroupedAsymGradRejectsBadDivisibility);
    RUN_TEST(testSkipSerializedGroupedSymGradRejectsBadDivisibility);
    RUN_TEST(testDeserializeArithmeticRoundTripsBfp);
    RUN_TEST(testDeserializeArithmeticRejectsUnknownTypeTag);
    RUN_TEST(testDeserializeArithmeticRejectsUnknownRoundingModeTag);
    RUN_TEST(testDeserializeBatchNormRejectsChannelMismatch);
    RUN_TEST(testDeserializeBatchNormRejectsAffineMismatch);
    RUN_TEST(testDeserializeBatchNormRejectsNonFiniteEps);
    RUN_TEST(testDeserializeBatchNormRejectsMomentumOutOfRange);
    RUN_TEST(testDeserializeBatchNormRejectsUnknownMomentumMode);
    RUN_TEST(testDeserializeBatchNormRejectsNaNRunningMean);
    RUN_TEST(testDeserializeBatchNormRejectsNegativeRunningVar);
    RUN_TEST(testDeserializeBatchNormFileHyperparametersWin);
    RUN_TEST(testDeserializeBatchNormRejectsNonFloat32ForwardMath);
    RUN_TEST(testDeserializeBatchNormRejectsNonFloat32PropLossMath);
    RUN_TEST(testDeserializeGroupNormRejectsIncoherentGeometry);
    RUN_TEST(testDeserializeLayerNormRejectsNormalizedShapeGammaMismatch);
    RUN_TEST(testDeserializeSoftmaxRejectsBfpMathOverANonBfpWire);
    RUN_TEST(testDeserializeDropoutRejectsAnOutOfRangeP);
    RUN_TEST(testDeserializeConv1dTransposedRejectsNonValidPadding);
    RUN_TEST(testReadTensorRecordHeaderFloat32);
    RUN_TEST(testReadTensorRecordHeaderGroupedSym);
    RUN_TEST(testReadTensorRecordHeaderGroupedAsym);
    RUN_TEST(testReadTensorRecordHeaderGroupedBfp);
    RUN_TEST(testReadTensorRecordHeaderRejectsRankAboveCap);
    RUN_TEST(testDeserializeModelValidatesTheWiring);
    RUN_TEST(testDeserializeRejectsADifferentlyWiredSkeleton);
    RUN_TEST(testDeserializeRejectsADifferentInputStart);
    RUN_TEST(testDeserializeRejectsATruncatedWiringRecord);
    RUN_TEST(testDeserializeAddRejectsNonFloat32Math);
    RUN_TEST(testDeserializeRejectsInvalidDyadicScale);
    return UNITY_END();
}
